use std::collections::HashMap;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio_rustls::TlsAcceptor;
use rustls::ServerConfig;
use rustls::server::WebPkiClientVerifier;
use rustls::RootCertStore;
use rkyv::Deserialize;
use SmartCameraTethering2_shared_types::{
    ArchivedMessageToPostProcessor, ArchivedOutputDestination, ArchivedPostProcessingStep,
    MessageToCameraServer,
};

const BIND_ADDR: &str = "0.0.0.0:9000";
const CERT_PATH: &str = "/certs/client2.crt";
const KEY_PATH: &str = "/certs/client2.key";
const CA_PATH: &str = "/certs/root.crt";
const TEMP_DIR: &str = "temp";
const PERMANENT_DIR: &str = "/saved";

/// Per-config trigger info extracted from archived PostProcessingConfig
#[derive(Clone)]
struct SessionConfig {
    trigger_every_n_images: u32,
    steps: Vec<ProcessingStep>,
}

#[derive(Clone, Debug)]
enum OutputDest {
    Camera,
    SystemStorage,
    ServerStorage(String),
}

#[derive(Clone, Debug)]
enum ProcessingStep {
    Blend { number_of_images: u32, blending_mode: rsraw_utils::blending::BlendingMode },
    Save { destination: OutputDest },
    Return,
    Upload { base_url: String, username: Option<String>, password: Option<String> },
}

const SESSION_INACTIVITY_TIMEOUT_SECS: u64 = 30 * 60;

/// Returns a unique file path in `dir` based on the datetime stem.
/// If `<stem>.tiff` already exists, tries `<stem>_1.tiff`, `<stem>_2.tiff`, …
fn unique_tiff_path(dir: &std::path::Path, stem: &str) -> PathBuf {
    let candidate = dir.join(format!("{}.tiff", stem));
    if !candidate.exists() {
        return candidate;
    }
    let mut counter = 1u32;
    loop {
        let candidate = dir.join(format!("{}_{}.tiff", stem, counter));
        if !candidate.exists() {
            return candidate;
        }
        counter += 1;
    }
}

/// Returns a unique file path in `dir` based on the datetime stem for JPEG files.
fn unique_jpeg_path(dir: &std::path::Path, stem: &str) -> PathBuf {
    let candidate = dir.join(format!("{}.jpg", stem));
    if !candidate.exists() {
        return candidate;
    }
    let mut counter = 1u32;
    loop {
        let candidate = dir.join(format!("{}_{}.jpg", stem, counter));
        if !candidate.exists() {
            return candidate;
        }
        counter += 1;
    }
}

/// Per-session state
struct Session {
    /// Paths to raw image files stored in temp/<session_id>/
    raw_image_paths: Vec<PathBuf>,
    configs: Vec<SessionConfig>,
    last_activity: Instant,
}

impl Session {
    fn new() -> Self {
        Self {
            raw_image_paths: Vec::new(),
            configs: Vec::new(),
            last_activity: Instant::now(),
        }
    }
}

type Sessions = Arc<Mutex<HashMap<u64, Session>>>;

#[tokio::main]
async fn main() {
    rustls::crypto::ring::default_provider().install_default()
        .expect("Failed to install rustls crypto provider");

    // Clear temp folder on startup
    if std::path::Path::new(TEMP_DIR).exists() {
        std::fs::remove_dir_all(TEMP_DIR).expect("Failed to clear temp dir");
    }
    std::fs::create_dir_all(TEMP_DIR).expect("Failed to create temp dir");
    std::fs::create_dir_all(PERMANENT_DIR).expect("Failed to create permanent dir");

    let tls_acceptor = build_tls_acceptor();
    let sessions: Sessions = Arc::new(Mutex::new(HashMap::new()));

    // Background task: clean up raw images for inactive sessions every minute
    {
        let sessions = sessions.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
                let mut map = sessions.lock().await;
                for (session_id, session) in map.iter_mut() {
                    if session.last_activity.elapsed().as_secs() >= SESSION_INACTIVITY_TIMEOUT_SECS
                        && !session.raw_image_paths.is_empty()
                    {
                        println!("Session {} inactive for 30 minutes, deleting temp raw images.", session_id);
                        let temp_session_dir = PathBuf::from(TEMP_DIR).join(session_id.to_string());
                        let _ = std::fs::remove_dir_all(&temp_session_dir);
                        session.raw_image_paths.clear();
                    }
                }
            }
        });
    }

    let listener = TcpListener::bind(BIND_ADDR).await
        .expect("Failed to bind TCP listener");
    println!("Post-processing server listening on {}", BIND_ADDR);

    let mut next_session_id: u64 = 0;

    loop {
        match listener.accept().await {
            Ok((stream, peer_addr)) => {
                println!("New connection from {}", peer_addr);
                let acceptor = tls_acceptor.clone();
                let sessions = sessions.clone();
                let session_id = next_session_id;
                next_session_id += 1;

                tokio::spawn(async move {
                    match acceptor.accept(stream).await {
                        Ok(tls_stream) => {
                            if let Err(e) = handle_connection(tls_stream, sessions, session_id).await {
                                eprintln!("Connection error from {}: {:?}", peer_addr, e);
                            }
                        }
                        Err(e) => {
                            eprintln!("TLS handshake failed from {}: {:?}", peer_addr, e);
                        }
                    }
                });
            }
            Err(e) => {
                eprintln!("Accept error: {:?}", e);
            }
        }
    }
}

fn build_tls_acceptor() -> TlsAcceptor {
    let cert_file = std::fs::File::open(CERT_PATH).expect("Cannot open server cert");
    let certs: Vec<rustls::pki_types::CertificateDer> =
        rustls_pemfile::certs(&mut BufReader::new(cert_file))
            .map(|r| r.expect("Invalid cert PEM"))
            .collect();

    let key_file = std::fs::File::open(KEY_PATH).expect("Cannot open server key");
    let key = rustls_pemfile::private_key(&mut BufReader::new(key_file))
        .expect("Failed to read private key")
        .expect("No private key found");

    let ca_file = std::fs::File::open(CA_PATH).expect("Cannot open CA cert");
    let ca_certs: Vec<rustls::pki_types::CertificateDer> =
        rustls_pemfile::certs(&mut BufReader::new(ca_file))
            .map(|r| r.expect("Invalid CA cert PEM"))
            .collect();

    let mut root_store = RootCertStore::empty();
    for ca_cert in ca_certs {
        root_store.add(ca_cert).expect("Failed to add CA cert");
    }

    let client_verifier = WebPkiClientVerifier::builder(Arc::new(root_store))
        .build()
        .expect("Failed to build client verifier");

    let config = ServerConfig::builder()
        .with_client_cert_verifier(client_verifier)
        .with_single_cert(certs, key)
        .expect("Failed to build TLS server config");

    TlsAcceptor::from(Arc::new(config))
}

async fn handle_connection<S>(
    mut stream: S,
    sessions: Sessions,
    assigned_session_id: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    S: AsyncReadExt + AsyncWriteExt + Unpin,
{
    // The actual session_id may be overridden by ResumeSession
    let mut session_id = assigned_session_id;
    loop {
        let mut len_buf = [0u8; 4];
        match stream.read_exact(&mut len_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                println!("Session {} client disconnected (session kept alive for reconnect).", session_id);
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        }
        let msg_len = u32::from_be_bytes(len_buf) as usize;

        let mut buf = vec![0u8; msg_len];
        stream.read_exact(&mut buf).await?;

        let archived = rkyv::check_archived_root::<SmartCameraTethering2_shared_types::MessageToPostProcessor>(&buf)
            .map_err(|e| format!("Archive check error: {:?}", e))?;

        match archived {
            ArchivedMessageToPostProcessor::StartSession => {
                session_id = assigned_session_id;
                println!("Session {} started.", session_id);
                {
                    let mut map = sessions.lock().await;
                    map.insert(session_id, Session::new());
                }
                send_message(&mut stream, &MessageToCameraServer::SessionId(session_id)).await
                    .unwrap_or_else(|e| eprintln!("Failed to send SessionId: {:?}", e));
            }
            ArchivedMessageToPostProcessor::ResumeSession(archived_id) => {
                let requested_id: u64 = (*archived_id).into();
                let exists = sessions.lock().await.contains_key(&requested_id);
                if exists {
                    session_id = requested_id;
                    println!("Session {} resumed.", session_id);
                } else {
                    // Session no longer exists, start a fresh one
                    session_id = assigned_session_id;
                    println!("Session {} not found, starting new session {} instead.", requested_id, session_id);
                    let mut map = sessions.lock().await;
                    map.insert(session_id, Session::new());
                }
                send_message(&mut stream, &MessageToCameraServer::SessionId(session_id)).await
                    .unwrap_or_else(|e| eprintln!("Failed to send SessionId after resume: {:?}", e));
            }
            ArchivedMessageToPostProcessor::SetPostProcessorConfig(archived_configs) => {
                let mut new_configs: Vec<SessionConfig> = Vec::new();
                for archived_config in archived_configs.iter() {
                    let trigger_every_n_images: u32 = archived_config.trigger_every_n_images.into();
                    let mut steps: Vec<ProcessingStep> = Vec::new();
                    for archived_step in archived_config.steps.iter() {
                        match archived_step {
                            ArchivedPostProcessingStep::Blend(b) => {
                                let n: u32 = b.number_of_images.into();
                                let shared_mode: SmartCameraTethering2_shared_types::BlendingMode =
                                    b.blending_mode.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new())
                                    .unwrap_or(SmartCameraTethering2_shared_types::BlendingMode::Average);
                                let mode: rsraw_utils::blending::BlendingMode = shared_mode.into();
                                steps.push(ProcessingStep::Blend { number_of_images: n, blending_mode: mode });
                            }
                            ArchivedPostProcessingStep::Save(s) => {
                                let dest = match &s.output_destination {
                                    ArchivedOutputDestination::Camera => OutputDest::Camera,
                                    ArchivedOutputDestination::SystemStorage => OutputDest::SystemStorage,
                                    ArchivedOutputDestination::ServerStorage(path) => OutputDest::ServerStorage(path.as_str().to_string()),
                                };
                                steps.push(ProcessingStep::Save { destination: dest });
                            }
                            ArchivedPostProcessingStep::Return => {
                                steps.push(ProcessingStep::Return);
                            }
                            ArchivedPostProcessingStep::Upload(u) => {
                                let dest = u.upload_destination.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new())
                                    .map_err(|e| format!("Deserialize upload dest error: {:?}", e))?;
                                if let SmartCameraTethering2_shared_types::UploadDestination::Webdav(w) = dest {
                                    steps.push(ProcessingStep::Upload {
                                        base_url: w.base_url,
                                        username: w.username,
                                        password: w.password,
                                    });
                                }
                            }
                        }
                    }
                    println!("Session {} received config: trigger every {} images, {} steps.", session_id, trigger_every_n_images, steps.len());
                    new_configs.push(SessionConfig { trigger_every_n_images, steps });
                }
                let mut map = sessions.lock().await;
                let session = map.entry(session_id).or_insert_with(Session::new);
                session.configs = new_configs;
                session.raw_image_paths.clear();
                session.last_activity = Instant::now();
                let temp_session_dir = PathBuf::from(TEMP_DIR).join(session_id.to_string());
                if temp_session_dir.exists() {
                    if let Err(e) = std::fs::remove_dir_all(&temp_session_dir) {
                        eprintln!("Session {} failed to clear temp dir on config update: {:?}", session_id, e);
                    }
                }
                if let Err(e) = std::fs::create_dir_all(&temp_session_dir) {
                    eprintln!("Session {} failed to recreate temp dir on config update: {:?}", session_id, e);
                }
            }
            ArchivedMessageToPostProcessor::CompressedRawImage(compressed_data) => {
                let compressed_bytes: Vec<u8> = compressed_data
                    .deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new())
                    .map_err(|e| format!("Deserialize compressed raw bytes error: {:?}", e))?;
                let raw_bytes = zstd::decode_all(std::io::Cursor::new(&compressed_bytes))
                    .map_err(|e| format!("Zstd decompression error: {:?}", e))?;
                println!("Session {} received compressed raw image ({} -> {} bytes).", session_id, compressed_bytes.len(), raw_bytes.len());
                handle_raw_bytes(raw_bytes, session_id, &sessions, &mut stream).await?;
            }
            ArchivedMessageToPostProcessor::RawImage(raw_data) => {
                let raw_bytes: Vec<u8> = raw_data
                    .deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new())
                    .map_err(|e| format!("Deserialize raw bytes error: {:?}", e))?;
                handle_raw_bytes(raw_bytes, session_id, &sessions, &mut stream).await?;
            }
        }
    }
}

async fn handle_raw_bytes<S>(
    raw_bytes: Vec<u8>,
    session_id: u64,
    sessions: &Arc<Mutex<HashMap<u64, Session>>>,
    stream: &mut S,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    S: AsyncWriteExt + Unpin,
{
    // Write raw bytes to temp/<session_id>/<index>.raw
    let temp_session_dir = PathBuf::from(TEMP_DIR).join(session_id.to_string());
    std::fs::create_dir_all(&temp_session_dir)
        .map_err(|e| format!("Failed to create session temp dir: {:?}", e))?;

    let (image_count, configs_snapshot): (usize, Vec<SessionConfig>) = {
        let mut map = sessions.lock().await;
        let session = map.entry(session_id).or_insert_with(Session::new);
        let index = session.raw_image_paths.len();
        let raw_path = temp_session_dir.join(format!("{}.raw", index));
        std::fs::write(&raw_path, &raw_bytes)
            .map_err(|e| format!("Failed to write raw to temp: {:?}", e))?;
        println!(
            "Session {} received raw image ({} bytes), stored at {:?}.",
            session_id, raw_bytes.len(), raw_path
        );
        session.raw_image_paths.push(raw_path);
        session.last_activity = Instant::now();
        (session.raw_image_paths.len(), session.configs.clone())
    };

    // Check each config's trigger
    for config in &configs_snapshot {
        let n = config.trigger_every_n_images as usize;
        if n == 0 {
            continue;
        }
        if image_count % n == 0 {
            println!(
                "Session {} auto-triggering post-processing after {} images.",
                session_id, image_count
            );
            // Collect the last N raw image paths for processing
            let raws_to_process = {
                let map = sessions.lock().await;
                if let Some(session) = map.get(&session_id) {
                    let start = session.raw_image_paths.len().saturating_sub(n);
                    session.raw_image_paths[start..].to_vec()
                } else {
                    vec![]
                }
            };

            if raws_to_process.is_empty() {
                continue;
            }

            match process_raw_images(raws_to_process, config.steps.clone(), session_id, image_count).await {
                Ok((maybe_jpeg, system_files)) => {
                    for (filename, data) in system_files {
                        let msg = MessageToCameraServer::SaveToSystemStorage { filename, data };
                        send_message(stream, &msg).await?;
                    }
                    if let Some(processed_bytes) = maybe_jpeg {
                        let response = MessageToCameraServer::ReturnedImage(processed_bytes);
                        send_message(stream, &response).await?;
                    }
                }
                Err(e) => {
                    eprintln!("Session {} post-processing error: {:?}", session_id, e);
                }
            }
        }
    }
    Ok(())
}


/// A pending upload: the data and URL/auth needed to PUT a file via WebDAV.
struct PendingUpload {
    url: String,
    data: Vec<u8>,
    content_type: &'static str,
    username: Option<String>,
    password: Option<String>,
}

/// Process raw images by executing the configured steps pipeline.
/// Returns `(Option<jpeg_bytes>, Vec<(filename, tiff_bytes)>)` for system storage files.
async fn process_raw_images(
    raw_paths: Vec<PathBuf>,
    steps: Vec<ProcessingStep>,
    session_id: u64,
    image_index: usize,
) -> Result<(Option<Vec<u8>>, Vec<(String, Vec<u8>)>), Box<dyn std::error::Error + Send + Sync>> {
    let (result, pending_uploads) = tokio::task::spawn_blocking(move || -> Result<((Option<Vec<u8>>, Vec<(String, Vec<u8>)>), Vec<PendingUpload>), Box<dyn std::error::Error + Send + Sync>> {
        let raw_images_count = raw_paths.len();

        let load_raw = |path: &PathBuf| -> Result<rsraw::RawImage, Box<dyn std::error::Error + Send + Sync>> {
            let data = std::fs::read(path).map_err(|e| format!("Failed to read raw {:?}: {}", path, e))?;
            let mut raw = rsraw::RawImage::open(&data).map_err(|e| format!("Failed to open raw {:?}: {:?}", path, e))?;
            raw.unpack().map_err(|e| format!("Failed to unpack raw {:?}: {}", path, e))?;
            Ok(raw)
        };

        // current_raw holds a RawImage (before any conversion).
        // current_tiff holds a path to a TIFF produced by a previous step (e.g. Blend).
        // Once current_tiff is set, subsequent steps (Save, Return) use it directly
        // instead of re-converting from the original raw.
        let mut current_raw: Option<rsraw::RawImage> = if let Some(last_path) = raw_paths.last() {
            Some(load_raw(last_path)?)
        } else {
            None
        };
        let mut current_tiff: Option<PathBuf> = None;
        let mut return_jpeg: Option<Vec<u8>> = None;
        let mut system_storage_files: Vec<(String, Vec<u8>)> = Vec::new();
        // Collect upload/save configs so we can also upload/save the JPEG after the loop.
        let mut upload_configs: Vec<(String, Option<String>, Option<String>)> = Vec::new();
        let mut save_destinations: Vec<OutputDest> = Vec::new();
        let mut pending_uploads: Vec<PendingUpload> = Vec::new();

        for step in &steps {
            match step {
                ProcessingStep::Blend { number_of_images, blending_mode } => {
                    let n = *number_of_images as usize;
                    let take = n.saturating_sub(1).min(raw_images_count.saturating_sub(1));
                    let start = raw_images_count.saturating_sub(1).saturating_sub(take);
                    let mut to_blend: Vec<rsraw::RawImage> = raw_paths[start..raw_paths.len().saturating_sub(1)]
                        .iter()
                        .map(|path| load_raw(path))
                        .collect::<Result<Vec<_>, _>>()?;
                    if let Some(raw) = current_raw.take() {
                        to_blend.push(raw);
                    }
                    if to_blend.is_empty() {
                        return Err("No images to blend".into());
                    }
                    let blended = rsraw_utils::blend_raw_images(to_blend, blending_mode.clone())
                        .map_err(|e| format!("Blend error: {:?}", e))?;
                    // Convert blended result to TIFF immediately so subsequent steps use it.
                    let tiff_path = PathBuf::from(TEMP_DIR)
                        .join(format!("session{}_{}_blended.tiff", session_id, image_index));
                    rsraw_utils::convert_raw(blended, rsraw_utils::OutputFormat::TIFF, &tiff_path)
                        .map_err(|e| format!("Failed to convert blended to TIFF: {:?}", e))?;
                    // Drop any previous intermediate TIFF.
                    if let Some(old) = current_tiff.take() {
                        if old != tiff_path { let _ = std::fs::remove_file(&old); }
                    }
                    current_tiff = Some(tiff_path);
                    current_raw = None;
                }
                ProcessingStep::Save { destination } => {
                    save_destinations.push(destination.clone());
                    let tiff_path = if let Some(ref p) = current_tiff {
                        p.clone()
                    } else {
                        let raw = current_raw.take().ok_or("No image for save step")?;
                        let p = PathBuf::from(TEMP_DIR)
                            .join(format!("session{}_{}_intermediate.tiff", session_id, image_index));
                        rsraw_utils::convert_raw(raw, rsraw_utils::OutputFormat::TIFF, &p)
                            .map_err(|e| format!("Failed to convert to TIFF for save: {:?}", e))?;
                        current_tiff = Some(p.clone());
                        p
                    };
                    match destination {
                        OutputDest::ServerStorage(dest_path) => {
                            let dir = PathBuf::from(PERMANENT_DIR).join(dest_path);
                            std::fs::create_dir_all(&dir)?;
                            let stem = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
                            let out_path = unique_tiff_path(&dir, &stem);
                            let _filename = out_path.file_name().unwrap().to_string_lossy().into_owned();
                            std::fs::copy(&tiff_path, &out_path)
                                .map_err(|e| format!("Failed to copy to server storage: {}", e))?;
                            println!("Session {} saved permanently to {:?}", session_id, out_path);
                        }
                        OutputDest::SystemStorage => {
                            let stem = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
                            let filename = format!("{}.tiff", stem);
                            let data = std::fs::read(&tiff_path)
                                .map_err(|e| format!("Failed to read TIFF for system storage: {}", e))?;
                            system_storage_files.push((filename.clone(), data));
                            println!("Session {} queued for system storage: {}", session_id, filename);
                        }
                        OutputDest::Camera => {
                            println!("Session {} Camera destination not supported on server, skipping.", session_id);
                        }
                    }
                    // Keep current_tiff intact so subsequent steps (e.g. Return) can still use it.
                }
                ProcessingStep::Return => {
                    let jpeg_path = PathBuf::from(TEMP_DIR)
                        .join(format!("session{}_{}_return.jpg", session_id, image_index));
                    if let Some(ref tiff_path) = current_tiff {
                        // Convert the existing TIFF to JPEG using the image crate.
                        let img = image::open(tiff_path)
                            .map_err(|e| format!("Failed to open TIFF for return: {:?}", e))?;
                        img.save_with_format(&jpeg_path, image::ImageFormat::Jpeg)
                            .map_err(|e| format!("Failed to save JPEG from TIFF: {:?}", e))?;
                    } else {
                        let raw = current_raw.take().ok_or("No image for return step")?;
                        rsraw_utils::convert_raw(raw, rsraw_utils::OutputFormat::JPEG, &jpeg_path)
                            .map_err(|e| format!("Failed to convert to JPEG: {:?}", e))?;
                    }
                    let jpeg_bytes = std::fs::read(&jpeg_path)
                        .map_err(|e| format!("Failed to read JPEG: {:?}", e))?;
                    let _ = std::fs::remove_file(&jpeg_path);
                    println!("Session {} returning JPEG ({} bytes) to client.", session_id, jpeg_bytes.len());
                    // Store for return after all steps complete (don't return early — Save steps may follow).
                    return_jpeg = Some(jpeg_bytes);
                }
                ProcessingStep::Upload { base_url, username, password } => {
                    upload_configs.push((base_url.clone(), username.clone(), password.clone()));
                    let tiff_path = if let Some(ref p) = current_tiff {
                        p.clone()
                    } else {
                        let raw = current_raw.take().ok_or("No image for upload step")?;
                        let p = PathBuf::from(TEMP_DIR)
                            .join(format!("session{}_{}_upload.tiff", session_id, image_index));
                        rsraw_utils::convert_raw(raw, rsraw_utils::OutputFormat::TIFF, &p)
                            .map_err(|e| format!("Failed to convert to TIFF for upload: {:?}", e))?;
                        current_tiff = Some(p.clone());
                        p
                    };
                    let stem = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
                    let filename = format!("{}.tiff", stem);
                    let upload_url = format!("{}/{}", base_url.trim_end_matches('/'), filename);
                    let data = std::fs::read(&tiff_path)
                        .map_err(|e| format!("Failed to read TIFF for upload: {}", e))?;
                    pending_uploads.push(PendingUpload {
                        url: upload_url,
                        data,
                        content_type: "image/tiff",
                        username: username.clone(),
                        password: password.clone(),
                    });
                    println!("Session {} queued TIFF {} for background upload", session_id, filename);
                }
            }
        }

        // Clean up any intermediate TIFF.
        if let Some(tiff_path) = current_tiff.take() {
            let _ = std::fs::remove_file(&tiff_path);
        }

        // If a JPEG was generated via Return, also save/upload it.
        if let Some(ref jpeg_bytes) = return_jpeg {
            let stem = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
            for dest in &save_destinations {
                match dest {
                    OutputDest::ServerStorage(dest_path) => {
                        let dir = PathBuf::from(PERMANENT_DIR).join(dest_path);
                        std::fs::create_dir_all(&dir)?;
                        let out_path = unique_jpeg_path(&dir, &stem);
                        std::fs::write(&out_path, jpeg_bytes)
                            .map_err(|e| format!("Failed to save JPEG to server storage: {}", e))?;
                        println!("Session {} saved JPEG permanently to {:?}", session_id, out_path);
                    }
                    OutputDest::SystemStorage => {
                        let filename = format!("{}.jpg", stem);
                        system_storage_files.push((filename.clone(), jpeg_bytes.clone()));
                        println!("Session {} queued JPEG for system storage: {}", session_id, filename);
                    }
                    OutputDest::Camera => {}
                }
            }
            for (base_url, username, password) in &upload_configs {
                let filename = format!("{}.jpg", stem);
                let upload_url = format!("{}/{}", base_url.trim_end_matches('/'), filename);
                pending_uploads.push(PendingUpload {
                    url: upload_url.clone(),
                    data: jpeg_bytes.clone(),
                    content_type: "image/jpeg",
                    username: username.clone(),
                    password: password.clone(),
                });
                println!("Session {} queued JPEG {} for background upload", session_id, filename);
            }
        }

        Ok(((return_jpeg, system_storage_files), pending_uploads))
    })
    .await??;

    // Fire off all uploads in the background — do not block returning/saving.
    for upload in pending_uploads {
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            let mut req = client.put(&upload.url)
                .header("Content-Type", upload.content_type)
                .body(upload.data);
            if let (Some(user), Some(pass)) = (upload.username.as_deref(), upload.password.as_deref()) {
                req = req.basic_auth(user, Some(pass));
            } else if let Some(user) = upload.username.as_deref() {
                req = req.basic_auth(user, None::<&str>);
            }
            match req.send().await {
                Ok(response) if response.status().is_success() =>
                    println!("Background upload to {} succeeded", upload.url),
                Ok(response) =>
                    eprintln!("Background upload to {} failed with status: {}", upload.url, response.status()),
                Err(e) =>
                    eprintln!("Background upload to {} failed: {}", upload.url, e),
            }
        });
    }

    Ok(result)
}

async fn send_message<S, M>(stream: &mut S, message: &M) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    S: AsyncWriteExt + Unpin,
    M: rkyv::Serialize<rkyv::ser::serializers::AllocSerializer<4096>>,
{
    let bytes = rkyv::to_bytes::<_, 4096>(message)
        .map_err(|e| format!("Serialization error: {:?}", e))?;
    let len = bytes.len() as u32;
    stream.write_all(&len.to_be_bytes()).await?;
    stream.write_all(&bytes).await?;
    Ok(())
}
