// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! The on-disk recording format.
//!
//! A results directory holds one directory per recorded exchange under `recordings/`, named by
//! the exchange's zero-padded sequence number. Each holds `meta.json` ([`RecordingMeta`]) and
//! the request and response bodies as verbatim files, whose names `meta.json` gives. An exchange
//! refused because a credential was found in a body leaves an [`Incident`] under `incidents/`
//! instead, and no recording.
//!
//! A recording is committed atomically: it is written in full under `.staging/` and then renamed
//! into `recordings/`, so a reader never observes a partial one. A staging directory left behind
//! by a crash is never read. Incident records are committed the same way.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use http::header::{CONNECTION, HeaderMap, HeaderName};
use serde::{Deserialize, Serialize};

/// Version of the recording format written by this crate. Readers reject any other version.
pub const FORMAT_VERSION: u32 = 1;

/// Name of the metadata file inside each recording directory.
pub const META_FILE: &str = "meta.json";

/// File name used for a recorded request body.
pub const REQUEST_BODY_FILE: &str = "request.body";

/// File name used for a buffered recorded response body.
pub const BUFFERED_RESPONSE_FILE: &str = "response.body";

/// File name used for a streamed (server-sent events) recorded response body.
pub const STREAM_RESPONSE_FILE: &str = "response.sse";

const RECORDINGS_DIR: &str = "recordings";
const INCIDENTS_DIR: &str = "incidents";
const STAGING_DIR: &str = ".staging";

/// Headers that describe one connection rather than the exchange: `Host`, `Connection`,
/// `Keep-Alive`, `TE`, `Trailer`, `Transfer-Encoding`, `Upgrade`, `Content-Length` and every
/// `Proxy-*`. They are neither forwarded as received, nor recorded, nor compared on replay; the
/// HTTP stack on each hop writes its own. Case-insensitive.
pub fn is_transport_header(name: &str) -> bool {
    const TRANSPORT: &[&str] = &[
        "host",
        "connection",
        "keep-alive",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
    ];
    let lower = name.to_ascii_lowercase();
    TRANSPORT.contains(&lower.as_str()) || lower.starts_with("proxy-")
}

/// The end-to-end headers of `headers`: every header except transport headers and any header
/// the `Connection` header names. These are what the capture forwards and records, and what
/// the replay compares.
pub fn end_to_end_headers(headers: &HeaderMap) -> HeaderMap {
    let connection_scoped: Vec<HeaderName> = headers
        .get_all(CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect();
    let mut out = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        if !is_transport_header(name.as_str()) && !connection_scoped.contains(name) {
            out.append(name.clone(), value.clone());
        }
    }
    out
}

/// Splits a request target as the gateway sends it, `/<route>/<rest>`, into the route name and
/// the upstream request target `/<rest>` — which is what [`RecordingMeta::uri`] records, before
/// redaction. `None` when the target does not start with `/`.
pub fn split_route_target(target: &str) -> Option<(&str, &str)> {
    let after_slash = target.strip_prefix('/')?;
    let end = after_slash.find(['/', '?']).unwrap_or(after_slash.len());
    let rest = &after_slash[end..];
    Some((
        &after_slash[..end],
        if rest.is_empty() { "/" } else { rest },
    ))
}

/// A header value as recorded: text when it is valid UTF-8, raw bytes otherwise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FieldValue {
    /// A UTF-8 value, recorded as a JSON string.
    Text(String),
    /// A value that is not valid UTF-8, recorded as a JSON array of bytes.
    Bytes(Vec<u8>),
}

impl FieldValue {
    /// Records `bytes` as text when they are valid UTF-8 and as raw bytes otherwise.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        match String::from_utf8(bytes) {
            Ok(text) => Self::Text(text),
            Err(err) => Self::Bytes(err.into_bytes()),
        }
    }

    /// The value's bytes, as they appear on the wire (after redaction).
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Text(text) => text.as_bytes(),
            Self::Bytes(bytes) => bytes,
        }
    }
}

/// One header line, in the order it appeared. Repeated names stay separate entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeaderField {
    /// Header name, lower-cased as HTTP/1.1 and HTTP/2 stacks present it.
    pub name: String,
    /// Header value after redaction.
    pub value: FieldValue,
}

/// How the response body was delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BodyKind {
    /// A complete, non-streamed body.
    Buffered,
    /// A `text/event-stream` body.
    EventStream,
}

/// Which side ended the exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Termination {
    /// The upstream body reached its end; the recording holds the complete response.
    UpstreamComplete,
    /// The downstream client stopped reading before the upstream body ended, and the upstream
    /// connection was closed. The recording holds only the prefix the client consumed and must
    /// never be presented as a complete response.
    DownstreamClosedFirst,
}

/// Contents of `meta.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordingMeta {
    /// Always [`FORMAT_VERSION`] for a recording this crate writes.
    pub format_version: u32,
    /// Position of the exchange in the capture session, starting at 0.
    pub sequence: u64,
    /// Name of the capture route the exchange arrived on.
    pub route: String,
    /// Upstream origin the route forwards to, e.g. `https://api.example.com`.
    pub upstream_origin: String,
    /// Request method.
    pub method: String,
    /// Path and query sent upstream, after redaction.
    pub uri: String,
    /// Response status code.
    pub status: u16,
    /// Request headers as sent upstream, after redaction.
    pub request_headers: Vec<HeaderField>,
    /// Response headers as received from upstream, after redaction.
    pub response_headers: Vec<HeaderField>,
    /// File holding the request body, relative to the recording directory.
    pub request_body_file: String,
    /// File holding the response body, relative to the recording directory.
    pub response_body_file: String,
    /// Whether the response was buffered or streamed.
    pub response_kind: BodyKind,
    /// Ordered sizes of the response body chunks as the capture observed them from upstream.
    /// They sum to the length of the response body file.
    pub response_chunks: Vec<u64>,
    /// Which side ended the exchange.
    pub termination: Termination,
}

/// Where a credential was found in a refused exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Surface {
    /// The request body.
    RequestBody,
    /// A buffered response body.
    ResponseBody,
    /// A streamed (server-sent events) response body, across all its chunks.
    ResponseStream,
}

/// Record of an exchange that was forwarded but not recorded, because a credential value was
/// found in one of its bodies. It names the exchange and where the value was found, and holds no
/// payload: no body, header, URI or credential.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Incident {
    /// Always [`FORMAT_VERSION`] for an incident this crate writes.
    pub format_version: u32,
    /// Position of the exchange in the capture session.
    pub sequence: u64,
    /// Name of the capture route the exchange arrived on.
    pub route: String,
    /// Request method.
    pub method: String,
    /// Response status code; `None` when no response arrived.
    pub status: Option<u16>,
    /// Every surface a credential value was found in.
    pub surfaces: Vec<Surface>,
}

/// A complete recorded exchange: metadata plus both bodies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recording {
    /// The exchange's metadata.
    pub meta: RecordingMeta,
    /// Request body bytes, verbatim.
    pub request_body: Vec<u8>,
    /// Response body bytes, verbatim.
    pub response_body: Vec<u8>,
}

/// Why a recording could not be written or read.
#[derive(Debug, thiserror::Error)]
pub enum RecordingError {
    /// A filesystem operation failed.
    #[error("{op} {}: {source}", path.display())]
    Io {
        /// What was being attempted.
        op: &'static str,
        /// The path it was attempted on.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// `meta.json` could not be encoded or decoded.
    #[error("meta file {}: {source}", path.display())]
    Json {
        /// The metadata file.
        path: PathBuf,
        /// The underlying error.
        source: serde_json::Error,
    },
    /// The recording was written by a different, unsupported version of the format.
    #[error(
        "recording {} has format version {found}; this reader supports only {FORMAT_VERSION}",
        path.display()
    )]
    UnsupportedVersion {
        /// The recording directory.
        path: PathBuf,
        /// The version it declares.
        found: u32,
    },
    /// The recording is internally inconsistent.
    #[error("recording {} is invalid: {reason}", path.display())]
    Invalid {
        /// The recording directory.
        path: PathBuf,
        /// What is wrong with it.
        reason: String,
    },
    /// A recording with this sequence number is already committed.
    #[error("a recording with sequence {sequence} is already committed")]
    AlreadyCommitted {
        /// The duplicated sequence number.
        sequence: u64,
    },
}

/// A results directory: where recordings are committed and read back from.
#[derive(Debug, Clone)]
pub struct RecordingStore {
    root: PathBuf,
}

impl RecordingStore {
    /// Opens `root` as a results directory, creating it and its layout if absent.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, RecordingError> {
        let root = root.into();
        for dir in [
            root.join(RECORDINGS_DIR),
            root.join(INCIDENTS_DIR),
            root.join(STAGING_DIR),
        ] {
            fs::create_dir_all(&dir).map_err(|source| io_err("create", &dir, source))?;
        }
        Ok(Self { root })
    }

    /// Opens `root` as an existing results directory, for reading: nothing is created, and a
    /// directory without a `recordings/` layout is an error.
    pub fn existing(root: impl Into<PathBuf>) -> Result<Self, RecordingError> {
        let root = root.into();
        let recordings = root.join(RECORDINGS_DIR);
        fs::read_dir(&recordings).map_err(|source| io_err("open", &recordings, source))?;
        Ok(Self { root })
    }

    /// The results directory this store writes to.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Commits `recording` atomically and returns its directory.
    ///
    /// The recording is validated first — current format version, plain body file names, chunk
    /// manifest summing to the response body length — so an invalid recording is never written.
    pub fn commit(&self, recording: &Recording) -> Result<PathBuf, RecordingError> {
        let meta = &recording.meta;
        let name = dir_name(meta.sequence);
        let target = self.root.join(RECORDINGS_DIR).join(&name);
        validate(recording, &target)?;
        if target.exists() {
            return Err(RecordingError::AlreadyCommitted {
                sequence: meta.sequence,
            });
        }

        let staging = self.root.join(STAGING_DIR).join(&name);
        if staging.exists() {
            fs::remove_dir_all(&staging).map_err(|source| io_err("remove", &staging, source))?;
        }
        fs::create_dir(&staging).map_err(|source| io_err("create", &staging, source))?;
        let meta_path = staging.join(META_FILE);
        let meta_json = serde_json::to_vec_pretty(meta).map_err(|source| RecordingError::Json {
            path: meta_path.clone(),
            source,
        })?;
        write_synced(
            &staging.join(&meta.request_body_file),
            &recording.request_body,
        )?;
        write_synced(
            &staging.join(&meta.response_body_file),
            &recording.response_body,
        )?;
        write_synced(&meta_path, &meta_json)?;
        sync_dir(&staging)?;

        fs::rename(&staging, &target).map_err(|source| io_err("commit", &target, source))?;
        sync_dir(&self.root.join(RECORDINGS_DIR))?;
        Ok(target)
    }

    /// Whether no recording and no incident record has been committed here.
    pub fn is_empty(&self) -> Result<bool, RecordingError> {
        for dir in [
            self.root.join(RECORDINGS_DIR),
            self.root.join(INCIDENTS_DIR),
        ] {
            let mut entries = fs::read_dir(&dir).map_err(|source| io_err("list", &dir, source))?;
            if entries.next().is_some() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Commits `incident` atomically and returns its path.
    pub fn commit_incident(&self, incident: &Incident) -> Result<PathBuf, RecordingError> {
        let name = format!("{}.json", dir_name(incident.sequence));
        let target = self.root.join(INCIDENTS_DIR).join(&name);
        if target.exists() {
            return Err(RecordingError::AlreadyCommitted {
                sequence: incident.sequence,
            });
        }
        let staging = self.root.join(STAGING_DIR).join(&name);
        let json = serde_json::to_vec_pretty(incident).map_err(|source| RecordingError::Json {
            path: staging.clone(),
            source,
        })?;
        write_synced(&staging, &json)?;
        fs::rename(&staging, &target).map_err(|source| io_err("commit", &target, source))?;
        sync_dir(&self.root.join(INCIDENTS_DIR))?;
        Ok(target)
    }

    /// Reads every committed incident record, ordered by sequence number.
    pub fn load_incidents(&self) -> Result<Vec<Incident>, RecordingError> {
        let dir = self.root.join(INCIDENTS_DIR);
        let entries = fs::read_dir(&dir).map_err(|source| io_err("list", &dir, source))?;
        let mut incidents = Vec::new();
        for entry in entries {
            let path = entry.map_err(|source| io_err("list", &dir, source))?.path();
            let raw = fs::read(&path).map_err(|source| io_err("read", &path, source))?;
            let incident: Incident =
                serde_json::from_slice(&raw).map_err(|source| RecordingError::Json {
                    path: path.clone(),
                    source,
                })?;
            if incident.format_version != FORMAT_VERSION {
                return Err(RecordingError::UnsupportedVersion {
                    path,
                    found: incident.format_version,
                });
            }
            incidents.push(incident);
        }
        incidents.sort_by_key(|incident| incident.sequence);
        Ok(incidents)
    }

    /// Reads every committed recording, ordered by sequence number.
    pub fn load_all(&self) -> Result<Vec<Recording>, RecordingError> {
        let dir = self.root.join(RECORDINGS_DIR);
        let entries = fs::read_dir(&dir).map_err(|source| io_err("list", &dir, source))?;
        let mut recordings = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| io_err("list", &dir, source))?;
            recordings.push(load(&entry.path())?);
        }
        recordings.sort_by_key(|recording| recording.meta.sequence);
        Ok(recordings)
    }
}

/// Reads the recording in `dir`, rejecting an unknown format version or an inconsistent one.
pub fn load(dir: &Path) -> Result<Recording, RecordingError> {
    /// Just enough of `meta.json` to learn its version before trusting the rest of its shape.
    #[derive(Deserialize)]
    struct VersionProbe {
        format_version: u32,
    }

    let meta_path = dir.join(META_FILE);
    let raw = fs::read(&meta_path).map_err(|source| io_err("read", &meta_path, source))?;
    let json_err = |source| RecordingError::Json {
        path: meta_path.clone(),
        source,
    };
    let probe: VersionProbe = serde_json::from_slice(&raw).map_err(json_err)?;
    if probe.format_version != FORMAT_VERSION {
        return Err(RecordingError::UnsupportedVersion {
            path: dir.to_path_buf(),
            found: probe.format_version,
        });
    }
    let meta: RecordingMeta = serde_json::from_slice(&raw).map_err(json_err)?;
    // Checked before the body names are joined onto `dir`, so a hostile name reads nothing.
    check_file_names(&meta, dir)?;

    let read = |name: &str| {
        let path = dir.join(name);
        fs::read(&path).map_err(|source| io_err("read", &path, source))
    };
    let recording = Recording {
        request_body: read(&meta.request_body_file)?,
        response_body: read(&meta.response_body_file)?,
        meta,
    };
    validate(&recording, dir)?;
    Ok(recording)
}

fn dir_name(sequence: u64) -> String {
    format!("{sequence:08}")
}

/// Rejects a recording that is not of the current version, whose body file names are not plain
/// names distinct from each other and from the meta file, or whose chunk manifest does not
/// account for exactly the recorded response body.
fn validate(recording: &Recording, path: &Path) -> Result<(), RecordingError> {
    let meta = &recording.meta;
    if meta.format_version != FORMAT_VERSION {
        return Err(RecordingError::UnsupportedVersion {
            path: path.to_path_buf(),
            found: meta.format_version,
        });
    }
    check_file_names(meta, path)?;
    let manifest_total = meta
        .response_chunks
        .iter()
        .try_fold(0u64, |total, &size| total.checked_add(size));
    if manifest_total != Some(recording.response_body.len() as u64) {
        return Err(invalid(
            path,
            format!(
                "chunk manifest totals {manifest_total:?} bytes; the response body has {}",
                recording.response_body.len()
            ),
        ));
    }
    Ok(())
}

fn check_file_names(meta: &RecordingMeta, path: &Path) -> Result<(), RecordingError> {
    for name in [&meta.request_body_file, &meta.response_body_file] {
        let plain = !name.is_empty()
            && name != "."
            && name != ".."
            && !name.contains(['/', '\\'])
            && name != META_FILE;
        if !plain {
            return Err(invalid(
                path,
                format!("body file name `{name}` is not allowed"),
            ));
        }
    }
    if meta.request_body_file == meta.response_body_file {
        return Err(invalid(
            path,
            "request and response share one body file".into(),
        ));
    }
    Ok(())
}

fn invalid(path: &Path, reason: String) -> RecordingError {
    RecordingError::Invalid {
        path: path.to_path_buf(),
        reason,
    }
}

fn io_err(op: &'static str, path: &Path, source: std::io::Error) -> RecordingError {
    RecordingError::Io {
        op,
        path: path.to_path_buf(),
        source,
    }
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), RecordingError> {
    let mut file = File::create(path).map_err(|source| io_err("create", path, source))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|source| io_err("write", path, source))
}

/// Flushes a directory's entries, so a rename into it survives a crash.
fn sync_dir(path: &Path) -> Result<(), RecordingError> {
    File::open(path)
        .and_then(|dir| dir.sync_all())
        .map_err(|source| io_err("sync", path, source))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(sequence: u64) -> Recording {
        let response_body = b"data: {\"a\":1}\n\ndata: [DONE]\n\n".to_vec();
        Recording {
            meta: RecordingMeta {
                format_version: FORMAT_VERSION,
                sequence,
                route: "openai".into(),
                upstream_origin: "https://api.example.com".into(),
                method: "POST".into(),
                uri: "/v1/chat/completions".into(),
                status: 200,
                request_headers: vec![HeaderField {
                    name: "content-type".into(),
                    value: FieldValue::Text("application/json".into()),
                }],
                response_headers: vec![
                    HeaderField {
                        name: "x-ratelimit-remaining-tokens".into(),
                        value: FieldValue::Text("149984".into()),
                    },
                    HeaderField {
                        name: "x-odd".into(),
                        value: FieldValue::Bytes(vec![0x66, 0xff]),
                    },
                ],
                request_body_file: REQUEST_BODY_FILE.into(),
                response_body_file: STREAM_RESPONSE_FILE.into(),
                response_kind: BodyKind::EventStream,
                response_chunks: vec![10, response_body.len() as u64 - 10],
                termination: Termination::UpstreamComplete,
            },
            request_body: b"{\"model\":\"m\",\"stream\":true}".to_vec(),
            response_body,
        }
    }

    fn store() -> (tempfile::TempDir, RecordingStore) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = RecordingStore::open(dir.path().join("results")).expect("open");
        (dir, store)
    }

    #[test]
    fn committed_recording_reads_back_identically() {
        let (_dir, store) = store();
        let recording = sample(0);
        let path = store.commit(&recording).expect("commit");

        assert_eq!(load(&path).expect("load"), recording);
        assert_eq!(
            fs::read(path.join(STREAM_RESPONSE_FILE)).expect("body"),
            recording.response_body
        );
        assert_eq!(
            fs::read(path.join(REQUEST_BODY_FILE)).expect("body"),
            recording.request_body
        );
    }

    #[test]
    fn meta_file_names_every_field_of_the_format() {
        let (_dir, store) = store();
        let path = store.commit(&sample(0)).expect("commit");
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join(META_FILE)).expect("meta")).expect("json");
        for field in [
            "format_version",
            "sequence",
            "route",
            "upstream_origin",
            "method",
            "uri",
            "status",
            "request_headers",
            "response_headers",
            "request_body_file",
            "response_body_file",
            "response_kind",
            "response_chunks",
            "termination",
        ] {
            assert!(meta.get(field).is_some(), "meta.json lacks `{field}`");
        }
        assert_eq!(meta["format_version"], FORMAT_VERSION);
    }

    #[test]
    fn load_all_orders_by_sequence() {
        let (_dir, store) = store();
        for sequence in [2, 0, 1] {
            store.commit(&sample(sequence)).expect("commit");
        }
        let sequences: Vec<u64> = store
            .load_all()
            .expect("load_all")
            .iter()
            .map(|r| r.meta.sequence)
            .collect();
        assert_eq!(sequences, [0, 1, 2]);
    }

    #[test]
    fn unknown_format_version_is_rejected() {
        let (_dir, store) = store();
        let path = store.commit(&sample(0)).expect("commit");
        let meta_path = path.join(META_FILE);
        let mut meta: serde_json::Value =
            serde_json::from_slice(&fs::read(&meta_path).expect("meta")).expect("json");
        meta["format_version"] = (FORMAT_VERSION + 1).into();
        fs::write(&meta_path, serde_json::to_vec(&meta).expect("encode")).expect("write");

        assert!(matches!(
            load(&path),
            Err(RecordingError::UnsupportedVersion { found, .. }) if found == FORMAT_VERSION + 1
        ));
    }

    #[test]
    fn staging_leftovers_are_never_read() {
        let (_dir, store) = store();
        store.commit(&sample(0)).expect("commit");
        // What a crash mid-write leaves behind: a partial directory with no meta file.
        let partial = store.root().join(STAGING_DIR).join("00000001");
        fs::create_dir_all(&partial).expect("mkdir");
        fs::write(partial.join(REQUEST_BODY_FILE), b"{").expect("write");

        let loaded = store.load_all().expect("load_all");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].meta.sequence, 0);
    }

    #[test]
    fn a_sequence_is_committed_at_most_once() {
        let (_dir, store) = store();
        store.commit(&sample(3)).expect("commit");
        assert!(matches!(
            store.commit(&sample(3)),
            Err(RecordingError::AlreadyCommitted { sequence: 3 })
        ));
    }

    #[test]
    fn inconsistent_recordings_are_never_written() {
        let (_dir, store) = store();

        let mut short_manifest = sample(0);
        short_manifest.meta.response_chunks = vec![1];
        let mut escaping_name = sample(1);
        escaping_name.meta.response_body_file = "../outside".into();
        let mut clashing_names = sample(2);
        clashing_names.meta.response_body_file = REQUEST_BODY_FILE.into();
        let mut old_version = sample(3);
        old_version.meta.format_version = 0;

        for recording in [short_manifest, escaping_name, clashing_names, old_version] {
            assert!(store.commit(&recording).is_err(), "{:?}", recording.meta);
        }
        assert!(store.load_all().expect("load_all").is_empty());
        assert!(
            !store
                .root()
                .parent()
                .expect("parent")
                .join("outside")
                .exists()
        );
    }

    #[test]
    fn a_body_file_name_outside_the_recording_is_rejected_on_read() {
        let (_dir, store) = store();
        let path = store.commit(&sample(0)).expect("commit");
        let meta_path = path.join(META_FILE);
        let mut meta: serde_json::Value =
            serde_json::from_slice(&fs::read(&meta_path).expect("meta")).expect("json");
        meta["request_body_file"] = "../../secret".into();
        fs::write(&meta_path, serde_json::to_vec(&meta).expect("encode")).expect("write");

        assert!(matches!(load(&path), Err(RecordingError::Invalid { .. })));
    }
}
