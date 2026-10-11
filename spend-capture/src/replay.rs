// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! The replay server.
//!
//! Serves a capture session's recordings back to the gateway, in the order they were captured.
//! The gateway is pointed at `http://<replay>/<route>` exactly as it was pointed at the capture.
//!
//! Each request must match the next recording: same route, method, redacted path and query,
//! end-to-end headers after redaction, and body bytes. Transport headers are ignored, and
//! credential values never cause a mismatch: the request is redacted as the capture redacted it,
//! and each redaction marker in the recording stands for any value. The capture may have redacted
//! a value the replay cannot recognise from the request alone — one the response carried, or one
//! declared only to the capture — so the marker, not the replay's own redaction, decides. Every
//! byte outside a marker must be equal, and in the URI a marker's value never spans a path
//! segment or query parameter. Values declared to the replay are only scrubbed from what it says,
//! never used to normalize a request. A request that does not match — or arrives after every
//! recording has been served — is refused with an explicit error naming the field, and the
//! session stops: every later request is refused too, so a diverged replay can never quietly
//! serve a recording to the wrong request. The error, in the response and on standard error, has
//! every credential the request is known to carry scrubbed from it.
//!
//! A response is served as it was recorded: status, end-to-end headers, and the body as the
//! sequence of chunks the capture observed. A recording whose downstream closed first holds only
//! a prefix; after that prefix the body fails, so a client reading past it gets an error rather
//! than a clean end of stream.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use bytes::Bytes;
use http::header::{HeaderMap, HeaderName, HeaderValue};
use http::{Request, Response, StatusCode};
use http_body::{Body, Frame, SizeHint};
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use secrecy::{ExposeSecret, SecretString};
use tokio::net::TcpListener;

use crate::recording::{
    BodyKind, HeaderField, Recording, RecordingError, RecordingStore, Termination,
    split_route_target,
};
use crate::redact::{
    KnownCredentials, REDACTED, URI_DELIMITERS, matches_redacted, recorded_headers, redact_uri,
};
use crate::serve::Server;

/// Status of the response to a request the replay refuses.
pub const MISMATCH_STATUS: StatusCode = StatusCode::CONFLICT;

/// Everything the replay needs to start.
#[derive(Debug)]
pub struct ReplayConfig {
    /// Results directory of a capture session. It is only read.
    pub results_dir: PathBuf,
    /// Credential values to redact from each request, and from anything said about it, beyond
    /// those in its own headers and URI — the ones declared to the capture.
    pub declared_credentials: Vec<SecretString>,
}

/// Why the replay could not start.
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    /// The recordings could not be read.
    #[error(transparent)]
    Store(#[from] RecordingError),
    /// The results directory holds no recording.
    #[error("results directory {} holds no recordings", .0.display())]
    Empty(PathBuf),
    /// A recording cannot be served as an HTTP response.
    #[error("recording {sequence} cannot be served: {reason}")]
    Unservable {
        /// The recording's sequence number.
        sequence: u64,
        /// What is wrong with it.
        reason: String,
    },
    /// The listen address could not be bound.
    #[error("listen on {addr}: {source}")]
    Bind {
        /// The address.
        addr: SocketAddr,
        /// The underlying error.
        source: std::io::Error,
    },
}

/// The part of a request that differed from the next recording.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MismatchField {
    /// The request matches a later recording, not the next one.
    Sequence,
    /// The capture route: the first path segment.
    Route,
    /// The request method.
    Method,
    /// The path and query after the route, after redaction.
    Uri,
    /// One end-to-end request header, by lower-case name: its values after redaction differ, or
    /// it is present on one side only.
    Header(String),
    /// The request body bytes.
    Body,
    /// The exchange due at this position arrived at the capture but was not recorded.
    NotRecorded,
    /// Every recording has already been served.
    Unexpected,
}

impl std::fmt::Display for MismatchField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sequence => f.write_str("sequence"),
            Self::Route => f.write_str("route"),
            Self::Method => f.write_str("method"),
            Self::Uri => f.write_str("uri"),
            Self::Header(name) => write!(f, "header `{name}`"),
            Self::Body => f.write_str("body"),
            Self::NotRecorded => f.write_str("not recorded"),
            Self::Unexpected => f.write_str("unexpected request"),
        }
    }
}

/// A request the replay refused. Its detail quotes only recorded values and redacted request
/// values — never a request value where the recording holds a redaction — and never body bytes,
/// and has every credential value known for the request scrubbed from it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("request {request} (capture position {}): {field}: {detail}",
    .expected.map_or_else(|| "none; all served".to_owned(), |sequence| sequence.to_string()))]
pub struct Mismatch {
    /// Position of the refused request in the replay session, starting at 0.
    pub request: u64,
    /// Capture sequence position that was due: the next recording's sequence number, or the
    /// position of an exchange that was not recorded. `None` when every recording had been served.
    pub expected: Option<u64>,
    /// What differed.
    pub field: MismatchField,
    /// The two sides of the difference.
    pub detail: String,
}

/// A running replay.
pub struct ReplayServer {
    local_addr: SocketAddr,
    shared: Arc<Shared>,
    server: Server,
}

impl ReplayServer {
    /// Loads every recording in the results directory and starts serving them on `addr`.
    pub async fn bind(addr: SocketAddr, config: ReplayConfig) -> Result<Self, ReplayError> {
        let recordings = RecordingStore::existing(&config.results_dir)?
            .load_all()?
            .into_iter()
            .map(Prepared::new)
            .collect::<Result<Vec<_>, _>>()?;
        if recordings.is_empty() {
            return Err(ReplayError::Empty(config.results_dir));
        }
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|source| ReplayError::Bind { addr, source })?;
        let local_addr = listener
            .local_addr()
            .map_err(|source| ReplayError::Bind { addr, source })?;
        log(&format!(
            "serving {} recording(s) from {} on {local_addr}",
            recordings.len(),
            config.results_dir.display()
        ));
        let shared = Arc::new(Shared {
            declared: config.declared_credentials,
            session: Mutex::new(Session {
                recordings,
                next: 0,
                requests: 0,
                mismatch: None,
            }),
        });
        let server = Server::spawn(listener, Arc::clone(&shared), handle, |_, err| {
            log(&format!("accept failed: {err}"));
        });
        Ok(Self {
            local_addr,
            shared,
            server,
        })
    }

    /// The address the replay is listening on.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Recordings served so far.
    pub fn served(&self) -> usize {
        self.shared.session().next
    }

    /// Recordings not yet served.
    pub fn remaining(&self) -> usize {
        self.shared.session().remaining()
    }

    /// The request that stopped the session, if one did.
    pub fn mismatch(&self) -> Option<Mismatch> {
        self.shared.session().mismatch.clone()
    }

    /// Stops accepting connections, waits for the open ones to close, and returns the state the
    /// session ended in — taken only once no request can still be served or refused.
    pub async fn shutdown(mut self) -> SessionEnd {
        self.server.shutdown().await;
        let session = self.shared.session();
        SessionEnd {
            mismatch: session.mismatch.clone(),
            remaining: session.remaining(),
        }
    }
}

/// The state a replay session ended in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionEnd {
    /// The request that stopped the session, if one did.
    pub mismatch: Option<Mismatch>,
    /// Recordings never served.
    pub remaining: usize,
}

struct Shared {
    declared: Vec<SecretString>,
    session: Mutex<Session>,
}

impl Shared {
    fn session(&self) -> std::sync::MutexGuard<'_, Session> {
        // A poisoned lock only means a request panicked mid-match; the state is still whole.
        self.session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// A recording ready to serve, its response head built at start so serving cannot fail.
struct Prepared {
    recording: Recording,
    status: StatusCode,
    headers: HeaderMap,
}

impl Prepared {
    fn new(recording: Recording) -> Result<Self, ReplayError> {
        let sequence = recording.meta.sequence;
        let unservable = |reason: String| ReplayError::Unservable { sequence, reason };
        let status = StatusCode::from_u16(recording.meta.status)
            .map_err(|_| unservable(format!("status {} is not valid", recording.meta.status)))?;
        let mut headers = HeaderMap::with_capacity(recording.meta.response_headers.len());
        for field in &recording.meta.response_headers {
            let name = HeaderName::from_bytes(field.name.as_bytes())
                .map_err(|_| unservable(format!("header name `{}` is not valid", field.name)))?;
            let value = HeaderValue::from_bytes(field.value.as_bytes())
                .map_err(|_| unservable(format!("header `{}` has an invalid value", field.name)))?;
            headers.append(name, value);
        }
        Ok(Self {
            recording,
            status,
            headers,
        })
    }

    fn response(&self) -> Response<ReplayBody> {
        let mut response = Response::new(RecordedChunks::new(&self.recording).boxed());
        *response.status_mut() = self.status;
        *response.headers_mut() = self.headers.clone();
        response
    }
}

struct Session {
    recordings: Vec<Prepared>,
    /// Index of the next recording to serve.
    next: usize,
    /// Requests received so far.
    requests: u64,
    /// Set by the first refused request; from then on every request is refused.
    mismatch: Option<Mismatch>,
}

impl Session {
    fn remaining(&self) -> usize {
        self.recordings.len() - self.next
    }

    /// Serves the next recording if `observed` matches it, and refuses it otherwise.
    fn serve(&mut self, observed: &Observed) -> Response<ReplayBody> {
        let request = self.requests;
        self.requests += 1;
        if let Some(earlier) = &self.mismatch {
            let message = format!("the session stopped at an earlier mismatch: {earlier}");
            log(&format!("request {request} refused: {message}"));
            return mismatch_response(&message);
        }
        let Some(expected) = self.recordings.get(self.next) else {
            let detail = format!("{} {}", observed.method, observed.target());
            return self.refuse(
                observed,
                Mismatch {
                    request,
                    expected: None,
                    field: MismatchField::Unexpected,
                    detail,
                },
            );
        };
        let expected_sequence = expected.recording.meta.sequence;
        // Every exchange that reached the capture took the next sequence number, so a number
        // missing here is an exchange the capture forwarded but did not record. Serving the next
        // recording in its place would answer this request with another exchange's response.
        let due = self
            .next
            .checked_sub(1)
            .map_or(0, |last| self.recordings[last].recording.meta.sequence + 1);
        if expected_sequence != due {
            return self.refuse(
                observed,
                Mismatch {
                    request,
                    expected: Some(due),
                    field: MismatchField::NotRecorded,
                    detail: format!(
                        "exchange {due} reached the capture but was not recorded (see the capture \
                     log and incidents), so it cannot be replayed"
                    ),
                },
            );
        }
        if let Some((field, detail)) = difference(&expected.recording, observed) {
            let later = self.recordings[self.next + 1..]
                .iter()
                .find(|later| difference(&later.recording, observed).is_none());
            let (field, detail) = match later {
                Some(later) => (
                    MismatchField::Sequence,
                    format!(
                        "matches recording {}, but recording {expected_sequence} is next",
                        later.recording.meta.sequence
                    ),
                ),
                None => (field, detail),
            };
            return self.refuse(
                observed,
                Mismatch {
                    request,
                    expected: Some(expected_sequence),
                    field,
                    detail,
                },
            );
        }
        self.next += 1;
        // Described by the recording, which holds the request as the capture redacted it: with
        // values redacted that the request alone does not mark as credentials. Then scrubbed,
        // for values declared only to the replay, which the recording holds as sent.
        let meta = &expected.recording.meta;
        log(&observed.scrub(&format!(
            "request {request}: recording {expected_sequence} served ({} /{}{}, {:?})",
            meta.method, meta.route, meta.uri, meta.termination
        )));
        expected.response()
    }

    /// Stops the session at `mismatch`, its text scrubbed of the request's known credentials:
    /// the route, method and header names are compared and quoted as sent.
    fn refuse(&mut self, observed: &Observed, mut mismatch: Mismatch) -> Response<ReplayBody> {
        mismatch.detail = observed.scrub(&mismatch.detail);
        if let MismatchField::Header(name) = &mut mismatch.field {
            *name = observed.scrub(name);
        }
        log(&format!("refused, and the session stops: {mismatch}"));
        let response = mismatch_response(&mismatch.to_string());
        self.mismatch = Some(mismatch);
        response
    }
}

/// A request in the form a recording holds it: route split off, URI and headers redacted.
struct Observed {
    route: String,
    method: String,
    uri: String,
    headers: Vec<HeaderField>,
    body: Bytes,
    /// The credential values the request carries, plus the declared ones: scrubbed from
    /// anything said about the request, never used to normalize it.
    known: KnownCredentials,
}

impl Observed {
    fn new(parts: &http::request::Parts, body: Bytes, declared: &[SecretString]) -> Self {
        let target = parts.uri.path_and_query().map_or("/", |pq| pq.as_str());
        // Normalized with its own carriers only. A declared value redacted here would turn into
        // a marker where the recording may hold the value as sent; where the capture did redact
        // it, the recorded marker already matches it.
        let collect = |mut known: KnownCredentials| {
            for (name, value) in &parts.headers {
                known.collect_header(name.as_str(), value.as_bytes());
            }
            known.collect_uri(target);
            known
        };
        let carried = collect(KnownCredentials::default());
        let known = collect(KnownCredentials::with_declared(
            declared
                .iter()
                .map(|value| value.expose_secret().as_bytes()),
        ));
        let (route, rest) = split_route_target(target).unwrap_or(("", target));
        Self {
            route: route.to_owned(),
            method: parts.method.to_string(),
            uri: redact_uri(rest, &carried),
            headers: recorded_headers(&parts.headers, &carried),
            body,
            known,
        }
    }

    /// The request target, route included, its URI redacted. The route is as sent: pass any
    /// text quoting it through [`Self::scrub`].
    fn target(&self) -> String {
        format!("/{}{}", self.route, self.uri)
    }

    /// `text` with every credential value known for this request replaced by [`REDACTED`].
    fn scrub(&self, text: &str) -> String {
        String::from_utf8_lossy(&self.known.scrub(text.as_bytes())).into_owned()
    }
}

/// The first field in which `observed` differs from `recording`, with both sides of it — the
/// request's side only where the recorded side holds no redaction (see [`holds_redaction`]).
fn difference(recording: &Recording, observed: &Observed) -> Option<(MismatchField, String)> {
    let meta = &recording.meta;
    let differs = |field, expected: &str, got: &str| {
        // Route and method never hold a marker; for the URI the delimiters keep its structure.
        (!matches_redacted(expected.as_bytes(), got.as_bytes(), URI_DELIMITERS)).then(|| {
            let got = if holds_redaction(expected.as_bytes()) {
                "a different value".to_owned()
            } else {
                format!("`{got}`")
            };
            (field, format!("expected `{expected}`, got {got}"))
        })
    };
    differs(MismatchField::Route, &meta.route, &observed.route)
        .or_else(|| differs(MismatchField::Method, &meta.method, &observed.method))
        .or_else(|| differs(MismatchField::Uri, &meta.uri, &observed.uri))
        .or_else(|| header_difference(&meta.request_headers, &observed.headers))
        .or_else(|| body_difference(&recording.request_body, &observed.body))
}

/// The first header, by name, whose values differ between the two sides — order among values
/// of one name counts, order among names does not. A recorded value matches an observed one as
/// [`matches_redacted`] does.
fn header_difference(
    expected: &[HeaderField],
    observed: &[HeaderField],
) -> Option<(MismatchField, String)> {
    fn by_name(fields: &[HeaderField]) -> BTreeMap<&str, Vec<&[u8]>> {
        let mut map: BTreeMap<&str, Vec<&[u8]>> = BTreeMap::new();
        for field in fields {
            map.entry(field.name.as_str())
                .or_default()
                .push(field.value.as_bytes());
        }
        map
    }
    fn show(values: Option<&Vec<&[u8]>>) -> String {
        values.map_or_else(
            || "no such header".to_owned(),
            |values| {
                values
                    .iter()
                    .map(|value| format!("`{}`", String::from_utf8_lossy(value)))
                    .collect::<Vec<_>>()
                    .join(", ")
            },
        )
    }
    fn same(want: Option<&Vec<&[u8]>>, got: Option<&Vec<&[u8]>>) -> bool {
        match (want, got) {
            (Some(want), Some(got)) => {
                want.len() == got.len()
                    && want
                        .iter()
                        .zip(got)
                        .all(|(want, got)| matches_redacted(want, got, b""))
            }
            (None, None) => true,
            _ => false,
        }
    }
    let (expected, observed) = (by_name(expected), by_name(observed));
    let names: BTreeSet<&str> = expected.keys().chain(observed.keys()).copied().collect();
    names.into_iter().find_map(|name| {
        let (want, got) = (expected.get(name), observed.get(name));
        (!same(want, got)).then(|| {
            let redacted = want.is_some_and(|values| values.iter().any(|v| holds_redaction(v)));
            let got = match got {
                Some(values) if redacted => format!("{} different value(s)", values.len()),
                _ => show(got),
            };
            (
                MismatchField::Header(name.to_owned()),
                format!("expected {}, got {got}", show(want)),
            )
        })
    })
}

/// Whether a recorded value had something redacted from it. The request's differing value is
/// then never quoted: the difference may be a credential the replay was not told about.
fn holds_redaction(recorded: &[u8]) -> bool {
    recorded
        .windows(REDACTED.len())
        .any(|window| window == REDACTED.as_bytes())
}

/// Where two bodies differ, by length and offset only: a body is never quoted.
fn body_difference(expected: &[u8], observed: &[u8]) -> Option<(MismatchField, String)> {
    (expected != observed).then(|| {
        let at = expected
            .iter()
            .zip(observed)
            .position(|(a, b)| a != b)
            .unwrap_or(expected.len().min(observed.len()));
        (
            MismatchField::Body,
            format!(
                "expected {} bytes, got {}; first difference at byte {at}",
                expected.len(),
                observed.len()
            ),
        )
    })
}

type ReplayBody = BoxBody<Bytes, IncompleteRecording>;

async fn handle(
    shared: Arc<Shared>,
    request: Request<Incoming>,
) -> Result<Response<ReplayBody>, Infallible> {
    let (parts, body) = request.into_parts();
    let body = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => {
            log(&format!(
                "a request body could not be read ({err}); not matched"
            ));
            return Ok(local_response(
                StatusCode::BAD_REQUEST,
                "request body unreadable",
            ));
        }
    };
    let observed = Observed::new(&parts, body, &shared.declared);
    Ok(shared.session().serve(&observed))
}

fn mismatch_response(message: &str) -> Response<ReplayBody> {
    local_response(MISMATCH_STATUS, &format!("request refused: {message}"))
}

fn local_response(status: StatusCode, message: &str) -> Response<ReplayBody> {
    let body = Full::new(Bytes::from(format!("spend-capture replay: {message}\n")))
        .map_err(|never| match never {})
        .boxed();
    let mut response = Response::new(body);
    *response.status_mut() = status;
    response
}

/// How a body served from a recording whose downstream closed first ends: the capture saw only
/// the prefix the gateway read, so there is nothing after it to serve, and no clean end either.
#[derive(Debug, thiserror::Error)]
#[error("recording {sequence} holds only the prefix read at capture; the response never completed")]
struct IncompleteRecording {
    sequence: u64,
}

/// A recorded response body, served as the chunks the capture observed: one frame per chunk
/// manifest entry, each handed to the connection to write before the next is produced. A body
/// whose downstream closed first fails after its last chunk instead of ending.
struct RecordedChunks {
    chunks: VecDeque<Bytes>,
    /// Bytes left to serve, for a complete buffered body: its length is then declared up front.
    exact_remaining: Option<u64>,
    /// Set for a recording whose downstream closed first, until its failure is produced.
    incomplete: Option<u64>,
    /// A frame was just produced; the next poll yields once so the connection writes it first.
    yielded: bool,
}

impl RecordedChunks {
    fn new(recording: &Recording) -> Self {
        let meta = &recording.meta;
        let mut body = Bytes::from(recording.response_body.clone());
        // `load` checked that the manifest sums to the body length.
        let chunks = meta
            .response_chunks
            .iter()
            .map(|&size| body.split_to(usize::try_from(size).unwrap_or(usize::MAX).min(body.len())))
            .collect();
        let complete = meta.termination == Termination::UpstreamComplete;
        Self {
            chunks,
            exact_remaining: (complete && meta.response_kind == BodyKind::Buffered)
                .then_some(recording.response_body.len() as u64),
            incomplete: (!complete).then_some(meta.sequence),
            yielded: false,
        }
    }
}

impl Body for RecordedChunks {
    type Data = Bytes;
    type Error = IncompleteRecording;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, IncompleteRecording>>> {
        let this = &mut *self;
        if std::mem::take(&mut this.yielded) {
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        if let Some(chunk) = this.chunks.pop_front() {
            this.yielded = true;
            if let Some(remaining) = &mut this.exact_remaining {
                *remaining -= chunk.len() as u64;
            }
            return Poll::Ready(Some(Ok(Frame::data(chunk))));
        }
        match this.incomplete.take() {
            Some(sequence) => {
                log(&format!(
                    "recording {sequence}: the client read past the captured prefix; failing the body"
                ));
                Poll::Ready(Some(Err(IncompleteRecording { sequence })))
            }
            None => Poll::Ready(None),
        }
    }

    fn size_hint(&self) -> SizeHint {
        self.exact_remaining
            .map_or_else(SizeHint::default, SizeHint::with_exact)
    }
}

/// Writes one line to standard error. Every line is built from redacted values only.
fn log(message: &str) {
    eprintln!("spend-capture replay: {message}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::{
        BUFFERED_RESPONSE_FILE, FORMAT_VERSION, REQUEST_BODY_FILE, RecordingMeta,
        STREAM_RESPONSE_FILE,
    };

    fn recording(chunks: &[&[u8]], kind: BodyKind, termination: Termination) -> Recording {
        Recording {
            meta: RecordingMeta {
                format_version: FORMAT_VERSION,
                sequence: 7,
                route: "openai".into(),
                upstream_origin: "https://api.example.com".into(),
                method: "POST".into(),
                uri: "/v1/chat/completions".into(),
                status: 200,
                request_headers: vec![],
                response_headers: vec![],
                request_body_file: REQUEST_BODY_FILE.into(),
                response_body_file: match kind {
                    BodyKind::Buffered => BUFFERED_RESPONSE_FILE,
                    BodyKind::EventStream => STREAM_RESPONSE_FILE,
                }
                .into(),
                response_kind: kind,
                response_chunks: chunks.iter().map(|chunk| chunk.len() as u64).collect(),
                termination,
            },
            request_body: vec![],
            response_body: chunks.concat(),
        }
    }

    /// Every frame the body produces, and how it ended.
    async fn frames(mut body: RecordedChunks) -> (Vec<Vec<u8>>, Result<(), IncompleteRecording>) {
        let mut out = Vec::new();
        loop {
            match body.frame().await {
                Some(Ok(frame)) => out.push(frame.into_data().expect("data frame").to_vec()),
                Some(Err(err)) => return (out, Err(err)),
                None => return (out, Ok(())),
            }
        }
    }

    #[tokio::test]
    async fn a_stream_is_served_in_its_recorded_chunks() {
        // A JSON line split across two chunks, then two events coalesced into one.
        let split: [&[u8]; 3] = [
            b"data: {\"choices\":[{\"delta\":{\"cont",
            b"ent\":\"hi\"}}]}\n\n",
            b"data: {\"usage\":{\"total_tokens\":5}}\n\ndata: [DONE]\n\n",
        ];
        let body = RecordedChunks::new(&recording(
            &split,
            BodyKind::EventStream,
            Termination::UpstreamComplete,
        ));
        let (served, ended) = frames(body).await;
        assert_eq!(served, split);
        assert!(ended.is_ok());
    }

    #[tokio::test]
    async fn a_body_whose_downstream_closed_first_fails_after_its_prefix() {
        let prefix: [&[u8]; 2] = [b"data: {\"a\":1}\n\n", b"data: [DONE]\n\n"];
        let body = RecordedChunks::new(&recording(
            &prefix,
            BodyKind::EventStream,
            Termination::DownstreamClosedFirst,
        ));
        let (served, ended) = frames(body).await;
        assert_eq!(served, prefix);
        assert!(matches!(ended, Err(IncompleteRecording { sequence: 7 })));
    }

    #[test]
    fn only_a_complete_buffered_body_declares_its_length() {
        let declared = |kind, termination| {
            RecordedChunks::new(&recording(&[b"{\"a\":", b"1}"], kind, termination))
                .size_hint()
                .exact()
        };
        assert_eq!(
            declared(BodyKind::Buffered, Termination::UpstreamComplete),
            Some(7)
        );
        assert_eq!(
            declared(BodyKind::EventStream, Termination::UpstreamComplete),
            None
        );
        assert_eq!(
            declared(BodyKind::Buffered, Termination::DownstreamClosedFirst),
            None
        );
    }
}
