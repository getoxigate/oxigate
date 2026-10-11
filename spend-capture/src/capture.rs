// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! The capture proxy.
//!
//! The gateway reaches the capture by configuration alone: a provider's base URL is pointed at
//! `http://<capture>/<route>`, and the capture forwards everything under that prefix to the
//! route's upstream origin over TLS. Each exchange is relayed as it arrives — response chunks
//! reach the gateway as the upstream sends them — and recorded once it ends.
//!
//! Forwarding is never altered: no header is added, no body re-encoded, no redirect followed.
//! Only transport headers ([`crate::recording::is_transport_header`]) are left to each hop's HTTP
//! stack. Redaction
//! and the credential scan apply to what is written, never to what is forwarded.

use std::collections::HashMap;
use std::convert::Infallible;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http::header::HeaderMap;
use http::{Request, Response, StatusCode, Uri};
use http_body::{Body, Frame, SizeHint};
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper_rustls::HttpsConnector;
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;
use rustls_pki_types::CertificateDer;
use rustls_pki_types::pem::PemObject;
use secrecy::{ExposeSecret, SecretString};
use tokio::net::TcpListener;
use tokio::sync::watch;

use crate::recording::{
    BUFFERED_RESPONSE_FILE, BodyKind, FORMAT_VERSION, Incident, REQUEST_BODY_FILE, Recording,
    RecordingError, RecordingMeta, RecordingStore, STREAM_RESPONSE_FILE, Surface, Termination,
    end_to_end_headers, split_route_target,
};
use crate::redact::{KnownCredentials, recorded_headers, redact_uri};
use crate::serve::Server;

/// Name of the harness's own log file inside the results directory.
pub const LOG_FILE: &str = "capture.log";

/// One capture route: requests under `/<name>` are forwarded to `upstream_origin`.
#[derive(Debug, Clone)]
pub struct Route {
    name: String,
    upstream_origin: String,
}

impl Route {
    /// A route named `name` forwarding to `upstream_origin`, which must be a bare `https://` or
    /// `http://` origin: scheme, host and optional port, with no userinfo, path or query.
    pub fn new(name: &str, upstream_origin: &str) -> Result<Self, CaptureError> {
        let invalid = |reason: &str| CaptureError::InvalidRoute {
            route: format!("{name}={upstream_origin}"),
            reason: reason.to_owned(),
        };
        let name_ok = !name.is_empty()
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'));
        if !name_ok {
            return Err(invalid(
                "a route name is letters, digits, `-`, `_` and `.` only",
            ));
        }
        let origin = upstream_origin.trim_end_matches('/');
        let uri: Uri = origin
            .parse()
            .map_err(|_| invalid("the upstream origin is not a URI"))?;
        let bare = matches!(uri.scheme_str(), Some("https" | "http"))
            && uri
                .authority()
                .is_some_and(|authority| !authority.as_str().contains('@'))
            && uri.path_and_query().is_none_or(|pq| pq.as_str() == "/");
        if !bare {
            return Err(invalid(
                "the upstream origin must be scheme://host[:port] with nothing after it",
            ));
        }
        Ok(Self {
            name: name.to_owned(),
            upstream_origin: origin.to_owned(),
        })
    }

    /// Parses `NAME=ORIGIN`, as given on the command line.
    pub fn parse(spec: &str) -> Result<Self, CaptureError> {
        let (name, origin) = spec
            .split_once('=')
            .ok_or_else(|| CaptureError::InvalidRoute {
                route: spec.to_owned(),
                reason: "expected NAME=ORIGIN".to_owned(),
            })?;
        Self::new(name, origin)
    }

    /// The route's name: the first path segment the gateway sends.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The upstream origin requests on this route are forwarded to.
    pub fn upstream_origin(&self) -> &str {
        &self.upstream_origin
    }
}

/// Everything the capture needs to start.
#[derive(Debug)]
pub struct CaptureConfig {
    /// Routes to serve; names must be distinct.
    pub routes: Vec<Route>,
    /// Results directory. Must not already hold recordings or incident records.
    pub results_dir: PathBuf,
    /// Credential values to scan bodies for, beyond those seen in each exchange's own headers
    /// and URI.
    pub declared_credentials: Vec<SecretString>,
    /// PEM certificates to trust for upstream TLS in addition to the bundled web roots.
    pub extra_root_certificates_pem: Option<Vec<u8>>,
}

/// Why the capture could not start.
#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    /// A route specification is malformed.
    #[error("invalid route `{route}`: {reason}")]
    InvalidRoute {
        /// The route as given.
        route: String,
        /// What is wrong with it.
        reason: String,
    },
    /// Two routes share a name.
    #[error("route `{0}` is configured twice")]
    DuplicateRoute(String),
    /// The results directory already holds a session's output.
    #[error(
        "results directory {} already holds recordings or incident records; use an empty one",
        .0.display()
    )]
    ResultsNotEmpty(PathBuf),
    /// The results directory could not be prepared.
    #[error(transparent)]
    Store(#[from] RecordingError),
    /// The log file could not be opened.
    #[error("log file {}: {source}", path.display())]
    Log {
        /// The log file.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The upstream TLS configuration could not be built.
    #[error("upstream TLS: {0}")]
    Tls(String),
    /// The listen address could not be bound.
    #[error("listen on {addr}: {source}")]
    Bind {
        /// The address.
        addr: SocketAddr,
        /// The underlying error.
        source: std::io::Error,
    },
}

/// A running capture.
pub struct CaptureServer {
    local_addr: SocketAddr,
    shared: Arc<Shared>,
    finished: watch::Receiver<u64>,
    server: Server,
}

impl CaptureServer {
    /// Prepares the results directory and starts serving on `addr`.
    pub async fn bind(addr: SocketAddr, config: CaptureConfig) -> Result<Self, CaptureError> {
        let mut routes = HashMap::new();
        for route in config.routes {
            if routes.contains_key(&route.name) {
                return Err(CaptureError::DuplicateRoute(route.name));
            }
            routes.insert(route.name.clone(), route);
        }
        let store = RecordingStore::open(&config.results_dir)?;
        if !store.is_empty()? {
            return Err(CaptureError::ResultsNotEmpty(config.results_dir));
        }
        let log = EventLog::open(config.results_dir.join(LOG_FILE))?;
        let client = upstream_client(config.extra_root_certificates_pem.as_deref())?;
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|source| CaptureError::Bind { addr, source })?;
        let local_addr = listener
            .local_addr()
            .map_err(|source| CaptureError::Bind { addr, source })?;

        let mut names: Vec<&Route> = routes.values().collect();
        names.sort_by(|a, b| a.name.cmp(&b.name));
        for route in names {
            log.line(&format!(
                "route {} -> {}",
                route.name, route.upstream_origin
            ));
        }
        log.line(&format!("listening on {local_addr}"));

        let (finished_tx, finished) = watch::channel(0);
        let shared = Arc::new(Shared {
            routes,
            client,
            store,
            log,
            declared: config.declared_credentials,
            started: AtomicU64::new(0),
            finished: finished_tx,
        });
        let server = Server::spawn(listener, Arc::clone(&shared), handle, |shared, err| {
            shared.log.line(&format!("accept failed: {err}"));
        });
        Ok(Self {
            local_addr,
            shared,
            finished,
            server,
        })
    }

    /// The address the capture is listening on.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Exchanges that have arrived and not yet finished.
    pub fn in_flight(&self) -> u64 {
        let finished = *self.finished.borrow();
        self.shared
            .started
            .load(Ordering::SeqCst)
            .saturating_sub(finished)
    }

    /// Waits until at least `count` exchanges have ended and been recorded, refused, or logged
    /// as failed.
    pub async fn wait_finished(&self, count: u64) {
        let mut finished = self.finished.clone();
        // The sender lives in `self.shared`, so it cannot close while `self` exists.
        let _ = finished.wait_for(|done| *done >= count).await;
    }

    /// Stops accepting connections and stops reading new requests on open ones — idle
    /// connections close at once, busy ones after their current exchange — then waits for every
    /// exchange that has arrived to finish and be written.
    pub async fn shutdown(mut self) {
        self.server.shutdown().await;
        let mut finished = self.finished.clone();
        let shared = Arc::clone(&self.shared);
        let _ = finished
            .wait_for(|done| *done >= shared.started.load(Ordering::SeqCst))
            .await;
    }
}

type UpstreamClient = Client<HttpsConnector<HttpConnector>, Full<Bytes>>;
type ProxyBody = BoxBody<Bytes, hyper::Error>;

/// State shared by every connection.
struct Shared {
    routes: HashMap<String, Route>,
    client: UpstreamClient,
    store: RecordingStore,
    log: EventLog,
    declared: Vec<SecretString>,
    started: AtomicU64,
    finished: watch::Sender<u64>,
}

impl Shared {
    fn mark_finished(&self) {
        self.finished.send_modify(|done| *done += 1);
    }

    /// Credential values known before an exchange is seen: the operator-declared ones.
    fn declared_credentials(&self) -> KnownCredentials {
        KnownCredentials::with_declared(
            self.declared
                .iter()
                .map(|value| value.expose_secret().as_bytes()),
        )
    }
}

/// Builds the upstream client: HTTP/1.1 over TLS with the bundled web roots, plus any extra
/// certificates, and no redirect handling (the legacy client never follows redirects).
fn upstream_client(extra_pem: Option<&[u8]>) -> Result<UpstreamClient, CaptureError> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    if let Some(pem) = extra_pem {
        let mut added = 0;
        for cert in CertificateDer::pem_slice_iter(pem) {
            let cert = cert.map_err(|err| CaptureError::Tls(err.to_string()))?;
            roots
                .add(cert)
                .map_err(|err| CaptureError::Tls(err.to_string()))?;
            added += 1;
        }
        if added == 0 {
            return Err(CaptureError::Tls(
                "the extra root certificate file holds no certificate".to_owned(),
            ));
        }
    }
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let tls = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|err| CaptureError::Tls(err.to_string()))?
        .with_root_certificates(roots)
        .with_no_client_auth();
    let connector = hyper_rustls::HttpsConnectorBuilder::new()
        .with_tls_config(tls)
        .https_or_http()
        .enable_http1()
        .build();
    Ok(Client::builder(TokioExecutor::new()).build(connector))
}

/// Accounts for one exchange: it is marked finished exactly once, when it is recorded, refused
/// or given up on. Dropped unconsumed — the client went away before its request body was read —
/// it logs that and marks the exchange finished. From the moment the body is read, the
/// [`Exchange`] owns the ticket and every ending goes through [`Exchange::end`].
struct Ticket {
    shared: Arc<Shared>,
    sequence: u64,
    done: bool,
}

impl Ticket {
    fn issue(shared: &Arc<Shared>) -> Self {
        Self {
            shared: Arc::clone(shared),
            sequence: shared.started.fetch_add(1, Ordering::SeqCst),
            done: false,
        }
    }

    /// Logs the exchange's outcome and marks it finished.
    fn finish(mut self, outcome: &str) {
        self.shared
            .log
            .line(&format!("seq={} {outcome}", self.sequence));
        self.done = true;
        self.shared.mark_finished();
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if !self.done {
            self.shared.log.line(&format!(
                "seq={} abandoned: the client went away before its request was read; not recorded",
                self.sequence
            ));
            self.shared.mark_finished();
        }
    }
}

async fn handle(
    shared: Arc<Shared>,
    request: Request<Incoming>,
) -> Result<Response<ProxyBody>, Infallible> {
    let ticket = Ticket::issue(&shared);
    let (parts, body) = request.into_parts();
    let mut known = shared.declared_credentials();
    for (name, value) in &parts.headers {
        known.collect_header(name.as_str(), value.as_bytes());
    }
    let target = parts
        .uri
        .path_and_query()
        .map_or("/", |pq| pq.as_str())
        .to_owned();
    known.collect_uri(&target);

    let Some((route, upstream_target)) =
        split_route_target(&target).and_then(|(name, rest)| Some((shared.routes.get(name)?, rest)))
    else {
        ticket.finish(&format!(
            "{} {} -> 404 no such route; not recorded",
            parts.method,
            redact_uri(&target, &known)
        ));
        return Ok(local_response(
            StatusCode::NOT_FOUND,
            "no such capture route",
        ));
    };
    let route = route.clone();
    let label = format!(
        "route={} {} {}",
        route.name,
        parts.method,
        redact_uri(upstream_target, &known)
    );
    let upstream_uri: Uri = match format!("{}{}", route.upstream_origin, upstream_target).parse() {
        Ok(uri) => uri,
        Err(_) => {
            ticket.finish(&format!(
                "{label} -> not a valid upstream URI; not recorded"
            ));
            return Ok(local_response(
                StatusCode::BAD_REQUEST,
                "invalid request target",
            ));
        }
    };
    // A body that cannot be read is never held whole, so there is nothing to scan or write.
    let request_body = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => {
            ticket.finish(&format!(
                "{label} -> request body unreadable ({}); not recorded",
                scrubbed(&known, &err.to_string()),
            ));
            return Ok(local_response(
                StatusCode::BAD_REQUEST,
                "request body unreadable",
            ));
        }
    };

    let mut upstream_request = Request::new(Full::new(request_body.clone()));
    *upstream_request.method_mut() = parts.method.clone();
    *upstream_request.uri_mut() = upstream_uri;
    *upstream_request.headers_mut() = end_to_end_headers(&parts.headers);
    let mut awaiting = AwaitingResponse(Some(Exchange {
        ticket,
        route,
        method: parts.method.to_string(),
        upstream_target: upstream_target.to_owned(),
        request_headers: parts.headers,
        request_body,
        status: None,
        response_headers: HeaderMap::new(),
        response_body: Vec::new(),
        response_chunks: Vec::new(),
    }));

    let result = shared.client.request(upstream_request).await;
    let Some(mut exchange) = awaiting.0.take() else {
        // `awaiting` is built holding the exchange and nothing else takes it.
        return Ok(local_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "exchange lost",
        ));
    };
    let response = match result {
        Ok(response) => response,
        Err(err) => {
            exchange.end(Outcome::Failed(format!(
                "upstream request failed ({})",
                error_chain(&err)
            )));
            return Ok(local_response(
                StatusCode::BAD_GATEWAY,
                "upstream request failed",
            ));
        }
    };

    let (response_parts, upstream_body) = response.into_parts();
    let mut downstream_headers = end_to_end_headers(&response_parts.headers);
    exchange.status = Some(response_parts.status.as_u16());
    exchange.response_headers = response_parts.headers;
    let relay = RecordedBody {
        inner: upstream_body,
        exchange: Some(exchange),
    };
    let mut downstream = Response::new(relay.boxed());
    *downstream.status_mut() = response_parts.status;
    std::mem::swap(downstream.headers_mut(), &mut downstream_headers);
    Ok(downstream)
}

/// Holds an exchange while its upstream request is in flight. Dropped still holding it — the
/// client went away and the handler with it — it ends the exchange, so the request body is still
/// scanned and an incident written if it carries a credential.
struct AwaitingResponse(Option<Exchange>);

impl Drop for AwaitingResponse {
    fn drop(&mut self) {
        if let Some(exchange) = self.0.take() {
            exchange.end(Outcome::Failed(
                "the client went away before the upstream responded".to_owned(),
            ));
        }
    }
}

fn local_response(status: StatusCode, message: &str) -> Response<ProxyBody> {
    let body = Full::new(Bytes::from(format!("spend-capture: {message}\n")))
        .map_err(|never| match never {})
        .boxed();
    let mut response = Response::new(body);
    *response.status_mut() = status;
    response
}

/// An error and its sources, joined.
fn error_chain(err: &dyn std::error::Error) -> String {
    let mut text = err.to_string();
    let mut source = err.source();
    while let Some(next) = source {
        text.push_str(": ");
        text.push_str(&next.to_string());
        source = next.source();
    }
    text
}

fn scrubbed(known: &KnownCredentials, text: &str) -> String {
    String::from_utf8_lossy(&known.scrub(text.as_bytes())).into_owned()
}

/// How an exchange ended.
enum Outcome {
    /// A response arrived and its body ended one way or the other.
    Ended(Termination),
    /// No response arrived, or its body failed mid-way. Unscrubbed; scrubbed before logging.
    Failed(String),
}

/// One exchange as it passes through: everything needed to record it once it ends.
struct Exchange {
    ticket: Ticket,
    route: Route,
    method: String,
    upstream_target: String,
    /// As received from the gateway, transport headers included: they are scanned for
    /// credentials, though only the forwarded ones are recorded.
    request_headers: HeaderMap,
    request_body: Bytes,
    /// `None` until the upstream responds.
    status: Option<u16>,
    /// As received from upstream, transport headers included, for the same reason.
    response_headers: HeaderMap,
    response_body: Vec<u8>,
    response_chunks: Vec<u64>,
}

impl Exchange {
    /// Every credential value this exchange carries, plus the declared ones.
    fn known_credentials(&self) -> KnownCredentials {
        let mut known = self.ticket.shared.declared_credentials();
        for (name, value) in self.request_headers.iter().chain(&self.response_headers) {
            known.collect_header(name.as_str(), value.as_bytes());
        }
        known.collect_uri(&self.upstream_target);
        known
    }

    fn response_kind(&self) -> BodyKind {
        let event_stream = self
            .response_headers
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                value
                    .trim_start()
                    .to_ascii_lowercase()
                    .starts_with("text/event-stream")
            });
        if event_stream {
            BodyKind::EventStream
        } else {
            BodyKind::Buffered
        }
    }

    /// Hands the ended exchange to a blocking thread to scan and write, so neither the
    /// filesystem nor the scan holds up an async worker. Runs inline outside a runtime.
    fn end(self, outcome: Outcome) {
        let work = move || self.write(outcome);
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => drop(runtime.spawn_blocking(work)),
            Err(_) => work(),
        }
    }

    /// Scans whatever of both bodies arrived, then writes a payload-free incident record when a
    /// credential value was found — however the exchange ended — or otherwise the redacted
    /// recording of an exchange that ended with a response. Nothing is written before the scan.
    fn write(self, outcome: Outcome) {
        let known = self.known_credentials();
        let kind = self.response_kind();
        let ending = match &outcome {
            Outcome::Ended(termination) => format!("{termination:?}"),
            Outcome::Failed(reason) => scrubbed(&known, reason),
        };
        let summary = format!(
            "route={} {} {} -> {} [{ending}]",
            self.route.name,
            self.method,
            redact_uri(&self.upstream_target, &known),
            self.status
                .map_or_else(|| "no response".to_owned(), |status| status.to_string()),
        );
        let store = &self.ticket.shared.store;

        let mut surfaces = Vec::new();
        if known.found_in(&self.request_body) {
            surfaces.push(Surface::RequestBody);
        }
        if known.found_in(&self.response_body) {
            surfaces.push(match kind {
                BodyKind::Buffered => Surface::ResponseBody,
                BodyKind::EventStream => Surface::ResponseStream,
            });
        }
        if !surfaces.is_empty() {
            let incident = Incident {
                format_version: FORMAT_VERSION,
                sequence: self.ticket.sequence,
                route: self.route.name.clone(),
                method: self.method.clone(),
                status: self.status,
                surfaces: surfaces.clone(),
            };
            let written = match store.commit_incident(&incident) {
                Ok(_) => "incident record written".to_owned(),
                Err(err) => format!(
                    "incident record FAILED: {}",
                    scrubbed(&known, &err.to_string())
                ),
            };
            self.ticket.finish(&format!(
                "{summary} refused: a credential value is in {surfaces:?}; {written}; not recorded"
            ));
            return;
        }

        // Only an exchange that ended with a response is recorded: a failed one could not be
        // replayed as it happened.
        let (Outcome::Ended(termination), Some(status)) = (outcome, self.status) else {
            self.ticket.finish(&format!("{summary} not recorded"));
            return;
        };
        let recording = Recording {
            meta: RecordingMeta {
                format_version: FORMAT_VERSION,
                sequence: self.ticket.sequence,
                route: self.route.name.clone(),
                upstream_origin: self.route.upstream_origin.clone(),
                method: self.method.clone(),
                uri: redact_uri(&self.upstream_target, &known),
                status,
                request_headers: recorded_headers(&self.request_headers, &known),
                response_headers: recorded_headers(&self.response_headers, &known),
                request_body_file: REQUEST_BODY_FILE.to_owned(),
                response_body_file: match kind {
                    BodyKind::Buffered => BUFFERED_RESPONSE_FILE,
                    BodyKind::EventStream => STREAM_RESPONSE_FILE,
                }
                .to_owned(),
                response_kind: kind,
                response_chunks: self.response_chunks,
                termination,
            },
            request_body: self.request_body.to_vec(),
            response_body: self.response_body,
        };
        let written = match store.commit(&recording) {
            Ok(_) => "recorded".to_owned(),
            Err(err) => format!("recording FAILED: {}", scrubbed(&known, &err.to_string())),
        };
        self.ticket.finish(&format!("{summary} {written}"));
    }
}

/// The upstream response body, relayed frame by frame and kept for the recording. Each data
/// frame is one entry of the chunk manifest. The exchange is written when the upstream body
/// ends, or otherwise when this body is dropped. Dropped before the upstream body ended — the
/// client stopped reading first — it drops the upstream body too, which closes the upstream
/// connection rather than draining it.
struct RecordedBody {
    inner: Incoming,
    exchange: Option<Exchange>,
}

impl Body for RecordedBody {
    type Data = Bytes;
    type Error = hyper::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, hyper::Error>>> {
        let polled = Pin::new(&mut self.inner).poll_frame(cx);
        match &polled {
            Poll::Ready(Some(Ok(frame))) => {
                if let (Some(data), Some(exchange)) = (frame.data_ref(), self.exchange.as_mut()) {
                    exchange.response_body.extend_from_slice(data);
                    exchange.response_chunks.push(data.len() as u64);
                }
            }
            Poll::Ready(Some(Err(err))) => {
                // What arrived has been relayed; the error reaches the client as a broken body.
                if let Some(exchange) = self.exchange.take() {
                    let received = exchange.response_body.len();
                    exchange.end(Outcome::Failed(format!(
                        "upstream body failed after {received} bytes ({})",
                        error_chain(err)
                    )));
                }
            }
            Poll::Ready(None) => {
                if let Some(exchange) = self.exchange.take() {
                    exchange.end(Outcome::Ended(Termination::UpstreamComplete));
                }
            }
            Poll::Pending => {}
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for RecordedBody {
    fn drop(&mut self) {
        if let Some(exchange) = self.exchange.take() {
            // A body of known length can be complete without ever yielding its end: the server
            // stops polling once the declared length is written.
            exchange.end(Outcome::Ended(if self.inner.is_end_stream() {
                Termination::UpstreamComplete
            } else {
                Termination::DownstreamClosedFirst
            }));
        }
    }
}

/// The harness's own log: one line per event, appended to [`LOG_FILE`] in the results
/// directory and echoed to standard error. Every line is built from redacted values only.
struct EventLog {
    file: Mutex<File>,
}

impl EventLog {
    fn open(path: PathBuf) -> Result<Self, CaptureError> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| CaptureError::Log { path, source })?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }

    fn line(&self, message: &str) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let line = format!("t={}.{:03} {message}", now.as_secs(), now.subsec_millis());
        eprintln!("{line}");
        // A poisoned lock only means another line panicked mid-write; keep logging.
        let mut file = self
            .file
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Logging must never take an exchange down; a failed write is reported on stderr.
        if let Err(err) = writeln!(file, "{line}") {
            eprintln!("spend-capture: log write failed: {err}");
        }
    }
}
