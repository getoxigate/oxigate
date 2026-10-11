// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Shared fixtures: a scripted TLS upstream on a raw listener, and helpers around the harness.
//!
//! The upstream is a raw `TcpListener` behind TLS rather than a mock server, so a test controls
//! exactly which bytes are written when — the only way to pin chunk boundaries.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use spend_capture::capture::{CaptureConfig, CaptureServer, Route};
use spend_capture::replay::{ReplayConfig, ReplayServer};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::TlsAcceptor;
use tokio_rustls::rustls::ServerConfig;
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio_rustls::server::TlsStream;

/// A TLS listener on 127.0.0.1 with a freshly generated self-signed certificate.
pub struct TlsUpstream {
    listener: TcpListener,
    acceptor: TlsAcceptor,
    /// `https://127.0.0.1:<port>`.
    pub origin: String,
    /// The certificate, PEM-encoded, for the harness to trust.
    pub ca_pem: Vec<u8>,
}

impl TlsUpstream {
    /// Binds the listener and generates its certificate.
    pub async fn start() -> Self {
        let certified =
            rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_owned()]).expect("cert");
        let cert = CertificateDer::from(certified.cert.der().to_vec());
        let key = PrivateKeyDer::from(PrivatePkcs8KeyDer::from(certified.key_pair.serialize_der()));
        let provider = Arc::new(tokio_rustls::rustls::crypto::ring::default_provider());
        let config = ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .expect("protocol versions")
            .with_no_client_auth()
            .with_single_cert(vec![cert], key)
            .expect("server config");
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let origin = format!(
            "https://127.0.0.1:{}",
            listener.local_addr().expect("addr").port()
        );
        Self {
            listener,
            acceptor: TlsAcceptor::from(Arc::new(config)),
            origin,
            ca_pem: certified.cert.pem().into_bytes(),
        }
    }

    /// Accepts the next connection and completes its TLS handshake.
    pub async fn accept(&self) -> UpstreamConn {
        let (tcp, _) = self.listener.accept().await.expect("accept");
        let stream = self.acceptor.accept(tcp).await.expect("tls handshake");
        UpstreamConn {
            stream,
            buf: Vec::new(),
        }
    }
}

/// One HTTP/1.1 request as the upstream received it.
#[derive(Debug, Clone)]
pub struct RawRequest {
    /// Request line and header lines, as sent.
    pub head: String,
    /// Body bytes, as sent.
    pub body: Vec<u8>,
}

impl RawRequest {
    /// The request target of the request line.
    pub fn target(&self) -> &str {
        self.head.split(' ').nth(1).expect("request target")
    }

    /// Every value of header `name`, compared case-insensitively.
    pub fn header(&self, name: &str) -> Vec<&str> {
        self.head
            .lines()
            .skip(1)
            .filter_map(|line| line.split_once(':'))
            .filter(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.trim())
            .collect()
    }
}

/// The upstream side of one accepted connection.
pub struct UpstreamConn {
    stream: TlsStream<TcpStream>,
    buf: Vec<u8>,
}

impl UpstreamConn {
    /// Reads one request with a `Content-Length` body (or none).
    pub async fn read_request(&mut self) -> RawRequest {
        let head_end = loop {
            if let Some(at) = self.buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break at + 4;
            }
            self.fill().await;
        };
        let head = String::from_utf8(self.buf[..head_end].to_vec()).expect("utf-8 head");
        let length = head
            .lines()
            .filter_map(|line| line.split_once(':'))
            .find(|(n, _)| n.eq_ignore_ascii_case("content-length"))
            .map_or(0, |(_, v)| v.trim().parse::<usize>().expect("length"));
        while self.buf.len() < head_end + length {
            self.fill().await;
        }
        let body = self.buf[head_end..head_end + length].to_vec();
        self.buf.drain(..head_end + length);
        RawRequest { head, body }
    }

    async fn fill(&mut self) {
        let mut chunk = [0u8; 4096];
        let n = self.stream.read(&mut chunk).await.expect("read");
        assert!(n > 0, "connection closed mid-request");
        self.buf.extend_from_slice(&chunk[..n]);
    }

    /// Writes `bytes` in one write and flushes.
    pub async fn write(&mut self, bytes: &[u8]) {
        self.stream.write_all(bytes).await.expect("write");
        self.stream.flush().await.expect("flush");
    }

    /// Writes a buffered response with a `Content-Length`.
    pub async fn respond(&mut self, status: u16, headers: &[(&str, &str)], body: &[u8]) {
        let mut out = format!("HTTP/1.1 {status} X\r\ncontent-length: {}\r\n", body.len());
        for (name, value) in headers {
            out.push_str(&format!("{name}: {value}\r\n"));
        }
        out.push_str("\r\n");
        let mut bytes = out.into_bytes();
        bytes.extend_from_slice(body);
        self.write(&bytes).await;
    }

    /// Writes the head of a chunked `text/event-stream` response.
    pub async fn start_stream(&mut self, headers: &[(&str, &str)]) {
        let mut out = String::from(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n",
        );
        for (name, value) in headers {
            out.push_str(&format!("{name}: {value}\r\n"));
        }
        out.push_str("\r\n");
        self.write(out.as_bytes()).await;
    }

    /// Writes `data` as one HTTP chunk, in one write.
    pub async fn chunk(&mut self, data: &[u8]) {
        let mut bytes = format!("{:x}\r\n", data.len()).into_bytes();
        bytes.extend_from_slice(data);
        bytes.extend_from_slice(b"\r\n");
        self.write(&bytes).await;
    }

    /// Writes the terminating chunk.
    pub async fn end_stream(&mut self) {
        self.write(b"0\r\n\r\n").await;
    }

    /// Waits until the peer closes the connection, without sending anything.
    pub async fn wait_closed(&mut self) {
        let mut chunk = [0u8; 4096];
        loop {
            match self.stream.read(&mut chunk).await {
                Ok(0) | Err(_) => return,
                Ok(_) => {}
            }
        }
    }
}

/// Starts the harness on an ephemeral port with `routes`, trusting the upstream's certificate.
pub async fn start_capture(
    results: &Path,
    routes: &[(&str, &str)],
    ca_pem: &[u8],
    declared: &[&str],
) -> CaptureServer {
    let config = CaptureConfig {
        routes: routes
            .iter()
            .map(|(name, origin)| Route::new(name, origin).expect("route"))
            .collect(),
        results_dir: results.to_path_buf(),
        declared_credentials: declared
            .iter()
            .map(|value| secrecy::SecretString::new((*value).to_owned()))
            .collect(),
        extra_root_certificates_pem: Some(ca_pem.to_vec()),
    };
    CaptureServer::bind("127.0.0.1:0".parse().expect("addr"), config)
        .await
        .expect("capture starts")
}

/// Starts a replay of `results` on an ephemeral port, with `declared` credentials.
pub async fn start_replay(results: &Path, declared: &[&str]) -> ReplayServer {
    ReplayServer::bind(
        "127.0.0.1:0".parse().expect("addr"),
        ReplayConfig {
            results_dir: results.to_path_buf(),
            declared_credentials: declared
                .iter()
                .map(|value| secrecy::SecretString::new((*value).to_owned()))
                .collect(),
        },
    )
    .await
    .expect("replay starts")
}

/// A client that talks to the harness directly, never through an environment proxy, and never
/// follows a redirect.
pub fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("client")
}

/// Every file under `dir`, recursively.
pub fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next).expect("read_dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files
}

/// Files under `dir` whose bytes contain `needle`.
pub fn files_containing(dir: &Path, needle: &[u8]) -> Vec<PathBuf> {
    files_under(dir)
        .into_iter()
        .filter(|path| {
            let bytes = std::fs::read(path).expect("read");
            bytes.windows(needle.len()).any(|w| w == needle)
        })
        .collect()
}
