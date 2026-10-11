// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! The `spend-capture replay` command, run as a process: what it writes to standard error and how
//! it exits once interrupted.

#![cfg(unix)]

mod support;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use support::{TlsUpstream, client, start_capture};

const KEY: &str = "sk-capture-0001";
const BODY: &[u8] = br#"{"model":"claude","max_tokens":8}"#;

/// Captures one Anthropic-shaped exchange into `results`.
async fn capture_one_exchange(results: &Path) {
    capture_exchange(results, "/anthropic/v1/messages", &[]).await;
}

/// Captures one exchange sent to `target`, answered with `response_headers` besides the usual.
async fn capture_exchange(results: &Path, target: &str, response_headers: &[(&str, &str)]) {
    let upstream = TlsUpstream::start().await;
    let capture = start_capture(
        results,
        &[("anthropic", &upstream.origin)],
        &upstream.ca_pem,
        &[],
    )
    .await;
    let mut headers = vec![
        ("content-type".to_owned(), "application/json".to_owned()),
        ("connection".to_owned(), "close".to_owned()),
    ];
    headers.extend(
        response_headers
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned())),
    );
    let upstream_task = tokio::spawn(async move {
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        let headers: Vec<(&str, &str)> = headers
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        conn.respond(200, &headers, b"{}").await;
    });
    let response = send(capture.local_addr(), target, KEY).await;
    assert_eq!(response.status(), 200);
    upstream_task.await.expect("upstream");
    capture.wait_finished(1).await;
    capture.shutdown().await;
}

async fn send(addr: SocketAddr, target: &str, key: &str) -> reqwest::Response {
    client()
        .post(format!("http://{addr}{target}"))
        .header("x-api-key", key)
        .header("content-type", "application/json")
        .body(BODY)
        .send()
        .await
        .expect("send")
}

/// A running `spend-capture replay`, with its standard error read line by line.
struct ReplayProcess {
    child: Child,
    addr: SocketAddr,
    stderr: Receiver<String>,
    seen: Vec<String>,
}

impl ReplayProcess {
    /// Starts a replay of `results`, declaring each `(variable, value)` with `--declare-env`.
    fn start(results: &Path, declared: &[(&str, &str)]) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_spend-capture"));
        command
            .args(["replay", "--listen", "127.0.0.1:0", "--results"])
            .arg(results);
        for (var, value) in declared {
            command.args(["--declare-env", var]).env(var, value);
        }
        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");
        let stderr = child.stderr.take().expect("stderr");
        let (lines, received) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { return };
                if lines.send(line).is_err() {
                    return;
                }
            }
        });
        let mut process = Self {
            child,
            addr: "0.0.0.0:0".parse().expect("addr"),
            stderr: received,
            seen: Vec::new(),
        };
        let serving = process.wait_for("serving ");
        let (_, addr) = serving.rsplit_once(" on ").expect("listen address");
        process.addr = addr.parse().expect("listen address");
        process
    }

    /// The next line of standard error containing `needle`.
    fn wait_for(&mut self, needle: &str) -> String {
        loop {
            let line = self
                .stderr
                .recv_timeout(Duration::from_secs(30))
                .unwrap_or_else(|_| panic!("no line containing `{needle}`; saw {:?}", self.seen));
            self.seen.push(line.clone());
            if line.contains(needle) {
                return line;
            }
        }
    }

    /// Delivers Ctrl-C.
    fn interrupt(&self) {
        let status = Command::new("kill")
            .args(["-INT", &self.child.id().to_string()])
            .status()
            .expect("kill");
        assert!(status.success());
    }

    /// Waits for the process to exit; returns its status and everything it wrote to standard
    /// error.
    fn finish(mut self) -> (ExitStatus, String) {
        let status = self.child.wait().expect("wait");
        self.seen.extend(self.stderr.iter());
        (status, self.seen.join("\n"))
    }
}

impl Drop for ReplayProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

#[tokio::test]
async fn a_credential_in_the_route_is_never_written_by_the_replay() {
    // Sent as the route segment and as the request's own `x-api-key`.
    const SECRET: &str = "sk-route-leak-4242";
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    capture_one_exchange(&results).await;

    let replay = ReplayProcess::start(&results, &[]);
    let response = send(replay.addr, &format!("/{SECRET}/v1/messages"), SECRET).await;
    assert_eq!(response.status(), 409);
    let message = response.text().await.expect("text");
    assert!(message.contains("route"), "{message}");
    assert!(!message.contains(SECRET), "{message}");

    replay.interrupt();
    let (status, stderr) = replay.finish();
    assert!(!status.success());
    assert!(stderr.contains("route"), "{stderr}");
    assert!(!stderr.contains(SECRET), "{stderr}");
}

#[tokio::test]
async fn a_request_refused_while_stopping_fails_the_replay() {
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    capture_one_exchange(&results).await;

    let mut replay = ReplayProcess::start(&results, &[]);
    let served = send(replay.addr, "/anthropic/v1/messages", KEY).await;
    assert_eq!(served.status(), 200);

    // An extra request whose handler is waiting for its body: the server answers
    // `100 Continue` only once the handler reads the body.
    let mut extra = TcpStream::connect(replay.addr).expect("connect");
    write!(
        extra,
        "POST /anthropic/v1/messages HTTP/1.1\r\nhost: replay\r\nx-api-key: {KEY}\r\n\
         content-type: application/json\r\nexpect: 100-continue\r\ncontent-length: {}\r\n\r\n",
        BODY.len()
    )
    .expect("write head");
    let mut continued = [0u8; 25];
    extra.read_exact(&mut continued).expect("read 100");
    assert!(continued.starts_with(b"HTTP/1.1 100"), "{continued:?}");

    replay.interrupt();
    replay.wait_for("stopping");
    extra.write_all(BODY).expect("write body");
    let mut response = String::new();
    extra.read_to_string(&mut response).expect("read response");
    assert!(response.starts_with("HTTP/1.1 409"), "{response}");

    let (status, stderr) = replay.finish();
    assert!(!status.success(), "{stderr}");
    assert!(stderr.contains("diverged"), "{stderr}");
}

#[tokio::test]
async fn a_credential_from_the_response_is_never_written_when_its_request_is_served() {
    // Issued in `Set-Cookie` and present in the request's query: the capture redacts it there,
    // though the request alone does not mark it as a credential.
    const ISSUED: &str = "issued-session-6161";
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    let target = format!("/anthropic/v1/messages?request_id={ISSUED}");
    let cookie = format!("sid={ISSUED}; Path=/");
    capture_exchange(&results, &target, &[("set-cookie", &cookie)]).await;

    let mut replay = ReplayProcess::start(&results, &[]);
    assert_eq!(send(replay.addr, &target, KEY).await.status(), 200);
    replay.wait_for("served");
    replay.interrupt();
    let (status, stderr) = replay.finish();
    assert!(status.success(), "{stderr}");
    assert!(!stderr.contains(ISSUED), "{stderr}");
}

#[tokio::test]
async fn a_value_declared_only_to_the_replay_is_never_written_when_its_request_is_served() {
    // Not a credential to the capture, so recorded as sent; declared to the replay.
    const DECLARED: &str = "declared-to-replay-7272";
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    let target = format!("/anthropic/v1/messages?request_id={DECLARED}");
    capture_exchange(&results, &target, &[]).await;

    let mut replay = ReplayProcess::start(&results, &[("SPEND_CAPTURE_TEST_DECLARED", DECLARED)]);
    assert_eq!(send(replay.addr, &target, KEY).await.status(), 200);
    replay.wait_for("served");
    replay.interrupt();
    let (status, stderr) = replay.finish();
    assert!(status.success(), "{stderr}");
    assert!(!stderr.contains(DECLARED), "{stderr}");
}
