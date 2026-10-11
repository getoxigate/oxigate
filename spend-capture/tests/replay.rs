// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! The replay server end to end, over corpora captured from a scripted upstream — so the capture
//! and the replay are held to one recorded form.

mod support;

use std::net::SocketAddr;
use std::path::Path;

use futures::StreamExt;
use reqwest::Method;
use spend_capture::recording::{HeaderField, RecordingStore, Termination};
use spend_capture::redact::REDACTED;
use spend_capture::replay::{
    MISMATCH_STATUS, MismatchField, ReplayConfig, ReplayError, ReplayServer,
};
use support::{TlsUpstream, client, start_capture, start_replay};

/// One request as a test sends it.
#[derive(Clone)]
struct Call {
    method: Method,
    target: String,
    headers: Vec<(&'static str, String)>,
    body: Vec<u8>,
}

impl Call {
    async fn send(&self, addr: SocketAddr) -> reqwest::Response {
        let mut request = client()
            .request(self.method.clone(), format!("http://{addr}{}", self.target))
            .body(self.body.clone());
        for (name, value) in &self.headers {
            request = request.header(*name, value);
        }
        request.send().await.expect("send")
    }

    fn with_header(mut self, name: &'static str, value: &str) -> Self {
        self.headers.retain(|(n, _)| !n.eq_ignore_ascii_case(name));
        self.headers.push((name, value.to_owned()));
        self
    }

    fn without_header(mut self, name: &str) -> Self {
        self.headers.retain(|(n, _)| !n.eq_ignore_ascii_case(name));
        self
    }
}

const ANTHROPIC_BODY: &[u8] =
    br#"{"model":"claude","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#;
const GEMINI_BODY: &[u8] = br#"{"contents":[{"parts":[{"text":"hi"}]}]}"#;

fn anthropic(key: &str) -> Call {
    Call {
        method: Method::POST,
        target: "/anthropic/v1/messages".to_owned(),
        headers: vec![
            ("x-api-key", key.to_owned()),
            ("anthropic-version", "2023-06-01".to_owned()),
            ("content-type", "application/json".to_owned()),
        ],
        body: ANTHROPIC_BODY.to_vec(),
    }
}

fn gemini(key: &str) -> Call {
    Call {
        method: Method::POST,
        target: format!("/gemini/v1beta/models/m:generateContent?key={key}"),
        headers: vec![("content-type", "application/json".to_owned())],
        body: GEMINI_BODY.to_vec(),
    }
}

/// Captures `anthropic` then `gemini` into `results`; each answered `{"n":<position>}`.
async fn capture_two_exchanges(results: &Path) {
    let upstream = TlsUpstream::start().await;
    let capture = start_capture(
        results,
        &[
            ("anthropic", &upstream.origin),
            ("gemini", &upstream.origin),
        ],
        &upstream.ca_pem,
        &[],
    )
    .await;
    let upstream_task = tokio::spawn(async move {
        for n in 0..2 {
            let mut conn = upstream.accept().await;
            conn.read_request().await;
            conn.respond(
                200,
                &[
                    ("content-type", "application/json"),
                    ("connection", "close"),
                ],
                format!("{{\"n\":{n}}}").as_bytes(),
            )
            .await;
        }
    });
    for call in [anthropic("sk-capture-0001"), gemini("capture-key-0002")] {
        assert_eq!(call.send(capture.local_addr()).await.status(), 200);
    }
    upstream_task.await.expect("upstream");
    capture.wait_finished(2).await;
    capture.shutdown().await;
}

#[tokio::test]
async fn a_replay_serves_status_headers_and_body_as_captured() {
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    let upstream = TlsUpstream::start().await;
    let capture = start_capture(
        &results,
        &[("openai", &upstream.origin)],
        &upstream.ca_pem,
        &[],
    )
    .await;
    let upstream_task = tokio::spawn(async move {
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        conn.respond(
            201,
            &[
                ("content-type", "application/json"),
                ("x-request-id", "req_1"),
                ("x-ratelimit-remaining-tokens", "149984"),
                ("connection", "close"),
            ],
            b"{\"id\":\"x\",\"usage\":{\"total_tokens\":5}}",
        )
        .await;
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        conn.start_stream(&[("openai-processing-ms", "12"), ("connection", "close")])
            .await;
        conn.chunk(b"data: {\"choices\":[{\"delta\":{\"cont").await;
        conn.chunk(b"ent\":\"hi\"}}]}\n\n").await;
        conn.chunk(b"data: {\"usage\":{\"total_tokens\":5}}\n\ndata: [DONE]\n\n")
            .await;
        conn.end_stream().await;
    });
    let calls = [
        Call {
            method: Method::POST,
            target: "/openai/v1/chat/completions".to_owned(),
            headers: vec![("content-type", "application/json".to_owned())],
            body: b"{\"stream\":false}".to_vec(),
        },
        Call {
            method: Method::POST,
            target: "/openai/v1/chat/completions".to_owned(),
            headers: vec![("content-type", "application/json".to_owned())],
            body: b"{\"stream\":true}".to_vec(),
        },
    ];
    let mut captured = Vec::new();
    for call in &calls {
        let response = call.send(capture.local_addr()).await;
        let status = response.status();
        let headers = response.headers().clone();
        captured.push((status, headers, response.bytes().await.expect("body")));
    }
    upstream_task.await.expect("upstream");
    capture.wait_finished(2).await;
    capture.shutdown().await;
    let recordings = RecordingStore::open(&results)
        .expect("store")
        .load_all()
        .expect("load");
    assert_eq!(recordings.len(), 2);

    let replay = start_replay(&results, &[]).await;
    for ((call, (status, headers, body)), recording) in calls.iter().zip(&captured).zip(&recordings)
    {
        let response = call.send(replay.local_addr()).await;
        assert_eq!(response.status(), *status);
        for field in &recording.meta.response_headers {
            let values = |map: &reqwest::header::HeaderMap| -> Vec<Vec<u8>> {
                map.get_all(field.name.as_str())
                    .iter()
                    .map(|value| value.as_bytes().to_vec())
                    .collect()
            };
            assert_eq!(
                values(response.headers()),
                values(headers),
                "header `{}`",
                field.name
            );
        }
        assert_eq!(&response.bytes().await.expect("body"), body);
    }
    assert_eq!(replay.served(), 2);
    assert_eq!(replay.mismatch(), None);
    replay.shutdown().await;
}

#[tokio::test]
async fn a_downstream_first_recording_is_never_presented_as_complete() {
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    let upstream = TlsUpstream::start().await;
    let capture = start_capture(
        &results,
        &[("openai", &upstream.origin)],
        &upstream.ca_pem,
        &[],
    )
    .await;
    let sent: &[u8] = b"data: {\"a\":1}\n\ndata: [DONE]\n\n";
    let upstream_task = tokio::spawn(async move {
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        conn.start_stream(&[]).await;
        conn.chunk(sent).await;
        // The provider has said [DONE] but never ends its body.
        conn.wait_closed().await;
    });
    let call = Call {
        method: Method::POST,
        target: "/openai/v1/chat/completions".to_owned(),
        headers: vec![],
        body: b"{\"stream\":true}".to_vec(),
    };
    let read_to_terminator = |response: reqwest::Response| async move {
        let mut stream = response.bytes_stream();
        let mut received = Vec::new();
        while !received.ends_with(b"data: [DONE]\n\n") {
            received.extend_from_slice(&stream.next().await.expect("more").expect("chunk"));
        }
        received
    };
    assert_eq!(
        read_to_terminator(call.send(capture.local_addr()).await).await,
        sent
    );
    upstream_task.await.expect("upstream");
    capture.wait_finished(1).await;
    capture.shutdown().await;
    let recording = RecordingStore::open(&results)
        .expect("store")
        .load_all()
        .expect("load")
        .remove(0);
    assert_eq!(
        recording.meta.termination,
        Termination::DownstreamClosedFirst
    );

    // A consumer that stops at the recorded terminator replays exactly as captured.
    let replay = start_replay(&results, &[]).await;
    assert_eq!(
        read_to_terminator(call.send(replay.local_addr()).await).await,
        sent
    );
    assert_eq!(replay.mismatch(), None);
    replay.shutdown().await;

    // A consumer that reads past the captured prefix gets an error, never a clean end.
    let replay = start_replay(&results, &[]).await;
    let mut stream = call.send(replay.local_addr()).await.bytes_stream();
    let mut received = Vec::new();
    let ended = loop {
        match stream.next().await {
            Some(Ok(chunk)) => received.extend_from_slice(&chunk),
            Some(Err(err)) => break Err(err),
            None => break Ok(()),
        }
    };
    assert!(ended.is_err(), "a captured prefix ended cleanly");
    assert_eq!(received, sent);
    replay.shutdown().await;
}

#[tokio::test]
async fn a_matching_session_is_served_whatever_its_credentials_host_and_transport() {
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    capture_two_exchanges(&results).await;

    let replay = start_replay(&results, &[]).await;
    let first = anthropic("sk-replay-other-9999")
        .with_header("host", "elsewhere.invalid:9")
        .with_header("connection", "keep-alive")
        .send(replay.local_addr())
        .await;
    assert_eq!(first.status(), 200);
    assert_eq!(first.bytes().await.expect("body"), &b"{\"n\":0}"[..]);
    let second = gemini("replay-key-other-9999")
        .send(replay.local_addr())
        .await;
    assert_eq!(second.status(), 200);
    assert_eq!(second.bytes().await.expect("body"), &b"{\"n\":1}"[..]);
    assert_eq!(replay.served(), 2);
    assert_eq!(replay.mismatch(), None);

    // Every recording is served: one more request is unexpected.
    let extra = anthropic("sk-capture-0001").send(replay.local_addr()).await;
    assert_eq!(extra.status(), MISMATCH_STATUS);
    let mismatch = replay.mismatch().expect("mismatch");
    assert_eq!(mismatch.field, MismatchField::Unexpected);
    assert_eq!((mismatch.request, mismatch.expected), (2, None));
    replay.shutdown().await;
}

#[tokio::test]
async fn a_request_that_differs_from_the_next_recording_is_refused_by_field() {
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    capture_two_exchanges(&results).await;

    let key = "sk-capture-0001";
    let mut changed_body = anthropic(key);
    changed_body.body[10] ^= 1;
    let mut changed_second_body = gemini("capture-key-0002");
    changed_second_body.body.push(b' ');
    let retarget = |target: &str| Call {
        target: target.to_owned(),
        ..anthropic(key)
    };
    let cases: Vec<(&str, Vec<Call>, MismatchField)> = vec![
        (
            "method",
            vec![Call {
                method: Method::PUT,
                ..anthropic(key)
            }],
            MismatchField::Method,
        ),
        (
            "route",
            vec![retarget("/openai/v1/messages")],
            MismatchField::Route,
        ),
        (
            "path",
            vec![retarget("/anthropic/v1/complete")],
            MismatchField::Uri,
        ),
        (
            "query",
            vec![retarget("/anthropic/v1/messages?beta=true")],
            MismatchField::Uri,
        ),
        ("body", vec![changed_body], MismatchField::Body),
        (
            "changed anthropic-version",
            vec![anthropic(key).with_header("anthropic-version", "2024-01-01")],
            MismatchField::Header("anthropic-version".to_owned()),
        ),
        (
            "added anthropic-beta",
            vec![anthropic(key).with_header("anthropic-beta", "output-128k-2025-02-19")],
            MismatchField::Header("anthropic-beta".to_owned()),
        ),
        (
            "missing header",
            vec![anthropic(key).without_header("content-type")],
            MismatchField::Header("content-type".to_owned()),
        ),
        (
            "out of order",
            vec![gemini("capture-key-0002")],
            MismatchField::Sequence,
        ),
        (
            "second request",
            vec![anthropic(key), changed_second_body],
            MismatchField::Body,
        ),
        (
            "added query parameter",
            vec![
                anthropic(key),
                Call {
                    target: format!("{}&alt=sse", gemini("other-key").target),
                    ..gemini("other-key")
                },
            ],
            MismatchField::Uri,
        ),
    ];

    for (name, calls, field) in cases {
        let replay = start_replay(&results, &[]).await;
        let (last, earlier) = calls.split_last().expect("a call");
        for call in earlier {
            assert_eq!(call.send(replay.local_addr()).await.status(), 200, "{name}");
        }
        let response = last.send(replay.local_addr()).await;
        assert_eq!(response.status(), MISMATCH_STATUS, "{name}");
        let message = response.text().await.expect("text");
        assert!(message.contains(&field.to_string()), "{name}: {message}");

        let mismatch = replay.mismatch().expect(name);
        assert_eq!(mismatch.field, field, "{name}: {mismatch}");
        assert_eq!(mismatch.request, earlier.len() as u64, "{name}");
        assert_eq!(mismatch.expected, Some(earlier.len() as u64), "{name}");
        assert_eq!(replay.served(), earlier.len(), "{name}");

        // The session has stopped: even the request that was due is refused, not served.
        let due = [anthropic(key), gemini("capture-key-0002")][earlier.len()].clone();
        assert_eq!(
            due.send(replay.local_addr()).await.status(),
            MISMATCH_STATUS,
            "{name}"
        );
        assert_eq!(replay.served(), earlier.len(), "{name}");
        replay.shutdown().await;
    }
}

#[tokio::test]
async fn a_results_directory_without_recordings_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    RecordingStore::open(&results).expect("store");
    let refused = ReplayServer::bind(
        "127.0.0.1:0".parse().expect("addr"),
        ReplayConfig {
            results_dir: results,
            declared_credentials: vec![],
        },
    )
    .await;
    assert!(matches!(refused, Err(ReplayError::Empty(_))));
}

#[tokio::test]
async fn an_exchange_missing_from_the_corpus_stops_the_replay_at_its_position() {
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    let upstream = TlsUpstream::start().await;
    let capture = start_capture(
        &results,
        &[("anthropic", &upstream.origin)],
        &upstream.ca_pem,
        &[],
    )
    .await;
    let upstream_task = tokio::spawn(async move {
        // The first attempt gets no response; the identical retry succeeds.
        let mut failed = upstream.accept().await;
        failed.read_request().await;
        drop(failed);
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        conn.respond(
            200,
            &[
                ("content-type", "application/json"),
                ("connection", "close"),
            ],
            b"{\"n\":1}",
        )
        .await;
    });
    let call = anthropic("sk-capture-0001");
    assert_eq!(call.send(capture.local_addr()).await.status(), 502);
    assert_eq!(call.send(capture.local_addr()).await.status(), 200);
    upstream_task.await.expect("upstream");
    capture.wait_finished(2).await;
    capture.shutdown().await;
    let recordings = RecordingStore::open(&results)
        .expect("store")
        .load_all()
        .expect("load");
    assert_eq!(recordings.len(), 1);
    assert_eq!(recordings[0].meta.sequence, 1);

    // The retry's response is never served in place of the failed attempt.
    let replay = start_replay(&results, &[]).await;
    let response = call.send(replay.local_addr()).await;
    assert_eq!(response.status(), MISMATCH_STATUS);
    let mismatch = replay.mismatch().expect("mismatch");
    assert_eq!(mismatch.field, MismatchField::NotRecorded, "{mismatch}");
    assert_eq!((mismatch.request, mismatch.expected), (0, Some(0)));
    assert_eq!(replay.served(), 0);
    replay.shutdown().await;
}

#[tokio::test]
async fn a_declared_credential_is_redacted_on_replay_and_never_quoted() {
    const DECLARED: &str = "declared-secret-value-77";
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    let upstream = TlsUpstream::start().await;
    let capture = start_capture(
        &results,
        &[("anthropic", &upstream.origin)],
        &upstream.ca_pem,
        &[DECLARED],
    )
    .await;
    let upstream_task = tokio::spawn(async move {
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        conn.respond(
            200,
            &[
                ("content-type", "application/json"),
                ("connection", "close"),
            ],
            b"{}",
        )
        .await;
    });
    let call =
        anthropic("sk-capture-0001").with_header("x-custom-id", &format!("id={DECLARED};v=1"));
    assert_eq!(call.send(capture.local_addr()).await.status(), 200);
    upstream_task.await.expect("upstream");
    capture.wait_finished(1).await;
    capture.shutdown().await;

    let recording = RecordingStore::open(&results)
        .expect("store")
        .load_all()
        .expect("load")
        .remove(0);
    let custom = header(&recording.meta.request_headers, "x-custom-id");
    assert_eq!(custom, format!("id={REDACTED};v=1"));

    // Declared to the replay or not, the identical request matches: a credential value never
    // causes a mismatch.
    for declared in [&[DECLARED][..], &[]] {
        let replay = start_replay(&results, declared).await;
        assert_eq!(call.send(replay.local_addr()).await.status(), 200);
        assert_eq!(replay.mismatch(), None);
        replay.shutdown().await;
    }

    // A difference outside the redacted part is refused by header name, and the request's
    // value is quoted nowhere.
    let replay = start_replay(&results, &[]).await;
    let changed = call.with_header("x-custom-id", &format!("id={DECLARED};v=2"));
    let response = changed.send(replay.local_addr()).await;
    assert_eq!(response.status(), MISMATCH_STATUS);
    let message = response.text().await.expect("text");
    assert!(!message.contains(DECLARED), "{message}");
    let mismatch = replay.mismatch().expect("mismatch");
    assert_eq!(
        mismatch.field,
        MismatchField::Header("x-custom-id".to_owned())
    );
    assert!(!mismatch.to_string().contains(DECLARED), "{mismatch}");
    replay.shutdown().await;
}

/// The single recorded value of request header `name`, as text.
fn header(fields: &[HeaderField], name: &str) -> String {
    let values: Vec<_> = fields.iter().filter(|field| field.name == name).collect();
    assert_eq!(values.len(), 1, "{name}");
    String::from_utf8_lossy(values[0].value.as_bytes()).into_owned()
}

#[tokio::test]
async fn a_request_redacted_with_a_credential_from_its_response_replays() {
    // Issued by the upstream in `Set-Cookie`, and already present in a request header the
    // gateway sent: the capture redacts it there, though the request alone does not mark it.
    const ISSUED: &str = "issued-session-5150";
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    let upstream = TlsUpstream::start().await;
    let capture = start_capture(
        &results,
        &[("anthropic", &upstream.origin)],
        &upstream.ca_pem,
        &[],
    )
    .await;
    let upstream_task = tokio::spawn(async move {
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        conn.respond(
            200,
            &[
                ("content-type", "application/json"),
                ("set-cookie", &format!("sid={ISSUED}; Path=/; HttpOnly")),
                ("connection", "close"),
            ],
            b"{}",
        )
        .await;
    });
    let call = anthropic("sk-capture-0001").with_header("x-custom-id", ISSUED);
    assert_eq!(call.send(capture.local_addr()).await.status(), 200);
    upstream_task.await.expect("upstream");
    capture.wait_finished(1).await;
    capture.shutdown().await;
    let recording = RecordingStore::open(&results)
        .expect("store")
        .load_all()
        .expect("load")
        .remove(0);
    assert_eq!(
        header(&recording.meta.request_headers, "x-custom-id"),
        REDACTED
    );

    let replay = start_replay(&results, &[]).await;
    assert_eq!(call.send(replay.local_addr()).await.status(), 200);
    assert_eq!(replay.mismatch(), None);
    replay.shutdown().await;
}

#[tokio::test]
async fn a_value_declared_only_to_the_replay_does_not_change_matching() {
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    capture_two_exchanges(&results).await;

    // `messages` is in the first request's path; declaring it only scrubs what the replay says.
    let replay = start_replay(&results, &["messages"]).await;
    let first = anthropic("sk-capture-0001").send(replay.local_addr()).await;
    assert_eq!(first.status(), 200);
    let second = gemini("capture-key-0002").send(replay.local_addr()).await;
    assert_eq!(second.status(), 200);
    assert_eq!(replay.mismatch(), None);
    replay.shutdown().await;
}
