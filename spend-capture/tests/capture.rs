// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! The capture proxy end to end: plain HTTP in, TLS to a scripted upstream, recordings out.

mod support;

use futures::StreamExt;
use spend_capture::recording::{
    BodyKind, FieldValue, HeaderField, RecordingStore, Surface, Termination,
};
use support::{TlsUpstream, client, files_containing, start_capture};
use tokio::sync::oneshot;

const REQUEST_BODY: &[u8] =
    b"{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";

fn header<'a>(headers: &'a [HeaderField], name: &str) -> Option<&'a FieldValue> {
    headers.iter().find(|h| h.name == name).map(|h| &h.value)
}

fn text(value: &str) -> FieldValue {
    FieldValue::Text(value.to_owned())
}

#[tokio::test]
async fn a_buffered_exchange_is_recorded_verbatim() {
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
    let response_body = b"{\"id\":\"x\",\"usage\":{\"prompt_tokens\":3}}\n".to_vec();
    let origin = upstream.origin.clone();

    let served = response_body.clone();
    let upstream_task = tokio::spawn(async move {
        let mut conn = upstream.accept().await;
        let request = conn.read_request().await;
        conn.respond(
            200,
            &[
                ("content-type", "application/json"),
                ("x-ratelimit-remaining-tokens", "149984"),
            ],
            &served,
        )
        .await;
        request
    });

    let response = client()
        .post(format!(
            "http://{}/openai/v1/chat/completions",
            capture.local_addr()
        ))
        .header("content-type", "application/json")
        .header("openai-project", "proj_1")
        .body(REQUEST_BODY)
        .send()
        .await
        .expect("send");
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-ratelimit-remaining-tokens"], "149984");
    assert_eq!(response.bytes().await.expect("body"), response_body);
    let received = upstream_task.await.expect("upstream");
    assert_eq!(received.target(), "/v1/chat/completions");
    assert_eq!(received.body, REQUEST_BODY);
    assert_eq!(received.header("openai-project"), ["proj_1"]);

    capture.wait_finished(1).await;
    let recordings = RecordingStore::open(&results)
        .expect("store")
        .load_all()
        .expect("load");
    assert_eq!(recordings.len(), 1);
    let recording = &recordings[0];
    let meta = &recording.meta;
    assert_eq!(recording.request_body, REQUEST_BODY);
    assert_eq!(recording.response_body, response_body);
    assert_eq!(meta.route, "openai");
    assert_eq!(meta.upstream_origin, origin);
    assert_eq!(meta.method, "POST");
    assert_eq!(meta.uri, "/v1/chat/completions");
    assert_eq!(meta.status, 200);
    assert_eq!(meta.response_kind, BodyKind::Buffered);
    assert_eq!(meta.termination, Termination::UpstreamComplete);
    assert_eq!(
        header(&meta.request_headers, "openai-project"),
        Some(&text("proj_1"))
    );
    assert_eq!(
        header(&meta.response_headers, "x-ratelimit-remaining-tokens"),
        Some(&text("149984"))
    );
    assert!(
        header(&meta.request_headers, "host").is_none(),
        "transport headers are not recorded"
    );
    capture.shutdown().await;
}

#[tokio::test]
async fn a_stream_is_recorded_with_its_observed_chunk_boundaries() {
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

    // A JSON line split across two writes; then two whole events coalesced into one write.
    let first: &[u8] = b"data: {\"choices\":[{\"delta\":{\"cont";
    let second: &[u8] = b"ent\":\"hi\"}}]}\n\n";
    let third: &[u8] = b"data: {\"usage\":{\"total_tokens\":5}}\n\ndata: [DONE]\n\n";
    let (got_first_tx, got_first_rx) = oneshot::channel::<()>();
    let (got_second_tx, got_second_rx) = oneshot::channel::<()>();
    let upstream_task = tokio::spawn(async move {
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        conn.start_stream(&[]).await;
        conn.chunk(first).await;
        got_first_rx.await.expect("client saw first");
        conn.chunk(second).await;
        got_second_rx.await.expect("client saw second");
        conn.chunk(third).await;
        conn.end_stream().await;
    });

    let response = client()
        .post(format!(
            "http://{}/openai/v1/chat/completions",
            capture.local_addr()
        ))
        .body(REQUEST_BODY)
        .send()
        .await
        .expect("send");
    let mut stream = response.bytes_stream();
    let mut received = Vec::new();
    let mut signals = [Some(got_first_tx), Some(got_second_tx)].into_iter();
    let mut wanted = first.len();
    while let Some(chunk) = stream.next().await {
        received.extend_from_slice(&chunk.expect("chunk"));
        if received.len() == wanted {
            if let Some(Some(signal)) = signals.next() {
                signal.send(()).expect("signal");
            }
            wanted += second.len();
        }
    }
    upstream_task.await.expect("upstream");
    let expected = [first, second, third].concat();
    assert_eq!(received, expected);

    capture.wait_finished(1).await;
    let recording = RecordingStore::open(&results)
        .expect("store")
        .load_all()
        .expect("load")
        .remove(0);
    assert_eq!(recording.response_body, expected);
    assert_eq!(recording.meta.response_kind, BodyKind::EventStream);
    assert_eq!(
        recording.meta.response_chunks,
        [first.len() as u64, second.len() as u64, third.len() as u64]
    );
    assert_eq!(recording.meta.termination, Termination::UpstreamComplete);
    capture.shutdown().await;
}

#[tokio::test]
async fn the_first_chunk_reaches_the_client_before_anything_is_committed() {
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
    let (release_tx, release_rx) = oneshot::channel::<()>();
    let upstream_task = tokio::spawn(async move {
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        conn.start_stream(&[]).await;
        conn.chunk(b"data: {\"a\":1}\n\n").await;
        release_rx.await.expect("release");
        conn.chunk(b"data: [DONE]\n\n").await;
        conn.end_stream().await;
    });

    let response = client()
        .post(format!("http://{}/openai/v1/x", capture.local_addr()))
        .body(REQUEST_BODY)
        .send()
        .await
        .expect("send");
    let mut stream = response.bytes_stream();
    let first = stream.next().await.expect("first chunk").expect("ok");
    assert_eq!(&first[..], b"data: {\"a\":1}\n\n");

    let store = RecordingStore::open(&results).expect("store");
    assert!(store.load_all().expect("load").is_empty());

    release_tx.send(()).expect("release");
    while let Some(chunk) = stream.next().await {
        chunk.expect("chunk");
    }
    upstream_task.await.expect("upstream");
    capture.wait_finished(1).await;
    assert_eq!(store.load_all().expect("load").len(), 1);
    capture.shutdown().await;
}

#[tokio::test]
async fn a_client_stopping_at_the_terminator_closes_upstream_and_is_recorded() {
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
        // No terminating chunk: the provider has said [DONE] but not closed the body. Returns
        // only once the capture closes this connection.
        conn.wait_closed().await;
    });

    let response = client()
        .post(format!("http://{}/openai/v1/x", capture.local_addr()))
        .body(REQUEST_BODY)
        .send()
        .await
        .expect("send");
    let mut stream = response.bytes_stream();
    let mut received = Vec::new();
    while !received.ends_with(b"data: [DONE]\n\n") {
        received.extend_from_slice(&stream.next().await.expect("more").expect("chunk"));
    }
    drop(stream);

    upstream_task
        .await
        .expect("upstream saw the connection close");
    capture.wait_finished(1).await;
    let recordings = RecordingStore::open(&results)
        .expect("store")
        .load_all()
        .expect("load");
    assert_eq!(recordings.len(), 1);
    assert_eq!(recordings[0].response_body, sent);
    assert_eq!(
        recordings[0].meta.termination,
        Termination::DownstreamClosedFirst
    );
    capture.shutdown().await;
}

#[tokio::test]
async fn a_credential_in_a_body_refuses_the_recording_but_not_the_exchange() {
    const KEY: &str = "sk-proj-body-echo-0001";
    const DECLARED: &str = "declared-secret-0002";
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    let upstream = TlsUpstream::start().await;
    let capture = start_capture(
        &results,
        &[("openai", &upstream.origin)],
        &upstream.ca_pem,
        &[DECLARED],
    )
    .await;

    // Split across two stream chunks, so no single chunk holds the whole value.
    let (head, tail) = KEY.split_at(9);
    let stream_first = format!("data: {{\"echo\":\"{head}");
    let stream_second = format!("{tail}\"}}\n\ndata: [DONE]\n\n");
    let request_echo = format!("{{\"prompt\":\"{KEY}\"}}");
    let response_echo = format!("{{\"echo\":\"{KEY}\"}}");
    let declared_echo = format!("{{\"echo\":\"{DECLARED}\"}}");

    let (got_first_tx, got_first_rx) = oneshot::channel::<()>();
    let script = (
        response_echo.clone(),
        declared_echo.clone(),
        stream_first.clone(),
        stream_second.clone(),
    );
    let upstream_task = tokio::spawn(async move {
        let (response_echo, declared_echo, stream_first, stream_second) = script;
        let mut requests = Vec::new();
        // `connection: close` makes each exchange its own upstream connection, in order.
        for body in ["{}".to_owned(), response_echo, declared_echo] {
            let mut conn = upstream.accept().await;
            requests.push(conn.read_request().await);
            conn.respond(200, &[("connection", "close")], body.as_bytes())
                .await;
        }
        let mut conn = upstream.accept().await;
        requests.push(conn.read_request().await);
        conn.start_stream(&[]).await;
        conn.chunk(stream_first.as_bytes()).await;
        got_first_rx.await.expect("client saw first");
        conn.chunk(stream_second.as_bytes()).await;
        conn.end_stream().await;
        requests
    });

    let url = format!("http://{}/openai/v1/chat/completions", capture.local_addr());
    let http = client();
    let send = |body: String| http.post(&url).header("x-api-key", KEY).body(body).send();
    let bodies = [
        send(request_echo.clone())
            .await
            .expect("send")
            .bytes()
            .await
            .expect("body"),
        send("{}".into())
            .await
            .expect("send")
            .bytes()
            .await
            .expect("body"),
        send("{}".into())
            .await
            .expect("send")
            .bytes()
            .await
            .expect("body"),
    ];
    assert_eq!(bodies[0], "{}");
    assert_eq!(bodies[1], response_echo);
    assert_eq!(bodies[2], declared_echo);

    let mut stream = send("{}".into()).await.expect("send").bytes_stream();
    let mut streamed = Vec::new();
    let mut signal = Some(got_first_tx);
    while let Some(chunk) = stream.next().await {
        streamed.extend_from_slice(&chunk.expect("chunk"));
        if let Some(tx) = signal.take_if(|_| streamed.len() >= stream_first.len()) {
            tx.send(()).expect("signal");
        }
    }
    assert_eq!(
        streamed,
        format!("{stream_first}{stream_second}").as_bytes()
    );

    let requests = upstream_task.await.expect("upstream");
    assert_eq!(requests[0].body, request_echo.as_bytes());
    assert!(requests.iter().all(|r| r.header("x-api-key") == [KEY]));

    capture.wait_finished(4).await;
    let store = RecordingStore::open(&results).expect("store");
    assert!(store.load_all().expect("load").is_empty());
    let incidents = store.load_incidents().expect("incidents");
    let surfaces: Vec<(u64, Vec<Surface>)> = incidents
        .iter()
        .map(|incident| (incident.sequence, incident.surfaces.clone()))
        .collect();
    assert_eq!(
        surfaces,
        [
            (0, vec![Surface::RequestBody]),
            (1, vec![Surface::ResponseBody]),
            (2, vec![Surface::ResponseBody]),
            (3, vec![Surface::ResponseStream]),
        ]
    );
    for needle in [KEY, DECLARED, head, tail] {
        assert!(
            files_containing(&results, needle.as_bytes()).is_empty(),
            "`{needle}` reached the results directory"
        );
    }
    capture.shutdown().await;
}

#[tokio::test]
async fn an_unknown_route_is_refused_and_nothing_is_recorded() {
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

    let response = client()
        .post(format!(
            "http://{}/anthropic/v1/messages",
            capture.local_addr()
        ))
        .body(REQUEST_BODY)
        .send()
        .await
        .expect("send");
    assert_eq!(response.status(), 404);
    capture.wait_finished(1).await;
    let store = RecordingStore::open(&results).expect("store");
    assert!(store.load_all().expect("load").is_empty());
    capture.shutdown().await;
}

#[tokio::test]
async fn a_results_directory_that_already_holds_recordings_is_refused() {
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
        conn.respond(200, &[], b"{}").await;
    });
    client()
        .post(format!("http://{}/openai/v1/x", capture.local_addr()))
        .body(REQUEST_BODY)
        .send()
        .await
        .expect("send");
    upstream_task.await.expect("upstream");
    capture.wait_finished(1).await;
    capture.shutdown().await;

    let config = spend_capture::capture::CaptureConfig {
        routes: vec![],
        results_dir: results,
        declared_credentials: vec![],
        extra_root_certificates_pem: None,
    };
    assert!(
        spend_capture::capture::CaptureServer::bind("127.0.0.1:0".parse().expect("addr"), config)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn a_credential_in_a_failed_exchange_still_leaves_an_incident() {
    const KEY: &str = "sk-proj-failed-exchange-0003";
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
    let url = format!("http://{}/openai/v1/chat/completions", capture.local_addr());
    let echo = format!("{{\"prompt\":\"{KEY}\"}}");
    let send = |body: String| {
        client()
            .post(&url)
            .header("x-api-key", KEY)
            .body(body)
            .send()
    };
    let (got_request_tx, got_request_rx) = oneshot::channel::<()>();

    let truncated = format!("{{\"echo\":\"{KEY}\"");
    let upstream_task = tokio::spawn(async move {
        // 0: the upstream closes without responding.
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        drop(conn);
        // 1: the upstream promises more body than it sends, then closes.
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        conn.write(format!("HTTP/1.1 200 OK\r\ncontent-length: 500\r\n\r\n{truncated}").as_bytes())
            .await;
        drop(conn);
        // 2: the client goes away while the upstream has not answered yet.
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        got_request_tx.send(()).expect("signal");
        conn.wait_closed().await;
    });

    let no_response = send(echo.clone()).await.expect("send");
    assert_eq!(no_response.status(), 502);
    let truncated_response = send("{}".into()).await.expect("send");
    assert!(
        truncated_response.bytes().await.is_err(),
        "truncation reaches the client"
    );
    let abandoned = tokio::spawn(send(echo.clone()));
    got_request_rx.await.expect("upstream got the request");
    abandoned.abort();
    upstream_task
        .await
        .expect("upstream saw the abandoned connection close");

    capture.wait_finished(3).await;
    let store = RecordingStore::open(&results).expect("store");
    assert!(store.load_all().expect("load").is_empty());
    let surfaces: Vec<(u64, Vec<Surface>)> = store
        .load_incidents()
        .expect("incidents")
        .into_iter()
        .map(|incident| (incident.sequence, incident.surfaces))
        .collect();
    assert_eq!(
        surfaces,
        [
            (0, vec![Surface::RequestBody]),
            (1, vec![Surface::ResponseBody]),
            (2, vec![Surface::RequestBody]),
        ]
    );
    assert!(files_containing(&results, KEY.as_bytes()).is_empty());
    capture.shutdown().await;
}

#[tokio::test]
async fn shutdown_stops_serving_connections_already_open() {
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
        conn.respond(200, &[], b"{}").await;
        // Listener and connection close here, so a request served after shutdown would get a
        // 502 rather than wait on an upstream that never answers.
    });
    let http = client();
    let url = format!("http://{}/openai/v1/x", capture.local_addr());
    let first = http.post(&url).body("{}").send().await.expect("send");
    assert_eq!(first.bytes().await.expect("body"), "{}");
    upstream_task.await.expect("upstream");
    capture.wait_finished(1).await;
    capture.shutdown().await;

    // The same client, whose keep-alive connection to the capture was open at shutdown.
    assert!(http.post(&url).body("{}").send().await.is_err());
    let log = std::fs::read_to_string(results.join("capture.log")).expect("log");
    assert!(!log.contains("seq=1"), "{log}");
}
