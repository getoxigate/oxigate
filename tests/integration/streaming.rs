// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! T-cancel: upstream streaming read is cancelled when the gateway client disconnects.
//! Proved by a mock upstream that sends on a channel immediately before emitting chunk 2; after the
//! client drops the response, `recv` must not observe that signal within a deadline derived from the
//! same upstream delay (no paired wall-clock sleeps).
//!
//! Uses a minimal axum SSE upstream and `reqwest` reading only the first chunk. `axum-test`
//! buffers the full response body on await; wiremock 0.6 cannot interleave chunk/sleep/chunk.

use std::sync::Arc;
use std::time::Duration;

use async_stream::stream;
use axum::Router;
use axum::body::Body;
use axum::http::header;
use axum::response::Response;
use axum::routing::post;
use bytes::Bytes;
use oxigate::api::CHAT_COMPLETIONS_PATH;
use oxigate::config::{OpenAIConfig, SecretString};
use oxigate::providers::OpenAiAdapter;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::fixtures;
use crate::common::gateway::TestGateway;

/// Seconds the mock upstream waits before signaling “about to send chunk 2”.
const UPSTREAM_BEFORE_CHUNK2_SIGNAL_SECS: u64 = 5;
/// Milliseconds of slack on top of upstream delay; recv deadline must exceed upstream delay so a
/// regression (no cancel) always delivers the signal in time and fails the test.
const RECV_DEADLINE_SLACK_MS: u64 = 500;
/// `timeout` on `mpsc::Receiver::recv`: upstream delay + slack (keep in sync with pre-send sleep).
const RECV_DEADLINE: Duration =
    Duration::from_millis(UPSTREAM_BEFORE_CHUNK2_SIGNAL_SECS * 1000 + RECV_DEADLINE_SLACK_MS);

fn openai_config_upstream(base: &str) -> OpenAIConfig {
    OpenAIConfig {
        api_key: Some(SecretString::new("sk-test-key")),
        default_model: Some("gpt-4o".into()),
        api_base_url: Some(base.trim_end_matches('/').to_string()),
        timeout_secs: Some(30),
        supported_models: None,
        organization: None,
        project: None,
    }
}

fn slow_sse_router(chunk2_attempt_tx: mpsc::Sender<()>) -> Router {
    Router::new().route(
        CHAT_COMPLETIONS_PATH,
        post(move || {
            let tx = chunk2_attempt_tx.clone();
            async move {
                let body_stream = stream! {
                    yield Result::<Bytes, std::convert::Infallible>::Ok(Bytes::from(
                        fixtures::openai_stream_chunk("gpt-4o", "first"),
                    ));
                    tokio::time::sleep(Duration::from_secs(UPSTREAM_BEFORE_CHUNK2_SIGNAL_SECS)).await;
                    let _ = tx.send(()).await;
                    yield Ok(Bytes::from(fixtures::openai_stream_chunk(
                        "gpt-4o", "second",
                    )));
                };
                Response::builder()
                    .status(200)
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .header(header::CACHE_CONTROL, "no-cache")
                    .body(Body::from_stream(body_stream))
                    .expect("streaming mock response")
            }
        }),
    )
}

#[tokio::test]
async fn streaming_client_disconnect_releases_upstream_before_slow_chunk() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind mock upstream");
    let upstream_addr = listener.local_addr().expect("local addr");

    let (chunk2_attempt_tx, mut chunk2_attempt_rx) = mpsc::channel(1);
    let app = slow_sse_router(chunk2_attempt_tx);

    let serve = axum::serve(listener, app);
    let upstream_handle = tokio::spawn(async move {
        let _ = serve.await;
    });

    let base = format!("http://{}", upstream_addr);
    let config = openai_config_upstream(&base);
    let adapter = OpenAiAdapter::new(config)
        .await
        .expect("OpenAI adapter must build");
    let gateway =
        TestGateway::spawn_random_http_port(pg.pool.clone(), redis.pool.clone(), Arc::new(adapter))
            .await;

    let url = gateway
        .server
        .server_url(CHAT_COMPLETIONS_PATH)
        .expect("TestServer must expose HTTP URL");

    let client = reqwest::Client::new();
    let mut resp = client
        .post(url)
        .header("Authorization", "Bearer sk-test-key")
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "x"}],
            "stream": true
        }))
        .send()
        .await
        .expect("gateway request");

    assert!(
        resp.status().is_success(),
        "expected 200 from streaming endpoint, got {}",
        resp.status()
    );

    let first = resp
        .chunk()
        .await
        .expect("read chunk")
        .expect("first body chunk");
    assert!(
        !first.is_empty(),
        "expected first SSE bytes from upstream through gateway"
    );
    drop(resp);

    let outcome = tokio::time::timeout(RECV_DEADLINE, chunk2_attempt_rx.recv()).await;
    assert!(
        !matches!(outcome, Ok(Some(()))),
        "upstream must not signal chunk-2 attempt after client disconnect (cancel regression)"
    );

    upstream_handle.abort();
    let _ = upstream_handle.await;
}
