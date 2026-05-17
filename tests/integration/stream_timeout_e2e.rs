// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration test for inter-chunk stream timeout.
//!
//! When an upstream stream sends one chunk then hangs, the gateway must:
//!   1. Terminate the stream after `stream_chunk_timeout_ms`.
//!   2. Emit an `event: oxigate.error` SSE event before closing.
//!
//! Uses a real TCP upstream (axum serve on ephemeral port) and `reqwest` to read the
//! SSE stream body, since axum-test buffers the full response and cannot observe partial
//! SSE output. Mirrors the technique used by `streaming.rs` for the disconnect test.

use std::collections::HashMap;
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
use oxigate::config::{
    OpenAIConfig, PricingConfig, RetryConfig, RoutingConfig, SecretString, SecurityConfig,
};
use oxigate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
use oxigate::domain::routing::WeightedRandom;
use oxigate::providers::{OpenAiAdapter, ProviderHealthTracker, ProviderRouter};
use tokio::net::TcpListener;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::fixtures;

/// Milliseconds the mock upstream hangs after the first SSE chunk.
/// Must exceed `stream_chunk_timeout_ms` configured on the router.
const UPSTREAM_HANG_MS: u64 = 10_000;

/// The stream_chunk_timeout_ms set on the router — short for fast tests.
const ROUTER_CHUNK_TIMEOUT_MS: u64 = 100;

/// Builds a mock upstream router that streams one chunk then hangs indefinitely.
fn one_chunk_then_hang_router() -> Router {
    Router::new().route(
        CHAT_COMPLETIONS_PATH,
        post(|| async {
            let body_stream = stream! {
                yield Result::<Bytes, std::convert::Infallible>::Ok(Bytes::from(
                    fixtures::openai_stream_chunk("gpt-4o", "first"),
                ));
                // Hang: simulate upstream stall — exceeds router's inter-chunk timeout.
                tokio::time::sleep(Duration::from_millis(UPSTREAM_HANG_MS)).await;
                // This chunk is never delivered — timeout fires first.
                yield Ok(Bytes::from(fixtures::openai_stream_chunk("gpt-4o", "unreachable")));
            };
            Response::builder()
                .status(200)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .header(header::CACHE_CONTROL, "no-cache")
                .body(Body::from_stream(body_stream))
                .expect("mock stream response")
        }),
    )
}

/// When a provider stream hangs between chunks, the gateway must close the stream
/// with an `oxigate.error` SSE event after the inter-chunk timeout fires.
#[tokio::test(flavor = "multi_thread")]
async fn stream_inter_chunk_timeout_emits_error_event() {
    let pg = PgContainer::start().await.expect("pg must start");
    let redis = RedisContainer::start().await.expect("redis must start");

    // Bind the mock upstream to a random port.
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind mock upstream");
    let upstream_addr = listener.local_addr().expect("local addr");
    let base_url = format!("http://{}", upstream_addr);

    let serve = axum::serve(listener, one_chunk_then_hang_router());
    let upstream_handle = tokio::spawn(async move {
        let _ = serve.await;
    });

    // Build the real adapter pointed at the mock upstream.
    let adapter = OpenAiAdapter::new(OpenAIConfig {
        api_key: Some(SecretString::new("sk-test")),
        default_model: Some("gpt-4o".into()),
        api_base_url: Some(base_url.trim_end_matches('/').to_string()),
        timeout_secs: Some(30),
        supported_models: None,
        organization: None,
        project: None,
    })
    .await
    .expect("OpenAI adapter must build");

    let pricing = Arc::new(std::sync::RwLock::new(
        PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing must parse"),
    ));
    let health = ProviderHealthTracker::new_for_test(&["openai"]);

    // Configure a very short inter-chunk timeout so the test completes quickly.
    let retry = RetryConfig {
        stream_chunk_timeout_ms: ROUTER_CHUNK_TIMEOUT_MS,
        max_retries: 0, // no retries — timeout is a hard stop for streaming
        ..Default::default()
    };

    let router = ProviderRouter::new_with_resilience(
        vec![Arc::new(adapter)],
        Arc::new(WeightedRandom),
        health,
        pricing,
        RoutingConfig {
            weights: HashMap::new(),
            ..Default::default()
        },
        retry,
        vec![],
        SecurityConfig::default(),
    );

    let gateway = crate::common::gateway::TestGateway::spawn_random_http_port(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(router),
    )
    .await;

    let url = gateway
        .server
        .server_url(CHAT_COMPLETIONS_PATH)
        .expect("TestServer must expose HTTP URL");

    // Use reqwest to consume the stream body; the response completes when the gateway
    // closes the connection after the timeout.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("reqwest client");

    let resp = client
        .post(url.as_str())
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
        "expected 200 from gateway, got {}",
        resp.status()
    );

    let body = resp.text().await.expect("read body");

    assert!(
        body.contains("oxigate.error"),
        "stream body must contain oxigate.error SSE event after inter-chunk timeout; got:\n{body}"
    );

    upstream_handle.abort();
    let _ = upstream_handle.await;
}
