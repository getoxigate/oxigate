// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for request body size limits.
//!
//! Verifies DefaultBodyLimit enforcement — 413 before handler on oversized bodies,
//! and pass-through for bodies within the limit.

use std::sync::Arc;

use axum::http::StatusCode;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::StubAdapter;
use oxigate::api::CHAT_COMPLETIONS_PATH;

/// Small limit used for tests — avoids sending megabytes of data.
const TEST_LIMIT_BYTES: usize = 1024;

/// POST body exceeding the configured limit returns 413 before the handler runs.
///
/// The handler uses `StubAdapter` which returns 501 for any real request;
/// a 413 response proves `DefaultBodyLimit` rejected the body before handler dispatch.
#[tokio::test]
async fn test_oversized_body_returns_413() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let gateway = TestGateway::spawn_with_body_limit(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        TEST_LIMIT_BYTES,
    )
    .await;

    // Build a JSON body that exceeds TEST_LIMIT_BYTES. The message content is padded
    // so the full serialized JSON is definitely over the limit.
    let large_content = "x".repeat(TEST_LIMIT_BYTES + 128);
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": large_content}]
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::PAYLOAD_TOO_LARGE);
}

/// POST body within the configured limit passes through to the handler (not 413).
///
/// Uses a minimal well-formed JSON body (well under TEST_LIMIT_BYTES). StubAdapter returns
/// `ProviderError::NotImplemented` → `ChatError::NotImplemented` → 501. Asserting 501
/// proves the request reached the handler, not just that it wasn't rejected as 413.
#[tokio::test]
async fn test_body_within_limit_reaches_handler() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let gateway = TestGateway::spawn_with_body_limit(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        TEST_LIMIT_BYTES,
    )
    .await;

    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    // 501 proves the request reached the handler (StubAdapter → NotImplemented).
    // A 413 here would mean DefaultBodyLimit rejected it before dispatch.
    response.assert_status(StatusCode::NOT_IMPLEMENTED);
}
