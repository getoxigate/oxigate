// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for request tagging .
//!
//! Verifies that tagged requests are accepted and processed end-to-end.
//! `RequestIdentity.tags` capture is verified by unit tests in `src/middleware/tagger.rs`
//! (no mechanism to inspect internal extensions from the HTTP boundary without a debug
//! endpoint or tracing capture infrastructure).
//! Spend record persistence (tags JSONB) is in scope.

use std::sync::Arc;

use axum::http::StatusCode;

use oxigate::api::CHAT_COMPLETIONS_PATH;
use oxigate::config::OpenAICompatConfig;
use oxigate::providers::{CompatHttpClient, OpenAICompatAdapter};
use oxigate::utils::CostHeader;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::wiremock_stubs;

#[tokio::test]
async fn test_tag_headers_flow_through_full_stack() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_chat(&mock, "gpt-4", 1, 1).await;

    let provider = Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: "compat-test".to_string(),
                base_url: mock.uri().trim_end_matches('/').to_string(),
                api_key: None,
                supported_models: None,
                stream_options_support: false,
                supports_tools: false,
                timeout_secs: None,
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("compat adapter must build"),
    );

    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("X-OxiGate-Team", "engineering")
        .add_header("X-OxiGate-Project", "chat")
        .json(&serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .await;

    response.assert_status(StatusCode::OK);
    assert!(
        response.headers().get(CostHeader::REQUEST_COST).is_some(),
        "cost headers must be present when request flows through full stack with tag headers"
    );
}

#[tokio::test]
async fn test_no_tag_headers_request_succeeds() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_chat(&mock, "gpt-4", 1, 1).await;

    let provider = Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: "compat-test".to_string(),
                base_url: mock.uri().trim_end_matches('/').to_string(),
                api_key: None,
                supported_models: None,
                stream_options_support: false,
                supports_tools: false,
                timeout_secs: None,
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("compat adapter must build"),
    );

    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .json(&serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .await;

    response.assert_status(StatusCode::OK);
}
