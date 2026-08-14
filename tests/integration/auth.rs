// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for auth Tower layer.
//!
//! Verifies Bearer token validation on /v1/* routes and that health routes bypass auth.

use std::sync::Arc;

use axum::http::StatusCode;

use oxigate::api::CHAT_COMPLETIONS_PATH;
use oxigate::config::{AuthConfig, OpenAICompatConfig, SecretString};
use oxigate::providers::{CompatHttpClient, OpenAICompatAdapter};

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::StubAdapter;
use crate::common::wiremock_stubs;

#[tokio::test]
async fn test_v1_bypass_when_key_absent() {
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

    // key: None — bypass mode; /v1/* must accept requests with no Authorization header.
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

#[tokio::test]
async fn test_health_routes_bypass_auth() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let auth = AuthConfig {
        key: Some(SecretString::from("secret-token")),
    };
    let gateway = TestGateway::spawn_with_auth(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        auth,
    )
    .await;

    let health = gateway.server.get("/health").await;
    health.assert_status(StatusCode::OK);

    let ready = gateway.server.get("/health/ready").await;
    ready.assert_status(StatusCode::OK);
}

#[tokio::test]
async fn test_auth_integration_correct_token() {
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

    let auth = AuthConfig {
        key: Some(SecretString::from("correct-token")),
    };
    let gateway =
        TestGateway::spawn_with_auth(pg.pool.clone(), redis.pool.clone(), provider, auth).await;

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer correct-token")
        .json(&serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .await;

    response.assert_status(StatusCode::OK);
}

#[tokio::test]
async fn test_auth_integration_missing_token() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let auth = AuthConfig {
        key: Some(SecretString::from("required-token")),
    };
    let gateway = TestGateway::spawn_with_auth(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        auth,
    )
    .await;

    let response = gateway
        .server
        .post("/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .await;

    response.assert_status(StatusCode::UNAUTHORIZED);
    response.assert_json(&serde_json::json!({
        "error": "unauthorized",
        "message": "missing Authorization header"
    }));
}

#[tokio::test]
async fn test_auth_integration_wrong_token() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let auth = AuthConfig {
        key: Some(SecretString::from("correct-token")),
    };
    let gateway = TestGateway::spawn_with_auth(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        auth,
    )
    .await;

    let response = gateway
        .server
        .post("/v1/chat/completions")
        .add_header("Authorization", "Bearer wrong-token")
        .json(&serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .await;

    response.assert_status(StatusCode::UNAUTHORIZED);
    response.assert_json(&serde_json::json!({
        "error": "unauthorized",
        "message": "invalid API key"
    }));
}
