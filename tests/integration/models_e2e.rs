// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for GET /v1/models .
//!
//! Verifies OpenAI-compatible model list, auth, and response shape.

use std::sync::Arc;

use axum::http::StatusCode;

use oxigate::config::{AuthConfig, SecretString};
use oxigate::domain::ports::HealthStatus;
use oxigate::providers::ProviderHealthTracker;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::{ModelsTestAdapter, StubAdapter};

#[tokio::test]
async fn test_get_models_200_with_valid_auth() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let provider = Arc::new(ModelsTestAdapter::new("openai", vec!["gpt-4o", "gpt-4"]));
    let auth = AuthConfig {
        key: Some(SecretString::from("test-token")),
    };

    let gateway =
        TestGateway::spawn_with_auth(pg.pool.clone(), redis.pool.clone(), provider, auth).await;

    let response = gateway
        .server
        .get("/v1/models")
        .add_header("Authorization", "Bearer test-token")
        .await;

    response.assert_status(StatusCode::OK);
    let body: serde_json::Value = response.json();
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("data must be array");
    assert!(!data.is_empty(), "data must not be empty");
}

#[tokio::test]
async fn test_get_models_401_without_auth() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let auth = AuthConfig {
        key: Some(SecretString::from("required")),
    };
    let gateway = TestGateway::spawn_with_auth(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        auth,
    )
    .await;

    let response = gateway.server.get("/v1/models").await;

    response.assert_status(StatusCode::UNAUTHORIZED);
    let body: serde_json::Value = response.json();
    assert!(body.get("error").is_some());
}

#[tokio::test]
async fn test_get_models_401_wrong_token() {
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
        .get("/v1/models")
        .add_header("Authorization", "Bearer wrong-token")
        .await;

    response.assert_status(StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_get_models_entry_shape() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let provider = Arc::new(ModelsTestAdapter::new("openai", vec!["gpt-4o"]));
    let tracker = ProviderHealthTracker::new_for_test(&["openai"]);
    tracker.update_health("openai", HealthStatus::Healthy).await;
    let auth = AuthConfig {
        key: Some(SecretString::from("shape-test-token")),
    };

    let gateway = TestGateway::spawn_with_health_tracker(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        auth,
        tracker,
    )
    .await;

    let response = gateway
        .server
        .get("/v1/models")
        .add_header("Authorization", "Bearer shape-test-token")
        .await;

    response.assert_status(StatusCode::OK);
    let body: serde_json::Value = response.json();
    let data = body["data"].as_array().expect("data must be array");
    assert!(!data.is_empty(), "data must not be empty");

    let entry = &data[0];
    assert_eq!(entry["id"], "gpt-4o", "entry id must match model");
    assert_eq!(entry["object"], "model", "object must be \"model\"");
    assert!(
        entry["created"].as_u64().unwrap_or(0) > 0,
        "created must be > 0"
    );
    assert_eq!(
        entry["owned_by"], "openai",
        "owned_by must be provider name"
    );

    let oxi = &entry["oxigate"];
    assert!(oxi.is_object(), "oxigate extension must be present");
    assert_eq!(oxi["provider"], "openai");
    assert_eq!(
        oxi["health_status"], "available",
        "health_status must reflect injected health_map"
    );
    assert!(
        oxi.get("supports_streaming").is_some(),
        "supports_streaming must be present"
    );
    assert!(
        oxi.get("supports_tools").is_some(),
        "supports_tools must be present"
    );
    assert!(
        oxi.get("supports_vision").is_some(),
        "supports_vision must be present"
    );
    assert!(
        oxi.get("supports_embeddings").is_some(),
        "supports_embeddings must be present"
    );
    assert!(
        oxi.get("supports_thinking").is_some(),
        "supports_thinking must be present"
    );
}

#[tokio::test]
async fn test_get_models_health_status_available() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let provider = Arc::new(ModelsTestAdapter::new(
        "anthropic",
        vec!["claude-3-5-sonnet"],
    ));
    let tracker = ProviderHealthTracker::new_for_test(&["anthropic"]);
    tracker
        .update_health("anthropic", HealthStatus::Healthy)
        .await;
    let auth = AuthConfig {
        key: Some(SecretString::from("health-test-token")),
    };

    let gateway = TestGateway::spawn_with_health_tracker(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        auth,
        tracker,
    )
    .await;

    let response = gateway
        .server
        .get("/v1/models")
        .add_header("Authorization", "Bearer health-test-token")
        .await;

    response.assert_status(StatusCode::OK);
    let body: serde_json::Value = response.json();
    let data = body["data"].as_array().expect("data must be array");
    let entry = data
        .iter()
        .find(|e| e["id"] == "claude-3-5-sonnet")
        .expect("claude-3-5-sonnet must appear in response");
    assert_eq!(
        entry["oxigate"]["health_status"], "available",
        "model must report available when health_map says available"
    );
}

#[tokio::test]
async fn test_get_models_health_status_unknown_when_absent_from_map() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    // health_map is empty (default spawn) — all models must report "unknown".
    let provider = Arc::new(ModelsTestAdapter::new("gemini", vec!["gemini-2.0-flash"]));
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let response = gateway.server.get("/v1/models").await;

    response.assert_status(StatusCode::OK);
    let body: serde_json::Value = response.json();
    let data = body["data"].as_array().expect("data must be array");
    let entry = data
        .iter()
        .find(|e| e["id"] == "gemini-2.0-flash")
        .expect("gemini-2.0-flash must appear in response");
    assert_eq!(
        entry["oxigate"]["health_status"], "unknown",
        "model must report unknown when provider is absent from health_map"
    );
}

#[tokio::test]
async fn test_get_models_no_wildcard_entries() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    // Adapter with only wildcard — should contribute 0 entries.
    let provider = Arc::new(ModelsTestAdapter::new("compat-test", vec!["*"]));
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let response = gateway.server.get("/v1/models").await;

    response.assert_status(StatusCode::OK);
    let body: serde_json::Value = response.json();
    let data = body["data"].as_array().expect("data must be array");
    let ids: Vec<&str> = data.iter().filter_map(|e| e["id"].as_str()).collect();
    assert!(
        !ids.contains(&"*"),
        "wildcard must not appear in response: {:?}",
        ids
    );
}
