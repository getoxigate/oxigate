// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! E2E tests for POST /v1/embeddings.
//!
//! Verifies that the full axum stack produces correct cost headers for embedding requests.

use std::sync::Arc;

use axum::http::StatusCode;
use oxigate::api::EMBEDDINGS_PATH;
use oxigate::config::{AuthConfig, BudgetConfig, OpenAIConfig, SecretString};
use oxigate::providers::openai::OpenAiAdapter;
use oxigate::utils::CostHeader;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::StubAdapter;
use crate::common::wiremock_stubs;

// Matches RequestIdentity::default() key path used by auth-disabled test flows.
const DEFAULT_SPEND_KEY: &str = "oxigate:org:default:spend:default";

/// POST /v1/embeddings → cost headers present and non-zero for a known model.
///
/// Exercises the path: request → auth middleware → embeddings handler →
/// build_embedding_cost_headers → response headers.
#[tokio::test]
async fn test_embeddings_e2e_cost_headers() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_embeddings(&mock, "text-embedding-3-small", 42).await;

    let provider = Arc::new(
        OpenAiAdapter::new(OpenAIConfig {
            api_key: Some(SecretString::new("sk-test")),
            api_base_url: Some(mock.uri().trim_end_matches('/').to_string()),
            default_model: None,
            timeout_secs: Some(10),
            supported_models: None,
            organization: None,
            project: None,
        })
        .await
        .expect("OpenAiAdapter must build"),
    );

    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "text-embedding-3-small",
        "input": "hello world"
    });

    let response = gateway
        .server
        .post(EMBEDDINGS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::OK);

    let headers = response.headers();
    assert!(
        headers.contains_key(CostHeader::REQUEST_COST),
        "missing X-Oxigate-Request-Cost header"
    );
    assert!(
        headers.contains_key(CostHeader::INPUT_TOKENS),
        "missing X-Oxigate-Input-Tokens header"
    );
    assert!(
        headers.contains_key(CostHeader::MODEL_USED),
        "missing X-Oxigate-Model-Used header"
    );

    let cost_val = headers
        .get(CostHeader::REQUEST_COST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("0.000000");
    assert_ne!(
        cost_val, "0.000000",
        "text-embedding-3-small (42 tokens) must produce non-zero cost"
    );

    let input_tokens = headers
        .get(CostHeader::INPUT_TOKENS)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    assert_eq!(input_tokens, 42, "input token count must match fixture");

    assert_eq!(
        headers
            .get(CostHeader::OUTPUT_TOKENS)
            .and_then(|v| v.to_str().ok()),
        Some("0"),
        "embeddings must always report zero output tokens"
    );
}

/// POST /v1/embeddings with a provider error → zero cost headers present (no panic).
#[tokio::test]
async fn test_embeddings_e2e_provider_error_zero_cost_headers() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    // Return 500 from the upstream provider.
    let err_mock = wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path(EMBEDDINGS_PATH))
        .respond_with(wiremock::ResponseTemplate::new(500).set_body_string("upstream error"));
    mock.register(err_mock).await;

    let provider = Arc::new(
        OpenAiAdapter::new(OpenAIConfig {
            api_key: Some(SecretString::new("sk-test")),
            api_base_url: Some(mock.uri().trim_end_matches('/').to_string()),
            default_model: None,
            timeout_secs: Some(10),
            supported_models: None,
            organization: None,
            project: None,
        })
        .await
        .expect("OpenAiAdapter must build"),
    );

    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "text-embedding-3-small",
        "input": "hello world"
    });

    let response = gateway
        .server
        .post(EMBEDDINGS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    // Provider 500 → ProviderUnavailable → gateway 503.
    response.assert_status(StatusCode::SERVICE_UNAVAILABLE);
    // inject_zero_cost_headers must emit exactly "0.000000" on error.
    assert_eq!(
        response
            .headers()
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok()),
        Some("0.000000"),
        "zero cost header must be present and equal '0.000000' on error path"
    );
}

fn openai_adapter(base_url: &str) -> OpenAIConfig {
    OpenAIConfig {
        api_key: Some(SecretString::new("sk-test")),
        api_base_url: Some(base_url.trim_end_matches('/').to_string()),
        default_model: None,
        timeout_secs: Some(10),
        supported_models: None,
        organization: None,
        project: None,
    }
}

/// POST /v1/embeddings with array input → cost headers reflect total token count.
#[tokio::test]
async fn test_embeddings_e2e_batch_input_cost_headers() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    // 3 inputs, 90 total tokens
    wiremock_stubs::stub_openai_embeddings_batch(&mock, "text-embedding-3-small", 3, 90).await;

    let provider = Arc::new(
        OpenAiAdapter::new(openai_adapter(&mock.uri()))
            .await
            .expect("adapter must build"),
    );
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "text-embedding-3-small",
        "input": ["foo", "bar", "baz"]
    });

    let response = gateway
        .server
        .post(EMBEDDINGS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::OK);
    let headers = response.headers();

    let input_tokens = headers
        .get(CostHeader::INPUT_TOKENS)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    assert_eq!(input_tokens, 90, "batch token count must sum all inputs");

    let cost_val = headers
        .get(CostHeader::REQUEST_COST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("0.000000");
    assert_ne!(
        cost_val, "0.000000",
        "batch embedding must produce non-zero cost"
    );

    let json: serde_json::Value = response.json();
    assert_eq!(
        json["data"].as_array().map(|a| a.len()),
        Some(3),
        "response must contain 3 embedding vectors"
    );
}

/// HardCapLayer blocks POST /v1/embeddings when spend exceeds the cap.
///
/// Proves the embeddings route passes through the same HardCapLayer middleware as chat
/// completions. The provider is never called — HardCapLayer short-circuits on seeded spend.
#[tokio::test]
async fn test_embeddings_budget_enforced() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let provider = Arc::new(StubAdapter::new());
    let budget = BudgetConfig {
        hard_cap_usd: Some(0.001), // $0.001 = 1_000_000 nano USD
        ..BudgetConfig::default()
    };
    let gateway = TestGateway::spawn_with_budget(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    // Seed spend at exactly the hard cap — HardCapLayer uses >= so boundary triggers 429.
    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_SPEND_KEY)
        .arg(1_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed spend key");

    let response = gateway
        .server
        .post(EMBEDDINGS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&serde_json::json!({
            "model": "text-embedding-3-small",
            "input": "hello"
        }))
        .await;

    response.assert_status(StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("0.000000"),
        "budget_remaining must be '0.000000' on hard-cap rejection for embeddings"
    );
}

/// POST /v1/embeddings with dimensions param → forwarded to provider, response 200.
#[tokio::test]
async fn test_embeddings_e2e_dimensions_param_forwarded() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_embeddings(&mock, "text-embedding-3-large", 10).await;

    let provider = Arc::new(
        OpenAiAdapter::new(openai_adapter(&mock.uri()))
            .await
            .expect("adapter must build"),
    );
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "text-embedding-3-large",
        "input": "hello",
        "dimensions": 1024
    });

    let response = gateway
        .server
        .post(EMBEDDINGS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    // Gateway must return 200 and forward the dimensions param without rejecting it.
    response.assert_status(StatusCode::OK);
    assert!(
        response.headers().contains_key(CostHeader::REQUEST_COST),
        "cost header must be present even with dimensions param"
    );
}
