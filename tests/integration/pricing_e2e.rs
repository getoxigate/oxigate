// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! E2E tests for provider-specific pricing .
//!
//! Cache tokens, batch discount, and tier override flow through to the request cost header (`CostHeader::REQUEST_COST`).

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
async fn anthropic_cache_tokens_reflected_in_cost_header() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    // No cache: 1500 input at full rate + 500 completion
    wiremock_stubs::stub_openai_chat_once(&mock, "claude-sonnet-4-6", 1500, 500).await;
    // Anthropic semantics: prompt_tokens = input-only (plain), cache_read additive
    // 500 plain + 1000 cache_read + 500 completion; cache_read at 0.1x
    wiremock_stubs::stub_openai_chat_with_anthropic_cache(
        &mock,
        "claude-sonnet-4-6",
        500,
        500,
        0,
        1000,
    )
    .await;

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

    let body = serde_json::json!({
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response_without_cache = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response_without_cache.assert_status(StatusCode::OK);
    let cost_without_cache: f64 = response_without_cache
        .headers()
        .get(CostHeader::REQUEST_COST)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .expect("request cost header present");

    let response_with_cache = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response_with_cache.assert_status(StatusCode::OK);
    let cost_with_cache: f64 = response_with_cache
        .headers()
        .get(CostHeader::REQUEST_COST)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .expect("request cost header present");

    // 500@1x + 1000@0.1x should cost less than 1500@1x (same output)
    assert!(
        cost_with_cache < cost_without_cache,
        "cache_read at 0.1x must reduce cost vs full rate: with_cache={cost_with_cache}, without={cost_without_cache}"
    );
}

// Remove the duplicate block - we only need two requests

#[tokio::test]
async fn openai_batch_flag_ignored_until_batch_api() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_chat_with_cache(&mock, "gpt-4.1", 1000, 500, 0).await;

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

    let body_batch = serde_json::json!({
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hello"}],
        "batch": true
    });
    let body_non_batch = serde_json::json!({
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response_batch = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body_batch)
        .await;

    let response_non_batch = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body_non_batch)
        .await;

    response_batch.assert_status(StatusCode::OK);
    response_non_batch.assert_status(StatusCode::OK);

    let cost_batch: f64 = response_batch
        .headers()
        .get(CostHeader::REQUEST_COST)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .expect("request cost header present");

    let cost_non_batch: f64 = response_non_batch
        .headers()
        .get(CostHeader::REQUEST_COST)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .expect("request cost header present");

    // SECURITY: batch flag from user input is ignored until we support /v1/batches.
    // Both requests must get the same cost to prevent budget bypass.
    assert!(
        (cost_batch - cost_non_batch).abs() < 0.000001,
        "batch=true must be ignored: batch={cost_batch}, non_batch={cost_non_batch}"
    );
}

#[tokio::test]
async fn openai_cached_tokens_discount_applied() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_chat_with_cache_once(&mock, "gpt-4.1", 100, 50, 50).await;

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

    let body = serde_json::json!({
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response_cached = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    wiremock_stubs::stub_openai_chat(&mock, "gpt-4.1", 100, 50).await;
    let response_uncached = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response_cached.assert_status(StatusCode::OK);
    response_uncached.assert_status(StatusCode::OK);

    let cost_cached: f64 = response_cached
        .headers()
        .get(CostHeader::REQUEST_COST)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .expect("request cost header present");

    let cost_uncached: f64 = response_uncached
        .headers()
        .get(CostHeader::REQUEST_COST)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .expect("request cost header present");

    assert!(
        cost_cached < cost_uncached,
        "cached tokens must reduce cost: cached={cost_cached}, uncached={cost_uncached}"
    );
}

/// Verifies Gemini tiered pricing: requests above 128k tokens use tier-2 rate.
/// gemini-1.5-pro-002 tier 0: 0–128k @ 1.25/1M input; tier 1: 128k+ @ 2.5/1M input.
#[tokio::test]
async fn gemini_tiered_pricing_applies_higher_rate() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    // Tier 0: 1000 input tokens
    wiremock_stubs::stub_openai_chat_once(&mock, "gemini-1.5-pro", 1000, 100).await;
    // Tier 1: 150000 input tokens (>128k threshold)
    wiremock_stubs::stub_openai_chat(&mock, "gemini-1.5-pro", 150_000, 100).await;

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

    let body = serde_json::json!({
        "model": "gemini-1.5-pro",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response_tier0 = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    let response_tier1 = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response_tier0.assert_status(StatusCode::OK);
    response_tier1.assert_status(StatusCode::OK);

    let cost_tier0: f64 = response_tier0
        .headers()
        .get(CostHeader::REQUEST_COST)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .expect("request cost header present");
    let cost_tier1: f64 = response_tier1
        .headers()
        .get(CostHeader::REQUEST_COST)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .expect("request cost header present");

    // Tier 0: 1000*1.25/1M + 100*5/1M ≈ 0.00175. Tier 1: 150000*2.5/1M + 100*10/1M ≈ 0.376
    assert!(
        cost_tier1 > 0.3 && cost_tier1 < 0.5,
        "tier-1 cost must reflect higher rate: cost={cost_tier1}"
    );
    assert!(
        cost_tier1 > cost_tier0 * 100.0,
        "tier-1 cost must be far higher than tier-0: tier0={cost_tier0}, tier1={cost_tier1}"
    );
}
