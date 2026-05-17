// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for budget soft-cap middleware.

use std::sync::Arc;

use axum::http::StatusCode;
use bytes::Bytes;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::{StreamStubAdapter, StubAdapter};
use crate::common::wiremock_stubs;
use oxigate::api::CHAT_COMPLETIONS_PATH;
use oxigate::config::OpenAICompatConfig;
use oxigate::config::{AuthConfig, BudgetConfig, RedisConfig, SecretString};
use oxigate::domain::chat::{StreamChunk, Usage};
use oxigate::providers::{CompatHttpClient, OpenAICompatAdapter};
use oxigate::redis_pool::create_pool;
use oxigate::utils::CostHeader;

// Matches RequestIdentity::default() key path used by auth-disabled test flows.
const DEFAULT_SPEND_KEY: &str = "oxigate:org:default:spend:default";
// Identity dedup keys include "identity:" segment prefix.
const DEFAULT_WARNED_KEY_PREFIX: &str = "oxigate:budget:warned:default:identity:default:";

#[tokio::test]
async fn test_budget_remaining_header_on_non_streaming_response() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    let budget = BudgetConfig {
        soft_cap_usd: None,
        hard_cap_usd: Some(10.0),
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

    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_SPEND_KEY)
        .arg(2_500_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed spend key");

    let response = gateway.server.get("/v1/models").await;
    response.assert_status(StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("7.500000")
    );
}

#[tokio::test]
async fn test_budget_remaining_header_on_streaming_response() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let chunks = vec![Ok(StreamChunk::new(
        Bytes::from("data: {\"choices\":[]}\n\n"),
        Some(Usage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            ..Default::default()
        }),
        Some("gpt-4.1".to_string()),
    ))];
    let provider = Arc::new(StreamStubAdapter::new(chunks));
    let budget = BudgetConfig {
        soft_cap_usd: None,
        hard_cap_usd: Some(10.0),
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

    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_SPEND_KEY)
        .arg(9_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed spend key");

    let body = serde_json::json!({
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    });
    let response = gateway.server.post(CHAT_COMPLETIONS_PATH).json(&body).await;
    response.assert_status(StatusCode::OK);
    assert!(
        response.content_type().contains("text/event-stream"),
        "expected SSE response content type"
    );
    assert_eq!(
        response
            .headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("1.000000")
    );
}

#[tokio::test]
async fn test_budget_is_retrospective_next_request_observes_previous_spend() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_chat(&mock, "gpt-4.1", 20, 30).await;
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
    let budget = BudgetConfig {
        soft_cap_usd: None,
        hard_cap_usd: Some(100.0),
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

    let body = serde_json::json!({
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let first = gateway.server.post(CHAT_COMPLETIONS_PATH).json(&body).await;
    first.assert_status(StatusCode::OK);
    assert_eq!(
        first
            .headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("100.000000")
    );

    let mut observed = None;
    for _ in 0..20 {
        let mut conn = redis.pool.get().await.expect("redis conn");
        let spend: Option<u64> = redis::cmd("GET")
            .arg(DEFAULT_SPEND_KEY)
            .query_async(&mut *conn)
            .await
            .expect("redis GET");
        if let Some(value) = spend
            && value > 0
        {
            observed = Some(value);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(
        observed.is_some(),
        "spend write should update Redis counter"
    );

    let second = gateway.server.get("/v1/models").await;
    second.assert_status(StatusCode::OK);
    let remaining = second
        .headers()
        .get(CostHeader::BUDGET_REMAINING)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok())
        .expect("X-Oxigate-Budget-Remaining should parse");
    assert!(
        remaining < 100.0,
        "second request should see spent budget from previous request"
    );
}

#[tokio::test]
async fn test_budget_fail_open_when_redis_unavailable() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let bad_redis_cfg = RedisConfig {
        url: SecretString::new("redis://127.0.0.1:19999"),
        pool_size: Some(1),
        pool_timeout_secs: Some(1),
    };
    let bad_redis = create_pool(&bad_redis_cfg).expect("pool struct should still construct");

    let provider = Arc::new(StubAdapter::new());
    let budget = BudgetConfig {
        soft_cap_usd: None,
        hard_cap_usd: Some(10.0),
        ..BudgetConfig::default()
    };
    let gateway = TestGateway::spawn_with_budget(
        pg.pool.clone(),
        bad_redis,
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    let response = gateway.server.get("/v1/models").await;
    response.assert_status(StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("10.000000")
    );
}

#[tokio::test]
async fn test_budget_warn_dedup_is_atomic_under_concurrent_requests() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    let budget = BudgetConfig {
        soft_cap_usd: Some(10.0),
        hard_cap_usd: None,
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

    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_SPEND_KEY)
        .arg(10_500_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed spend key");

    let (first, second) = tokio::join!(async { gateway.server.get("/v1/models").await }, async {
        gateway.server.get("/v1/models").await
    });
    first.assert_status(StatusCode::OK);
    second.assert_status(StatusCode::OK);

    let keys: Vec<String> = redis::cmd("KEYS")
        .arg(format!("{DEFAULT_WARNED_KEY_PREFIX}*"))
        .query_async(&mut *conn)
        .await
        .expect("list warned keys");
    assert_eq!(keys.len(), 3, "expected one warned key per threshold");

    for threshold in [80_u8, 90_u8, 100_u8] {
        let key = format!("{DEFAULT_WARNED_KEY_PREFIX}{threshold}");
        let exists: u8 = redis::cmd("EXISTS")
            .arg(&key)
            .query_async(&mut *conn)
            .await
            .expect("check warned key exists");
        assert_eq!(
            exists, 1,
            "warned key should exist for threshold {threshold}"
        );

        let ttl: i64 = redis::cmd("TTL")
            .arg(&key)
            .query_async(&mut *conn)
            .await
            .expect("read warned key ttl");
        assert!(
            ttl > 0,
            "warned key should have positive ttl for threshold {threshold}"
        );
    }
}

/// Full AC path: seed spend >= hard_cap → 429 with CostHeader::BUDGET_REMAINING: 0.000000.
#[tokio::test]
async fn test_hard_cap_returns_429_when_exceeded() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    let budget = BudgetConfig {
        soft_cap_usd: None,
        hard_cap_usd: Some(10.0),
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

    // Seed spend at exactly the hard cap (10 USD = 10_000_000_000 nano USD).
    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_SPEND_KEY)
        .arg(10_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed spend key");

    let response = gateway.server.get("/v1/models").await;
    response.assert_status(StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("0.000000")
    );
}

/// Header non-overwrite: when hard_cap fires (429), BudgetResponseLayer must not
/// overwrite CostHeader::BUDGET_REMAINING with a soft-cap-based value.
///
/// Setup: soft_cap=15, hard_cap=10 (hard fires first). Spend at hard_cap.
/// If BudgetResponseLayer ran, remaining = soft_cap - spend = 5.0 → "5.000000".
/// Correct behavior: HardCapLayer short-circuits before BudgetResponseLayer runs → "0.000000".
#[tokio::test]
async fn test_hard_cap_header_not_overwritten_by_budget_response_layer() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    let budget = BudgetConfig {
        soft_cap_usd: Some(15.0), // higher than hard_cap — fires last
        hard_cap_usd: Some(10.0),
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

    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_SPEND_KEY)
        .arg(10_000_000_000_u64) // at hard cap, 5 USD below soft cap
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed spend key");

    let response = gateway.server.get("/v1/models").await;
    response.assert_status(StatusCode::TOO_MANY_REQUESTS);
    // Must be "0.000000", not "5.000000" (which would indicate BudgetResponseLayer ran).
    assert_eq!(
        response
            .headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("0.000000")
    );
}

/// Only hard_cap_usd set (no soft_cap_usd): non-blocked request still gets
/// CostHeader::BUDGET_REMAINING via effective_response_cap_nano_usd fallback to hard_cap.
#[tokio::test]
async fn test_hard_cap_only_non_blocked_request_has_budget_remaining_header() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    let budget = BudgetConfig {
        soft_cap_usd: None, // no soft cap — hard cap is the only cap
        hard_cap_usd: Some(10.0),
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

    // Seed spend below hard cap.
    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_SPEND_KEY)
        .arg(2_500_000_000_u64) // 2.5 USD spent
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed spend key");

    let response = gateway.server.get("/v1/models").await;
    response.assert_status(StatusCode::OK);
    // effective_response_cap = hard_cap = 10 USD. remaining = 10 - 2.5 = 7.5
    assert_eq!(
        response
            .headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("7.500000")
    );
}

#[tokio::test]
async fn test_soft_and_hard_configured_header_shows_remaining_to_hard_cap() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    let budget = BudgetConfig {
        soft_cap_usd: Some(10.0),
        hard_cap_usd: Some(15.0),
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

    // Seed spend between soft and hard caps: $12 (warn zone, but not blocked).
    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_SPEND_KEY)
        .arg(12_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed spend key");

    let response = gateway.server.get("/v1/models").await;
    response.assert_status(StatusCode::OK);
    let remaining = response
        .headers()
        .get("X-Oxigate-Budget-Remaining")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok())
        .expect("X-Oxigate-Budget-Remaining must parse");
    // Remaining to hard cap: 15 - 12 = 3.
    assert!(
        (remaining - 3.0).abs() < 0.01,
        "expected ~$3.0 remaining to hard cap, got {remaining}"
    );
}
