// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for GET /v1/spend/{daily,providers,models} .
//!
//! All tests use `TestGateway::spawn()` (auth bypass → RequestIdentity::default()
//! → org_id = "default"). Spend rows are seeded via direct `sqlx::query!` INSERT
//! with an explicit `created_at` to control which window they fall into.

use std::sync::Arc;

use axum::http::StatusCode;
use chrono::{DateTime, Duration, Utc};

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::StubAdapter;
use oxigate::db::DbPool;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Insert a single spend row with a caller-supplied `created_at` timestamp.
/// All other fields are filled with deterministic defaults so tests can focus
/// on the dimension they care about.
async fn seed_spend(
    pool: &DbPool,
    org_id: &str,
    provider: &str,
    model: &str,
    cost_nano_usd: i64,
    created_at: DateTime<Utc>,
) {
    sqlx::query(
        r#"
        INSERT INTO spend_records
            (org_id, identity_id, model, provider,
             prompt_tokens, completion_tokens,
             cache_read_tokens, cache_write_5m_tokens, cache_write_1h_tokens, thinking_tokens,
             cost_nano_usd, latency_ms, tags, created_at)
        VALUES ($1, 'test-key', $2, $3, 10, 5, 0, 0, 0, 0, $4, 10, '{}', $5)
        "#,
    )
    .bind(org_id)
    .bind(model)
    .bind(provider)
    .bind(cost_nano_usd)
    .bind(created_at)
    .execute(pool)
    .await
    .expect("seed_spend: INSERT failed");
}

// ---------------------------------------------------------------------------
// daily aggregation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_daily_spend_returns_aggregated_rows() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");
    let gateway = TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
    )
    .await;

    let today = Utc::now()
        .date_naive()
        .and_time(chrono::NaiveTime::MIN)
        .and_utc();
    // Two rows on the same day — should be summed into a single entry.
    seed_spend(
        &pg.pool,
        "default",
        "openai",
        "gpt-4.1",
        1_000_000_000,
        today,
    )
    .await;
    seed_spend(&pg.pool, "default", "openai", "gpt-4.1", 500_000_000, today).await;

    let response = gateway.server.get("/v1/spend/daily").await;
    response.assert_status(StatusCode::OK);

    let body: serde_json::Value = response.json();
    let data = body["data"].as_array().expect("data must be array");
    assert_eq!(
        data.len(),
        1,
        "two rows on same day must aggregate into one entry"
    );
    assert_eq!(data[0]["cost_nano_usd"], 1_500_000_000_i64);
}

// ---------------------------------------------------------------------------
// provider aggregation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_spend_by_provider_groups_correctly() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");
    let gateway = TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
    )
    .await;

    let now = Utc::now();
    seed_spend(&pg.pool, "default", "openai", "gpt-4.1", 2_000_000_000, now).await;
    seed_spend(
        &pg.pool,
        "default",
        "anthropic",
        "claude-3-5-sonnet",
        1_000_000_000,
        now,
    )
    .await;

    let response = gateway.server.get("/v1/spend/providers").await;
    response.assert_status(StatusCode::OK);

    let body: serde_json::Value = response.json();
    let data = body["data"].as_array().expect("data must be array");
    assert_eq!(data.len(), 2);

    let anthropic = data
        .iter()
        .find(|e| e["dimension"] == "anthropic")
        .expect("anthropic entry must be present");
    assert_eq!(anthropic["cost_nano_usd"], 1_000_000_000_i64);

    let openai = data
        .iter()
        .find(|e| e["dimension"] == "openai")
        .expect("openai entry must be present");
    assert_eq!(openai["cost_nano_usd"], 2_000_000_000_i64);
}

// ---------------------------------------------------------------------------
// model aggregation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_spend_by_model_groups_correctly() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");
    let gateway = TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
    )
    .await;

    let now = Utc::now();
    seed_spend(&pg.pool, "default", "openai", "gpt-4.1", 3_000_000_000, now).await;
    seed_spend(
        &pg.pool,
        "default",
        "openai",
        "gpt-4o-mini",
        500_000_000,
        now,
    )
    .await;

    let response = gateway.server.get("/v1/spend/models").await;
    response.assert_status(StatusCode::OK);

    let body: serde_json::Value = response.json();
    let data = body["data"].as_array().expect("data must be array");
    assert_eq!(data.len(), 2);

    let gpt4 = data
        .iter()
        .find(|e| e["dimension"] == "gpt-4.1")
        .expect("gpt-4.1 must be present");
    assert_eq!(gpt4["cost_nano_usd"], 3_000_000_000_i64);
}

// ---------------------------------------------------------------------------
// default window is last 30 days
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_default_window_is_last_30_days() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");
    let gateway = TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
    )
    .await;

    let now = Utc::now();
    let sixty_days_ago = now - Duration::days(60);
    let today = now;

    // Row 60 days ago — outside the 30-day default window.
    seed_spend(
        &pg.pool,
        "default",
        "openai",
        "gpt-4.1",
        999_999_999,
        sixty_days_ago,
    )
    .await;
    // Row today — inside the window.
    seed_spend(
        &pg.pool,
        "default",
        "openai",
        "gpt-4.1",
        1_000_000_000,
        today,
    )
    .await;

    let response = gateway.server.get("/v1/spend/daily").await;
    response.assert_status(StatusCode::OK);

    let body: serde_json::Value = response.json();
    let data = body["data"].as_array().expect("data must be array");
    assert_eq!(
        data.len(),
        1,
        "only the row within the 30-day window must appear"
    );
    assert_eq!(data[0]["cost_nano_usd"], 1_000_000_000_i64);
}

// ---------------------------------------------------------------------------
// empty result when no records
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_empty_result_when_no_records() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");
    let gateway = TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
    )
    .await;

    for path in &["/v1/spend/daily", "/v1/spend/providers", "/v1/spend/models"] {
        let response = gateway.server.get(path).await;
        response.assert_status(StatusCode::OK);
        let body: serde_json::Value = response.json();
        let data = body["data"].as_array().expect("data must be array");
        assert!(
            data.is_empty(),
            "{path}: data must be empty when no records exist"
        );
    }
}

// ---------------------------------------------------------------------------
// invalid from format → 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_invalid_from_format_returns_400() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");
    let gateway = TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
    )
    .await;

    let response = gateway
        .server
        .get("/v1/spend/daily")
        .add_query_params([("from", "not-a-date")])
        .await;

    response.assert_status(StatusCode::BAD_REQUEST);
    let body: serde_json::Value = response.json();
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("invalid date format")
    );
}

// ---------------------------------------------------------------------------
// invalid to format → 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_invalid_to_format_returns_400() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");
    let gateway = TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
    )
    .await;

    let response = gateway
        .server
        .get("/v1/spend/daily")
        .add_query_params([("to", "2025-13-01")])
        .await;

    response.assert_status(StatusCode::BAD_REQUEST);
    let body: serde_json::Value = response.json();
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("invalid date format")
    );
}

// ---------------------------------------------------------------------------
// range > 365 days → 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_range_over_365_days_returns_400() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");
    let gateway = TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
    )
    .await;

    let response = gateway
        .server
        .get("/v1/spend/daily")
        .add_query_params([("from", "2020-01-01"), ("to", "2021-12-31")])
        .await;

    response.assert_status(StatusCode::BAD_REQUEST);
    let body: serde_json::Value = response.json();
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("invalid date range")
    );
}

// ---------------------------------------------------------------------------
// from after to → 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_from_after_to_returns_400() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");
    let gateway = TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
    )
    .await;

    let response = gateway
        .server
        .get("/v1/spend/daily")
        .add_query_params([("from", "2025-06-01"), ("to", "2025-01-01")])
        .await;

    response.assert_status(StatusCode::BAD_REQUEST);
    let body: serde_json::Value = response.json();
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("invalid date range")
    );
}

// ---------------------------------------------------------------------------
// tenant isolation — org_a rows not visible when querying as org_b
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_tenant_isolation() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");
    // TestGateway::spawn() uses auth bypass → RequestIdentity::default() → org_id = "default"
    let gateway = TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
    )
    .await;

    // Seed rows for a different org — must not appear in the "default" org's response.
    let now = Utc::now();
    seed_spend(
        &pg.pool,
        "other-org",
        "openai",
        "gpt-4.1",
        9_999_999_999,
        now,
    )
    .await;

    for path in &["/v1/spend/daily", "/v1/spend/providers", "/v1/spend/models"] {
        let response = gateway.server.get(path).await;
        response.assert_status(StatusCode::OK);
        let body: serde_json::Value = response.json();
        let data = body["data"].as_array().expect("data must be array");
        assert!(
            data.is_empty(),
            "{path}: other-org rows must not appear for the 'default' org"
        );
    }
}

// ---------------------------------------------------------------------------
// explicit date range excludes rows outside [from, to]
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_explicit_date_range_excludes_outside_rows() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");
    let gateway = TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
    )
    .await;

    // Row inside the explicit range.
    let inside = chrono::NaiveDate::from_ymd_opt(2025, 6, 15)
        .unwrap()
        .and_time(chrono::NaiveTime::MIN)
        .and_utc();
    // Row outside (before from).
    let before = chrono::NaiveDate::from_ymd_opt(2025, 5, 31)
        .unwrap()
        .and_time(chrono::NaiveTime::MIN)
        .and_utc();
    // Row outside (after to).
    let after = chrono::NaiveDate::from_ymd_opt(2025, 7, 1)
        .unwrap()
        .and_time(chrono::NaiveTime::MIN)
        .and_utc();

    seed_spend(
        &pg.pool,
        "default",
        "openai",
        "gpt-4.1",
        1_000_000_000,
        inside,
    )
    .await;
    seed_spend(
        &pg.pool,
        "default",
        "openai",
        "gpt-4.1",
        999_000_000,
        before,
    )
    .await;
    seed_spend(&pg.pool, "default", "openai", "gpt-4.1", 888_000_000, after).await;

    let response = gateway
        .server
        .get("/v1/spend/daily")
        .add_query_params([("from", "2025-06-01"), ("to", "2025-06-30")])
        .await;

    response.assert_status(StatusCode::OK);
    let body: serde_json::Value = response.json();
    let data = body["data"].as_array().expect("data must be array");
    assert_eq!(data.len(), 1, "only the row inside [from, to] must appear");
    assert_eq!(data[0]["cost_nano_usd"], 1_000_000_000_i64);
}
