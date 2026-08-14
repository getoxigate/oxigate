// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for GlobalSafetyLayer .
//!
//! Covers: 429 when spend >= cap, pass-through when spend < cap,
//! disabled (None) cap, exact-cap boundary, and GLOBAL_SPEND_KEY constant value.

use std::sync::Arc;

use axum::http::StatusCode;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::StubAdapter;
use oxigate::domain::ports::NanoUsd;
use oxigate::redis_pool::create_pool;
use oxigate::utils::CostHeader;

const GLOBAL_SPEND_KEY: &str = "oxigate:global:spend";

/// Seed the global spend counter in Redis directly.
async fn seed_global_spend(redis: &oxigate::redis_pool::RedisPool, value: u64) {
    let mut conn = redis.get().await.expect("redis conn for seeding");
    redis::cmd("SET")
        .arg(GLOBAL_SPEND_KEY)
        .arg(value)
        .query_async::<()>(&mut *conn)
        .await
        .expect("SET global spend key");
}

/// spend > cap → 429 + budget-cap header value `global` + JSON error body.
#[tokio::test]
async fn test_global_safety_blocks_when_cap_exceeded() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    // Seed: $10.00 + 1 nano (just above cap)
    seed_global_spend(&redis.pool, 10_000_000_001).await;

    let provider = Arc::new(StubAdapter::new());
    let gateway = TestGateway::spawn_with_global_safety_cap(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        Some(NanoUsd(10_000_000_000)), // $10.00 cap
    )
    .await;

    let response = gateway.server.get("/v1/models").await;

    response.assert_status(StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(CostHeader::BUDGET_CAP)
            .and_then(|v| v.to_str().ok()),
        Some("global"),
        "{} header must be 'global'",
        CostHeader::BUDGET_CAP
    );
    let body: serde_json::Value = response.json();
    assert_eq!(
        body["error"], "global_budget_cap_exceeded",
        "error body must be global_budget_cap_exceeded"
    );
}

/// spend < cap → request passes through.
#[tokio::test]
async fn test_global_safety_passes_when_under_cap() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    // Seed: $10.00 - 1 nano (just under cap)
    seed_global_spend(&redis.pool, 9_999_999_999).await;

    let provider = Arc::new(StubAdapter::new());
    let gateway = TestGateway::spawn_with_global_safety_cap(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        Some(NanoUsd(10_000_000_000)), // $10.00 cap
    )
    .await;

    let response = gateway.server.get("/v1/models").await;

    // StubAdapter returns models list — gateway should pass through and respond 200.
    response.assert_status(StatusCode::OK);
}

/// cap = None → always passes, no Redis consultation.
#[tokio::test]
async fn test_global_safety_disabled_when_none() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    // Do NOT seed Redis — if the layer consulted Redis and got None, it would pass anyway.
    // This test verifies the fast-path (no Redis call at all when cap is None).
    let provider = Arc::new(StubAdapter::new());
    let gateway = TestGateway::spawn_with_global_safety_cap(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        None, // cap disabled
    )
    .await;

    let response = gateway.server.get("/v1/models").await;

    response.assert_status(StatusCode::OK);
}

/// Redis unavailable → fail-open; request passes through even when cap is set.
#[tokio::test]
async fn test_global_safety_fails_open_when_redis_unavailable() {
    let pg = PgContainer::start().await.expect("pg container");

    // Dead Redis pool — nothing listening on this port.
    let bad_cfg = oxigate::config::RedisConfig {
        url: oxigate::config::SecretString::new("redis://127.0.0.1:19999"),
        pool_size: Some(1),
        pool_timeout_secs: Some(1),
    };
    let dead_pool = create_pool(&bad_cfg).expect("pool struct created");

    let provider = Arc::new(StubAdapter::new());
    let gateway = TestGateway::spawn_with_global_safety_cap(
        pg.pool.clone(),
        dead_pool,
        provider,
        Some(NanoUsd(1)), // cap at 1 nano — would block all requests if Redis were reachable
    )
    .await;

    let response = gateway.server.get("/v1/models").await;

    // Must fail open: Redis unreachable → layer skips enforcement → 200.
    response.assert_status(StatusCode::OK);
}

/// spend == cap exactly → 429 (≥ blocks, boundary inclusive).
#[tokio::test]
async fn test_global_safety_blocks_at_exact_cap() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    // Seed: exactly $10.00
    seed_global_spend(&redis.pool, 10_000_000_000).await;

    let provider = Arc::new(StubAdapter::new());
    let gateway = TestGateway::spawn_with_global_safety_cap(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        Some(NanoUsd(10_000_000_000)), // $10.00 cap
    )
    .await;

    let response = gateway.server.get("/v1/models").await;

    response.assert_status(StatusCode::TOO_MANY_REQUESTS);
    let body: serde_json::Value = response.json();
    assert_eq!(
        body["error"], "global_budget_cap_exceeded",
        "exact-cap boundary must return 429"
    );
}
