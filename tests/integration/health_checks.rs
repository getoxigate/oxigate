// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for health endpoints .

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use sqlx::postgres::PgPoolOptions;

use oxigate::config::{AuthConfig, RedisConfig, SecretString};
use oxigate::db::DbPool;
use oxigate::domain::ports::HealthStatus;
use oxigate::providers::ProviderHealthTracker;
use oxigate::redis_pool::{RedisPool, create_pool, health_check as redis_health_check};

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::StubAdapter;

fn unavailable_db_pool() -> DbPool {
    PgPoolOptions::new()
        .acquire_timeout(Duration::from_millis(100))
        .connect_lazy("postgres://postgres:postgres@127.0.0.1:1/postgres")
        .expect("lazy PG pool must build for unreachable URL")
}

fn unavailable_redis_pool() -> RedisPool {
    create_pool(&RedisConfig {
        url: SecretString::new("redis://127.0.0.1:1"),
        pool_size: Some(1),
        pool_timeout_secs: Some(1),
    })
    .expect("lazy Redis pool must build for unreachable URL")
}

fn providers_all_available() -> Arc<ProviderHealthTracker> {
    ProviderHealthTracker::new_for_test(&["compat-test"])
}

async fn wait_for_redis_ready(pool: &RedisPool) {
    for _ in 0..20 {
        if redis_health_check(pool).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("redis did not become healthy within 2 seconds");
}

#[tokio::test]
async fn test_health_live_returns_200() {
    let gateway = TestGateway::spawn(
        unavailable_db_pool(),
        unavailable_redis_pool(),
        Arc::new(StubAdapter::new()),
    )
    .await;

    let response = gateway.server.get("/health").await;
    response.assert_status(StatusCode::OK);
    response.assert_json(&serde_json::json!({"status": "ok"}));
}

#[tokio::test]
async fn test_health_ready_returns_200_when_dependencies_are_healthy() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    wait_for_redis_ready(&redis.pool).await;
    let gateway = TestGateway::spawn_with_health_tracker(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        AuthConfig::default(),
        providers_all_available(),
    )
    .await;

    let response = gateway.server.get("/health/ready").await;
    response.assert_status(StatusCode::OK);
    response.assert_json(&serde_json::json!({"status": "ok"}));
}

#[tokio::test]
async fn test_health_ready_single_healthy_provider() {
    // A single provider reporting Healthy should keep the providers dimension at "ok".
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    wait_for_redis_ready(&redis.pool).await;
    let gateway = TestGateway::spawn_with_health_tracker(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        AuthConfig::default(),
        providers_all_available(),
    )
    .await;

    let response = gateway.server.get("/health/ready").await;
    response.assert_status(StatusCode::OK);
    response.assert_json(&serde_json::json!({"status": "ok"}));
}

#[tokio::test]
async fn test_health_ready_returns_503_when_postgres_unreachable() {
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    wait_for_redis_ready(&redis.pool).await;
    let gateway = TestGateway::spawn_with_health_tracker(
        unavailable_db_pool(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        AuthConfig::default(),
        providers_all_available(),
    )
    .await;

    let response = gateway.server.get("/health/ready").await;
    response.assert_status(StatusCode::SERVICE_UNAVAILABLE);
    let body: serde_json::Value = response.json();
    assert_eq!(body["status"], "degraded");
    assert_eq!(body["checks"]["postgres"], "unreachable");
    assert_eq!(body["checks"]["redis"], "ok");
    assert_eq!(body["checks"]["providers"], "ok");
}

#[tokio::test]
async fn test_health_ready_returns_503_when_redis_unreachable() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let gateway = TestGateway::spawn_with_health_tracker(
        pg.pool.clone(),
        unavailable_redis_pool(),
        Arc::new(StubAdapter::new()),
        AuthConfig::default(),
        providers_all_available(),
    )
    .await;

    let response = gateway.server.get("/health/ready").await;
    response.assert_status(StatusCode::SERVICE_UNAVAILABLE);
    let body: serde_json::Value = response.json();
    assert_eq!(body["status"], "degraded");
    assert_eq!(body["checks"]["postgres"], "ok");
    assert_eq!(body["checks"]["redis"], "unreachable");
    assert_eq!(body["checks"]["providers"], "ok");
}

#[tokio::test]
async fn test_health_ready_completes_within_1500ms_when_db_and_redis_are_down() {
    // Note: this exercises fail-fast pool acquisition for unreachable hosts.
    // It does not prove the 800ms tokio::time::timeout branch fires first.
    // The goal is bounded latency in the degraded path for CI-safe conditions.
    let gateway = TestGateway::spawn_with_health_tracker(
        unavailable_db_pool(),
        unavailable_redis_pool(),
        Arc::new(StubAdapter::new()),
        AuthConfig::default(),
        providers_all_available(),
    )
    .await;

    let start = Instant::now();
    let response = gateway.server.get("/health/ready").await;
    let elapsed = start.elapsed();

    response.assert_status(StatusCode::SERVICE_UNAVAILABLE);
    let body: serde_json::Value = response.json();
    assert_eq!(body["status"], "degraded");
    assert_eq!(body["checks"]["postgres"], "unreachable");
    assert_eq!(body["checks"]["redis"], "unreachable");
    assert!(
        elapsed <= Duration::from_millis(1500),
        "readiness probe exceeded 1500ms: {elapsed:?}"
    );
}

#[tokio::test]
async fn test_health_ready_returns_503_when_provider_startup_check_failed() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    wait_for_redis_ready(&redis.pool).await;
    // openai is Unhealthy; compat-test is Healthy → 1 unhealthy provider.
    let tracker = ProviderHealthTracker::new_for_test(&["openai", "compat-test"]);
    tracker
        .update_health("openai", HealthStatus::Unhealthy)
        .await;
    let gateway = TestGateway::spawn_with_health_tracker(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        AuthConfig::default(),
        tracker,
    )
    .await;

    let response = gateway.server.get("/health/ready").await;
    response.assert_status(StatusCode::SERVICE_UNAVAILABLE);
    let body: serde_json::Value = response.json();
    assert_eq!(body["status"], "degraded");
    assert_eq!(body["checks"]["postgres"], "ok");
    assert_eq!(body["checks"]["redis"], "ok");
    assert_eq!(body["checks"]["providers"], "1 provider(s) unhealthy");
}

#[tokio::test]
async fn test_health_routes_bypass_auth_with_key_configured() {
    let auth = AuthConfig {
        key: Some(SecretString::from("required-token")),
    };
    let gateway = TestGateway::spawn_with_health_tracker(
        unavailable_db_pool(),
        unavailable_redis_pool(),
        Arc::new(StubAdapter::new()),
        auth,
        providers_all_available(),
    )
    .await;

    let live = gateway.server.get("/health").await;
    live.assert_status(StatusCode::OK);

    let ready = gateway.server.get("/health/ready").await;
    ready.assert_status(StatusCode::SERVICE_UNAVAILABLE);
}
