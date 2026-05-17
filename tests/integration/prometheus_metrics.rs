// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for the Prometheus metrics endpoint .
//!
//! Verifies:
//! - `GET /metrics` returns 200 with valid Prometheus text when a handle is present.
//! - Required metric families appear in scrape output after being emitted.
//! - Auth bypass: `/metrics` accessible without Authorization; `/v1/*` returns 401 when key is set.
//! - Provider label values are stable lowercase strings.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::http::StatusCode;
use metrics_exporter_prometheus::PrometheusHandle;
use sqlx::postgres::PgPoolOptions;

use oxigate::api::{CHAT_COMPLETIONS_PATH, router_with_metrics};
use oxigate::config::{
    AuthConfig, BudgetConfig, PricingConfig, RedisConfig, SecretString, SecurityConfig,
};
use oxigate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
use oxigate::middleware::global_safety::GlobalSafetyRuntimeConfig;
use oxigate::observability::metrics::{
    ACTIVE_CONNECTIONS, COST_USD_TOTAL, FALLBACK_RESOLUTION_ATTEMPTS, FALLBACK_RESOLUTION_SECONDS,
    FALLBACK_SKIP_TOTAL, FALLBACK_TRIGGER_TOTAL, REQUEST_DURATION_SECONDS, REQUESTS_TOTAL,
    RETRY_ATTEMPT_TOTAL,
};
use oxigate::providers::ProviderHealthTracker;
use oxigate::redis_pool::create_pool;

use crate::common::stub_adapter::StubAdapter;

// ---------------------------------------------------------------------------
// Shared recorder — can only be installed once per process.
// ---------------------------------------------------------------------------

fn test_prometheus_handle() -> PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE
        .get_or_init(|| {
            metrics_exporter_prometheus::PrometheusBuilder::new()
                .install_recorder()
                .expect("prometheus recorder must install once in test process")
        })
        .clone()
}

// ---------------------------------------------------------------------------
// Minimal lazy pools — metrics endpoint does not touch DB or Redis.
// ---------------------------------------------------------------------------

fn lazy_pg_pool() -> oxigate::db::DbPool {
    PgPoolOptions::new()
        .acquire_timeout(Duration::from_millis(100))
        .connect_lazy("postgres://postgres:postgres@127.0.0.1:1/postgres")
        .expect("lazy PG pool must build")
}

fn lazy_redis_pool() -> oxigate::redis_pool::RedisPool {
    create_pool(&RedisConfig {
        url: SecretString::new("redis://127.0.0.1:1"),
        pool_size: Some(1),
        pool_timeout_secs: Some(1),
    })
    .expect("lazy Redis pool must build")
}

/// Builds a test AppState with lazy (non-connecting) pools and a StubAdapter.
fn metrics_test_app_state(auth: AuthConfig) -> oxigate::api::AppState {
    let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
        .expect("bundled pricing DB must parse");
    oxigate::api::AppState {
        pool: Arc::new(tokio::sync::RwLock::new(lazy_pg_pool())),
        redis_pool: Arc::new(tokio::sync::RwLock::new(lazy_redis_pool())),
        pricing_db: Arc::new(std::sync::RwLock::new(pricing_db)),
        provider: Arc::new(tokio::sync::RwLock::new(
            Arc::new(StubAdapter::default()) as Arc<dyn oxigate::domain::ports::ProviderAdapterExt>
        )),
        auth: Arc::new(tokio::sync::RwLock::new(auth)),
        global_safety: Arc::new(tokio::sync::RwLock::new(
            GlobalSafetyRuntimeConfig::default(),
        )),
        budget_settings: Arc::new(tokio::sync::RwLock::new(BudgetConfig::default())),
        budget: Arc::new(tokio::sync::RwLock::new(
            oxigate::middleware::budget::BudgetRuntimeConfig::default(),
        )),
        startup_time: 1,
        health: ProviderHealthTracker::new_for_test(&[]),
        security: Arc::new(tokio::sync::RwLock::new(SecurityConfig::default())),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `GET /metrics` returns 200 when a PrometheusHandle Extension is present.
#[tokio::test]
async fn test_metrics_endpoint_returns_200() {
    let handle = test_prometheus_handle();
    let state = metrics_test_app_state(AuthConfig::default());
    let app = router_with_metrics(state, handle);
    let server = axum_test::TestServer::new(app).expect("TestServer must build");

    let response = server.get("/metrics").await;
    assert_eq!(
        response.status_code(),
        StatusCode::OK,
        "GET /metrics must return 200"
    );
}

/// Scrape output contains all required metric families after they have been emitted.
///
/// Since we cannot make a real LLM request in this test, we directly emit each metric via
/// the shared global recorder and verify the names appear in the scrape output.
#[tokio::test]
async fn test_metrics_output_contains_required_metric_families() {
    let handle = test_prometheus_handle();

    // Emit each baseline metric so it appears in the registry even if no real request has run.
    metrics::counter!(REQUESTS_TOTAL, "method" => "POST", "status" => "200", "provider" => "test")
        .increment(1);
    metrics::histogram!(REQUEST_DURATION_SECONDS, "provider" => "test").record(0.1);
    metrics::counter!(COST_USD_TOTAL, "provider" => "test").increment(1_000_000);
    metrics::gauge!(ACTIVE_CONNECTIONS).set(0.0);
    // fallback metrics
    metrics::counter!(FALLBACK_TRIGGER_TOTAL, "trigger" => "timeout").increment(1);
    metrics::counter!(FALLBACK_SKIP_TOTAL, "reason" => "in_cooldown").increment(1);
    metrics::counter!(RETRY_ATTEMPT_TOTAL, "provider" => "test", "trigger" => "timeout")
        .increment(1);
    metrics::histogram!(FALLBACK_RESOLUTION_SECONDS).record(0.0);
    metrics::histogram!(FALLBACK_RESOLUTION_ATTEMPTS).record(0.0);

    let state = metrics_test_app_state(AuthConfig::default());
    let app = router_with_metrics(state, handle);
    let server = axum_test::TestServer::new(app).expect("TestServer must build");

    let response = server.get("/metrics").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body = response.text();

    for metric in [
        REQUESTS_TOTAL,
        REQUEST_DURATION_SECONDS,
        COST_USD_TOTAL,
        ACTIVE_CONNECTIONS,
        FALLBACK_TRIGGER_TOTAL,
        FALLBACK_SKIP_TOTAL,
        RETRY_ATTEMPT_TOTAL,
        FALLBACK_RESOLUTION_SECONDS,
        FALLBACK_RESOLUTION_ATTEMPTS,
    ] {
        assert!(
            body.contains(metric),
            "scrape output must contain metric {metric:?}; got:\n{body}"
        );
    }
}

/// `GET /metrics` returns 503 when no PrometheusHandle Extension is present (base router).
#[tokio::test]
async fn test_metrics_endpoint_returns_503_without_handle() {
    let state = metrics_test_app_state(AuthConfig::default());
    let app = oxigate::api::router(state);
    let server = axum_test::TestServer::new(app).expect("TestServer must build");

    let response = server.get("/metrics").await;
    assert_eq!(
        response.status_code(),
        StatusCode::SERVICE_UNAVAILABLE,
        "GET /metrics without handle must return 503"
    );
}

/// Auth bypass: `GET /metrics` without Authorization returns 200; `POST /v1/*` without
/// Authorization returns 401 when an auth key is configured.
#[tokio::test]
async fn test_metrics_bypasses_auth() {
    let handle = test_prometheus_handle();
    let auth = AuthConfig {
        key: Some(SecretString::from("secret-key")),
    };
    let state = metrics_test_app_state(auth);
    let app = router_with_metrics(state, handle);
    let server = axum_test::TestServer::new(app).expect("TestServer must build");

    // /metrics — no auth header — must be 200
    let metrics_resp = server.get("/metrics").await;
    assert_eq!(
        metrics_resp.status_code(),
        StatusCode::OK,
        "/metrics must bypass auth and return 200"
    );

    // /v1/chat/completions — no auth header — must be 401
    let v1_resp = server
        .post(CHAT_COMPLETIONS_PATH)
        .json(&serde_json::json!({"model": "gpt-4", "messages": []}))
        .await;
    assert_eq!(
        v1_resp.status_code(),
        StatusCode::UNAUTHORIZED,
        "/v1/chat/completions without auth header must return 401"
    );
}

/// Provider names that flow into the `provider` label must be stable lowercase strings.
///
/// Values come from `ProviderAdapter::metadata().name`. Verify the contract holds for
/// the test stub adapter used in this test module.
#[test]
fn test_provider_label_values_are_lowercase() {
    use oxigate::domain::ports::ProviderAdapter;
    let stub = StubAdapter::default();
    let name = stub.metadata().name.clone();
    assert_eq!(
        name,
        name.to_lowercase(),
        "provider name {name:?} must be stable lowercase to satisfy label cardinality contract"
    );
}
