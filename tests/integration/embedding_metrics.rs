// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Tests for embedding-specific Prometheus counters and the endpoint label on shared metrics.
//!
//! Verifies:
//! - EMBEDDINGS_TOTAL, EMBEDDINGS_DURATION_SECONDS, EMBEDDINGS_VECTORS_TOTAL fire on success.
//! - COST_USD_TOTAL{endpoint="embeddings"} fires only when usage data is present.
//! - EMBEDDINGS_TOTAL{status="error"} fires on provider error; cost/vector counters do not.
//! - EMBEDDINGS_TOTAL{status="success"} fires on no-usage response; vector/cost counters do not.
//! - REQUESTS_TOTAL and REQUEST_DURATION_SECONDS carry an endpoint label on all /v1/* routes.

use std::sync::Arc;

use axum::http::StatusCode;

use oxigate::api::{CHAT_COMPLETIONS_PATH, EMBEDDINGS_PATH, router_with_metrics};
use oxigate::config::{
    AuthConfig, BudgetConfig, OpenAIConfig, PricingConfig, SecretString, SecurityConfig,
};
use oxigate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
use oxigate::middleware::global_safety::GlobalSafetyRuntimeConfig;
use oxigate::observability::metrics::{
    COST_USD_TOTAL, EMBEDDINGS_DURATION_SECONDS, EMBEDDINGS_TOTAL, EMBEDDINGS_VECTORS_TOTAL,
    ENDPOINT_CHAT, ENDPOINT_EMBEDDINGS, ENDPOINT_OTHER, REQUESTS_TOTAL,
};
use oxigate::providers::ProviderHealthTracker;
use oxigate::providers::openai::OpenAiAdapter;

use crate::common::stub_adapter::StubAdapter;
use crate::common::wiremock_stubs;

// ---------------------------------------------------------------------------
// Serialisation lock
//
// All tests in this file share the same process-global Prometheus recorder.
// Running them concurrently produces racy deltas (two tests fire the same
// counter between one test's before/after snapshot).  A single mutex
// serialises all metric assertions without requiring an external crate.
// ---------------------------------------------------------------------------

fn metrics_test_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

// ---------------------------------------------------------------------------
// Infrastructure helpers
// ---------------------------------------------------------------------------

fn app_state_with_provider(
    provider: Arc<dyn oxigate::domain::ports::ProviderAdapterExt>,
) -> oxigate::api::AppState {
    let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
        .expect("bundled pricing DB must parse");
    oxigate::api::AppState {
        pool: Arc::new(tokio::sync::RwLock::new(crate::common::lazy_pg_pool())),
        redis_pool: Arc::new(tokio::sync::RwLock::new(crate::common::lazy_redis_pool())),
        pricing_db: Arc::new(std::sync::RwLock::new(pricing_db)),
        provider: Arc::new(tokio::sync::RwLock::new(provider)),
        auth: Arc::new(tokio::sync::RwLock::new(AuthConfig::default())),
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

fn openai_config(base_url: &str) -> OpenAIConfig {
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

/// Build the Prometheus label-filter substring for an endpoint value.
/// Using the constant avoids test breakage if label values are renamed.
fn endpoint_label_filter(val: &str) -> String {
    format!("endpoint=\"{val}\"")
}

/// Parse the most recent value of a counter from Prometheus text output.
/// Returns 0.0 when the time series does not yet exist in the registry.
fn counter_value(text: &str, metric_name: &str, label_substr: &str) -> f64 {
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        if line.starts_with(metric_name) && line.contains(label_substr) {
            return line
                .split_whitespace()
                .next_back()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
        }
    }
    0.0
}

/// Parse the _count line of a histogram family from Prometheus text output.
fn histogram_count(text: &str, metric_name: &str, label_substr: &str) -> f64 {
    let count_key = format!("{metric_name}_count");
    counter_value(text, &count_key, label_substr)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Successful embedding call emits EMBEDDINGS_TOTAL{status="success"}, duration histogram,
/// EMBEDDINGS_VECTORS_TOTAL, and COST_USD_TOTAL{endpoint="embeddings"}.
#[tokio::test]
async fn test_embeddings_metrics_incremented_on_success() {
    let _guard = metrics_test_lock().lock().await;
    let handle = crate::common::test_prometheus_handle();

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_embeddings(&mock, "text-embedding-3-small", 42).await;

    let provider = Arc::new(
        OpenAiAdapter::new(openai_config(&mock.uri()))
            .await
            .expect("OpenAiAdapter must build"),
    );
    let state = app_state_with_provider(provider);
    let app = router_with_metrics(state, handle.clone());
    let server = axum_test::TestServer::new(app).expect("TestServer must build");

    let before = handle.render();

    let response = server
        .post(EMBEDDINGS_PATH)
        .json(&serde_json::json!({ "model": "text-embedding-3-small", "input": "hello" }))
        .await;
    response.assert_status(StatusCode::OK);

    let after = handle.render();

    let delta_total = counter_value(&after, EMBEDDINGS_TOTAL, "status=\"success\"")
        - counter_value(&before, EMBEDDINGS_TOTAL, "status=\"success\"");
    assert_eq!(
        delta_total, 1.0,
        "EMBEDDINGS_TOTAL{{status=success}} must increment by 1"
    );

    let delta_vectors = counter_value(&after, EMBEDDINGS_VECTORS_TOTAL, "provider=")
        - counter_value(&before, EMBEDDINGS_VECTORS_TOTAL, "provider=");
    assert!(
        delta_vectors >= 1.0,
        "EMBEDDINGS_VECTORS_TOTAL must increment (got delta {delta_vectors})"
    );

    let delta_cost = counter_value(
        &after,
        COST_USD_TOTAL,
        &endpoint_label_filter(ENDPOINT_EMBEDDINGS),
    ) - counter_value(
        &before,
        COST_USD_TOTAL,
        &endpoint_label_filter(ENDPOINT_EMBEDDINGS),
    );
    assert!(
        delta_cost > 0.0,
        "COST_USD_TOTAL{{endpoint=embeddings}} must be > 0 after a successful call (got delta {delta_cost})"
    );

    let delta_hist = histogram_count(&after, EMBEDDINGS_DURATION_SECONDS, "provider=")
        - histogram_count(&before, EMBEDDINGS_DURATION_SECONDS, "provider=");
    assert_eq!(
        delta_hist, 1.0,
        "EMBEDDINGS_DURATION_SECONDS must record one observation"
    );
}

/// Provider error: EMBEDDINGS_TOTAL{status="error"} and duration histogram fire;
/// COST_USD_TOTAL{endpoint="embeddings"} and EMBEDDINGS_VECTORS_TOTAL must NOT change.
#[tokio::test]
async fn test_embeddings_metrics_cost_not_incremented_on_error() {
    let _guard = metrics_test_lock().lock().await;
    let handle = crate::common::test_prometheus_handle();

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_embeddings_error(&mock, 500).await;

    let provider = Arc::new(
        OpenAiAdapter::new(openai_config(&mock.uri()))
            .await
            .expect("OpenAiAdapter must build"),
    );
    let state = app_state_with_provider(provider);
    let app = router_with_metrics(state, handle.clone());
    let server = axum_test::TestServer::new(app).expect("TestServer must build");

    let before = handle.render();

    let response = server
        .post(EMBEDDINGS_PATH)
        .json(&serde_json::json!({ "model": "text-embedding-3-small", "input": "hello" }))
        .await;
    // Provider 500 maps to a gateway 5xx; just check it's not 200.
    assert_ne!(
        response.status_code(),
        StatusCode::OK,
        "provider error must not return 200"
    );

    let after = handle.render();

    let delta_error = counter_value(&after, EMBEDDINGS_TOTAL, "status=\"error\"")
        - counter_value(&before, EMBEDDINGS_TOTAL, "status=\"error\"");
    assert_eq!(
        delta_error, 1.0,
        "EMBEDDINGS_TOTAL{{status=error}} must increment by 1 on error"
    );

    let delta_hist = histogram_count(&after, EMBEDDINGS_DURATION_SECONDS, "provider=")
        - histogram_count(&before, EMBEDDINGS_DURATION_SECONDS, "provider=");
    assert_eq!(
        delta_hist, 1.0,
        "EMBEDDINGS_DURATION_SECONDS must record one observation on error"
    );

    let delta_cost = counter_value(
        &after,
        COST_USD_TOTAL,
        &endpoint_label_filter(ENDPOINT_EMBEDDINGS),
    ) - counter_value(
        &before,
        COST_USD_TOTAL,
        &endpoint_label_filter(ENDPOINT_EMBEDDINGS),
    );
    assert_eq!(
        delta_cost, 0.0,
        "COST_USD_TOTAL{{endpoint=embeddings}} must NOT change on error"
    );

    let delta_vectors = counter_value(&after, EMBEDDINGS_VECTORS_TOTAL, "provider=")
        - counter_value(&before, EMBEDDINGS_VECTORS_TOTAL, "provider=");
    assert_eq!(
        delta_vectors, 0.0,
        "EMBEDDINGS_VECTORS_TOTAL must NOT change on error"
    );
}

/// Provider returns 200 with usage=null: EMBEDDINGS_TOTAL{status="success"} and duration
/// histogram fire; EMBEDDINGS_VECTORS_TOTAL and COST_USD_TOTAL{endpoint="embeddings"} must NOT change.
#[tokio::test]
async fn test_embeddings_metrics_suppressed_on_no_usage() {
    let _guard = metrics_test_lock().lock().await;
    let handle = crate::common::test_prometheus_handle();

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_embeddings_no_usage(&mock, "text-embedding-3-small").await;

    let provider = Arc::new(
        OpenAiAdapter::new(openai_config(&mock.uri()))
            .await
            .expect("OpenAiAdapter must build"),
    );
    let state = app_state_with_provider(provider);
    let app = router_with_metrics(state, handle.clone());
    let server = axum_test::TestServer::new(app).expect("TestServer must build");

    let before = handle.render();

    let response = server
        .post(EMBEDDINGS_PATH)
        .json(&serde_json::json!({ "model": "text-embedding-3-small", "input": "hello" }))
        .await;
    response.assert_status(StatusCode::OK);

    let after = handle.render();

    let delta_success = counter_value(&after, EMBEDDINGS_TOTAL, "status=\"success\"")
        - counter_value(&before, EMBEDDINGS_TOTAL, "status=\"success\"");
    assert_eq!(
        delta_success, 1.0,
        "EMBEDDINGS_TOTAL{{status=success}} must fire even without usage"
    );

    let delta_hist = histogram_count(&after, EMBEDDINGS_DURATION_SECONDS, "provider=")
        - histogram_count(&before, EMBEDDINGS_DURATION_SECONDS, "provider=");
    assert_eq!(
        delta_hist, 1.0,
        "EMBEDDINGS_DURATION_SECONDS must fire even without usage"
    );

    let delta_vectors = counter_value(&after, EMBEDDINGS_VECTORS_TOTAL, "provider=")
        - counter_value(&before, EMBEDDINGS_VECTORS_TOTAL, "provider=");
    assert_eq!(
        delta_vectors, 0.0,
        "EMBEDDINGS_VECTORS_TOTAL must NOT fire when usage is absent"
    );

    let delta_cost = counter_value(
        &after,
        COST_USD_TOTAL,
        &endpoint_label_filter(ENDPOINT_EMBEDDINGS),
    ) - counter_value(
        &before,
        COST_USD_TOTAL,
        &endpoint_label_filter(ENDPOINT_EMBEDDINGS),
    );
    assert_eq!(
        delta_cost, 0.0,
        "COST_USD_TOTAL{{endpoint=embeddings}} must NOT fire when usage is absent"
    );
}

/// Integration: one chat call + one embedding call → COST_USD_TOTAL and REQUESTS_TOTAL
/// both carry distinct endpoint labels with value > 0.
///
/// Uses lazy (non-connecting) pools since DB writes are fire-and-forget; the response and
/// metric emission complete before any pool connection is attempted.
#[tokio::test]
async fn test_embedding_and_chat_cost_labels_distinct() {
    let _guard = metrics_test_lock().lock().await;
    let handle = crate::common::test_prometheus_handle();

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_chat(&mock, "gpt-4o-mini", 10, 5).await;
    wiremock_stubs::stub_openai_embeddings(&mock, "text-embedding-3-small", 20).await;

    let provider = Arc::new(
        OpenAiAdapter::new(openai_config(&mock.uri()))
            .await
            .expect("OpenAiAdapter must build"),
    );
    let state = app_state_with_provider(provider);
    let app = router_with_metrics(state, handle.clone());
    let server = axum_test::TestServer::new(app).expect("TestServer must build");

    // Chat completion call.
    let chat_resp = server
        .post(CHAT_COMPLETIONS_PATH)
        .json(&serde_json::json!({
            "model": "gpt-4o-mini",
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .await;
    chat_resp.assert_status(StatusCode::OK);

    // Embedding call.
    let emb_resp = server
        .post(EMBEDDINGS_PATH)
        .json(&serde_json::json!({ "model": "text-embedding-3-small", "input": "hello" }))
        .await;
    emb_resp.assert_status(StatusCode::OK);

    let output = handle.render();

    assert!(
        counter_value(
            &output,
            COST_USD_TOTAL,
            &endpoint_label_filter(ENDPOINT_CHAT)
        ) > 0.0,
        "COST_USD_TOTAL{{endpoint=chat}} must be > 0 after a chat call; scrape:\n{output}"
    );
    assert!(
        counter_value(
            &output,
            COST_USD_TOTAL,
            &endpoint_label_filter(ENDPOINT_EMBEDDINGS)
        ) > 0.0,
        "COST_USD_TOTAL{{endpoint=embeddings}} must be > 0 after an embedding call; scrape:\n{output}"
    );
    assert!(
        counter_value(
            &output,
            REQUESTS_TOTAL,
            &endpoint_label_filter(ENDPOINT_CHAT)
        ) > 0.0,
        "REQUESTS_TOTAL{{endpoint=chat}} must be > 0; scrape:\n{output}"
    );
    assert!(
        counter_value(
            &output,
            REQUESTS_TOTAL,
            &endpoint_label_filter(ENDPOINT_EMBEDDINGS)
        ) > 0.0,
        "REQUESTS_TOTAL{{endpoint=embeddings}} must be > 0; scrape:\n{output}"
    );
}

/// GET /v1/models fires REQUESTS_TOTAL{endpoint="other"} — verifies the catch-all branch
/// in the path matcher for routes that are neither chat nor embeddings.
///
/// /health and /metrics are on the outer router outside RequestMetricsLayer and must not be
/// used here; only /v1/* routes are instrumented.
#[tokio::test]
async fn test_models_request_endpoint_other_label() {
    let _guard = metrics_test_lock().lock().await;
    let handle = crate::common::test_prometheus_handle();

    let state = app_state_with_provider(
        Arc::new(StubAdapter::default()) as Arc<dyn oxigate::domain::ports::ProviderAdapterExt>
    );
    let app = router_with_metrics(state, handle.clone());
    let server = axum_test::TestServer::new(app).expect("TestServer must build");

    let before = handle.render();

    let response = server.get("/v1/models").await;
    response.assert_status(StatusCode::OK);

    let after = handle.render();

    let delta = counter_value(
        &after,
        REQUESTS_TOTAL,
        &endpoint_label_filter(ENDPOINT_OTHER),
    ) - counter_value(
        &before,
        REQUESTS_TOTAL,
        &endpoint_label_filter(ENDPOINT_OTHER),
    );
    assert!(
        delta >= 1.0,
        "REQUESTS_TOTAL{{endpoint=other}} must increment for GET /v1/models (got delta {delta})"
    );
}
