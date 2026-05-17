// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Axum handlers and DTOs.
//!
//! Router builder, health endpoints, chat completions, and fallback 404 handler.

pub mod auth;
pub mod chat;
pub mod embeddings;
pub mod health;
pub mod metrics_endpoint;
pub mod models;
pub mod spend;

/// Chat completions endpoint path. Used by router and provider adapters.
pub const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";
/// Embeddings endpoint path.
pub const EMBEDDINGS_PATH: &str = "/v1/embeddings";

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::http::{HeaderValue, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use tokio::sync::RwLock;

use chrono::Utc;

use crate::config::{AuthConfig, BudgetConfig, SecurityConfig};
use crate::db::DbPool;
use crate::domain::ports::ProviderAdapterExt;
use crate::domain::pricing::PricingDb;
use crate::domain::spend::SpendRecord;
use crate::middleware::active_connections::ActiveConnectionsLayer;
use crate::middleware::auth::AuthLayer;
use crate::middleware::budget::{BudgetLayer, BudgetResponseLayer, BudgetRuntimeConfig};
use crate::middleware::global_safety::{GlobalSafetyLayer, GlobalSafetyRuntimeConfig};
use crate::middleware::hard_cap::HardCapLayer;
use crate::middleware::request_metrics::RequestMetricsLayer;
use crate::middleware::tagger::TaggerLayer;
use crate::middleware::team_tag_budget::{TeamTagBudgetLayer, TeamTagBudgetResponseLayer};
use crate::providers::ProviderHealthTracker;
use crate::redis_pool::RedisPool;

/// Application state shared across handlers.
///
/// Uses `Arc<RwLock<...>>` so pools and provider can be rebuilt on SIGHUP
/// (Class A/B reload) without restarting the process.
#[derive(Clone)]
pub struct AppState {
    /// PostgreSQL connection pool. Handlers acquire via `pool.read().await`.
    pub pool: Arc<RwLock<DbPool>>,
    /// Redis connection pool. Handlers acquire via `redis_pool.read().await`.
    pub redis_pool: Arc<RwLock<RedisPool>>,
    /// Pricing DB. Outer RwLock enables atomic swap on SIGHUP Class A reload;
    /// not a general concurrency primitive — writes only at startup/reload.
    pub pricing_db: Arc<std::sync::RwLock<PricingDb>>,
    /// Provider adapter. Wrapped in RwLock so SIGHUP can rebuild it on config change.
    pub provider: Arc<RwLock<Arc<dyn ProviderAdapterExt>>>,
    /// Auth config. Wrapped in RwLock for SIGHUP Class A reload.
    ///
    /// The read lock is held for only microseconds per request (key bytes are copied out
    /// immediately; the guard is dropped before the inner service call). Multiple concurrent
    /// readers never block each other; the only writer is the SIGHUP reload task, which runs
    /// at most once per reload event. Lock contention on the hot path is negligible.
    pub auth: Arc<RwLock<AuthConfig>>,
    /// Global safety cap config holder for SIGHUP reload.
    pub global_safety: Arc<RwLock<GlobalSafetyRuntimeConfig>>,
    /// Budget config for spend keying; hot-reloaded with gateway config.
    pub budget_settings: Arc<RwLock<BudgetConfig>>,
    /// Budget runtime config holder for BudgetLayer + HardCapLayer SIGHUP reload.
    pub budget: Arc<RwLock<BudgetRuntimeConfig>>,
    /// Gateway process startup time (Unix seconds). Used for model `created` field.
    pub startup_time: u64,
    /// Provider health tracker. Tracks health status, 429-cooldown, EWMA latency,
    /// and in-flight counts. Thread-safe internally — NOT wrapped in outer RwLock.
    /// Mutated in-place on SIGHUP via `sync_providers()`; never swapped.
    pub health: Arc<ProviderHealthTracker>,
    /// Security config for opt-in visibility features.
    /// Wrapped in RwLock for Class A SIGHUP hot-reload.
    pub security: Arc<RwLock<SecurityConfig>>,
}

/// Default safety cap for request bodies (50 MiB).
/// Operators requiring a different limit should configure `client_max_body_size` at their
/// reverse proxy until provides a configurable SecurityConfig.
pub const MAX_REQUEST_BODY_BYTES: usize = 50 * 1024 * 1024;

/// Builds the application router with health routes, chat completions, and 404 fallback.
/// Health routes are public; /v1/* routes require Bearer auth when auth.key is configured.
///
/// Tower layers apply in reverse declaration order: last .layer() is outermost.
/// Request flow (innermost → outermost):
///   Router → BudgetResponseLayer → HardCapLayer → BudgetLayer
///          → TeamTagBudgetResponseLayer → TeamTagBudgetLayer
///          → TaggerLayer → AuthLayer → GlobalSafetyLayer → DefaultBodyLimit
///          → RequestMetricsLayer → ActiveConnectionsLayer
pub fn router(state: AppState) -> Router {
    router_with_body_limit(state, MAX_REQUEST_BODY_BYTES)
}

/// Like [`router`] but with a [`metrics_exporter_prometheus::PrometheusHandle`] Extension.
///
/// Mounts `GET /metrics` (already present in `router()`) and adds the `PrometheusHandle`
/// as a Tower `Extension` layer on the outer router so the handler can render the scrape.
/// Call this in `main.rs` after `init_metrics()`.
pub fn router_with_metrics(
    state: AppState,
    handle: metrics_exporter_prometheus::PrometheusHandle,
) -> Router {
    router(state).layer(axum::Extension(handle))
}

/// Like [`router`] but with a configurable body limit.
/// Intended for integration tests that need to verify the 413 enforcement path
/// with a small limit without sending 50 MiB of data.
pub fn router_with_body_limit(state: AppState, max_request_body_bytes: usize) -> Router {
    // Step 1 — routes only (both tiers).
    let v1_routes = Router::new()
        .route("/chat/completions", post(chat::chat_completions))
        .route("/embeddings", post(embeddings::embeddings))
        .route("/models", get(models::list_models))
        .route("/spend/daily", get(spend::daily))
        .route("/spend/providers", get(spend::by_provider))
        .route("/spend/models", get(spend::by_model));

    // Step 2 — per-identity budget + enforcement layers (innermost of this block).
    // Layer declaration order (innermost → outermost):
    //   BudgetResponseLayer (innermost: only sees responses from Router)
    //   HardCapLayer        (middle: short-circuits on 429; never calls BudgetResponseLayer on that path)
    //   BudgetLayer         (outermost of this block: runs first in request flow, sets BudgetCheckResult)
    let v1_routes = v1_routes
        .layer(BudgetResponseLayer::new())
        .layer(HardCapLayer::new(
            Arc::clone(&state.redis_pool),
            Arc::clone(&state.budget),
        ))
        // BudgetLayer must be declared last here (outermost of this block).
        // It runs first in request flow, reads Redis, and sets BudgetCheckResult in extensions
        // before HardCapLayer or BudgetResponseLayer execute.
        .layer(BudgetLayer::new(
            Arc::clone(&state.redis_pool),
            Arc::clone(&state.budget),
        ));

    // Step 3 — COM: per-team + per-tag budget enforcement (Community, no feature gate).
    // Layer declaration order (innermost → outermost):
    //   TeamTagBudgetResponseLayer (innermost: most-restrictive-wins CostHeader::BUDGET_REMAINING header)
    //   TeamTagBudgetLayer         (outermost: hard-cap 429 + soft-cap threshold logging)
    //
    // Request execution order: TaggerLayer → TeamTagBudgetLayer (hard-cap check)
    //   → BudgetLayer/HardCapLayer (per-identity) → BudgetResponseLayer
    //   → TeamTagBudgetResponseLayer (header merge) → Router
    let v1_routes =
        v1_routes
            .layer(TeamTagBudgetResponseLayer::new())
            .layer(TeamTagBudgetLayer::new(
                Arc::clone(&state.redis_pool),
                Arc::clone(&state.budget_settings),
            ));

    // Step 4 — auth + global safety + request metrics (both tiers).
    // Layer declaration order (innermost → outermost):
    //   TaggerLayer             (innermost of this block)
    //   AuthLayer
    //   GlobalSafetyLayer
    //   DefaultBodyLimit        (outermost body check; rejects 413 before any middleware)
    //   RequestMetricsLayer     (outermost: measures full e2e latency incl. auth + body limit)
    //   ActiveConnectionsLayer  (outermost: counts connections for the full v1 scope)
    //
    //: DefaultBodyLimit rejects oversized bodies with 413 before other middleware.
    let v1_routes = v1_routes
        .layer(TaggerLayer::new())
        .layer(AuthLayer::new(Arc::clone(&state.auth)))
        .layer(GlobalSafetyLayer::new(
            Arc::clone(&state.redis_pool),
            Arc::clone(&state.global_safety),
        ))
        .layer(DefaultBodyLimit::max(max_request_body_bytes))
        .layer(RequestMetricsLayer::new())
        .layer(ActiveConnectionsLayer::new())
        .with_state(state.clone());

    Router::new()
        .route("/health", get(health::health_live))
        .route("/health/ready", get(health::health_ready))
        // GET /metrics — no auth; protected at network level (see docs/guides/prometheus-metrics.md)
        .route("/metrics", get(metrics_endpoint::metrics_handler))
        // Health routes are GET-only (no body), so DefaultBodyLimit is not applied here.
        .nest("/v1", v1_routes)
        .fallback(fallback_404)
        .with_state(state)
}

async fn fallback_404() -> (StatusCode, Json<Value>) {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": "not_found",
            "message": "route does not exist"
        })),
    )
}

/// Injects routing observability headers when `security.expose_provider_names` is true.
///
/// `X-Oxigate-Attempted-Providers` and `X-Oxigate-Attempted-Models`.
/// `X-Fallback-Reason` — present only when ≥1 fallback target was dispatched.
/// Shared by chat and embeddings handlers.
pub(crate) fn inject_attempted_headers(
    headers: &mut axum::http::HeaderMap,
    providers: &[String],
    models: &[String],
    fallback_trigger: Option<&str>,
    fallback_dispatched: bool,
) {
    if !providers.is_empty() {
        let value = providers.join(",");
        if let Ok(v) = HeaderValue::from_str(&value) {
            headers.insert("x-oxigate-attempted-providers", v);
        }
    }
    if !models.is_empty() {
        let value = models.join(",");
        if let Ok(v) = HeaderValue::from_str(&value) {
            headers.insert("x-oxigate-attempted-models", v);
        }
    }
    if fallback_dispatched
        && let Some(trigger) = fallback_trigger
        && let Ok(v) = HeaderValue::from_str(trigger)
    {
        headers.insert("x-fallback-reason", v);
    }
}

/// Emit a structured cost log line and spawn a fire-and-forget spend write.
///
/// Called from both the chat and embeddings handlers after usage data is available.
/// `event_name` becomes the tracing message (e.g. `"chat_completion_cost"`, `"embedding_cost"`).
/// `request_body_bytes` is intentionally omitted — that debug field stays local to the chat handler.
pub(crate) fn spawn_cost_log_and_spend(
    event_name: &'static str,
    record: SpendRecord,
    request_id: &str,
    cost_usd_display: &str,
    pool: Arc<RwLock<DbPool>>,
    redis: Arc<RwLock<RedisPool>>,
    budget: BudgetConfig,
) {
    tracing::info!(
        request_id = %request_id,
        org_id = %record.org_id,
        identity_id = %record.identity_id,
        model = %record.model,
        provider = %record.provider,
        prompt_tokens = record.prompt_tokens,
        completion_tokens = record.completion_tokens,
        cost_usd = %cost_usd_display,
        latency_ms = record.latency_ms,
        tags = %record.tags,
        "{}", event_name
    );
    tokio::spawn(async move {
        let duration = budget.resolved_duration();
        let tz = budget.resolved_timezone();
        let now = Utc::now();
        crate::db::spend_writer::write_spend(record, pool, redis, duration, tz, now).await;
    });
}
