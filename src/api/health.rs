// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Health check endpoints for liveness and readiness probes.
//!
//! `GET /health` is liveness (no I/O); `GET /health/ready` is readiness.
//!
//! ## Kubernetes probe configuration
//!
//! ```yaml
//! livenessProbe:
//!   httpGet: { path: /health, port: 8080 }
//!   initialDelaySeconds: 5
//!   periodSeconds: 10
//!
//! readinessProbe:
//!   httpGet: { path: /health/ready, port: 8080 }
//!   initialDelaySeconds: 10
//!   periodSeconds: 5
//!   failureThreshold: 3
//!   timeoutSeconds: 2
//! ```

use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde_json::{Value, json};
use tokio::time::timeout;

use crate::api::AppState;
use crate::domain::ports::HealthStatus;

const READINESS_CHECK_TIMEOUT_MS: u64 = 800;

fn resolve_check_result<E: std::fmt::Debug>(
    result: Result<Result<(), E>, tokio::time::error::Elapsed>,
    check: &'static str,
) -> &'static str {
    match result {
        Ok(Ok(())) => "ok",
        Ok(Err(err)) => {
            tracing::warn!(
                event = "readiness_check_failed",
                check,
                error = ?err,
                "readiness dependency check failed"
            );
            "unreachable"
        }
        Err(err) => {
            tracing::warn!(
                event = "readiness_check_failed",
                check,
                error = ?err,
                "readiness dependency check timed out"
            );
            "unreachable"
        }
    }
}

/// GET /health — liveness probe. Always 200. No I/O checks.
///
/// Shape is stable: `{"status":"ok"}` — do not add fields​.
pub async fn health_live() -> (StatusCode, Json<Value>) {
    (StatusCode::OK, Json(json!({"status": "ok"})))
}

/// GET /health/ready — readiness probe.
///
/// Returns 200 `{"status":"ok"}` when PostgreSQL and Redis are reachable and
/// all provider startup checks are available.
///
/// Returns 503 `{"status":"degraded","checks":{...}}` when any check fails.
/// DB and Redis checks run concurrently and are individually bounded to 800 ms.
/// The asymmetry is intentional: successful responses omit `checks` per spec.
pub async fn health_ready(State(state): State<AppState>) -> (StatusCode, Json<Value>) {
    let pool = state.pool.read().await.clone();
    let redis_pool = state.redis_pool.read().await.clone();

    let db_check = timeout(
        Duration::from_millis(READINESS_CHECK_TIMEOUT_MS),
        crate::db::health_check(&pool),
    );
    let redis_check = timeout(
        Duration::from_millis(READINESS_CHECK_TIMEOUT_MS),
        crate::redis_pool::health_check(&redis_pool),
    );
    let (db_result, redis_result) = tokio::join!(db_check, redis_check);

    let postgres_status = resolve_check_result(db_result, "postgres");
    let redis_status = resolve_check_result(redis_result, "redis");

    let statuses = state.health.provider_statuses().await;
    let failed_count = statuses
        .iter()
        .filter(|(_, s)| *s == HealthStatus::Unhealthy)
        .count();
    let (providers_ok, providers_msg) = if failed_count == 0 {
        (true, "ok".to_string())
    } else {
        (false, format!("{failed_count} provider(s) unhealthy"))
    };
    if !providers_ok {
        tracing::warn!(
            event = "readiness_check_failed",
            check = "providers",
            status = %providers_msg,
            "provider health check is degraded"
        );
    }

    if postgres_status == "ok" && redis_status == "ok" && providers_ok {
        return (StatusCode::OK, Json(json!({"status": "ok"})));
    }

    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "status": "degraded",
            "checks": {
                "postgres": postgres_status,
                "redis": redis_status,
                "providers": providers_msg
            }
        })),
    )
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use axum::extract::State;
    use axum::http::StatusCode;
    use sqlx::postgres::PgPoolOptions;
    use tracing_test::traced_test;

    use crate::api::AppState;
    use crate::config::{AuthConfig, PricingConfig, RedisConfig, SecretString};
    use crate::domain::chat::{ChatRequest, ChatResponse};
    use crate::domain::ports::{
        HealthStatus, ProviderAdapter, ProviderAdapterExt, ProviderError, ProviderMetadata,
    };
    use crate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
    use crate::middleware::global_safety::GlobalSafetyRuntimeConfig;
    use crate::providers::ProviderHealthTracker;
    use crate::redis_pool::{RedisPool, create_pool};

    use super::{health_live, health_ready, resolve_check_result};

    struct TestProvider {
        meta: ProviderMetadata,
    }

    impl TestProvider {
        fn new() -> Self {
            Self {
                meta: ProviderMetadata {
                    name: "test-provider".to_string(),
                    supported_models: vec!["*".to_string()],
                    supports_streaming: false,
                    supports_tools: false,
                    supports_vision: false,
                    supports_embeddings: false,
                    supports_thinking: false,
                    kind: Default::default(),
                    ..Default::default()
                },
            }
        }
    }

    #[async_trait]
    impl ProviderAdapter for TestProvider {
        async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
            Err(ProviderError::NotImplemented)
        }

        fn metadata(&self) -> &ProviderMetadata {
            &self.meta
        }

        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }

    impl ProviderAdapterExt for TestProvider {}

    fn unavailable_db_pool() -> crate::db::DbPool {
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

    fn test_state(health: Arc<ProviderHealthTracker>) -> AppState {
        let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing DB must parse");
        AppState {
            pool: Arc::new(tokio::sync::RwLock::new(unavailable_db_pool())),
            redis_pool: Arc::new(tokio::sync::RwLock::new(unavailable_redis_pool())),
            pricing_db: Arc::new(std::sync::RwLock::new(pricing_db)),
            provider: Arc::new(tokio::sync::RwLock::new(Arc::new(TestProvider::new()))),
            auth: Arc::new(tokio::sync::RwLock::new(AuthConfig::default())),
            global_safety: Arc::new(tokio::sync::RwLock::new(
                GlobalSafetyRuntimeConfig::default(),
            )),
            budget_settings: Arc::new(tokio::sync::RwLock::new(
                crate::config::BudgetConfig::default(),
            )),
            budget: Arc::new(tokio::sync::RwLock::new(
                crate::middleware::budget::BudgetRuntimeConfig::default(),
            )),
            startup_time: 1,
            health,
            security: Arc::new(tokio::sync::RwLock::new(
                crate::config::SecurityConfig::default(),
            )),
        }
    }

    #[tokio::test]
    async fn test_health_live_returns_200_ok_shape() {
        let (status, body) = health_live().await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body.0, serde_json::json!({"status": "ok"}));
    }

    #[test]
    #[traced_test]
    fn test_resolve_check_result_logs_failed_dependency() {
        let status = resolve_check_result::<io::Error>(
            Ok(Err(io::Error::new(
                io::ErrorKind::ConnectionRefused,
                "db down",
            ))),
            "postgres",
        );
        assert_eq!(status, "unreachable");
        assert!(logs_contain("readiness_check_failed"));
        assert!(logs_contain("postgres"));
    }

    #[tokio::test]
    #[traced_test]
    async fn test_resolve_check_result_logs_timeout() {
        let elapsed = tokio::time::timeout(Duration::from_millis(1), async {
            tokio::time::sleep(Duration::from_millis(5)).await;
        })
        .await
        .expect_err("timeout must elapse");
        let status = resolve_check_result::<io::Error>(Err(elapsed), "redis");
        assert_eq!(status, "unreachable");
        assert!(logs_contain("readiness_check_failed"));
        assert!(logs_contain("redis"));
    }

    #[tokio::test]
    async fn test_health_ready_minimal_state_returns_degraded_when_backends_unreachable() {
        // Tracker with a single provider as Healthy — providers check should be "ok".
        let tracker = ProviderHealthTracker::new_for_test(&["compat-test"]);
        let state = test_state(tracker);

        let (status, body) = health_ready(State(state)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body.0["status"], "degraded");
        assert_eq!(body.0["checks"]["postgres"], "unreachable");
        assert_eq!(body.0["checks"]["redis"], "unreachable");
        assert_eq!(body.0["checks"]["providers"], "ok");
    }

    #[tokio::test]
    async fn test_health_ready_unhealthy_provider_is_degraded() {
        let tracker = ProviderHealthTracker::new_for_test(&["openai"]);
        // Mark openai as Unhealthy.
        tracker
            .update_health("openai", HealthStatus::Unhealthy)
            .await;
        let state = test_state(tracker);

        let (status, body) = health_ready(State(state)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body.0["status"], "degraded");
        assert_ne!(body.0["checks"]["providers"], "ok");
    }
}
