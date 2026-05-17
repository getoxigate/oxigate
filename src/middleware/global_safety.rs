// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! GlobalSafetyLayer — instance-wide budget circuit breaker .
//!
//! Community feature (no feature gate). Position: outermost Tower layer, before AuthLayer.
//! Reads `oxigate:global:spend` from Redis; blocks with 429 if spend >= cap.
//! Fail-open on Redis unavailability. Zero overhead when cap is None.

use std::task::{Context, Poll};

use axum::extract::Request;
use axum::http::HeaderValue;
use axum::response::{IntoResponse, Response};
use tower::{Layer, Service};

use crate::config::BudgetConfig;
use crate::db::spend_writer::GLOBAL_SPEND_KEY;
use crate::domain::ports::NanoUsd;
use crate::redis_pool::RedisPool;
use crate::utils::CostHeader;

/// Runtime config for GlobalSafetyLayer — NanoUsd conversion done once at init.
#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalSafetyRuntimeConfig {
    cap_nano_usd: Option<NanoUsd>,
}

impl GlobalSafetyRuntimeConfig {
    /// Build runtime config from budget config. Converts USD to NanoUsd once.
    #[must_use]
    pub fn from_budget_config(config: &BudgetConfig) -> Self {
        Self {
            cap_nano_usd: config.global_safety_cap_usd.map(NanoUsd::from_f64_usd),
        }
    }

    /// Constructs a config directly from a NanoUsd cap value.
    /// Used by integration test helpers to bypass BudgetConfig conversion.
    #[must_use]
    pub fn with_nano_usd_cap(cap_nano_usd: Option<NanoUsd>) -> Self {
        Self { cap_nano_usd }
    }
}

/// Tower layer that wraps a service with the global safety cap check.
#[derive(Clone)]
pub struct GlobalSafetyLayer {
    redis_pool: std::sync::Arc<tokio::sync::RwLock<RedisPool>>,
    config: std::sync::Arc<tokio::sync::RwLock<GlobalSafetyRuntimeConfig>>,
}

impl GlobalSafetyLayer {
    /// Build a GlobalSafetyLayer from shared Redis pool and runtime config.
    #[must_use]
    pub fn new(
        redis_pool: std::sync::Arc<tokio::sync::RwLock<RedisPool>>,
        config: std::sync::Arc<tokio::sync::RwLock<GlobalSafetyRuntimeConfig>>,
    ) -> Self {
        Self { redis_pool, config }
    }
}

impl<S> Layer<S> for GlobalSafetyLayer {
    type Service = GlobalSafetyService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        GlobalSafetyService {
            inner,
            redis_pool: std::sync::Arc::clone(&self.redis_pool),
            config: std::sync::Arc::clone(&self.config),
        }
    }
}

/// Inner service for pre-dispatch global safety cap checks.
#[derive(Clone)]
pub struct GlobalSafetyService<S> {
    inner: S,
    redis_pool: std::sync::Arc<tokio::sync::RwLock<RedisPool>>,
    config: std::sync::Arc<tokio::sync::RwLock<GlobalSafetyRuntimeConfig>>,
}

impl<S, E> Service<Request> for GlobalSafetyService<S>
where
    S: Service<Request, Response = Response, Error = E> + Clone + Send + 'static,
    S::Future: Send + 'static,
    E: Send + 'static,
{
    type Response = Response;
    type Error = E;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        let redis_pool = std::sync::Arc::clone(&self.redis_pool);
        let config = std::sync::Arc::clone(&self.config);
        let mut inner = self.inner.clone();
        std::mem::swap(&mut inner, &mut self.inner);

        Box::pin(async move {
            let cfg = *config.read().await;
            if let Some(cap_nano_usd) = cfg.cap_nano_usd
                && let Some(spend) = get_global_spend(&redis_pool).await
                && spend >= cap_nano_usd
            {
                let mut response = (
                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                    axum::Json(serde_json::json!({"error": "global_budget_cap_exceeded"})),
                )
                    .into_response();
                response
                    .headers_mut()
                    .insert(CostHeader::BUDGET_CAP, HeaderValue::from_static("global"));
                return Ok(response);
            }
            inner.call(req).await
        })
    }
}

async fn get_global_spend(
    redis_pool: &std::sync::Arc<tokio::sync::RwLock<RedisPool>>,
) -> Option<NanoUsd> {
    let pool = redis_pool.read().await.clone();
    let mut conn = match pool.get().await {
        Ok(conn) => conn,
        Err(error) => {
            tracing::warn!(
                event = "global_safety_check_skipped",
                reason = "redis_unavailable",
                error = %error,
            );
            return None;
        }
    };
    match redis::cmd("GET")
        .arg(GLOBAL_SPEND_KEY)
        .query_async::<Option<u64>>(&mut *conn)
        .await
    {
        Ok(Some(raw)) => Some(NanoUsd(raw)),
        Ok(None) => Some(NanoUsd::zero()),
        Err(error) => {
            tracing::warn!(
                event = "global_safety_check_skipped",
                reason = "redis_unavailable",
                error = %error,
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_spend_key_constant() {
        assert_eq!(GLOBAL_SPEND_KEY, "oxigate:global:spend");
    }

    #[test]
    fn test_runtime_config_none_when_cap_none() {
        let config = BudgetConfig {
            global_safety_cap_usd: None,
            ..Default::default()
        };
        let runtime = GlobalSafetyRuntimeConfig::from_budget_config(&config);
        assert_eq!(runtime.cap_nano_usd, None);
    }

    #[test]
    fn test_runtime_config_converts_usd_to_nanos() {
        let config = BudgetConfig {
            global_safety_cap_usd: Some(10.0),
            ..Default::default()
        };
        let runtime = GlobalSafetyRuntimeConfig::from_budget_config(&config);
        assert_eq!(runtime.cap_nano_usd, Some(NanoUsd(10_000_000_000)));
    }
}
