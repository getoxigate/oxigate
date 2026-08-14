// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Reads current spend from Redis, returns 429 if `spend >= hard_cap`.
//! Fail-open if Redis is unavailable.

use std::sync::Arc;

use axum::extract::Request;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::json;
use tokio::sync::RwLock;
use tower::{Layer, Service};

use crate::domain::auth::RequestIdentity;
use crate::middleware::budget::{BudgetCheckResult, BudgetRuntimeConfig};
use crate::redis_pool::RedisPool;
use crate::utils::CostHeader;
use crate::utils::{identity_spend_key, period_key, read_identity_spend};

/// Tower layer that enforces per-identity hard cap via Redis read and 429 on breach.
#[derive(Clone)]
pub struct HardCapLayer {
    redis_pool: Arc<RwLock<RedisPool>>,
    budget_config: Arc<RwLock<BudgetRuntimeConfig>>,
}

impl HardCapLayer {
    /// Create a HardCapLayer from shared Redis pool and budget config.
    #[must_use]
    pub fn new(
        redis_pool: Arc<RwLock<RedisPool>>,
        budget_config: Arc<RwLock<BudgetRuntimeConfig>>,
    ) -> Self {
        Self {
            redis_pool,
            budget_config,
        }
    }
}

impl<S> Layer<S> for HardCapLayer {
    type Service = HardCapService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        HardCapService {
            inner,
            redis_pool: Arc::clone(&self.redis_pool),
            budget_config: Arc::clone(&self.budget_config),
        }
    }
}

/// Inner service for hard cap enforcement.
#[derive(Clone)]
pub struct HardCapService<S> {
    inner: S,
    redis_pool: Arc<RwLock<RedisPool>>,
    budget_config: Arc<RwLock<BudgetRuntimeConfig>>,
}

impl<S> Service<Request> for HardCapService<S>
where
    S: Service<Request, Response = Response> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        let redis_pool = Arc::clone(&self.redis_pool);
        let budget_config = Arc::clone(&self.budget_config);

        let mut inner = self.inner.clone();
        std::mem::swap(&mut inner, &mut self.inner);

        Box::pin(async move {
            let budget_cfg = budget_config.read().await.clone();

            // Step 1: If hard_cap not configured, pass through (no-op).
            let hard_cap_nano_usd = match budget_cfg.hard_cap_nano_usd() {
                Some(cap) => cap,
                None => return inner.call(req).await,
            };

            // Step 2: Get identity (fail-safe default if auth is disabled).
            let identity = req
                .extensions()
                .get::<RequestIdentity>()
                .cloned()
                .unwrap_or_default();

            // Step 3: Read identity spend. BudgetLayer (earlier in the Pro stack) already read
            // this key and stored the result in BudgetCheckResult — reuse it to avoid a second
            // Redis GET for the same key. Fall back to Redis only if absent.
            let spend_nano_usd = if let Some(r) = req.extensions().get::<BudgetCheckResult>() {
                r.spend_nano_usd
            } else {
                let now = budget_cfg.resolved_now();
                let period = period_key(budget_cfg.duration, now, budget_cfg.tz);
                let spend_key = identity_spend_key(&identity.org_id, &identity.id, &period);
                match read_identity_spend(
                    &redis_pool,
                    &spend_key,
                    &identity,
                    "hard_cap_check_skipped",
                )
                .await
                {
                    Some(spend) => spend,
                    None => {
                        // Redis unavailable — fail-open, structured WARN already emitted.
                        return inner.call(req).await;
                    }
                }
            };

            // Step 4: Enforce hard cap (spend >= cap → 429).
            if spend_nano_usd >= hard_cap_nano_usd {
                tracing::info!(
                    event = "hard_cap_enforced",
                    identity_id = %identity.id,
                    org_id = %identity.org_id,
                    spend_nano_usd = spend_nano_usd.as_u64(),
                    cap_nano_usd = hard_cap_nano_usd.as_u64(),
                );

                let body = json!({
                    "error": {
                        "message": format!(
                            "Budget hard cap exceeded for identity '{}' (org: '{}')",
                            identity.id, identity.org_id,
                        ),
                        "type": "insufficient_quota",
                        "code": "budget_hard_cap_exceeded",
                        "param": null,
                    }
                });
                let mut response =
                    (StatusCode::TOO_MANY_REQUESTS, axum::Json(body)).into_response();
                response.headers_mut().insert(
                    CostHeader::BUDGET_REMAINING,
                    HeaderValue::from_static("0.000000"),
                );
                return Ok(response);
            }

            // Step 5: Under cap — pass to inner service.
            inner.call(req).await
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::task::{Context, Poll};

    use axum::body::Body;
    use axum::extract::Request;
    use axum::http::StatusCode;
    use axum::response::Response;
    use deadpool_redis::Pool;
    use tokio::sync::RwLock;
    use tower::{Layer, Service, ServiceExt};

    use crate::domain::auth::RequestIdentity;
    use crate::middleware::budget::BudgetRuntimeConfig;
    use crate::middleware::hard_cap::HardCapLayer;
    use crate::redis_pool::RedisPool;
    use crate::utils::CostHeader;

    /// Clonable stub inner service that always returns 200 OK.
    #[derive(Clone)]
    struct OkService;

    impl Service<Request> for OkService {
        type Response = Response;
        type Error = std::convert::Infallible;
        type Future = std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Response, Self::Error>> + Send>,
        >;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: Request) -> Self::Future {
            Box::pin(async {
                Ok(axum::response::Response::builder()
                    .status(StatusCode::OK)
                    .body(Body::empty())
                    .unwrap())
            })
        }
    }

    fn make_budget_config(hard_cap_usd: Option<f64>) -> Arc<RwLock<BudgetRuntimeConfig>> {
        use crate::config::BudgetConfig;
        Arc::new(RwLock::new(BudgetRuntimeConfig::from_budget_config(
            BudgetConfig {
                global_safety_cap_usd: None,
                soft_cap_usd: None,
                hard_cap_usd,
                ..BudgetConfig::default()
            },
        )))
    }

    fn make_redis_pool_failing() -> Arc<RwLock<RedisPool>> {
        // Use an invalid URL so every connection attempt fails.
        // Build Config directly to avoid `UrlAndConnectionSpecified` error from Default.
        let cfg = deadpool_redis::Config {
            url: Some("redis://127.0.0.1:1".into()),
            connection: None,
            pool: None,
        };
        let pool: Pool = cfg
            .create_pool(Some(deadpool_redis::Runtime::Tokio1))
            .expect("pool create");
        Arc::new(RwLock::new(pool))
    }

    async fn call_layer(hard_cap_usd: Option<f64>, identity: Option<RequestIdentity>) -> Response {
        let budget_config = make_budget_config(hard_cap_usd);
        let layer = HardCapLayer::new(make_redis_pool_failing(), budget_config);
        let mut svc = layer.layer(OkService);

        let mut req = Request::new(Body::empty());
        if let Some(id) = identity {
            req.extensions_mut().insert(id);
        }

        svc.ready().await.unwrap();
        svc.call(req).await.unwrap()
    }

    #[tokio::test]
    async fn hard_cap_none_is_noop() {
        // No hard cap configured — layer must be transparent regardless of Redis state.
        let response = call_layer(None, None).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn redis_unavailable_fail_open() {
        // Redis down → fail-open, request passes through.
        let response = call_layer(Some(10.0), None).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn missing_request_identity_no_panic() {
        // No RequestIdentity in extensions → default identity used, no panic.
        let response = call_layer(Some(10.0), None).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn x_budget_remaining_header_constant_is_correct() {
        // Verifies the header constant used in 429 responses is the expected header name.
        let header_name = CostHeader::BUDGET_REMAINING;
        assert_eq!(
            header_name,
            axum::http::header::HeaderName::from_static("x-oxigate-budget-remaining")
        );
    }
}

#[cfg(test)]
mod config_tests {
    use crate::config::BudgetConfig;
    use crate::domain::ports::NanoUsd;
    use crate::middleware::budget::BudgetRuntimeConfig;

    fn make_config(hard_cap_usd: Option<f64>, soft_cap_usd: Option<f64>) -> BudgetRuntimeConfig {
        BudgetRuntimeConfig::from_budget_config(BudgetConfig {
            soft_cap_usd,
            hard_cap_usd,
            ..BudgetConfig::default()
        })
    }

    #[test]
    fn hard_cap_usd_converts_to_nanos() {
        let cfg = make_config(Some(10.0), None);
        assert_eq!(cfg.hard_cap_nano_usd(), Some(NanoUsd(10_000_000_000)));
    }

    #[test]
    fn hard_cap_usd_none_remains_none() {
        let cfg = make_config(None, None);
        assert_eq!(cfg.hard_cap_nano_usd(), None);
    }

    #[test]
    fn effective_response_cap_hard_only_when_soft_and_hard_set() {
        // soft=8, hard=10 → Some(10) (header reflects enforcement boundary only)
        let cfg = make_config(Some(10.0), Some(8.0));
        assert_eq!(
            cfg.effective_response_cap_nano_usd(),
            Some(NanoUsd(10_000_000_000))
        );
    }

    #[test]
    fn effective_response_cap_hard_only_when_soft_absent() {
        // soft=None, hard=10 → Some(10_000_000_000)
        let cfg = make_config(Some(10.0), None);
        assert_eq!(
            cfg.effective_response_cap_nano_usd(),
            Some(NanoUsd(10_000_000_000))
        );
    }

    #[test]
    fn effective_response_cap_none_when_both_none() {
        let cfg = make_config(None, None);
        assert_eq!(cfg.effective_response_cap_nano_usd(), None);
    }
}
