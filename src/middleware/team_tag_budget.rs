// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Per-team and per-tag budget enforcement middleware .
//!
//! Community tier — no feature gate.
//!
//! `TeamTagBudgetLayer` runs pre-dispatch:
//!   - Builds a check_list from the request's team tag + all matching tag kv entries in config.
//!   - Redis GET pipeline fetches current spend for all matched keys (single round trip).
//!   - Hard-cap enforcement: returns 429 immediately if any hard cap is breached.
//!   - Soft-cap threshold logging: emits deduplicated `warn!` at 80/90/100% of soft caps.
//!   - Stores `TeamTagCheckResult` in request extensions for response layer.
//!
//! `TeamTagBudgetResponseLayer` runs post-dispatch:
//!   - Reads `TeamTagCheckResult` and injects `CostHeader::BUDGET_REMAINING` header using
//!     most-restrictive-wins semantics (lower value wins vs. any existing header from
//!     `BudgetResponseLayer`).

use std::task::{Context, Poll};

use axum::extract::Request;
use axum::http::HeaderValue;
use axum::response::Response;
use serde_json::json;
use tower::Layer;
use tower::Service;

use crate::config::BudgetConfig;
use crate::domain::auth::RequestIdentity;
use crate::domain::ports::BudgetScope;
use crate::domain::ports::NanoUsd;
use crate::middleware::budget::check_soft_cap_thresholds;
use crate::redis_pool::RedisPool;
use crate::utils::CostHeader;
use crate::utils::{nanos_to_usd_display, period_key, tag_spend_key, team_spend_key};

/// Stored in request extensions by `TeamTagBudgetService`.
/// Read by `TeamTagBudgetResponseService` to inject the response header.
#[derive(Clone)]
pub(crate) struct TeamTagCheckResult {
    /// Minimum remaining budget across all active **hard** team/tag caps.
    /// `None` = no team/tag hard caps are active for this request (header omitted).
    pub min_remaining_nano_usd: Option<NanoUsd>,
}

// ─── TeamTagBudgetLayer ──────────────────────────────────────────────────────

/// Tower layer that performs per-team and per-tag budget checks before dispatch.
#[derive(Clone)]
pub struct TeamTagBudgetLayer {
    redis_pool: std::sync::Arc<tokio::sync::RwLock<RedisPool>>,
    budget_config: std::sync::Arc<tokio::sync::RwLock<BudgetConfig>>,
}

impl TeamTagBudgetLayer {
    /// Create a `TeamTagBudgetLayer` from shared pools.
    #[must_use]
    pub fn new(
        redis_pool: std::sync::Arc<tokio::sync::RwLock<RedisPool>>,
        budget_config: std::sync::Arc<tokio::sync::RwLock<BudgetConfig>>,
    ) -> Self {
        Self {
            redis_pool,
            budget_config,
        }
    }
}

impl<S> Layer<S> for TeamTagBudgetLayer {
    type Service = TeamTagBudgetService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TeamTagBudgetService {
            inner,
            redis_pool: std::sync::Arc::clone(&self.redis_pool),
            budget_config: std::sync::Arc::clone(&self.budget_config),
        }
    }
}

/// Inner service for pre-dispatch team/tag budget checks.
#[derive(Clone)]
pub struct TeamTagBudgetService<S> {
    inner: S,
    redis_pool: std::sync::Arc<tokio::sync::RwLock<RedisPool>>,
    budget_config: std::sync::Arc<tokio::sync::RwLock<BudgetConfig>>,
}

impl<S, E> Service<Request> for TeamTagBudgetService<S>
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

    fn call(&mut self, mut req: Request) -> Self::Future {
        let redis_pool = std::sync::Arc::clone(&self.redis_pool);
        let budget_config = std::sync::Arc::clone(&self.budget_config);

        let mut inner = self.inner.clone();
        std::mem::swap(&mut inner, &mut self.inner);

        Box::pin(async move {
            // Step 1: Read BudgetConfig; drop lock immediately.
            let cfg = budget_config.read().await.clone();

            // Early-exit: no team/tag caps configured → pass-through.
            if cfg.teams.is_empty() && cfg.tag_budgets.is_empty() {
                return inner.call(req).await;
            }

            // Step 2: Extract RequestIdentity (default fallback for auth-disabled deploys).
            let identity = req
                .extensions()
                .get::<RequestIdentity>()
                .cloned()
                .unwrap_or_default();

            // Step 3: Compute period key.
            let now = chrono::Utc::now();
            let duration = cfg.resolved_duration();
            let tz = cfg.resolved_timezone();
            let period = period_key(duration, now, tz);
            let warn_dedup_secs = cfg.warn_dedup_period_secs();
            let org_id = identity.org_id.clone();

            // Step 4: Build check_list.
            // Tuple: (redis_key, soft_cap, hard_cap, scope)
            let mut check_list: Vec<(String, Option<NanoUsd>, Option<NanoUsd>, BudgetScope)> =
                Vec::new();

            // 4a: Team entry — look up tags["team"] in config.teams.
            if let Some(team_name) = identity.tags.get("team")
                && let Some(entry) = cfg.teams.get(team_name)
            {
                check_list.push((
                    team_spend_key(&org_id, team_name, &period),
                    entry.soft_cap_usd.map(NanoUsd::from_f64_usd),
                    entry.hard_cap_usd.map(NanoUsd::from_f64_usd),
                    BudgetScope::Team(team_name.clone()),
                ));
            }

            // 4b: Tag entries — collect matching tags, sort deterministically, extend.
            let mut tag_entries: Vec<(String, Option<NanoUsd>, Option<NanoUsd>, BudgetScope)> =
                Vec::new();
            for (k, v) in &identity.tags {
                let kv = format!("{k}:{v}");
                if let Some(entry) = cfg.tag_budgets.get(&kv) {
                    tag_entries.push((
                        tag_spend_key(&org_id, &kv, &period),
                        entry.soft_cap_usd.map(NanoUsd::from_f64_usd),
                        entry.hard_cap_usd.map(NanoUsd::from_f64_usd),
                        BudgetScope::Tag(kv),
                    ));
                }
            }
            tag_entries.sort_by(|a, b| a.3.sort_key().cmp(&b.3.sort_key()));
            check_list.extend(tag_entries);

            // Early-exit: no configured caps match this request.
            if check_list.is_empty() {
                req.extensions_mut().insert(TeamTagCheckResult {
                    min_remaining_nano_usd: None,
                });
                return inner.call(req).await;
            }

            // Step 5: Redis pipeline GET all keys (single round trip).
            let spend_values: Vec<NanoUsd> = {
                let rp = redis_pool.read().await.clone();
                match rp.get().await {
                    Err(e) => {
                        tracing::warn!(
                            event = "team_tag_budget_check_skipped",
                            error = %e,
                            "TeamTagBudgetLayer: Redis connection failed — passing through (fail-open)"
                        );
                        req.extensions_mut().insert(TeamTagCheckResult {
                            min_remaining_nano_usd: None,
                        });
                        return inner.call(req).await;
                    }
                    Ok(mut conn) => {
                        let mut pipe = redis::pipe();
                        for (key, _, _, _) in &check_list {
                            pipe.cmd("GET").arg(key);
                        }
                        match pipe.query_async::<Vec<Option<i64>>>(&mut *conn).await {
                            Err(e) => {
                                tracing::warn!(
                                    event = "team_tag_budget_check_skipped",
                                    error = %e,
                                    "TeamTagBudgetLayer: Redis GET pipeline failed — passing through (fail-open)"
                                );
                                req.extensions_mut().insert(TeamTagCheckResult {
                                    min_remaining_nano_usd: None,
                                });
                                return inner.call(req).await;
                            }
                            Ok(raw) => raw
                                .into_iter()
                                .map(|v| NanoUsd(v.unwrap_or(0).max(0) as u64))
                                .collect(),
                        }
                    }
                }
            };

            // Step 6: Hard-cap enforcement (team first by construction, then sorted tags).
            for ((_, _, hard_cap, scope), spend) in check_list.iter().zip(spend_values.iter()) {
                if let Some(hard_cap) = hard_cap
                    && spend >= hard_cap
                {
                    let body = match scope {
                        BudgetScope::Team(name) => {
                            json!({"error": "team_budget_exceeded", "team": name})
                        }
                        BudgetScope::Tag(kv) => {
                            json!({"error": "tag_budget_exceeded", "tag": kv})
                        }
                        BudgetScope::Identity(_) => {
                            // Identity scope is never added to check_list in this middleware.
                            // Log and skip rather than panic to stay safe under future refactors.
                            tracing::error!(
                                event = "team_tag_budget_unexpected_identity_scope",
                                "TeamTagBudgetLayer: unexpected Identity scope in check_list \
                                 — skipping hard-cap enforcement for this entry"
                            );
                            continue;
                        }
                    };
                    let mut response = match axum::response::Response::builder()
                        .status(axum::http::StatusCode::TOO_MANY_REQUESTS)
                        .header(axum::http::header::CONTENT_TYPE, "application/json")
                        .header(CostHeader::BUDGET_REMAINING, "0.000000")
                        .body(axum::body::Body::from(body.to_string()))
                    {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::error!(
                                error = %e,
                                event = "team_tag_budget_response_build_failed",
                                "TeamTagBudgetLayer: 429 response build failed — passing through"
                            );
                            return inner.call(req).await;
                        }
                    };
                    response.extensions_mut().insert(TeamTagCheckResult {
                        min_remaining_nano_usd: Some(NanoUsd::zero()),
                    });
                    return Ok(response);
                }
            }

            // Step 7: Soft-cap threshold logging (non-blocking).
            // Delegates to the shared `check_soft_cap_thresholds` helper (budget.rs), which
            // deduplicates via Redis SET NX and emits a structured warn per scope type.
            {
                let rp = redis_pool.read().await.clone();
                if let Ok(mut conn) = rp.get().await {
                    for ((_, soft_cap, _, scope), spend) in
                        check_list.iter().zip(spend_values.iter())
                    {
                        if let Some(soft_cap) = soft_cap {
                            check_soft_cap_thresholds(
                                &mut *conn,
                                scope,
                                *spend,
                                *soft_cap,
                                &org_id,
                                warn_dedup_secs,
                            )
                            .await;
                        }
                    }
                }
            }

            // Step 8: Compute min_remaining across all entries with a hard cap only.
            let min_remaining = check_list
                .iter()
                .zip(spend_values.iter())
                .filter_map(|((_, _soft, hard, _), spend)| hard.map(|cap| cap - *spend))
                .min();

            // Step 9: Insert result into extensions; call inner service.
            req.extensions_mut().insert(TeamTagCheckResult {
                min_remaining_nano_usd: min_remaining,
            });
            inner.call(req).await
        })
    }
}

// ─── TeamTagBudgetResponseLayer ──────────────────────────────────────────────

/// Tower layer that injects `CostHeader::BUDGET_REMAINING` using most-restrictive-wins semantics.
///
/// No constructor arguments — reads `TeamTagCheckResult` from request extensions only.
#[derive(Clone, Default)]
pub struct TeamTagBudgetResponseLayer;

impl TeamTagBudgetResponseLayer {
    /// Create a `TeamTagBudgetResponseLayer`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for TeamTagBudgetResponseLayer {
    type Service = TeamTagBudgetResponseService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TeamTagBudgetResponseService { inner }
    }
}

/// Inner response service for most-restrictive-wins `CostHeader::BUDGET_REMAINING` header.
#[derive(Clone)]
pub struct TeamTagBudgetResponseService<S> {
    inner: S,
}

impl<S, E> Service<Request> for TeamTagBudgetResponseService<S>
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
        // Capture TeamTagCheckResult before the request is consumed by inner.
        let check = req.extensions().get::<TeamTagCheckResult>().cloned();

        let mut inner = self.inner.clone();
        std::mem::swap(&mut inner, &mut self.inner);

        Box::pin(async move {
            let mut response = inner.call(req).await?;

            let Some(result) = check else {
                return Ok(response);
            };
            let Some(remaining) = result.min_remaining_nano_usd else {
                return Ok(response);
            };

            let our_value = nanos_to_usd_display(remaining);

            // Most-restrictive-wins: only set our value if it is strictly lower than
            // any existing CostHeader::BUDGET_REMAINING header (e.g. from BudgetResponseLayer).
            let existing_f64: Option<f64> = response
                .headers()
                .get(CostHeader::BUDGET_REMAINING)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<f64>().ok());

            let our_f64: Option<f64> = our_value.parse::<f64>().ok();

            let should_set = match (existing_f64, our_f64) {
                (None, _) => true,
                (Some(_), None) => false,
                (Some(existing), Some(ours)) => ours < existing,
            };

            if should_set {
                match HeaderValue::from_str(&our_value) {
                    Ok(value) => {
                        response
                            .headers_mut()
                            .insert(CostHeader::BUDGET_REMAINING, value);
                    }
                    Err(error) => {
                        tracing::error!(
                            event = "team_tag_budget_remaining_header_invalid",
                            value = %our_value,
                            error = %error,
                        );
                    }
                }
            }

            Ok(response)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::budget::THRESHOLDS_PCT;
    use crate::utils::tag_spend_key;

    #[test]
    fn test_team_tag_check_list_tag_key_format() {
        // Verify that format!("{k}:{v}") produces the correct lookup key for tag_spend_key.
        let k = "project";
        let v = "chat-bot";
        let kv = format!("{k}:{v}");
        assert_eq!(kv, "project:chat-bot");
        assert_eq!(
            tag_spend_key("acme", &kv, ""),
            "oxigate:org:acme:tag:project:chat-bot:spend"
        );
    }

    #[test]
    fn test_crossed_thresholds_team_tag() {
        // Verify spend.pct_of(soft_cap) threshold logic directly (not via budget.rs).
        let soft_cap = NanoUsd::from_f64_usd(10.0); // $10
        let scenarios: &[(u64, &[u8])] = &[
            (7_900_000_000, &[]),
            (8_000_000_000, &[80]),
            (9_500_000_000, &[80, 90]),
            (10_000_000_000, &[80, 90, 100]),
            (10_500_000_000, &[80, 90, 100]),
        ];
        for (spend_nano, expected_pcts) in scenarios {
            let spend = NanoUsd(*spend_nano);
            let crossed: Vec<u8> = THRESHOLDS_PCT
                .into_iter()
                .filter(|pct| spend.pct_of(soft_cap) >= u64::from(*pct))
                .collect();
            assert_eq!(&crossed, expected_pcts, "spend={spend_nano}");
        }
    }

    #[test]
    fn test_min_remaining_calculation() {
        // Simulate check_list with multiple hard-capped entries so min() selection is exercised.
        let team_hard_cap = NanoUsd::from_f64_usd(100.0);
        let tag1_hard_cap = NanoUsd::from_f64_usd(50.0);
        let tag2_hard_cap = NanoUsd::from_f64_usd(200.0);

        let check_list: Vec<(String, Option<NanoUsd>, Option<NanoUsd>, BudgetScope)> = vec![
            (
                "k1".into(),
                None,
                Some(team_hard_cap),
                BudgetScope::Team("eng".into()),
            ),
            (
                "k2".into(),
                None,
                Some(tag1_hard_cap),
                BudgetScope::Tag("project:a".into()),
            ),
            (
                "k3".into(),
                None,
                Some(tag2_hard_cap),
                BudgetScope::Tag("project:b".into()),
            ),
        ];
        let spend_values = vec![
            NanoUsd::from_f64_usd(80.0), // team: $80/$100 → remaining $20
            NanoUsd::from_f64_usd(60.0), // tag1: $60/$50  → saturated to $0 (over hard cap)
            NanoUsd::from_f64_usd(10.0), // tag2: $10/$200 → remaining $190
        ];
        let min_remaining = check_list
            .iter()
            .zip(spend_values.iter())
            // Production semantics: min remaining is computed across hard caps only.
            .filter_map(|((_, _soft, hard, _), spend)| hard.map(|cap| cap - *spend))
            .min();
        // Minimum is $0 (tag1) because it is over its hard cap and NanoUsd saturates at zero.
        assert_eq!(min_remaining, Some(NanoUsd::zero()));
    }

    #[test]
    fn test_tag_sort_order_deterministic() {
        // Two exhausted tags — alphabetically-first exhausted tag should always be reported.
        let mut entries: Vec<(String, Option<NanoUsd>, Option<NanoUsd>, BudgetScope)> = vec![
            (
                "k_b".into(),
                None,
                Some(NanoUsd::from_f64_usd(10.0)),
                BudgetScope::Tag("project:b".into()),
            ),
            (
                "k_a".into(),
                None,
                Some(NanoUsd::from_f64_usd(10.0)),
                BudgetScope::Tag("project:a".into()),
            ),
        ];
        entries.sort_by(|a, b| a.3.sort_key().cmp(&b.3.sort_key()));

        // After sort, "project:a" must be first.
        assert_eq!(entries[0].3, BudgetScope::Tag("project:a".into()));
        assert_eq!(entries[1].3, BudgetScope::Tag("project:b".into()));
    }
}
