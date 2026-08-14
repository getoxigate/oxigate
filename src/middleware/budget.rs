// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Budget middleware for per-identity soft-cap checks.
//!
//! Community tier — compiled for all builds.
//!
//! `BudgetLayer` runs pre-dispatch, reads current spend from Redis, emits deduplicated
//! threshold warnings (80/90/100), and stores `BudgetCheckResult` in request extensions.
//! `BudgetResponseLayer` runs post-dispatch and injects CostHeader::BUDGET_REMAINING header.

use std::task::{Context, Poll};

use axum::extract::Request;
use axum::http::HeaderValue;
use axum::response::Response;
use chrono::{Duration, Utc};
use tower::Layer;
use tower::Service;

use crate::config::{BudgetConfig, BudgetDuration};
use crate::domain::auth::RequestIdentity;
use crate::domain::ports::{BudgetScope, NanoUsd};
use crate::redis_pool::RedisPool;
use crate::utils::CostHeader;
use crate::utils::read_identity_spend;
use crate::utils::{get_next_standardized_reset_time, identity_spend_key, period_key};

pub(crate) const THRESHOLDS_PCT: [u8; 3] = [80, 90, 100];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DedupAction {
    EmitWarn,
    SkipWarn,
    ContinueOnFailure,
}

/// Request-scoped budget check output used by `BudgetResponseLayer` for header injection.
///
/// Shared within the Pro feature set: `HardCapLayer` reads `spend_nano_usd` from this
/// struct (set in request extensions by `BudgetLayer`) to avoid a second Redis GET for
/// the same identity spend key per request. Remains `pub(crate)` — no public API.
#[derive(Debug, Clone)]
pub(crate) struct BudgetCheckResult {
    /// Current spend for this identity in NanoUsd.
    pub(crate) spend_nano_usd: NanoUsd,
    /// Configured soft cap in NanoUsd.
    pub(crate) cap_nano_usd: NanoUsd,
}

/// Runtime budget configuration consumed by budget middleware.
///
/// Keeps soft-cap and hard-cap converted to NanoUsd so request handling remains integer-only.
/// stores resolved `budget_duration` timezone and next reset instant for lazy rollover.
#[derive(Debug, Clone)]
pub struct BudgetRuntimeConfig {
    soft_cap_nano_usd: Option<NanoUsd>,
    warn_dedup_period_secs: u32,
    hard_cap_nano_usd: Option<NanoUsd>,
    /// Resolved reset cadence (from `budget.budget_duration`).
    pub duration: BudgetDuration,
    /// IANA timezone for period boundaries.
    pub tz: chrono_tz::Tz,
    /// Next automatic boundary (UTC). May be explicit `budget_reset_at` from config.
    pub next_reset_at: chrono::DateTime<chrono::Utc>,
    /// Pro scheduler wake interval (seconds).
    pub scheduler_interval_secs: u64,
    /// When set, used instead of `Utc::now()` for period keys and lazy reset.
    /// **Never enable `test-hooks` in production** — this field exists only under
    /// `cfg(any(test, feature = "test-hooks"))` so integration tests can pin time.
    #[cfg(any(test, feature = "test-hooks"))]
    pub now_override: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for BudgetRuntimeConfig {
    fn default() -> Self {
        Self::from_budget_config(BudgetConfig::default())
    }
}

impl BudgetRuntimeConfig {
    /// Build runtime budget config from user-facing config.
    #[must_use]
    pub fn from_budget_config(config: BudgetConfig) -> Self {
        let duration = config.resolved_duration();
        let tz = config.resolved_timezone();
        let now = Utc::now();
        let next_reset_at = config
            .budget_reset_at
            .unwrap_or_else(|| get_next_standardized_reset_time(duration, now, tz));
        Self {
            soft_cap_nano_usd: config.soft_cap_usd.map(NanoUsd::from_f64_usd),
            warn_dedup_period_secs: config.warn_dedup_period_secs(),
            hard_cap_nano_usd: config.hard_cap_usd.map(NanoUsd::from_f64_usd),
            duration,
            tz,
            next_reset_at,
            scheduler_interval_secs: config.scheduler_interval_secs,
            #[cfg(any(test, feature = "test-hooks"))]
            now_override: None,
        }
    }

    pub fn resolved_now(&self) -> chrono::DateTime<chrono::Utc> {
        #[cfg(any(test, feature = "test-hooks"))]
        if let Some(t) = self.now_override {
            return t;
        }
        Utc::now()
    }

    /// Returns the hard cap in NanoUsd, if configured.
    #[must_use]
    pub fn hard_cap_nano_usd(&self) -> Option<NanoUsd> {
        self.hard_cap_nano_usd
    }

    /// Effective cap for BudgetCheckResult population and CostHeader::BUDGET_REMAINING header:
    /// hard_cap only — response header reflects the enforcement boundary.
    /// Returns None when hard_cap is not configured (header omitted).
    #[must_use]
    pub fn effective_response_cap_nano_usd(&self) -> Option<NanoUsd> {
        self.hard_cap_nano_usd
    }
}

/// Tower layer that performs budget check before dispatch.
#[derive(Clone)]
pub struct BudgetLayer {
    redis_pool: std::sync::Arc<tokio::sync::RwLock<RedisPool>>,
    budget_config: std::sync::Arc<tokio::sync::RwLock<BudgetRuntimeConfig>>,
}

impl BudgetLayer {
    /// Build a BudgetLayer from shared runtime budget config.
    #[must_use]
    pub fn new(
        redis_pool: std::sync::Arc<tokio::sync::RwLock<RedisPool>>,
        budget_config: std::sync::Arc<tokio::sync::RwLock<BudgetRuntimeConfig>>,
    ) -> Self {
        Self {
            redis_pool,
            budget_config,
        }
    }
}

impl<S> Layer<S> for BudgetLayer {
    type Service = BudgetService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        BudgetService {
            inner,
            redis_pool: std::sync::Arc::clone(&self.redis_pool),
            budget_config: std::sync::Arc::clone(&self.budget_config),
        }
    }
}

/// Inner service for pre-dispatch budget checks.
#[derive(Clone)]
pub struct BudgetService<S> {
    inner: S,
    redis_pool: std::sync::Arc<tokio::sync::RwLock<RedisPool>>,
    budget_config: std::sync::Arc<tokio::sync::RwLock<BudgetRuntimeConfig>>,
}

impl<S, E> Service<Request> for BudgetService<S>
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
            let budget_cfg = budget_config.read().await.clone();
            // Run when either soft_cap or hard_cap is configured:
            // - soft_cap: threshold WARNs only (non-blocking)
            // - hard_cap: enforcement (HardCapLayer) + response header injection (BudgetResponseLayer)
            let hard_cap_nano_usd = budget_cfg.hard_cap_nano_usd;
            let soft_cap_nano_usd = budget_cfg.soft_cap_nano_usd;

            if hard_cap_nano_usd.is_some() || soft_cap_nano_usd.is_some() {
                // Auth-disabled deployments can omit RequestIdentity; default identity
                // intentionally collapses spend into a single shared budget bucket.
                let identity = req
                    .extensions()
                    .get::<RequestIdentity>()
                    .cloned()
                    .unwrap_or_default();

                let now = budget_cfg.resolved_now();
                let period = period_key(budget_cfg.duration, now, budget_cfg.tz);
                let spend_key = identity_spend_key(&identity.org_id, &identity.id, &period);
                let spend_nano_usd: NanoUsd =
                    read_identity_spend(&redis_pool, &spend_key, &identity, "budget_check_skipped")
                        .await
                        .unwrap_or_default();

                if budget_cfg.duration != BudgetDuration::None && now >= budget_cfg.next_reset_at {
                    let old_period = period_key(
                        budget_cfg.duration,
                        budget_cfg.next_reset_at - Duration::seconds(1),
                        budget_cfg.tz,
                    );
                    let mut w = budget_config.write().await;
                    if now >= w.next_reset_at {
                        w.next_reset_at = get_next_standardized_reset_time(w.duration, now, w.tz);
                        log_budget_period_crossed(&identity.id, &old_period, &period);
                    }
                }

                // Threshold WARNs fire only when soft_cap is configured.
                // When only hard_cap is set, warn_cap is None and no WARNs are emitted.
                if let Some(warn_cap) = soft_cap_nano_usd {
                    let scope = BudgetScope::Identity(identity.id.clone());
                    let pool = redis_pool.read().await.clone();
                    match pool.get().await {
                        Ok(mut conn) => {
                            check_soft_cap_thresholds(
                                &mut *conn,
                                &scope,
                                spend_nano_usd,
                                warn_cap,
                                &identity.org_id,
                                budget_cfg.warn_dedup_period_secs,
                            )
                            .await;
                        }
                        Err(error) => {
                            tracing::warn!(
                                event = "budget_check_skipped",
                                reason = "redis_unavailable",
                                identity_id = %identity.id,
                                org_id = %identity.org_id,
                                error = %error,
                            );
                        }
                    }
                }

                // Response header reflects ONLY the hard-cap enforcement boundary.
                // When no hard cap is configured, omit the header (BudgetResponseLayer sees None).
                if let Some(hard_cap_nano_usd) = hard_cap_nano_usd {
                    req.extensions_mut().insert(BudgetCheckResult {
                        spend_nano_usd,
                        cap_nano_usd: hard_cap_nano_usd,
                    });
                }
            }

            inner.call(req).await
        })
    }
}

/// Tower response wrapper that injects CostHeader::BUDGET_REMAINING when budget is configured.
#[derive(Clone, Default)]
pub struct BudgetResponseLayer;

impl BudgetResponseLayer {
    /// Create a budget response wrapper layer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for BudgetResponseLayer {
    type Service = BudgetResponseService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        BudgetResponseService { inner }
    }
}

/// Inner response service that injects `CostHeader::BUDGET_REMAINING`.
#[derive(Clone)]
pub struct BudgetResponseService<S> {
    inner: S,
}

impl<S, E> Service<Request> for BudgetResponseService<S>
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
        let check = req.extensions().get::<BudgetCheckResult>().cloned();

        let mut inner = self.inner.clone();
        std::mem::swap(&mut inner, &mut self.inner);

        Box::pin(async move {
            let mut response = inner.call(req).await?;
            if let Some(check) = check {
                let remaining_nano = check.cap_nano_usd - check.spend_nano_usd;
                let remaining_display = crate::utils::nanos_to_usd_display(remaining_nano);
                match HeaderValue::from_str(&remaining_display) {
                    Ok(value) => {
                        response
                            .headers_mut()
                            .insert(CostHeader::BUDGET_REMAINING, value);
                    }
                    Err(error) => {
                        tracing::error!(
                            event = "budget_remaining_header_invalid",
                            value = %remaining_display,
                            error = %error,
                        );
                    }
                }
            }
            Ok(response)
        })
    }
}

/// Check soft-cap thresholds for a budget scope and emit deduplicated warnings.
///
/// For each threshold in `THRESHOLDS_PCT` that `spend` has crossed relative to `soft_cap`,
/// attempts a Redis `SET NX` on a dedup key. Emits a structured `tracing::warn!` only on the
/// first crossing (i.e., when SET NX returns `Some(_)`). Per-threshold Redis errors are treated
/// as `ContinueOnFailure` so higher thresholds are still attempted. Connection-level errors are
/// the caller's responsibility to surface before calling this function.
///
/// Shared by `BudgetService` (identity scope) and `TeamTagBudgetService` (team/tag scopes).
pub(crate) async fn check_soft_cap_thresholds(
    conn: &mut impl redis::aio::ConnectionLike,
    scope: &BudgetScope,
    spend: NanoUsd,
    soft_cap: NanoUsd,
    org_id: &str,
    warn_dedup_secs: u32,
) {
    for pct in crossed_thresholds(spend, soft_cap) {
        let dedup_key = scope.warn_dedup_key(org_id, pct);
        let reply: Option<bool> = redis::cmd("SET")
            .arg(&dedup_key)
            .arg("1")
            .arg("NX")
            .arg("EX")
            .arg(warn_dedup_secs)
            .query_async::<Option<String>>(conn)
            .await
            .ok()
            .map(|r| r.is_some());
        match dedup_action(reply) {
            DedupAction::EmitWarn => match scope {
                BudgetScope::Identity(id) => tracing::warn!(
                    event = "budget_threshold_reached",
                    threshold_pct = pct,
                    identity_id = id,
                    org_id,
                    spend_nano_usd = spend.as_u64(),
                    cap_nano_usd = soft_cap.as_u64(),
                    spend_usd = spend.to_display_string(),
                    cap_usd = soft_cap.to_display_string(),
                ),
                BudgetScope::Team(name) => tracing::warn!(
                    event = "budget_threshold_reached",
                    team = name,
                    threshold_pct = pct,
                    spend_nano_usd = spend.as_u64(),
                    cap_nano_usd = soft_cap.as_u64(),
                    spend_usd = spend.to_display_string(),
                    cap_usd = soft_cap.to_display_string(),
                    org_id,
                    "team budget threshold crossed"
                ),
                BudgetScope::Tag(kv) => tracing::warn!(
                    event = "budget_threshold_reached",
                    tag_kv = kv,
                    threshold_pct = pct,
                    spend_nano_usd = spend.as_u64(),
                    cap_nano_usd = soft_cap.as_u64(),
                    spend_usd = spend.to_display_string(),
                    cap_usd = soft_cap.to_display_string(),
                    org_id,
                    "tag budget threshold crossed"
                ),
            },
            DedupAction::SkipWarn => {}
            DedupAction::ContinueOnFailure => continue,
        }
    }
}

fn crossed_thresholds(spend: NanoUsd, cap: NanoUsd) -> impl Iterator<Item = u8> {
    let spend_pct = spend.pct_of(cap);
    THRESHOLDS_PCT
        .into_iter()
        .filter(move |threshold| spend_pct >= u64::from(*threshold))
}

fn dedup_action(dedup_result: Option<bool>) -> DedupAction {
    match dedup_result {
        Some(true) => DedupAction::EmitWarn,
        Some(false) => DedupAction::SkipWarn,
        None => DedupAction::ContinueOnFailure,
    }
}

pub(crate) fn log_budget_period_crossed(identity_id: &str, old_period: &str, new_period: &str) {
    tracing::info!(
        identity_id = %identity_id,
        old_period = %old_period,
        new_period = %new_period,
        "budget period boundary crossed"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_test::traced_test;

    #[traced_test]
    #[test]
    fn test_budget_period_boundary_log_shape() {
        log_budget_period_crossed("id-1", "2026-01", "2026-02");
        assert!(logs_contain("budget period boundary crossed"));
        assert!(logs_contain("id-1"));
    }

    #[test]
    fn test_budget_runtime_config_converts_once() {
        let runtime = BudgetRuntimeConfig::from_budget_config(BudgetConfig {
            budget_reset_at: None,
            global_safety_cap_usd: None,
            budget_duration: Some("1d".into()),
            timezone: "UTC".into(),
            soft_cap_usd: Some(10.0),
            hard_cap_usd: None,
            ..BudgetConfig::default()
        });
        assert_eq!(runtime.soft_cap_nano_usd, Some(NanoUsd(10_000_000_000)));
        assert_eq!(runtime.warn_dedup_period_secs, 86_400);
    }

    #[test]
    fn test_remaining_display_saturating() {
        let remaining = NanoUsd(10_000_000_000) - NanoUsd(12_000_000_000);
        assert_eq!(remaining, NanoUsd(0));
        assert_eq!(crate::utils::nanos_to_usd_display(remaining), "0.000000");
    }

    #[test]
    fn test_budget_key_shapes() {
        use crate::domain::ports::BudgetScope;
        assert_eq!(
            identity_spend_key("acme", "k1", ""),
            "oxigate:org:acme:spend:k1".to_string()
        );
        // Identity dedup key includes "identity:" segment prefix.
        assert_eq!(
            BudgetScope::Identity("k1".to_owned()).warn_dedup_key("acme", 90),
            "oxigate:budget:warned:acme:identity:k1:90".to_string()
        );
    }

    #[test]
    fn test_crossed_thresholds_required_scenarios() {
        let scenarios = [
            (7_900_000_000_u64, vec![]),
            (8_000_000_000_u64, vec![80_u8]),
            (9_500_000_000_u64, vec![80_u8, 90_u8]),
            (10_000_000_000_u64, vec![80_u8, 90_u8, 100_u8]),
            (10_500_000_000_u64, vec![80_u8, 90_u8, 100_u8]),
        ];

        for (spend_nano_usd, expected) in scenarios {
            let actual: Vec<u8> =
                crossed_thresholds(NanoUsd(spend_nano_usd), NanoUsd(10_000_000_000)).collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_dedup_failure_action_is_continue() {
        assert_eq!(dedup_action(None), DedupAction::ContinueOnFailure);
    }

    #[test]
    fn test_dedup_failure_does_not_short_circuit_higher_thresholds() {
        let crossed: Vec<u8> =
            crossed_thresholds(NanoUsd(10_500_000_000), NanoUsd(10_000_000_000)).collect();
        let dedup_results = [None, Some(true), Some(true)];
        let mut emitted = Vec::new();

        for (index, threshold) in crossed.into_iter().enumerate() {
            let action = dedup_action(dedup_results[index]);
            match action {
                DedupAction::EmitWarn => emitted.push(threshold),
                DedupAction::SkipWarn => {}
                DedupAction::ContinueOnFailure => continue,
            }
        }

        assert_eq!(emitted, vec![90_u8, 100_u8]);
    }

    #[test]
    fn test_crossed_thresholds_handles_large_values_without_overflow() {
        let crossed: Vec<u8> = crossed_thresholds(NanoUsd(u64::MAX), NanoUsd(u64::MAX)).collect();
        assert_eq!(crossed, vec![80_u8, 90_u8, 100_u8]);
    }

    #[test]
    fn test_jump_to_over_cap_crosses_all_thresholds_in_one_evaluation() {
        let crossed: Vec<u8> =
            crossed_thresholds(NanoUsd(10_500_000_000), NanoUsd(10_000_000_000)).collect();
        let mut emitted = Vec::new();

        for threshold in crossed {
            if dedup_action(Some(true)) == DedupAction::EmitWarn {
                emitted.push(threshold);
            }
        }

        assert_eq!(emitted, vec![80_u8, 90_u8, 100_u8]);
    }
}
