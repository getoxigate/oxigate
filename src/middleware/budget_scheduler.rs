// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Pro-only background task: zero current-period per-identity spend keys at boundaries .
//!
//! Complements lazy reset in `BudgetLayer` by proactively clearing Redis counters on a timer.

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::config::BudgetDuration;
use crate::middleware::budget::BudgetRuntimeConfig;
use crate::redis_pool::RedisPool;
use crate::utils::{
    get_next_standardized_reset_time, identity_spend_key, period_key, spend_key_ttl_secs,
};

/// Wakes periodically and, when effective `now` has passed `next_reset_at`, sets each
/// discovered identity's **current** period spend key to `0` (with TTL).
pub struct BudgetResetScheduler {
    runtime_cfg: Arc<RwLock<BudgetRuntimeConfig>>,
    redis_pool: Arc<RwLock<RedisPool>>,
}

impl BudgetResetScheduler {
    /// Create a scheduler bound to shared runtime config and Redis.
    #[must_use]
    pub fn new(
        runtime_cfg: Arc<RwLock<BudgetRuntimeConfig>>,
        redis_pool: Arc<RwLock<RedisPool>>,
    ) -> Self {
        Self {
            runtime_cfg,
            redis_pool,
        }
    }

    /// Run until process exit. Sleeps `scheduler_interval_secs` between cycles.
    pub async fn run(self) {
        loop {
            let wait_secs = self.runtime_cfg.read().await.scheduler_interval_secs.max(1);
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            let failures = self.wake_cycle().await;
            if failures > 0 {
                tracing::warn!(failures, "budget scheduler wake cycle encountered errors");
            }
        }
    }

    async fn wake_cycle(&self) -> usize {
        let rt = self.runtime_cfg.read().await.clone();
        if rt.duration == BudgetDuration::None {
            return 0;
        }
        let now = rt.resolved_now();
        if now < rt.next_reset_at {
            return 0;
        }

        let current_period = period_key(rt.duration, now, rt.tz);
        let ttl = spend_key_ttl_secs(rt.duration);

        let pool = self.redis_pool.read().await.clone();
        let mut conn = match pool.get().await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "budget scheduler: redis unavailable");
                return 1;
            }
        };

        let pairs = match scan_spend_pairs(&mut conn, rt.duration, &current_period).await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(error = %e, "budget scheduler: scan failed");
                return 1;
            }
        };

        let mut failures = 0usize;
        for (org_id, identity_id) in pairs {
            let key = identity_spend_key(&org_id, &identity_id, &current_period);
            let mut pipe = redis::pipe();
            // SET NX: initialize only; lazy reset may already hold spend in this key.
            // EXPIRE always runs — refreshing TTL on existing keys is intentional (harmless).
            pipe.cmd("SET").arg(&key).arg(0i64).arg("NX").ignore();
            pipe.cmd("EXPIRE").arg(&key).arg(ttl).ignore();
            if let Err(e) = pipe.query_async::<()>(&mut *conn).await {
                tracing::warn!(
                    key = %key,
                    error = %e,
                    "failed to reset budget for identity"
                );
                failures += 1;
            }
        }

        let mut w = self.runtime_cfg.write().await;
        if now >= w.next_reset_at {
            w.next_reset_at = get_next_standardized_reset_time(w.duration, now, w.tz);
        }

        failures
    }
}

#[cfg(feature = "test-hooks")]
impl BudgetResetScheduler {
    /// Integration-test entry point for [`Self::wake_cycle`] (not part of production API).
    pub async fn wake_cycle_for_test(&self) -> usize {
        self.wake_cycle().await
    }
}

async fn scan_spend_pairs(
    conn: &mut redis::aio::MultiplexedConnection,
    duration: BudgetDuration,
    current_period: &str,
) -> Result<HashSet<(String, String)>, redis::RedisError> {
    let mut out = HashSet::new();
    let period_suffix = format!(":{current_period}");
    let mut cursor: u64 = 0;
    loop {
        let (next, keys): (u64, Vec<String>) = redis::cmd("SCAN")
            .arg(cursor)
            .arg("MATCH")
            .arg("oxigate:org:*:spend:*")
            .arg("COUNT")
            .arg(1000)
            .query_async(conn)
            .await?;
        for key in keys {
            if !spend_key_matches_period_scan(&key, duration, &period_suffix) {
                continue;
            }
            if let Some(pair) = parse_org_identity_from_spend_key(&key) {
                out.insert(pair);
            }
        }
        cursor = next;
        if cursor == 0 {
            break;
        }
    }
    Ok(out)
}

/// Limit SCAN to keys relevant for this wake: current period suffix or legacy unprefixed keys.
fn spend_key_matches_period_scan(key: &str, duration: BudgetDuration, period_suffix: &str) -> bool {
    if duration == BudgetDuration::None {
        return false;
    }
    if key.ends_with(period_suffix) {
        return true;
    }
    // Legacy (pre-period-keyed) migration: `oxigate:org:{org}:spend:{id}` (exactly five ':'-separated segments).
    let parts: Vec<&str> = key.split(':').collect();
    parts.len() == 5
        && parts.first() == Some(&"oxigate")
        && parts.get(1) == Some(&"org")
        && parts.get(3) == Some(&"spend")
}

/// Parse `oxigate:org:{org_id}:spend:{identity_id}` or the same with `:{period}` appended.
///
/// **Assumption:** `identity_id` does not contain `:` (API key hashes, UUIDs without colons).
/// If that ever changes, this must join `parts[4..]` up to the optional period segment instead
/// of using `parts[4]` alone.
fn parse_org_identity_from_spend_key(key: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = key.split(':').collect();
    if parts.len() < 5 {
        return None;
    }
    if parts.first() != Some(&"oxigate")
        || parts.get(1) != Some(&"org")
        || parts.get(3) != Some(&"spend")
    {
        return None;
    }
    let org_id = parts.get(2)?.to_string();
    let identity_id = parts.get(4)?.to_string();
    Some((org_id, identity_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_spend_key_with_period_suffix() {
        assert_eq!(
            parse_org_identity_from_spend_key("oxigate:org:acme:spend:key1:2026-03"),
            Some(("acme".into(), "key1".into()))
        );
    }

    #[test]
    fn parse_spend_key_unprefixed_period() {
        assert_eq!(
            parse_org_identity_from_spend_key("oxigate:org:acme:spend:key1"),
            Some(("acme".into(), "key1".into()))
        );
    }
}
