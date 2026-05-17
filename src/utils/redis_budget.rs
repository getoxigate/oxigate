// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Shared Redis spend-read helper for Pro budget middleware.
//!
//! Extracted to eliminate the duplicated `get_spend_nano_usd` in `BudgetLayer` and
//! `HardCapLayer`. Both layers call this with a caller-specific `skip_event` so that
//! structured log events remain distinguishable in ops tooling.
//!
//! Gated under `#[cfg(feature = "pro")]` via `utils/mod.rs` — not compiled for Community.

use std::sync::Arc;

use tokio::sync::RwLock;

use crate::domain::auth::RequestIdentity;
use crate::domain::ports::NanoUsd;
use crate::redis_pool::RedisPool;

/// Read the current spend for an identity key from Redis.
///
/// Returns `None` on any Redis failure (connection or query); callers must fail-open.
/// `skip_event` is emitted as the structured `event` field on warn — use a caller-specific
/// string (e.g. `"budget_check_skipped"` or `"hard_cap_check_skipped"`) so ops can
/// distinguish which layer triggered the skip.
pub async fn read_identity_spend(
    redis_pool: &Arc<RwLock<RedisPool>>,
    key: &str,
    identity: &RequestIdentity,
    skip_event: &'static str,
) -> Option<NanoUsd> {
    let pool = redis_pool.read().await.clone();
    let mut conn = match pool.get().await {
        Ok(conn) => conn,
        Err(error) => {
            tracing::warn!(
                event = skip_event,
                reason = "redis_unavailable",
                identity_id = %identity.id,
                org_id = %identity.org_id,
                error = %error,
            );
            return None;
        }
    };

    match redis::cmd("GET")
        .arg(key)
        .query_async::<Option<u64>>(&mut *conn)
        .await
    {
        Ok(Some(raw)) => Some(NanoUsd(raw)),
        Ok(None) => Some(NanoUsd::zero()),
        Err(error) => {
            tracing::warn!(
                event = skip_event,
                reason = "redis_unavailable",
                identity_id = %identity.id,
                org_id = %identity.org_id,
                error = %error,
            );
            None
        }
    }
}
