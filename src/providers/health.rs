// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Provider health tracker, in-flight guard, and model-matching helpers.
//!
//! Tracks per-provider health status, EWMA latency, 429-cooldown state,
//! and in-flight request counts. Provides the `candidates()` snapshot method
//! that strategies consume. All state mutations are internally synchronised;
//! callers hold no locks across await points.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

use crate::domain::ports::{
    HealthStatus, NanoUsd, ProviderAdapter, ProviderCandidate, ProviderKind,
};
use crate::domain::pricing::PricingDb;
use crate::redis_pool::RedisPool;

/// Circuit-breaker cooldown state for a provider .
///
/// State transitions:
/// - `Closed`  → `Open`     : on_rate_limit() called
/// - `Open`    → `HalfOpen` : cooldown elapsed; first candidates() call after expiry
/// - `HalfOpen`→ `Closed`   : on_response() called (probe succeeded)
/// - `HalfOpen`→ `Open`     : on_rate_limit() called (probe failed; reset cooldown)
#[derive(Clone)]
pub enum CooldownState {
    /// No cooldown — provider is healthy.
    Closed,
    /// Provider is in cooldown after a 429/5xx. Excluded from candidate lists until `cooldown_until` elapses.
    Open { cooldown_until: Instant },
    /// Cooldown elapsed; one probe request is allowed per node.
    /// `probe_taken` is set to `true` atomically by the first thread that claims the probe slot.
    HalfOpen { probe_taken: Arc<AtomicBool> },
}

/// Per-provider mutable state stored in the main RwLock.
pub struct ProviderHealthEntry {
    /// Last known health status (from startup / SIGHUP health checks).
    pub health_status: HealthStatus,
    /// Circuit-breaker cooldown state .
    pub cooldown_state: CooldownState,
    /// EWMA latency in milliseconds. 0.0 = no samples yet (cold-start sentinel).
    pub latency_ewma_ms: f64,
}

/// Tracks health, latency, cooldown, and in-flight counts for all providers.
///
/// Thread-safe — methods acquire internal locks and never expose guards.
/// `AppState.health: Arc<ProviderHealthTracker>` (no outer RwLock); the tracker
/// is mutated in-place on SIGHUP via `sync_providers()`, never swapped.
pub struct ProviderHealthTracker {
    /// Main mutable state: health status + cooldown timestamps + EWMA latency.
    state: Arc<RwLock<HashMap<String, ProviderHealthEntry>>>,
    /// In-flight counters, one `Arc<AtomicUsize>` per provider.
    ///
    /// Uses `std::sync::RwLock` (not tokio) so `InFlightGuard::new` can call
    /// `.read().unwrap()` from a sync context without blocking the async executor.
    /// The write lock is held only in `sync_providers` (SIGHUP), which has no
    /// `.await` inside the critical section.
    inflight: Arc<std::sync::RwLock<HashMap<String, Arc<AtomicUsize>>>>,
    /// Optional Redis pool for persistent cooldown keys (survives restart).
    redis: Option<Arc<RwLock<RedisPool>>>,
    /// Cooldown duration in seconds after a 429 response.
    /// Stored as `AtomicU64` so `update_routing_params` can update it on SIGHUP
    /// without needing a separate lock.
    cooldown_secs: AtomicU64,
    /// EWMA smoothing factor α (0 < α ≤ 1), stored as `f64::to_bits()`.
    /// Updated atomically on SIGHUP via `update_routing_params`.
    ewma_alpha: AtomicU64,
    /// Throttle Redis WARN logs — emit at most once per 60 seconds.
    last_redis_warn: tokio::sync::Mutex<Option<Instant>>,
    /// Throttle HALF-OPEN multi-node WARN — emit at most once per provider per 60 seconds.
    last_half_open_warn: tokio::sync::Mutex<HashMap<String, Instant>>,
}

impl ProviderHealthTracker {
    /// Creates a tracker for the given provider names.
    ///
    /// All providers start as `Healthy` with no cooldown and no latency samples.
    pub fn new(
        provider_names: &[String],
        redis: Option<Arc<RwLock<RedisPool>>>,
        cooldown_secs: u64,
        ewma_alpha: f64,
    ) -> Arc<Self> {
        let mut state = HashMap::new();
        let mut inflight = HashMap::new();
        for name in provider_names {
            state.insert(
                name.clone(),
                ProviderHealthEntry {
                    health_status: HealthStatus::Healthy,
                    cooldown_state: CooldownState::Closed,
                    latency_ewma_ms: 0.0,
                },
            );
            inflight.insert(name.clone(), Arc::new(AtomicUsize::new(0)));
        }
        // Only relevant for multi-node deployments. Single-node operators have no Redis
        // and the warning is noise for them.
        if redis.is_some() {
            tracing::warn!(
                "circuit breaker: HALF-OPEN probe coordination is node-local; \
                 in multi-node deployments up to N nodes may probe simultaneously. \
                 Distributed coordination via Redis is not yet implemented."
            );
        }
        Arc::new(Self {
            state: Arc::new(RwLock::new(state)),
            inflight: Arc::new(std::sync::RwLock::new(inflight)),
            redis,
            cooldown_secs: AtomicU64::new(cooldown_secs),
            ewma_alpha: AtomicU64::new(ewma_alpha.to_bits()),
            last_redis_warn: tokio::sync::Mutex::new(None),
            last_half_open_warn: tokio::sync::Mutex::new(HashMap::new()),
        })
    }

    /// Creates a minimal tracker for unit and integration tests (no Redis, all providers Healthy).
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn new_for_test(provider_names: &[&str]) -> Arc<Self> {
        let names: Vec<String> = provider_names.iter().map(|s| (*s).to_string()).collect();
        Self::new(&names, None, 60, 0.1)
    }

    /// Updates `cooldown_secs` and `ewma_alpha` in-place on SIGHUP config reload.
    ///
    /// Called by `build_from_config` when `existing_tracker` is `Some`. Ensures that
    /// operator changes to `routing.cooldown_secs` or `routing.latency_ewma_alpha` in
    /// YAML take effect without a full restart.
    pub fn update_routing_params(&self, cooldown_secs: u64, ewma_alpha: f64) {
        self.cooldown_secs.store(cooldown_secs, Ordering::Relaxed);
        self.ewma_alpha
            .store(ewma_alpha.to_bits(), Ordering::Relaxed);
    }

    // ---------------------------------------------------------------------------
    // In-flight counter (lock-free snapshot only)
    // ---------------------------------------------------------------------------

    /// Returns the current in-flight count for a provider (snapshot; may be stale by 1 req).
    fn read_in_flight(&self, name: &str) -> usize {
        self.inflight
            .read()
            .expect("inflight lock not poisoned")
            .get(name)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    // ---------------------------------------------------------------------------
    // Candidate snapshot
    // ---------------------------------------------------------------------------

    /// Builds a per-request candidate slice for use by routing strategies.
    ///
    /// **Lock ordering**: sync pricing lock acquired and released (no await), THEN
    /// async state/Redis operations. Never holds `std::sync::RwLock` across `.await`.
    ///
    /// **Cost map note**: `input_cost_per_million` is keyed by model name, not provider.
    /// All providers for the same model receive the same cost value. `LowestCost` therefore
    /// only differentiates when providers serve different model names (common for
    /// OpenAI/Anthropic/Gemini). Multi-provider same-model routing (e.g. two OpenAI-compatible
    /// backends) will see equal costs and fall back to `WeightedRandom`. custom routing
    /// may address per-provider pricing overrides.
    pub async fn candidates(
        &self,
        providers: &[Arc<dyn ProviderAdapter>],
        weights: &HashMap<String, f64>,
        model: &str,
        pricing_db: &Arc<std::sync::RwLock<PricingDb>>,
    ) -> Vec<ProviderCandidate> {
        // Step 1 — sync: acquire pricing lock, collect cost map, release immediately.
        // input_cost_per_million is keyed by model name (not provider), so the value is
        // identical for every provider in the loop. Compute it once before the iterator.
        let cost_map: HashMap<String, NanoUsd> = {
            let db_guard = pricing_db.read().expect("pricing lock poisoned");
            let inner = db_guard.read();
            let cost = inner.input_cost_per_million(model);
            providers
                .iter()
                .map(|p| (p.metadata().name.clone(), cost))
                .collect()
        }; // pricing lock dropped here — safe to .await below

        // Step 2+3 — per-provider: read latency + cooldown state (read lock released before
        // each resolve_cooldown call to avoid holding a read lock across potential write locks).
        let mut candidates = Vec::new();
        for provider in providers {
            let meta = provider.metadata();
            if !model_matches_provider(model, &meta.supported_models) {
                continue;
            }
            let name = meta.name.clone();

            // Snapshot latency and cooldown state under a brief read lock, then release.
            let (latency_ewma_ms, cooldown_snapshot) = {
                let state = self.state.read().await;
                let entry = state.get(&name);
                let latency = entry.map(|e| e.latency_ewma_ms).unwrap_or(0.0);
                let cs = entry
                    .map(|e| e.cooldown_state.clone())
                    .unwrap_or(CooldownState::Closed);
                (latency, cs)
            }; // read lock released

            // Determine cooldown state and whether a probe slot is available (HALF-OPEN).
            let (is_cooling_down, cooldown_remaining_secs) =
                self.resolve_cooldown(&name, cooldown_snapshot).await;

            // FallbackOnly providers default to weight 0.0 unless explicitly overridden in
            // routing.weights. This prevents them from appearing in normal routing while still
            // allowing them as fallback targets​.
            let weight = if meta.kind == ProviderKind::FallbackOnly {
                weights.get(&name).copied().unwrap_or(0.0)
            } else {
                weights.get(&name).copied().unwrap_or(1.0)
            };

            let in_flight = self.read_in_flight(&name);
            let cost = cost_map.get(&name).copied().unwrap_or(NanoUsd::MAX);

            candidates.push(ProviderCandidate {
                name,
                adapter: Arc::clone(provider),
                weight,
                in_flight,
                latency_ewma_ms,
                is_cooling_down,
                cooldown_remaining_secs,
                cost_per_million_tokens: cost,
            });
        }
        candidates
    }

    /// Resolves the cooldown state for a provider and returns `(is_cooling_down, remaining_secs)`.
    ///
    /// Handles the `Closed → Open → HalfOpen → Closed` circuit-breaker state machine .
    ///
    /// When `Open` and the cooldown has elapsed, the state is **mutated** to `HalfOpen` under
    /// a write lock and the first caller claims the probe slot via `compare_exchange`.
    /// Subsequent callers see `is_cooling_down = true` until the probe resolves.
    ///
    /// The HALF-OPEN probe claim is local-node only . In multi-node deployments, up to N
    /// nodes may probe simultaneously. A WARN is emitted when Redis is present to inform operators.
    ///
    /// **Takes a cloned `CooldownState`** (not a reference) to avoid holding the read lock
    /// across the potential write lock in the Open→HalfOpen transition path.
    async fn resolve_cooldown(&self, name: &str, state_snapshot: CooldownState) -> (bool, u64) {
        let cooldown_secs = self.cooldown_secs.load(Ordering::Relaxed);
        let now = Instant::now();

        match state_snapshot {
            CooldownState::Closed => {
                // No local cooldown — check Redis for cross-node cooldown signals.
                if self.redis_has_cooldown(name).await {
                    return (true, cooldown_secs);
                }
                (false, 0)
            }
            CooldownState::Open { cooldown_until } => {
                if now < cooldown_until {
                    let remaining = cooldown_until.duration_since(now).as_secs();
                    return (true, remaining);
                }
                // Cooldown elapsed — transition to HalfOpen under write lock.
                let probe_arc = {
                    let mut state_guard = self.state.write().await;
                    match state_guard.get_mut(name) {
                        Some(e) => match &e.cooldown_state {
                            CooldownState::Open { .. } => {
                                // We are the first to transition.
                                let probe = Arc::new(AtomicBool::new(false));
                                e.cooldown_state = CooldownState::HalfOpen {
                                    probe_taken: Arc::clone(&probe),
                                };
                                probe
                            }
                            CooldownState::HalfOpen { probe_taken } => {
                                // Another task already transitioned.
                                Arc::clone(probe_taken)
                            }
                            CooldownState::Closed => {
                                // Recovered between snapshot and write lock — treat as open.
                                return (false, 0);
                            }
                        },
                        None => return (false, 0),
                    }
                }; // write lock released

                // Emit multi-node WARN after releasing write lock (async op).
                if self.redis.is_some() {
                    self.maybe_warn_half_open(name).await;
                }

                // Claim the probe slot.
                if probe_arc
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    (false, 0) // This caller is the probe
                } else {
                    (true, 0) // Probe already claimed; exclude this provider
                }
            }
            CooldownState::HalfOpen { probe_taken } => {
                // Cloned from snapshot — shares the same AtomicBool as the live state.
                if probe_taken
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    (false, 0) // Probe slot claimed
                } else {
                    (true, 0) // Probe already taken; still cooling
                }
            }
        }
    }

    /// Checks Redis for a persistent cooldown key. Returns `false` on errors (fail-open).
    async fn redis_has_cooldown(&self, name: &str) -> bool {
        if let Some(redis) = &self.redis {
            let key = format!("oxigate:provider:{name}:cooldown");
            match redis.read().await.get().await {
                Ok(mut conn) => {
                    use redis::AsyncCommands;
                    let result: redis::RedisResult<Option<String>> = conn.get(&key).await;
                    match result {
                        Ok(Some(_)) => return true,
                        Ok(None) => {}
                        Err(e) => self.maybe_warn_redis(&e.to_string()).await,
                    }
                }
                Err(e) => self.maybe_warn_redis(&e.to_string()).await,
            }
        }
        false
    }

    // ---------------------------------------------------------------------------
    // Feedback from dispatch (called by ProviderRouter)
    // ---------------------------------------------------------------------------

    /// Records a 429/5xx response: sets local cooldown state to `Open` and writes a Redis key.
    ///
    /// When called in `HalfOpen` state (probe failed): resets to `Open` with a fresh cooldown.
    /// **Must be called BEFORE returning the error to the caller** (C4 from review).
    pub async fn on_rate_limit(&self, name: &str) {
        let cooldown_secs = self.cooldown_secs.load(Ordering::Relaxed);
        let until = Instant::now() + Duration::from_secs(cooldown_secs);
        {
            let mut state = self.state.write().await;
            if let Some(entry) = state.get_mut(name) {
                entry.cooldown_state = CooldownState::Open {
                    cooldown_until: until,
                };
            }
        }
        tracing::info!(provider = %name, cooldown_secs, "provider tripped OPEN");
        if let Some(redis) = &self.redis {
            let key = format!("oxigate:provider:{name}:cooldown");
            match redis.read().await.get().await {
                Ok(mut conn) => {
                    use redis::AsyncCommands;
                    let result: redis::RedisResult<()> =
                        conn.set_ex(&key, "1", cooldown_secs).await;
                    if let Err(e) = result {
                        self.maybe_warn_redis(&e.to_string()).await;
                    }
                }
                Err(e) => {
                    self.maybe_warn_redis(&e.to_string()).await;
                }
            }
        }
    }

    /// Records a successful response: updates EWMA latency and clears any HALF-OPEN state.
    ///
    /// When called in `HalfOpen` state (probe succeeded): transitions back to `Closed`.
    /// Cold-start: first sample sets EWMA to the full sample value (no dampening).
    pub async fn on_response(&self, name: &str, latency_ms: f64) {
        let ewma_alpha = f64::from_bits(self.ewma_alpha.load(Ordering::Relaxed));
        let mut state = self.state.write().await;
        if let Some(entry) = state.get_mut(name) {
            // Probe succeeded — transition from HALF-OPEN → CLOSED.
            if matches!(entry.cooldown_state, CooldownState::HalfOpen { .. }) {
                tracing::info!(provider = %name, "HALF-OPEN probe succeeded → CLOSED");
                entry.cooldown_state = CooldownState::Closed;
            }
            if entry.latency_ewma_ms == 0.0 {
                entry.latency_ewma_ms = latency_ms;
            } else {
                entry.latency_ewma_ms =
                    ewma_alpha * latency_ms + (1.0 - ewma_alpha) * entry.latency_ewma_ms;
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Health queries (called by api/health.rs and api/models.rs)
    // ---------------------------------------------------------------------------

    /// Returns a snapshot of (provider_name, health_status) pairs.
    pub async fn provider_statuses(&self) -> Vec<(String, HealthStatus)> {
        self.state
            .read()
            .await
            .iter()
            .map(|(name, entry)| (name.clone(), entry.health_status.clone()))
            .collect()
    }

    // ---------------------------------------------------------------------------
    // Lifecycle management
    // ---------------------------------------------------------------------------

    /// Updates the health status for a provider. Called by `lifecycle::refresh_health_tracker`.
    pub async fn update_health(&self, name: &str, status: HealthStatus) {
        let mut state = self.state.write().await;
        if let Some(entry) = state.get_mut(name) {
            entry.health_status = status;
        } else {
            // Provider not yet in the map — add it (can happen on SIGHUP with new provider).
            state.insert(
                name.to_string(),
                ProviderHealthEntry {
                    health_status: status,
                    cooldown_state: CooldownState::Closed,
                    latency_ewma_ms: 0.0,
                },
            );
        }
    }

    /// Synchronises the tracker with a new provider set (called on SIGHUP).
    ///
    /// - New providers are added (Healthy, no cooldown, no EWMA).
    /// - Removed providers are cleaned up.
    /// - Surviving providers retain their cooldown and EWMA state.
    pub async fn sync_providers(&self, new_names: &[String]) {
        let new_set: std::collections::HashSet<&String> = new_names.iter().collect();

        // Update state map
        {
            let mut state = self.state.write().await;
            // Add missing providers
            for name in new_names {
                state.entry(name.clone()).or_insert(ProviderHealthEntry {
                    health_status: HealthStatus::Healthy,
                    cooldown_state: CooldownState::Closed,
                    latency_ewma_ms: 0.0,
                });
            }
            // Remove stale providers
            state.retain(|k, _| new_set.contains(k));
        }

        // Update inflight map.
        //
        // Note: `InFlightGuard` clones `Arc<AtomicUsize>` at construction time, so guards
        // already in flight hold a direct Arc reference and are unaffected by this write lock.
        // New guards created after this block will use the updated map.
        {
            let mut inflight = self.inflight.write().expect("inflight lock not poisoned");
            for name in new_names {
                inflight
                    .entry(name.clone())
                    .or_insert_with(|| Arc::new(AtomicUsize::new(0)));
            }
            inflight.retain(|k, _| new_set.contains(k));
        }
    }

    // ---------------------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------------------

    /// Emits a WARN log for Redis errors at most once per 60 seconds.
    async fn maybe_warn_redis(&self, error: &str) {
        const WARN_THROTTLE_SECS: u64 = 60;
        let mut last = self.last_redis_warn.lock().await;
        let now = Instant::now();
        let should_warn = last
            .map(|t| now.duration_since(t).as_secs() >= WARN_THROTTLE_SECS)
            .unwrap_or(true);
        if should_warn {
            tracing::warn!(
                error,
                "health tracker: Redis unavailable; falling back to in-memory cooldown state"
            );
            *last = Some(now);
        }
    }

    /// Emits a WARN log when a HALF-OPEN probe is claimed, at most once per provider per 60 s.
    ///
    /// In multi-node deployments, each node independently manages circuit-breaker state
    /// (Redis coordinates cooldown presence but not the HALF-OPEN probe slot). This means
    /// up to N nodes may simultaneously probe a recovering provider. The warning informs
    /// operators so they can size probe traffic expectations accordingly.
    async fn maybe_warn_half_open(&self, name: &str) {
        const WARN_THROTTLE_SECS: u64 = 60;
        let mut last = self.last_half_open_warn.lock().await;
        let now = Instant::now();
        let should_warn = last
            .get(name)
            .map(|t| now.duration_since(*t).as_secs() >= WARN_THROTTLE_SECS)
            .unwrap_or(true);
        if should_warn {
            tracing::debug!(
                provider = %name,
                "provider {} entered HALF-OPEN: probe slot claimed on this node", name
            );
            last.insert(name.to_string(), now);
        }
    }
}

// ---------------------------------------------------------------------------
// InFlightGuard
// ---------------------------------------------------------------------------

/// RAII guard that decrements the in-flight counter on drop.
///
/// Constructed by `ProviderRouter::chat_completion()` before dispatching.
/// The counter is decremented even if the future is dropped (client disconnect, panic).
///
/// **Design**: clones the `Arc<AtomicUsize>` from the tracker at construction time.
/// `Drop` operates directly on the stored Arc — no lock is acquired during drop.
/// This is safe even when `sync_providers()` holds the inflight write lock during SIGHUP:
/// guards in flight hold their own Arc clone and can always decrement.
pub struct InFlightGuard {
    counter: Arc<AtomicUsize>,
}

impl InFlightGuard {
    /// Creates a guard and immediately increments the in-flight counter.
    ///
    /// Uses a blocking `std::sync::RwLock` read, which is uncontended on the hot path.
    /// The write lock is only held during `sync_providers` (SIGHUP, infrequent), and
    /// has no `.await` inside the critical section, so it releases quickly.
    /// `Drop` operates lock-free via the cloned `Arc<AtomicUsize>`.
    pub fn new(tracker: &ProviderHealthTracker, name: &str) -> Self {
        let counter = tracker
            .inflight
            .read()
            .expect("inflight lock not poisoned")
            .get(name)
            .cloned()
            .unwrap_or_else(|| Arc::new(AtomicUsize::new(0)));
        counter.fetch_add(1, Ordering::Relaxed);
        Self { counter }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        // Atomic saturating decrement — no lock needed since we hold the Arc directly.
        // fetch_update is used instead of a load+compare+fetch_sub sequence to prevent
        // the TOCTOU race where two concurrent decrements at count=1 both pass a > 0 check
        // and both subtract, wrapping to usize::MAX.
        let _ = self
            .counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            });
    }
}

// ---------------------------------------------------------------------------
// Model-matching helper
// ---------------------------------------------------------------------------

/// Returns true if `model` matches any entry in `supported_models`.
///
/// Four rules (applied to each entry in order — all rules checked for all entries):
/// - `"*"` → wildcard, matches any model
/// - exact match: `entry == model`
/// - prefix-dash: `entry.ends_with('-')` and `model.starts_with(entry)` (e.g. `"gpt-4-"`)
/// - prefix-glob: `entry.ends_with('*')` and `model.starts_with(entry_without_star)`
pub fn model_matches_provider(model: &str, supported: &[String]) -> bool {
    supported.iter().any(|entry| model_matches(model, entry))
}

fn model_matches(model: &str, entry: &str) -> bool {
    entry == "*"
        || model == entry
        || (entry.ends_with('-') && model.starts_with(entry))
        || entry
            .strip_suffix('*')
            .is_some_and(|prefix| model.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_matches_wildcard() {
        assert!(model_matches("gpt-4o", "*"));
        assert!(model_matches("anything", "*"));
    }

    #[test]
    fn test_model_matches_exact() {
        assert!(model_matches("gpt-4o", "gpt-4o"));
        assert!(!model_matches("gpt-4o", "gpt-4"));
    }

    #[test]
    fn test_model_matches_prefix_dash() {
        assert!(model_matches("gpt-4o-mini", "gpt-4o-"));
        assert!(model_matches("gpt-4-turbo", "gpt-4-"));
        assert!(!model_matches("gpt-5", "gpt-4-"));
    }

    #[test]
    fn test_model_matches_glob() {
        assert!(model_matches("claude-3-5-sonnet", "claude-*"));
        assert!(!model_matches("gpt-4", "claude-*"));
    }

    #[tokio::test]
    async fn test_inflight_guard_decrements_on_drop() {
        let tracker = ProviderHealthTracker::new_for_test(&["openai"]);
        {
            let _g = InFlightGuard::new(&tracker, "openai");
            assert_eq!(tracker.read_in_flight("openai"), 1);
        }
        assert_eq!(tracker.read_in_flight("openai"), 0);
    }

    #[tokio::test]
    async fn test_inflight_guard_decrements_on_panic() {
        let tracker = ProviderHealthTracker::new_for_test(&["openai"]);
        let tracker_ref = Arc::clone(&tracker);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = InFlightGuard::new(&tracker_ref, "openai");
            panic!("intentional panic");
        }));
        assert!(result.is_err(), "panic should have propagated");
        assert_eq!(
            tracker.read_in_flight("openai"),
            0,
            "counter must be decremented on panic"
        );
    }

    // -------------------------------------------------------------------------
    // HALF-OPEN circuit breaker tests
    // -------------------------------------------------------------------------

    /// Provider enters Open after on_rate_limit(); transitions to HalfOpen once the
    /// cooldown expires. Verified by using cooldown_secs=0 + brief sleep.
    #[tokio::test]
    async fn test_half_open_after_cooldown_expires() {
        let tracker = ProviderHealthTracker::new_for_test(&["openai"]);
        // Set cooldown to 0 seconds so it expires immediately after on_rate_limit().
        tracker.update_routing_params(0, 0.1);
        tracker.on_rate_limit("openai").await;

        // Allow Instant::now() to advance past cooldown_until (which = now + 0s).
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;

        // candidates() triggers the Open → HalfOpen transition for the first caller.
        let pricing = Arc::new(std::sync::RwLock::new(
            crate::domain::pricing::PricingDb::load(
                crate::domain::pricing::BUNDLED_PRICING_JSON,
                &crate::config::PricingConfig::default(),
            )
            .unwrap(),
        ));
        let adapter: Arc<dyn crate::domain::ports::ProviderAdapter> = Arc::new({
            struct Dummy;
            #[async_trait::async_trait]
            impl crate::domain::ports::ProviderAdapter for Dummy {
                async fn chat_completion(
                    &self,
                    _: &crate::domain::chat::ChatRequest,
                ) -> Result<crate::domain::chat::ChatResponse, crate::domain::ports::ProviderError>
                {
                    unimplemented!()
                }
                fn metadata(&self) -> &crate::domain::ports::ProviderMetadata {
                    static M: std::sync::LazyLock<crate::domain::ports::ProviderMetadata> =
                        std::sync::LazyLock::new(|| crate::domain::ports::ProviderMetadata {
                            name: "openai".into(),
                            supported_models: vec!["gpt-4o".into()],
                            supports_streaming: false,
                            supports_tools: false,
                            supports_vision: false,
                            supports_embeddings: false,
                            supports_thinking: false,
                            kind: Default::default(),
                            ..Default::default()
                        });
                    &M
                }
                async fn health_check(&self) -> HealthStatus {
                    HealthStatus::Healthy
                }
            }
            Dummy
        });

        let default_weights = std::collections::HashMap::new();
        let candidates = tracker
            .candidates(&[adapter], &default_weights, "gpt-4o", &pricing)
            .await;
        assert!(
            !candidates.is_empty(),
            "probe slot must be granted when cooldown expires"
        );
        assert!(
            !candidates[0].is_cooling_down,
            "probe-granted candidate must not be marked as cooling"
        );
    }

    /// After probe succeeds (on_response called), provider returns to Closed.
    #[tokio::test]
    async fn test_half_open_probe_success_transitions_to_closed() {
        let tracker = ProviderHealthTracker::new_for_test(&["openai"]);
        tracker.update_routing_params(0, 0.1);
        tracker.on_rate_limit("openai").await;
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;

        // candidates() is required to trigger the Open → HalfOpen transition.
        let pricing = Arc::new(std::sync::RwLock::new(
            crate::domain::pricing::PricingDb::load(
                crate::domain::pricing::BUNDLED_PRICING_JSON,
                &crate::config::PricingConfig::default(),
            )
            .unwrap(),
        ));
        let adapter: Arc<dyn crate::domain::ports::ProviderAdapter> = Arc::new({
            struct Dummy2;
            #[async_trait::async_trait]
            impl crate::domain::ports::ProviderAdapter for Dummy2 {
                async fn chat_completion(
                    &self,
                    _: &crate::domain::chat::ChatRequest,
                ) -> Result<crate::domain::chat::ChatResponse, crate::domain::ports::ProviderError>
                {
                    unimplemented!()
                }
                fn metadata(&self) -> &crate::domain::ports::ProviderMetadata {
                    static M: std::sync::LazyLock<crate::domain::ports::ProviderMetadata> =
                        std::sync::LazyLock::new(|| crate::domain::ports::ProviderMetadata {
                            name: "openai".into(),
                            supported_models: vec!["gpt-4o".into()],
                            supports_streaming: false,
                            supports_tools: false,
                            supports_vision: false,
                            supports_embeddings: false,
                            supports_thinking: false,
                            kind: Default::default(),
                            ..Default::default()
                        });
                    &M
                }
                async fn health_check(&self) -> HealthStatus {
                    HealthStatus::Healthy
                }
            }
            Dummy2
        });
        let default_weights = std::collections::HashMap::new();
        // This call transitions Open → HalfOpen (probe slot granted).
        let _ = tracker
            .candidates(&[adapter], &default_weights, "gpt-4o", &pricing)
            .await;

        // Now on_response simulates a successful probe — must transition HalfOpen → Closed.
        tracker.on_response("openai", 100.0).await;

        let state = tracker.state.read().await;
        let entry = state.get("openai").expect("openai must exist");
        assert!(
            matches!(entry.cooldown_state, CooldownState::Closed),
            "probe success must transition back to Closed"
        );
    }

    /// After probe fails (on_rate_limit called in HalfOpen), cooldown resets to Open.
    #[tokio::test]
    async fn test_half_open_probe_failure_resets_cooldown() {
        let tracker = ProviderHealthTracker::new_for_test(&["openai"]);
        // Set cooldown to 0 so it expires immediately.
        tracker.update_routing_params(0, 0.1);
        tracker.on_rate_limit("openai").await;
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;

        // candidates() transitions Open → HalfOpen (probe slot granted).
        let pricing = Arc::new(std::sync::RwLock::new(
            crate::domain::pricing::PricingDb::load(
                crate::domain::pricing::BUNDLED_PRICING_JSON,
                &crate::config::PricingConfig::default(),
            )
            .unwrap(),
        ));
        let adapter: Arc<dyn crate::domain::ports::ProviderAdapter> = Arc::new({
            struct Dummy3;
            #[async_trait::async_trait]
            impl crate::domain::ports::ProviderAdapter for Dummy3 {
                async fn chat_completion(
                    &self,
                    _: &crate::domain::chat::ChatRequest,
                ) -> Result<crate::domain::chat::ChatResponse, crate::domain::ports::ProviderError>
                {
                    unimplemented!()
                }
                fn metadata(&self) -> &crate::domain::ports::ProviderMetadata {
                    static M: std::sync::LazyLock<crate::domain::ports::ProviderMetadata> =
                        std::sync::LazyLock::new(|| crate::domain::ports::ProviderMetadata {
                            name: "openai".into(),
                            supported_models: vec!["gpt-4o".into()],
                            supports_streaming: false,
                            supports_tools: false,
                            supports_vision: false,
                            supports_embeddings: false,
                            supports_thinking: false,
                            kind: Default::default(),
                            ..Default::default()
                        });
                    &M
                }
                async fn health_check(&self) -> HealthStatus {
                    HealthStatus::Healthy
                }
            }
            Dummy3
        });
        let default_weights = std::collections::HashMap::new();
        let _ = tracker
            .candidates(&[adapter], &default_weights, "gpt-4o", &pricing)
            .await;

        // Reset cooldown_secs to 60 — next re-open will use this value.
        tracker.update_routing_params(60, 0.1);

        // Simulate probe failure — should re-open with fresh 60s cooldown.
        tracker.on_rate_limit("openai").await;

        let state = tracker.state.read().await;
        let entry = state.get("openai").expect("openai must exist");
        assert!(
            matches!(entry.cooldown_state, CooldownState::Open { .. }),
            "probe failure must return provider to Open state"
        );
    }

    #[tokio::test]
    async fn test_update_routing_params_takes_effect() {
        let tracker = ProviderHealthTracker::new_for_test(&["openai"]);
        tracker.update_routing_params(120, 0.5);
        assert_eq!(
            tracker.cooldown_secs.load(Ordering::Relaxed),
            120,
            "cooldown_secs must update"
        );
        let alpha = f64::from_bits(tracker.ewma_alpha.load(Ordering::Relaxed));
        assert!((alpha - 0.5).abs() < 1e-10, "ewma_alpha must update");
    }
}
