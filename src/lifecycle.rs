// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Resource lifecycle management: initialization and hot-reload.
//!
//! Encapsulates DB and Redis pool creation, health checks, and SIGHUP-driven
//! Class B reload logic. Keeps `main.rs` focused on high-level orchestration.

use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;
use tokio::sync::RwLock;

use crate::config::{
    BudgetConfig, GatewayConfig, HotReloadClass, SecurityConfig, apply_config_reload,
    classify_reload,
};
use crate::db::{DbPool, init_db};
use crate::domain::ports::ProviderAdapterExt;
use crate::middleware::budget::BudgetRuntimeConfig;
use crate::middleware::global_safety::GlobalSafetyRuntimeConfig;
use crate::providers::{self, ProviderHealthTracker};
use crate::redis_pool::{RedisPool, create_pool, health_check};

/// Errors during resource initialization or reload.
#[derive(Debug, Error)]
pub enum LifecycleError {
    #[error("DB connection failed: {0}")]
    Db(#[from] crate::db::DbError),
    #[error("Redis pool creation failed: {0}")]
    Redis(#[from] crate::redis_pool::RedisError),
    #[error("{0}")]
    Other(String),
}

/// Shareable application resources (DB and Redis pools).
///
/// Wrapped in `Arc<RwLock<...>>` so pools can be rebuilt on SIGHUP (Class B reload)
/// without restarting the process.
#[derive(Clone)]
pub struct AppResources {
    /// PostgreSQL connection pool.
    pub pool: Arc<RwLock<DbPool>>,
    /// Redis connection pool.
    pub redis_pool: Arc<RwLock<RedisPool>>,
}

/// Shared state updated by SIGHUP reload.
#[derive(Clone)]
pub struct ReloadHandles {
    /// Handle for hot-reloading the active log filter on SIGHUP (Class A).
    pub log_level_handle: crate::observability::LogLevelHandle,
    pub pricing_db_holder: Arc<std::sync::RwLock<crate::domain::pricing::PricingDb>>,
    pub provider_holder: Arc<RwLock<Arc<dyn ProviderAdapterExt>>>,
    pub auth_holder: Arc<RwLock<crate::config::AuthConfig>>,
    pub health_tracker: Arc<ProviderHealthTracker>,
    pub global_safety_holder: Arc<RwLock<GlobalSafetyRuntimeConfig>>,
    /// User budget section: spend keying + seeding; hot-reloaded on Class A/B.
    pub budget_config_holder: Arc<RwLock<BudgetConfig>>,
    /// Budget runtime config for BudgetLayer + HardCapLayer hot-reload.
    pub budget_holder: Arc<RwLock<BudgetRuntimeConfig>>,
    /// Security config for opt-in visibility features. Class A hot-reload.
    pub security_holder: Arc<RwLock<SecurityConfig>>,
}

/// Timeout for Redis health check during startup (milliseconds).
const REDIS_HEALTH_TIMEOUT_MS: u64 = 500;

/// Returns the start of the current calendar month at UTC midnight.
///
/// Used as the default Postgres aggregation lower bound when `budget_reset_at` is unset.
fn month_start_utc() -> chrono::DateTime<chrono::Utc> {
    use chrono::{Datelike, TimeZone, Utc};
    let now = Utc::now();
    Utc.with_ymd_and_hms(now.year(), now.month(), 1, 0, 0, 0)
        .single()
        .unwrap_or(now)
}

/// Initialize database and Redis pools, run migrations and health checks.
///
/// Returns shareable resources ready for wiring into the router and SIGHUP reload.
pub async fn init_resources(config: &GatewayConfig) -> Result<AppResources, LifecycleError> {
    let pool = init_db(config).await.map_err(LifecycleError::Db)?;
    let pool: Arc<RwLock<DbPool>> = Arc::new(RwLock::new(pool));
    tracing::info!("database pool ready, migrations applied");

    let redis_pool = create_pool(&config.redis).map_err(LifecycleError::Redis)?;
    let redis_pool_inner: Arc<RwLock<RedisPool>> = Arc::new(RwLock::new(redis_pool));

    // Clone pool out of lock — do not hold RwLockReadGuard across I/O.
    // At startup no writers exist; if this pattern were copied to the request hot-path,
    // it would cause writer starvation during SIGHUP pool rebuilds.
    let redis_pool_for_health = redis_pool_inner.read().await.clone();
    tokio::time::timeout(
        std::time::Duration::from_millis(REDIS_HEALTH_TIMEOUT_MS),
        health_check(&redis_pool_for_health),
    )
    .await
    .map_err(|_| LifecycleError::Other("Redis health check timed out (>500 ms)".into()))?
    .map_err(LifecycleError::Redis)?;
    tracing::info!("redis pool ready");

    // Seed Redis spend counters from Postgres​.
    // Runs after health check so Redis is confirmed reachable. Failures are
    // non-fatal — logged at WARN; gateway startup continues regardless.
    let aggregate_since = config
        .budget
        .budget_reset_at
        .unwrap_or_else(month_start_utc);
    let duration = config.budget.resolved_duration();
    let tz = config.budget.resolved_timezone();
    let budget_reset_at_is_explicit = config.budget.budget_reset_at.is_some();
    let db_for_seed = pool.read().await.clone();
    let redis_for_seed = redis_pool_inner.read().await.clone();
    crate::db::spend_writer::seed_redis_from_db(
        &db_for_seed,
        &redis_for_seed,
        aggregate_since,
        duration,
        tz,
        budget_reset_at_is_explicit,
    )
    .await;

    Ok(AppResources {
        pool,
        redis_pool: redis_pool_inner,
    })
}

/// Spawn a background task that listens for SIGHUP and reloads config.
///
/// On Class B changes (DB/Redis URL, provider endpoints), rebuilds pools and provider.
/// On Class A (pricing, auth.key, providers.gemini.api_key), reloads pricing,
/// auth config, and rebuilds provider.
/// Class C (port/host) is rejected with a warning (restart required).
///
/// Unix-only; no-op on non-Unix platforms.
#[cfg(unix)]
pub fn spawn_sighup_reload(
    config_path: PathBuf,
    active_config: Arc<RwLock<GatewayConfig>>,
    resources: AppResources,
    handles: ReloadHandles,
) {
    fn apply_log_level_reload(
        handles: &ReloadHandles,
        old_cfg: &GatewayConfig,
        new_cfg: &GatewayConfig,
    ) {
        if old_cfg.log_level == new_cfg.log_level {
            return;
        }
        match tracing_subscriber::EnvFilter::try_new(&new_cfg.log_level) {
            Ok(new_filter) => {
                if let Err(e) = handles.log_level_handle.reload(new_filter) {
                    tracing::warn!(error = %e, "log level reload failed");
                } else {
                    tracing::info!(level = %new_cfg.log_level, "log level updated");
                }
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    new_level = %new_cfg.log_level,
                    "invalid log level in reloaded config — keeping previous"
                );
            }
        }
    }

    tokio::spawn(async move {
        let mut sighup = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())
            .expect("failed to install SIGHUP handler");
        loop {
            sighup.recv().await;
            tracing::info!("SIGHUP received — attempting config reload");
            match crate::config::load_and_validate_config(&config_path) {
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "config reload failed — keeping previous config"
                    );
                }
                Ok(new_cfg) => {
                    let old_cfg = active_config.read().await.clone();
                    let class = classify_reload(&old_cfg, &new_cfg);
                    match class {
                        HotReloadClass::ClassC => {
                            tracing::warn!(
                                "config reload rejected: server.port/host change requires restart"
                            );
                        }
                        HotReloadClass::ClassB => {
                            tracing::info!("Class B reload: rebuilding DB and Redis pools");
                            // NOTE: Migrations are intentionally not re-run on Class B reload.
                            // Startup uses init_db() which runs migrations; reload uses connect()
                            // only. Re-running sqlx::migrate!() on every SIGHUP would be
                            // redundant and could interfere with in-flight connections.
                            match crate::db::connect(&new_cfg.database).await {
                                Err(e) => {
                                    tracing::error!(
                                        error = %e,
                                        "DB pool rebuild failed — keeping previous config"
                                    );
                                }
                                Ok(new_db_pool) => match create_pool(&new_cfg.redis) {
                                    Err(e) => {
                                        tracing::error!(
                                            error = %e,
                                            "Redis pool rebuild failed — keeping previous config"
                                        );
                                    }
                                    Ok(new_redis_pool) => {
                                        *resources.pool.write().await = new_db_pool;
                                        *resources.redis_pool.write().await = new_redis_pool;
                                        match providers::build_from_config(
                                            &new_cfg,
                                            Arc::clone(&handles.pricing_db_holder),
                                            Some(Arc::clone(&resources.redis_pool)),
                                            Some(Arc::clone(&handles.health_tracker)),
                                        )
                                        .await
                                        {
                                            Ok((new_provider, _)) => {
                                                refresh_health_tracker(
                                                    &new_provider,
                                                    &handles.health_tracker,
                                                )
                                                .await;
                                                *handles.provider_holder.write().await =
                                                    new_provider;
                                                tracing::info!("provider rebuilt");
                                            }
                                            Err(e) => {
                                                tracing::error!(
                                                    error = %e,
                                                    "provider rebuild failed — keeping previous"
                                                );
                                            }
                                        }
                                        // Class B supersedes Class A: also apply auth and global
                                        // safety so that simultaneous changes aren't silently
                                        // dropped (classify_reload returns the highest class).
                                        apply_log_level_reload(&handles, &old_cfg, &new_cfg);
                                        *handles.auth_holder.write().await = new_cfg.auth.clone();
                                        *handles.global_safety_holder.write().await =
                                            GlobalSafetyRuntimeConfig::from_budget_config(
                                                &new_cfg.budget,
                                            );
                                        *handles.budget_config_holder.write().await =
                                            new_cfg.budget.clone();
                                        *handles.security_holder.write().await =
                                            new_cfg.security.clone();
                                        *handles.budget_holder.write().await =
                                            BudgetRuntimeConfig::from_budget_config(
                                                new_cfg.budget.clone(),
                                            );
                                        apply_config_reload(&active_config, new_cfg).await;
                                        tracing::info!("Class B reload complete");
                                    }
                                },
                            }
                        }
                        HotReloadClass::ClassA => {
                            tracing::info!(
                                "Class A reload: applying log level, config, pricing, auth, and provider"
                            );
                            apply_log_level_reload(&handles, &old_cfg, &new_cfg);
                            *handles.auth_holder.write().await = new_cfg.auth.clone();
                            *handles.global_safety_holder.write().await =
                                GlobalSafetyRuntimeConfig::from_budget_config(&new_cfg.budget);
                            *handles.budget_config_holder.write().await = new_cfg.budget.clone();
                            *handles.security_holder.write().await = new_cfg.security.clone();
                            *handles.budget_holder.write().await =
                                BudgetRuntimeConfig::from_budget_config(new_cfg.budget.clone());
                            match crate::domain::pricing::PricingDb::load(
                                crate::domain::pricing::BUNDLED_PRICING_JSON,
                                &new_cfg.pricing,
                            ) {
                                Ok(new_db) => {
                                    let holder = Arc::clone(&handles.pricing_db_holder);
                                    tokio::task::spawn_blocking(move || {
                                        *holder.write().expect("pricing holder lock poisoned") =
                                            new_db;
                                    })
                                    .await
                                    .expect("spawn_blocking");
                                    tracing::info!("pricing DB reloaded");
                                }
                                Err(e) => {
                                    tracing::error!(
                                        error = %e,
                                        "pricing DB reload failed — keeping previous"
                                    );
                                }
                            }
                            match providers::build_from_config(
                                &new_cfg,
                                Arc::clone(&handles.pricing_db_holder),
                                Some(Arc::clone(&resources.redis_pool)),
                                Some(Arc::clone(&handles.health_tracker)),
                            )
                            .await
                            {
                                Ok((new_provider, _)) => {
                                    refresh_health_tracker(&new_provider, &handles.health_tracker)
                                        .await;
                                    *handles.provider_holder.write().await = new_provider;
                                    tracing::info!("provider rebuilt (api_key/default_model)");
                                }
                                Err(e) => {
                                    tracing::error!(
                                        error = %e,
                                        "provider rebuild failed — keeping previous"
                                    );
                                }
                            }
                            apply_config_reload(&active_config, new_cfg).await;
                        }
                    }
                }
            }
        }
    });
}

/// Re-runs health checks for all leaf adapters and updates the tracker in-place.
/// Called after every provider rebuild so the tracker reflects the new adapter set.
async fn refresh_health_tracker(
    provider: &Arc<dyn ProviderAdapterExt>,
    tracker: &Arc<ProviderHealthTracker>,
) {
    for p in providers::leaf_adapters(provider) {
        let status = p.health_check().await;
        tracker.update_health(&p.metadata().name, status).await;
    }
    tracing::info!("health tracker refreshed after provider rebuild");
}
