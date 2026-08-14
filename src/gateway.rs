// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Two-phase gateway builder: [`Gateway::build`] assembles resources, [`Gateway::serve`] binds and runs.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use miette::{Result, miette};
use tokio::sync::RwLock as AsyncRwLock;
use tracing::info;

use crate::config::{AuthConfig, BudgetConfig, GatewayConfig, SecurityConfig};
use crate::domain::ports::ProviderAdapterExt;
use crate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
use crate::lifecycle::{AppResources, ReloadHandles};
use crate::middleware::budget::BudgetRuntimeConfig;
use crate::middleware::global_safety::GlobalSafetyRuntimeConfig;
use crate::providers::ProviderHealthTracker;
use crate::redis_pool::RedisPool;

/// Shared runtime handles exposed to gateway extensions via [`Gateway::resources`].
///
/// Fields are stable public API; additions must be backward-compatible with existing callers.
/// Do not add fields speculatively.
pub struct GatewayResources {
    pub budget_runtime: Arc<AsyncRwLock<BudgetRuntimeConfig>>,
    pub redis_pool: Arc<AsyncRwLock<RedisPool>>,
}

/// Fully assembled gateway. Consume via [`Gateway::serve`]; cannot be served twice.
pub struct Gateway {
    config: GatewayConfig,
    app_resources: AppResources,
    pricing_db_holder: Arc<RwLock<PricingDb>>,
    provider_holder: Arc<AsyncRwLock<Arc<dyn ProviderAdapterExt>>>,
    auth_holder: Arc<AsyncRwLock<AuthConfig>>,
    global_safety_holder: Arc<AsyncRwLock<GlobalSafetyRuntimeConfig>>,
    budget_settings_holder: Arc<AsyncRwLock<BudgetConfig>>,
    budget_holder: Arc<AsyncRwLock<BudgetRuntimeConfig>>,
    security_holder: Arc<AsyncRwLock<SecurityConfig>>,
    health_tracker: Arc<ProviderHealthTracker>,
    prometheus_handle: metrics_exporter_prometheus::PrometheusHandle,
    startup_time: u64,
    resources: GatewayResources,
}

impl Gateway {
    /// Assemble all gateway resources from config. Does not bind any port.
    ///
    /// `config_path` is used in the startup log and retained for SIGHUP hot-reload (Unix);
    /// it is not stored in `Self`.
    pub async fn build(config: GatewayConfig, config_path: PathBuf) -> Result<Self> {
        let log_level_handle = crate::observability::init_tracing(&config.log_level)?;
        let prometheus_handle = crate::observability::init_metrics()?;

        let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &config.pricing)
            .map_err(|e| miette!("pricing DB load failed: {e}"))?;
        let pricing_db_holder = Arc::new(RwLock::new(pricing_db));
        info!("pricing DB loaded (anchor set + overrides)");

        info!(
            version = env!("CARGO_PKG_VERSION"),
            config = %config_path.display(),
            "oxigate startup"
        );

        let app_resources = crate::lifecycle::init_resources(&config)
            .await
            .map_err(|e| miette!("{e}"))?;

        let startup_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let (provider, health_tracker) = crate::providers::build_from_config(
            &config,
            Arc::clone(&pricing_db_holder),
            Some(Arc::clone(&app_resources.redis_pool)),
            None,
        )
        .await
        .map_err(|e| miette!("provider init failed: {e}"))?;

        for p in crate::providers::leaf_adapters(&provider) {
            let status = p.health_check().await;
            health_tracker
                .update_health(&p.metadata().name, status)
                .await;
        }

        let provider_holder = Arc::new(AsyncRwLock::new(provider));
        let auth_holder = Arc::new(AsyncRwLock::new(config.auth.clone()));
        if config.auth.key.is_none() {
            tracing::warn!(
                "auth.key not configured — all /v1/* requests accepted; set for production"
            );
        }
        let global_safety_holder = Arc::new(AsyncRwLock::new(
            GlobalSafetyRuntimeConfig::from_budget_config(&config.budget),
        ));
        let budget_settings_holder = Arc::new(AsyncRwLock::new(config.budget.clone()));
        let security_holder = Arc::new(AsyncRwLock::new(config.security.clone()));
        let budget_holder = Arc::new(AsyncRwLock::new(BudgetRuntimeConfig::from_budget_config(
            config.budget.clone(),
        )));

        #[cfg(unix)]
        {
            let active_config = Arc::new(AsyncRwLock::new(config.clone()));
            crate::lifecycle::spawn_sighup_reload(
                config_path,
                Arc::clone(&active_config),
                app_resources.clone(),
                ReloadHandles {
                    log_level_handle,
                    pricing_db_holder: Arc::clone(&pricing_db_holder),
                    provider_holder: Arc::clone(&provider_holder),
                    auth_holder: Arc::clone(&auth_holder),
                    health_tracker: Arc::clone(&health_tracker),
                    global_safety_holder: Arc::clone(&global_safety_holder),
                    budget_config_holder: Arc::clone(&budget_settings_holder),
                    budget_holder: Arc::clone(&budget_holder),
                    security_holder: Arc::clone(&security_holder),
                },
            );
        }
        // On non-Unix, SIGHUP hot-reload is unavailable; log_level_handle is not used.
        #[cfg(not(unix))]
        let _ = log_level_handle;

        let resources = GatewayResources {
            budget_runtime: Arc::clone(&budget_holder),
            redis_pool: Arc::clone(&app_resources.redis_pool),
        };

        Ok(Self {
            config,
            app_resources,
            pricing_db_holder,
            provider_holder,
            auth_holder,
            global_safety_holder,
            budget_settings_holder,
            budget_holder,
            security_holder,
            health_tracker,
            prometheus_handle,
            startup_time,
            resources,
        })
    }

    /// Shared runtime handles for gateway extensions. Call before [`Gateway::serve`].
    pub fn resources(&self) -> &GatewayResources {
        &self.resources
    }

    /// Bind, serve, and block until shutdown. Consumes `self`.
    pub async fn serve(self) -> Result<()> {
        let addr = std::net::SocketAddr::from((
            self.config
                .server
                .host
                .parse::<std::net::IpAddr>()
                .map_err(|e| miette!("invalid server.host '{}': {e}", self.config.server.host))?,
            self.config.server.port,
        ));
        let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
            miette!(
                "failed to bind on {}: {e} — is the port already in use?",
                addr
            )
        })?;

        info!(
            port = self.config.server.port,
            version = env!("CARGO_PKG_VERSION"),
            "oxigate listening"
        );

        let drain_secs = self.config.server.drain_timeout_secs;
        let (drain_tx, drain_rx) = tokio::sync::oneshot::channel::<()>();
        let (shutdown_requested_tx, shutdown_requested_rx) = tokio::sync::oneshot::channel::<()>();

        tokio::spawn(async move {
            shutdown_signal().await;
            let _ = drain_tx.send(());
            let _ = shutdown_requested_tx.send(());
        });

        let pool = Arc::clone(&self.app_resources.pool);
        let redis_pool = Arc::clone(&self.app_resources.redis_pool);
        let state = crate::api::AppState {
            pool,
            redis_pool,
            pricing_db: self.pricing_db_holder,
            provider: self.provider_holder,
            auth: self.auth_holder,
            global_safety: self.global_safety_holder,
            budget_settings: self.budget_settings_holder,
            budget: self.budget_holder,
            startup_time: self.startup_time,
            health: self.health_tracker,
            security: self.security_holder,
        };
        let router = crate::api::router_with_metrics(state, self.prometheus_handle);
        let serve = axum::serve(listener, router).with_graceful_shutdown(async move {
            drain_rx.await.ok();
        });
        let mut serve_handle =
            tokio::spawn(async move { serve.await.map_err(|e| miette!("server error: {e}")) });

        tokio::select! {
            Ok(()) = shutdown_requested_rx => {
                tokio::select! {
                    result = (&mut serve_handle) => {
                        result.map_err(|e| miette!("server task failed: {e}"))??;
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_secs(drain_secs)) => {
                        tracing::warn!("drain timeout reached — forcing shutdown");
                        serve_handle.abort();
                        let _ = serve_handle.await;
                    }
                }
            }
            result = (&mut serve_handle) => {
                result.map_err(|e| miette!("server task failed: {e}"))??;
            }
        }

        info!("server shutdown complete");
        Ok(())
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c    => info!("received Ctrl-C"),
        _ = terminate => info!("received SIGTERM"),
    }
    info!("shutdown signal received — draining in-flight requests");
}

#[cfg(all(test, feature = "test-hooks"))]
mod tests {
    use super::*;

    use testcontainers_modules::postgres::Postgres;
    use testcontainers_modules::redis::Redis;
    use testcontainers_modules::testcontainers::runners::AsyncRunner;

    use crate::config::{DatabaseConfig, RedisConfig, SecretString};

    /// Asserts that `resources()` Arc fields alias their internal counterparts after
    /// `Gateway::build`. Guards against a future refactor that creates independent Arcs.
    ///
    /// Requires Docker and a clean process (global tracing/metrics subscribers).
    /// Run explicitly: `cargo test -p oxigate --features test-hooks -- --ignored build_resources`
    #[ignore = "requires Docker and an isolated process (global tracing/metrics subscribers)"]
    #[tokio::test]
    async fn build_resources_aliased_to_budget_holder() {
        let pg = Postgres::default()
            .start()
            .await
            .expect("postgres container must start");
        let redis = Redis::default()
            .start()
            .await
            .expect("redis container must start");
        let pg_port = pg.get_host_port_ipv4(5432).await.expect("pg port");
        let redis_port = redis.get_host_port_ipv4(6379).await.expect("redis port");

        let mut config = GatewayConfig::default();
        config.database = DatabaseConfig {
            url: SecretString::new(format!(
                "postgres://postgres:postgres@localhost:{pg_port}/postgres"
            )),
            max_connections: Some(2),
            pool_acquire_timeout_secs: Some(10),
        };
        config.redis = RedisConfig {
            url: SecretString::new(format!("redis://127.0.0.1:{redis_port}")),
            pool_size: Some(2),
            pool_timeout_secs: Some(2),
        };

        let gw = Gateway::build(config, PathBuf::from("config/test.yaml"))
            .await
            .expect("Gateway::build must succeed");

        assert!(
            Arc::ptr_eq(&gw.resources.budget_runtime, &gw.budget_holder),
            "resources().budget_runtime must alias the internal budget_holder Arc"
        );
        assert!(
            Arc::ptr_eq(&gw.resources.redis_pool, &gw.app_resources.redis_pool),
            "resources().redis_pool must alias the internal app_resources.redis_pool Arc"
        );
    }
}
