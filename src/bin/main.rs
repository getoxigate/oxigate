// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! OxiGate gateway binary — composition root.
//!
//! Wires axum server, config parsing, graceful shutdown, and SIGHUP hot-reload.
//! Resource initialization and reload logic live in `lifecycle`.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use miette::{Result, miette};
use tracing::info;

use oxigate::domain::ports::ProviderAdapterExt;
use oxigate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
use oxigate::providers;

/// Run mode determined by CLI args.
enum RunMode {
    Serve,
    ValidateConfig,
    Migrate,
}

#[tokio::main]
async fn main() -> Result<()> {
    let (mode, config_path) = parse_args()?;

    match mode {
        RunMode::ValidateConfig => return run_validate_config(&config_path),
        RunMode::Migrate => return run_migrate(&config_path).await,
        RunMode::Serve => {}
    }

    let config =
        oxigate::config::load_and_validate_config(&config_path).map_err(|e| miette!("{e}"))?;
    let log_level_handle = oxigate::observability::init_tracing(&config.log_level)?;
    //: install the Prometheus recorder before any metrics are emitted.
    // Fatal if it fails (MVP Critical observability — running without metrics is unacceptable).
    let prometheus_handle = oxigate::observability::init_metrics()?;

    let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &config.pricing)
        .map_err(|e| miette!("pricing DB load failed: {e}"))?;
    let pricing_db_holder: Arc<RwLock<PricingDb>> = Arc::new(std::sync::RwLock::new(pricing_db));
    tracing::info!("pricing DB loaded (anchor set + overrides)");

    info!(
        version = env!("CARGO_PKG_VERSION"),
        config = %config_path.display(),
        "oxigate startup"
    );

    let resources = oxigate::lifecycle::init_resources(&config)
        .await
        .map_err(|e| miette!("{e}"))?;

    let addr = std::net::SocketAddr::from((
        config
            .server
            .host
            .parse::<std::net::IpAddr>()
            .map_err(|e| miette!("invalid server.host '{}': {e}", config.server.host))?,
        config.server.port,
    ));
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        miette!(
            "failed to bind on {}: {e} — is the port already in use?",
            addr
        )
    })?;

    info!(
        port = config.server.port,
        version = env!("CARGO_PKG_VERSION"),
        "oxigate listening"
    );

    let startup_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let (provider, health_tracker) = providers::build_from_config(
        &config,
        Arc::clone(&pricing_db_holder),
        Some(Arc::clone(&resources.redis_pool)),
        None,
    )
    .await
    .map_err(|e| miette!("provider init failed: {e}"))?;

    // Run initial health checks on all leaf adapters.
    for p in oxigate::providers::leaf_adapters(&provider) {
        let status = p.health_check().await;
        health_tracker
            .update_health(&p.metadata().name, status)
            .await;
    }

    let provider_holder: Arc<tokio::sync::RwLock<Arc<dyn ProviderAdapterExt>>> =
        Arc::new(tokio::sync::RwLock::new(provider));

    let auth_holder: Arc<tokio::sync::RwLock<oxigate::config::AuthConfig>> =
        Arc::new(tokio::sync::RwLock::new(config.auth.clone()));
    if config.auth.key.is_none() {
        tracing::warn!("auth.key not configured — all /v1/* requests accepted; set for production");
    }

    let global_safety_holder: Arc<
        tokio::sync::RwLock<oxigate::middleware::global_safety::GlobalSafetyRuntimeConfig>,
    > = Arc::new(tokio::sync::RwLock::new(
        oxigate::middleware::global_safety::GlobalSafetyRuntimeConfig::from_budget_config(
            &config.budget,
        ),
    ));

    let budget_settings_holder: Arc<tokio::sync::RwLock<oxigate::config::BudgetConfig>> =
        Arc::new(tokio::sync::RwLock::new(config.budget.clone()));

    let security_holder: Arc<tokio::sync::RwLock<oxigate::config::SecurityConfig>> =
        Arc::new(tokio::sync::RwLock::new(config.security.clone()));

    let budget_holder: Arc<tokio::sync::RwLock<oxigate::middleware::budget::BudgetRuntimeConfig>> =
        Arc::new(tokio::sync::RwLock::new(
            oxigate::middleware::budget::BudgetRuntimeConfig::from_budget_config(
                config.budget.clone(),
            ),
        ));

    #[cfg(unix)]
    {
        let active_config: Arc<tokio::sync::RwLock<oxigate::config::GatewayConfig>> =
            Arc::new(tokio::sync::RwLock::new(config.clone()));
        oxigate::lifecycle::spawn_sighup_reload(
            config_path.clone(),
            Arc::clone(&active_config),
            resources.clone(),
            oxigate::lifecycle::ReloadHandles {
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

    #[cfg(feature = "pro")]
    {
        let sched = oxigate::middleware::budget_scheduler::BudgetResetScheduler::new(
            Arc::clone(&budget_holder),
            Arc::clone(&resources.redis_pool),
        );
        tokio::spawn(sched.run());
    }

    let drain_secs = config.server.drain_timeout_secs;
    let (drain_tx, drain_rx) = tokio::sync::oneshot::channel::<()>();
    let (drain_started_tx, drain_started_rx) = tokio::sync::oneshot::channel::<()>();

    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = drain_tx.send(());
        let _ = drain_started_tx.send(());
    });

    let state = oxigate::api::AppState {
        pool: Arc::clone(&resources.pool),
        redis_pool: Arc::clone(&resources.redis_pool),
        pricing_db: pricing_db_holder,
        provider: provider_holder,
        auth: auth_holder,
        global_safety: global_safety_holder,
        budget_settings: Arc::clone(&budget_settings_holder),
        budget: budget_holder,
        startup_time,
        health: health_tracker,
        security: security_holder,
    };
    let router = oxigate::api::router_with_metrics(state, prometheus_handle);
    let serve = axum::serve(listener, router).with_graceful_shutdown(async move {
        drain_rx.await.ok();
    });
    let mut serve_handle =
        tokio::spawn(async move { serve.await.map_err(|e| miette!("server error: {e}")) });

    // Race shutdown vs server exit: if serve_handle completes first (e.g. listener error),
    // we exit immediately instead of hanging on drain_started_rx.
    // Err from drain_started_rx means sender dropped (e.g. shutdown task panicked) — fall through to serve_handle arm.
    tokio::select! {
        Ok(()) = drain_started_rx => {
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

/// Runs migrate mode: connect, run migrations, exit 0 on success.
async fn run_migrate(config_path: &std::path::Path) -> Result<()> {
    let config =
        oxigate::config::load_and_validate_config(config_path).map_err(|e| miette!("{e}"))?;
    let _handle = oxigate::observability::init_tracing(&config.log_level)?;
    oxigate::db::init_db(&config)
        .await
        .map_err(|e| miette!("DB connection failed: {e}"))?;
    tracing::info!("migrations applied successfully");
    Ok(())
}

/// Runs validate-config mode: load and validate, exit 0 on success, 1 on failure.
/// Uses eprintln! for error output (allowed: tracing not initialized in offline validation mode).
#[allow(clippy::print_stdout, clippy::print_stderr)]
fn run_validate_config(config_path: &std::path::Path) -> Result<()> {
    match oxigate::config::load_and_validate_config(config_path) {
        Ok(_) => {
            println!("config is valid");
            Ok(())
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
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

/// Parses CLI args. Returns (run_mode, config_path).
fn parse_args() -> Result<(RunMode, PathBuf)> {
    let args: Vec<String> = std::env::args().collect();
    let mode = match args.get(1).map(String::as_str) {
        Some("validate-config") => RunMode::ValidateConfig,
        Some("migrate") if args.get(2).map(String::as_str) == Some("run") => RunMode::Migrate,
        _ => RunMode::Serve,
    };
    let config_path = parse_config_arg(&args)?;
    Ok((mode, config_path))
}

/// Extracts --config path from args. Uses default config/oxigate.yaml when omitted.
fn parse_config_arg(args: &[String]) -> Result<PathBuf> {
    let mut iter = args.iter().skip(1);
    while let Some(arg) = iter.next() {
        if arg == "--config" {
            let path: PathBuf = iter
                .next()
                .ok_or_else(|| miette!("--config requires a path argument"))?
                .into();
            if path.exists() && !path.is_file() {
                return Err(miette!("{} is a directory, not a file", path.display()));
            }
            std::fs::File::open(&path)
                .map_err(|e| miette!("cannot open config file {}: {}", path.display(), e))?;
            return Ok(path);
        }
    }
    let default = PathBuf::from("config/oxigate.yaml");
    std::fs::File::open(&default).map_err(|e| {
        miette!(
            "no --config path given and default config/oxigate.yaml unavailable: {}; \
             pass --config <path> or mount a config file",
            e
        )
    })?;
    Ok(default)
}
