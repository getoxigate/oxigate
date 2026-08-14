// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! OxiGate gateway binary — composition root.
//!
//! Wires axum server, config parsing, graceful shutdown, and SIGHUP hot-reload.
//! Resource initialization and reload logic live in `lifecycle`.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use miette::{Result, miette};

#[derive(Parser)]
#[command(version, about = "OxiGate — LLM FinOps gateway")]
struct Cli {
    /// Path to the config file.
    #[arg(long, global = true, default_value = "config/oxigate.yaml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Validate the config file and exit (0 = valid, 1 = invalid).
    ValidateConfig,
    /// Database migration commands.
    Migrate {
        #[command(subcommand)]
        action: MigrateAction,
    },
}

#[derive(Subcommand)]
enum MigrateAction {
    /// Apply all pending migrations and exit.
    Run,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = oxigate::cli::validate_config_path(cli.config)?;

    match cli.command {
        Some(Command::ValidateConfig) => return run_validate_config(&config_path),
        Some(Command::Migrate {
            action: MigrateAction::Run,
        }) => return run_migrate(&config_path).await,
        None => {}
    }

    let config =
        oxigate::config::load_and_validate_config(&config_path).map_err(|e| miette!("{e}"))?;
    oxigate::Gateway::build(config, config_path)
        .await?
        .serve()
        .await
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
