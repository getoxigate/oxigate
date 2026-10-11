// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! `spend-capture` command line.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use miette::{IntoDiagnostic, WrapErr, miette};
use secrecy::SecretString;
use spend_capture::capture::{CaptureConfig, CaptureServer, Route};
use spend_capture::replay::{ReplayConfig, ReplayServer, SessionEnd};

/// Default results directory: the crate's own `results/`, which is git- and Docker-ignored.
const DEFAULT_RESULTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/results");

/// Default listen address, shared by both subcommands so the gateway's configuration is the same
/// for a capture and for its replay.
const DEFAULT_LISTEN: &str = "127.0.0.1:8480";

#[derive(Parser)]
#[command(name = "spend-capture", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Forward the gateway's provider traffic and record every exchange.
    Capture(CaptureArgs),
    /// Serve a capture session's recordings back to the gateway, in order, refusing any request
    /// that does not match the next one.
    Replay(ReplayArgs),
}

#[derive(Args)]
struct CaptureArgs {
    /// Address to accept plain-HTTP requests from the gateway on.
    #[arg(long, default_value = DEFAULT_LISTEN)]
    listen: SocketAddr,
    /// A route, as NAME=ORIGIN: requests under /NAME are forwarded to ORIGIN
    /// (e.g. openai=https://api.openai.com). Repeatable.
    #[arg(long = "route", value_name = "NAME=ORIGIN", required = true, value_parser = Route::parse)]
    routes: Vec<Route>,
    /// Results directory. Must not already hold recordings or incident records.
    #[arg(long, value_name = "DIR", default_value = DEFAULT_RESULTS)]
    results: PathBuf,
    /// Name of an environment variable holding a credential to scan bodies for, in addition to
    /// those each exchange carries in its headers and URI. Repeatable. The value is read from the
    /// environment so it never appears on the command line.
    #[arg(long = "declare-env", value_name = "VAR")]
    declare_env: Vec<String>,
    /// PEM file of extra certificates to trust for upstream TLS, besides the bundled web roots.
    #[arg(long, value_name = "PEM")]
    upstream_ca: Option<PathBuf>,
}

#[derive(Args)]
struct ReplayArgs {
    /// Address to accept plain-HTTP requests from the gateway on.
    #[arg(long, default_value = DEFAULT_LISTEN)]
    listen: SocketAddr,
    /// Results directory of the capture session to replay. It is only read.
    #[arg(long, value_name = "DIR", default_value = DEFAULT_RESULTS)]
    results: PathBuf,
    /// Name of an environment variable holding a credential declared to the capture, so it is
    /// scrubbed from everything the replay says about a request. Repeatable.
    #[arg(long = "declare-env", value_name = "VAR")]
    declare_env: Vec<String>,
}

#[tokio::main]
async fn main() -> miette::Result<()> {
    match Cli::parse().command {
        Command::Capture(args) => capture(args).await,
        Command::Replay(args) => replay(args).await,
    }
}

/// Reads each `--declare-env` variable's value from the environment.
fn declared_credentials(vars: &[String]) -> miette::Result<Vec<SecretString>> {
    vars.iter()
        .map(|var| {
            std::env::var(var)
                .ok()
                .filter(|value| !value.is_empty())
                .map(SecretString::new)
                .ok_or_else(|| miette!("--declare-env {var}: the variable is unset or empty"))
        })
        .collect()
}

async fn capture(args: CaptureArgs) -> miette::Result<()> {
    let declared = declared_credentials(&args.declare_env)?;
    let extra_root_certificates_pem = match &args.upstream_ca {
        Some(path) => Some(
            std::fs::read(path)
                .into_diagnostic()
                .wrap_err_with(|| format!("reading {}", path.display()))?,
        ),
        None => None,
    };
    let server = CaptureServer::bind(
        args.listen,
        CaptureConfig {
            routes: args.routes,
            results_dir: args.results,
            declared_credentials: declared,
            extra_root_certificates_pem,
        },
    )
    .await
    .into_diagnostic()?;

    tokio::signal::ctrl_c().await.into_diagnostic()?;
    eprintln!(
        "spend-capture: stopping; waiting for {} exchange(s) in flight (Ctrl-C again to abort)",
        server.in_flight()
    );
    tokio::select! {
        () = server.shutdown() => Ok(()),
        interrupted = tokio::signal::ctrl_c() => {
            interrupted.into_diagnostic()?;
            Err(miette!("aborted with exchanges still in flight; they were not recorded"))
        }
    }
}

/// Serves until interrupted, then — once the requests still open have been answered — fails if
/// the session diverged or left recordings unserved.
async fn replay(args: ReplayArgs) -> miette::Result<()> {
    let server = ReplayServer::bind(
        args.listen,
        ReplayConfig {
            results_dir: args.results,
            declared_credentials: declared_credentials(&args.declare_env)?,
        },
    )
    .await
    .into_diagnostic()?;
    tokio::signal::ctrl_c().await.into_diagnostic()?;
    eprintln!("spend-capture replay: stopping; waiting for open requests to finish");
    let SessionEnd {
        mismatch,
        remaining,
    } = server.shutdown().await;
    if let Some(mismatch) = mismatch {
        return Err(miette!(
            "the replay diverged from the recordings: {mismatch}"
        ));
    }
    if remaining > 0 {
        return Err(miette!("{remaining} recording(s) were never requested"));
    }
    Ok(())
}
