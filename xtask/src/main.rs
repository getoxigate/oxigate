// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
use anyhow::{Context, Result, bail};
use std::{
    env,
    process::{Command, ExitStatus},
};

mod usage_scan;

fn main() -> Result<()> {
    let task = env::args().nth(1);
    match task.as_deref() {
        Some("check") => check(),
        Some("ci") => ci(),
        Some("audit") => audit(),
        Some("doc") => doc(),
        Some("bench") => bench(),
        Some("sqlx-prepare") => sqlx_prepare(),
        Some("usage-scan") => usage_scan::run(),
        Some(unknown) => bail!("unknown xtask: `{unknown}`"),
        None => {
            eprintln!("Usage: cargo xtask <check|ci|audit|doc|bench|sqlx-prepare|usage-scan>");
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

/// Local pre-commit gate: provider usage scan → fmt → clippy → test. Exits on first failure.
fn check() -> Result<()> {
    // First: it parses source and runs no toolchain, so it costs well under a second.
    usage_scan::run()?;
    // The scan's own fixtures. The test runs below cover only the root package, and a scanner
    // whose tests never run can start passing vacuously without anyone noticing. Run as its own
    // invocation so the xtask's parser features do not unify into the library's build.
    run("cargo", &["test", "--package", "xtask"])?;
    run("cargo", &["fmt", "--all", "--check"])?;
    run(
        "cargo",
        &["clippy", "--all-features", "--", "-D", "warnings"],
    )?;
    // Use nextest if available, fall back to cargo test
    // Note: `cargo --list` shows subcommand names (e.g. "nextest"), not binary names
    if command_exists("nextest") {
        run("cargo", &["nextest", "run", "--all-features"])?;
    } else {
        run("cargo", &["test", "--all-features"])?;
    }
    Ok(())
}

/// Full CI gate: check-all → audit → doc.
fn ci() -> Result<()> {
    check()?;
    audit()?;
    doc()?;
    Ok(())
}

/// Dependency and license audit.
fn audit() -> Result<()> {
    run("cargo", &["audit"])?;
    run("cargo", &["deny", "check"])?;
    Ok(())
}

/// Documentation build (zero warnings required).
fn doc() -> Result<()> {
    run_env(
        "cargo",
        &["doc", "--no-deps", "--all-features"],
        &[("RUSTDOCFLAGS", "-D warnings")],
    )
}

/// Runs `cargo sqlx prepare` for offline compilation.
///
/// Requires `DATABASE_URL` to be set. Run against a live PG instance with migrations applied.
/// CI uses `SQLX_OFFLINE=true` and `cargo sqlx check` — not this command.
fn sqlx_prepare() -> Result<()> {
    let status = Command::new("cargo")
        .args(["sqlx", "prepare", "--", "--all-targets", "--all-features"])
        .status()
        .with_context(|| "failed to launch `cargo sqlx prepare`")?;
    if !status.success() {
        eprintln!("cargo sqlx prepare failed");
        std::process::exit(1);
    }
    Ok(())
}

/// Criterion benchmark suite (advisory — does not block CI merge).
/// Skips gracefully if no benches/ directory exists.
fn bench() -> Result<()> {
    if !std::path::Path::new("benches").exists() {
        eprintln!("xtask bench: no benches/ directory found — skipping");
        return Ok(());
    }
    // TODO: replace `cargo bench` with `cargo criterion` once benches/ harness exists.
    run("cargo", &["bench", "--all-features"])
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run(program: &str, args: &[&str]) -> Result<()> {
    run_env(program, args, &[])
}

fn run_env(program: &str, args: &[&str], env_vars: &[(&str, &str)]) -> Result<()> {
    let status: ExitStatus = Command::new(program)
        .args(args)
        .envs(env_vars.iter().copied())
        .status()
        .with_context(|| format!("failed to launch `{program}`"))?;
    if !status.success() {
        bail!("`{program} {}` exited with {status}", args.join(" "));
    }
    Ok(())
}

fn command_exists(name: &str) -> bool {
    Command::new("cargo")
        .args(["--list"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(name))
        .unwrap_or(false)
}
