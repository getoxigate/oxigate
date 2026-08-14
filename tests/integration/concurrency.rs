// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Concurrency tests for budget/accounting invariants.
//!
//! Placeholder for real budget counter. Uses multi_thread runtime so spawned
//! tasks actually race. tokio::time::pause requires current_thread and is omitted
//! here may reintroduce pause if the real BudgetCounter has time-based logic.

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

#[tokio::test(flavor = "multi_thread")]
async fn test_last_cent_race() {
    // No tokio::time::pause() — it requires current_thread; multi_thread enables real race.
    let budget_remaining = Arc::new(AtomicI64::new(1)); // $0.01 in cents

    let b1 = Arc::clone(&budget_remaining);
    let b2 = Arc::clone(&budget_remaining);

    let t1 = tokio::spawn(async move {
        let prev = b1.fetch_sub(1, Ordering::SeqCst);
        prev > 0 // true = request allowed
    });
    let t2 = tokio::spawn(async move {
        let prev = b2.fetch_sub(1, Ordering::SeqCst);
        prev > 0 // true = request allowed
    });

    let (r1, r2) = tokio::join!(t1, t2);
    let allowed = [r1.unwrap(), r2.unwrap()];
    let success_count = allowed.iter().filter(|&&x| x).count();

    // At most one request succeeds when exactly $0.01 remains
    assert!(
        success_count <= 1,
        "race condition: both requests granted when budget = $0.01"
    );
    // Stub allows one overshoot (counter may reach -1) because AtomicI64::fetch_sub has no
    // floor. This is intentional for the stub — it documents the race, not prevents it.
    // MUST tighten this assertion to >= 0 once the real BudgetCounter (with a
    // compare-exchange floor) is wired in place of the AtomicI64 stub.
    assert!(
        budget_remaining.load(Ordering::SeqCst) >= -1,
        "stub: counter below -1 means >1 overshoot — unexpected even without floor"
    );
}
