// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Property-based invariant tests for budget logic.
//!
//! Cost calculator invariants are tested in unit tests (domain::pricing::tests).
//! Budget stubs remain here until wires real BudgetCounter.

use proptest::prelude::*;

/// Stub: simulates balance after spend. Re-wire after to real BudgetCounter.
fn stub_balance_after_reset(initial: u64, spend: u64) -> i64 {
    let to_spend = initial.min(spend);
    let remaining = initial.saturating_sub(to_spend);
    remaining as i64
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn prop_reset_never_produces_negative_balance(
        initial in 0u64..1_000_000u64,
        spend in 0u64..1_000_000u64,
    ) {
        let balance = stub_balance_after_reset(initial, spend);
        // Invariant: reset on zeroed counter must never produce negative balance.
        // Stub: saturating_sub + max(0) ensures non-negative.
        // After: wire to real BudgetCounter and verify same invariant under concurrent access.
        prop_assert!(balance >= 0, "balance={} must be >= 0", balance);
    }
}
