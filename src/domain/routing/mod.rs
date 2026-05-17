// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Routing strategies for multi-provider dispatch .
//!
//! Each strategy implements `RoutingStrategy` from `domain::ports` and is a pure
//! function — no I/O, no locks. Input is a pre-built `&[&ProviderCandidate]` slice
//! produced by `ProviderHealthTracker::candidates()`.

pub mod lowest_cost;
pub mod rate_limit_aware;
pub mod weighted_random;

pub use lowest_cost::LowestCost;
pub use rate_limit_aware::RateLimitAware;
pub use weighted_random::WeightedRandom;
