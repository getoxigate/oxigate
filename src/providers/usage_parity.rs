// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Shared test support for provider usage projections: field-by-field comparison of two
//! projected `Usage` values, and one synthetic price entry for exact-cost assertions.
//!
//! A wire shape with both a buffered and a streamed path must project the same facts into the
//! same `Usage` on either. These assertions are provider-neutral — they read only the domain
//! types — so every lane's parity tests share one definition of "the same".

use std::sync::{Arc, RwLock};

use crate::config::PricingConfig;
use crate::domain::chat::{CompletionTokensDetails, PromptTokensDetails, Usage};
use crate::domain::pricing::PricingDb;
use crate::domain::usage_accounting::CacheWriteAccounting;

/// An invented model, priced only by [`synthetic_pricing_holder`].
pub(super) const SYNTHETIC_MODEL: &str = "usage-projection-fixture";

/// Arbitrary, permanent rates for [`SYNTHETIC_MODEL`], one tier: input $2/Mtok (2,000 nano-USD a
/// token), output $10/Mtok (10,000), cache reads at 0.5x input, cache writes at 1.5x (`5m`) and
/// 3.0x (`1h`) input, and no separate thinking rate, so thinking bills at the output rate.
///
/// Synthetic on purpose. An exact-cost test checks a projection and the arithmetic downstream of
/// it; pricing it against a bundled catalogue entry would couple it to prices that legitimately
/// move, and a refresh would turn it red for a reason unrelated to what it checks. The rates are
/// deliberately unlike any real entry, so a test cannot pass on catalogue values by accident.
const SYNTHETIC_PRICING_JSON: &str = r#"{"models":{"usage-projection-fixture":{
    "provider":"test","context_window":1000000,"aliases":[],
    "tiers":[
      {"threshold":0,"input_per_token":0.000002,"output_per_token":0.00001,
       "cache_read_multiplier":0.5,"cache_write_multipliers":{"5m":1.5,"1h":3.0}}
    ]}}}"#;

/// A pricing holder over [`SYNTHETIC_PRICING_JSON`], for exact-cost tests. Take the request's
/// pricing context from the same holder with `snapshot_pricing_context`, so the projection and
/// the finalization price from one generation.
pub(super) fn synthetic_pricing_holder() -> Arc<RwLock<PricingDb>> {
    Arc::new(RwLock::new(
        PricingDb::load(SYNTHETIC_PRICING_JSON.as_bytes(), &PricingConfig::default())
            .expect("the synthetic pricing fixture loads"),
    ))
}

/// Asserts two `Usage` values agree on every field, including every component of the
/// cache-write accounting.
///
/// Each field is named rather than compared as a whole value: `Usage` and
/// `CacheWriteAccounting` carry no `PartialEq`, and the failure this guards against is one
/// path populating a field the other forgets, which is only actionable when the report says
/// which field moved.
pub(super) fn assert_usage_parity(buffered: &Usage, streamed: &Usage, case: &str) {
    // Destructured exhaustively, with no `..`: a field added to `Usage` or to either details
    // struct is a compile error here until this helper compares it, so no parity suite can stay
    // green while ignoring it.
    let Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        completion_tokens_details,
        cache_creation_input_tokens,
        cache_read_input_tokens,
        prompt_tokens_details,
        accounting,
        inference_geo,
        cache_write,
        image_units,
        audio_seconds,
    } = buffered;
    let reasoning = |details: &Option<CompletionTokensDetails>| {
        details
            .as_ref()
            .map(|CompletionTokensDetails { reasoning_tokens }| *reasoning_tokens)
    };
    let prompt_details = |details: &Option<PromptTokensDetails>| {
        details.as_ref().map(
            |PromptTokensDetails {
                 cached_tokens,
                 cache_write_tokens,
             }| (*cached_tokens, *cache_write_tokens),
        )
    };

    assert_eq!(
        *prompt_tokens, streamed.prompt_tokens,
        "{case}: prompt_tokens"
    );
    assert_eq!(
        *completion_tokens, streamed.completion_tokens,
        "{case}: completion_tokens"
    );
    assert_eq!(*total_tokens, streamed.total_tokens, "{case}: total_tokens");
    assert_eq!(
        reasoning(completion_tokens_details),
        reasoning(&streamed.completion_tokens_details),
        "{case}: completion_tokens_details.reasoning_tokens"
    );
    assert_eq!(
        *cache_creation_input_tokens, streamed.cache_creation_input_tokens,
        "{case}: cache_creation_input_tokens"
    );
    assert_eq!(
        *cache_read_input_tokens, streamed.cache_read_input_tokens,
        "{case}: cache_read_input_tokens"
    );
    assert_eq!(
        prompt_details(prompt_tokens_details),
        prompt_details(&streamed.prompt_tokens_details),
        "{case}: prompt_tokens_details"
    );
    assert_eq!(*accounting, streamed.accounting, "{case}: accounting");
    assert_eq!(
        *inference_geo, streamed.inference_geo,
        "{case}: inference_geo"
    );
    assert_eq!(*image_units, streamed.image_units, "{case}: image_units");
    assert_eq!(
        *audio_seconds, streamed.audio_seconds,
        "{case}: audio_seconds"
    );
    assert_cache_write_parity(cache_write, &streamed.cache_write, case);
}

/// Every component of one response's cache-write accounting, compared field by field.
///
/// The pricing generation is compared by presence only: `PricingContext` is a snapshot of a
/// loaded catalogue, and the two paths take their own snapshot of the same one, so identity
/// is not the property under test — that each path attached one is.
fn assert_cache_write_parity(
    buffered: &CacheWriteAccounting,
    streamed: &CacheWriteAccounting,
    case: &str,
) {
    assert_eq!(
        buffered.reported_tokens(),
        streamed.reported_tokens(),
        "{case}: cache_write.reported_tokens"
    );
    assert_eq!(
        buffered.detail_tokens(),
        streamed.detail_tokens(),
        "{case}: cache_write.detail_tokens"
    );
    assert_eq!(
        buffered.accounted_tokens(),
        streamed.accounted_tokens(),
        "{case}: cache_write.accounted_tokens"
    );
    assert_eq!(
        buffered.class_totals(),
        streamed.class_totals(),
        "{case}: cache_write.class_totals"
    );
    assert_eq!(
        buffered.unknown_tokens(),
        streamed.unknown_tokens(),
        "{case}: cache_write.unknown_tokens"
    );
    assert_eq!(
        buffered.unmatched_residual_tokens(),
        streamed.unmatched_residual_tokens(),
        "{case}: cache_write.unmatched_residual_tokens"
    );
    assert_eq!(
        buffered.fallback_tokens(),
        streamed.fallback_tokens(),
        "{case}: cache_write.fallback_tokens"
    );
    assert_eq!(
        buffered.quantity_overflow(),
        streamed.quantity_overflow(),
        "{case}: cache_write.quantity_overflow"
    );
    assert_eq!(
        buffered.partition_is_exact(),
        streamed.partition_is_exact(),
        "{case}: cache_write.partition_is_exact"
    );
    assert_eq!(
        buffered.observation_count(),
        streamed.observation_count(),
        "{case}: cache_write.observation_count"
    );
    assert_eq!(
        buffered.published_tokens(),
        streamed.published_tokens(),
        "{case}: cache_write.published_tokens"
    );
    assert_eq!(
        buffered.duplicate(),
        streamed.duplicate(),
        "{case}: cache_write.duplicate"
    );
    assert_eq!(
        buffered.outcome(),
        streamed.outcome(),
        "{case}: cache_write.outcome"
    );
    assert_eq!(
        buffered.evidence_entries(),
        streamed.evidence_entries(),
        "{case}: cache_write.evidence_entries"
    );
    assert_eq!(
        buffered.evidence_truncated(),
        streamed.evidence_truncated(),
        "{case}: cache_write.evidence_truncated"
    );
    assert_eq!(
        buffered.pricing_context().is_some(),
        streamed.pricing_context().is_some(),
        "{case}: cache_write.pricing_context presence"
    );
}
