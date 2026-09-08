// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Cost header construction for chat completions responses.
//!
//! Builds [`CostHeader::REQUEST_COST`], [`CostHeader::INPUT_TOKENS`], [`CostHeader::OUTPUT_TOKENS`],
//! [`CostHeader::MODEL_USED`] and [`CostHeader::COST_STATUS`] using the bundled pricing DB.
//!
//! It is also where a completed request's single structured warning is decided:
//! [`report_finalized_warning`] reads the one finalization result this module produces, and is
//! the only site that emits one.

use std::sync::Arc;

use axum::http::header::{HeaderMap, HeaderValue};
use axum::response::Response;
use tracing::warn;

use crate::domain::chat::{CacheAccounting, ReasoningAccounting, Usage};
use crate::domain::ports::{CostCalculator, TokenUsage};
use crate::domain::pricing::BundledCostCalculator;
use crate::domain::usage_accounting::{
    CACHE_WRITE_EVIDENCE_MAX_BYTES, CostStatus, FinalizedAccounting, ReconciliationFacts,
    WarningFacts, WarningReason,
};
use crate::observability::metrics;

/// The `cost-unavailable` spelling as a `'static` literal, for the header paths that need one.
///
/// [`HeaderValue::from_static`] panics on an invalid value, and nothing on a request path may
/// panic, so the two sites that cannot build their value from a fallible conversion use this
/// instead. `header_fallback_matches_the_status_spelling` keeps it equal to
/// [`CostStatus::CostUnavailable`]'s own wire spelling.
const COST_UNAVAILABLE_HEADER: &str = "cost-unavailable";

pub struct CostHeader;

impl CostHeader {
    pub const REQUEST_COST: &'static str = "X-Oxigate-Request-Cost";
    pub const INPUT_TOKENS: &'static str = "X-Oxigate-Input-Tokens";
    pub const OUTPUT_TOKENS: &'static str = "X-Oxigate-Output-Tokens";
    pub const MODEL_USED: &'static str = "X-Oxigate-Model-Used";
    pub const BUDGET_REMAINING: &'static str = "X-Oxigate-Budget-Remaining";
    pub const BUDGET_CAP: &'static str = "X-Oxigate-Budget-Cap";
    /// How much confidence this request's cost carries — see [`CostStatus`].
    ///
    /// Emitted wherever OxiGate emits its cost headers at all, including embeddings, requests
    /// with no cache write, and the zero-cost path that has no calculation behind it. A cost
    /// number without it is a number whose confidence the client has to guess at.
    pub const COST_STATUS: &'static str = "X-Oxigate-Cost-Status";
}

/// Builds cost headers for the response.
///
/// Returns `(HeaderMap, FinalizedAccounting)`: the headers, and the one authoritative
/// accounting result — cost, status, reconciliation facts, evidence and warning facts computed
/// together. Callers pass that value on for spend writing, structured cost logging and budget
/// accounting rather than reassembling it from loose parts.
///
/// Takes `Arc<RwLock<PricingDb>>` because `BundledCostCalculator` retains it for
/// Class A hot-reload (SIGHUP) semantics; the clone is cheap.
#[must_use]
pub fn build_cost_headers(
    model: &str,
    usage: &Usage,
    pricing_db: Arc<std::sync::RwLock<crate::domain::pricing::PricingDb>>,
    batch: bool,
) -> (HeaderMap, FinalizedAccounting) {
    let thinking_tokens = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|d| d.reasoning_tokens)
        .unwrap_or(0);
    // OpenAI: prompt_tokens is total (plain + cached); cached come from prompt_tokens_details.
    // Anthropic/Gemini: cache_read/cache_creation are additive; input_tokens excludes them.
    let cache_read = usage
        .cache_read_input_tokens
        .or_else(|| {
            // Fallback for OpenAI: when cache_read_input_tokens not yet normalized from
            // prompt_tokens_details.cached_tokens (e.g. raw JSON from compat provider).
            usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
        })
        .unwrap_or(0);
    // The single cache-write quantity this request is billed on. Read once: the carve-out below,
    // the invariant check and the priced accumulator must all see the same number.
    let accounted_cache_write = usage.cache_write.accounted_tokens();
    // Under `Inclusive` every bucket the provider counts inside `prompt_tokens` is carved out
    // here and charged once at its own multiplier — cache reads at the read multiplier, cache
    // writes at their class multiplier. Leaving either inside would charge it a second time at
    // the plain input rate; leaving the written tokens in is what would make OpenAI's documented
    // 1.25x cache write bill at 2.25x. Under `Additive` the provider reports the buckets
    // disjointly and `prompt_tokens` is already the plain portion.
    //
    // Saturating, and deliberately so: a provider reporting more cached tokens than prompt
    // tokens yields no plain input charge rather than wrapping. `detect_usage_invariants` raises
    // `cache_exceeds_prompt` for exactly that case, so the clamp is never silent.
    //
    // `TokenUsage::context_input_tokens` re-adds both buckets, so the tier comparator counts each
    // input bucket exactly once under either contract: under `Inclusive` it reconstructs the
    // provider-reported prompt total, and under `Additive` it sums the disjoint buckets the
    // provider reported separately.
    let input_tokens = match usage.accounting.cache {
        CacheAccounting::Additive => usage.prompt_tokens,
        CacheAccounting::Inclusive => usage
            .prompt_tokens
            .saturating_sub(cache_read)
            .saturating_sub(accounted_cache_write),
    };
    let token_usage = TokenUsage {
        input_tokens,
        output_tokens: usage.completion_tokens,
        cache_read_input_tokens: cache_read,
        // Already reconciled by the provider lane that parsed the payload, against the pricing
        // generation the request was dispatched under. Rebuilding it here from published
        // quantities would reconcile a second time, against a registry that may since have been
        // reloaded. A provider that reports no cache-write classes carries the empty default.
        cache_write: usage.cache_write.clone(),
        thinking_tokens,
        batch,
        image_count: usage.image_units.unwrap_or(0),
        audio_seconds: usage.audio_seconds.unwrap_or(0.0),
        reasoning_accounting: usage.accounting.reasoning,
    };
    let invariants =
        detect_usage_invariants(usage, cache_read, accounted_cache_write, thinking_tokens);
    record_usage_invariant_metrics(invariants);
    let extra_reconciliation = ReconciliationFacts {
        cache_exceeds_prompt: invariants.cache_exceeds_prompt,
        reasoning_exceeds_completion: invariants.reasoning_exceeds_completion,
        ..Default::default()
    };
    assemble_cost_headers(
        model,
        &token_usage,
        usage.prompt_tokens,
        usage.completion_tokens,
        pricing_db,
        extra_reconciliation,
    )
}

/// Finalizes a request whose provider reported no usage at all.
///
/// Returns the accounting **only** — no `HeaderMap`. A streamed response has no cost-header
/// channel: its headers are sent before the first chunk, so there is nothing to attach them to,
/// and streaming is this function's only caller. The map is also the half that would be wrong:
/// [`assemble_cost_headers`] bakes the status into `X-Oxigate-Cost-Status` while building it,
/// *before* [`mark_usage_missing`] runs, so returning both would hand the caller a pair that can
/// disagree. Should a buffered caller ever need headers here, build them **after** the force.
///
/// No usage is not the same fact as zero usage: every quantity is unreported, so the result is
/// an all-zero cost carrying [`CostStatus::CostUnavailable`] rather than a confident zero.
#[must_use]
pub(crate) fn finalize_missing_usage(
    model: &str,
    pricing_db: Arc<std::sync::RwLock<crate::domain::pricing::PricingDb>>,
    batch: bool,
) -> FinalizedAccounting {
    let token_usage = TokenUsage {
        batch,
        ..Default::default()
    };
    let (_headers, mut finalized) = assemble_cost_headers(
        model,
        &token_usage,
        0,
        0,
        pricing_db,
        ReconciliationFacts::default(),
    );
    mark_usage_missing(&mut finalized);
    finalized
}

/// Forces the missing-usage verdict onto a finalization result.
///
/// Both arms of [`assemble_cost_headers`] already land on `cost-unavailable` for an all-zero
/// usage — the priced arm through the calculator's no-usable-usage gate, the unpriced arm through
/// `CostBreakdown::default()`. This states the guarantee once, at one site, so it does not rest on
/// two unrelated code paths continuing to agree. It is extracted rather than inlined so that
/// force is testable: driven end to end it cannot be pinned, because both arms produce the value
/// it forces.
fn mark_usage_missing(finalized: &mut FinalizedAccounting) {
    finalized.cost.status = finalized.cost.status.worst(CostStatus::CostUnavailable);
    // `WarningFacts::add` ignores a repeat, so the status reason the assembler already recorded
    // is not duplicated here.
    finalized.warning.add(WarningReason::CostUnavailable);
    finalized.warning.add(WarningReason::ProviderUsageMissing);
}

/// Reported-usage contradictions found in a single provider payload.
///
/// Detection is kept separate from emission so it can be exercised directly, and so the metric
/// leaves through the observability wrapper at exactly one site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct UsageInvariantFlags {
    /// An `Inclusive` contract reported more cached tokens — read plus accounted write — than the
    /// prompt total documented to contain them; the billable input clamps to zero.
    cache_exceeds_prompt: bool,
    /// An `IncludedInOutput` contract reported more reasoning tokens than the completion total
    /// documented to contain them; the standard output charge clamps to zero.
    reasoning_exceeds_completion: bool,
}

/// Detects reported numbers that contradict the accounting the provider contract declares.
///
/// Only a *subset* relation can be contradicted. A contract that reports its buckets disjointly
/// cannot violate one, so nothing is checked on that axis — checking anyway would fire on every
/// normal request to such a provider.
///
/// The cache check is scoped to the buckets the billable input is actually derived from — under
/// `Inclusive` that is cache read *and* accounted cache write, the two `build_cost_headers`
/// subtracts. A check narrower than the subtraction would let a provider reporting
/// `read + write > prompt` clamp the input bucket to zero with no flag raised.
///
/// `accounted_cache_write` is passed rather than read from `usage` so the check and the
/// subtraction cannot come to see different quantities.
fn detect_usage_invariants(
    usage: &Usage,
    cache_read: u64,
    accounted_cache_write: u64,
    thinking_tokens: u64,
) -> UsageInvariantFlags {
    // Matched exhaustively, like the clamps they mirror: a new accounting variant must fail the
    // build here too, rather than compile silently as "never violates".
    UsageInvariantFlags {
        cache_exceeds_prompt: match usage.accounting.cache {
            // Both operands are provider-controlled, and `accounted_tokens()` itself saturates,
            // so the sum is checked. An unrepresentable total is a violation, not a pass: a
            // wrapped sum would read as *below* `prompt_tokens` and suppress the clamp it should
            // raise. The subtraction stays saturating — it is only reached once this has fired.
            CacheAccounting::Inclusive => match cache_read.checked_add(accounted_cache_write) {
                Some(cached_portion) => cached_portion > usage.prompt_tokens,
                None => true,
            },
            CacheAccounting::Additive => false,
        },
        reasoning_exceeds_completion: match usage.accounting.reasoning {
            ReasoningAccounting::IncludedInOutput => thinking_tokens > usage.completion_tokens,
            ReasoningAccounting::Additive => false,
        },
    }
}

/// Increments the violation metric for each detected contradiction.
///
/// The operator-visible *log* for these two is no longer emitted here. Both clamps are
/// conservative quantity policies, so both become reconciliation facts on the one finalization
/// result and reach the log through the single warning [`report_finalized_warning`] emits —
/// otherwise a request that also reconciled its cache-write evidence would WARN twice about the
/// same anomaly. The metric is untouched: it is a different consumer with a different cardinality
/// contract, and it keeps firing exactly as it did.
///
/// Takes the already-detected `flags` so one payload is judged exactly once, and what is counted
/// here cannot diverge from the reconciliation facts derived from the same detection.
fn record_usage_invariant_metrics(flags: UsageInvariantFlags) {
    if flags.cache_exceeds_prompt {
        metrics::record_usage_invariant_violation(metrics::INVARIANT_CACHE_EXCEEDS_PROMPT);
    }

    if flags.reasoning_exceeds_completion {
        metrics::record_usage_invariant_violation(metrics::INVARIANT_REASONING_EXCEEDS_COMPLETION);
    }
}

/// Shared computation: cost calculation, the one authoritative finalization result and header
/// assembly, for both chat and embedding paths.
///
/// `prompt_tokens_display` feeds `INPUT_TOKENS` header; `token_usage.input_tokens` feeds cost calc.
/// For embeddings they are equal. For chat they diverge (cached tokens split).
///
/// `extra_reconciliation` carries conservative-quantity facts this function cannot derive from
/// `token_usage` alone — today, chat's two usage-invariant clamps; embeddings passes
/// [`ReconciliationFacts::default()`]. Cache-write reconciliation, evidence and the warning
/// reason set are all derived here, from `token_usage.cache_write`, so cost, status,
/// reconciliation, evidence completeness and warning facts are one result computed at one point
/// rather than related facts assembled at different times by different callers.
fn assemble_cost_headers(
    model: &str,
    token_usage: &TokenUsage,
    prompt_tokens_display: u64,
    completion_tokens_display: u64,
    pricing_db: Arc<std::sync::RwLock<crate::domain::pricing::PricingDb>>,
    extra_reconciliation: ReconciliationFacts,
) -> (HeaderMap, FinalizedAccounting) {
    let calc = BundledCostCalculator::new(pricing_db);
    let mut pricing_error = None;
    let cost = match calc.calculate(model, token_usage) {
        Ok(c) => c,
        Err(e) => {
            // Recorded, not emitted: this arm is reached by every checked failure on the money
            // path and by an unpriced model, which are exactly the requests the one consolidated
            // warning exists to describe. Warning here as well would double-WARN each of them.
            // The model and this reason both appear as fields on that single event.
            pricing_error = Some(e.to_string());
            // `CostBreakdown::default()` carries `CostStatus::default()` == `CostUnavailable` —
            // the same all-zero, unavailable result required for every checked failure on
            // the money path, including the quantity-overflow trigger routed here.
            crate::domain::ports::CostBreakdown::default()
        }
    };

    let cache_write = &token_usage.cache_write;
    let reconciliation = ReconciliationFacts {
        outcome: cache_write.outcome(),
        duplicate: cache_write.duplicate(),
        unknown_classes_present: cache_write.unknown_tokens() > 0,
        cache_exceeds_prompt: extra_reconciliation.cache_exceeds_prompt,
        reasoning_exceeds_completion: extra_reconciliation.reasoning_exceeds_completion,
    };

    let mut status = cost.status;
    if reconciliation.requires_reconciled() {
        status = status.worst(CostStatus::Reconciled);
    }

    let mut evidence = cache_write.to_evidence(cost.cache_write_cost.as_u64());
    let mut evidence_incomplete = false;
    if let Some(doc) = evidence.as_mut() {
        // The document is integers, booleans and strings — serialization cannot fail here.
        let _ = doc.limit_to_bytes(CACHE_WRITE_EVIDENCE_MAX_BYTES);
        evidence_incomplete = doc.cache_write.incomplete;
    }

    let mut warning = WarningFacts::default();
    if reconciliation.outcome.is_contradiction() {
        warning.add(WarningReason::AggregateDetailContradiction);
    }
    if reconciliation.duplicate.any() {
        warning.add(WarningReason::DuplicateAmbiguity);
    }
    if reconciliation.cache_exceeds_prompt {
        warning.add(WarningReason::CacheExceedsPrompt);
    }
    if reconciliation.reasoning_exceeds_completion {
        warning.add(WarningReason::ReasoningExceedsCompletion);
    }
    match status {
        CostStatus::RateFallback => warning.add(WarningReason::RateFallback),
        CostStatus::CostUnavailable => warning.add(WarningReason::CostUnavailable),
        CostStatus::Exact | CostStatus::Reconciled => {}
    }
    if evidence_incomplete {
        warning.add(WarningReason::IncompleteEvidence);
    }
    if let Some(reason) = pricing_error.as_deref() {
        warning.set_pricing_error(reason);
    }

    let mut map = HeaderMap::new();
    map.insert(
        CostHeader::REQUEST_COST,
        HeaderValue::from_str(&cost.total_cost.to_display_string())
            .unwrap_or_else(|_| HeaderValue::from_static("0")),
    );
    map.insert(
        CostHeader::INPUT_TOKENS,
        HeaderValue::from_str(&prompt_tokens_display.to_string())
            .unwrap_or_else(|_| HeaderValue::from_static("0")),
    );
    map.insert(
        CostHeader::OUTPUT_TOKENS,
        HeaderValue::from_str(&completion_tokens_display.to_string())
            .unwrap_or_else(|_| HeaderValue::from_static("0")),
    );
    map.insert(
        CostHeader::MODEL_USED,
        HeaderValue::from_str(model).unwrap_or_else(|_| HeaderValue::from_static("unknown")),
    );
    map.insert(
        CostHeader::COST_STATUS,
        HeaderValue::from_str(status.as_str())
            .unwrap_or_else(|_| HeaderValue::from_static(COST_UNAVAILABLE_HEADER)),
    );

    let finalized = FinalizedAccounting {
        token_usage: token_usage.clone(),
        cost: crate::domain::ports::CostBreakdown { status, ..cost },
        evidence,
        reconciliation,
        warning,
    };
    (map, finalized)
}

/// Builds cost headers for an embeddings response.
///
/// Returns `(HeaderMap, FinalizedAccounting)` — the same one authoritative accounting result the
/// chat path produces — for spend writing and structured logging.
#[must_use]
pub fn build_embedding_cost_headers(
    model: &str,
    usage: &crate::domain::embedding::EmbeddingUsage,
    pricing_db: Arc<std::sync::RwLock<crate::domain::pricing::PricingDb>>,
    is_batch: bool,
) -> (HeaderMap, FinalizedAccounting) {
    let token_usage = TokenUsage {
        input_tokens: usage.prompt_tokens,
        output_tokens: 0,
        cache_read_input_tokens: 0,
        thinking_tokens: 0,
        batch: is_batch,
        image_count: 0,
        audio_seconds: 0.0,
        reasoning_accounting: ReasoningAccounting::Additive,
        ..Default::default()
    };
    assemble_cost_headers(
        model,
        &token_usage,
        usage.prompt_tokens,
        0,
        pricing_db,
        ReconciliationFacts::default(),
    )
}

/// Injects zero-cost headers into an error response.
///
/// Used when provider requests fail before any usage data is available.
/// Sets [`CostHeader::REQUEST_COST`] to `0.000000`, [`CostHeader::INPUT_TOKENS`] and
/// [`CostHeader::OUTPUT_TOKENS`] to `0`, and [`CostHeader::MODEL_USED`] to the attempted model name. If the model string
/// contains invalid header characters (CR/LF/NUL), falls back to "unknown".
pub fn inject_zero_cost_headers(resp: &mut Response, model: &str) {
    resp.headers_mut().insert(
        CostHeader::REQUEST_COST,
        HeaderValue::from_static("0.000000"),
    );
    resp.headers_mut()
        .insert(CostHeader::INPUT_TOKENS, HeaderValue::from_static("0"));
    resp.headers_mut()
        .insert(CostHeader::OUTPUT_TOKENS, HeaderValue::from_static("0"));
    resp.headers_mut().insert(
        CostHeader::MODEL_USED,
        HeaderValue::from_str(model).unwrap_or_else(|_| HeaderValue::from_static("unknown")),
    );
    // There is no calculation behind these headers, so the status is not a degraded cost — it is
    // the absence of one. A client that reads the zero without it cannot tell a free request from
    // a request whose cost OxiGate never established.
    resp.headers_mut().insert(
        CostHeader::COST_STATUS,
        HeaderValue::from_static(COST_UNAVAILABLE_HEADER),
    );
}

/// Emits the single structured warning for one completed request, when it was anomalous.
///
/// Every path that finalizes a request calls this exactly once, reading the one finalization
/// result rather than re-deriving any part of it. It is now the only warning site for cache-write
/// reconciliation, duplicate ambiguity, rate fallback, unavailable cost, the two usage-invariant
/// clamps and incomplete evidence: the four sites that used to warn independently — the two
/// invariant warnings here, the calculator's `model_not_in_pricing_db` event and the cost-error
/// arm in [`assemble_cost_headers`] — all report facts into [`FinalizedAccounting`] and emit
/// nothing themselves. Every metric they fed still fires.
///
/// An exact, consistent, complete request emits nothing at all. The event carries counts and
/// reason codes, and also the request id, the model and the bounded pricing failure reason (when
/// there is one) — but never a provider member name or prompt content.
///
/// `request_id` is the join key, and it is required rather than optional: the per-request cost
/// `INFO` line carries the same value, and without it the two events cannot be correlated at all
/// under concurrent requests on the same model — which is the ordinary case, not an edge one.
pub fn report_finalized_warning(request_id: &str, model: &str, accounting: &FinalizedAccounting) {
    if !accounting.warning.should_warn() {
        return;
    }
    let reasons = accounting
        .warning
        .reasons()
        .iter()
        .map(|reason| reason.as_str())
        .collect::<Vec<_>>()
        .join(",");
    let reconciliation = &accounting.reconciliation;
    let cache_write = &accounting.token_usage.cache_write;
    let evidence_incomplete = accounting
        .evidence
        .as_ref()
        .is_some_and(|doc| doc.cache_write.incomplete);
    warn!(
        request_id = %request_id,
        model = %model,
        cost_status = accounting.cost.status.as_str(),
        reasons = %reasons,
        pricing_error = accounting.warning.pricing_error().unwrap_or(""),
        cache_exceeds_prompt = reconciliation.cache_exceeds_prompt,
        reasoning_exceeds_completion = reconciliation.reasoning_exceeds_completion,
        configured_duplicate = reconciliation.duplicate.configured_duplicate,
        unknown_duplicates_indeterminate = reconciliation.duplicate.unknown_indeterminate,
        unknown_classes_present = reconciliation.unknown_classes_present,
        quantity_overflow = cache_write.quantity_overflow(),
        observation_count = cache_write.observation_count(),
        retained_evidence_entries = cache_write.evidence_entries().len(),
        accounted_cache_write_tokens = cache_write.accounted_tokens(),
        evidence_incomplete,
        evidence_limit_bytes = CACHE_WRITE_EVIDENCE_MAX_BYTES,
        "request accounting anomaly"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PricingConfig;
    use crate::domain::chat::UsageAccounting;
    use crate::domain::embedding::EmbeddingUsage;
    use crate::domain::ports::{CostError, NanoUsd};
    use crate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
    use crate::domain::usage_accounting::{
        CacheWriteAccumulator, CacheWriteClass, MAX_PRICING_ERROR_BYTES,
    };
    use axum::response::IntoResponse;
    use tracing_test::traced_test;

    fn holder() -> Arc<std::sync::RwLock<PricingDb>> {
        Arc::new(std::sync::RwLock::new(
            PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
                .expect("bundled pricing must load"),
        ))
    }

    /// Counts the "request accounting anomaly" lines among the captured logs and asserts there
    /// is exactly one, then asserts every `needle` appears on that one line.
    ///
    /// AC33/AC33a/AC33b require exactly one WARN per completed request, not merely that the
    /// expected content is present somewhere in the log — a `logs_contain` check alone cannot
    /// tell "one bounded event" from "one of several". Counting lines is what proves it.
    fn assert_exactly_one_anomaly_warning(lines: &[&str], needles: &[&str]) -> Result<(), String> {
        // Counts every WARN-level line, not only ones matching the consolidated event's own
        // message: a stray old-style site restored under a different message must fail this
        // just as loudly as a duplicate of the consolidated one.
        let warn_lines: Vec<&&str> = lines
            .iter()
            .filter(|line| line.contains(" WARN "))
            .collect();
        if warn_lines.len() != 1 {
            return Err(format!(
                "expected exactly one WARN, found {}: {:?}",
                warn_lines.len(),
                warn_lines
            ));
        }
        let line = warn_lines[0];
        if !line.contains("request accounting anomaly") {
            return Err(format!(
                "expected the single WARN to be the consolidated event, got: {line}"
            ));
        }
        for needle in needles {
            if !line.contains(needle) {
                return Err(format!(
                    "expected the single warning to contain {needle:?}, got: {line}"
                ));
            }
        }
        Ok(())
    }

    // --- The one finalization result: status, reconciliation, evidence and warning facts ---

    // --- The status header, and the one warning ---

    /// The fallback literal the two panic-free header paths use must stay the status enum's own
    /// spelling; a drift here would report a status no consumer recognises.
    #[test]
    fn test_header_fallback_matches_the_status_spelling() {
        assert_eq!(
            COST_UNAVAILABLE_HEADER,
            CostStatus::CostUnavailable.as_str()
        );
    }

    /// Every response that carries OxiGate's cost headers carries the status beside them —
    /// chat and embeddings alike, and a request with no cache write at all.
    #[test]
    fn test_cost_status_header_accompanies_every_cost_response() {
        let usage = Usage {
            prompt_tokens: 1_000,
            completion_tokens: 100,
            total_tokens: 1_100,
            ..Default::default()
        };
        let (headers, finalized) = build_cost_headers("gpt-4.1", &usage, holder(), false);
        assert_eq!(
            headers
                .get(CostHeader::COST_STATUS)
                .and_then(|v| v.to_str().ok()),
            Some(CostStatus::Exact.as_str()),
            "a priced chat request with no cache write must still report its status"
        );
        assert_eq!(finalized.cost.status, CostStatus::Exact);

        let embedding_usage = EmbeddingUsage {
            prompt_tokens: 1_000,
            total_tokens: 1_000,
        };
        let (headers, finalized) = build_embedding_cost_headers(
            "text-embedding-3-small",
            &embedding_usage,
            holder(),
            false,
        );
        assert_eq!(
            headers
                .get(CostHeader::COST_STATUS)
                .and_then(|v| v.to_str().ok()),
            Some(finalized.cost.status.as_str()),
            "embeddings report the same status their finalization concluded"
        );
    }

    /// The zero-cost path has no calculation behind it, so it reports `cost-unavailable` rather
    /// than presenting an unestablished zero as a confident one.
    #[test]
    fn test_zero_cost_headers_report_cost_unavailable() {
        let mut resp = (axum::http::StatusCode::BAD_GATEWAY, "error").into_response();
        inject_zero_cost_headers(&mut resp, "gpt-4.1");
        let headers = resp.headers();
        assert_eq!(
            headers
                .get(CostHeader::COST_STATUS)
                .and_then(|v| v.to_str().ok()),
            Some(CostStatus::CostUnavailable.as_str())
        );
    }

    /// An exact, consistent, complete request says nothing at all.
    #[traced_test]
    #[test]
    fn test_a_clean_request_emits_no_warning() {
        let token_usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 100,
            cache_write: cache_write_of(&[("5m", 500)]),
            ..Default::default()
        };
        let (_, finalized) = assemble_cost_headers(
            "claude-sonnet-4-6",
            &token_usage,
            1_000,
            100,
            holder(),
            ReconciliationFacts::default(),
        );
        report_finalized_warning("req-test", "claude-sonnet-4-6", &finalized);
        assert!(!logs_contain("request accounting anomaly"));
    }

    /// One anomalous request produces one bounded event, and no provider member name reaches it.
    #[traced_test]
    #[test]
    fn test_an_anomalous_request_emits_one_bounded_warning_without_raw_keys() {
        let token_usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 100,
            // The second member names no valid duration, so it is an unknown class: fallback
            // rate, and its raw spelling is retained as evidence but must never be logged.
            cache_write: cache_write_of(&[("5m", 500), ("ephemeral_zzz_input_tokens", 100)]),
            ..Default::default()
        };
        let (_, finalized) = assemble_cost_headers(
            "claude-sonnet-4-6",
            &token_usage,
            1_000,
            100,
            holder(),
            ReconciliationFacts::default(),
        );
        assert_eq!(finalized.cost.status, CostStatus::RateFallback);
        report_finalized_warning("req-test", "claude-sonnet-4-6", &finalized);

        assert!(logs_contain("request accounting anomaly"));
        assert!(logs_contain("rate-fallback"));
        assert!(
            !logs_contain("ephemeral_zzz_input_tokens"),
            "a provider member name must never reach the log"
        );
    }

    /// A missing-model request emits the consolidated event and nothing else, and the
    /// reconciliation facts it also carries survive onto that one event.
    ///
    /// The other half of "one WARN, not two" is asserted at the source: the calculator's own
    /// `test_calculate_unknown_reports_the_fact_and_emits_nothing` proves it no longer emits.
    #[traced_test]
    #[test]
    fn test_a_missing_model_emits_one_warning_carrying_coexisting_facts() {
        let token_usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 100,
            cache_write: cache_write_of(&[("5m", 500), ("5m", 200)]),
            ..Default::default()
        };
        let (headers, finalized) = assemble_cost_headers(
            "no-such-model-anywhere",
            &token_usage,
            1_000,
            100,
            holder(),
            ReconciliationFacts {
                cache_exceeds_prompt: true,
                ..Default::default()
            },
        );
        assert_eq!(finalized.cost.status, CostStatus::CostUnavailable);
        assert_eq!(
            headers
                .get(CostHeader::COST_STATUS)
                .and_then(|v| v.to_str().ok()),
            Some(CostStatus::CostUnavailable.as_str())
        );
        report_finalized_warning("req-test", "no-such-model-anywhere", &finalized);

        logs_assert(|lines| {
            assert_exactly_one_anomaly_warning(
                lines,
                &[
                    "model_not_in_pricing_db",
                    "duplicate-ambiguity",
                    "cache-exceeds-prompt",
                ],
            )
        });
    }

    /// A checked-arithmetic failure emits the consolidated event once. The model and the error
    /// reason the cost-error arm used to log are fields on it, and that arm emits nothing.
    #[traced_test]
    #[test]
    fn test_a_cost_error_emits_one_warning_carrying_the_model_and_reason() {
        let json = r#"{"models":{"no-img":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.00001,"output_per_token":0.00003}]}}}"#;
        let token_usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 100,
            image_count: 3,
            ..Default::default()
        };
        let (_, finalized) = assemble_cost_headers(
            "no-img",
            &token_usage,
            1_000,
            100,
            holder_from(json),
            ReconciliationFacts::default(),
        );
        assert_eq!(finalized.cost.status, CostStatus::CostUnavailable);
        assert!(
            finalized.warning.pricing_error().is_some(),
            "the failure reason must reach the finalization result"
        );
        report_finalized_warning("req-test", "no-img", &finalized);

        logs_assert(|lines| assert_exactly_one_anomaly_warning(lines, &["no-img", "image"]));
        assert!(
            !logs_contain("cost calculation failed"),
            "the cost-error arm must no longer emit its own WARN"
        );
    }

    // --- Finalizing a request whose provider reported no usage ---

    /// A priced model with nothing reported is `cost-unavailable` at zero cost, and says why
    /// twice over: the status reason, and the reason that distinguishes *unreported* from
    /// *established as zero*.
    ///
    /// Output-contract coverage only. It is deliberately insensitive to the status force in
    /// [`mark_usage_missing`] — the calculator's no-usable-usage gate already returns
    /// `CostUnavailable` here, so deleting the force leaves this green.
    /// `test_the_missing_usage_force_moves_a_status_that_disagrees` is what protects it.
    #[test]
    fn test_finalize_missing_usage_on_a_priced_model_is_zero_and_unavailable() {
        let finalized = finalize_missing_usage("gpt-4.1", holder(), false);

        assert_eq!(finalized.cost.status, CostStatus::CostUnavailable);
        assert_eq!(finalized.cost.total_cost, NanoUsd::zero());
        assert_eq!(
            finalized.warning.reasons(),
            &[
                WarningReason::CostUnavailable,
                WarningReason::ProviderUsageMissing
            ],
            "each reason exactly once, in the order they are recorded"
        );
    }

    /// A model absent from the pricing DB lands on the same verdict, rather than reporting only
    /// that pricing failed — the request's usage is missing either way.
    ///
    /// Mutation-insensitive to the force for the same reason as the test above: this arm reaches
    /// `CostBreakdown::default()`, whose status is already `CostUnavailable`.
    #[test]
    fn test_finalize_missing_usage_on_an_unpriced_model_is_zero_and_unavailable() {
        let finalized = finalize_missing_usage("no-such-model-anywhere", holder(), false);

        assert_eq!(finalized.cost.status, CostStatus::CostUnavailable);
        assert_eq!(finalized.cost.total_cost, NanoUsd::zero());
        assert!(
            finalized
                .warning
                .reasons()
                .contains(&WarningReason::ProviderUsageMissing)
        );
    }

    /// The force itself: a result that arrived claiming `exact` is moved, not trusted.
    ///
    /// Neither calculator arm produces such a result today, which is exactly why this test
    /// constructs one directly. It is the only test here that fails if the force is deleted.
    #[test]
    fn test_the_missing_usage_force_moves_a_status_that_disagrees() {
        let mut finalized = FinalizedAccounting {
            token_usage: TokenUsage::default(),
            cost: crate::domain::ports::CostBreakdown {
                status: CostStatus::Exact,
                ..Default::default()
            },
            evidence: None,
            reconciliation: ReconciliationFacts::default(),
            warning: WarningFacts::default(),
        };

        mark_usage_missing(&mut finalized);

        assert_eq!(
            finalized.cost.status,
            CostStatus::CostUnavailable,
            "an unreported quantity cannot be priced exactly, whatever the calculator concluded"
        );
        assert_eq!(
            finalized.warning.reasons(),
            &[
                WarningReason::CostUnavailable,
                WarningReason::ProviderUsageMissing
            ]
        );
    }

    /// One missing-usage request produces exactly one WARN, carrying both reason codes and the
    /// request id.
    ///
    /// This is the half of the one-WARN rule that lives at finalization; the compat lane's half —
    /// that it no longer warns about the same fact itself — is asserted in
    /// `providers::openai_compat`.
    ///
    /// `request_id` is asserted here because consolidating the lane-local warnings into this one
    /// event dropped a field the deleted `stream_eof_no_usage` warning carried. The per-request
    /// cost `INFO` line is not a substitute: without a shared key, two concurrent requests on the
    /// same model produce a warning and two cost lines that cannot be joined.
    #[traced_test]
    #[test]
    fn test_a_missing_usage_request_emits_one_warning_naming_both_reasons() {
        let finalized = finalize_missing_usage("gpt-4.1", holder(), false);
        report_finalized_warning("req-correlation-key", "gpt-4.1", &finalized);

        logs_assert(|lines| {
            assert_exactly_one_anomaly_warning(
                lines,
                &[
                    "cost-unavailable",
                    "provider-usage-missing",
                    "req-correlation-key",
                ],
            )
        });
    }

    /// A usage-invariant clamp reports at least `reconciled`, never `exact`, and folds into the
    /// single event instead of warning on its own.
    ///
    /// `USAGE_INVARIANT_VIOLATION_TOTAL` is still incremented from the same detection — the
    /// metric call is untouched — but this crate has no recorder harness to observe a counter,
    /// so the assertion here is on the status, the reason set and the absence of the old event.
    #[traced_test]
    #[test]
    fn test_an_invariant_clamp_folds_into_the_single_warning() {
        let usage = Usage {
            prompt_tokens: 1_000,
            completion_tokens: 100,
            total_tokens: 1_100,
            cache_read_input_tokens: Some(5_000),
            accounting: UsageAccounting {
                cache: CacheAccounting::Inclusive,
                reasoning: ReasoningAccounting::Additive,
            },
            ..Default::default()
        };
        let (_, finalized) = build_cost_headers("gpt-4.1", &usage, holder(), false);
        assert_ne!(finalized.cost.status, CostStatus::Exact);
        report_finalized_warning("req-test", "gpt-4.1", &finalized);

        logs_assert(|lines| assert_exactly_one_anomaly_warning(lines, &["cache-exceeds-prompt"]));
        assert!(
            !logs_contain("billable input clamped to zero"),
            "the invariant site must no longer emit its own WARN"
        );
    }

    /// The retained pricing message is bounded, so one event cannot carry an unbounded string.
    #[test]
    fn test_the_pricing_failure_message_is_bounded() {
        let mut facts = WarningFacts::default();
        facts.set_pricing_error(&"é".repeat(1_000));
        let retained = facts.pricing_error().expect("message retained");
        assert!(retained.len() <= MAX_PRICING_ERROR_BYTES);
        assert!(retained.chars().all(|c| c == 'é'), "trimmed at a boundary");
    }

    /// A holder over a purpose-built pricing DB, for the cases the bundled table cannot express.
    fn holder_from(json: &str) -> Arc<std::sync::RwLock<PricingDb>> {
        Arc::new(std::sync::RwLock::new(
            PricingDb::load(json.as_bytes(), &PricingConfig::default()).expect("pricing must load"),
        ))
    }

    /// Every cost component of an all-zero `cost-unavailable` finalization, so the assertions
    /// below cannot silently miss a dimension that a future component adds.
    fn assert_all_components_zero(cost: &crate::domain::ports::CostBreakdown) {
        assert_eq!(cost.input_cost, NanoUsd::zero(), "input_cost");
        assert_eq!(cost.output_cost, NanoUsd::zero(), "output_cost");
        assert_eq!(cost.cached_input_cost, NanoUsd::zero(), "cached_input_cost");
        assert_eq!(cost.cache_write_cost, NanoUsd::zero(), "cache_write_cost");
        assert_eq!(cost.thinking_cost, NanoUsd::zero(), "thinking_cost");
        assert_eq!(cost.image_cost, NanoUsd::zero(), "image_cost");
        assert_eq!(cost.audio_cost, NanoUsd::zero(), "audio_cost");
        assert_eq!(cost.total_cost, NanoUsd::zero(), "total_cost");
    }

    /// Builds accounting state through the accumulator, the only path that produces one.
    fn cache_write_of(
        pairs: &[(&str, u64)],
    ) -> crate::domain::usage_accounting::CacheWriteAccounting {
        let registry = crate::domain::usage_accounting::CacheWriteClassRegistry::from_classes(
            pairs
                .iter()
                .filter_map(|(k, _)| CacheWriteClass::canonicalize(k)),
        )
        .expect("registry");
        let mut acc = CacheWriteAccumulator::new(registry);
        for (raw_key, tokens) in pairs {
            acc.observe_detail(raw_key, CacheWriteClass::canonicalize(raw_key), *tokens);
        }
        acc.finish()
    }

    /// A clean request with configured classes and no contradictions finalizes as `exact` and
    /// raises no warning at all.
    #[test]
    fn test_finalization_is_exact_and_silent_for_a_clean_request() {
        let token_usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 100,
            cache_write: cache_write_of(&[("5m", 500)]),
            ..Default::default()
        };
        let (_, finalized) = assemble_cost_headers(
            "claude-sonnet-4-6",
            &token_usage,
            1_000,
            100,
            holder(),
            ReconciliationFacts::default(),
        );
        assert_eq!(finalized.cost.status, CostStatus::Exact);
        assert!(
            !finalized.warning.should_warn(),
            "a clean request must raise no warning reason"
        );
    }

    /// The two usage-invariant clamps are conservative *quantity* policies, so they force at
    /// least `reconciled` and each contributes its own warning reason.
    #[test]
    fn test_clamps_force_reconciled_and_carry_their_warning_reasons() {
        let token_usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 100,
            cache_write: cache_write_of(&[("5m", 500)]),
            ..Default::default()
        };
        let (_, finalized) = assemble_cost_headers(
            "claude-sonnet-4-6",
            &token_usage,
            1_000,
            100,
            holder(),
            ReconciliationFacts {
                cache_exceeds_prompt: true,
                reasoning_exceeds_completion: true,
                ..Default::default()
            },
        );
        assert_eq!(finalized.cost.status, CostStatus::Reconciled);
        assert!(
            finalized
                .warning
                .reasons()
                .contains(&WarningReason::CacheExceedsPrompt)
        );
        assert!(
            finalized
                .warning
                .reasons()
                .contains(&WarningReason::ReasoningExceedsCompletion)
        );
        assert!(
            finalized.reconciliation.cache_exceeds_prompt
                && finalized.reconciliation.reasoning_exceeds_completion,
            "the clamps must survive onto the result as reconciliation facts"
        );
    }

    /// A rate fallback outranks a reconciliation clamp: the worse status wins, so a fallback rate
    /// can never be hidden behind a merely-reconciled request.
    #[test]
    fn test_rate_fallback_outranks_a_reconciliation_clamp() {
        // "unknownclass" is no valid duration, so it lands in the unknown bucket and prices at
        // the tier-local fallback rate.
        let registry = crate::domain::usage_accounting::CacheWriteClassRegistry::from_classes(
            ["5m"]
                .iter()
                .filter_map(|k| CacheWriteClass::canonicalize(k)),
        )
        .expect("registry");
        let mut acc = CacheWriteAccumulator::new(registry);
        acc.observe_detail("cache_creation_unknownclass_tokens", None, 400);
        let token_usage = TokenUsage {
            input_tokens: 1_000,
            cache_write: acc.finish(),
            ..Default::default()
        };
        let (_, finalized) = assemble_cost_headers(
            "claude-sonnet-4-6",
            &token_usage,
            1_000,
            0,
            holder(),
            ReconciliationFacts {
                cache_exceeds_prompt: true,
                ..Default::default()
            },
        );
        assert_eq!(finalized.cost.status, CostStatus::RateFallback);
        assert!(
            finalized.reconciliation.unknown_classes_present,
            "the unknown class must be recorded as a reconciliation fact"
        );
        assert!(
            finalized
                .warning
                .reasons()
                .contains(&WarningReason::RateFallback)
        );
    }

    /// A real `CostError::Pricing` raised inside `calculate` — here a priced model reporting
    /// image units it has no rate for — must finalize as an all-zero `cost-unavailable` result.
    ///
    /// This is the phase's common error mapping, and it is the *error* arm specifically: a
    /// partly-priced breakdown must not survive, so every component is asserted zero rather
    /// than only the total. Note that an unknown model is a different path — it returns
    /// `Ok(CostBreakdown::zero())`, and is covered separately below.
    #[test]
    fn test_cost_error_finalizes_as_all_zero_cost_unavailable() {
        // Priced input and output, but no `image_per_unit`: the positive image quantity below
        // cannot be defended at any rate, so `calculate` fails the whole request.
        let json = r#"{"models":{"no-img":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.00001,"output_per_token":0.00003}]}}}"#;
        let holder = holder_from(json);

        // Precondition: this really is the `Err` arm, not a zero that happens to look like one.
        let err = BundledCostCalculator::new(Arc::clone(&holder))
            .calculate(
                "no-img",
                &TokenUsage {
                    input_tokens: 1_000,
                    output_tokens: 100,
                    image_count: 3,
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, CostError::Pricing(_)), "{err:?}");

        let token_usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 100,
            image_count: 3,
            ..Default::default()
        };
        let (_, finalized) = assemble_cost_headers(
            "no-img",
            &token_usage,
            1_000,
            100,
            holder,
            ReconciliationFacts::default(),
        );
        assert_eq!(finalized.cost.status, CostStatus::CostUnavailable);
        assert_all_components_zero(&finalized.cost);
        assert!(
            finalized
                .warning
                .reasons()
                .contains(&WarningReason::CostUnavailable)
        );
    }

    /// A model absent from the pricing DB reaches the same request-wide failure path as any
    /// other unusable pricing, and finalizes all-zero `cost-unavailable` — a zero charge nobody
    /// established is not a confident zero.
    #[test]
    fn test_unknown_model_finalizes_as_zero_cost_unavailable() {
        let token_usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 100,
            ..Default::default()
        };
        let (_, finalized) = assemble_cost_headers(
            "no-such-model-anywhere",
            &token_usage,
            1_000,
            100,
            holder(),
            ReconciliationFacts::default(),
        );
        assert_eq!(finalized.cost.status, CostStatus::CostUnavailable);
        assert_all_components_zero(&finalized.cost);
        assert!(
            finalized
                .warning
                .reasons()
                .contains(&WarningReason::CostUnavailable)
        );
    }

    /// A saturated cache-write quantity must travel the same finalization path: the partition
    /// gate refuses to price, the result is all-zero `cost-unavailable`, and the overflow itself
    /// survives into the evidence document so the anomaly is still legible after the fact.
    ///
    /// The zero-rate fixture is deliberate — every priced component is zero however large the
    /// quantity, so no downstream monetary check can fire and the partition gate is the only
    /// thing that can reject this request. That keeps the test discriminating under mutation.
    #[test]
    fn test_overflowed_partition_finalizes_as_cost_unavailable_and_survives_in_evidence() {
        let json = r#"{"models":{"cw-free":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.0,"output_per_token":0.0,
              "cache_write_multipliers":{"5m":1.0,"1h":1.0}}]}}}"#;
        let holder = holder_from(json);

        let registry = crate::domain::usage_accounting::CacheWriteClassRegistry::from_classes(
            ["5m", "1h"]
                .iter()
                .filter_map(|k| CacheWriteClass::canonicalize(k)),
        )
        .expect("registry");
        let mut acc = CacheWriteAccumulator::new(registry);
        acc.observe_detail("5m", CacheWriteClass::canonicalize("5m"), u64::MAX);
        acc.observe_detail("1h", CacheWriteClass::canonicalize("1h"), 1);
        let cache_write = acc.finish();
        assert!(cache_write.quantity_overflow(), "precondition: saturated");
        assert!(
            !cache_write.partition_is_exact(),
            "precondition: does not partition exactly"
        );

        let token_usage = TokenUsage {
            cache_write,
            ..Default::default()
        };
        let (_, finalized) = assemble_cost_headers(
            "cw-free",
            &token_usage,
            0,
            0,
            holder,
            ReconciliationFacts::default(),
        );

        assert_eq!(finalized.cost.status, CostStatus::CostUnavailable);
        assert_all_components_zero(&finalized.cost);
        let evidence = finalized.evidence.expect("evidence must be produced");
        assert!(
            evidence.cache_write.quantity_overflow,
            "the saturation must survive into the persisted evidence document"
        );
        assert_eq!(
            evidence.cache_write.component_cost_nano_usd, 0,
            "the evidence must not report a cache-write charge the request refused to make"
        );
    }

    /// Evidence completeness is resolved at the same instant as the cost, so a truncated document
    /// carries its warning reason on the very result that describes it.
    #[test]
    fn test_incomplete_evidence_is_resolved_with_the_cost() {
        let registry = crate::domain::usage_accounting::CacheWriteClassRegistry::from_classes(
            ["5m"]
                .iter()
                .filter_map(|k| CacheWriteClass::canonicalize(k)),
        )
        .expect("registry");
        let mut acc = CacheWriteAccumulator::new(registry);
        // Far more observations than the retention cap, so the document cannot describe them all.
        for i in 0..512 {
            acc.observe_detail(
                &format!("cache_creation_{i}_tokens"),
                CacheWriteClass::canonicalize("5m"),
                1,
            );
        }
        let token_usage = TokenUsage {
            input_tokens: 1_000,
            cache_write: acc.finish(),
            ..Default::default()
        };
        let (_, finalized) = assemble_cost_headers(
            "claude-sonnet-4-6",
            &token_usage,
            1_000,
            0,
            holder(),
            ReconciliationFacts::default(),
        );
        let evidence = finalized.evidence.expect("evidence must be produced");
        assert!(
            evidence.cache_write.incomplete,
            "a document that dropped observations must be flagged incomplete"
        );
        assert!(
            finalized
                .warning
                .reasons()
                .contains(&WarningReason::IncompleteEvidence)
        );
    }

    /// AC27 (headers half): whether the retained evidence document is `incomplete` must never
    /// change the request-cost or cost-status headers, or the finalized cost/status those headers
    /// are built from — evidence completeness is a fact about the document, not an input to
    /// pricing. Two requests here carry identical billable quantities (one `5m` observation of
    /// 1,000 tokens), identical rates (same model, same pricing DB) and identical reconciliation
    /// state (a single observation of a configured class, no aggregate, no duplicates) — the only
    /// difference is the byte length of the one observation's raw key, which is long enough on one
    /// side to force key truncation and therefore `incomplete: true`.
    ///
    /// Negative control: if a future change let cost or status read `incomplete` (a second
    /// authority this module must never introduce), the header and finalized-result assertions
    /// below would diverge and fail.
    #[test]
    fn test_evidence_completeness_does_not_change_cost_or_status_headers() {
        let single_observation = |raw_key: &str| {
            let registry = crate::domain::usage_accounting::CacheWriteClassRegistry::from_classes(
                ["5m"]
                    .iter()
                    .filter_map(|k| CacheWriteClass::canonicalize(k)),
            )
            .expect("registry");
            let mut acc = CacheWriteAccumulator::new(registry);
            acc.observe_detail(raw_key, CacheWriteClass::canonicalize("5m"), 1_000);
            acc.finish()
        };

        // Short key: fits well within MAX_RAW_KEY_BYTES (128), so nothing is truncated.
        let complete_usage = TokenUsage {
            input_tokens: 1_000,
            cache_write: single_observation("cache_creation_5m_tokens"),
            ..Default::default()
        };
        // Long key: exceeds MAX_RAW_KEY_BYTES, so the copied evidence entry is truncated and the
        // document is marked incomplete — the one and only difference from `complete_usage`.
        let incomplete_usage = TokenUsage {
            input_tokens: 1_000,
            cache_write: single_observation(&"x".repeat(200)),
            ..Default::default()
        };

        let (complete_headers, complete_finalized) = assemble_cost_headers(
            "claude-sonnet-4-6",
            &complete_usage,
            1_000,
            0,
            holder(),
            ReconciliationFacts::default(),
        );
        let (incomplete_headers, incomplete_finalized) = assemble_cost_headers(
            "claude-sonnet-4-6",
            &incomplete_usage,
            1_000,
            0,
            holder(),
            ReconciliationFacts::default(),
        );

        // Sanity: the fixtures actually differ in evidence completeness, and only in that —
        // otherwise the assertions below would pass vacuously.
        assert!(
            !complete_finalized
                .evidence
                .as_ref()
                .expect("evidence")
                .cache_write
                .incomplete
        );
        assert!(
            incomplete_finalized
                .evidence
                .as_ref()
                .expect("evidence")
                .cache_write
                .incomplete
        );
        assert_eq!(
            complete_finalized.reconciliation, incomplete_finalized.reconciliation,
            "a single observation of one configured class must reconcile identically regardless \
             of raw-key length"
        );

        assert_eq!(
            complete_headers.get(CostHeader::REQUEST_COST),
            incomplete_headers.get(CostHeader::REQUEST_COST),
            "request-cost header must not depend on evidence completeness"
        );
        assert_eq!(
            complete_headers.get(CostHeader::COST_STATUS),
            incomplete_headers.get(CostHeader::COST_STATUS),
            "cost-status header must not depend on evidence completeness"
        );
        assert_eq!(
            complete_finalized.cost.total_cost, incomplete_finalized.cost.total_cost,
            "finalized cost must not depend on evidence completeness"
        );
        assert_eq!(
            complete_finalized.cost.status, incomplete_finalized.cost.status,
            "finalized cost_status must not depend on evidence completeness"
        );
    }

    /// A cache-inclusive contract reporting more cache-read tokens than prompt tokens is
    /// flagged, the reported cache bucket is preserved, and the billable input clamps to zero
    /// rather than wrapping to a near-`u64::MAX` charge.
    #[test]
    fn test_cache_read_exceeding_prompt_is_flagged_and_clamps_input_to_zero() {
        let usage = Usage {
            prompt_tokens: 1_000,
            completion_tokens: 100,
            total_tokens: 1_100,
            cache_read_input_tokens: Some(5_000),
            accounting: crate::domain::chat::UsageAccounting {
                cache: CacheAccounting::Inclusive,
                reasoning: ReasoningAccounting::Additive,
            },
            ..Default::default()
        };

        let flags = detect_usage_invariants(&usage, 5_000, 0, 0);
        assert!(flags.cache_exceeds_prompt);
        assert!(!flags.reasoning_exceeds_completion);

        let (_, finalized) = build_cost_headers("gpt-4.1", &usage, holder(), false);
        let token_usage = &finalized.token_usage;
        assert_eq!(token_usage.input_tokens, 0, "billable input must clamp");
        assert_eq!(
            token_usage.cache_read_input_tokens, 5_000,
            "the reported cache bucket must be preserved as reported"
        );
    }

    /// A contract that reports reasoning inside the completion total, reporting more
    /// reasoning than completion, is flagged and the standard output charge clamps to zero.
    #[test]
    fn test_reasoning_exceeding_completion_is_flagged_and_clamps_output_to_zero() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 500,
            total_tokens: 600,
            completion_tokens_details: Some(crate::domain::chat::CompletionTokensDetails {
                reasoning_tokens: Some(900),
            }),
            accounting: crate::domain::chat::UsageAccounting {
                cache: CacheAccounting::Inclusive,
                reasoning: ReasoningAccounting::IncludedInOutput,
            },
            ..Default::default()
        };

        let flags = detect_usage_invariants(&usage, 0, 0, 900);
        assert!(flags.reasoning_exceeds_completion);
        assert!(!flags.cache_exceeds_prompt);

        let (_, finalized) = build_cost_headers("gpt-4.1", &usage, holder(), false);
        let token_usage = &finalized.token_usage;
        assert_eq!(token_usage.standard_output_tokens(), 0);
        assert_eq!(
            token_usage.output_tokens, 500,
            "output_tokens keeps meaning the provider-reported completion total"
        );
        assert_eq!(
            token_usage.thinking_tokens, 900,
            "reasoning kept as reported"
        );
    }

    /// A contract that reports its buckets disjointly cannot violate a subset relation.
    /// Large cache and reasoning buckets beside a small prompt and completion are normal there,
    /// and a false positive would fire on every such request.
    #[test]
    fn test_additive_contract_with_large_buckets_is_not_a_violation() {
        let usage = Usage {
            prompt_tokens: 5_000,
            completion_tokens: 500,
            total_tokens: 5_500,
            cache_read_input_tokens: Some(100_000),
            cache_write: cache_write_of(&[("5m", 10_000), ("1h", 10_000)]),
            completion_tokens_details: Some(crate::domain::chat::CompletionTokensDetails {
                reasoning_tokens: Some(9_000),
            }),
            accounting: crate::domain::chat::UsageAccounting {
                cache: CacheAccounting::Additive,
                reasoning: ReasoningAccounting::Additive,
            },
            ..Default::default()
        };

        assert_eq!(
            detect_usage_invariants(&usage, 100_000, 20_000, 9_000),
            UsageInvariantFlags::default(),
            "an additive contract cannot contradict a subset relation"
        );
    }

    /// The default accounting leaves the standard output charge equal to the reported completion
    /// total, so introducing the reasoning axis moves no billed amount on its own.
    #[test]
    fn test_default_accounting_leaves_standard_output_equal_to_reported_total() {
        let usage = Usage {
            prompt_tokens: 1_000,
            completion_tokens: 500,
            total_tokens: 1_500,
            completion_tokens_details: Some(crate::domain::chat::CompletionTokensDetails {
                reasoning_tokens: Some(400),
            }),
            ..Default::default()
        };

        let (_, finalized) = build_cost_headers("gpt-4.1", &usage, holder(), false);
        let token_usage = &finalized.token_usage;
        assert_eq!(token_usage.standard_output_tokens(), 500);
        assert_eq!(token_usage.thinking_tokens, 400);
    }

    /// The reasoning carve-out is a pricing concern only. `output_tokens` keeps meaning the
    /// provider-reported completion total, so the persisted `completion_tokens` column still
    /// records what the provider called completion — carving reasoning out of it would report a
    /// decrease in usage as a side effect of a cost correction.
    #[test]
    fn test_reasoning_carve_out_does_not_change_the_persisted_completion_total() {
        use crate::domain::spend::SpendRecord;

        let usage = Usage {
            prompt_tokens: 1_000,
            completion_tokens: 1_000,
            total_tokens: 2_000,
            completion_tokens_details: Some(crate::domain::chat::CompletionTokensDetails {
                reasoning_tokens: Some(800),
            }),
            accounting: crate::domain::chat::UsageAccounting {
                cache: CacheAccounting::Inclusive,
                reasoning: ReasoningAccounting::IncludedInOutput,
            },
            ..Default::default()
        };

        let (_, finalized) = build_cost_headers("gpt-4.1", &usage, holder(), false);
        let token_usage = &finalized.token_usage;
        assert_eq!(
            token_usage.standard_output_tokens(),
            200,
            "pricing sees the reasoning carved out"
        );

        let identity = crate::domain::auth::RequestIdentity {
            id: "key-1".into(),
            org_id: "acme".into(),
            label: None,
            tags: std::collections::HashMap::new(),
        };
        let record = SpendRecord::build(&identity, "gpt-4.1", "openai", &finalized, 1);
        assert_eq!(
            record.completion_tokens, 1_000,
            "persistence sees the provider-reported completion total"
        );
        assert_eq!(record.thinking_tokens, 800);
    }

    // -----------------------------------------------------------------------
    // Double-charge regressions.
    //
    // Oracles are exact integer nano-USD and assert every component, not only the total, so a
    // compensating pair of errors cannot pass. Each imports the adapter's own declaration rather
    // than restating its value, so a wrong constant fails here too.
    // -----------------------------------------------------------------------

    /// Reasoning tokens are a breakdown of OpenAI's completion total, so charging the full
    /// completion total beside them billed the reasoning subset twice.
    ///
    /// `gpt-5.6-sol`, completion 1,000 of which 800 reasoning, output rate $30/Mtok.
    /// Previously 54_000_000 nano-USD; 1.80x the 30_000_000 owed.
    #[test]
    fn test_openai_reasoning_double_charge_regression() {
        use crate::providers::openai::utils::OPENAI_ACCOUNTING;

        let usage = Usage {
            prompt_tokens: 0,
            completion_tokens: 1_000,
            total_tokens: 1_000,
            completion_tokens_details: Some(crate::domain::chat::CompletionTokensDetails {
                reasoning_tokens: Some(800),
            }),
            accounting: OPENAI_ACCOUNTING,
            ..Default::default()
        };

        let (_, finalized) = build_cost_headers("gpt-5.6-sol", &usage, holder(), false);
        let (cost, token_usage) = (&finalized.cost, &finalized.token_usage);

        assert_eq!(token_usage.standard_output_tokens(), 200);
        assert_eq!(cost.output_cost, NanoUsd(6_000_000));
        assert_eq!(cost.thinking_cost, NanoUsd(24_000_000));
        assert_eq!(cost.total_cost, NanoUsd(30_000_000));
    }

    /// The reasoning axis is per contract, not global.
    ///
    /// Identical raw numbers priced under a contract that reports reasoning inside the completion
    /// total and one that reports it beside the total produce different — and individually
    /// correct — costs. Gemini is the second kind: `totalTokenCount` is documented as
    /// "prompt + thoughts + response candidates", so its thoughts sit outside
    /// `candidatesTokenCount` and carving them out would create an *under*charge.
    ///
    /// This is the guard against "finishing the job" by flipping Gemini too.
    #[test]
    fn test_reasoning_axis_is_per_contract_not_global() {
        use crate::providers::openai::utils::OPENAI_ACCOUNTING;

        let raw = |accounting| Usage {
            prompt_tokens: 0,
            completion_tokens: 1_000,
            total_tokens: 1_000,
            completion_tokens_details: Some(crate::domain::chat::CompletionTokensDetails {
                reasoning_tokens: Some(800),
            }),
            accounting,
            ..Default::default()
        };

        let included = build_cost_headers("gpt-5.6-sol", &raw(OPENAI_ACCOUNTING), holder(), false)
            .1
            .cost;
        let additive = build_cost_headers(
            "gpt-5.6-sol",
            &raw(UsageAccounting {
                reasoning: ReasoningAccounting::Additive,
                ..OPENAI_ACCOUNTING
            }),
            holder(),
            false,
        )
        .1
        .cost;

        assert_eq!(included.total_cost, NanoUsd(30_000_000));
        assert_eq!(additive.total_cost, NanoUsd(54_000_000));
        assert!(included.total_cost < additive.total_cost);
    }

    /// A generic OpenAI-compatible backend is deliberately left where it was.
    ///
    /// The pair is the assertion: native OpenAI drops on these numbers while an unconfigured
    /// compat instance does not. Either half alone would pass under a design that moved both, and
    /// the reasoning double-charge therefore persists on compat backends as a known, documented
    /// cost until a captured payload justifies declaring one.
    #[test]
    fn test_generic_compat_cost_does_not_move_while_native_openai_does() {
        use crate::providers::openai::utils::{COMPAT_DEFAULT_ACCOUNTING, OPENAI_ACCOUNTING};

        let raw = |accounting| Usage {
            prompt_tokens: 0,
            completion_tokens: 1_000,
            total_tokens: 1_000,
            completion_tokens_details: Some(crate::domain::chat::CompletionTokensDetails {
                reasoning_tokens: Some(800),
            }),
            accounting,
            ..Default::default()
        };

        let compat = build_cost_headers(
            "gpt-5.6-sol",
            &raw(COMPAT_DEFAULT_ACCOUNTING),
            holder(),
            false,
        )
        .1
        .cost;
        let native = build_cost_headers("gpt-5.6-sol", &raw(OPENAI_ACCOUNTING), holder(), false)
            .1
            .cost;

        assert_eq!(
            compat.total_cost,
            NanoUsd(54_000_000),
            "unchanged from the pre-correction value"
        );
        assert_eq!(native.total_cost, NanoUsd(30_000_000));
    }

    /// A cache-inclusive contract counts written tokens inside `prompt_tokens`, so charging the
    /// prompt whole *and* the cache-write class beside it billed those tokens twice.
    ///
    /// `gpt-5.6-terra` base tier: input $2/Mtok, output $12/Mtok, cache read 0.1x,
    /// `30m` cache write 1.25x. A 10,000-token prompt containing 2,000 read and 1,000 written:
    ///
    /// | Component | Quantity | Rate | Nano-USD |
    /// |---|---|---|---|
    /// | plain input | 10,000 − 2,000 − 1,000 = 7,000 | 2,000 | 14,000,000 |
    /// | output | 500 | 12,000 | 6,000,000 |
    /// | cache read | 2,000 | 2,000 × 0.1 | 400,000 |
    /// | cache write | 1,000 | 2,000 × 1.25 | 2,500,000 |
    /// | **total** | | | **22,900,000** |
    ///
    /// Without the carve-out the written tokens stay in the plain bucket: input_cost 16,000,000
    /// and a 24,900,000 total — 2.25x on the written portion instead of the documented 1.25x.
    #[test]
    fn test_inclusive_cache_write_is_billed_once_at_the_class_rate() {
        use crate::providers::openai::utils::OPENAI_ACCOUNTING;

        let usage = Usage {
            prompt_tokens: 10_000,
            completion_tokens: 500,
            total_tokens: 10_500,
            cache_read_input_tokens: Some(2_000),
            cache_write: cache_write_of(&[("30m", 1_000)]),
            accounting: OPENAI_ACCOUNTING,
            ..Default::default()
        };

        let (_, finalized) = build_cost_headers("gpt-5.6-terra", &usage, holder(), false);
        let (cost, token_usage) = (&finalized.cost, &finalized.token_usage);

        assert_eq!(token_usage.input_tokens, 7_000, "both buckets carved out");
        assert_eq!(cost.input_cost, NanoUsd(14_000_000));
        assert_eq!(cost.output_cost, NanoUsd(6_000_000));
        assert_eq!(cost.cached_input_cost, NanoUsd(400_000));
        assert_eq!(cost.cache_write_cost, NanoUsd(2_500_000));
        assert_eq!(cost.total_cost, NanoUsd(22_900_000));
        assert_eq!(
            finalized.cost.status,
            CostStatus::Exact,
            "the tier prices the 30m class, so nothing falls back"
        );
    }

    /// The tier comparator must see each prompt token exactly once.
    ///
    /// `context_input_tokens()` re-adds both cache buckets to the plain input bucket, so under a
    /// cache-inclusive contract it reconstructs the provider's own `prompt_tokens` — but only if
    /// the carve-out took the written tokens out first. This fixture straddles `gpt-5.6-terra`'s
    /// 272,001 threshold: a 272,000-token prompt with 1,000 of them written stays on the base
    /// tier, where a double-counted comparator would see 273,000 and reprice the whole request at
    /// the long-context rate (input $4/Mtok rather than $2/Mtok).
    ///
    /// Base tier: 271,000 × 2,000 = 542,000,000 input, 1,000 × 2,000 × 1.25 = 2,500,000 write.
    #[test]
    fn test_inclusive_comparator_counts_each_prompt_token_exactly_once() {
        use crate::providers::openai::utils::OPENAI_ACCOUNTING;

        let usage = Usage {
            prompt_tokens: 272_000,
            completion_tokens: 0,
            total_tokens: 272_000,
            cache_write: cache_write_of(&[("30m", 1_000)]),
            accounting: OPENAI_ACCOUNTING,
            ..Default::default()
        };

        let (_, finalized) = build_cost_headers("gpt-5.6-terra", &usage, holder(), false);
        let (cost, token_usage) = (&finalized.cost, &finalized.token_usage);

        assert_eq!(
            token_usage.context_input_tokens(),
            272_000,
            "the comparator reconstructs the provider-reported prompt total"
        );
        assert_eq!(token_usage.input_tokens, 271_000);
        assert_eq!(cost.input_cost, NanoUsd(542_000_000), "base tier rate");
        assert_eq!(cost.cache_write_cost, NanoUsd(2_500_000));
        assert_eq!(cost.total_cost, NanoUsd(544_500_000));
    }

    /// An `Inclusive` contract reporting more cached tokens than prompt tokens clamps the plain
    /// input bucket to zero, and the clamp must never be silent.
    ///
    /// Cache *write* tokens now leave that bucket too, so the invariant has to compare the sum of
    /// both cache buckets against the prompt: 600 read plus 500 written against a 1,000-token
    /// prompt is a contradiction that a read-only comparison does not see.
    #[test]
    fn test_inclusive_cache_buckets_exceeding_the_prompt_raise_the_clamp() {
        use crate::providers::openai::utils::OPENAI_ACCOUNTING;

        let usage = Usage {
            prompt_tokens: 1_000,
            completion_tokens: 0,
            total_tokens: 1_000,
            cache_read_input_tokens: Some(600),
            cache_write: cache_write_of(&[("30m", 500)]),
            accounting: OPENAI_ACCOUNTING,
            ..Default::default()
        };

        let (_, finalized) = build_cost_headers("gpt-5.6-terra", &usage, holder(), false);

        assert_eq!(
            finalized.token_usage.input_tokens, 0,
            "the plain bucket clamps rather than wrapping"
        );
        assert_eq!(finalized.cost.input_cost, NanoUsd::zero());
        assert!(
            finalized.reconciliation.cache_exceeds_prompt,
            "the clamp must be reported, not silent"
        );
        assert_eq!(finalized.cost.status, CostStatus::Reconciled);
        assert!(
            finalized
                .warning
                .reasons()
                .contains(&WarningReason::CacheExceedsPrompt)
        );
    }

    /// Both operands of the widened invariant are provider-controlled `u64`s, and
    /// `accounted_tokens()` itself saturates. A wrapping sum would read as *below* `prompt_tokens`
    /// and suppress the very clamp it should raise, so overflow is treated as a violation.
    ///
    /// The fixture discriminates against both wrong implementations: `cache_read` alone does not
    /// exceed the prompt, and the wrapped sum (399) is far below it.
    #[test]
    fn test_the_widened_cache_invariant_treats_an_overflowing_sum_as_a_violation() {
        use crate::providers::openai::utils::OPENAI_ACCOUNTING;

        let cache_read = u64::MAX - 100;
        let cache_write = cache_write_of(&[("30m", 500)]);
        assert_eq!(cache_write.accounted_tokens(), 500, "precondition");

        let usage = Usage {
            prompt_tokens: u64::MAX,
            cache_write,
            accounting: OPENAI_ACCOUNTING,
            ..Default::default()
        };

        assert!(
            cache_read <= usage.prompt_tokens,
            "precondition: the read bucket alone does not violate the invariant"
        );
        assert!(
            detect_usage_invariants(&usage, cache_read, 500, 0).cache_exceeds_prompt,
            "an unrepresentable cached total is a violation, not a pass"
        );
    }

    /// Characterization: every `Inclusive` request on `master` carries an empty cache-write
    /// accumulator, so the carve-out is a no-op until a lane starts populating one.
    ///
    /// Same `gpt-5.6-terra` fixture as the oracle above with the written tokens removed:
    /// 8,000 × 2,000 input, 500 × 12,000 output, 2,000 × 2,000 × 0.1 cache read.
    ///
    /// The persisted row is asserted beside the cost, not assumed from it. `SpendRecord`'s
    /// `prompt_tokens` column is `token_usage.input_tokens` — the one field the carve-out moves —
    /// so a regression would reach persistence whether or not the totals still matched.
    #[test]
    fn test_an_inclusive_request_with_no_cache_write_is_unchanged() {
        use crate::domain::spend::SpendRecord;
        use crate::providers::openai::utils::OPENAI_ACCOUNTING;

        let usage = Usage {
            prompt_tokens: 10_000,
            completion_tokens: 500,
            total_tokens: 10_500,
            cache_read_input_tokens: Some(2_000),
            accounting: OPENAI_ACCOUNTING,
            ..Default::default()
        };

        let (_, finalized) = build_cost_headers("gpt-5.6-terra", &usage, holder(), false);
        let (cost, token_usage) = (&finalized.cost, &finalized.token_usage);

        assert_eq!(token_usage.input_tokens, 8_000);
        assert_eq!(cost.input_cost, NanoUsd(16_000_000));
        assert_eq!(cost.output_cost, NanoUsd(6_000_000));
        assert_eq!(cost.cached_input_cost, NanoUsd(400_000));
        assert_eq!(cost.cache_write_cost, NanoUsd::zero());
        assert_eq!(cost.total_cost, NanoUsd(22_400_000));
        assert_eq!(finalized.cost.status, CostStatus::Exact);
        assert!(!finalized.reconciliation.cache_exceeds_prompt);

        let identity = crate::domain::auth::RequestIdentity {
            id: "key-1".into(),
            org_id: "acme".into(),
            label: None,
            tags: std::collections::HashMap::new(),
        };
        let record = SpendRecord::build(&identity, "gpt-5.6-terra", "openai", &finalized, 1);
        assert_eq!(record.prompt_tokens, 8_000, "the persisted billable input");
        assert_eq!(record.completion_tokens, 500);
        assert_eq!(record.cache_read_tokens, 2_000);
        assert_eq!(record.thinking_tokens, 0);
        assert_eq!(record.cost_nano_usd, NanoUsd(22_400_000));
        assert_eq!(record.cost_status, CostStatus::Exact);
        assert!(
            record.usage_evidence.is_none(),
            "no cache write, so nothing to evidence"
        );
    }

    /// The carve-out belongs to the cache-inclusive contract alone.
    ///
    /// An `Additive` contract reports `prompt_tokens` as the plain portion already, so subtracting
    /// either cache bucket from it would undercharge. `claude-sonnet-4-6`: 10,000 × 3,000 input on
    /// the full prompt, 500 × 15,000 output, 2,000 × 3,000 × 0.1 read, 1,000 × 3,000 × 1.25 write.
    ///
    /// The accounting constant and the priced model are deliberately from different lanes: the
    /// axis under test is the declared contract, and importing a real `Additive` declaration
    /// rather than restating one is what makes a wrong constant fail here too.
    #[test]
    fn test_the_carve_out_does_not_cross_into_the_additive_contract() {
        use crate::providers::bedrock::translate::BEDROCK_ACCOUNTING;

        let usage = Usage {
            prompt_tokens: 10_000,
            completion_tokens: 500,
            total_tokens: 10_500,
            cache_read_input_tokens: Some(2_000),
            cache_write: cache_write_of(&[("5m", 1_000)]),
            accounting: BEDROCK_ACCOUNTING,
            ..Default::default()
        };

        let (_, finalized) = build_cost_headers("claude-sonnet-4-6", &usage, holder(), false);
        let (cost, token_usage) = (&finalized.cost, &finalized.token_usage);

        assert_eq!(
            token_usage.input_tokens, 10_000,
            "an additive prompt total is already the plain portion"
        );
        assert_eq!(token_usage.context_input_tokens(), 13_000);
        assert_eq!(cost.input_cost, NanoUsd(30_000_000));
        assert_eq!(cost.output_cost, NanoUsd(7_500_000));
        assert_eq!(cost.cached_input_cost, NanoUsd(600_000));
        assert_eq!(cost.cache_write_cost, NanoUsd(3_750_000));
        assert_eq!(cost.total_cost, NanoUsd(41_850_000));
        assert!(!finalized.reconciliation.cache_exceeds_prompt);
    }

    /// Verifies zero-cost headers on error path.
    #[test]
    fn test_inject_zero_cost_headers() {
        let mut resp = (axum::http::StatusCode::BAD_GATEWAY, "error").into_response();
        inject_zero_cost_headers(&mut resp, "gpt-4");
        let headers = resp.headers();
        assert_eq!(
            headers
                .get(CostHeader::REQUEST_COST)
                .and_then(|v| v.to_str().ok()),
            Some("0.000000")
        );
        assert_eq!(
            headers
                .get(CostHeader::INPUT_TOKENS)
                .and_then(|v| v.to_str().ok()),
            Some("0")
        );
        assert_eq!(
            headers
                .get(CostHeader::OUTPUT_TOKENS)
                .and_then(|v| v.to_str().ok()),
            Some("0")
        );
        assert_eq!(
            headers
                .get(CostHeader::MODEL_USED)
                .and_then(|v| v.to_str().ok()),
            Some("gpt-4")
        );
    }

    /// Invalid model characters (CR/LF/NUL) must fall back to "unknown", not panic.
    #[test]
    fn test_inject_zero_cost_headers_sanitizes_invalid_model() {
        let mut resp = (axum::http::StatusCode::BAD_GATEWAY, "error").into_response();
        inject_zero_cost_headers(&mut resp, "gpt-4\n\r\x00");
        assert_eq!(
            resp.headers()
                .get(CostHeader::MODEL_USED)
                .and_then(|v| v.to_str().ok()),
            Some("unknown"),
            "invalid model chars must fallback to unknown"
        );
    }

    #[test]
    fn test_build_cost_headers_non_zero_usage() {
        let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing must load");
        let holder = Arc::new(std::sync::RwLock::new(pricing_db));
        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
            ..Default::default()
        };
        let (headers, _) = build_cost_headers("gpt-4.1", &usage, holder, false);
        let cost_val = headers
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .expect("request cost header must be present");
        assert_ne!(
            cost_val, "0.000000",
            "known model gpt-4.1 must produce non-zero cost"
        );
        // Edge case: prompt_tokens_details None, no cache hit → input_tokens = prompt_tokens
        assert_eq!(
            headers
                .get(CostHeader::INPUT_TOKENS)
                .and_then(|v| v.to_str().ok()),
            Some("1000"),
            "{} must equal prompt_tokens when no cache",
            CostHeader::INPUT_TOKENS,
        );
    }

    /// Verifies that thinking_tokens flows through to cost calculation (e.g. Gemini 2.5).
    #[test]
    fn test_thinking_tokens_increase_cost() {
        let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing must load");
        let holder = Arc::new(std::sync::RwLock::new(pricing_db));

        let usage_without_thinking = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            ..Default::default()
        };
        let usage_with_thinking = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 200,
            completion_tokens_details: Some(crate::domain::chat::CompletionTokensDetails {
                reasoning_tokens: Some(50),
            }),
            ..Default::default()
        };

        let (headers_without, _) = build_cost_headers(
            "gpt-4.1",
            &usage_without_thinking,
            Arc::clone(&holder),
            false,
        );
        let (headers_with, _) =
            build_cost_headers("gpt-4.1", &usage_with_thinking, Arc::clone(&holder), false);

        let cost_without: f64 = headers_without
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        let cost_with: f64 = headers_with
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();

        assert!(
            cost_with > cost_without,
            "reasoning_tokens must increase cost (without={cost_without}, with={cost_with})"
        );
    }

    /// cache_read tokens at reduced rate (Anthropic cache_read_multiplier 0.1).
    /// Uses Anthropic semantics: prompt_tokens is plain-only, cache_read is additive.
    #[test]
    fn test_cache_read_reduces_effective_cost() {
        use crate::domain::chat::PromptTokensDetails;

        let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing must load");
        let holder = Arc::new(std::sync::RwLock::new(pricing_db));

        // Anthropic path: 1500 prompt (all at full rate) + 500 completion, no cache
        let usage_no_cache = Usage {
            prompt_tokens: 1500,
            completion_tokens: 500,
            total_tokens: 2000,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            prompt_tokens_details: None,
            accounting: crate::domain::chat::UsageAccounting {
                cache: CacheAccounting::Additive,
                reasoning: ReasoningAccounting::Additive,
            },
            ..Default::default()
        };
        // Anthropic path: 500 input + 1000 cache_read + 500 completion
        // (prompt_tokens excludes cached; cache_read is additive)
        let usage_with_cache = Usage {
            prompt_tokens: 500,
            completion_tokens: 500,
            total_tokens: 2000,
            cache_read_input_tokens: Some(1000),
            cache_creation_input_tokens: None,
            prompt_tokens_details: None,
            accounting: crate::domain::chat::UsageAccounting {
                cache: CacheAccounting::Additive,
                reasoning: ReasoningAccounting::Additive,
            },
            ..Default::default()
        };
        let (headers_no_cache, _) = build_cost_headers(
            "claude-sonnet-4-6",
            &usage_no_cache,
            Arc::clone(&holder),
            false,
        );
        let (headers_with_cache, _) = build_cost_headers(
            "claude-sonnet-4-6",
            &usage_with_cache,
            Arc::clone(&holder),
            false,
        );
        let cost_no: f64 = headers_no_cache
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        let cost_with: f64 = headers_with_cache
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        // 500@1x + 1000@0.1x should cost less than 1500@1x
        assert!(
            cost_with < cost_no,
            "cache_read should reduce cost (with={cost_with}, no_cache={cost_no})"
        );

        // OpenAI path: prompt_tokens is total; cached from prompt_tokens_details → subtract
        let usage_openai_no_cache = Usage {
            prompt_tokens: 1500,
            completion_tokens: 500,
            total_tokens: 2000,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            prompt_tokens_details: None,
            ..Default::default()
        };
        let usage_openai_with_cache = Usage {
            prompt_tokens: 1500,
            completion_tokens: 500,
            total_tokens: 2000,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: Some(1000),
                cache_write_tokens: None,
            }),
            ..Default::default()
        };
        let (headers_openai_no, _) = build_cost_headers(
            "gpt-4.1",
            &usage_openai_no_cache,
            Arc::clone(&holder),
            false,
        );
        let (headers_openai_with, _) = build_cost_headers(
            "gpt-4.1",
            &usage_openai_with_cache,
            Arc::clone(&holder),
            false,
        );
        let cost_openai_no: f64 = headers_openai_no
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        let cost_openai_with: f64 = headers_openai_with
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        // Same model: 1500 at full rate vs 500@1x + 1000@0.1x — OpenAI path must subtract
        assert!(
            cost_openai_with < cost_openai_no,
            "OpenAI cache path should reduce cost (with={cost_openai_with}, no_cache={cost_openai_no})"
        );
    }

    /// cache_creation tokens at 1.25x (Anthropic cache_write_5m_multiplier).
    #[test]
    fn test_cache_write_5m_increases_cost() {
        let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing must load");
        let holder = Arc::new(std::sync::RwLock::new(pricing_db));
        let usage_plain = Usage {
            prompt_tokens: 1000,
            completion_tokens: 100,
            total_tokens: 1100,
            cache_creation_input_tokens: None,
            ..Default::default()
        };
        let usage_cache_write = Usage {
            prompt_tokens: 1000,
            completion_tokens: 100,
            total_tokens: 1100,
            cache_creation_input_tokens: Some(1000),
            cache_write: cache_write_of(&[("5m", 1000)]),
            ..Default::default()
        };
        let (headers_plain, _) = build_cost_headers(
            "claude-sonnet-4-6",
            &usage_plain,
            Arc::clone(&holder),
            false,
        );
        let (headers_cache, _) = build_cost_headers(
            "claude-sonnet-4-6",
            &usage_cache_write,
            Arc::clone(&holder),
            false,
        );
        let cost_plain: f64 = headers_plain
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        let cost_cache: f64 = headers_cache
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        assert!(
            cost_cache > cost_plain,
            "cache_write_5m should increase cost"
        );
    }

    /// cache_creation_1h_tokens at 2.0× produces higher cost than 5m at 1.25×.
    #[test]
    fn test_cache_creation_1h_costs_more_than_5m() {
        let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing must load");
        let holder = Arc::new(std::sync::RwLock::new(pricing_db));
        // 1000 tokens at 5m rate (1.25×)
        let usage_5m = Usage {
            prompt_tokens: 1000,
            completion_tokens: 100,
            total_tokens: 1100,
            cache_creation_input_tokens: Some(1000),
            cache_write: cache_write_of(&[("5m", 1000)]),
            ..Default::default()
        };
        // 1000 tokens at 1h rate (2.0×)
        let usage_1h = Usage {
            prompt_tokens: 1000,
            completion_tokens: 100,
            total_tokens: 1100,
            cache_creation_input_tokens: Some(1000),
            cache_write: cache_write_of(&[("1h", 1000)]),
            ..Default::default()
        };
        let cost_5m =
            build_cost_headers("claude-sonnet-4-6", &usage_5m, Arc::clone(&holder), false)
                .1
                .cost;
        let cost_1h =
            build_cost_headers("claude-sonnet-4-6", &usage_1h, Arc::clone(&holder), false)
                .1
                .cost;
        // 1h rate (2.0×) should be 1.6× more expensive than 5m rate (1.25×)
        // 2.0 / 1.25 = 1.6
        // Each usage sets only one class, so the merged `cache_write_cost` is that class's cost.
        assert!(
            cost_1h.cache_write_cost > cost_5m.cache_write_cost,
            "1h cache creation should cost more than 5m"
        );
        let ratio = cost_1h.cache_write_cost.0 as f64 / cost_5m.cache_write_cost.0 as f64;
        assert!(
            (ratio - 1.6).abs() < 0.01,
            "1h/5m cost ratio should be ~1.6 (got {})",
            ratio
        );
    }

    /// batch=true halves cost for OpenAI model with batch multipliers.
    #[test]
    fn test_batch_flag_halves_cost() {
        let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing must load");
        let holder = Arc::new(std::sync::RwLock::new(pricing_db));
        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
            ..Default::default()
        };
        let (headers_batch, _) = build_cost_headers("gpt-4.1", &usage, Arc::clone(&holder), true);
        let (headers_no_batch, _) =
            build_cost_headers("gpt-4.1", &usage, Arc::clone(&holder), false);
        let cost_batch: f64 = headers_batch
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        let cost_no_batch: f64 = headers_no_batch
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        assert!(
            (cost_batch * 2.0 - cost_no_batch).abs() < 0.000001,
            "batch=true should halve cost (batch={}, no_batch={})",
            cost_batch,
            cost_no_batch
        );
    }

    /// image_units flows to TokenUsage and increases cost when model has image_per_unit.
    #[test]
    fn test_image_units_produce_nonzero_cost() {
        let json = r#"{"models":{"img-model":{"provider":"test","context_window":1000,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0,"output_per_token":0,"image_per_unit":0.01}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &PricingConfig::default()).unwrap();
        let holder = Arc::new(std::sync::RwLock::new(db));
        let usage_no_img = Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            image_units: None,
            audio_seconds: None,
            ..Default::default()
        };
        let usage_with_img = Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            image_units: Some(3),
            audio_seconds: None,
            ..Default::default()
        };
        let (headers_no, _) =
            build_cost_headers("img-model", &usage_no_img, Arc::clone(&holder), false);
        let (headers_with, _) =
            build_cost_headers("img-model", &usage_with_img, Arc::clone(&holder), false);
        let cost_no: f64 = headers_no
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        let cost_with: f64 = headers_with
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        assert!(
            cost_with > cost_no && cost_with > 0.0,
            "image_units must increase cost (no_img={}, with_img={})",
            cost_no,
            cost_with
        );
    }

    /// audio_seconds flows to TokenUsage and increases cost when model has audio_per_second.
    #[test]
    fn test_audio_seconds_produce_nonzero_cost() {
        let json = r#"{"models":{"audio-model":{"provider":"test","context_window":1000,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0,"output_per_token":0,"audio_per_second":0.006}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &PricingConfig::default()).unwrap();
        let holder = Arc::new(std::sync::RwLock::new(db));
        let usage_no_audio = Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            image_units: None,
            audio_seconds: None,
            ..Default::default()
        };
        let usage_with_audio = Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            image_units: None,
            audio_seconds: Some(5.0),
            ..Default::default()
        };
        let (headers_no, _) =
            build_cost_headers("audio-model", &usage_no_audio, Arc::clone(&holder), false);
        let (headers_with, _) =
            build_cost_headers("audio-model", &usage_with_audio, Arc::clone(&holder), false);
        let cost_no: f64 = headers_no
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        let cost_with: f64 = headers_with
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        assert!(
            cost_with > cost_no && cost_with > 0.0,
            "audio_seconds must increase cost (no_audio={}, with_audio={})",
            cost_no,
            cost_with
        );
    }

    /// build_embedding_cost_headers sets non-zero cost for known model.
    #[test]
    fn test_build_embedding_cost_headers_known_model() {
        let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing must load");
        let holder = Arc::new(std::sync::RwLock::new(pricing_db));
        let usage = EmbeddingUsage {
            prompt_tokens: 1000,
            total_tokens: 1000,
        };
        let (headers, finalized) =
            build_embedding_cost_headers("text-embedding-3-small", &usage, holder, false);
        let (breakdown, token_usage) = (&finalized.cost, &finalized.token_usage);
        let cost_val: f64 = headers
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        assert!(
            cost_val > 0.0,
            "known embedding model must produce non-zero cost"
        );
        assert_eq!(token_usage.input_tokens, 1000);
        assert_eq!(token_usage.output_tokens, 0);
        assert!(breakdown.total_cost.as_u64() > 0);
    }

    /// build_embedding_cost_headers sets zero cost for unknown model without panicking.
    #[test]
    fn test_build_embedding_cost_headers_unknown_model() {
        let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing must load");
        let holder = Arc::new(std::sync::RwLock::new(pricing_db));
        let usage = EmbeddingUsage {
            prompt_tokens: 500,
            total_tokens: 500,
        };
        let (headers, finalized) =
            build_embedding_cost_headers("unknown-embed-model", &usage, holder, false);
        let token_usage = &finalized.token_usage;
        assert!(
            headers.contains_key(CostHeader::REQUEST_COST),
            "REQUEST_COST header must always be present"
        );
        assert_eq!(token_usage.input_tokens, 500);
        assert_eq!(token_usage.output_tokens, 0);
    }

    /// completion_tokens is always 0 for embedding cost tracking.
    #[test]
    fn test_build_embedding_cost_headers_output_tokens_zero() {
        let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing must load");
        let holder = Arc::new(std::sync::RwLock::new(pricing_db));
        let usage = EmbeddingUsage {
            prompt_tokens: 200,
            total_tokens: 200,
        };
        let (headers, finalized) =
            build_embedding_cost_headers("text-embedding-3-large", &usage, holder, false);
        let token_usage = &finalized.token_usage;
        assert_eq!(
            headers
                .get(CostHeader::OUTPUT_TOKENS)
                .and_then(|v| v.to_str().ok()),
            Some("0"),
            "OUTPUT_TOKENS must always be 0 for embeddings"
        );
        assert_eq!(token_usage.output_tokens, 0);
    }
}
