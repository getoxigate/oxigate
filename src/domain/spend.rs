// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Spend record domain type .
//!
//! Pure data — no I/O. Constructed from the request's [`FinalizedAccounting`] plus its
//! [`RequestIdentity`] by the chat handler, then passed to `spend_writer::write_spend` for
//! persistence.

use crate::domain::auth::RequestIdentity;
use crate::domain::ports::NanoUsd;
use crate::domain::usage_accounting::{CostStatus, FinalizedAccounting, UsageEvidence};

/// One row to persist in `spend_records` .
///
/// Constructed after a completed provider call; never partially populated.
/// Monetary value (`cost_nano_usd`) is stored as integer nano-USD.
#[derive(Debug, Clone)]
pub struct SpendRecord {
    pub org_id: String,
    pub identity_id: String,
    pub model: String,
    pub provider: String,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub cache_read_tokens: i64,
    pub thinking_tokens: i64,
    /// Total cost in nano-USD. Typed as NanoUsd; converted to i64 at the
    /// db boundary via `NanoUsd::as_i64()`.
    pub cost_nano_usd: NanoUsd,
    /// The confidence this row's cost carries. See [`CostStatus`].
    pub cost_status: CostStatus,
    /// The persisted evidence document backing `cost_status`, or `None` when nothing was cached.
    pub usage_evidence: Option<UsageEvidence>,
    pub latency_ms: i32,
    /// Attribution tags from RequestIdentity (JSON object). Empty `{}` when no tags.
    pub tags: serde_json::Value,
}

impl SpendRecord {
    /// Build a SpendRecord from the completed-request context.
    ///
    /// `identity`   — injected by the auth+tagger Tower layers.
    /// `model`      — actual model returned in the provider response (not the requested model).
    /// `provider`   — name from `ProviderMetadata::name`.
    /// `accounting` — the one authoritative finalization result returned by `build_cost_headers`
    ///                or `build_embedding_cost_headers`; quantities, cost, status and evidence
    ///                are read from it together so the row cannot mix facts from two different
    ///                computations.
    /// `latency_ms` — wall-clock milliseconds from handler entry to provider response.
    pub fn build(
        identity: &RequestIdentity,
        model: &str,
        provider: &str,
        accounting: &FinalizedAccounting,
        latency_ms: i32,
    ) -> Self {
        let token_usage = &accounting.token_usage;
        Self {
            org_id: identity.org_id.clone(),
            identity_id: identity.id.clone(),
            model: model.to_owned(),
            provider: provider.to_owned(),
            prompt_tokens: token_usage.input_tokens as i64,
            completion_tokens: token_usage.output_tokens as i64,
            cache_read_tokens: token_usage.cache_read_input_tokens as i64,
            thinking_tokens: token_usage.thinking_tokens as i64,
            cost_nano_usd: accounting.cost.total_cost,
            cost_status: accounting.cost.status,
            usage_evidence: accounting.evidence.clone(),
            latency_ms,
            tags: match serde_json::to_value(&identity.tags) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        identity_id = %identity.id,
                        "SpendRecord: failed to serialize tags; storing empty object"
                    );
                    serde_json::Value::Object(Default::default())
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::domain::ports::{CostBreakdown, NanoUsd, TokenUsage};
    use crate::domain::usage_accounting::{
        CacheWriteAccounting, CacheWriteAccumulator, CacheWriteClass, CacheWriteClassRegistry,
        ReconciliationFacts, WarningFacts,
    };

    /// The finalization result a completed request hands to [`SpendRecord::build`].
    ///
    /// Only the two fields the row reads are varied here; the rest carry their clean defaults.
    fn finalized(token_usage: TokenUsage, cost: CostBreakdown) -> FinalizedAccounting {
        FinalizedAccounting {
            token_usage,
            cost,
            evidence: None,
            reconciliation: ReconciliationFacts::default(),
            warning: WarningFacts::default(),
        }
    }

    fn make_identity(org: &str, id: &str) -> RequestIdentity {
        RequestIdentity {
            id: id.into(),
            org_id: org.into(),
            label: None,
            tags: HashMap::new(),
        }
    }

    /// Builds cache-write accounting state through the only path that produces one — the
    /// accumulator — rather than a struct literal `TokenUsage::cache_write` has no fields for.
    fn cache_write_of(pairs: &[(&str, u64)]) -> CacheWriteAccounting {
        let registry = CacheWriteClassRegistry::from_classes(
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

    #[test]
    fn test_build_maps_all_fields() {
        let identity = make_identity("acme", "key-123");
        let token_usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_input_tokens: 20,
            cache_write: cache_write_of(&[("5m", 10), ("1h", 5)]),
            thinking_tokens: 8,
            ..Default::default()
        };
        let cost = CostBreakdown {
            total_cost: NanoUsd(1_500_000_000),
            status: CostStatus::Exact,
            ..Default::default()
        };
        let mut accounting = finalized(token_usage, cost);
        accounting.evidence = accounting.token_usage.cache_write.to_evidence(0);
        let record = SpendRecord::build(&identity, "gpt-4.1", "openai", &accounting, 123);

        assert_eq!(record.org_id, "acme");
        assert_eq!(record.identity_id, "key-123");
        assert_eq!(record.model, "gpt-4.1");
        assert_eq!(record.provider, "openai");
        assert_eq!(record.prompt_tokens, 100);
        assert_eq!(record.completion_tokens, 50);
        assert_eq!(record.cache_read_tokens, 20);
        assert_eq!(record.thinking_tokens, 8);
        assert_eq!(record.cost_nano_usd, NanoUsd(1_500_000_000));
        assert_eq!(record.cost_status, CostStatus::Exact);
        assert_eq!(record.usage_evidence, accounting.evidence);
        assert_eq!(record.latency_ms, 123);
    }

    /// AC27: evidence completeness is a fact *about* the request, not an input to its cost.
    /// Two accountings that share the same `cost` but disagree only on whether the retained
    /// evidence document is `incomplete` must produce the same billed `cost_nano_usd` and the
    /// same `cost_status` — the value `SpendRecord` writes to the row, and the same value
    /// `build_cost_headers`/`build_embedding_cost_headers` write to headers and the terminal SSE
    /// event, and the value `write_spend` uses for the Redis budget counter. All three read
    /// `accounting.cost` directly; none of them reads `accounting.evidence` to decide an amount.
    ///
    /// Negative control: if a future change made cost or status depend on evidence completeness
    /// (a second authority, or a recomputation this module must never introduce), the two
    /// `cost_nano_usd`/`cost_status` assertions below would diverge and fail.
    #[test]
    fn test_evidence_completeness_does_not_change_billed_cost() {
        let identity = make_identity("acme", "key-9");
        let cost = CostBreakdown {
            total_cost: NanoUsd(2_000_000_000),
            status: CostStatus::Reconciled,
            ..Default::default()
        };

        let evidence_of = |incomplete: bool| UsageEvidence {
            schema_version: crate::domain::usage_accounting::EVIDENCE_SCHEMA_VERSION,
            cache_write: crate::domain::usage_accounting::CacheWriteEvidence {
                reported_tokens: 100,
                detail_tokens: 100,
                accounted_tokens: 100,
                component_cost_nano_usd: 500,
                reconciliation: crate::domain::usage_accounting::ReconciliationOutcome::Consistent,
                unknown_duplicates_indeterminate: false,
                quantity_overflow: false,
                entries: Vec::new(),
                incomplete,
            },
        };

        let mut complete = finalized(TokenUsage::default(), cost.clone());
        complete.evidence = Some(evidence_of(false));
        let mut incomplete = finalized(TokenUsage::default(), cost);
        incomplete.evidence = Some(evidence_of(true));

        let complete_record = SpendRecord::build(&identity, "m", "p", &complete, 10);
        let incomplete_record = SpendRecord::build(&identity, "m", "p", &incomplete, 10);

        // Sanity: the fixtures actually differ in evidence completeness — otherwise this test
        // would pass vacuously.
        assert_ne!(
            complete_record
                .usage_evidence
                .as_ref()
                .expect("evidence")
                .cache_write
                .incomplete,
            incomplete_record
                .usage_evidence
                .as_ref()
                .expect("evidence")
                .cache_write
                .incomplete,
        );

        assert_eq!(
            complete_record.cost_nano_usd, incomplete_record.cost_nano_usd,
            "billed cost must not depend on evidence completeness"
        );
        assert_eq!(
            complete_record.cost_status, incomplete_record.cost_status,
            "cost_status must not depend on evidence completeness"
        );
    }

    #[test]
    fn test_redis_key_format_string() {
        // Delegates to spend_writer's canonical helper — a typo there will fail this test.
        let key = crate::utils::identity_spend_key("acme", "key-abc", "");
        assert_eq!(key, "oxigate:org:acme:spend:key-abc");
    }

    #[test]
    fn test_tags_serialized_to_json_object() {
        let mut tags = HashMap::new();
        tags.insert("team".to_string(), "ml".to_string());
        tags.insert("project".to_string(), "rag".to_string());
        let identity = RequestIdentity {
            id: "x".into(),
            org_id: "o".into(),
            label: None,
            tags,
        };
        let record = SpendRecord::build(
            &identity,
            "m",
            "p",
            &finalized(TokenUsage::default(), CostBreakdown::default()),
            0,
        );
        assert!(record.tags.is_object());
        assert_eq!(record.tags["team"], "ml");
        assert_eq!(record.tags["project"], "rag");
    }
}
