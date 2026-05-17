// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Spend record domain type .
//!
//! Pure data — no I/O. Constructed from CostBreakdown + RequestIdentity + TokenUsage
//! by the chat handler, then passed to `spend_writer::write_spend` for persistence.

use crate::domain::auth::RequestIdentity;
use crate::domain::ports::{CostBreakdown, NanoUsd, TokenUsage};

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
    pub cache_write_5m_tokens: i64,
    pub cache_write_1h_tokens: i64,
    pub thinking_tokens: i64,
    /// Total cost in nano-USD. Typed as NanoUsd; converted to i64 at the
    /// db boundary via `NanoUsd::as_i64()`.
    pub cost_nano_usd: NanoUsd,
    pub latency_ms: i32,
    /// Attribution tags from RequestIdentity (JSON object). Empty `{}` when no tags.
    pub tags: serde_json::Value,
}

impl SpendRecord {
    /// Build a SpendRecord from the completed-request context.
    ///
    /// `identity`  — injected by the auth+tagger Tower layers.
    /// `model`     — actual model returned in the provider response (not the requested model).
    /// `provider`  — name from `ProviderMetadata::name`.
    /// `token_usage` — parsed token counts returned by `build_cost_headers`.
    /// `cost`        — cost breakdown returned by `build_cost_headers`.
    /// `latency_ms`  — wall-clock milliseconds from handler entry to provider response.
    pub fn build(
        identity: &RequestIdentity,
        model: &str,
        provider: &str,
        token_usage: &TokenUsage,
        cost: &CostBreakdown,
        latency_ms: i32,
    ) -> Self {
        Self {
            org_id: identity.org_id.clone(),
            identity_id: identity.id.clone(),
            model: model.to_owned(),
            provider: provider.to_owned(),
            prompt_tokens: token_usage.input_tokens as i64,
            completion_tokens: token_usage.output_tokens as i64,
            cache_read_tokens: token_usage.cache_read_input_tokens as i64,
            cache_write_5m_tokens: token_usage.cache_write_5m_tokens as i64,
            cache_write_1h_tokens: token_usage.cache_write_1h_tokens as i64,
            thinking_tokens: token_usage.thinking_tokens as i64,
            cost_nano_usd: cost.total_cost,
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

    use crate::domain::ports::NanoUsd;

    fn make_identity(org: &str, id: &str) -> RequestIdentity {
        RequestIdentity {
            id: id.into(),
            org_id: org.into(),
            label: None,
            tags: HashMap::new(),
        }
    }

    #[test]
    fn test_build_maps_all_fields() {
        let identity = make_identity("acme", "key-123");
        let token_usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_input_tokens: 20,
            cache_write_5m_tokens: 10,
            cache_write_1h_tokens: 5,
            thinking_tokens: 8,
            ..Default::default()
        };
        let cost = CostBreakdown {
            total_cost: NanoUsd(1_500_000_000),
            ..Default::default()
        };
        let record = SpendRecord::build(&identity, "gpt-4.1", "openai", &token_usage, &cost, 123);

        assert_eq!(record.org_id, "acme");
        assert_eq!(record.identity_id, "key-123");
        assert_eq!(record.model, "gpt-4.1");
        assert_eq!(record.provider, "openai");
        assert_eq!(record.prompt_tokens, 100);
        assert_eq!(record.completion_tokens, 50);
        assert_eq!(record.cache_read_tokens, 20);
        assert_eq!(record.cache_write_5m_tokens, 10);
        assert_eq!(record.cache_write_1h_tokens, 5);
        assert_eq!(record.thinking_tokens, 8);
        assert_eq!(record.cost_nano_usd, NanoUsd(1_500_000_000));
        assert_eq!(record.latency_ms, 123);
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
            &TokenUsage::default(),
            &CostBreakdown::default(),
            0,
        );
        assert!(record.tags.is_object());
        assert_eq!(record.tags["team"], "ml");
        assert_eq!(record.tags["project"], "rag");
    }
}
