// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Cost header construction for chat completions responses.
//!
//! Builds [`CostHeader::REQUEST_COST`], [`CostHeader::INPUT_TOKENS`], [`CostHeader::OUTPUT_TOKENS`],
//! and [`CostHeader::MODEL_USED`] using the bundled pricing DB.

use std::sync::Arc;

use axum::http::header::{HeaderMap, HeaderValue};
use axum::response::Response;
use tracing::warn;

use crate::domain::chat::Usage;
use crate::domain::ports::{CostCalculator, TokenUsage};
use crate::domain::pricing::BundledCostCalculator;

pub struct CostHeader;

impl CostHeader {
    pub const REQUEST_COST: &'static str = "X-Oxigate-Request-Cost";
    pub const INPUT_TOKENS: &'static str = "X-Oxigate-Input-Tokens";
    pub const OUTPUT_TOKENS: &'static str = "X-Oxigate-Output-Tokens";
    pub const MODEL_USED: &'static str = "X-Oxigate-Model-Used";
    pub const BUDGET_REMAINING: &'static str = "X-Oxigate-Budget-Remaining";
    pub const BUDGET_CAP: &'static str = "X-Oxigate-Budget-Cap";
}

/// Builds cost headers for the response.
///
/// Returns `(HeaderMap, CostBreakdown, TokenUsage)` so callers can use the
/// cost breakdown and token usage for spend writing  and structured
/// cost logging  without re-computing.
///
/// Takes `Arc<RwLock<PricingDb>>` because `BundledCostCalculator` retains it for
/// Class A hot-reload (SIGHUP) semantics; the clone is cheap.
#[must_use]
pub fn build_cost_headers(
    model: &str,
    usage: &Usage,
    pricing_db: Arc<std::sync::RwLock<crate::domain::pricing::PricingDb>>,
    batch: bool,
) -> (HeaderMap, crate::domain::ports::CostBreakdown, TokenUsage) {
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
    let input_tokens = match usage.cache_accounting {
        crate::domain::chat::CacheAccounting::Additive => usage.prompt_tokens,
        crate::domain::chat::CacheAccounting::Inclusive => {
            usage.prompt_tokens.saturating_sub(cache_read)
        }
    };
    let token_usage = TokenUsage {
        input_tokens,
        output_tokens: usage.completion_tokens,
        cache_read_input_tokens: cache_read,
        cache_write_5m_tokens: usage.cache_creation_5m_tokens,
        cache_write_1h_tokens: usage.cache_creation_1h_tokens,
        thinking_tokens,
        tier_threshold_override: usage.tier_threshold_override,
        batch,
        image_count: usage.image_units.unwrap_or(0),
        audio_seconds: usage.audio_seconds.unwrap_or(0.0),
    };
    assemble_cost_headers(
        model,
        &token_usage,
        usage.prompt_tokens,
        usage.completion_tokens,
        pricing_db,
    )
}

/// Shared computation: cost calculation + header assembly for both chat and embedding paths.
///
/// `prompt_tokens_display` feeds `INPUT_TOKENS` header; `token_usage.input_tokens` feeds cost calc.
/// For embeddings they are equal. For chat they diverge (cached tokens split).
fn assemble_cost_headers(
    model: &str,
    token_usage: &TokenUsage,
    prompt_tokens_display: u64,
    completion_tokens_display: u64,
    pricing_db: Arc<std::sync::RwLock<crate::domain::pricing::PricingDb>>,
) -> (HeaderMap, crate::domain::ports::CostBreakdown, TokenUsage) {
    let calc = BundledCostCalculator::new(pricing_db);
    let cost = match calc.calculate(model, token_usage) {
        Ok(c) => c,
        Err(e) => {
            warn!(model = %model, error = %e, "cost calculation failed, using zero; check pricing config");
            Default::default()
        }
    };

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
    (map, cost, token_usage.clone())
}

/// Builds cost headers for an embeddings response.
///
/// Returns `(HeaderMap, CostBreakdown, TokenUsage)` for spend writing and structured logging.
#[must_use]
pub fn build_embedding_cost_headers(
    model: &str,
    usage: &crate::domain::embedding::EmbeddingUsage,
    pricing_db: Arc<std::sync::RwLock<crate::domain::pricing::PricingDb>>,
    is_batch: bool,
) -> (HeaderMap, crate::domain::ports::CostBreakdown, TokenUsage) {
    let token_usage = TokenUsage {
        input_tokens: usage.prompt_tokens,
        output_tokens: 0,
        cache_read_input_tokens: 0,
        cache_write_5m_tokens: 0,
        cache_write_1h_tokens: 0,
        thinking_tokens: 0,
        tier_threshold_override: None,
        batch: is_batch,
        image_count: 0,
        audio_seconds: 0.0,
    };
    assemble_cost_headers(model, &token_usage, usage.prompt_tokens, 0, pricing_db)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PricingConfig;
    use crate::domain::embedding::EmbeddingUsage;
    use crate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
    use axum::response::IntoResponse;

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
        let (headers, _, _) = build_cost_headers("gpt-4.1", &usage, holder, false);
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

        let (headers_without, _, _) = build_cost_headers(
            "gpt-4.1",
            &usage_without_thinking,
            Arc::clone(&holder),
            false,
        );
        let (headers_with, _, _) =
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
            cache_accounting: crate::domain::chat::CacheAccounting::Additive,
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
            cache_accounting: crate::domain::chat::CacheAccounting::Additive,
            ..Default::default()
        };
        let (headers_no_cache, _, _) = build_cost_headers(
            "claude-sonnet-4-6",
            &usage_no_cache,
            Arc::clone(&holder),
            false,
        );
        let (headers_with_cache, _, _) = build_cost_headers(
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
            }),
            ..Default::default()
        };
        let (headers_openai_no, _, _) = build_cost_headers(
            "gpt-4.1",
            &usage_openai_no_cache,
            Arc::clone(&holder),
            false,
        );
        let (headers_openai_with, _, _) = build_cost_headers(
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
            cache_creation_5m_tokens: 0,
            cache_creation_1h_tokens: 0,
            ..Default::default()
        };
        let usage_cache_write = Usage {
            prompt_tokens: 1000,
            completion_tokens: 100,
            total_tokens: 1100,
            cache_creation_input_tokens: Some(1000),
            cache_creation_5m_tokens: 1000,
            cache_creation_1h_tokens: 0,
            ..Default::default()
        };
        let (headers_plain, _, _) = build_cost_headers(
            "claude-sonnet-4-6",
            &usage_plain,
            Arc::clone(&holder),
            false,
        );
        let (headers_cache, _, _) = build_cost_headers(
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
            cache_creation_5m_tokens: 1000,
            cache_creation_1h_tokens: 0,
            ..Default::default()
        };
        // 1000 tokens at 1h rate (2.0×)
        let usage_1h = Usage {
            prompt_tokens: 1000,
            completion_tokens: 100,
            total_tokens: 1100,
            cache_creation_input_tokens: Some(1000),
            cache_creation_5m_tokens: 0,
            cache_creation_1h_tokens: 1000,
            ..Default::default()
        };
        let (_, cost_5m, _) =
            build_cost_headers("claude-sonnet-4-6", &usage_5m, Arc::clone(&holder), false);
        let (_, cost_1h, _) =
            build_cost_headers("claude-sonnet-4-6", &usage_1h, Arc::clone(&holder), false);
        // 1h rate (2.0×) should be 1.6× more expensive than 5m rate (1.25×)
        // 2.0 / 1.25 = 1.6
        assert!(
            cost_1h.cache_write_1h_cost > cost_5m.cache_write_5m_cost,
            "1h cache creation should cost more than 5m"
        );
        let ratio = cost_1h.cache_write_1h_cost.0 as f64 / cost_5m.cache_write_5m_cost.0 as f64;
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
        let (headers_batch, _, _) =
            build_cost_headers("gpt-4.1", &usage, Arc::clone(&holder), true);
        let (headers_no_batch, _, _) =
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
        let (headers_no, _, _) =
            build_cost_headers("img-model", &usage_no_img, Arc::clone(&holder), false);
        let (headers_with, _, _) =
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
        let (headers_no, _, _) =
            build_cost_headers("audio-model", &usage_no_audio, Arc::clone(&holder), false);
        let (headers_with, _, _) =
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
        let (headers, breakdown, token_usage) =
            build_embedding_cost_headers("text-embedding-3-small", &usage, holder, false);
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
        let (headers, _, token_usage) =
            build_embedding_cost_headers("unknown-embed-model", &usage, holder, false);
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
        let (headers, _, token_usage) =
            build_embedding_cost_headers("text-embedding-3-large", &usage, holder, false);
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
