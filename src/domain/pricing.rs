// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Pricing domain: bundled model DB, tiered rates, and cost calculation.
//!
//! compile-time embedded anchor set, YAML overrides, fail-fast startup.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Bundled pricing JSON (compile-time embed). Used at startup and for Class A reload.
pub const BUNDLED_PRICING_JSON: &[u8] = include_bytes!("../../assets/model_prices.json");

use serde::Deserialize;
use thiserror::Error;
use tracing::warn;

use crate::config::PricingConfig;
use crate::domain::ports::{CostBreakdown, CostCalculator, CostError, NanoUsd, TokenUsage};

/// One threshold band in a model's tiered pricing.
#[derive(Debug, Clone, Deserialize)]
pub struct PricingTier {
    /// Input token threshold; 0 = base rate.
    pub threshold: u64,
    /// Cost per input token (USD).
    pub input_per_token: f64,
    /// Cost per output token (USD).
    pub output_per_token: f64,
    /// Fraction of input_per_token for cache-read tokens.
    #[serde(default)]
    pub cache_read_multiplier: Option<f64>,
    /// Multiplier for 5m cache write.
    #[serde(default)]
    pub cache_write_5m_multiplier: Option<f64>,
    /// Multiplier for 1h cache write.
    #[serde(default)]
    pub cache_write_1h_multiplier: Option<f64>,
    /// Cost per thinking token (if applicable).
    #[serde(default)]
    pub thinking_per_token: Option<f64>,
    /// Cost per image unit.
    #[serde(default)]
    pub image_per_unit: Option<f64>,
    /// Cost per second of audio (USD). When adding to model_prices.json, use sufficient
    /// precision (e.g. 6+ decimal places) for very cheap rates — rates below ~1e-9 USD/sec
    /// round to zero in nano-USD conversion.
    #[serde(default)]
    pub audio_per_second: Option<f64>,
    /// Batch input discount (e.g. 0.5 = 50%).
    #[serde(default)]
    pub batch_input_multiplier: Option<f64>,
    /// Batch output discount.
    #[serde(default)]
    pub batch_output_multiplier: Option<f64>,
}

/// Model pricing record.
#[derive(Debug, Clone)]
pub struct PricingEntry {
    /// Canonical model ID.
    pub model_id: String,
    /// Provider name (openai, anthropic, etc.).
    pub provider: String,
    /// Alternative names for lookup.
    pub aliases: Vec<String>,
    /// Context window size; 0 = unknown/unconstrained.
    pub context_window: u32,
    /// Max output tokens if known.
    pub max_output_tokens: Option<u32>,
    /// Tiers sorted ascending by threshold.
    pub tiers: Vec<PricingTier>,
}

/// Inner DB state — canonical map and alias map.
pub(crate) struct PricingDbInner {
    /// Canonical model ID → entry.
    pub(crate) by_canonical: HashMap<String, PricingEntry>,
    /// Alias → canonical model ID.
    pub(crate) by_alias: HashMap<String, String>,
}

/// In-memory pricing DB. Uses `std::sync::RwLock` (read-heavy, write at startup
/// only) to avoid `tokio::RwLock::blocking_read` panic risk in async context.
#[derive(Clone)]
pub struct PricingDb(Arc<RwLock<PricingDbInner>>);

impl std::fmt::Debug for PricingDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PricingDb").field(&"[..]").finish()
    }
}

/// Pricing load or validation error.
#[derive(Debug, Error)]
pub enum PricingError {
    /// JSON parse failure.
    #[error("pricing parse failure: {0}")]
    ParseFailure(#[from] serde_json::Error),
    /// Validation errors (all collected, concatenated).
    #[error("invalid pricing DB: {0}")]
    InvalidDb(String),
}

/// Raw JSON model record (for deserialization).
#[derive(Debug, Deserialize)]
struct JsonModel {
    provider: String,
    context_window: u32,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    aliases: Vec<String>,
    tiers: Vec<PricingTier>,
}

/// Root JSON structure.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct JsonRoot {
    #[serde(default)]
    schema_version: u32,
    #[serde(default)]
    snapshot_date: Option<String>,
    models: HashMap<String, JsonModel>,
}

impl PricingDbInner {
    /// Lookups model by canonical ID or alias. `provider` reserved for.
    ///
    /// When the exact model ID is not found (e.g. `gpt-4o-2024-08-06` from streaming),
    /// falls back to stripping a trailing `-YYYY-MM-DD` suffix so provider-specific
    /// revision IDs resolve to base-model pricing (e.g. `gpt-4o`).
    ///
    /// This is a proven corner case of Open AI, when streaming responses return
    /// provider-specific IDs like `gpt-4o-2024-08-06`.
    pub fn lookup<'a>(&'a self, model: &str, _provider: Option<&str>) -> Option<&'a PricingEntry> {
        if let Some(entry) = self.by_canonical.get(model) {
            return Some(entry);
        }
        if let Some(entry) = self
            .by_alias
            .get(model)
            .and_then(|canon| self.by_canonical.get(canon))
        {
            return Some(entry);
        }
        // Fallback: strip trailing -YYYY-MM-DD (provider revision suffix)
        if model.len() > 11 {
            let suffix = &model[model.len() - 11..]; // "-YYYY-MM-DD"
            if suffix.starts_with('-')
                && suffix[1..5].chars().all(|c| c.is_ascii_digit())
                && suffix.chars().nth(5) == Some('-')
                && suffix[6..8].chars().all(|c| c.is_ascii_digit())
                && suffix.chars().nth(8) == Some('-')
                && suffix[9..11].chars().all(|c| c.is_ascii_digit())
            {
                let base = &model[..model.len() - 11];
                if !base.is_empty() {
                    if let Some(entry) = self.by_canonical.get(base) {
                        return Some(entry);
                    }
                    if let Some(entry) = self
                        .by_alias
                        .get(base)
                        .and_then(|canon| self.by_canonical.get(canon))
                    {
                        return Some(entry);
                    }
                }
            }
        }
        None
    }

    /// Returns the input cost per million tokens for the given model.
    ///
    /// Uses the first (base) tier's `input_per_token` rate. Returns `NanoUsd::MAX`
    /// when the model is not in the pricing DB — signals "unknown cost" to routing strategies.
    pub fn input_cost_per_million(&self, model: &str) -> crate::domain::ports::NanoUsd {
        match self.lookup(model, None) {
            Some(entry) => {
                let rate = entry
                    .tiers
                    .first()
                    .map(|t| t.input_per_token)
                    .unwrap_or(0.0);
                crate::domain::ports::NanoUsd::from_f64_usd(rate * 1_000_000.0)
            }
            None => crate::domain::ports::NanoUsd::MAX,
        }
    }
}

impl PricingEntry {
    /// Returns the highest tier where `total_input_tokens >= tier.threshold`.
    pub fn get_tier(&self, total_input_tokens: u64) -> &PricingTier {
        let mut highest_tier_idx = 0;
        for (i, t) in self.tiers.iter().enumerate() {
            if total_input_tokens >= t.threshold {
                highest_tier_idx = i;
            }
        }
        &self.tiers[highest_tier_idx]
    }
}

impl PricingDb {
    /// Acquires a read lock for lookups. Safe from async context (no blocking_read).
    pub(crate) fn read(&self) -> std::sync::RwLockReadGuard<'_, PricingDbInner> {
        self.0.read().expect("pricing lock poisoned")
    }

    /// Loads and validates the pricing DB from bytes, merging YAML overrides.
    ///
    /// Overrides always win. Fails if JSON is invalid or validation fails.
    pub fn load(bytes: &[u8], config: &PricingConfig) -> Result<Self, PricingError> {
        let root: JsonRoot = serde_json::from_slice(bytes)?;
        let mut by_canonical: HashMap<String, PricingEntry> = HashMap::new();
        let mut by_alias: HashMap<String, String> = HashMap::new();
        let mut errors: Vec<String> = Vec::new();

        for (id, m) in &root.models {
            let tiers = m.tiers.clone();
            let entry = PricingEntry {
                model_id: id.clone(),
                provider: m.provider.clone(),
                aliases: m.aliases.clone(),
                context_window: m.context_window,
                max_output_tokens: m.max_output_tokens,
                tiers: tiers.clone(),
            };
            let mut entry_errors = validate_entry(&entry);
            if entry_errors.is_empty() {
                let mut tiers = tiers;
                tiers.sort_by_key(|t| t.threshold);
                let entry = PricingEntry { tiers, ..entry };
                if by_canonical.contains_key(id) {
                    errors.push(format!("duplicate canonical model ID: {}", id));
                } else {
                    for a in &entry.aliases {
                        if let Some(existing) = by_alias.insert(a.clone(), id.clone())
                            && existing != *id
                        {
                            errors.push(format!("alias '{}' collides ({} vs {})", a, existing, id));
                        }
                    }
                    by_canonical.insert(id.clone(), entry);
                }
            } else {
                errors.append(&mut entry_errors);
            }
        }

        if !errors.is_empty() {
            return Err(PricingError::InvalidDb(errors.join("; ")));
        }

        // Apply overrides
        for (model_key, ov) in &config.overrides {
            apply_override(&mut by_canonical, &mut by_alias, model_key, ov);
        }

        Ok(Self(Arc::new(RwLock::new(PricingDbInner {
            by_canonical,
            by_alias,
        }))))
    }
}

fn validate_entry(entry: &PricingEntry) -> Vec<String> {
    let mut errors = Vec::new();
    if entry.tiers.is_empty() {
        errors.push(format!("model {} has no tiers", entry.model_id));
    }
    for (i, t) in entry.tiers.iter().enumerate() {
        if t.input_per_token < 0.0 || t.output_per_token < 0.0 {
            errors.push(format!(
                "model {} tier {}: input/output per token must be >= 0",
                entry.model_id, i
            ));
        }
        for (name, opt) in [
            ("cache_read_multiplier", t.cache_read_multiplier),
            ("cache_write_5m_multiplier", t.cache_write_5m_multiplier),
            ("cache_write_1h_multiplier", t.cache_write_1h_multiplier),
            ("batch_input_multiplier", t.batch_input_multiplier),
            ("batch_output_multiplier", t.batch_output_multiplier),
        ] {
            if let Some(v) = opt
                && !(0.0..=10.0).contains(&v)
            {
                errors.push(format!(
                    "model {} tier {}: {} must be in [0.0, 10.0]",
                    entry.model_id, i, name
                ));
            }
        }
    }
    for i in 1..entry.tiers.len() {
        if entry.tiers[i].threshold <= entry.tiers[i - 1].threshold {
            errors.push(format!(
                "model {}: tier thresholds must be strictly ascending",
                entry.model_id
            ));
            break;
        }
    }
    errors
}

/// Applies a YAML override to the pricing DB, creating or replacing an entry.
///
/// Config layer should catch invalid overrides first; this validation keeps the
/// domain self-protecting.
fn apply_override(
    by_canonical: &mut HashMap<String, PricingEntry>,
    by_alias: &mut HashMap<String, String>,
    model_key: &str,
    ov: &crate::config::PricingOverride,
) {
    let tier = PricingTier {
        threshold: 0,
        input_per_token: ov.input_per_token,
        output_per_token: ov.output_per_token,
        cache_read_multiplier: ov.cache_read_multiplier,
        cache_write_5m_multiplier: None,
        cache_write_1h_multiplier: None,
        thinking_per_token: None,
        image_per_unit: None,
        audio_per_second: None,
        batch_input_multiplier: None,
        batch_output_multiplier: None,
    };

    let aliases = if let Some(existing) = by_canonical.get(model_key) {
        existing.aliases.clone()
    } else {
        Vec::new()
    };

    // Remove old aliases so we don't leave stale entries
    for a in &aliases {
        by_alias.remove(a);
    }

    let entry = PricingEntry {
        model_id: model_key.to_string(),
        provider: by_canonical
            .get(model_key)
            .map(|e| e.provider.clone())
            .unwrap_or_else(|| "override".to_string()),
        aliases: aliases.clone(),
        context_window: ov.context_window,
        max_output_tokens: None,
        tiers: vec![tier],
    };

    let entry_errors = validate_entry(&entry);
    if !entry_errors.is_empty() {
        warn!(
            model_key = model_key,
            issues = %entry_errors.join("; "),
            "pricing override validation failed; skipping override. Config layer should catch invalid overrides first."
        );
        for a in &aliases {
            by_alias.insert(a.clone(), model_key.to_string());
        }
        return;
    }

    by_canonical.insert(model_key.to_string(), entry);
    for a in &aliases {
        by_alias.insert(a.clone(), model_key.to_string());
    }
}

/// Default cost calculator using the bundled pricing DB.
///
/// Holds `Arc` to the pricing DB holder (not a snapshot) so it always sees the
/// current DB after Class A SIGHUP reload.
pub struct BundledCostCalculator {
    db_holder: Arc<RwLock<PricingDb>>,
}

impl BundledCostCalculator {
    /// Creates a calculator that reads from the given holder on each `calculate()`.
    /// The holder enables Class A hot-reload; calculator always sees current DB.
    #[must_use]
    pub fn new(db_holder: Arc<RwLock<PricingDb>>) -> Self {
        Self { db_holder }
    }
}

/// Convert USD-per-token rate to nano-USD per token.
fn rate_to_nano_usd_per_token(rate_usd: f64) -> u64 {
    (rate_usd * 1_000_000_000.0).round().max(0.0) as u64
}

/// Multiplier as 1e9 scale (0.5 -> 500_000_000). Warns when clamp activates.
fn mult_to_1e9(m: f64) -> u64 {
    if m > 10.0 {
        warn!(
            multiplier = m,
            "pricing multiplier exceeds 10x, clamping to 10x"
        );
    }
    (m.clamp(0.0, 10.0) * 1_000_000_000.0).round() as u64
}

impl CostCalculator for BundledCostCalculator {
    fn calculate(&self, model: &str, usage: &TokenUsage) -> Result<CostBreakdown, CostError> {
        let db = self.db_holder.read().expect("pricing holder lock poisoned");
        let inner = db.read();
        let entry = inner.lookup(model, None);

        match entry {
            Some(e) => {
                //: use tier_threshold_override for Gemini (input+cached); else input_tokens.
                let tier_comparator = usage.tier_threshold_override.unwrap_or(usage.input_tokens);
                let tier = e.get_tier(tier_comparator);

                // Integer arithmetic: rates in nano-USD per token
                let input_rate = rate_to_nano_usd_per_token(tier.input_per_token);
                let output_rate = rate_to_nano_usd_per_token(tier.output_per_token);
                let mut input_cost = NanoUsd(usage.input_tokens.saturating_mul(input_rate));
                let mut output_cost = NanoUsd(usage.output_tokens.saturating_mul(output_rate));

                let cache_read_mult = mult_to_1e9(tier.cache_read_multiplier.unwrap_or(1.0));
                let mut cached_input_cost = NanoUsd(
                    ((usage.cache_read_input_tokens as u128)
                        .saturating_mul(input_rate as u128)
                        .saturating_mul(cache_read_mult as u128)
                        / 1_000_000_000u128) as u64,
                );

                let cache_5m_mult = mult_to_1e9(tier.cache_write_5m_multiplier.unwrap_or(1.0));
                let mut cache_write_5m_cost = NanoUsd(
                    ((usage.cache_write_5m_tokens as u128)
                        .saturating_mul(input_rate as u128)
                        .saturating_mul(cache_5m_mult as u128)
                        / 1_000_000_000u128) as u64,
                );

                let cache_1h_mult = mult_to_1e9(tier.cache_write_1h_multiplier.unwrap_or(1.0));
                let mut cache_write_1h_cost = NanoUsd(
                    ((usage.cache_write_1h_tokens as u128)
                        .saturating_mul(input_rate as u128)
                        .saturating_mul(cache_1h_mult as u128)
                        / 1_000_000_000u128) as u64,
                );

                let thinking_rate = rate_to_nano_usd_per_token(
                    tier.thinking_per_token.unwrap_or(tier.output_per_token),
                );
                let mut thinking_cost =
                    NanoUsd(usage.thinking_tokens.saturating_mul(thinking_rate));

                //: image and audio costs (multimodal).
                let image_rate_nano =
                    rate_to_nano_usd_per_token(tier.image_per_unit.unwrap_or(0.0));
                let mut image_cost = NanoUsd(usage.image_count.saturating_mul(image_rate_nano));

                // rate_to_nano_usd_per_token converts USD→nano-USD; unit is seconds not tokens here
                let audio_rate_nano =
                    rate_to_nano_usd_per_token(tier.audio_per_second.unwrap_or(0.0));
                let mut audio_cost = NanoUsd(
                    ((usage.audio_seconds * audio_rate_nano as f64)
                        .round()
                        .max(0.0)) as u64,
                );

                //: apply batch discount to all token cost components when usage.batch.
                // Cache costs are input-token-based; thinking is output-token-based.
                // Image (vision input) uses batch_input_multiplier; audio (e.g. TTS output) uses batch_output_multiplier.
                if usage.batch {
                    let batch_in = mult_to_1e9(tier.batch_input_multiplier.unwrap_or(1.0));
                    let batch_out = mult_to_1e9(tier.batch_output_multiplier.unwrap_or(1.0));
                    let apply_in = |c: NanoUsd| {
                        NanoUsd(((c.0 as u128 * batch_in as u128) / 1_000_000_000u128) as u64)
                    };
                    let apply_out = |c: NanoUsd| {
                        NanoUsd(((c.0 as u128 * batch_out as u128) / 1_000_000_000u128) as u64)
                    };
                    input_cost = apply_in(input_cost);
                    output_cost = apply_out(output_cost);
                    cached_input_cost = apply_in(cached_input_cost);
                    cache_write_5m_cost = apply_in(cache_write_5m_cost);
                    cache_write_1h_cost = apply_in(cache_write_1h_cost);
                    thinking_cost = apply_out(thinking_cost);
                    image_cost = apply_in(image_cost);
                    audio_cost = apply_out(audio_cost);
                }

                let total_cost = input_cost
                    + output_cost
                    + cached_input_cost
                    + cache_write_5m_cost
                    + cache_write_1h_cost
                    + thinking_cost
                    + image_cost
                    + audio_cost;

                Ok(CostBreakdown {
                    input_cost,
                    output_cost,
                    cached_input_cost,
                    cache_write_5m_cost,
                    cache_write_1h_cost,
                    thinking_cost,
                    image_cost,
                    audio_cost,
                    total_cost,
                })
            }
            None => {
                warn!(model = model, "model_not_in_pricing_db");
                Ok(CostBreakdown::zero())
            }
        }
    }

    fn handles_model(&self, _model: &str) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PricingOverride;
    use crate::domain::ports::{NanoUsd, TokenUsage};
    use proptest::prelude::*;
    use tracing_test::traced_test;

    fn token_usage_strategy() -> impl Strategy<Value = TokenUsage> {
        (
            0u64..500_000u64,
            0u64..500_000u64,
            0u64..100_000u64,
            0u64..50_000u64,
            0u64..50_000u64,
            0u64..100_000u64,
            0u64..10u64,
            0.0f64..30.0f64,
        )
            .prop_map(|(i, o, cr, c5, c1, th, img, aud)| TokenUsage {
                input_tokens: i,
                output_tokens: o,
                cache_read_input_tokens: cr,
                cache_write_5m_tokens: c5,
                cache_write_1h_tokens: c1,
                thinking_tokens: th,
                image_count: img,
                audio_seconds: aud,
                batch: false,
                tier_threshold_override: None,
            })
    }

    fn default_config() -> PricingConfig {
        PricingConfig::default()
    }

    fn db_holder(db: PricingDb) -> Arc<RwLock<PricingDb>> {
        Arc::new(RwLock::new(db))
    }

    #[test]
    fn test_parse_bundled_json() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let guard = db.read();
        // 2026-05-09 snapshot: 56 models (+5: text-embedding-3-large, text-embedding-ada-002,
        // text-embedding-004, text-multilingual-embedding-002, gemini-embedding-exp-03-07)
        assert_eq!(guard.by_canonical.len(), 56);
    }

    #[test]
    fn test_lookup_canonical() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let guard = db.read();
        let entry = guard.lookup("gpt-4.1-2025-04-14", None);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().model_id, "gpt-4.1-2025-04-14");
    }

    #[test]
    fn test_lookup_alias() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let guard = db.read();
        let entry = guard.lookup("gpt-4.1", None);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().model_id, "gpt-4.1-2025-04-14");
    }

    #[test]
    fn test_lookup_unknown_returns_none() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let guard = db.read();
        assert!(guard.lookup("unknown-xyz", None).is_none());
    }

    #[test]
    fn test_lookup_strips_date_suffix_fallback() {
        // Streaming responses return provider-specific IDs like gpt-4o-2024-08-06.
        // Fallback strips -YYYY-MM-DD and resolves via alias (gpt-4o -> gpt-4o-2024-11-20).
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let guard = db.read();
        let entry = guard.lookup("gpt-4o-2024-08-06", None);
        assert!(
            entry.is_some(),
            "gpt-4o-2024-08-06 should resolve via date-suffix fallback"
        );
        assert_eq!(entry.unwrap().model_id, "gpt-4o-2024-11-20");
    }

    #[test]
    fn test_calculate_known_model() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 500_000,
            ..Default::default()
        };
        let cost = calc.calculate("gpt-4.1", &usage).unwrap();
        // $2/M input = 0.000002, $8/M output = 0.000008 → 6 USD = 6_000_000_000 nano
        assert_eq!(cost.total_cost, NanoUsd(6_000_000_000));
    }

    #[traced_test]
    #[test]
    fn test_calculate_unknown_emits_warn() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage::default();
        let cost = calc.calculate("unknown-xyz", &usage).unwrap();
        assert_eq!(cost.total_cost, NanoUsd::zero());
        assert!(logs_contain("model_not_in_pricing_db"));
    }

    #[traced_test]
    #[test]
    fn test_calculate_local_no_config_emits_warn() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let cost = calc
            .calculate("ollama/llama3.2", &TokenUsage::default())
            .unwrap();
        assert_eq!(cost.total_cost, NanoUsd::zero());
        assert!(logs_contain("model_not_in_pricing_db"));
    }

    #[test]
    fn test_override_wins_over_db() {
        let mut config = default_config();
        config.overrides.insert(
            "gpt-4.1-mini-2025-04-14".into(),
            PricingOverride {
                input_per_token: 0.001,
                output_per_token: 0.001,
                context_window: 1_000_000,
                cache_read_multiplier: None,
            },
        );
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &config).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 1000,
            output_tokens: 500,
            ..Default::default()
        };
        let cost = calc.calculate("gpt-4.1-mini", &usage).unwrap();
        assert_eq!(cost.total_cost, NanoUsd(1_500_000_000));
    }

    #[traced_test]
    #[test]
    fn test_override_zero_suppresses_warn() {
        let mut config = default_config();
        config.overrides.insert(
            "ollama/llama3.2".into(),
            PricingOverride {
                input_per_token: 0.0,
                output_per_token: 0.0,
                context_window: 128_000,
                cache_read_multiplier: None,
            },
        );
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &config).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let cost = calc
            .calculate("ollama/llama3.2", &TokenUsage::default())
            .unwrap();
        assert_eq!(cost.total_cost, NanoUsd::zero());
        assert!(!logs_contain("model_not_in_pricing_db"));
    }

    #[test]
    fn test_override_creates_new_entry() {
        let mut config = default_config();
        config.overrides.insert(
            "ollama/llama3.2".into(),
            PricingOverride {
                input_per_token: 0.0000005,
                output_per_token: 0.0000005,
                context_window: 128_000,
                cache_read_multiplier: None,
            },
        );
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &config).unwrap();
        {
            let guard = db.read();
            assert!(guard.lookup("ollama/llama3.2", None).is_some());
        }
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 500_000,
            ..Default::default()
        };
        let cost = calc.calculate("ollama/llama3.2", &usage).unwrap();
        // 0.0000005 → 500 nano/token → 1M*500 + 500K*500 = 750_000_000 nano = $0.75
        assert_eq!(cost.total_cost, NanoUsd(750_000_000));
    }

    #[test]
    fn test_tiered_below_threshold() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 50_000,
            output_tokens: 1000,
            ..Default::default()
        };
        let cost = calc.calculate("gemini-1.5-pro-002", &usage).unwrap();
        // Tier 0: 1250 nano input, 5000 nano output → 50_000*1250 + 1000*5000 = 67_500_000
        assert_eq!(cost.total_cost, NanoUsd(67_500_000));
    }

    #[test]
    fn test_tiered_above_threshold() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 200_000,
            output_tokens: 1000,
            ..Default::default()
        };
        let cost = calc.calculate("gemini-1.5-pro-002", &usage).unwrap();
        // Tier 1: 2500 nano input, 10000 nano output → 200_000*2500 + 1000*10000 = 510_000_000
        assert_eq!(cost.total_cost, NanoUsd(510_000_000));
    }

    #[test]
    fn test_tiered_at_exact_boundary() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 128_001,
            output_tokens: 1,
            ..Default::default()
        };
        let cost = calc.calculate("gemini-1.5-pro-002", &usage).unwrap();
        // 128001 >= 128001 → tier 1: 2500 nano input, 10000 nano output → 320_012_500
        assert_eq!(cost.total_cost, NanoUsd(320_012_500));
    }

    /// batch discount halves cost for OpenAI models with batch multipliers.
    #[test]
    fn test_batch_discount_applied() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage_batch = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 500_000,
            batch: true,
            ..Default::default()
        };
        let usage_non_batch = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 500_000,
            batch: false,
            ..Default::default()
        };
        let cost_batch = calc.calculate("gpt-4.1", &usage_batch).unwrap();
        let cost_non_batch = calc.calculate("gpt-4.1", &usage_non_batch).unwrap();
        assert_eq!(
            cost_batch.total_cost,
            NanoUsd(cost_non_batch.total_cost.0 / 2),
            "batch=true must halve cost when batch_input_multiplier=0.5 and batch_output_multiplier=0.5"
        );
    }

    /// Batch discount applies to cache read, cache write 5m/1h, and thinking costs too.
    #[test]
    fn test_batch_discount_applies_to_cache_costs() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage_batch = TokenUsage {
            cache_read_input_tokens: 500_000,
            cache_write_5m_tokens: 200_000,
            cache_write_1h_tokens: 100_000,
            batch: true,
            ..Default::default()
        };
        let usage_non_batch = TokenUsage {
            cache_read_input_tokens: 500_000,
            cache_write_5m_tokens: 200_000,
            cache_write_1h_tokens: 100_000,
            batch: false,
            ..Default::default()
        };
        // Use claude-sonnet-4-6 which has cache multipliers + batch multipliers
        let cost_batch = calc.calculate("claude-sonnet-4-6", &usage_batch).unwrap();
        let cost_non_batch = calc
            .calculate("claude-sonnet-4-6", &usage_non_batch)
            .unwrap();
        assert_eq!(
            cost_batch.cached_input_cost,
            NanoUsd(cost_non_batch.cached_input_cost.0 / 2),
            "batch must halve cached_input_cost"
        );
        assert_eq!(
            cost_batch.cache_write_5m_cost,
            NanoUsd(cost_non_batch.cache_write_5m_cost.0 / 2),
            "batch must halve cache_write_5m_cost"
        );
        assert_eq!(
            cost_batch.cache_write_1h_cost,
            NanoUsd(cost_non_batch.cache_write_1h_cost.0 / 2),
            "batch must halve cache_write_1h_cost"
        );
    }

    /// batch=false receives no discount.
    #[test]
    fn test_batch_no_discount_when_false() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 500_000,
            batch: false,
            ..Default::default()
        };
        let cost = calc.calculate("gpt-4.1", &usage).unwrap();
        assert_eq!(cost.total_cost, NanoUsd(6_000_000_000));
    }

    /// tier_threshold_override selects tier based on input+cached (Google AI Studio).
    #[test]
    fn test_tier_threshold_override() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 50_000,
            output_tokens: 1_000,
            cache_read_input_tokens: 100_000,
            tier_threshold_override: Some(150_001),
            ..Default::default()
        };
        let cost = calc.calculate("gemini-1.5-pro-002", &usage).unwrap();
        // 150001 >= 128001 → tier 1: 2500 nano input, 10000 output, cache_read_mult 0.25
        // input: 50_000*2500=125M, output: 1_000*10000=10M, cache: 100_000*2500*0.25=62.5M
        assert_eq!(cost.total_cost, NanoUsd(197_500_000));
    }

    /// image cost applied when tier has image_per_unit.
    #[test]
    fn test_image_cost_applied() {
        let json = r#"{"models":{"test-img":{"provider":"test","context_window":1000,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0,"output_per_token":0,"image_per_unit":0.01}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            image_count: 2,
            ..Default::default()
        };
        let cost = calc.calculate("test-img", &usage).unwrap();
        // 2 × $0.01 = 20_000_000 nano-USD
        assert_eq!(cost.image_cost, NanoUsd(20_000_000));
        assert_eq!(cost.total_cost, NanoUsd(20_000_000));
    }

    /// audio cost applied when tier has audio_per_second.
    #[test]
    fn test_audio_cost_applied() {
        let json = r#"{"models":{"test-audio":{"provider":"test","context_window":1000,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0,"output_per_token":0,"audio_per_second":0.006}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            audio_seconds: 5.0,
            ..Default::default()
        };
        let cost = calc.calculate("test-audio", &usage).unwrap();
        // 5.0 × $0.006 = 30_000_000 nano-USD
        assert_eq!(cost.audio_cost, NanoUsd(30_000_000));
        assert_eq!(cost.total_cost, NanoUsd(30_000_000));
    }

    /// image and audio costs combined.
    #[test]
    fn test_image_audio_combined() {
        let json = r#"{"models":{"test-multimodal":{"provider":"test","context_window":1000,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0,"output_per_token":0,"image_per_unit":0.01,"audio_per_second":0.006}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            image_count: 2,
            audio_seconds: 5.0,
            ..Default::default()
        };
        let cost = calc.calculate("test-multimodal", &usage).unwrap();
        assert_eq!(cost.image_cost, NanoUsd(20_000_000));
        assert_eq!(cost.audio_cost, NanoUsd(30_000_000));
        assert_eq!(cost.total_cost, NanoUsd(50_000_000));
    }

    /// image_cost zero when tier has no image_per_unit.
    #[test]
    fn test_image_zero_when_no_rate() {
        let json = r#"{"models":{"test-no-img":{"provider":"test","context_window":1000,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0,"output_per_token":0}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            image_count: 5,
            ..Default::default()
        };
        let cost = calc.calculate("test-no-img", &usage).unwrap();
        assert_eq!(cost.image_cost, NanoUsd::zero());
    }

    /// batch discount applies to image_cost (batch_input_multiplier) and audio_cost
    /// (batch_output_multiplier). Documents invariant that multimodal costs get batch discount.
    #[test]
    fn test_batch_discount_applies_to_image_and_audio_cost() {
        let json = r#"{"models":{"batch-multimodal":{"provider":"test","context_window":1000,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0,"output_per_token":0,"image_per_unit":0.01,"audio_per_second":0.006,"batch_input_multiplier":0.5,"batch_output_multiplier":0.5}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage_batch = TokenUsage {
            image_count: 4,
            audio_seconds: 10.0,
            batch: true,
            ..Default::default()
        };
        let usage_non_batch = TokenUsage {
            image_count: 4,
            audio_seconds: 10.0,
            batch: false,
            ..Default::default()
        };
        let cost_batch = calc.calculate("batch-multimodal", &usage_batch).unwrap();
        let cost_non_batch = calc
            .calculate("batch-multimodal", &usage_non_batch)
            .unwrap();
        // 4 × $0.01 = 40M, 10 × $0.006 = 60M → non-batch total 100M. Batch halves both → 50M.
        assert_eq!(cost_batch.total_cost, NanoUsd(50_000_000));
        assert_eq!(cost_non_batch.total_cost, NanoUsd(100_000_000));
        assert_eq!(cost_batch.image_cost, NanoUsd(20_000_000)); // 40M × 0.5
        assert_eq!(cost_batch.audio_cost, NanoUsd(30_000_000)); // 60M × 0.5
    }

    /// tier_threshold_override None uses input_tokens only (Vertex AI).
    #[test]
    fn test_tier_no_override_uses_input_tokens() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 50_000,
            output_tokens: 1_000,
            cache_read_input_tokens: 100_000,
            tier_threshold_override: None,
            ..Default::default()
        };
        let cost = calc.calculate("gemini-1.5-pro-002", &usage).unwrap();
        // 50K < 128001 → tier 0: 1250 nano input, 5000 output, cache_read_mult 0.25
        // input: 50_000*1250=62.5M, output: 1_000*5000=5M, cache: 100_000*1250*0.25=31.25M
        assert_eq!(cost.total_cost, NanoUsd(98_750_000));
    }

    #[test]
    fn test_validation_non_ascending_thresholds() {
        let json = r#"{"models":{"x":{"provider":"p","context_window":1000,"aliases":[],"tiers":[{"threshold":100,"input_per_token":0.001,"output_per_token":0.001},{"threshold":50,"input_per_token":0.002,"output_per_token":0.002}]}}}"#;
        let err = PricingDb::load(json.as_bytes(), &default_config()).unwrap_err();
        assert!(matches!(err, PricingError::InvalidDb(_)));
        assert!(err.to_string().contains("ascending"));
    }

    /// Documents that canonical model ID uniqueness is enforced. JSON object-key
    /// uniqueness makes true duplicate canonicals impossible from the JSON source.
    /// The alias collision path (test_validation_alias_collision) exercises the same
    /// insert rejection logic when two entries claim the same alias.
    #[test]
    fn test_canonical_uniqueness_documented() {
        let json = r#"{"models":{"a":{"provider":"p","context_window":1000,"aliases":["x"],"tiers":[{"threshold":0,"input_per_token":0.001,"output_per_token":0.001}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let guard = db.read();
        assert_eq!(
            guard.by_canonical.len(),
            1,
            "single model yields single canonical"
        );
        assert_eq!(
            guard.by_canonical.get("a").map(|e| e.model_id.as_str()),
            Some("a")
        );
    }

    #[test]
    fn test_validation_alias_collision() {
        let json = r#"{"models":{"a":{"provider":"p","context_window":1000,"aliases":["x"],"tiers":[{"threshold":0,"input_per_token":0.001,"output_per_token":0.001}]},"b":{"provider":"q","context_window":2000,"aliases":["x"],"tiers":[{"threshold":0,"input_per_token":0.002,"output_per_token":0.002}]}}}"#;
        let err = PricingDb::load(json.as_bytes(), &default_config()).unwrap_err();
        assert!(matches!(err, PricingError::InvalidDb(_)));
        assert!(err.to_string().contains("collides"));
    }

    #[test]
    fn test_startup_parse_failure() {
        let err = PricingDb::load(b"{ invalid json", &default_config()).unwrap_err();
        assert!(matches!(err, PricingError::ParseFailure(_)));
    }

    /// Invalid overrides (e.g. negative prices) are validated in apply_override.
    /// Config layer catches these first; domain logs WARN and skips the override.
    #[traced_test]
    #[test]
    fn test_apply_override_invalid_logs_warn() {
        let mut config = default_config();
        config.overrides.insert(
            "ollama/llama3.2".into(),
            PricingOverride {
                input_per_token: -0.01,
                output_per_token: 0.0,
                context_window: 128_000,
                cache_read_multiplier: None,
            },
        );
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &config).unwrap();
        assert!(
            logs_contain("pricing override validation failed"),
            "expected WARN when override has invalid values"
        );
        let guard = db.read();
        assert!(
            guard.lookup("ollama/llama3.2", None).is_none(),
            "invalid override must be skipped; model should not be in DB"
        );
    }

    // --- Property-based invariants (proptest) ---

    fn assert_component_sum_invariant(cost: &CostBreakdown) {
        let sum = cost
            .input_cost
            .0
            .saturating_add(cost.output_cost.0)
            .saturating_add(cost.cached_input_cost.0)
            .saturating_add(cost.cache_write_5m_cost.0)
            .saturating_add(cost.cache_write_1h_cost.0)
            .saturating_add(cost.thinking_cost.0)
            .saturating_add(cost.image_cost.0)
            .saturating_add(cost.audio_cost.0);
        assert_eq!(
            cost.total_cost.0, sum,
            "total_cost must equal sum of components"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        #[test]
        fn prop_cost_breakdown_total_equals_sum_of_components(usage in token_usage_strategy()) {
            let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
            let calc = BundledCostCalculator::new(db_holder(db));
            let cost = calc.calculate("gpt-4.1", &usage).unwrap();
            assert_component_sum_invariant(&cost);
        }

        #[test]
        fn prop_zero_usage_zero_cost(_ in token_usage_strategy()) {
            let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
            let calc = BundledCostCalculator::new(db_holder(db));
            let cost = calc.calculate("gpt-4.1", &TokenUsage::default()).unwrap();
            assert_eq!(cost.total_cost, NanoUsd::zero());
        }

        /// Fields 6 (image_count) and 7 (audio_seconds) are vacuously monotone on gpt-4.1
        /// since it has no image_per_unit/audio_per_second; covered by isolated unit tests.
        #[test]
        fn prop_monotonic_more_tokens_higher_cost(
            base in token_usage_strategy(),
            field in 0u32..8u32,
        ) {
            let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
            let calc = BundledCostCalculator::new(db_holder(db));
            let cost_base = calc.calculate("gpt-4.1", &base).unwrap();

            let mut inc = base.clone();
            match field {
                0 => inc.input_tokens = inc.input_tokens.saturating_add(1),
                1 => inc.output_tokens = inc.output_tokens.saturating_add(1),
                2 => inc.cache_read_input_tokens = inc.cache_read_input_tokens.saturating_add(1),
                3 => inc.cache_write_5m_tokens = inc.cache_write_5m_tokens.saturating_add(1),
                4 => inc.cache_write_1h_tokens = inc.cache_write_1h_tokens.saturating_add(1),
                5 => inc.thinking_tokens = inc.thinking_tokens.saturating_add(1),
                6 => inc.image_count = inc.image_count.saturating_add(1),
                _ => inc.audio_seconds = (inc.audio_seconds + 0.1).max(0.0),
            }
            let cost_inc = calc.calculate("gpt-4.1", &inc).unwrap();
            assert!(
                cost_inc.total_cost >= cost_base.total_cost,
                "more tokens must not decrease cost"
            );
        }
    }

    // --- Table-driven varied-pattern tests ---

    #[test]
    fn test_table_gpt41_plain_input_output() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 10_000,
            output_tokens: 2_000,
            ..Default::default()
        };
        let cost = calc.calculate("gpt-4.1", &usage).unwrap();
        assert_component_sum_invariant(&cost);
        assert!(cost.total_cost > NanoUsd::zero());
    }

    #[test]
    fn test_table_gpt41_cache_read() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 5_000,
            output_tokens: 1_000,
            cache_read_input_tokens: 2_000,
            ..Default::default()
        };
        let cost = calc.calculate("gpt-4.1", &usage).unwrap();
        assert_component_sum_invariant(&cost);
        assert!(cost.cached_input_cost > NanoUsd::zero());
    }

    #[test]
    fn test_table_claude_sonnet_46_cache_write() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 500,
            cache_write_5m_tokens: 100,
            cache_write_1h_tokens: 50,
            ..Default::default()
        };
        let cost = calc.calculate("claude-sonnet-4-6", &usage).unwrap();
        assert_component_sum_invariant(&cost);
        assert!(cost.cache_write_5m_cost > NanoUsd::zero());
        assert!(cost.cache_write_1h_cost > NanoUsd::zero());
        // 1h cache should cost more than 5m per token (2.0x vs 1.25x multiplier)
        // 5m: 100 tokens × 3000 nano × 1.25 = 375_000 nano
        // 1h: 50 tokens × 3000 nano × 2.0 = 300_000 nano
        // Per-token: 1h rate (6000 nano/token) > 5m rate (3750 nano/token)
        let per_token_5m = cost.cache_write_5m_cost.0 / 100;
        let per_token_1h = cost.cache_write_1h_cost.0 / 50;
        assert!(
            per_token_1h > per_token_5m,
            "1h cache creation should cost more per token than 5m"
        );
    }

    #[test]
    fn test_table_gemini_25_pro_thinking() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            thinking_tokens: 30,
            ..Default::default()
        };
        let cost = calc.calculate("gemini-2.5-pro", &usage).unwrap();
        assert_component_sum_invariant(&cost);
        assert!(cost.thinking_cost > NanoUsd::zero());
    }

    #[test]
    fn test_table_inline_tiered_below_threshold() {
        let json = r#"{"models":{"test-tiered":{"provider":"test","context_window":200000,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000004},{"threshold":100000,"input_per_token":0.000002,"output_per_token":0.000008}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 50_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let cost = calc.calculate("test-tiered", &usage).unwrap();
        assert_component_sum_invariant(&cost);
        // Tier 0: 1000 nano input, 4000 nano output
        assert_eq!(cost.total_cost, NanoUsd(50_000 * 1000 + 1_000 * 4000));
    }

    #[test]
    fn test_table_inline_tiered_above_threshold() {
        let json = r#"{"models":{"test-tiered":{"provider":"test","context_window":200000,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000004},{"threshold":100000,"input_per_token":0.000002,"output_per_token":0.000008}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 150_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let cost = calc.calculate("test-tiered", &usage).unwrap();
        assert_component_sum_invariant(&cost);
        // Tier 1: 2000 nano input, 8000 nano output
        assert_eq!(cost.total_cost, NanoUsd(150_000 * 2000 + 1_000 * 8000));
    }
}
