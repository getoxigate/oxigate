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
use crate::domain::usage_accounting::{
    CacheWriteClass, CacheWriteClassRegistry, ClassRegistryError, CostStatus, PricingContext,
};

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
    /// Multiplier per cache-write class, keyed by canonical duration (e.g. `"5m"`, `"1h"`).
    ///
    /// A positive quantity in a configured class prices at that multiplier exactly, including an
    /// explicit `0.0`. An unconfigured class falls back to the tier-local highest configured
    /// multiplier, or `1.0` when the map is empty — never another model's or tier's price.
    #[serde(default)]
    pub cache_write_multipliers: HashMap<String, f64>,
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
///
/// Carries the [`CacheWriteClassRegistry`] derived from this generation alongside the catalogue
/// itself: the registry is computed once at load, from the *union* of configured classes across
/// every entry and tier after overrides are applied, so it never drifts from the catalogue a
/// request prices against.
#[derive(Clone)]
pub struct PricingDb(Arc<RwLock<PricingDbInner>>, Arc<CacheWriteClassRegistry>);

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
    /// The union of configured cache-write classes across the final effective database exceeds
    /// the reserved capacity.
    #[error(transparent)]
    ClassRegistry(#[from] ClassRegistryError),
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
    /// When the exact model ID is not found, falls back to stripping a trailing provider-revision
    /// suffix so a dated or versioned wire ID resolves to base-model pricing:
    /// - OpenAI streaming responses append `-YYYY-MM-DD` (e.g. `gpt-4o-2024-08-06` → `gpt-4o`).
    /// - AWS Bedrock model IDs append `-YYYYMMDD-vN:M`, no internal dashes in the date, unlike
    ///   OpenAI's suffix (e.g. `anthropic.claude-sonnet-4-6-20251001-v1:0` →
    ///   `anthropic.claude-sonnet-4-6`) — a real client sends this shape verbatim; the bundled
    ///   entry is keyed on the undated name.
    pub fn lookup<'a>(&'a self, model: &str, _provider: Option<&str>) -> Option<&'a PricingEntry> {
        if let Some(entry) = self.resolve_by_canonical_or_alias(model) {
            return Some(entry);
        }
        if let Some(base) = strip_openai_date_suffix(model)
            && let Some(entry) = self.resolve_by_canonical_or_alias(base)
        {
            return Some(entry);
        }
        if let Some(base) = strip_bedrock_version_suffix(model)
            && let Some(entry) = self.resolve_by_canonical_or_alias(base)
        {
            return Some(entry);
        }
        None
    }

    fn resolve_by_canonical_or_alias<'a>(&'a self, model: &str) -> Option<&'a PricingEntry> {
        if let Some(entry) = self.by_canonical.get(model) {
            return Some(entry);
        }
        self.by_alias
            .get(model)
            .and_then(|canon| self.by_canonical.get(canon))
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

/// Strips a trailing OpenAI-style `-YYYY-MM-DD` revision suffix (e.g. `-2024-08-06`), so a
/// streaming response's dated model ID resolves to base-model pricing (e.g. `gpt-4o`).
fn strip_openai_date_suffix(model: &str) -> Option<&str> {
    if model.len() <= 11 {
        return None;
    }
    let suffix = model.get(model.len() - 11..)?; // "-YYYY-MM-DD"
    let ok = suffix.starts_with('-')
        && suffix
            .get(1..5)
            .is_some_and(|s| s.bytes().all(|b| b.is_ascii_digit()))
        && suffix.as_bytes().get(5) == Some(&b'-')
        && suffix
            .get(6..8)
            .is_some_and(|s| s.bytes().all(|b| b.is_ascii_digit()))
        && suffix.as_bytes().get(8) == Some(&b'-')
        && suffix
            .get(9..11)
            .is_some_and(|s| s.bytes().all(|b| b.is_ascii_digit()));
    if !ok {
        return None;
    }
    let base = model.get(..model.len() - 11)?;
    (!base.is_empty()).then_some(base)
}

/// Strips a trailing AWS Bedrock revision suffix of the form `-YYYYMMDD-vN:M` — an 8-digit date
/// with no internal dashes, unlike the OpenAI-style suffix above, followed by a version and
/// revision number (e.g. `-20251001-v1:0`). A real Bedrock request carries this shape verbatim as
/// its model ID (e.g. `anthropic.claude-sonnet-4-6-20251001-v1:0`); the bundled pricing entry is
/// keyed on the undated canonical name (`anthropic.claude-sonnet-4-6`), so without this fallback
/// every such request misses the pricing lookup entirely and falls to cost-unavailable.
fn strip_bedrock_version_suffix(model: &str) -> Option<&str> {
    let (base, rest) = model.rsplit_once("-v")?;
    let (major, minor) = rest.split_once(':')?;
    if major.is_empty()
        || minor.is_empty()
        || !major.bytes().all(|b| b.is_ascii_digit())
        || !minor.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    if base.len() < 8 {
        return None;
    }
    let date = base.get(base.len() - 8..)?;
    if !date.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let prefix = base.get(..base.len() - 8)?.strip_suffix('-')?;
    (!prefix.is_empty()).then_some(prefix)
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
                for t in &mut tiers {
                    t.cache_write_multipliers =
                        canonicalize_cache_write_multipliers(&t.cache_write_multipliers);
                }
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

        // The registry is the union of configured classes across every entry and tier of the
        // *final* effective database — computed once, after every override, so a transient union
        // that would exceed the cap on one `HashMap` iteration order and not another can never
        // decide whether load succeeds.
        let classes = by_canonical
            .values()
            .flat_map(|e| e.tiers.iter())
            .flat_map(|t| t.cache_write_multipliers.keys())
            .filter_map(|k| CacheWriteClass::canonicalize(k));
        let registry = CacheWriteClassRegistry::from_classes(classes)?;

        Ok(Self(
            Arc::new(RwLock::new(PricingDbInner {
                by_canonical,
                by_alias,
            })),
            Arc::new(registry),
        ))
    }

    /// The cache-write class registry derived from this generation's final effective database.
    #[must_use]
    pub fn registry(&self) -> &CacheWriteClassRegistry {
        &self.1
    }
}

/// Exclusive upper bound for any value about to be cast `as u64`: `2^64`.
///
/// `u64::MAX as f64` is **not** this boundary. `u64::MAX` is `2^64 - 1`, which has 64 significant
/// bits and is not exactly representable in an `f64` mantissa (52 bits); it rounds *up* to `2^64`
/// when converted. A check written as `value <= u64::MAX as f64` therefore silently accepts
/// `value == 2^64` itself — and `2^64_f64 as u64` saturates to `u64::MAX` rather than being
/// rejected, the same silent-ceiling failure the representability policy exists to close. `2^64`
/// is a power of two and so is exactly representable in `f64`; comparing against it directly, with
/// a strict `<`, is the boundary that actually matches "fits in a `u64`."
const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;

/// Whether a value is finite, non-negative, and strictly below [`TWO_POW_64`] — the single
/// boundary check every `f64 -> u64` money conversion on this path applies before the cast.
fn representable_as_u64(value: f64) -> bool {
    // `Range::contains` alone is sufficient: every comparison against `NaN` is `false`, so a
    // non-finite value fails the range check without a separate `is_finite()` guard.
    (0.0..TWO_POW_64).contains(&value)
}

/// Whether a USD rate is finite, non-negative and representable as nano-USD in a `u64`.
///
/// Every base and optional per-token/per-unit rate must pass this at load — otherwise an
/// unusable catalogue fails loudly at startup instead of miscomputing at request time. `< 0.0`
/// alone is not sufficient: it is `false` for `NaN` too, so a NaN rate previously loaded clean.
///
/// `pub(crate)` so `config.rs` applies the same boundary to a `PricingOverride`'s
/// `input_per_token`/`output_per_token` before the override ever reaches `apply_override`'s own
/// (defensive) validation.
pub(crate) fn rate_is_representable(rate_usd: f64) -> bool {
    representable_as_u64(rate_usd * 1_000_000_000.0)
}

/// Validates one cache-write multiplier map: every key must canonicalize to a duration class,
/// every value must fall in the existing multiplier range, and two keys folding to the same
/// canonical class must agree on the value — otherwise which one is billed is ambiguous.
///
/// `context` names where the map came from (a model/tier pair, or a config override), so
/// `pricing.rs` and `config.rs` share one error message shape without duplicating this logic.
pub(crate) fn validate_cache_write_multipliers(
    context: &str,
    map: &HashMap<String, f64>,
) -> Vec<String> {
    let mut errors = Vec::new();
    let mut canonical: HashMap<CacheWriteClass, (&str, f64)> = HashMap::new();
    for (raw_key, &value) in map {
        let Some(class) = CacheWriteClass::canonicalize(raw_key) else {
            errors.push(format!(
                "{context}: cache_write_multipliers key '{raw_key}' is not a canonical cache-write duration"
            ));
            continue;
        };
        if !(0.0..=10.0).contains(&value) {
            errors.push(format!(
                "{context}: cache_write_multipliers['{raw_key}'] must be in [0.0, 10.0]"
            ));
            continue;
        }
        match canonical.get(&class) {
            Some(&(existing_key, existing_value)) if existing_value != value => {
                errors.push(format!(
                    "{context}: cache_write_multipliers keys '{existing_key}' and '{raw_key}' both name class '{class}' with different values"
                ));
            }
            Some(_) => {}
            None => {
                canonical.insert(class, (raw_key, value));
            }
        }
    }
    errors
}

/// Rewrites a cache-write multiplier map's keys to their canonical spelling (e.g. `"05m"` ->
/// `"5m"`), so a request can look a class up by its canonical name regardless of how the source
/// spelled it. Callers run this only after [`validate_cache_write_multipliers`] found no errors,
/// so every key canonicalizes.
fn canonicalize_cache_write_multipliers(map: &HashMap<String, f64>) -> HashMap<String, f64> {
    map.iter()
        .filter_map(|(k, &v)| CacheWriteClass::canonicalize(k).map(|c| (c.as_str().to_string(), v)))
        .collect()
}

/// The tier-local fallback multiplier: `max(highest configured cache-write multiplier in this
/// tier, 1.0)` — never another model's or tier's price, and never less than the full input
/// rate. An empty map has no multiplier to raise it, so it falls back to `1.0`.
///
/// Applied to any cache-write tokens this tier does not itself price exactly: a class absent
/// from the whole pricing DB, a class the DB configures elsewhere but not in this tier, and any
/// unmatched aggregate residual.
fn tier_cache_write_fallback_multiplier(tier: &PricingTier) -> f64 {
    tier.cache_write_multipliers
        .values()
        .copied()
        .fold(1.0_f64, f64::max)
}

/// The multiplier `calculate` prices one cache-write class at: the configured value when the
/// tier's map names the class, or [`tier_cache_write_fallback_multiplier`] otherwise.
fn cache_write_multiplier_or_fallback(tier: &PricingTier, class: &str) -> f64 {
    match tier.cache_write_multipliers.get(class) {
        Some(&v) => v,
        None => tier_cache_write_fallback_multiplier(tier),
    }
}

fn validate_entry(entry: &PricingEntry) -> Vec<String> {
    let mut errors = Vec::new();
    if entry.tiers.is_empty() {
        errors.push(format!("model {} has no tiers", entry.model_id));
    }
    for (i, t) in entry.tiers.iter().enumerate() {
        for (name, rate) in [
            ("input_per_token", Some(t.input_per_token)),
            ("output_per_token", Some(t.output_per_token)),
            ("thinking_per_token", t.thinking_per_token),
            ("image_per_unit", t.image_per_unit),
            ("audio_per_second", t.audio_per_second),
        ] {
            if let Some(v) = rate
                && !rate_is_representable(v)
            {
                errors.push(format!(
                    "model {} tier {}: {} must be finite, >= 0 and representable in nano-USD",
                    entry.model_id, i, name
                ));
            }
        }
        for (name, opt) in [
            ("cache_read_multiplier", t.cache_read_multiplier),
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
        errors.extend(validate_cache_write_multipliers(
            &format!("model {} tier {}", entry.model_id, i),
            &t.cache_write_multipliers,
        ));
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
    // A tier priced above the model's own context_window can never be reached — no request
    // can carry more input tokens than the model accepts. A stale re-import has shipped exactly
    // this before: unreachable pricing data, not merely undertested. context_window == 0 means
    // unknown/unconstrained (see the field doc above), so there is nothing to compare against.
    if entry.context_window != 0 {
        for t in &entry.tiers {
            if t.threshold > u64::from(entry.context_window) {
                errors.push(format!(
                    "model {}: tier threshold {} exceeds context_window {}",
                    entry.model_id, t.threshold, entry.context_window
                ));
            }
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
    // Complete-replacement semantics: an override that omits the map does not inherit the
    // baseline's, so the visible fallback applies rather than a merge with what was there before.
    let tier = PricingTier {
        threshold: 0,
        input_per_token: ov.input_per_token,
        output_per_token: ov.output_per_token,
        cache_read_multiplier: ov.cache_read_multiplier,
        cache_write_multipliers: ov.cache_write_multipliers.clone(),
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

    let mut tiers = entry.tiers.clone();
    for t in &mut tiers {
        t.cache_write_multipliers =
            canonicalize_cache_write_multipliers(&t.cache_write_multipliers);
    }
    by_canonical.insert(model_key.to_string(), PricingEntry { tiers, ..entry });
    for a in &aliases {
        by_alias.insert(a.clone(), model_key.to_string());
    }
}

/// Default cost calculator using the bundled pricing DB.
///
/// Prices against one of two generations, and which one is the request's choice, not this type's:
///
/// - a request whose usage carries a [`PricingContext`] is priced against **that** snapshot, so
///   the generation it was classified under is the generation it is billed under even if a Class
///   A SIGHUP reload lands mid-request;
/// - a request that carries none is priced against the holder, which always reflects the current
///   DB after a reload.
///
/// The holder is therefore an `Arc` to the live holder rather than a snapshot — that is what makes
/// the second case see reloads at all — but it is not consulted for the first.
pub struct BundledCostCalculator {
    db_holder: Arc<RwLock<PricingDb>>,
}

impl BundledCostCalculator {
    /// Creates a calculator over the live pricing holder.
    ///
    /// The holder is the fallback generation, read on each `calculate()` for requests that carry
    /// no pinned [`PricingContext`]; see the type docs for when it is bypassed.
    #[must_use]
    pub fn new(db_holder: Arc<RwLock<PricingDb>>) -> Self {
        Self { db_holder }
    }

    /// Snapshots this request's pricing generation — see [`snapshot_pricing_context`].
    #[must_use]
    pub fn pricing_context(&self) -> PricingContext {
        snapshot_pricing_context(&self.db_holder)
    }
}

/// Snapshots one request's pricing generation out of a hot-reload holder: a [`PricingDb`] clone
/// plus the [`CacheWriteClassRegistry`] derived from that same clone, acquired under one brief
/// read lock.
///
/// Taken once, before the upstream request is dispatched, so a request seeds its accumulator
/// against and prices against one generation. A reload installs a *new* [`PricingDb`] into the
/// holder rather than mutating this one, so the clone stays pinned to the generation it was taken
/// from — see [`PricingContext`].
///
/// This is the single way to derive a request generation from a holder; callers that hold the
/// holder directly (provider adapters) use it rather than reproducing the lock-and-derive steps.
#[must_use]
pub fn snapshot_pricing_context(holder: &Arc<RwLock<PricingDb>>) -> PricingContext {
    let db = holder.read().expect("pricing holder lock poisoned").clone();
    let registry = db.registry().clone();
    PricingContext::new(db, registry)
}

/// Converts a USD-per-token/unit rate, already validated at load as finite, non-negative and
/// representable (see [`rate_is_representable`]), into nano-USD.
///
/// Fallible rather than assumed: a rate reaching here that is not representable is a checked
/// failure on the money path, not a value for `as` to silently saturate or zero.
fn rate_to_nano_usd_per_token(rate_usd: f64) -> Result<u64, CostError> {
    if !rate_is_representable(rate_usd) {
        return Err(CostError::Pricing(format!(
            "rate {rate_usd} is not representable in nano-USD"
        )));
    }
    Ok((rate_usd * 1_000_000_000.0).round() as u64)
}

/// Multiplier as 1e9 scale (0.5 -> 500_000_000). Warns when clamp activates.
///
/// Every multiplier reaching here was already range-checked at load
/// ([`validate_entry`]/[`validate_cache_write_multipliers`]) or by `GatewayConfig::validate()`
/// for an override, so the clamp is defense in depth, not the primary check.
fn mult_to_1e9(m: f64) -> u64 {
    if m > 10.0 {
        warn!(
            multiplier = m,
            "pricing multiplier exceeds 10x, clamping to 10x"
        );
    }
    (m.clamp(0.0, 10.0) * 1_000_000_000.0).round() as u64
}

/// Prices one token quantity against a per-token rate and a dimensionless multiplier, both
/// already scaled to 1e9 fixed point. Every step is checked u128 arithmetic: this is the single
/// place a component's multiplication can leave the representable range, so it is also the
/// single place that turns that into `CostError::Pricing` for it.
fn priced_component(tokens: u64, rate_nano: u64, mult_1e9: u64) -> Result<NanoUsd, CostError> {
    let product = (tokens as u128)
        .checked_mul(rate_nano as u128)
        .and_then(|p| p.checked_mul(mult_1e9 as u128))
        .ok_or_else(|| CostError::Pricing("cost component overflowed u128".to_string()))?;
    u64::try_from(product / 1_000_000_000u128)
        .map(NanoUsd)
        .map_err(|_| CostError::Pricing("cost component not representable in nano-USD".to_string()))
}

/// Scales an already-computed cost by a dimensionless multiplier at 1e9 fixed point — the batch
/// discount application. Checked the same way as [`priced_component`].
fn apply_multiplier(cost: NanoUsd, mult_1e9: u64) -> Result<NanoUsd, CostError> {
    let product = (cost.0 as u128)
        .checked_mul(mult_1e9 as u128)
        .ok_or_else(|| CostError::Pricing("batch discount overflowed u128".to_string()))?;
    u64::try_from(product / 1_000_000_000u128)
        .map(NanoUsd)
        .map_err(|_| CostError::Pricing("batch discount not representable in nano-USD".to_string()))
}

/// Checked sum of every cost component. Deliberately not the `NanoUsd: Add` operator
/// (`ports.rs`), which saturates: a saturated total here would present a partial sum as a real
/// one, the same "wrong number wearing a confident status" the whole representability policy
/// exists to reject.
fn checked_total(components: [NanoUsd; 7]) -> Result<NanoUsd, CostError> {
    components
        .into_iter()
        .try_fold(0u64, |acc, c| acc.checked_add(c.0))
        .map(NanoUsd)
        .ok_or_else(|| CostError::Pricing("total cost overflowed nano-USD".to_string()))
}

/// Checked addition of two already-priced components — the cache-write accumulation loop's own
/// running total, kept separate from [`checked_total`] because it runs once per observed class
/// rather than once at the end.
fn checked_add(a: NanoUsd, b: NanoUsd) -> Result<NanoUsd, CostError> {
    a.0.checked_add(b.0)
        .map(NanoUsd)
        .ok_or_else(|| CostError::Pricing("cache-write cost overflowed nano-USD".to_string()))
}

impl CostCalculator for BundledCostCalculator {
    fn calculate(&self, model: &str, usage: &TokenUsage) -> Result<CostBreakdown, CostError> {
        // Price against the generation this request was accounted under, when one travelled with
        // it. A Class A reload can land between the pre-dispatch snapshot and this call; reading
        // the holder here instead would classify the cache write against one generation's
        // registry and bill it at another's rates — the split `PricingContext` exists to prevent.
        // Requests that carry no snapshot (providers that never seeded one) keep the live holder.
        let holder_guard;
        let db: &PricingDb = match usage.cache_write.pricing_context() {
            Some(context) => context.db(),
            None => {
                holder_guard = self.db_holder.read().expect("pricing holder lock poisoned");
                &holder_guard
            }
        };
        let inner = db.read();
        let entry = inner.lookup(model, None);

        match entry {
            Some(e) => {
                let tier = e.get_tier(usage.context_input_tokens());

                // A request whose accumulated components do not sum exactly to
                // `accounted_tokens` must not be priced — charging the parts that happen to fit
                // would bill a total that differs from the quantity persisted on the spend row
                // and counted against the budget. This is a quantity failure, one step upstream
                // of the money checks below, and takes the same request-wide path they do.
                if !usage.cache_write.partition_is_exact() {
                    return Err(CostError::Pricing(
                        "cache-write quantity does not partition exactly; refusing to price"
                            .to_string(),
                    ));
                }

                // Composite status: starts at the best value and is only ever pulled down,
                // never back up — see `CostStatus::worst`'s precedence.
                let mut status = CostStatus::Exact;
                if usage.cache_write.outcome().is_contradiction()
                    || usage.cache_write.duplicate().any()
                {
                    status = status.worst(CostStatus::Reconciled);
                }

                // Integer arithmetic: rates in nano-USD per token
                let input_rate = rate_to_nano_usd_per_token(tier.input_per_token)?;
                let output_rate = rate_to_nano_usd_per_token(tier.output_per_token)?;
                let mut input_cost = NanoUsd(
                    usage
                        .input_tokens
                        .checked_mul(input_rate)
                        .ok_or_else(|| CostError::Pricing("input cost overflowed".to_string()))?,
                );
                // Reasoning is charged separately below. Under a contract that reports it
                // inside the completion total, charging that total whole would bill the
                // reasoning subset twice.
                let mut output_cost = NanoUsd(
                    usage
                        .standard_output_tokens()
                        .checked_mul(output_rate)
                        .ok_or_else(|| CostError::Pricing("output cost overflowed".to_string()))?,
                );

                if usage.cache_read_input_tokens > 0 && tier.cache_read_multiplier.is_none() {
                    status = status.worst(CostStatus::RateFallback);
                }
                let cache_read_mult = mult_to_1e9(tier.cache_read_multiplier.unwrap_or(1.0));
                let mut cached_input_cost =
                    priced_component(usage.cache_read_input_tokens, input_rate, cache_read_mult)?;

                // Cache-write: exact for every class this tier itself configures, fallback for
                // everything else — a class the DB reserves elsewhere but this tier does not
                // price, a class unknown to the whole DB, and any unmatched aggregate residual
                // all share the same tier-local fallback rate.
                let mut cache_write_cost = NanoUsd::zero();
                for total in usage.cache_write.class_totals() {
                    let configured = tier
                        .cache_write_multipliers
                        .contains_key(total.class.as_str());
                    let mult = mult_to_1e9(cache_write_multiplier_or_fallback(
                        tier,
                        total.class.as_str(),
                    ));
                    let component = priced_component(total.tokens, input_rate, mult)?;
                    cache_write_cost = checked_add(cache_write_cost, component)?;
                    if !configured && total.tokens > 0 {
                        status = status.worst(CostStatus::RateFallback);
                    }
                }
                let fallback_tokens = usage.cache_write.fallback_tokens();
                if fallback_tokens > 0 {
                    let mult = mult_to_1e9(tier_cache_write_fallback_multiplier(tier));
                    let component = priced_component(fallback_tokens, input_rate, mult)?;
                    cache_write_cost = checked_add(cache_write_cost, component)?;
                    status = status.worst(CostStatus::RateFallback);
                }

                // Thinking is always charged at a documented rate — the tier's own
                // `thinking_per_token`, or the standard output rate when the model prices
                // reasoning the same as ordinary completion tokens. Both are contractual
                // derivations, not a fallback, so this never moves status.
                let thinking_rate = rate_to_nano_usd_per_token(
                    tier.thinking_per_token.unwrap_or(tier.output_per_token),
                )?;
                let mut thinking_cost = NanoUsd(
                    usage
                        .thinking_tokens
                        .checked_mul(thinking_rate)
                        .ok_or_else(|| {
                            CostError::Pricing("thinking cost overflowed".to_string())
                        })?,
                );

                // Image and audio have no fallback: a positive quantity with no configured rate
                // cannot be defended at any price, so the whole request fails closed instead of
                // silently billing zero for a real cost dimension.
                if usage.image_count > 0 && tier.image_per_unit.is_none() {
                    return Err(CostError::Pricing(
                        "image usage reported with no configured image rate".to_string(),
                    ));
                }
                let image_rate_nano =
                    rate_to_nano_usd_per_token(tier.image_per_unit.unwrap_or(0.0))?;
                let mut image_cost = NanoUsd(
                    usage
                        .image_count
                        .checked_mul(image_rate_nano)
                        .ok_or_else(|| CostError::Pricing("image cost overflowed".to_string()))?,
                );

                // rate_to_nano_usd_per_token converts USD→nano-USD; unit is seconds not tokens here
                if !usage.audio_seconds.is_finite() || usage.audio_seconds < 0.0 {
                    return Err(CostError::Pricing(format!(
                        "audio_seconds {} must be finite and >= 0",
                        usage.audio_seconds
                    )));
                }
                if usage.audio_seconds > 0.0 && tier.audio_per_second.is_none() {
                    return Err(CostError::Pricing(
                        "audio usage reported with no configured audio rate".to_string(),
                    ));
                }
                let audio_rate_nano =
                    rate_to_nano_usd_per_token(tier.audio_per_second.unwrap_or(0.0))?;
                let audio_product = usage.audio_seconds * audio_rate_nano as f64;
                if !representable_as_u64(audio_product) {
                    return Err(CostError::Pricing(
                        "audio cost is not representable in nano-USD".to_string(),
                    ));
                }
                let mut audio_cost = NanoUsd(audio_product.round() as u64);

                //: apply batch discount to all token cost components when usage.batch.
                // Cache costs are input-token-based; thinking is output-token-based.
                // Image (vision input) uses batch_input_multiplier; audio (e.g. TTS output) uses batch_output_multiplier.
                if usage.batch {
                    let batch_input_group = usage.input_tokens > 0
                        || usage.cache_read_input_tokens > 0
                        || usage.cache_write.accounted_tokens() > 0
                        || usage.image_count > 0;
                    let batch_output_group = usage.standard_output_tokens() > 0
                        || usage.thinking_tokens > 0
                        || usage.audio_seconds > 0.0;
                    if batch_input_group && tier.batch_input_multiplier.is_none() {
                        status = status.worst(CostStatus::RateFallback);
                    }
                    if batch_output_group && tier.batch_output_multiplier.is_none() {
                        status = status.worst(CostStatus::RateFallback);
                    }

                    let batch_in = mult_to_1e9(tier.batch_input_multiplier.unwrap_or(1.0));
                    let batch_out = mult_to_1e9(tier.batch_output_multiplier.unwrap_or(1.0));
                    input_cost = apply_multiplier(input_cost, batch_in)?;
                    output_cost = apply_multiplier(output_cost, batch_out)?;
                    cached_input_cost = apply_multiplier(cached_input_cost, batch_in)?;
                    cache_write_cost = apply_multiplier(cache_write_cost, batch_in)?;
                    thinking_cost = apply_multiplier(thinking_cost, batch_out)?;
                    image_cost = apply_multiplier(image_cost, batch_in)?;
                    audio_cost = apply_multiplier(audio_cost, batch_out)?;
                }

                let total_cost = checked_total([
                    input_cost,
                    output_cost,
                    cached_input_cost,
                    cache_write_cost,
                    thinking_cost,
                    image_cost,
                    audio_cost,
                ])?;

                // A successful response with no usable usage at all cannot claim confidence it
                // never established, even though nothing above found a reason to degrade it.
                let no_usable_usage = usage.input_tokens == 0
                    && usage.output_tokens == 0
                    && usage.cache_read_input_tokens == 0
                    && usage.cache_write.accounted_tokens() == 0
                    && usage.thinking_tokens == 0
                    && usage.image_count == 0
                    && usage.audio_seconds == 0.0;
                if no_usable_usage {
                    status = CostStatus::CostUnavailable;
                }

                Ok(CostBreakdown {
                    input_cost,
                    output_cost,
                    cached_input_cost,
                    cache_write_cost,
                    thinking_cost,
                    image_cost,
                    audio_cost,
                    total_cost,
                    status,
                })
            }
            // No usable pricing for this model. §2.2 treats missing base pricing exactly as it
            // treats a checked-arithmetic failure, so it takes the same request-wide path: the
            // caller maps this to the all-zero `cost-unavailable` breakdown this arm used to
            // return directly, and no billed amount moves. What changes is who says so — the
            // calculator reports the fact and finalization owns the one warning, instead of this
            // arm emitting a second event beside it. `model_not_in_pricing_db` stays the stable
            // token in the message so the operator signal survives the move.
            None => Err(CostError::Pricing(format!(
                "model_not_in_pricing_db: {model}"
            ))),
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
    use crate::domain::chat::ReasoningAccounting;
    use crate::domain::ports::{NanoUsd, TokenUsage};
    use crate::domain::usage_accounting::{CacheWriteAccounting, CacheWriteAccumulator};
    use proptest::prelude::*;
    use tracing_test::traced_test;

    /// Builds cache-write accounting state the way a request actually accumulates it: a
    /// registry seeded from the observed classes, then one detail observation per pair.
    ///
    /// `TokenUsage::cache_write` has no public constructor — the accumulator is the only way to
    /// produce one — so every test that needs cache-write tokens on a `TokenUsage` goes through
    /// this rather than a struct literal.
    fn cache_write_of(pairs: &[(&str, u64)]) -> CacheWriteAccounting {
        let registry = CacheWriteClassRegistry::from_classes(
            pairs
                .iter()
                .filter_map(|(k, _)| CacheWriteClass::canonicalize(k)),
        )
        .expect("registry");
        let mut acc = CacheWriteAccumulator::new(registry);
        for (raw_key, tokens) in pairs {
            if *tokens > 0 {
                acc.observe_detail(raw_key, CacheWriteClass::canonicalize(raw_key), *tokens);
            }
        }
        acc.finish()
    }

    /// The same shape `token_usage_strategy` used to build directly, kept as plain fields so a
    /// property test can mutate one dimension at a time before materializing a `TokenUsage`.
    #[derive(Clone, Debug)]
    struct RawUsage {
        input_tokens: u64,
        output_tokens: u64,
        cache_read_input_tokens: u64,
        cache_write_5m_tokens: u64,
        cache_write_1h_tokens: u64,
        thinking_tokens: u64,
        image_count: u64,
        audio_seconds: f64,
        reasoning_accounting: ReasoningAccounting,
    }

    impl RawUsage {
        fn to_token_usage(&self) -> TokenUsage {
            TokenUsage {
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
                cache_read_input_tokens: self.cache_read_input_tokens,
                cache_write: cache_write_of(&[
                    ("5m", self.cache_write_5m_tokens),
                    ("1h", self.cache_write_1h_tokens),
                ]),
                thinking_tokens: self.thinking_tokens,
                image_count: self.image_count,
                audio_seconds: self.audio_seconds,
                batch: false,
                reasoning_accounting: self.reasoning_accounting,
                ..Default::default()
            }
        }
    }

    fn token_usage_strategy() -> impl Strategy<Value = RawUsage> {
        (
            0u64..500_000u64,
            0u64..500_000u64,
            0u64..100_000u64,
            0u64..50_000u64,
            0u64..50_000u64,
            0u64..100_000u64,
            // gpt-4.1 (the model every user of this strategy prices against) has no configured
            // image_per_unit or audio_per_second, and a positive quantity with no
            // usable rate a request-wide `CostError::Pricing` rather than a silent zero — so both
            // stay fixed at zero here and are exercised by their own isolated unit tests instead.
            Just(0u64),
            Just(0.0f64),
            // Both contracts, so the reasoning carve-out is exercised against unconstrained
            // reasoning/completion pairs — including reasoning larger than the completion total,
            // where the standard output charge clamps to zero.
            prop_oneof![
                Just(ReasoningAccounting::Additive),
                Just(ReasoningAccounting::IncludedInOutput),
            ],
        )
            .prop_map(|(i, o, cr, c5, c1, th, img, aud, ra)| RawUsage {
                input_tokens: i,
                output_tokens: o,
                cache_read_input_tokens: cr,
                cache_write_5m_tokens: c5,
                cache_write_1h_tokens: c1,
                thinking_tokens: th,
                image_count: img,
                audio_seconds: aud,
                reasoning_accounting: ra,
            })
    }

    fn default_config() -> PricingConfig {
        PricingConfig::default()
    }

    fn db_holder(db: PricingDb) -> Arc<RwLock<PricingDb>> {
        Arc::new(RwLock::new(db))
    }

    /// Two-tier fixture for exercising tier selection, threshold boundaries and the cache
    /// multiplier.
    ///
    /// Synthetic on purpose. These tests are about *tier-selection logic*, so binding them to a
    /// real model's rates only couples them to data that legitimately moves: when a vendor
    /// reprices, the test fails for a reason unrelated to what it checks, and the standing fix is
    /// to edit the expected value to match — which detects change, not error. The constants below
    /// are arbitrary and permanent.
    const TIERED_FIXTURE: &str = r#"{"models":{"tiered-fixture":{
        "provider":"test","context_window":200000,"aliases":[],
        "tiers":[
          {"threshold":0,"input_per_token":0.00000125,"output_per_token":0.000005,
           "cache_read_multiplier":0.25},
          {"threshold":128001,"input_per_token":0.0000025,"output_per_token":0.00001,
           "cache_read_multiplier":0.25}
        ]}}}"#;

    fn tiered_fixture_calc() -> BundledCostCalculator {
        let db = PricingDb::load(TIERED_FIXTURE.as_bytes(), &default_config()).unwrap();
        BundledCostCalculator::new(db_holder(db))
    }

    #[test]
    fn test_parse_bundled_json() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let guard = db.read();
        // 2026-08-07 snapshot: 65 models (56 - 2 retired DeepSeek IDs + 11 new entries;
        // see docs/changelog.md "Refresh bundled pricing snapshot"), minus 7: seven Anthropic
        // entries retired by their vendor were removed from the bundled snapshot.
        assert_eq!(guard.by_canonical.len(), 58);
    }

    /// Every cache multiplier the Bedrock and OpenAI cache parsers price against is present, on
    /// **every** tier of the entry that carries it.
    ///
    /// This is a structural check on named entries, not a restatement of a price: it asserts the
    /// fields exist, never what they are worth, so a legitimate refresh cannot turn it red. It
    /// exists because absence here is silent and expensive. A missing `cache_read_multiplier`
    /// bills cache hits at `unwrap_or(1.0)` — the full input rate — and a missing cache-write
    /// class falls back to the tier's highest configured multiplier, which is not reliably
    /// conservative: an entry carrying `5m` but not `1h` prices a 1-hour write at 1.25x against
    /// a real 2.0x, under-billing it in the direction that lets budgets fail open.
    ///
    /// Per tier, not per entry: an earlier import left every long-context tier without its `1h`
    /// entry while the base tier looked complete.
    #[test]
    fn test_bundled_entries_carry_the_cache_multipliers_their_parsers_price() {
        // (model, cache-write classes every tier must configure)
        let required: &[(&str, &[&str])] = &[
            // Bedrock Converse reports cache-write TTLs as `5m` or `1h`, so both must be priced.
            ("anthropic.claude-sonnet-4-6", &["5m", "1h"]),
            // OpenAI's only supported cache TTL is 30 minutes. Both tiers, including the
            // long-context one — that is the tier a large cached prompt actually lands on.
            ("gpt-5.6-sol", &["30m"]),
            ("gpt-5.6-terra", &["30m"]),
            ("gpt-5.6-luna", &["30m"]),
        ];

        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let guard = db.read();

        for (model, classes) in required {
            let entry = guard
                .lookup(model, None)
                .unwrap_or_else(|| panic!("{model} is missing from the bundled pricing snapshot"));
            for tier in &entry.tiers {
                let at = format!("{model} tier {}", tier.threshold);
                assert!(
                    tier.cache_read_multiplier.is_some(),
                    "{at} has no cache_read_multiplier — cache reads would bill at the full \
                     input rate"
                );
                for class in *classes {
                    assert!(
                        tier.cache_write_multipliers.contains_key(*class),
                        "{at} has no `{class}` cache-write multiplier — a write in that class \
                         would price at the fallback rate, which is not reliably conservative"
                    );
                }
            }
        }
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

    /// A real AWS Bedrock request carries the fully versioned model ID verbatim
    /// (`anthropic.claude-sonnet-4-6-20251001-v1:0`), not the bundled entry's undated canonical
    /// name (`anthropic.claude-sonnet-4-6`). Before the Bedrock suffix fallback, this lookup
    /// missed entirely and fell to cost-unavailable for every real Bedrock request against this
    /// entry — the OpenAI-style `-YYYY-MM-DD` fallback cannot match it: the date has no internal
    /// dashes and the suffix also carries a `-vN:M` version tail.
    #[test]
    fn test_lookup_strips_bedrock_version_suffix_fallback() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let guard = db.read();
        let entry = guard.lookup("anthropic.claude-sonnet-4-6-20251001-v1:0", None);
        assert!(
            entry.is_some(),
            "the real Bedrock model ID should resolve via the version-suffix fallback"
        );
        assert_eq!(entry.unwrap().model_id, "anthropic.claude-sonnet-4-6");
    }

    /// The cross-region inference-profile prefix (`us.`) combines with the version suffix on real
    /// traffic. Stripping the suffix must land on the alias (`us.anthropic.claude-sonnet-4-6`),
    /// not just the bare canonical name.
    #[test]
    fn test_lookup_strips_bedrock_version_suffix_then_resolves_cross_region_alias() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let guard = db.read();
        let entry = guard.lookup("us.anthropic.claude-sonnet-4-6-20251001-v1:0", None);
        assert!(
            entry.is_some(),
            "the cross-region versioned ID should resolve via the alias after suffix-stripping"
        );
        assert_eq!(entry.unwrap().model_id, "anthropic.claude-sonnet-4-6");
    }

    /// A model ID that merely contains `-v` followed by digits and a colon, but does not end in
    /// an 8-digit date, must not be falsely stripped — the fallback must not fire on shapes it
    /// was not written for.
    #[test]
    fn test_lookup_bedrock_suffix_fallback_does_not_false_positive() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let guard = db.read();
        assert!(
            guard
                .lookup("anthropic.claude-sonnet-4-6-v1:0", None)
                .is_none()
        );
        assert!(
            guard
                .lookup("anthropic.claude-sonnet-4-6-2025100-v1:0", None)
                .is_none()
        ); // 7-digit date
        assert!(
            guard
                .lookup("anthropic.claude-sonnet-4-6-20251001-v1", None)
                .is_none()
        ); // no ":M"
    }

    /// An unpriced model is reported to the caller, not logged here: finalization owns the one
    /// warning, and a second event from this arm would double-WARN the same request.
    #[traced_test]
    #[test]
    fn test_calculate_unknown_reports_the_fact_and_emits_nothing() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage::default();
        let err = calc.calculate("unknown-xyz", &usage).unwrap_err();
        assert!(err.to_string().contains("model_not_in_pricing_db"));
        assert!(err.to_string().contains("unknown-xyz"));
        assert!(!logs_contain("model_not_in_pricing_db"));
    }

    #[traced_test]
    #[test]
    fn test_calculate_local_no_config_reports_the_fact_and_emits_nothing() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let err = calc
            .calculate("ollama/llama3.2", &TokenUsage::default())
            .unwrap_err();
        assert!(err.to_string().contains("model_not_in_pricing_db"));
        assert!(!logs_contain("model_not_in_pricing_db"));
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
                cache_write_multipliers: HashMap::new(),
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
                cache_write_multipliers: HashMap::new(),
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
                cache_write_multipliers: HashMap::new(),
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
        let calc = tiered_fixture_calc();
        let usage = TokenUsage {
            input_tokens: 50_000,
            output_tokens: 1000,
            ..Default::default()
        };
        let cost = calc.calculate("tiered-fixture", &usage).unwrap();
        // Tier 0: 1250 nano input, 5000 nano output → 50_000*1250 + 1000*5000 = 67_500_000
        assert_eq!(cost.total_cost, NanoUsd(67_500_000));
    }

    #[test]
    fn test_tiered_above_threshold() {
        let calc = tiered_fixture_calc();
        let usage = TokenUsage {
            input_tokens: 200_000,
            output_tokens: 1000,
            ..Default::default()
        };
        let cost = calc.calculate("tiered-fixture", &usage).unwrap();
        // Tier 1: 2500 nano input, 10000 nano output → 200_000*2500 + 1000*10000 = 510_000_000
        assert_eq!(cost.total_cost, NanoUsd(510_000_000));
    }

    /// The boundary is inclusive: `total_input_tokens >= tier.threshold` selects the upper tier,
    /// so 128_001 must price at tier 1 while 128_000 stays at tier 0.
    #[test]
    fn test_tiered_at_exact_boundary() {
        let calc = tiered_fixture_calc();
        let at_threshold = TokenUsage {
            input_tokens: 128_001,
            output_tokens: 1,
            ..Default::default()
        };
        let cost = calc.calculate("tiered-fixture", &at_threshold).unwrap();
        // 128001 >= 128001 → tier 1: 2500 nano input, 10000 nano output → 320_012_500
        assert_eq!(cost.total_cost, NanoUsd(320_012_500));

        // One token below: still tier 0, so the rate must be half.
        let below_threshold = TokenUsage {
            input_tokens: 128_000,
            output_tokens: 1,
            ..Default::default()
        };
        let cost = calc.calculate("tiered-fixture", &below_threshold).unwrap();
        assert_eq!(cost.total_cost, NanoUsd(128_000 * 1250 + 5000));
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

    /// Batch discount applies to cache read and the merged cache-write cost too.
    #[test]
    fn test_batch_discount_applies_to_cache_costs() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let cache_write = || cache_write_of(&[("5m", 200_000), ("1h", 100_000)]);
        let usage_batch = TokenUsage {
            cache_read_input_tokens: 500_000,
            cache_write: cache_write(),
            batch: true,
            ..Default::default()
        };
        let usage_non_batch = TokenUsage {
            cache_read_input_tokens: 500_000,
            cache_write: cache_write(),
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
            cost_batch.cache_write_cost,
            NanoUsd(cost_non_batch.cache_write_cost.0 / 2),
            "batch must halve the merged cache-write cost"
        );
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

    /// Positive image usage with no configured rate cannot be defended at any price, so
    /// the whole request fails closed rather than silently billing zero for a real cost
    /// dimension. Supersedes the old `test_image_zero_when_no_rate`, which asserted the
    /// silent-zero behaviour this story replaces.
    #[test]
    fn test_image_errors_when_no_rate() {
        let json = r#"{"models":{"test-no-img":{"provider":"test","context_window":1000,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0,"output_per_token":0}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            image_count: 5,
            ..Default::default()
        };
        let err = calc.calculate("test-no-img", &usage).unwrap_err();
        assert!(matches!(err, CostError::Pricing(_)));
    }

    /// The audio counterpart of [`test_image_errors_when_no_rate`].
    #[test]
    fn test_audio_errors_when_no_rate() {
        let json = r#"{"models":{"test-no-audio":{"provider":"test","context_window":1000,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0,"output_per_token":0}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            audio_seconds: 3.0,
            ..Default::default()
        };
        let err = calc.calculate("test-no-audio", &usage).unwrap_err();
        assert!(matches!(err, CostError::Pricing(_)));
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

    /// The tier comparator sums plain input and cache-read tokens, not plain input alone.
    #[test]
    fn test_tier_comparator_includes_cache_read_tokens() {
        let calc = tiered_fixture_calc();
        let usage = TokenUsage {
            input_tokens: 50_000,
            output_tokens: 1_000,
            cache_read_input_tokens: 100_000,
            ..Default::default()
        };
        let cost = calc.calculate("tiered-fixture", &usage).unwrap();
        // comparator = 50_000 + 100_000 = 150_000 >= 128_001 → tier 1:
        // 2500 nano input, 10000 output, cache_read_mult 0.25
        // input: 50_000*2500=125M, cache: 100_000*2500*0.25=62.5M, output: 1_000*10000=10M
        assert_eq!(cost.input_cost, NanoUsd(125_000_000));
        assert_eq!(cost.cached_input_cost, NanoUsd(62_500_000));
        assert_eq!(cost.output_cost, NanoUsd(10_000_000));
        assert_eq!(cost.total_cost, NanoUsd(197_500_000));
    }

    /// Cache-read tokens contribute to cost below the tier threshold — isolates cache
    /// arithmetic from tier movement.
    #[test]
    fn test_tier_comparator_below_threshold_cache_read_priced() {
        let calc = tiered_fixture_calc();
        let usage = TokenUsage {
            input_tokens: 10_000,
            output_tokens: 1_000,
            cache_read_input_tokens: 20_000,
            ..Default::default()
        };
        let cost = calc.calculate("tiered-fixture", &usage).unwrap();
        // comparator = 10_000 + 20_000 = 30_000 < 128_001 → tier 0
        // input: 10_000*1250=12.5M, cache: 20_000*1250*0.25=6.25M, output: 1_000*5000=5M
        assert_eq!(cost.input_cost, NanoUsd(12_500_000));
        assert_eq!(cost.cached_input_cost, NanoUsd(6_250_000));
        assert_eq!(cost.output_cost, NanoUsd(5_000_000));
        assert_eq!(cost.total_cost, NanoUsd(23_750_000));
    }

    /// Reasoning tokens bill additively at the output rate — today's behaviour, pinned so a
    /// future carve-out cannot land silently.
    #[test]
    fn test_tier_comparator_excludes_thinking_tokens() {
        let calc = tiered_fixture_calc();
        let usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 2_000,
            thinking_tokens: 500,
            ..Default::default()
        };
        let cost = calc.calculate("tiered-fixture", &usage).unwrap();
        // comparator = 1_000 (thinking is not a context-window bucket) < 128_001 → tier 0
        // input: 1_000*1250=1.25M, output: 2_000*5000=10M, thinking: 500*5000=2.5M
        assert_eq!(cost.total_cost, NanoUsd(13_750_000));
    }

    /// The cache-crossing-tier boundary: plain input alone is below the threshold, but plain
    /// input plus cache-read tokens crosses it. The sum must drive tier selection.
    #[test]
    fn test_tier_comparator_cache_read_crosses_threshold() {
        let calc = tiered_fixture_calc();
        let below = TokenUsage {
            input_tokens: 28_000,
            cache_read_input_tokens: 100_000,
            ..Default::default()
        };
        let cost_below = calc.calculate("tiered-fixture", &below).unwrap();
        // comparator = 128_000 < 128_001 → tier 0
        // input: 28_000*1250=35M, cache: 100_000*1250*0.25=31.25M
        assert_eq!(cost_below.total_cost, NanoUsd(66_250_000));

        let at_threshold = TokenUsage {
            input_tokens: 28_001,
            cache_read_input_tokens: 100_000,
            ..Default::default()
        };
        let cost_at = calc.calculate("tiered-fixture", &at_threshold).unwrap();
        // comparator = 128_001 → tier 1
        // input: 28_001*2500=70_002_500, cache: 100_000*2500*0.25=62.5M
        assert_eq!(cost_at.total_cost, NanoUsd(132_502_500));

        let past_threshold = TokenUsage {
            input_tokens: 28_002,
            cache_read_input_tokens: 100_000,
            ..Default::default()
        };
        let cost_past = calc.calculate("tiered-fixture", &past_threshold).unwrap();
        // comparator = 128_002 → tier 1
        // input: 28_002*2500=70_005_000, cache: 100_000*2500*0.25=62.5M
        assert_eq!(cost_past.total_cost, NanoUsd(132_505_000));
    }

    /// [`TokenUsage::context_input_tokens`] as a method, not a stored field: default
    /// construction reads 0, and partial construction reads only what was set.
    #[test]
    fn test_context_input_tokens_derived_not_stored() {
        assert_eq!(TokenUsage::default().context_input_tokens(), 0);
        let usage = TokenUsage {
            input_tokens: 42,
            ..Default::default()
        };
        assert_eq!(usage.context_input_tokens(), 42);
    }

    /// Buckets summing past `u64::MAX` saturate rather than wrap, so an overflow selects the
    /// highest tier instead of falling back to tier 0.
    #[test]
    fn test_context_input_tokens_saturates_on_overflow() {
        let usage = TokenUsage {
            input_tokens: u64::MAX,
            cache_read_input_tokens: 1,
            ..Default::default()
        };
        assert_eq!(usage.context_input_tokens(), u64::MAX);
    }

    /// Both cache-write buckets (ephemeral 5-minute and 1-hour) contribute to the comparator
    /// sum, not just cache-read — isolated from `input_tokens` and from each other.
    #[test]
    fn test_context_input_tokens_includes_cache_write_buckets() {
        let usage = TokenUsage {
            input_tokens: 100,
            cache_write: cache_write_of(&[("5m", 200), ("1h", 300)]),
            ..Default::default()
        };
        assert_eq!(usage.context_input_tokens(), 600);
    }

    /// Synthetic additive-bucket shape, not a claim about any live adapter's current output:
    /// when `input_tokens` and cache tokens are disjoint (no double count), the comparator is
    /// their sum. This is a no-movement check on the derived method itself.
    #[test]
    fn test_context_input_tokens_additive_bucket_shape_no_movement() {
        let usage = TokenUsage {
            input_tokens: 40_000,
            cache_read_input_tokens: 10_000,
            ..Default::default()
        };
        assert_eq!(usage.context_input_tokens(), 50_000);
    }

    /// Inclusive-by-default accounting (Bedrock): no cache buckets set until cache buckets are
    /// parsed, so the comparator equals `inputTokens` unchanged.
    #[test]
    fn test_context_input_tokens_matches_bedrock_no_cache_shape() {
        let usage = TokenUsage {
            input_tokens: 40_000,
            ..Default::default()
        };
        assert_eq!(usage.context_input_tokens(), 40_000);
    }

    #[test]
    fn test_validation_non_ascending_thresholds() {
        let json = r#"{"models":{"x":{"provider":"p","context_window":1000,"aliases":[],"tiers":[{"threshold":100,"input_per_token":0.001,"output_per_token":0.001},{"threshold":50,"input_per_token":0.002,"output_per_token":0.002}]}}}"#;
        let err = PricingDb::load(json.as_bytes(), &default_config()).unwrap_err();
        assert!(matches!(err, PricingError::InvalidDb(_)));
        assert!(err.to_string().contains("ascending"));
    }

    /// A tier priced above a model's declared `context_window` can never be reached — no
    /// request can carry more input tokens than the model accepts, so a threshold there is
    /// unreachable dead data. `claude-sonnet-4-5-20250929` shipped exactly this once: a
    /// premium tier priced above its actual context window.
    #[test]
    fn test_validation_tier_threshold_exceeds_context_window() {
        let json = r#"{"models":{"x":{"provider":"p","context_window":1000,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0.001,"output_per_token":0.001},{"threshold":1001,"input_per_token":0.002,"output_per_token":0.002}]}}}"#;
        let err = PricingDb::load(json.as_bytes(), &default_config()).unwrap_err();
        assert!(matches!(err, PricingError::InvalidDb(_)));
        assert!(err.to_string().contains("context_window"));
    }

    /// A tier threshold exactly at `context_window` is reachable (a request may carry up to
    /// and including that many tokens) and must not trip the check above.
    #[test]
    fn test_validation_tier_threshold_equal_to_context_window_is_valid() {
        let json = r#"{"models":{"x":{"provider":"p","context_window":1000,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0.001,"output_per_token":0.001},{"threshold":1000,"input_per_token":0.002,"output_per_token":0.002}]}}}"#;
        PricingDb::load(json.as_bytes(), &default_config()).unwrap();
    }

    /// `context_window == 0` is the documented unknown/unconstrained sentinel (see the field
    /// doc on `PricingEntry::context_window`), not a real zero-token window. A nonzero tier
    /// threshold under an unknown window is not unreachable — there is nothing to compare
    /// against — so the guard above must not fire here.
    #[test]
    fn test_validation_nonzero_threshold_under_unknown_context_window_is_valid() {
        let json = r#"{"models":{"x":{"provider":"p","context_window":0,"aliases":[],"tiers":[{"threshold":0,"input_per_token":0.001,"output_per_token":0.001},{"threshold":1000,"input_per_token":0.002,"output_per_token":0.002}]}}}"#;
        PricingDb::load(json.as_bytes(), &default_config()).unwrap();
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
                cache_write_multipliers: HashMap::new(),
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
            .saturating_add(cost.cache_write_cost.0)
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
            let cost = calc.calculate("gpt-4.1", &usage.to_token_usage()).unwrap();
            assert_component_sum_invariant(&cost);
        }

        #[test]
        fn prop_zero_usage_zero_cost(_ in token_usage_strategy()) {
            let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
            let calc = BundledCostCalculator::new(db_holder(db));
            let cost = calc.calculate("gpt-4.1", &TokenUsage::default()).unwrap();
            assert_eq!(cost.total_cost, NanoUsd::zero());
        }

        /// Image and audio are excluded from `token_usage_strategy` itself (gpt-4.1 has no
        /// configured rate for either, and a positive quantity with no rate is now a request-wide
        /// error rather than a silent zero — see the isolated cost-unavailable unit tests), so
        /// only the six dimensions that strategy generates are exercised here.
        #[test]
        fn prop_monotonic_more_tokens_higher_cost(
            base in token_usage_strategy(),
            field in 0u32..6u32,
        ) {
            let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
            let calc = BundledCostCalculator::new(db_holder(db));
            let cost_base = calc.calculate("gpt-4.1", &base.to_token_usage()).unwrap();

            let mut inc = base.clone();
            match field {
                0 => inc.input_tokens = inc.input_tokens.saturating_add(1),
                1 => inc.output_tokens = inc.output_tokens.saturating_add(1),
                2 => inc.cache_read_input_tokens = inc.cache_read_input_tokens.saturating_add(1),
                3 => inc.cache_write_5m_tokens = inc.cache_write_5m_tokens.saturating_add(1),
                4 => inc.cache_write_1h_tokens = inc.cache_write_1h_tokens.saturating_add(1),
                _ => inc.thinking_tokens = inc.thinking_tokens.saturating_add(1),
            }
            let cost_inc = calc.calculate("gpt-4.1", &inc.to_token_usage()).unwrap();
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
    fn test_table_gpt41_cache_read_smoke() {
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
            cache_write: cache_write_of(&[("5m", 100), ("1h", 50)]),
            ..Default::default()
        };
        let cost = calc.calculate("claude-sonnet-4-6", &usage).unwrap();
        assert_component_sum_invariant(&cost);
        assert!(cost.cache_write_cost > NanoUsd::zero());
        // 5m: 100 tokens × 3000 nano × 1.25 = 375_000 nano
        // 1h: 50 tokens × 3000 nano × 2.0 = 300_000 nano
        assert_eq!(cost.cache_write_cost, NanoUsd(375_000 + 300_000));

        // 1h costs more per token than 5m (2.0x vs 1.25x multiplier): isolate each class at an
        // equal token count so the merged `cache_write_cost` field can be compared directly.
        let usage_5m_only = TokenUsage {
            cache_write: cache_write_of(&[("5m", 1_000)]),
            ..Default::default()
        };
        let usage_1h_only = TokenUsage {
            cache_write: cache_write_of(&[("1h", 1_000)]),
            ..Default::default()
        };
        let cost_5m = calc.calculate("claude-sonnet-4-6", &usage_5m_only).unwrap();
        let cost_1h = calc.calculate("claude-sonnet-4-6", &usage_1h_only).unwrap();
        assert!(
            cost_1h.cache_write_cost > cost_5m.cache_write_cost,
            "1h cache creation should cost more per token than 5m"
        );
    }

    #[test]
    fn test_table_gemini_25_pro_thinking_smoke() {
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

    /// Extracts model IDs from `supported_models:` YAML list blocks in a markdown doc, using
    /// indentation (not just a "- " prefix) to find the end of the list — otherwise a sibling
    /// `- name: ...` entry one indent level out would be misread as another model ID.
    fn extract_supported_model_ids(content: &str) -> Vec<String> {
        let mut ids = Vec::new();
        let mut lines = content.lines().peekable();
        while let Some(line) = lines.next() {
            if line.trim_start() != "supported_models:" {
                continue;
            }
            let indent = line.len() - line.trim_start().len();
            while let Some(next) = lines.peek() {
                let next_indent = next.len() - next.trim_start().len();
                let next_trimmed = next.trim_start();
                if next_indent > indent
                    && let Some(id) = next_trimmed.strip_prefix("- ")
                {
                    let id = id.trim().trim_matches('"').to_string();
                    if !id.is_empty() {
                        ids.push(id);
                    }
                    lines.next();
                } else {
                    break;
                }
            }
        }
        ids
    }

    /// Regression guard, scoped specifically to `supported_models:` YAML list blocks in
    /// `docs/providers/*.md` — it does **not** scan prose, tables, or curl examples elsewhere in
    /// those docs, so a model ID mentioned only outside a `supported_models:` block can still go
    /// stale undetected. `supported_models:` is the one place a config example is meant to be
    /// copy-pasted verbatim, which is the actual failure mode behind two prior review findings on
    /// this file (stale `deepseek-chat` / `deepseek-reasoner` example values).
    #[test]
    fn test_provider_docs_supported_models_resolve_in_pricing_db() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let inner = db.read();

        // Keyless local/self-hosted examples are illustrative only and are never expected to
        // carry bundled pricing.
        let known_unpriced_examples: &[&str] = &["llama3"];

        let docs_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/providers");
        let mut checked = 0usize;
        for entry in std::fs::read_dir(&docs_dir).expect("docs/providers must exist") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let content = std::fs::read_to_string(&path).expect("doc must be readable");
            for model_id in extract_supported_model_ids(&content) {
                if known_unpriced_examples.contains(&model_id.as_str()) {
                    continue;
                }
                checked += 1;
                assert!(
                    inner.lookup(&model_id, None).is_some(),
                    "{path:?} lists `{model_id}` under supported_models, but it does not resolve in PricingDb"
                );
            }
        }
        assert!(
            checked > 0,
            "expected at least one supported_models entry across provider docs — extractor may be broken"
        );
    }

    // --- Composite request-wide cost status and the quantity-partition gate ---

    /// A request whose cache-write counters saturated does not partition exactly, and pricing
    /// the parts that happen to fit would bill a total that differs from the quantity persisted
    /// on the spend row. The whole request must refuse to price instead.
    #[test]
    fn test_inexact_partition_refuses_to_price() {
        // A zero-rate model: every priced component is zero however large the quantity, so the
        // monetary-overflow checks downstream cannot fire and the quantity gate is the only
        // thing that can reject this request.
        let json = r#"{"models":{"cw-free":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.0,"output_per_token":0.0,
              "cache_write_multipliers":{"5m":1.0,"1h":1.0}}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));

        let registry = CacheWriteClassRegistry::from_classes(
            ["5m", "1h"]
                .iter()
                .filter_map(|k| CacheWriteClass::canonicalize(k)),
        )
        .expect("registry");
        let mut acc = CacheWriteAccumulator::new(registry);
        // Saturates the running detail total, so the components can no longer sum to the
        // accounted quantity under checked arithmetic.
        acc.observe_detail("5m", CacheWriteClass::canonicalize("5m"), u64::MAX);
        acc.observe_detail("1h", CacheWriteClass::canonicalize("1h"), 1);
        let cache_write = acc.finish();
        assert!(
            !cache_write.partition_is_exact(),
            "precondition: the saturated accumulation must not partition exactly"
        );

        let usage = TokenUsage {
            cache_write,
            ..Default::default()
        };
        let err = calc.calculate("cw-free", &usage).unwrap_err();
        assert!(
            matches!(&err, CostError::Pricing(m) if m.contains("partition")),
            "must be rejected by the quantity gate, not by a downstream money check: {err:?}"
        );
    }

    /// A request whose every positive component priced at a configured rate, with self-consistent
    /// quantity evidence, reports `exact`.
    #[test]
    fn test_status_is_exact_when_every_class_is_configured() {
        let json = r#"{"models":{"cw-test":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001,
              "cache_write_multipliers":{"5m":1.25,"1h":2.0}}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 100,
            cache_write: cache_write_of(&[("5m", 1000), ("1h", 500)]),
            ..Default::default()
        };
        let cost = calc.calculate("cw-test", &usage).unwrap();
        assert_eq!(cost.status, CostStatus::Exact);
    }

    /// A positive quantity in a class this tier does not price falls back, and a fallback rate
    /// anywhere pulls the whole request's status down — it cannot hide behind exact components.
    #[test]
    fn test_status_is_rate_fallback_when_a_class_is_unconfigured() {
        let json = r#"{"models":{"cw-test":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001,
              "cache_write_multipliers":{"1h":3.0}}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 100,
            cache_write: cache_write_of(&[("5m", 1000), ("1h", 1000)]),
            ..Default::default()
        };
        let cost = calc.calculate("cw-test", &usage).unwrap();
        assert_eq!(cost.status, CostStatus::RateFallback);
    }

    /// A successful response carrying no usable usage at all cannot claim a confidence it never
    /// established, even though no component found a reason to degrade.
    #[test]
    fn test_status_is_cost_unavailable_when_no_usable_usage() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let cost = calc.calculate("gpt-4.1", &TokenUsage::default()).unwrap();
        assert_eq!(cost.status, CostStatus::CostUnavailable);
        assert_eq!(cost.total_cost, NanoUsd::zero());
    }

    /// An external `CostCalculator` filling a breakdown with `..Default::default()` must not
    /// assert `exact` by omission — the default is the worst status, not the best.
    #[test]
    fn test_cost_breakdown_default_status_is_cost_unavailable() {
        assert_eq!(
            crate::domain::ports::CostBreakdown::default().status,
            CostStatus::CostUnavailable
        );
    }

    /// A positive cache-read quantity priced through the `unwrap_or(1.0)` fallback — the tier
    /// configures no `cache_read_multiplier` — is a fallback rate, not an exact one.
    #[test]
    fn test_status_is_rate_fallback_for_cache_read_without_multiplier() {
        let json = r#"{"models":{"cr-test":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));

        // The same tier with no cache-read quantity is exact, so the status below is caused by
        // the unpriced quantity rather than by anything else about this fixture.
        let baseline = calc
            .calculate(
                "cr-test",
                &TokenUsage {
                    input_tokens: 100,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(baseline.status, CostStatus::Exact);

        let cost = calc
            .calculate(
                "cr-test",
                &TokenUsage {
                    input_tokens: 100,
                    cache_read_input_tokens: 50,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(cost.status, CostStatus::RateFallback);
    }

    /// A batch request whose tier configures no batch multipliers is priced through the
    /// `unwrap_or(1.0)` fallback on both groups; either group alone is enough to degrade status.
    #[test]
    fn test_status_is_rate_fallback_for_batch_without_multipliers() {
        let json = r#"{"models":{"b-test":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));

        // Input group only.
        let input_only = calc
            .calculate(
                "b-test",
                &TokenUsage {
                    input_tokens: 100,
                    batch: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(input_only.status, CostStatus::RateFallback);

        // Output group only.
        let output_only = calc
            .calculate(
                "b-test",
                &TokenUsage {
                    output_tokens: 100,
                    batch: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(output_only.status, CostStatus::RateFallback);

        // The identical usage without the batch flag is exact, so the degradation above is the
        // batch multipliers and nothing else.
        let non_batch = calc
            .calculate(
                "b-test",
                &TokenUsage {
                    input_tokens: 100,
                    output_tokens: 100,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(non_batch.status, CostStatus::Exact);
    }

    /// Thinking tokens on a tier with no `thinking_per_token` are charged at the standard output
    /// rate. That is a contractual derivation, not a fallback, so the request stays `exact`.
    #[test]
    fn test_thinking_without_specific_rate_derives_output_rate_and_stays_exact() {
        let json = r#"{"models":{"th-test":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000002}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let cost = calc
            .calculate(
                "th-test",
                &TokenUsage {
                    thinking_tokens: 1_000,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            cost.status,
            CostStatus::Exact,
            "the documented output-rate derivation must not read as a fallback"
        );
        // 1000 tokens x 2000 nano-USD per token: the output rate, applied exactly.
        assert_eq!(cost.thinking_cost, NanoUsd(2_000_000));
    }

    /// A provider aggregate that contradicts the details it also reported forces a conservative
    /// quantity choice, which forbids claiming `exact` even though every rate was configured.
    #[test]
    fn test_cache_write_contradiction_forces_reconciled() {
        let json = r#"{"models":{"cw-test":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001,
              "cache_write_multipliers":{"5m":1.0}}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));

        let registry = CacheWriteClassRegistry::from_classes(
            ["5m"]
                .iter()
                .filter_map(|k| CacheWriteClass::canonicalize(k)),
        )
        .expect("registry");
        let mut acc = CacheWriteAccumulator::new(registry);
        acc.observe_detail("5m", CacheWriteClass::canonicalize("5m"), 100);
        // The details exceed the aggregate, so the two views disagree. This direction is chosen
        // deliberately: an aggregate *larger* than the details leaves an unmatched residual that
        // is itself priced at the fallback rate, and `RateFallback` would then outrank the
        // `Reconciled` this test exists to pin. Here the residual is zero, so the contradiction
        // is the only thing that can move the status.
        acc.set_reported_aggregate(50);
        let cache_write = acc.finish();
        assert!(cache_write.outcome().is_contradiction(), "precondition");
        assert_eq!(
            cache_write.fallback_tokens(),
            0,
            "precondition: no fallback-priced tokens, so only the contradiction can move status"
        );

        let cost = calc
            .calculate(
                "cw-test",
                &TokenUsage {
                    cache_write,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(cost.status, CostStatus::Reconciled);
    }

    /// Observing one configured class twice is an exact duplicate, which is ambiguous quantity
    /// evidence and forces `reconciled` on its own.
    #[test]
    fn test_cache_write_duplicate_forces_reconciled() {
        let json = r#"{"models":{"cw-test":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001,
              "cache_write_multipliers":{"5m":1.0}}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));

        let registry = CacheWriteClassRegistry::from_classes(
            ["5m"]
                .iter()
                .filter_map(|k| CacheWriteClass::canonicalize(k)),
        )
        .expect("registry");
        let mut acc = CacheWriteAccumulator::new(registry);
        acc.observe_detail("5m", CacheWriteClass::canonicalize("5m"), 100);
        acc.observe_detail("5m", CacheWriteClass::canonicalize("5m"), 100);
        let cache_write = acc.finish();
        assert!(cache_write.duplicate().any(), "precondition");
        assert!(
            !cache_write.outcome().is_contradiction(),
            "precondition: duplicate alone, no aggregate contradiction"
        );

        let cost = calc
            .calculate(
                "cw-test",
                &TokenUsage {
                    cache_write,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(cost.status, CostStatus::Reconciled);
    }

    // --- Cache-write multiplier map, overrides and monetary representability ---

    /// AC10/AC11: a configured class — including an explicit `0.0` — prices exactly at its
    /// multiplier; zero is a real price, not a fallback.
    #[test]
    fn test_cache_write_configured_class_prices_exactly_including_zero() {
        let json = r#"{"models":{"cw-test":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001,
              "cache_write_multipliers":{"5m":1.25,"1h":0.0}}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            cache_write: cache_write_of(&[("5m", 1000), ("1h", 1000)]),
            ..Default::default()
        };
        let cost = calc.calculate("cw-test", &usage).unwrap();
        // 5m: 1000 × 1000 nano × 1.25 = 1_250_000; 1h: 1000 × 1000 nano × 0.0 = 0.
        assert_eq!(cost.cache_write_cost, NanoUsd(1_250_000));
    }

    /// AC11/AC12: an unconfigured class in a tier that configures at least one other class falls
    /// back to the tier-local highest multiplier, not a flat `1.0`.
    #[test]
    fn test_cache_write_unknown_class_falls_back_to_tier_local_highest() {
        let json = r#"{"models":{"cw-test":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001,
              "cache_write_multipliers":{"1h":3.0}}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            cache_write: cache_write_of(&[("5m", 1000)]),
            ..Default::default()
        };
        let cost = calc.calculate("cw-test", &usage).unwrap();
        // "5m" is not configured, so it falls back to max(3.0, 1.0) = 3.0.
        assert_eq!(cost.cache_write_cost, NanoUsd(3_000_000));
    }

    /// AC12: an absent map falls back to `1.0` — the floor of `max(highest configured, 1.0)`
    /// when nothing is configured to raise it.
    #[test]
    fn test_cache_write_absent_map_uses_1_0() {
        let json = r#"{"models":{"cw-test":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            cache_write: cache_write_of(&[("5m", 1000)]),
            ..Default::default()
        };
        let cost = calc.calculate("cw-test", &usage).unwrap();
        assert_eq!(cost.cache_write_cost, NanoUsd(1_000_000));
    }

    /// AC14: an override that supplies a map replaces the baseline map completely — a class the
    /// baseline priced exactly does not survive into the override's fallback.
    #[test]
    fn test_override_cache_write_map_replaces_baseline_completely() {
        let mut config = default_config();
        config.overrides.insert(
            "claude-sonnet-4-6".into(),
            PricingOverride {
                input_per_token: 0.000003,
                output_per_token: 0.000015,
                context_window: 200_000,
                cache_read_multiplier: None,
                cache_write_multipliers: HashMap::from([("5m".to_string(), 4.0)]),
            },
        );
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &config).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            cache_write: cache_write_of(&[("5m", 1000), ("1h", 1000)]),
            ..Default::default()
        };
        let cost = calc.calculate("claude-sonnet-4-6", &usage).unwrap();
        // 5m: 1000 * 3000 * 4.0 = 12_000_000. "1h" is absent from the override's map, so it
        // falls back to max(4.0, 1.0) = 4.0 too — the baseline's 2.0x for this model must not
        // survive the override — giving the same 12_000_000, for 24_000_000 combined.
        assert_eq!(cost.cache_write_cost, NanoUsd(24_000_000));
    }

    /// AC14: an override that omits the map does not inherit the baseline's — a class the
    /// baseline priced exactly is now fallback-priced under the override.
    #[test]
    fn test_override_omitted_cache_write_map_does_not_inherit_baseline() {
        let mut config = default_config();
        config.overrides.insert(
            "claude-sonnet-4-6".into(),
            PricingOverride {
                input_per_token: 0.000003,
                output_per_token: 0.000015,
                context_window: 200_000,
                cache_read_multiplier: None,
                cache_write_multipliers: HashMap::new(),
            },
        );
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &config).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            cache_write: cache_write_of(&[("5m", 1000)]),
            ..Default::default()
        };
        let cost = calc.calculate("claude-sonnet-4-6", &usage).unwrap();
        // Baseline configures 5m at 1.25x (375_000). The override's empty map means fallback
        // (1.0x = 300_000) applies instead — it must not inherit the baseline's map.
        assert_eq!(cost.cache_write_cost, NanoUsd(3_000_000));
    }

    /// AC14c: `"05m"` and `"5m"` canonicalize to one class, so a leading-zero spelling in
    /// config prices a plain `"5m"` request exactly.
    #[test]
    fn test_cache_write_map_key_leading_zero_canonicalizes() {
        let json = r#"{"models":{"cw-test":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001,
              "cache_write_multipliers":{"05m":2.0}}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            cache_write: cache_write_of(&[("5m", 1000)]),
            ..Default::default()
        };
        let cost = calc.calculate("cw-test", &usage).unwrap();
        assert_eq!(cost.cache_write_cost, NanoUsd(2_000_000));
    }

    /// AC14c: two keys folding to the same canonical class with different multipliers fail load.
    #[test]
    fn test_cache_write_map_conflicting_duplicate_keys_fail_load() {
        let json = r#"{"models":{"cw-test":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001,
              "cache_write_multipliers":{"5m":1.25,"05m":2.0}}]}}}"#;
        let err = PricingDb::load(json.as_bytes(), &default_config()).unwrap_err();
        assert!(matches!(err, PricingError::InvalidDb(_)));
        assert!(err.to_string().contains("different values"));
    }

    /// AC14c: an over-long or non-duration key fails load.
    #[test]
    fn test_cache_write_map_invalid_key_fails_load() {
        let json = r#"{"models":{"cw-test":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001,
              "cache_write_multipliers":{"not-a-duration":1.0}}]}}}"#;
        let err = PricingDb::load(json.as_bytes(), &default_config()).unwrap_err();
        assert!(matches!(err, PricingError::InvalidDb(_)));
        assert!(err.to_string().contains("not a canonical"));
    }

    /// AC14d/21c: a single tier under the cap while the *union* across all entries and tiers
    /// exceeds it must still fail load — the per-tier check alone does not satisfy this.
    #[test]
    fn test_cache_write_registry_union_across_entries_exceeding_cap_fails_load() {
        let base = r#"{"models":{"base":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001}]}}}"#;
        let mut config = default_config();
        // 17 distinct classes via 17 one-class overrides — MAX_CONFIGURED_CACHE_WRITE_CLASSES
        // (16) plus one.
        for i in 0..17u32 {
            config.overrides.insert(
                format!("model-{i}"),
                PricingOverride {
                    input_per_token: 0.000001,
                    output_per_token: 0.000001,
                    context_window: 1000,
                    cache_read_multiplier: None,
                    cache_write_multipliers: HashMap::from([(format!("{i}m"), 1.0)]),
                },
            );
        }
        let err = PricingDb::load(base.as_bytes(), &config).unwrap_err();
        assert!(matches!(err, PricingError::ClassRegistry(_)));
    }

    /// AC14d: a set of overrides whose union is exactly at the cap loads successfully — the
    /// union is computed once from the final effective database, so `HashMap` iteration order
    /// over `config.overrides` cannot make this flaky.
    #[test]
    fn test_cache_write_registry_union_at_cap_loads_regardless_of_order() {
        let base = r#"{"models":{"base":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001}]}}}"#;
        let mut config = default_config();
        for i in 0..16u32 {
            config.overrides.insert(
                format!("model-{i}"),
                PricingOverride {
                    input_per_token: 0.000001,
                    output_per_token: 0.000001,
                    context_window: 1000,
                    cache_read_multiplier: None,
                    cache_write_multipliers: HashMap::from([(format!("{i}m"), 1.0)]),
                },
            );
        }
        let db = PricingDb::load(base.as_bytes(), &config);
        assert!(
            db.is_ok(),
            "16 distinct classes at the cap must load: {:?}",
            db.err()
        );
        assert_eq!(db.unwrap().registry().len(), 16);
    }

    /// `pricing_context()` snapshots the current generation's DB and its derived
    /// registry together, under one read.
    #[test]
    fn test_pricing_context_snapshots_db_and_registry_together() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let expected_registry = db.registry().clone();
        let calc = BundledCostCalculator::new(db_holder(db));
        let ctx = calc.pricing_context();
        assert_eq!(ctx.registry(), &expected_registry);
    }

    /// A request prices against the generation it was accounted under, not the one the holder
    /// happens to carry at finalization.
    ///
    /// The reload window is real: the snapshot is taken before dispatch and the cost is computed
    /// after the response is parsed, so a Class A SIGHUP can land in between. Pinning only the
    /// class registry and re-reading the holder for rates would classify the cache write against
    /// the old generation and bill it at the new one — one request, two generations.
    #[test]
    fn test_calculate_prices_against_the_pinned_generation_not_the_reloaded_holder() {
        const GEN_ONE: &str = r#"{"models":{"reload-fixture":{
            "provider":"test","context_window":200000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000001,"output_per_token":0.000001,
                      "cache_write_multipliers":{"5m":1.0}}]}}}"#;
        const GEN_TWO: &str = r#"{"models":{"reload-fixture":{
            "provider":"test","context_window":200000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.000002,"output_per_token":0.000002,
                      "cache_write_multipliers":{"5m":1.0}}]}}}"#;

        let holder = db_holder(PricingDb::load(GEN_ONE.as_bytes(), &default_config()).unwrap());
        let before_dispatch = snapshot_pricing_context(&holder);

        let mut cache_write = cache_write_of(&[("5m", 1_000)]);
        cache_write.set_pricing_context(before_dispatch);
        let usage = TokenUsage {
            input_tokens: 1_000,
            cache_write,
            ..Default::default()
        };

        // The reload lands after the request was accounted and before it is priced.
        *holder.write().expect("holder lock") =
            PricingDb::load(GEN_TWO.as_bytes(), &default_config()).unwrap();

        let cost = BundledCostCalculator::new(Arc::clone(&holder))
            .calculate("reload-fixture", &usage)
            .expect("pinned generation prices the request");

        // 1_000 tokens x 1 nano-USD, twice over: input and the 1.0-multiplier cache write.
        assert_eq!(
            cost.input_cost,
            NanoUsd(1_000_000),
            "input must be billed at the pinned generation's rate, not the reloaded one"
        );
        assert_eq!(
            cost.cache_write_cost,
            NanoUsd(1_000_000),
            "cache write must be billed at the pinned generation's rate, not the reloaded one"
        );
        assert_eq!(cost.status, CostStatus::Exact);
    }

    /// The registry derived from the bundled asset contains the classes it configures today.
    #[test]
    fn test_registry_contains_bundled_cache_write_classes() {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &default_config()).unwrap();
        let classes: Vec<&str> = db.registry().classes().iter().map(|c| c.as_str()).collect();
        assert!(classes.contains(&"5m"));
        assert!(classes.contains(&"1h"));
    }

    /// AC14b: `rate_is_representable` rejects non-finite and unrepresentably large values — not
    /// only `< 0.0`, which is `false` for `NaN` too, so the previous check let `NaN` load clean.
    #[test]
    fn test_rate_is_representable_rejects_non_finite_and_huge() {
        assert!(!rate_is_representable(f64::NAN));
        assert!(!rate_is_representable(f64::INFINITY));
        assert!(!rate_is_representable(f64::NEG_INFINITY));
        assert!(!rate_is_representable(-0.0001));
        assert!(!rate_is_representable(1e30));
        assert!(rate_is_representable(0.0));
        assert!(rate_is_representable(1.0));
    }

    /// AC14b: a non-finite override rate is skipped with a WARN rather than loaded — the same
    /// mechanism `test_apply_override_invalid_logs_warn` pins for a negative one.
    #[traced_test]
    #[test]
    fn test_nan_override_rate_is_skipped_not_loaded() {
        let mut config = default_config();
        config.overrides.insert(
            "nan-model".into(),
            PricingOverride {
                input_per_token: f64::NAN,
                output_per_token: 0.001,
                context_window: 1000,
                cache_read_multiplier: None,
                cache_write_multipliers: HashMap::new(),
            },
        );
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &config).unwrap();
        assert!(logs_contain("pricing override validation failed"));
        assert!(db.read().lookup("nan-model", None).is_none());
    }

    /// AC14b: a finite but non-representably large base rate fails pricing load entirely — this
    /// is a bundled entry, not an optional override, so an invalid value cannot be skipped.
    #[test]
    fn test_astronomically_large_base_rate_fails_load() {
        let json = r#"{"models":{"x":{"provider":"p","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":1e30,"output_per_token":0.0}]}}}"#;
        let err = PricingDb::load(json.as_bytes(), &default_config()).unwrap_err();
        assert!(matches!(err, PricingError::InvalidDb(_)));
        assert!(err.to_string().contains("representable"));
    }

    /// AC14b: non-finite or negative `audio_seconds` is rejected where it enters accounting.
    #[test]
    fn test_non_finite_or_negative_audio_seconds_rejected() {
        let json = r#"{"models":{"aud":{"provider":"p","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.0,"output_per_token":0.0,
              "audio_per_second":0.01}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let usage = TokenUsage {
                audio_seconds: bad,
                ..Default::default()
            };
            let err = calc.calculate("aud", &usage).unwrap_err();
            assert!(
                matches!(err, CostError::Pricing(_)),
                "audio_seconds={bad} must be rejected"
            );
        }
    }

    /// AC14b-i: a finite, positive `audio_seconds` whose product with the configured rate
    /// exceeds the nano-USD domain is rejected too — not saturated to `u64::MAX`. This is the
    /// case that separates validated from saturated: it passes against a saturating `as u64`.
    #[test]
    fn test_astronomically_large_but_finite_audio_seconds_rejected() {
        let json = r#"{"models":{"aud":{"provider":"p","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.0,"output_per_token":0.0,
              "audio_per_second":0.01}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            audio_seconds: 1e30,
            ..Default::default()
        };
        let err = calc.calculate("aud", &usage).unwrap_err();
        assert!(matches!(err, CostError::Pricing(_)));
    }

    /// Review correction: `u64::MAX as f64` is not the representability boundary — `u64::MAX`
    /// (`2^64 - 1`) is not exactly representable in `f64` and rounds *up* to `2^64`, so the
    /// original `<= u64::MAX as f64` check silently accepted `2^64` itself, which `as u64` then
    /// saturates to `u64::MAX` instead of rejecting. A rate whose nano-USD conversion is exactly
    /// `2^64` must be rejected by the strict `< 2^64` boundary.
    #[test]
    fn test_rate_is_representable_rejects_exact_two_pow_64_boundary() {
        let boundary_rate = TWO_POW_64 / 1_000_000_000.0;
        assert_eq!(
            boundary_rate * 1_000_000_000.0,
            TWO_POW_64,
            "test precondition: this rate must round-trip to exactly 2^64"
        );
        assert!(!rate_is_representable(boundary_rate));
    }

    /// Nearby valid-value coverage for the boundary above: a rate whose nano-USD conversion sits
    /// just under `2^64` is still representable — the strict `<` rejects only the boundary and
    /// beyond, not values approaching it.
    #[test]
    fn test_rate_is_representable_accepts_just_under_two_pow_64_boundary() {
        let rate = (TWO_POW_64 - 8192.0) / 1_000_000_000.0;
        assert!(
            rate * 1_000_000_000.0 < TWO_POW_64,
            "test precondition: this rate must round-trip to just under 2^64"
        );
        assert!(rate_is_representable(rate));
    }

    /// Review correction: `audio_seconds * audio_rate_nano` exactly equal to `2^64` must be
    /// rejected — the same boundary bug as `rate_is_representable`, reached through the audio
    /// product rather than the rate conversion.
    #[test]
    fn test_audio_product_exactly_two_pow_64_rejected() {
        let json = r#"{"models":{"aud":{"provider":"p","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.0,"output_per_token":0.0,
              "audio_per_second":1.0}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        // audio_rate_nano = 1_000_000_000 (audio_per_second: 1.0). audio_seconds chosen so the
        // product round-trips to exactly 2^64.
        let audio_seconds = TWO_POW_64 / 1_000_000_000.0;
        assert_eq!(
            audio_seconds * 1_000_000_000.0,
            TWO_POW_64,
            "test precondition: this audio_seconds must round-trip to exactly 2^64"
        );
        let usage = TokenUsage {
            audio_seconds,
            ..Default::default()
        };
        let err = calc.calculate("aud", &usage).unwrap_err();
        assert!(matches!(err, CostError::Pricing(_)));
    }

    /// Nearby valid-value coverage: an audio product just under `2^64` is priced normally, not
    /// rejected.
    #[test]
    fn test_audio_product_just_under_two_pow_64_accepted() {
        let json = r#"{"models":{"aud":{"provider":"p","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":0.0,"output_per_token":0.0,
              "audio_per_second":1.0}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let audio_seconds = (TWO_POW_64 - 8192.0) / 1_000_000_000.0;
        let expected_product = audio_seconds * 1_000_000_000.0;
        assert!(
            expected_product < TWO_POW_64,
            "test precondition: this audio_seconds must round-trip to just under 2^64"
        );
        let usage = TokenUsage {
            audio_seconds,
            ..Default::default()
        };
        let cost = calc.calculate("aud", &usage).unwrap();
        assert_eq!(cost.audio_cost, NanoUsd(expected_product.round() as u64));
    }

    /// AC14a: a plain component multiplication that leaves the `u64` domain returns
    /// `CostError::Pricing` rather than the previous `saturating_mul` to `u64::MAX`.
    #[test]
    fn test_input_cost_overflow_returns_pricing_error() {
        let json = r#"{"models":{"huge-rate":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":1.0,"output_per_token":0.0}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: u64::MAX,
            ..Default::default()
        };
        let err = calc.calculate("huge-rate", &usage).unwrap_err();
        assert!(matches!(err, CostError::Pricing(_)));
    }

    /// AC14a: the batch-discount application is checked too — a cost within `u64` range scaled
    /// up by a >1x batch multiplier overflows rather than truncating through the previous
    /// `u128 -> u64 as` cast.
    #[test]
    fn test_batch_discount_overflow_returns_pricing_error() {
        let json = r#"{"models":{"batch-huge":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":1.0,"output_per_token":0.0,
              "batch_input_multiplier":10.0}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        // Base input_cost = 10_000_000_000 * 1_000_000_000 ≈ 1e19, within u64::MAX (~1.8446e19).
        // Scaled ×10 by the batch multiplier, it leaves the domain.
        let usage = TokenUsage {
            input_tokens: 10_000_000_000,
            batch: true,
            ..Default::default()
        };
        let err = calc.calculate("batch-huge", &usage).unwrap_err();
        assert!(matches!(err, CostError::Pricing(_)));
    }

    /// AC14a: the total addition across components is checked — two components each individually
    /// within range can still sum past the `u64` ceiling.
    #[test]
    fn test_total_cost_overflow_returns_pricing_error() {
        let json = r#"{"models":{"total-huge":{"provider":"test","context_window":1000,"aliases":[],
            "tiers":[{"threshold":0,"input_per_token":1.0,"output_per_token":1.0}]}}}"#;
        let db = PricingDb::load(json.as_bytes(), &default_config()).unwrap();
        let calc = BundledCostCalculator::new(db_holder(db));
        let usage = TokenUsage {
            input_tokens: 12_000_000_000,
            output_tokens: 12_000_000_000,
            ..Default::default()
        };
        let err = calc.calculate("total-huge", &usage).unwrap_err();
        assert!(matches!(err, CostError::Pricing(_)));
    }

    // -----------------------------------------------------------------------
    // No asset-level assertions live here, deliberately.
    //
    // `PricingDb::load` already rejects malformed JSON, non-monotonic tier
    // thresholds and alias collisions at startup. Beyond that, a test that
    // restates a figure from the asset cannot tell a wrong price from a right
    // one — it agrees with whatever was imported — and fails on every
    // legitimate refresh. Price correctness is established by diffing the
    // import against the pinned upstream revision recorded in the asset's
    // `_provenance` header; see docs/runbooks/pricing-refresh.md.
    //
    // Snapshot staleness is a runtime concern, not a build-time one: the asset
    // is embedded with `include_bytes!`, so a deployed binary keeps serving a
    // frozen snapshot long after any build. The check therefore belongs at
    // startup, where it reaches the operator whose budgets are affected.
    // -----------------------------------------------------------------------
}
