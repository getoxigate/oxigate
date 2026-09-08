// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Provider-neutral cache-write accounting: canonical classes, a bounded accumulator, the
//! evidence document, and the single finalization result.
//!
//! Cache-write tokens are reported by providers in two alternate views — an aggregate scalar and
//! a per-class detail object — that can disagree. This module owns the conservative
//! reconciliation between them, the bounded state the accumulation runs in, and the confidence
//! status the resulting cost carries.
//!
//! # What is bounded here
//!
//! Accounting-owned state only: the accumulator, the class registry it is seeded from, and every
//! copy accounting makes. Transport and event buffers — the response body, an SSE line — are
//! owned by the provider lanes and are deliberately not bounded by this module.
//!
//! # Layering
//!
//! Nothing here knows a provider member name. The canonical class is a *duration* (`5m`, `1h`,
//! `30m`); translating `ephemeral_5m_input_tokens` into one is the provider adapter's job, which
//! it does by implementing [`CacheWriteKeyGrammar`]. That keeps one provider's wire spelling out
//! of the shared accumulator.

use std::collections::BTreeSet;
use std::fmt;

use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

use crate::domain::ports::{CostBreakdown, TokenUsage};
use crate::domain::pricing::PricingDb;

/// Ceiling on the union of configured cache-write class names across the whole pricing DB.
///
/// The accumulator reserves one exact counter per configured class, so this also bounds the
/// accumulator's per-request state.
pub const MAX_CONFIGURED_CACHE_WRITE_CLASSES: usize = 16;

/// Ceiling on one canonical class name, so a count bound is also a memory bound.
pub const MAX_CANONICAL_CLASS_BYTES: usize = 16;

/// Ceiling on how many observations the accumulator retains as evidence.
///
/// Exceeding it marks the evidence document incomplete; it never changes an accounted quantity
/// or a cost.
pub const MAX_RETAINED_EVIDENCE_ENTRIES: usize = 32;

/// Ceiling on one retained raw provider key, applied at copy time.
///
/// Accounting never allocates a copy of a provider key longer than this, however long the key on
/// the wire was.
pub const MAX_RAW_KEY_BYTES: usize = 128;

/// Ceiling on the serialized evidence document.
///
/// A code constant, not configuration: it bounds a database column, and an operator lowering it
/// would silently change what is provable about past rows.
pub const CACHE_WRITE_EVIDENCE_MAX_BYTES: usize = 2 * 1024;

/// Version of the persisted evidence document shape.
pub const EVIDENCE_SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------------------------
// Cost status
// ---------------------------------------------------------------------------------------------

/// How much confidence the gateway has in one request's final cost.
///
/// Exactly one value describes a whole request. Where components disagree the worst one wins —
/// see [`CostStatus::worst`] — so a single fallback rate anywhere cannot be hidden behind exact
/// components elsewhere.
///
/// The declaration order *is* the precedence order, and [`Ord`] is derived from it:
/// `Exact < Reconciled < RateFallback < CostUnavailable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CostStatus {
    /// Every positive component used a configured or documented contractual rate, and the
    /// quantity evidence was self-consistent.
    Exact,
    /// Rates were exact, but contradictory or ambiguous quantity evidence required a
    /// conservative quantity policy.
    Reconciled,
    /// At least one positive quantity was priced at a fallback rate.
    RateFallback,
    /// No defensible complete request cost could be produced.
    CostUnavailable,
}

impl CostStatus {
    /// The worse of two statuses, under the precedence in [`CostStatus`].
    #[must_use]
    pub fn worst(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }

    /// The wire spelling used in headers, the terminal SSE event and the spend row.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Reconciled => "reconciled",
            Self::RateFallback => "rate-fallback",
            Self::CostUnavailable => "cost-unavailable",
        }
    }
}

/// Defaults to the *worst* status, so a cost assembled without deciding one cannot claim
/// confidence it never established.
impl Default for CostStatus {
    fn default() -> Self {
        Self::CostUnavailable
    }
}

impl fmt::Display for CostStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------------------------
// Canonical class
// ---------------------------------------------------------------------------------------------

/// A canonical cache-write class: a duration such as `5m`, `1h` or `30m`.
///
/// Stored inline with a compile-time capacity of [`MAX_CANONICAL_CLASS_BYTES`], so a bound on how
/// many classes exist is also a bound on how much memory they occupy. Values only come from
/// [`CacheWriteClass::canonicalize`], so every instance is already normalized.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CacheWriteClass {
    bytes: [u8; MAX_CANONICAL_CLASS_BYTES],
    len: u8,
}

impl CacheWriteClass {
    /// Normalizes a duration string into a canonical class, or returns `None` when it is not one.
    ///
    /// Accepts `[0-9]+[smhd]`, ASCII-case-insensitively. Leading zeros are stripped, so `05m` and
    /// `5m` are one class rather than two. Input longer than [`MAX_CANONICAL_CLASS_BYTES`] is
    /// rejected rather than normalized, so the bound cannot be reached through a padded spelling.
    #[must_use]
    pub fn canonicalize(raw: &str) -> Option<Self> {
        if raw.len() > MAX_CANONICAL_CLASS_BYTES || !raw.is_ascii() {
            return None;
        }
        let (digits, unit) = raw.split_at_checked(raw.len().checked_sub(1)?)?;
        let unit = unit.as_bytes().first()?.to_ascii_lowercase();
        if !matches!(unit, b's' | b'm' | b'h' | b'd') {
            return None;
        }
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        // Keep one digit so "0m" and "00m" both canonicalize to "0m" rather than to "m".
        let trimmed = digits.trim_start_matches('0');
        let significant = if trimmed.is_empty() { "0" } else { trimmed };

        let mut bytes = [0u8; MAX_CANONICAL_CLASS_BYTES];
        let len = significant.len() + 1;
        if len > MAX_CANONICAL_CLASS_BYTES {
            return None;
        }
        bytes[..significant.len()].copy_from_slice(significant.as_bytes());
        bytes[significant.len()] = unit;
        Some(Self {
            bytes,
            len: len as u8,
        })
    }

    /// The canonical name, e.g. `"5m"`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len as usize]).unwrap_or("")
    }
}

impl fmt::Debug for CacheWriteClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CacheWriteClass({:?})", self.as_str())
    }
}

impl fmt::Display for CacheWriteClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for CacheWriteClass {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for CacheWriteClass {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ClassVisitor;

        impl Visitor<'_> for ClassVisitor {
            type Value = CacheWriteClass;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a canonical cache-write duration such as \"5m\"")
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                CacheWriteClass::canonicalize(value)
                    .ok_or_else(|| E::invalid_value(serde::de::Unexpected::Str(value), &self))
            }
        }

        deserializer.deserialize_str(ClassVisitor)
    }
}

// ---------------------------------------------------------------------------------------------
// Registry and per-request pricing context
// ---------------------------------------------------------------------------------------------

/// Rejection reasons when building the configured-class registry.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ClassRegistryError {
    /// The union of configured class names across the pricing DB exceeds the reserved capacity.
    #[error("configured cache-write classes ({count}) exceed the maximum of {max}")]
    TooManyClasses {
        /// How many distinct canonical classes were configured.
        count: usize,
        /// The ceiling, [`MAX_CONFIGURED_CACHE_WRITE_CLASSES`].
        max: usize,
    },
}

/// The cache-write classes the effective pricing database configures.
///
/// This is the *union* across every entry and every tier, not one model's classes: the tier is
/// selected from a token total that does not exist until accumulation has finished, so the
/// accumulator cannot know which tier's classes to reserve while it is running.
///
/// A class in this registry gets a reserved exact counter in the accumulator. Nothing a provider
/// reports can take that counter away from it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheWriteClassRegistry {
    /// Sorted and deduplicated, so slot lookup is a binary search and slot indices are stable.
    classes: Vec<CacheWriteClass>,
}

impl CacheWriteClassRegistry {
    /// Builds a registry from configured canonical classes, deduplicating them.
    ///
    /// # Errors
    ///
    /// Returns [`ClassRegistryError::TooManyClasses`] when more than
    /// [`MAX_CONFIGURED_CACHE_WRITE_CLASSES`] distinct classes are configured. Callers fail the
    /// pricing load or reload on this rather than degrading at request time.
    pub fn from_classes<I>(classes: I) -> Result<Self, ClassRegistryError>
    where
        I: IntoIterator<Item = CacheWriteClass>,
    {
        let unique: BTreeSet<CacheWriteClass> = classes.into_iter().collect();
        if unique.len() > MAX_CONFIGURED_CACHE_WRITE_CLASSES {
            return Err(ClassRegistryError::TooManyClasses {
                count: unique.len(),
                max: MAX_CONFIGURED_CACHE_WRITE_CLASSES,
            });
        }
        Ok(Self {
            classes: unique.into_iter().collect(),
        })
    }

    /// The reserved slot index for a class, or `None` when it is not configured.
    #[must_use]
    pub fn slot_of(&self, class: &CacheWriteClass) -> Option<usize> {
        self.classes.binary_search(class).ok()
    }

    /// How many classes are configured.
    #[must_use]
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    /// Whether no class is configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// The configured classes, in canonical order.
    #[must_use]
    pub fn classes(&self) -> &[CacheWriteClass] {
        &self.classes
    }
}

/// One request's pricing generation: a pricing snapshot and the registry derived from it.
///
/// A `PricingDb` clone keeps pointing at the generation it was taken from, because a reload
/// installs a *new* database into the holder rather than mutating this one. Taking this snapshot
/// once, before dispatch, is what stops a request from seeding its accumulator from one
/// generation and pricing the result against another — a transition that would send a class to
/// unknown overflow that the newer generation configures exactly.
///
/// This is an ephemeral in-memory snapshot, taken for reload consistency inside one request. It
/// is not a persisted catalogue snapshot for historical replay, which remains out of scope.
#[derive(Debug, Clone)]
pub struct PricingContext {
    db: PricingDb,
    registry: CacheWriteClassRegistry,
}

impl PricingContext {
    /// Pairs a pricing snapshot with the registry derived from that same snapshot.
    #[must_use]
    pub fn new(db: PricingDb, registry: CacheWriteClassRegistry) -> Self {
        Self { db, registry }
    }

    /// The pricing database this request prices against.
    #[must_use]
    pub fn db(&self) -> &PricingDb {
        &self.db
    }

    /// The configured-class registry derived from [`PricingContext::db`].
    #[must_use]
    pub fn registry(&self) -> &CacheWriteClassRegistry {
        &self.registry
    }
}

// ---------------------------------------------------------------------------------------------
// Reconciliation facts
// ---------------------------------------------------------------------------------------------

/// How the aggregate and detail views of the cache-write quantity related to each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReconciliationOutcome {
    /// The two views agreed, or only one of them was reported.
    #[default]
    Consistent,
    /// The provider's aggregate exceeded the sum of the details it also reported.
    AggregateExceedsDetail,
    /// The sum of the reported details exceeded the provider's aggregate.
    DetailExceedsAggregate,
}

impl ReconciliationOutcome {
    /// Whether the two views contradicted each other and the maximum had to be selected.
    #[must_use]
    pub fn is_contradiction(self) -> bool {
        !matches!(self, Self::Consistent)
    }
}

/// Whether one class was observed more than once, and how confidently that can be said.
///
/// A configured class has a reserved slot, so repeats of it are counted exactly. Unknown classes
/// collapse into one bucket by design — keeping their identities apart to compare them is exactly
/// the unbounded state this module refuses to hold — so once more than one unknown observation
/// exists, their duplicate identity is *indeterminate* rather than duplicated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DuplicateAmbiguity {
    /// A configured class was observed more than once. Exact.
    pub configured_duplicate: bool,
    /// More than one unknown observation was recorded; whether they were the same class cannot
    /// be established.
    pub unknown_indeterminate: bool,
}

impl DuplicateAmbiguity {
    /// Whether either form of duplicate ambiguity applies.
    #[must_use]
    pub fn any(self) -> bool {
        self.configured_duplicate || self.unknown_indeterminate
    }
}

/// Every conservative quantity policy this request had to apply.
///
/// These force at least [`CostStatus::Reconciled`]: exact rates must not hide self-contradictory
/// quantity evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReconciliationFacts {
    /// Aggregate versus detail relation for cache-write tokens.
    pub outcome: ReconciliationOutcome,
    /// Duplicate observation ambiguity.
    pub duplicate: DuplicateAmbiguity,
    /// At least one observed class was not configured and was priced at the fallback rate.
    pub unknown_classes_present: bool,
    /// A cache-inclusive contract reported more cache-read than prompt tokens; billable input
    /// clamped to zero.
    pub cache_exceeds_prompt: bool,
    /// A reasoning-in-output contract reported more reasoning than completion tokens; the
    /// standard output charge clamped to zero.
    pub reasoning_exceeds_completion: bool,
}

impl ReconciliationFacts {
    /// Whether a conservative quantity policy was applied, which forbids reporting `exact`.
    ///
    /// Unknown classes are deliberately not counted here: they are a *rate* problem, and
    /// [`CostStatus::RateFallback`] already outranks [`CostStatus::Reconciled`].
    #[must_use]
    pub fn requires_reconciled(self) -> bool {
        self.outcome.is_contradiction()
            || self.duplicate.any()
            || self.cache_exceeds_prompt
            || self.reasoning_exceeds_completion
    }
}

// ---------------------------------------------------------------------------------------------
// Warning facts
// ---------------------------------------------------------------------------------------------

/// Why one completed request was anomalous enough to warrant the single structured warning.
///
/// Marked `#[non_exhaustive]` so a future reason code is an additive change for callers outside
/// this crate: an exhaustive `match` needs a wildcard arm once, and never again.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningReason {
    /// Aggregate and detail quantities contradicted each other.
    AggregateDetailContradiction,
    /// One class was observed more than once, exactly or indeterminately.
    DuplicateAmbiguity,
    /// A quantity was priced at a fallback rate.
    RateFallback,
    /// No defensible complete cost could be produced.
    CostUnavailable,
    /// Billable input clamped to zero by the cache-versus-prompt contradiction.
    CacheExceedsPrompt,
    /// Standard output charge clamped to zero by the reasoning-versus-completion contradiction.
    ReasoningExceedsCompletion,
    /// The persisted evidence document does not describe every observation.
    IncompleteEvidence,
    /// The provider reported no usage at all, so every billing quantity is unreported rather
    /// than zero.
    ProviderUsageMissing,
}

impl WarningReason {
    /// Stable reason code for the structured event.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AggregateDetailContradiction => "aggregate-detail-contradiction",
            Self::DuplicateAmbiguity => "duplicate-ambiguity",
            Self::RateFallback => "rate-fallback",
            Self::CostUnavailable => "cost-unavailable",
            Self::CacheExceedsPrompt => "cache-exceeds-prompt",
            Self::ReasoningExceedsCompletion => "reasoning-exceeds-completion",
            Self::IncompleteEvidence => "incomplete-evidence",
            Self::ProviderUsageMissing => "provider-usage-missing",
        }
    }
}

/// Ceiling on the retained pricing-failure message.
///
/// The message is the gateway's own, so this is a bound on the event rather than a defence
/// against a provider — but a warning that promises to be bounded has to be bounded by
/// construction, not by who happens to write its inputs.
pub const MAX_PRICING_ERROR_BYTES: usize = 256;

/// The bounded reason set for one request's single warning.
///
/// Reason codes, plus the calculator's own message when pricing failed outright — the detail
/// that used to be logged from the cost-error arm and would otherwise be lost when emission
/// moved to finalization. Every count the event reports — observations, retained entries,
/// accounted tokens, the byte limit — is read from the finalization result itself rather than
/// copied, so the warning cannot contradict the row it describes. No provider key and no
/// provider-supplied text ever reaches this type.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WarningFacts {
    reasons: Vec<WarningReason>,
    pricing_error: Option<String>,
}

impl WarningFacts {
    /// Records a reason, ignoring a repeat of one already present.
    pub fn add(&mut self, reason: WarningReason) {
        if !self.reasons.contains(&reason) {
            self.reasons.push(reason);
        }
    }

    /// Records why pricing failed outright, replacing any message already recorded.
    ///
    /// Trimmed at a UTF-8 boundary to [`MAX_PRICING_ERROR_BYTES`].
    pub fn set_pricing_error(&mut self, message: &str) {
        let end = floor_char_boundary(message, MAX_PRICING_ERROR_BYTES);
        self.pricing_error = Some(message[..end].to_owned());
    }

    /// The calculator's message, when pricing failed outright rather than degraded.
    #[must_use]
    pub fn pricing_error(&self) -> Option<&str> {
        self.pricing_error.as_deref()
    }

    /// The recorded reasons, in the order they were added.
    #[must_use]
    pub fn reasons(&self) -> &[WarningReason] {
        &self.reasons
    }

    /// Whether this request should emit the structured warning at all.
    #[must_use]
    pub fn should_warn(&self) -> bool {
        !self.reasons.is_empty()
    }
}

// ---------------------------------------------------------------------------------------------
// Evidence document
// ---------------------------------------------------------------------------------------------

/// One retained cache-write observation, as persisted.
///
/// `raw_key` is the provider's own member name, already capped at [`MAX_RAW_KEY_BYTES`] when it
/// was copied. It is evidence only: no quantity and no cost is ever derived from it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceEntry {
    /// The provider member name that carried this observation.
    pub raw_key: String,
    /// The canonical class it normalized to, or `null` when it named no valid duration.
    pub canonical_class: Option<CacheWriteClass>,
    /// Tokens the member reported.
    pub tokens: u64,
}

/// The cache-write half of the persisted evidence document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheWriteEvidence {
    /// The provider's aggregate, or zero when it reported none.
    pub reported_tokens: u64,
    /// The sum of every detail observation.
    pub detail_tokens: u64,
    /// The quantity actually billed, persisted and counted against budgets.
    pub accounted_tokens: u64,
    /// The cache-write component of this request's cost.
    pub component_cost_nano_usd: u64,
    /// How the two quantity views related.
    pub reconciliation: ReconciliationOutcome,
    /// Whether more than one unknown-class observation was recorded.
    ///
    /// Unknown classes collapse into one bucket, so whether two of them were the same class
    /// cannot be established. The document says *indeterminate* rather than claiming duplication
    /// it cannot see — a reader must not treat this as an exact duplicate count. It is a
    /// document-level fact, so entry omission cannot lose it.
    pub unknown_duplicates_indeterminate: bool,
    /// Whether an accumulated token total saturated, which makes the priceable decomposition
    /// untrustworthy and forces the request to `cost-unavailable`.
    pub quantity_overflow: bool,
    /// Retained observations, in arrival order.
    pub entries: Vec<EvidenceEntry>,
    /// Whether observations or key bytes were dropped to stay inside the bounds.
    pub incomplete: bool,
}

/// The versioned document persisted alongside a spend row.
///
/// The request-wide `cost_status` scalar on the row is authoritative and is deliberately not
/// duplicated in here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageEvidence {
    /// Shape version, so a later reader can tell what it is looking at.
    pub schema_version: u32,
    /// Cache-write evidence.
    pub cache_write: CacheWriteEvidence,
}

impl UsageEvidence {
    /// Shrinks the document until it serializes within `max_bytes`.
    ///
    /// Trims retained `raw_key` strings at UTF-8 boundaries first, keeping every observation's
    /// tokens and canonical class; only if that is not enough are whole entries omitted from the
    /// tail, which preserves the relative order of the ones retained. Either step sets
    /// `incomplete`.
    ///
    /// The floor is the entry-less document: its scalars are what the request concluded and are
    /// never dropped, so a `max_bytes` below roughly 300 cannot be met and the document is
    /// returned at that floor. [`CACHE_WRITE_EVIDENCE_MAX_BYTES`] is an order of magnitude above
    /// it, so the case does not arise for the cap this exists to enforce.
    ///
    /// This runs during finalization, not at the database boundary, so the flag the warning
    /// reports and the document the writer binds are decided at the same instant.
    ///
    /// # Errors
    ///
    /// Propagates a serialization failure; the document is made of integers, booleans and
    /// strings, so this is not reachable in practice.
    pub fn limit_to_bytes(&mut self, max_bytes: usize) -> Result<(), serde_json::Error> {
        if self.serialized_len()? <= max_bytes {
            return Ok(());
        }

        let mut cap = MAX_RAW_KEY_BYTES;
        while cap > 0 {
            cap /= 2;
            let mut trimmed = false;
            for entry in &mut self.cache_write.entries {
                let end = floor_char_boundary(&entry.raw_key, cap);
                if end < entry.raw_key.len() {
                    entry.raw_key.truncate(end);
                    trimmed = true;
                }
            }
            if trimmed {
                self.cache_write.incomplete = true;
            }
            if self.serialized_len()? <= max_bytes {
                return Ok(());
            }
        }

        while self.cache_write.entries.pop().is_some() {
            self.cache_write.incomplete = true;
            if self.serialized_len()? <= max_bytes {
                return Ok(());
            }
        }
        Ok(())
    }

    fn serialized_len(&self) -> Result<usize, serde_json::Error> {
        serde_json::to_string(self).map(|s| s.len())
    }
}

/// Largest index at or below `max` that is a UTF-8 character boundary of `s`.
fn floor_char_boundary(s: &str, max: usize) -> usize {
    if s.len() <= max {
        return s.len();
    }
    let mut i = max;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

// ---------------------------------------------------------------------------------------------
// Accumulator and accounting state
// ---------------------------------------------------------------------------------------------

/// One configured class's reserved counter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ClassSlot {
    tokens: u64,
    observations: u64,
}

/// Tokens for one configured class, as accounted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassTotal {
    /// The canonical class.
    pub class: CacheWriteClass,
    /// Tokens credited to it across the request.
    pub tokens: u64,
}

/// Bounded cache-write accumulator, seeded from the configured-class registry.
///
/// Slot assignment is by class *identity*, never by arrival order: a configured class is credited
/// to its reserved counter whether it arrives first or after an adversarial run of unknown
/// classes, so nothing a provider sends can cost a configured class its exact rate. Unknown
/// classes never take a reserved slot — they collapse into one saturating overflow bucket.
///
/// Total state is `O(MAX_CONFIGURED_CACHE_WRITE_CLASSES + MAX_RETAINED_EVIDENCE_ENTRIES)` and is
/// independent of how many members a provider sends or how long their names are.
#[derive(Debug, Clone)]
pub struct CacheWriteAccumulator {
    registry: CacheWriteClassRegistry,
    slots: Vec<ClassSlot>,
    unknown_tokens: u64,
    unknown_observations: u64,
    reported_tokens: Option<u64>,
    detail_tokens: u64,
    quantity_overflow: bool,
    evidence: Vec<EvidenceEntry>,
    evidence_truncated: bool,
}

/// Adds `tokens` to `total`, recording on `overflowed` when the true sum left the token domain.
///
/// The saturated value is kept so the request stays describable, but the flag is what the cost
/// path reads: a saturated total is not the quantity the provider reported, and pricing it would
/// charge a number nobody can defend.
fn accumulate(total: &mut u64, tokens: u64, overflowed: &mut bool) {
    match total.checked_add(tokens) {
        Some(sum) => *total = sum,
        None => {
            *total = u64::MAX;
            *overflowed = true;
        }
    }
}

impl CacheWriteAccumulator {
    /// Creates an accumulator with one reserved counter per configured class.
    #[must_use]
    pub fn new(registry: CacheWriteClassRegistry) -> Self {
        let slots = vec![ClassSlot::default(); registry.len()];
        Self {
            registry,
            slots,
            unknown_tokens: 0,
            unknown_observations: 0,
            reported_tokens: None,
            detail_tokens: 0,
            quantity_overflow: false,
            evidence: Vec::new(),
            evidence_truncated: false,
        }
    }

    /// Records the provider's cache-write aggregate, replacing any previously recorded one.
    ///
    /// Streaming providers restate the aggregate as the request progresses; the latest statement
    /// wins. The aggregate is never added to the detail sum — they are alternate views of the
    /// same quantity.
    pub fn set_reported_aggregate(&mut self, tokens: u64) {
        self.reported_tokens = Some(tokens);
    }

    /// Starts a new detail snapshot, discarding the previous one.
    ///
    /// A provider detail object is a cumulative snapshot rather than a delta: a new object
    /// replaces the previous one, while an omitted object leaves the previous one standing. Only
    /// the detail side is reset — the aggregate keeps its own last-write-wins semantics.
    pub fn begin_detail_snapshot(&mut self) {
        for slot in &mut self.slots {
            *slot = ClassSlot::default();
        }
        self.unknown_tokens = 0;
        self.unknown_observations = 0;
        self.detail_tokens = 0;
        self.quantity_overflow = false;
        self.evidence.clear();
        self.evidence_truncated = false;
    }

    /// Credits one detail observation.
    ///
    /// `class` is the canonical class the provider adapter's grammar derived from `raw_key`, or
    /// `None` when the key named no valid duration. A class the registry does not configure goes
    /// to overflow whether or not it is a valid duration.
    ///
    /// `raw_key` is borrowed from the transport; any copy this makes is capped at
    /// [`MAX_RAW_KEY_BYTES`] and is only made while evidence retention is still open.
    pub fn observe_detail(&mut self, raw_key: &str, class: Option<CacheWriteClass>, tokens: u64) {
        accumulate(&mut self.detail_tokens, tokens, &mut self.quantity_overflow);

        match class.and_then(|c| self.registry.slot_of(&c)) {
            Some(slot_index) => {
                let slot = &mut self.slots[slot_index];
                accumulate(&mut slot.tokens, tokens, &mut self.quantity_overflow);
                slot.observations = slot.observations.saturating_add(1);
            }
            None => {
                accumulate(
                    &mut self.unknown_tokens,
                    tokens,
                    &mut self.quantity_overflow,
                );
                self.unknown_observations = self.unknown_observations.saturating_add(1);
            }
        }

        if self.evidence.len() < MAX_RETAINED_EVIDENCE_ENTRIES {
            let end = floor_char_boundary(raw_key, MAX_RAW_KEY_BYTES);
            if end < raw_key.len() {
                self.evidence_truncated = true;
            }
            self.evidence.push(EvidenceEntry {
                raw_key: raw_key[..end].to_owned(),
                canonical_class: class,
                tokens,
            });
        } else {
            self.evidence_truncated = true;
        }
    }

    /// Finishes accumulation and returns the accounting state the cost path consumes.
    #[must_use]
    pub fn finish(self) -> CacheWriteAccounting {
        let classes: Vec<ClassTotal> = self
            .registry
            .classes()
            .iter()
            .zip(&self.slots)
            .filter(|(_, slot)| slot.observations > 0)
            .map(|(class, slot)| ClassTotal {
                class: *class,
                tokens: slot.tokens,
            })
            .collect();

        let configured_duplicate = self.slots.iter().any(|s| s.observations > 1);
        let observation_count = self.slots.iter().fold(self.unknown_observations, |acc, s| {
            acc.saturating_add(s.observations)
        });

        let outcome = match self.reported_tokens {
            Some(reported) if observation_count > 0 => match reported.cmp(&self.detail_tokens) {
                std::cmp::Ordering::Greater => ReconciliationOutcome::AggregateExceedsDetail,
                std::cmp::Ordering::Less => ReconciliationOutcome::DetailExceedsAggregate,
                std::cmp::Ordering::Equal => ReconciliationOutcome::Consistent,
            },
            _ => ReconciliationOutcome::Consistent,
        };

        CacheWriteAccounting {
            reported_tokens: self.reported_tokens,
            detail_tokens: self.detail_tokens,
            classes,
            unknown_tokens: self.unknown_tokens,
            observation_count,
            duplicate: DuplicateAmbiguity {
                configured_duplicate,
                unknown_indeterminate: self.unknown_observations > 1,
            },
            outcome,
            quantity_overflow: self.quantity_overflow,
            evidence: self.evidence,
            evidence_truncated: self.evidence_truncated,
            pricing_context: None,
        }
    }
}

/// Generalized cache-write accounting state for one request.
///
/// Replaces the fixed 5-minute / 1-hour token pair: the classes a provider reports are data, not
/// struct fields, so a new class becomes billable through pricing data alone.
///
/// The accounted quantity is *intended* to partition exactly into the parts the cost path prices
/// — `accounted_tokens == sum(class tokens) + unknown_tokens + unmatched_residual_tokens` — but
/// that is a conditional property, not an invariant. The counters are independent `u64`s, so a
/// provider reporting absurd quantities can saturate them separately and leave the components
/// summing to more than the accounted total. [`CacheWriteAccounting::partition_is_exact`] is the
/// authority on whether the identity actually holds for this request; the cost path must check it
/// rather than assume it.
#[derive(Debug, Clone, Default)]
pub struct CacheWriteAccounting {
    reported_tokens: Option<u64>,
    detail_tokens: u64,
    classes: Vec<ClassTotal>,
    unknown_tokens: u64,
    observation_count: u64,
    duplicate: DuplicateAmbiguity,
    outcome: ReconciliationOutcome,
    quantity_overflow: bool,
    evidence: Vec<EvidenceEntry>,
    evidence_truncated: bool,
    pricing_context: Option<PricingContext>,
}

impl CacheWriteAccounting {
    /// The provider's aggregate, when it reported one.
    #[must_use]
    pub fn reported_tokens(&self) -> Option<u64> {
        self.reported_tokens
    }

    /// The sum of every detail observation.
    #[must_use]
    pub fn detail_tokens(&self) -> u64 {
        self.detail_tokens
    }

    /// The single quantity used for tier selection, cost, spend, budgets and the public
    /// `cache_creation_input_tokens` field.
    ///
    /// The maximum of the two views, never their sum: contradictory evidence is resolved
    /// conservatively rather than by trusting whichever view is smaller.
    #[must_use]
    pub fn accounted_tokens(&self) -> u64 {
        self.reported_tokens.unwrap_or(0).max(self.detail_tokens)
    }

    /// Tokens credited to configured classes, priced at their exact multipliers.
    #[must_use]
    pub fn class_totals(&self) -> &[ClassTotal] {
        &self.classes
    }

    /// Tokens observed in classes the pricing context does not configure.
    #[must_use]
    pub fn unknown_tokens(&self) -> u64 {
        self.unknown_tokens
    }

    /// The part of a larger aggregate that the reported details did not account for.
    ///
    /// Not assigned to any class — a residual has no identity — so it is priced at the fallback
    /// rate rather than defaulted.
    #[must_use]
    pub fn unmatched_residual_tokens(&self) -> u64 {
        self.reported_tokens
            .unwrap_or(0)
            .saturating_sub(self.detail_tokens)
    }

    /// Tokens that must be priced at the fallback rate: overflow plus unmatched residual.
    ///
    /// Only meaningful when [`CacheWriteAccounting::partition_is_exact`] holds; the saturating
    /// addition here exists so the value stays describable, not so it can be charged.
    #[must_use]
    pub fn fallback_tokens(&self) -> u64 {
        self.unknown_tokens
            .saturating_add(self.unmatched_residual_tokens())
    }

    /// Whether any accumulated token total left the `u64` domain and was saturated.
    ///
    /// A saturated total is not the quantity the provider reported, and no later arithmetic can
    /// recover what it was.
    #[must_use]
    pub fn quantity_overflow(&self) -> bool {
        self.quantity_overflow
    }

    /// Whether the priceable components sum exactly to [`CacheWriteAccounting::accounted_tokens`].
    ///
    /// The cost path must not price the components when this is false. Charging them would bill a
    /// sum that differs from the quantity persisted on the spend row and counted against the
    /// budget — one request carrying two different cache-write quantities. Pricing returns
    /// `CostError::Pricing` instead, which finalization maps to an all-zero breakdown with
    /// `cost-unavailable`, the same treatment every other unrepresentable money value gets.
    ///
    /// False in two cases: a counter saturated during accumulation
    /// ([`CacheWriteAccounting::quantity_overflow`]), or the components do not add up under
    /// checked arithmetic. The second check is not redundant — a total that saturated to
    /// `u64::MAX` can still *appear* to balance, which is exactly why the flag is tracked as well.
    #[must_use]
    pub fn partition_is_exact(&self) -> bool {
        if self.quantity_overflow {
            return false;
        }
        let mut sum: u64 = 0;
        for total in &self.classes {
            match sum.checked_add(total.tokens) {
                Some(next) => sum = next,
                None => return false,
            }
        }
        sum.checked_add(self.unknown_tokens)
            .and_then(|s| s.checked_add(self.unmatched_residual_tokens()))
            .is_some_and(|total| total == self.accounted_tokens())
    }

    /// How many detail observations were recorded.
    #[must_use]
    pub fn observation_count(&self) -> u64 {
        self.observation_count
    }

    /// The accounted cache-write quantity to publish on the response's
    /// `cache_creation_input_tokens` field, or `None` when the response had no cache write.
    ///
    /// `None` and `Some(0)` are different statements — the first is a provider that said
    /// nothing, the second one that reported a zero — so a response that mentioned cache writes
    /// at all (an aggregate, or at least one detail observation with no aggregate) keeps saying
    /// so.
    #[must_use]
    pub fn published_tokens(&self) -> Option<u64> {
        if self.reported_tokens.is_some() || self.observation_count > 0 {
            Some(self.accounted_tokens())
        } else {
            None
        }
    }

    /// Duplicate observation ambiguity.
    #[must_use]
    pub fn duplicate(&self) -> DuplicateAmbiguity {
        self.duplicate
    }

    /// How the aggregate and detail views related.
    #[must_use]
    pub fn outcome(&self) -> ReconciliationOutcome {
        self.outcome
    }

    /// The retained observations, in arrival order.
    #[must_use]
    pub fn evidence_entries(&self) -> &[EvidenceEntry] {
        &self.evidence
    }

    /// Whether observations or key bytes were dropped to stay inside the accumulator bounds.
    #[must_use]
    pub fn evidence_truncated(&self) -> bool {
        self.evidence_truncated
    }

    /// The pricing generation this request accumulated against, when one travelled with it.
    #[must_use]
    pub fn pricing_context(&self) -> Option<&PricingContext> {
        self.pricing_context.as_ref()
    }

    /// Attaches the pricing generation the accumulator was seeded from, so finalization prices
    /// against the same generation rather than re-reading the holder.
    pub fn set_pricing_context(&mut self, context: PricingContext) {
        self.pricing_context = Some(context);
    }

    /// Builds the persisted evidence document, or `None` when nothing was cached.
    ///
    /// The returned document is not yet size-limited; finalization applies
    /// [`UsageEvidence::limit_to_bytes`] so the `incomplete` flag and the cost are decided
    /// together.
    #[must_use]
    pub fn to_evidence(&self, component_cost_nano_usd: u64) -> Option<UsageEvidence> {
        if self.accounted_tokens() == 0 {
            return None;
        }
        Some(UsageEvidence {
            schema_version: EVIDENCE_SCHEMA_VERSION,
            cache_write: CacheWriteEvidence {
                reported_tokens: self.reported_tokens.unwrap_or(0),
                detail_tokens: self.detail_tokens,
                accounted_tokens: self.accounted_tokens(),
                component_cost_nano_usd,
                reconciliation: self.outcome,
                unknown_duplicates_indeterminate: self.duplicate.unknown_indeterminate,
                quantity_overflow: self.quantity_overflow,
                entries: self.evidence.clone(),
                incomplete: self.evidence_truncated,
            },
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Seeded parse
// ---------------------------------------------------------------------------------------------

/// Interprets one provider member name in a cache-write detail object.
///
/// Each adapter owns its own wire spelling — Anthropic's `ephemeral_<duration>_input_tokens` is
/// not a cross-provider grammar — and returns the canonical duration it denotes, or `None` when
/// the member does not match. A `None` member is an unknown class and goes to overflow; it is
/// never guessed at.
pub trait CacheWriteKeyGrammar {
    /// The canonical cache-write class this member name denotes.
    fn class_of(&self, raw_key: &str) -> Option<CacheWriteClass>;
}

/// A [`DeserializeSeed`] that credits a provider's cache-write detail object straight into an
/// accumulator.
///
/// Deserializing into an intermediate collection first is what this exists to avoid: the entry
/// count is provider-controlled, each member name is an unbounded allocation, and duplicate
/// members allocate independently. Here each member is canonicalized and credited as it streams,
/// so accounting materializes no unbounded intermediate — and duplicate members are both counted
/// rather than silently collapsed the way a map type would collapse them.
///
/// Beginning the map starts a new detail snapshot, which is the replacement semantics a
/// cumulative provider snapshot requires.
pub struct CacheWriteDetailsSeed<'a, G: CacheWriteKeyGrammar> {
    accumulator: &'a mut CacheWriteAccumulator,
    grammar: &'a G,
}

impl<'a, G: CacheWriteKeyGrammar> CacheWriteDetailsSeed<'a, G> {
    /// Seeds a deserialization of one detail object with the accumulator to credit and the
    /// adapter grammar that names its members.
    pub fn new(accumulator: &'a mut CacheWriteAccumulator, grammar: &'a G) -> Self {
        Self {
            accumulator,
            grammar,
        }
    }
}

impl<'de, G: CacheWriteKeyGrammar> DeserializeSeed<'de> for CacheWriteDetailsSeed<'_, G> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de, G: CacheWriteKeyGrammar> Visitor<'de> for CacheWriteDetailsSeed<'_, G> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a cache-write detail object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        self.accumulator.begin_detail_snapshot();
        while let Some(member) = map.next_key_seed(MemberKeySeed {
            grammar: self.grammar,
        })? {
            let value: MemberTokenCount = map.next_value()?;
            if let Some(tokens) = value.0 {
                self.accumulator
                    .observe_detail(&member.raw_key, member.class, tokens);
            }
        }
        Ok(())
    }
}

/// A member name, already classified and copied within [`MAX_RAW_KEY_BYTES`].
struct ObservedMember {
    raw_key: String,
    class: Option<CacheWriteClass>,
}

/// Classifies and bounds a member name at the moment the deserializer produces it, so no
/// unbounded copy of a provider key is ever owned by accounting.
struct MemberKeySeed<'g, G: CacheWriteKeyGrammar> {
    grammar: &'g G,
}

impl<'de, G: CacheWriteKeyGrammar> DeserializeSeed<'de> for MemberKeySeed<'_, G> {
    type Value = ObservedMember;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_str(self)
    }
}

impl<G: CacheWriteKeyGrammar> Visitor<'_> for MemberKeySeed<'_, G> {
    type Value = ObservedMember;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a cache-write detail member name")
    }

    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
        let end = floor_char_boundary(value, MAX_RAW_KEY_BYTES);
        Ok(ObservedMember {
            raw_key: value[..end].to_owned(),
            class: self.grammar.class_of(value),
        })
    }
}

/// A member value: the token count it carries, or `None` when it does not carry one.
///
/// Values of an unexpected shape are drained without being retained, so an unfamiliar member
/// cannot make the parse fail and cannot make accounting hold provider-controlled data.
///
/// A token count is a non-negative integer inside the `u64` domain, and nothing else is coerced
/// into one. A negative count is not zero tokens — recording it as an observation of zero would
/// invent a fact about a class the provider never asserted, and it would count towards duplicate
/// ambiguity. A value too large to represent is not clamped, and **no float is accepted at all**,
/// whole-valued or not: each would put a quantity on the money path that the provider did not
/// report.
struct MemberTokenCount(Option<u64>);

impl<'de> Deserialize<'de> for MemberTokenCount {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(MemberTokenCountVisitor)
    }
}

struct MemberTokenCountVisitor;

impl<'de> Visitor<'de> for MemberTokenCountVisitor {
    type Value = MemberTokenCount;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a token count")
    }

    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
        Ok(MemberTokenCount(Some(v)))
    }

    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
        Ok(MemberTokenCount(u64::try_from(v).ok()))
    }

    fn visit_u128<E: serde::de::Error>(self, v: u128) -> Result<Self::Value, E> {
        Ok(MemberTokenCount(u64::try_from(v).ok()))
    }

    fn visit_i128<E: serde::de::Error>(self, v: i128) -> Result<Self::Value, E> {
        Ok(MemberTokenCount(u64::try_from(v).ok()))
    }

    /// No float is a token count.
    ///
    /// Not even a whole-valued one: above `2^53` an `f64` cannot represent consecutive integers,
    /// so by the time the value reaches here the deserializer has already rounded it and the
    /// original digits are unrecoverable. `9007199254740993` written as a JSON float arrives as
    /// `9007199254740992`, and `.fract() == 0.0` is true of that rounded value — an integrality
    /// check reads as a precision guarantee while proving nothing. Tokens are counted in
    /// integers end to end; a provider that means a count writes one.
    fn visit_f64<E: serde::de::Error>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(MemberTokenCount(None))
    }

    fn visit_bool<E: serde::de::Error>(self, _v: bool) -> Result<Self::Value, E> {
        Ok(MemberTokenCount(None))
    }

    fn visit_str<E: serde::de::Error>(self, _v: &str) -> Result<Self::Value, E> {
        Ok(MemberTokenCount(None))
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(MemberTokenCount(None))
    }

    fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(MemberTokenCount(None))
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_any(self)
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        while seq.next_element::<IgnoredAny>()?.is_some() {}
        Ok(MemberTokenCount(None))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(MemberTokenCount(None))
    }
}

// ---------------------------------------------------------------------------------------------
// Finalization result
// ---------------------------------------------------------------------------------------------

/// Everything one request's accounting concluded, computed at one point.
///
/// Cost, status, reconciliation facts, evidence completeness and warning facts are one value
/// rather than related facts assembled at different times by different callers: a warning cannot
/// report evidence as incomplete if the evidence is built later, and a cost cannot claim `exact`
/// if a quantity was clamped somewhere else. Headers, the terminal SSE event, the spend row,
/// structured logging, metrics and budget accounting all read this one value.
#[derive(Debug, Clone)]
pub struct FinalizedAccounting {
    /// The billing quantities this request was priced on.
    pub token_usage: TokenUsage,
    /// The resulting cost.
    pub cost: CostBreakdown,
    /// The persisted evidence document, already size-limited, or `None` when nothing was cached.
    pub evidence: Option<UsageEvidence>,
    /// Every conservative quantity policy that was applied.
    pub reconciliation: ReconciliationFacts,
    /// The bounded reason set for the single structured warning.
    pub warning: WarningFacts,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Anthropic's member grammar, used here only to drive the seed. The real one lands with the
    /// Anthropic lane.
    struct EphemeralGrammar;

    impl CacheWriteKeyGrammar for EphemeralGrammar {
        fn class_of(&self, raw_key: &str) -> Option<CacheWriteClass> {
            let duration = raw_key
                .strip_prefix("ephemeral_")?
                .strip_suffix("_input_tokens")?;
            CacheWriteClass::canonicalize(duration)
        }
    }

    fn class(raw: &str) -> CacheWriteClass {
        CacheWriteClass::canonicalize(raw).expect("canonical class")
    }

    fn registry(names: &[&str]) -> CacheWriteClassRegistry {
        CacheWriteClassRegistry::from_classes(names.iter().map(|n| class(n))).expect("registry")
    }

    fn default_registry() -> CacheWriteClassRegistry {
        registry(&["5m", "1h"])
    }

    // -- status --------------------------------------------------------------------------------

    #[test]
    fn worst_status_precedence_holds() {
        assert_eq!(
            CostStatus::Exact.worst(CostStatus::Reconciled),
            CostStatus::Reconciled
        );
        assert_eq!(
            CostStatus::Reconciled.worst(CostStatus::RateFallback),
            CostStatus::RateFallback
        );
        assert_eq!(
            CostStatus::RateFallback.worst(CostStatus::CostUnavailable),
            CostStatus::CostUnavailable
        );
        assert_eq!(
            CostStatus::CostUnavailable.worst(CostStatus::Exact),
            CostStatus::CostUnavailable
        );
        assert_eq!(
            CostStatus::Exact.worst(CostStatus::Exact),
            CostStatus::Exact
        );
    }

    /// An omitted status must not read as confidence nobody established.
    #[test]
    fn status_defaults_to_the_worst_value() {
        assert_eq!(CostStatus::default(), CostStatus::CostUnavailable);
    }

    #[test]
    fn status_wire_spelling_is_kebab_case() {
        assert_eq!(CostStatus::RateFallback.as_str(), "rate-fallback");
        assert_eq!(CostStatus::CostUnavailable.as_str(), "cost-unavailable");
        assert_eq!(
            serde_json::to_string(&CostStatus::RateFallback).expect("serialize"),
            "\"rate-fallback\""
        );
    }

    // -- canonicalization ----------------------------------------------------------------------

    #[test]
    fn canonicalization_folds_case_and_leading_zeros() {
        assert_eq!(class("5m").as_str(), "5m");
        assert_eq!(class("05m").as_str(), "5m");
        assert_eq!(class("5M").as_str(), "5m");
        assert_eq!(class("0005M").as_str(), "5m");
        assert_eq!(class("1H").as_str(), "1h");
        assert_eq!(class("30m").as_str(), "30m");
        assert_eq!(class("0m").as_str(), "0m");
        assert_eq!(class("00m").as_str(), "0m");
        assert_eq!(class("90s").as_str(), "90s");
        assert_eq!(class("7d").as_str(), "7d");
        assert_eq!(class("05m"), class("5M"));
    }

    #[test]
    fn canonicalization_rejects_non_durations() {
        for raw in [
            "", "m", "5", "5x", "5m5", "a5m", "5 m", "-5m", "5.5m", "５m",
        ] {
            assert!(
                CacheWriteClass::canonicalize(raw).is_none(),
                "expected {raw:?} to be rejected"
            );
        }
    }

    #[test]
    fn canonicalization_rejects_over_long_input() {
        let long = format!("{}m", "9".repeat(MAX_CANONICAL_CLASS_BYTES));
        assert!(CacheWriteClass::canonicalize(&long).is_none());
        let at_limit = format!("{}m", "9".repeat(MAX_CANONICAL_CLASS_BYTES - 1));
        assert!(CacheWriteClass::canonicalize(&at_limit).is_some());
    }

    #[test]
    fn canonical_class_round_trips_through_json() {
        let json = serde_json::to_string(&class("30m")).expect("serialize");
        assert_eq!(json, "\"30m\"");
        let back: CacheWriteClass = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, class("30m"));
        assert!(serde_json::from_str::<CacheWriteClass>("\"nope\"").is_err());
    }

    // -- registry ------------------------------------------------------------------------------

    #[test]
    fn registry_deduplicates_and_assigns_stable_slots() {
        let reg = registry(&["1h", "5m", "05m"]);
        assert_eq!(reg.len(), 2);
        assert!(reg.slot_of(&class("5m")).is_some());
        assert!(reg.slot_of(&class("1h")).is_some());
        assert_ne!(reg.slot_of(&class("5m")), reg.slot_of(&class("1h")));
        assert!(reg.slot_of(&class("30m")).is_none());
    }

    #[test]
    fn registry_rejects_more_classes_than_the_reserved_capacity() {
        let over: Vec<CacheWriteClass> = (0..=MAX_CONFIGURED_CACHE_WRITE_CLASSES)
            .map(|i| class(&format!("{i}m")))
            .collect();
        let count = over.len();
        assert_eq!(
            CacheWriteClassRegistry::from_classes(over),
            Err(ClassRegistryError::TooManyClasses {
                count,
                max: MAX_CONFIGURED_CACHE_WRITE_CLASSES,
            })
        );
    }

    // -- quantity reconciliation ---------------------------------------------------------------

    #[test]
    fn aggregate_only_is_consistent() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.set_reported_aggregate(2_000);
        let accounting = acc.finish();

        assert_eq!(accounting.accounted_tokens(), 2_000);
        assert_eq!(accounting.outcome(), ReconciliationOutcome::Consistent);
        assert!(!accounting.duplicate().any());
    }

    #[test]
    fn details_only_accounts_the_detail_sum() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 1_000);
        acc.observe_detail("ephemeral_1h_input_tokens", Some(class("1h")), 2_000);
        let accounting = acc.finish();

        assert_eq!(accounting.detail_tokens(), 3_000);
        assert_eq!(accounting.accounted_tokens(), 3_000);
        assert_eq!(accounting.outcome(), ReconciliationOutcome::Consistent);
        assert_eq!(accounting.class_totals().len(), 2);
        assert_eq!(accounting.unmatched_residual_tokens(), 0);
    }

    #[test]
    fn equal_aggregate_and_details_are_consistent() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.set_reported_aggregate(3_000);
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 3_000);
        let accounting = acc.finish();

        assert_eq!(accounting.accounted_tokens(), 3_000);
        assert_eq!(accounting.outcome(), ReconciliationOutcome::Consistent);
    }

    /// Aggregate 2,000 against details 1,000 + 2,000: the larger view wins and the request is
    /// reconciled rather than exact.
    #[test]
    fn details_exceeding_the_aggregate_select_the_detail_sum() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.set_reported_aggregate(2_000);
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 1_000);
        acc.observe_detail("ephemeral_1h_input_tokens", Some(class("1h")), 2_000);
        let accounting = acc.finish();

        assert_eq!(accounting.accounted_tokens(), 3_000);
        assert_eq!(
            accounting.outcome(),
            ReconciliationOutcome::DetailExceedsAggregate
        );
        assert_eq!(accounting.unmatched_residual_tokens(), 0);
    }

    /// A larger aggregate keeps its quantity, and the part no detail explained is a residual with
    /// no class — which is what makes it fallback-priced rather than defaulted.
    #[test]
    fn aggregate_exceeding_partial_details_leaves_an_unmatched_residual() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.set_reported_aggregate(5_000);
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 1_000);
        let accounting = acc.finish();

        assert_eq!(accounting.accounted_tokens(), 5_000);
        assert_eq!(
            accounting.outcome(),
            ReconciliationOutcome::AggregateExceedsDetail
        );
        assert_eq!(accounting.unmatched_residual_tokens(), 4_000);
        assert_eq!(accounting.fallback_tokens(), 4_000);
    }

    /// The accounted quantity partitions exactly into the parts the cost path prices — for
    /// ordinary quantities, which is the only case in which the identity is claimed.
    #[test]
    fn accounted_tokens_partition_into_priceable_parts() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.set_reported_aggregate(10_000);
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 1_000);
        acc.observe_detail("ephemeral_30m_input_tokens", Some(class("30m")), 2_000);
        let accounting = acc.finish();

        let class_sum: u64 = accounting.class_totals().iter().map(|t| t.tokens).sum();
        assert_eq!(
            accounting.accounted_tokens(),
            class_sum + accounting.unknown_tokens() + accounting.unmatched_residual_tokens()
        );
        assert!(accounting.partition_is_exact());
        assert!(!accounting.quantity_overflow());
    }

    #[test]
    fn duplicate_members_are_both_summed_for_a_configured_class() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 1_000);
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 500);
        let accounting = acc.finish();

        assert_eq!(accounting.detail_tokens(), 1_500);
        assert_eq!(accounting.class_totals()[0].tokens, 1_500);
        assert!(accounting.duplicate().configured_duplicate);
        assert!(!accounting.duplicate().unknown_indeterminate);
        assert_eq!(accounting.evidence_entries().len(), 2);
    }

    /// Two different raw keys folding to one canonical class are the same duplicate hazard.
    #[test]
    fn distinct_raw_keys_folding_to_one_class_raise_duplicate_ambiguity() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 1_000);
        acc.observe_detail("ephemeral_05m_input_tokens", Some(class("05m")), 400);
        let accounting = acc.finish();

        assert_eq!(accounting.class_totals()[0].tokens, 1_400);
        assert!(accounting.duplicate().configured_duplicate);
        assert_eq!(accounting.evidence_entries().len(), 2);
    }

    /// A later detail object replaces the previous one; the aggregate keeps its own semantics.
    #[test]
    fn a_new_detail_snapshot_replaces_the_previous_one() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.set_reported_aggregate(1_000);
        acc.begin_detail_snapshot();
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 1_000);
        acc.begin_detail_snapshot();
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 4_000);
        acc.set_reported_aggregate(4_000);
        let accounting = acc.finish();

        assert_eq!(accounting.detail_tokens(), 4_000);
        assert_eq!(accounting.accounted_tokens(), 4_000);
        assert_eq!(accounting.evidence_entries().len(), 1);
        assert!(!accounting.duplicate().any());
    }

    #[test]
    fn token_totals_saturate_rather_than_wrap() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), u64::MAX);
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 1_000);
        acc.observe_detail("ephemeral_9d_input_tokens", Some(class("9d")), u64::MAX);
        acc.observe_detail("ephemeral_9d_input_tokens", Some(class("9d")), 7);
        let accounting = acc.finish();

        assert_eq!(accounting.detail_tokens(), u64::MAX);
        assert_eq!(accounting.class_totals()[0].tokens, u64::MAX);
        assert_eq!(accounting.unknown_tokens(), u64::MAX);
        assert_eq!(accounting.accounted_tokens(), u64::MAX);
        assert!(accounting.quantity_overflow());
        assert!(!accounting.partition_is_exact());
    }

    /// Two configured classes, each inside the domain on its own, whose independent counters
    /// carry more than the accounted total can hold. Nothing wrapped and nothing saturated — the
    /// components simply out-sum `accounted_tokens`, and pricing them would charge a quantity
    /// that never reaches the spend row or the budget.
    #[test]
    fn independently_valid_class_totals_can_out_sum_the_accounted_quantity() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail(
            "ephemeral_5m_input_tokens",
            Some(class("5m")),
            u64::MAX / 2 + 1,
        );
        acc.observe_detail(
            "ephemeral_1h_input_tokens",
            Some(class("1h")),
            u64::MAX / 2 + 1,
        );
        let accounting = acc.finish();

        // The detail sum is where the truth was lost: the two halves add to 2^64.
        assert!(accounting.quantity_overflow());
        assert_eq!(accounting.detail_tokens(), u64::MAX);
        assert_eq!(accounting.accounted_tokens(), u64::MAX);

        let class_sum: u128 = accounting
            .class_totals()
            .iter()
            .map(|t| u128::from(t.tokens))
            .sum();
        assert!(class_sum > u128::from(accounting.accounted_tokens()));
        assert!(!accounting.partition_is_exact());
    }

    /// The case a checked sum alone cannot catch, and the reason the flag is tracked as well.
    ///
    /// One class saturates on its own, so every component still balances against
    /// `accounted_tokens` — `u64::MAX == u64::MAX` — while the true quantity was 1,000 higher.
    /// Verifying the partition with checked arithmetic would call this exact and price it.
    #[test]
    fn a_saturated_single_class_still_balances_and_only_the_flag_catches_it() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), u64::MAX);
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 1_000);
        let accounting = acc.finish();

        let class_sum: u64 = accounting.class_totals().iter().map(|t| t.tokens).sum();
        assert_eq!(
            class_sum + accounting.unknown_tokens() + accounting.unmatched_residual_tokens(),
            accounting.accounted_tokens(),
            "the components balance, which is exactly the trap"
        );
        assert!(accounting.quantity_overflow());
        assert!(!accounting.partition_is_exact());
    }

    /// The overflow fact survives into the accounting state on the unknown side too, where a
    /// single bucket absorbs every unconfigured class.
    #[test]
    fn overflow_in_the_unknown_bucket_is_reported() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail("first", None, u64::MAX);
        acc.observe_detail("second", None, 1);
        let accounting = acc.finish();

        assert!(accounting.quantity_overflow());
        assert!(!accounting.partition_is_exact());
        assert_eq!(accounting.unknown_tokens(), u64::MAX);
    }

    /// A single observation at the domain ceiling is not an overflow: nothing was lost.
    #[test]
    fn a_maximal_single_observation_is_not_an_overflow() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), u64::MAX);
        let accounting = acc.finish();

        assert!(!accounting.quantity_overflow());
        assert!(accounting.partition_is_exact());
        assert_eq!(accounting.accounted_tokens(), u64::MAX);
    }

    /// A new snapshot clears the overflow fact along with the quantities that caused it.
    #[test]
    fn a_new_detail_snapshot_clears_the_overflow_fact() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), u64::MAX);
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 1);
        acc.begin_detail_snapshot();
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 42);
        let accounting = acc.finish();

        assert!(!accounting.quantity_overflow());
        assert!(accounting.partition_is_exact());
        assert_eq!(accounting.accounted_tokens(), 42);
    }

    // -- bounded state -------------------------------------------------------------------------

    /// The reserved-slot design exists for exactly this: a configured class arriving after an
    /// adversarial run of unknown classes is still credited in full.
    #[test]
    fn a_configured_class_is_exact_however_late_it_arrives() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        for i in 0..(MAX_RETAINED_EVIDENCE_ENTRIES * 4) {
            let raw = format!("ephemeral_{}s_input_tokens", i + 1);
            acc.observe_detail(
                &raw,
                CacheWriteClass::canonicalize(&format!("{}s", i + 1)),
                3,
            );
        }
        acc.observe_detail("ephemeral_1h_input_tokens", Some(class("1h")), 12_345);
        let accounting = acc.finish();

        let one_hour = accounting
            .class_totals()
            .iter()
            .find(|t| t.class == class("1h"))
            .expect("configured class credited");
        assert_eq!(one_hour.tokens, 12_345);
        assert!(accounting.evidence_truncated());
        assert_eq!(
            accounting.evidence_entries().len(),
            MAX_RETAINED_EVIDENCE_ENTRIES
        );
    }

    #[test]
    fn a_configured_class_accumulates_after_overflow_and_exhausted_retention() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        for i in 0..(MAX_RETAINED_EVIDENCE_ENTRIES * 2) {
            let raw = format!("ephemeral_{}s_input_tokens", i + 1);
            acc.observe_detail(
                &raw,
                CacheWriteClass::canonicalize(&format!("{}s", i + 1)),
                1,
            );
        }
        for _ in 0..1_000 {
            acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 7);
        }
        let accounting = acc.finish();

        let five_minute = accounting
            .class_totals()
            .iter()
            .find(|t| t.class == class("5m"))
            .expect("configured class credited");
        assert_eq!(five_minute.tokens, 7_000);
    }

    /// Accounting-owned state is independent of how many members a provider sends and of how long
    /// their names are. Transport buffers are out of scope and are not asserted.
    #[test]
    fn accounting_owned_state_is_independent_of_provider_entry_count() {
        let long_key = format!("ephemeral_{}_input_tokens", "z".repeat(4_096));

        let mut small = CacheWriteAccumulator::new(default_registry());
        small.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 1);
        small.observe_detail("ephemeral_1h_input_tokens", Some(class("1h")), 1);
        let small = small.finish();

        let mut large = CacheWriteAccumulator::new(default_registry());
        for i in 0..5_000 {
            large.observe_detail(&format!("{long_key}{i}"), None, 1);
        }
        let large = large.finish();

        assert!(large.evidence_entries().len() <= MAX_RETAINED_EVIDENCE_ENTRIES);
        assert!(large.class_totals().len() <= default_registry().len());
        let large_key_bytes: usize = large
            .evidence_entries()
            .iter()
            .map(|e| e.raw_key.len())
            .sum();
        assert!(large_key_bytes <= MAX_RETAINED_EVIDENCE_ENTRIES * MAX_RAW_KEY_BYTES);
        assert!(small.evidence_entries().len() <= MAX_RETAINED_EVIDENCE_ENTRIES);
    }

    #[test]
    fn a_long_raw_key_is_copied_only_up_to_the_cap() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        let raw = "e".repeat(MAX_RAW_KEY_BYTES * 10);
        acc.observe_detail(&raw, None, 5);
        let accounting = acc.finish();

        assert_eq!(
            accounting.evidence_entries()[0].raw_key.len(),
            MAX_RAW_KEY_BYTES
        );
        assert!(accounting.evidence_truncated());
    }

    /// A multibyte key is cut at a character boundary, so the retained copy is still valid UTF-8.
    #[test]
    fn a_multibyte_raw_key_is_cut_at_a_character_boundary() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        let raw = "\u{4e00}".repeat(MAX_RAW_KEY_BYTES);
        acc.observe_detail(&raw, None, 5);
        let accounting = acc.finish();

        let retained = &accounting.evidence_entries()[0].raw_key;
        assert!(retained.len() <= MAX_RAW_KEY_BYTES);
        assert!(retained.chars().all(|c| c == '\u{4e00}'));
    }

    /// Unknown classes collapse into one bucket, so their duplicate identity cannot be claimed.
    #[test]
    fn unknown_duplicate_identity_is_indeterminate_not_duplicated() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail("ephemeral_30m_input_tokens", Some(class("30m")), 10);
        let single = acc.clone().finish();
        assert!(!single.duplicate().unknown_indeterminate);

        acc.observe_detail("who_knows", None, 20);
        let accounting = acc.finish();

        assert!(accounting.duplicate().unknown_indeterminate);
        assert!(!accounting.duplicate().configured_duplicate);
        assert_eq!(accounting.unknown_tokens(), 30);
    }

    // -- seeded parse --------------------------------------------------------------------------

    fn parse_details(json: &str, reg: CacheWriteClassRegistry) -> CacheWriteAccounting {
        let mut acc = CacheWriteAccumulator::new(reg);
        let mut de = serde_json::Deserializer::from_str(json);
        CacheWriteDetailsSeed::new(&mut acc, &EphemeralGrammar)
            .deserialize(&mut de)
            .expect("seeded parse");
        acc.finish()
    }

    /// Two identical members must both survive: a map type in this position would keep only the
    /// last one and silently undercharge.
    #[test]
    fn seeded_parse_keeps_both_of_two_identical_members() {
        const DUPLICATED: &str =
            r#"{"ephemeral_5m_input_tokens": 1000, "ephemeral_5m_input_tokens": 500}"#;

        let accounting = parse_details(DUPLICATED, default_registry());

        assert_eq!(accounting.detail_tokens(), 1_500);
        assert_eq!(accounting.class_totals()[0].tokens, 1_500);
        assert!(accounting.duplicate().configured_duplicate);

        // What the seeded parse is protecting against: a map in this position keeps only the
        // last member, so the same payload would silently undercharge by 1,000 tokens.
        let as_map: std::collections::HashMap<String, u64> =
            serde_json::from_str(DUPLICATED).expect("map parse");
        assert_eq!(as_map["ephemeral_5m_input_tokens"], 500);
    }

    #[test]
    fn seeded_parse_sends_unfamiliar_members_to_overflow() {
        let accounting = parse_details(
            r#"{"ephemeral_5m_input_tokens": 100, "ephemeral_30m_input_tokens": 200, "surprise": 300}"#,
            default_registry(),
        );

        assert_eq!(accounting.detail_tokens(), 600);
        assert_eq!(accounting.class_totals()[0].tokens, 100);
        assert_eq!(accounting.unknown_tokens(), 500);
        assert_eq!(accounting.evidence_entries().len(), 3);
        assert_eq!(
            accounting.evidence_entries()[1].canonical_class,
            Some(class("30m"))
        );
        assert_eq!(accounting.evidence_entries()[2].canonical_class, None);
    }

    /// A member whose value is not a token count is drained without becoming an observation.
    #[test]
    fn seeded_parse_skips_values_that_are_not_token_counts() {
        let accounting = parse_details(
            r#"{"ephemeral_5m_input_tokens": 100, "notes": {"a": [1, 2, {"b": 3}]}, "flag": true, "missing": null}"#,
            default_registry(),
        );

        assert_eq!(accounting.detail_tokens(), 100);
        assert_eq!(accounting.observation_count(), 1);
        assert_eq!(accounting.unknown_tokens(), 0);
    }

    // -- published_tokens ----------------------------------------------------------------------
    //
    // Three provider lanes each computed this independently before it moved onto
    // `CacheWriteAccounting` itself; one of the three (OpenAI/Azure) checked only
    // `reported_tokens().is_some()`, silently dropping the `observation_count() > 0` disjunct the
    // other two carried. Not reachable through `normalize_openai_usage` today — that caller always
    // sets the aggregate and the one detail together — but the two views are independent on the
    // type itself, so a provider that ever reports one without the other must not depend on which
    // lane happens to call it.

    /// Neither an aggregate nor any detail observation: nothing to publish.
    #[test]
    fn published_tokens_is_none_with_no_aggregate_and_no_details() {
        let accounting = CacheWriteAccumulator::new(default_registry()).finish();
        assert_eq!(accounting.reported_tokens(), None);
        assert_eq!(accounting.observation_count(), 0);
        assert_eq!(accounting.published_tokens(), None);
    }

    /// A detail observation with no aggregate must still publish — the exact case the divergent
    /// OpenAI-lane implementation silently returned `None` for.
    #[test]
    fn published_tokens_is_some_with_details_but_no_aggregate() {
        let accounting = parse_details(r#"{"ephemeral_5m_input_tokens": 100}"#, default_registry());
        assert_eq!(accounting.reported_tokens(), None);
        assert!(accounting.observation_count() > 0);
        assert_eq!(
            accounting.published_tokens(),
            Some(accounting.accounted_tokens())
        );
        assert_eq!(accounting.published_tokens(), Some(100));
    }

    /// An aggregate with no detail observations must still publish.
    #[test]
    fn published_tokens_is_some_with_aggregate_but_no_details() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.set_reported_aggregate(250);
        let accounting = acc.finish();
        assert_eq!(accounting.observation_count(), 0);
        assert_eq!(
            accounting.published_tokens(),
            Some(accounting.accounted_tokens())
        );
        assert_eq!(accounting.published_tokens(), Some(250));
    }

    /// A negative count is not an observation of zero tokens. Recording one would assert a fact
    /// about a class the provider never reported, and would feed duplicate detection.
    #[test]
    fn seeded_parse_rejects_negative_counts_rather_than_zeroing_them() {
        let accounting = parse_details(
            r#"{"ephemeral_5m_input_tokens": -1, "ephemeral_1h_input_tokens": -9007199254740993}"#,
            default_registry(),
        );

        assert_eq!(accounting.observation_count(), 0);
        assert_eq!(accounting.detail_tokens(), 0);
        assert!(accounting.class_totals().is_empty());
        assert!(accounting.evidence_entries().is_empty());
        assert!(!accounting.duplicate().any());
    }

    /// No float is a token count — a fractional one is not rounded, and a whole-valued one is
    /// not accepted either.
    #[test]
    fn seeded_parse_rejects_every_float_count() {
        for json in [
            r#"{"ephemeral_5m_input_tokens": 100.5}"#,
            r#"{"ephemeral_5m_input_tokens": 0.4}"#,
            r#"{"ephemeral_5m_input_tokens": 100.0}"#,
            r#"{"ephemeral_5m_input_tokens": 1e2}"#,
        ] {
            let accounting = parse_details(json, default_registry());
            assert_eq!(accounting.observation_count(), 0, "{json}");
            assert_eq!(accounting.detail_tokens(), 0, "{json}");
        }
    }

    /// The precision-loss regression, and why whole-valued floats cannot be trusted.
    ///
    /// `9007199254740993` is `2^53 + 1`: the first integer `f64` cannot represent. Written as a
    /// JSON float it reaches the visitor already rounded down to `9007199254740992`, and
    /// `.fract() == 0.0` holds of that rounded value — so an integrality check would accept it
    /// and bill one token less than the payload states, with nothing anywhere recording that a
    /// digit was lost. The same integer written without a decimal point parses as `u64` and is
    /// billed exactly.
    #[test]
    fn a_whole_valued_float_that_lost_a_digit_is_not_billed() {
        const EXACT: u64 = 9_007_199_254_740_993;

        let rounded = 9_007_199_254_740_993.0_f64;
        assert_eq!(rounded as u64, EXACT - 1, "f64 rounds 2^53 + 1 down");
        assert_eq!(rounded.fract(), 0.0, "and the rounded value looks integral");

        let as_float = parse_details(
            r#"{"ephemeral_5m_input_tokens": 9007199254740993.0}"#,
            default_registry(),
        );
        assert_eq!(as_float.observation_count(), 0);
        assert_eq!(as_float.detail_tokens(), 0);

        let as_integer = parse_details(
            r#"{"ephemeral_5m_input_tokens": 9007199254740993}"#,
            default_registry(),
        );
        assert_eq!(as_integer.detail_tokens(), EXACT);
    }

    /// A count outside the token domain is not clamped to the ceiling; it is not a count.
    #[test]
    fn seeded_parse_rejects_counts_outside_the_token_domain() {
        // `1e30`, and `2^64` — the first integer above the domain.
        let accounting = parse_details(
            r#"{"ephemeral_5m_input_tokens": 1e30, "ephemeral_1h_input_tokens": 18446744073709551616}"#,
            default_registry(),
        );

        assert_eq!(accounting.observation_count(), 0);
        assert_eq!(accounting.detail_tokens(), 0);
        assert!(!accounting.quantity_overflow());
    }

    /// `u64::MAX` itself is inside the domain and is accepted.
    #[test]
    fn seeded_parse_accepts_the_largest_representable_count() {
        let accounting = parse_details(
            r#"{"ephemeral_5m_input_tokens": 18446744073709551615}"#,
            default_registry(),
        );

        assert_eq!(accounting.detail_tokens(), u64::MAX);
        assert!(!accounting.quantity_overflow());
    }

    /// JSON cannot spell a non-finite number, so the visitor is driven directly — a
    /// self-describing format that can produce one must not turn it into a quantity either.
    #[test]
    fn the_value_visitor_rejects_every_float_including_non_finite_ones() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.5, 1.0] {
            let parsed = MemberTokenCountVisitor
                .visit_f64::<serde_json::Error>(value)
                .expect("visitor never fails");
            assert!(parsed.0.is_none(), "expected {value} to be rejected");
        }
    }

    #[test]
    fn seeded_parse_bounds_a_long_member_name() {
        let key = format!("ephemeral_{}_input_tokens", "y".repeat(2_000));
        let json = format!("{{{:?}: 42}}", key);
        let accounting = parse_details(&json, default_registry());

        assert_eq!(accounting.unknown_tokens(), 42);
        assert_eq!(
            accounting.evidence_entries()[0].raw_key.len(),
            MAX_RAW_KEY_BYTES
        );
    }

    /// A syntactically valid duration that the pricing DB does not configure is unknown.
    #[test]
    fn seeded_parse_treats_a_valid_but_unconfigured_class_as_unknown() {
        let accounting = parse_details(
            r#"{"ephemeral_30m_input_tokens": 900}"#,
            registry(&["5m", "1h"]),
        );

        assert_eq!(accounting.unknown_tokens(), 900);
        assert!(accounting.class_totals().is_empty());

        let configured = parse_details(
            r#"{"ephemeral_30m_input_tokens": 900}"#,
            registry(&["5m", "1h", "30m"]),
        );
        assert_eq!(configured.unknown_tokens(), 0);
        assert_eq!(configured.class_totals()[0].tokens, 900);
    }

    // -- evidence document ---------------------------------------------------------------------

    #[test]
    fn no_cache_write_tokens_produce_no_evidence_document() {
        let acc = CacheWriteAccumulator::new(default_registry());
        assert!(acc.finish().to_evidence(0).is_none());
    }

    #[test]
    fn evidence_document_carries_the_reconciled_quantities() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.set_reported_aggregate(2_000);
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 1_000);
        acc.observe_detail("ephemeral_1h_input_tokens", Some(class("1h")), 2_000);
        let evidence = acc.finish().to_evidence(7_500_000).expect("evidence");

        assert_eq!(evidence.schema_version, EVIDENCE_SCHEMA_VERSION);
        assert_eq!(evidence.cache_write.reported_tokens, 2_000);
        assert_eq!(evidence.cache_write.detail_tokens, 3_000);
        assert_eq!(evidence.cache_write.accounted_tokens, 3_000);
        assert_eq!(evidence.cache_write.component_cost_nano_usd, 7_500_000);
        assert_eq!(
            evidence.cache_write.reconciliation,
            ReconciliationOutcome::DetailExceedsAggregate
        );
        assert!(!evidence.cache_write.incomplete);

        let json = serde_json::to_value(&evidence).expect("serialize");
        assert_eq!(
            json["cache_write"]["reconciliation"],
            "detail-exceeds-aggregate"
        );
        assert_eq!(json["cache_write"]["entries"][0]["canonical_class"], "5m");
        assert_eq!(
            json["cache_write"]["unknown_duplicates_indeterminate"],
            false
        );
        assert_eq!(json["cache_write"]["quantity_overflow"], false);
        let back: UsageEvidence = serde_json::from_value(json).expect("round trip");
        assert_eq!(back, evidence);
    }

    /// The indeterminate marker is a persisted fact, not only a warning-time one, and it round
    /// trips as itself rather than as a duplicate count.
    #[test]
    fn the_evidence_document_persists_the_indeterminate_unknown_duplicate_marker() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail("first_unfamiliar", None, 10);
        acc.observe_detail("second_unfamiliar", None, 20);
        let accounting = acc.finish();
        assert!(accounting.duplicate().unknown_indeterminate);

        let evidence = accounting.to_evidence(1_234).expect("evidence");
        assert!(evidence.cache_write.unknown_duplicates_indeterminate);

        let json = serde_json::to_string(&evidence).expect("serialize");
        assert!(json.contains("\"unknown_duplicates_indeterminate\":true"));
        let back: UsageEvidence = serde_json::from_str(&json).expect("round trip");
        assert_eq!(back, evidence);
    }

    /// A single unknown observation asserts nothing about duplication.
    #[test]
    fn one_unknown_observation_leaves_the_document_marker_clear() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail("only_unfamiliar", None, 10);
        let evidence = acc.finish().to_evidence(0).expect("evidence");

        assert!(!evidence.cache_write.unknown_duplicates_indeterminate);
    }

    /// Document-level facts survive the size limit, however many entries it has to drop —
    /// entry omission must not be able to erase what the request concluded.
    #[test]
    fn document_level_facts_survive_entry_omission() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), u64::MAX);
        for i in 0..MAX_RETAINED_EVIDENCE_ENTRIES {
            acc.observe_detail(&format!("{}{i}", "u".repeat(MAX_RAW_KEY_BYTES)), None, 1);
        }
        let accounting = acc.finish();
        assert!(accounting.duplicate().unknown_indeterminate);
        assert!(accounting.quantity_overflow());

        let mut evidence = accounting.to_evidence(99).expect("evidence");
        let full = serde_json::to_string(&evidence).expect("serialize").len();
        evidence.limit_to_bytes(400).expect("limit");

        let serialized = serde_json::to_string(&evidence).expect("serialize");
        assert!(full > 400, "the document must actually need shrinking");
        assert!(serialized.len() <= 400);
        assert!(evidence.cache_write.entries.len() < MAX_RETAINED_EVIDENCE_ENTRIES);
        assert!(evidence.cache_write.unknown_duplicates_indeterminate);
        assert!(evidence.cache_write.quantity_overflow);
        assert!(evidence.cache_write.incomplete);
        let back: UsageEvidence = serde_json::from_str(&serialized).expect("valid json");
        assert_eq!(back, evidence);
    }

    #[test]
    fn an_undersized_document_is_left_alone() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        acc.observe_detail("ephemeral_5m_input_tokens", Some(class("5m")), 1_000);
        let mut evidence = acc.finish().to_evidence(10).expect("evidence");
        let before = evidence.clone();

        evidence
            .limit_to_bytes(CACHE_WRITE_EVIDENCE_MAX_BYTES)
            .expect("limit");
        assert_eq!(evidence, before);
        assert!(!evidence.cache_write.incomplete);
    }

    /// Trimming keys is enough here, so every observation is still described.
    #[test]
    fn oversized_keys_are_trimmed_before_entries_are_dropped() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        for i in 0..MAX_RETAINED_EVIDENCE_ENTRIES {
            acc.observe_detail(&format!("{}{i}", "k".repeat(MAX_RAW_KEY_BYTES)), None, 1);
        }
        let mut evidence = acc.finish().to_evidence(0).expect("evidence");
        assert!(serde_json::to_string(&evidence).expect("serialize").len() > 2 * 1024);

        evidence
            .limit_to_bytes(CACHE_WRITE_EVIDENCE_MAX_BYTES)
            .expect("limit");

        let serialized = serde_json::to_string(&evidence).expect("serialize");
        assert!(serialized.len() <= CACHE_WRITE_EVIDENCE_MAX_BYTES);
        assert!(evidence.cache_write.incomplete);
        assert_eq!(
            evidence.cache_write.entries.len(),
            MAX_RETAINED_EVIDENCE_ENTRIES
        );
        assert_eq!(evidence.cache_write.accounted_tokens, 32);
    }

    /// A multibyte key survives trimming as valid UTF-8 inside a valid JSON document, and only
    /// the document-level flag records it.
    #[test]
    fn a_multibyte_key_is_trimmed_safely_by_the_size_limit() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        for _ in 0..MAX_RETAINED_EVIDENCE_ENTRIES {
            acc.observe_detail(&"\u{1f600}".repeat(MAX_RAW_KEY_BYTES / 4), None, 1);
        }
        let mut evidence = acc.finish().to_evidence(0).expect("evidence");

        evidence
            .limit_to_bytes(CACHE_WRITE_EVIDENCE_MAX_BYTES)
            .expect("limit");

        let serialized = serde_json::to_string(&evidence).expect("serialize");
        assert!(serialized.len() <= CACHE_WRITE_EVIDENCE_MAX_BYTES);
        let back: UsageEvidence = serde_json::from_str(&serialized).expect("valid json");
        assert!(
            back.cache_write
                .entries
                .iter()
                .all(|e| e.raw_key.chars().all(|c| c == '\u{1f600}'))
        );
        assert!(back.cache_write.incomplete);
    }

    /// When trimming cannot get there, whole entries go from the tail and the retained ones keep
    /// their relative order.
    #[test]
    fn entries_are_omitted_from_the_tail_when_trimming_is_not_enough() {
        let mut acc = CacheWriteAccumulator::new(registry(&["5m", "1h"]));
        for i in 0..MAX_RETAINED_EVIDENCE_ENTRIES {
            acc.observe_detail(
                &format!("{}{i}", "k".repeat(MAX_RAW_KEY_BYTES)),
                Some(class("5m")),
                (i as u64) + 1,
            );
        }
        let full = acc.finish();
        let first_tokens: Vec<u64> = full.evidence_entries().iter().map(|e| e.tokens).collect();
        let mut evidence = full.to_evidence(0).expect("evidence");

        evidence.limit_to_bytes(512).expect("limit");

        let serialized = serde_json::to_string(&evidence).expect("serialize");
        assert!(serialized.len() <= 512);
        assert!(evidence.cache_write.incomplete);
        assert!(evidence.cache_write.entries.len() < MAX_RETAINED_EVIDENCE_ENTRIES);
        let retained: Vec<u64> = evidence
            .cache_write
            .entries
            .iter()
            .map(|e| e.tokens)
            .collect();
        assert_eq!(retained, first_tokens[..retained.len()]);
    }

    /// Evidence limiting happens after accounting, so it can never move a billed quantity.
    #[test]
    fn size_limiting_never_moves_an_accounted_quantity() {
        let mut acc = CacheWriteAccumulator::new(default_registry());
        for i in 0..(MAX_RETAINED_EVIDENCE_ENTRIES * 3) {
            acc.observe_detail(&format!("{}{i}", "k".repeat(MAX_RAW_KEY_BYTES)), None, 10);
        }
        let accounting = acc.finish();
        let accounted = accounting.accounted_tokens();
        let mut evidence = accounting.to_evidence(4_242).expect("evidence");

        evidence.limit_to_bytes(256).expect("limit");

        assert_eq!(evidence.cache_write.accounted_tokens, accounted);
        assert_eq!(evidence.cache_write.component_cost_nano_usd, 4_242);
    }

    // -- reconciliation and warning facts -------------------------------------------------------

    #[test]
    fn reconciliation_facts_force_reconciled_for_every_conservative_policy() {
        assert!(!ReconciliationFacts::default().requires_reconciled());

        for facts in [
            ReconciliationFacts {
                outcome: ReconciliationOutcome::DetailExceedsAggregate,
                ..Default::default()
            },
            ReconciliationFacts {
                duplicate: DuplicateAmbiguity {
                    configured_duplicate: true,
                    unknown_indeterminate: false,
                },
                ..Default::default()
            },
            ReconciliationFacts {
                duplicate: DuplicateAmbiguity {
                    configured_duplicate: false,
                    unknown_indeterminate: true,
                },
                ..Default::default()
            },
            ReconciliationFacts {
                cache_exceeds_prompt: true,
                ..Default::default()
            },
            ReconciliationFacts {
                reasoning_exceeds_completion: true,
                ..Default::default()
            },
        ] {
            assert!(facts.requires_reconciled(), "{facts:?}");
        }
    }

    /// Unknown classes are a rate problem, and rate fallback already outranks reconciled.
    #[test]
    fn unknown_classes_alone_do_not_force_reconciled() {
        let facts = ReconciliationFacts {
            unknown_classes_present: true,
            ..Default::default()
        };
        assert!(!facts.requires_reconciled());
    }

    #[test]
    fn warning_facts_record_each_reason_once() {
        let mut warning = WarningFacts::default();
        assert!(!warning.should_warn());

        warning.add(WarningReason::RateFallback);
        warning.add(WarningReason::RateFallback);
        warning.add(WarningReason::IncompleteEvidence);

        assert!(warning.should_warn());
        assert_eq!(
            warning.reasons(),
            [
                WarningReason::RateFallback,
                WarningReason::IncompleteEvidence
            ]
        );
        assert_eq!(WarningReason::RateFallback.as_str(), "rate-fallback");
    }
}
