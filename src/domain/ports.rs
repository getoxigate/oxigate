// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Domain port traits and shared types for extension points.
//!
//! Cost calculation, routing, provider adaptation, and other extension points
//! are defined here. Adapters in `src/providers/`, `src/plugins/`, and
//! `src/domain/` implement these traits.
//!
//! Cost representation: — all monetary values use integer nano-USD
//! (1 USD = 1_000_000_000 nano-USD) for exact accumulation and budget comparison.

use std::ops::{Add, Sub};
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;
use thiserror::Error;

use crate::domain::chat::{ChatRequest, ChatResponse, StreamChunk};
use crate::domain::embedding::{EmbeddingRequest, EmbeddingResponse};

/// Nano-dollars: 1 USD = 1_000_000_000 nano-USD.
///
/// Integer representation avoids floating-point rounding errors in cost
/// accumulation and budget enforcement. Handles sub-micro per-token rates. Config/JSON stay human-readable
/// (floats); conversion happens at load or display boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct NanoUsd(pub u64);

impl NanoUsd {
    /// Zero cost.
    pub const fn zero() -> Self {
        Self(0)
    }

    /// Sentinel value used when pricing data is unavailable.
    /// `LowestCost` strategy treats this as "unknown cost" and falls back to `WeightedRandom`.
    pub const MAX: Self = Self(u64::MAX);

    /// Convert from USD (float) at config/JSON boundary. Clamps negative to 0.
    #[must_use]
    pub fn from_f64_usd(v: f64) -> Self {
        if v <= 0.0 {
            return Self::zero();
        }
        let nano = (v * 1_000_000_000.0).round();
        Self(nano as u64)
    }

    /// Format for display in the [`crate::utils::CostHeader::REQUEST_COST`] header: 6 decimal places.
    #[must_use]
    pub fn to_display_string(&self) -> String {
        format!("{:.6}", self.0 as f64 / 1_000_000_000.0)
    }

    /// DB boundary only: convert to PostgreSQL BIGINT (`i64`).
    /// Values above `i64::MAX` (unreachable at per-request granularity ~$9.2B) are capped.
    #[must_use]
    pub fn as_i64(self) -> i64 {
        i64::try_from(self.0).unwrap_or_else(|_| {
            tracing::warn!(
                value = self.0,
                "NanoUsd::as_i64 overflow — capping at i64::MAX (unreachable in normal operation)"
            );
            i64::MAX
        })
    }

    /// Raw inner value for structured log fields. Prefer this over `.0` at call sites.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Returns true if this value is zero.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }

    /// Returns `floor(self * 100 / cap)` — the percentage of `cap` that `self` represents.
    ///
    /// Range: [0, ∞). Computed in u128 to avoid u64 overflow above ~$184 M spend.
    /// Returns `u64::MAX` if cap is zero (defensive guard; `GatewayConfig::validate()` rejects
    /// zero caps, so this branch should never be reached in production).
    #[must_use]
    pub fn pct_of(self, cap: NanoUsd) -> u64 {
        if cap.is_zero() {
            return u64::MAX;
        }
        // u128 headroom: u64::MAX * 100 ≈ 1.8e21, well within u128::MAX (≈ 3.4e38).
        ((self.0 as u128 * 100) / cap.0 as u128).min(u64::MAX as u128) as u64
    }

    /// DB boundary only: construct from PostgreSQL BIGINT (`i64`).
    /// This is the **required** entry point for all PostgreSQL reads of `cost_nano_usd`
    /// columns — do not bypass with `NanoUsd(v as u64)` which would silently accept
    /// corrupt negative values. `NanoUsd(raw)` is only correct when `raw` is a `u64`
    /// already (e.g. a Redis counter).
    /// Negative values indicate data corruption — treated as zero with a warning.
    #[must_use]
    pub fn from_i64(v: i64) -> Self {
        if v < 0 {
            tracing::warn!(
                value = v,
                "NanoUsd::from_i64 received negative value — data corruption? Treating as zero"
            );
            Self::zero()
        } else {
            Self(v as u64)
        }
    }
}

impl Add for NanoUsd {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.saturating_add(rhs.0))
    }
}

impl Sub for NanoUsd {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.saturating_sub(rhs.0))
    }
}

/// Token usage extracted from a provider API response.
///
/// Supports input/output and cache fields, with provider-specific extensions
/// (thinking, modalities, etc.).
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    /// Total input (prompt) tokens.
    pub input_tokens: u64,
    /// Total output (completion) tokens.
    pub output_tokens: u64,
    /// Cache-read input tokens (billed at input × cache_read_multiplier).
    pub cache_read_input_tokens: u64,
    /// Ephemeral 5-minute cache write tokens.
    pub cache_write_5m_tokens: u64,
    /// Ephemeral 1-hour cache write tokens.
    pub cache_write_1h_tokens: u64,
    /// Thinking/reasoning tokens (model-specific rate).
    pub thinking_tokens: u64,
    /// Image units billed for this request .
    pub image_count: u64,
    /// Seconds of audio billed for this request .
    pub audio_seconds: f64,
    /// Batch API request (50% discount when true).
    pub batch: bool,
    /// Tier threshold override for Gemini (input+cached for tier selection).
    pub tier_threshold_override: Option<u64>,
}

/// Cost breakdown for a single request.
///
/// Input/output/cache/modality/batch cost breakdown per request.
/// All fields use integer nano-USD.
///
/// `#[non_exhaustive]` allows new cost dimensions without breaking plugin authors
/// implementing `CostCalculator` — they use `..Default::default()` for future fields.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct CostBreakdown {
    /// Cost from input tokens.
    pub input_cost: NanoUsd,
    /// Cost from output tokens.
    pub output_cost: NanoUsd,
    /// Cost from cache-read tokens.
    pub cached_input_cost: NanoUsd,
    /// Cost from 5m cache write tokens.
    pub cache_write_5m_cost: NanoUsd,
    /// Cost from 1h cache write tokens.
    pub cache_write_1h_cost: NanoUsd,
    /// Cost from thinking tokens.
    pub thinking_cost: NanoUsd,
    /// Cost from image units .
    pub image_cost: NanoUsd,
    /// Cost from audio seconds .
    pub audio_cost: NanoUsd,
    /// Total cost (sum of components).
    pub total_cost: NanoUsd,
}

impl CostBreakdown {
    /// Returns a zero cost breakdown.
    #[must_use]
    pub fn zero() -> Self {
        Self::default()
    }
}

/// Cost calculation error.
#[derive(Debug, Error)]
pub enum CostError {
    /// Internal pricing DB or config error.
    #[error("pricing error: {0}")]
    Pricing(String),
}

/// Provider adapter error.
#[derive(Debug, Clone, Error)]
pub enum ProviderError {
    /// Provider unreachable (network, DNS, etc.).
    #[error("provider unreachable: {0}")]
    Unreachable(String),
    /// Provider returned an HTTP error.
    #[error("provider returned error {status}: {body}")]
    ProviderHttpError { status: u16, body: String },
    /// Serialization/deserialization failure.
    #[error("serialization error: {0}")]
    Serialization(String),
    /// Feature not implemented by this adapter.
    #[error("not implemented")]
    NotImplemented,
    /// Invalid request (e.g. bad parameters).
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    /// Authentication or authorization failure.
    #[error("auth error: {0}")]
    Auth(String),
    /// Model not found or not available.
    #[error("unknown model: {0}")]
    UnknownModel(String),
    /// Rate limited; retry after optional seconds.
    #[error("rate limited")]
    RateLimited {
        /// Seconds to wait before retry, if provided by provider.
        retry_after: Option<u64>,
    },
    /// Provider temporarily unavailable (5xx).
    #[error("provider unavailable: {0}")]
    ProviderUnavailable(String),
    /// Content filtered (safety, recitation, etc.).
    #[error("content filtered: {0}")]
    ContentFiltered(String),
    /// Feature not supported by this adapter.
    #[error("not supported: {0}")]
    NotSupported(String),
    /// Translation between formats failed.
    #[error("translation error: {0}")]
    Translate(String),
    /// All eligible providers are in 429 cooldown simultaneously. → HTTP 503 + Retry-After.
    #[error("all providers rate limited; retry after {retry_after}s")]
    AllProvidersRateLimited { retry_after: u64 },
    /// Internal routing misconfiguration (e.g. all weights zero). → HTTP 500.
    #[error("internal routing error: {0}")]
    Internal(String),
    /// Provider request timed out, or inter-chunk streaming deadline exceeded.
    #[error("provider timeout after {elapsed_ms}ms")]
    Timeout { elapsed_ms: u64 },
    /// `tool_choice` value not supported by the target provider. → HTTP 400.
    #[error(
        "tool_choice not supported by {provider}: '{requested}' is not supported; \
         accepted string values: {supported_values:?}; \
         for named function: {{\"type\":\"function\",\"function\":{{\"name\":\"X\"}}}}"
    )]
    ToolChoiceUnsupported {
        provider: &'static str,
        requested: String,
        supported_values: &'static [&'static str],
    },
    /// Request declares more tools than the provider allows per call. → HTTP 400.
    #[error("tool count exceeded for {provider}: requested {requested}, limit {limit}")]
    ToolCountExceeded {
        provider: &'static str,
        requested: usize,
        limit: usize,
    },
    /// One or more tool definitions are malformed. → HTTP 400.
    #[error("malformed tool schema for {provider}: {reason}")]
    MalformedToolSchema {
        provider: &'static str,
        reason: &'static str,
    },
    /// Tool-argument JSON buffer exceeded the per-call cap.
    /// Non-streaming: HTTP 502. Streaming: terminal SSE event then graceful close.
    #[error(
        "tool call buffer overflow for {provider}: \
         tool_call_id '{tool_call_id}', cap {cap_bytes} bytes"
    )]
    ToolCallBufferOverflow {
        provider: &'static str,
        tool_call_id: String,
        cap_bytes: usize,
    },
    /// Requested capability not yet supported by this adapter. → HTTP 400.
    #[error("not yet supported: {feature}")]
    NotYetSupported { feature: &'static str },
}

/// Classifies a provider's role in routing.
///
/// `FallbackOnly` providers default to weight 0.0 in `candidates()` — they are excluded
/// from normal weighted routing but remain eligible as explicit fallback targets .
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ProviderKind {
    /// Standard provider — participates in weighted routing.
    #[default]
    Primary,
    /// Fallback-only provider (e.g. an OpenAI-compat instance without `supported_models`) —
    /// excluded from normal routing; weight defaults to 0.0 unless the operator overrides
    /// via `routing.weights`.
    FallbackOnly,
}

/// Embedding capabilities metadata for a provider adapter.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct EmbeddingCapabilities {
    /// Supported output dimensions (common presets; treat as hints, not constraints).
    /// Empty = provider decides / not configurable.
    pub dimensions: Vec<u32>,
    /// Maximum input tokens for a single embedding input.
    pub max_input_tokens: u32,
    /// Whether the provider accepts a batch of inputs in a single API call.
    pub supports_batch: bool,
}

/// Provider metadata for routing and discovery.
#[derive(Debug, Clone, Default)]
pub struct ProviderMetadata {
    /// Provider display name.
    pub name: String,
    /// Models this adapter supports (empty or ["*"] for wildcard).
    pub supported_models: Vec<String>,
    /// Whether the adapter supports streaming.
    pub supports_streaming: bool,
    /// Whether the adapter supports tool/function calling.
    pub supports_tools: bool,
    /// Whether the adapter supports vision (multimodal image input).
    pub supports_vision: bool,
    /// Whether the adapter supports embeddings.
    pub supports_embeddings: bool,
    /// Whether the adapter includes thinking-capable models (Gemini 2.5+, Claude extended thinking).
    pub supports_thinking: bool,
    /// Routing role: Primary (normal routing) or FallbackOnly (excluded from weighted routing).
    pub kind: ProviderKind,
    /// Embedding capabilities metadata (None if adapter does not implement embeddings).
    pub embedding_capabilities: Option<EmbeddingCapabilities>,
}

/// Health status for provider readiness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    /// No active probe performed; status indeterminate. Routing treats this as
    /// "available, do not prefer or deprecate." Used by adapters that have not yet
    /// implemented an upstream health probe (e.g. `OpenAICompatAdapter`).
    Unknown,
}

/// Per-request snapshot of a provider's routing state.
///
/// Built by `ProviderHealthTracker::candidates()` and passed as a read-only slice
/// to `RoutingStrategy::select()`. Strategies are pure functions — no I/O.
pub struct ProviderCandidate {
    /// Provider name (matches `ProviderMetadata::name`).
    pub name: String,
    /// The adapter to dispatch to when this candidate is selected.
    pub adapter: Arc<dyn ProviderAdapter>,
    /// Configured weight (0.0 = excluded from weighted strategies).
    pub weight: f64,
    /// Current in-flight request count (lock-free snapshot).
    pub in_flight: usize,
    /// Exponentially-weighted moving average latency in milliseconds. 0.0 = no samples yet.
    pub latency_ewma_ms: f64,
    /// True when this provider is in a 429-triggered cooldown window.
    pub is_cooling_down: bool,
    /// Seconds remaining in the current cooldown window.
    /// 0 when not cooling. When cooling, contains the remaining seconds derived from the local
    /// `cooldown_until` timestamp, or falls back to the configured `cooldown_secs` when the
    /// cooldown was signalled via Redis only (no local timestamp available).
    /// Used by `RateLimitAware` to return an accurate `Retry-After` value.
    pub cooldown_remaining_secs: u64,
    /// Pre-computed input cost per million tokens for this model. `NanoUsd::MAX` = unknown.
    pub cost_per_million_tokens: NanoUsd,
}

impl std::fmt::Debug for ProviderCandidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderCandidate")
            .field("name", &self.name)
            .field("weight", &self.weight)
            .field("in_flight", &self.in_flight)
            .field("latency_ewma_ms", &self.latency_ewma_ms)
            .field("is_cooling_down", &self.is_cooling_down)
            .field("cooldown_remaining_secs", &self.cooldown_remaining_secs)
            .field("cost_per_million_tokens", &self.cost_per_million_tokens)
            .finish_non_exhaustive()
    }
}

/// Request context passed alongside candidates to `RoutingStrategy::select()`.
pub struct RoutingContext<'a> {
    /// Model name from the incoming request.
    pub model: &'a str,
}

/// Error returned by `RoutingStrategy::select()`.
#[derive(Debug)]
pub enum StrategyError {
    /// All candidates are in 429 cooldown. → HTTP 503 + Retry-After header.
    AllProvidersRateLimited { retry_after: u64 },
    /// No eligible candidates (e.g. all weights zero). → HTTP 500 / misconfiguration.
    NoEligibleCandidates,
}

/// Port: routing strategy for multi-provider dispatch.
///
/// Implementations are pure functions — they receive a pre-built candidate slice
/// and return a reference to the selected provider. No I/O, no locks.
pub trait RoutingStrategy: Send + Sync + 'static {
    /// Selects a provider from `candidates` for the given request context.
    ///
    /// `candidates` is a slice of references so `RateLimitAware` and other strategies
    /// can build sub-slices (e.g. non-cooling only) without cloning the owned values.
    ///
    /// `'s` is the lifetime of the `ProviderCandidate` values (owned by the caller's
    /// snapshot Vec). The container slice can be shorter-lived (e.g. a local `Vec<&'s _>`
    /// in `RateLimitAware`). The returned reference borrows from the candidates, not from
    /// the container slice, so callers can safely return it after dropping the sub-slice.
    fn select<'s>(
        &self,
        candidates: &[&'s ProviderCandidate],
        ctx: &RoutingContext<'_>,
    ) -> Result<&'s ProviderCandidate, StrategyError>;
}

/// Stream of SSE chunks for streaming chat completions.
pub type ChatCompletionStream =
    Pin<Box<dyn Stream<Item = Result<StreamChunk, ProviderError>> + Send>>;

/// Routing metadata returned alongside a response by [`ProviderAdapterExt`].
///
/// `providers[i]` attempted `models[i]`; the last entry is the provider that succeeded.
/// Injected as response headers by `api/chat.rs` when `expose_provider_names` is true.
#[derive(Debug, Clone, Default)]
pub struct AttemptedMeta {
    /// Providers tried in attempt order (primary first, then fallback targets).
    pub providers: Vec<String>,
    /// Models tried in attempt order — same index as `providers`.
    pub models: Vec<String>,
    /// Trigger that caused the fallback (e.g. `"rate_limit"`, `"timeout"`).
    /// `None` when primary succeeded or no fallback was attempted.
    /// Used to inject the `X-Fallback-Reason` response header .
    pub fallback_trigger: Option<String>,
    /// Whether at least one non-primary fallback target was actually dispatched.
    /// Determines `X-Fallback-Reason` presence (absent when all targets were skipped).
    pub fallback_dispatched: bool,
}

impl AttemptedMeta {
    /// Constructs metadata for a single-provider, single-attempt dispatch (no fallback).
    pub fn single(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            providers: vec![provider.into()],
            models: vec![model.into()],
            fallback_trigger: None,
            fallback_dispatched: false,
        }
    }
}

/// Port: LLM provider adapter for chat completions.
///
/// Implementations forward requests to upstream APIs (OpenAI, Anthropic, etc.).
/// First-party adapters cover OpenAI, Anthropic, Gemini, Bedrock, and Azure;
/// OpenAICompatAdapter handles third-party OpenAI-compatible providers.
///
/// Gateway-level tool schema validation has already run before dispatch; implement only
/// provider-specific constraints (e.g. tool count limits, supported tool_choice values).
#[async_trait]
pub trait ProviderAdapter: Send + Sync + 'static {
    /// Performs a non-streaming chat completion.
    async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError>;
    /// Performs a streaming chat completion. Returns NotImplemented if not supported.
    async fn chat_completion_stream(
        &self,
        _req: &ChatRequest,
    ) -> Result<ChatCompletionStream, ProviderError> {
        Err(ProviderError::NotImplemented)
    }
    /// Performs an embedding request. Returns NotImplemented if not supported.
    async fn embeddings(
        &self,
        _req: &EmbeddingRequest,
    ) -> Result<EmbeddingResponse, ProviderError> {
        Err(ProviderError::NotImplemented)
    }
    /// Returns provider metadata for routing.
    fn metadata(&self) -> &ProviderMetadata;
    /// Checks provider health.
    async fn health_check(&self) -> HealthStatus;
    /// Returns the list of underlying providers when this is a composite router (e.g. ProviderRouter).
    /// Single adapters return `None`; the caller treats the adapter as a list of one.
    fn as_providers_slice(&self) -> Option<&[std::sync::Arc<dyn ProviderAdapter>]> {
        None
    }

    /// Attempts zero-copy forwarding of raw inbound bytes to upstream (non-streaming).
    ///
    /// Default returns `None` — translation adapters (OpenAI, Anthropic, Gemini, Bedrock)
    /// inherit this and their call sites are unchanged. Only `OpenAICompatAdapter` overrides.
    /// `ChatRequest` is immutable from handler entry through adapter dispatch;
    /// `raw_body` and `req` are guaranteed to be consistent at this call site.
    async fn try_forward_raw(
        &self,
        _req: &ChatRequest,
        _raw_body: &Bytes,
    ) -> Option<Result<ChatResponse, ProviderError>> {
        None
    }

    /// Streaming counterpart of [`ProviderAdapter::try_forward_raw`].
    ///
    /// Returns `None` when stream_options injection is required (`stream_options_support: true`),
    /// `req.stream != Some(true)`, or the adapter does not support raw forwarding.
    async fn try_forward_raw_stream(
        &self,
        _req: &ChatRequest,
        _raw_body: &Bytes,
    ) -> Option<Result<ChatCompletionStream, ProviderError>> {
        None
    }
}

/// Gateway-internal extension trait: adds routing-metadata tracing on top of
/// the stable [`ProviderAdapter`] plugin ABI.
///
/// `chat_completion_with_trace` and `chat_completion_stream_with_trace` return
/// an [`AttemptedMeta`] alongside the response, allowing `api/chat.rs` to inject
/// `X-Oxigate-Attempted-Providers` / `X-Oxigate-Attempted-Models` headers without
/// polluting the `ChatResponse` domain type.
///
/// Default implementations handle single-adapter leaf nodes (providers, plugins).
/// Leaf adapters implement this trait with an empty body — they get the defaults.
/// [`crate::providers::router::ProviderRouter`] overrides both methods with full
/// fallback + retry dispatch .
///
/// `AppState.provider` is typed as `Arc<dyn ProviderAdapterExt>`, so all stored
/// adapters must implement this trait. The empty `impl ProviderAdapterExt for X {}`
/// is the only boilerplate required for leaf adapters and plugins.
///
/// # Migration for existing `ProviderAdapter` implementations
///
/// `ProviderKind` has `#[default]` set to `Primary`, so existing `ProviderMetadata`
/// literals that omit `kind` are source-compatible — just add `kind: Default::default()`.
/// The only additional step is adding `impl ProviderAdapterExt for MyAdapter {}` (empty
/// body; all methods delegate to `ProviderAdapter` via the default implementations).
/// No other changes are required to existing adapters or plugins.
#[async_trait]
pub trait ProviderAdapterExt: ProviderAdapter {
    /// Performs a non-streaming chat completion and returns routing metadata.
    ///
    /// Default: delegates to `chat_completion`; populates `AttemptedMeta` with the
    /// single adapter name and model.
    async fn chat_completion_with_trace(
        &self,
        req: &ChatRequest,
    ) -> Result<(ChatResponse, AttemptedMeta), ProviderError> {
        let response = self.chat_completion(req).await?;
        Ok((
            response,
            AttemptedMeta::single(self.metadata().name.clone(), &req.model),
        ))
    }

    /// Performs a streaming chat completion and returns routing metadata.
    ///
    /// Metadata is populated before the stream starts (primary selection is synchronous).
    /// Default: delegates to `chat_completion_stream`; populates `AttemptedMeta` with the
    /// single adapter name and model.
    ///
    /// **Resilience contract :** `ProviderRouter`'s override provides pre-stream
    /// fallback — errors before the first chunk trigger the same fallback cascade as
    /// non-streaming. Mid-stream failures (errors after the first chunk is yielded) are
    /// surfaced as stream errors; they cannot be retried without buffering the entire
    /// response, which is not yet implemented.
    async fn chat_completion_stream_with_trace(
        &self,
        req: &ChatRequest,
    ) -> Result<(ChatCompletionStream, AttemptedMeta), ProviderError> {
        let stream = self.chat_completion_stream(req).await?;
        Ok((
            stream,
            AttemptedMeta::single(self.metadata().name.clone(), &req.model),
        ))
    }

    /// Non-streaming dispatch with raw-bytes fast path and routing metadata.
    ///
    /// Tries [`ProviderAdapter::try_forward_raw`] first; on `None`, falls back to
    /// [`chat_completion_with_trace`]. `ProviderRouter` overrides this to route raw bytes
    /// through its full dispatch + fallback pipeline.
    async fn chat_completion_raw_with_trace(
        &self,
        req: &ChatRequest,
        raw_body: &Bytes,
    ) -> Result<(ChatResponse, AttemptedMeta), ProviderError> {
        if let Some(result) = self.try_forward_raw(req, raw_body).await {
            return Ok((
                result?,
                AttemptedMeta::single(self.metadata().name.clone(), &req.model),
            ));
        }
        self.chat_completion_with_trace(req).await
    }

    /// Streaming dispatch with raw-bytes fast path and routing metadata.
    ///
    /// Tries [`ProviderAdapter::try_forward_raw_stream`] first; on `None`, falls back to
    /// [`chat_completion_stream_with_trace`]. `ProviderRouter` overrides this to route raw
    /// bytes through its full streaming dispatch + fallback pipeline.
    async fn chat_completion_stream_raw_with_trace(
        &self,
        req: &ChatRequest,
        raw_body: &Bytes,
    ) -> Result<(ChatCompletionStream, AttemptedMeta), ProviderError> {
        if let Some(result) = self.try_forward_raw_stream(req, raw_body).await {
            return Ok((
                result?,
                AttemptedMeta::single(self.metadata().name.clone(), &req.model),
            ));
        }
        self.chat_completion_stream_with_trace(req).await
    }

    /// Performs an embeddings request and returns routing metadata.
    ///
    /// Default: delegates to `embeddings`; populates `AttemptedMeta` with the
    /// single adapter name and model.
    async fn embeddings_with_trace(
        &self,
        req: &EmbeddingRequest,
    ) -> Result<(EmbeddingResponse, AttemptedMeta), ProviderError> {
        let response = self.embeddings(req).await?;
        Ok((
            response,
            AttemptedMeta::single(self.metadata().name.clone(), &req.model),
        ))
    }
}

/// Port: custom cost calculation (overrides pricing database per model/provider).
///
/// Implementations consult the pricing DB, YAML overrides, or provider-specific
/// logic. `BundledCostCalculator` is the default catch-all impl.
pub trait CostCalculator: Send + Sync + 'static {
    /// Calculates cost from token usage for the given model.
    fn calculate(&self, model: &str, usage: &TokenUsage) -> Result<CostBreakdown, CostError>;
    /// Returns true if this calculator handles the model (catch-all impls return true).
    fn handles_model(&self, model: &str) -> bool;
}

/// Identifies the budget enforcement dimension for a single budget check.
///
/// This is a domain concept (business rule: team precedes tags; identity is a separate
/// budget dimension) and lives here rather than in `src/utils/` where it was initially
/// scaffolded alongside Redis key formatters.
///
/// Does NOT derive `Ord` — sort order is a business rule, not an implicit discriminant
/// property. Use `BudgetScope::sort_key` everywhere ordering is needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetScope {
    /// Per-identity (API key / user). Value = identity_id.
    Identity(String),
    /// Per-team. Value = team name from `RequestIdentity.tags["team"]`.
    Team(String),
    /// Per-tag. Value = full `"key:value"` string, e.g. `"project:chat-bot"`.
    Tag(String),
}

impl BudgetScope {
    /// Explicit sort key for check_list ordering.
    ///
    /// Business rule: team precedes all tags; within tags, alphabetical by kv string.
    /// Identity is not used in team_tag_budget check_lists but is included so this
    /// method is the single, complete ordering for any future mixed-scope list.
    #[must_use]
    pub fn sort_key(&self) -> (u8, &str) {
        match self {
            BudgetScope::Identity(id) => (0, id),
            BudgetScope::Team(name) => (1, name),
            BudgetScope::Tag(kv) => (2, kv),
        }
    }

    /// Redis key for a deduplicated budget threshold warning.
    ///
    /// Canonical format: `oxigate:budget:warned:{org_id}:{scope_type}:{value}:{pct}`
    #[must_use]
    pub fn warn_dedup_key(&self, org_id: &str, pct: u8) -> String {
        match self {
            BudgetScope::Identity(id) => {
                format!("oxigate:budget:warned:{org_id}:identity:{id}:{pct}")
            }
            BudgetScope::Team(name) => {
                format!("oxigate:budget:warned:{org_id}:team:{name}:{pct}")
            }
            BudgetScope::Tag(kv) => format!("oxigate:budget:warned:{org_id}:tag:{kv}:{pct}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use tracing_test::traced_test;

    use super::NanoUsd;

    // --- NanoUsd::sub (saturating) ---

    #[test]
    fn sub_normal() {
        assert_eq!(NanoUsd(10) - NanoUsd(3), NanoUsd(7));
    }

    #[test]
    fn sub_saturates_at_zero() {
        assert_eq!(NanoUsd(5) - NanoUsd(10), NanoUsd(0));
    }

    // --- NanoUsd::as_i64 ---

    #[test]
    fn as_i64_round_trip_normal() {
        let v = NanoUsd(1_500_000_000);
        assert_eq!(v.as_i64(), 1_500_000_000_i64);
        assert_eq!(NanoUsd::from_i64(v.as_i64()), v);
    }

    #[test]
    fn as_i64_zero() {
        assert_eq!(NanoUsd::zero().as_i64(), 0_i64);
    }

    #[traced_test]
    #[test]
    fn as_i64_overflow_caps_at_max_and_emits_warn() {
        let overflow = NanoUsd(u64::MAX);
        assert_eq!(overflow.as_i64(), i64::MAX);
        assert!(logs_contain("NanoUsd::as_i64 overflow"));
    }

    // --- NanoUsd::from_i64 ---

    #[test]
    fn from_i64_zero_is_zero() {
        assert_eq!(NanoUsd::from_i64(0), NanoUsd::zero());
    }

    #[test]
    fn from_i64_positive_round_trips() {
        let v = 42_000_000_000_i64;
        assert_eq!(NanoUsd::from_i64(v), NanoUsd(42_000_000_000));
    }

    #[traced_test]
    #[test]
    fn from_i64_negative_returns_zero_and_emits_warn() {
        assert_eq!(NanoUsd::from_i64(-1), NanoUsd::zero());
        assert!(logs_contain("NanoUsd::from_i64 received negative value"));
    }

    // --- NanoUsd::pct_of ---

    #[test]
    fn pct_of_normal() {
        let spend = NanoUsd::from_f64_usd(0.75);
        let cap = NanoUsd::from_f64_usd(1.0);
        assert_eq!(spend.pct_of(cap), 75);
    }

    #[test]
    fn pct_of_zero_spend() {
        let cap = NanoUsd::from_f64_usd(100.0);
        assert_eq!(NanoUsd::zero().pct_of(cap), 0);
    }

    #[test]
    fn pct_of_at_cap_is_100() {
        let cap = NanoUsd::from_f64_usd(50.0);
        assert_eq!(cap.pct_of(cap), 100);
    }

    #[test]
    fn pct_of_over_cap_exceeds_100() {
        let spend = NanoUsd::from_f64_usd(2.0);
        let cap = NanoUsd::from_f64_usd(1.0);
        assert_eq!(spend.pct_of(cap), 200);
    }

    #[test]
    fn pct_of_zero_cap_returns_u64_max() {
        let spend = NanoUsd::from_f64_usd(1.0);
        assert_eq!(spend.pct_of(NanoUsd::zero()), u64::MAX);
    }

    #[test]
    fn pct_of_zero_spend_zero_cap_returns_u64_max() {
        // Zero cap is always u64::MAX regardless of spend (defensive guard).
        assert_eq!(NanoUsd::zero().pct_of(NanoUsd::zero()), u64::MAX);
    }
}
