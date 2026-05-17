// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Configuration types and loader.
//!
//! Loaded with precedence: env vars > YAML file > code defaults.
//! Env var format: `OXIGATE__<SECTION>__<KEY>` (double-underscore nesting separator).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use figment::{
    Figment,
    providers::{Env, Format, Serialized, Yaml},
};
use secrecy::{ExposeSecret, Secret};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// Opaque string that masks its value and zeroizes memory on drop.
///
/// Wraps `secrecy::Secret<String>` for config credentials (DB URL, Redis URL).
/// Call `expose_secret()` only when the value is needed for a connection/auth call.
#[derive(Clone)]
pub struct SecretString(Secret<String>);

impl SecretString {
    /// Creates a SecretString from the given value. Use for tests or when
    /// constructing config programmatically (e.g. connection URLs).
    #[must_use]
    pub fn new(s: impl Into<String>) -> Self {
        Self(Secret::new(s.into()))
    }
}

impl Default for SecretString {
    fn default() -> Self {
        Self(Secret::new(String::new()))
    }
}

impl From<&str> for SecretString {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for SecretString {
    fn from(s: String) -> Self {
        Self(Secret::new(s))
    }
}

impl ExposeSecret<String> for SecretString {
    fn expose_secret(&self) -> &String {
        self.0.expose_secret()
    }
}

impl Serialize for SecretString {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Empty -> "" so figment defaults produce missing-value semantics; validate() then fails.
        // Non-empty -> "***" so we never persist real secrets.
        if self.0.expose_secret().is_empty() {
            serializer.serialize_str("")
        } else {
            serializer.serialize_str("***")
        }
    }
}

impl<'de> Deserialize<'de> for SecretString {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self(Secret::new(s)))
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("***")
    }
}

impl std::fmt::Display for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("***")
    }
}

/// Server bind and shutdown configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    /// TCP port to bind. Env: OXIGATE__SERVER__PORT
    pub port: u16,
    /// Bind address. Env: OXIGATE__SERVER__HOST
    pub host: String,
    /// Seconds to wait for in-flight requests to complete after shutdown signal.
    /// Env: OXIGATE__SERVER__DRAIN_TIMEOUT_SECS
    pub drain_timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "0.0.0.0".into(),
            drain_timeout_secs: 30,
        }
    }
}

/// Database connection configuration.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct DatabaseConfig {
    /// PostgreSQL connection URL. Credentials are masked in logs via SecretString.
    /// Env: OXIGATE__DATABASE__URL
    pub url: SecretString,
    /// Maximum pool connections. Env: OXIGATE__DATABASE__MAX_CONNECTIONS
    pub max_connections: Option<u32>,
    /// Seconds to wait when acquiring a connection from the pool (not TCP connect timeout).
    /// Env: OXIGATE__DATABASE__POOL_ACQUIRE_TIMEOUT_SECS
    pub pool_acquire_timeout_secs: Option<u64>,
}

/// Per-model pricing override (YAML). Creates or fully replaces an entry with a
/// single flat tier-0. `context_window` is required (0 = unknown/unconstrained).
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PricingOverride {
    /// Cost per input token (USD).
    pub input_per_token: f64,
    /// Cost per output token (USD).
    pub output_per_token: f64,
    /// Context window; 0 = unknown/unconstrained. Required for all overrides.
    pub context_window: u32,
    /// Fraction of input_per_token for cache-read tokens.
    #[serde(default)]
    pub cache_read_multiplier: Option<f64>,
}

/// Pricing configuration (YAML overrides). Empty by default.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Default)]
pub struct PricingConfig {
    /// Model ID → override. Overrides always win over bundled DB.
    #[serde(default)]
    pub overrides: std::collections::HashMap<String, PricingOverride>,
}

/// Auth configuration. When key is absent, all /v1/* requests are accepted (bypass mode).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct AuthConfig {
    /// Optional Bearer token. When absent, auth is bypassed with a WARN log
    /// (local dev / CI). When set, all /v1/* requests must carry a matching token.
    /// Hot-reload class: A (in-memory swap). Must not appear in logs.
    pub key: Option<SecretString>,
}

/// Redis connection configuration.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct RedisConfig {
    /// Redis connection URL (redis:// or rediss:// for TLS). Credentials masked in logs.
    /// Env: OXIGATE__REDIS__URL
    pub url: SecretString,
    /// Maximum pool connections. Env: OXIGATE__REDIS__POOL_SIZE
    pub pool_size: Option<u32>,
    /// Seconds to wait for a free pool connection before returning PoolError::Timeout.
    /// Env: OXIGATE__REDIS__POOL_TIMEOUT_SECS. Default: 5. Hot-reload class: B.
    pub pool_timeout_secs: Option<u64>,
}

// ---------------------------------------------------------------------------
// Routing configuration
// ---------------------------------------------------------------------------

fn default_cooldown_secs() -> u64 {
    60
}

fn default_latency_ewma_alpha() -> f64 {
    0.1
}

/// Which routing algorithm to use for multi-provider dispatch.
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategyKind {
    /// Weighted random selection (Walker's Alias Method). Default.
    #[default]
    WeightedRandom,
    /// Excludes cooling-down (429) providers; falls back to WeightedRandom for the rest.
    RateLimitAware,
    /// Selects the provider with the lowest input cost; tiebreaks by stable order.
    LowestCost,
}

/// Gateway-level routing configuration.
///
/// Hot-reload class: A (in-memory swap; no pool rebuild needed).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RoutingConfig {
    /// Routing strategy. Default: `weighted_random`.
    #[serde(default)]
    pub strategy: RoutingStrategyKind,
    /// Seconds to cool down a provider after a 429 response. Default: 60.
    #[serde(default = "default_cooldown_secs")]
    pub cooldown_secs: u64,
    /// EWMA smoothing factor α for latency estimates (0 < α ≤ 1). Default: 0.1.
    #[serde(default = "default_latency_ewma_alpha")]
    pub latency_ewma_alpha: f64,
    /// Provider name → routing weight. Unset providers get weight 1.0.
    #[serde(default)]
    pub weights: std::collections::HashMap<String, f64>,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            strategy: RoutingStrategyKind::default(),
            cooldown_secs: default_cooldown_secs(),
            latency_ewma_alpha: default_latency_ewma_alpha(),
            weights: std::collections::HashMap::new(),
        }
    }
}

impl RoutingConfig {
    /// Validates routing configuration against known provider names.
    ///
    /// Returns an error string (collected into the parent `GatewayConfig::validate()` errors vec)
    /// rather than `ConfigError` directly, matching the existing validation pattern.
    pub fn collect_errors(&self, provider_names: &[&str], errors: &mut Vec<String>) {
        for (key, &weight) in &self.weights {
            if !provider_names.contains(&key.as_str()) {
                errors.push(format!(
                    "routing.weights[{}]: unknown provider name (known: {})",
                    key,
                    provider_names.join(", ")
                ));
            }
            if weight < 0.0 || !weight.is_finite() {
                errors.push(format!(
                    "routing.weights[{}]: weight {} must be finite and >= 0.0",
                    key, weight
                ));
            } else if weight > f64::from(f32::MAX) {
                // WeightedRandom casts weights to f32 for Walker's Alias Method.
                // Values above f32::MAX overflow to +Inf on cast, breaking the table.
                errors.push(format!(
                    "routing.weights[{}]: weight {} exceeds f32::MAX ({}) and would overflow on cast",
                    key, weight, f32::MAX
                ));
            }
        }
        if self.latency_ewma_alpha <= 0.0 || self.latency_ewma_alpha > 1.0 {
            errors.push(format!(
                "routing.latency_ewma_alpha {} must be in (0.0, 1.0]",
                self.latency_ewma_alpha
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Fallback + retry + security configuration
// ---------------------------------------------------------------------------

/// Fallback trigger type. Parsed and stored but not acted on by default;
/// fallback fires on any error. This field activates per-trigger filtering.
///
/// Stored so configs written for filter-enabled nodes don't fail parsing on
/// nodes without filtering, during a rolling upgrade.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FallbackTrigger {
    RateLimit,
    ProviderUnavailable,
    Timeout,
    /// context-window exceeded fallback.
    ContextWindow,
    /// content-filter fallback (operator opt-in only).
    ContentFilter,
    /// Auth failure (e.g. invalid API key for a provider).
    Authentication,
    /// Model not found or not available on the provider.
    ModelNotFound,
    /// Forward-compatibility: unknown trigger values are stored but ignored.
    #[serde(other)]
    Unknown,
}

/// A single fallback target: either a provider name shorthand or an explicit provider+model pair.
///
/// String shorthand (`"azure_openai"`) forwards the original request model unchanged —
/// appropriate only for same-protocol / OpenAI-compatible setups.
/// Object form (`{ provider: "openai", model: "gpt-4o" }`) overrides the model for
/// cross-provider fallback.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum FallbackTarget {
    /// String shorthand: forward original request model to this provider.
    Provider(String),
    /// Object form: explicit provider + optional model override.
    Explicit {
        provider: String,
        /// Model to use for this fallback attempt. When absent, uses the original request model.
        #[serde(default)]
        model: Option<String>,
    },
}

impl FallbackTarget {
    /// Returns the provider name for this target.
    #[must_use]
    pub fn provider_name(&self) -> &str {
        match self {
            FallbackTarget::Provider(s) => s,
            FallbackTarget::Explicit { provider, .. } => provider,
        }
    }

    /// Returns the model override, if set. When `None`, caller uses the original request model.
    #[must_use]
    pub fn model_override(&self) -> Option<&str> {
        match self {
            FallbackTarget::Provider(_) => None,
            FallbackTarget::Explicit { model, .. } => model.as_deref(),
        }
    }
}

/// A single fallback rule: when a request to (provider, model) fails after retries,
/// try each target in order.
///
/// Match precedence (most specific wins):
///   - `provider` + `model` both set  → score 2
///   - `provider` only or `model` only → score 1
///   - neither → invalid (caught at startup)
///
/// Trigger filter. `None` = any error triggers fallback (default). `Some([...])` = only listed
/// triggers fire fallback; `Some([])` is a config error.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct FallbackRule {
    /// Source provider name (exact match). Optional when `model` is set.
    #[serde(default)]
    pub provider: Option<String>,
    /// Source model pattern. Exact match or `*`-suffix glob (e.g. `claude-*`).
    /// Optional when `provider` is set.
    #[serde(default)]
    pub model: Option<String>,
    /// Ordered fallback targets. Each tried once (no retry on fallback targets).
    pub targets: Vec<FallbackTarget>,
    /// Optional stable identifier for debugging across config reorders (non-enforced, advisory).
    #[serde(default)]
    pub key: Option<String>,
    /// Trigger filter . `None` = any error triggers fallback (backward-compat).
    /// `Some([])` = config error (rejected at startup). `Some([...])` = only listed triggers.
    #[serde(default)]
    pub on: Option<Vec<FallbackTrigger>>,
}

/// Retry policy for same-provider retries on transient errors.
///
/// Hot-reload class: A (provider rebuild on SIGHUP picks up new values).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RetryConfig {
    /// Max additional attempts after the first failure. 0 = no retry. Default: 3.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Base delay before first retry (milliseconds). Default: 100.
    #[serde(default = "default_retry_base_ms")]
    pub base_delay_ms: u64,
    /// Exponential backoff multiplier. Default: 2.0.
    #[serde(default = "default_retry_multiplier")]
    pub multiplier: f64,
    /// Maximum delay cap (milliseconds). Default: 10_000 (10 s).
    #[serde(default = "default_retry_max_delay_ms")]
    pub max_delay_ms: u64,
    /// Maximum random jitter added to each delay (milliseconds). Default: 100.
    #[serde(default = "default_retry_jitter_ms")]
    pub jitter_ms: u64,
    /// Inter-chunk silence timeout for streaming (milliseconds).
    /// If no chunk arrives within this window the stream is terminated with an error.
    /// Default: 30_000 (30 s).
    ///
    /// TODO: move to a dedicated `StreamingConfig` when mid-stream fallback adds
    /// `buffer_limit` and `commitment_point` settings — this field has no semantic
    /// relationship with retry policy and only lives here for historical reasons.
    #[serde(default = "default_stream_chunk_timeout_ms")]
    pub stream_chunk_timeout_ms: u64,
    /// Optional trigger filter for retries. `None` = retry any retryable error (backward-compat).
    /// `Some([])` = config error. `Some([...])` = only retry when trigger is in the list.
    #[serde(default)]
    pub on: Option<Vec<FallbackTrigger>>,
}

fn default_max_retries() -> u32 {
    3
}
fn default_retry_base_ms() -> u64 {
    100
}
fn default_retry_multiplier() -> f64 {
    2.0
}
fn default_retry_max_delay_ms() -> u64 {
    10_000
}
fn default_retry_jitter_ms() -> u64 {
    100
}
fn default_stream_chunk_timeout_ms() -> u64 {
    30_000
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: default_max_retries(),
            base_delay_ms: default_retry_base_ms(),
            multiplier: default_retry_multiplier(),
            max_delay_ms: default_retry_max_delay_ms(),
            jitter_ms: default_retry_jitter_ms(),
            stream_chunk_timeout_ms: default_stream_chunk_timeout_ms(),
            on: None,
        }
    }
}

/// Security and observability visibility settings.
///
/// Hot-reload class: A.
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
pub struct SecurityConfig {
    /// When `true`, inject `X-Oxigate-Attempted-Providers` and
    /// `X-Oxigate-Attempted-Models` response headers showing every
    /// provider+model attempted for this request (primary + fallbacks).
    ///
    /// Default: `false`. Exposing provider names reveals operator
    /// infrastructure topology. Also gates the planned `X-Oxigate-Provider`
    /// header from.
    #[serde(default)]
    pub expose_provider_names: bool,
}

/// Google Gemini/Vertex AI provider mode.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GeminiMode {
    /// Gemini Developer API — requires api_key.
    Api,
    /// Vertex AI — requires vertex_project, vertex_location, vertex_service_account_json.
    Vertex,
}

/// Google Gemini/Vertex AI provider configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GeminiConfig {
    /// Which Google backend to use.
    pub mode: GeminiMode,
    /// Gemini Developer API key (required when mode = Api). Class A reload.
    pub api_key: Option<SecretString>,
    /// Vertex AI GCP project ID (required when mode = Vertex). Class B — restart required.
    pub vertex_project: Option<String>,
    /// Vertex AI region, e.g. "us-central1" (required when mode = Vertex). Class B.
    pub vertex_location: Option<String>,
    /// Service account JSON (inline or path to file). Class B.
    pub vertex_service_account_json: Option<SecretString>,
    /// Default model name to use if none specified in request (optional).
    pub default_model: Option<String>,
    /// Request timeout in seconds (defaults to 120).
    pub timeout_secs: Option<u64>,
    /// API base URL for Gemini Developer API. Default: Google. Override for wiremock or custom endpoints. Class B.
    /// Backward-compat: alias for former api_base_url_override (was #[serde(skip)], test-only).
    #[serde(default, alias = "api_base_url_override")]
    pub api_base_url: Option<String>,
    /// Override Vertex base URL for testing (e.g. wiremock). When None, uses Google defaults.
    #[serde(skip)]
    pub vertex_base_url_override: Option<String>,
    /// Optional list of model IDs to report in ProviderMetadata.supported_models.
    /// Defaults to the built-in KNOWN_GEMINI_MODELS list (excluding deprecated).
    #[serde(default)]
    pub supported_models: Option<Vec<String>>,
    /// Default thinking budget for Gemini 2.5 models. 0 = disable (opt-in), -1 = dynamic, N = explicit tokens.
    /// When absent from YAML, defaults to 0 (thinking disabled).
    #[serde(default)]
    pub default_thinking_budget: Option<i32>,
    /// Gemini embedding API version. Default: "v1". Applies to API-key arm only; Vertex hardcodes /v1/. Class A reload.
    #[serde(default)]
    pub embed_api_version: Option<String>,
}

impl GeminiConfig {
    /// Resolved thinking budget: config value or 0 (opt-in default).
    #[must_use]
    pub fn resolved_thinking_budget(&self) -> i32 {
        self.default_thinking_budget.unwrap_or(0)
    }
}

/// OpenAI provider configuration.
///
/// Supports OpenAI API and compatible backends (vLLM, Ollama, Together AI) via
/// api_base_url override. Class B reload for endpoint/auth; Class A for api_key.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIConfig {
    /// API key for Authorization: Bearer. Required when providers.openai is set.
    #[serde(default)]
    pub api_key: Option<SecretString>,
    /// Default model when request omits model (e.g. gpt-4o). Class A reload.
    #[serde(default)]
    pub default_model: Option<String>,
    /// API base URL. Default: https://api.openai.com. Override for vLLM/Ollama.
    /// Class B reload.
    #[serde(default)]
    pub api_base_url: Option<String>,
    /// Request timeout in seconds. Default: 120. Class B reload.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Override known model list for supported_models. Class A reload.
    #[serde(default)]
    pub supported_models: Option<Vec<String>>,
    /// OpenAI-Organization header when using org-scoped keys.
    #[serde(default)]
    pub organization: Option<String>,
    /// OpenAI-Project header (project-scoped billing).
    #[serde(default)]
    pub project: Option<String>,
}

/// Anthropic Claude provider configuration.
///
/// Translates OpenAI-compatible requests to Anthropic Messages API.
/// Anthropic requires max_tokens on every request; default_max_tokens applies when absent.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicConfig {
    /// API key for x-api-key header. Required when providers.anthropic is set.
    #[serde(default)]
    pub api_key: Option<SecretString>,
    /// API base URL. Default: https://api.anthropic.com. Class B reload.
    #[serde(default)]
    pub api_base_url: Option<String>,
    /// Anthropic API version header. Default: 2023-06-01. Class B reload.
    #[serde(default)]
    pub anthropic_version: Option<String>,
    /// Default model when request omits model. Default: claude-sonnet-4-6. Class A reload.
    #[serde(default)]
    pub default_model: Option<String>,
    /// Default max_tokens when request has neither max_tokens nor max_completion_tokens.
    /// Anthropic rejects requests without max_tokens. Default: 4096. Class A reload.
    #[serde(default)]
    pub default_max_tokens: Option<u32>,
    /// Request timeout in seconds. Default: 120. Class B reload.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Override known model list for supported_models. Class A reload.
    #[serde(default)]
    pub supported_models: Option<Vec<String>>,
    /// Per-call streaming buffer cap for Anthropic tool-argument JSON (bytes).
    /// Default: 1 MiB (1048576). Class B reload. Env-var: PROVIDERS__ANTHROPIC__TOOL_CALL_BUFFER_CAP_BYTES.
    /// Reject 0 at config load. Very large values are untested and increase gateway memory pressure.
    #[serde(default)]
    pub tool_call_buffer_cap_bytes: Option<usize>,
}

impl AnthropicConfig {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(cap) = self.tool_call_buffer_cap_bytes {
            if cap == 0 {
                return Err(
                    "providers.anthropic.tool_call_buffer_cap_bytes must not be 0".to_string(),
                );
            }
            if cap > crate::providers::tool_limits::MAX_TOOL_CALL_BUFFER_CAP_BYTES {
                return Err(format!(
                    "providers.anthropic.tool_call_buffer_cap_bytes ({cap}) exceeds the \
                     maximum of {} bytes (64 MiB)",
                    crate::providers::tool_limits::MAX_TOOL_CALL_BUFFER_CAP_BYTES,
                ));
            }
        }
        Ok(())
    }
}

/// AWS Bedrock provider configuration .
///
/// Converse API only; phase 1 supports Claude (anthropic.*) models.
/// Credentials resolve: config fields → env vars (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).
/// Reload class: B (adapter rebuild required on credential/region/endpoint change).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BedrockConfig {
    /// AWS region, e.g. "us-east-1". Required.
    pub region: String,
    /// Explicit AWS access key ID. Falls back to AWS_ACCESS_KEY_ID env var.
    #[serde(default)]
    pub access_key_id: Option<SecretString>,
    /// Explicit AWS secret access key. Falls back to AWS_SECRET_ACCESS_KEY env var.
    #[serde(default)]
    pub secret_access_key: Option<SecretString>,
    /// Temporary STS session token. Falls back to AWS_SESSION_TOKEN env var. Optional.
    #[serde(default)]
    pub session_token: Option<SecretString>,
    /// Override endpoint URL for VPC endpoints or local mocks.
    #[serde(default)]
    pub endpoint_url: Option<String>,
    /// Default model when request omits model. Default: anthropic.claude-3-5-sonnet-20241022-v2:0.
    #[serde(default)]
    pub default_model: Option<String>,
    /// Request timeout in seconds. Default: 120.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Override known model list for supported_models.
    #[serde(default)]
    pub supported_models: Option<Vec<String>>,
}

/// OpenAI-compatible provider instance configuration (DeepSeek, OpenRouter, Kimi, etc.).
///
/// These providers speak the OpenAI wire format but are not the real OpenAI API.
/// The adapter forwards requests with zero transformation; `stream_options` injection
/// is opt-in per instance because unsupported providers return 400 on unknown fields.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAICompatConfig {
    /// Instance identifier, e.g. "deepseek" or "openrouter". Must be unique cross-provider.
    pub name: String,
    /// Upstream base URL, e.g. "https://api.deepseek.com". Must start with http:// or https://.
    pub base_url: String,
    /// API key for `Authorization: Bearer {key}`. Omit for keyless providers.
    #[serde(default)]
    pub api_key: Option<SecretString>,
    /// Models this instance handles. `None` → `FallbackOnly` (no primary routing).
    /// `Some([...])` → `Primary`; the instance participates in routing for those models.
    /// `Some([])` is a config-time error — empty list produces no selectable models.
    #[serde(default)]
    pub supported_models: Option<Vec<String>>,
    /// Inject `stream_options.include_usage: true` before forwarding streaming requests.
    /// Default: false. Set to true only for providers known to support this field
    /// (e.g. OpenRouter). Unsupported providers will return 400 if injected.
    #[serde(default)]
    pub stream_options_support: bool,
    /// Whether this provider supports tool use / function calling.
    /// Default: false. Set to true for providers that implement the OpenAI tools spec
    /// (e.g. deepseek, openrouter, together-ai, groq). Affects `/v1/models` metadata
    /// and future capability-aware routing filters.
    #[serde(default)]
    pub supports_tools: bool,
    /// Per-request HTTP timeout in seconds. Default: 120.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

impl OpenAICompatConfig {
    /// Returns true if any field differs from `other`, including the api_key secret.
    ///
    /// `#[derive(PartialEq)]` cannot be used here because `SecretString` intentionally
    /// does not implement `PartialEq` (equality on secrets creates timing-attack surface).
    /// This method is the controlled exposure point: it compares exposed values and is used
    /// only by `classify_reload()` to detect Class B changes.
    pub fn differs_from(&self, other: &Self) -> bool {
        self.name != other.name
            || self.base_url != other.base_url
            || self.supported_models != other.supported_models
            || self.stream_options_support != other.stream_options_support
            || self.supports_tools != other.supports_tools
            || self.timeout_secs != other.timeout_secs
            || self.api_key.as_ref().map(|k| k.expose_secret())
                != other.api_key.as_ref().map(|k| k.expose_secret())
    }
}

/// Azure OpenAI provider instance configuration .
///
/// Deployment-based URL, `api-key` header auth, and always-on `stream_options` injection
/// for non-zero streaming cost. Embeddings →; tool use/vision →.
/// Reload class: B (adapter rebuild required on any field change).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AzureConfig {
    /// Routing name — must be unique across all providers. Conventionally "azure-{deployment}".
    pub name: String,
    /// Azure resource endpoint, e.g. "https://my-resource.openai.azure.com". Must use https://.
    pub endpoint: String,
    /// Deployment name as shown in Azure AI Studio, e.g. "gpt-4o".
    pub deployment_name: String,
    /// API version string, e.g. "2024-10-21" (current GA stable).
    pub api_version: String,
    /// Azure API key for the `api-key` header. Do not set `Authorization`.
    pub api_key: SecretString,
    /// `None` → FallbackOnly (safe default). `Some([...])` → Primary routing for those models.
    /// `Some([])` is a config-time error.
    #[serde(default)]
    pub supported_models: Option<Vec<String>>,
    /// Per-request HTTP timeout in seconds. Default: 120.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

impl AzureConfig {
    /// Returns true if any field differs from `other`, including the api_key secret.
    ///
    /// `#[derive(PartialEq)]` cannot be used because `SecretString` intentionally does not
    /// implement `PartialEq` (equality on secrets creates timing-attack surface). Used only
    /// by `classify_reload()` to detect Class B changes.
    pub(crate) fn differs_from(&self, other: &Self) -> bool {
        self.name != other.name
            || self.endpoint != other.endpoint
            || self.deployment_name != other.deployment_name
            || self.api_version != other.api_version
            || self.supported_models != other.supported_models
            || self.timeout_secs != other.timeout_secs
            || self.api_key.expose_secret() != other.api_key.expose_secret()
    }
}

/// Provider configurations. Each provider is optional.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ProvidersConfig {
    /// OpenAI configuration.
    #[serde(default)]
    pub openai: Option<OpenAIConfig>,
    /// Anthropic Claude configuration.
    #[serde(default)]
    pub anthropic: Option<AnthropicConfig>,
    /// Google Gemini/Vertex AI configuration.
    #[serde(default)]
    pub gemini: Option<GeminiConfig>,
    /// AWS Bedrock configuration .
    #[serde(default)]
    pub bedrock: Option<BedrockConfig>,
    /// OpenAI-compatible provider instances (DeepSeek, OpenRouter, Kimi, Qwen, etc.).
    #[serde(default)]
    pub openai_compat: Vec<OpenAICompatConfig>,
    /// Azure OpenAI provider instances .
    #[serde(default)]
    pub azure: Vec<AzureConfig>,
}

/// Parsed budget reset cadence . Not serialized — derived from `budget_duration`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BudgetDuration {
    /// No automatic period suffix / reset.
    #[default]
    None,
    /// `"1d"` — daily reset at local midnight.
    Daily,
    /// `"7d"` — weekly reset Monday local midnight.
    Weekly,
    /// `"30d"` / `"1mo"` — calendar month.
    Monthly,
}

/// Budget enforcement settings​.
/// Budget cap entry for team or tag dimensions.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BudgetCapEntry {
    pub soft_cap_usd: Option<f64>,
    pub hard_cap_usd: Option<f64>,
}

///
/// Hot-reload class: A (in-memory swap — no pool rebuild needed).
///
/// Community: crash-recovery Redis seeding, `global_safety_cap_usd`, and optional
/// `budget_duration` / `timezone` for period-keyed identity spend .
/// Soft/hard per-identity caps, dedup TTL derived from `budget_duration`, and optional scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Lower bound for Postgres aggregation during Redis seeding (crash recovery).
    /// When set, also forces **unprefixed** identity Redis keys during seed (P1 —):
    /// operators cannot combine explicit reset-at with auto period suffixes.
    ///
    /// YAML may still use the legacy key `period_start` (serde alias).
    ///
    /// When absent, the current calendar month start (UTC midnight on day 1) is used for SQL.
    ///
    /// Env: `OXIGATE__BUDGET__BUDGET_RESET_AT` (RFC 3339). Legacy: `OXIGATE__BUDGET__PERIOD_START`.
    #[serde(default, alias = "period_start")]
    pub budget_reset_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Instance-wide global safety cap in USD. When set, GlobalSafetyLayer blocks all
    /// /v1/* requests with 429 if aggregate instance spend exceeds this threshold.
    /// Community feature — no feature gate. Set to opt-in; leave as None to disable.
    ///
    /// Env: OXIGATE__BUDGET__GLOBAL_SAFETY_CAP_USD
    pub global_safety_cap_usd: Option<f64>,
    /// Budget reset cadence: `"1d"`, `"7d"`, `"30d"`, `"1mo"`.
    /// `None` → no period suffix (backward-compatible Redis keys).
    ///
    /// Env: OXIGATE__BUDGET__BUDGET_DURATION
    #[serde(default)]
    pub budget_duration: Option<String>,
    /// IANA timezone for period boundaries (e.g. `US/Eastern`). Default `UTC`.
    ///
    /// Env: OXIGATE__BUDGET__TIMEZONE
    #[serde(default = "default_budget_timezone")]
    pub timezone: String,
    /// Lazily parsed IANA timezone (avoids repeated string parse on hot paths).
    /// Public so struct update (`..Default::default()`) works from integration tests; treat as opaque.
    #[serde(skip, default = "default_budget_tz_cache")]
    pub parsed_tz: Arc<OnceLock<chrono_tz::Tz>>,
    /// Per-identity soft cap threshold in USD. Converted to NanoUsd once at middleware init.
    /// Env: OXIGATE__BUDGET__SOFT_CAP_USD
    pub soft_cap_usd: Option<f64>,
    /// Per-identity hard cap in USD. When spend equals or exceeds this value,
    /// HardCapLayer returns 429. Converted to NanoUsd once at middleware init.
    /// Must be > 0 if set. May equal or exceed soft_cap_usd (hard enforcement fires
    /// at hard_cap even when soft warning fires earlier at soft_cap).
    /// Env: OXIGATE__BUDGET__HARD_CAP_USD
    pub hard_cap_usd: Option<f64>,
    /// Interval between background budget-reset scheduler wakes (seconds). Default 60.
    ///
    /// Env: OXIGATE__BUDGET__SCHEDULER_INTERVAL_SECS
    #[serde(default = "default_scheduler_interval_secs")]
    pub scheduler_interval_secs: u64,
    /// Per-team soft/hard caps (USD). Key = team name from `X-Oxigate-Team` header or tags.
    ///
    /// Community tier. Inherits `budget_duration` and `timezone` from this `BudgetConfig` —
    /// no per-team custom cadence.
    ///
    /// **Warning**: do not configure keys of the form `"team:*"` in `tag_budgets` alongside
    /// team entries — the spend writer writes a `tag:team:{name}:spend` key unconditionally,
    /// which would double-count the team spend. Use `teams` for team-scoped enforcement.
    ///
    /// Env: not individually overridable via env vars (YAML only for maps).
    #[serde(default)]
    pub teams: HashMap<String, BudgetCapEntry>,
    /// Per-tag soft/hard caps (USD). Key = full `"key:value"` string (e.g. `"project:chat-bot"`),
    /// matching the format produced by `format!("{k}:{v}")` from `RequestIdentity.tags` entries.
    ///
    /// Community tier. Inherits `budget_duration` and `timezone` from this `BudgetConfig`.
    #[serde(default)]
    pub tag_budgets: HashMap<String, BudgetCapEntry>,
}

fn default_budget_timezone() -> String {
    "UTC".to_string()
}

fn default_budget_tz_cache() -> Arc<OnceLock<chrono_tz::Tz>> {
    Arc::new(OnceLock::new())
}

fn default_scheduler_interval_secs() -> u64 {
    60
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            budget_reset_at: None,
            global_safety_cap_usd: None,
            budget_duration: None,
            timezone: default_budget_timezone(),
            parsed_tz: default_budget_tz_cache(),
            soft_cap_usd: None,
            hard_cap_usd: None,
            scheduler_interval_secs: default_scheduler_interval_secs(),
            teams: HashMap::new(),
            tag_budgets: HashMap::new(),
        }
    }
}

impl BudgetConfig {
    /// Parsed reset cadence. Invalid strings must not occur after `GatewayConfig::validate()`.
    #[must_use]
    pub fn resolved_duration(&self) -> BudgetDuration {
        match &self.budget_duration {
            None => BudgetDuration::None,
            Some(s) => crate::utils::parse_budget_duration(s).unwrap_or(BudgetDuration::None),
        }
    }

    /// Resolved IANA timezone. Invalid names must not occur after `GatewayConfig::validate()`.
    #[must_use]
    pub fn resolved_timezone(&self) -> chrono_tz::Tz {
        *self
            .parsed_tz
            .get_or_init(|| self.timezone.parse().unwrap_or(chrono_tz::UTC))
    }

    /// Dedup TTL for budget threshold WARN keys, derived from `budget_duration`.
    #[must_use]
    pub fn warn_dedup_period_secs(&self) -> u32 {
        match self.resolved_duration() {
            BudgetDuration::Daily => 86_400,
            BudgetDuration::Weekly => 604_800,
            BudgetDuration::Monthly | BudgetDuration::None => 2_592_000,
        }
    }
}

/// Top-level gateway configuration.
///
/// Loaded with precedence: env vars > YAML file > code defaults.
/// Env var format: `OXIGATE__<SECTION>__<KEY>` (double-underscore nesting separator).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GatewayConfig {
    /// Server bind and shutdown settings.
    pub server: ServerConfig,
    /// Database connection settings.
    pub database: DatabaseConfig,
    /// Redis connection settings.
    pub redis: RedisConfig,
    /// Tracing/log level. Accepted values: "trace", "debug", "info", "warn", "error".
    /// Env: OXIGATE__LOG_LEVEL. Hot-reload class: A (in-memory apply).
    pub log_level: String,
    /// Pricing overrides. Hot-reload class: A (in-memory swap).
    #[serde(default)]
    pub pricing: PricingConfig,
    /// Provider adapters. Hot-reload class: B (provider client rebuild).
    #[serde(default)]
    pub providers: ProvidersConfig,
    /// Auth config. When auth.key is absent, /v1/* bypasses validation.
    #[serde(default)]
    pub auth: AuthConfig,
    /// Budget enforcement settings. Hot-reload class: A (in-memory).
    #[serde(default)]
    pub budget: BudgetConfig,
    /// Routing strategy and load-balancing settings . Hot-reload class: A.
    #[serde(default)]
    pub routing: RoutingConfig,
    /// Ordered fallback rules . Hot-reload class: A (provider rebuild picks up changes).
    #[serde(default)]
    pub fallbacks: Vec<FallbackRule>,
    /// Retry policy for transient provider failures . Hot-reload class: A.
    #[serde(default)]
    pub retry: RetryConfig,
    /// Security and observability visibility settings . Hot-reload class: A.
    #[serde(default)]
    pub security: SecurityConfig,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            database: DatabaseConfig::default(),
            redis: RedisConfig::default(),
            log_level: "info".into(),
            pricing: PricingConfig::default(),
            providers: ProvidersConfig::default(),
            auth: AuthConfig::default(),
            budget: BudgetConfig::default(),
            routing: RoutingConfig::default(),
            fallbacks: Vec::new(),
            retry: RetryConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Load and validate
// ---------------------------------------------------------------------------

/// Configuration loading or validation error.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Failed to load or parse configuration.
    /// Boxed to avoid clippy::result_large_err — figment::Error is ~200 bytes.
    #[error("failed to load config: {0}")]
    Load(#[from] Box<figment::Error>),
    /// Post-load validation failed.
    #[error("config validation failed:\n{0}")]
    Invalid(String),
}

/// Load config from `path` with precedence: defaults → YAML file → env vars.
///
/// The `OXIGATE__` prefix maps: `OXIGATE__SERVER__PORT` → `server.port`.
pub fn load_config(path: &Path) -> Result<GatewayConfig, ConfigError> {
    let cfg: GatewayConfig = Figment::from(Serialized::defaults(GatewayConfig::default()))
        .merge(Yaml::file(path))
        .merge(Env::prefixed("OXIGATE__").split("__"))
        .extract()
        .map_err(Box::new)?;
    Ok(cfg)
}

/// Load config from `path`, validate it, and emit any advisory warnings
/// (e.g. `hard_cap_usd < soft_cap_usd`). Returns a validated config or a
/// structured error listing all problems found.
///
/// **All config startup paths must go through this function**, not through
/// `load_config` + `validate` directly, so that advisory warns are never silently skipped.
pub fn load_and_validate_config(path: &Path) -> Result<GatewayConfig, ConfigError> {
    let cfg = load_config(path)?;
    cfg.validate()?;
    Ok(cfg)
}

impl GatewayConfig {
    /// Validates soft/hard cap pairs for any budget dimension.
    fn validate_budget_caps(
        path: &str,
        soft: Option<f64>,
        hard: Option<f64>,
        errors: &mut Vec<String>,
    ) {
        if let Some(cap) = soft
            && (!cap.is_finite() || cap <= 0.0)
        {
            errors.push(format!("{path}.soft_cap_usd must be a finite number > 0"));
        }
        if let Some(cap) = hard
            && (!cap.is_finite() || cap <= 0.0)
        {
            errors.push(format!("{path}.hard_cap_usd must be a finite number > 0"));
        }
        if let (Some(h), Some(s)) = (hard, soft)
            && h < s
        {
            errors.push(format!(
                "{path}.hard_cap_usd ({h}) must be >= soft_cap_usd ({s})"
            ));
        }
    }

    /// Post-load validation: check required fields and invariants.
    /// Returns `Err(ConfigError::Invalid)` with all problems concatenated.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut errors: Vec<String> = Vec::new();

        if self.database.url.expose_secret().is_empty() {
            errors.push(
                "database.url is required (set OXIGATE__DATABASE__URL or database.url in YAML)"
                    .into(),
            );
        }

        if self.redis.url.expose_secret().is_empty() {
            errors.push(
                "redis.url is required (set OXIGATE__REDIS__URL or redis.url in YAML)".into(),
            );
        }

        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.log_level.to_lowercase().as_str()) {
            errors.push(format!(
                "log_level '{}' is not valid; accepted: {}",
                self.log_level,
                valid_levels.join(", ")
            ));
        }

        if self.server.port == 0 {
            errors.push("server.port must be > 0".into());
        }

        // auth.key, when present, must be 1–256 bytes.
        // Zero-length key: constant_time_eq would accept "Bearer " (empty token) — reject.
        // Keys >256 bytes: constant_time_eq silently truncates — reject.
        if let Some(k) = self.auth.key.as_ref() {
            let len = k.expose_secret().len();
            if len == 0 {
                errors.push("auth.key must not be empty".into());
            } else if len > 256 {
                errors.push("auth.key must be ≤256 bytes".into());
            }
        }

        for (model_key, ov) in &self.pricing.overrides {
            if ov.input_per_token < 0.0 {
                errors.push(format!(
                    "pricing.overrides[{}]: input_per_token must be >= 0",
                    model_key
                ));
            }
            if ov.output_per_token < 0.0 {
                errors.push(format!(
                    "pricing.overrides[{}]: output_per_token must be >= 0",
                    model_key
                ));
            }
            if let Some(v) = ov.cache_read_multiplier
                && !(0.0..=10.0).contains(&v)
            {
                errors.push(format!(
                    "pricing.overrides[{}]: cache_read_multiplier must be in [0.0, 10.0]",
                    model_key
                ));
            }
        }

        if let Some(cap) = self.budget.global_safety_cap_usd
            && (!cap.is_finite() || cap <= 0.0)
        {
            errors.push("budget.global_safety_cap_usd must be a finite number > 0".into());
        }

        Self::validate_budget_caps(
            "budget",
            self.budget.soft_cap_usd,
            self.budget.hard_cap_usd,
            &mut errors,
        );

        for (name, entry) in &self.budget.teams {
            Self::validate_budget_caps(
                &format!("budget.teams[{name}]"),
                entry.soft_cap_usd,
                entry.hard_cap_usd,
                &mut errors,
            );
        }

        for (kv, entry) in &self.budget.tag_budgets {
            Self::validate_budget_caps(
                &format!("budget.tag_budgets[{kv}]"),
                entry.soft_cap_usd,
                entry.hard_cap_usd,
                &mut errors,
            );
        }

        // N-1: team names must not contain ':' — it is the Redis key separator and would
        // produce ambiguous keys (e.g. oxigate:org:x:team:a:b:spend is unresolvable).
        for name in self.budget.teams.keys() {
            if name.contains(':') || name.contains('"') || name.contains('\'') {
                errors.push(format!(
                    "budget.teams[{name}]: team name must not contain ':', '\"', or '\\''"
                ));
            }
        }

        // N-1: tag_budgets keys must be in "tag_key:tag_value" format with exactly one ':'.
        // Extra colons make the Redis key ambiguous and the config lookup unreliable.
        for kv in self.budget.tag_budgets.keys() {
            let parts: Vec<&str> = kv.splitn(3, ':').collect();
            if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
                errors.push(format!(
                    "budget.tag_budgets[{kv}]: key must be exactly 'tag_key:tag_value' \
                     with one ':' separator; neither part may be empty"
                ));
            }
        }

        // N-3: tag_budgets["team:{name}"] + teams["{name}"] would double-count spend
        // for the same team (spend_writer writes both a team key and a tag key).
        for kv in self.budget.tag_budgets.keys() {
            if let Some(team_name) = kv.strip_prefix("team:")
                && self.budget.teams.contains_key(team_name)
            {
                errors.push(format!(
                    "budget.tag_budgets[\"{kv}\"] and budget.teams[\"{team_name}\"] \
                     both enforce the same team spend — this causes double-counting; \
                     remove one"
                ));
            }
        }

        if self.budget.timezone.parse::<chrono_tz::Tz>().is_err() {
            errors.push(format!(
                "budget.timezone '{}' is not a valid IANA timezone name",
                self.budget.timezone
            ));
        }
        if let Some(ref s) = self.budget.budget_duration
            && crate::utils::parse_budget_duration(s).is_err()
        {
            errors.push(format!(
                "budget.budget_duration '{s}' is not a valid duration string (e.g. \"1d\", \"7d\", \"30d\", \"1mo\")"
            ));
        }

        if self.budget.budget_reset_at.is_some() && self.budget.budget_duration.is_some() {
            errors.push(
                "budget.budget_reset_at and budget.budget_duration cannot both be set; \
                 use budget_duration alone for automatic period resets, or \
                 budget_reset_at alone for manual crash-recovery seeding"
                    .into(),
            );
        }

        // providers.openai validation
        if let Some(ref openai) = self.providers.openai {
            #[allow(clippy::unnecessary_map_or)]
            if openai.api_key.is_none()
                || openai
                    .api_key
                    .as_ref()
                    .map_or(true, |k| k.expose_secret().is_empty())
            {
                errors.push("providers.openai.api_key is required".into());
            }
        }

        // providers.anthropic validation
        if let Some(ref anthropic) = self.providers.anthropic {
            #[allow(clippy::unnecessary_map_or)]
            if anthropic.api_key.is_none()
                || anthropic
                    .api_key
                    .as_ref()
                    .map_or(true, |k| k.expose_secret().is_empty())
            {
                errors.push("providers.anthropic.api_key is required".into());
            }
            if let Err(e) = anthropic.validate() {
                errors.push(e);
            }
        }

        // providers.gemini validation
        if let Some(ref gemini) = self.providers.gemini {
            match gemini.mode {
                GeminiMode::Api => {
                    #[allow(clippy::unnecessary_map_or)]
                    if gemini.api_key.is_none()
                        || gemini
                            .api_key
                            .as_ref()
                            .map_or(true, |k| k.expose_secret().is_empty())
                    {
                        errors.push("providers.gemini.api_key is required when mode is api".into());
                    }
                }
                GeminiMode::Vertex => {
                    #[allow(clippy::unnecessary_map_or)]
                    if gemini.vertex_project.is_none()
                        || gemini
                            .vertex_project
                            .as_ref()
                            .map_or(true, |s| s.is_empty())
                    {
                        errors.push(
                            "providers.gemini.vertex_project is required when mode is vertex"
                                .into(),
                        );
                    }
                    #[allow(clippy::unnecessary_map_or)]
                    if gemini.vertex_location.is_none()
                        || gemini
                            .vertex_location
                            .as_ref()
                            .map_or(true, |s| s.is_empty())
                    {
                        errors.push(
                            "providers.gemini.vertex_location is required when mode is vertex"
                                .into(),
                        );
                    }
                    #[allow(clippy::unnecessary_map_or)]
                    if gemini.vertex_service_account_json.is_none()
                        || gemini
                            .vertex_service_account_json
                            .as_ref()
                            .map_or(true, |s| s.expose_secret().is_empty())
                    {
                        errors.push(
                            "providers.gemini.vertex_service_account_json is required when mode is vertex"
                                .into(),
                        );
                    }
                }
            }
            if let Some(ref v) = gemini.embed_api_version
                && (v.trim().is_empty() || v.contains(char::is_whitespace))
            {
                errors.push(
                    "providers.gemini.embed_api_version must not be empty or contain whitespace"
                        .into(),
                );
            }
        }

        // providers.openai_compat[] validation
        // reserved_names mirrors the named provider fields on ProvidersConfig (openai, anthropic,
        // gemini, bedrock, azure) plus retired names (passthrough). Update both together.
        let reserved_names = [
            "openai",
            "anthropic",
            "gemini",
            "bedrock",
            "azure",
            "passthrough",
        ];
        let mut seen_compat_names: std::collections::HashSet<&str> =
            std::collections::HashSet::new();
        for (i, compat) in self.providers.openai_compat.iter().enumerate() {
            if compat.name.is_empty() {
                errors.push(format!(
                    "providers.openai_compat[{i}].name must not be empty"
                ));
            }
            if compat.base_url.is_empty() {
                errors.push(format!(
                    "providers.openai_compat[{i}].base_url must not be empty"
                ));
            } else if !compat.base_url.starts_with("http://")
                && !compat.base_url.starts_with("https://")
            {
                errors.push(format!(
                    "providers.openai_compat[{i}].base_url must start with http:// or https://; got {:?}",
                    compat.base_url
                ));
            }
            if reserved_names.contains(&compat.name.as_str()) {
                errors.push(format!(
                    "providers.openai_compat[{i}].name {:?} is reserved; choose a different name",
                    compat.name
                ));
            }
            if !compat.name.is_empty() && !seen_compat_names.insert(compat.name.as_str()) {
                errors.push(format!(
                    "providers.openai_compat: duplicate name {:?}",
                    compat.name
                ));
            }
            if matches!(&compat.supported_models, Some(ms) if ms.is_empty()) {
                errors.push(format!(
                    "providers.openai_compat[{i}] ({:?}): supported_models must not be an empty list; \
                     omit the field for FallbackOnly routing, or list at least one model",
                    compat.name
                ));
            }
        }

        // providers.azure[] validation
        let mut seen_azure_names: std::collections::HashSet<&str> =
            std::collections::HashSet::new();
        for (i, azure) in self.providers.azure.iter().enumerate() {
            if azure.name.is_empty() {
                errors.push(format!("providers.azure[{i}].name must not be empty"));
            }
            if reserved_names.contains(&azure.name.as_str()) {
                errors.push(format!(
                    "providers.azure[{i}].name {:?} is reserved; choose a different name",
                    azure.name
                ));
            }
            if !azure.name.is_empty() && seen_compat_names.contains(azure.name.as_str()) {
                errors.push(format!(
                    "providers.azure[{i}].name {:?} conflicts with an openai_compat name",
                    azure.name
                ));
            }
            if !azure.name.is_empty() && !seen_azure_names.insert(azure.name.as_str()) {
                errors.push(format!("providers.azure: duplicate name {:?}", azure.name));
            }
            if !azure.endpoint.starts_with("https://") {
                errors.push(format!(
                    "providers.azure[{i}].endpoint must start with https://; got {:?}",
                    azure.endpoint
                ));
            }
            const URL_UNSAFE: &[char] = &['/', '?', '#', '&', '%'];
            if azure.deployment_name.is_empty() {
                errors.push(format!(
                    "providers.azure[{i}].deployment_name must not be empty"
                ));
            } else if azure.deployment_name.contains(URL_UNSAFE)
                || azure
                    .deployment_name
                    .chars()
                    .any(|c| c.is_ascii_whitespace())
            {
                errors.push(format!(
                    "providers.azure[{i}].deployment_name {:?} contains an unsafe URL character (/, ?, #, &, %, whitespace) — OWASP A03",
                    azure.deployment_name
                ));
            }
            if azure.api_version.is_empty() {
                errors.push(format!(
                    "providers.azure[{i}].api_version must not be empty"
                ));
            } else if azure.api_version.contains(URL_UNSAFE)
                || azure.api_version.chars().any(|c| c.is_ascii_whitespace())
            {
                errors.push(format!(
                    "providers.azure[{i}].api_version {:?} contains an unsafe character (/, ?, #, &, %, whitespace) — OWASP A03",
                    azure.api_version
                ));
            }
            let api_key = azure.api_key.expose_secret();
            if api_key.is_empty() {
                errors.push(format!("providers.azure[{i}].api_key must not be empty"));
            } else if api_key.contains('\r') || api_key.contains('\n') {
                errors.push(format!(
                    "providers.azure[{i}].api_key contains a CR or LF character — OWASP A03 CRLF injection guard"
                ));
            }
            if matches!(&azure.supported_models, Some(ms) if ms.is_empty()) {
                errors.push(format!(
                    "providers.azure[{i}] ({:?}): supported_models must not be an empty list; \
                     omit the field for FallbackOnly routing, or list at least one model",
                    azure.name
                ));
            }
        }

        // routing validation
        let mut provider_names: Vec<&str> = Vec::new();
        if self.providers.openai.is_some() {
            provider_names.push("openai");
        }
        if self.providers.anthropic.is_some() {
            provider_names.push("anthropic");
        }
        if self.providers.gemini.is_some() {
            provider_names.push("gemini");
        }
        if self.providers.bedrock.is_some() {
            provider_names.push("bedrock");
        }
        // compat instances by name
        let compat_names: Vec<String> = self
            .providers
            .openai_compat
            .iter()
            .filter(|c| !c.name.is_empty())
            .map(|c| c.name.clone())
            .collect();
        for n in &compat_names {
            provider_names.push(n.as_str());
        }
        let azure_names: Vec<String> = self
            .providers
            .azure
            .iter()
            .filter(|a| !a.name.is_empty())
            .map(|a| a.name.clone())
            .collect();
        for n in &azure_names {
            provider_names.push(n.as_str());
        }
        self.routing.collect_errors(&provider_names, &mut errors);

        // fallback + retry validation
        self.validate_fallbacks(&provider_names, &mut errors);
        self.validate_retry(&mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Invalid(errors.join("\n")))
        }
    }

    /// Validates that a name contains only safe characters for use in HTTP header values.
    /// Rejects commas, newlines, control characters — prevents header injection in
    /// `X-Oxigate-Attempted-Providers` and `X-Oxigate-Attempted-Models`.
    fn is_safe_header_name(s: &str) -> bool {
        s.chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
    }

    fn validate_fallbacks(&self, provider_names: &[&str], errors: &mut Vec<String>) {
        use std::collections::HashMap;

        for (i, rule) in self.fallbacks.iter().enumerate() {
            if rule.provider.is_none() && rule.model.is_none() {
                errors.push(format!(
                    "fallbacks[{i}]: must have at least one of `provider` or `model`"
                ));
            }
            if rule.targets.is_empty() {
                errors.push(format!("fallbacks[{i}]: `targets` must not be empty"));
            }
            if matches!(&rule.on, Some(list) if list.is_empty()) {
                errors.push(format!(
                    "fallbacks[{i}]: `on` list is empty — omit the field for any-error fallback, or list at least one trigger"
                ));
            }
            // Validate source provider name for header safety.
            if let Some(ref p) = rule.provider
                && !Self::is_safe_header_name(p)
            {
                errors.push(format!(
                    "fallbacks[{i}].provider {p:?}: only alphanumeric, -, _, ., / allowed"
                ));
            }
            // Validate model pattern: only exact match or `*`-suffix glob.
            if let Some(ref pat) = rule.model {
                let star_count = pat.chars().filter(|&c| c == '*').count();
                if star_count > 1 || (star_count == 1 && !pat.ends_with('*')) {
                    errors.push(format!(
                        "fallbacks[{i}].model {pat:?}: only `*`-suffix glob is supported (e.g. `claude-*`)"
                    ));
                }
            }
            for (j, target) in rule.targets.iter().enumerate() {
                let name = target.provider_name();
                if !Self::is_safe_header_name(name) {
                    errors.push(format!(
                        "fallbacks[{i}].targets[{j}]: provider name {name:?}: only alphanumeric, -, _, ., / allowed"
                    ));
                }
                if let Some(m) = target.model_override()
                    && !Self::is_safe_header_name(m)
                {
                    errors.push(format!(
                        "fallbacks[{i}].targets[{j}]: model {m:?}: only alphanumeric, -, _, ., / allowed"
                    ));
                }
                if !provider_names.contains(&name) {
                    errors.push(format!(
                        "fallbacks[{i}].targets[\"{name}\"]: unknown provider (configured: {})",
                        provider_names.join(", ")
                    ));
                }
            }
        }

        // Cycle detection across provider-level rules.
        // Same-provider targets (source == target provider) are NOT edges — they are
        // same-provider model downgrades, not routing loops.
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for rule in &self.fallbacks {
            if let Some(ref src) = rule.provider {
                let targets: Vec<&str> = rule
                    .targets
                    .iter()
                    .map(|t| t.provider_name())
                    .filter(|&t| t != src.as_str())
                    .collect();
                if !targets.is_empty() {
                    adj.entry(src.as_str()).or_default().extend(targets);
                }
            }
        }
        if let Some(cycle) = detect_cycle(&adj) {
            errors.push(format!(
                "fallbacks: cycle detected: {} — fallback chains must be acyclic",
                cycle.join(" → ")
            ));
        }
    }

    fn validate_retry(&self, errors: &mut Vec<String>) {
        if self.retry.multiplier < 1.0 || !self.retry.multiplier.is_finite() {
            errors.push(format!(
                "retry.multiplier {} must be finite and >= 1.0",
                self.retry.multiplier
            ));
        }
        if self.retry.max_delay_ms < self.retry.base_delay_ms {
            errors.push(format!(
                "retry.max_delay_ms ({}) must be >= base_delay_ms ({})",
                self.retry.max_delay_ms, self.retry.base_delay_ms
            ));
        }
        if self.retry.stream_chunk_timeout_ms == 0 {
            errors.push(
                "retry.stream_chunk_timeout_ms must be > 0 (zero makes all streaming instantly fail)"
                    .into(),
            );
        }
        if matches!(&self.retry.on, Some(list) if list.is_empty()) {
            errors.push(
                "retry.on list is empty — omit the field to retry any retryable error, or list at least one trigger".into(),
            );
        }
    }
}

/// Detects a cycle in a directed graph represented as an adjacency list.
/// Returns the cycle path if found (nodes in traversal order), or `None`.
fn detect_cycle<'a>(adj: &std::collections::HashMap<&'a str, Vec<&'a str>>) -> Option<Vec<String>> {
    use std::collections::HashSet;

    fn dfs<'a>(
        node: &'a str,
        adj: &std::collections::HashMap<&'a str, Vec<&'a str>>,
        visited: &mut HashSet<&'a str>,
        in_stack: &mut Vec<&'a str>,
    ) -> Option<Vec<String>> {
        if in_stack.contains(&node) {
            // Found a cycle — return path from the repeated node.
            let start = in_stack.iter().position(|&n| n == node).unwrap_or(0);
            let mut cycle: Vec<String> =
                in_stack[start..].iter().map(|s| (*s).to_string()).collect();
            cycle.push(node.to_string());
            return Some(cycle);
        }
        if visited.contains(node) {
            return None;
        }
        visited.insert(node);
        in_stack.push(node);
        if let Some(neighbours) = adj.get(node) {
            for &next in neighbours {
                if let Some(cycle) = dfs(next, adj, visited, in_stack) {
                    return Some(cycle);
                }
            }
        }
        in_stack.pop();
        None
    }

    let mut visited = std::collections::HashSet::new();
    for &node in adj.keys() {
        if !visited.contains(node) {
            let mut in_stack = Vec::new();
            if let Some(cycle) = dfs(node, adj, &mut visited, &mut in_stack) {
                return Some(cycle);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Hot-reload classification
// ---------------------------------------------------------------------------

/// Reload class for a config change, per.
#[derive(Debug, PartialEq, Eq)]
pub enum HotReloadClass {
    /// In-memory atomic swap — no I/O required. (e.g. log_level, routing weights, budget caps)
    ClassA,
    /// Rebuild client/pool and atomically swap. (e.g. provider endpoints, DB/Redis pool)
    ClassB,
    /// Restart required — cannot be hot-reloaded. (e.g. bind port, TLS identity)
    ClassC,
}

/// Classify the most restrictive reload class needed to apply `new` over `old`.
///
/// Returns `ClassC` if any Class C field changed, else `ClassB` if any Class B field
/// changed, else `ClassA`.
pub fn classify_reload(old: &GatewayConfig, new: &GatewayConfig) -> HotReloadClass {
    if old.server.port != new.server.port || old.server.host != new.server.host {
        return HotReloadClass::ClassC;
    }

    // database.*, redis.*: Class B — rebuild provider client / pool (per oxigate.yaml)
    if old.database.url.expose_secret() != new.database.url.expose_secret()
        || old.database.max_connections != new.database.max_connections
        || old.database.pool_acquire_timeout_secs != new.database.pool_acquire_timeout_secs
        || old.redis.url.expose_secret() != new.redis.url.expose_secret()
        || old.redis.pool_size != new.redis.pool_size
        || old.redis.pool_timeout_secs != new.redis.pool_timeout_secs
    {
        return HotReloadClass::ClassB;
    }

    // providers.openai_compat[]: any addition, removal, or field change → Class B
    if old.providers.openai_compat.len() != new.providers.openai_compat.len()
        || old
            .providers
            .openai_compat
            .iter()
            .zip(new.providers.openai_compat.iter())
            .any(|(o, n)| o.differs_from(n))
    {
        return HotReloadClass::ClassB;
    }

    // providers.azure[]: any addition, removal, reorder, or field change → Class B
    {
        let old_map: std::collections::HashMap<&str, &AzureConfig> = old
            .providers
            .azure
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect();
        let changed = old.providers.azure.len() != new.providers.azure.len()
            || new.providers.azure.iter().any(|n| {
                old_map
                    .get(n.name.as_str())
                    .is_none_or(|o| o.differs_from(n))
            });
        if changed {
            return HotReloadClass::ClassB;
        }
    }

    // providers.openai Class B fields (api_base_url, timeout, organization, project)
    let old_openai = old.providers.openai.as_ref();
    let new_openai = new.providers.openai.as_ref();
    if old_openai.map(|o| o.api_base_url.as_ref()) != new_openai.map(|o| o.api_base_url.as_ref())
        || old_openai.map(|o| o.timeout_secs) != new_openai.map(|o| o.timeout_secs)
        || old_openai.and_then(|o| o.organization.as_ref())
            != new_openai.and_then(|o| o.organization.as_ref())
        || old_openai.and_then(|o| o.project.as_ref())
            != new_openai.and_then(|o| o.project.as_ref())
    {
        return HotReloadClass::ClassB;
    }

    // providers.anthropic Class B fields (api_base_url, timeout, anthropic_version,
    // tool_call_buffer_cap_bytes — cap_bytes is resolved in AnthropicAdapter::new, so
    // a change requires adapter reconstruction, not just an in-memory swap)
    let old_anthropic = old.providers.anthropic.as_ref();
    let new_anthropic = new.providers.anthropic.as_ref();
    if old_anthropic.map(|a| a.api_base_url.as_ref())
        != new_anthropic.map(|a| a.api_base_url.as_ref())
        || old_anthropic.map(|a| a.timeout_secs) != new_anthropic.map(|a| a.timeout_secs)
        || old_anthropic.map(|a| a.anthropic_version.as_ref())
            != new_anthropic.map(|a| a.anthropic_version.as_ref())
        || old_anthropic.map(|a| a.tool_call_buffer_cap_bytes)
            != new_anthropic.map(|a| a.tool_call_buffer_cap_bytes)
    {
        return HotReloadClass::ClassB;
    }

    // providers.gemini Class B fields (mode, vertex_*)
    let old_gemini = old.providers.gemini.as_ref();
    let new_gemini = new.providers.gemini.as_ref();
    if old_gemini.map(|g| &g.mode) != new_gemini.map(|g| &g.mode)
        || old_gemini.and_then(|g| g.vertex_project.as_ref())
            != new_gemini.and_then(|g| g.vertex_project.as_ref())
        || old_gemini.and_then(|g| g.vertex_location.as_ref())
            != new_gemini.and_then(|g| g.vertex_location.as_ref())
        || old_gemini
            .and_then(|g| g.vertex_service_account_json.as_ref())
            .map(SecretString::expose_secret)
            != new_gemini
                .and_then(|g| g.vertex_service_account_json.as_ref())
                .map(SecretString::expose_secret)
    {
        return HotReloadClass::ClassB;
    }

    // providers.gemini Class A fields (api_key, embed_api_version — triggers adapter rebuild on SIGHUP)
    if old_gemini
        .and_then(|g| g.api_key.as_ref())
        .map(SecretString::expose_secret)
        != new_gemini
            .and_then(|g| g.api_key.as_ref())
            .map(SecretString::expose_secret)
        || old_gemini.and_then(|g| g.embed_api_version.as_deref())
            != new_gemini.and_then(|g| g.embed_api_version.as_deref())
    {
        return HotReloadClass::ClassA;
    }

    // Pricing changes, auth.key, providers.gemini Class A: Class A (in-memory swap)
    if old.pricing.overrides != new.pricing.overrides {
        return HotReloadClass::ClassA;
    }
    // auth.key: Class A — swap Arc<AuthConfig> in AppState
    let old_key = old.auth.key.as_ref().map(SecretString::expose_secret);
    let new_key = new.auth.key.as_ref().map(SecretString::expose_secret);
    if old_key != new_key {
        return HotReloadClass::ClassA;
    }

    // routing config : Class A — strategy, weights, and cooldown params are in-memory.
    // PartialEq on RoutingConfig uses f64 == comparison; NaN/infinite are rejected at validate().
    if old.routing != new.routing {
        return HotReloadClass::ClassA;
    }

    // fallback/retry/security : Class A — provider is rebuilt on SIGHUP picking up new values.
    // RetryConfig.multiplier uses f64 == comparison; NaN/infinite rejected at validate().
    if old.fallbacks != new.fallbacks || old.retry != new.retry || old.security != new.security {
        return HotReloadClass::ClassA;
    }

    HotReloadClass::ClassA
}

/// Atomically applies a new config to the active shared config.
/// Used by both Class A and Class B SIGHUP reload paths.
pub async fn apply_config_reload(
    active_config: &tokio::sync::RwLock<GatewayConfig>,
    new_cfg: GatewayConfig,
) {
    let mut guard = active_config.write().await;
    *guard = new_cfg;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn yaml_path(content: &str) -> std::path::PathBuf {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(content.as_bytes()).expect("write");
        f.flush().expect("flush");
        f.into_temp_path().keep().expect("keep")
    }

    #[test]
    fn test_env_overrides_yaml() {
        figment::Jail::expect_with(|jail| {
            jail.create_file(
                "config.yaml",
                r#"
server:
  port: 8080
database:
  url: "postgres://oxigate:changeme@localhost:5432/oxigate"
redis:
  url: "redis://localhost:6379"
log_level: "info"
"#,
            )?;
            jail.set_env("OXIGATE__SERVER__PORT", "9000");
            let path = std::path::Path::new("config.yaml");
            let cfg = load_config(path).map_err(|e| e.to_string())?;
            assert_eq!(cfg.server.port, 9000);
            Ok(())
        });
    }

    #[test]
    fn test_yaml_overrides_defaults() {
        let path = yaml_path(
            r#"
server:
  port: 8080
database:
  url: "postgres://oxigate:changeme@localhost:5432/oxigate"
redis:
  url: "redis://localhost:6379"
log_level: "debug"
"#,
        );
        let cfg = load_config(&path).expect("load");
        assert_eq!(cfg.log_level, "debug");
    }

    #[test]
    fn test_invalid_yaml_exits_error() {
        let path = yaml_path("server: [");
        let err = load_config(&path).unwrap_err();
        assert!(matches!(err, ConfigError::Load(_)));
        let s = err.to_string();
        assert!(s.to_lowercase().contains("yaml") || s.contains("load"));
    }

    #[test]
    fn test_secret_string_expose() {
        let secret = SecretString::from("my_secret");
        assert_eq!(secret.expose_secret(), "my_secret");
    }

    #[test]
    fn test_validate_missing_db_url() {
        let cfg = GatewayConfig::default();
        assert!(cfg.validate().is_err());
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)));
        let s = err.to_string();
        assert!(s.contains("database.url"));
        assert!(s.contains("redis.url"));
    }

    /// Ensures that when YAML omits database/redis URLs AND no env vars supply them,
    /// validate() rejects the config. Uses `figment::Jail` to isolate the test from
    /// any OXIGATE__DATABASE__URL / OXIGATE__REDIS__URL set in the CI or developer
    /// environment (which would otherwise make validation pass — correct behaviour,
    /// but it would make this assertion spuriously succeed or fail).
    #[test]
    fn test_load_and_validate_fails_when_urls_absent_from_yaml_and_env() {
        figment::Jail::expect_with(|jail| {
            jail.create_file(
                "config.yaml",
                r#"
server:
  port: 8080
  host: "0.0.0.0"
  drain_timeout_secs: 30
log_level: "info"
"#,
            )?;
            let path = std::path::Path::new("config.yaml");
            let result = load_and_validate_config(path);
            assert!(
                result.is_err(),
                "expected validation failure when URLs are absent from both YAML and env"
            );
            let err = result.unwrap_err().to_string();
            assert!(err.contains("database.url"), "got: {err}");
            assert!(err.contains("redis.url"), "got: {err}");
            Ok(())
        });
    }

    #[test]
    fn test_validate_missing_redis_url() {
        let mut cfg = GatewayConfig::default();
        cfg.database.url = SecretString::from("postgres://localhost/db");
        let err = cfg.validate().unwrap_err();
        let s = err.to_string();
        assert!(s.contains("redis.url"));
    }

    #[test]
    fn test_validate_invalid_log_level() {
        let mut cfg = GatewayConfig::default();
        cfg.database.url = SecretString::from("postgres://x/y");
        cfg.redis.url = SecretString::from("redis://x");
        cfg.log_level = "verbose".into();
        let err = cfg.validate().unwrap_err();
        let s = err.to_string();
        assert!(s.contains("verbose"));
        assert!(s.contains("trace"));
        assert!(s.contains("info"));
    }

    #[test]
    fn test_validate_port_zero() {
        let mut cfg = GatewayConfig::default();
        cfg.database.url = SecretString::from("postgres://x/y");
        cfg.redis.url = SecretString::from("redis://x");
        cfg.server.port = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("server.port"));
    }

    #[test]
    fn test_validate_auth_key_too_long() {
        let mut cfg = GatewayConfig::default();
        cfg.database.url = SecretString::from("postgres://x/y");
        cfg.redis.url = SecretString::from("redis://x");
        cfg.auth.key = Some(SecretString::from("a".repeat(257).as_str()));
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("auth.key"));
    }

    #[test]
    fn test_validate_auth_key_empty() {
        let mut cfg = GatewayConfig::default();
        cfg.database.url = SecretString::from("postgres://x/y");
        cfg.redis.url = SecretString::from("redis://x");
        cfg.auth.key = Some(SecretString::from(""));
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("auth.key must not be empty"));
    }

    #[test]
    fn test_validate_auth_key_at_max_length_ok() {
        let mut cfg = GatewayConfig::default();
        cfg.database.url = SecretString::from("postgres://x/y");
        cfg.redis.url = SecretString::from("redis://x");
        cfg.auth.key = Some(SecretString::from("a".repeat(256).as_str()));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_ok() {
        let cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_gemini_api_mode_missing_key_fails() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.providers.gemini = Some(GeminiConfig {
            mode: GeminiMode::Api,
            api_key: None,
            vertex_project: None,
            vertex_location: None,
            vertex_service_account_json: None,
            default_model: None,
            timeout_secs: None,
            api_base_url: None,
            vertex_base_url_override: None,
            supported_models: None,
            default_thinking_budget: None,
            embed_api_version: None,
        });
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("api_key"));
        assert!(err.to_string().contains("gemini"));
    }

    #[test]
    fn test_openai_config_missing_key_fails() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.providers.openai = Some(OpenAIConfig {
            api_key: None,
            default_model: None,
            api_base_url: None,
            timeout_secs: None,
            supported_models: None,
            organization: None,
            project: None,
        });
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("api_key"));
        assert!(err.to_string().contains("openai"));
    }

    #[test]
    fn test_openai_config_valid() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.providers.openai = Some(OpenAIConfig {
            api_key: Some(SecretString::new("sk-test")),
            default_model: Some("gpt-4o".into()),
            api_base_url: None,
            timeout_secs: Some(60),
            supported_models: None,
            organization: None,
            project: None,
        });
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_anthropic_config_missing_key_fails() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.providers.anthropic = Some(AnthropicConfig {
            api_key: None,
            api_base_url: None,
            anthropic_version: None,
            default_model: None,
            default_max_tokens: None,
            timeout_secs: None,
            supported_models: None,
            tool_call_buffer_cap_bytes: None,
        });
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("api_key"));
        assert!(err.to_string().contains("anthropic"));
    }

    #[test]
    fn test_anthropic_config_buffer_cap_zero_fails() {
        let cfg = AnthropicConfig {
            api_key: None,
            api_base_url: None,
            anthropic_version: None,
            default_model: None,
            default_max_tokens: None,
            timeout_secs: None,
            supported_models: None,
            tool_call_buffer_cap_bytes: Some(0),
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("must not be 0"), "unexpected message: {err}");
    }

    #[test]
    fn test_anthropic_config_buffer_cap_over_limit_fails() {
        use crate::providers::tool_limits::MAX_TOOL_CALL_BUFFER_CAP_BYTES;
        let cfg = AnthropicConfig {
            api_key: None,
            api_base_url: None,
            anthropic_version: None,
            default_model: None,
            default_max_tokens: None,
            timeout_secs: None,
            supported_models: None,
            tool_call_buffer_cap_bytes: Some(MAX_TOOL_CALL_BUFFER_CAP_BYTES + 1),
        };
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("exceeds the maximum"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn test_anthropic_config_buffer_cap_at_limit_ok() {
        use crate::providers::tool_limits::MAX_TOOL_CALL_BUFFER_CAP_BYTES;
        let cfg = AnthropicConfig {
            api_key: None,
            api_base_url: None,
            anthropic_version: None,
            default_model: None,
            default_max_tokens: None,
            timeout_secs: None,
            supported_models: None,
            tool_call_buffer_cap_bytes: Some(MAX_TOOL_CALL_BUFFER_CAP_BYTES),
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_anthropic_config_valid() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.providers.anthropic = Some(AnthropicConfig {
            api_key: Some(SecretString::new("sk-ant-test")),
            api_base_url: None,
            anthropic_version: None,
            default_model: Some("claude-sonnet-4-6".into()),
            default_max_tokens: Some(4096),
            timeout_secs: Some(120),
            supported_models: None,
            tool_call_buffer_cap_bytes: None,
        });
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_vertex_config_missing_project_fails_validation() {
        let minimal_sa = r#"{"client_email":"x@y.iam.gserviceaccount.com","private_key":"test-private-key-placeholder","token_uri":"https://oauth2.googleapis.com/token"}"#;
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.providers.gemini = Some(GeminiConfig {
            mode: GeminiMode::Vertex,
            api_key: None,
            vertex_project: None,
            vertex_location: Some("us-central1".into()),
            vertex_service_account_json: Some(SecretString::new(minimal_sa)),
            default_model: None,
            timeout_secs: None,
            api_base_url: None,
            vertex_base_url_override: None,
            supported_models: None,
            default_thinking_budget: None,
            embed_api_version: None,
        });
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("vertex_project"));
    }

    #[test]
    fn test_validate_pricing_override_negative_input() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.pricing.overrides.insert(
            "gpt-4".into(),
            PricingOverride {
                input_per_token: -0.01,
                output_per_token: 0.04,
                context_window: 128_000,
                cache_read_multiplier: None,
            },
        );
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)));
        let s = err.to_string();
        assert!(s.contains("pricing"));
        assert!(s.contains("input_per_token"));
    }

    #[test]
    fn test_validate_pricing_override_negative_output() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.pricing.overrides.insert(
            "gpt-4".into(),
            PricingOverride {
                input_per_token: 0.01,
                output_per_token: -0.04,
                context_window: 128_000,
                cache_read_multiplier: None,
            },
        );
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)));
        let s = err.to_string();
        assert!(s.contains("pricing"));
        assert!(s.contains("output_per_token"));
    }

    #[test]
    fn test_validate_pricing_override_invalid_cache_multiplier() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.pricing.overrides.insert(
            "gpt-4".into(),
            PricingOverride {
                input_per_token: 0.01,
                output_per_token: 0.04,
                context_window: 128_000,
                cache_read_multiplier: Some(11.0),
            },
        );
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)));
        let s = err.to_string();
        assert!(s.contains("pricing"));
        assert!(s.contains("cache_read_multiplier"));
    }

    #[test]
    fn test_validate_pricing_override_valid() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.pricing.overrides.insert(
            "ollama/llama3.2".into(),
            PricingOverride {
                input_per_token: 0.0,
                output_per_token: 0.0,
                context_window: 128_000,
                cache_read_multiplier: Some(0.5),
            },
        );
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_classify_reload_class_a() {
        let mut old = GatewayConfig::default();
        old.database.url = SecretString::from("postgres://x/y");
        old.redis.url = SecretString::from("redis://x");
        let mut new = old.clone();
        new.log_level = "debug".into();
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassA);
    }

    #[test]
    fn test_classify_reload_class_b() {
        let mut old = GatewayConfig::default();
        old.database.url = SecretString::from("postgres://x/y");
        old.redis.url = SecretString::from("redis://x");
        let mut new = old.clone();
        new.database.url = SecretString::from("postgres://other/db");
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassB);
    }

    #[test]
    fn test_classify_reload_class_c() {
        let mut old = GatewayConfig::default();
        old.database.url = SecretString::from("postgres://x/y");
        old.redis.url = SecretString::from("redis://x");
        let mut new = old.clone();
        new.server.port = 9999;
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassC);
    }

    #[test]
    fn test_classify_reload_class_b_pool_acquire_timeout() {
        let mut old = GatewayConfig::default();
        old.database.url = SecretString::from("postgres://x/y");
        old.redis.url = SecretString::from("redis://x");
        let mut new = old.clone();
        new.database.pool_acquire_timeout_secs = Some(60);
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassB);
    }

    #[test]
    fn test_classify_reload_class_b_redis_pool_timeout_secs() {
        let mut old = GatewayConfig::default();
        old.database.url = SecretString::from("postgres://x/y");
        old.redis.url = SecretString::from("redis://x");
        let mut new = old.clone();
        new.redis.pool_timeout_secs = Some(10);
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassB);
    }

    #[test]
    fn test_classify_reload_class_b_redis_pool_size() {
        let mut old = GatewayConfig::default();
        old.database.url = SecretString::from("postgres://x/y");
        old.redis.url = SecretString::from("redis://x");
        let mut new = old.clone();
        new.redis.pool_size = Some(32);
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassB);
    }

    #[test]
    fn test_classify_reload_class_a_auth_key_change() {
        let mut old = GatewayConfig::default();
        old.database.url = SecretString::from("postgres://x/y");
        old.redis.url = SecretString::from("redis://x");
        let mut new = old.clone();
        new.auth.key = Some(SecretString::new("new-token"));
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassA);
    }

    #[test]
    fn test_classify_reload_class_a_auth_key_removal() {
        let mut old = GatewayConfig::default();
        old.database.url = SecretString::from("postgres://x/y");
        old.redis.url = SecretString::from("redis://x");
        old.auth.key = Some(SecretString::new("existing-token"));
        let mut new = old.clone();
        new.auth.key = None;
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassA);
    }

    #[test]
    fn test_classify_reload_class_a_auth_key_swap() {
        let mut old = GatewayConfig::default();
        old.database.url = SecretString::from("postgres://x/y");
        old.redis.url = SecretString::from("redis://x");
        old.auth.key = Some(SecretString::new("token-a"));
        let mut new = old.clone();
        new.auth.key = Some(SecretString::new("token-b"));
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassA);
    }

    #[test]
    fn test_classify_reload_pricing_change() {
        let mut old = GatewayConfig::default();
        old.database.url = SecretString::from("postgres://x/y");
        old.redis.url = SecretString::from("redis://x");
        let mut new = old.clone();
        new.pricing.overrides.insert(
            "gpt-5".into(),
            PricingOverride {
                input_per_token: 0.01,
                output_per_token: 0.04,
                context_window: 200_000,
                cache_read_multiplier: None,
            },
        );
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassA);
    }

    #[test]
    fn test_validate_global_safety_cap_negative_fails() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.budget.global_safety_cap_usd = Some(-1.0);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("global_safety_cap_usd"),
            "expected error about global_safety_cap_usd, got: {}",
            err
        );
    }

    #[test]
    fn test_validate_global_safety_cap_nan_fails() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.budget.global_safety_cap_usd = Some(f64::NAN);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("global_safety_cap_usd"),
            "expected error about global_safety_cap_usd, got: {}",
            err
        );
    }

    #[test]
    fn test_validate_global_safety_cap_zero_fails() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.budget.global_safety_cap_usd = Some(0.0);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("global_safety_cap_usd"),
            "expected error about global_safety_cap_usd, got: {}",
            err
        );
    }

    #[test]
    fn test_validate_global_safety_cap_valid_passes() {
        let cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            budget: BudgetConfig {
                global_safety_cap_usd: Some(10.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    /// context_window is required in YAML overrides. Omission is rejected at
    /// config load (Figment/serde) before PricingDb::load. Error is ConfigError::Load.
    #[test]
    fn test_validation_context_window_required_in_override() {
        let path = yaml_path(
            r#"
server:
  port: 8080
  host: "0.0.0.0"
  drain_timeout_secs: 30
database:
  url: "postgres://oxigate:changeme@localhost:5432/oxigate"
redis:
  url: "redis://localhost:6379"
log_level: "info"
pricing:
  overrides:
    ollama/llama3.2:
      input_per_token: 0.0
      output_per_token: 0.0
      # context_window omitted — required, must fail
"#,
        );
        let err = load_config(&path).unwrap_err();
        assert!(
            matches!(err, ConfigError::Load(_)),
            "rejection at config layer: {:?}",
            err
        );
        let s = err.to_string();
        assert!(
            s.to_lowercase().contains("context_window") || s.contains("missing"),
            "error should mention context_window or missing field: {}",
            s
        );
    }

    #[test]
    fn test_validate_hard_cap_negative_fails() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.budget.hard_cap_usd = Some(-1.0);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("hard_cap_usd"),
            "expected error about hard_cap_usd, got: {}",
            err
        );
    }

    #[test]
    fn test_validate_hard_cap_zero_fails() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.budget.hard_cap_usd = Some(0.0);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("hard_cap_usd"),
            "expected error about hard_cap_usd, got: {}",
            err
        );
    }

    #[test]
    fn test_validate_hard_cap_nan_fails() {
        let mut cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.budget.hard_cap_usd = Some(f64::NAN);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("hard_cap_usd"),
            "expected error about hard_cap_usd, got: {}",
            err
        );
    }

    #[test]
    fn test_validate_hard_cap_positive_passes() {
        let cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            budget: BudgetConfig {
                hard_cap_usd: Some(10.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_hard_cap_none_passes() {
        let cfg = GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    fn minimal_gateway_for_budget_validate() -> GatewayConfig {
        GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_validate_budget_timezone_invalid() {
        let mut cfg = minimal_gateway_for_budget_validate();
        cfg.budget.timezone = "invalid/zone".into();
        let err = cfg.validate().unwrap_err();
        let s = err.to_string();
        assert!(s.contains("budget.timezone"), "{s}");
        assert!(s.contains("invalid/zone"), "{s}");
    }

    #[test]
    fn test_validate_budget_duration_invalid() {
        let mut cfg = minimal_gateway_for_budget_validate();
        cfg.budget.budget_duration = Some("bad_value".into());
        let err = cfg.validate().unwrap_err();
        let s = err.to_string();
        assert!(s.contains("budget.budget_duration"), "{s}");
        assert!(s.contains("bad_value"), "{s}");
    }

    #[test]
    fn test_validate_budget_reset_at_and_duration_mutually_exclusive() {
        let mut cfg = minimal_gateway_for_budget_validate();
        cfg.budget.budget_duration = Some("30d".into());
        cfg.budget.budget_reset_at = Some(chrono::Utc::now());
        let err = cfg.validate().unwrap_err();
        let s = err.to_string();
        assert!(
            s.contains("budget_reset_at") && s.contains("budget_duration"),
            "{s}"
        );
    }

    #[test]
    fn resolved_timezone_uses_lazy_cache() {
        let mut c = BudgetConfig::default();
        c.timezone = "Europe/Berlin".into();
        assert_eq!(c.resolved_timezone(), c.resolved_timezone());
    }

    // -------------------------------------------------------------------------
    //: Fallback + retry config validation tests
    // -------------------------------------------------------------------------

    /// Minimal valid config with an openai provider — used by validation tests.
    fn base_cfg_with_openai() -> GatewayConfig {
        GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            providers: ProvidersConfig {
                openai: Some(OpenAIConfig {
                    api_key: Some(SecretString::from("key")),
                    default_model: None,
                    api_base_url: None,
                    timeout_secs: None,
                    supported_models: None,
                    organization: None,
                    project: None,
                }),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_fallback_unknown_target_provider_rejected() {
        let mut cfg = base_cfg_with_openai();
        cfg.fallbacks = vec![FallbackRule {
            provider: Some("openai".into()),
            model: None,
            targets: vec![FallbackTarget::Provider("anthropic".into())], // not configured
            key: None,
            on: None,
        }];
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("unknown provider"),
            "expected unknown provider error, got: {err}"
        );
    }

    #[test]
    fn test_fallback_empty_targets_rejected() {
        let mut cfg = base_cfg_with_openai();
        cfg.fallbacks = vec![FallbackRule {
            provider: Some("openai".into()),
            model: None,
            targets: vec![],
            key: None,
            on: None,
        }];
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("targets` must not be empty"),
            "got: {err}"
        );
    }

    #[test]
    fn test_fallback_neither_provider_nor_model_rejected() {
        let mut cfg = base_cfg_with_openai();
        cfg.fallbacks = vec![FallbackRule {
            provider: None,
            model: None,
            targets: vec![FallbackTarget::Provider("openai".into())],
            key: None,
            on: None,
        }];
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("provider` or `model"),
            "got: {err}"
        );
    }

    #[test]
    fn test_fallback_invalid_model_pattern_rejected() {
        // Star not at the suffix (e.g. "cla*ude") is rejected.
        let mut cfg = base_cfg_with_openai();
        cfg.fallbacks = vec![FallbackRule {
            provider: None,
            model: Some("cla*ude".into()),
            targets: vec![FallbackTarget::Provider("openai".into())],
            key: None,
            on: None,
        }];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("suffix glob"), "got: {err}");
    }

    #[test]
    fn test_fallback_cycle_detected() {
        // A → B and B → A forms a cycle.
        let mut cfg = base_cfg_with_openai();
        cfg.providers.anthropic = Some(AnthropicConfig {
            api_key: Some(SecretString::from("key")),
            api_base_url: None,
            anthropic_version: None,
            default_model: None,
            default_max_tokens: None,
            timeout_secs: None,
            supported_models: None,
            tool_call_buffer_cap_bytes: None,
        });
        cfg.fallbacks = vec![
            FallbackRule {
                provider: Some("openai".into()),
                model: None,
                targets: vec![FallbackTarget::Provider("anthropic".into())],
                key: None,
                on: None,
            },
            FallbackRule {
                provider: Some("anthropic".into()),
                model: None,
                targets: vec![FallbackTarget::Provider("openai".into())],
                key: None,
                on: None,
            },
        ];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("cycle detected"), "got: {err}");
    }

    #[test]
    fn test_fallback_same_provider_not_a_cycle() {
        // anthropic → anthropic with different model is valid (model downgrade, not routing loop).
        let mut cfg = base_cfg_with_openai();
        cfg.providers.anthropic = Some(AnthropicConfig {
            api_key: Some(SecretString::from("key")),
            api_base_url: None,
            anthropic_version: None,
            default_model: None,
            default_max_tokens: None,
            timeout_secs: None,
            supported_models: None,
            tool_call_buffer_cap_bytes: None,
        });
        cfg.fallbacks = vec![FallbackRule {
            provider: Some("anthropic".into()),
            model: None,
            targets: vec![FallbackTarget::Explicit {
                provider: "anthropic".into(),
                model: Some("claude-haiku-4-5".into()),
            }],
            key: None,
            on: None,
        }];
        assert!(
            cfg.validate().is_ok(),
            "same-provider target must not be flagged as a cycle"
        );
    }

    #[test]
    fn test_provider_name_with_comma_rejected() {
        let mut cfg = base_cfg_with_openai();
        cfg.fallbacks = vec![FallbackRule {
            provider: Some("open,ai".into()), // comma — unsafe for headers
            model: None,
            targets: vec![FallbackTarget::Provider("openai".into())],
            key: None,
            on: None,
        }];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("alphanumeric"), "got: {err}");
    }

    #[test]
    fn test_retry_multiplier_below_one_rejected() {
        let mut cfg = base_cfg_with_openai();
        cfg.retry.multiplier = 0.5;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("multiplier"), "got: {err}");
    }

    #[test]
    fn test_retry_max_delay_less_than_base_rejected() {
        let mut cfg = base_cfg_with_openai();
        cfg.retry.base_delay_ms = 1000;
        cfg.retry.max_delay_ms = 500;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("max_delay_ms"), "got: {err}");
    }

    #[test]
    fn test_warn_dedup_period_secs_from_budget_duration() {
        let mut c = BudgetConfig::default();
        c.budget_duration = Some("1d".into());
        assert_eq!(c.warn_dedup_period_secs(), 86_400);
        c.budget_duration = Some("7d".into());
        assert_eq!(c.warn_dedup_period_secs(), 604_800);
    }

    #[test]
    fn test_validate_team_budget_negative_soft_fails() {
        let mut cfg = base_cfg_with_openai();
        cfg.budget.teams.insert(
            "eng".into(),
            BudgetCapEntry {
                soft_cap_usd: Some(-1.0),
                hard_cap_usd: None,
            },
        );
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("soft_cap_usd"), "got: {err}");
    }

    #[test]
    fn test_validate_team_budget_zero_hard_fails() {
        let mut cfg = base_cfg_with_openai();
        cfg.budget.teams.insert(
            "eng".into(),
            BudgetCapEntry {
                soft_cap_usd: None,
                hard_cap_usd: Some(0.0),
            },
        );
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("hard_cap_usd"), "got: {err}");
    }

    #[test]
    fn test_validate_team_budget_hard_below_soft_fails() {
        let mut cfg = base_cfg_with_openai();
        cfg.budget.teams.insert(
            "eng".into(),
            BudgetCapEntry {
                soft_cap_usd: Some(100.0),
                hard_cap_usd: Some(50.0),
            },
        );
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("hard_cap_usd"), "got: {err}");
    }

    #[test]
    fn test_validate_tag_budget_valid_passes() {
        let mut cfg = base_cfg_with_openai();
        cfg.budget.tag_budgets.insert(
            "project:chat-bot".into(),
            BudgetCapEntry {
                soft_cap_usd: Some(50.0),
                hard_cap_usd: Some(100.0),
            },
        );
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_tag_budget_nan_fails() {
        let mut cfg = base_cfg_with_openai();
        cfg.budget.tag_budgets.insert(
            "project:chat-bot".into(),
            BudgetCapEntry {
                soft_cap_usd: Some(f64::NAN),
                hard_cap_usd: None,
            },
        );
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("soft_cap_usd"), "got: {err}");
    }

    // --- N-1: tag key/value colon validation ---

    #[test]
    fn test_validate_tag_budget_extra_colon_in_key_fails() {
        let mut cfg = base_cfg_with_openai();
        cfg.budget.tag_budgets.insert(
            "project:chat:bot".into(), // two colons → ambiguous Redis key
            BudgetCapEntry {
                soft_cap_usd: Some(50.0),
                hard_cap_usd: None,
            },
        );
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("tag_budgets[project:chat:bot]"),
            "got: {err}"
        );
    }

    #[test]
    fn test_validate_tag_budget_no_colon_fails() {
        let mut cfg = base_cfg_with_openai();
        cfg.budget.tag_budgets.insert(
            "projectchatbot".into(), // no separator
            BudgetCapEntry {
                soft_cap_usd: Some(50.0),
                hard_cap_usd: None,
            },
        );
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("tag_budgets"), "got: {err}");
    }

    #[test]
    fn test_validate_team_name_with_colon_fails() {
        let mut cfg = base_cfg_with_openai();
        cfg.budget.teams.insert(
            "eng:team".into(),
            BudgetCapEntry {
                soft_cap_usd: Some(50.0),
                hard_cap_usd: None,
            },
        );
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("teams[eng:team]"), "got: {err}");
    }

    // --- N-3: double-counting detection ---

    #[test]
    fn test_validate_tag_team_double_counting_fails() {
        let mut cfg = base_cfg_with_openai();
        cfg.budget.teams.insert(
            "engineering".into(),
            BudgetCapEntry {
                soft_cap_usd: Some(100.0),
                hard_cap_usd: None,
            },
        );
        cfg.budget.tag_budgets.insert(
            "team:engineering".into(),
            BudgetCapEntry {
                soft_cap_usd: Some(80.0),
                hard_cap_usd: None,
            },
        );
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("double-count"), "got: {err}");
    }

    #[test]
    fn test_validate_tag_team_prefix_without_matching_teams_entry_passes() {
        // "team:engineering" in tag_budgets is fine if budget.teams has no "engineering" entry.
        let mut cfg = base_cfg_with_openai();
        cfg.budget.tag_budgets.insert(
            "team:engineering".into(),
            BudgetCapEntry {
                soft_cap_usd: Some(80.0),
                hard_cap_usd: None,
            },
        );
        assert!(
            cfg.validate().is_ok(),
            "no double-counting without a teams entry"
        );
    }

    #[test]
    fn test_validate_top_level_hard_below_soft_now_fails() {
        let mut cfg = base_cfg_with_openai();
        cfg.budget.soft_cap_usd = Some(100.0);
        cfg.budget.hard_cap_usd = Some(50.0);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("hard_cap_usd"),
            "expected hard<soft to be a hard error, got: {err}"
        );
    }

    /// `FallbackRule.on = Some([])` must be rejected by validation.
    #[test]
    fn test_fallback_rule_empty_on_list_is_rejected() {
        let mut cfg = base_cfg_with_openai();
        cfg.fallbacks = vec![FallbackRule {
            provider: Some("openai".into()),
            model: None,
            targets: vec![FallbackTarget::Provider("openai".into())],
            key: None,
            on: Some(vec![]),
        }];
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("on"),
            "error should mention 'on' field; got: {err}"
        );
    }

    /// `RetryConfig.on = Some([])` must be rejected by validation.
    #[test]
    fn test_retry_on_empty_list_is_rejected() {
        let mut cfg = base_cfg_with_openai();
        cfg.retry.on = Some(vec![]);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("on"),
            "error should mention 'on' field; got: {err}"
        );
    }

    /// `FallbackRule.on = Some([RateLimit])` with matching trigger must pass validation.
    #[test]
    fn test_fallback_rule_on_with_valid_trigger_passes_validation() {
        let mut cfg = base_cfg_with_openai();
        cfg.fallbacks = vec![FallbackRule {
            provider: Some("openai".into()),
            model: None,
            targets: vec![FallbackTarget::Provider("openai".into())],
            key: Some("rate-limit-only".into()),
            on: Some(vec![FallbackTrigger::RateLimit]),
        }];
        assert!(
            cfg.validate().is_ok(),
            "non-empty on-list should pass validation"
        );
    }

    // --- Azure config validation  ---

    fn base_azure() -> AzureConfig {
        AzureConfig {
            name: "azure-prod".into(),
            endpoint: "https://my-resource.openai.azure.com".into(),
            deployment_name: "gpt-4o".into(),
            api_version: "2024-10-21".into(),
            api_key: SecretString::new("sk-azure-test"),
            supported_models: None,
            timeout_secs: None,
        }
    }

    fn base_cfg() -> GatewayConfig {
        GatewayConfig {
            database: DatabaseConfig {
                url: SecretString::from("postgres://x/y"),
                ..Default::default()
            },
            redis: RedisConfig {
                url: SecretString::from("redis://x"),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn azure_valid_config_passes_validation() {
        let mut cfg = base_cfg();
        cfg.providers.azure.push(base_azure());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn azure_empty_name_rejected() {
        let mut cfg = base_cfg();
        let mut az = base_azure();
        az.name = "".into();
        cfg.providers.azure.push(az);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("azure") && err.contains("name"), "got: {err}");
    }

    #[test]
    fn azure_reserved_name_rejected() {
        let mut cfg = base_cfg();
        let mut az = base_azure();
        az.name = "azure".into();
        cfg.providers.azure.push(az);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("reserved"), "got: {err}");
    }

    #[test]
    fn azure_duplicate_name_rejected() {
        let mut cfg = base_cfg();
        cfg.providers.azure.push(base_azure());
        cfg.providers.azure.push(base_azure()); // same name
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("duplicate") || err.contains("azure-prod"),
            "got: {err}"
        );
    }

    #[test]
    fn azure_http_endpoint_rejected() {
        let mut cfg = base_cfg();
        let mut az = base_azure();
        az.endpoint = "http://my-resource.openai.azure.com".into();
        cfg.providers.azure.push(az);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("https"), "got: {err}");
    }

    #[test]
    fn azure_empty_deployment_name_rejected() {
        let mut cfg = base_cfg();
        let mut az = base_azure();
        az.deployment_name = "".into();
        cfg.providers.azure.push(az);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("deployment_name"), "got: {err}");
    }

    #[test]
    fn azure_deployment_name_with_slash_rejected() {
        let mut cfg = base_cfg();
        let mut az = base_azure();
        az.deployment_name = "gpt-4o/turbo".into();
        cfg.providers.azure.push(az);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("deployment_name") && err.contains("unsafe"),
            "got: {err}"
        );
    }

    #[test]
    fn azure_deployment_name_with_whitespace_rejected() {
        let mut cfg = base_cfg();
        let mut az = base_azure();
        az.deployment_name = "gpt 4o".into();
        cfg.providers.azure.push(az);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("deployment_name"), "got: {err}");
    }

    #[test]
    fn azure_empty_api_version_rejected() {
        let mut cfg = base_cfg();
        let mut az = base_azure();
        az.api_version = "".into();
        cfg.providers.azure.push(az);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("api_version"), "got: {err}");
    }

    #[test]
    fn azure_api_version_with_unsafe_char_rejected() {
        let mut cfg = base_cfg();
        let mut az = base_azure();
        az.api_version = "2024-10-21?foo=bar".into();
        cfg.providers.azure.push(az);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("api_version") && err.contains("unsafe"),
            "got: {err}"
        );
    }

    #[test]
    fn azure_supported_models_empty_list_rejected() {
        let mut cfg = base_cfg();
        let mut az = base_azure();
        az.supported_models = Some(vec![]);
        cfg.providers.azure.push(az);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("supported_models"), "got: {err}");
    }

    // --- Azure classify_reload  ---

    fn azure_cfg_pair() -> (GatewayConfig, GatewayConfig) {
        let mut cfg = base_cfg();
        cfg.providers.azure.push(base_azure());
        (cfg.clone(), cfg)
    }

    #[test]
    fn azure_no_change_is_class_a() {
        let (old, new) = azure_cfg_pair();
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassA);
    }

    #[test]
    fn azure_deployment_name_change_is_class_b() {
        let (old, mut new) = azure_cfg_pair();
        new.providers.azure[0].deployment_name = "gpt-35-turbo".into();
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassB);
    }

    #[test]
    fn azure_api_key_change_is_class_b() {
        let (old, mut new) = azure_cfg_pair();
        new.providers.azure[0].api_key = SecretString::new("new-key");
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassB);
    }

    #[test]
    fn azure_added_entry_is_class_b() {
        let (old, mut new) = azure_cfg_pair();
        let mut extra = base_azure();
        extra.name = "azure-secondary".into();
        new.providers.azure.push(extra);
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassB);
    }

    #[test]
    fn azure_removed_entry_is_class_b() {
        let (old, mut new) = azure_cfg_pair();
        new.providers.azure.clear();
        assert_eq!(classify_reload(&old, &new), HotReloadClass::ClassB);
    }

    #[test]
    fn azure_reorder_is_not_class_b() {
        let mut old = base_cfg();
        let mut az1 = base_azure();
        az1.name = "azure-a".into();
        let mut az2 = base_azure();
        az2.name = "azure-b".into();
        az2.deployment_name = "gpt-35-turbo".into();
        old.providers.azure.push(az1.clone());
        old.providers.azure.push(az2.clone());

        let mut new = old.clone();
        new.providers.azure = vec![az2, az1]; // reversed order, same content
        assert_eq!(
            classify_reload(&old, &new),
            HotReloadClass::ClassA,
            "reordering azure entries without field changes must not trigger rebuild"
        );
    }
}
