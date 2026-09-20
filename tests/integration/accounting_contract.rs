// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Gateway accounting contract matrix.
//!
//! Follows one provider payload per contract through the real adapter and the real gateway to
//! every surface that reports its money: the buffered cost headers or the terminal
//! `oxigate.usage` event, the persisted `spend_records` row, and the Redis budget counter. Each
//! row prices against a synthetic catalogue whose rates match no bundled entry, and is held to an
//! integer oracle derived by hand from the provider's documented accounting contract and those
//! rates — never read back from the code under test.
//!
//! Beside the contract rows, the matrix carries one row per degraded cost status and a pair of
//! evidence twins that differ only in whether the persisted evidence document is complete. A
//! separate test, [`hard_cap_trips_on_the_cache_inclusive_cost`], follows repeated requests into
//! a hard budget cap.
//!
//! Each row is held to seven assertions, named by letter where the code below refers to them:
//!
//! - **(a)** the persisted `spend_records.cost_nano_usd` equals the oracle;
//! - **(b)** the Redis budget counter moved by exactly the oracle;
//! - **(c)** the client-facing cost — the buffered cost header, or the member of the streamed
//!   `oxigate.usage` event — equals the oracle's display string;
//! - **(d)** the client-facing status and the persisted `cost_status` equal the expected status;
//! - **(e)** the persisted prompt, completion, cache-read and thinking token columns equal their
//!   hand-derived values;
//! - **(f)** `usage_evidence` is SQL NULL exactly when no positive cache-write quantity was
//!   accounted, and otherwise its `accounted_tokens` equals both the hand-derived quantity and the
//!   published `cache_creation_input_tokens`;
//! - **(g)** on buffered rows the body's `usage.prompt_tokens` and `usage.completion_tokens` equal
//!   the input and output token headers.
//!
//! Rows are data; [`run_row`] is the only place that asserts on them. They run in sequence on one
//! Postgres/Redis pair, because `spend_records` carries no request identifier: a row is told
//! apart from the one before it by clearing the table and the counter first.
//!
//! A contract's buffered and streamed rows are held to one shared [`Accounting`] value, so the
//! pair is the buffered-versus-streamed parity check: both must produce the same cost, status and
//! persisted columns, or one of them fails against the shared oracle.
//!
//! Each fixture is one of two kinds, and the doc comment of the function building it says which:
//!
//! - **Contract fixtures** claim something about the usage object only: every usage member they
//!   carry, where it sits, and how the counts relate to one another is something the provider
//!   documents it can report. The model ID, deployment name, count magnitudes and everything
//!   outside the usage object (IDs, timestamps, message text) are synthetic and claim nothing.
//! - **Robustness fixtures** use a first-party shape with values the provider does not document,
//!   to exercise how the gateway handles malformed-but-deserializable upstream data.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::http::StatusCode;
use bytes::Bytes;
use futures::StreamExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use oxigate::api::CHAT_COMPLETIONS_PATH;
use oxigate::config::{
    AnthropicConfig, AzureConfig, BedrockConfig, BudgetConfig, GeminiConfig, GeminiMode,
    OpenAICompatConfig, OpenAIConfig, PricingConfig, SecretString,
};
use oxigate::domain::chat::{ChatRequest, ChatResponse, Usage};
use oxigate::domain::embedding::{EmbeddingRequest, EmbeddingResponse};
use oxigate::domain::ports::{
    ChatCompletionStream, HealthStatus, NanoUsd, ProviderAdapter, ProviderAdapterExt,
    ProviderError, ProviderMetadata,
};
use oxigate::domain::pricing::PricingDb;
use oxigate::domain::usage_accounting::{
    CACHE_WRITE_EVIDENCE_MAX_BYTES, CostStatus, MAX_RAW_KEY_BYTES, MAX_RETAINED_EVIDENCE_ENTRIES,
    UsageEvidence,
};
use oxigate::providers::bedrock::eventstream::build_frame;
use oxigate::providers::{
    AnthropicAdapter, AzureAdapter, BedrockAdapter, CompatHttpClient, GeminiAdapter,
    OpenAICompatAdapter, OpenAiAdapter,
};
use oxigate::utils::CostHeader;

// Matches RequestIdentity::default() key path used by auth-disabled test flows, as in
// budget_e2e.rs and bedrock_cache_e2e.rs.
const DEFAULT_SPEND_KEY: &str = "oxigate:org:default:spend:default";

// ---------------------------------------------------------------------------------------------
// Synthetic catalogue
// ---------------------------------------------------------------------------------------------

/// The model every primary row prices against.
///
/// Starts with `anthropic.` because the Bedrock lane refuses any other prefix before dispatch;
/// one ID then serves every lane.
const PRIMARY_MODEL: &str = "anthropic.accounting-contract-fixture";

/// The model the `rate-fallback` row and both evidence twins price against: one tier that
/// configures the `5m` cache-write class and no other.
const FIVE_M_ONLY_MODEL: &str = "anthropic.accounting-contract-fixture-5m-only";

/// A model deliberately absent from the synthetic catalogue, for the `cost-unavailable` row.
const UNCATALOGUED_MODEL: &str = "anthropic.accounting-contract-uncatalogued";

/// The model the inference-geo rows price against.
///
/// Its catalogue entry is the only one here declaring `"provider": "anthropic"`, because the
/// geographic surcharge is gated on the provider that publishes it. Priced against
/// [`PRIMARY_MODEL`]'s `"synthetic"` entry the surcharge would never apply, and the rows would
/// pass green while proving nothing.
const GEO_MODEL: &str = "anthropic.accounting-contract-fixture-geo";

/// Where [`PRIMARY_MODEL`]'s upper tier starts.
///
/// Every primary fixture keeps its plain input below this and reaches it only through its cache
/// tokens, so every oracle is priced at the upper tier. A tier comparator that left a cache
/// bucket out of the context size would price the row at tier 0 and miss the oracle.
const UPPER_TIER_THRESHOLD: u64 = 30_000;

/// The synthetic catalogue, in the bundled asset's schema.
///
/// [`PRIMARY_MODEL`] — every rate is chosen so each per-token price is a whole number of
/// nano-USD, and every fixture count is a multiple of 1,000, so each oracle is a multiple of
/// 1,000 nano-USD and the six-decimal cost header carries it losslessly. Thinking has no rate of
/// its own and bills at the output rate, which is a documented derivation, not a fallback.
///
/// | Rate | Tier 0 (from 0) | Tier 1 (from 30,000) | Tier 1, nano-USD per token |
/// |---|---|---|---|
/// | input | 3.7e-06 | 5.3e-06 | 5,300 |
/// | output (and thinking) | 1.9e-05 | 2.3e-05 | 23,000 |
/// | cache read | × 0.15 | × 0.15 | 795 |
/// | cache write `5m` | × 1.35 | × 1.35 | 7,155 |
/// | cache write `1h` | × 2.2 | × 2.2 | 11,660 |
/// | cache write `30m` | × 1.45 | × 1.45 | 7,685 |
///
/// [`FIVE_M_ONLY_MODEL`] — one tier, from 0. Its only configured cache-write class is `5m`, so
/// its fallback multiplier is `max(1.3, 1.0)` = 1.3: any other class, and any unknown one, is
/// priced at the `5m` rate and reported `rate-fallback`.
///
/// | Rate | Tier 0 (from 0) | nano-USD per token |
/// |---|---|---|
/// | input | 4.1e-06 | 4,100 |
/// | output | 1.7e-05 | 17,000 |
/// | cache read | × 0.2 | 820 |
/// | cache write `5m`, and the fallback | × 1.3 | 5,330 |
///
/// [`GEO_MODEL`] — one tier, from 0, and the only entry declaring `"provider": "anthropic"`. No
/// cache or thinking rates: the inference-geo rows are about a surcharge reaching the spend row
/// and the budget counter, and a plain input/output oracle makes the 1.1x unmistakable rather
/// than something to be picked out of a stack of multipliers.
///
/// | Rate | Tier 0 (from 0) | nano-USD per token |
/// |---|---|---|
/// | input | 6.0e-06 | 6,000 |
/// | output | 3.0e-05 | 30,000 |
///
/// [`UNCATALOGUED_MODEL`] is absent on purpose.
fn synthetic_catalogue() -> String {
    let cache_writes = serde_json::json!({"5m": 1.35, "1h": 2.2, "30m": 1.45});
    serde_json::json!({
        "schema_version": 1,
        "models": {
            FIVE_M_ONLY_MODEL: {
                "provider": "synthetic",
                "context_window": 1_000_000,
                "tiers": [
                    {
                        "threshold": 0,
                        "input_per_token": 4.1e-06,
                        "output_per_token": 1.7e-05,
                        "cache_read_multiplier": 0.2,
                        "cache_write_multipliers": {"5m": 1.3}
                    }
                ]
            },
            GEO_MODEL: {
                "provider": "anthropic",
                "context_window": 1_000_000,
                "tiers": [
                    {
                        "threshold": 0,
                        "input_per_token": 6.0e-06,
                        "output_per_token": 3.0e-05
                    }
                ]
            },
            PRIMARY_MODEL: {
                "provider": "synthetic",
                "context_window": 1_000_000,
                "tiers": [
                    {
                        "threshold": 0,
                        "input_per_token": 3.7e-06,
                        "output_per_token": 1.9e-05,
                        "cache_read_multiplier": 0.15,
                        "cache_write_multipliers": cache_writes
                    },
                    {
                        "threshold": UPPER_TIER_THRESHOLD,
                        "input_per_token": 5.3e-06,
                        "output_per_token": 2.3e-05,
                        "cache_read_multiplier": 0.15,
                        "cache_write_multipliers": cache_writes
                    }
                ]
            }
        }
    })
    .to_string()
}

/// One holder, shared by the adapter and the gateway, so every lane prices against the synthetic
/// catalogue whether it pins its own pricing generation or leaves pricing to the gateway.
fn synthetic_pricing_holder() -> Arc<std::sync::RwLock<PricingDb>> {
    let db = PricingDb::load(synthetic_catalogue().as_bytes(), &PricingConfig::default())
        .expect("synthetic catalogue must load");
    Arc::new(std::sync::RwLock::new(db))
}

// ---------------------------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------------------------

/// The provider contract a row dispatches through.
#[derive(Clone, Copy, Debug)]
enum Lane {
    OpenAi,
    Azure,
    /// A generic OpenAI-compatible backend. `stream_options_support` picks the streaming path:
    /// `true` re-serializes the request to inject `stream_options`, `false` forwards the client's
    /// bytes verbatim. Buffered requests always take the verbatim path.
    Compat {
        stream_options_support: bool,
    },
    /// The Anthropic Messages API.
    Anthropic,
    /// Gemini through AI Studio. Vertex shares the projection and the accounting declaration, so
    /// it has no row of its own.
    Gemini,
    /// Bedrock Converse and ConverseStream.
    Bedrock,
}

/// The deployment the Azure rows address. Synthetic, outside every fixture's claim.
const AZURE_DEPLOYMENT: &str = "accounting-contract";

impl Lane {
    /// The upstream path the adapter calls. Gemini and Bedrock name the model and choose between
    /// buffered and streamed generation in the path; the others do both in the request body.
    fn upstream_path(self, model: &str, delivery: Delivery) -> String {
        let streamed = delivery == Delivery::Streamed;
        match self {
            Lane::OpenAi | Lane::Compat { .. } => CHAT_COMPLETIONS_PATH.to_string(),
            Lane::Azure => format!("/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"),
            Lane::Anthropic => "/v1/messages".to_string(),
            Lane::Gemini if streamed => format!("/v1beta/models/{model}:streamGenerateContent"),
            Lane::Gemini => format!("/v1beta/models/{model}:generateContent"),
            Lane::Bedrock if streamed => format!("/model/{model}/converse-stream"),
            Lane::Bedrock => format!("/model/{model}/converse"),
        }
    }

    /// The media type the upstream streams its response in.
    fn streamed_content_type(self) -> &'static str {
        match self {
            Lane::Bedrock => "application/vnd.amazon.eventstream",
            Lane::OpenAi | Lane::Azure | Lane::Compat { .. } | Lane::Anthropic | Lane::Gemini => {
                "text/event-stream"
            }
        }
    }

    /// Builds the lane's real adapter against the mock upstream and the shared pricing holder.
    async fn adapter(
        self,
        upstream: &str,
        pricing: Arc<std::sync::RwLock<PricingDb>>,
    ) -> Arc<dyn ProviderAdapterExt> {
        let upstream = upstream.trim_end_matches('/').to_string();
        let models = Some(
            [PRIMARY_MODEL, FIVE_M_ONLY_MODEL, UNCATALOGUED_MODEL]
                .map(str::to_string)
                .to_vec(),
        );
        match self {
            Lane::OpenAi => Arc::new(
                OpenAiAdapter::new(
                    OpenAIConfig {
                        api_key: Some(SecretString::new("sk-accounting-contract")),
                        default_model: None,
                        api_base_url: Some(upstream),
                        timeout_secs: Some(10),
                        supported_models: models,
                        organization: None,
                        project: None,
                    },
                    pricing,
                )
                .await
                .expect("openai adapter must build"),
            ),
            Lane::Azure => Arc::new(
                AzureAdapter::new(
                    AzureConfig {
                        name: "azure-accounting-contract".to_string(),
                        endpoint: upstream,
                        deployment_name: AZURE_DEPLOYMENT.to_string(),
                        api_version: "2025-02-01-preview".to_string(),
                        api_key: SecretString::new("azure-accounting-contract"),
                        supported_models: models,
                        timeout_secs: Some(10),
                    },
                    Arc::new(CompatHttpClient::new().expect("http client must build")),
                    pricing,
                )
                .await
                .expect("azure adapter must build"),
            ),
            Lane::Compat {
                stream_options_support,
            } => Arc::new(
                OpenAICompatAdapter::new(
                    OpenAICompatConfig {
                        name: "compat-accounting-contract".to_string(),
                        base_url: upstream,
                        api_key: None,
                        supported_models: models,
                        stream_options_support,
                        supports_tools: false,
                        timeout_secs: Some(10),
                    },
                    Arc::new(CompatHttpClient::new().expect("http client must build")),
                )
                .await
                .expect("compat adapter must build"),
            ),
            Lane::Anthropic => Arc::new(
                AnthropicAdapter::new(
                    AnthropicConfig {
                        api_key: Some(SecretString::new("sk-ant-accounting-contract")),
                        api_base_url: Some(upstream),
                        anthropic_version: None,
                        default_model: None,
                        default_max_tokens: None,
                        timeout_secs: Some(10),
                        supported_models: models,
                        tool_call_buffer_cap_bytes: None,
                    },
                    pricing,
                )
                .await
                .expect("anthropic adapter must build"),
            ),
            Lane::Gemini => Arc::new(
                GeminiAdapter::new(GeminiConfig {
                    mode: GeminiMode::Api,
                    api_key: Some(SecretString::new("gemini-accounting-contract")),
                    vertex_project: None,
                    vertex_location: None,
                    vertex_service_account_json: None,
                    default_model: None,
                    timeout_secs: Some(10),
                    api_base_url: Some(upstream),
                    vertex_base_url_override: None,
                    supported_models: models,
                    default_thinking_budget: None,
                    embed_api_version: None,
                })
                .await
                .expect("gemini adapter must build"),
            ),
            Lane::Bedrock => Arc::new(
                BedrockAdapter::new(
                    BedrockConfig {
                        region: "us-east-1".to_string(),
                        access_key_id: Some(SecretString::new("AKIDACCOUNTINGCONTRACT")),
                        secret_access_key: Some(SecretString::new("accounting-contract-secret")),
                        session_token: None,
                        endpoint_url: Some(upstream),
                        default_model: None,
                        timeout_secs: Some(10),
                        supported_models: models,
                    },
                    pricing,
                )
                .await
                .expect("bedrock adapter must build"),
            ),
        }
    }

    /// What the request this lane sends upstream must show about the dispatch path it took.
    ///
    /// The compat rows differ only in the path the adapter takes, so without this the matrix
    /// would stay green if the verbatim path were never taken or `stream_options` stopped being
    /// injected. The OpenAI and Azure streamed rows are held to the injection as well: their
    /// fixture's final usage chunk is what the provider sends only when the request asks for it.
    fn upstream_request(self, delivery: Delivery) -> UpstreamRequest {
        match (self, delivery) {
            (Lane::Compat { .. }, Delivery::Buffered) => UpstreamRequest::Verbatim,
            (
                Lane::Compat {
                    stream_options_support: false,
                },
                Delivery::Streamed,
            ) => UpstreamRequest::Verbatim,
            (
                Lane::Compat {
                    stream_options_support: true,
                }
                | Lane::OpenAi
                | Lane::Azure,
                Delivery::Streamed,
            ) => UpstreamRequest::RequestsStreamUsage,
            (Lane::OpenAi | Lane::Azure, Delivery::Buffered)
            | (Lane::Anthropic | Lane::Gemini | Lane::Bedrock, _) => UpstreamRequest::Unconstrained,
        }
    }
}

/// What the upstream request must show about the dispatch path that produced it.
#[derive(Clone, Copy, Debug)]
enum UpstreamRequest {
    /// The client's request bytes, forwarded unmodified: the raw-forward path.
    Verbatim,
    /// A re-serialized request carrying `stream_options.include_usage: true`, which the client did
    /// not send: the injection path.
    RequestsStreamUsage,
    /// Re-serialized or translated, with nothing about it that this matrix depends on.
    Unconstrained,
}

/// Wraps a lane's adapter and records the last normalized `Usage` its stream reports — the value
/// the gateway finalizes a streamed request from.
///
/// Streamed chunks reach the client as the bytes the adapter yielded — the gateway forwards them
/// and never re-serializes the normalized `Usage` onto the wire — so on every lane whose chunk
/// bytes are the provider's own, that `Usage` is observable only here.
/// Every `ProviderAdapter` method is delegated and the `ProviderAdapterExt` dispatch defaults are
/// taken exactly as every leaf adapter takes them, so the wrapped adapter's dispatch — raw-forward
/// paths included — is unchanged.
///
/// That delegation is hand-written, so it is only exhaustive as long as it is kept so: a method
/// added to `ProviderAdapter` with a default body compiles here without a forwarder and answers
/// from the trait default instead of the wrapped adapter. Add the forwarder with the method.
struct StreamUsageTap {
    inner: Arc<dyn ProviderAdapterExt>,
    last_usage: Arc<Mutex<Option<Usage>>>,
}

impl StreamUsageTap {
    fn new(inner: Arc<dyn ProviderAdapterExt>) -> Self {
        Self {
            inner,
            last_usage: Arc::new(Mutex::new(None)),
        }
    }

    /// The last usage any chunk reported — the rule the gateway's own stream finalization uses.
    fn last_usage(&self) -> Option<Usage> {
        self.last_usage.lock().expect("usage tap lock").clone()
    }

    fn tap(&self, stream: ChatCompletionStream) -> ChatCompletionStream {
        let slot = Arc::clone(&self.last_usage);
        Box::pin(stream.inspect(move |item| {
            if let Ok(chunk) = item
                && let Some(usage) = &chunk.usage
            {
                *slot.lock().expect("usage tap lock") = Some(usage.clone());
            }
        }))
    }
}

#[async_trait]
impl ProviderAdapter for StreamUsageTap {
    async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        self.inner.chat_completion(req).await
    }

    async fn chat_completion_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<ChatCompletionStream, ProviderError> {
        self.inner
            .chat_completion_stream(req)
            .await
            .map(|s| self.tap(s))
    }

    async fn embeddings(&self, req: &EmbeddingRequest) -> Result<EmbeddingResponse, ProviderError> {
        self.inner.embeddings(req).await
    }

    fn metadata(&self) -> &ProviderMetadata {
        self.inner.metadata()
    }

    async fn health_check(&self) -> HealthStatus {
        self.inner.health_check().await
    }

    fn as_providers_slice(&self) -> Option<&[Arc<dyn ProviderAdapter>]> {
        self.inner.as_providers_slice()
    }

    async fn try_forward_raw(
        &self,
        req: &ChatRequest,
        raw_body: &Bytes,
    ) -> Option<Result<ChatResponse, ProviderError>> {
        self.inner.try_forward_raw(req, raw_body).await
    }

    async fn try_forward_raw_stream(
        &self,
        req: &ChatRequest,
        raw_body: &Bytes,
    ) -> Option<Result<ChatCompletionStream, ProviderError>> {
        self.inner
            .try_forward_raw_stream(req, raw_body)
            .await
            .map(|r| r.map(|s| self.tap(s)))
    }
}

impl ProviderAdapterExt for StreamUsageTap {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Delivery {
    Buffered,
    Streamed,
}

/// The persisted cache-write evidence a row must produce.
#[derive(Clone, Copy, Debug)]
struct Evidence {
    /// The accounted cache-write quantity: billed, persisted and counted against budgets.
    accounted_tokens: u64,
    /// Whether the evidence bound dropped observations or key bytes.
    incomplete: bool,
}

/// What the gateway must account for one upstream usage object, on every surface.
#[derive(Clone, Copy, Debug)]
struct Accounting {
    /// The integer oracle, in nano-USD. A multiple of 1,000, so the cost header carries it
    /// losslessly.
    cost_nano_usd: u64,
    status: CostStatus,
    /// `spend_records.prompt_tokens`: the finalized billable input bucket, after this contract's
    /// own carve-outs — not the provider-reported prompt total.
    prompt_tokens: i64,
    /// `spend_records.completion_tokens`: the provider-reported completion total.
    completion_tokens: i64,
    cache_read_tokens: i64,
    thinking_tokens: i64,
    /// `None` exactly when no positive cache-write quantity was accounted, in which case
    /// `usage_evidence` must be SQL NULL and no `cache_creation_input_tokens` may be published.
    cache_write: Option<Evidence>,
}

/// One row of the matrix.
struct Row {
    name: &'static str,
    lane: Lane,
    delivery: Delivery,
    /// The model the client requests, which the upstream response names back where its wire
    /// format carries one — the model the gateway prices against.
    model: &'static str,
    /// The upstream response body, exactly as the provider sends it.
    upstream: Vec<u8>,
    /// Shared by the contract's buffered and streamed rows — see the module doc.
    accounting: Accounting,
    /// The provider's raw `prompt_tokens_details.cache_write_tokens`, which the client-visible
    /// usage must echo whether or not the lane accounts it. `None` on a lane whose wire format
    /// has no such member, where the client-visible usage must not invent one.
    raw_cache_write_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------------------------
// OpenAI-shaped fixtures: OpenAI, Azure OpenAI, generic compat
// ---------------------------------------------------------------------------------------------

/// The usage object the OpenAI, Azure and compat rows all receive, byte for byte.
///
/// A 40,000-token prompt of which 12,000 were read from cache and 6,000 written to it, and a
/// 3,000-token completion of which 1,000 were reasoning. Plain input is 22,000 under the OpenAI
/// contract and 28,000 under compat's, both below [`UPPER_TIER_THRESHOLD`]; the reconstructed
/// prompt of 40,000 is above it, so the upper tier is reached only through the cache tokens.
const OPENAI_SHAPE_USAGE: &str = r#"{"prompt_tokens":40000,"completion_tokens":3000,"total_tokens":43000,"prompt_tokens_details":{"cached_tokens":12000,"cache_write_tokens":6000},"completion_tokens_details":{"reasoning_tokens":1000}}"#;

/// The cache write [`OPENAI_SHAPE_USAGE`] reports, which every OpenAI-shaped row must echo.
const OPENAI_SHAPE_CACHE_WRITE: u64 = 6_000;

/// A buffered chat-completions response from `model`, carrying [`OPENAI_SHAPE_USAGE`].
fn openai_shape_buffered_body(model: &str) -> Vec<u8> {
    format!(
        r#"{{"id":"chatcmpl-accounting-contract","object":"chat.completion","created":1757592000,"model":"{model}","choices":[{{"index":0,"message":{{"role":"assistant","content":"Summary."}},"finish_reason":"stop"}}],"usage":{OPENAI_SHAPE_USAGE}}}"#
    )
    .into_bytes()
}

/// A streamed chat-completions response carrying [`OPENAI_SHAPE_USAGE`] on its final chunk.
///
/// With `stream_options.include_usage` set, the usage arrives on an additional last chunk whose
/// `choices` is empty, and every earlier chunk carries `"usage": null`.
fn openai_shape_streamed_body() -> Vec<u8> {
    let chunk = |choices: &str, usage: &str| {
        format!(
            r#"data: {{"id":"chatcmpl-accounting-contract","object":"chat.completion.chunk","created":1757592000,"model":"{PRIMARY_MODEL}","choices":{choices},"usage":{usage}}}"#
        )
    };
    format!(
        "{}\n\n{}\n\n{}\n\ndata: [DONE]\n\n",
        chunk(
            r#"[{"index":0,"delta":{"role":"assistant","content":"Summary."},"finish_reason":null}]"#,
            "null"
        ),
        chunk(
            r#"[{"index":0,"delta":{},"finish_reason":"stop"}]"#,
            "null"
        ),
        chunk("[]", OPENAI_SHAPE_USAGE),
    )
    .into_bytes()
}

/// The oracle for [`OPENAI_SHAPE_USAGE`] under the OpenAI contract, shared by the OpenAI and
/// Azure rows.
///
/// Cache is inclusive — `prompt_tokens` contains both cache buckets, which are carved out and
/// charged once at their own multipliers. Reasoning is contained in `completion_tokens` — the
/// standard output charge excludes it and it is charged once at the thinking rate. The write is
/// accounted as class `30m`, the only TTL the contract supports. Context size is 22,000 +
/// 12,000 + 6,000 = 40,000, so tier 1 prices.
///
/// | Component | Quantity × rate | nano-USD |
/// |---|---|---|
/// | plain input | (40,000 − 12,000 − 6,000) = 22,000 × 5,300 | 116,600,000 |
/// | cache read | 12,000 × 795 | 9,540,000 |
/// | cache write `30m` | 6,000 × 7,685 | 46,110,000 |
/// | standard output | (3,000 − 1,000) = 2,000 × 23,000 | 46,000,000 |
/// | thinking | 1,000 × 23,000 | 23,000,000 |
/// | **total** | | **241,250,000** |
const OPENAI_CONTRACT_ACCOUNTING: Accounting = Accounting {
    cost_nano_usd: 241_250_000,
    status: CostStatus::Exact,
    prompt_tokens: 22_000,
    completion_tokens: 3_000,
    cache_read_tokens: 12_000,
    thinking_tokens: 1_000,
    cache_write: Some(Evidence {
        accounted_tokens: 6_000,
        incomplete: false,
    }),
};

/// The oracle for [`OPENAI_SHAPE_USAGE`] under the generic compat contract.
///
/// Compat speaks the OpenAI wire format but inherits none of its semantics, and prices the same
/// bytes differently in two places. Cache reads are carved out of `prompt_tokens`, but the cache
/// write is not accounted at all: it stays inside the plain input bucket and is charged at the
/// plain input rate, nothing is published on `cache_creation_input_tokens`, and no evidence is
/// persisted. Reasoning is charged beside the whole completion total. Context size is 28,000 +
/// 12,000 = 40,000, so tier 1 prices.
///
/// | Component | Quantity × rate | nano-USD |
/// |---|---|---|
/// | plain input, write included | (40,000 − 12,000) = 28,000 × 5,300 | 148,400,000 |
/// | cache read | 12,000 × 795 | 9,540,000 |
/// | output | 3,000 × 23,000 | 69,000,000 |
/// | reasoning, additive | 1,000 × 23,000 | 23,000,000 |
/// | **total** | | **249,940,000** |
const COMPAT_ACCOUNTING: Accounting = Accounting {
    cost_nano_usd: 249_940_000,
    status: CostStatus::Exact,
    prompt_tokens: 28_000,
    completion_tokens: 3_000,
    cache_read_tokens: 12_000,
    thinking_tokens: 1_000,
    cache_write: None,
};

/// OpenAI, buffered and streamed. **Contract fixture.**
///
/// Usage members and their relationships, per OpenAI: `prompt_tokens` contains
/// `prompt_tokens_details.cached_tokens`, and `prompt_tokens_details.cache_write_tokens` is the
/// number of prompt tokens written to cache, whose only TTL is `30m`
/// (`developers.openai.com/api/docs/guides/prompt-caching`);
/// `completion_tokens_details.reasoning_tokens` is a breakdown of `completion_tokens`, billed as
/// output (`developers.openai.com/api/docs/guides/reasoning`); with
/// `stream_options.include_usage`, usage arrives on one additional chunk before `data: [DONE]`,
/// whose `choices` is empty, while every other chunk carries `"usage": null` (chat completions API
/// reference, `stream_options.include_usage`). Oracle: [`OPENAI_CONTRACT_ACCOUNTING`].
fn openai_rows() -> [Row; 2] {
    [
        Row {
            name: "openai buffered",
            lane: Lane::OpenAi,
            delivery: Delivery::Buffered,
            model: PRIMARY_MODEL,
            upstream: openai_shape_buffered_body(PRIMARY_MODEL),
            accounting: OPENAI_CONTRACT_ACCOUNTING,
            raw_cache_write_tokens: Some(OPENAI_SHAPE_CACHE_WRITE),
        },
        Row {
            name: "openai streamed",
            lane: Lane::OpenAi,
            delivery: Delivery::Streamed,
            model: PRIMARY_MODEL,
            upstream: openai_shape_streamed_body(),
            accounting: OPENAI_CONTRACT_ACCOUNTING,
            raw_cache_write_tokens: Some(OPENAI_SHAPE_CACHE_WRITE),
        },
    ]
}

/// Azure OpenAI, buffered and streamed. **Contract fixture.**
///
/// Usage members and their relationships, per Microsoft: cache hits appear as
/// `prompt_tokens_details.cached_tokens`, inside `prompt_tokens`, and Standard deployments of
/// GPT-5.6 and later also report `prompt_tokens_details.cache_write_tokens`
/// (`learn.microsoft.com/en-us/azure/foundry/openai/how-to/prompt-caching`);
/// `completion_tokens_details.reasoning_tokens` is a breakdown of `completion_tokens`, billed as
/// output (`learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning`). The streamed
/// final-chunk placement is the OpenAI wire format's, which the adapter always requests.
///
/// Oracle: [`OPENAI_CONTRACT_ACCOUNTING`], the same as OpenAI's. The upstream bytes are identical
/// to compat's, so this pair is also what shows Azure's streaming path — which shares compat's
/// stream reader — does not carry compat's accounting.
fn azure_rows() -> [Row; 2] {
    [
        Row {
            name: "azure buffered",
            lane: Lane::Azure,
            delivery: Delivery::Buffered,
            model: PRIMARY_MODEL,
            upstream: openai_shape_buffered_body(PRIMARY_MODEL),
            accounting: OPENAI_CONTRACT_ACCOUNTING,
            raw_cache_write_tokens: Some(OPENAI_SHAPE_CACHE_WRITE),
        },
        Row {
            name: "azure streamed",
            lane: Lane::Azure,
            delivery: Delivery::Streamed,
            model: PRIMARY_MODEL,
            upstream: openai_shape_streamed_body(),
            accounting: OPENAI_CONTRACT_ACCOUNTING,
            raw_cache_write_tokens: Some(OPENAI_SHAPE_CACHE_WRITE),
        },
    ]
}

/// Generic OpenAI-compatible backend: buffered, and streamed on both streaming paths.
/// **Contract fixture** — of the OpenAI wire format, which is the only schema a generic backend
/// is known to speak; the usage members are cited as for [`openai_rows`].
///
/// No billing meaning is inherited from OpenAI. The raw `cache_write_tokens` must still reach the
/// client, because it is an OpenAI-standard member the client asked its own backend for.
/// Oracle: [`COMPAT_ACCOUNTING`].
fn compat_rows() -> [Row; 3] {
    [
        Row {
            name: "compat buffered",
            lane: Lane::Compat {
                stream_options_support: true,
            },
            delivery: Delivery::Buffered,
            model: PRIMARY_MODEL,
            upstream: openai_shape_buffered_body(PRIMARY_MODEL),
            accounting: COMPAT_ACCOUNTING,
            raw_cache_write_tokens: Some(OPENAI_SHAPE_CACHE_WRITE),
        },
        Row {
            name: "compat streamed, stream_options injected",
            lane: Lane::Compat {
                stream_options_support: true,
            },
            delivery: Delivery::Streamed,
            model: PRIMARY_MODEL,
            upstream: openai_shape_streamed_body(),
            accounting: COMPAT_ACCOUNTING,
            raw_cache_write_tokens: Some(OPENAI_SHAPE_CACHE_WRITE),
        },
        Row {
            name: "compat streamed, request forwarded verbatim",
            lane: Lane::Compat {
                stream_options_support: false,
            },
            delivery: Delivery::Streamed,
            model: PRIMARY_MODEL,
            upstream: openai_shape_streamed_body(),
            accounting: COMPAT_ACCOUNTING,
            raw_cache_write_tokens: Some(OPENAI_SHAPE_CACHE_WRITE),
        },
    ]
}

// ---------------------------------------------------------------------------------------------
// Anthropic Messages
// ---------------------------------------------------------------------------------------------

/// The usage object the Anthropic rows receive.
///
/// 20,000 uncached input tokens, 8,000 read from cache and 4,000 written to it — 3,000 at the
/// 5-minute TTL and 1,000 at the 1-hour TTL — and a 3,000-token completion of which 1,000 were
/// thinking. Plain input is below [`UPPER_TIER_THRESHOLD`]; the total prompt of 32,000 is above
/// it, so the upper tier is reached only through the cache tokens.
const ANTHROPIC_USAGE: &str = r#"{"input_tokens":20000,"cache_creation_input_tokens":4000,"cache_read_input_tokens":8000,"cache_creation":{"ephemeral_5m_input_tokens":3000,"ephemeral_1h_input_tokens":1000},"output_tokens":3000,"output_tokens_details":{"thinking_tokens":1000}}"#;

/// A Messages response carrying `usage`, with the thinking block extended thinking returns
/// ahead of the text.
fn anthropic_buffered_body(model: &str, usage: &str) -> Vec<u8> {
    format!(
        r#"{{"id":"msg_accounting_contract","type":"message","role":"assistant","model":"{model}","content":[{{"type":"thinking","thinking":"Reviewing the cached document.","signature":"accounting-contract-signature"}},{{"type":"text","text":"Summary."}}],"stop_reason":"end_turn","stop_sequence":null,"usage":{usage}}}"#
    )
    .into_bytes()
}

/// A streamed Messages response stating [`ANTHROPIC_USAGE`].
///
/// `message_start` carries the input side and the per-TTL breakdown; `message_delta` restates the
/// input aggregates cumulatively, as the provider does, and carries the final output count and its
/// thinking breakdown. The breakdown is stated once, so a stream reader that added the restated
/// aggregates to the first statement would double-count the cache.
fn anthropic_streamed_body() -> Vec<u8> {
    let message_start = format!(
        r#"{{"type":"message_start","message":{{"id":"msg_accounting_contract","type":"message","role":"assistant","content":[],"model":"{PRIMARY_MODEL}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":20000,"cache_creation_input_tokens":4000,"cache_read_input_tokens":8000,"cache_creation":{{"ephemeral_5m_input_tokens":3000,"ephemeral_1h_input_tokens":1000}},"output_tokens":1}}}}}}"#
    );
    let events: [(&str, &str); 10] = [
        ("message_start", &message_start),
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Reviewing the cached document."}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"accounting-contract-signature"}}"#,
        ),
        (
            "content_block_stop",
            r#"{"type":"content_block_stop","index":0}"#,
        ),
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Summary."}}"#,
        ),
        (
            "content_block_stop",
            r#"{"type":"content_block_stop","index":1}"#,
        ),
        (
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":20000,"cache_creation_input_tokens":4000,"cache_read_input_tokens":8000,"output_tokens":3000,"output_tokens_details":{"thinking_tokens":1000}}}"#,
        ),
        ("message_stop", r#"{"type":"message_stop"}"#),
    ];
    events
        .iter()
        .map(|(kind, data)| format!("event: {kind}\ndata: {data}\n\n"))
        .collect::<String>()
        .into_bytes()
}

/// The oracle for [`ANTHROPIC_USAGE`] under the Anthropic contract.
///
/// Cache is additive — `input_tokens` is the uncached remainder, so nothing is carved out of it and
/// each cache bucket is charged once at its own multiplier. Each TTL class is charged at its own
/// write multiplier. Thinking is contained in `output_tokens` — the standard output charge
/// excludes it and it is charged once at the thinking rate. Context size is 20,000 + 8,000 +
/// 4,000 = 32,000, so tier 1 prices.
///
/// | Component | Quantity × rate | nano-USD |
/// |---|---|---|
/// | plain input | 20,000 × 5,300 | 106,000,000 |
/// | cache read | 8,000 × 795 | 6,360,000 |
/// | cache write `5m` | 3,000 × 7,155 | 21,465,000 |
/// | cache write `1h` | 1,000 × 11,660 | 11,660,000 |
/// | standard output | (3,000 − 1,000) = 2,000 × 23,000 | 46,000,000 |
/// | thinking | 1,000 × 23,000 | 23,000,000 |
/// | **total** | | **214,485,000** |
const ANTHROPIC_CONTRACT_ACCOUNTING: Accounting = Accounting {
    cost_nano_usd: 214_485_000,
    status: CostStatus::Exact,
    prompt_tokens: 20_000,
    completion_tokens: 3_000,
    cache_read_tokens: 8_000,
    thinking_tokens: 1_000,
    cache_write: Some(Evidence {
        accounted_tokens: 4_000,
        incomplete: false,
    }),
};

/// Anthropic, buffered and streamed. **Contract fixture.**
///
/// Usage members and their relationships, per Anthropic: with prompt caching, the input is split
/// across `input_tokens`, `cache_read_input_tokens` and `cache_creation_input_tokens`, which sum
/// to the total input, and `cache_creation_input_tokens` equals the sum of the
/// `cache_creation.ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens` breakdown
/// (`platform.claude.com/docs/en/build-with-claude/prompt-caching`);
/// `output_tokens_details.thinking_tokens` is the part of `output_tokens` spent on internal
/// reasoning, and `output_tokens` remains the inclusive total used for billing (the `Usage`,
/// `MessageDeltaUsage` and `OutputTokensDetails` types of `anthropics/anthropic-sdk-python` @
/// `eb21a4352015686c30f5759e8c2f02d70f5371e2`); the counts on a `message_delta` event's `usage`,
/// at the event root, are cumulative (`platform.claude.com/docs/en/build-with-claude/streaming`).
/// Oracle: [`ANTHROPIC_CONTRACT_ACCOUNTING`].
fn anthropic_rows() -> [Row; 2] {
    [
        Row {
            name: "anthropic buffered",
            lane: Lane::Anthropic,
            delivery: Delivery::Buffered,
            model: PRIMARY_MODEL,
            upstream: anthropic_buffered_body(PRIMARY_MODEL, ANTHROPIC_USAGE),
            accounting: ANTHROPIC_CONTRACT_ACCOUNTING,
            raw_cache_write_tokens: None,
        },
        Row {
            name: "anthropic streamed",
            lane: Lane::Anthropic,
            delivery: Delivery::Streamed,
            model: PRIMARY_MODEL,
            upstream: anthropic_streamed_body(),
            accounting: ANTHROPIC_CONTRACT_ACCOUNTING,
            raw_cache_write_tokens: None,
        },
    ]
}

// ---------------------------------------------------------------------------------------------
// Anthropic inference geo
// ---------------------------------------------------------------------------------------------

/// The usage object both inference-geo rows receive, differing only in the geography stated.
///
/// 10,000 input and 2,000 output tokens, and nothing else — the surcharge is the subject, so
/// every other dimension is left at zero rather than added to the pile the oracle has to unwind.
fn anthropic_geo_usage(inference_geo: &str) -> String {
    format!(r#"{{"input_tokens":10000,"output_tokens":2000,"inference_geo":"{inference_geo}"}}"#)
}

/// A streamed Messages response stating a geography on `message_start` and nothing on
/// `message_delta`, which is where the provider states it.
///
/// That asymmetry is the point on this path: the terminal event carries the final counts but no
/// geography, so a streamed projection that read geo from the terminal event would silently drop
/// the surcharge — and the buffered row beside this one would still pass.
fn anthropic_geo_streamed_body(inference_geo: &str) -> Vec<u8> {
    let message_start = format!(
        r#"{{"type":"message_start","message":{{"id":"msg_accounting_contract","type":"message","role":"assistant","content":[],"model":"{GEO_MODEL}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":10000,"output_tokens":1,"inference_geo":"{inference_geo}"}}}}}}"#
    );
    let events: [(&str, &str); 5] = [
        ("message_start", &message_start),
        (
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        ),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Summary."}}"#,
        ),
        (
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":10000,"output_tokens":2000}}"#,
        ),
        ("message_stop", r#"{"type":"message_stop"}"#),
    ];
    events
        .iter()
        .map(|(kind, data)| format!("event: {kind}\ndata: {data}\n\n"))
        .collect::<String>()
        .into_bytes()
}

/// The oracle for a `us` request against [`GEO_MODEL`].
///
/// | Component | Quantity × rate | Standard | × 1.1 |
/// |---|---|---|---|
/// | input | 10,000 × 6,000 | 60,000,000 | 66,000,000 |
/// | output | 2,000 × 30,000 | 60,000,000 | 66,000,000 |
/// | **total** | | **120,000,000** | **132,000,000** |
///
/// The defect this row exists for is that the persisted and budgeted figures are short, not that
/// a header is: 120,000,000 is what both surfaces carried before the fix, and it is what they
/// would carry again if the surcharge reached only the response headers.
const ANTHROPIC_GEO_US_ACCOUNTING: Accounting = Accounting {
    cost_nano_usd: 132_000_000,
    status: CostStatus::Exact,
    prompt_tokens: 10_000,
    completion_tokens: 2_000,
    cache_read_tokens: 0,
    thinking_tokens: 0,
    cache_write: None,
};

/// Anthropic US-only inference, buffered and streamed. **Contract fixture.**
///
/// Usage members and their relationships, per Anthropic: the response `usage` object carries an
/// `inference_geo` member stating where inference ran, whose documented values are `us`,
/// `global` and `not_available`; US-only inference on Claude 4.6 and later is priced at 1.1x
/// across all token pricing categories
/// (`platform.claude.com/docs/en/manage-claude/data-residency`,
/// `platform.claude.com/docs/en/manage-claude/usage-cost-api`, accessed 2026-09-17). The member
/// is stated on the buffered `usage` object and on `message_start`, and is not restated on
/// `message_delta`. Oracle: [`ANTHROPIC_GEO_US_ACCOUNTING`].
fn anthropic_geo_rows() -> [Row; 2] {
    [
        Row {
            name: "anthropic inference geo us buffered",
            lane: Lane::Anthropic,
            delivery: Delivery::Buffered,
            model: GEO_MODEL,
            upstream: anthropic_buffered_body(GEO_MODEL, &anthropic_geo_usage("us")),
            accounting: ANTHROPIC_GEO_US_ACCOUNTING,
            raw_cache_write_tokens: None,
        },
        Row {
            name: "anthropic inference geo us streamed",
            lane: Lane::Anthropic,
            delivery: Delivery::Streamed,
            model: GEO_MODEL,
            upstream: anthropic_geo_streamed_body("us"),
            accounting: ANTHROPIC_GEO_US_ACCOUNTING,
            raw_cache_write_tokens: None,
        },
    ]
}

/// The same request reporting `global` bills the standard rate, on the same two surfaces.
///
/// Its oracle is [`ANTHROPIC_GEO_US_ACCOUNTING`]'s standard column. Without it the `us` rows
/// would prove that 132,000,000 is persisted, but not that the 12,000,000 difference is the
/// geography rather than the fixture's rates.
fn anthropic_geo_control_row() -> Row {
    Row {
        name: "anthropic inference geo global buffered",
        lane: Lane::Anthropic,
        delivery: Delivery::Buffered,
        model: GEO_MODEL,
        upstream: anthropic_buffered_body(GEO_MODEL, &anthropic_geo_usage("global")),
        accounting: Accounting {
            cost_nano_usd: 120_000_000,
            ..ANTHROPIC_GEO_US_ACCOUNTING
        },
        raw_cache_write_tokens: None,
    }
}

// ---------------------------------------------------------------------------------------------
// Gemini generateContent
// ---------------------------------------------------------------------------------------------

/// The usage metadata the Gemini rows receive.
///
/// A 40,000-token prompt of which 12,000 are cached content, 2,000 candidate tokens and 1,000
/// thought tokens beside them. Plain input is 28,000, below [`UPPER_TIER_THRESHOLD`]; the
/// prompt of 40,000 is above it, so the upper tier is reached only through the cached tokens.
const GEMINI_USAGE: &str = r#"{"promptTokenCount":40000,"candidatesTokenCount":2000,"totalTokenCount":43000,"cachedContentTokenCount":12000,"thoughtsTokenCount":1000}"#;

/// A `generateContent` response carrying [`GEMINI_USAGE`].
fn gemini_buffered_body() -> Vec<u8> {
    format!(
        r#"{{"candidates":[{{"content":{{"role":"model","parts":[{{"text":"Summary."}}]}},"finishReason":"STOP","index":0}}],"usageMetadata":{GEMINI_USAGE},"modelVersion":"{PRIMARY_MODEL}","responseId":"accounting-contract"}}"#
    )
    .into_bytes()
}

/// A `streamGenerateContent?alt=sse` response whose final chunk — the one carrying
/// `finishReason` — states [`GEMINI_USAGE`].
fn gemini_streamed_body() -> Vec<u8> {
    format!(
        concat!(
            r#"data: {{"candidates":[{{"content":{{"role":"model","parts":[{{"text":"Summ"}}]}},"index":0}}],"modelVersion":"{model}","responseId":"accounting-contract"}}"#,
            "\n\n",
            r#"data: {{"candidates":[{{"content":{{"role":"model","parts":[{{"text":"ary."}}]}},"finishReason":"STOP","index":0}}],"usageMetadata":{usage},"modelVersion":"{model}","responseId":"accounting-contract"}}"#,
            "\n\n",
        ),
        model = PRIMARY_MODEL,
        usage = GEMINI_USAGE,
    )
    .into_bytes()
}

/// The oracle for [`GEMINI_USAGE`] under the Gemini contract.
///
/// Cache is inclusive — `promptTokenCount` contains the cached content, which is carved out and
/// charged once at the cache-read multiplier. Thoughts are additive — `thoughtsTokenCount` sits
/// beside `candidatesTokenCount`, not inside it, so both are charged in full. Gemini reports no
/// cache write. Context size is 28,000 + 12,000 = 40,000, so tier 1 prices.
///
/// | Component | Quantity × rate | nano-USD |
/// |---|---|---|
/// | plain input | (40,000 − 12,000) = 28,000 × 5,300 | 148,400,000 |
/// | cache read | 12,000 × 795 | 9,540,000 |
/// | output | 2,000 × 23,000 | 46,000,000 |
/// | thoughts | 1,000 × 23,000 | 23,000,000 |
/// | **total** | | **226,940,000** |
const GEMINI_CONTRACT_ACCOUNTING: Accounting = Accounting {
    cost_nano_usd: 226_940_000,
    status: CostStatus::Exact,
    prompt_tokens: 28_000,
    completion_tokens: 2_000,
    cache_read_tokens: 12_000,
    thinking_tokens: 1_000,
    cache_write: None,
};

/// Gemini, buffered and streamed. **Contract fixture.**
///
/// Usage members and their relationships, per Google's `v1beta` discovery document
/// (`generativelanguage.googleapis.com/$discovery/rest?version=v1beta`, revision `20260910`),
/// `UsageMetadata`: `promptTokenCount` is the total effective prompt size and includes the
/// cached content, `cachedContentTokenCount` is the cached part of the prompt, and
/// `totalTokenCount` is prompt + thoughts + response candidates — so `thoughtsTokenCount` is not
/// part of `candidatesTokenCount`. Oracle: [`GEMINI_CONTRACT_ACCOUNTING`].
fn gemini_rows() -> [Row; 2] {
    [
        Row {
            name: "gemini buffered",
            lane: Lane::Gemini,
            delivery: Delivery::Buffered,
            model: PRIMARY_MODEL,
            upstream: gemini_buffered_body(),
            accounting: GEMINI_CONTRACT_ACCOUNTING,
            raw_cache_write_tokens: None,
        },
        Row {
            name: "gemini streamed",
            lane: Lane::Gemini,
            delivery: Delivery::Streamed,
            model: PRIMARY_MODEL,
            upstream: gemini_streamed_body(),
            accounting: GEMINI_CONTRACT_ACCOUNTING,
            raw_cache_write_tokens: None,
        },
    ]
}

// ---------------------------------------------------------------------------------------------
// Bedrock Converse
// ---------------------------------------------------------------------------------------------

/// The usage the Bedrock rows receive.
///
/// 18,000 non-cached input tokens, 9,000 read from cache and 5,000 written to it — 2,000 at the
/// 1-hour TTL and 3,000 at the 5-minute TTL — and 2,000 output tokens. Plain input is below
/// [`UPPER_TIER_THRESHOLD`]; the total input of 32,000 is above it, so the upper tier is reached
/// only through the cache tokens.
fn bedrock_usage() -> serde_json::Value {
    serde_json::json!({
        "inputTokens": 18_000,
        "outputTokens": 2_000,
        "totalTokens": 34_000,
        "cacheReadInputTokens": 9_000,
        "cacheWriteInputTokens": 5_000,
        "cacheDetails": [
            {"ttl": "1h", "inputTokens": 2_000},
            {"ttl": "5m", "inputTokens": 3_000}
        ]
    })
}

/// A `Converse` response carrying `usage`.
fn bedrock_buffered_body(usage: serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "output": {"message": {"role": "assistant", "content": [{"text": "Summary."}]}},
        "stopReason": "end_turn",
        "usage": usage,
        "metrics": {"latencyMs": 1_200}
    }))
    .expect("fixture serializes")
}

/// A `ConverseStream` response, in the binary event-stream framing, whose `metadata` event
/// carries [`bedrock_usage`].
fn bedrock_streamed_body() -> Vec<u8> {
    let events = [
        ("messageStart", serde_json::json!({"role": "assistant"})),
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex": 0, "delta": {"text": "Summary."}}),
        ),
        (
            "contentBlockStop",
            serde_json::json!({"contentBlockIndex": 0}),
        ),
        ("messageStop", serde_json::json!({"stopReason": "end_turn"})),
        (
            "metadata",
            serde_json::json!({"usage": bedrock_usage(), "metrics": {"latencyMs": 1_200}}),
        ),
    ];
    events
        .iter()
        .flat_map(|(kind, payload)| {
            build_frame(
                kind,
                &serde_json::to_vec(payload).expect("event serializes"),
            )
        })
        .collect()
}

/// The oracle for [`bedrock_usage`] under the Bedrock contract.
///
/// Cache is additive — `inputTokens` holds only the tokens neither read from nor written to the
/// cache, so nothing is carved out of it and each cache bucket is charged once at its own
/// multiplier. Each TTL class is charged at its own write multiplier. Converse reports no
/// reasoning count. Context size is 18,000 + 9,000 + 5,000 = 32,000, so tier 1 prices.
///
/// | Component | Quantity × rate | nano-USD |
/// |---|---|---|
/// | plain input | 18,000 × 5,300 | 95,400,000 |
/// | cache read | 9,000 × 795 | 7,155,000 |
/// | cache write `5m` | 3,000 × 7,155 | 21,465,000 |
/// | cache write `1h` | 2,000 × 11,660 | 23,320,000 |
/// | output | 2,000 × 23,000 | 46,000,000 |
/// | **total** | | **193,340,000** |
const BEDROCK_CONTRACT_ACCOUNTING: Accounting = Accounting {
    cost_nano_usd: 193_340_000,
    status: CostStatus::Exact,
    prompt_tokens: 18_000,
    completion_tokens: 2_000,
    cache_read_tokens: 9_000,
    thinking_tokens: 0,
    cache_write: Some(Evidence {
        accounted_tokens: 5_000,
        incomplete: false,
    }),
};

/// Bedrock, buffered and streamed. **Contract fixture.**
///
/// Usage members and their relationships, per AWS: when prompt caching is enabled, `inputTokens`
/// represents only the tokens not read from or written to the cache, and the total input is
/// `inputTokens + cacheReadInputTokens + cacheWriteInputTokens`
/// (`docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html`); `cacheDetails` is the
/// breakdown of cache writes by TTL, sorted `1h` before `5m`, each `ttl` one of `5m` and `1h`;
/// `totalTokens` is the total of the input tokens and the tokens generated — set here from the
/// total input above, and read by no accounting path; and a
/// `ConverseStream` response reports usage on its `metadata` event (`TokenUsage`, `CacheDetail`,
/// `CacheTTL` and `ConverseStreamMetadataEvent` in `boto/botocore` @
/// `e3091ae01b90bb7fb4838f91a1fe545abbb47dd5`, `bedrock-runtime/2023-09-30`). Oracle:
/// [`BEDROCK_CONTRACT_ACCOUNTING`].
fn bedrock_rows() -> [Row; 2] {
    [
        Row {
            name: "bedrock buffered",
            lane: Lane::Bedrock,
            delivery: Delivery::Buffered,
            model: PRIMARY_MODEL,
            upstream: bedrock_buffered_body(bedrock_usage()),
            accounting: BEDROCK_CONTRACT_ACCOUNTING,
            raw_cache_write_tokens: None,
        },
        Row {
            name: "bedrock streamed",
            lane: Lane::Bedrock,
            delivery: Delivery::Streamed,
            model: PRIMARY_MODEL,
            upstream: bedrock_streamed_body(),
            accounting: BEDROCK_CONTRACT_ACCOUNTING,
            raw_cache_write_tokens: None,
        },
    ]
}

// ---------------------------------------------------------------------------------------------
// Status and bounded-evidence rows
// ---------------------------------------------------------------------------------------------

/// The Bedrock usage the `rate-fallback` row, the complete evidence twin, receives.
///
/// 6,000 non-cached input tokens, 10,000 read from cache and 5,000 written to it — 2,000 at the
/// 1-hour TTL and 3,000 at the 5-minute TTL — and 2,000 output tokens.
fn rate_fallback_usage() -> serde_json::Value {
    serde_json::json!({
        "inputTokens": 6_000,
        "outputTokens": 2_000,
        "totalTokens": 23_000,
        "cacheReadInputTokens": 10_000,
        "cacheWriteInputTokens": 5_000,
        "cacheDetails": [
            {"ttl": "1h", "inputTokens": 2_000},
            {"ttl": "5m", "inputTokens": 3_000}
        ]
    })
}

/// How many unfamiliar `ttl` entries the incomplete twin spreads its 2,000 fallback-priced
/// tokens across, [`UNFAMILIAR_TTL_TOKENS`] each: more entries than the evidence document
/// retains.
const UNFAMILIAR_TTL_ENTRIES: usize = 40;
const UNFAMILIAR_TTL_TOKENS: u64 = 50;
const _: () = assert!(UNFAMILIAR_TTL_ENTRIES > MAX_RETAINED_EVIDENCE_ENTRIES);
const _: () = assert!(UNFAMILIAR_TTL_ENTRIES as u64 * UNFAMILIAR_TTL_TOKENS == 2_000);

/// [`rate_fallback_usage`], with the 2,000 tokens it reports at `1h` spread instead across
/// [`UNFAMILIAR_TTL_ENTRIES`] entries whose `ttl` names no duration and is longer than the
/// evidence document retains of a key. The configured `5m` entry arrives last, after the whole
/// run of unknown classes, and still owes only its own exact rate. Aggregate, input, cache read
/// and output are unchanged.
fn incomplete_evidence_usage() -> serde_json::Value {
    let mut details: Vec<serde_json::Value> = (0..UNFAMILIAR_TTL_ENTRIES)
        .map(|i| {
            serde_json::json!({
                "ttl": format!("unfamiliar-ttl-{i:02}-{}", "x".repeat(MAX_RAW_KEY_BYTES)),
                "inputTokens": UNFAMILIAR_TTL_TOKENS
            })
        })
        .collect();
    details.push(serde_json::json!({"ttl": "5m", "inputTokens": 3_000}));
    let mut usage = rate_fallback_usage();
    usage["cacheDetails"] = serde_json::Value::Array(details);
    usage
}

/// The oracle for [`rate_fallback_usage`] against [`FIVE_M_ONLY_MODEL`].
///
/// Cache is additive, as for [`BEDROCK_CONTRACT_ACCOUNTING`]. The tier prices `5m` exactly; `1h`
/// is a class this tier does not configure, so it takes the tier's fallback multiplier, 1.3, and
/// the request reports `rate-fallback`. The model has one tier, so no tier crossing is at stake.
///
/// | Component | Quantity × rate | nano-USD |
/// |---|---|---|
/// | plain input | 6,000 × 4,100 | 24,600,000 |
/// | cache read | 10,000 × 820 | 8,200,000 |
/// | cache write `5m` | 3,000 × 5,330 | 15,990,000 |
/// | cache write `1h`, fallback rate | 2,000 × 5,330 | 10,660,000 |
/// | output | 2,000 × 17,000 | 34,000,000 |
/// | **total** | | **93,450,000** |
const RATE_FALLBACK_ACCOUNTING: Accounting = Accounting {
    cost_nano_usd: 93_450_000,
    status: CostStatus::RateFallback,
    prompt_tokens: 6_000,
    completion_tokens: 2_000,
    cache_read_tokens: 10_000,
    thinking_tokens: 0,
    cache_write: Some(Evidence {
        accounted_tokens: 5_000,
        incomplete: false,
    }),
};

/// The oracle for [`incomplete_evidence_usage`]: [`RATE_FALLBACK_ACCOUNTING`], with only the
/// evidence document's completeness flipped.
///
/// An unknown class takes the same tier fallback multiplier as an unconfigured known one, so the
/// 2,000 unknown-class tokens cost what the 2,000 `1h` tokens did: 2,000 × 5,330 = 10,660,000.
/// Aggregate and details still agree at 5,000. The many unknown observations also make duplicate
/// identity indeterminate, which alone would report `reconciled`; `rate-fallback` outranks it.
const INCOMPLETE_EVIDENCE_ACCOUNTING: Accounting = Accounting {
    cache_write: Some(Evidence {
        accounted_tokens: 5_000,
        incomplete: true,
    }),
    ..RATE_FALLBACK_ACCOUNTING
};

/// `rate-fallback`, and the complete evidence twin: Bedrock, buffered, against
/// [`FIVE_M_ONLY_MODEL`]. **Contract fixture.**
///
/// Usage members and their relationships as cited for [`bedrock_rows`]; `1h` and `5m` are the two
/// values botocore's `CacheTTL` enumerates. Only the `5m`-only tier is synthetic, and that sits
/// outside the usage object. Oracle: [`RATE_FALLBACK_ACCOUNTING`].
fn rate_fallback_row() -> Row {
    Row {
        name: "rate-fallback, complete evidence twin",
        lane: Lane::Bedrock,
        delivery: Delivery::Buffered,
        model: FIVE_M_ONLY_MODEL,
        upstream: bedrock_buffered_body(rate_fallback_usage()),
        accounting: RATE_FALLBACK_ACCOUNTING,
        raw_cache_write_tokens: None,
    }
}

/// The incomplete evidence twin: Bedrock, buffered, against [`FIVE_M_ONLY_MODEL`].
/// **Robustness fixture.**
///
/// The shape is Bedrock's `cacheDetails` array of `{ttl, inputTokens}` and deserializes, but its
/// values are not provider-documented: botocore's `CacheTTL` enumerates only `5m` and `1h`, and
/// these `ttl` strings are neither. It exercises how the gateway bounds evidence from
/// malformed-but-deserializable upstream data, and claims nothing about what Bedrock sends.
/// Oracle: [`INCOMPLETE_EVIDENCE_ACCOUNTING`] — identical cost, status, token columns and budget
/// increment to [`rate_fallback_row`], with the persisted document marked incomplete and still
/// inside the byte bound.
fn incomplete_evidence_row() -> Row {
    Row {
        name: "incomplete evidence twin",
        lane: Lane::Bedrock,
        delivery: Delivery::Buffered,
        model: FIVE_M_ONLY_MODEL,
        upstream: bedrock_buffered_body(incomplete_evidence_usage()),
        accounting: INCOMPLETE_EVIDENCE_ACCOUNTING,
        raw_cache_write_tokens: None,
    }
}

/// The oracle for [`OPENAI_SHAPE_USAGE`] from [`UNCATALOGUED_MODEL`].
///
/// With no price there is no defensible cost: the cost is zero and says so with
/// `cost-unavailable` rather than posing as a confident zero, and the budget counter moves by
/// that zero. The quantities do not depend on the model's price, so the token columns and the
/// accounted cache write, with its evidence, are [`OPENAI_CONTRACT_ACCOUNTING`]'s.
const COST_UNAVAILABLE_ACCOUNTING: Accounting = Accounting {
    cost_nano_usd: 0,
    status: CostStatus::CostUnavailable,
    ..OPENAI_CONTRACT_ACCOUNTING
};

/// `cost-unavailable`: OpenAI, buffered, from [`UNCATALOGUED_MODEL`]. **Contract fixture.**
///
/// Usage members and their relationships as cited for [`openai_rows`]; the model ID is synthetic
/// and absent from the catalogue by design. Oracle: [`COST_UNAVAILABLE_ACCOUNTING`].
fn cost_unavailable_row() -> Row {
    Row {
        name: "cost-unavailable",
        lane: Lane::OpenAi,
        delivery: Delivery::Buffered,
        model: UNCATALOGUED_MODEL,
        upstream: openai_shape_buffered_body(UNCATALOGUED_MODEL),
        accounting: COST_UNAVAILABLE_ACCOUNTING,
        raw_cache_write_tokens: Some(OPENAI_SHAPE_CACHE_WRITE),
    }
}

/// An Anthropic usage object whose cache-write breakdown — 3,000 at `5m` and 2,000 at `1h` — sums
/// to 5,000, above its own `cache_creation_input_tokens` aggregate of 4,000. Everything else is
/// [`ANTHROPIC_USAGE`].
const ANTHROPIC_CONTRADICTORY_USAGE: &str = r#"{"input_tokens":20000,"cache_creation_input_tokens":4000,"cache_read_input_tokens":8000,"cache_creation":{"ephemeral_5m_input_tokens":3000,"ephemeral_1h_input_tokens":2000},"output_tokens":3000,"output_tokens_details":{"thinking_tokens":1000}}"#;

/// The oracle for [`ANTHROPIC_CONTRADICTORY_USAGE`].
///
/// Aggregate and breakdown contradict each other, so the accounted cache write is the larger of
/// the two, 5,000, never their sum and never the smaller; every class is configured, so each is
/// priced at its own multiplier and nothing is left to a fallback rate. The request reports
/// `reconciled`. Otherwise the Anthropic contract applies as for
/// [`ANTHROPIC_CONTRACT_ACCOUNTING`]. Context size is 20,000 + 8,000 + 5,000 = 33,000, so tier 1
/// prices.
///
/// | Component | Quantity × rate | nano-USD |
/// |---|---|---|
/// | plain input | 20,000 × 5,300 | 106,000,000 |
/// | cache read | 8,000 × 795 | 6,360,000 |
/// | cache write `5m` | 3,000 × 7,155 | 21,465,000 |
/// | cache write `1h` | 2,000 × 11,660 | 23,320,000 |
/// | standard output | (3,000 − 1,000) = 2,000 × 23,000 | 46,000,000 |
/// | thinking | 1,000 × 23,000 | 23,000,000 |
/// | **total** | | **226,145,000** |
const RECONCILED_ACCOUNTING: Accounting = Accounting {
    cost_nano_usd: 226_145_000,
    status: CostStatus::Reconciled,
    prompt_tokens: 20_000,
    completion_tokens: 3_000,
    cache_read_tokens: 8_000,
    thinking_tokens: 1_000,
    cache_write: Some(Evidence {
        accounted_tokens: 5_000,
        incomplete: false,
    }),
};

/// `reconciled`: Anthropic, buffered. **Robustness fixture.**
///
/// The shape is Anthropic's Messages `usage` with its `cache_creation` breakdown, as cited for
/// [`anthropic_rows`], and deserializes; the values are not provider-documented, because Anthropic
/// documents `cache_creation_input_tokens` as the sum of that breakdown. It exercises the
/// conservative quantity policy for self-contradictory upstream data and claims nothing about what
/// Anthropic sends. Oracle: [`RECONCILED_ACCOUNTING`].
fn reconciled_row() -> Row {
    Row {
        name: "reconciled",
        lane: Lane::Anthropic,
        delivery: Delivery::Buffered,
        model: PRIMARY_MODEL,
        upstream: anthropic_buffered_body(PRIMARY_MODEL, ANTHROPIC_CONTRADICTORY_USAGE),
        accounting: RECONCILED_ACCOUNTING,
        raw_cache_write_tokens: None,
    }
}

// ---------------------------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------------------------

/// The containers and pricing every row shares.
struct Env {
    pg: PgContainer,
    redis: RedisContainer,
    pricing: Arc<std::sync::RwLock<PricingDb>>,
}

impl Env {
    async fn start() -> Self {
        Self {
            pg: PgContainer::start().await.expect("pg container must start"),
            redis: RedisContainer::start()
                .await
                .expect("redis container must start"),
            pricing: synthetic_pricing_holder(),
        }
    }

    /// Clears every row and the per-identity counter, so the next row's are the only ones.
    async fn reset(&self) {
        sqlx::query("DELETE FROM spend_records")
            .execute(&self.pg.pool)
            .await
            .expect("clear spend_records");
        let mut conn = self.redis.pool.get().await.expect("redis conn");
        redis::cmd("DEL")
            .arg(DEFAULT_SPEND_KEY)
            .query_async::<()>(&mut *conn)
            .await
            .expect("clear spend counter");
    }
}

/// The persisted spend row, as the runner reads it back.
#[derive(sqlx::FromRow)]
struct PersistedRow {
    prompt_tokens: i64,
    completion_tokens: i64,
    cache_read_tokens: i64,
    thinking_tokens: i64,
    cost_nano_usd: i64,
    cost_status: String,
    usage_evidence: Option<serde_json::Value>,
}

/// The one spend row the request wrote, polled with a bound because the write is spawned.
async fn persisted_row(env: &Env, name: &str) -> PersistedRow {
    for _ in 0..40 {
        let rows: Vec<PersistedRow> = sqlx::query_as(
            "SELECT prompt_tokens, completion_tokens, cache_read_tokens, thinking_tokens, \
             cost_nano_usd, cost_status, usage_evidence FROM spend_records",
        )
        .fetch_all(&env.pg.pool)
        .await
        .expect("read spend_records");
        match rows.len() {
            0 => tokio::time::sleep(std::time::Duration::from_millis(50)).await,
            1 => return rows.into_iter().next().expect("one row"),
            n => panic!("{name}: one request must write one spend row, found {n}"),
        }
    }
    panic!("{name}: no spend row was persisted within 2s");
}

/// The per-identity budget counter. Read after the row exists: the spend writer increments Redis
/// before it inserts the row, in the same task.
async fn budget_counter(env: &Env) -> Option<i64> {
    let mut conn = env.redis.pool.get().await.expect("redis conn");
    redis::cmd("GET")
        .arg(DEFAULT_SPEND_KEY)
        .query_async(&mut *conn)
        .await
        .expect("read spend counter")
}

/// Formats nano-USD as the cost header does, with six decimal places — independently of the code
/// under test, and only for amounts it can carry losslessly.
fn display_usd(nano_usd: u64) -> String {
    assert_eq!(
        nano_usd % 1_000,
        0,
        "every oracle must be a multiple of 1,000 nano-USD, got {nano_usd}"
    );
    format!(
        "{}.{:06}",
        nano_usd / 1_000_000_000,
        (nano_usd % 1_000_000_000) / 1_000
    )
}

/// Splits an SSE body into `(event type, data)` records.
fn sse_records(body: &str) -> Vec<(Option<&str>, &str)> {
    body.split("\n\n")
        .filter_map(|record| {
            let mut event = None;
            let mut data = None;
            for line in record.lines() {
                if let Some(rest) = line.strip_prefix("event: ") {
                    event = Some(rest);
                } else if let Some(rest) = line.strip_prefix("data: ") {
                    data = Some(rest);
                }
            }
            data.map(|d| (event, d))
        })
        .collect()
}

/// What the client-facing surface reported for one row.
struct Reported {
    cost: String,
    status: String,
    /// The usage object the client received. `null` for a stream with no usage-bearing chunk:
    /// the client did not ask for `stream_options.include_usage`, and a lane that translates its
    /// provider's stream may then leave usage off the chunks it emits.
    usage: serde_json::Value,
}

/// Reads cost, status and usage off a buffered response, and checks the token headers against the
/// body's usage — item (g).
fn reported_buffered(name: &str, response: &axum_test::TestResponse) -> Reported {
    let header = |key: &str| {
        response
            .headers()
            .get(key)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_else(|| panic!("{name}: missing {key} header"))
            .to_string()
    };
    let body: serde_json::Value = response.json();
    let usage = body["usage"].clone();
    assert_eq!(
        header(CostHeader::INPUT_TOKENS),
        usage["prompt_tokens"].to_string(),
        "{name}: the input-tokens header must equal the body's usage.prompt_tokens"
    );
    assert_eq!(
        header(CostHeader::OUTPUT_TOKENS),
        usage["completion_tokens"].to_string(),
        "{name}: the output-tokens header must equal the body's usage.completion_tokens"
    );
    Reported {
        cost: header(CostHeader::REQUEST_COST),
        status: header(CostHeader::COST_STATUS),
        usage,
    }
}

/// Reads cost and status off the terminal `oxigate.usage` event, and usage off the last provider
/// chunk that carried one.
fn reported_streamed(name: &str, body: &str) -> Reported {
    let records = sse_records(body);
    let usage_events: Vec<&str> = records
        .iter()
        .filter(|(event, _)| *event == Some("oxigate.usage"))
        .map(|(_, data)| *data)
        .collect();
    assert_eq!(
        usage_events.len(),
        1,
        "{name}: exactly one oxigate.usage event must be emitted"
    );
    let event: serde_json::Value =
        serde_json::from_str(usage_events[0]).expect("oxigate.usage data must be JSON");
    let usage = records
        .iter()
        .filter(|(event, data)| event.is_none() && *data != "[DONE]")
        .filter_map(|(_, data)| serde_json::from_str::<serde_json::Value>(data).ok())
        .rev()
        .find_map(|chunk| chunk.get("usage").filter(|u| !u.is_null()).cloned())
        .unwrap_or(serde_json::Value::Null);
    let member = |key: &str| {
        event[key]
            .as_str()
            .unwrap_or_else(|| panic!("{name}: oxigate.usage has no {key}"))
            .to_string()
    };
    Reported {
        cost: member(CostHeader::REQUEST_COST),
        status: member("cost_status"),
        usage,
    }
}

/// Runs one row end to end and asserts every surface against its oracle.
async fn run_row(env: &Env, row: &Row) {
    let name = row.name;
    let expected = &row.accounting;
    env.reset().await;

    let upstream = MockServer::start().await;
    let content_type = match row.delivery {
        Delivery::Buffered => "application/json",
        Delivery::Streamed => row.lane.streamed_content_type(),
    };
    Mock::given(method("POST"))
        .and(path(row.lane.upstream_path(row.model, row.delivery)))
        .respond_with(ResponseTemplate::new(200).set_body_raw(row.upstream.clone(), content_type))
        .expect(1)
        .mount(&upstream)
        .await;

    let tap = Arc::new(StreamUsageTap::new(
        row.lane
            .adapter(&upstream.uri(), Arc::clone(&env.pricing))
            .await,
    ));
    let gateway = TestGateway::spawn_with_pricing(
        env.pg.pool.clone(),
        env.redis.pool.clone(),
        Arc::clone(&tap) as Arc<dyn ProviderAdapterExt>,
        Arc::clone(&env.pricing),
        None,
    )
    .await;

    // Pretty-printed on purpose: every re-serializing path emits compact JSON, so only the
    // raw-forward path can deliver these exact bytes upstream. No `stream_options` either, so its
    // presence upstream can only mean the adapter injected it.
    let client_body = serde_json::to_vec_pretty(&serde_json::json!({
        "model": row.model,
        "messages": [{"role": "user", "content": "Summarize the cached document."}],
        "stream": row.delivery == Delivery::Streamed
    }))
    .expect("request serializes");
    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .content_type("application/json")
        .bytes(Bytes::from(client_body.clone()))
        .await;
    response.assert_status(StatusCode::OK);

    // The dispatch path the adapter took, read off the request it sent.
    let received = upstream
        .received_requests()
        .await
        .expect("wiremock records requests");
    assert_eq!(received.len(), 1, "{name}: one upstream request");
    let sent_upstream = &received[0].body;
    match row.lane.upstream_request(row.delivery) {
        UpstreamRequest::Verbatim => assert!(
            *sent_upstream == client_body,
            "{name}: the client's request must be forwarded byte for byte, got {}",
            String::from_utf8_lossy(sent_upstream)
        ),
        UpstreamRequest::RequestsStreamUsage => {
            let sent: serde_json::Value =
                serde_json::from_slice(sent_upstream).expect("upstream request is JSON");
            assert_eq!(
                sent["stream_options"]["include_usage"],
                serde_json::json!(true),
                "{name}: the adapter must ask the provider for streamed usage, sent {sent}"
            );
        }
        UpstreamRequest::Unconstrained => {}
    }

    let reported = match row.delivery {
        Delivery::Buffered => reported_buffered(name, &response),
        Delivery::Streamed => reported_streamed(name, &response.text()),
    };

    // (c) and (d): the client-facing cost and status.
    assert_eq!(
        reported.cost,
        display_usd(expected.cost_nano_usd),
        "{name}: reported cost"
    );
    assert_eq!(
        reported.status,
        expected.status.as_str(),
        "{name}: reported status"
    );

    // (a), (d), (e) and (f): the persisted row.
    let persisted = persisted_row(env, name).await;
    assert_eq!(
        persisted.cost_nano_usd,
        i64::try_from(expected.cost_nano_usd).expect("oracle fits BIGINT"),
        "{name}: persisted cost"
    );
    assert_eq!(
        persisted.cost_status,
        expected.status.as_str(),
        "{name}: persisted status"
    );
    assert_eq!(
        (
            persisted.prompt_tokens,
            persisted.completion_tokens,
            persisted.cache_read_tokens,
            persisted.thinking_tokens,
        ),
        (
            expected.prompt_tokens,
            expected.completion_tokens,
            expected.cache_read_tokens,
            expected.thinking_tokens,
        ),
        "{name}: persisted (prompt, completion, cache_read, thinking) token columns"
    );
    match (expected.cache_write, persisted.usage_evidence) {
        (None, None) => {}
        (None, Some(doc)) => {
            panic!("{name}: no cache write was accounted, but evidence was persisted: {doc}")
        }
        (Some(_), None) => panic!("{name}: a cache write was accounted, but no evidence persisted"),
        (Some(evidence), Some(doc)) => {
            let doc: UsageEvidence =
                serde_json::from_value(doc).expect("usage_evidence must deserialize");
            assert_eq!(
                (doc.cache_write.accounted_tokens, doc.cache_write.incomplete),
                (evidence.accounted_tokens, evidence.incomplete),
                "{name}: persisted evidence (accounted_tokens, incomplete)"
            );
            // Measured as the gateway serialized it, not as Postgres renders JSONB back, which
            // adds whitespace the bound never counted.
            let serialized = serde_json::to_string(&doc).expect("evidence serializes");
            assert!(
                serialized.len() <= CACHE_WRITE_EVIDENCE_MAX_BYTES,
                "{name}: persisted evidence is {} bytes, above the {CACHE_WRITE_EVIDENCE_MAX_BYTES}-byte bound",
                serialized.len()
            );
        }
    }

    // (f), public half: `cache_creation_input_tokens` carries exactly the accounted quantity, and
    // is absent when nothing was accounted. Buffered, it is read off the response body the
    // gateway serialized. Streamed, it is read off the normalized `Usage` the gateway finalized
    // from: the gateway forwards the adapter's chunk bytes and never re-serializes that `Usage`
    // onto the wire, so on most lanes the client's copy is the provider's own representation and
    // this member is not on it at all.
    let expected_published = expected.cache_write.map(|e| e.accounted_tokens);
    let published = match row.delivery {
        Delivery::Buffered => reported
            .usage
            .get("cache_creation_input_tokens")
            .map(|v| v.as_u64().expect("an integer token count")),
        Delivery::Streamed => {
            let finalized = tap
                .last_usage()
                .unwrap_or_else(|| panic!("{name}: the adapter's stream reported no usage"))
                .cache_creation_input_tokens;
            // Bedrock's adapter builds its OpenAI-shaped chunks from the normalized usage, so
            // there the client's copy must carry exactly what was finalized. On the other
            // streamed lanes it cannot: the gateway forwards the adapter's chunk bytes and never
            // re-serializes the normalized `Usage` onto the wire, so what the client sees is the
            // provider's own representation, checked as the raw echo below.
            if matches!(row.lane, Lane::Bedrock) {
                assert_eq!(
                    reported
                        .usage
                        .get("cache_creation_input_tokens")
                        .map(|v| v.as_u64().expect("an integer token count")),
                    finalized,
                    "{name}: the finalized cache_creation_input_tokens must reach the client"
                );
            }
            finalized
        }
    };
    assert_eq!(
        published, expected_published,
        "{name}: published cache_creation_input_tokens"
    );
    assert_eq!(
        reported.usage["prompt_tokens_details"]
            .get("cache_write_tokens")
            .and_then(|v| v.as_u64()),
        row.raw_cache_write_tokens,
        "{name}: the provider's raw cache_write_tokens must reach the client unchanged"
    );

    // (b): the budget counter moved by exactly the oracle.
    assert_eq!(
        budget_counter(env).await,
        Some(i64::try_from(expected.cost_nano_usd).expect("oracle fits i64")),
        "{name}: budget counter"
    );
}

/// Every contract row, buffered and streamed, and every status and evidence row, through the real
/// gateway.
#[tokio::test]
async fn accounting_contract_matrix() {
    let env = Env::start().await;
    let rows = openai_rows()
        .into_iter()
        .chain(azure_rows())
        .chain(compat_rows())
        .chain(anthropic_rows())
        .chain(anthropic_geo_rows())
        .chain(gemini_rows())
        .chain(bedrock_rows())
        .chain([
            anthropic_geo_control_row(),
            rate_fallback_row(),
            incomplete_evidence_row(),
            cost_unavailable_row(),
            reconciled_row(),
        ]);
    for row in rows {
        run_row(&env, &row).await;
    }
}

// ---------------------------------------------------------------------------------------------
// Budget trip
// ---------------------------------------------------------------------------------------------

/// The hard cap the budget trip test configures: 0.3 USD, 300,000,000 nano-USD.
const BUDGET_TRIP_HARD_CAP_USD: f64 = 0.3;
const BUDGET_TRIP_HARD_CAP_NANO_USD: u64 = 300_000_000;

/// A heavily cached Bedrock prompt: 2,000 non-cached input tokens, 30,000 read from cache and
/// 4,000 written to it at the 5-minute TTL, and 1,000 output tokens.
fn heavily_cached_bedrock_usage() -> serde_json::Value {
    serde_json::json!({
        "inputTokens": 2_000,
        "outputTokens": 1_000,
        "totalTokens": 37_000,
        "cacheReadInputTokens": 30_000,
        "cacheWriteInputTokens": 4_000,
        "cacheDetails": [{"ttl": "5m", "inputTokens": 4_000}]
    })
}

/// The cost of one [`heavily_cached_bedrock_usage`] request under the Bedrock contract.
///
/// Context size is 2,000 + 30,000 + 4,000 = 36,000, so tier 1 prices.
///
/// | Component | Quantity × rate | nano-USD |
/// |---|---|---|
/// | plain input | 2,000 × 5,300 | 10,600,000 |
/// | cache read | 30,000 × 795 | 23,850,000 |
/// | cache write `5m` | 4,000 × 7,155 | 28,620,000 |
/// | output | 1,000 × 23,000 | 23,000,000 |
/// | **total** | | **86,070,000** |
const BUDGET_TRIP_REQUEST_COST: u64 = 86_070_000;

/// What the same request cost when the Bedrock lane dropped both cache buckets: a context of
/// 2,000 prices at tier 0 — 2,000 × 3,700 + 1,000 × 19,000 = 26,400,000.
const CACHE_BLIND_REQUEST_COST: u64 = 26_400_000;

/// Waits, with a bound, for the budget counter to read `expected`. The spend write is spawned,
/// and the next request's budget check has to see it.
async fn await_budget_counter(env: &Env, expected: u64, context: &str) {
    let expected = i64::try_from(expected).expect("spend fits i64");
    for _ in 0..40 {
        if budget_counter(env).await == Some(expected) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!(
        "{context}: budget counter did not reach {expected} within 2s, read {:?}",
        budget_counter(env).await
    );
}

/// A hard cap trips on the request the cache-inclusive cost predicts, not on the later one a
/// cache-blind cost would. **Contract fixture** — usage members and their relationships as cited
/// for [`bedrock_rows`].
///
/// The cap admits a request while recorded spend is below it and rejects once spend reaches it.
/// With `c` = 86,070,000 and `C` = 300,000,000, request `k = ceil(C / c)` = 4 is admitted, on a
/// spend of 3c = 258,210,000, and request 5 is rejected on 4c = 344,280,000. At the cache-blind
/// 26,400,000 the same cap would have admitted `ceil(C / 26,400,000)` = 12 requests.
#[tokio::test]
async fn hard_cap_trips_on_the_cache_inclusive_cost() {
    assert_eq!(
        NanoUsd::from_f64_usd(BUDGET_TRIP_HARD_CAP_USD).as_u64(),
        BUDGET_TRIP_HARD_CAP_NANO_USD,
        "precondition: the configured cap is the one the derivation uses"
    );
    let k = BUDGET_TRIP_HARD_CAP_NANO_USD.div_ceil(BUDGET_TRIP_REQUEST_COST);
    let cache_blind_k = BUDGET_TRIP_HARD_CAP_NANO_USD.div_ceil(CACHE_BLIND_REQUEST_COST);
    assert_eq!(
        (k, cache_blind_k),
        (4, 12),
        "the derivation's admitted-request counts"
    );

    let env = Env::start().await;
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(
            Lane::Bedrock.upstream_path(PRIMARY_MODEL, Delivery::Buffered),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            bedrock_buffered_body(heavily_cached_bedrock_usage()),
            "application/json",
        ))
        .mount(&upstream)
        .await;
    let gateway = TestGateway::spawn_with_pricing(
        env.pg.pool.clone(),
        env.redis.pool.clone(),
        Lane::Bedrock
            .adapter(&upstream.uri(), Arc::clone(&env.pricing))
            .await,
        Arc::clone(&env.pricing),
        Some(BudgetConfig {
            soft_cap_usd: None,
            hard_cap_usd: Some(BUDGET_TRIP_HARD_CAP_USD),
            ..BudgetConfig::default()
        }),
    )
    .await;
    let request = serde_json::json!({
        "model": PRIMARY_MODEL,
        "messages": [{"role": "user", "content": "Summarize the cached document."}]
    });

    for i in 1..=k {
        gateway
            .server
            .post(CHAT_COMPLETIONS_PATH)
            .json(&request)
            .await
            .assert_status(StatusCode::OK);
        await_budget_counter(
            &env,
            i * BUDGET_TRIP_REQUEST_COST,
            &format!("request {i} of {k}"),
        )
        .await;
    }

    gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .json(&request)
        .await
        .assert_status(StatusCode::TOO_MANY_REQUESTS);
    let dispatched = upstream
        .received_requests()
        .await
        .expect("wiremock records requests")
        .len();
    assert_eq!(
        dispatched as u64, k,
        "the rejected request must never reach the provider"
    );
    assert_eq!(
        budget_counter(&env).await,
        Some(i64::try_from(k * BUDGET_TRIP_REQUEST_COST).expect("spend fits i64")),
        "a rejected request records no spend"
    );
}
