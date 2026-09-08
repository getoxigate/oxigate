// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Shared utilities extracted from the OpenAI adapter for use by OpenAICompatAdapter.

use futures::StreamExt;
use tracing::debug;

use crate::domain::chat::{
    CacheAccounting, ChatRequest, ReasoningAccounting, Usage, UsageAccounting,
};
use crate::domain::ports::ProviderError;
use crate::domain::usage_accounting::{CacheWriteAccumulator, CacheWriteClass, PricingContext};

/// Maximum bytes read from an upstream error body. Prevents hostile upstreams from
/// forcing large allocations on the error path.
const ERROR_BODY_CAP: usize = 4 * 1024;

// ---------------------------------------------------------------------------
// Accounting declarations
//
// One per provider *contract*, not per wire shape. The three below share the OpenAI response
// schema and hold identical values today, but they are not the same contract: OpenAI's and
// Azure's accounting is documented by their vendors, while a generic OpenAI-compatible backend
// is an arbitrary third party whose counting is unverified. They diverge when the documented
// values are applied, so they are declared separately from the start.
// ---------------------------------------------------------------------------

/// Token accounting declared by the OpenAI chat completions contract.
///
/// Cache is **inclusive**: `developers.openai.com/api/docs/guides/prompt-caching` documents
/// `prompt_tokens` as containing `prompt_tokens_details.cached_tokens` (accessed 2026-08-10).
///
/// Reasoning is **contained in** the completion total:
/// `developers.openai.com/api/docs/guides/reasoning` states that reasoning tokens "still occupy
/// space in the model's context window and are billed as output tokens" (accessed 2026-08-10).
/// `completion_tokens_details.reasoning_tokens` is therefore a breakdown of `completion_tokens`,
/// not an addition to it.
pub(crate) const OPENAI_ACCOUNTING: UsageAccounting = UsageAccounting {
    cache: CacheAccounting::Inclusive,
    reasoning: ReasoningAccounting::IncludedInOutput,
};

/// Token accounting declared by the Azure OpenAI Service contract.
///
/// Cache is **inclusive**:
/// `learn.microsoft.com/en-us/azure/foundry/openai/how-to/prompt-caching` documents that "cache
/// hits show up as `cached_tokens` under `prompt_tokens_details` in the chat completions
/// response", and its worked response carries `prompt_tokens: 1566` against
/// `cached_tokens: 1408` — the cached count is part of the prompt total, not an addition to it
/// (accessed 2026-08-23).
///
/// Reasoning is **contained in** the completion total:
/// `learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning` states that reasoning tokens
/// "occupy space in the context window and are billed as output tokens" and directs callers to
/// `completion_tokens_details.reasoning_tokens`; its worked response carries
/// `completion_tokens: 1843` against `reasoning_tokens: 448` (accessed 2026-08-23).
///
/// Held separately from `OPENAI_ACCOUNTING` because it is a separate vendor contract with its own
/// documentation, not because the values differ.
pub(crate) const AZURE_ACCOUNTING: UsageAccounting = UsageAccounting {
    cache: CacheAccounting::Inclusive,
    reasoning: ReasoningAccounting::IncludedInOutput,
};

/// Token accounting assumed for a generic OpenAI-compatible backend.
///
/// Deliberately the gateway's historical behaviour on both axes, and deliberately **not** copied
/// from `OPENAI_ACCOUNTING`. Speaking the OpenAI wire format proves what fields a backend emits,
/// not how it counts them: a third-party backend may report reasoning beside the completion total
/// rather than inside it, and no first-party reference covers "any backend that accepts this
/// schema".
///
/// The consequence is stated rather than hidden: where a compat backend does count reasoning
/// inside its completion total, that subset is charged twice until a captured payload from that
/// backend justifies a per-instance declaration. Changing this value on the strength of the wire
/// format alone would be an unverified billing change.
pub(crate) const COMPAT_DEFAULT_ACCOUNTING: UsageAccounting = UsageAccounting {
    cache: CacheAccounting::Inclusive,
    reasoning: ReasoningAccounting::Additive,
};

/// The `prompt_tokens_details` member the OpenAI-shaped contracts report cache writes under.
///
/// Retained as the raw evidence key so a persisted evidence document names the wire field the
/// observation came from, exactly as the per-class lanes do.
pub(crate) const OPENAI_CACHE_WRITE_KEY: &str = "cache_write_tokens";

/// The only cache-write TTL the OpenAI and Azure contracts support.
///
/// `developers.openai.com/api/docs/guides/prompt-caching`: "The only supported value, `30m`, is
/// also the default." Microsoft documents the same single value for Azure OpenAI. A closed
/// one-value vocabulary is what licenses crediting the reported aggregate to a class at all —
/// there is no other duration the quantity could belong to.
pub(crate) const OPENAI_CACHE_WRITE_CLASS: &str = "30m";

/// Normalizes an OpenAI-shaped `Usage` and stamps the caller's accounting declaration onto it.
///
/// Maps `prompt_tokens_details.cached_tokens` → `cache_read_input_tokens`, because the domain
/// `Usage` model prices from the latter.
///
/// `accounting` is a **required parameter, never inferred here**. Several provider contracts
/// share this response schema and this function, and the schema does not identify which one
/// produced the payload. Each caller passes its own constant; a caller that guessed from the
/// wire shape would give every contract the same answer, which is the defect this parameter
/// exists to prevent.
///
/// `pricing_context` is the lane's declared position on cache-write accounting, not a default a
/// forgetful caller can inherit. A lane that supplies one has
/// `prompt_tokens_details.cache_write_tokens` credited as a [`OPENAI_CACHE_WRITE_CLASS`]
/// observation and published on `cache_creation_input_tokens`; a lane that supplies `None` — a
/// generic OpenAI-compatible backend, whose cache-write semantics no first-party reference
/// covers — leaves cost, status, spend and budget exactly as they were. The `Option` therefore
/// separates *different* lanes rather than giving one lane two behaviours.
pub fn normalize_openai_usage(
    usage: &mut Usage,
    accounting: UsageAccounting,
    pricing_context: Option<&PricingContext>,
) {
    if let Some(ref d) = usage.prompt_tokens_details
        && d.cached_tokens.is_some()
    {
        usage.cache_read_input_tokens = d.cached_tokens;
    }

    if let Some(context) = pricing_context {
        // The generation is pinned whether or not this response wrote to cache: the lane
        // snapshotted it before dispatch, and pricing must not drift to a newer one because a
        // reload landed while the request was in flight.
        let mut accumulator = CacheWriteAccumulator::new(context.registry().clone());
        if let Some(written) = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cache_write_tokens)
        {
            // One field, reported as the total written and — because the contract supports
            // exactly one TTL — wholly of that class. Recording it as both the aggregate and the
            // single detail is not double counting: `accounted_tokens` is the maximum of the two
            // views, never their sum, and the two agreeing is what makes the partition exact.
            accumulator.set_reported_aggregate(written);
            accumulator.observe_detail(
                OPENAI_CACHE_WRITE_KEY,
                CacheWriteClass::canonicalize(OPENAI_CACHE_WRITE_CLASS),
                written,
            );
        }
        let mut cache_write = accumulator.finish();
        cache_write.set_pricing_context(context.clone());
        // A declared lane republishes what it accounted, so the public field cannot disagree with
        // the spend row and the budget counter. The assignment is unconditional in both
        // directions: `cache_creation_input_tokens` deserializes from the wire like any other
        // member, so an OpenAI-shaped upstream that emits it hands over a quantity this
        // accumulator never saw, and leaving that standing would publish a number nothing priced.
        usage.cache_creation_input_tokens = cache_write.published_tokens();
        usage.cache_write = cache_write;
    }

    usage.accounting = accounting;
}

/// Injects `stream_options.include_usage: true` unless the client already set it to any value.
///
/// Without this injection OpenAI (and Azure) emit NO usage data in any streaming chunk
/// (`usage: null` on every chunk). The injection is mandatory for any cost tracking.
///
/// Cases:
/// - `Some(true)` → already set; log debug, no-op.
/// - `Some(false)` → client opted out; log debug, no-op (client value wins).
/// - `None` → inject `true`.
pub fn inject_stream_options(req: &mut ChatRequest) {
    let existing = req
        .extra
        .get("stream_options")
        .and_then(|o| o.get("include_usage"))
        .and_then(|v| v.as_bool());
    match existing {
        Some(true) => {
            debug!("stream_options.include_usage already true; cost tracking will be precise");
        }
        Some(false) => {
            debug!(
                "stream_options.include_usage=false from client; cost tracking will be imprecise for this request"
            );
        }
        None => {
            let mut opts = req
                .extra
                .get("stream_options")
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_default();
            opts.insert("include_usage".into(), serde_json::json!(true));
            req.extra.insert("stream_options".into(), opts.into());
        }
    }
}

/// Maps an HTTP status code + pre-extracted message to a [`ProviderError`].
///
/// Shared primitive used by [`map_openai_error_response`] and `azure::map_error_response`.
/// Callers are responsible for reading the response body, extracting the message string,
/// and reading the `Retry-After` header before calling this function.
pub(crate) fn map_status_to_provider_error(
    status: reqwest::StatusCode,
    msg: String,
    retry_after: Option<u64>,
) -> ProviderError {
    match status.as_u16() {
        400 => ProviderError::InvalidRequest(msg),
        401 => ProviderError::Auth(msg),
        403 => ProviderError::Auth(format!("forbidden: {msg}")),
        404 => ProviderError::UnknownModel(msg),
        429 => ProviderError::RateLimited { retry_after },
        500 | 502 | 503 => ProviderError::ProviderUnavailable(msg),
        _ => ProviderError::ProviderHttpError {
            status: status.as_u16(),
            body: msg,
        },
    }
}

/// Shared error-response mapper for the OpenAI adapter family (OpenAI, compat).
///
/// Reads the `Retry-After` header and body (bounded at `ERROR_BODY_CAP`), extracts
/// `error.message` from the JSON body, then delegates to [`map_status_to_provider_error`].
///
/// Azure uses its own `map_error_response` to perform content-filter detection before
/// delegating to [`map_status_to_provider_error`] for the generic status table.
pub async fn map_openai_error_response(
    status: reqwest::StatusCode,
    resp: reqwest::Response,
) -> ProviderError {
    let retry_after = resp
        .headers()
        .get("Retry-After")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    // Bounded read: stop after ERROR_BODY_CAP bytes so a hostile upstream cannot force
    // a large heap allocation by sending a multi-MB error body.
    let mut body_bytes: Vec<u8> = Vec::with_capacity(ERROR_BODY_CAP);
    let mut stream = resp.bytes_stream();
    while let Some(Ok(chunk)) = stream.next().await {
        let remaining = ERROR_BODY_CAP.saturating_sub(body_bytes.len());
        if remaining == 0 {
            break;
        }
        body_bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    let msg = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        .ok()
        .and_then(|j| {
            j.get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(String::from)
        })
        .unwrap_or_else(|| String::from_utf8_lossy(&body_bytes).into_owned());

    map_status_to_provider_error(status, msg, retry_after)
}

/// One cache-write fixture and one exact-cost oracle, shared by the adapter-seam tests.
///
/// Every OpenAI-shaped lane is asserted against the *same* payload — that is the point of the
/// tests that use it: OpenAI and Azure must account and price it identically, and a generic
/// compat instance must not account it at all. Holding the fixture in one place is what makes
/// "identically" and "not at all" statements about one number rather than about three
/// independently maintained copies.
#[cfg(test)]
pub(crate) mod cache_write_fixture {
    use crate::domain::chat::Usage;

    /// The bundled entry these tests price against, and the only tier they reach.
    ///
    /// `gpt-5.6-sol` tier 0: input `5e-6`/token, output `3e-5`/token, cache read `0.1x`,
    /// cache write `30m` `1.25x`.
    pub(crate) const MODEL: &str = "gpt-5.6-sol";

    /// The hand-computed charge for [`usage_json`] against [`MODEL`] tier 0, in nano-USD.
    ///
    /// Rates in nano-USD per token: input 5,000; output 30,000; cache read 5,000 × 0.1 = 500;
    /// cache write 5,000 × 1.25 = 6,250.
    ///
    /// | Component | Tokens | Rate | Cost |
    /// |---|---|---|---|
    /// | plain input (10,000 − 2,000 − 1,000) | 7,000 | 5,000 | 35,000,000 |
    /// | cache read | 2,000 | 500 | 1,000,000 |
    /// | cache write `30m` | 1,000 | 6,250 | 6,250,000 |
    /// | output | 500 | 30,000 | 15,000,000 |
    ///
    /// A lane that leaves the written tokens in the plain-input bucket at `1.0x` charges
    /// [`UNACCOUNTED_ORACLE_NANO_USD`]; one that credits the class without carving it back out of
    /// that bucket charges 62,250,000. Only crediting *and* carving out yields this number.
    pub(crate) const ORACLE_NANO_USD: u64 = 57_250_000;

    /// The charge for the same payload on a lane that accounts no cache write.
    ///
    /// The written tokens stay inside `prompt_tokens` and bill at the plain input rate:
    /// 8,000 × 5,000 + 1,000,000 + 15,000,000. This is the compat lane's declared position, and
    /// it is also what every OpenAI-shaped lane charged before the pricing context was supplied.
    pub(crate) const UNACCOUNTED_ORACLE_NANO_USD: u64 = 56_000_000;

    /// The usage block on the OpenAI wire shape.
    ///
    /// 10,000 prompt tokens of which 2,000 were read from cache and 1,000 written to it — an
    /// `Inclusive` contract, so all three counts live inside `prompt_tokens`.
    pub(crate) fn usage_json() -> serde_json::Value {
        serde_json::json!({
            "prompt_tokens": 10_000,
            "completion_tokens": 500,
            "total_tokens": 10_500,
            "prompt_tokens_details": {
                "cached_tokens": 2_000,
                "cache_write_tokens": 1_000
            }
        })
    }

    /// A hot-reload holder over the bundled snapshot, for adapters and for pricing.
    pub(crate) fn pricing_holder()
    -> std::sync::Arc<std::sync::RwLock<crate::domain::pricing::PricingDb>> {
        std::sync::Arc::new(std::sync::RwLock::new(
            crate::domain::pricing::PricingDb::load(
                crate::domain::pricing::BUNDLED_PRICING_JSON,
                &crate::config::PricingConfig::default(),
            )
            .expect("bundled pricing must load"),
        ))
    }

    /// Prices a lane's normalized usage the way the request path does.
    fn finalize(usage: &Usage) -> crate::domain::usage_accounting::FinalizedAccounting {
        crate::utils::cost_headers::build_cost_headers(MODEL, usage, pricing_holder(), false).1
    }

    /// Asserts a declared lane accounted the write as class `30m` and billed it at 1.25x net.
    pub(crate) fn assert_accounted_and_billed(usage: &Usage) {
        assert_eq!(
            usage.cache_creation_input_tokens,
            Some(1_000),
            "the adapter publishes the accounted cache-write quantity"
        );
        let classes = usage.cache_write.class_totals();
        assert_eq!(classes.len(), 1, "one observation, one class");
        assert_eq!(classes[0].class.as_str(), "30m");
        assert_eq!(classes[0].tokens, 1_000);
        assert_eq!(
            usage.cache_write.fallback_tokens(),
            0,
            "the class is configured on this tier; nothing prices at the fallback rate"
        );
        assert!(
            usage.cache_write.pricing_context().is_some(),
            "the adapter pinned the generation it accounted against"
        );

        let finalized = finalize(usage);
        assert_eq!(
            finalized.cost.total_cost.0, ORACLE_NANO_USD,
            "cache writes bill at 1.25x net of the input charge, not on top of it"
        );
        assert_eq!(finalized.cost.cache_write_cost.0, 6_250_000);
        assert_eq!(
            finalized.cost.status,
            crate::domain::usage_accounting::CostStatus::Exact
        );
    }

    /// Asserts an undeclared lane accounted nothing and its money did not move.
    ///
    /// The echoed wire field is asserted by the caller, which is the half that *does* change.
    pub(crate) fn assert_unaccounted_and_unbilled(usage: &Usage) {
        assert_eq!(
            usage.cache_creation_input_tokens, None,
            "an undeclared lane publishes no accounted quantity"
        );
        assert_eq!(usage.cache_write.observation_count(), 0);
        assert!(usage.cache_write.pricing_context().is_none());

        let finalized = finalize(usage);
        assert_eq!(
            finalized.cost.total_cost.0, UNACCOUNTED_ORACLE_NANO_USD,
            "the written tokens stay in the plain-input bucket, exactly as before"
        );
        assert_eq!(finalized.cost.cache_write_cost.0, 0);
        assert_eq!(
            finalized.cost.status,
            crate::domain::usage_accounting::CostStatus::Exact
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::{ChatRequest, Message, MessageContent, Role};
    use crate::domain::usage_accounting::CacheWriteClassRegistry;

    fn minimal_request() -> ChatRequest {
        ChatRequest {
            model: "gpt-4o".into(),
            messages: vec![Message {
                role: Role::User,
                content: Some(MessageContent::Text("hi".into())),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn normalize_usage_maps_cached_tokens() {
        let mut usage = Usage {
            prompt_tokens: 100,
            prompt_tokens_details: Some(crate::domain::chat::PromptTokensDetails {
                cached_tokens: Some(40),
                cache_write_tokens: None,
            }),
            cache_read_input_tokens: None,
            ..Default::default()
        };
        normalize_openai_usage(&mut usage, OPENAI_ACCOUNTING, None);
        assert_eq!(usage.cache_read_input_tokens, Some(40));
        assert_eq!(usage.accounting, OPENAI_ACCOUNTING);
    }

    #[test]
    fn normalize_usage_noop_when_no_details() {
        let mut usage = Usage {
            prompt_tokens: 100,
            prompt_tokens_details: None,
            cache_read_input_tokens: None,
            ..Default::default()
        };
        normalize_openai_usage(&mut usage, OPENAI_ACCOUNTING, None);
        assert_eq!(usage.cache_read_input_tokens, None);
        assert_eq!(usage.accounting, OPENAI_ACCOUNTING);
    }

    /// The accounting is stamped from the parameter and never inferred from the payload. The
    /// value used here is held by none of the three constants, so a re-introduced inference
    /// cannot coincidentally satisfy it.
    #[test]
    fn normalize_usage_stamps_exactly_the_accounting_it_is_given() {
        let caller_declared = UsageAccounting {
            cache: CacheAccounting::Additive,
            reasoning: ReasoningAccounting::IncludedInOutput,
        };
        let mut usage = Usage {
            prompt_tokens: 100,
            ..Default::default()
        };
        normalize_openai_usage(&mut usage, caller_declared, None);
        assert_eq!(usage.accounting, caller_declared);
    }

    /// The three contracts sharing this wire shape do **not** share an accounting contract.
    ///
    /// OpenAI and Azure each document reasoning as part of the completion total; a generic compat
    /// backend documents nothing, so it keeps the gateway's prior behaviour. A change that
    /// collapsed the three into one constant would have to break this.
    #[test]
    fn openai_shape_contracts_do_not_share_one_declaration() {
        assert_eq!(
            OPENAI_ACCOUNTING.reasoning,
            ReasoningAccounting::IncludedInOutput
        );
        assert_eq!(
            AZURE_ACCOUNTING.reasoning,
            ReasoningAccounting::IncludedInOutput
        );
        assert_eq!(
            COMPAT_DEFAULT_ACCOUNTING.reasoning,
            ReasoningAccounting::Additive,
            "a generic compat backend's counting is unverified; it must not inherit OpenAI's"
        );
        assert_ne!(AZURE_ACCOUNTING, COMPAT_DEFAULT_ACCOUNTING);

        // The cache axis is documented identically for all three.
        for accounting in [
            OPENAI_ACCOUNTING,
            AZURE_ACCOUNTING,
            COMPAT_DEFAULT_ACCOUNTING,
        ] {
            assert_eq!(accounting.cache, CacheAccounting::Inclusive);
        }
    }

    /// The model `priced_at` writes, used to tell one pricing generation from another.
    const RATE_PROBE_MODEL: &str = "rate-probe";

    /// A pricing generation identifiable by its rate alone.
    ///
    /// Every generation this builds configures the same single `30m` class, so the class registry
    /// is constant across them and only `input_per_token` moves. That is what lets a test
    /// distinguish two generations the registry cannot.
    fn priced_at(input_per_token: f64) -> PricingContext {
        let json = format!(
            r#"{{"models":{{"{RATE_PROBE_MODEL}":{{"provider":"test","context_window":1000,
               "aliases":[],"tiers":[{{"threshold":0,"input_per_token":{input_per_token},
               "output_per_token":0.00003,"cache_write_multipliers":{{"30m":1.25}}}}]}}}}}}"#
        );
        let db = crate::domain::pricing::PricingDb::load(
            json.as_bytes(),
            &crate::config::PricingConfig::default(),
        )
        .expect("probe pricing must load");
        let registry = db.registry().clone();
        PricingContext::new(db, registry)
    }

    /// A pricing context whose registry configures exactly `classes`, over the bundled snapshot.
    ///
    /// The registry is built independently of the snapshot so a test can state which classes are
    /// configured rather than inherit whatever the bundled data happens to define.
    fn context_configuring(classes: &[&str]) -> PricingContext {
        let db = crate::domain::pricing::PricingDb::load(
            crate::domain::pricing::BUNDLED_PRICING_JSON,
            &crate::config::PricingConfig::default(),
        )
        .expect("bundled pricing must load");
        let registry = CacheWriteClassRegistry::from_classes(
            classes
                .iter()
                .filter_map(|c| CacheWriteClass::canonicalize(c)),
        )
        .expect("registry must build");
        PricingContext::new(db, registry)
    }

    fn usage_with_cache_write(written: Option<u64>) -> Usage {
        Usage {
            prompt_tokens: 10_000,
            completion_tokens: 500,
            total_tokens: 10_500,
            prompt_tokens_details: Some(crate::domain::chat::PromptTokensDetails {
                cached_tokens: Some(2_000),
                cache_write_tokens: written,
            }),
            ..Default::default()
        }
    }

    /// The OpenAI contract publishes one cache-write quantity under one documented TTL, so a lane
    /// that supplies its pricing generation gets an exactly-classified `30m` observation.
    #[test]
    fn cache_write_tokens_are_credited_to_the_thirty_minute_class() {
        let context = context_configuring(&["30m", "5m", "1h"]);
        let mut usage = usage_with_cache_write(Some(1_000));

        normalize_openai_usage(&mut usage, OPENAI_ACCOUNTING, Some(&context));

        assert_eq!(
            usage.cache_creation_input_tokens,
            Some(1_000),
            "the accounted quantity is published on the public usage field"
        );
        let classes = usage.cache_write.class_totals();
        assert_eq!(classes.len(), 1, "one observation, one class");
        assert_eq!(
            classes[0].class,
            CacheWriteClass::canonicalize(OPENAI_CACHE_WRITE_CLASS).expect("30m canonicalizes")
        );
        assert_eq!(classes[0].tokens, 1_000);
        assert_eq!(
            usage.cache_write.unknown_tokens(),
            0,
            "a configured class never reaches the unknown bucket"
        );
        assert_eq!(
            usage.cache_write.fallback_tokens(),
            0,
            "nothing is priced at the fallback rate"
        );
        assert!(usage.cache_write.partition_is_exact());
        assert!(
            usage.cache_write.pricing_context().is_some(),
            "the request is pinned to the generation it was accounted under"
        );

        let evidence = usage.cache_write.evidence_entries();
        assert_eq!(evidence.len(), 1);
        assert_eq!(
            evidence[0].raw_key, OPENAI_CACHE_WRITE_KEY,
            "the evidence names the wire field the observation came from"
        );
    }

    /// On a declared lane the normalized field is authoritative, in both directions.
    ///
    /// `Usage.cache_creation_input_tokens` deserializes from the wire like any other member, so an
    /// OpenAI-shaped upstream that emits it — a proxy, a relabelled backend, a future field — hands
    /// the gateway a quantity this accumulator never saw. Publishing it unchanged would put a
    /// number on the response that nothing priced, persisted or counted against a budget, while
    /// `usage.cache_write` said zero. Clearing the field when nothing was reported is as
    /// load-bearing as setting it when something was.
    #[test]
    fn a_declared_lane_overwrites_an_unaccounted_upstream_quantity() {
        let context = context_configuring(&["30m"]);

        let mut replaced = usage_with_cache_write(Some(1_000));
        replaced.cache_creation_input_tokens = Some(9_999);
        normalize_openai_usage(&mut replaced, OPENAI_ACCOUNTING, Some(&context));
        assert_eq!(
            replaced.cache_creation_input_tokens,
            Some(1_000),
            "the accounted quantity replaces whatever the upstream put there"
        );

        let mut cleared = usage_with_cache_write(None);
        cleared.cache_creation_input_tokens = Some(9_999);
        normalize_openai_usage(&mut cleared, OPENAI_ACCOUNTING, Some(&context));
        assert_eq!(
            cleared.cache_creation_input_tokens, None,
            "a stale quantity the lane did not account must not survive on the response"
        );
        assert_eq!(cleared.cache_write.accounted_tokens(), 0);
    }

    /// A lane that supplies no pricing generation must not account the field at all — that is the
    /// compat lane's declared position, not an oversight.
    #[test]
    fn cache_write_tokens_are_not_accounted_without_a_pricing_context() {
        let mut usage = usage_with_cache_write(Some(1_000));

        normalize_openai_usage(&mut usage, COMPAT_DEFAULT_ACCOUNTING, None);

        assert_eq!(usage.cache_creation_input_tokens, None);
        assert_eq!(usage.cache_write.observation_count(), 0);
        assert!(usage.cache_write.pricing_context().is_none());

        // Passthrough cuts both ways: an undeclared lane does not rewrite the public field
        // either, because it has accounted nothing to put there.
        let mut passthrough = usage_with_cache_write(Some(1_000));
        passthrough.cache_creation_input_tokens = Some(9_999);
        normalize_openai_usage(&mut passthrough, COMPAT_DEFAULT_ACCOUNTING, None);
        assert_eq!(passthrough.cache_creation_input_tokens, Some(9_999));

        assert_eq!(
            usage.cache_read_input_tokens,
            Some(2_000),
            "the cache-read mapping is unaffected"
        );
    }

    /// A generation that does not configure `30m` must send the quantity to the unknown bucket
    /// rather than price it at a class it never defined.
    #[test]
    fn an_unconfigured_thirty_minute_class_falls_back_rather_than_guessing() {
        let context = context_configuring(&["5m", "1h"]);
        let mut usage = usage_with_cache_write(Some(1_000));

        normalize_openai_usage(&mut usage, OPENAI_ACCOUNTING, Some(&context));

        assert_eq!(usage.cache_creation_input_tokens, Some(1_000));
        assert!(usage.cache_write.class_totals().is_empty());
        assert_eq!(usage.cache_write.unknown_tokens(), 1_000);
        assert_eq!(usage.cache_write.fallback_tokens(), 1_000);
    }

    /// Azure Provisioned (PTU-M) deployments do not expose `cache_write_tokens` at all. The field
    /// is simply absent, and an absent field is not a zero: nothing is published.
    ///
    /// The pricing generation is pinned anyway. A declared lane snapshots it before dispatch, and
    /// `BundledCostCalculator::calculate` prices against the carried snapshot when there is one
    /// and re-reads the live holder when there is not — so a request that happened not to write to
    /// cache would otherwise still be exposed to a reload landing mid-flight. This assertion is
    /// what makes that unconditional pin a tested decision rather than an incidental one.
    #[test]
    fn an_absent_cache_write_field_publishes_no_quantity_but_still_pins_the_generation() {
        let context = context_configuring(&["30m"]);
        let mut usage = usage_with_cache_write(None);

        normalize_openai_usage(&mut usage, AZURE_ACCOUNTING, Some(&context));

        assert_eq!(usage.cache_creation_input_tokens, None);
        assert_eq!(usage.cache_write.observation_count(), 0);

        let pinned = usage
            .cache_write
            .pricing_context()
            .expect("the generation is pinned even with nothing written");
        assert_eq!(pinned.registry().classes(), context.registry().classes());

        // The registry alone cannot prove this. Two generations can configure identical classes
        // and still price differently — that is precisely the reload the pin exists to survive —
        // so the carried *database* is what has to be identified. `priced_at` holds the class set
        // fixed and moves only the rate, so a pin that carried the wrong generation, or that
        // re-read a live holder, cannot satisfy both assertions.
        let supplied = priced_at(0.000_007);
        let mut usage = usage_with_cache_write(None);
        normalize_openai_usage(&mut usage, AZURE_ACCOUNTING, Some(&supplied));

        let pinned = usage
            .cache_write
            .pricing_context()
            .expect("pinned")
            .db()
            .read();
        assert_eq!(
            pinned
                .lookup(RATE_PROBE_MODEL, None)
                .expect("the probe model resolves in the pinned generation")
                .tiers[0]
                .input_per_token,
            0.000_007,
            "the pinned database is the generation the lane supplied, rate and all"
        );
    }

    /// A provider that reports a zero *is* saying something, and the distinction survives: Azure's
    /// own worked example carries `cache_write_tokens: 0`.
    #[test]
    fn a_reported_zero_cache_write_is_published_as_zero() {
        let context = context_configuring(&["30m"]);
        let mut usage = usage_with_cache_write(Some(0));

        normalize_openai_usage(&mut usage, AZURE_ACCOUNTING, Some(&context));

        assert_eq!(
            usage.cache_creation_input_tokens,
            Some(0),
            "`None` and `Some(0)` are different statements"
        );
        assert_eq!(usage.cache_write.fallback_tokens(), 0);
    }

    #[test]
    fn inject_stream_options_adds_include_usage_when_absent() {
        let mut req = minimal_request();
        inject_stream_options(&mut req);
        let v = req
            .extra
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(|v| v.as_bool());
        assert_eq!(v, Some(true));
    }

    #[test]
    fn inject_stream_options_noop_when_already_true() {
        let mut req = minimal_request();
        req.extra.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": true}),
        );
        inject_stream_options(&mut req);
        let v = req
            .extra
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(|v| v.as_bool());
        assert_eq!(v, Some(true));
    }

    #[test]
    fn inject_stream_options_respects_client_false() {
        let mut req = minimal_request();
        req.extra.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": false}),
        );
        inject_stream_options(&mut req);
        let v = req
            .extra
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(|v| v.as_bool());
        // Client opted out — must be preserved as false, not overridden.
        assert_eq!(v, Some(false));
    }
}
