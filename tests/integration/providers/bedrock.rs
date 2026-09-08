// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Bedrock Converse adapter integration tests .
//!
//! Uses wiremock to mock the Bedrock endpoint. Tests verify:
//! - Request/response translation (non-streaming and streaming)
//! - SigV4 headers are present on every upstream request
//! - Error mapping (ThrottlingException → RateLimited)
//! - Config validation (missing region fails fast)

use crate::common::bundled_pricing_holder;
use futures::StreamExt;
use oxigate::config::{BedrockConfig, SecretString};
use oxigate::domain::chat::{ChatRequest, Message, MessageContent, Role, Tool, ToolFunction};
use oxigate::domain::ports::{ProviderAdapter, ProviderError};
use oxigate::providers::BedrockAdapter;
use oxigate::providers::bedrock::eventstream::build_frame;
use wiremock::matchers::{header_exists, method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn bedrock_config(mock_uri: &str) -> BedrockConfig {
    BedrockConfig {
        region: "us-east-1".to_string(),
        access_key_id: Some(SecretString::from("AKIDEXAMPLE")),
        secret_access_key: Some(SecretString::from(
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        )),
        session_token: None,
        endpoint_url: Some(mock_uri.trim_end_matches('/').to_string()),
        default_model: Some("anthropic.claude-3-5-sonnet-20241022-v2:0".to_string()),
        timeout_secs: Some(10),
        supported_models: None,
    }
}

fn user_request(model: &str, text: &str) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        max_completion_tokens: None,
        stream: None,
        tools: None,
        parallel_tool_calls: None,
        request_id: Some("test-req-001".to_string()),
        extra: serde_json::Map::new(),
    }
}

fn tool_request(model: &str, tool_name: &str) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("What is the weather?".to_string())),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        max_completion_tokens: None,
        stream: None,
        tools: Some(vec![Tool {
            type_: "function".to_string(),
            function: ToolFunction {
                name: tool_name.to_string(),
                description: Some("Get weather".to_string()),
                parameters: Some(serde_json::json!({"type":"object","properties":{}})),
            },
        }]),
        parallel_tool_calls: None,
        request_id: Some("test-tool-bedrock".to_string()),
        extra: serde_json::Map::new(),
    }
}

fn converse_response_body(text: &str, input_tokens: u64, output_tokens: u64) -> serde_json::Value {
    serde_json::json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}]
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens
        }
    })
}

/// Builds a minimal EventStream byte sequence with the given frames.
fn event_stream_bytes(frames: &[(&str, serde_json::Value)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (event_type, payload) in frames {
        let payload_bytes = serde_json::to_vec(payload).unwrap();
        bytes.extend_from_slice(&build_frame(event_type, &payload_bytes));
    }
    bytes
}

#[tokio::test]
async fn test_bedrock_chat_non_streaming() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse$"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(converse_response_body(
                "Hello from Bedrock",
                10,
                5,
            )),
        )
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()), bundled_pricing_holder())
        .await
        .expect("adapter must build");

    let req = user_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "Say hi");
    let resp = adapter.chat_completion(&req).await.unwrap();

    assert_eq!(resp.choices.len(), 1);
    assert_eq!(resp.choices[0].message.role, Role::Assistant);
    if let Some(MessageContent::Text(t)) = &resp.choices[0].message.content {
        assert_eq!(t, "Hello from Bedrock");
    } else {
        panic!("expected text content");
    }
    assert_eq!(resp.usage.prompt_tokens, 10);
    assert_eq!(resp.usage.completion_tokens, 5);
    assert_eq!(resp.usage.total_tokens, 15);
}

/// A pricing holder distinguishable from the bundled one by its rates alone.
///
/// Same two cache-write classes and the same multipliers, so the class registry is identical and
/// cannot be what a test keys on; both base rates exactly doubled, so *which generation was
/// pinned* is visible in the money and the expected total is the bundled oracle times two. An
/// adapter that ignored the holder it was handed and re-read the live database would price this
/// request at the bundled rate instead.
fn probe_pricing_holder() -> std::sync::Arc<std::sync::RwLock<oxigate::domain::pricing::PricingDb>>
{
    use oxigate::config::{PricingConfig, PricingOverride};

    let mut overrides = std::collections::HashMap::new();
    overrides.insert(
        "anthropic.claude-sonnet-4-6".to_string(),
        PricingOverride {
            input_per_token: 0.000_006,
            output_per_token: 0.000_030,
            context_window: 1_000_000,
            cache_read_multiplier: Some(0.1),
            cache_write_multipliers: std::collections::HashMap::from([
                ("5m".to_string(), 1.25),
                ("1h".to_string(), 2.0),
            ]),
        },
    );
    let db = oxigate::domain::pricing::PricingDb::load(
        oxigate::domain::pricing::BUNDLED_PRICING_JSON,
        &PricingConfig { overrides },
    )
    .expect("probe pricing DB loads");
    std::sync::Arc::new(std::sync::RwLock::new(db))
}

/// The buffered adapter accounts what the wire reported, against the generation it was handed.
///
/// Asserted at the adapter seam rather than on the translation helper: a helper test cannot tell
/// a lane that passes a pricing context from one that was never wired, and an unpinned generation
/// is what silently sends every cache-write class to the fallback rate.
///
/// The adapter is built with [`probe_pricing_holder`] and the result is then finalized against
/// the **bundled** holder. If the adapter pinned the generation it was given, the cost comes out
/// at the probe's doubled input rate; if it re-read the live database instead, it comes out at
/// the bundled rate. Merely asserting that *some* context was pinned would pass either way.
#[tokio::test]
async fn test_bedrock_buffered_accounts_cache_writes_from_the_adapter() {
    let mock = MockServer::start().await;
    let body = serde_json::json!({
        "output": {"message": {"role": "assistant", "content": [{"text": "cached"}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10_000,
            "outputTokens": 500,
            "totalTokens": 10_500,
            "cacheReadInputTokens": 2_000,
            "cacheWriteInputTokens": 1_500,
            "cacheDetails": [
                {"ttl": "5m", "inputTokens": 1_000},
                {"ttl": "1h", "inputTokens": 500}
            ]
        }
    });
    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()), probe_pricing_holder())
        .await
        .expect("adapter must build");

    let req = user_request("anthropic.claude-sonnet-4-6", "Say hi");
    let resp = adapter.chat_completion(&req).await.unwrap();

    assert_eq!(resp.usage.cache_read_input_tokens, Some(2_000));
    assert_eq!(resp.usage.cache_creation_input_tokens, Some(1_500));
    // Registry order, not arrival order, so the pairs are sorted before comparison.
    let mut credited: Vec<(&str, u64)> = resp
        .usage
        .cache_write
        .class_totals()
        .iter()
        .map(|t| (t.class.as_str(), t.tokens))
        .collect();
    credited.sort_unstable();
    assert_eq!(credited, vec![("1h", 500), ("5m", 1_000)]);
    assert_eq!(resp.usage.cache_write.fallback_tokens(), 0);

    // Finalize against the *bundled* holder. The pinned generation must win, so the cost lands at
    // the probe's rates — every component exactly double the bundled oracle's 44,850,000.
    let (_, finalized) = oxigate::utils::cost_headers::build_cost_headers(
        "anthropic.claude-sonnet-4-6",
        &resp.usage,
        bundled_pricing_holder(),
        false,
    );
    assert_eq!(
        finalized.cost.total_cost.0, 89_700_000,
        "the adapter must price against the holder it was constructed with, not the live one"
    );
    assert_eq!(
        finalized.cost.status,
        oxigate::domain::usage_accounting::CostStatus::Exact
    );
}

/// The streaming adapter accounts what the terminal `metadata` frame reported, against the
/// generation it was handed — the streaming half of
/// [`test_bedrock_buffered_accounts_cache_writes_from_the_adapter`]'s coverage.
///
/// Same reasoning, same probe holder: a translation-helper test cannot show that
/// `chat_completion_stream` actually snapshots a context and passes it to
/// `converse_cache_write`, only that the helper works when handed one.
///
/// The fixture reports the aggregate with **no** `cacheDetails` breakdown, deliberately — a
/// fixture where the aggregate equals the sum of the details cannot tell "the adapter forwarded
/// `cacheWriteInputTokens`" apart from "the adapter dropped it and the details alone produced the
/// same total". With the details absent, the published quantity, `reported_tokens()`,
/// `accounted_tokens()`, `unmatched_residual_tokens()` / `fallback_tokens()`, and the finalized
/// total cost and status all only reach their asserted values if the aggregate specifically
/// reaches `converse_cache_write` — a dropped aggregate leaves them at `None`/`0`/`Exact` instead.
/// `class_totals()` and `outcome()` do **not** discriminate this: an aggregate with zero detail
/// observations credits no class and reports `Consistent` either way (nothing to reconcile
/// against), and the plain input/output/cache-read components still price regardless, so the
/// total cost is never zero.
#[tokio::test]
async fn test_bedrock_streaming_accounts_cache_writes_from_the_adapter() {
    let mock = MockServer::start().await;
    let frames: &[(&str, serde_json::Value)] = &[
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex": 0, "delta": {"text": "cached"}}),
        ),
        ("messageStop", serde_json::json!({"stopReason": "end_turn"})),
        (
            "metadata",
            serde_json::json!({
                "usage": {
                    "inputTokens": 10_000,
                    "outputTokens": 500,
                    "totalTokens": 10_500,
                    "cacheReadInputTokens": 2_000,
                    "cacheWriteInputTokens": 1_500
                }
            }),
        ),
    ];
    let stream_bytes = event_stream_bytes(frames);

    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse-stream$"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(stream_bytes)
                .append_header("content-type", "application/vnd.amazon.eventstream"),
        )
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()), probe_pricing_holder())
        .await
        .expect("adapter must build");

    let mut req = user_request("anthropic.claude-sonnet-4-6", "Say hi");
    req.stream = Some(true);

    let stream = adapter.chat_completion_stream(&req).await.unwrap();
    let chunks: Vec<_> = stream.collect().await;
    let usage = chunks
        .iter()
        .filter_map(|c| c.as_ref().ok())
        .find_map(|c| c.usage.as_ref())
        .expect("metadata frame must produce a usage chunk");

    assert_eq!(usage.cache_read_input_tokens, Some(2_000));
    assert_eq!(
        usage.cache_creation_input_tokens,
        Some(1_500),
        "the aggregate must be published even though no per-class detail credited it"
    );
    assert_eq!(
        usage.cache_write.reported_tokens(),
        Some(1_500),
        "the reported aggregate must reach the accumulator"
    );
    assert_eq!(
        usage.cache_write.detail_tokens(),
        0,
        "no cacheDetails means no detail-view tokens"
    );
    assert_eq!(usage.cache_write.accounted_tokens(), 1_500);
    assert!(
        usage.cache_write.class_totals().is_empty(),
        "an aggregate with no detail breakdown must not be credited to a class"
    );
    assert_eq!(
        usage.cache_write.unmatched_residual_tokens(),
        1_500,
        "the whole aggregate stands as a residual with nothing to reconcile it against"
    );
    assert_eq!(usage.cache_write.fallback_tokens(), 1_500);
    assert_eq!(
        usage.cache_write.outcome(),
        oxigate::domain::usage_accounting::ReconciliationOutcome::Consistent,
        "with zero detail observations there is nothing for the aggregate to disagree with — \
         it stands as an unmatched residual rather than a reconciliation conflict"
    );

    // Finalize against the *bundled* holder, mirroring the buffered test. The pinned generation
    // must win, so the cost lands at the probe's doubled rates: plain input 10,000 * 6,000 +
    // output 500 * 30,000 + cache read 2,000 * 600 + the 1,500-token residual at the tier
    // fallback rate (max configured multiplier, 2.0x) * 12,000 = 94,200,000.
    let (_, finalized) = oxigate::utils::cost_headers::build_cost_headers(
        "anthropic.claude-sonnet-4-6",
        usage,
        bundled_pricing_holder(),
        false,
    );
    assert_eq!(
        finalized.cost.total_cost.0, 94_200_000,
        "the streaming adapter must price against the holder it was constructed with, not the \
         live one"
    );
    assert_eq!(
        finalized.cost.status,
        oxigate::domain::usage_accounting::CostStatus::RateFallback,
        "a residual with no established rate must not report exact"
    );
}

/// Buffered and streaming must produce numerically identical `Usage` from the same reported
/// cache-write payload. Both paths funnel through the same `converse_cache_write`, but only a
/// test driving both from one fixture proves neither adapter seam drops or duplicates a field
/// on the way there.
///
/// As in [`test_bedrock_streaming_accounts_cache_writes_from_the_adapter`], the fixture reports
/// the aggregate with no `cacheDetails` breakdown so that a seam which silently dropped
/// `cacheWriteInputTokens` cannot coincidentally agree with one that forwarded it.
#[tokio::test]
async fn test_bedrock_cache_surfaces_agree_between_buffered_and_streaming() {
    let cache_usage = serde_json::json!({
        "inputTokens": 10_000,
        "outputTokens": 500,
        "totalTokens": 10_500,
        "cacheReadInputTokens": 2_000,
        "cacheWriteInputTokens": 1_500
    });

    let buffered_mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "cached"}]}},
            "stopReason": "end_turn",
            "usage": cache_usage
        })))
        .mount(&buffered_mock)
        .await;
    let buffered_adapter =
        BedrockAdapter::new(bedrock_config(&buffered_mock.uri()), probe_pricing_holder())
            .await
            .expect("adapter must build");
    let buffered_resp = buffered_adapter
        .chat_completion(&user_request("anthropic.claude-sonnet-4-6", "Say hi"))
        .await
        .unwrap();

    let streaming_mock = MockServer::start().await;
    let frames: &[(&str, serde_json::Value)] =
        &[("metadata", serde_json::json!({"usage": cache_usage}))];
    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse-stream$"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(event_stream_bytes(frames))
                .append_header("content-type", "application/vnd.amazon.eventstream"),
        )
        .mount(&streaming_mock)
        .await;
    let streaming_adapter = BedrockAdapter::new(
        bedrock_config(&streaming_mock.uri()),
        probe_pricing_holder(),
    )
    .await
    .expect("adapter must build");
    let mut streaming_req = user_request("anthropic.claude-sonnet-4-6", "Say hi");
    streaming_req.stream = Some(true);
    let chunks: Vec<_> = streaming_adapter
        .chat_completion_stream(&streaming_req)
        .await
        .unwrap()
        .collect()
        .await;
    let streaming_usage = chunks
        .iter()
        .filter_map(|c| c.as_ref().ok())
        .find_map(|c| c.usage.clone())
        .expect("metadata frame must produce a usage chunk");

    assert_eq!(
        buffered_resp.usage.prompt_tokens, streaming_usage.prompt_tokens,
        "prompt_tokens must agree"
    );
    assert_eq!(
        buffered_resp.usage.completion_tokens, streaming_usage.completion_tokens,
        "completion_tokens must agree"
    );
    assert_eq!(
        buffered_resp.usage.total_tokens, streaming_usage.total_tokens,
        "total_tokens must agree — neither path folds the cache buckets into it"
    );
    assert_eq!(
        buffered_resp.usage.cache_read_input_tokens, streaming_usage.cache_read_input_tokens,
        "cache_read_input_tokens must agree"
    );
    assert_eq!(
        buffered_resp.usage.cache_creation_input_tokens,
        streaming_usage.cache_creation_input_tokens,
        "cache_creation_input_tokens must agree"
    );
    assert_eq!(
        buffered_resp.usage.cache_write.reported_tokens(),
        streaming_usage.cache_write.reported_tokens(),
        "reported_tokens must agree"
    );
    assert_eq!(
        buffered_resp.usage.cache_write.detail_tokens(),
        streaming_usage.cache_write.detail_tokens(),
        "detail_tokens must agree"
    );
    assert_eq!(
        buffered_resp.usage.cache_write.accounted_tokens(),
        streaming_usage.cache_write.accounted_tokens(),
        "accounted_tokens must agree"
    );
    let mut buffered_credited: Vec<(&str, u64)> = buffered_resp
        .usage
        .cache_write
        .class_totals()
        .iter()
        .map(|t| (t.class.as_str(), t.tokens))
        .collect();
    buffered_credited.sort_unstable();
    let mut streaming_credited: Vec<(&str, u64)> = streaming_usage
        .cache_write
        .class_totals()
        .iter()
        .map(|t| (t.class.as_str(), t.tokens))
        .collect();
    streaming_credited.sort_unstable();
    assert_eq!(
        buffered_credited, streaming_credited,
        "per-class credited totals must agree"
    );
    assert_eq!(
        buffered_resp.usage.cache_write.unmatched_residual_tokens(),
        streaming_usage.cache_write.unmatched_residual_tokens(),
        "unmatched_residual_tokens must agree"
    );
    assert_eq!(
        buffered_resp.usage.cache_write.fallback_tokens(),
        streaming_usage.cache_write.fallback_tokens(),
        "fallback_tokens must agree"
    );
    assert_eq!(
        buffered_resp.usage.cache_write.outcome(),
        streaming_usage.cache_write.outcome(),
        "reconciliation outcome must agree"
    );

    let (_, buffered_finalized) = oxigate::utils::cost_headers::build_cost_headers(
        "anthropic.claude-sonnet-4-6",
        &buffered_resp.usage,
        bundled_pricing_holder(),
        false,
    );
    let (_, streaming_finalized) = oxigate::utils::cost_headers::build_cost_headers(
        "anthropic.claude-sonnet-4-6",
        &streaming_usage,
        bundled_pricing_holder(),
        false,
    );
    assert_eq!(
        buffered_finalized.cost.total_cost, streaming_finalized.cost.total_cost,
        "finalized cost must agree"
    );
    assert_eq!(
        buffered_finalized.cost.status, streaming_finalized.cost.status,
        "finalized cost status must agree"
    );
}

#[tokio::test]
async fn test_bedrock_chat_streaming() {
    let mock = MockServer::start().await;

    let frames: &[(&str, serde_json::Value)] = &[
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex":0,"delta":{"text":"Hello "}}),
        ),
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex":0,"delta":{"text":"world"}}),
        ),
        ("messageStop", serde_json::json!({"stopReason":"end_turn"})),
    ];
    let stream_bytes = event_stream_bytes(frames);

    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse-stream$"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(stream_bytes)
                .append_header("content-type", "application/vnd.amazon.eventstream"),
        )
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()), bundled_pricing_holder())
        .await
        .expect("adapter must build");

    let mut req = user_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "Count");
    req.stream = Some(true);

    let stream = adapter.chat_completion_stream(&req).await.unwrap();
    let chunks: Vec<_> = stream.collect().await;

    for chunk in &chunks {
        assert!(chunk.is_ok(), "no error chunks expected");
    }

    // Derive expected count from the input frames so this assertion stays in sync
    // automatically if the frame list is ever changed.
    // Formula: 1 preamble + N content deltas + 1 fallback (no metadata frame in this stream).
    let n_deltas = frames
        .iter()
        .filter(|(t, _)| *t == "contentBlockDelta")
        .count();
    assert_eq!(
        chunks.len(),
        1 + n_deltas + 1,
        "expected 1 preamble + {n_deltas} deltas + 1 fallback"
    );

    // First ok-chunk must carry the role preamble with empty content.
    let first_data = String::from_utf8_lossy(
        &chunks
            .iter()
            .filter_map(|c| c.as_ref().ok())
            .next()
            .unwrap()
            .data,
    )
    .to_string();
    assert!(
        first_data.contains("\"role\":\"assistant\""),
        "first chunk must carry role"
    );
    assert!(
        first_data.contains("\"content\":\"\""),
        "first chunk content must be empty string"
    );

    // Content-delta chunks (skip preamble at index 0) must not carry role.
    // Delta chunks have "content" in their delta; the fallback final chunk has "delta":{} so
    // no "content" key — filtering on "content" alone correctly isolates just the deltas.
    let content_chunks: Vec<_> = chunks
        .iter()
        .skip(1)
        .filter_map(|c| c.as_ref().ok())
        .filter(|c| String::from_utf8_lossy(&c.data).contains("\"content\""))
        .collect();
    assert!(
        !content_chunks.is_empty(),
        "expected at least one content-delta chunk after the preamble"
    );
    for c in &content_chunks {
        let s = String::from_utf8_lossy(&c.data).to_string();
        assert!(
            !s.contains("\"role\""),
            "content delta must not carry role: {s}"
        );
    }

    let all_data: String = chunks
        .iter()
        .filter_map(|c| c.as_ref().ok())
        .map(|c| String::from_utf8_lossy(&c.data).to_string())
        .collect();
    assert!(
        all_data.contains("Hello ") || all_data.contains("world"),
        "stream data must contain text deltas"
    );
    assert!(
        all_data.contains("[DONE]"),
        "stream must terminate with [DONE] even without a metadata frame"
    );
    assert!(
        all_data.contains("finish_reason"),
        "final chunk must carry finish_reason"
    );
}

#[tokio::test]
async fn test_bedrock_streaming_usage_in_final_chunk() {
    let mock = MockServer::start().await;

    // Real AWS order: contentBlockDelta(s) → messageStop → metadata.
    // metadata is the last event and carries billing token counts.
    let frames: &[(&str, serde_json::Value)] = &[
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex":0,"delta":{"text":"hi"}}),
        ),
        ("messageStop", serde_json::json!({"stopReason":"end_turn"})),
        (
            "metadata",
            serde_json::json!({"usage":{"inputTokens":20,"outputTokens":8,"totalTokens":28}}),
        ),
    ];
    let stream_bytes = event_stream_bytes(frames);

    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse-stream$"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(stream_bytes)
                .append_header("content-type", "application/vnd.amazon.eventstream"),
        )
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()), bundled_pricing_holder())
        .await
        .expect("adapter must build");

    let mut req = user_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "hi");
    req.stream = Some(true);

    let stream = adapter.chat_completion_stream(&req).await.unwrap();
    let chunks: Vec<_> = stream.collect().await;

    // Find the chunk that carries usage
    let usage_chunk = chunks
        .iter()
        .filter_map(|c| c.as_ref().ok())
        .find(|c| c.usage.is_some());

    assert!(
        usage_chunk.is_some(),
        "at least one stream chunk must carry usage data"
    );
    let usage = usage_chunk.unwrap().usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, 20);
    assert_eq!(usage.completion_tokens, 8);
}

#[tokio::test]
async fn test_bedrock_sigv4_headers_on_mock() {
    let mock = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse$"))
        .and(header_exists("authorization"))
        .and(header_exists("x-amz-date"))
        .and(header_exists("x-amz-content-sha256"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(converse_response_body("signed", 5, 3)),
        )
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()), bundled_pricing_holder())
        .await
        .expect("adapter must build");

    let req = user_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "test signing");
    // If SigV4 headers are missing, wiremock returns 404 (no mock matches).
    let resp = adapter.chat_completion(&req).await;
    assert!(
        resp.is_ok(),
        "request with SigV4 headers must succeed: {resp:?}"
    );
}

#[tokio::test]
async fn test_bedrock_429_throttling() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse$"))
        .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
            "__type": "ThrottlingException",
            "message": "Rate exceeded"
        })))
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()), bundled_pricing_holder())
        .await
        .expect("adapter must build");

    let req = user_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "hi");
    let err = adapter.chat_completion(&req).await.unwrap_err();
    assert!(
        matches!(err, ProviderError::RateLimited { retry_after: None }),
        "ThrottlingException must map to RateLimited{{retry_after: None}}, got: {err:?}"
    );
}

#[tokio::test]
async fn test_bedrock_streaming_fallback_no_metadata() {
    // Exercises the post-loop fallback: stream ends after messageStop with no metadata frame.
    // Validates that [DONE] is still emitted and no usage chunk is produced.
    let mock = MockServer::start().await;

    let frames: &[(&str, serde_json::Value)] = &[
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex":0,"delta":{"text":"ok"}}),
        ),
        ("messageStop", serde_json::json!({"stopReason":"end_turn"})),
        // intentionally no metadata frame
    ];
    let stream_bytes = event_stream_bytes(frames);

    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse-stream$"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(stream_bytes)
                .append_header("content-type", "application/vnd.amazon.eventstream"),
        )
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()), bundled_pricing_holder())
        .await
        .expect("adapter must build");

    let mut req = user_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "hi");
    req.stream = Some(true);

    let stream = adapter.chat_completion_stream(&req).await.unwrap();
    let chunks: Vec<_> = stream.collect().await;

    for chunk in &chunks {
        assert!(chunk.is_ok(), "no error chunks expected in fallback path");
    }

    let all_data: String = chunks
        .iter()
        .filter_map(|c| c.as_ref().ok())
        .map(|c| String::from_utf8_lossy(&c.data).to_string())
        .collect();
    assert!(
        all_data.contains("[DONE]"),
        "fallback path must still emit [DONE]"
    );

    // No metadata frame means no usage chunk.
    let has_usage = chunks
        .iter()
        .filter_map(|c| c.as_ref().ok())
        .any(|c| c.usage.is_some());
    assert!(
        !has_usage,
        "no usage chunk expected when metadata frame is absent"
    );
}

#[tokio::test]
async fn test_bedrock_config_missing_region_fails() {
    let config = BedrockConfig {
        region: "".to_string(), // intentionally empty
        access_key_id: Some(SecretString::from("AKID")),
        secret_access_key: Some(SecretString::from("SECRET")),
        session_token: None,
        endpoint_url: None,
        default_model: None,
        timeout_secs: None,
        supported_models: None,
    };

    let err = match BedrockAdapter::new(config, bundled_pricing_holder()).await {
        Ok(_) => panic!("missing region must fail at startup"),
        Err(e) => e,
    };
    let msg = format!("{err:?}");
    assert!(
        msg.contains("region") || matches!(err, ProviderError::InvalidRequest(_)),
        "missing region must produce an actionable error, got: {err:?}"
    );
}

// ── M3: Bedrock tool use integration tests ────────────────────────────────────────────────────

#[tokio::test]
async fn test_bedrock_tool_use_non_streaming_round_trip() {
    let mock = MockServer::start().await;
    let response_body = serde_json::json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_01",
                            "name": "get_weather",
                            "input": {"city": "London"}
                        }
                    }
                ]
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 20, "outputTokens": 10}
    });

    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()), bundled_pricing_holder())
        .await
        .expect("adapter must build");

    let req = tool_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "get_weather");
    let resp = adapter.chat_completion(&req).await.expect("must succeed");

    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("tool_calls"));
    let tool_calls = resp.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("tool_calls present");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "tooluse_01");
    assert_eq!(tool_calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "London");
}

#[tokio::test]
async fn test_bedrock_streaming_with_tools_returns_not_yet_supported() {
    let mock = MockServer::start().await;
    // Server should not be called; the error happens before dispatch.
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()), bundled_pricing_holder())
        .await
        .expect("adapter must build");

    let mut req = tool_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "get_weather");
    req.stream = Some(true);

    let err = match adapter.chat_completion_stream(&req).await {
        Err(e) => e,
        Ok(_) => panic!("expected NotYetSupported error but got Ok"),
    };
    assert!(
        matches!(err, ProviderError::NotYetSupported { .. }),
        "expected NotYetSupported, got: {err:?}"
    );
}

// ── Terminal-chunk marking ────────────────────────────────────────────────────────────
//
// Bedrock closes a stream at one of two places, reached by different upstream event sequences:
// the metadata frame when the provider reports usage, and the post-loop fallback when it stops
// after `messageStop` without one. Both emit the `[DONE]` terminator, so both are clean
// completions and both must be marked — one passing says nothing about the other.

/// Drives a Converse event-stream through the adapter and returns the yielded chunks.
async fn bedrock_stream_chunks(
    frames: &[(&str, serde_json::Value)],
) -> Vec<oxigate::domain::chat::StreamChunk> {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse-stream$"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(event_stream_bytes(frames))
                .append_header("content-type", "application/vnd.amazon.eventstream"),
        )
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()), bundled_pricing_holder())
        .await
        .expect("adapter must build");
    let mut req = user_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "hi");
    req.stream = Some(true);

    adapter
        .chat_completion_stream(&req)
        .await
        .expect("stream must start")
        .map(|r| r.expect("no error chunks expected"))
        .collect()
        .await
}

use crate::common::assert_only_last_is_final;

#[tokio::test]
async fn test_bedrock_metadata_terminated_stream_marks_its_last_chunk() {
    let chunks = bedrock_stream_chunks(&[
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex":0,"delta":{"text":"hi"}}),
        ),
        ("messageStop", serde_json::json!({"stopReason":"end_turn"})),
        (
            "metadata",
            serde_json::json!({"usage":{"inputTokens":20,"outputTokens":8,"totalTokens":28}}),
        ),
    ])
    .await;

    assert_only_last_is_final(&chunks);
    assert!(
        chunks.last().expect("chunks").usage.is_some(),
        "the metadata terminator carries the usage it just reported"
    );
}

/// A stream that stops after `messageStop` without a metadata frame still completed — the
/// provider simply reported no usage. The fallback terminator closes it, so it is terminal too.
#[tokio::test]
async fn test_bedrock_fallback_terminated_stream_marks_its_last_chunk() {
    let chunks = bedrock_stream_chunks(&[
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex":0,"delta":{"text":"hi"}}),
        ),
        ("messageStop", serde_json::json!({"stopReason":"end_turn"})),
    ])
    .await;

    assert_only_last_is_final(&chunks);
    assert!(
        chunks.last().expect("chunks").usage.is_none(),
        "the fallback path has no usage to report — accounting falls back to what came earlier"
    );
}
