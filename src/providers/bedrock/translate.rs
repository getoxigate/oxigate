// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Converse API ↔ OxiGate ChatRequest/ChatResponse translation .
//!
//! The Converse API is NOT the same as the Anthropic Messages API — do not reuse
//! src/providers/anthropic/translate.rs. Key differences:
//! - model goes in the URL path, not the body
//! - streaming selected by URL path (/converse-stream), not a body flag
//! - system messages live in a top-level `system` array (array of {text}, not a string)

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::warn;

use crate::domain::chat::{
    CacheAccounting, ChatRequest, ChatResponse, Choice, Message, MessageContent,
    ReasoningAccounting, Role, ToolCall, ToolCallFunction, Usage, UsageAccounting,
};
use crate::domain::ports::ProviderError;
use crate::domain::tool_schema::{ToolChoiceKind, parse_tool_choice_value, truncate_for_error};
use crate::domain::usage_accounting::{
    CacheWriteAccounting, CacheWriteAccumulator, CacheWriteClass, PricingContext,
};
use crate::providers::tool_limits::{BEDROCK_MAX_TOOLS, TOOL_ARGS_MAX_BYTES};

/// Token accounting declared by the AWS Bedrock Converse API contract.
///
/// Cache is **additive**:
/// `docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html` states that "when prompt
/// caching is enabled, the `inputTokens` field represents only the non-cached input tokens" and
/// gives `total input tokens = inputTokens + cacheReadInputTokens + cacheWriteInputTokens`
/// (accessed 2026-08-10).
///
/// This is the one declaration in the family that differs from what the gateway assumed. Both
/// Converse construction sites previously inherited the cache-inclusive type default, which is
/// wrong for this contract. It bills identically today only because neither site parses
/// `cacheReadInputTokens`, so there is nothing for an inclusive reading to subtract — the
/// declaration is corrected here, before the parsing that would make it live money.
///
/// Reasoning is `Additive`, the neutral value: neither Converse path parses a reasoning token
/// count, so nothing is charged on that axis and no first-party reference has been captured for
/// it. Capture one before this axis is relied upon.
pub(crate) const BEDROCK_ACCOUNTING: UsageAccounting = UsageAccounting {
    cache: CacheAccounting::Additive,
    reasoning: ReasoningAccounting::Additive,
};

// Bedrock Converse stop reason values (AWS Converse API spec).
pub(crate) mod bedrock_stop {
    pub const END_TURN: &str = "end_turn";
    pub const MAX_TOKENS: &str = "max_tokens";
    pub const STOP_SEQUENCE: &str = "stop_sequence";
    pub const TOOL_USE: &str = "tool_use";
}

// OpenAI-compatible finish reason values.
mod openai_finish {
    pub const STOP: &str = "stop";
    pub const LENGTH: &str = "length";
    pub const TOOL_CALLS: &str = "tool_calls";
}

// Converse wire role values (user/assistant only; system is top-level).
mod role {
    pub const USER: &str = "user";
    pub const ASSISTANT: &str = "assistant";
}

/// Converse API request body. `model` and `stream` are intentionally absent.
#[derive(Debug, Serialize)]
pub struct ConverseRequest {
    pub messages: Vec<ConverseMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub system: Vec<ConverseSystemBlock>,
    #[serde(rename = "inferenceConfig", skip_serializing_if = "Option::is_none")]
    pub inference_config: Option<InferenceConfig>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
}

/// Bedrock Converse toolConfig wrapper.
#[derive(Debug, Serialize)]
pub struct ToolConfig {
    pub tools: Vec<ConverseToolItem>,
    #[serde(rename = "toolChoice", skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolConfigToolChoice>,
}

/// Single item in the toolConfig tools array.
#[derive(Debug, Serialize)]
pub struct ConverseToolItem {
    #[serde(rename = "toolSpec")]
    pub tool_spec: ToolSpecInner,
}

/// Tool specification sent to Bedrock.
#[derive(Debug, Serialize)]
pub struct ToolSpecInner {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: InputSchema,
}

/// JSON schema wrapper for Bedrock tool input.
#[derive(Debug, Serialize)]
pub struct InputSchema {
    pub json: serde_json::Value,
}

/// Bedrock toolChoice value — externally tagged so each variant serializes as `{"variant": {...}}`.
#[derive(Debug, Serialize)]
pub enum ToolConfigToolChoice {
    #[serde(rename = "auto")]
    Auto {},
    #[serde(rename = "any")]
    Any {},
    #[serde(rename = "tool")]
    Tool { name: String },
}

#[derive(Debug, Serialize)]
pub struct ConverseMessage {
    pub role: String,
    pub content: Vec<ConverseContentBlock>,
}

/// Content block for a Converse message. Externally-tagged so each variant serializes
/// as the correct Bedrock wire shape (`{"text":"..."}`, `{"toolUse":{...}}`,
/// or `{"toolResult":{...}}`).
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ConverseContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        #[serde(rename = "toolUse")]
        tool_use: ConverseToolUse,
    },
    ToolResult {
        #[serde(rename = "toolResult")]
        tool_result: ConverseToolResultBlock,
    },
}

/// Wire shape for a Bedrock tool result block.
#[derive(Debug, Serialize)]
pub struct ConverseToolResultBlock {
    #[serde(rename = "toolUseId")]
    pub tool_use_id: String,
    pub content: Vec<ConverseToolResultContent>,
}

#[derive(Debug, Serialize)]
pub struct ConverseToolResultContent {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ConverseSystemBlock {
    pub text: String,
}

#[derive(Debug, Serialize, Default)]
pub struct InferenceConfig {
    #[serde(rename = "maxTokens", skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(rename = "stopSequences", skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
}

/// Converse non-streaming response.
#[derive(Debug, Deserialize)]
pub struct ConverseResponse {
    pub output: ConverseOutput,
    #[serde(rename = "stopReason")]
    pub stop_reason: Option<String>,
    pub usage: Option<ConverseUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ConverseOutput {
    pub message: ConverseOutputMessage,
}

#[derive(Debug, Deserialize)]
pub struct ConverseOutputMessage {
    pub role: String,
    pub content: Vec<ConverseOutputBlock>,
}

#[derive(Debug, Default, Deserialize)]
pub struct ConverseOutputBlock {
    pub text: Option<String>,
    #[serde(rename = "toolUse")]
    pub tool_use: Option<ConverseToolUse>,
}

/// Tool use block in a Converse message (request and response share the same wire shape).
#[derive(Debug, Serialize, Deserialize)]
pub struct ConverseToolUse {
    #[serde(rename = "toolUseId")]
    pub tool_use_id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// One entry of the Converse `cacheDetails` array.
///
/// A fixed-shape `{ttl, inputTokens}` pair, unlike the Anthropic Messages breakdown whose members
/// are arbitrary object keys — which is why the derived `Deserialize` is sufficient here and no
/// seeded parse is introduced. The array is bounded by the already-buffered response body.
#[derive(Debug, Deserialize)]
pub struct CacheDetail {
    /// The cache duration this entry's tokens were written for, verbatim from the wire.
    ///
    /// Kept as the raw string: it is both the class to canonicalize and the evidence key, and a
    /// value that names no known duration must stay legible in the persisted evidence rather than
    /// being normalized into a guess.
    pub ttl: String,
    #[serde(rename = "inputTokens")]
    pub input_tokens: u64,
}

#[derive(Debug, Default, Deserialize)]
pub struct ConverseUsage {
    #[serde(rename = "inputTokens")]
    pub input_tokens: u64,
    #[serde(rename = "outputTokens")]
    pub output_tokens: u64,
    /// Tokens served from cache. Additive: not part of `inputTokens`.
    #[serde(rename = "cacheReadInputTokens")]
    pub cache_read_input_tokens: Option<u64>,
    /// The provider's cache-write aggregate — an alternate view of `cacheDetails`, not an
    /// addition to it.
    #[serde(rename = "cacheWriteInputTokens")]
    pub cache_write_input_tokens: Option<u64>,
    /// The per-class breakdown of the cache write. Absent on API versions that predate it.
    #[serde(rename = "cacheDetails")]
    pub cache_details: Option<Vec<CacheDetail>>,
}

/// Translates an OxiGate `ChatRequest` to a Converse `ConverseRequest`.
///
/// `model` is excluded from the body — it goes in the URL path.
/// `stream` is excluded from the body — it is selected by URL path.
pub fn chat_request_to_converse(req: &ChatRequest) -> Result<ConverseRequest, ProviderError> {
    let (system_blocks, messages) = extract_system_and_messages(&req.messages)?;

    let max_tokens = req.max_completion_tokens.or(req.max_tokens);
    let stop_sequences = stop_from_extra(&req.extra);

    let has_inference = max_tokens.is_some()
        || req.temperature.is_some()
        || req.extra.contains_key("top_p")
        || !stop_sequences.is_empty();

    let inference_config = if has_inference {
        Some(InferenceConfig {
            max_tokens,
            temperature: req.temperature,
            top_p: req.extra.get("top_p").and_then(Value::as_f64),
            stop_sequences,
        })
    } else {
        None
    };

    let tool_config = build_converse_tool_config(req)?;

    Ok(ConverseRequest {
        messages,
        system: system_blocks,
        inference_config,
        tool_config,
    })
}

fn extract_system_and_messages(
    openai_messages: &[Message],
) -> Result<(Vec<ConverseSystemBlock>, Vec<ConverseMessage>), ProviderError> {
    let mut system: Vec<ConverseSystemBlock> = Vec::new();
    let mut messages: Vec<ConverseMessage> = Vec::new();
    // tool_call_id → function name, built from prior assistant turns for orphan detection.
    let mut tool_call_ids: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    for msg in openai_messages {
        match &msg.role {
            Role::System => {
                let text = message_content_to_text(msg);
                if !text.is_empty() {
                    system.push(ConverseSystemBlock { text });
                }
            }
            Role::Assistant => {
                if let Some(ref tcs) = msg.tool_calls {
                    for tc in tcs {
                        tool_call_ids.insert(tc.id.clone(), tc.function.name.clone());
                    }
                }
                let mut content: Vec<ConverseContentBlock> = Vec::new();
                let text = message_content_to_text(msg);
                if !text.is_empty() {
                    content.push(ConverseContentBlock::Text { text });
                }
                if let Some(ref tcs) = msg.tool_calls {
                    for tc in tcs {
                        if tc.function.arguments.len() > TOOL_ARGS_MAX_BYTES {
                            return Err(ProviderError::InvalidRequest(format!(
                                "tool_call '{}' arguments exceed the {} KiB limit",
                                truncate_for_error(tc.id.clone()),
                                TOOL_ARGS_MAX_BYTES / 1024,
                            )));
                        }
                        let input =
                            match serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                            {
                                Ok(v) => v,
                                Err(e) => {
                                    warn!(
                                        tool_call_id = %tc.id,
                                        arguments = %tc.function.arguments,
                                        error = %e,
                                        "tool call arguments are not valid JSON; forwarding null"
                                    );
                                    serde_json::Value::Null
                                }
                            };
                        content.push(ConverseContentBlock::ToolUse {
                            tool_use: ConverseToolUse {
                                tool_use_id: tc.id.clone(),
                                name: tc.function.name.clone(),
                                input,
                            },
                        });
                    }
                }
                if !content.is_empty() {
                    messages.push(ConverseMessage {
                        role: role::ASSISTANT.to_string(),
                        content,
                    });
                }
            }
            Role::Tool => {
                // Bedrock requires tool results as user-role messages with toolResult blocks.
                let text = message_content_to_text(msg);
                let tool_use_id = msg.tool_call_id.clone().ok_or_else(|| {
                    ProviderError::InvalidRequest(
                        "tool message is missing tool_call_id".to_string(),
                    )
                })?;
                if tool_use_id.is_empty() {
                    return Err(ProviderError::InvalidRequest(
                        "tool message tool_call_id must not be empty".to_string(),
                    ));
                }
                if !tool_call_ids.contains_key(&tool_use_id) {
                    return Err(ProviderError::InvalidRequest(format!(
                        "tool_call_id '{}' has no matching prior assistant tool_call in this \
                         request; include the full conversation history (assistant message with \
                         tool_calls[])",
                        truncate_for_error(tool_use_id.clone())
                    )));
                }
                messages.push(ConverseMessage {
                    role: role::USER.to_string(),
                    content: vec![ConverseContentBlock::ToolResult {
                        tool_result: ConverseToolResultBlock {
                            tool_use_id,
                            content: vec![ConverseToolResultContent { text }],
                        },
                    }],
                });
            }
            _ => {
                // Converse maps user and other → user wire role
                let text = message_content_to_text(msg);
                if !text.is_empty() {
                    messages.push(ConverseMessage {
                        role: role::USER.to_string(),
                        content: vec![ConverseContentBlock::Text { text }],
                    });
                }
            }
        }
    }
    Ok((system, messages))
}

fn message_content_to_text(msg: &Message) -> String {
    match &msg.content {
        Some(MessageContent::Text(t)) => t.clone(),
        Some(MessageContent::Parts(parts)) => parts
            .iter()
            .filter_map(|p| {
                if p.get("type").and_then(Value::as_str) == Some("text") {
                    p.get("text").and_then(Value::as_str).map(String::from)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        None => String::new(),
    }
}

fn stop_from_extra(extra: &serde_json::Map<String, Value>) -> Vec<String> {
    extra
        .get("stop")
        .and_then(|v| {
            if let Some(s) = v.as_str() {
                Some(vec![s.to_string()])
            } else {
                v.as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
            }
        })
        .unwrap_or_default()
}

/// Translates a Converse response to an OxiGate `ChatResponse`.
///
/// `pricing_context` is the generation the request was dispatched under, snapshotted before
/// dispatch. Cache-write observations are credited against its class registry and the context is
/// pinned onto the resulting accounting, so a reload landing while the request was in flight
/// cannot change what it is priced against.
pub fn converse_response_to_chat(
    resp: &ConverseResponse,
    model: &str,
    request_id: &str,
    pricing_context: &PricingContext,
) -> ChatResponse {
    let mut text_parts: Vec<&str> = Vec::new();
    let mut tool_calls_out: Vec<ToolCall> = Vec::new();

    for block in &resp.output.message.content {
        if let Some(ref t) = block.text {
            text_parts.push(t.as_str());
        }
        if let Some(ref tu) = block.tool_use {
            tool_calls_out.push(ToolCall {
                id: tu.tool_use_id.clone(),
                type_: "function".to_string(),
                function: ToolCallFunction {
                    name: tu.name.clone(),
                    arguments: serde_json::to_string(&tu.input)
                        .unwrap_or_else(|_| "{}".to_string()),
                },
            });
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(MessageContent::Text(text_parts.join("")))
    };

    let tool_calls = if tool_calls_out.is_empty() {
        None
    } else {
        Some(tool_calls_out)
    };

    let finish_reason = resp
        .stop_reason
        .as_deref()
        .map(map_stop_reason)
        .map(String::from);

    let usage = resp.usage.as_ref();
    let (prompt_tokens, completion_tokens, total_tokens) = usage
        .map(|u| {
            (
                u.input_tokens,
                u.output_tokens,
                // Deliberately excludes both cache buckets, matching the sibling Additive lane.
                u.input_tokens + u.output_tokens,
            )
        })
        .unwrap_or((0, 0, 0));

    let cache_write = converse_cache_write(
        usage.and_then(|u| u.cache_write_input_tokens),
        usage
            .and_then(|u| u.cache_details.as_deref())
            .unwrap_or_default(),
        pricing_context,
    );

    ChatResponse {
        id: format!("chatcmpl-{}", request_id),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: Role::Assistant,
                content,
                tool_calls,
                tool_call_id: None,
            },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cache_creation_input_tokens: cache_write.published_tokens(),
            cache_read_input_tokens: usage.and_then(|u| u.cache_read_input_tokens),
            accounting: BEDROCK_ACCOUNTING,
            cache_write,
            ..Default::default()
        },
    }
}

/// Closes cache-write accounting for one Converse response, buffered or streamed.
///
/// Both Converse paths report the same three things — a per-class detail list, an aggregate, and
/// nothing else — so both close accounting here. The two views are never summed:
/// `accounted_tokens` takes the maximum, and details that fall short of the aggregate leave an
/// unmatched residual priced at the tier fallback.
///
/// **An aggregate with no details gets no default class**, which is where this diverges from the
/// Anthropic Messages path. AWS's default-TTL statement is about the *request* — a response that
/// omits `cacheDetails` says nothing about what the request asked for — and AWS documents
/// `cacheDetails` as empty only when no cache creation occurred. An aggregate reported without a
/// breakdown is therefore an older API version, a non-conforming implementation, or an
/// undocumented shape, and the tokens may well have been written for the longer duration.
/// Crediting them to the shorter class would undercharge while reporting `exact`; leaving them as
/// an unmatched residual overcharges visibly and reports `rate-fallback`, which is the true
/// statement that no exact rate was established.
///
/// A `ttl` that does not canonicalize is one unknown class, never guessed at.
pub(crate) fn converse_cache_write(
    reported_aggregate: Option<u64>,
    details: &[CacheDetail],
    pricing_context: &PricingContext,
) -> CacheWriteAccounting {
    let mut accumulator = CacheWriteAccumulator::new(pricing_context.registry().clone());
    for detail in details {
        accumulator.observe_detail(
            &detail.ttl,
            CacheWriteClass::canonicalize(&detail.ttl),
            detail.input_tokens,
        );
    }
    if let Some(total) = reported_aggregate {
        accumulator.set_reported_aggregate(total);
    }
    let mut accounting = accumulator.finish();
    accounting.set_pricing_context(pricing_context.clone());
    accounting
}

/// Builds the Bedrock `toolConfig` from a `ChatRequest`.
/// Returns `None` when tools are absent or tool_choice is "none".
fn build_converse_tool_config(req: &ChatRequest) -> Result<Option<ToolConfig>, ProviderError> {
    let Some(ref tools) = req.tools else {
        return Ok(None);
    };
    if tools.is_empty() {
        return Ok(None);
    }

    let tool_choice_val = req.extra.get("tool_choice");

    if crate::domain::tool_schema::is_tool_choice_none(tool_choice_val) {
        return Ok(None);
    }

    let converse_tools: Vec<ConverseToolItem> = tools
        .iter()
        .filter(|t| t.type_ == "function")
        .map(|t| ConverseToolItem {
            tool_spec: ToolSpecInner {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                input_schema: InputSchema {
                    json: t
                        .function
                        .parameters
                        .clone()
                        .unwrap_or(serde_json::json!({"type": "object"})),
                },
            },
        })
        .collect();

    if converse_tools.is_empty() {
        return Ok(None);
    }

    if converse_tools.len() > BEDROCK_MAX_TOOLS {
        return Err(ProviderError::ToolCountExceeded {
            provider: "bedrock",
            requested: converse_tools.len(),
            limit: BEDROCK_MAX_TOOLS,
        });
    }

    let tool_choice = map_bedrock_tool_choice(tool_choice_val)?;

    Ok(Some(ToolConfig {
        tools: converse_tools,
        tool_choice,
    }))
}

/// Maps OpenAI `tool_choice` to a Bedrock `ToolConfigToolChoice`.
fn map_bedrock_tool_choice(
    val: Option<&serde_json::Value>,
) -> Result<Option<ToolConfigToolChoice>, ProviderError> {
    // absent tool_choice → no explicit constraint in ToolConfig
    if val.is_none() {
        return Ok(None);
    }
    match parse_tool_choice_value(val, "bedrock")? {
        ToolChoiceKind::Auto => Ok(Some(ToolConfigToolChoice::Auto {})),
        ToolChoiceKind::Required => Ok(Some(ToolConfigToolChoice::Any {})),
        ToolChoiceKind::Function { name } => Ok(Some(ToolConfigToolChoice::Tool { name })),
    }
}

/// Maps Bedrock stop reasons to OpenAI-compatible finish reasons.
pub fn map_stop_reason(stop_reason: &str) -> &str {
    match stop_reason {
        bedrock_stop::END_TURN | bedrock_stop::STOP_SEQUENCE => openai_finish::STOP,
        bedrock_stop::MAX_TOKENS => openai_finish::LENGTH,
        bedrock_stop::TOOL_USE => openai_finish::TOOL_CALLS,
        _ => openai_finish::STOP,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PricingConfig;
    use crate::domain::chat::Message;
    use crate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
    use crate::domain::usage_accounting::{CostStatus, FinalizedAccounting};

    /// A minimal successful `ConverseResponse` carrying the given usage.
    ///
    /// The accounting tests below care only about the usage numbers, so the message body is fixed
    /// and shared — two copies of it would let the pair drift on a field neither test is about.
    fn converse_response(input_tokens: u64, output_tokens: u64) -> ConverseResponse {
        converse_response_with_usage(ConverseUsage {
            input_tokens,
            output_tokens,
            ..Default::default()
        })
    }

    /// The same fixed message body, carrying a caller-supplied usage payload.
    fn converse_response_with_usage(usage: ConverseUsage) -> ConverseResponse {
        ConverseResponse {
            output: ConverseOutput {
                message: ConverseOutputMessage {
                    role: "assistant".to_string(),
                    content: vec![ConverseOutputBlock {
                        text: Some("hi".to_string()),
                        ..Default::default()
                    }],
                },
            },
            stop_reason: Some("end_turn".to_string()),
            usage: Some(usage),
        }
    }

    /// A pricing context over the bundled snapshot — the same database the gateway ships and
    /// prices with, so the class registry these tests account against is the production one.
    fn test_pricing_context() -> PricingContext {
        crate::domain::pricing::snapshot_pricing_context(&bundled_pricing_holder())
    }

    fn bundled_pricing_holder() -> std::sync::Arc<std::sync::RwLock<PricingDb>> {
        std::sync::Arc::new(std::sync::RwLock::new(
            PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
                .expect("bundled pricing must load"),
        ))
    }

    fn detail(ttl: &str, input_tokens: u64) -> CacheDetail {
        CacheDetail {
            ttl: ttl.to_string(),
            input_tokens,
        }
    }

    /// The Converse contract's accounting reaches the non-stream construction site.
    ///
    /// The streaming site in `bedrock/mod.rs` applies the same constant; the two are checked
    /// separately because they are separate literals until they are collapsed.
    #[test]
    fn test_converse_response_declares_bedrock_accounting() {
        let converse_resp = converse_response(5_000, 500);
        let chat = converse_response_to_chat(
            &converse_resp,
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "req-001",
            &test_pricing_context(),
        );

        assert_eq!(chat.usage.accounting, BEDROCK_ACCOUNTING);
        assert_eq!(chat.usage.accounting.cache, CacheAccounting::Additive);
    }

    /// Correcting the declaration moves no billed amount, because neither Converse path parses a
    /// cache-read token count: there is nothing for the previous cache-inclusive reading to
    /// subtract, so both readings price the same prompt.
    ///
    /// This is what makes it safe to correct the declaration ahead of the parsing work. It stops
    /// being true the moment those buckets are populated — which is exactly why the declaration
    /// is corrected first.
    #[test]
    fn test_bedrock_declaration_correction_moves_no_cost() {
        use crate::utils::cost_headers::build_cost_headers;

        let converse_resp = converse_response(5_000, 500);
        let model = "anthropic.claude-3-5-sonnet-20241022-v2:0";
        let corrected =
            converse_response_to_chat(&converse_resp, model, "req-001", &test_pricing_context())
                .usage;

        assert_eq!(
            corrected.cache_read_input_tokens, None,
            "no cache-read bucket is parsed on this path"
        );
        assert!(corrected.prompt_tokens_details.is_none());

        // The pre-correction reading, reconstructed: everything identical but cache-inclusive.
        let previous = Usage {
            accounting: UsageAccounting {
                cache: CacheAccounting::Inclusive,
                ..corrected.accounting
            },
            ..corrected.clone()
        };

        let (_, corrected_finalized) =
            build_cost_headers(model, &corrected, bundled_pricing_holder(), false);
        let (_, previous_finalized) =
            build_cost_headers(model, &previous, bundled_pricing_holder(), false);
        let (corrected_cost, corrected_tokens) =
            (&corrected_finalized.cost, &corrected_finalized.token_usage);
        let (previous_cost, previous_tokens) =
            (&previous_finalized.cost, &previous_finalized.token_usage);

        assert_eq!(corrected_cost.total_cost, previous_cost.total_cost);
        assert_eq!(corrected_cost.input_cost, previous_cost.input_cost);
        assert_eq!(
            corrected_cost.cached_input_cost,
            previous_cost.cached_input_cost
        );
        assert_eq!(corrected_tokens.input_tokens, previous_tokens.input_tokens);
        assert_eq!(
            corrected_tokens.context_input_tokens(),
            previous_tokens.context_input_tokens(),
            "the tier comparator must not move either"
        );
    }

    // -----------------------------------------------------------------------------------------
    // Cache-write accounting on the buffered path
    // -----------------------------------------------------------------------------------------

    /// The bundled Bedrock entry, the only one these cost assertions can be written against.
    const PRICED_MODEL: &str = "anthropic.claude-sonnet-4-6";

    /// Finalizes a Converse response the way the request path does, and returns its usage and
    /// finalization together — the pair every cost, status and spend assertion below reads from.
    fn finalize(usage: ConverseUsage) -> (Usage, FinalizedAccounting) {
        let (usage, _, finalized) = finalize_with_headers(usage);
        (usage, finalized)
    }

    /// The same finalization, keeping the emitted headers — used where a surface, not just a
    /// number, is what is under test.
    fn finalize_with_headers(
        usage: ConverseUsage,
    ) -> (Usage, axum::http::header::HeaderMap, FinalizedAccounting) {
        use crate::utils::cost_headers::build_cost_headers;

        let resp = converse_response_with_usage(usage);
        let chat =
            converse_response_to_chat(&resp, PRICED_MODEL, "req-cache", &test_pricing_context());
        let (headers, finalized) =
            build_cost_headers(PRICED_MODEL, &chat.usage, bundled_pricing_holder(), false);
        (chat.usage, headers, finalized)
    }

    fn class_tokens(usage: &Usage, class: &str) -> u64 {
        usage
            .cache_write
            .class_totals()
            .iter()
            .find(|t| t.class.as_str() == class)
            .map_or(0, |t| t.tokens)
    }

    /// Per-class details are credited to their canonical classes, the aggregate is the reported
    /// view of the same quantity rather than an addition to it, and the read bucket arrives.
    #[test]
    fn test_bedrock_buffered_credits_cache_details_per_class() {
        let (usage, finalized) = finalize(ConverseUsage {
            input_tokens: 10_000,
            output_tokens: 500,
            cache_read_input_tokens: Some(2_000),
            cache_write_input_tokens: Some(1_500),
            cache_details: Some(vec![detail("5m", 1_000), detail("1h", 500)]),
        });

        assert_eq!(usage.cache_read_input_tokens, Some(2_000));
        assert_eq!(
            usage.cache_creation_input_tokens,
            Some(1_500),
            "the published quantity is the accounted one, not the sum of both views"
        );
        assert_eq!(class_tokens(&usage, "5m"), 1_000);
        assert_eq!(class_tokens(&usage, "1h"), 500);
        assert_eq!(usage.cache_write.fallback_tokens(), 0);
        assert!(usage.cache_write.partition_is_exact());
        assert_eq!(finalized.cost.status, CostStatus::Exact);
    }

    /// A positive aggregate with no detail breakdown is an unmatched residual, not a write to the
    /// contract's shorter default class.
    ///
    /// A Converse response omitting `cacheDetails` says nothing about the TTL the request asked
    /// for, so crediting it to `5m` would undercharge a possible `1h` write while reporting
    /// `exact`. The residual is priced at the tier fallback and the request says `rate-fallback`.
    #[test]
    fn test_bedrock_buffered_aggregate_without_details_is_an_unmatched_residual() {
        let (usage, finalized) = finalize(ConverseUsage {
            input_tokens: 10_000,
            output_tokens: 500,
            cache_write_input_tokens: Some(1_500),
            ..Default::default()
        });

        assert_eq!(usage.cache_creation_input_tokens, Some(1_500));
        assert!(
            usage.cache_write.class_totals().is_empty(),
            "no class was observed, so none may be credited"
        );
        assert_eq!(usage.cache_write.unmatched_residual_tokens(), 1_500);
        assert_eq!(usage.cache_write.fallback_tokens(), 1_500);
        assert_eq!(finalized.cost.status, CostStatus::RateFallback);
    }

    /// A zero aggregate is a provider saying it wrote nothing, which is an exact statement.
    ///
    /// Only a *positive* fallback quantity degrades status, so this case must stay `Exact` — the
    /// discriminator that keeps the test above about the residual rather than about the field
    /// merely being present.
    #[test]
    fn test_bedrock_buffered_zero_aggregate_stays_exact() {
        let (usage, finalized) = finalize(ConverseUsage {
            input_tokens: 10_000,
            output_tokens: 500,
            cache_write_input_tokens: Some(0),
            ..Default::default()
        });

        assert_eq!(
            usage.cache_creation_input_tokens,
            Some(0),
            "a reported zero is a statement, not silence"
        );
        assert_eq!(usage.cache_write.fallback_tokens(), 0);
        assert_eq!(finalized.cost.status, CostStatus::Exact);
    }

    /// Details falling short of the aggregate leave the shortfall as a residual — priced at the
    /// fallback rate, and specifically not topped up into the class that was reported.
    #[test]
    fn test_bedrock_buffered_partial_details_leave_a_residual() {
        let (usage, finalized) = finalize(ConverseUsage {
            input_tokens: 10_000,
            output_tokens: 500,
            cache_write_input_tokens: Some(1_500),
            cache_details: Some(vec![detail("5m", 1_000)]),
            ..Default::default()
        });

        assert_eq!(usage.cache_creation_input_tokens, Some(1_500));
        assert_eq!(
            class_tokens(&usage, "5m"),
            1_000,
            "the residual must not be absorbed into the observed class"
        );
        assert_eq!(usage.cache_write.unmatched_residual_tokens(), 500);
        assert_eq!(usage.cache_write.fallback_tokens(), 500);
        assert_eq!(finalized.cost.status, CostStatus::RateFallback);
    }

    /// A `ttl` that is not a duration at all is one unknown class, never guessed at, and its raw
    /// spelling survives into the evidence.
    ///
    /// The fixture has to name something the shared grammar genuinely rejects. A value like
    /// `"10m"` would not do: it canonicalizes fine and is merely a class the Bedrock tier does
    /// not configure, which is the tier-fallback path rather than this one. `canonical_class` is
    /// the assertion that separates them — `None` here, `Some` there.
    #[test]
    fn test_bedrock_buffered_uncanonicalizable_ttl_is_one_unknown_class() {
        let (usage, finalized) = finalize(ConverseUsage {
            input_tokens: 10_000,
            output_tokens: 500,
            cache_write_input_tokens: Some(1_000),
            cache_details: Some(vec![detail("forever", 1_000)]),
            ..Default::default()
        });

        assert!(usage.cache_write.class_totals().is_empty());
        assert_eq!(usage.cache_write.unknown_tokens(), 1_000);
        assert_eq!(usage.cache_write.fallback_tokens(), 1_000);
        assert_eq!(finalized.cost.status, CostStatus::RateFallback);

        // Priced at the tier fallback — `max(configured, 1.0)` = 2.0x over a 3,000 input rate.
        assert_eq!(finalized.cost.cache_write_cost.0, 1_000 * 6_000);

        let evidence: Vec<(String, Option<String>)> = usage
            .cache_write
            .evidence_entries()
            .iter()
            .map(|e| {
                (
                    e.raw_key.clone(),
                    e.canonical_class.map(|c| c.as_str().to_string()),
                )
            })
            .collect();
        assert_eq!(
            evidence,
            vec![("forever".to_string(), None)],
            "the raw spelling is retained and no class is invented for it"
        );
    }

    /// Contradictory views reconcile to the larger quantity and say so, rather than trusting
    /// whichever view happens to be smaller.
    #[test]
    fn test_bedrock_buffered_details_exceeding_aggregate_reconcile_upward() {
        let (usage, finalized) = finalize(ConverseUsage {
            input_tokens: 10_000,
            output_tokens: 500,
            cache_write_input_tokens: Some(1_000),
            cache_details: Some(vec![detail("5m", 2_000)]),
            ..Default::default()
        });

        assert_eq!(usage.cache_creation_input_tokens, Some(2_000));
        assert_eq!(class_tokens(&usage, "5m"), 2_000);
        assert_eq!(usage.cache_write.fallback_tokens(), 0);
        assert_eq!(finalized.cost.status, CostStatus::Reconciled);
    }

    /// The exact-cost oracle: mixed `5m`/`1h` writes plus reads, hand-computed against the
    /// bundled Bedrock entry.
    ///
    /// Rates in nano-USD per token — input 3,000; output 15,000; cache read `3,000 x 0.1` = 300;
    /// `5m` write `3,000 x 1.25` = 3,750; `1h` write `3,000 x 2.0` = 6,000.
    ///
    /// | Component  | Tokens | Rate   | Cost       |
    /// |------------|--------|--------|------------|
    /// | input      | 10,000 |  3,000 | 30,000,000 |
    /// | cache read |  2,000 |    300 |    600,000 |
    /// | `5m` write |  1,000 |  3,750 |  3,750,000 |
    /// | `1h` write |    500 |  6,000 |  3,000,000 |
    /// | output     |    500 | 15,000 |  7,500,000 |
    /// | **total**  |        |        | 44,850,000 |
    ///
    /// The contract is `Additive`, so `inputTokens` already excludes both cache buckets and no
    /// carve-out applies: the accumulator's charge is the whole cache charge.
    #[test]
    fn test_bedrock_buffered_mixed_class_cost_oracle() {
        let (_, finalized) = finalize(ConverseUsage {
            input_tokens: 10_000,
            output_tokens: 500,
            cache_read_input_tokens: Some(2_000),
            cache_write_input_tokens: Some(1_500),
            cache_details: Some(vec![detail("5m", 1_000), detail("1h", 500)]),
        });

        assert_eq!(finalized.cost.input_cost.0, 30_000_000);
        assert_eq!(finalized.cost.cached_input_cost.0, 600_000);
        assert_eq!(finalized.cost.cache_write_cost.0, 6_750_000);
        assert_eq!(finalized.cost.output_cost.0, 7_500_000);
        assert_eq!(finalized.cost.total_cost.0, 44_850_000);
        assert_eq!(finalized.cost.status, CostStatus::Exact);
    }

    /// A response with no cache fields bills exactly as it did before the fields were parsed:
    /// nothing is published, nothing is accounted, and the cost is input plus output alone.
    #[test]
    fn test_bedrock_buffered_no_cache_fields_bills_as_before() {
        let (usage, finalized) = finalize(ConverseUsage {
            input_tokens: 10_000,
            output_tokens: 500,
            ..Default::default()
        });

        assert_eq!(
            usage.cache_creation_input_tokens, None,
            "silence must not be published as a zero"
        );
        assert_eq!(usage.cache_read_input_tokens, None);
        assert_eq!(usage.cache_write.accounted_tokens(), 0);
        assert_eq!(finalized.cost.total_cost.0, 30_000_000 + 7_500_000);
        assert_eq!(finalized.cost.status, CostStatus::Exact);
    }

    /// `total_tokens` stays `inputTokens + outputTokens` and does not absorb the cache buckets.
    ///
    /// Understated once the buckets are parsed, and deliberately so: the sibling Additive lane
    /// publishes the same convention, and changing it is a cross-provider wire change rather than
    /// something to fix on one lane in passing.
    #[test]
    fn test_bedrock_total_tokens_excludes_the_cache_buckets() {
        let (usage, _) = finalize(ConverseUsage {
            input_tokens: 10_000,
            output_tokens: 500,
            cache_read_input_tokens: Some(2_000),
            cache_write_input_tokens: Some(1_500),
            cache_details: Some(vec![detail("5m", 1_500)]),
        });

        assert_eq!(usage.total_tokens, 10_500);
    }

    /// The surfaces that carry a cache-write quantity agree, and so do the ones that carry cost.
    ///
    /// Covers three of the four surfaces the criterion names: the response body, the emitted
    /// `X-Oxigate-Request-Cost` / `X-Oxigate-Cost-Status` headers, and the persisted row. The
    /// spend row has no cache-write column by design, so the quantity pair is the response field
    /// and the persisted evidence; cache *reads* are a column and are checked against it.
    ///
    /// **Two legs are not covered here and are not claimed:** the terminal `oxigate.usage` event
    /// and the Redis budget increment are emitted by the request handler, not by anything this
    /// test can reach, so proving a Bedrock cache cost reaches them needs handler-level coverage.
    /// The generic budget tests exercise that path but not this lane's newly parsed quantity.
    #[test]
    fn test_bedrock_cache_surfaces_agree() {
        use crate::domain::auth::RequestIdentity;
        use crate::domain::spend::SpendRecord;
        use crate::utils::cost_headers::CostHeader;

        let (usage, headers, finalized) = finalize_with_headers(ConverseUsage {
            input_tokens: 10_000,
            output_tokens: 500,
            cache_read_input_tokens: Some(2_000),
            cache_write_input_tokens: Some(1_500),
            cache_details: Some(vec![detail("5m", 1_000), detail("1h", 500)]),
        });
        let record = SpendRecord::build(
            &RequestIdentity::default(),
            PRICED_MODEL,
            "bedrock",
            &finalized,
            7,
        );

        // Quantity: the response field and the persisted evidence are the only two surfaces that
        // carry a cache-write quantity at all.
        let evidence = record
            .usage_evidence
            .as_ref()
            .expect("an accounted cache write persists its evidence");
        assert_eq!(
            usage.cache_creation_input_tokens,
            Some(evidence.cache_write.accounted_tokens)
        );
        // Cache reads are a column.
        assert_eq!(
            record.cache_read_tokens as u64,
            usage.cache_read_input_tokens.unwrap_or(0)
        );
        // Cost and status: the emitted headers, the finalization and the row carry the same two
        // values. Asserted against the absolute oracle, not just against each other, so three
        // equal-but-wrong surfaces cannot satisfy it.
        assert_eq!(record.cost_nano_usd, finalized.cost.total_cost);
        assert_eq!(record.cost_status, finalized.cost.status);
        assert_eq!(record.cost_nano_usd.0, 44_850_000);
        assert_eq!(
            headers
                .get(CostHeader::REQUEST_COST)
                .and_then(|v| v.to_str().ok()),
            Some(finalized.cost.total_cost.to_display_string()).as_deref()
        );
        assert_eq!(
            headers
                .get(CostHeader::COST_STATUS)
                .and_then(|v| v.to_str().ok()),
            Some("exact")
        );
    }

    fn make_request(messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            model: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
            messages,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra: serde_json::Map::new(),
        }
    }

    fn user_msg(text: &str) -> Message {
        Message {
            role: Role::User,
            content: Some(MessageContent::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn system_msg(text: &str) -> Message {
        Message {
            role: Role::System,
            content: Some(MessageContent::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn test_converse_request_no_model_in_body() {
        let req = make_request(vec![user_msg("hello")]);
        let converse = chat_request_to_converse(&req).expect("must translate");
        let json = serde_json::to_value(&converse).unwrap();
        assert!(
            json.get("model").is_none(),
            "model must not appear in Converse body"
        );
    }

    #[test]
    fn test_converse_request_no_stream_in_body() {
        let mut req = make_request(vec![user_msg("hello")]);
        req.stream = Some(true);
        let converse = chat_request_to_converse(&req).expect("must translate");
        let json = serde_json::to_value(&converse).unwrap();
        assert!(
            json.get("stream").is_none(),
            "stream must not appear in Converse body"
        );
    }

    #[test]
    fn test_converse_request_system_extracted() {
        let req = make_request(vec![system_msg("You are helpful"), user_msg("hi")]);
        let converse = chat_request_to_converse(&req).expect("must translate");
        assert_eq!(converse.system.len(), 1);
        assert_eq!(converse.system[0].text, "You are helpful");
        assert_eq!(converse.messages.len(), 1);
        assert_eq!(converse.messages[0].role, "user");
    }

    #[test]
    fn test_converse_request_inference_config() {
        let mut req = make_request(vec![user_msg("hi")]);
        req.max_tokens = Some(512);
        req.temperature = Some(0.7);
        let converse = chat_request_to_converse(&req).expect("must translate");

        // Verify struct values.
        let ic = converse.inference_config.as_ref().unwrap();
        assert_eq!(ic.max_tokens, Some(512));
        assert!((ic.temperature.unwrap() - 0.7).abs() < 1e-9);

        // Verify wire key is "inferenceConfig" (camelCase), not "inference_config".
        let json = serde_json::to_value(&converse).unwrap();
        assert!(
            json.get("inferenceConfig").is_some(),
            "wire key must be 'inferenceConfig', got: {json}"
        );
        assert!(
            json.get("inference_config").is_none(),
            "snake_case key must not appear on wire"
        );
        let ic_json = &json["inferenceConfig"];
        assert_eq!(ic_json["maxTokens"], 512);
    }

    #[test]
    fn test_converse_response_translates_to_chat_response() {
        let converse_resp = ConverseResponse {
            output: ConverseOutput {
                message: ConverseOutputMessage {
                    role: "assistant".to_string(),
                    content: vec![ConverseOutputBlock {
                        text: Some("Hi there".to_string()),
                        ..Default::default()
                    }],
                },
            },
            stop_reason: Some("end_turn".to_string()),
            usage: Some(ConverseUsage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            }),
        };
        let chat = converse_response_to_chat(
            &converse_resp,
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "req-001",
            &test_pricing_context(),
        );
        assert_eq!(chat.choices.len(), 1);
        let msg = &chat.choices[0].message;
        assert_eq!(msg.role, Role::Assistant);
        if let Some(MessageContent::Text(t)) = &msg.content {
            assert_eq!(t, "Hi there");
        } else {
            panic!("expected text content");
        }
    }

    #[test]
    fn test_converse_response_multi_block_concatenated() {
        let converse_resp = ConverseResponse {
            output: ConverseOutput {
                message: ConverseOutputMessage {
                    role: "assistant".to_string(),
                    content: vec![
                        ConverseOutputBlock {
                            text: Some("Hello ".to_string()),
                            ..Default::default()
                        },
                        ConverseOutputBlock {
                            text: Some("world".to_string()),
                            ..Default::default()
                        },
                    ],
                },
            },
            stop_reason: Some("end_turn".to_string()),
            usage: Some(ConverseUsage {
                input_tokens: 5,
                output_tokens: 3,
                ..Default::default()
            }),
        };
        let chat = converse_response_to_chat(
            &converse_resp,
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "req-002",
            &test_pricing_context(),
        );
        assert_eq!(chat.choices.len(), 1);
        if let Some(MessageContent::Text(t)) = &chat.choices[0].message.content {
            assert_eq!(t, "Hello world");
        } else {
            panic!("expected text content");
        }
    }

    #[test]
    fn test_converse_stop_reason_mapping() {
        assert_eq!(map_stop_reason("end_turn"), "stop");
        assert_eq!(map_stop_reason("max_tokens"), "length");
        assert_eq!(map_stop_reason("stop_sequence"), "stop");
    }

    #[test]
    fn test_converse_usage_mapped() {
        let converse_resp = ConverseResponse {
            output: ConverseOutput {
                message: ConverseOutputMessage {
                    role: "assistant".to_string(),
                    content: vec![ConverseOutputBlock {
                        text: Some("ok".to_string()),
                        ..Default::default()
                    }],
                },
            },
            stop_reason: Some("end_turn".to_string()),
            usage: Some(ConverseUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            }),
        };
        let chat =
            converse_response_to_chat(&converse_resp, "model", "id", &test_pricing_context());
        assert_eq!(chat.usage.prompt_tokens, 100);
        assert_eq!(chat.usage.completion_tokens, 50);
        assert_eq!(chat.usage.total_tokens, 150);
    }

    // ── map_bedrock_tool_choice tests ────────────────────────────────────────────

    #[test]
    fn test_bedrock_tool_choice_absent_returns_none() {
        let result = map_bedrock_tool_choice(None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_bedrock_tool_choice_auto() {
        use serde_json::json;
        let result = map_bedrock_tool_choice(Some(&json!("auto"))).unwrap();
        assert!(matches!(result, Some(ToolConfigToolChoice::Auto {})));
    }

    #[test]
    fn test_bedrock_tool_choice_required_maps_to_any() {
        use serde_json::json;
        let result = map_bedrock_tool_choice(Some(&json!("required"))).unwrap();
        assert!(matches!(result, Some(ToolConfigToolChoice::Any {})));
    }

    #[test]
    fn test_bedrock_tool_choice_any_string_maps_to_any() {
        use serde_json::json;
        let result = map_bedrock_tool_choice(Some(&json!("any"))).unwrap();
        assert!(matches!(result, Some(ToolConfigToolChoice::Any {})));
    }

    #[test]
    fn test_bedrock_tool_choice_function_object() {
        use serde_json::json;
        let v = json!({"type": "function", "function": {"name": "search"}});
        let result = map_bedrock_tool_choice(Some(&v)).unwrap();
        match result {
            Some(ToolConfigToolChoice::Tool { name }) => assert_eq!(name, "search"),
            other => panic!("expected Tool{{name}}, got {other:?}"),
        }
    }

    #[test]
    fn test_bedrock_tool_choice_round_trip_serializes_correctly() {
        // Verify the wire format matches what Bedrock Converse expects.
        use serde_json::json;
        let auto = map_bedrock_tool_choice(Some(&json!("auto")))
            .unwrap()
            .unwrap();
        assert_eq!(serde_json::to_value(auto).unwrap(), json!({"auto": {}}));

        let any = map_bedrock_tool_choice(Some(&json!("required")))
            .unwrap()
            .unwrap();
        assert_eq!(serde_json::to_value(any).unwrap(), json!({"any": {}}));

        let v = json!({"type": "function", "function": {"name": "fn_x"}});
        let tool = map_bedrock_tool_choice(Some(&v)).unwrap().unwrap();
        assert_eq!(
            serde_json::to_value(tool).unwrap(),
            json!({"tool": {"name": "fn_x"}})
        );
    }

    // ── NEW-A: end-to-end chat_request_to_converse with tools + tool_choice ─────

    #[test]
    fn test_full_converse_with_tools_and_tool_choice_required() {
        use crate::domain::chat::{Tool, ToolFunction};
        use serde_json::json;

        let mut req = make_request(vec![user_msg("What's the weather?")]);
        req.tools = Some(vec![Tool {
            type_: "function".to_string(),
            function: ToolFunction {
                name: "get_weather".to_string(),
                description: Some("Returns current weather".to_string()),
                parameters: Some(json!({"type": "object", "properties": {}})),
            },
        }]);
        req.extra
            .insert("tool_choice".to_string(), json!("required"));

        let converse = chat_request_to_converse(&req).expect("must translate");

        let cfg = converse
            .tool_config
            .as_ref()
            .expect("tool_config must be present");
        assert_eq!(cfg.tools.len(), 1);
        assert_eq!(cfg.tools[0].tool_spec.name, "get_weather");
        assert!(
            matches!(cfg.tool_choice, Some(ToolConfigToolChoice::Any {})),
            "tool_choice 'required' must map to Bedrock Any"
        );
    }

    #[test]
    fn test_full_converse_with_tools_and_tool_choice_function() {
        use crate::domain::chat::{Tool, ToolFunction};
        use serde_json::json;

        let mut req = make_request(vec![user_msg("hello")]);
        req.tools = Some(vec![Tool {
            type_: "function".to_string(),
            function: ToolFunction {
                name: "search".to_string(),
                description: None,
                parameters: None,
            },
        }]);
        req.extra.insert(
            "tool_choice".to_string(),
            json!({"type": "function", "function": {"name": "search"}}),
        );

        let converse = chat_request_to_converse(&req).expect("must translate");

        let cfg = converse
            .tool_config
            .as_ref()
            .expect("tool_config must be present");
        match &cfg.tool_choice {
            Some(ToolConfigToolChoice::Tool { name }) => assert_eq!(name, "search"),
            other => panic!("expected Tool{{name}}, got {other:?}"),
        }
    }

    // ──: orphaned tool_call_id guard ──────────────────────────────────

    fn request_with_tool_result(
        tool_call_id: Option<&str>,
        include_assistant: bool,
    ) -> ChatRequest {
        use crate::domain::chat::{ToolCall, ToolCallFunction};
        let mut messages = vec![user_msg("Weather?")];
        if include_assistant {
            messages.push(Message {
                role: Role::Assistant,
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: tool_call_id.unwrap_or("call_x").to_string(),
                    type_: "function".to_string(),
                    function: ToolCallFunction {
                        name: "get_weather".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
            });
        }
        messages.push(Message {
            role: Role::Tool,
            content: Some(crate::domain::chat::MessageContent::Text("{}".into())),
            tool_calls: None,
            tool_call_id: tool_call_id.map(str::to_string),
        });
        make_request(messages)
    }

    #[test]
    fn test_matched_tool_call_id_converse_ok() {
        let req = request_with_tool_result(Some("call_abc"), true);
        let converse = chat_request_to_converse(&req).expect("must translate");
        // The assistant message must be present and contain a toolUse block.
        let assistant = converse
            .messages
            .iter()
            .find(|m| m.role == role::ASSISTANT)
            .expect("assistant message must be present");
        assert!(
            assistant
                .content
                .iter()
                .any(|b| matches!(b, ConverseContentBlock::ToolUse { .. })),
            "assistant message must contain a toolUse block"
        );
    }

    #[test]
    fn test_pure_tool_call_assistant_message_emits_tool_use_block() {
        use crate::domain::chat::{ToolCall, ToolCallFunction};
        let req = make_request(vec![
            user_msg("call the function"),
            Message {
                role: Role::Assistant,
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    type_: "function".to_string(),
                    function: ToolCallFunction {
                        name: "my_func".to_string(),
                        arguments: r#"{"x":1}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
            },
        ]);
        let converse = chat_request_to_converse(&req).expect("must translate");
        let assistant = converse
            .messages
            .iter()
            .find(|m| m.role == role::ASSISTANT)
            .expect("pure-tool-call assistant message must be present — was previously dropped");
        match &assistant.content[0] {
            ConverseContentBlock::ToolUse { tool_use } => {
                assert_eq!(tool_use.tool_use_id, "call_1");
                assert_eq!(tool_use.name, "my_func");
                assert_eq!(tool_use.input.get("x").and_then(|v| v.as_i64()), Some(1));
            }
            other => panic!("expected ToolUse block, got {other:?}"),
        }
    }

    #[test]
    fn test_orphaned_tool_call_id_converse_invalid_request() {
        // Tool message references an ID not in any prior assistant turn.
        let req = request_with_tool_result(Some("call_orphan"), false);
        let err = chat_request_to_converse(&req).expect_err("orphaned ID must error");
        match &err {
            ProviderError::InvalidRequest(msg) => {
                assert!(msg.contains("call_orphan"), "error must name the ID: {msg}");
                assert!(
                    msg.contains("no matching prior assistant tool_call"),
                    "{msg}"
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn test_orphaned_long_tool_call_id_is_truncated_in_error() {
        let long_id = "y".repeat(300);
        let mut messages = vec![user_msg("go")];
        messages.push(Message {
            role: Role::Tool,
            content: Some(crate::domain::chat::MessageContent::Text("{}".into())),
            tool_calls: None,
            tool_call_id: Some(long_id.clone()),
        });
        let req = make_request(messages);
        let err = chat_request_to_converse(&req).expect_err("orphaned long ID must error");
        match &err {
            ProviderError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("no matching prior assistant tool_call"),
                    "{msg}"
                );
                assert!(
                    msg.contains("<truncated>"),
                    "300-byte ID must be truncated: {msg}"
                );
                assert!(
                    msg.len() < 512,
                    "error must be bounded, got {} bytes",
                    msg.len()
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn test_missing_tool_call_id_converse_invalid_request() {
        let req = request_with_tool_result(None, false);
        let err = chat_request_to_converse(&req).expect_err("missing ID must error");
        assert!(
            matches!(err, ProviderError::InvalidRequest(_)),
            "expected InvalidRequest, got {err:?}"
        );
    }

    #[test]
    fn test_assistant_message_with_text_and_tool_calls() {
        use crate::domain::chat::{ToolCall, ToolCallFunction};
        let req = make_request(vec![
            user_msg("Weather in NYC?"),
            Message {
                role: Role::Assistant,
                content: Some(crate::domain::chat::MessageContent::Text(
                    "I'll check that for you.".into(),
                )),
                tool_calls: Some(vec![ToolCall {
                    id: "call_wx".to_string(),
                    type_: "function".to_string(),
                    function: ToolCallFunction {
                        name: "get_weather".to_string(),
                        arguments: r#"{"city":"NYC"}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
            },
        ]);
        let converse = chat_request_to_converse(&req).expect("must translate");
        let assistant = converse
            .messages
            .iter()
            .find(|m| m.role == role::ASSISTANT)
            .expect("assistant message must be present");
        assert_eq!(
            assistant.content.len(),
            2,
            "must have text + tool_use blocks"
        );
        match &assistant.content[0] {
            ConverseContentBlock::Text { text } => {
                assert_eq!(text, "I'll check that for you.");
            }
            other => panic!("first block must be Text, got {other:?}"),
        }
        match &assistant.content[1] {
            ConverseContentBlock::ToolUse { tool_use } => {
                assert_eq!(tool_use.tool_use_id, "call_wx");
                assert_eq!(tool_use.name, "get_weather");
            }
            other => panic!("second block must be ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn test_tool_args_over_limit_returns_invalid_request() {
        use crate::domain::chat::{ToolCall, ToolCallFunction};
        use crate::providers::tool_limits::TOOL_ARGS_MAX_BYTES;
        let oversized = "x".repeat(TOOL_ARGS_MAX_BYTES + 1);
        let req = make_request(vec![
            user_msg("call it"),
            Message {
                role: Role::Assistant,
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_big".to_string(),
                    type_: "function".to_string(),
                    function: ToolCallFunction {
                        name: "big_func".to_string(),
                        arguments: oversized,
                    },
                }]),
                tool_call_id: None,
            },
        ]);
        let err = chat_request_to_converse(&req).expect_err("over-limit args must error");
        assert!(
            matches!(err, ProviderError::InvalidRequest(_)),
            "expected InvalidRequest, got {err:?}"
        );
    }
}
