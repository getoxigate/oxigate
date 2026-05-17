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
    ChatRequest, ChatResponse, Choice, Message, MessageContent, Role, ToolCall, ToolCallFunction,
    Usage,
};
use crate::domain::ports::ProviderError;
use crate::domain::tool_schema::{ToolChoiceKind, parse_tool_choice_value, truncate_for_error};
use crate::providers::tool_limits::{BEDROCK_MAX_TOOLS, TOOL_ARGS_MAX_BYTES};

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

#[derive(Debug, Deserialize)]
pub struct ConverseUsage {
    #[serde(rename = "inputTokens")]
    pub input_tokens: u64,
    #[serde(rename = "outputTokens")]
    pub output_tokens: u64,
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
pub fn converse_response_to_chat(
    resp: &ConverseResponse,
    model: &str,
    request_id: &str,
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

    let (prompt_tokens, completion_tokens, total_tokens) = resp
        .usage
        .as_ref()
        .map(|u| {
            (
                u.input_tokens,
                u.output_tokens,
                u.input_tokens + u.output_tokens,
            )
        })
        .unwrap_or((0, 0, 0));

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
            ..Default::default()
        },
    }
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
    use crate::domain::chat::Message;

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
            }),
        };
        let chat = converse_response_to_chat(
            &converse_resp,
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "req-001",
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
            }),
        };
        let chat = converse_response_to_chat(
            &converse_resp,
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "req-002",
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
            }),
        };
        let chat = converse_response_to_chat(&converse_resp, "model", "id");
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
