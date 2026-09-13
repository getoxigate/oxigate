// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! OpenAI <-> Gemini format translation.
//!
//! Pure functions, no I/O.

use bytes::Bytes;
use metrics::counter;
use thiserror::Error;
use tracing::warn;

use crate::domain::chat::{
    CacheAccounting, ChatRequest, ChatResponse, Choice, Message, MessageContent,
    ReasoningAccounting, Role, StreamChunk, ToolCall, ToolCallFunction, Usage, UsageAccounting,
};
use crate::domain::embedding::{EmbeddingData, EmbeddingResponse, EmbeddingUsage};
use crate::providers::gemini::types::EmbedContentItem;

use crate::domain::ports::ProviderError;
use crate::domain::tool_schema::{ToolChoiceKind, parse_tool_choice_value, truncate_for_error};
use crate::providers::tool_limits::{GEMINI_MAX_TOOLS, TOOL_ARGS_MAX_BYTES};

use super::types::{
    Content, FunctionCallingConfig, FunctionDeclaration, GeminiChatRequest, GeminiChatResponse,
    GeminiFunctionCall, GeminiFunctionResponse, GeminiTool, GenerationConfig, Part, ThinkingConfig,
    ThinkingLevel, ToolConfig, UsageMetadata,
};
use super::{ModelFlags, model_flags};
use crate::domain::chat::CompletionTokensDetails;
use crate::utils::sse;

/// Token accounting declared by the Google Gemini `generateContent` contract.
///
/// Cache is **inclusive**: `ai.google.dev/api/generate-content` documents `promptTokenCount` as
/// "the total effective prompt size meaning this includes the number of tokens in the cached
/// content" (accessed 2026-08-10), so the cached portion is subtracted from the reported prompt
/// before the remainder is charged at the full input rate.
///
/// Reasoning is **additive** and stays that way: the same page documents `totalTokenCount` as
/// "prompt + thoughts + response candidates", so `thoughtsTokenCount` sits outside
/// `candidatesTokenCount` — the opposite of OpenAI and Anthropic. Carving thoughts out of the
/// completion total here would turn a correct charge into an undercharge.
const GEMINI_ACCOUNTING: UsageAccounting = UsageAccounting {
    cache: CacheAccounting::Inclusive,
    reasoning: ReasoningAccounting::Additive,
};

/// Translation error.
#[derive(Debug, Error)]
pub enum TranslateError {
    #[allow(dead_code)]
    #[error("translation failed: {0}")]
    Failed(String),
    #[error("missing required field: {0}")]
    Missing(String),
    #[error("invalid thinking_level '{value}'; accepted values: LOW, MEDIUM, HIGH, MINIMAL")]
    InvalidThinkingLevel { value: String },
}

fn resolve_thinking_config(
    model: &str,
    extra: &serde_json::Map<String, serde_json::Value>,
    default_thinking_budget: i32,
) -> Result<Option<ThinkingConfig>, TranslateError> {
    let flags = model_flags(model);

    if flags.contains(ModelFlags::THINKING_BUDGET) {
        let budget = extra
            .get("thinking_budget")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
            .unwrap_or(default_thinking_budget);
        return Ok(Some(ThinkingConfig {
            thinking_budget: Some(budget),
            thinking_level: None,
        }));
    }

    if flags.contains(ModelFlags::THINKING_LEVEL) {
        let level = if let Some(s) = extra.get("thinking_level").and_then(|v| v.as_str()) {
            serde_json::from_value::<ThinkingLevel>(serde_json::Value::String(s.to_uppercase()))
                .map_err(|_| TranslateError::InvalidThinkingLevel {
                    value: s.to_owned(),
                })?
        } else {
            ThinkingLevel::Medium
        };
        return Ok(Some(ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some(level),
        }));
    }

    Ok(None)
}

/// Converts OpenAI ChatRequest to Gemini GenerateContent request.
pub fn openai_to_gemini(
    req: &ChatRequest,
    default_thinking_budget: i32,
) -> Result<GeminiChatRequest, ProviderError> {
    let mut system_instruction = None;
    let mut contents = Vec::new();
    let mut tool_call_ids: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    let te = |e: TranslateError| ProviderError::Translate(e.to_string());

    for msg in &req.messages {
        match &msg.role {
            Role::System => {
                let text = message_content_to_text(msg).map_err(te)?;
                system_instruction = Some(Content {
                    role: None,
                    parts: vec![Part::Text { text }],
                });
            }
            Role::User => {
                let parts = message_to_parts(msg).map_err(te)?;
                if !parts.is_empty() {
                    contents.push(Content {
                        role: Some("user".into()),
                        parts,
                    });
                }
            }
            Role::Assistant => {
                if let Some(ref tcs) = msg.tool_calls {
                    for tc in tcs {
                        tool_call_ids.insert(tc.id.clone(), tc.function.name.clone());
                        if tc.function.arguments.len() > TOOL_ARGS_MAX_BYTES {
                            return Err(ProviderError::InvalidRequest(format!(
                                "tool_call '{}' arguments exceed the {} KiB limit",
                                truncate_for_error(tc.id.clone()),
                                TOOL_ARGS_MAX_BYTES / 1024,
                            )));
                        }
                    }
                }
                let parts = assistant_message_to_parts(msg).map_err(te)?;
                if !parts.is_empty() {
                    contents.push(Content {
                        role: Some("model".into()),
                        parts,
                    });
                }
            }
            Role::Tool => {
                let parts = tool_message_to_parts(msg, &tool_call_ids)
                    .map_err(|e| ProviderError::InvalidRequest(e.to_string()))?;
                if !parts.is_empty() {
                    contents.push(Content {
                        role: Some("user".into()),
                        parts,
                    });
                }
            }
            Role::Other(_) => {
                let text = message_content_to_text(msg).unwrap_or_else(|_| String::new());
                if !text.is_empty() {
                    contents.push(Content {
                        role: Some(msg.role.as_str().to_string()),
                        parts: vec![Part::Text { text }],
                    });
                }
            }
        }
    }

    let tool_choice_val = req.extra.get("tool_choice");

    let is_none = crate::domain::tool_schema::is_tool_choice_none(tool_choice_val);

    let tools = if is_none {
        None
    } else {
        req.tools.as_ref().and_then(|tls| {
            let decls: Vec<FunctionDeclaration> = tls
                .iter()
                .filter_map(|t| {
                    if t.type_ == "function" {
                        Some(FunctionDeclaration {
                            name: t.function.name.clone(),
                            description: t.function.description.clone(),
                            parameters: t.function.parameters.clone(),
                        })
                    } else {
                        None
                    }
                })
                .collect();
            if decls.is_empty() {
                None
            } else {
                Some(vec![GeminiTool {
                    function_declarations: decls,
                }])
            }
        })
    };

    // Count check against function-type tools only — non-function types are silently dropped.
    if let Some(ref gem_tools) = tools {
        let count: usize = gem_tools
            .iter()
            .map(|t| t.function_declarations.len())
            .sum();
        if count > GEMINI_MAX_TOOLS {
            return Err(ProviderError::ToolCountExceeded {
                provider: "gemini",
                requested: count,
                limit: GEMINI_MAX_TOOLS,
            });
        }
    }

    let tool_config = if is_none {
        None
    } else {
        build_gemini_tool_config(tool_choice_val)?
    };

    let max_out = req.max_tokens.or(req.max_completion_tokens).or_else(|| {
        req.extra
            .get("max_completion_tokens")
            .and_then(|v| v.as_u64())
            .map(|u| u as u32)
    });

    let temperature = req.temperature.map(|f| f as f32);
    let top_p = req
        .extra
        .get("top_p")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    let stop = req.extra.get("stop").and_then(|v| {
        v.as_str().map(|s| vec![s.to_string()]).or_else(|| {
            v.as_array().map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
        })
    });

    let thinking_config =
        resolve_thinking_config(&req.model, &req.extra, default_thinking_budget).map_err(te)?;

    let needs_generation_config = max_out.is_some()
        || temperature.is_some()
        || top_p.is_some()
        || stop.is_some()
        || thinking_config.is_some();

    let generation_config = if needs_generation_config {
        Some(GenerationConfig {
            max_output_tokens: max_out.or(Some(8192)),
            temperature,
            top_p,
            top_k: None,
            candidate_count: Some(1),
            stop_sequences: stop,
            thinking_config,
        })
    } else {
        None
    };

    Ok(GeminiChatRequest {
        contents,
        tools,
        tool_config,
        generation_config,
        system_instruction,
    })
}

/// Maps an OpenAI `tool_choice` value to a Gemini `ToolConfig`.
/// Returns `None` when `tool_choice` is absent (no tool_config sent).
/// Returns `Err(ToolChoiceUnsupported)` for unrecognised values.
fn build_gemini_tool_config(
    val: Option<&serde_json::Value>,
) -> Result<Option<ToolConfig>, ProviderError> {
    // absent tool_choice → no toolConfig constraint sent to Gemini
    if val.is_none() {
        return Ok(None);
    }
    // G4: mode = ANY + allowed_function_names = [name] for named-function choice.
    match parse_tool_choice_value(val, "gemini")? {
        ToolChoiceKind::Auto => Ok(Some(ToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode: "AUTO".to_string(),
                allowed_function_names: None,
            },
        })),
        ToolChoiceKind::Required => Ok(Some(ToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode: "ANY".to_string(),
                allowed_function_names: None,
            },
        })),
        ToolChoiceKind::Function { name } => Ok(Some(ToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode: "ANY".to_string(),
                allowed_function_names: Some(vec![name]),
            },
        })),
    }
}

fn message_content_to_text(msg: &Message) -> Result<String, TranslateError> {
    match &msg.content {
        Some(MessageContent::Text(s)) => Ok(s.clone()),
        Some(MessageContent::Parts(parts)) => {
            let texts: Vec<String> = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|v| v.as_str()).map(String::from))
                .collect();
            Ok(texts.join("\n"))
        }
        None => Ok(String::new()),
    }
}

fn message_to_parts(msg: &Message) -> Result<Vec<Part>, TranslateError> {
    let text = message_content_to_text(msg)?;
    if text.is_empty() {
        return Ok(vec![]);
    }
    Ok(vec![Part::Text { text }])
}

fn assistant_message_to_parts(msg: &Message) -> Result<Vec<Part>, TranslateError> {
    let mut parts = Vec::new();

    if let Some(MessageContent::Text(ref t)) = msg.content
        && !t.is_empty()
    {
        parts.push(Part::Text { text: t.clone() });
    }

    if let Some(ref tool_calls) = msg.tool_calls {
        for tc in tool_calls {
            let args = match serde_json::from_str(&tc.function.arguments) {
                Ok(v) => Some(v),
                Err(e) => {
                    warn!(
                        tool_call_id = %tc.id,
                        arguments = %tc.function.arguments,
                        error = %e,
                        "tool call arguments are not valid JSON; forwarding null"
                    );
                    None
                }
            };
            parts.push(Part::FunctionCall {
                function_call: GeminiFunctionCall {
                    name: tc.function.name.clone(),
                    args,
                },
            });
        }
    }

    Ok(parts)
}

fn tool_message_to_parts(
    msg: &Message,
    tool_call_ids: &std::collections::HashMap<String, String>,
) -> Result<Vec<Part>, TranslateError> {
    let tool_call_id = msg
        .tool_call_id
        .as_ref()
        .ok_or_else(|| TranslateError::Missing("tool_call_id for tool message".into()))?;
    let name = match tool_call_ids.get(tool_call_id) {
        Some(n) => n.clone(),
        None => {
            // Lookup miss means the client sent a Role::Tool message referencing a tool_call_id
            // from a prior turn without including that assistant message in this request.
            // Gemini requires FunctionResponse.name to match a declared FunctionDeclaration.name;
            // using the raw ID as name would produce a guaranteed 4xx from Gemini.
            return Err(TranslateError::Missing(format!(
                "tool_call_id '{}' has no matching prior assistant tool_call in this request; \
                 include the full conversation history (assistant message with tool_calls[])",
                tool_call_id
            )));
        }
    };
    let response = message_content_to_text(msg).unwrap_or_else(|_| "{}".to_string());
    let response_json: serde_json::Value =
        serde_json::from_str(&response).unwrap_or(serde_json::json!({}));

    Ok(vec![Part::FunctionResponse {
        function_response: GeminiFunctionResponse {
            name,
            response: response_json,
        },
    }])
}

/// Converts Gemini response to OpenAI ChatResponse.
pub fn gemini_to_openai(
    resp: &GeminiChatResponse,
    model: &str,
    request_id: &str,
) -> Result<ChatResponse, TranslateError> {
    let candidate = resp
        .candidates
        .first()
        .ok_or_else(|| TranslateError::Missing("candidates".into()))?;

    let finish_reason = candidate.finish_reason.as_deref().map(map_finish_reason);

    if let Some(ref reason) = candidate.finish_reason
        && reason == "RECITATION"
    {
        counter!(
            "llm_provider_content_blocks_total",
            "provider" => "google",
            "reason" => "recitation"
        )
        .increment(1);
        warn!(
            provider = "google",
            reason = "recitation",
            "content blocked due to training data recitation"
        );
    }

    let (content, tool_calls) = match &candidate.content {
        Some(c) => {
            let mut text_parts = Vec::new();
            let mut tool_calls_out = Vec::new();
            let mut idx = 0u32;

            for part in &c.parts {
                match part {
                    Part::Thought { .. } => {
                        tracing::trace!("skipping thought part in gemini_to_openai translation");
                    }
                    Part::Text { text } => text_parts.push(text.clone()),
                    Part::FunctionCall { function_call } => {
                        let args = function_call
                            .args
                            .as_ref()
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "{}".to_string());
                        tool_calls_out.push(ToolCall {
                            id: format!("call_{request_id}_{idx}"),
                            type_: "function".to_string(),
                            function: ToolCallFunction {
                                name: function_call.name.clone(),
                                arguments: args,
                            },
                        });
                        idx += 1;
                    }
                    Part::ExecutableCode { executable_code } => {
                        text_parts.push(format!(
                            "[code] {}\n{}",
                            executable_code.language, executable_code.code
                        ));
                    }
                    Part::CodeExecutionResult {
                        code_execution_result,
                    } => {
                        let s = match code_execution_result.output.as_deref() {
                            Some(out) => {
                                format!("[output] {} {}", code_execution_result.outcome, out)
                            }
                            None => format!("[output] {}", code_execution_result.outcome),
                        };
                        text_parts.push(s);
                    }
                    _ => {
                        tracing::debug!(
                            part_type = ?std::mem::discriminant(part),
                            "non-Text/FunctionCall/Thought part in gemini_to_openai"
                        );
                    }
                }
            }

            let content = if text_parts.is_empty() {
                None
            } else {
                Some(MessageContent::Text(text_parts.join("")))
            };

            (
                content,
                if tool_calls_out.is_empty() {
                    None
                } else {
                    Some(tool_calls_out)
                },
            )
        }
        None => (None, None),
    };

    let usage = resp
        .usage_metadata
        .as_ref()
        .map(|u| usage_metadata_to_usage(u, GEMINI_ACCOUNTING))
        .unwrap_or_else(|| usage_metadata_to_usage(&UsageMetadata::default(), GEMINI_ACCOUNTING));

    let id = format!("chatcmpl-{request_id}");

    Ok(ChatResponse {
        id: id.clone(),
        object: "chat.completion".into(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
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
        usage,
    })
}

fn map_finish_reason(reason: &str) -> String {
    match reason {
        "STOP" => "stop".to_string(),
        "MAX_TOKENS" => "length".to_string(),
        "SAFETY" | "RECITATION" => "content_filter".to_string(),
        _ => "stop".to_string(),
    }
}

/// The single projection for the `generateContent` wire shape, buffered and streamed.
///
/// `accounting` is the contract the counts are billed under; every caller passes
/// `GEMINI_ACCOUNTING`. A response without `usageMetadata` is projected from
/// `UsageMetadata::default()`, which yields zero counts and no optional member.
fn usage_metadata_to_usage(u: &UsageMetadata, accounting: UsageAccounting) -> Usage {
    let prompt = u.prompt_token_count.unwrap_or(0) as u64;
    let completion = u.candidates_token_count.unwrap_or(0) as u64;
    let cached = u.cached_content_token_count.unwrap_or(0) as u64;
    let total = u
        .total_token_count
        .map(u64::from)
        .unwrap_or(prompt + completion);
    let completion_tokens_details = u.thoughts_token_count.map(|t| CompletionTokensDetails {
        reasoning_tokens: Some(t.into()),
    });
    Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: total,
        completion_tokens_details,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: if cached > 0 { Some(cached) } else { None },
        prompt_tokens_details: None,
        accounting,
        image_units: None,
        audio_seconds: None,
        ..Default::default()
    }
}

/// Converts a Gemini stream chunk to OpenAI SSE format.
/// `created` should be captured once before the stream starts so all chunks share the same timestamp.
///
/// Gemini Vertex sends usage_metadata in a separate chunk after the one with finish_reason.
/// When `usage_from` is provided, it supplies usage for the final chunk; otherwise `chunk` is used.
pub fn gemini_stream_chunk_to_sse(
    chunk: &GeminiChatResponse,
    model: &str,
    request_id: &str,
    created: u64,
    is_last: bool,
    usage_from: Option<&GeminiChatResponse>,
) -> Result<Option<StreamChunk>, TranslateError> {
    let candidate = chunk.candidates.first();

    let (delta_content, tool_calls_delta, finish_reason, usage) = if let Some(c) = candidate {
        let mut delta_content = String::new();
        let mut tool_calls_out: Vec<serde_json::Value> = Vec::new();
        let mut tc_idx = 0u32;

        for part in c.content.as_ref().map(|x| &x.parts).unwrap_or(&vec![]) {
            match part {
                Part::Text { text } => delta_content.push_str(text),
                Part::FunctionCall { function_call } => {
                    let args = function_call
                        .args
                        .as_ref()
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "{}".to_string());
                    tool_calls_out.push(serde_json::json!({
                        "index": tc_idx,
                        "id": format!("call_{request_id}_{tc_idx}"),
                        "type": "function",
                        "function": { "name": function_call.name, "arguments": args }
                    }));
                    tc_idx += 1;
                }
                _ => {}
            }
        }

        let finish = c.finish_reason.as_deref().map(map_finish_reason);

        if let Some(ref reason) = c.finish_reason
            && reason == "RECITATION"
        {
            counter!(
                "llm_provider_content_blocks_total",
                "provider" => "google",
                "reason" => "recitation"
            )
            .increment(1);
            warn!(
                provider = "google",
                reason = "recitation",
                "content blocked due to training data recitation"
            );
        }

        let usage = if is_last {
            usage_from
                .and_then(|u| u.usage_metadata.as_ref())
                .or(chunk.usage_metadata.as_ref())
                .map(|u| usage_metadata_to_usage(u, GEMINI_ACCOUNTING))
        } else {
            None
        };

        (delta_content, tool_calls_out, finish, usage)
    } else {
        let usage = if is_last {
            usage_from
                .and_then(|u| u.usage_metadata.as_ref())
                .or(chunk.usage_metadata.as_ref())
                .map(|u| usage_metadata_to_usage(u, GEMINI_ACCOUNTING))
        } else {
            None
        };
        (String::new(), vec![], None, usage)
    };

    // OpenAI SSE spec: finish_reason at choice level, not inside delta
    let mut delta = serde_json::json!({ "content": delta_content });
    if !tool_calls_delta.is_empty() {
        delta["tool_calls"] = serde_json::json!(tool_calls_delta);
    }
    let mut choice = serde_json::json!({
        "index": 0,
        "delta": delta
    });
    if let Some(ref fr) = finish_reason {
        choice["finish_reason"] = serde_json::json!(fr);
    }

    let mut chunk_json = sse::openai_chat_completion_envelope(created, model, request_id, choice);
    if let Some(ref u) = usage {
        let mut usage_obj = serde_json::json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        });
        if let Some(details) = &u.completion_tokens_details
            && let Some(reasoning) = details.reasoning_tokens
        {
            usage_obj["completion_tokens_details"] = serde_json::json!({
                "reasoning_tokens": reasoning
            });
        }
        chunk_json.insert("usage".to_string(), usage_obj);
    }

    let data = format!("data: {}\n\n", serde_json::Value::Object(chunk_json));
    let model_used = chunk
        .model_version
        .clone()
        .or_else(|| Some(model.to_string()));
    Ok(Some(StreamChunk::new(Bytes::from(data), usage, model_used)))
}

/// Converts Gemini embedding items to OpenAI format.
pub fn gemini_embedding_to_openai(
    embeddings: Vec<EmbedContentItem>,
    model: &str,
    token_count: u64,
) -> EmbeddingResponse {
    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(i, item)| EmbeddingData {
            object: "embedding".to_string(),
            embedding: item.values,
            index: i as u32,
        })
        .collect();

    EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: model.to_string(),
        usage: Some(EmbeddingUsage {
            prompt_tokens: token_count,
            total_tokens: token_count,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::MessageContent;
    use crate::domain::ports::TokenUsage;
    use crate::providers::gemini::types::Candidate;
    use crate::providers::usage_parity::assert_usage_parity;
    use crate::utils::cost_headers::build_cost_headers;

    fn pricing_holder() -> std::sync::Arc<std::sync::RwLock<crate::domain::pricing::PricingDb>> {
        std::sync::Arc::new(std::sync::RwLock::new(
            crate::domain::pricing::PricingDb::load(
                crate::domain::pricing::BUNDLED_PRICING_JSON,
                &crate::config::PricingConfig::default(),
            )
            .expect("bundled pricing must load"),
        ))
    }

    #[test]
    fn test_openai_tool_call_message_translated_to_function_call_part() {
        let req = ChatRequest {
            model: "gemini-2.0-flash".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: Some(MessageContent::Text("Weather?".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: Role::Assistant,
                    content: None,
                    tool_calls: Some(vec![ToolCall {
                        id: "call_abc".to_string(),
                        type_: "function".to_string(),
                        function: ToolCallFunction {
                            name: "get_weather".to_string(),
                            arguments: r#"{"location":"London"}"#.to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
            ],
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra: Default::default(),
        };
        let gemini = openai_to_gemini(&req, 0).expect("must translate");
        let model_content = gemini
            .contents
            .iter()
            .find(|c| c.role.as_deref() == Some("model"));
        let model_content = model_content.expect("must have model content");
        let func_call_part = model_content.parts.iter().find_map(|p| match p {
            Part::FunctionCall { function_call } => Some(function_call),
            _ => None,
        });
        let fc = func_call_part.expect("must have FunctionCall part");
        assert_eq!(fc.name, "get_weather");
        assert!(fc.args.is_some());
    }

    #[test]
    fn test_openai_tool_result_message_translated_to_function_response_part() {
        let req = ChatRequest {
            model: "gemini-2.0-flash".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: Some(MessageContent::Text("Weather?".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: Role::Assistant,
                    content: None,
                    tool_calls: Some(vec![ToolCall {
                        id: "call_xyz".to_string(),
                        type_: "function".to_string(),
                        function: ToolCallFunction {
                            name: "get_weather".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                Message {
                    role: Role::Tool,
                    content: Some(MessageContent::Text(r#"{"temp":22}"#.to_string())),
                    tool_calls: None,
                    tool_call_id: Some("call_xyz".to_string()),
                },
            ],
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra: Default::default(),
        };
        let gemini = openai_to_gemini(&req, 0).expect("must translate");
        let func_resp_part = gemini.contents.iter().find_map(|c| {
            c.parts.iter().find_map(|p| match p {
                Part::FunctionResponse { function_response } => Some(function_response),
                _ => None,
            })
        });
        let fr = func_resp_part.expect("must have FunctionResponse part");
        assert_eq!(fr.name, "get_weather");
        assert_eq!(fr.response.get("temp").and_then(|v| v.as_i64()), Some(22));
    }

    #[test]
    fn test_gemini_function_call_response_maps_to_tool_calls() {
        let resp = GeminiChatResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: Some("model".into()),
                    parts: vec![Part::FunctionCall {
                        function_call: GeminiFunctionCall {
                            name: "get_weather".to_string(),
                            args: Some(serde_json::json!({"location":"Paris"})),
                        },
                    }],
                }),
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }],
            usage_metadata: Some(UsageMetadata::default()),
            model_version: None,
            prompt_feedback: None,
        };
        let openai = gemini_to_openai(&resp, "gemini-2.0-flash", "req-1").expect("must translate");
        let tool_calls = openai
            .choices
            .first()
            .and_then(|c| c.message.tool_calls.as_ref())
            .expect("must have tool_calls");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert!(tool_calls[0].function.arguments.contains("Paris"));
    }

    #[test]
    fn test_openai_to_gemini_max_tokens_field() {
        let req = ChatRequest {
            model: "gemini-2.0-flash".into(),
            messages: vec![Message {
                role: Role::User,
                content: Some(MessageContent::Text("Hi".into())),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            max_tokens: Some(256),
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra: Default::default(),
        };
        let gemini = openai_to_gemini(&req, 0).expect("must translate");
        let gc = gemini
            .generation_config
            .as_ref()
            .expect("must have generation_config");
        assert_eq!(gc.max_output_tokens, Some(256));

        let req2 = ChatRequest {
            max_tokens: None,
            max_completion_tokens: Some(512),
            ..req.clone()
        };
        let gemini2 = openai_to_gemini(&req2, 0).expect("must translate");
        let gc2 = gemini2
            .generation_config
            .as_ref()
            .expect("must have generation_config");
        assert_eq!(gc2.max_output_tokens, Some(512));
    }

    #[test]
    fn test_thinking_config_budget_override_to_dynamic() {
        let mut extra = serde_json::Map::new();
        extra.insert("thinking_budget".into(), serde_json::json!(-1));
        let req = ChatRequest {
            model: "gemini-2.5-pro".into(),
            messages: vec![Message {
                role: Role::User,
                content: Some(MessageContent::Text("Hi".into())),
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
            extra,
        };
        let gemini = openai_to_gemini(&req, 0).expect("must translate");
        let tc = gemini
            .generation_config
            .as_ref()
            .and_then(|g| g.thinking_config.as_ref())
            .expect("must have thinking_config");
        assert_eq!(tc.thinking_budget, Some(-1), "per-request override wins");
        assert!(tc.thinking_level.is_none());
    }

    #[test]
    fn test_thinking_budget_override_from_extra() {
        let mut extra = serde_json::Map::new();
        extra.insert("thinking_budget".into(), serde_json::json!(1024));
        let req = ChatRequest {
            model: "gemini-2.5-flash".into(),
            messages: vec![Message {
                role: Role::User,
                content: Some(MessageContent::Text("Hi".into())),
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
            extra,
        };
        let gemini = openai_to_gemini(&req, 0).expect("must translate");
        let tc = gemini
            .generation_config
            .as_ref()
            .and_then(|g| g.thinking_config.as_ref())
            .expect("must have thinking_config");
        assert_eq!(tc.thinking_budget, Some(1024));
    }

    #[test]
    fn test_streaming_usage_no_completion_tokens_details_when_no_thinking() {
        let chunk = GeminiChatResponse {
            candidates: vec![],
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(5),
                total_token_count: Some(15),
                cached_content_token_count: None,
                thoughts_token_count: None,
            }),
            model_version: None,
            prompt_feedback: None,
        };
        let sse =
            gemini_stream_chunk_to_sse(&chunk, "gemini-2.0-flash", "req-1", 12345, true, None)
                .expect("must produce SSE");
        let sse = sse.expect("must have chunk");
        let data = String::from_utf8_lossy(&sse.data);
        assert!(
            !data.contains("completion_tokens_details"),
            "SSE must NOT have completion_tokens_details when no thinking tokens, got: {data}"
        );
    }

    #[test]
    fn test_usage_metadata_to_usage_populates_completion_tokens_details() {
        let resp = GeminiChatResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: Some("model".into()),
                    parts: vec![Part::Text {
                        text: "ok".to_string(),
                    }],
                }),
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }],
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: Some(100),
                candidates_token_count: Some(50),
                total_token_count: Some(250),
                cached_content_token_count: None,
                thoughts_token_count: Some(100),
            }),
            model_version: None,
            prompt_feedback: None,
        };
        let openai = gemini_to_openai(&resp, "gemini-2.5-pro", "req-1").expect("must translate");
        assert_eq!(
            openai
                .usage
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            Some(100)
        );
    }

    #[test]
    fn test_usage_metadata_to_usage_no_details_when_no_thoughts() {
        let resp = GeminiChatResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: Some("model".into()),
                    parts: vec![Part::Text {
                        text: "ok".to_string(),
                    }],
                }),
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }],
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(5),
                total_token_count: Some(15),
                cached_content_token_count: None,
                thoughts_token_count: None,
            }),
            model_version: None,
            prompt_feedback: None,
        };
        let openai = gemini_to_openai(&resp, "gemini-2.0-flash", "req-1").expect("must translate");
        assert!(openai.usage.completion_tokens_details.is_none());
    }

    /// `promptTokenCount` already contains the cached tokens, so treating cache reads as additive
    /// billed them at the full input rate inside the prompt *and* again at the cache rate — and
    /// counted them twice for tier selection, pushing the request into a higher-priced tier.
    ///
    /// `gemini-2.5-pro`, promptTokenCount 150,000 containing 100,000 cached, candidates 1,000.
    /// Previously 640_000_000 nano-USD at tier 1, then 197_500_000 at tier 0 once the double
    /// charge was fixed, and now 85_000_000 with the entry's imported `cache_read_multiplier`
    /// (0.1x) applied on top — every component is asserted, not only the total, so a
    /// compensating pair of errors cannot pass.
    #[test]
    fn test_cache_double_charge_regression() {
        let usage = Usage {
            prompt_tokens: 150_000,
            completion_tokens: 1_000,
            total_tokens: 151_000,
            cache_read_input_tokens: Some(100_000),
            accounting: GEMINI_ACCOUNTING,
            ..Default::default()
        };

        let (_, finalized) = build_cost_headers("gemini-2.5-pro", &usage, pricing_holder(), false);
        let (cost, token_usage) = (&finalized.cost, &finalized.token_usage);

        assert_eq!(
            token_usage.input_tokens, 50_000,
            "the cached portion is carved out of the reported prompt"
        );
        assert_eq!(
            token_usage.context_input_tokens(),
            150_000,
            "tier selection reads the reported prompt once, not prompt plus cached"
        );
        assert_eq!(cost.input_cost, crate::domain::ports::NanoUsd(62_500_000));
        assert_eq!(
            cost.cached_input_cost,
            crate::domain::ports::NanoUsd(12_500_000)
        );
        assert_eq!(cost.output_cost, crate::domain::ports::NanoUsd(10_000_000));
        assert_eq!(cost.total_cost, crate::domain::ports::NanoUsd(85_000_000));
    }

    /// Gemini's reasoning axis stays `Additive`, and that is a correction in its own right.
    ///
    /// `totalTokenCount` is documented as "prompt + thoughts + response candidates", so
    /// `thoughtsTokenCount` sits **outside** `candidatesTokenCount` — the opposite of OpenAI and
    /// Anthropic. Carving thoughts out of the completion total here would turn a correct charge
    /// into an undercharge.
    ///
    /// This is the guard against "finishing the job" by flipping Gemini alongside the others.
    #[test]
    fn test_reasoning_stays_additive_because_thoughts_sit_outside_candidates() {
        assert_eq!(GEMINI_ACCOUNTING.reasoning, ReasoningAccounting::Additive);

        let usage = Usage {
            prompt_tokens: 0,
            completion_tokens: 1_000,
            total_tokens: 1_800,
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: Some(800),
            }),
            accounting: GEMINI_ACCOUNTING,
            ..Default::default()
        };

        let (_, finalized) = build_cost_headers("gemini-2.5-pro", &usage, pricing_holder(), false);
        let token_usage = &finalized.token_usage;

        assert_eq!(
            token_usage.standard_output_tokens(),
            1_000,
            "candidatesTokenCount is charged whole; thoughts are charged beside it"
        );
    }

    /// Populated and absent `usageMetadata` apply the same accounting declaration, so the
    /// zero-usage case is not billed under the type default while the populated case is billed
    /// under Gemini's contract.
    #[test]
    fn test_both_usage_paths_apply_the_same_accounting() {
        let populated = gemini_to_openai(
            &text_response(Some(UsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(5),
                total_token_count: Some(15),
                cached_content_token_count: None,
                thoughts_token_count: None,
            })),
            "gemini-2.5-pro",
            "req-1",
        )
        .expect("must translate");
        let absent = gemini_to_openai(&text_response(None), "gemini-2.5-pro", "req-1")
            .expect("must translate");

        assert_eq!(populated.usage.accounting, GEMINI_ACCOUNTING);
        assert_eq!(absent.usage.accounting, GEMINI_ACCOUNTING);
    }

    /// A one-candidate text response carrying `usage_metadata`, or none.
    fn text_response(usage_metadata: Option<UsageMetadata>) -> GeminiChatResponse {
        GeminiChatResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: Some("model".into()),
                    parts: vec![Part::Text {
                        text: "ok".to_string(),
                    }],
                }),
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }],
            usage_metadata,
            model_version: None,
            prompt_feedback: None,
        }
    }

    /// A response with no `usageMetadata` projects exactly the value the translator has always
    /// produced for it: every count zero, no optional member, Gemini's accounting.
    ///
    /// The expected value is written out as a literal so the comparison is against a fixed
    /// value, not against whatever the projection happens to return for an empty input.
    #[test]
    fn test_absent_usage_metadata_projects_zero_usage_under_gemini_accounting() {
        let absent = gemini_to_openai(&text_response(None), "gemini-2.5-pro", "req-1")
            .expect("must translate");
        let expected = Usage {
            accounting: GEMINI_ACCOUNTING,
            ..Default::default()
        };

        assert_usage_parity(&expected, &absent.usage, "absent usageMetadata");
    }

    /// The buffered translator and both terminal stream arms — a chunk that still carries a
    /// candidate, and a candidate-less trailing chunk — project one `usageMetadata` identically,
    /// including when Vertex delivers it on a separate chunk (`usage_from`).
    #[test]
    fn test_buffered_and_streamed_usage_metadata_project_identically() {
        let metadata = UsageMetadata {
            prompt_token_count: Some(150_000),
            candidates_token_count: Some(1_000),
            total_token_count: Some(151_200),
            cached_content_token_count: Some(100_000),
            thoughts_token_count: Some(200),
        };
        let buffered = gemini_to_openai(
            &text_response(Some(metadata.clone())),
            "gemini-2.5-pro",
            "req-1",
        )
        .expect("must translate")
        .usage;

        let candidate_less = GeminiChatResponse {
            candidates: vec![],
            ..text_response(Some(metadata.clone()))
        };
        let streamed = |chunk: &GeminiChatResponse, usage_from: Option<&GeminiChatResponse>| {
            gemini_stream_chunk_to_sse(chunk, "gemini-2.5-pro", "req-1", 12345, true, usage_from)
                .expect("must produce SSE")
                .expect("the last chunk is emitted")
                .usage
                .expect("the last chunk carries usage")
        };

        assert_usage_parity(
            &buffered,
            &streamed(&text_response(Some(metadata.clone())), None),
            "terminal chunk with a candidate",
        );
        assert_usage_parity(
            &buffered,
            &streamed(&candidate_less, None),
            "candidate-less terminal chunk",
        );
        assert_usage_parity(
            &buffered,
            &streamed(&text_response(None), Some(&candidate_less)),
            "usage delivered on a separate chunk",
        );
    }

    /// Every downstream surface of one fully-populated Gemini response, pinned to absolute
    /// values.
    ///
    /// Priced against the synthetic fixture's fixed rates — $2/Mtok input, $10/Mtok output, cache
    /// reads at 0.5x input, no separate thinking rate — so a catalogue price refresh cannot move
    /// these values. promptTokenCount 150,000 containing 100,000 cached, candidates 1,000,
    /// thoughts 200:
    ///
    /// - input: (150,000 − 100,000) × 2,000 nano-USD = 100,000,000
    /// - cached input: 100,000 × 2,000 × 0.5 = 100,000,000
    /// - output: 1,000 × 10,000 = 10,000,000
    /// - thinking, additive beside the candidates: 200 × 10,000 = 2,000,000
    /// - total: 212,000,000
    ///
    /// **Two legs are deliberately not claimed:** the terminal `oxigate.usage` SSE event and the
    /// budget increment are emitted by the request handler, which nothing in this module can
    /// drive.
    #[test]
    fn test_every_accounted_surface_of_a_full_gemini_response_is_pinned() {
        use crate::domain::auth::RequestIdentity;
        use crate::domain::ports::NanoUsd;
        use crate::domain::spend::SpendRecord;
        use crate::domain::usage_accounting::CostStatus;
        use crate::providers::usage_parity::{SYNTHETIC_MODEL, synthetic_pricing_holder};
        use crate::utils::cost_headers::CostHeader;

        const MODEL: &str = SYNTHETIC_MODEL;
        let usage = gemini_to_openai(
            &text_response(Some(UsageMetadata {
                prompt_token_count: Some(150_000),
                candidates_token_count: Some(1_000),
                total_token_count: Some(151_200),
                cached_content_token_count: Some(100_000),
                thoughts_token_count: Some(200),
            })),
            MODEL,
            "req-1",
        )
        .expect("must translate")
        .usage;

        // --- Usage, every field: this shape reports no cache writes, so its cache-write
        // accounting is the empty default with no pricing generation attached ---
        let expected = Usage {
            prompt_tokens: 150_000,
            completion_tokens: 1_000,
            total_tokens: 151_200,
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: Some(200),
            }),
            cache_creation_input_tokens: None,
            cache_read_input_tokens: Some(100_000),
            prompt_tokens_details: None,
            accounting: GEMINI_ACCOUNTING,
            cache_write: Default::default(),
            image_units: None,
            audio_seconds: None,
        };
        assert_usage_parity(&expected, &usage, "full Gemini response");

        let (headers, finalized) =
            build_cost_headers(MODEL, &usage, synthetic_pricing_holder(), false);

        // --- TokenUsage, every field ---
        let tu = &finalized.token_usage;
        assert_eq!(tu.input_tokens, 50_000);
        assert_eq!(tu.output_tokens, 1_000);
        assert_eq!(tu.standard_output_tokens(), 1_000);
        assert_eq!(tu.cache_read_input_tokens, 100_000);
        assert_eq!(tu.cache_write.accounted_tokens(), 0);
        assert_eq!(tu.thinking_tokens, 200);
        assert_eq!(tu.image_count, 0);
        assert_eq!(tu.audio_seconds, 0.0);
        assert!(!tu.batch);
        assert_eq!(tu.reasoning_accounting, ReasoningAccounting::Additive);
        // Inclusive: the reported prompt already contains the cached portion.
        assert_eq!(tu.context_input_tokens(), 150_000);

        // --- Cost, every component ---
        let cost = &finalized.cost;
        assert_eq!(cost.input_cost, NanoUsd(100_000_000));
        assert_eq!(cost.cached_input_cost, NanoUsd(100_000_000));
        assert_eq!(cost.cache_write_cost, NanoUsd(0));
        assert_eq!(cost.output_cost, NanoUsd(10_000_000));
        assert_eq!(cost.thinking_cost, NanoUsd(2_000_000));
        assert_eq!(cost.image_cost, NanoUsd(0));
        assert_eq!(cost.audio_cost, NanoUsd(0));
        assert_eq!(cost.total_cost, NanoUsd(212_000_000));
        assert_eq!(cost.status, CostStatus::Exact);

        // --- The emitted headers ---
        assert_eq!(
            headers
                .get(CostHeader::REQUEST_COST)
                .and_then(|v| v.to_str().ok()),
            Some(cost.total_cost.to_display_string()).as_deref()
        );
        assert_eq!(
            headers
                .get(CostHeader::COST_STATUS)
                .and_then(|v| v.to_str().ok()),
            Some("exact")
        );

        // --- The persisted row, every column ---
        let record =
            SpendRecord::build(&RequestIdentity::default(), MODEL, "google", &finalized, 7);
        assert_eq!(record.org_id, "default");
        assert_eq!(record.identity_id, "default");
        assert_eq!(record.tags, serde_json::json!({}));
        assert_eq!(record.model, MODEL);
        assert_eq!(record.provider, "google");
        assert_eq!(record.prompt_tokens, 50_000);
        assert_eq!(record.completion_tokens, 1_000);
        assert_eq!(record.cache_read_tokens, 100_000);
        assert_eq!(record.thinking_tokens, 200);
        assert_eq!(record.cost_nano_usd, NanoUsd(212_000_000));
        assert_eq!(record.cost_status, CostStatus::Exact);
        assert!(
            record.usage_evidence.is_none(),
            "no cache write was reported, so there is no evidence document"
        );
        assert_eq!(record.latency_ms, 7);
    }

    /// Tier selection reads the total prompt context of a cached Gemini request.
    ///
    /// The payload is one Gemini can actually emit: `promptTokenCount` is documented as the total
    /// effective prompt size *including* the cached content, so a 250,000-token prompt containing
    /// 200,000 cached tokens is the shape to test. The earlier fixture asserted a 50,000 prompt
    /// alongside 200,000 cached tokens, which this contract cannot produce.
    #[test]
    fn test_cached_prompt_tier_comparator() {
        let resp = GeminiChatResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: Some("model".into()),
                    parts: vec![Part::Text {
                        text: "ok".to_string(),
                    }],
                }),
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }],
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: Some(250_000),
                candidates_token_count: Some(100),
                total_token_count: None,
                cached_content_token_count: Some(200_000),
                thoughts_token_count: None,
            }),
            model_version: None,
            prompt_feedback: None,
        };
        let openai = gemini_to_openai(&resp, "gemini-2.5-pro", "req-1").expect("must translate");
        let (_, finalized) =
            build_cost_headers("gemini-2.5-pro", &openai.usage, pricing_holder(), false);
        let token_usage = &finalized.token_usage;

        // The cached tokens are already inside `promptTokenCount`, so the comparator is the
        // reported prompt itself — counting them again selected a higher-priced tier.
        assert_eq!(token_usage.context_input_tokens(), 250_000);
    }

    /// The billable plain input of the same cached Gemini request.
    ///
    /// Paired with the comparator test above on an identical payload: the two dimensions move
    /// independently under a change of cache declaration, so each is asserted on its own.
    #[test]
    fn test_cached_prompt_billable_input() {
        let resp = GeminiChatResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: Some("model".into()),
                    parts: vec![Part::Text {
                        text: "ok".to_string(),
                    }],
                }),
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }],
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: Some(250_000),
                candidates_token_count: Some(100),
                total_token_count: None,
                cached_content_token_count: Some(200_000),
                thoughts_token_count: None,
            }),
            model_version: None,
            prompt_feedback: None,
        };
        let openai = gemini_to_openai(&resp, "gemini-2.5-pro", "req-1").expect("must translate");
        let (_, finalized) =
            build_cost_headers("gemini-2.5-pro", &openai.usage, pricing_holder(), false);
        let token_usage = &finalized.token_usage;

        // The cached portion is carved out of the reported prompt, so it is charged once at the
        // cache rate instead of at the full input rate as well.
        assert_eq!(token_usage.input_tokens, 50_000);
        assert_eq!(token_usage.cache_read_input_tokens, 200_000);
        assert_eq!(
            token_usage.input_tokens + token_usage.cache_read_input_tokens,
            250_000,
            "the two buckets partition the reported prompt; neither token is in both"
        );
    }

    #[test]
    fn test_thought_part_excluded_from_openai_translation() {
        let resp = GeminiChatResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: Some("model".into()),
                    parts: vec![
                        Part::Thought {
                            thought: true,
                            text: "internal reasoning".to_string(),
                        },
                        Part::Text {
                            text: "answer".to_string(),
                        },
                    ],
                }),
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }],
            usage_metadata: Some(UsageMetadata::default()),
            model_version: None,
            prompt_feedback: None,
        };
        let openai = gemini_to_openai(&resp, "gemini-2.5-pro", "req-1").expect("must translate");
        let content = openai
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref());
        match content {
            Some(MessageContent::Text(s)) => assert_eq!(s, "answer", "thought must be excluded"),
            _ => panic!("expected text content"),
        }
    }

    #[test]
    fn test_thinking_config_injected_for_gemini_25_pro() {
        let req = ChatRequest {
            model: "gemini-2.5-pro".into(),
            messages: vec![Message {
                role: Role::User,
                content: Some(MessageContent::Text("Hi".into())),
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
        };
        let gemini = openai_to_gemini(&req, 0).expect("must translate");
        let gc = gemini
            .generation_config
            .as_ref()
            .expect("must have generation_config");
        let tc = gc
            .thinking_config
            .as_ref()
            .expect("must have thinking_config");
        assert_eq!(tc.thinking_budget, Some(0));
        assert!(tc.thinking_level.is_none());
    }

    #[test]
    fn test_thinking_config_not_injected_for_gemini_20() {
        let req = ChatRequest {
            model: "gemini-2.0-flash".into(),
            messages: vec![Message {
                role: Role::User,
                content: Some(MessageContent::Text("Hi".into())),
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
        };
        let gemini = openai_to_gemini(&req, 0).expect("must translate");
        let thinking_config = gemini
            .generation_config
            .as_ref()
            .and_then(|g| g.thinking_config.as_ref());
        assert!(
            thinking_config.is_none(),
            "gemini-2.0 must not get ThinkingConfig"
        );
    }

    #[test]
    fn test_streaming_usage_includes_reasoning_tokens() {
        let chunk = GeminiChatResponse {
            candidates: vec![],
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(5),
                total_token_count: Some(215),
                cached_content_token_count: None,
                thoughts_token_count: Some(200),
            }),
            model_version: None,
            prompt_feedback: None,
        };
        let sse = gemini_stream_chunk_to_sse(&chunk, "gemini-2.5-pro", "req-1", 12345, true, None)
            .expect("must produce SSE");
        let sse = sse.expect("must have chunk");
        let data = String::from_utf8_lossy(&sse.data);
        assert!(
            data.contains("\"reasoning_tokens\":200"),
            "SSE must include reasoning_tokens in usage, got: {data}"
        );
    }

    #[test]
    fn test_openai_system_message_becomes_system_instruction() {
        let req = ChatRequest {
            model: "gemini-2.0-flash".into(),
            messages: vec![Message {
                role: Role::System,
                content: Some(MessageContent::Text("You are helpful.".into())),
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
        };
        let gemini = openai_to_gemini(&req, 0).expect("must translate");
        assert!(gemini.system_instruction.is_some());
        let si = gemini.system_instruction.as_ref().unwrap();
        assert_eq!(si.parts.len(), 1);
        if let Part::Text { text } = &si.parts[0] {
            assert_eq!(text, "You are helpful.");
        } else {
            panic!("expected text part");
        }
    }

    #[test]
    fn test_gemini_finish_reason_stop_maps_correctly() {
        let resp = GeminiChatResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: Some("model".into()),
                    parts: vec![Part::Text {
                        text: "Hi".to_string(),
                    }],
                }),
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }],
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: Some(1),
                candidates_token_count: Some(2),
                total_token_count: Some(3),
                cached_content_token_count: None,
                thoughts_token_count: None,
            }),
            model_version: None,
            prompt_feedback: None,
        };
        let openai = gemini_to_openai(&resp, "gemini-2.0-flash", "req-1").expect("must translate");
        assert_eq!(openai.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn test_gemini_finish_reason_safety_maps_to_content_filter() {
        let resp = GeminiChatResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: Some("model".into()),
                    parts: vec![Part::Text {
                        text: "".to_string(),
                    }],
                }),
                finish_reason: Some("SAFETY".into()),
                index: Some(0),
            }],
            usage_metadata: Some(UsageMetadata::default()),
            model_version: None,
            prompt_feedback: None,
        };
        let openai = gemini_to_openai(&resp, "gemini-2.0-flash", "req-1").expect("must translate");
        assert_eq!(
            openai.choices[0].finish_reason.as_deref(),
            Some("content_filter")
        );
    }

    #[test]
    fn test_gemini_thinking_tokens_tracked_separately() {
        let resp = GeminiChatResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: Some("model".into()),
                    parts: vec![Part::Text {
                        text: "42".to_string(),
                    }],
                }),
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }],
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(5),
                total_token_count: Some(20),
                cached_content_token_count: None,
                thoughts_token_count: Some(5),
            }),
            model_version: None,
            prompt_feedback: None,
        };
        let openai = gemini_to_openai(&resp, "gemini-2.5-pro", "req-1").expect("must translate");
        // thoughts_token_count is included in total_token_count; completion_tokens = candidates_token_count
        assert_eq!(openai.usage.completion_tokens, 5);
        assert_eq!(openai.usage.prompt_tokens, 10);
        assert_eq!(openai.usage.total_tokens, 20);
        // thoughts_token_count must flow to Usage.completion_tokens_details.reasoning_tokens for cost calculation
        assert_eq!(
            openai
                .usage
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            Some(5),
            "reasoning_tokens must be wired for Gemini 2.5 billing"
        );
        // Assert the same path used by build_cost_headers reaches TokenUsage (internal cost struct)
        let thinking_tokens = openai
            .usage
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens)
            .unwrap_or(0);
        let token_usage = TokenUsage {
            input_tokens: openai.usage.prompt_tokens,
            output_tokens: openai.usage.completion_tokens,
            thinking_tokens,
            ..Default::default()
        };
        assert_eq!(
            token_usage.thinking_tokens, 5,
            "TokenUsage.thinking_tokens must receive thoughts_token_count for cost calculation"
        );
    }

    #[test]
    fn test_missing_usage_metadata_does_not_panic() {
        let resp = GeminiChatResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: Some("model".into()),
                    parts: vec![Part::Text {
                        text: "ok".to_string(),
                    }],
                }),
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }],
            usage_metadata: None,
            model_version: None,
            prompt_feedback: None,
        };
        let openai = gemini_to_openai(&resp, "gemini-2.0-flash", "req-1").expect("must translate");
        assert_eq!(openai.usage.prompt_tokens, 0);
        assert_eq!(openai.usage.completion_tokens, 0);
        assert_eq!(openai.usage.total_tokens, 0);
    }

    // ── F4: orphaned tool_call_id tests ─────────────────────────────────────────

    #[test]
    fn test_tool_message_without_tool_call_id_returns_err() {
        // A Role::Tool message with no tool_call_id is a malformed client request.
        let req = crate::domain::chat::ChatRequest {
            model: "gemini-2.0-flash".into(),
            messages: vec![
                crate::domain::chat::Message {
                    role: crate::domain::chat::Role::User,
                    content: Some(crate::domain::chat::MessageContent::Text("hi".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                crate::domain::chat::Message {
                    role: crate::domain::chat::Role::Tool,
                    content: Some(crate::domain::chat::MessageContent::Text("{}".into())),
                    tool_calls: None,
                    tool_call_id: None, // missing — must error
                },
            ],
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra: Default::default(),
        };
        let err = openai_to_gemini(&req, 0).expect_err("missing tool_call_id must be Err");
        match &err {
            crate::domain::ports::ProviderError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("tool_call_id"),
                    "error should name the missing field: {msg}"
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn test_tool_message_with_orphaned_tool_call_id_returns_err() {
        // tool_call_id present but no matching assistant tool_call in this request.
        use crate::domain::chat::*;
        let req = ChatRequest {
            model: "gemini-2.0-flash".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: Some(MessageContent::Text("Weather?".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                // No assistant message with tool_calls — so lookup map is empty.
                Message {
                    role: Role::Tool,
                    content: Some(MessageContent::Text(r#"{"temp":22}"#.to_string())),
                    tool_calls: None,
                    tool_call_id: Some("call_orphan".to_string()),
                },
            ],
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra: Default::default(),
        };
        let err = openai_to_gemini(&req, 0).expect_err("orphaned tool_call_id must be Err");
        match &err {
            crate::domain::ports::ProviderError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("call_orphan"),
                    "error message should include the ID: {msg}"
                );
                assert!(
                    msg.contains("no matching prior assistant tool_call"),
                    "error should explain cause: {msg}"
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn test_gemini_tool_args_over_limit_returns_invalid_request() {
        use crate::domain::chat::{Message, MessageContent, Role, ToolCall, ToolCallFunction};
        use crate::providers::tool_limits::TOOL_ARGS_MAX_BYTES;

        let oversized = "z".repeat(TOOL_ARGS_MAX_BYTES + 1);
        let req = crate::domain::chat::ChatRequest {
            model: "gemini-2.0-flash".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: Some(MessageContent::Text("call it".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: Role::Assistant,
                    content: None,
                    tool_calls: Some(vec![ToolCall {
                        id: "call_big".into(),
                        type_: "function".into(),
                        function: ToolCallFunction {
                            name: "big_func".into(),
                            arguments: oversized,
                        },
                    }]),
                    tool_call_id: None,
                },
            ],
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra: Default::default(),
        };
        let err = openai_to_gemini(&req, 0).expect_err("over-limit args must error");
        assert!(
            matches!(err, crate::domain::ports::ProviderError::InvalidRequest(_)),
            "expected InvalidRequest, got {err:?}"
        );
    }
}
