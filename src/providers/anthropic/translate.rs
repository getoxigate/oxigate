// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! OpenAI <-> Anthropic format translation.
//!
//! Pure functions for request/response and streaming translation.

use std::collections::HashMap;

use bytes::Bytes;
use serde_json;
use tracing::{debug, warn};

use crate::domain::chat::{
    ChatRequest, ChatResponse, Choice, CompletionTokensDetails, Message, MessageContent, Role,
    StreamChunk, ToolCall, ToolCallFunction, Usage,
};
use crate::domain::ports::ProviderError;
use crate::domain::tool_schema::{ERR_TOOL_CALL_BUFFER_OVERFLOW, ERR_TYPE_GATEWAY_ERROR};
use crate::domain::tool_schema::{ToolChoiceKind, parse_tool_choice_value, truncate_for_error};
use crate::providers::tool_limits::{ANTHROPIC_MAX_TOOLS, TOOL_ARGS_MAX_BYTES};
use crate::utils::sse;

use super::types::{
    AnthropicMessage, AnthropicTool, AnthropicToolChoice, AnthropicUsage, ContentBlock,
    MessagesRequest, MessagesResponse, StreamEvent, ThinkingConfig,
};

pub(crate) const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Translates OpenAI ChatRequest to Anthropic MessagesRequest.
pub fn chat_request_to_anthropic(
    req: &ChatRequest,
    default_model: &str,
    default_max_tokens: u32,
) -> Result<MessagesRequest, ProviderError> {
    let model = if req.model.is_empty() {
        default_model.to_string()
    } else {
        req.model.clone()
    };

    let (system, messages) = extract_system_and_messages(&req.messages)?;

    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or_else(|| {
            debug!("anthropic: request omits max_tokens and max_completion_tokens, using default_max_tokens={}", default_max_tokens);
            default_max_tokens
        });

    let stop_sequences = stop_from_extra(&req.extra);

    let (tools, tool_choice) = tools_from_request(req)?;

    let thinking = thinking_from_extra(&req.extra);

    Ok(MessagesRequest {
        model,
        max_tokens,
        system,
        messages,
        tools,
        tool_choice,
        stop_sequences,
        stream: req.stream,
        thinking,
    })
}

fn extract_system_and_messages(
    openai_messages: &[Message],
) -> Result<(Option<String>, Vec<AnthropicMessage>), ProviderError> {
    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<AnthropicMessage> = Vec::new();
    // tool_call_id → function name, built from prior assistant turns for orphan detection.
    let mut tool_call_ids: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    for msg in openai_messages {
        match &msg.role {
            Role::System => {
                let text = message_content_to_text(msg);
                if !text.is_empty() {
                    system_parts.push(text);
                }
            }
            Role::User => {
                let blocks = message_to_content_blocks(msg)?;
                if !blocks.is_empty() {
                    messages.push(AnthropicMessage {
                        role: msg.role.as_str().to_string(),
                        content: blocks,
                    });
                }
            }
            Role::Assistant => {
                if let Some(ref tcs) = msg.tool_calls {
                    for tc in tcs {
                        tool_call_ids.insert(tc.id.clone(), tc.function.name.clone());
                    }
                }
                let blocks = message_to_content_blocks(msg)?;
                if !blocks.is_empty() {
                    messages.push(AnthropicMessage {
                        role: msg.role.as_str().to_string(),
                        content: blocks,
                    });
                }
            }
            Role::Tool => {
                // Anthropic requires tool results as user-role messages with tool_result blocks.
                let content = message_content_to_text(msg);
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
                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                    }],
                });
            }
            Role::Other(_) => {
                let text = message_content_to_text(msg);
                if !text.is_empty() {
                    messages.push(AnthropicMessage {
                        role: msg.role.as_str().to_string(),
                        content: vec![ContentBlock::Text { text }],
                    });
                }
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };

    Ok((system, messages))
}

fn message_content_to_text(msg: &Message) -> String {
    match &msg.content {
        Some(MessageContent::Text(s)) => s.clone(),
        Some(MessageContent::Parts(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        None => String::new(),
    }
}

fn message_to_content_blocks(msg: &Message) -> Result<Vec<ContentBlock>, ProviderError> {
    let mut blocks = Vec::new();

    // Add text content if present (OpenAI clients may send both text and tool_calls)
    match &msg.content {
        Some(MessageContent::Text(s)) if !s.is_empty() => {
            blocks.push(ContentBlock::Text { text: s.clone() });
        }
        Some(MessageContent::Parts(_)) => {
            let text = message_content_to_text(msg);
            if !text.is_empty() {
                blocks.push(ContentBlock::Text { text });
            }
        }
        _ => {}
    }

    // Add tool calls if present
    if let Some(ref tcs) = msg.tool_calls {
        for tc in tcs {
            if tc.function.arguments.len() > TOOL_ARGS_MAX_BYTES {
                return Err(ProviderError::InvalidRequest(format!(
                    "tool_call '{}' arguments exceed the {} KiB limit",
                    truncate_for_error(tc.id.clone()),
                    TOOL_ARGS_MAX_BYTES / 1024,
                )));
            }
            let input = match serde_json::from_str::<serde_json::Value>(&tc.function.arguments) {
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
            blocks.push(ContentBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input,
            });
        }
    }

    Ok(blocks)
}

fn stop_from_extra(extra: &serde_json::Map<String, serde_json::Value>) -> Option<Vec<String>> {
    let stop = extra.get("stop")?;
    match stop {
        serde_json::Value::String(s) => Some(vec![s.clone()]),
        serde_json::Value::Array(arr) => {
            let seqs: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            if seqs.is_empty() { None } else { Some(seqs) }
        }
        _ => None,
    }
}

fn tools_from_request(
    req: &ChatRequest,
) -> Result<(Option<Vec<AnthropicTool>>, Option<AnthropicToolChoice>), ProviderError> {
    let tool_choice_val = req.extra.get("tool_choice");

    if crate::domain::tool_schema::is_tool_choice_none(tool_choice_val) {
        return Ok((None, None));
    }

    let raw_tools = match req.tools.as_ref() {
        Some(tls) if !tls.is_empty() => tls,
        _ => return Ok((None, None)),
    };

    let tools: Vec<AnthropicTool> = raw_tools
        .iter()
        .filter(|t| t.type_ == "function")
        .map(|t| AnthropicTool {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            input_schema: t
                .function
                .parameters
                .clone()
                .unwrap_or(serde_json::json!({})),
        })
        .collect();

    if tools.is_empty() {
        return Ok((None, None));
    }

    if tools.len() > ANTHROPIC_MAX_TOOLS {
        return Err(ProviderError::ToolCountExceeded {
            provider: "anthropic",
            requested: tools.len(),
            limit: ANTHROPIC_MAX_TOOLS,
        });
    }

    let tool_choice = map_anthropic_tool_choice(tool_choice_val)?;
    Ok((Some(tools), Some(tool_choice)))
}

/// Maps an OpenAI `tool_choice` value to an Anthropic `AnthropicToolChoice`.
fn map_anthropic_tool_choice(
    val: Option<&serde_json::Value>,
) -> Result<AnthropicToolChoice, ProviderError> {
    match parse_tool_choice_value(val, "anthropic")? {
        ToolChoiceKind::Auto => Ok(AnthropicToolChoice::Auto),
        ToolChoiceKind::Required => Ok(AnthropicToolChoice::Any),
        ToolChoiceKind::Function { name } => Ok(AnthropicToolChoice::Tool { name }),
    }
}

fn thinking_from_extra(
    extra: &serde_json::Map<String, serde_json::Value>,
) -> Option<ThinkingConfig> {
    let thinking = extra.get("thinking")?;
    let budget = match thinking {
        serde_json::Value::Number(n) => {
            let u = n.as_u64().or_else(|| {
                n.as_i64()
                    .and_then(|i| if i >= 0 { Some(i as u64) } else { None })
            })?;
            if u > u32::MAX as u64 {
                debug!(
                    "anthropic: thinking budget {} exceeds u32::MAX, clamping",
                    u
                );
                u32::MAX
            } else {
                u as u32
            }
        }
        _ => {
            debug!(
                "anthropic: extra.thinking must be a positive number, got {:?}; ignoring",
                thinking
            );
            return None;
        }
    };
    Some(ThinkingConfig {
        type_: "enabled".to_string(),
        budget_tokens: budget,
    })
}

/// Maps Anthropic stop_reason to OpenAI finish_reason.
fn map_stop_reason(reason: Option<&str>) -> String {
    match reason {
        Some("end_turn") | Some("stop_sequence") => "stop".to_string(),
        Some("max_tokens") => "length".to_string(),
        Some("tool_use") => "tool_calls".to_string(),
        _ => "stop".to_string(),
    }
}

/// Translates Anthropic MessagesResponse to OpenAI ChatResponse.
///
/// `cap_bytes`: per-call tool-argument buffer cap. Returns `ToolCallBufferOverflow`
/// when a single `tool_use.input` serialises to more bytes than the cap.
pub fn anthropic_to_chat_response(
    resp: &MessagesResponse,
    model: &str,
    request_id: &str,
    cap_bytes: usize,
) -> Result<ChatResponse, ProviderError> {
    let mut content_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut reasoning_tokens: Option<u64> = None;

    for block in &resp.content {
        match block {
            ContentBlock::Text { text } => content_parts.push(text.clone()),
            ContentBlock::ToolUse { id, name, input } => {
                let args = serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
                if args.len() > cap_bytes {
                    tracing::error!(
                        provider = "anthropic",
                        tool_call_id = %id,
                        cap_bytes,
                        actual_bytes = args.len(),
                        "tool call input exceeded buffer cap (non-streaming)"
                    );
                    return Err(ProviderError::ToolCallBufferOverflow {
                        provider: "anthropic",
                        tool_call_id: id.clone(),
                        cap_bytes,
                    });
                }
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    type_: "function".to_string(),
                    function: ToolCallFunction {
                        name: name.clone(),
                        arguments: args,
                    },
                });
            }
            ContentBlock::Thinking { thinking: _ } => {
                if let Some(ref details) = resp.usage.output_tokens_details {
                    reasoning_tokens = details.thinking_tokens;
                }
                debug!(
                    "anthropic: stripping thinking block, reasoning_tokens={:?}",
                    reasoning_tokens
                );
            }
            ContentBlock::ToolResult { .. } => {
                // tool_result blocks are request-side only; Anthropic never returns them.
                debug!("anthropic: unexpected tool_result block in response; skipping");
            }
        }
    }

    let content = if content_parts.is_empty() && tool_calls.is_empty() {
        None
    } else if content_parts.len() == 1 && tool_calls.is_empty() {
        Some(MessageContent::Text(
            content_parts
                .into_iter()
                .next()
                .expect("infallible: len checked above"),
        ))
    } else if !content_parts.is_empty() {
        Some(MessageContent::Text(content_parts.join("")))
    } else {
        None
    };

    let usage = anthropic_usage_to_usage(&resp.usage, reasoning_tokens);

    let finish_reason = map_stop_reason(resp.stop_reason.as_deref());

    Ok(ChatResponse {
        id: format!("chatcmpl-{}", request_id),
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
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                tool_call_id: None,
            },
            finish_reason: Some(finish_reason),
        }],
        usage,
    })
}

fn anthropic_usage_to_usage(u: &AnthropicUsage, reasoning_tokens: Option<u64>) -> Usage {
    let reasoning = reasoning_tokens.or_else(|| {
        u.output_tokens_details
            .as_ref()
            .and_then(|d| d.thinking_tokens)
    });
    let completion_tokens_details = reasoning.map(|r| CompletionTokensDetails {
        reasoning_tokens: Some(r),
    });
    let total = u.input_tokens + u.output_tokens;
    // Extract cache creation breakdown by TTL
    // If cache_creation object is missing but cache_creation_input_tokens is present,
    // fall back to 5m bucket (legacy semantics) to avoid undercharging.
    let (cache_creation_5m, cache_creation_1h) = u
        .cache_creation
        .as_ref()
        .map(|cc| (cc.ephemeral_5m_input_tokens, cc.ephemeral_1h_input_tokens))
        .unwrap_or_else(|| {
            // Fallback: attribute all cache creation to 5m bucket
            (u.cache_creation_input_tokens.unwrap_or(0), 0)
        });
    // Invariant guard: 5m + 1h should equal total (debug-only check for API drift)
    if let Some(total_cache) = u.cache_creation_input_tokens {
        let sum = cache_creation_5m + cache_creation_1h;
        debug_assert!(
            sum == total_cache,
            "cache_creation breakdown sum ({sum}) != total ({total_cache}); Anthropic API may have changed"
        );
        if sum != total_cache {
            tracing::warn!(
                total = total_cache,
                sum,
                "cache_creation breakdown sum != total; using breakdown values (API drift?)"
            );
        }
    }
    Usage {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
        total_tokens: total,
        completion_tokens_details,
        cache_creation_input_tokens: u.cache_creation_input_tokens,
        cache_read_input_tokens: u.cache_read_input_tokens,
        prompt_tokens_details: None,
        tier_threshold_override: None,
        cache_accounting: crate::domain::chat::CacheAccounting::Additive,
        image_units: None,
        audio_seconds: None,
        cache_creation_5m_tokens: cache_creation_5m,
        cache_creation_1h_tokens: cache_creation_1h,
    }
}

/// Accumulates per-block state for a single concurrent tool call during streaming.
struct ToolAccumulator {
    /// OpenAI `tool_calls[index]` assigned monotonically at `ContentBlockStart`.
    openai_index: u32,
    id: String,
    name: String,
    /// Running byte count of `partial_json` chunks seen — checked against cap, not buffered.
    bytes_seen: usize,
}

/// Error returned by `StreamTranslator::process_event`.
#[derive(Debug)]
pub enum StreamErr {
    /// Anthropic SSE error event (optional message).
    ProviderError(Option<String>),
    /// Tool-argument buffer cap exceeded. Always mid-stream for Anthropic streaming.
    BufferOverflow(ProviderError),
}

/// Stateful translator for Anthropic SSE stream -> OpenAI SSE.
pub struct StreamTranslator {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_5m_tokens: Option<u64>,
    cache_creation_1h_tokens: Option<u64>,
    reasoning_tokens: Option<u64>,
    /// Concurrent tool-call accumulators keyed by Anthropic SSE block index.
    /// Entries are removed by ContentBlockStop. On mid-stream network drop the map is not
    /// explicitly drained — but it is dropped with the StreamTranslator at request end, so
    /// this is request-scoped memory, not a process-level leak.
    tool_blocks: HashMap<u32, ToolAccumulator>,
    /// Monotonic counter assigns each new ToolUse block a unique OpenAI `index`.
    next_openai_index: u32,
    /// Per-call buffer cap for tool-argument JSON (bytes). Set once at construction.
    cap_bytes: usize,
    emitted_role: bool,
    created: u64,
    model: String,
    request_id: String,
}

impl StreamTranslator {
    pub fn new(model: String, request_id: String, cap_bytes: usize) -> Self {
        Self {
            input_tokens: None,
            output_tokens: None,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            cache_creation_5m_tokens: None,
            cache_creation_1h_tokens: None,
            reasoning_tokens: None,
            tool_blocks: HashMap::new(),
            next_openai_index: 0,
            cap_bytes,
            emitted_role: false,
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model,
            request_id,
        }
    }

    /// Process an Anthropic stream event and optionally emit an OpenAI-format chunk.
    pub fn process_event(&mut self, event: &StreamEvent) -> Result<Option<StreamChunk>, StreamErr> {
        match event {
            StreamEvent::MessageStart { message } => {
                if let Some(ref u) = message.usage {
                    self.input_tokens = Some(u.input_tokens);
                    self.cache_creation_input_tokens = u.cache_creation_input_tokens;
                    self.cache_read_input_tokens = u.cache_read_input_tokens;
                    // Extract cache creation breakdown by TTL
                    // If cache_creation object is missing, fall back to 5m bucket (legacy semantics)
                    if let Some(ref cc) = u.cache_creation {
                        self.cache_creation_5m_tokens = Some(cc.ephemeral_5m_input_tokens);
                        self.cache_creation_1h_tokens = Some(cc.ephemeral_1h_input_tokens);
                    } else if let Some(total) = u.cache_creation_input_tokens {
                        // Fallback: attribute all to 5m bucket
                        self.cache_creation_5m_tokens = Some(total);
                        self.cache_creation_1h_tokens = Some(0);
                    }
                    // Invariant guard: 5m + 1h should equal total (debug-only check for API drift)
                    if let Some(total_cache) = u.cache_creation_input_tokens {
                        let sum_5m = self.cache_creation_5m_tokens.unwrap_or(0);
                        let sum_1h = self.cache_creation_1h_tokens.unwrap_or(0);
                        let sum = sum_5m + sum_1h;
                        debug_assert!(
                            sum == total_cache,
                            "cache_creation breakdown sum ({sum}) != total ({total_cache}); Anthropic API may have changed"
                        );
                        if sum != total_cache {
                            tracing::warn!(
                                total = total_cache,
                                sum,
                                "cache_creation breakdown sum != total in message_start; using breakdown values (API drift?)"
                            );
                        }
                    }
                }
                Ok(None)
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                super::types::ContentBlockStartBlock::Text => {
                    if !self.emitted_role {
                        self.emitted_role = true;
                        let chunk = make_delta_chunk(
                            self.created,
                            &self.model,
                            &self.request_id,
                            Some("assistant"),
                            Some(""),
                            None,
                            None,
                        );
                        Ok(Some(chunk))
                    } else {
                        Ok(None)
                    }
                }
                super::types::ContentBlockStartBlock::ToolUse { id, name, .. } => {
                    let openai_index = self.next_openai_index;
                    self.next_openai_index += 1;
                    self.tool_blocks.insert(
                        *index as u32,
                        ToolAccumulator {
                            openai_index,
                            id: id.clone(),
                            name: name.clone(),
                            bytes_seen: 0,
                        },
                    );
                    if !self.emitted_role {
                        self.emitted_role = true;
                        let role_chunk = make_delta_chunk(
                            self.created,
                            &self.model,
                            &self.request_id,
                            Some("assistant"),
                            Some(""),
                            None,
                            None,
                        );
                        let tool_chunk = make_tool_call_delta_chunk(
                            self.created,
                            &self.model,
                            &self.request_id,
                            id,
                            name,
                            "",
                            openai_index,
                        );
                        Ok(Some(StreamChunk::new(
                            Bytes::from(
                                [role_chunk.data.as_ref(), tool_chunk.data.as_ref()].concat(),
                            ),
                            None,
                            Some(self.model.clone()),
                        )))
                    } else {
                        let chunk = make_tool_call_delta_chunk(
                            self.created,
                            &self.model,
                            &self.request_id,
                            id,
                            name,
                            "",
                            openai_index,
                        );
                        Ok(Some(chunk))
                    }
                }
                super::types::ContentBlockStartBlock::Thinking => Ok(None),
            },
            StreamEvent::ContentBlockDelta { index, delta } => match delta {
                super::types::ContentBlockDelta::Thinking { .. } => {
                    debug!(
                        "anthropic: stripping thinking_delta block (content dropped, tokens surfaced via usage)"
                    );
                    Ok(None)
                }
                super::types::ContentBlockDelta::Text { text } => {
                    let chunk = make_delta_chunk(
                        self.created,
                        &self.model,
                        &self.request_id,
                        None,
                        Some(text),
                        None,
                        None,
                    );
                    Ok(Some(chunk))
                }
                super::types::ContentBlockDelta::InputJson { partial_json } => {
                    let acc = match self.tool_blocks.get_mut(&(*index as u32)) {
                        Some(a) => a,
                        None => {
                            warn!(
                                block_index = index,
                                "InputJson delta for unknown tool block; skipping"
                            );
                            return Ok(None);
                        }
                    };
                    // bytes_seen counts raw partial_json chunk bytes as a conservative proxy for
                    // the final serialized argument size. Anthropic streams compact JSON so the
                    // difference is negligible; we intentionally over-count rather than under-count.
                    acc.bytes_seen += partial_json.len();
                    if acc.bytes_seen > self.cap_bytes {
                        tracing::error!(
                            provider = "anthropic",
                            tool_call_id = %acc.id,
                            cap_bytes = self.cap_bytes,
                            bytes_seen = acc.bytes_seen,
                            "tool call buffer cap exceeded (mid-stream)"
                        );
                        return Err(StreamErr::BufferOverflow(
                            ProviderError::ToolCallBufferOverflow {
                                provider: "anthropic",
                                tool_call_id: acc.id.clone(),
                                cap_bytes: self.cap_bytes,
                            },
                        ));
                    }
                    let openai_index = acc.openai_index;
                    let id = acc.id.clone();
                    let name = acc.name.clone();
                    let chunk = make_tool_call_delta_chunk(
                        self.created,
                        &self.model,
                        &self.request_id,
                        &id,
                        &name,
                        partial_json,
                        openai_index,
                    );
                    Ok(Some(chunk))
                }
            },
            StreamEvent::ContentBlockStop { index } => {
                self.tool_blocks.remove(&(*index as u32));
                Ok(None)
            }
            StreamEvent::MessageDelta { delta } => {
                if let Some(ref u) = delta.usage {
                    self.output_tokens = Some(u.output_tokens);
                    self.cache_creation_input_tokens = u
                        .cache_creation_input_tokens
                        .or(self.cache_creation_input_tokens);
                    self.cache_read_input_tokens =
                        u.cache_read_input_tokens.or(self.cache_read_input_tokens);
                    // Extract cache creation breakdown by TTL
                    //
                    // IMPORTANT: Anthropic sends CUMULATIVE (repeated) values in message_delta,
                    // not incremental. The same cache_creation values appear in both message_start
                    // and message_delta events.
                    //
                    // We use assignment (overwrite) instead of saturating_add to avoid double-counting.
                    // This was confirmed via LangChain's bug report on cache token double-counting:
                    // https://github.com/langchain-ai/langchainjs/issues/10249
                    //
                    // If Anthropic changes their API to send incremental values in the future,
                    // the debug_assert! on sum == total will catch the drift.
                    if let Some(ref cc) = u.cache_creation {
                        self.cache_creation_5m_tokens = Some(cc.ephemeral_5m_input_tokens);
                        self.cache_creation_1h_tokens = Some(cc.ephemeral_1h_input_tokens);
                    } else if let Some(total) = u.cache_creation_input_tokens {
                        // Fallback: attribute all to 5m bucket, but ONLY if we haven't
                        // already received a breakdown from message_start.
                        if self.cache_creation_5m_tokens.is_none() {
                            self.cache_creation_5m_tokens = Some(total);
                            self.cache_creation_1h_tokens = Some(0);
                        }
                    }
                    // Invariant guard: verify 5m+1h==total in message_delta
                    if let Some(ref d) = u.output_tokens_details {
                        self.reasoning_tokens = d.thinking_tokens.or(self.reasoning_tokens);
                    }
                }
                let usage = self.build_usage();
                let finish_reason = map_stop_reason(delta.stop_reason.as_deref());
                let chunk = make_delta_chunk(
                    self.created,
                    &self.model,
                    &self.request_id,
                    None,
                    None,
                    Some(&finish_reason),
                    Some(&usage),
                );
                Ok(Some(chunk))
            }
            StreamEvent::MessageStop => {
                let chunk = StreamChunk::new(
                    Bytes::from_static(b"data: [DONE]\n\n"),
                    None,
                    Some(self.model.clone()),
                );
                Ok(Some(chunk))
            }
            StreamEvent::Error { error } => Err(StreamErr::ProviderError(error.message.clone())),
            StreamEvent::Ping => Ok(None),
        }
    }

    fn build_usage(&self) -> Usage {
        let input = self.input_tokens.unwrap_or(0);
        let output = self.output_tokens.unwrap_or(0);
        let completion_tokens_details = self.reasoning_tokens.map(|r| CompletionTokensDetails {
            reasoning_tokens: Some(r),
        });
        Usage {
            prompt_tokens: input,
            completion_tokens: output,
            total_tokens: input + output,
            completion_tokens_details,
            cache_creation_input_tokens: self.cache_creation_input_tokens,
            cache_read_input_tokens: self.cache_read_input_tokens,
            prompt_tokens_details: None,
            tier_threshold_override: None,
            cache_accounting: crate::domain::chat::CacheAccounting::Additive,
            image_units: None,
            audio_seconds: None,
            cache_creation_5m_tokens: self.cache_creation_5m_tokens.unwrap_or(0),
            cache_creation_1h_tokens: self.cache_creation_1h_tokens.unwrap_or(0),
        }
    }
}

fn make_delta_chunk(
    created: u64,
    model: &str,
    request_id: &str,
    role: Option<&str>,
    content: Option<&str>,
    finish_reason: Option<&str>,
    usage: Option<&Usage>,
) -> StreamChunk {
    let mut delta = serde_json::Map::new();
    if let Some(r) = role {
        delta.insert("role".to_string(), serde_json::Value::String(r.to_string()));
    }
    if let Some(c) = content {
        delta.insert(
            "content".to_string(),
            serde_json::Value::String(c.to_string()),
        );
    }

    let mut choice = serde_json::Map::new();
    choice.insert("index".to_string(), serde_json::json!(0));
    choice.insert("delta".to_string(), serde_json::Value::Object(delta));
    choice.insert(
        "finish_reason".to_string(),
        finish_reason
            .map(|fr| serde_json::Value::String(fr.to_string()))
            .unwrap_or(serde_json::Value::Null),
    );

    let choice_value = serde_json::Value::Object(choice);
    let root = sse::openai_chat_completion_envelope(created, model, request_id, choice_value);

    let data = match serde_json::to_string(&root) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "SSE delta chunk serialization failed, emitting empty data");
            String::new()
        }
    };
    let data = format!("data: {data}\n\n");
    let usage = usage.cloned();
    StreamChunk::new(Bytes::from(data), usage, Some(model.to_string()))
}

fn make_tool_call_delta_chunk(
    created: u64,
    model: &str,
    request_id: &str,
    tool_id: &str,
    tool_name: &str,
    arguments_delta: &str,
    openai_index: u32,
) -> StreamChunk {
    let mut func = serde_json::Map::new();
    func.insert("name".to_string(), serde_json::json!(tool_name));
    func.insert("arguments".to_string(), serde_json::json!(arguments_delta));

    let mut tc = serde_json::Map::new();
    tc.insert("index".to_string(), serde_json::json!(openai_index));
    tc.insert("id".to_string(), serde_json::json!(tool_id));
    tc.insert("type".to_string(), serde_json::json!("function"));
    tc.insert("function".to_string(), serde_json::Value::Object(func));

    let mut delta = serde_json::Map::new();
    delta.insert("tool_calls".to_string(), serde_json::json!([tc]));

    let mut choice = serde_json::Map::new();
    choice.insert("index".to_string(), serde_json::json!(0));
    choice.insert("delta".to_string(), serde_json::Value::Object(delta));
    choice.insert("finish_reason".to_string(), serde_json::Value::Null);

    let choice_value = serde_json::Value::Object(choice);
    let root = sse::openai_chat_completion_envelope(created, model, request_id, choice_value);

    let data = match serde_json::to_string(&root) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "SSE tool_call chunk serialization failed, emitting empty data");
            String::new()
        }
    };
    let data = format!("data: {data}\n\n");
    StreamChunk::new(Bytes::from(data), None, Some(model.to_string()))
}

/// Builds the terminal SSE error event for mid-stream tool-call buffer overflow.
///
/// Emitted as the last chunk before stream close — no `[DONE]` follows.
pub fn overflow_sse_event(e: &ProviderError) -> StreamChunk {
    let (provider, tool_call_id, cap_bytes) = match e {
        ProviderError::ToolCallBufferOverflow {
            provider,
            tool_call_id,
            cap_bytes,
        } => (*provider, tool_call_id.as_str(), *cap_bytes),
        _ => ("unknown", "unknown", 0),
    };
    let payload = serde_json::json!({
        "error": {
            "message": "tool call JSON exceeded the per-call buffer cap",
            "type": ERR_TYPE_GATEWAY_ERROR,
            "code": ERR_TOOL_CALL_BUFFER_OVERFLOW,
            "provider": provider,
            "tool_call_id": tool_call_id,
            "cap_bytes": cap_bytes,
        }
    });
    let data = format!(
        "data: {}\n\n",
        serde_json::to_string(&payload).unwrap_or_default()
    );
    StreamChunk::new(Bytes::from(data), None, None)
}

/// Parse a single SSE event line into StreamEvent.
pub fn parse_stream_event(line: &str) -> Option<StreamEvent> {
    let line = line.trim();
    if line.starts_with("data: ") {
        let payload = line.strip_prefix("data: ")?;
        if payload == "[DONE]" {
            return None;
        }
        serde_json::from_str(payload).ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::anthropic::types::{
        AnthropicUsage, ContentBlock, MessagesResponse, OutputTokensDetails,
    };

    fn chat_request(
        messages: Vec<Message>,
        extra: serde_json::Map<String, serde_json::Value>,
    ) -> ChatRequest {
        ChatRequest {
            model: "claude-sonnet-4-5-20251022".into(),
            messages,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra,
        }
    }

    #[test]
    fn test_system_message_lifted() {
        let extra = serde_json::Map::new();
        let req = chat_request(
            vec![Message {
                role: Role::System,
                content: Some(MessageContent::Text("You are helpful.".into())),
                tool_calls: None,
                tool_call_id: None,
            }],
            extra,
        );
        let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect("must translate");
        assert_eq!(out.system, Some("You are helpful.".into()));
        assert!(out.messages.is_empty());
    }

    #[test]
    fn test_multiple_system_messages_concatenated() {
        let extra = serde_json::Map::new();
        let req = chat_request(
            vec![
                Message {
                    role: Role::System,
                    content: Some(MessageContent::Text("First.".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: Role::System,
                    content: Some(MessageContent::Text("Second.".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            extra,
        );
        let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect("must translate");
        assert_eq!(out.system, Some("First.\n\nSecond.".into()));
    }

    #[test]
    fn test_max_completion_tokens_takes_precedence() {
        let extra = serde_json::Map::new();
        let req = ChatRequest {
            model: "claude-sonnet-4-5-20251022".into(),
            messages: vec![Message {
                role: Role::User,
                content: Some(MessageContent::Text("Hi".into())),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            max_tokens: Some(100),
            max_completion_tokens: Some(200),
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra,
        };
        let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect("must translate");
        assert_eq!(out.max_tokens, 200);
    }

    #[test]
    fn test_max_tokens_default_applied() {
        let extra = serde_json::Map::new();
        let req = chat_request(
            vec![Message {
                role: Role::User,
                content: Some(MessageContent::Text("Hi".into())),
                tool_calls: None,
                tool_call_id: None,
            }],
            extra,
        );
        let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 2048)
            .expect("must translate");
        assert_eq!(out.max_tokens, 2048);
    }

    #[test]
    fn test_stop_string_to_array() {
        let mut extra = serde_json::Map::new();
        extra.insert("stop".into(), serde_json::json!("foo"));
        let req = chat_request(
            vec![Message {
                role: Role::User,
                content: Some(MessageContent::Text("Hi".into())),
                tool_calls: None,
                tool_call_id: None,
            }],
            extra,
        );
        let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect("must translate");
        assert_eq!(out.stop_sequences, Some(vec!["foo".into()]));
    }

    #[test]
    fn test_stop_array_passthrough() {
        let mut extra = serde_json::Map::new();
        extra.insert("stop".into(), serde_json::json!(["foo", "bar"]));
        let req = chat_request(
            vec![Message {
                role: Role::User,
                content: Some(MessageContent::Text("Hi".into())),
                tool_calls: None,
                tool_call_id: None,
            }],
            extra,
        );
        let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect("must translate");
        assert_eq!(out.stop_sequences, Some(vec!["foo".into(), "bar".into()]));
    }

    #[test]
    fn test_stop_reason_mapping() {
        assert_eq!(map_stop_reason(Some("end_turn")), "stop");
        assert_eq!(map_stop_reason(Some("max_tokens")), "length");
        assert_eq!(map_stop_reason(Some("tool_use")), "tool_calls");
        assert_eq!(map_stop_reason(Some("stop_sequence")), "stop");
    }

    #[test]
    fn test_cache_tokens_surfaced() {
        let resp = MessagesResponse {
            id: "msg_01".into(),
            type_: Some("message".into()),
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text { text: "Hi".into() }],
            stop_reason: Some("end_turn".into()),
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: Some(2),
                cache_read_input_tokens: Some(3),
                output_tokens_details: None,
                cache_creation: None,
            },
        };
        let chat =
            anthropic_to_chat_response(&resp, "claude-sonnet-4-5-20251022", "req-1", usize::MAX)
                .expect("must translate");
        assert_eq!(chat.usage.cache_creation_input_tokens, Some(2));
        assert_eq!(chat.usage.cache_read_input_tokens, Some(3));
        // No breakdown object → fall back to 5m bucket (legacy semantics)
        assert_eq!(chat.usage.cache_creation_5m_tokens, 2);
        assert_eq!(chat.usage.cache_creation_1h_tokens, 0);
    }

    #[test]
    fn test_cache_creation_1h_breakdown() {
        //: Anthropic reports cache creation breakdown by TTL
        use crate::providers::anthropic::types::CacheCreationBreakdown;
        let resp = MessagesResponse {
            id: "msg_01".into(),
            type_: Some("message".into()),
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text { text: "Hi".into() }],
            stop_reason: Some("end_turn".into()),
            usage: AnthropicUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_creation_input_tokens: Some(3500),
                cache_read_input_tokens: Some(2000),
                output_tokens_details: None,
                cache_creation: Some(CacheCreationBreakdown {
                    ephemeral_5m_input_tokens: 1000,
                    ephemeral_1h_input_tokens: 2500,
                }),
            },
        };
        let chat =
            anthropic_to_chat_response(&resp, "claude-sonnet-4-5-20251022", "req-1", usize::MAX)
                .expect("must translate");
        // Total cache creation preserved
        assert_eq!(chat.usage.cache_creation_input_tokens, Some(3500));
        // Breakdown correctly split
        assert_eq!(chat.usage.cache_creation_5m_tokens, 1000);
        assert_eq!(chat.usage.cache_creation_1h_tokens, 2500);
    }

    #[test]
    fn test_cache_creation_fallback_to_5m_when_breakdown_absent() {
        //: If cache_creation object is missing but cache_creation_input_tokens
        // is present, fall back to 5m bucket (legacy semantics) to avoid undercharging.
        let resp = MessagesResponse {
            id: "msg_01".into(),
            type_: Some("message".into()),
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text { text: "Hi".into() }],
            stop_reason: Some("end_turn".into()),
            usage: AnthropicUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_creation_input_tokens: Some(3500),
                cache_read_input_tokens: Some(2000),
                output_tokens_details: None,
                cache_creation: None, // No breakdown object
            },
        };
        let chat =
            anthropic_to_chat_response(&resp, "claude-sonnet-4-5-20251022", "req-1", usize::MAX)
                .expect("must translate");
        // Total cache creation preserved
        assert_eq!(chat.usage.cache_creation_input_tokens, Some(3500));
        // All attributed to 5m bucket (fallback)
        assert_eq!(chat.usage.cache_creation_5m_tokens, 3500);
        assert_eq!(chat.usage.cache_creation_1h_tokens, 0);
    }

    #[test]
    fn test_tool_translation_request() {
        let extra = serde_json::Map::new();
        let req = ChatRequest {
            model: "claude-sonnet-4-5-20251022".into(),
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
            tools: Some(vec![crate::domain::chat::Tool {
                type_: "function".into(),
                function: crate::domain::chat::ToolFunction {
                    name: "get_weather".into(),
                    description: Some("Get weather".into()),
                    parameters: Some(serde_json::json!({"type":"object","properties":{"city":{}}})),
                },
            }]),
            parallel_tool_calls: None,
            request_id: None,
            extra,
        };
        let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect("must translate");
        let tools = out.tools.expect("tools present");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert!(tools[0].input_schema.get("type").is_some());
    }

    #[test]
    fn test_assistant_message_with_text_and_tool_calls() {
        let extra = serde_json::Map::new();
        let req = ChatRequest {
            model: "claude-sonnet-4-5-20251022".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: Some(MessageContent::Text("Weather in NYC?".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: Role::Assistant,
                    content: Some(MessageContent::Text("I'll check that for you.".into())),
                    tool_calls: Some(vec![crate::domain::chat::ToolCall {
                        id: "toolu_01".into(),
                        type_: "function".into(),
                        function: crate::domain::chat::ToolCallFunction {
                            name: "get_weather".into(),
                            arguments: r#"{"city":"NYC"}"#.into(),
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
            extra,
        };
        let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect("must translate");
        let assistant_msg = out
            .messages
            .iter()
            .find(|m| m.role == "assistant")
            .expect("assistant message");
        assert_eq!(assistant_msg.content.len(), 2);
        match &assistant_msg.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "I'll check that for you."),
            _ => panic!("first block must be Text"),
        }
        match &assistant_msg.content[1] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_01");
                assert_eq!(name, "get_weather");
                assert_eq!(input.get("city").and_then(|v| v.as_str()), Some("NYC"));
            }
            _ => panic!("second block must be ToolUse"),
        }
    }

    #[test]
    fn test_tool_translation_response() {
        let resp = MessagesResponse {
            id: "msg_01".into(),
            type_: Some("message".into()),
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "I'll check.".into(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_01".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"city":"NYC"}),
                },
            ],
            stop_reason: Some("tool_use".into()),
            usage: AnthropicUsage {
                input_tokens: 5,
                output_tokens: 10,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                output_tokens_details: None,
                cache_creation: None,
            },
        };
        let chat =
            anthropic_to_chat_response(&resp, "claude-sonnet-4-5-20251022", "req-1", usize::MAX)
                .expect("must translate");
        let tcs = chat.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("tool_calls");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "toolu_01");
        assert_eq!(tcs[0].function.name, "get_weather");
        assert_eq!(tcs[0].function.arguments, r#"{"city":"NYC"}"#);
        assert_eq!(chat.choices[0].finish_reason, Some("tool_calls".into()));
    }

    #[test]
    fn test_tool_choice_none_removes_tools() {
        let mut extra = serde_json::Map::new();
        extra.insert("tool_choice".into(), serde_json::json!({"type": "none"}));
        let req = ChatRequest {
            model: "claude-sonnet-4-5-20251022".into(),
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
            tools: Some(vec![crate::domain::chat::Tool {
                type_: "function".into(),
                function: crate::domain::chat::ToolFunction {
                    name: "x".into(),
                    description: None,
                    parameters: None,
                },
            }]),
            parallel_tool_calls: None,
            request_id: None,
            extra,
        };
        let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect("must translate");
        assert!(out.tools.is_none());
        assert!(out.tool_choice.is_none());
    }

    #[test]
    fn test_tool_choice_specific() {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "tool_choice".into(),
            serde_json::json!({"type": "function", "function": {"name": "get_weather"}}),
        );
        let req = ChatRequest {
            model: "claude-sonnet-4-5-20251022".into(),
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
            tools: Some(vec![crate::domain::chat::Tool {
                type_: "function".into(),
                function: crate::domain::chat::ToolFunction {
                    name: "get_weather".into(),
                    description: None,
                    parameters: None,
                },
            }]),
            parallel_tool_calls: None,
            request_id: None,
            extra,
        };
        let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect("must translate");
        match out.tool_choice.as_ref().expect("tool_choice") {
            AnthropicToolChoice::Tool { name } => assert_eq!(name, "get_weather"),
            _ => panic!("expected Tool {{ name }}"),
        }
    }

    // ──: orphaned tool_call_id guard ──────────────────────────────────

    fn make_request_with_tool_call(tool_call_id: &str) -> ChatRequest {
        ChatRequest {
            model: "claude-sonnet-4-5-20251022".into(),
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
                        id: tool_call_id.to_string(),
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
                    tool_call_id: Some(tool_call_id.to_string()),
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
        }
    }

    #[test]
    fn test_matched_tool_call_id_translates_ok() {
        let req = make_request_with_tool_call("call_abc");
        let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect("must translate");
        // The assistant message must be present and contain the toolUse block.
        let assistant = out
            .messages
            .iter()
            .find(|m| m.role == "assistant")
            .expect("assistant message must be present");
        assert!(
            assistant
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { .. })),
            "assistant message must contain a tool_use block"
        );
    }

    #[test]
    fn test_pure_tool_call_assistant_message_emits_tool_use_block() {
        let req = ChatRequest {
            model: "claude-sonnet-4-5-20251022".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: Some(MessageContent::Text("call the function".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: Role::Assistant,
                    content: None,
                    tool_calls: Some(vec![ToolCall {
                        id: "tu_01".into(),
                        type_: "function".into(),
                        function: ToolCallFunction {
                            name: "my_func".into(),
                            arguments: r#"{"x":1}"#.into(),
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
        let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect("must translate");
        let assistant = out
            .messages
            .iter()
            .find(|m| m.role == "assistant")
            .expect("pure-tool-call assistant message must be present");
        assert_eq!(assistant.content.len(), 1);
        match &assistant.content[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "tu_01");
                assert_eq!(name, "my_func");
                assert_eq!(input.get("x").and_then(|v| v.as_i64()), Some(1));
            }
            other => panic!("expected ToolUse block, got {other:?}"),
        }
    }

    #[test]
    fn test_orphaned_tool_call_id_returns_invalid_request() {
        let req = ChatRequest {
            model: "claude-sonnet-4-5-20251022".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: Some(MessageContent::Text("hi".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                // No assistant message with tool_calls — lookup map is empty.
                Message {
                    role: Role::Tool,
                    content: Some(MessageContent::Text("{}".into())),
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
        let err = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect_err("orphaned ID must error");
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
        let long_id = "x".repeat(300);
        let req = ChatRequest {
            model: "claude-sonnet-4-5-20251022".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: Some(MessageContent::Text("hi".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: Role::Tool,
                    content: Some(MessageContent::Text("{}".into())),
                    tool_calls: None,
                    tool_call_id: Some(long_id.clone()),
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
        let err = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect_err("orphaned long ID must error");
        match &err {
            ProviderError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("no matching prior assistant tool_call"),
                    "{msg}"
                );
                assert!(
                    msg.contains("<truncated>"),
                    "300-byte ID must be truncated in error: {msg}"
                );
                assert!(
                    msg.len() < 512,
                    "error message must be bounded, got {} bytes",
                    msg.len()
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn test_missing_tool_call_id_returns_invalid_request() {
        let req = ChatRequest {
            model: "claude-sonnet-4-5-20251022".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: Some(MessageContent::Text("hi".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: Role::Tool,
                    content: Some(MessageContent::Text("{}".into())),
                    tool_calls: None,
                    tool_call_id: None, // None — must error
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
        let err = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect_err("missing ID must error");
        assert!(
            matches!(err, ProviderError::InvalidRequest(_)),
            "expected InvalidRequest, got {err:?}"
        );
    }

    #[test]
    fn test_tool_args_over_limit_returns_invalid_request() {
        use crate::providers::tool_limits::TOOL_ARGS_MAX_BYTES;
        let oversized = "x".repeat(TOOL_ARGS_MAX_BYTES + 1);
        let req = ChatRequest {
            model: "claude-sonnet-4-5-20251022".into(),
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
        let err = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect_err("over-limit args must error");
        assert!(
            matches!(err, ProviderError::InvalidRequest(_)),
            "expected InvalidRequest, got {err:?}"
        );
    }

    #[test]
    fn test_thinking_from_extra_validates_type() {
        let mut extra = serde_json::Map::new();
        extra.insert("thinking".into(), serde_json::json!(1000));
        let req = chat_request(
            vec![Message {
                role: Role::User,
                content: Some(MessageContent::Text("Hi".into())),
                tool_calls: None,
                tool_call_id: None,
            }],
            extra,
        );
        let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
            .expect("must translate");
        assert!(out.thinking.is_some());
        assert_eq!(out.thinking.as_ref().unwrap().budget_tokens, 1000);

        for (label, val) in [
            ("bool", serde_json::json!(true)),
            ("string", serde_json::json!("1000")),
            ("negative", serde_json::json!(-5)),
            ("object", serde_json::json!({"budget": 1000})),
        ] {
            let mut extra = serde_json::Map::new();
            extra.insert("thinking".into(), val);
            let req = chat_request(
                vec![Message {
                    role: Role::User,
                    content: Some(MessageContent::Text("Hi".into())),
                    tool_calls: None,
                    tool_call_id: None,
                }],
                extra,
            );
            let out = chat_request_to_anthropic(&req, "claude-sonnet-4-5-20251022", 4096)
                .expect("must translate");
            assert!(
                out.thinking.is_none(),
                "extra.thinking={} should be ignored",
                label
            );
        }
    }

    #[test]
    fn test_thinking_tokens_surfaced() {
        let resp = MessagesResponse {
            id: "msg_01".into(),
            type_: Some("message".into()),
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Thinking {
                    thinking: "internal reasoning...".into(),
                },
                ContentBlock::Text {
                    text: "The answer is 42.".into(),
                },
            ],
            stop_reason: Some("end_turn".into()),
            usage: AnthropicUsage {
                input_tokens: 5,
                output_tokens: 20,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                output_tokens_details: Some(OutputTokensDetails {
                    thinking_tokens: Some(15),
                }),
                cache_creation: None,
            },
        };
        let chat =
            anthropic_to_chat_response(&resp, "claude-sonnet-4-5-20251022", "req-1", usize::MAX)
                .expect("must translate");
        assert_eq!(
            chat.usage
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            Some(15)
        );
    }

    #[test]
    fn test_thinking_content_stripped() {
        let resp = MessagesResponse {
            id: "msg_01".into(),
            type_: Some("message".into()),
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Thinking {
                    thinking: "internal reasoning...".into(),
                },
                ContentBlock::Text {
                    text: "The answer is 42.".into(),
                },
            ],
            stop_reason: Some("end_turn".into()),
            usage: AnthropicUsage {
                input_tokens: 5,
                output_tokens: 20,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                output_tokens_details: Some(OutputTokensDetails {
                    thinking_tokens: Some(15),
                }),
                cache_creation: None,
            },
        };
        let chat =
            anthropic_to_chat_response(&resp, "claude-sonnet-4-5-20251022", "req-1", usize::MAX)
                .expect("must translate");
        let content = chat.choices[0].message.content.as_ref().expect("content");
        let text = match content {
            MessageContent::Text(s) => s.as_str(),
            _ => "",
        };
        assert_eq!(text, "The answer is 42.");
        assert!(!text.contains("internal reasoning"));
    }

    // --- StreamTranslator unit tests ---

    fn stream_event(s: &str) -> StreamEvent {
        parse_stream_event(&format!("data: {s}")).expect("valid event")
    }

    #[test]
    fn test_stream_text_delta() {
        let ev = stream_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
        );
        let mut tr = StreamTranslator::new("claude".into(), "rid".into(), usize::MAX);
        tr.emitted_role = true;
        let out = tr.process_event(&ev).unwrap();
        let chunk = out.expect("chunk");
        let s = String::from_utf8_lossy(&chunk.data);
        assert!(s.contains("Hello"));
        assert!(s.contains("content"));
    }

    #[test]
    fn test_stream_message_start_extracts_input_tokens() {
        let ev = stream_event(
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":42,"output_tokens":0}}}"#,
        );
        let mut tr = StreamTranslator::new("claude".into(), "rid".into(), usize::MAX);
        let out = tr.process_event(&ev).unwrap();
        assert!(out.is_none());
        assert_eq!(tr.input_tokens, Some(42));
    }

    /// Streaming cache creation breakdown — MessageStart with cache_creation object.
    #[test]
    fn test_stream_message_start_extracts_cache_creation_breakdown() {
        let ev = stream_event(
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":100,"output_tokens":0,"cache_creation_input_tokens":3500,"cache_read_input_tokens":2000,"cache_creation":{"ephemeral_5m_input_tokens":1000,"ephemeral_1h_input_tokens":2500}}}}"#,
        );
        let mut tr = StreamTranslator::new(
            "claude-sonnet-4-5-20251022".into(),
            "rid".into(),
            usize::MAX,
        );
        let out = tr.process_event(&ev).unwrap();
        assert!(out.is_none());
        assert_eq!(tr.input_tokens, Some(100));
        assert_eq!(tr.cache_creation_input_tokens, Some(3500));
        assert_eq!(tr.cache_read_input_tokens, Some(2000));
        assert_eq!(tr.cache_creation_5m_tokens, Some(1000));
        assert_eq!(tr.cache_creation_1h_tokens, Some(2500));
    }

    /// Streaming cache creation fallback when cache_creation object is absent.
    #[test]
    fn test_stream_message_start_fallback_to_5m_without_breakdown() {
        let ev = stream_event(
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":100,"output_tokens":0,"cache_creation_input_tokens":3500,"cache_read_input_tokens":2000}}}"#,
        );
        let mut tr = StreamTranslator::new(
            "claude-sonnet-4-5-20251022".into(),
            "rid".into(),
            usize::MAX,
        );
        let out = tr.process_event(&ev).unwrap();
        assert!(out.is_none());
        assert_eq!(tr.cache_creation_input_tokens, Some(3500));
        // Fallback: all to 5m bucket
        assert_eq!(tr.cache_creation_5m_tokens, Some(3500));
        assert_eq!(tr.cache_creation_1h_tokens, Some(0));
    }

    #[test]
    fn test_stream_message_delta_emits_usage() {
        let start = stream_event(
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":10,"output_tokens":0}}}"#,
        );
        let delta = stream_event(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}}"#,
        );
        let mut tr = StreamTranslator::new("claude".into(), "rid".into(), usize::MAX);
        tr.process_event(&start).unwrap();
        let out = tr.process_event(&delta).unwrap();
        let chunk = out.expect("chunk");
        let usage = chunk.usage.expect("usage in chunk");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        let s = String::from_utf8_lossy(&chunk.data);
        assert!(s.contains("finish_reason"));
        assert!(s.contains("stop"));
    }

    /// Streaming final chunk carries complete Usage. Assert image_units/audio_seconds
    /// are present (currently None); future provider population must not silently drop them.
    #[test]
    fn test_stream_message_delta_usage_carries_multimodal_fields() {
        let start = stream_event(
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":10,"output_tokens":0}}}"#,
        );
        let delta = stream_event(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}}"#,
        );
        let mut tr = StreamTranslator::new("claude".into(), "rid".into(), usize::MAX);
        tr.process_event(&start).unwrap();
        let out = tr.process_event(&delta).unwrap();
        let chunk = out.expect("chunk");
        let usage = chunk.usage.expect("Usage in final SSE chunk");
        // Current state: Anthropic does not yet populate multimodal fields; assert they exist.
        assert_eq!(
            usage.image_units, None,
            "image_units must be present (None until provider populates)"
        );
        assert_eq!(
            usage.audio_seconds, None,
            "audio_seconds must be present (None until provider populates)"
        );
    }

    #[test]
    fn test_stream_message_stop_emits_done() {
        let ev = stream_event(r#"{"type":"message_stop"}"#);
        let mut tr = StreamTranslator::new("claude".into(), "rid".into(), usize::MAX);
        let out = tr.process_event(&ev).unwrap();
        let chunk = out.expect("chunk");
        assert_eq!(chunk.data.as_ref(), b"data: [DONE]\n\n");
    }

    #[test]
    fn test_stream_tool_input_json_delta_forwarded() {
        let block = stream_event(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu1","name":"f","input":null}}"#,
        );
        let delta = stream_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"x\":1}"}}"#,
        );
        let mut tr = StreamTranslator::new("claude".into(), "rid".into(), usize::MAX);
        let first = tr.process_event(&block).unwrap();
        let first_chunk = first.expect("chunk from content_block_start");
        let first_s = String::from_utf8_lossy(&first_chunk.data);
        assert!(
            first_s.contains("assistant"),
            "stream starting with tool_use must emit role: assistant first"
        );
        let out = tr.process_event(&delta).unwrap();
        let chunk = out.expect("chunk");
        let s = String::from_utf8_lossy(&chunk.data);
        assert!(s.contains(r#"{"x":1}"#) || s.contains("x"));
    }

    #[test]
    fn test_stream_ping_ignored() {
        let ev = stream_event(r#"{"type":"ping"}"#);
        let mut tr = StreamTranslator::new("claude".into(), "rid".into(), usize::MAX);
        let out = tr.process_event(&ev).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_stream_error_event() {
        let ev = stream_event(r#"{"type":"error","error":{"message":"overloaded"}}"#);
        let mut tr = StreamTranslator::new("claude".into(), "rid".into(), usize::MAX);
        let out = tr.process_event(&ev);
        assert!(out.is_err());
    }

    // ── M4: buffer cap enforcement ────────────────────────────────────────────

    #[test]
    fn test_non_streaming_cap_exceeded_returns_overflow() {
        use crate::domain::ports::ProviderError;
        let resp = MessagesResponse {
            id: "msg_01".into(),
            type_: Some("message".into()),
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: "tu1".into(),
                name: "get_weather".into(),
                input: serde_json::json!({"a": 1}), // serializes to 7 bytes — exceeds cap of 3
            }],
            stop_reason: Some("tool_use".into()),
            usage: AnthropicUsage {
                input_tokens: 5,
                output_tokens: 10,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                output_tokens_details: None,
                cache_creation: None,
            },
        };
        let err = anthropic_to_chat_response(&resp, "claude", "req-1", 3).unwrap_err();
        match err {
            ProviderError::ToolCallBufferOverflow {
                provider,
                tool_call_id,
                cap_bytes,
            } => {
                assert_eq!(provider, "anthropic");
                assert_eq!(tool_call_id, "tu1");
                assert_eq!(cap_bytes, 3);
            }
            _ => panic!("expected ToolCallBufferOverflow, got {:?}", err),
        }
    }

    #[test]
    fn test_non_streaming_exactly_at_cap_is_ok() {
        // Boundary: input serializes to exactly cap_bytes — must pass.
        let resp = MessagesResponse {
            id: "msg_01".into(),
            type_: Some("message".into()),
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: "tu1".into(),
                name: "f".into(),
                input: serde_json::json!({}), // serializes to 2 bytes "{}"
            }],
            stop_reason: Some("tool_use".into()),
            usage: AnthropicUsage {
                input_tokens: 1,
                output_tokens: 1,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                output_tokens_details: None,
                cache_creation: None,
            },
        };
        // "{}" is 2 bytes; cap of 2 means len == cap, which is NOT > cap, so must be Ok.
        assert!(anthropic_to_chat_response(&resp, "claude", "req-1", 2).is_ok());
    }

    #[test]
    fn test_streaming_tool_input_json_delta_cap_exceeded() {
        use crate::domain::ports::ProviderError;
        let block = stream_event(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu1","name":"f","input":null}}"#,
        );
        // "abcd" is 4 bytes — exceeds cap of 3
        let delta = stream_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"abcd"}}"#,
        );
        let mut tr = StreamTranslator::new("claude".into(), "rid".into(), 3);
        tr.process_event(&block).unwrap();
        let err = tr.process_event(&delta).unwrap_err();
        match err {
            StreamErr::BufferOverflow(ProviderError::ToolCallBufferOverflow {
                provider,
                tool_call_id,
                cap_bytes,
            }) => {
                assert_eq!(provider, "anthropic");
                assert_eq!(tool_call_id, "tu1");
                assert_eq!(cap_bytes, 3);
            }
            _ => panic!(
                "expected StreamErr::BufferOverflow(ToolCallBufferOverflow), got {:?}",
                err
            ),
        }
    }

    #[test]
    fn test_streaming_cap_not_exceeded_across_two_deltas() {
        // Two deltas of 2 bytes each = 4 total, cap = 4 — must pass.
        let block = stream_event(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu1","name":"f","input":null}}"#,
        );
        let delta1 = stream_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ab"}}"#,
        );
        let delta2 = stream_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"cd"}}"#,
        );
        let mut tr = StreamTranslator::new("claude".into(), "rid".into(), 4);
        tr.process_event(&block).unwrap();
        assert!(tr.process_event(&delta1).is_ok());
        assert!(tr.process_event(&delta2).is_ok());
    }
}
