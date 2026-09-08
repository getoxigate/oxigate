// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! OpenAI <-> Anthropic format translation.
//!
//! Pure functions for request/response and streaming translation.

use std::collections::HashMap;

use bytes::Bytes;
use serde::de::DeserializeSeed;
use serde_json;
use tracing::{debug, warn};

use crate::domain::chat::{
    CacheAccounting, ChatRequest, ChatResponse, Choice, CompletionTokensDetails, Message,
    MessageContent, ReasoningAccounting, Role, StreamChunk, ToolCall, ToolCallFunction, Usage,
    UsageAccounting,
};
use crate::domain::ports::ProviderError;
use crate::domain::tool_schema::{ERR_TOOL_CALL_BUFFER_OVERFLOW, ERR_TYPE_GATEWAY_ERROR};
use crate::domain::tool_schema::{ToolChoiceKind, parse_tool_choice_value, truncate_for_error};
use crate::domain::usage_accounting::{
    CacheWriteAccounting, CacheWriteAccumulator, CacheWriteClass, CacheWriteClassRegistry,
    PricingContext,
};
use crate::providers::tool_limits::{ANTHROPIC_MAX_TOOLS, TOOL_ARGS_MAX_BYTES};
use crate::utils::sse;

use super::types::{
    ANTHROPIC_CACHE_WRITE_AGGREGATE_KEY, ANTHROPIC_DEFAULT_CACHE_WRITE_CLASS, AnthropicMessage,
    AnthropicTool, AnthropicToolChoice, AnthropicUsage, ContentBlock, MessagesRequest,
    MessagesResponse, StreamEvent, StreamEventSeed, ThinkingConfig,
};

pub(crate) const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Token accounting declared by the Anthropic Messages API contract.
///
/// Cache is **additive**: `platform.claude.com/docs/en/docs/build-with-claude/context-windows`
/// states that with prompt caching "the input count is split across `input_tokens`,
/// `cache_read_input_tokens`, and `cache_creation_input_tokens`", so `input_tokens` carries the
/// uncached remainder only (accessed 2026-08-10).
///
/// Reasoning is **contained in** the completion total:
/// `platform.claude.com/docs/en/docs/build-with-claude/extended-thinking` describes
/// `usage.output_tokens_details.thinking_tokens` as reporting "how many of the billed output
/// tokens were internal reasoning" (accessed 2026-08-10) — a breakdown of the billed output, not
/// an addition to it.
const ANTHROPIC_ACCOUNTING: UsageAccounting = UsageAccounting {
    cache: CacheAccounting::Additive,
    reasoning: ReasoningAccounting::IncludedInOutput,
};

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
///
/// `accumulator` is the one the seeded parse of `resp` credited; it carries the response's
/// cache-write details, which are not on `resp` itself.
pub fn anthropic_to_chat_response(
    resp: &MessagesResponse,
    model: &str,
    request_id: &str,
    cap_bytes: usize,
    pricing_context: &PricingContext,
    accumulator: CacheWriteAccumulator,
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

    let usage =
        anthropic_usage_to_usage(&resp.usage, reasoning_tokens, pricing_context, accumulator);

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

/// Closes cache-write accounting for one Anthropic response.
///
/// The per-class details are already in `accumulator` — the seeded parse credited them as the
/// payload streamed. What is left is the aggregate, which is an alternate view of the same
/// quantity rather than something to add to it:
///
/// - it is recorded as the reported total, so a provider contradicting itself reconciles to the
///   larger of the two views instead of tripping an assertion; and
/// - when the provider stated no per-class breakdown at all, the aggregate is a write to
///   Anthropic's documented default class, so it keeps that class's exact rate rather than
///   falling back.
fn finish_cache_write(
    mut accumulator: CacheWriteAccumulator,
    reported_aggregate: Option<u64>,
    details_seen: bool,
    pricing_context: &PricingContext,
) -> CacheWriteAccounting {
    if let Some(total) = reported_aggregate {
        accumulator.set_reported_aggregate(total);
        if !details_seen {
            accumulator.observe_detail(
                ANTHROPIC_CACHE_WRITE_AGGREGATE_KEY,
                CacheWriteClass::canonicalize(ANTHROPIC_DEFAULT_CACHE_WRITE_CLASS),
                total,
            );
        }
    }
    let mut accounting = accumulator.finish();
    accounting.set_pricing_context(pricing_context.clone());
    accounting
}

fn anthropic_usage_to_usage(
    u: &AnthropicUsage,
    reasoning_tokens: Option<u64>,
    pricing_context: &PricingContext,
    accumulator: CacheWriteAccumulator,
) -> Usage {
    let reasoning = reasoning_tokens.or_else(|| {
        u.output_tokens_details
            .as_ref()
            .and_then(|d| d.thinking_tokens)
    });
    let completion_tokens_details = reasoning.map(|r| CompletionTokensDetails {
        reasoning_tokens: Some(r),
    });
    let total = u.input_tokens + u.output_tokens;
    let cache_write = finish_cache_write(
        accumulator,
        u.cache_creation_input_tokens,
        u.cache_creation_present,
        pricing_context,
    );

    Usage {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
        total_tokens: total,
        completion_tokens_details,
        cache_creation_input_tokens: cache_write.published_tokens(),
        cache_read_input_tokens: u.cache_read_input_tokens,
        prompt_tokens_details: None,
        accounting: ANTHROPIC_ACCOUNTING,
        image_units: None,
        audio_seconds: None,
        cache_write,
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
    /// Cache-write details credited by the seeded parse of each event, retained across the stream
    /// because a provider detail object is a cumulative snapshot: a later object replaces the
    /// previous one, an omitted object leaves it standing.
    cache_write: CacheWriteAccumulator,
    /// Whether any event has stated a per-class breakdown, which is what stops a later aggregate
    /// from being attributed to the default class on top of details already credited.
    cache_write_details_seen: bool,
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
    /// The pricing generation this stream was dispatched under, snapshotted once by the adapter.
    /// Held so the cache-write classes reported by the stream are resolved against the same
    /// database the response is priced with.
    pricing_context: PricingContext,
}

impl StreamTranslator {
    /// Creates a translator for one streamed response.
    ///
    /// `pricing_context` is the generation snapshotted before dispatch; it is never re-read from
    /// the holder, so one stream cannot mix two pricing generations.
    pub fn new(
        model: String,
        request_id: String,
        cap_bytes: usize,
        pricing_context: PricingContext,
    ) -> Self {
        let cache_write = CacheWriteAccumulator::new(pricing_context.registry().clone());
        Self {
            input_tokens: None,
            output_tokens: None,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            cache_write,
            cache_write_details_seen: false,
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
            pricing_context,
        }
    }

    /// Parses one SSE line, committing any cache-write details it carries into this stream's
    /// accumulator.
    ///
    /// Parsing is a method rather than a free call at the caller because the accumulator it
    /// commits into is this translator's own state; exposing it for the caller to pass back in
    /// would let a stream be parsed against an accumulator that is not its own.
    pub fn parse_event(&mut self, line: &str) -> Option<StreamEvent> {
        parse_stream_event(line, self.pricing_context.registry(), &mut self.cache_write)
    }

    /// Process an Anthropic stream event and optionally emit an OpenAI-format chunk.
    pub fn process_event(&mut self, event: &StreamEvent) -> Result<Option<StreamChunk>, StreamErr> {
        match event {
            StreamEvent::MessageStart { message } => {
                if let Some(ref u) = message.usage {
                    self.input_tokens = Some(u.input_tokens);
                    self.cache_creation_input_tokens = u.cache_creation_input_tokens;
                    self.cache_read_input_tokens = u.cache_read_input_tokens;
                    // The per-class detail tokens were credited into `cache_write` while this
                    // event was parsed; only the fact that a breakdown was stated is recorded
                    // here. An aggregate that disagrees with the details is legal input and
                    // reconciles at finalization rather than being asserted away.
                    self.cache_write_details_seen |= u.cache_creation_present;
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
            StreamEvent::MessageDelta { delta, usage: u } => {
                self.output_tokens = Some(u.output_tokens);
                // Both aggregates are restatements, not increments: the same counts appear on
                // message_start and again here, so the final statement stands and an event that
                // omits a member leaves the standing value alone. The provider's last word is
                // taken as its word even when it is smaller than the first — resolving a
                // disagreement in either direction is a quantity decision, and a quantity the
                // gateway chose rather than read may not be reported as an exact one.
                self.cache_creation_input_tokens = u
                    .cache_creation_input_tokens
                    .or(self.cache_creation_input_tokens);
                self.cache_read_input_tokens =
                    u.cache_read_input_tokens.or(self.cache_read_input_tokens);
                // Anthropic restates cache creation cumulatively rather than incrementally:
                // the same values appear in message_start and again here. The seeded parse
                // therefore replaces the detail snapshot instead of adding to it, which is
                // what keeps the tokens from being counted twice — see
                // https://github.com/langchain-ai/langchainjs/issues/10249. An event that
                // omits the object leaves the previous snapshot standing — which is what this
                // event does in practice: it restates the cache-write aggregate but never the
                // per-class breakdown, so message_start's snapshot stands.
                self.cache_write_details_seen |= u.cache_creation_present;
                if let Some(ref d) = u.output_tokens_details {
                    self.reasoning_tokens = d.thinking_tokens.or(self.reasoning_tokens);
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
                // The provider says the response completed, and nothing follows `message_stop` on
                // the wire — so this terminator is the stream's clean end.
                let chunk = StreamChunk {
                    is_final: true,
                    ..StreamChunk::new(
                        Bytes::from_static(b"data: [DONE]\n\n"),
                        None,
                        Some(self.model.clone()),
                    )
                };
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
        // Cloned rather than consumed: usage is rebuilt on every terminal-bearing event, and the
        // stream may still restate its cache-write snapshot afterwards. The accumulator is
        // bounded by construction, so the copy is too.
        let cache_write = finish_cache_write(
            self.cache_write.clone(),
            self.cache_creation_input_tokens,
            self.cache_write_details_seen,
            &self.pricing_context,
        );

        Usage {
            prompt_tokens: input,
            completion_tokens: output,
            total_tokens: input + output,
            completion_tokens_details,
            cache_creation_input_tokens: cache_write.published_tokens(),
            cache_read_input_tokens: self.cache_read_input_tokens,
            prompt_tokens_details: None,
            accounting: ANTHROPIC_ACCOUNTING,
            image_units: None,
            audio_seconds: None,
            cache_write,
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

/// Parses a single SSE event line into a [`StreamEvent`], committing what it accounts for.
///
/// The parse is seeded rather than run over an intermediate value: a `cache_creation` object is
/// classified member by member as it is read, because no type the parse could produce first would
/// preserve duplicate members and stay bounded at the same time.
///
/// It is also transactional, which is the other half of that. The seed proposes a snapshot into a
/// bounded candidate and this function commits it, only once three things hold: the event
/// deserialized, nothing followed it on the line, and the detail object belonged to the usage the
/// resolved event actually carries. A line that is malformed, names an event type this lane does
/// not know, hangs a usage-shaped member off a `ping` or a `content_block_delta`, or carries
/// trailing JSON after the object therefore leaves accounting exactly as it found it.
///
/// Committing replaces the accumulator wholesale, which is the cumulative-snapshot semantics a
/// restated detail object requires: while a stream is running this accumulator holds detail-side
/// state only — the reported aggregate is applied to a clone at finalization — so there is no
/// earlier statement for a replacement to discard.
pub fn parse_stream_event(
    line: &str,
    registry: &CacheWriteClassRegistry,
    accumulator: &mut CacheWriteAccumulator,
) -> Option<StreamEvent> {
    let line = line.trim();
    let payload = line.strip_prefix("data: ")?;
    if payload == "[DONE]" {
        return None;
    }
    let mut de = serde_json::Deserializer::from_str(payload);
    let parsed = StreamEventSeed::new(registry).deserialize(&mut de).ok()?;
    de.end().ok()?;
    if let Some(details) = parsed.cache_write {
        *accumulator = details;
    }
    Some(parsed.event)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::anthropic::types::{
        AnthropicUsage, ContentBlock, MessagesResponse, MessagesResponseSeed, OutputTokensDetails,
    };

    /// Parses a buffered response body exactly as the adapter does — seeded, so the cache-write
    /// detail object reaches the accumulator instead of an intermediate that would flatten it.
    fn parse_buffered(
        body: &str,
        context: &PricingContext,
    ) -> (MessagesResponse, CacheWriteAccumulator) {
        let mut de = serde_json::Deserializer::from_str(body);
        let parsed = MessagesResponseSeed::new(context.registry())
            .deserialize(&mut de)
            .expect("valid response body");
        de.end().expect("no trailing data");
        let accumulator = parsed
            .cache_write
            .unwrap_or_else(|| CacheWriteAccumulator::new(context.registry().clone()));
        (parsed.response, accumulator)
    }

    /// Translates a buffered response body end to end, the way the adapter does.
    fn buffered_usage(body: &str, context: &PricingContext) -> Usage {
        let (resp, accumulator) = parse_buffered(body, context);
        anthropic_to_chat_response(
            &resp,
            "claude-sonnet-4-5-20251022",
            "req-1",
            usize::MAX,
            context,
            accumulator,
        )
        .expect("must translate")
        .usage
    }

    /// Tokens credited to one canonical cache-write class.
    fn class_tokens(accounting: &CacheWriteAccounting, class: &str) -> u64 {
        accounting
            .class_totals()
            .iter()
            .find(|total| total.class.as_str() == class)
            .map_or(0, |total| total.tokens)
    }

    /// Parses one SSE line into the translator's own accumulator and processes it.
    fn feed(tr: &mut StreamTranslator, event: &str) -> Option<StreamChunk> {
        let parsed = tr
            .parse_event(&format!("data: {event}"))
            .expect("valid event");
        tr.process_event(&parsed).expect("event processes")
    }

    /// A pricing generation taken from the bundled catalogue, for tests whose subject is not
    /// pricing. Its registry is the bundled union — today `{"5m", "1h"}`.
    fn bundled_pricing_context() -> PricingContext {
        use crate::config::PricingConfig;
        use crate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};

        let db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing DB loads");
        let registry = db.registry().clone();
        PricingContext::new(db, registry)
    }

    /// Thinking tokens are part of Anthropic's billed output total, so charging that total whole
    /// beside them billed the thinking subset twice.
    ///
    /// `claude-sonnet-5`, output 2,000 of which 1,500 thinking, output rate $15/Mtok.
    /// Previously 52_500_000 nano-USD; 1.75x the 30_000_000 owed. Every component is asserted,
    /// not only the total, so a compensating pair of errors cannot pass.
    #[test]
    fn test_thinking_double_charge_regression() {
        use crate::config::PricingConfig;
        use crate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
        use crate::utils::cost_headers::build_cost_headers;

        let usage = Usage {
            prompt_tokens: 0,
            completion_tokens: 2_000,
            total_tokens: 2_000,
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: Some(1_500),
            }),
            accounting: ANTHROPIC_ACCOUNTING,
            ..Default::default()
        };
        let holder = std::sync::Arc::new(std::sync::RwLock::new(
            PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
                .expect("bundled pricing must load"),
        ));

        let (_, finalized) = build_cost_headers("claude-sonnet-5", &usage, holder, false);
        let (cost, token_usage) = (&finalized.cost, &finalized.token_usage);

        assert_eq!(token_usage.standard_output_tokens(), 500);
        assert_eq!(
            token_usage.output_tokens, 2_000,
            "the reported completion total is unchanged; only what is priced at the output rate moves"
        );
        assert_eq!(cost.output_cost, crate::domain::ports::NanoUsd(7_500_000));
        assert_eq!(
            cost.thinking_cost,
            crate::domain::ports::NanoUsd(22_500_000)
        );
        assert_eq!(cost.total_cost, crate::domain::ports::NanoUsd(30_000_000));
    }

    /// Both `Usage` construction paths — the non-stream parse and the stream accumulator —
    /// apply the same accounting declaration. One declaration, two application sites.
    #[test]
    fn test_both_usage_paths_apply_the_same_accounting() {
        let non_stream = {
            let context = bundled_pricing_context();
            let accumulator = CacheWriteAccumulator::new(context.registry().clone());
            anthropic_usage_to_usage(&AnthropicUsage::default(), None, &context, accumulator)
        };
        let stream = StreamTranslator::new(
            "claude-sonnet-4-5".into(),
            "req-1".into(),
            4096,
            bundled_pricing_context(),
        )
        .build_usage();

        assert_eq!(non_stream.accounting, ANTHROPIC_ACCOUNTING);
        assert_eq!(stream.accounting, ANTHROPIC_ACCOUNTING);
    }

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
        let usage = buffered_usage(
            r#"{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"Hi"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":2,"cache_read_input_tokens":3}}"#,
            &bundled_pricing_context(),
        );
        assert_eq!(usage.cache_creation_input_tokens, Some(2));
        assert_eq!(usage.cache_read_input_tokens, Some(3));
        // No breakdown stated, so the aggregate is a write to the documented default class and
        // keeps that class's exact rate.
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 2);
        assert_eq!(class_tokens(&usage.cache_write, "1h"), 0);
    }

    /// A stated per-class breakdown is credited class by class, through the seeded parse.
    #[test]
    fn test_cache_creation_1h_breakdown() {
        let usage = buffered_usage(
            r#"{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"Hi"}],"stop_reason":"end_turn","usage":{"input_tokens":100,"output_tokens":50,"cache_creation_input_tokens":3500,"cache_read_input_tokens":2000,"cache_creation":{"ephemeral_5m_input_tokens":1000,"ephemeral_1h_input_tokens":2500}}}"#,
            &bundled_pricing_context(),
        );
        assert_eq!(usage.cache_creation_input_tokens, Some(3500));
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 1000);
        assert_eq!(class_tokens(&usage.cache_write, "1h"), 2500);
        assert_eq!(usage.cache_write.unknown_tokens(), 0);
    }

    /// An aggregate with no breakdown behind it is the documented default class, not a fallback.
    #[test]
    fn test_cache_creation_defaults_to_documented_class_when_breakdown_absent() {
        let usage = buffered_usage(
            r#"{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"Hi"}],"stop_reason":"end_turn","usage":{"input_tokens":100,"output_tokens":50,"cache_creation_input_tokens":3500,"cache_read_input_tokens":2000}}"#,
            &bundled_pricing_context(),
        );
        assert_eq!(usage.cache_creation_input_tokens, Some(3500));
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 3500);
        assert_eq!(class_tokens(&usage.cache_write, "1h"), 0);
        assert_eq!(usage.cache_write.unknown_tokens(), 0);
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
                cache_creation_present: false,
                input_tokens_present: true,
            },
        };
        let chat = anthropic_to_chat_response(
            &resp,
            "claude-sonnet-4-5-20251022",
            "req-1",
            usize::MAX,
            &bundled_pricing_context(),
            CacheWriteAccumulator::new(CacheWriteClassRegistry::default()),
        )
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
                cache_creation_present: false,
                input_tokens_present: true,
            },
        };
        let chat = anthropic_to_chat_response(
            &resp,
            "claude-sonnet-4-5-20251022",
            "req-1",
            usize::MAX,
            &bundled_pricing_context(),
            CacheWriteAccumulator::new(CacheWriteClassRegistry::default()),
        )
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
                cache_creation_present: false,
                input_tokens_present: true,
            },
        };
        let chat = anthropic_to_chat_response(
            &resp,
            "claude-sonnet-4-5-20251022",
            "req-1",
            usize::MAX,
            &bundled_pricing_context(),
            CacheWriteAccumulator::new(CacheWriteClassRegistry::default()),
        )
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

    /// Parses an event against a throwaway accumulator, for tests whose subject is not
    /// cache-write accounting. Tests that are use [`feed`], which parses into the translator's
    /// own accumulator the way the adapter does.
    fn stream_event(s: &str) -> StreamEvent {
        let registry = CacheWriteClassRegistry::default();
        let mut discarded = CacheWriteAccumulator::new(registry.clone());
        parse_stream_event(&format!("data: {s}"), &registry, &mut discarded).expect("valid event")
    }

    #[test]
    fn test_stream_text_delta() {
        let ev = stream_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
        );
        let mut tr = StreamTranslator::new(
            "claude".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
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
        let mut tr = StreamTranslator::new(
            "claude".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
        let out = tr.process_event(&ev).unwrap();
        assert!(out.is_none());
        assert_eq!(tr.input_tokens, Some(42));
    }

    /// Streaming cache creation breakdown — MessageStart with cache_creation object.
    /// A pricing database whose cache-write class union is wider than the bundled one, so a
    /// registry taken from it is distinguishable from a bundled registry by content alone.
    /// The generation a mid-request reload installs: a wider class registry *and* different
    /// rates. Both halves matter — pinning the registry while re-reading the holder for rates
    /// would still bill one request across two generations, and only a rate change catches that.
    fn reloaded_pricing_db() -> crate::domain::pricing::PricingDb {
        use crate::config::{PricingConfig, PricingOverride};
        use crate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};

        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            "claude-sonnet-4-5".to_string(),
            PricingOverride {
                input_per_token: 0.000_030,
                output_per_token: 0.000_150,
                context_window: 200_000,
                cache_read_multiplier: Some(0.1),
                cache_write_multipliers: std::collections::HashMap::from([
                    ("5m".to_string(), 1.25),
                    ("1h".to_string(), 2.0),
                    ("2h".to_string(), 3.0),
                ]),
            },
        );
        PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig { overrides })
            .expect("widened pricing DB loads")
    }

    fn class_names(context: &PricingContext) -> Vec<String> {
        context
            .registry()
            .classes()
            .iter()
            .map(|c| c.as_str().to_string())
            .collect()
    }

    /// One request never mixes two pricing generations.
    ///
    /// The adapter snapshots the generation before dispatch and the translator holds that
    /// snapshot; a reload landing mid-stream installs a new database into the holder, but the
    /// in-flight response is still accounted against the generation it started under. An
    /// implementation that read the holder at usage-build time instead would pick up the wider
    /// registry here and fail.
    #[test]
    fn test_stream_usage_pins_the_generation_acquired_before_dispatch() {
        use crate::domain::ports::NanoUsd;
        use crate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb, snapshot_pricing_context};
        use crate::utils::cost_headers::build_cost_headers;
        use std::sync::{Arc, RwLock};

        let holder = Arc::new(RwLock::new(
            PricingDb::load(
                BUNDLED_PRICING_JSON,
                &crate::config::PricingConfig::default(),
            )
            .expect("bundled pricing DB loads"),
        ));
        let before_dispatch = snapshot_pricing_context(&holder);
        // The pre-dispatch class set, whatever the bundled snapshot happens to configure. What
        // matters is that the pinned registry is still *this* set after the reload — spelling
        // the classes out here would restate the snapshot's contents and go red on any refresh
        // that adds a class, which says nothing about generation pinning.
        let before_classes = class_names(&before_dispatch);
        assert!(
            !before_classes.contains(&"2h".to_string()),
            "the reload-only class must not already be configured, or this test proves nothing"
        );

        let mut tr = StreamTranslator::new(
            "claude-sonnet-4-5".into(),
            "rid".into(),
            usize::MAX,
            before_dispatch,
        );

        let start = stream_event(
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":100,"output_tokens":0}}}"#,
        );
        tr.process_event(&start).unwrap();

        // Reload lands between message_start and the final message_delta.
        *holder.write().expect("holder lock") = reloaded_pricing_db();
        assert!(class_names(&snapshot_pricing_context(&holder)).contains(&"2h".to_string()));

        let delta = stream_event(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}"#,
        );
        let usage = tr
            .process_event(&delta)
            .unwrap()
            .expect("final chunk")
            .usage
            .expect("usage in final chunk");

        let pinned = usage
            .cache_write
            .pricing_context()
            .expect("usage carries the request generation");
        assert_eq!(class_names(pinned), before_classes);

        // The classification is only half the invariant. Finalization is handed the *reloaded*
        // holder, exactly as the live handler would be, and must still bill the pre-dispatch
        // rates — otherwise the request is classified under one generation and charged under
        // another.
        let (_, finalized) =
            build_cost_headers("claude-sonnet-4-5", &usage, Arc::clone(&holder), false);
        assert_eq!(
            finalized.cost.input_cost,
            NanoUsd(300_000),
            "100 input tokens at the pinned generation's 3e-06/token, not the reloaded 3e-05"
        );
        assert_eq!(
            finalized.cost.output_cost,
            NanoUsd(300_000),
            "20 output tokens at the pinned generation's 15e-06/token, not the reloaded 15e-05"
        );
        assert_eq!(
            finalized.cost.total_cost,
            NanoUsd(600_000),
            "the whole bill comes from the pinned generation, on both axes"
        );
    }

    /// The buffered path carries the same generation, taken at the same point in the request.
    #[test]
    fn test_buffered_usage_carries_the_request_generation() {
        let context = bundled_pricing_context();
        let resp = MessagesResponse {
            id: "m1".into(),
            type_: Some("message".into()),
            role: "assistant".into(),
            content: vec![ContentBlock::Text { text: "hi".into() }],
            stop_reason: Some("end_turn".into()),
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            },
        };

        let out = anthropic_to_chat_response(
            &resp,
            "claude-sonnet-4-5-20251022",
            "req-1",
            usize::MAX,
            &context,
            CacheWriteAccumulator::new(CacheWriteClassRegistry::default()),
        )
        .expect("buffered translation");

        let pinned = out
            .usage
            .cache_write
            .pricing_context()
            .expect("usage carries the request generation");
        assert_eq!(class_names(pinned), class_names(&context));
    }

    #[test]
    fn test_stream_message_start_extracts_cache_creation_breakdown() {
        let mut tr = StreamTranslator::new(
            "claude-sonnet-4-5-20251022".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
        let out = feed(
            &mut tr,
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":100,"output_tokens":0,"cache_creation_input_tokens":3500,"cache_read_input_tokens":2000,"cache_creation":{"ephemeral_5m_input_tokens":1000,"ephemeral_1h_input_tokens":2500}}}}"#,
        );
        assert!(out.is_none());
        assert_eq!(tr.input_tokens, Some(100));
        assert_eq!(tr.cache_creation_input_tokens, Some(3500));
        assert_eq!(tr.cache_read_input_tokens, Some(2000));

        let usage = tr.build_usage();
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 1000);
        assert_eq!(class_tokens(&usage.cache_write, "1h"), 2500);
        assert_eq!(usage.cache_creation_input_tokens, Some(3500));
    }

    /// Streaming aggregate with no breakdown behind it is the documented default class.
    #[test]
    fn test_stream_message_start_defaults_class_without_breakdown() {
        let mut tr = StreamTranslator::new(
            "claude-sonnet-4-5-20251022".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
        let out = feed(
            &mut tr,
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":100,"output_tokens":0,"cache_creation_input_tokens":3500,"cache_read_input_tokens":2000}}}"#,
        );
        assert!(out.is_none());
        assert_eq!(tr.cache_creation_input_tokens, Some(3500));

        let usage = tr.build_usage();
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 3500);
        assert_eq!(class_tokens(&usage.cache_write, "1h"), 0);
    }

    /// A response body carrying `usage`, for the reconciliation cases below.
    fn buffered_body(usage_json: &str) -> String {
        format!(
            r#"{{"id":"msg_01","type":"message","role":"assistant","content":[{{"type":"text","text":"Hi"}}],"stop_reason":"end_turn","usage":{usage_json}}}"#
        )
    }

    /// The message a seeded buffered parse fails with. Panics if the body parses instead —
    /// `SeededMessagesResponse` is deliberately not `Debug`, so `expect_err` cannot be used.
    fn buffered_parse_error(body: &str) -> String {
        let context = bundled_pricing_context();
        let mut de = serde_json::Deserializer::from_str(body);
        match MessagesResponseSeed::new(context.registry()).deserialize(&mut de) {
            Ok(_) => panic!("expected a parse error, but the body parsed"),
            Err(e) => e.to_string(),
        }
    }

    /// Aggregate and details are alternate views of one quantity, so a provider contradicting
    /// itself bills the larger view rather than the stated aggregate — and reconciles rather than
    /// tripping an assertion. This test runs in a debug build, where the removed `debug_assert!`
    /// would have fired.
    #[test]
    fn test_details_exceeding_the_aggregate_bill_the_larger_view() {
        let usage = buffered_usage(
            &buffered_body(
                r#"{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":2000,"cache_creation":{"ephemeral_5m_input_tokens":1000,"ephemeral_1h_input_tokens":2000}}"#,
            ),
            &bundled_pricing_context(),
        );
        assert_eq!(usage.cache_write.accounted_tokens(), 3000);
        assert_eq!(usage.cache_creation_input_tokens, Some(3000));
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 1000);
        assert_eq!(class_tokens(&usage.cache_write, "1h"), 2000);
        assert!(usage.cache_write.outcome().is_contradiction());
        assert_eq!(usage.cache_write.unmatched_residual_tokens(), 0);
    }

    /// An aggregate larger than the details it came with keeps its quantity, and the part no
    /// class claimed is a residual — priced at fallback, never defaulted into a class.
    #[test]
    fn test_aggregate_exceeding_partial_details_leaves_a_residual() {
        let usage = buffered_usage(
            &buffered_body(
                r#"{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":5000,"cache_creation":{"ephemeral_5m_input_tokens":1000}}"#,
            ),
            &bundled_pricing_context(),
        );
        assert_eq!(usage.cache_write.accounted_tokens(), 5000);
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 1000);
        assert_eq!(usage.cache_write.unmatched_residual_tokens(), 4000);
        assert!(usage.cache_write.outcome().is_contradiction());
        assert!(usage.cache_write.partition_is_exact());
    }

    /// A repeated billing scalar is rejected, not last-value-wins.
    ///
    /// The hand-written visitors replaced derived deserializers, which fail on a repeated struct
    /// member. Overwriting instead would let a second `input_tokens` replace a larger quantity
    /// with a smaller one and undercharge the request — a money-path input the previous
    /// implementation failed closed on.
    #[test]
    fn test_duplicate_billing_scalar_in_buffered_usage_is_rejected() {
        let err = buffered_parse_error(&buffered_body(
            r#"{"input_tokens":9000,"input_tokens":1,"output_tokens":5}"#,
        ));
        assert!(
            err.contains("duplicate field `input_tokens`"),
            "expected a duplicate-field error, got: {err}"
        );
    }

    /// The same rule for the cache-write aggregate, which is priced directly.
    #[test]
    fn test_duplicate_cache_write_aggregate_in_buffered_usage_is_rejected() {
        let err = buffered_parse_error(&buffered_body(
            r#"{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":9000,"cache_creation_input_tokens":1}"#,
        ));
        assert!(
            err.contains("duplicate field `cache_creation_input_tokens`"),
            "expected a duplicate-field error, got: {err}"
        );
    }

    /// A repeated `cache_creation` *wrapper* is a duplicate member like any other. Only the
    /// members inside one accumulate — see `test_duplicate_members_are_both_summed`.
    #[test]
    fn test_duplicate_cache_creation_wrapper_in_buffered_usage_is_rejected() {
        let err = buffered_parse_error(&buffered_body(
            r#"{"input_tokens":10,"output_tokens":5,"cache_creation":{"ephemeral_5m_input_tokens":9000},"cache_creation":{"ephemeral_5m_input_tokens":1}}"#,
        ));
        assert!(
            err.contains("duplicate field `cache_creation`"),
            "expected a duplicate-field error, got: {err}"
        );
    }

    /// The streaming lane rejects the same shape. A rejected event is transactional: it commits
    /// nothing to the accumulator, so the smaller repeat never reaches billing either.
    #[test]
    fn test_duplicate_billing_scalar_in_stream_event_is_rejected() {
        let registry = CacheWriteClassRegistry::default();
        let mut accumulator = CacheWriteAccumulator::new(registry.clone());
        let event = parse_stream_event(
            r#"data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":9000,"input_tokens":1,"output_tokens":0}}}"#,
            &registry,
            &mut accumulator,
        );
        assert!(
            event.is_none(),
            "a repeated input_tokens must not yield a stream event"
        );
        let finished = accumulator.finish();
        assert_eq!(finished.accounted_tokens(), 0, "nothing may be committed");
    }

    /// A repeated wrapper member on the event object itself, one level above usage.
    ///
    /// The fixture carries a valid top-level `usage` so the event is rejected for the duplicate
    /// `delta` this test is named after, rather than for a missing required member — otherwise it
    /// would keep passing while no longer testing its subject.
    #[test]
    fn test_duplicate_wrapper_member_in_stream_event_is_rejected() {
        let registry = CacheWriteClassRegistry::default();
        let mut accumulator = CacheWriteAccumulator::new(registry.clone());
        let event = parse_stream_event(
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"delta":{"stop_reason":"max_tokens"},"usage":{"input_tokens":10,"output_tokens":5}}"#,
            &registry,
            &mut accumulator,
        );
        assert!(
            event.is_none(),
            "a repeated delta member must not yield a stream event"
        );
    }

    /// Two identical members are two observations of the same class, not one — a map type in this
    /// position would keep the last and under-bill by exactly the first.
    #[test]
    fn test_duplicate_members_are_both_summed() {
        let usage = buffered_usage(
            &buffered_body(
                r#"{"input_tokens":10,"output_tokens":5,"cache_creation":{"ephemeral_5m_input_tokens":100,"ephemeral_5m_input_tokens":100}}"#,
            ),
            &bundled_pricing_context(),
        );
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 200);
        assert_eq!(usage.cache_write.accounted_tokens(), 200);
        assert!(usage.cache_write.duplicate().configured_duplicate);
    }

    /// An unfamiliar member reaches the accumulator as an unknown class, so a class the gateway
    /// has never priced still lands on the money path instead of being dropped.
    ///
    /// `12h` is the unconfigured duration: it canonicalizes cleanly but no bundled entry prices
    /// it. The class has to be one the *whole* snapshot leaves unconfigured — the registry is the
    /// union across every entry, so a duration another provider prices is not unfamiliar here.
    #[test]
    fn test_unconfigured_and_malformed_members_land_in_overflow() {
        let usage = buffered_usage(
            &buffered_body(
                r#"{"input_tokens":10,"output_tokens":5,"cache_creation":{"ephemeral_12h_input_tokens":700,"ephemeral_5m_tokens":300}}"#,
            ),
            &bundled_pricing_context(),
        );
        assert_eq!(usage.cache_write.unknown_tokens(), 1000);
        assert_eq!(usage.cache_write.accounted_tokens(), 1000);
        assert!(usage.cache_write.class_totals().is_empty());
        // Unknown classes share one bucket by design, so their duplicate identity cannot be
        // asserted — it is reported indeterminate rather than claimed either way.
        assert!(usage.cache_write.duplicate().unknown_indeterminate);
        assert!(!usage.cache_write.duplicate().configured_duplicate);
    }

    /// A later detail object replaces the previous snapshot; an event that omits one leaves the
    /// previous snapshot standing. Anthropic restates cache creation cumulatively, so adding the
    /// two would double-bill the whole request.
    #[test]
    fn test_stream_detail_snapshots_replace_and_persist() {
        let mut tr = StreamTranslator::new(
            "claude-sonnet-4-5-20251022".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
        feed(
            &mut tr,
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":10,"output_tokens":0,"cache_creation":{"ephemeral_5m_input_tokens":1000}}}}"#,
        );
        feed(
            &mut tr,
            r#"{"type":"message_delta","delta":{"stop_reason":null},"usage":{"input_tokens":10,"output_tokens":3,"cache_creation":{"ephemeral_1h_input_tokens":2500}}}"#,
        );
        let replaced = tr.build_usage();
        assert_eq!(class_tokens(&replaced.cache_write, "5m"), 0);
        assert_eq!(class_tokens(&replaced.cache_write, "1h"), 2500);

        // A final event with no detail object at all must not disturb the standing snapshot.
        feed(
            &mut tr,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":5}}"#,
        );
        let retained = tr.build_usage();
        assert_eq!(class_tokens(&retained.cache_write, "1h"), 2500);
        assert_eq!(retained.cache_write.accounted_tokens(), 2500);
    }

    /// A stream event with a standing detail snapshot behind it, for the transactional cases.
    fn translator_with_a_standing_snapshot() -> StreamTranslator {
        let mut tr = StreamTranslator::new(
            "claude-sonnet-4-5-20251022".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
        feed(
            &mut tr,
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":10,"output_tokens":0,"cache_creation":{"ephemeral_5m_input_tokens":1000}}}}"#,
        );
        tr
    }

    /// An event the parse rejects must leave accounting exactly as it found it.
    ///
    /// The seed reads a `cache_creation` object before the outer event has resolved, so an event
    /// that turns out to name an unknown type or to be missing a member it requires has already
    /// been read by the time it fails. Proposing into a candidate and committing only after the
    /// event resolves is what keeps a payload the gateway refuses from moving a billed quantity.
    #[test]
    fn test_a_rejected_stream_event_does_not_move_accounting() {
        let mut tr = translator_with_a_standing_snapshot();

        // An event type this lane does not know, carrying a detail object.
        assert!(
            tr.parse_event(
                r#"data: {"type":"message_finished","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":10,"output_tokens":0,"cache_creation":{"ephemeral_5m_input_tokens":999999}}}}"#
            )
            .is_none()
        );
        // A known event type missing a member it requires, carrying a detail object.
        assert!(
            tr.parse_event(
                r#"data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"x","usage":{"input_tokens":10,"output_tokens":0,"cache_creation":{"ephemeral_1h_input_tokens":999999}}}}"#
            )
            .is_none()
        );
        // A line that is not JSON at all.
        assert!(
            tr.parse_event(r#"data: {"type":"message_delta",}"#)
                .is_none()
        );

        let usage = tr.build_usage();
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 1000);
        assert_eq!(class_tokens(&usage.cache_write, "1h"), 0);
        assert_eq!(usage.cache_write.accounted_tokens(), 1000);
    }

    /// A usage-shaped member hanging off an event that reports no usage is not accounting.
    ///
    /// `ping` and `content_block_delta` carry no usage in Anthropic's contract, so a detail
    /// object reached through them belongs to nothing the event states.
    #[test]
    fn test_usage_shaped_members_on_irrelevant_events_do_not_account() {
        let mut tr = translator_with_a_standing_snapshot();

        feed(
            &mut tr,
            r#"{"type":"ping","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":10,"output_tokens":0,"cache_creation":{"ephemeral_1h_input_tokens":999999}}}}"#,
        );
        feed(
            &mut tr,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x","usage":{"input_tokens":10,"output_tokens":0,"cache_creation":{"ephemeral_1h_input_tokens":999999}}}}"#,
        );

        let usage = tr.build_usage();
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 1000);
        assert_eq!(class_tokens(&usage.cache_write, "1h"), 0);
        assert_eq!(usage.cache_write.accounted_tokens(), 1000);
    }

    /// Trailing JSON after the object means the line was not the event it appeared to be, so
    /// nothing it stated may bill — including the part that parsed cleanly.
    #[test]
    fn test_trailing_data_after_a_stream_event_is_not_accounted() {
        let mut tr = translator_with_a_standing_snapshot();

        assert!(
            tr.parse_event(
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":5,"cache_creation":{"ephemeral_1h_input_tokens":999999}}} {"type":"ping"}"#
            )
            .is_none()
        );

        let usage = tr.build_usage();
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 1000);
        assert_eq!(class_tokens(&usage.cache_write, "1h"), 0);
    }

    /// The transactional parse still commits what it should: a later snapshot replaces the
    /// standing one even after rejected events in between.
    #[test]
    fn test_a_valid_snapshot_replaces_the_standing_one_after_rejections() {
        let mut tr = translator_with_a_standing_snapshot();

        assert!(tr.parse_event(r#"data: {"type":"nonsense"}"#).is_none());
        feed(
            &mut tr,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":5,"cache_creation":{"ephemeral_1h_input_tokens":2500}}}"#,
        );

        let usage = tr.build_usage();
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 0);
        assert_eq!(class_tokens(&usage.cache_write, "1h"), 2500);
        assert_eq!(usage.cache_write.accounted_tokens(), 2500);
    }

    /// The seeded path is actually invoked on the streaming lane: a class no pricing generation
    /// configures reaches unknown accounting through `parse_event`, not through a direct call on
    /// the accumulator. `12h` is unconfigured across the whole bundled snapshot — see
    /// `test_unconfigured_and_malformed_members_land_in_overflow`.
    #[test]
    fn test_unfamiliar_stream_member_reaches_unknown_accounting() {
        let mut tr = StreamTranslator::new(
            "claude-sonnet-4-5-20251022".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
        feed(
            &mut tr,
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":10,"output_tokens":0,"cache_creation":{"ephemeral_12h_input_tokens":1500}}}}"#,
        );

        let usage = tr.build_usage();
        assert_eq!(usage.cache_write.unknown_tokens(), 1500);
        assert!(usage.cache_write.class_totals().is_empty());
        assert_eq!(usage.cache_write.accounted_tokens(), 1500);
    }

    /// Optional members that the removed derived impls resolved to `None` when absent must still
    /// resolve to `None` — a hand-written visitor that demands them rejects payloads the gateway
    /// accepted before this lane stopped deriving.
    #[test]
    fn test_absent_optional_buffered_members_are_none() {
        let (resp, _) = parse_buffered(
            r#"{"id":"msg_01","role":"assistant","content":[{"type":"text","text":"Hi"}],"usage":{"input_tokens":10,"output_tokens":5}}"#,
            &bundled_pricing_context(),
        );
        assert!(resp.type_.is_none());
        assert!(resp.stop_reason.is_none());
    }

    /// Anthropic omits `stop_reason` on every `message_delta` but the last one.
    ///
    /// The usage object sits beside `delta`, so an intermediate delta is an empty `delta` object
    /// with a populated sibling — not a `delta` carrying usage of its own.
    #[test]
    fn test_message_delta_without_a_stop_reason_still_parses() {
        let ev = stream_event(
            r#"{"type":"message_delta","delta":{},"usage":{"input_tokens":10,"output_tokens":5}}"#,
        );
        match ev {
            StreamEvent::MessageDelta { delta, usage } => {
                assert!(delta.stop_reason.is_none());
                assert_eq!(usage.output_tokens, 5);
            }
            other => panic!("expected a message_delta, got {other:?}"),
        }
    }

    /// The same facts reported over SSE and in a buffered body account identically. The two
    /// paths run different deserializers, so this is the test that keeps them one behaviour.
    #[test]
    fn test_buffered_and_streaming_payloads_account_identically() {
        let usage_json = r#"{"input_tokens":100,"output_tokens":20,"cache_creation_input_tokens":3500,"cache_creation":{"ephemeral_5m_input_tokens":1000,"ephemeral_1h_input_tokens":2500}}"#;

        let buffered = buffered_usage(&buffered_body(usage_json), &bundled_pricing_context());

        let mut tr = StreamTranslator::new(
            "claude-sonnet-4-5-20251022".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
        feed(
            &mut tr,
            &format!(
                r#"{{"type":"message_start","message":{{"id":"m1","type":"message","role":"assistant","usage":{usage_json}}}}}"#
            ),
        );
        feed(
            &mut tr,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":100,"output_tokens":20}}"#,
        );
        let streamed = tr.build_usage();

        assert_eq!(
            buffered.cache_creation_input_tokens,
            streamed.cache_creation_input_tokens
        );
        assert_eq!(
            buffered.cache_write.accounted_tokens(),
            streamed.cache_write.accounted_tokens()
        );
        assert_eq!(
            buffered.cache_write.class_totals(),
            streamed.cache_write.class_totals()
        );
        assert_eq!(
            buffered.cache_write.outcome(),
            streamed.cache_write.outcome()
        );
        assert_eq!(
            buffered.cache_write.duplicate(),
            streamed.cache_write.duplicate()
        );
        assert_eq!(
            buffered.cache_write.evidence_entries(),
            streamed.cache_write.evidence_entries()
        );
    }

    /// Anthropic reports `message_delta` usage as a **sibling** of `delta`, at the event's root.
    ///
    /// Both frames are transcribed from a live capture, so this test states what the provider
    /// actually sends rather than what the parser expects of it. Only the SSE line's trailing
    /// padding is dropped; the JSON payloads are verbatim, including members this lane does not
    /// model (`stop_details`, `service_tier`, `inference_geo`).
    #[test]
    fn test_message_delta_reports_usage_from_the_event_root() {
        let mut tr = StreamTranslator::new(
            "claude-haiku-4-5-20251001".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
        feed(
            &mut tr,
            r#"{"type":"message_start","message":{"model":"claude-haiku-4-5-20251001","id":"msg_011CeZDce3bhSggocHJ8tbai","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"stop_details":null,"usage":{"input_tokens":8,"cache_creation_input_tokens":7601,"cache_read_input_tokens":0,"cache_creation":{"ephemeral_5m_input_tokens":7601,"ephemeral_1h_input_tokens":0},"output_tokens":1,"service_tier":"standard","inference_geo":"not_available"}}}"#,
        );
        let chunk = feed(
            &mut tr,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null,"stop_details":null},"usage":{"input_tokens":8,"cache_creation_input_tokens":7601,"cache_read_input_tokens":0,"output_tokens":16}}"#,
        )
        .expect("the terminal event emits a chunk");

        let usage = chunk.usage.expect("usage in the final chunk");
        assert_eq!(
            usage.completion_tokens, 16,
            "the captured frame reports 16 output tokens at the event root"
        );
        assert_eq!(usage.prompt_tokens, 8);
        assert_eq!(usage.total_tokens, 24);
    }

    /// A translator with no `message_start` behind it, for the usage-shape cases.
    fn bare_translator() -> StreamTranslator {
        StreamTranslator::new(
            "claude-sonnet-4-5-20251022".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        )
    }

    /// Anthropic's streaming documentation shows a `message_delta` usage carrying `output_tokens`
    /// alone. `input_tokens` is not read from this event, so demanding it would drop the terminal
    /// event — and `finish_reason` with it — over a value nothing consumes.
    #[test]
    fn test_message_delta_usage_may_state_output_tokens_alone() {
        let mut tr = bare_translator();
        let chunk = feed(
            &mut tr,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":14}}"#,
        )
        .expect("the terminal event emits a chunk");

        assert_eq!(chunk.usage.expect("usage in chunk").completion_tokens, 14);
        let body = String::from_utf8_lossy(&chunk.data);
        assert!(
            body.contains(r#""finish_reason":"stop""#),
            "the terminal event must still carry its finish_reason: {body}"
        );
    }

    /// `MessageDeltaUsage.input_tokens` is typed nullable, so an explicit `null` is a shape the
    /// provider is entitled to send. It resolves to zero rather than failing the event.
    #[test]
    fn test_message_delta_usage_tolerates_an_explicitly_null_input_tokens() {
        let mut tr = bare_translator();
        let chunk = feed(
            &mut tr,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":null,"output_tokens":14}}"#,
        )
        .expect("an explicit null must not drop the terminal event");

        assert_eq!(chunk.usage.expect("usage in chunk").completion_tokens, 14);
    }

    /// The relaxation is position-specific. A buffered `Message.usage` is the request's only
    /// input-token source, so omitting the member there must still fail loudly.
    #[test]
    fn test_buffered_usage_still_requires_input_tokens() {
        let context = bundled_pricing_context();
        let body = r#"{"id":"msg_01","role":"assistant","content":[{"type":"text","text":"Hi"}],"usage":{"output_tokens":5}}"#;
        let mut de = serde_json::Deserializer::from_str(body);
        let parsed = MessagesResponseSeed::new(context.registry()).deserialize(&mut de);
        assert!(
            parsed.is_err(),
            "a buffered response with no input_tokens must be rejected, not defaulted to zero"
        );
    }

    /// Reading `usage` from the event root means every event's root member is parsed, whatever
    /// its tag. A member the resolved event never reads must not be able to fail it: a
    /// `content_block_delta` carrying a root `usage` without `output_tokens` still yields its
    /// text, rather than the generated content being discarded over an accounting member that
    /// belongs to another event.
    #[test]
    fn test_root_usage_without_output_tokens_does_not_fail_a_non_terminal_event() {
        let mut tr = bare_translator();
        let chunk = feed(
            &mut tr,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"},"usage":{"input_tokens":5}}"#,
        )
        .expect("the text delta must still emit its chunk");

        let body = String::from_utf8_lossy(&chunk.data);
        assert!(
            body.contains(r#""content":"Hi""#),
            "the generated text must survive the stray usage member: {body}"
        );
        assert!(
            tr.parse_event(r#"data: {"type":"ping","usage":{"input_tokens":5}}"#)
                .is_some(),
            "a ping carrying the same member must parse too"
        );
    }

    /// The deferral covers the whole member, not one of its rules. A root `usage` stated twice is
    /// a contradiction only for the event that reads the member; on any other event it is noise,
    /// and rejecting the frame would discard generated text over it.
    #[test]
    fn test_duplicate_root_usage_does_not_fail_a_non_terminal_event() {
        let mut tr = bare_translator();
        let chunk = feed(
            &mut tr,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"},"usage":{"output_tokens":1},"usage":{"output_tokens":2}}"#,
        )
        .expect("the text delta must still emit its chunk");

        let body = String::from_utf8_lossy(&chunk.data);
        assert!(
            body.contains(r#""content":"Hi""#),
            "the generated text must survive the repeated usage member: {body}"
        );
    }

    /// Nor does the deferral stop at members the object is missing. A root `usage` that is not
    /// even an object, or whose counts are not numbers, is still a member only `message_delta`
    /// reads — so it is buffered unexamined and dropped, not parsed.
    #[test]
    fn test_type_invalid_root_usage_does_not_fail_a_non_terminal_event() {
        let mut tr = bare_translator();
        for line in [
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"},"usage":"not-an-object"}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"},"usage":{"output_tokens":"five"}}"#,
        ] {
            let chunk = feed(&mut tr, line).expect("the text delta must still emit its chunk");
            let body = String::from_utf8_lossy(&chunk.data);
            assert!(
                body.contains(r#""content":"Hi""#),
                "the generated text must survive an unreadable usage member: {body}"
            );
        }
    }

    /// The same shapes on the event that *does* read the member are fatal, which is what makes
    /// the deferral a deferral rather than a relaxation. A repeat is fatal there too, in
    /// `test_duplicate_top_level_usage_in_stream_event_is_rejected`.
    #[test]
    fn test_message_delta_rejects_a_usage_it_cannot_read() {
        for line in [
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":"not-an-object"}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":"five"}}"#,
        ] {
            let mut tr = bare_translator();
            assert!(
                tr.parse_event(line).is_none(),
                "a usage object message_delta cannot read must not yield an event: {line}"
            );
            assert_eq!(tr.build_usage().completion_tokens, 0);
        }
    }

    /// The deferral is a deferral, not a relaxation: `message_delta` is the one event that reads
    /// output tokens, and there an absent count still fails the event rather than resolving to a
    /// zero indistinguishable from a genuinely empty response.
    #[test]
    fn test_message_delta_still_requires_output_tokens() {
        let mut tr = bare_translator();
        assert!(
            tr.parse_event(
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10}}"#
            )
            .is_none(),
            "a message_delta with no output_tokens must not yield an event"
        );
    }

    /// The cache aggregates are cumulative restatements, so the two frames normally agree and the
    /// restatement changes nothing. Where they disagree the final snapshot stands in either
    /// direction: the gateway reports the count the provider last stated rather than choosing
    /// between two of them, because a quantity it picked is no longer a quantity it read, and
    /// this lane has no channel for saying so on the request's cost status.
    #[test]
    fn test_message_delta_cache_restatement_takes_the_final_snapshot() {
        let mut tr = bare_translator();
        feed(
            &mut tr,
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":8,"output_tokens":1,"cache_creation_input_tokens":7601,"cache_read_input_tokens":2000}}}"#,
        );
        feed(
            &mut tr,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":8,"output_tokens":16,"cache_creation_input_tokens":10,"cache_read_input_tokens":5}}"#,
        );

        assert_eq!(tr.cache_creation_input_tokens, Some(10));
        assert_eq!(tr.cache_read_input_tokens, Some(5));
    }

    /// A member the event omits leaves the standing value alone rather than clearing it: only a
    /// stated count restates, and an absent one states nothing.
    #[test]
    fn test_message_delta_cache_restatement_never_clears_an_omitted_member() {
        let mut tr = bare_translator();
        feed(
            &mut tr,
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":8,"output_tokens":1,"cache_creation_input_tokens":100,"cache_read_input_tokens":2000}}}"#,
        );
        feed(
            &mut tr,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":8,"output_tokens":16,"cache_creation_input_tokens":7601}}"#,
        );

        assert_eq!(tr.cache_creation_input_tokens, Some(7601));
        assert_eq!(tr.cache_read_input_tokens, Some(2000));
    }

    /// The old nested position is no longer honoured. A frame that puts `usage` inside `delta`
    /// states no usage at the event root, so it is rejected outright rather than silently
    /// yielding a chunk that reports zero completion tokens.
    #[test]
    fn test_usage_nested_inside_delta_no_longer_supplies_usage() {
        let mut tr = bare_translator();
        assert!(
            tr.parse_event(
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}}"#
            )
            .is_none(),
            "the nested shape must not parse into a usage-bearing event"
        );
        assert_eq!(
            tr.build_usage().completion_tokens,
            0,
            "and it must not have moved the output count on its way through"
        );
    }

    /// `message_delta` restates the cache-write **aggregate** but never the `cache_creation`
    /// breakdown, so `message_start`'s per-class snapshot has to survive it. Both frames are
    /// transcribed from a live capture of a cache-writing request.
    #[test]
    fn test_message_start_cache_write_detail_survives_an_aggregate_only_message_delta() {
        let mut tr = bare_translator();
        feed(
            &mut tr,
            r#"{"type":"message_start","message":{"id":"msg_011CeZDce3bhSggocHJ8tbai","type":"message","role":"assistant","usage":{"input_tokens":8,"cache_creation_input_tokens":7601,"cache_read_input_tokens":0,"cache_creation":{"ephemeral_5m_input_tokens":7601,"ephemeral_1h_input_tokens":0},"output_tokens":1}}}"#,
        );
        feed(
            &mut tr,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":8,"cache_creation_input_tokens":7601,"cache_read_input_tokens":0,"output_tokens":16}}"#,
        );

        let usage = tr.build_usage();
        assert_eq!(usage.completion_tokens, 16);
        assert_eq!(
            class_tokens(&usage.cache_write, "5m"),
            7601,
            "the per-class snapshot from message_start must stand"
        );
        assert_eq!(usage.cache_write.accounted_tokens(), 7601);
    }

    /// The event tag stays authoritative for the new top-level candidate exactly as for the
    /// existing two: `ping` and `content_block_delta` report no usage, so a usage object hung off
    /// one of them belongs to nothing and proposes nothing.
    #[test]
    fn test_top_level_usage_on_irrelevant_events_does_not_account() {
        let mut tr = translator_with_a_standing_snapshot();

        feed(
            &mut tr,
            r#"{"type":"ping","usage":{"input_tokens":10,"output_tokens":0,"cache_creation":{"ephemeral_1h_input_tokens":999999}}}"#,
        );
        feed(
            &mut tr,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"},"usage":{"input_tokens":10,"output_tokens":0,"cache_creation":{"ephemeral_1h_input_tokens":999999}}}"#,
        );

        let usage = tr.build_usage();
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 1000);
        assert_eq!(class_tokens(&usage.cache_write, "1h"), 0);
        assert_eq!(usage.cache_write.accounted_tokens(), 1000);
        assert_eq!(
            usage.completion_tokens, 0,
            "neither event reports output tokens"
        );
    }

    /// The new member joins the duplicate-rejection rule the other wrapper members follow: two
    /// `usage` objects are two contradictory statements, not one to be resolved last-wins.
    #[test]
    fn test_duplicate_top_level_usage_in_stream_event_is_rejected() {
        let mut tr = translator_with_a_standing_snapshot();
        assert!(
            tr.parse_event(
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":5},"usage":{"input_tokens":10,"output_tokens":999}}"#
            )
            .is_none(),
            "a repeated usage member must not yield a stream event"
        );
        assert_eq!(tr.build_usage().completion_tokens, 0);
    }

    /// Deferring the root `usage` must not cost the duplicate rejection *inside* it.
    ///
    /// The member is buffered until the `type` tag resolves, and a buffer that parsed the object
    /// into a map would collapse these pairs to their last value before any seed saw them —
    /// approving `output_tokens: 1` where the provider stated two contradictory counts, which is
    /// the undercount the seeded visitors exist to fail closed on. Each fixture repeats one member
    /// of one root usage object, which is the shape a repeated *wrapper* test cannot reach.
    #[test]
    fn test_duplicate_members_inside_one_root_usage_are_rejected() {
        for (member, line) in [
            (
                "output_tokens",
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9000,"output_tokens":1}}"#,
            ),
            (
                "input_tokens",
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":9000,"input_tokens":1,"output_tokens":5}}"#,
            ),
            (
                "cache_creation_input_tokens",
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5,"cache_creation_input_tokens":9000,"cache_creation_input_tokens":1}}"#,
            ),
            (
                "cache_read_input_tokens",
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5,"cache_read_input_tokens":9000,"cache_read_input_tokens":1}}"#,
            ),
            (
                "cache_creation",
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5,"cache_creation":{"ephemeral_5m_input_tokens":9000},"cache_creation":{"ephemeral_5m_input_tokens":1}}}"#,
            ),
        ] {
            let mut tr = translator_with_a_standing_snapshot();
            assert!(
                tr.parse_event(line).is_none(),
                "a repeated {member} inside the root usage must not yield a stream event"
            );
            let usage = tr.build_usage();
            assert_eq!(
                usage.completion_tokens, 0,
                "a rejected {member} must report no output tokens"
            );
            assert_eq!(
                class_tokens(&usage.cache_write, "5m"),
                1000,
                "a rejected {member} must leave the standing snapshot untouched"
            );
        }
    }

    /// The other half of the same rule: members repeated *inside* one `cache_creation` are two
    /// observations of one class and still sum, on the event root as on a buffered body. The
    /// duplicate rejection above is about struct members; this grammar is deliberately different,
    /// and a lossless buffer has to preserve both behaviours rather than trade one for the other.
    #[test]
    fn test_repeated_classes_inside_one_root_cache_creation_still_sum() {
        let mut tr = translator_with_a_standing_snapshot();
        assert!(
            tr.parse_event(
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5,"cache_creation":{"ephemeral_5m_input_tokens":100,"ephemeral_5m_input_tokens":100}}}"#
            )
            .is_some(),
            "repeated class members are accumulated, not a duplicate-member error"
        );
        let usage = tr.build_usage();
        assert_eq!(class_tokens(&usage.cache_write, "5m"), 200);
        assert_eq!(usage.cache_write.accounted_tokens(), 200);
        assert!(usage.cache_write.duplicate().configured_duplicate);
    }

    #[test]
    fn test_stream_message_delta_emits_usage() {
        let start = stream_event(
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":10,"output_tokens":0}}}"#,
        );
        let delta = stream_event(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":5}}"#,
        );
        let mut tr = StreamTranslator::new(
            "claude".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
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
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":5}}"#,
        );
        let mut tr = StreamTranslator::new(
            "claude".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
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
        let mut tr = StreamTranslator::new(
            "claude".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
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
        let mut tr = StreamTranslator::new(
            "claude".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
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
        let mut tr = StreamTranslator::new(
            "claude".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
        let out = tr.process_event(&ev).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_stream_error_event() {
        let ev = stream_event(r#"{"type":"error","error":{"message":"overloaded"}}"#);
        let mut tr = StreamTranslator::new(
            "claude".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
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
                cache_creation_present: false,
                input_tokens_present: true,
            },
        };
        let err = anthropic_to_chat_response(
            &resp,
            "claude",
            "req-1",
            3,
            &bundled_pricing_context(),
            CacheWriteAccumulator::new(CacheWriteClassRegistry::default()),
        )
        .unwrap_err();
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
                cache_creation_present: false,
                input_tokens_present: true,
            },
        };
        // "{}" is 2 bytes; cap of 2 means len == cap, which is NOT > cap, so must be Ok.
        assert!(
            anthropic_to_chat_response(
                &resp,
                "claude",
                "req-1",
                2,
                &bundled_pricing_context(),
                CacheWriteAccumulator::new(CacheWriteClassRegistry::default())
            )
            .is_ok()
        );
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
        let mut tr =
            StreamTranslator::new("claude".into(), "rid".into(), 3, bundled_pricing_context());
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
        let mut tr =
            StreamTranslator::new("claude".into(), "rid".into(), 4, bundled_pricing_context());
        tr.process_event(&block).unwrap();
        assert!(tr.process_event(&delta1).is_ok());
        assert!(tr.process_event(&delta2).is_ok());
    }

    /// `message_stop` is the provider saying the response completed, and it is the last thing the
    /// wire protocol carries — so the terminator it translates to is the stream's clean end.
    #[test]
    fn test_message_stop_chunk_is_terminal() {
        let mut tr = StreamTranslator::new(
            "claude-sonnet-4-5-20251022".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
        let chunk = feed(&mut tr, r#"{"type":"message_stop"}"#).expect("message_stop emits");
        assert!(
            chunk.is_final,
            "the translated [DONE] is the clean end of the response"
        );
    }

    /// Only the terminator is terminal. A delta is mid-response no matter what follows it, and a
    /// `message_delta` carrying the final `stop_reason` is still not the end of the stream.
    #[test]
    fn test_mid_stream_chunks_are_not_terminal() {
        let mut tr = StreamTranslator::new(
            "claude-sonnet-4-5-20251022".into(),
            "rid".into(),
            usize::MAX,
            bundled_pricing_context(),
        );
        feed(
            &mut tr,
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":10,"output_tokens":0}}}"#,
        );
        let content = feed(
            &mut tr,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        )
        .expect("content delta emits");
        assert!(!content.is_final, "a content delta is mid-response");

        let final_delta = feed(
            &mut tr,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":5}}"#,
        )
        .expect("message_delta emits");
        assert!(
            !final_delta.is_final,
            "a stop_reason is not the end of the stream — message_stop is"
        );
    }

    /// A degraded overflow termination is not a clean provider completion.
    ///
    /// The gateway generates this event because it gave up buffering a tool call, not because the
    /// upstream finished. Marking it would assert a completion that never happened, so a later
    /// change that marks it must fail here rather than pass silently.
    #[test]
    fn test_overflow_termination_is_not_terminal() {
        let chunk = overflow_sse_event(&ProviderError::ToolCallBufferOverflow {
            provider: "anthropic",
            tool_call_id: "tu1".into(),
            cap_bytes: 16,
        });
        assert!(
            !chunk.is_final,
            "an aborted stream must never claim the upstream completed"
        );
    }
}
