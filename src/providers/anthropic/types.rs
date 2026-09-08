// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Anthropic Messages API wire types.
//!
//! Serde structs for POST /v1/messages request and response.

use std::fmt;

use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::domain::usage_accounting::{
    CacheWriteAccumulator, CacheWriteClass, CacheWriteClassRegistry, CacheWriteDetailsSeed,
    CacheWriteKeyGrammar,
};

/// Anthropic Messages API request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Extended thinking (beta). Requires anthropic-beta header.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

/// Single message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Vec<ContentBlock>,
}

/// Content block: text, tool_use, tool_result, or thinking.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Tool result returned by the caller; maps from OpenAI Role::Tool messages.
    /// Anthropic requires this to be a user-role message with tool_use_id set.
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
    /// Extended thinking block (beta). Stripped from response; tokens surfaced only.
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

/// Tool definition for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Tool choice: auto, any (forced), or specific tool by name.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicToolChoice {
    #[serde(rename = "auto")]
    Auto,
    /// Forces the model to call at least one tool (OpenAI "required").
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
}

/// Extended thinking config (beta).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub type_: String,
    pub budget_tokens: u32,
}

/// Anthropic Messages API response body (non-streaming).
#[derive(Debug, Clone, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

/// Token usage from Anthropic response.
///
/// There is deliberately no derived `Deserialize`: the `cache_creation` detail object is credited
/// straight into the request's cache-write accumulator by [`AnthropicUsageSeed`], and a derived
/// impl has no accumulator to credit — it could only drop the per-class breakdown silently and
/// under-bill. Parsing goes through the seed, and the compiler is what enforces that.
#[derive(Debug, Clone, Serialize, Default)]
pub struct AnthropicUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    /// Thinking tokens (extended thinking beta).
    pub output_tokens_details: Option<OutputTokensDetails>,
    /// Whether the payload carried a `cache_creation` detail object at all.
    ///
    /// The tokens themselves are in the accumulator, not here. This records only whether the
    /// provider stated a per-class breakdown, which is what decides whether an aggregate may be
    /// attributed to the documented default class or must reconcile against stated details.
    #[serde(skip)]
    pub cache_creation_present: bool,
    /// Whether the payload stated a usable `input_tokens` — absent and explicit `null` both
    /// count as unstated.
    ///
    /// [`Self::input_tokens`] is a bare `u64`, so a position that tolerates the member's absence
    /// resolves it to `0` and the fabricated zero is otherwise indistinguishable from a stated
    /// one. Only `message_delta` fabricates, and nothing reads input tokens from that event;
    /// this records the difference so that nothing ever bills the fabrication.
    #[serde(skip)]
    pub input_tokens_present: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub thinking_tokens: Option<u64>,
}

/// SSE stream event types.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartMessage },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlockStartBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: ContentBlockDelta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    /// `usage` is a **sibling** of `delta` on this event, not a member of it, and Anthropic
    /// always sends it. It is required rather than optional deliberately: an `Option` would let a
    /// malformed or drifted event yield a *successful* final chunk reporting zero completion
    /// tokens, which is indistinguishable from a genuinely empty response. A missing member fails
    /// the parse instead, so the stream produces no usage rather than a confident wrong one.
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDelta,
        usage: AnthropicUsage,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "error")]
    Error { error: StreamError },
    #[serde(rename = "ping")]
    Ping,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageStartMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub role: String,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlockStartBlock {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Option<serde_json::Value>,
    },
    #[serde(rename = "thinking")]
    Thinking,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlockDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
    /// Extended thinking (beta). Stripped from stream; tokens surfaced only. Logged when seen.
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
}

/// The `delta` member of a `message_delta` event.
///
/// It carries the terminal `stop_reason` and nothing billable — the event's `usage` sits beside
/// this object, not inside it.
#[derive(Debug, Clone, Serialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamError {
    #[serde(default)]
    pub message: Option<String>,
}

// ---------------------------------------------------------------------------------------------
// Seeded deserialization
// ---------------------------------------------------------------------------------------------
//
// Anthropic reports cache-write tokens as an open set of `ephemeral_<duration>_input_tokens`
// members. Deserializing that object into any collection loses billing facts — a map keeps only
// the last of two identical members, and a vector is an unbounded provider-controlled allocation
// — so the object is credited into the request's bounded accumulator as it streams, by a
// `DeserializeSeed`. A derive never invokes a seed, so every type between the payload root and
// that object is deserialized by hand here.

/// The member Anthropic reports the cache-write aggregate under.
pub const ANTHROPIC_CACHE_WRITE_AGGREGATE_KEY: &str = "cache_creation_input_tokens";

/// Anthropic's documented default cache duration.
///
/// An aggregate that arrives with no per-class breakdown behind it is a write to this class, so
/// it stays exactly priced instead of falling back. See
/// <https://docs.anthropic.com/en/docs/about-claude/pricing>.
pub const ANTHROPIC_DEFAULT_CACHE_WRITE_CLASS: &str = "5m";

/// Anthropic's cache-write member naming: `ephemeral_<duration>_input_tokens`.
///
/// The spelling is Anthropic's own — other providers name the same concept nothing like it — so
/// it lives in this lane rather than in the accumulator that consumes its output. A member that
/// does not match names no class and is never guessed at.
pub struct AnthropicCacheWriteGrammar;

/// The single grammar instance, borrowed for `'static` by the seeds below.
pub static ANTHROPIC_CACHE_WRITE_GRAMMAR: AnthropicCacheWriteGrammar = AnthropicCacheWriteGrammar;

impl CacheWriteKeyGrammar for AnthropicCacheWriteGrammar {
    fn class_of(&self, raw_key: &str) -> Option<CacheWriteClass> {
        let duration = raw_key
            .strip_prefix("ephemeral_")?
            .strip_suffix("_input_tokens")?;
        CacheWriteClass::canonicalize(duration)
    }
}

/// Matches a member name against a fixed table without allocating a copy of it.
///
/// Yields the index of the matching entry, or `None` for a member the table does not name — which
/// the callers below drain. Serde's derive generates the same thing per struct; one shared seed
/// keeps the five hand-written visitors here from repeating it five times.
struct FieldIndexSeed(&'static [&'static str]);

impl<'de> DeserializeSeed<'de> for FieldIndexSeed {
    type Value = Option<usize>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_str(self)
    }
}

impl Visitor<'_> for FieldIndexSeed {
    type Value = Option<usize>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a member name")
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        Ok(self.0.iter().position(|name| *name == value))
    }
}

/// Rejects a member the visitor has already consumed, the way a derived deserializer does.
///
/// The visitors below replaced derived impls on the money path, and a derived impl fails on a
/// repeated struct member. Last-value-wins would let a second `input_tokens` — or a second
/// `cache_creation_input_tokens` — replace a larger quantity with a smaller one and undercharge
/// the request, silently broadening accepted input where the previous implementation failed
/// closed. Unknown members are drained without tracking, matching a derive that has no
/// `deny_unknown_fields`.
///
/// This is about *struct members*. The members **inside** a `cache_creation` object repeat
/// deliberately and are summed, because two identical member names there are two observations of
/// one class — a different grammar, handled by its own seed.
struct SeenFields {
    seen: u32,
    names: &'static [&'static str],
}

impl SeenFields {
    /// Starts tracking a field table. The table indexes a `u32` bitset, so it must be small
    /// enough to fit one; every table in this module has at most seven entries.
    fn new(names: &'static [&'static str]) -> Self {
        debug_assert!(names.len() <= u32::BITS as usize, "field table too wide");
        Self { seen: 0, names }
    }

    /// Records that member `index` was consumed, or fails if it already had been.
    fn mark<E: de::Error>(&mut self, index: usize) -> Result<(), E> {
        let bit = 1u32 << index;
        if self.seen & bit != 0 {
            return Err(de::Error::duplicate_field(self.names[index]));
        }
        self.seen |= bit;
        Ok(())
    }
}

/// Where a seeded parse puts the cache-write details it reads, before anything commits them.
///
/// The seeds below never write into a request's live accumulator. They fill a candidate, and the
/// caller commits it only once the whole payload has resolved *and* the usage object that carried
/// the detail object turns out to belong to the event that was actually parsed. A payload the
/// parse rejects — an unknown event type, a missing member, trailing data after the object — has
/// therefore moved no billing, and neither has a usage-shaped member hanging off an event that
/// carries no usage of its own.
///
/// A candidate is bounded by exactly what bounds the live accumulator: it is seeded from the same
/// registry, so its state stays `O(configured classes + retained evidence entries)` however many
/// members a provider sends.
struct CacheWriteCandidate<'a> {
    registry: &'a CacheWriteClassRegistry,
    accumulator: &'a mut Option<CacheWriteAccumulator>,
}

impl<'a> CacheWriteCandidate<'a> {
    /// Points a candidate at the slot a resolved parse will read its proposal out of.
    fn new(
        registry: &'a CacheWriteClassRegistry,
        accumulator: &'a mut Option<CacheWriteAccumulator>,
    ) -> Self {
        Self {
            registry,
            accumulator,
        }
    }

    /// Lends this candidate to a nested seed without giving up the slot.
    fn reborrow(&mut self) -> CacheWriteCandidate<'_> {
        CacheWriteCandidate {
            registry: self.registry,
            accumulator: &mut *self.accumulator,
        }
    }

    /// Starts a fresh detail snapshot, discarding any earlier proposal in this slot.
    ///
    /// A provider detail object is cumulative rather than incremental, so a later object replaces
    /// an earlier one instead of adding to it.
    fn begin(&mut self) -> &mut CacheWriteAccumulator {
        self.accumulator
            .insert(CacheWriteAccumulator::new(self.registry.clone()))
    }
}

/// Deserializes a `cache_creation` member into a candidate, tolerating an explicit `null`.
///
/// Yields whether a detail object was actually present, which is what distinguishes a provider
/// that stated a breakdown from one that stated only an aggregate.
struct CacheCreationMemberSeed<'a> {
    candidate: CacheWriteCandidate<'a>,
}

impl<'de> DeserializeSeed<'de> for CacheCreationMemberSeed<'_> {
    type Value = bool;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<bool, D::Error> {
        deserializer.deserialize_option(self)
    }
}

impl<'de> Visitor<'de> for CacheCreationMemberSeed<'_> {
    type Value = bool;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a cache creation detail object or null")
    }

    fn visit_none<E: de::Error>(self) -> Result<bool, E> {
        Ok(false)
    }

    fn visit_unit<E: de::Error>(self) -> Result<bool, E> {
        Ok(false)
    }

    fn visit_some<D: Deserializer<'de>>(mut self, deserializer: D) -> Result<bool, D::Error> {
        CacheWriteDetailsSeed::new(self.candidate.begin(), &ANTHROPIC_CACHE_WRITE_GRAMMAR)
            .deserialize(deserializer)?;
        Ok(true)
    }
}

/// Whether a usage object must state `input_tokens`, which depends on where it appears.
///
/// Anthropic types the member as required and non-nullable on a buffered `Message.usage` and on
/// `message_start`, and in both positions it is the request's only input-token source, so a
/// payload that omits it is a payload the gateway cannot bill and must reject loudly.
///
/// On `message_delta` the same member is documented as optional and typed nullable. This lane
/// never reads input tokens from that event — `message_start` remains the source — so demanding
/// it there would drop the whole terminal event over a value that is not used, losing
/// `finish_reason` and the output count with it.
#[derive(Clone, Copy)]
enum InputTokensRule {
    /// An absent or explicitly null `input_tokens` fails the parse.
    Required,
    /// An absent or explicitly null `input_tokens` resolves to `0`.
    DefaultsToZero,
}

/// Deserializes an [`AnthropicUsage`], proposing its cache-write detail object into `candidate`.
///
/// `output_tokens` is required in every position, because every position that parses a usage
/// object is one that reads it. That holds because no seed runs until the payload the object
/// belongs to has been identified — see [`StreamEventSeed`], which buffers the SSE event root's
/// `usage` rather than seeding it while the `type` tag is still unknown.
struct AnthropicUsageSeed<'a> {
    candidate: CacheWriteCandidate<'a>,
    input_tokens_rule: InputTokensRule,
}

impl<'a> AnthropicUsageSeed<'a> {
    /// Seeds a usage deserialization in a position where `input_tokens` is the request's only
    /// source for that count and is therefore required.
    fn new(candidate: CacheWriteCandidate<'a>) -> Self {
        Self {
            candidate,
            input_tokens_rule: InputTokensRule::Required,
        }
    }
}

const USAGE_FIELDS: &[&str] = &[
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens_details",
    "cache_creation",
];

impl<'de> DeserializeSeed<'de> for AnthropicUsageSeed<'_> {
    type Value = AnthropicUsage;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for AnthropicUsageSeed<'_> {
    type Value = AnthropicUsage;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an Anthropic usage object")
    }

    fn visit_map<A: MapAccess<'de>>(mut self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut usage = AnthropicUsage::default();
        let mut input_tokens = None;
        let mut output_tokens = None;
        let mut seen = SeenFields::new(USAGE_FIELDS);
        while let Some(field) = map.next_key_seed(FieldIndexSeed(USAGE_FIELDS))? {
            if let Some(index) = field {
                seen.mark(index)?;
            }
            match field {
                Some(0) => {
                    input_tokens = match self.input_tokens_rule {
                        // `u64` rejects an explicit null, which is what the strict positions
                        // want: there, a null input_tokens is as unbillable as an absent one.
                        InputTokensRule::Required => Some(map.next_value()?),
                        InputTokensRule::DefaultsToZero => map.next_value::<Option<u64>>()?,
                    };
                }
                Some(1) => output_tokens = Some(map.next_value()?),
                Some(2) => usage.cache_creation_input_tokens = map.next_value()?,
                Some(3) => usage.cache_read_input_tokens = map.next_value()?,
                Some(4) => usage.output_tokens_details = map.next_value()?,
                Some(5) => {
                    usage.cache_creation_present =
                        map.next_value_seed(CacheCreationMemberSeed {
                            candidate: self.candidate.reborrow(),
                        })?;
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        usage.input_tokens_present = input_tokens.is_some();
        usage.input_tokens = match (input_tokens, self.input_tokens_rule) {
            (Some(tokens), _) => tokens,
            (None, InputTokensRule::DefaultsToZero) => 0,
            (None, InputTokensRule::Required) => {
                return Err(de::Error::missing_field("input_tokens"));
            }
        };
        // An absent count is fatal rather than zero: a confident zero is indistinguishable from a
        // genuinely empty response, which is the failure this lane exists to keep off the bill.
        usage.output_tokens =
            output_tokens.ok_or_else(|| de::Error::missing_field("output_tokens"))?;
        Ok(usage)
    }
}

/// Deserializes an optional usage member, seeding the object when one is present.
struct OptionalUsageSeed<'a> {
    candidate: CacheWriteCandidate<'a>,
    /// Carried through to the object itself, because whether `input_tokens` is required is a
    /// property of the position this member occupies, not of the usage shape.
    input_tokens_rule: InputTokensRule,
}

impl<'de> DeserializeSeed<'de> for OptionalUsageSeed<'_> {
    type Value = Option<AnthropicUsage>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_option(self)
    }
}

impl<'de> Visitor<'de> for OptionalUsageSeed<'_> {
    type Value = Option<AnthropicUsage>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an Anthropic usage object or null")
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        AnthropicUsageSeed {
            candidate: self.candidate,
            input_tokens_rule: self.input_tokens_rule,
        }
        .deserialize(deserializer)
        .map(Some)
    }
}

/// A parsed buffered response and the cache-write snapshot its usage object proposed.
///
/// The snapshot is handed back uncommitted so the caller can reject the whole body first — a
/// response with trailing data after it is not a response, and nothing it stated may bill.
pub struct SeededMessagesResponse {
    /// The response body.
    pub response: MessagesResponse,
    /// The detail snapshot `usage.cache_creation` stated, or `None` when it stated none.
    pub cache_write: Option<CacheWriteAccumulator>,
}

/// Deserializes a buffered [`MessagesResponse`], proposing its cache-write details alongside it.
pub struct MessagesResponseSeed<'a> {
    registry: &'a CacheWriteClassRegistry,
}

impl<'a> MessagesResponseSeed<'a> {
    /// Seeds a buffered response parse with the request's configured-class registry.
    pub fn new(registry: &'a CacheWriteClassRegistry) -> Self {
        Self { registry }
    }
}

const RESPONSE_FIELDS: &[&str] = &["id", "type", "role", "content", "stop_reason", "usage"];

impl<'de> DeserializeSeed<'de> for MessagesResponseSeed<'_> {
    type Value = SeededMessagesResponse;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for MessagesResponseSeed<'_> {
    type Value = SeededMessagesResponse;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an Anthropic messages response")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut id = None;
        // `type` and `stop_reason` are optional on the wire and stay optional here: a derived
        // impl resolves a missing `Option` member to `None` rather than failing, and a response
        // that omits either one parsed before this lane was hand-written.
        let mut type_ = None;
        let mut role = None;
        let mut content = None;
        let mut stop_reason = None;
        let mut usage = None;
        let mut cache_write = None;
        let mut seen = SeenFields::new(RESPONSE_FIELDS);
        while let Some(field) = map.next_key_seed(FieldIndexSeed(RESPONSE_FIELDS))? {
            if let Some(index) = field {
                seen.mark(index)?;
            }
            match field {
                Some(0) => id = Some(map.next_value()?),
                Some(1) => type_ = map.next_value()?,
                Some(2) => role = Some(map.next_value()?),
                Some(3) => content = Some(map.next_value()?),
                Some(4) => stop_reason = map.next_value()?,
                Some(5) => {
                    usage = Some(map.next_value_seed(AnthropicUsageSeed::new(
                        CacheWriteCandidate::new(self.registry, &mut cache_write),
                    ))?);
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(SeededMessagesResponse {
            response: MessagesResponse {
                id: id.ok_or_else(|| de::Error::missing_field("id"))?,
                type_,
                role: role.ok_or_else(|| de::Error::missing_field("role"))?,
                content: content.ok_or_else(|| de::Error::missing_field("content"))?,
                stop_reason,
                usage: usage.ok_or_else(|| de::Error::missing_field("usage"))?,
            },
            cache_write,
        })
    }
}

/// Deserializes the `message` object of a `message_start` event.
struct MessageStartMessageSeed<'a> {
    candidate: CacheWriteCandidate<'a>,
}

const MESSAGE_START_FIELDS: &[&str] = &["id", "type", "role", "usage"];

impl<'de> DeserializeSeed<'de> for MessageStartMessageSeed<'_> {
    type Value = MessageStartMessage;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for MessageStartMessageSeed<'_> {
    type Value = MessageStartMessage;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a message_start message object")
    }

    fn visit_map<A: MapAccess<'de>>(mut self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut id = None;
        let mut type_ = None;
        let mut role = None;
        let mut usage = None;
        let mut seen = SeenFields::new(MESSAGE_START_FIELDS);
        while let Some(field) = map.next_key_seed(FieldIndexSeed(MESSAGE_START_FIELDS))? {
            if let Some(index) = field {
                seen.mark(index)?;
            }
            match field {
                Some(0) => id = Some(map.next_value()?),
                Some(1) => type_ = Some(map.next_value()?),
                Some(2) => role = Some(map.next_value()?),
                Some(3) => {
                    // `message_start` is this lane's only input-token source, so a usage object
                    // here that states no input_tokens is rejected rather than defaulted.
                    usage = map.next_value_seed(OptionalUsageSeed {
                        candidate: self.candidate.reborrow(),
                        input_tokens_rule: InputTokensRule::Required,
                    })?;
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(MessageStartMessage {
            id: id.ok_or_else(|| de::Error::missing_field("id"))?,
            type_: type_.ok_or_else(|| de::Error::missing_field("type"))?,
            role: role.ok_or_else(|| de::Error::missing_field("role"))?,
            usage,
        })
    }
}

/// Every member either delta shape can carry, collected before the event tag decides which it is.
///
/// `delta` is the one member name two stream variants share, and JSON member order is not a
/// contract — the tag may arrive after it. Collecting the union and letting the tag choose keeps
/// the outer tag authoritative without depending on where in the object it appeared.
#[derive(Default)]
struct AnyDelta {
    /// Index into [`CONTENT_DELTA_TYPES`], or `None` for an inner `type` that names no variant.
    delta_type: Option<usize>,
    saw_type: bool,
    text: Option<String>,
    partial_json: Option<String>,
    thinking: Option<String>,
    stop_reason: Option<String>,
}

const CONTENT_DELTA_TYPES: &[&str] = &["text_delta", "input_json_delta", "thinking_delta"];

impl AnyDelta {
    fn into_content_block_delta<E: de::Error>(self) -> Result<ContentBlockDelta, E> {
        match self.delta_type {
            Some(0) => Ok(ContentBlockDelta::Text {
                text: self.text.ok_or_else(|| de::Error::missing_field("text"))?,
            }),
            Some(1) => Ok(ContentBlockDelta::InputJson {
                partial_json: self
                    .partial_json
                    .ok_or_else(|| de::Error::missing_field("partial_json"))?,
            }),
            Some(2) => Ok(ContentBlockDelta::Thinking {
                thinking: self
                    .thinking
                    .ok_or_else(|| de::Error::missing_field("thinking"))?,
            }),
            _ if self.saw_type => Err(de::Error::custom("unknown content block delta type")),
            _ => Err(de::Error::missing_field("type")),
        }
    }

    /// `stop_reason` is optional on the wire — Anthropic omits it until the final delta — and a
    /// derived impl resolved a missing `Option` member to `None` rather than failing the event.
    fn into_message_delta(self) -> MessageDelta {
        MessageDelta {
            stop_reason: self.stop_reason,
        }
    }
}

/// Neither delta shape carries usage, so this seed proposes no cache-write candidate: a
/// usage-shaped member found inside a `delta` belongs to nothing the event reports and is
/// ignored rather than read.
struct AnyDeltaSeed;

const DELTA_FIELDS: &[&str] = &["type", "text", "partial_json", "thinking", "stop_reason"];

impl<'de> DeserializeSeed<'de> for AnyDeltaSeed {
    type Value = AnyDelta;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for AnyDeltaSeed {
    type Value = AnyDelta;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a stream delta object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut delta = AnyDelta::default();
        let mut seen = SeenFields::new(DELTA_FIELDS);
        while let Some(field) = map.next_key_seed(FieldIndexSeed(DELTA_FIELDS))? {
            if let Some(index) = field {
                seen.mark(index)?;
            }
            match field {
                Some(0) => {
                    delta.saw_type = true;
                    delta.delta_type = map.next_value_seed(FieldIndexSeed(CONTENT_DELTA_TYPES))?;
                }
                Some(1) => delta.text = Some(map.next_value()?),
                Some(2) => delta.partial_json = Some(map.next_value()?),
                Some(3) => delta.thinking = Some(map.next_value()?),
                Some(4) => delta.stop_reason = map.next_value()?,
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(delta)
    }
}

/// One parsed SSE event and the cache-write snapshot its own usage object proposed.
///
/// `cache_write` is `Some` only when the resolved event is the one that carried the detail
/// object: a `cache_creation` reached through a `message` member on a `ping`, or through a
/// top-level `usage` on a `content_block_delta`, belongs to no usage this event reports and never
/// reaches billing — the first is parsed and discarded, the second is not parsed at all.
pub struct SeededStreamEvent {
    /// The event.
    pub event: StreamEvent,
    /// The detail snapshot the event's usage object stated, or `None` when it stated none.
    pub cache_write: Option<CacheWriteAccumulator>,
}

/// Deserializes one SSE [`StreamEvent`], proposing any cache-write details alongside it.
pub struct StreamEventSeed<'a> {
    registry: &'a CacheWriteClassRegistry,
}

impl<'a> StreamEventSeed<'a> {
    /// Seeds a stream event parse with the request's configured-class registry.
    pub fn new(registry: &'a CacheWriteClassRegistry) -> Self {
        Self { registry }
    }
}

const EVENT_FIELDS: &[&str] = &[
    "type",
    "message",
    "index",
    "content_block",
    "delta",
    "error",
    "usage",
];

/// Index of `"usage"` in [`EVENT_FIELDS`], which is the one member handled after the tag.
const EVENT_USAGE_FIELD: usize = 6;

const EVENT_TYPES: &[&str] = &[
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "error",
    "ping",
];

impl<'de> DeserializeSeed<'de> for StreamEventSeed<'_> {
    type Value = SeededStreamEvent;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for StreamEventSeed<'_> {
    type Value = SeededStreamEvent;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an Anthropic stream event")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut event_type = None;
        let mut saw_type = false;
        let mut message = None;
        let mut index = None;
        let mut content_block = None;
        let mut delta = None;
        let mut error = None;
        // The event root's `usage` is *buffered*, not parsed. Every event carries this position
        // and only `message_delta` reads it, so nothing about the object may be judged until the
        // `type` tag has resolved: a repeat of the member, or a value that is not a usage object
        // at all, would otherwise fail an event that never consumes it — throwing away a
        // `content_block_delta`'s generated text over an accounting member that is not its own.
        // The buffer is bounded by the SSE line the caller already holds in memory.
        //
        // It holds the member's *original bytes*, not a `Value`. A `Value`'s object is a map, so
        // parking billing input in one would silently resolve `{"output_tokens": 9000,
        // "output_tokens": 1}` last-wins and hand the seed a single member to approve — deferring
        // the judgement would have destroyed the evidence it is deferred for.
        let mut usage: Option<Box<serde_json::value::RawValue>> = None;
        let mut duplicate_usage = false;
        // `message_start`'s candidate, held uncommitted until the tag confirms the event carries
        // it. The event root has no candidate here: its usage object is not deserialized at all
        // unless the tag turns out to be `message_delta`, and the candidate is created there.
        let mut message_cache_write = None;
        let mut seen = SeenFields::new(EVENT_FIELDS);
        while let Some(field) = map.next_key_seed(FieldIndexSeed(EVENT_FIELDS))? {
            // `usage` is deliberately outside the duplicate bitset: rejecting a repeat here would
            // fail every event carrying one. The repeat is recorded and rejected below, once the
            // tag says the member is this event's own.
            if let Some(index) = field
                && index != EVENT_USAGE_FIELD
            {
                seen.mark(index)?;
            }
            match field {
                Some(0) => {
                    saw_type = true;
                    event_type = map.next_value_seed(FieldIndexSeed(EVENT_TYPES))?;
                }
                Some(1) => {
                    message = Some(map.next_value_seed(MessageStartMessageSeed {
                        candidate: CacheWriteCandidate::new(
                            self.registry,
                            &mut message_cache_write,
                        ),
                    })?);
                }
                Some(2) => index = Some(map.next_value()?),
                Some(3) => content_block = Some(map.next_value()?),
                Some(4) => delta = Some(map.next_value_seed(AnyDeltaSeed)?),
                Some(5) => error = Some(map.next_value()?),
                Some(EVENT_USAGE_FIELD) => {
                    duplicate_usage |= usage.is_some();
                    usage = Some(map.next_value()?);
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }

        let missing = |field| de::Error::missing_field(field);
        let (event, cache_write) = match event_type {
            Some(0) => (
                StreamEvent::MessageStart {
                    message: message.ok_or_else(|| missing("message"))?,
                },
                message_cache_write,
            ),
            Some(1) => (
                StreamEvent::ContentBlockStart {
                    index: index.ok_or_else(|| missing("index"))?,
                    content_block: content_block.ok_or_else(|| missing("content_block"))?,
                },
                None,
            ),
            Some(2) => (
                StreamEvent::ContentBlockDelta {
                    index: index.ok_or_else(|| missing("index"))?,
                    delta: delta
                        .ok_or_else(|| missing("delta"))?
                        .into_content_block_delta()?,
                },
                None,
            ),
            Some(3) => (
                StreamEvent::ContentBlockStop {
                    index: index.ok_or_else(|| missing("index"))?,
                },
                None,
            ),
            Some(4) => {
                // Everything deferred at the member is judged here, where the tag has confirmed
                // the object is this event's own: two usage members are two contradictory
                // statements rather than one to resolve last-wins, and the object itself is now
                // parsed for the first time. `input_tokens` is the one member tolerated as absent
                // or explicitly null — Anthropic types it nullable on this event, and this lane
                // reads input tokens from `message_start` instead.
                if duplicate_usage {
                    return Err(de::Error::duplicate_field("usage"));
                }
                let raw = usage.ok_or_else(|| missing("usage"))?;
                let mut event_cache_write = None;
                // Re-read the buffered bytes rather than a parsed value, so the seed sees the
                // member sequence the provider actually sent — duplicates included.
                let usage = OptionalUsageSeed {
                    candidate: CacheWriteCandidate::new(self.registry, &mut event_cache_write),
                    input_tokens_rule: InputTokensRule::DefaultsToZero,
                }
                .deserialize(&mut serde_json::Deserializer::from_str(raw.get()))
                .map_err(de::Error::custom)?
                .ok_or_else(|| missing("usage"))?;
                (
                    StreamEvent::MessageDelta {
                        delta: delta.ok_or_else(|| missing("delta"))?.into_message_delta(),
                        usage,
                    },
                    event_cache_write,
                )
            }
            Some(5) => (StreamEvent::MessageStop, None),
            Some(6) => (
                StreamEvent::Error {
                    error: error.ok_or_else(|| missing("error"))?,
                },
                None,
            ),
            Some(7) => (StreamEvent::Ping, None),
            _ if saw_type => return Err(de::Error::custom("unknown stream event type")),
            _ => return Err(missing("type")),
        };
        Ok(SeededStreamEvent { event, cache_write })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class_of(raw_key: &str) -> Option<String> {
        ANTHROPIC_CACHE_WRITE_GRAMMAR
            .class_of(raw_key)
            .map(|c| c.as_str().to_owned())
    }

    /// The lane's own member grammar, independent of which classes pricing happens to configure.
    #[test]
    fn test_grammar_reads_the_duration_out_of_the_member_name() {
        assert_eq!(class_of("ephemeral_5m_input_tokens").as_deref(), Some("5m"));
        assert_eq!(class_of("ephemeral_1h_input_tokens").as_deref(), Some("1h"));
        // A duration the gateway has never seen still names a class — that is what makes a new
        // class priceable through pricing data alone.
        assert_eq!(
            class_of("ephemeral_30m_input_tokens").as_deref(),
            Some("30m")
        );
        // Spelling variants fold into one class rather than splitting the quantity across two.
        assert_eq!(
            class_of("ephemeral_05M_input_tokens").as_deref(),
            Some("5m")
        );
    }

    /// A member that does not match the grammar names no class and is never guessed at.
    #[test]
    fn test_grammar_rejects_members_it_does_not_recognise() {
        assert_eq!(class_of("ephemeral_5m_tokens"), None);
        assert_eq!(class_of("cache_creation_input_tokens"), None);
        assert_eq!(class_of("ephemeral__input_tokens"), None);
        assert_eq!(class_of("ephemeral_5x_input_tokens"), None);
        assert_eq!(class_of("ephemeral_input_tokens"), None);
    }
}
