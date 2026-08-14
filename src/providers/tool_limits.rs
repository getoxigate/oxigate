// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Per-provider tool-use limits and shared constants.
//!
//! Values track upstream API documentation; reviewed quarterly per provider docs.

/// Maximum tools per request for Anthropic.
/// Source: https://docs.anthropic.com/en/docs/tool-use (reviewed 2026-05-05).
pub const ANTHROPIC_MAX_TOOLS: usize = 64;

/// Maximum function declarations per request for Gemini.
/// Source: https://ai.google.dev/gemini-api/docs/function-calling (reviewed 2026-05-05).
pub const GEMINI_MAX_TOOLS: usize = 128;

/// Maximum tools per request for Bedrock Converse.
/// Note: this is model-dependent; const is the documented upper bound.
/// Source: Bedrock Converse API service quotas (reviewed 2026-05-05).
pub const BEDROCK_MAX_TOOLS: usize = 64;

/// Default per-call tool-argument streaming buffer cap for Anthropic.
///
/// Operators override via `providers.anthropic.tool_call_buffer_cap_bytes` in YAML.
/// Very large values (e.g. above 64 MiB) are untested and can increase gateway memory pressure.
pub const DEFAULT_TOOL_CALL_BUFFER_CAP_BYTES: usize = 1048576;

/// Hard upper bound on `tool_call_buffer_cap_bytes`.
///
/// With `ANTHROPIC_MAX_TOOLS = 64` slots, a cap at this limit caps in-flight memory at 4 GiB
/// per request in the worst case — anything higher is an operator misconfiguration.
pub const MAX_TOOL_CALL_BUFFER_CAP_BYTES: usize = 64 * 1024 * 1024; // 64 MiB

/// Maximum byte length accepted for a single `tool_call.function.arguments` string.
///
/// A global 50 MiB `DefaultBodyLimit` already bounds entire request bodies, but an unbounded
/// arguments string can still trigger deep serde_json recursion before the JSON is valid.
/// 64 KiB covers all real-world tool-call payloads; values above this indicate a client bug.
pub const TOOL_ARGS_MAX_BYTES: usize = 64 * 1024; // 64 KiB

// ── NotYetSupported feature identifiers ──────────────────────────────────────

/// Bedrock streaming tool deltas — tracked in.
pub const FEATURE_BEDROCK_STREAMING_TOOL_USE: &str = "bedrock_streaming_tool_use";
