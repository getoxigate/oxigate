// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Gateway-level tool-schema validation.
//!
//! Lives in the domain layer; has no knowledge of specific providers or HTTP.
//! Called once in the API handler before any provider dispatch so all adapters
//! are covered without per-adapter boilerplate.

use tracing::warn;

use crate::domain::chat::ChatRequest;
use crate::domain::ports::ProviderError;

// ── Protocol error-code strings (appear in JSON response bodies) ──────────────

pub const ERR_TOOL_CHOICE_UNSUPPORTED: &str = "tool_choice_unsupported";
pub const ERR_TOOL_COUNT_EXCEEDED: &str = "tool_count_exceeded";
pub const ERR_MALFORMED_TOOL_SCHEMA: &str = "malformed_tool_schema";
pub const ERR_TOOL_CALL_BUFFER_OVERFLOW: &str = "tool_call_buffer_overflow";
pub const ERR_NOT_YET_SUPPORTED: &str = "not_yet_supported";
pub const ERR_TYPE_GATEWAY_ERROR: &str = "gateway_error";

/// Valid string-form `tool_choice` values. Named-function form uses the object
/// `{"type":"function","function":{"name":"X"}}` and is not listed here.
pub const SUPPORTED_TOOL_CHOICE_VALUES: &[&str] = &["auto", "none", "required", "any"];

// ── Reason codes ──────────────────────────────────────────────────────────────

pub(crate) const REASON_NAME_INVALID: &str = "name_invalid";
pub(crate) const REASON_NAME_TOO_LONG: &str = "name_too_long";
pub(crate) const REASON_DESCRIPTION_TOO_LONG: &str = "description_too_long";
pub(crate) const REASON_PARAMETERS_NOT_OBJECT: &str = "parameters_not_object";
pub(crate) const REASON_SCHEMA_TOO_LARGE: &str = "schema_too_large";
pub(crate) const REASON_SCHEMA_TOO_DEEP: &str = "schema_too_deep";
/// Returned when `tool_choice` demands at least one tool but `tools[]` is absent or empty.
pub(crate) const REASON_TOOL_CHOICE_REQUIRES_TOOLS: &str = "tool_choice_requires_tools";

// ── Schema size / depth limits ────────────────────────────────────────────────

/// Maximum byte length of a serialised `parameters` schema per tool (128 KiB).
pub const TOOL_SCHEMA_MAX_BYTES: usize = 128 * 1024;
/// Maximum JSON nesting depth of a `parameters` schema.
pub const TOOL_SCHEMA_MAX_DEPTH: usize = 10;
/// Maximum byte length of a tool function name (mirrors OpenAI's documented limit).
pub const TOOL_NAME_MAX_LEN: usize = 64;
/// Maximum byte length of a tool function description.
pub const TOOL_DESCRIPTION_MAX_LEN: usize = 1024;

/// Validates `req.tools[]` at the gateway level before any provider dispatch.
///
/// Returns `Ok(())` when the request is valid or requires no validation.
/// Returns `Err(reason)` with a `REASON_*` constant on the first violation.
///
/// **Skipped when `tool_choice="none"`** — per OpenAI behaviour, tools provided
/// for context or grounding do not need to be schema-valid when they will not be called.
///
/// **Non-`"function"` tool types** (e.g. `"computer_use_preview"`) are skipped;
/// they are forwarded to upstream verbatim.
///
/// ## Limits enforced (fixed constants)
/// - Function name: non-empty, ≤ [`TOOL_NAME_MAX_LEN`] bytes
/// - Description: ≤ [`TOOL_DESCRIPTION_MAX_LEN`] bytes
/// - `parameters` schema: must be a JSON object, ≤ [`TOOL_SCHEMA_MAX_BYTES`] serialised,
///   JSON nesting depth ≤ [`TOOL_SCHEMA_MAX_DEPTH`]
pub fn validate_request_tools(req: &ChatRequest) -> Result<(), &'static str> {
    let tool_choice_val = req.extra.get("tool_choice");

    if is_tool_choice_none(tool_choice_val) {
        return Ok(());
    }

    let tools = match req.tools.as_deref() {
        Some(t) if !t.is_empty() => t,
        _ => {
            if tool_choice_demands_tools(tool_choice_val) {
                warn!(
                    reason = REASON_TOOL_CHOICE_REQUIRES_TOOLS,
                    "gateway: tool_choice demands tools but tools[] is absent/empty"
                );
                return Err(REASON_TOOL_CHOICE_REQUIRES_TOOLS);
            }
            return Ok(());
        }
    };

    for (i, tool) in tools.iter().enumerate() {
        if tool.type_ != "function" {
            continue; // non-"function" types forwarded verbatim
        }

        if tool.function.name.is_empty() {
            warn!(
                reason = REASON_NAME_INVALID,
                tool_index = i,
                "gateway: malformed function tool rejected"
            );
            return Err(REASON_NAME_INVALID);
        }

        if tool.function.name.len() > TOOL_NAME_MAX_LEN {
            warn!(
                reason = REASON_NAME_TOO_LONG,
                tool_index = i,
                name_len = tool.function.name.len(),
                limit = TOOL_NAME_MAX_LEN,
                "gateway: function name exceeds length limit"
            );
            return Err(REASON_NAME_TOO_LONG);
        }

        if let Some(ref desc) = tool.function.description
            && desc.len() > TOOL_DESCRIPTION_MAX_LEN
        {
            warn!(
                reason = REASON_DESCRIPTION_TOO_LONG,
                tool_index = i,
                desc_len = desc.len(),
                limit = TOOL_DESCRIPTION_MAX_LEN,
                "gateway: function description exceeds length limit"
            );
            return Err(REASON_DESCRIPTION_TOO_LONG);
        }

        if let Some(ref params) = tool.function.parameters {
            if !params.is_object() {
                warn!(
                    reason = REASON_PARAMETERS_NOT_OBJECT,
                    tool_index = i,
                    "gateway: malformed function tool rejected"
                );
                return Err(REASON_PARAMETERS_NOT_OBJECT);
            }

            let serialised = serde_json::to_string(params).unwrap_or_default();
            if serialised.len() > TOOL_SCHEMA_MAX_BYTES {
                warn!(
                    reason = REASON_SCHEMA_TOO_LARGE,
                    tool_index = i,
                    schema_bytes = serialised.len(),
                    limit = TOOL_SCHEMA_MAX_BYTES,
                    "gateway: tool schema exceeds byte limit"
                );
                return Err(REASON_SCHEMA_TOO_LARGE);
            }

            if json_depth(params) > TOOL_SCHEMA_MAX_DEPTH {
                warn!(
                    reason = REASON_SCHEMA_TOO_DEEP,
                    tool_index = i,
                    limit = TOOL_SCHEMA_MAX_DEPTH,
                    "gateway: tool schema exceeds nesting depth limit"
                );
                return Err(REASON_SCHEMA_TOO_DEEP);
            }
        }
    }

    Ok(())
}

/// Returns the maximum JSON nesting depth of `val` (1 = scalar, 2 = object/array with scalars).
fn json_depth(val: &serde_json::Value) -> usize {
    match val {
        serde_json::Value::Object(map) => 1 + map.values().map(json_depth).max().unwrap_or(0),
        serde_json::Value::Array(arr) => 1 + arr.iter().map(json_depth).max().unwrap_or(0),
        _ => 1,
    }
}

pub(crate) fn is_tool_choice_none(val: Option<&serde_json::Value>) -> bool {
    match val {
        Some(serde_json::Value::String(s)) if s == "none" => true,
        Some(serde_json::Value::Object(obj))
            if obj.get("type").and_then(|t| t.as_str()) == Some("none") =>
        {
            true
        }
        _ => false,
    }
}

fn tool_choice_demands_tools(val: Option<&serde_json::Value>) -> bool {
    match val {
        Some(serde_json::Value::String(s)) if s == "required" => true,
        Some(serde_json::Value::Object(obj)) => matches!(
            obj.get("type").and_then(|t| t.as_str()),
            Some("required" | "function")
        ),
        _ => false,
    }
}

// ── Tool choice parsing ───────────────────────────────────────────────────────

/// Provider-independent representation of an OpenAI `tool_choice` value.
///
/// `"none"` is not represented here — callers use `is_tool_choice_none()` to
/// short-circuit before calling `parse_tool_choice_value()`.
#[derive(Debug, Clone)]
pub enum ToolChoiceKind {
    /// Model decides whether to call a tool (`"auto"` / absent on Anthropic).
    Auto,
    /// Model must call at least one tool (`"required"` / `"any"`).
    Required,
    /// Model must call this specific function.
    Function { name: String },
}

/// Parse an OpenAI `tool_choice` value into a provider-independent [`ToolChoiceKind`].
///
/// `None` (absent key) → `Auto`. `"none"` must be handled by the caller via
/// [`is_tool_choice_none()`] before this function is called.
pub fn parse_tool_choice_value(
    val: Option<&serde_json::Value>,
    provider: &'static str,
) -> Result<ToolChoiceKind, ProviderError> {
    match val {
        None => Ok(ToolChoiceKind::Auto),
        Some(serde_json::Value::String(s)) => match s.as_str() {
            "auto" => Ok(ToolChoiceKind::Auto),
            // "any" is the Gemini-native synonym for "required"; accepted in string and object form.
            "required" | "any" => Ok(ToolChoiceKind::Required),
            other => {
                warn!(
                    provider,
                    requested = other,
                    "unsupported tool_choice string"
                );
                Err(ProviderError::ToolChoiceUnsupported {
                    provider,
                    requested: truncate_for_error(other.to_string()),
                    supported_values: SUPPORTED_TOOL_CHOICE_VALUES,
                })
            }
        },
        Some(serde_json::Value::Object(obj)) => {
            let type_ = obj.get("type").and_then(|t| t.as_str());
            match type_ {
                Some("auto") => Ok(ToolChoiceKind::Auto),
                // "any" is the Gemini-native wire synonym for "required".
                Some("required") | Some("any") => Ok(ToolChoiceKind::Required),
                Some("function") => {
                    let name = obj
                        .get("function")
                        .and_then(|f| f.as_object())
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("");
                    if name.is_empty() {
                        let requested = truncate_for_error(
                            serde_json::to_string(&serde_json::Value::Object(obj.clone()))
                                .unwrap_or_else(|_| "<invalid>".to_string()),
                        );
                        warn!(
                            provider,
                            requested = %requested,
                            "tool_choice.function.name is missing or empty"
                        );
                        return Err(ProviderError::ToolChoiceUnsupported {
                            provider,
                            requested,
                            supported_values: SUPPORTED_TOOL_CHOICE_VALUES,
                        });
                    }
                    Ok(ToolChoiceKind::Function {
                        name: name.to_string(),
                    })
                }
                _ => {
                    let requested = truncate_for_error(
                        serde_json::to_string(&serde_json::Value::Object(obj.clone()))
                            .unwrap_or_else(|_| "<invalid>".to_string()),
                    );
                    warn!(provider, requested = %requested, "unsupported tool_choice object type");
                    Err(ProviderError::ToolChoiceUnsupported {
                        provider,
                        requested,
                        supported_values: SUPPORTED_TOOL_CHOICE_VALUES,
                    })
                }
            }
        }
        Some(other) => {
            let requested = truncate_for_error(
                serde_json::to_string(other).unwrap_or_else(|_| "<invalid>".to_string()),
            );
            warn!(provider, requested = %requested, "unsupported tool_choice value type");
            Err(ProviderError::ToolChoiceUnsupported {
                provider,
                requested,
                supported_values: SUPPORTED_TOOL_CHOICE_VALUES,
            })
        }
    }
}

/// Truncates a user-supplied string for inclusion in error messages and logs.
///
/// Caps at 256 bytes to prevent large payloads being echoed in 400 response bodies.
/// Respects UTF-8 char boundaries.
pub(crate) fn truncate_for_error(s: String) -> String {
    const MAX: usize = 256;
    if s.len() <= MAX {
        return s;
    }
    let mut end = MAX;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}<truncated>", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::{ChatRequest, Tool, ToolFunction};
    use serde_json::json;

    // ── validate_request_tools helpers ───────────────────────────────────────

    fn make_req(tools: Option<Vec<Tool>>, tool_choice: Option<serde_json::Value>) -> ChatRequest {
        let mut extra = serde_json::Map::new();
        if let Some(tc) = tool_choice {
            extra.insert("tool_choice".into(), tc);
        }
        ChatRequest {
            model: "test".into(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            tools,
            parallel_tool_calls: None,
            request_id: None,
            extra,
        }
    }

    fn func_tool(name: &str, params: Option<serde_json::Value>) -> Tool {
        Tool {
            type_: "function".into(),
            function: ToolFunction {
                name: name.into(),
                description: None,
                parameters: params,
            },
        }
    }

    // ── validate_request_tools tests ─────────────────────────────────────────

    #[test]
    fn no_tools_ok() {
        assert!(validate_request_tools(&make_req(None, None)).is_ok());
    }

    #[test]
    fn empty_tools_ok() {
        assert!(validate_request_tools(&make_req(Some(vec![]), None)).is_ok());
    }

    #[test]
    fn valid_function_tool_ok() {
        let req = make_req(
            Some(vec![func_tool("my_fn", Some(json!({"type": "object"})))]),
            None,
        );
        assert!(validate_request_tools(&req).is_ok());
    }

    #[test]
    fn tool_choice_none_skips_validation_even_with_bad_schema() {
        let bad_tool = func_tool("", None); // empty name — would normally be rejected
        let req = make_req(Some(vec![bad_tool]), Some(json!("none")));
        assert!(validate_request_tools(&req).is_ok());
    }

    #[test]
    fn tool_choice_none_object_skips_validation() {
        let bad_tool = func_tool("", None);
        let req = make_req(Some(vec![bad_tool]), Some(json!({"type": "none"})));
        assert!(validate_request_tools(&req).is_ok());
    }

    #[test]
    fn tool_choice_required_with_empty_tools_rejected() {
        let req = make_req(Some(vec![]), Some(json!("required")));
        assert_eq!(
            validate_request_tools(&req),
            Err(REASON_TOOL_CHOICE_REQUIRES_TOOLS)
        );
    }

    #[test]
    fn tool_choice_required_with_no_tools_rejected() {
        let req = make_req(None, Some(json!("required")));
        assert_eq!(
            validate_request_tools(&req),
            Err(REASON_TOOL_CHOICE_REQUIRES_TOOLS)
        );
    }

    #[test]
    fn tool_choice_function_object_with_empty_tools_rejected() {
        let req = make_req(
            None,
            Some(json!({"type": "function", "function": {"name": "fn"}})),
        );
        assert_eq!(
            validate_request_tools(&req),
            Err(REASON_TOOL_CHOICE_REQUIRES_TOOLS)
        );
    }

    #[test]
    fn tool_choice_auto_with_empty_tools_ok() {
        let req = make_req(Some(vec![]), Some(json!("auto")));
        assert!(validate_request_tools(&req).is_ok());
    }

    // "any" is a Gemini-native wire value, not OpenAI spec — gateway must not treat it as
    // demanding tools; provider adapters handle it themselves.
    #[test]
    fn tool_choice_any_object_with_no_tools_passes_through() {
        let req = make_req(None, Some(json!({"type": "any"})));
        assert!(validate_request_tools(&req).is_ok());
    }

    #[test]
    fn empty_function_name_rejected() {
        let req = make_req(Some(vec![func_tool("", None)]), None);
        assert_eq!(validate_request_tools(&req), Err(REASON_NAME_INVALID));
    }

    #[test]
    fn non_object_params_rejected() {
        let req = make_req(
            Some(vec![func_tool("fn", Some(json!("not-an-object")))]),
            None,
        );
        assert_eq!(
            validate_request_tools(&req),
            Err(REASON_PARAMETERS_NOT_OBJECT)
        );
    }

    #[test]
    fn non_function_type_passes_through() {
        let non_fn_tool = Tool {
            type_: "computer_use_preview".into(),
            function: ToolFunction {
                name: "".into(),
                description: None,
                parameters: None,
            },
        };
        let req = make_req(Some(vec![non_fn_tool]), None);
        assert!(validate_request_tools(&req).is_ok());
    }

    #[test]
    fn mixed_non_function_and_valid_function_ok() {
        let non_fn = Tool {
            type_: "computer_use_preview".into(),
            function: ToolFunction {
                name: "".into(),
                description: None,
                parameters: None,
            },
        };
        let req = make_req(Some(vec![non_fn, func_tool("valid_fn", None)]), None);
        assert!(validate_request_tools(&req).is_ok());
    }

    #[test]
    fn mixed_non_function_and_invalid_function_rejected() {
        let non_fn = Tool {
            type_: "computer_use_preview".into(),
            function: ToolFunction {
                name: "".into(),
                description: None,
                parameters: None,
            },
        };
        let req = make_req(Some(vec![non_fn, func_tool("", None)]), None);
        assert_eq!(validate_request_tools(&req), Err(REASON_NAME_INVALID));
    }

    // ──: size / depth / length limit tests ─────────────────────────────

    #[test]
    fn name_at_limit_ok() {
        let name = "a".repeat(TOOL_NAME_MAX_LEN);
        let req = make_req(Some(vec![func_tool(&name, None)]), None);
        assert!(validate_request_tools(&req).is_ok());
    }

    #[test]
    fn name_over_limit_rejected() {
        let name = "a".repeat(TOOL_NAME_MAX_LEN + 1);
        let req = make_req(Some(vec![func_tool(&name, None)]), None);
        assert_eq!(validate_request_tools(&req), Err(REASON_NAME_TOO_LONG));
    }

    #[test]
    fn description_at_limit_ok() {
        let tool = Tool {
            type_: "function".into(),
            function: ToolFunction {
                name: "fn".into(),
                description: Some("x".repeat(TOOL_DESCRIPTION_MAX_LEN)),
                parameters: None,
            },
        };
        let req = make_req(Some(vec![tool]), None);
        assert!(validate_request_tools(&req).is_ok());
    }

    #[test]
    fn description_over_limit_rejected() {
        let tool = Tool {
            type_: "function".into(),
            function: ToolFunction {
                name: "fn".into(),
                description: Some("x".repeat(TOOL_DESCRIPTION_MAX_LEN + 1)),
                parameters: None,
            },
        };
        let req = make_req(Some(vec![tool]), None);
        assert_eq!(
            validate_request_tools(&req),
            Err(REASON_DESCRIPTION_TOO_LONG)
        );
    }

    #[test]
    fn schema_over_size_limit_rejected() {
        // Build a params object large enough to exceed 128 KiB.
        let big_val: String = "x".repeat(TOOL_SCHEMA_MAX_BYTES + 1);
        let params = json!({"type": "object", "description": big_val});
        let req = make_req(Some(vec![func_tool("fn", Some(params))]), None);
        assert_eq!(validate_request_tools(&req), Err(REASON_SCHEMA_TOO_LARGE));
    }

    #[test]
    fn schema_at_max_depth_ok() {
        // Build a schema at exactly TOOL_SCHEMA_MAX_DEPTH.
        // Each wrap adds 1 level; starting scalar is depth 1.
        let mut schema = json!("leaf");
        for _ in 0..(TOOL_SCHEMA_MAX_DEPTH - 1) {
            schema = json!({"x": schema});
        }
        // depth = TOOL_SCHEMA_MAX_DEPTH — must pass.
        let req = make_req(Some(vec![func_tool("fn", Some(schema))]), None);
        assert!(validate_request_tools(&req).is_ok());
    }

    #[test]
    fn schema_over_depth_limit_rejected() {
        // One level beyond the limit.
        let mut schema = json!("leaf");
        for _ in 0..TOOL_SCHEMA_MAX_DEPTH {
            schema = json!({"x": schema});
        }
        // depth = TOOL_SCHEMA_MAX_DEPTH + 1 — must fail.
        let req = make_req(Some(vec![func_tool("fn", Some(schema))]), None);
        assert_eq!(validate_request_tools(&req), Err(REASON_SCHEMA_TOO_DEEP));
    }

    // ── parse_tool_choice_value tests ────────────────────────────────────────

    fn parse(val: Option<serde_json::Value>) -> Result<ToolChoiceKind, ProviderError> {
        parse_tool_choice_value(val.as_ref(), "test-provider")
    }

    fn assert_unsupported(result: Result<ToolChoiceKind, ProviderError>) {
        match result {
            Err(ProviderError::ToolChoiceUnsupported { .. }) => {}
            other => panic!("expected ToolChoiceUnsupported, got {other:?}"),
        }
    }

    #[test]
    fn absent_returns_auto() {
        assert!(matches!(parse(None).unwrap(), ToolChoiceKind::Auto));
    }

    #[test]
    fn string_auto_returns_auto() {
        assert!(matches!(
            parse(Some(json!("auto"))).unwrap(),
            ToolChoiceKind::Auto
        ));
    }

    #[test]
    fn string_required_returns_required() {
        assert!(matches!(
            parse(Some(json!("required"))).unwrap(),
            ToolChoiceKind::Required
        ));
    }

    #[test]
    fn string_any_returns_required() {
        assert!(matches!(
            parse(Some(json!("any"))).unwrap(),
            ToolChoiceKind::Required
        ));
    }

    #[test]
    fn object_type_auto_returns_auto() {
        assert!(matches!(
            parse(Some(json!({"type": "auto"}))).unwrap(),
            ToolChoiceKind::Auto
        ));
    }

    #[test]
    fn object_type_required_returns_required() {
        assert!(matches!(
            parse(Some(json!({"type": "required"}))).unwrap(),
            ToolChoiceKind::Required
        ));
    }

    #[test]
    fn object_type_any_returns_required() {
        assert!(matches!(
            parse(Some(json!({"type": "any"}))).unwrap(),
            ToolChoiceKind::Required
        ));
    }

    #[test]
    fn object_function_with_name_returns_function() {
        let v = json!({"type": "function", "function": {"name": "my_fn"}});
        match parse(Some(v)).unwrap() {
            ToolChoiceKind::Function { name } => assert_eq!(name, "my_fn"),
            other => panic!("expected Function, got {other:?}"),
        }
    }

    #[test]
    fn object_function_empty_name_returns_err() {
        let v = json!({"type": "function", "function": {"name": ""}});
        assert_unsupported(parse(Some(v)));
    }

    #[test]
    fn object_function_missing_function_obj_returns_err() {
        let v = json!({"type": "function"});
        assert_unsupported(parse(Some(v)));
    }

    #[test]
    fn object_unknown_type_returns_err() {
        let v = json!({"type": "unknown"});
        assert_unsupported(parse(Some(v)));
    }

    #[test]
    fn wrong_value_type_returns_err() {
        assert_unsupported(parse(Some(json!(42))));
    }

    #[test]
    fn large_input_is_truncated_in_error() {
        let big = "x".repeat(512);
        let err = parse(Some(serde_json::Value::String(big))).unwrap_err();
        if let ProviderError::ToolChoiceUnsupported { requested, .. } = err {
            assert!(
                requested.len() <= 256 + "<truncated>".len(),
                "requested field too long: {} bytes",
                requested.len()
            );
            assert!(
                requested.ends_with("<truncated>"),
                "expected truncation suffix"
            );
        } else {
            panic!("expected ToolChoiceUnsupported");
        }
    }

    // ── truncate_for_error ───────────────────────────────────────────────────

    #[test]
    fn truncate_for_error_short_string_unchanged() {
        let s = "call_abc".to_string();
        assert_eq!(truncate_for_error(s.clone()), s);
    }

    #[test]
    fn truncate_for_error_exactly_256_bytes_unchanged() {
        let s = "a".repeat(256);
        let result = truncate_for_error(s.clone());
        assert_eq!(result, s);
        assert!(!result.ends_with("<truncated>"));
    }

    #[test]
    fn truncate_for_error_over_256_gets_truncated() {
        let s = "x".repeat(300);
        let result = truncate_for_error(s);
        assert!(result.ends_with("<truncated>"), "must end with <truncated>");
        assert!(result.len() <= 256 + "<truncated>".len());
    }

    #[test]
    fn truncate_for_error_respects_utf8_boundary() {
        // Build a string where the 256-byte boundary falls inside a 3-byte char (€ = 0xE2 0x82 0xAC).
        // 85 * 3 = 255 bytes — last char starts at byte 252.
        let euros = "€".repeat(85); // 255 bytes
        let padding = "a".repeat(10);
        let s = euros + &padding; // 265 bytes total; cut point must not land mid-char
        let result = truncate_for_error(s);
        assert!(
            std::str::from_utf8(result.as_bytes()).is_ok(),
            "result must be valid UTF-8"
        );
        assert!(result.ends_with("<truncated>"));
    }
}
