// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! FetchAttempt and FallbackDecisionTrace telemetry types for ProviderRouter .
//!
//! Records every dispatch attempt (retries + fallback cascade) in a structured trace
//! consumed by logging, metrics, and X-Fallback-Reason header injection.

use crate::config::FallbackTrigger;
use crate::domain::ports::ProviderError;

// ---------------------------------------------------------------------------
// Trace types
// ---------------------------------------------------------------------------

/// Record of a single dispatch attempt (retry or fallback target).
#[derive(Debug, Clone)]
pub struct FetchAttempt {
    /// Provider name.
    pub provider: String,
    /// Model name used for this attempt.
    pub model: String,
    /// `true` for attempt index ≥ 1 in the retry loop; `false` for the initial attempt.
    pub is_retry: bool,
    /// The trigger that caused the previous failure leading to this attempt.
    /// `None` for the initial primary attempt (no prior failure).
    pub trigger: Option<FallbackTrigger>,
    /// Whether the attempt was actually dispatched (`true`) or skipped pre-dispatch (`false`).
    pub attempted: bool,
    /// Reason the attempt was skipped (present when `attempted = false`).
    pub skip_reason: Option<SkipReason>,
    /// Normalised error class string (e.g. `"rate_limit"`, `"timeout"`). Set on dispatch failure.
    pub error_class: Option<String>,
    /// Dispatch latency in milliseconds. Set when `attempted = true`.
    pub latency_ms: Option<f64>,
}

/// Full telemetry trace for one routed request.
#[derive(Debug, Clone)]
pub struct FallbackDecisionTrace {
    /// Source provider selected by the routing strategy.
    pub source_provider: String,
    /// Request model.
    pub source_model: String,
    /// Trigger derived from the primary failure. `None` when primary succeeded.
    pub trigger: Option<FallbackTrigger>,
    /// 0-based index into `GatewayConfig::fallbacks[]` for the matched rule. `None` = no match.
    pub matched_rule_index: Option<usize>,
    /// `FallbackRule::key` of the matched rule, if set.
    pub matched_rule_key: Option<String>,
    /// All attempts in order (primary retries first, then fallback targets).
    pub attempts: Vec<FetchAttempt>,
    /// Terminal outcome derived from `attempts`.
    pub outcome: DecisionOutcome,
}

/// Terminal outcome of a dispatch pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionOutcome {
    /// At least one attempt succeeded.
    Success,
    /// All dispatched attempts failed; no attempts were skipped by policy.
    Exhausted,
    /// No fallback was dispatched because the trigger didn't match the rule's `on` filter.
    AbortedByPolicy,
}

/// Reason a fallback target was skipped without being dispatched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// Error trigger not in the rule's `on` list.
    TriggerNotAllowed,
    /// Provider does not support the resolved model.
    ModelUnsupported,
    /// Provider name referenced in the fallback rule is not registered in the router.
    ProviderNotFound,
    /// Provider+model pair already attempted (cycle guard).
    DuplicateTarget,
    /// Provider is in 429 cooldown (candidate list was empty).
    InCooldown,
}

impl std::fmt::Display for SkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkipReason::TriggerNotAllowed => f.write_str("trigger_not_allowed"),
            SkipReason::ModelUnsupported => f.write_str("model_unsupported"),
            SkipReason::ProviderNotFound => f.write_str("provider_not_found"),
            SkipReason::DuplicateTarget => f.write_str("duplicate_target"),
            SkipReason::InCooldown => f.write_str("in_cooldown"),
        }
    }
}

impl std::fmt::Display for DecisionOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecisionOutcome::Success => f.write_str("success"),
            DecisionOutcome::Exhausted => f.write_str("exhausted"),
            DecisionOutcome::AbortedByPolicy => f.write_str("aborted_by_policy"),
        }
    }
}

// ---------------------------------------------------------------------------
// ProviderError → FallbackTrigger + error_class
// ---------------------------------------------------------------------------

impl ProviderError {
    /// Maps this error to the `FallbackTrigger` used for `on`-filter matching.
    ///
    /// `Unreachable` → `ProviderUnavailable`: DNS/connect failures are intentionally collapsed
    /// with upstream-5xx. If finer granularity is needed later, introduce a `Connectivity`
    /// trigger in a future revision.
    #[must_use]
    pub fn to_trigger(&self) -> FallbackTrigger {
        match self {
            ProviderError::RateLimited { .. } => FallbackTrigger::RateLimit,
            ProviderError::ProviderUnavailable(_) => FallbackTrigger::ProviderUnavailable,
            ProviderError::Timeout { .. } => FallbackTrigger::Timeout,
            ProviderError::ContentFiltered(_) => FallbackTrigger::ContentFilter,
            ProviderError::Auth(_) => FallbackTrigger::Authentication,
            ProviderError::UnknownModel(_) => FallbackTrigger::ModelNotFound,
            ProviderError::Unreachable(_) => FallbackTrigger::ProviderUnavailable,
            ProviderError::ProviderHttpError { status: 429, .. } => FallbackTrigger::RateLimit,
            ProviderError::ProviderHttpError { status, .. } if *status >= 500 => {
                FallbackTrigger::ProviderUnavailable
            }
            _ => FallbackTrigger::Unknown,
        }
    }

    /// Returns a short normalised string tag for this error (used in metric labels and
    /// `FetchAttempt::error_class`). Never contains secrets, URLs, or request bodies.
    #[must_use]
    pub fn error_class(&self) -> &'static str {
        match self {
            ProviderError::RateLimited { .. } => "rate_limit",
            ProviderError::ProviderUnavailable(_) => "provider_unavailable",
            ProviderError::Timeout { .. } => "timeout",
            ProviderError::ContentFiltered(_) => "content_filter",
            ProviderError::Auth(_) => "auth",
            ProviderError::UnknownModel(_) => "model_not_found",
            ProviderError::Unreachable(_) => "unreachable",
            ProviderError::ProviderHttpError { status: 429, .. } => "rate_limit",
            ProviderError::ProviderHttpError { status, .. } if *status >= 500 => {
                "provider_unavailable"
            }
            ProviderError::ProviderHttpError { .. } => "provider_http_error",
            ProviderError::Serialization(_) => "serialization",
            ProviderError::NotImplemented => "not_implemented",
            ProviderError::InvalidRequest(_) => "invalid_request",
            ProviderError::NotSupported(_) => "not_supported",
            ProviderError::Translate(_) => "translate",
            ProviderError::AllProvidersRateLimited { .. } => "all_rate_limited",
            ProviderError::Internal(_) => "internal",
            ProviderError::ToolChoiceUnsupported { .. } => "invalid_request",
            ProviderError::ToolCountExceeded { .. } => "invalid_request",
            ProviderError::MalformedToolSchema { .. } => "invalid_request",
            ProviderError::ToolCallBufferOverflow { .. } => "tool_buffer_overflow",
            ProviderError::NotYetSupported { .. } => "not_yet_supported",
        }
    }
}

// ---------------------------------------------------------------------------
// FallbackTrigger display helper (for header values)
// ---------------------------------------------------------------------------

/// Returns the snake_case header value for a trigger (e.g. for `X-Fallback-Reason`).
#[must_use]
pub fn trigger_header_value(trigger: &FallbackTrigger) -> &'static str {
    match trigger {
        FallbackTrigger::RateLimit => "rate_limit",
        FallbackTrigger::ProviderUnavailable => "provider_unavailable",
        FallbackTrigger::Timeout => "timeout",
        FallbackTrigger::ContentFilter => "content_filter",
        FallbackTrigger::Authentication => "authentication",
        FallbackTrigger::ModelNotFound => "model_not_found",
        FallbackTrigger::ContextWindow => "context_window",
        FallbackTrigger::Unknown => "unknown",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_trigger_rate_limited() {
        assert_eq!(
            ProviderError::RateLimited { retry_after: None }.to_trigger(),
            FallbackTrigger::RateLimit
        );
    }

    #[test]
    fn to_trigger_timeout() {
        assert_eq!(
            ProviderError::Timeout { elapsed_ms: 5000 }.to_trigger(),
            FallbackTrigger::Timeout
        );
    }

    #[test]
    fn to_trigger_unreachable_collapses_to_unavailable() {
        assert_eq!(
            ProviderError::Unreachable("dns failed".into()).to_trigger(),
            FallbackTrigger::ProviderUnavailable
        );
    }

    #[test]
    fn to_trigger_http_429() {
        assert_eq!(
            ProviderError::ProviderHttpError {
                status: 429,
                body: String::new()
            }
            .to_trigger(),
            FallbackTrigger::RateLimit
        );
    }

    #[test]
    fn to_trigger_http_503() {
        assert_eq!(
            ProviderError::ProviderHttpError {
                status: 503,
                body: String::new()
            }
            .to_trigger(),
            FallbackTrigger::ProviderUnavailable
        );
    }

    #[test]
    fn to_trigger_content_filtered() {
        assert_eq!(
            ProviderError::ContentFiltered("blocked".into()).to_trigger(),
            FallbackTrigger::ContentFilter
        );
    }

    #[test]
    fn to_trigger_auth() {
        assert_eq!(
            ProviderError::Auth("401".into()).to_trigger(),
            FallbackTrigger::Authentication
        );
    }

    #[test]
    fn to_trigger_unknown_model() {
        assert_eq!(
            ProviderError::UnknownModel("gpt-99".into()).to_trigger(),
            FallbackTrigger::ModelNotFound
        );
    }

    #[test]
    fn to_trigger_internal_maps_to_unknown() {
        assert_eq!(
            ProviderError::Internal("bug".into()).to_trigger(),
            FallbackTrigger::Unknown
        );
    }

    #[test]
    fn to_trigger_unknown_catch_all_variants() {
        // Several error variants that are not specifically mapped must all resolve to Unknown.
        assert_eq!(
            ProviderError::Serialization("bad json".into()).to_trigger(),
            FallbackTrigger::Unknown
        );
        assert_eq!(
            ProviderError::NotImplemented.to_trigger(),
            FallbackTrigger::Unknown
        );
        assert_eq!(
            ProviderError::InvalidRequest("bad request".into()).to_trigger(),
            FallbackTrigger::Unknown
        );
        assert_eq!(
            ProviderError::NotSupported("vision".into()).to_trigger(),
            FallbackTrigger::Unknown
        );
    }

    /// DecisionOutcome derivation: verify the three canonical outcomes are represented
    /// correctly in the Display impl (used in structured log events and headers).
    #[test]
    fn decision_outcome_display() {
        assert_eq!(DecisionOutcome::Success.to_string(), "success");
        assert_eq!(DecisionOutcome::Exhausted.to_string(), "exhausted");
        assert_eq!(
            DecisionOutcome::AbortedByPolicy.to_string(),
            "aborted_by_policy"
        );
    }

    /// A trace built from a successful primary has outcome = Success and no trigger.
    #[test]
    fn trace_success_has_no_trigger() {
        let trace = FallbackDecisionTrace {
            source_provider: "openai".into(),
            source_model: "gpt-4o".into(),
            trigger: None,
            matched_rule_index: None,
            matched_rule_key: None,
            attempts: vec![FetchAttempt {
                provider: "openai".into(),
                model: "gpt-4o".into(),
                is_retry: false,
                trigger: None,
                attempted: true,
                skip_reason: None,
                error_class: None,
                latency_ms: Some(42.0),
            }],
            outcome: DecisionOutcome::Success,
        };
        assert!(trace.trigger.is_none());
        assert_eq!(trace.outcome, DecisionOutcome::Success);
        assert_eq!(trace.attempts.iter().filter(|a| a.attempted).count(), 1);
    }

    /// A trace built from AbortedByPolicy has all targets skipped with TriggerNotAllowed.
    #[test]
    fn trace_aborted_by_policy_all_skipped() {
        let trace = FallbackDecisionTrace {
            source_provider: "openai".into(),
            source_model: "gpt-4o".into(),
            trigger: Some(FallbackTrigger::ProviderUnavailable),
            matched_rule_index: Some(0),
            matched_rule_key: Some("rate-limit-only".into()),
            attempts: vec![
                FetchAttempt {
                    provider: "openai".into(),
                    model: "gpt-4o".into(),
                    is_retry: false,
                    trigger: None,
                    attempted: true,
                    skip_reason: None,
                    error_class: Some("provider_unavailable".into()),
                    latency_ms: Some(100.0),
                },
                FetchAttempt {
                    provider: "anthropic".into(),
                    model: "gpt-4o".into(),
                    is_retry: false,
                    trigger: Some(FallbackTrigger::ProviderUnavailable),
                    attempted: false,
                    skip_reason: Some(SkipReason::TriggerNotAllowed),
                    error_class: None,
                    latency_ms: None,
                },
            ],
            outcome: DecisionOutcome::AbortedByPolicy,
        };
        assert_eq!(trace.outcome, DecisionOutcome::AbortedByPolicy);
        assert_eq!(
            trace.attempts.iter().filter(|a| !a.attempted).count(),
            1,
            "one target should be skipped"
        );
        assert_eq!(
            trace.attempts[1].skip_reason,
            Some(SkipReason::TriggerNotAllowed)
        );
    }

    #[test]
    fn error_class_round_trip() {
        assert_eq!(
            ProviderError::RateLimited {
                retry_after: Some(5)
            }
            .error_class(),
            "rate_limit"
        );
        assert_eq!(
            ProviderError::Timeout { elapsed_ms: 100 }.error_class(),
            "timeout"
        );
        assert_eq!(ProviderError::Auth("x".into()).error_class(), "auth");
        assert_eq!(
            ProviderError::UnknownModel("x".into()).error_class(),
            "model_not_found"
        );
        assert_eq!(
            ProviderError::Unreachable("x".into()).error_class(),
            "unreachable"
        );
    }

    #[test]
    fn trigger_header_value_snake_case() {
        assert_eq!(
            trigger_header_value(&FallbackTrigger::RateLimit),
            "rate_limit"
        );
        assert_eq!(trigger_header_value(&FallbackTrigger::Timeout), "timeout");
        assert_eq!(
            trigger_header_value(&FallbackTrigger::ProviderUnavailable),
            "provider_unavailable"
        );
    }
}
