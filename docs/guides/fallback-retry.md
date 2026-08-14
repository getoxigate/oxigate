# Fallback & Retry Operator Guide

This guide covers the fallback cascade and retry engine: configuration, trigger semantics,
response header contracts, and log event fields.

---

## FallbackTrigger values

| Value | Meaning |
|---|---|
| `rate_limit` | Provider responded with HTTP 429 or `ProviderError::RateLimited` |
| `provider_unavailable` | Provider returned HTTP 5xx/502/503/504, OR a network connectivity failure (DNS resolution failure, connection refused, connection reset) via `ProviderError::Unreachable` — intentionally collapsed with upstream-5xx; introduce a `connectivity` trigger in a future revision if finer granularity is needed |
| `timeout` | Request timed out or inter-chunk streaming deadline exceeded (`ProviderError::Timeout`) |
| `content_filter` | Provider refused the request due to content policy (`ProviderError::ContentFiltered`) |
| `authentication` | API key or credentials were rejected (`ProviderError::Auth`) |
| `model_not_found` | Model is unknown to the provider (`ProviderError::UnknownModel`) |
| `context_window` | Prompt exceeded the provider's context window (not yet auto-detected; reserved) |
| `unknown` | Any other error that did not match the above categories |

---

## `fallbacks[].on` — trigger filter

Controls which error triggers cause the fallback cascade to fire.

| Config value | Behavior |
|---|---|
| field absent / `null` | Fallback fires for **any** error (default, backward compatible) |
| `on: []` | **Config error** — rejected at startup |
| `on: [rate_limit, timeout]` | Fallback fires **only** when the trigger is `rate_limit` or `timeout` |

### YAML example

```yaml
fallbacks:
  - provider: openai
    key: rate-limit-and-timeout-only
    on: [rate_limit, timeout]
    targets:
      - provider: anthropic
      - provider: gemini

  - provider: openai
    model: "gpt-4o*"
    # No `on` field — fires on any error for this model prefix
    targets:
      - provider: anthropic
        model: claude-sonnet-4-6
```

---

## `retry.on` — retry trigger filter

Controls which error triggers cause a retry attempt.

| Config value | Behavior |
|---|---|
| field absent / `null` | Retry fires for any retryable error (default) |
| `on: []` | **Config error** — rejected at startup |
| `on: [rate_limit]` | Retry fires only when the trigger is `rate_limit` |

Retryable errors (regardless of `on` filter): `rate_limit`, `provider_unavailable`, `timeout`,
`unknown` (when caused by a transient network error).

Non-retryable errors bypass the retry loop entirely: `authentication`, `content_filter`,
`model_not_found`.

### YAML example

```yaml
retry:
  max_retries: 3
  base_delay_ms: 200
  multiplier: 2.0
  max_delay_ms: 5000
  on: [rate_limit]   # only retry 429s; fail fast on 5xx / timeout
```

---

## `X-Fallback-Reason` response header

Injected when **all** of the following are true:
1. `security.expose_provider_names: true`
2. At least one non-primary fallback target was dispatched (and succeeded)

Contains the snake_case trigger string that caused the fallback (e.g. `rate_limit`, `timeout`).

**Not injected when:**
- The primary succeeded (no fallback triggered)
- The trigger was blocked by a `fallbacks[].on` filter (`AbortedByPolicy`)
- `security.expose_provider_names: false` (default)

### Example response headers

```
X-Oxigate-Attempted-Providers: openai,anthropic
X-Oxigate-Attempted-Models: gpt-4o,claude-sonnet-4-6
X-Fallback-Reason: rate_limit
```

---

## `X-Oxigate-Attempted-Providers` / `X-Oxigate-Attempted-Models`

Comma-separated list of providers/models that were **actually dispatched** (not skipped).
Primary attempt is always first; fallback targets follow in cascade order.

Gated by `security.expose_provider_names: true` (same as `X-Fallback-Reason`).

---

## OpenAI-compat adapter as a fallback target

Any `openai_compat[]` instance can act as a fallback target. Instances with no
`supported_models` list are registered as `FallbackOnly` (weight `0.0`) and are never
selected for primary routing — only as an explicit fallback. To configure one:

```yaml
providers:
  openai_compat:
    - name: openrouter
      base_url: https://openrouter.ai/api
      api_key: ${OPENROUTER_KEY}
      stream_options_support: true
      supports_tools: true

fallbacks:
  - provider: openai
    targets:
      - provider: openrouter
```

A `FallbackOnly` instance will not appear in primary model-based routing regardless of the
requested model. Set `routing.weights.openrouter: 1.0` only if you also want it eligible
for primary selection (requires `supported_models` to be set).

---

## Log event fields (`fallback dispatch terminal`)

Emitted at INFO level on failure/AbortedByPolicy, DEBUG level on success.

| Field | Type | Description |
|---|---|---|
| `source_provider` | string | Provider that was primary for this request |
| `source_model` | string | Model as specified in the request |
| `trigger` | string | snake_case trigger (e.g. `rate_limit`), or `none` if primary succeeded |
| `matched_rule_index` | int or null | 0-based index of the winning fallback rule |
| `matched_rule_key` | string | `key` field of the matched rule (empty if not set) |
| `attempt_count` | int | Number of actually-dispatched attempts (primary + fallbacks) |
| `skipped_count` | int | Number of targets skipped (trigger filter, cooldown, model unsupported, cycle) |
| `total_latency_ms` | float | Sum of per-attempt latencies (ms) |
| `outcome` | string | `success`, `exhausted`, or `aborted_by_policy` |
| `attempts` | string | Compact pipe-delimited list of all attempts |

### `attempts` compact format

```
openai:gpt-4o:retry=false:ok|anthropic:gpt-4o:retry=false:rate_limit|gemini:gpt-4o:skip=in_cooldown
```

Each segment is `provider:model:retry=<bool>:<error_class|ok>` for dispatched attempts, or
`provider:model:skip=<reason>` for skipped targets.

Skip reasons: `trigger_not_allowed`, `model_unsupported`, `duplicate_target`, `in_cooldown`.
