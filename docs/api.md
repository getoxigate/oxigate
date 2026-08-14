# OxiGate API Reference

This document covers OxiGate-specific behaviour and extension fields. For the base
OpenAI-compatible contract, refer to the [OpenAI API reference](https://platform.openai.com/docs/api-reference).

---

## GET /v1/spend/daily

Returns daily aggregated spend for the authenticated org over a date range.

### Query parameters

| Parameter | Format | Default | Description |
|-----------|--------|---------|-------------|
| `from` | `YYYY-MM-DD` (UTC) | today − 30 days | Start date (inclusive). Defaults to 30 days before `to` when only `to` is provided. |
| `to` | `YYYY-MM-DD` (UTC) | today | End date (inclusive). Defaults to today when only `from` is provided. |

**Validation rules:**
- Both parameters are independently optional.
- `from` must not be after `to`.
- The range must not exceed **365 days**. Requests spanning more than 365 days are rejected
  with `400` to prevent unbounded aggregation scans on high-volume deployments.
- Invalid date formats return `400`.

### Response shape

```json
{
  "data": [
    { "date": "2025-01-15", "cost_nano_usd": 1234567890 }
  ]
}
```

Days with zero spend are omitted. Rows are ordered ascending by date.
`cost_nano_usd` is the sum of all spend records for that org on that UTC calendar day
(1 USD = 1 000 000 000 nano-USD).

---

## GET /v1/spend/providers

Returns spend grouped by provider for the authenticated org.

Query parameters and validation rules are identical to `GET /v1/spend/daily`.

### Response shape

```json
{
  "data": [
    { "dimension": "openai",    "cost_nano_usd": 5000000000 },
    { "dimension": "anthropic", "cost_nano_usd": 2000000000 }
  ]
}
```

Rows are ordered ascending by provider name.

---

## GET /v1/spend/models

Returns spend grouped by model for the authenticated org.

Query parameters and validation rules are identical to `GET /v1/spend/daily`.

### Response shape

```json
{
  "data": [
    { "dimension": "gpt-4.1",           "cost_nano_usd": 4000000000 },
    { "dimension": "claude-sonnet-4-6", "cost_nano_usd": 1500000000 }
  ]
}
```

Rows are ordered ascending by model name.

### Auth (all three spend endpoints)

Requires `Authorization: Bearer <token>` when `OXIGATE__AUTH__KEY` is configured.
Returns `401` otherwise.

Tenant isolation is enforced server-side: the `org_id` is read from the authenticated
`RequestIdentity` injected by the auth middleware. Callers cannot query another org's
spend by supplying query parameters — no `org_id` parameter is accepted.

---

## GET /v1/models

Returns all models routable through configured providers in OpenAI-compatible format, with
OxiGate extensions under the `oxigate` key on each entry.

### Response shape

```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4o",
      "object": "model",
      "created": 1741876800,
      "owned_by": "openai",
      "oxigate": {
        "provider": "openai",
        "context_window": 128000,
        "supports_streaming": true,
        "supports_tools": true,
        "supports_vision": true,
        "supports_embeddings": false,
        "supports_thinking": false,
        "cost_per_input_token_usd": 0.0000025,
        "cost_per_output_token_usd": 0.000010,
        "health_status": "available"
      }
    }
  ]
}
```

### `owned_by` semantics

`owned_by` contains the **OxiGate provider adapter name** (e.g. `"openai"`, `"anthropic"`,
`"gemini"`, `"passthrough"`), not an organization name as in the upstream OpenAI API where
it is always `"openai"` or `"system"`.

Clients that branch on `owned_by` expecting OpenAI's values will need to use `oxigate.provider`
instead, which carries the same value and is the canonical field for OxiGate-aware consumers.
The duplication exists to satisfy OpenAI SDK clients that require the `owned_by` field to be
present.

### `oxigate` extension fields

| Field | Type | Description |
|---|---|---|
| `provider` | string | Adapter name that owns this model (same as `owned_by`). |
| `context_window` | integer \| null | Maximum context in tokens from pricing DB; null if model is not in pricing DB. |
| `supports_streaming` | bool | Whether the adapter supports streaming completions. |
| `supports_tools` | bool | Whether the adapter supports tool/function calling. |
| `supports_vision` | bool | Whether the adapter supports vision (multimodal image input). |
| `supports_embeddings` | bool | Whether the adapter supports the embeddings endpoint. |
| `supports_thinking` | bool | Whether the adapter includes extended-thinking models. |
| `cost_per_input_token_usd` | float \| null | Input token cost in USD from pricing DB (base tier); null if not in pricing DB. |
| `cost_per_output_token_usd` | float \| null | Output token cost in USD from pricing DB (base tier); null if not in pricing DB. |
| `health_status` | string | `"available"` if startup health check passed; `"unknown"` otherwise. Dynamic tracking is. |

**Known limitation:** capability flags (`supports_streaming`, `supports_tools`, etc.) are
adapter-level, not per-model. All models from the same adapter share the adapter's flags.
Per-model capability granularity is deferred to.

### Auth

Requires `Authorization: Bearer <token>` when `OXIGATE__AUTH__KEY` is configured.
Returns 401 otherwise. See `docs/smoke-tests.md §9` for curl examples.

### Wildcard entries

The passthrough adapter's `"*"` wildcard model never appears in the response. Only
explicitly named model IDs are returned.

### Pricing fields

Pricing data comes from the bundled pricing DB (`BUNDLED_PRICING_JSON`). The base (first)
pricing tier is used. Models that are routable but absent from the pricing DB appear in the
response with `context_window: null` and both cost fields null.

---

## POST /v1/embeddings

Proxies embedding requests to the configured provider with cost tracking.

### Cost headers

Every successful response includes:

| Header | Example | Description |
|--------|---------|-------------|
| `X-Oxigate-Request-Cost` | `0.000020` | Cost in USD for this embedding request. |
| `X-Oxigate-Input-Tokens` | `500` | Number of input (prompt) tokens. |
| `X-Oxigate-Output-Tokens` | `0` | Always `0` — embeddings produce no output tokens. |
| `X-Oxigate-Model-Used` | `text-embedding-3-small` | Actual model resolved after provider dispatch. |

On error (4xx/5xx from provider), zero-cost headers are injected (`0.000000` / `0` / `0`).

### `spend_records` note

Embedding spend rows always have `completion_tokens = 0`. Downstream queries on `spend_records` should not assume `completion_tokens > 0` for any row that may come from an embedding request.

---

## Fallback + retry headers

When `security.expose_provider_names: true` is set in the gateway config, two optional
response headers are injected on every chat completion response:

| Header | Example | Description |
|---|---|---|
| `X-Oxigate-Attempted-Providers` | `anthropic,openai` | Comma-separated provider names in attempt order (primary first, then fallback targets). |
| `X-Oxigate-Attempted-Models` | `claude-sonnet-4-6,gpt-4o` | Comma-separated model names at the same index as `X-Oxigate-Attempted-Providers`. |
| `X-Fallback-Reason` | `rate_limit` | Snake_case trigger that caused the fallback cascade to fire. Only present when ≥1 non-primary fallback target was dispatched. Not injected when fallback was blocked by policy (`fallbacks[].on` filter) or when primary succeeded. |

**Default: headers are suppressed** (`expose_provider_names: false`) to avoid leaking internal
routing topology to external callers.

See `docs/guides/fallback-retry.md` for the full list of trigger values and configuration
reference.

### X-Oxigate-Budget-Remaining Header

When a budget hard cap is configured, the gateway injects `X-Oxigate-Budget-Remaining`
with the remaining budget in USD before the request would be rejected with 429.

#### Soft Cap vs. Hard Cap

OxiGate supports two budget thresholds:

| Cap Type | Behavior |
|----------|----------|
| **Soft Cap** | Threshold warnings only (80%/90%/100%). Requests proceed normally. Useful for alerts and observability. |
| **Hard Cap** | Hard enforcement. Requests are rejected with HTTP 429 when exceeded. |

Both can be configured simultaneously. When both are set:
- Warnings fire at soft cap thresholds
- Blocking occurs at hard cap
- `X-Oxigate-Budget-Remaining` shows remaining to **hard cap** (the enforcement boundary)

#### Multiple Budget Dimensions

When a request matches multiple budgets (e.g., team + tag + identity), the gateway:
1. Tracks spend independently for each dimension
2. Enforces **all** applicable budgets (parallel enforcement)
3. Sets `X-Oxigate-Budget-Remaining` to the **most restrictive** (minimum) remaining amount

**Example:**
| Budget | Cap | Spend | Remaining |
|--------|-----|-------|-----------|
| Team "engineering" | $100 | $80 | $20 |
| Tag "project:chat" | $50 | $45 | $5 |

Result: `X-Oxigate-Budget-Remaining: 5.000000`

The request is blocked when **any** budget reaches its hard cap.

### SSE streaming error events

When a streaming response is interrupted (inter-chunk timeout, upstream disconnect), the
gateway emits an `event: oxigate.error` SSE event before closing the stream:

```
event: oxigate.error
data: {"message":"inter-chunk timeout: no data from provider within deadline"}

```

The `data` field is a JSON object with a `message` key. Clients should treat this event as a
terminal error and not expect further chunks.

### Streaming fallback behaviour and limitations

**Pre-stream fallback (implemented):** If a provider fails _before_ the first chunk is
yielded — connection refused, immediate 429, TLS error — the fallback cascade fires exactly
as for non-streaming requests. The client receives a seamless response from the fallback
provider.

**Mid-stream fallback (not yet implemented):** If a provider fails _after_ chunks have
already been sent to the client, the stream terminates with an `oxigate.error` event. No
automatic fallback to a backup provider is attempted. Full mid-stream fallback requires
buffering the entire response before forwarding, which changes TTFB semantics and is
planned for a future release.

#### FinOps impact of mid-stream failures

| Concern | Impact | Why |
|---|---|---|
| Wasted spend | Tiny | Mid-stream failures are rare (<5 % of all failures). Most 429s arrive before the first chunk. |
| Budget tracking | None | Every token delivered before failure is still tracked and charged. No spend disappears. |
| Double-charging | None | A failed stream is not retried automatically — no double spend. A manual user retry is a new request. |
| Cost accuracy | None | Partial delivery = partial charge. The gateway charges for what was delivered, which is correct. |

#### User experience impact

When a mid-stream failure occurs:

- The client receives all chunks delivered before the failure.
- A terminal `event: oxigate.error` SSE frame is emitted.
- No automatic continuation from a fallback provider.
- Manual retry by the user starts a fresh request (full response from a healthy provider).

Frequency: roughly 5 % of streaming failures are mid-stream. The other 95 % are caught
pre-stream and handled transparently via fallback.

#### Observability

The metric `streaming_mid_stream_failures_total` (planned, not yet emitted) will count
mid-stream failures where no fallback was applied. Monitor this counter to assess whether
mid-stream fallback would meaningfully improve reliability for your workload.

```
**Streaming fallback:** Fallback triggers before first chunk.
Mid-stream failures return partial response + error.
Full mid-stream fallback planned for future release.
```

---

## Tool schema validation

OxiGate validates `tools[]` at the gateway layer before any provider dispatch. Requests with
invalid tool schemas are rejected with **HTTP 400** regardless of which provider is targeted.

### Rejection criteria

Only `tools[i]` entries with `"type": "function"` are validated; other types (e.g.
`"computer_use_preview"`) are forwarded to the upstream provider verbatim.

| Violation | `code` | `reason` |
|-----------|--------|----------|
| `tools[i].function.name` is empty | `malformed_tool_schema` | `name_invalid` |
| `tools[i].function.parameters` is not a JSON object | `malformed_tool_schema` | `parameters_not_object` |
| `tool_choice` demands a tool call (`"required"`, `{"type":"required"}`, or `{"type":"function",…}`) but `tools[]` is absent or empty | `malformed_tool_schema` | `tool_choice_requires_tools` |

### Error shape

```json
{
  "error": {
    "message": "malformed tool schema for gateway: name_invalid",
    "type": "malformed_tool_schema",
    "code": "malformed_tool_schema",
    "param": null,
    "provider": "gateway",
    "reason": "name_invalid"
  }
}
```

### Behavior note

When `tool_choice="none"` (string or `{"type":"none"}`), tool schema validation is **skipped
entirely** — tools provided for context do not need to be schema-valid when the model will not
call them. This matches OpenAI API behaviour.

---

## Tool call buffer overflow (Anthropic)

When OxiGate streams an Anthropic response that includes tool-use JSON, it accumulates the
`input_json_delta` fragments in memory. If the accumulated JSON for a single tool call exceeds
the configured cap, OxiGate terminates the stream with a structured error.

### Configuration

| YAML key | Env var | Default |
|----------|---------|---------|
| `providers.anthropic.tool_call_buffer_cap_bytes` | `TOOL_CALL_BUFFER_CAP_BYTES` | `1048576` (1 MiB) |

Very large values (above 64 MiB) are untested and increase gateway memory pressure under
concurrent streaming requests.

### Streaming overflow (mid-stream)

OxiGate emits one terminal SSE error frame and then closes the stream. No further chunks,
including `data: [DONE]`, are sent — per OpenAI streaming contract, `[DONE]` signals success;
emitting it after an error would be a false signal.

```
data: {"error":{"message":"tool call JSON exceeded the per-call buffer cap","type":"gateway_error","code":"tool_call_buffer_overflow","provider":"anthropic","tool_call_id":"<id>","cap_bytes":<N>}}

```

### Non-streaming overflow (pre-stream)

Non-streaming requests detect overflow before the HTTP response body is sent and return
**HTTP 502 Bad Gateway** with the standard OxiGate error envelope:

```json
{
  "error": {
    "message": "tool call buffer overflow",
    "type": "gateway_error",
    "code": "tool_call_buffer_overflow"
  }
}
```

---

## Error codes reference

All OxiGate error responses use the standard envelope:

```json
{
  "error": {
    "message": "<human-readable description>",
    "type": "<code>",
    "code": "<code>",
    "param": null
  }
}
```

`type` and `code` carry the same value. Common codes:

| HTTP status | `code` | Meaning |
|-------------|--------|---------|
| 400 | `invalid_request_error` | Malformed request: missing field, invalid value, or orphaned `tool_call_id` |
| 400 | `malformed_tool_schema` | Tool schema validation failed (see §Tool schema validation) |
| 400 | `translation_error` | The gateway successfully parsed the request but could not translate it to the target provider's wire format (e.g. a tool argument was not expressible in the provider's schema dialect) |
| 400 | `tool_count_exceeded` | `tools[]` contains more entries than the target provider's per-request limit; includes `provider`, `requested`, and `limit` fields in the error body |
| 400 | `tool_choice_unsupported` | The `tool_choice` value is not supported by the target provider; includes `provider`, `requested`, and `supported_values` fields in the error body |
| 400 | `not_yet_supported` | The feature is recognized by the gateway but not yet implemented for the target provider; includes a `feature` field naming the capability |
| 400 | `content_filtered` | Provider rejected content via moderation |
| 400 | `not_supported` | Feature not supported for the selected provider/model |
| 401 | `unauthorized` / `authentication_error` | Missing or invalid API key |
| 404 | `invalid_request_error` | Unknown model |
| 429 | `rate_limit_exceeded` | Rate limit or hard budget cap exceeded |
| 500 | `internal_error` | Unexpected internal gateway error |
| 501 | `not_implemented` | Feature acknowledged but not yet implemented |
| 502 | `provider_error` | Upstream provider returned an error |
| 502 | `internal_error` | Response serialization failure |
| 503 | `provider_unreachable` / `provider_unavailable` | Provider could not be reached |
| 504 | `provider_timeout` | Upstream provider timed out |
