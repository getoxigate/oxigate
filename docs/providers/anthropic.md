# Anthropic Provider Implementation Notes

**Provider:** Anthropic (Claude)  
**Adapter:** `src/providers/anthropic/`  
**API Version:** 2023-06-01 (default)  

---

## Streaming API — Cache Token Semantics

### Key Finding: Cumulative vs Incremental

**Anthropic's streaming API sends CUMULATIVE (repeated) values, not incremental.**

This applies to:
- `cache_creation_input_tokens`
- `cache_read_input_tokens`
- `input_tokens` (in `message_delta`)

### Event Structure

#### `message_start` Event

```json
{
  "type": "message_start",
  "message": {
    "usage": {
      "input_tokens": 100,
      "output_tokens": 0,
      "cache_creation_input_tokens": 3500,
      "cache_read_input_tokens": 2000,
      "cache_creation": {
        "ephemeral_5m_input_tokens": 1000,
        "ephemeral_1h_input_tokens": 2500
      }
    }
  }
}
```

#### `message_delta` Event

**Note:** In practice, Anthropic typically sends only `output_tokens` in `message_delta`.
Cache token counts usually appear only in `message_start`. The gateway handles both cases:
- If `cache_creation` appears in `message_delta`, it overwrites (cumulative values)
- If absent, the breakdown from `message_start` is preserved

```json
{
  "type": "message_delta",
  "usage": {
    "output_tokens": 50
    // Typically no cache fields here — but gateway handles them if present
  }
}
```

### Implementation: Use Assignment, NOT Accumulation

**❌ WRONG (double-counts):**
```rust
self.cache_creation_5m_tokens = Some(
    self.cache_creation_5m_tokens.unwrap_or(0).saturating_add(cc.ephemeral_5m_input_tokens)
);
```

**✅ CORRECT (overwrite):**
```rust
self.cache_creation_5m_tokens = Some(cc.ephemeral_5m_input_tokens);
```

**Why:** Both events carry the same cumulative values. Using accumulation would result in double-counting (e.g., 1000 + 1000 = 2000 instead of 1000).

### Verification Sources

1. **LangChain Bug Report:** [cache tokens double-counted](https://github.com/langchain-ai/langchainjs/issues/10249)
2. **Anthropic Docs:** "The token counts shown in the usage field of the message_delta event are cumulative."

### Defensive Measures

The code includes `debug_assert!` + `warn!` logging to detect API drift:

```rust
if let Some(total_cache) = u.cache_creation_input_tokens {
    let sum = cache_creation_5m + cache_creation_1h;
    debug_assert!(
        sum == total_cache,
        "cache_creation breakdown sum ({sum}) != total ({total_cache}); Anthropic API may have changed"
    );
    if sum != total_cache {
        tracing::warn!(total, sum, "cache_creation breakdown sum != total");
    }
}
```

---

## Cache Token Pricing

Anthropic supports differential pricing for cache tokens:

| Token Type | Rate | Multiplier |
|------------|------|------------|
| Input (plain) | Base rate | 1.0× |
| Cache Read | Discounted | 0.1× (90% discount) |
| Cache Write (5m TTL) | Premium | 1.25× |
| Cache Write (1h TTL) | More Premium | 2.0× |

### Configuration

Pricing is configured in `assets/model_prices.json`. Example tier entry:

```json
{
  "model": "claude-sonnet-4-6",
  "tiers": [{
    "threshold": 0,
    "input_per_token": 0.000003,
    "output_per_token": 0.000015,
    "cache_read_multiplier": 0.1,
    "cache_write_5m_multiplier": 1.25,
    "cache_write_1h_multiplier": 2.0
  }]
}
```

### Billing Calculation

```
Total Cost = 
  (input_tokens × base_rate) +
  (output_tokens × base_rate) +
  (cache_read_tokens × base_rate × 0.1) +
  (cache_write_5m_tokens × base_rate × 1.25) +
  (cache_write_1h_tokens × base_rate × 2.0)
```

---

## Extended Thinking (Beta)

Anthropic's extended thinking feature produces `thinking_tokens` that are billed separately.

### Response Structure

```json
{
  "usage": {
    "input_tokens": 100,
    "output_tokens": 50,
    "output_tokens_details": {
      "thinking_tokens": 30
    }
  }
}
```

### Implementation

- `thinking_tokens` are extracted and billed at a configurable rate
- Default: Same as `output_per_token` (can be overridden via `thinking_per_token` in pricing tier)
- Surface in response: `completion_tokens_details.reasoning_tokens`

---

## Prompt Caching

Anthropic supports prompt caching with two TTL options:

| TTL | Use Case | Pricing |
|-----|----------|---------|
| 5-minute | Short-term session cache | 1.25× input rate |
| 1-hour | Long-term system prompt cache | 2.0× input rate |

### Client Request Format

To enable caching, clients must include `cache_control` hints in their request.
See [Anthropic's prompt caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) for the current API format.

Example (format may vary by API version):

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Long system context...",
          "cache_control": { "type": "ephemeral" }
        }
      ]
    }
  ]
}
```

### Gateway Behavior

The gateway does **not** automatically add cache hints. Clients must:
1. Include `cache_control` in their requests
2. Anthropic decides whether to cache (not guaranteed)
3. Gateway reads `cache_creation` and `cache_read` from response
4. Gateway bills at appropriate rates

---

## Error Handling

### Content Filtering

Anthropic may return errors for policy violations. These are mapped to `ProviderError::ContentFiltered` and can trigger fallbacks if configured.

### Rate Limits

Anthropic rate limits are mapped to `ProviderError::RateLimited` and trigger fallback per.

---

## Testing

### Unit Tests

```bash
cargo test --lib anthropic::translate::tests
```

Key tests:
- `test_cache_creation_1h_breakdown` — Verifies 5m/1h split parsing
- `test_cache_creation_fallback_to_5m_when_breakdown_absent` — Verifies fallback
- `test_stream_message_start_extracts_cache_creation_breakdown` — Streaming path
- `test_thinking_tokens_surfaced` — Extended thinking extraction

### Integration Tests

```bash
cargo test --test integration anthropic
```

Requires `OXIGATE__PROVIDERS__ANTHROPIC__API_KEY` set.

---

## Related Files

| File | Purpose |
|------|---------|
| `src/providers/anthropic/types.rs` | Request/response types |
| `src/providers/anthropic/translate.rs` | OpenAI ↔ Anthropic translation |
| `src/providers/anthropic/mod.rs` | Adapter implementation |
| `src/domain/pricing.rs` | Cost calculation logic |

---

## Tool Use

Anthropic's tool use is translated from OpenAI `tools[]` / `tool_choice` fields.

| OpenAI `tool_choice` | Anthropic `toolChoice` |
|----------------------|------------------------|
| `"auto"` / absent | `{"type":"auto"}` |
| `"required"` | `{"type":"any"}` |
| `"none"` | tools and tool_choice both omitted |
| `{"type":"function","function":{"name":"X"}}` | `{"type":"tool","name":"X"}` |

### Streaming buffer cap

Tool-argument JSON accumulates in memory during streaming. The cap is configurable:

```
OXIGATE__PROVIDERS__ANTHROPIC__TOOL_CALL_BUFFER_CAP_BYTES=1048576  # default 1 MiB
```

- **Pre-stream overflow (non-streaming):** HTTP 502 with `{"error":{"code":"tool_call_buffer_overflow",...}}`
- **Mid-stream overflow:** terminal SSE event `data: {"error":{...}}\n\n`, then graceful stream close. No `data: [DONE]` is sent — per OpenAI streaming contract, `[DONE]` signals success; emitting it after an error would be a false signal.

### Tool count limit

Maximum 64 tools per request. Exceeding this returns HTTP 400 `tool_count_exceeded`.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-05 | Tool use translation + streaming buffer cap |
| 2026-03-30 | Added cache token semantics (cumulative vs incremental) |
| 2026-03-30 | Added invariant guard for 5m+1h==total |
| 2026-03-18 | Extended thinking support |
| 2026-03-15 | Cache token pricing |
