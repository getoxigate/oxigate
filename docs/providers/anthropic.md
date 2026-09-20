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
- `input_tokens` — restated on `message_delta`, where it sits in that event's **top-level**
  `usage` object, a sibling of `delta` rather than a member of it

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

**Anthropic's documented wire shape** sends `usage` as a sibling of `delta` at the event's top
level. The documented example carries output tokens alone; captured responses also restate the
`cache_creation_input_tokens` **aggregate** here. What this event never carries is the
`cache_creation` **breakdown** behind that aggregate — the per-class detail is reported once, in
`message_start`:

```json
{
  "type": "message_delta",
  "delta": { "stop_reason": "end_turn" },
  "usage": { "output_tokens": 50 }
}
```

The parser reads `usage` from this top-level position. Both the member and its `output_tokens`
are **required** on `message_delta`: an event that omits either fails to parse rather than
yielding a final chunk that reports zero output tokens, because a confident zero is
indistinguishable from a genuinely empty response.

Those requirements — and every other judgement about the object — are applied once the event's
`type` has resolved, not while the member is read. The top-level position belongs to every event,
so the parser buffers a root `usage` unexamined and only reads it if the tag turns out to be
`message_delta`. A non-`message_delta` frame that carries one — a shape the first-party API does
not send, but a proxied `api_base_url` may — is delivered on its own terms whatever that member
contains, including a repeat of it or a value that is not a usage object at all. A
`content_block_delta`'s generated text is never discarded over an accounting member that only
`message_delta` consumes.

`input_tokens` is the one member tolerated as absent or explicitly `null` here — Anthropic's
streaming documentation shows an output-only example, and the member is typed nullable on this
event. It resolves to `0`, and nothing reads it from this position: `message_start` is the
input-token source. The same member stays strictly required on `message_start` and on buffered
responses, where it *is* the request's only input-token source.

**`message_delta` restates the cache-write aggregate but never the breakdown.** Captured
responses carry `cache_creation_input_tokens` on this event with no `cache_creation` object
behind it; the per-class breakdown is reported once, in `message_start`. The "a present
`cache_creation` object replaces the prior snapshot wholesale" rule below is therefore never
triggered from `message_delta` in practice — `message_start`'s detail snapshot stands for the
request.

**A restated aggregate is the provider's final word on that count.** Anthropic's counts are
cumulative, so `message_start` and `message_delta` agree in the normal case and the restatement
changes nothing. Where they disagree, the gateway reports what the provider stated last, for both
`cache_creation_input_tokens` and `cache_read_input_tokens` — it does not pick the larger or the
smaller of the two. Choosing between contradictory statements would make the billed quantity one
the gateway decided rather than one it read, and this lane has no way to say so on the request's
cost status; a quantity chosen conservatively but reported as `exact` is worse than a quantity
reported as stated. An event that omits a member states nothing about it and leaves the standing
value untouched.

### Implementation: Latest Snapshot Replaces, Never Accumulates

Cache-write detail is not limited to `5m`/`1h` — each TTL Anthropic reports is canonicalized to
its own class and accounted independently, so a future third class needs no code change here.

`message_start` is Anthropic's one documented, reliable source of cache-write detail, and the
parser treats it as authoritative for the request. Anthropic's own usage counts are cumulative
restatements rather than incremental deltas, so wherever the parser does encounter a
`cache_creation` object, a **present** later object — including an empty `{}` — **replaces the
entire prior detail snapshot**, not just the classes it names: a class the earlier snapshot
recorded but this object omits disappears — it is not carried over. Only when a later event's
`cache_creation` member is **absent** — the key missing entirely, or present as an explicit JSON
`null` — does the standing snapshot from the previous event stand unchanged.

**Why replace, not add:** treating a repeated cumulative snapshot as an increment would
double-count (e.g. 1000 + 1000 = 2000 instead of 1000).

### Verification Sources

1. **LangChain Bug Report:** [cache tokens double-counted](https://github.com/langchain-ai/langchainjs/issues/10249)
2. **Anthropic Docs:** "The token counts shown in the usage field of the message_delta event are cumulative."

### Reconciling the aggregate against the detail

An aggregate (`cache_creation_input_tokens`) with no per-class breakdown behind it is accounted
to Anthropic's documented default class (`5m`) — not treated as unclassified fallback pricing. When
both a breakdown and an aggregate are present but disagree, the gateway accounts the higher of the
two (never adds them) and the request's cost-confidence status reflects the disagreement rather
than asserting or silently dropping the difference — see
[Cost confidence and evidence](#cost-confidence-and-evidence).

---

## Cache Token Pricing

Anthropic supports differential pricing for cache tokens:

| Token Type | Rate | Multiplier |
|------------|------|------------|
| Input (plain) | Base rate | 1.0× |
| Cache Read | Discounted | 0.1× (90% discount) |
| Cache Write (5m TTL) | Premium | 1.25× |
| Cache Write (1h TTL) | More Premium | 2.0× |

Cache-write pricing is class-keyed rather than fixed to these two TTLs: any canonical class a
model's tier configures gets its own multiplier, and any class Anthropic reports without a
configured rate falls back to a conservative rate rather than billing zero — see
[Cost confidence and evidence](#cost-confidence-and-evidence).

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
    "cache_write_multipliers": {
      "5m": 1.25,
      "1h": 2.0
    }
  }]
}
```

`cache_write_multipliers` is a complete replacement of the tier's cache-write rates, not a merge
against a default — every class the tier bills at a non-conservative rate must be listed.

### Billing Calculation

```
Total Cost =
  (input_tokens × input_rate) +
  ((output_tokens - thinking_tokens) × output_rate) +
  (thinking_tokens × thinking_rate) +
  (cache_read_tokens × input_rate × 0.1) +
  Σ over each accounted cache-write class c:
    (tokens[c] × input_rate × multiplier[c])
```

`thinking_tokens` are subtracted from `output_tokens` before the remainder is charged, because
Anthropic reports them as part of the output total rather than beside it — see
[Token accounting](#token-accounting). `thinking_rate` defaults to `output_rate` unless the pricing
tier sets `thinking_per_token`.

---

## Token accounting

Anthropic's two usage axes point in **opposite** directions, so neither can be assumed from the
other. The gateway declares both once, in `src/providers/anthropic/translate.rs`, and applies the
declaration on the streaming and non-streaming paths alike.

| Axis | Semantics | Source |
|---|---|---|
| Cache | `input_tokens` **excludes** cached tokens; `cache_read_input_tokens` and `cache_creation_input_tokens` are reported beside it | `platform.claude.com/docs/en/docs/build-with-claude/context-windows` — "the input count is split across `input_tokens`, `cache_read_input_tokens`, and `cache_creation_input_tokens`" |
| Reasoning | `output_tokens` **contains** the thinking tokens | `platform.claude.com/docs/en/docs/build-with-claude/extended-thinking` — `thinking_tokens` reports "how many of the billed output tokens were internal reasoning" |

So the cache buckets are added to the input side, while the thinking count is carved out of the
output side. Charging the full `output_tokens` *and* the thinking count beside it charges the
thinking subset twice.

`X-Oxigate-Output-Tokens` still reports Anthropic's `output_tokens` unchanged; only what is charged
at the standard output rate is affected.

### Usage-field audit

Every member of the usage objects Anthropic documents for the Messages API, and what the gateway
does with it.

- **Audited objects:** `Usage` — on the Messages response and on `message_start` — and
  `MessageDeltaUsage` on `message_delta`, with the nested `CacheCreation`, `OutputTokensDetails` and
  `ServerToolUsage`. The adapter sends `anthropic-version: 2023-06-01`.
- **Members:** Anthropic publishes no machine-readable spec, so the member list is read from its
  Python SDK, `anthropics/anthropic-sdk-python` @ `eb21a4352015686c30f5759e8c2f02d70f5371e2`,
  `src/anthropic/types/{usage,message_delta_usage,cache_creation,output_tokens_details,server_tool_usage}.py`.
- **Semantics:** `https://platform.claude.com/docs/en/api/messages` and
  `https://platform.claude.com/docs/en/build-with-claude/streaming`; the pricing statements below
  from `https://platform.claude.com/docs/en/about-claude/pricing` and
  `https://platform.claude.com/docs/en/manage-claude/data-residency`; the documented
  `inference_geo` value set from
  `https://platform.claude.com/docs/en/manage-claude/usage-cost-api` (§Data residency).
- Accessed 2026-09-11; the data-residency, pricing and usage-and-cost pages re-read 2026-09-17.

**Not read** means the member plays no part in cost, cost status, the spend row or the budget
counter.

| Member | `Usage` | `MessageDeltaUsage` | Gateway use |
|---|---|---|---|
| `input_tokens` | ✓ | ✓ (cumulative) | `prompt_tokens`. Charged whole at the input rate — the cache buckets sit beside it |
| `output_tokens` | ✓ | ✓ (cumulative) | `completion_tokens`. Charged at the output rate after the thinking subset is carved out |
| `cache_read_input_tokens` | ✓ | ✓ (cumulative) | `cache_read_input_tokens`. Charged at the tier's `cache_read_multiplier` |
| `cache_creation_input_tokens` | ✓ | ✓ (cumulative) | The cache-write aggregate. Reconciled against `cache_creation`, never summed with it (see [Reconciling the aggregate against the detail](#reconciling-the-aggregate-against-the-detail)) |
| `cache_creation.ephemeral_5m_input_tokens` | ✓ | — | A `5m` cache write at the tier's `5m` multiplier |
| `cache_creation.ephemeral_1h_input_tokens` | ✓ | — | A `1h` cache write at the tier's `1h` multiplier |
| `output_tokens_details.thinking_tokens` | ✓ | ✓ | `completion_tokens_details.reasoning_tokens`. Charged once at the thinking rate |
| `server_tool_use.web_search_requests` | ✓ | ✓ | Not read. Web search is charged per search, beside tokens. Not reachable today: the request translation forwards function tools only, so no server tool can be declared through the gateway |
| `server_tool_use.web_fetch_requests` | ✓ | ✓ | Not read. Web fetch carries no charge beyond the tokens it adds, which are already in `input_tokens`. Not reachable, for the same reason |
| `service_tier` | ✓ | — | Not read. A `priority` request draws down a Priority Tier capacity commitment instead of being billed at the standard per-token rate the gateway applies to every request. Anthropic no longer sells new commitments |
| `inference_geo` | ✓ | — | The geographic pricing multiplier. `us` scales input, output, cache-read, cache-write and thinking cost by 1.1×; `global` and `not_available` price at the standard rate; an absent or `null` member does the same. Any other string prices at the standard rate and reports `rate-fallback` — see [Inference geography](#inference-geography) |

A `cache_creation` member Anthropic adds later in the `ephemeral_<duration>_input_tokens` form is
accounted as its own class without a code change, as described under
[Prompt Caching](#prompt-caching).

---

## Inference geography

Anthropic prices US-only inference above global routing on its newer models, and reports where
inference actually ran on the response's `usage.inference_geo`. The gateway reads that member and
prices the request accordingly.

**Sources.** `https://platform.claude.com/docs/en/manage-claude/data-residency` (§Pricing) and
`https://platform.claude.com/docs/en/about-claude/pricing` (§Data residency pricing): "For Claude
4.6 and later models, specifying US-only inference through the `inference_geo` parameter incurs a
1.1x multiplier on all token pricing categories, including input tokens, output tokens, cache
writes, and cache reads", global routing uses standard pricing, and earlier models do not support
the parameter and always use standard pricing.
`https://platform.claude.com/docs/en/manage-claude/usage-cost-api` (§Data residency) gives the
value set — "Valid values are `global`, `us`, and `not_available`" — and states that models
without `inference_geo` support report `not_available`. All three accessed 2026-09-17.

| Reported value | Multiplier | Cost status | Structured-warning reason |
|---|---|---|---|
| `us` | 1.1× | unchanged | none |
| `global` | 1.0× | unchanged | none |
| `not_available` | 1.0× | unchanged | none |
| absent, or `null` | 1.0× | unchanged | none |
| any other string, including `""` | 1.0× | `rate-fallback` | `inference-geo-unrecognized` |
| `us`, on a model with no applicable rate | 1.0× | `rate-fallback` | none |

The multiplier reaches the five **token-category** components — input, standard output, cache
reads, cache writes and thinking. Image units and audio seconds are not token categories and are
not scaled.

It applies last, after the cache and batch multipliers, so it scales their results rather than the
underlying rates — which is what Anthropic means by the data-residency multiplier stacking with
prompt caching and the Batch API. Each multiplier pass truncates toward zero in nano-USD
independently, so a request carrying both a batch discount and the geographic surcharge is billed
the twice-truncated integer, not an exact 1.1 ratio of its unbatched cost.

**An unrecognised value is not treated as global.** A geography this build has no rate for is
priced at the standard rate, because there is no other defensible figure — but the request's cost
status drops to `rate-fallback` and the warning names the reason. Billing an unknown geography
silently at the standard rate is how a surcharge stays invisible until it shows up on an invoice.
The unrecognised value itself is logged once at DEBUG, length-capped, so an operator can see which
geography arrived; it is not carried on the response or persisted.

**The same rule covers a model priced by a `pricing.overrides` entry.** An override for a model
the bundled catalogue does not contain — a Claude release newer than the last asset refresh, say —
produces an entry the gateway cannot attribute to a provider, and so cannot apply a published
surcharge to. Such a request reporting `us` is priced at the standard rate and reported
`rate-fallback`, never `exact`. If you override a model that Anthropic prices US-only inference
for, set the override to the rate you actually expect to be billed.

**The gateway never sends `inference_geo`.** The parameter controls where customer data is
processed, which is the operator's decision to make in their Anthropic workspace
(`default_inference_geo` and `allowed_inference_geos`), not the gateway's to make on their behalf.
Sending it would also fail closed on every model that predates the parameter, which rejects it
with a 400. The response member is read, never the request parameter written.

**Only the first-party Claude API reports this member.** On Bedrock and Vertex the inference
region is selected by the endpoint or inference profile and carries its own regional pricing,
which no usage member reports; those lanes report no geography and price at the standard rate.

---

## Extended Thinking (Beta)

Anthropic's extended thinking feature produces `thinking_tokens`, which are part of the reported
output total and are charged at their own rate rather than in addition to it.

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
- **Carved out of the output total before it is charged** — in the example above, 50 output tokens
  of which 30 are thinking are billed as 20 at the output rate plus 30 at the thinking rate, not as
  50 plus 30

---

## Prompt Caching

Anthropic currently offers two TTL options, and the gateway does not assume the set stays at two:

| TTL | Use Case | Pricing |
|-----|----------|---------|
| 5-minute | Short-term session cache | 1.25× input rate |
| 1-hour | Long-term system prompt cache | 2.0× input rate |

Any TTL class Anthropic adds is accounted the same way as these two — canonicalized, priced from
its own configured rate if one exists, and conservatively rate-fallback priced if not.

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

## Cost confidence and evidence

Every request — not just one with cache-write usage — carries a single composite cost-confidence
status, computed worst-wins across *every* priced component of that request (input, output,
reasoning, cache read, cache write, batch, image, audio):

| Status | Meaning (request-wide) |
|---|---|
| `exact` | Every positive component used a configured or documented contractual rate, and the quantity evidence was self-consistent. |
| `reconciled` | Rates were exact, but contradictory or ambiguous quantity evidence required a conservative quantity policy. This is not one trigger: it covers cache-write aggregate-vs-detail contradiction, duplicate observation ambiguity (a *configured* class observed more than once is known exactly; more than one *unknown*-class observation instead leaves duplicate identity **indeterminate** — the gateway cannot establish whether they name the same class or different ones — and either condition is conservatively treated as ambiguity), and either of the two invariant clamps — cache reads exceeding reported prompt tokens, or reasoning tokens exceeding reported completion tokens — each independently forcing at least `reconciled` because an exact rate must not hide self-contradictory quantity evidence. |
| `rate-fallback` | At least one positive quantity was priced at a fallback rate — a dimension with a *defined, defensible* fallback (an unrecognized cache-write class, or a missing cache-read/batch multiplier) rather than going unbilled. |
| `cost-unavailable` | No defensible complete request cost could be produced — a dimension with **no** defined fallback had a positive quantity and no configured rate (missing required image or audio rates, which the gateway refuses to guess a price for), accounting failed outright (e.g. a token count overflowed), or the provider reported **no usage at all** on a cleanly-ended stream, leaving every quantity unknown rather than zero. The whole request is billed zero rather than an invented or partial number. |

The status reaches the client on the `X-Oxigate-Cost-Status` response header and on the terminal
`oxigate.usage` streaming event, and is persisted on every spend row as a required `cost_status`
column, independent of whether that row involved a cache write.

Separately, a **nullable** `usage_evidence` document is persisted only when the request had
positive cache-write usage; it is `NULL` otherwise. It records the raw cache-write keys observed,
their canonical class, whether the cache-write aggregate and detail reconciled, and whether more
than one unknown-class observation left duplicate identity indeterminate — it records that fact,
never an asserted unknown-class duplicate, because the gateway cannot establish whether two
unknown observations name the same class. A configured class's exact repeat count is not
persisted in this document at all. Bounded to 2 KiB and possibly marked `incomplete` if trimming
a key or dropping an entry was needed to stay under that bound. **It explains only the cache-write portion of the request's status** — it does not cover the
two invariant clamps (which are not cache-write-specific), and it is not a complete record of
every possible cause of `rate-fallback` or `cost-unavailable` (a missing image or audio rate, for
instance, leaves no trace in this document).

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
- `test_cache_creation_1h_breakdown` — a stated per-class breakdown is credited class by class
- `test_cache_creation_defaults_to_documented_class_when_breakdown_absent` — an aggregate with no
  breakdown behind it is accounted to the documented default class, not treated as a fallback
- `test_stream_message_start_extracts_cache_creation_breakdown` — streaming path
- `test_thinking_tokens_surfaced` — extended thinking extraction

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
| 2026-08-30 | Cache-write accounting generalized from fixed 5m/1h fields to arbitrary canonicalized TTL classes; composite cost-confidence status and bounded evidence added |
| 2026-08-23 | Thinking tokens no longer charged twice — declared as contained in `output_tokens`. Cost drops on thinking-heavy requests |
| 2026-05-05 | Tool use translation + streaming buffer cap |
| 2026-03-30 | Added cache token semantics (cumulative vs incremental) |
| 2026-03-30 | Added invariant guard for 5m+1h==total |
| 2026-03-18 | Extended thinking support |
| 2026-03-15 | Cache token pricing |
