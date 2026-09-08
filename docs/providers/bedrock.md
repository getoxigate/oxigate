# Bedrock Provider Implementation Notes

**Provider:** AWS Bedrock Converse API  
**Adapter:** `src/providers/bedrock/`  

---

## Token accounting

Bedrock's Converse contract reports its input buckets **disjointly**, which is the opposite of the
OpenAI family. AWS documents that with prompt caching enabled "the `inputTokens` field represents
only the non-cached input tokens", and gives
`total input tokens = inputTokens + cacheReadInputTokens + cacheWriteInputTokens`.

| Axis | Semantics |
|---|---|
| Cache | `inputTokens` **excludes** cached tokens; the cache buckets are reported beside it |
| Reasoning | no reasoning token count is reported on the Converse path |

The adapter declares this in `src/providers/bedrock/translate.rs` and applies it on both the
buffered and the streaming path.

### Cache tokens

`cacheReadInputTokens`, `cacheWriteInputTokens` and the TTL-split `cacheDetails` are parsed on both
the buffered and the streaming path. Two quantities are billed — the cache read, and the cache
write — and because the contract is additive, neither is subtracted from `inputTokens`; each is
charged beside it.

| Field | Effect |
|---|---|
| `cacheReadInputTokens` | published as `cache_read_input_tokens`; charged at the tier's `cache_read_multiplier` |
| `cacheWriteInputTokens` | the provider's aggregate of what was written |
| `cacheDetails` | the per-class breakdown of that **same** write, not a second one |

`cacheDetails` is an array of fixed-shape entries:

```json
"cacheDetails": [
  {"ttl": "5m", "inputTokens": 1000},
  {"ttl": "1h", "inputTokens": 500}
]
```

Each entry's `ttl` names the cache duration its tokens were written for, and each duration is priced
at that tier's `cache_write_multipliers` entry for the class — or, where the tier does not configure
that class, at the fallback rate below.

#### The aggregate and the breakdown are never summed

They are two views of one quantity, not two quantities. The billed quantity is the **larger** of the
two, so a response whose aggregate and breakdown agree is priced exactly once, at each class's own
rate, and reports a cost status of `exact` — **provided every reported class has a configured rate
in the selected tier.**

The two ways they can disagree are not symmetrical:

| Disagreement | Billed quantity | Rate | Status |
|---|---|---|---|
| Aggregate **exceeds** the detail total | the aggregate | the classed tokens at their own rates; the unattributed remainder at the fallback rate | `rate-fallback` |
| Details **exceed** the aggregate | the detail total | every class at its own rate — nothing is left over | `reconciled`, when every class has a configured rate |

An aggregate larger than its breakdown leaves a **residual**: tokens the provider says were written
but did not attribute to any class. Nothing identifies which duration those tokens belong to, so
they are priced at the tier's cache-write fallback rate — the highest cache-write multiplier that
tier configures, never below the full input rate.

Details larger than the aggregate leave no residual. Every token is already attributed to a class,
so the request is billed on the larger, fully classed total rather than on whichever view happened
to be smaller; `reconciled` records that the two views contradicted each other and the conservative
one was taken.

#### A fallback rate anywhere outranks the rest of the status

The status describes the whole request, and the worst component wins:
`exact` < `reconciled` < `rate-fallback` < `cost-unavailable`.

So the `exact` and `reconciled` outcomes above hold only while the selected tier prices every
reported quantity. A **positive** quantity that tier does not price — a class reserved elsewhere in
the pricing data that this tier's `cache_write_multipliers` omits, a duration no tier configures
anywhere, or an unrecognised `ttl` — is priced at the fallback rate and pulls the whole request down
to `rate-fallback`, whatever the two views did. A quantity of zero does not degrade the status.

#### An aggregate without a breakdown gets no default class

A Converse response that reports `cacheWriteInputTokens` but omits `cacheDetails` is treated as
entirely residual: it is priced at the fallback rate, and a **positive** residual reports
`rate-fallback`. A reported aggregate of zero leaves nothing to price and does not degrade the
status.

This is deliberate, and it differs from the Anthropic Messages path. AWS's default-TTL statement
describes what a *request* asks for; a response that omits `cacheDetails` says nothing about which
TTL was actually written, and AWS documents `cacheDetails` as empty only when no cache creation
occurred. Crediting those tokens to the cheaper class would undercharge while claiming `exact`.
Pricing them at the fallback rate overcharges visibly and says so.

A `ttl` value that does not name a duration the pricing data reserves a class for is likewise never
guessed at — but it keeps no class identity of its own either. Such details accumulate into a
single shared unknown bucket, priced at the same fallback rate: several unrecognised durations in
one response are one bucket, not several classes. The raw values are preserved in the persisted
evidence so an unrecognised duration stays legible.

---

## Tool Use

Non-streaming tool use is supported. Streaming tool use is not yet implemented — the adapter returns HTTP 400 `not_yet_supported`.

### Request translation

OpenAI `tools[]` → Bedrock `toolConfig.tools[].toolSpec` with `inputSchema.json`.

| OpenAI `tool_choice` | Bedrock `toolConfig.toolChoice` |
|----------------------|----------------------------------|
| absent | not sent |
| `"auto"` | `{"auto":{}}` |
| `"required"` | `{"any":{}}` |
| `"none"` | tools and toolConfig both omitted |
| `{"type":"function","function":{"name":"X"}}` | `{"tool":{"name":"X"}}` |

### Response translation

`toolUse` blocks in the Converse response are mapped to OpenAI `tool_calls[]` using the Bedrock `toolUseId` as the call ID.

### Tool count limit

Maximum 64 tools per request. Exceeding this returns HTTP 400 `tool_count_exceeded`.

### Streaming guard

When `req.tools` is non-empty, `chat_completion_stream` returns immediately with:
```json
{"error":{"code":"not_yet_supported","feature":"bedrock_streaming_tool_use"}}
```
HTTP 400. Use non-streaming when tools are required with Bedrock.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-09-03 | Cache read, cache write and the `cacheDetails` breakdown are parsed and billed on both Converse paths. Cached spend that was previously invisible now costs money and counts against budgets |
| 2026-08-23 | Cache accounting declared additive, matching the Converse contract. No billed amount moves — cache tokens are not parsed yet |
| 2026-05-05 | Non-streaming tool use translation; streaming guard |
