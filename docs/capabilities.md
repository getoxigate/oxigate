# Provider Capability Status

Current implementation status for each provider adapter, by capability.

Keep this matrix in sync with implementation whenever a capability row changes.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Done — merged and validated |
| 🔄 | In progress |
| 📋 | Planned — not yet started |
| ❌ | Gap — not yet implemented |
| ⚡ | Supported but timing-limited or conditional — data may arrive late (end-of-stream) or only under certain backend conditions |
| N/A | Not applicable — provider does not support this capability |

---

## Matrix

| Capability | OpenAI | Anthropic | Gemini / Vertex | Bedrock (Converse) | Azure OpenAI |
|---|---|---|---|---|---|
| **Chat + streaming** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Tool use / function calling** | ✅ | ✅ | ✅ | ✅ non-streaming; ❌ streaming | ✅ validation; ❌ full support |
| **Vision / image inputs** | ❌ | ❌ | 🔄 | ❌ | ❌ |
| **Extended thinking / reasoning tokens** | ✅¹ | ✅¹ | ✅² | ❌ | ✅³ |
| **Embeddings** | ✅ | N/A⁴ | ✅ | 📋 | 📋 |
| **Structured outputs / JSON mode** | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Production credential chain** | ✅⁵ | ✅⁵ | ✅ | ❌ | 📋 partial⁶ |
| **Streaming usage reporting** | ⚡ final chunk⁷ (injected `stream_options`) | ✅ real-time (`message_start` + `message_delta`)⁸ | ⚡ most chunks; Vertex AI may trail after `finish_reason`⁹ | ⚡ near-end (`metadata` event) | ⚡ final chunk⁷ (injected `stream_options`) |
| **Cache token cost breakdown** | ✅ `cached_tokens` + `cache_write_tokens` (final chunk)¹⁰ | ✅ `cache_creation_input_tokens` + `cache_read_input_tokens` (`message_start` — first event) | ✅¹¹ | ✅ `cacheReadInputTokens` / `cacheWriteInputTokens` / `cacheDetails`, both Converse paths¹² | ✅ same as OpenAI¹⁰ |
| **Long-context tier selection**¹³ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Cost-confidence status**¹⁴ (`exact` / `reconciled` / `rate-fallback` / `cost-unavailable`) | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Mid-stream budget enforcement** | ❌¹⁵ | ✅ | ⚡⁹ | ❌¹⁵ | ❌¹⁵ |

**Notes:**

1. Reasoning tokens are read from `completion_tokens_details.reasoning_tokens` (Anthropic:
   `output_tokens_details.thinking_tokens`) and declared as **contained in** the reported
   completion total, which both vendors document. They are charged once, at the thinking rate,
   instead of being charged again at the output rate on top of the full completion total. See
   `providers/openai.md` and `providers/anthropic.md`.
2. Gemini's reasoning axis is the **opposite** of OpenAI's and Anthropic's and is declared
   separately for that reason: `totalTokenCount` is documented as "prompt + thoughts + response
   candidates", so `thoughtsTokenCount` sits *outside* `candidatesTokenCount` and is charged
   beside it rather than carved out of it. Applying the note-1 treatment here would subtract a
   quantity that was never included, producing an undercharge.
3. Azure OpenAI exposes `completion_tokens_details.reasoning_tokens` and documents those tokens as
   "billed as output tokens", so it carries the same declaration as OpenAI. See
   `providers/azure.md`.
4. Anthropic API has no embeddings endpoint.
5. Simple API key — no rotation or credential chain needed.
6. API key only; Azure Managed Identity (MSI) not in scope.
7. OpenAI and Azure emit NO usage data in any streaming chunk without `stream_options` injection; the adapters inject it unconditionally.
8. Anthropic sends `message_delta`'s `usage` object as a **sibling** of `delta`, at the event's
   top level, and that is where the parser reads it. This was previously read from inside
   `delta`, so the member was discarded and streamed Anthropic responses recorded
   `completion_tokens = 0`; `message_start` reporting — input tokens and the cache-write
   breakdown — was unaffected throughout. Cost recorded for streamed Anthropic requests **before
   this correction** understates the output portion, cannot be reconstructed, and hard-cap
   enforcement was correspondingly delayed for those requests.
9. Gemini API sends `usageMetadata` in most chunks; Vertex AI may trail it after `finish_reason`. Mid-stream enforcement applies when usage arrives before the terminal chunk; degrades to end-of-stream accounting on Vertex AI backends that trail usage.
10. `prompt_tokens_details.cached_tokens` is charged once at the tier's cache-read rate rather
    than also at the full input rate. GPT-5.6+ additionally reports
    `prompt_tokens_details.cache_write_tokens`, billed as a `30m` cache write — the only duration
    OpenAI documents — and likewise carved out of the reported prompt rather than added to it, so
    the documented 1.25× write is charged at 1.25× and not at 2.25×. The rate is per model: in the
    bundled pricing snapshot only the GPT-5.6 family carries a `30m` multiplier. A tier that
    configures none prices the write at that tier's cache-write fallback rate — the highest
    cache-write multiplier the tier itself configures, never below the full input rate — and a
    **positive** quantity priced that way reports `rate-fallback`. Azure inherits all of this on
    **Standard pay-as-you-go** deployments; provisioned (PTU-M) deployments do not expose
    `cache_write_tokens` at all and are unaffected. See `providers/openai.md` and
    `providers/azure.md`.
11. `usageMetadata.cachedContentTokenCount` is parsed, and `promptTokenCount` is documented as
    already containing it, so the cached portion is carved out of the reported prompt and charged
    once — at the cache rate rather than also at the full input rate. It is likewise counted once
    when selecting the long-context pricing tier. The bundled pricing snapshot carries
    model-specific `cache_read_multiplier` values for the applicable Gemini and Vertex entries, so
    cached tokens are charged at each model's own discounted rate. An entry configuring no
    multiplier charges them at 1.0× the selected tier's input rate, and a **positive** cached
    quantity priced that way reports `rate-fallback`; a reported zero does not degrade the status.
    See `providers/gemini.md`.
12. AWS defines `cacheReadInputTokens`, `cacheWriteInputTokens` and a TTL-split `cacheDetails` on
    the Converse `TokenUsage` shape, and documents `inputTokens` as the non-cached portion once
    caching is enabled — so the cache buckets are charged **beside** `inputTokens`, not carved out
    of it. All three fields are parsed on both the buffered and the streaming path. A
    `cacheDetails` entry **keeps its own class identity** when its `ttl` names a duration the
    pricing data reserves a slot for anywhere in the snapshot; reservation is global, so it decides
    identity, not price. **The rate is decided separately, by the selected tier**: a class that
    tier configures is charged at its multiplier, and everything else is fallback-priced at the
    tier's cache-write rate — the highest cache-write multiplier that tier configures, never below
    the full input rate — with a **positive** quantity priced that way pulling the request to
    `rate-fallback`. Three things reach that fallback: a class the pricing data reserves but the
    selected tier omits; a duration no tier configures anywhere, including an unrecognised `ttl`,
    which keeps no identity at all and accumulates into one shared unknown bucket; and the
    unmatched residual left when a reported aggregate exceeds the details that explain it,
    including an aggregate reported with no `cacheDetails` at all. See `providers/bedrock.md`.
13. Long-context pricing tiers reprice an entire request once the prompt crosses a threshold. The
    comparator is the **total prompt that occupied the context window** — plain input plus the
    cache-read and accounted cache-write buckets — applied uniformly to every provider, with no
    per-model override. Comparing only the uncached remainder would let a long cached prompt
    select the cheap tier and understate its cost. The buckets are disjoint by construction, so a
    cached token is counted once for tier selection, not twice. Pricing entries are validated at
    startup: a tier whose threshold exceeds its model's context window is rejected at load rather
    than shipped as a price no request can reach.
14. One composite status per request, worst-wins across every priced component of that request
    (input, output, reasoning, cache read, cache write, batch, image, audio), so a fallback
    anywhere cannot hide behind exact components elsewhere. `exact` — every positive component
    used a configured or documented contractual rate, and the quantity evidence was
    self-consistent. `reconciled` — the rates were exact, but contradictory or ambiguous quantity
    evidence required a conservative quantity policy. `rate-fallback` — at least one positive
    quantity was priced at a fallback rate. `cost-unavailable` — no defensible complete request
    cost could be produced, so the request-wide cost is zeroed rather than guessed. Reported usage
    quantities may still be present and non-zero: an unpriced model, or a positive image or audio
    quantity with no configured rate, leaves the token counts intact while the cost is
    unavailable. **A zero cost here does not mean the request was free.** Several conditions can
    produce each of the lower three values.
    **Where the status is delivered depends on how the request is made.** A buffered request
    carries it on the `X-Oxigate-Cost-Status` response header and on its persisted spend row;
    finalization runs before the response is sent, so neither depends on the client. A streamed
    request has no header twin — the status is not known when the headers are sent — so the
    terminal `oxigate.usage` SSE event is the only place a streaming client can read it. **The
    persisted row does not depend on the client.** When the provider reaches a clean, completed
    end of response, the gateway finalizes — cost, metric, log line and the scheduled spend row —
    *before* forwarding the terminating chunk, so a client that stops at the forwarded
    `data: [DONE]` still leaves its row. What such a client loses is only *observing* the status:
    the event is appended after the terminator it reports on, so reading `cost_status` requires
    reading past `[DONE]`. That guarantee covers the bundled adapters and any third-party adapter
    that marks its clean terminal chunk per the crate's adapter contract; an adapter that marks
    none is still accounted, but only once the consumer polls past its last chunk, so a client
    that stops early leaves no row for it. Two endings stay outside the guarantee, and they differ:
    an **error-interrupted** stream is charged nothing at all, and a **degraded non-error**
    termination is still accounted but only once the consumer polls past the last chunk, so a
    client that stops at one leaves no row.
    The status applies uniformly regardless of what a given adapter parses — an adapter with no
    cache-write parsing simply never produces a cache-write-caused outcome, while every other
    component can still drive it. See `api.md` for the wire format and the streaming limitations.
15. Usage arrives at or after stream end — pre-flight check is the only enforcement gate for this provider. Mid-stream termination is not applicable.

---

## OpenAI-compatible providers

Any OpenAI-compatible provider works through the shared `OpenAICompatAdapter` — **chat and
streaming are supported for all of them**. The table below tracks only *cost-tracking
fidelity*: whether usage data is reliably available during streaming. `stream_options` is
**not** injected by default — set `stream_options_support: true` per provider instance for
providers known to support it.

| Provider | `stream_options` supported | Usage in stream | Cache breakdown | Budget enforcement / accounting |
|---|---|---|---|---|
| DeepSeek | Not yet verified | Not yet verified | No | pre-flight |
| OpenRouter | Yes (normalises upstream) | Final chunk | No | pre-flight; final usage → end-of-stream accounting |
| Kimi (Moonshot) | Not yet verified | Not yet verified | No | pre-flight |
| Qwen | Not yet verified | Not yet verified | No | pre-flight |

*Not yet verified* means streaming cost-tracking fidelity has not been confirmed for that
provider — chat and streaming themselves work regardless.
