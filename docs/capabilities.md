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
| **Extended thinking / reasoning tokens** | 🔄 partial¹ | ❌ | ✅ | ❌ | N/A² |
| **Embeddings** | ✅ | N/A³ | ✅ | 📋 | 📋 |
| **Structured outputs / JSON mode** | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Production credential chain** | ✅⁴ | ✅⁴ | ✅ | ❌ | 📋 partial⁵ |
| **Streaming usage reporting** | ⚡ final chunk¹⁰ (injected `stream_options`) | ✅ real-time (`message_start` + `message_delta`) | ⚡ most chunks; Vertex AI may trail after `finish_reason`¹⁴ | ⚡ near-end (`metadata` event) | ⚡ final chunk¹⁰ (injected `stream_options`) |
| **Cache token cost breakdown** | ⚡ `prompt_tokens_details.cached_tokens` (final chunk) | ✅ `cache_creation_input_tokens` + `cache_read_input_tokens` (`message_start` — first event) | N/A¹¹ | N/A¹² | ⚡ same as OpenAI |
| **Mid-stream budget enforcement** | ❌¹³ | ✅ | ⚡¹⁴ | ❌¹³ | ❌¹³ |

**Notes:**

1. Maps `reasoning_effort` for o1/o3 but does not track thinking tokens in `completion_tokens_details` — partial gap.
2. Azure OpenAI does not yet expose thinking block tokens in its API surface.
3. Anthropic API has no embeddings endpoint.
4. Simple API key — no rotation or credential chain needed.
5. API key only; Azure Managed Identity (MSI) not in scope.
10. OpenAI and Azure emit NO usage data in any streaming chunk without `stream_options` injection; the adapters inject it unconditionally.
11. Gemini context caching is a separate API; token costs do not appear in `usageMetadata`.
12. Bedrock Converse has no cache pricing model.
13. Usage arrives at or after stream end — pre-flight check is the only enforcement gate for this provider. Mid-stream termination is not applicable.
14. Gemini API sends `usageMetadata` in most chunks; Vertex AI may trail it after `finish_reason`. Mid-stream enforcement applies when usage arrives before the terminal chunk; degrades to end-of-stream accounting on Vertex AI backends that trail usage.

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
