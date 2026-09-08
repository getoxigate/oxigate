# OpenAI-Compatible Provider Adapter

**Applies to:** `providers.openai_compat[]` entries  
**Adapter:** `OpenAICompatAdapter` (`src/providers/openai_compat/`)

Use this adapter for any provider that speaks the OpenAI chat completions wire format
but is not the real OpenAI API (e.g. DeepSeek, OpenRouter, Kimi, Qwen).

---

## Quick start

```yaml
providers:
  openai_compat:
    - name: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_KEY}
      supported_models:
        - deepseek-v4-flash
        - deepseek-v4-pro
      supports_tools: true           # DeepSeek supports OpenAI tool spec
    - name: openrouter
      base_url: https://openrouter.ai/api
      api_key: ${OPENROUTER_KEY}
      stream_options_support: true   # OpenRouter supports include_usage injection
      supports_tools: true           # OpenRouter forwards tool calls to upstream
    - name: local-llm
      base_url: http://localhost:11434   # keyless local endpoint
      # api_key omitted → no Authorization header sent
      supported_models:
        - llama3
```

Send requests via the standard endpoint — the router selects the adapter by model name:

```bash
curl -s -H "Authorization: Bearer $OXIGATE_KEY" \
     -H "Content-Type: application/json" \
     http://localhost:8080/v1/chat/completions \
     -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hi"}]}'
```

---

## Provider quick-reference

Each provider that ships as an `openai_compat[]` instance. Copy the YAML snippet, set your key,
and add the instance name to `routing.weights` (and `fallbacks` if desired).

> **Routing note:** `supported_models` is required for primary routing. If you omit it (or leave it
> `null`), the provider is assigned `ProviderKind::FallbackOnly` and will be skipped for all
> normal model-based routing — it only becomes reachable as an explicit fallback target. See
> [Routing: `FallbackOnly` vs `Primary`](#routing-fallbackonly-vs-primary) for the full decision
> table.

### Mistral

```yaml
providers:
  openai_compat:
    - name: mistral
      base_url: https://api.mistral.ai
      api_key: ${MISTRAL_API_KEY}
      supported_models:
        - mistral-large-latest
        - mistral-small-latest
        - codestral-latest
        - open-mistral-7b
        - open-mixtral-8x7b
      stream_options_support: false
      supports_tools: true
```

| Field | Value |
|---|---|
| `base_url` | `https://api.mistral.ai` |
| `stream_options_support` | `false` — injecting `stream_options` causes a 400 |
| `supports_tools` | `true` |

**Notes:** Codestral FIM endpoint (`/v1/fim/completions`) is a different wire format and is not
supported via `openai_compat` — tracked in. Chat window for `codestral-latest` is 32 k
tokens; the 262 k context is FIM-only.

---

### Groq

```yaml
providers:
  openai_compat:
    - name: groq
      base_url: https://api.groq.com/openai
      api_key: ${GROQ_API_KEY}
      supported_models:
        - llama-3.3-70b-versatile
        - llama-3.1-8b-instant
      stream_options_support: true
      supports_tools: true
```

| Field | Value |
|---|---|
| `base_url` | `https://api.groq.com/openai` — note the `/openai` path prefix |
| `stream_options_support` | `true` — Groq passes `include_usage` to the final streaming chunk |
| `supports_tools` | `true` |

**Notes:** Groq returns `x-ratelimit-*` headers on 429 responses; OxiGate captures the standard
`Retry-After` header and transitions the provider to cooldown. The `Retry-After` delay is not yet
honoured in the retry backoff.

---

### Together AI

```yaml
providers:
  openai_compat:
    - name: together-ai
      base_url: https://api.together.xyz
      api_key: ${TOGETHER_API_KEY}
      supported_models:
        - meta-llama/Llama-3.3-70B-Instruct-Turbo
        - meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
      stream_options_support: false
      supports_tools: true
```

| Field | Value |
|---|---|
| `base_url` | `https://api.together.xyz` |
| `stream_options_support` | `false` |
| `supports_tools` | `true` |

**Notes:** Together AI uses namespace-qualified model IDs (e.g. `meta-llama/Llama-3.3-70B-Instruct-Turbo`).
The gateway forwards the `model` field verbatim — no normalisation is applied. Ensure
`supported_models` entries match the exact IDs sent by clients.

---

### DeepSeek

```yaml
providers:
  openai_compat:
    - name: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      supported_models:
        - deepseek-v4-flash
        - deepseek-v4-pro
      stream_options_support: false
      supports_tools: true
```

| Field | Value |
|---|---|
| `base_url` | `https://api.deepseek.com` |
| `stream_options_support` | `false` — probe before enabling; may vary by model |
| `supports_tools` | `true` |

**Notes:** cached prompt tokens are discounted relative to the standard input rate:
`deepseek-v4-flash` carries `cache_read_multiplier: 0.02` (2 % of input rate) and
`deepseek-v4-pro` carries `cache_read_multiplier: ≈0.008333` (1/120th of input rate). Usage
fields follow the OpenAI schema; no Anthropic-style `cache_creation_input_tokens` is available.

---

### xAI (Grok)

```yaml
providers:
  openai_compat:
    - name: xai
      base_url: https://api.x.ai/v1
      api_key: ${XAI_API_KEY}
      supported_models:
        - grok-3-latest
        - grok-3-mini-latest
      stream_options_support: false
      supports_tools: true
```

| Field | Value |
|---|---|
| `base_url` | `https://api.x.ai/v1` |
| `stream_options_support` | `false` |
| `supports_tools` | `true` |

**Notes:** `grok-3-latest` and `grok-3-mini-latest` carry `cache_read_multiplier: 0.25` — cached
tokens cost 25 % of the standard input rate. xAI's first-class cache token breakdown, vision
inputs, and reasoning model parameters require a native adapter.

---

### Cerebras

```yaml
providers:
  openai_compat:
    - name: cerebras
      base_url: https://api.cerebras.ai/v1
      api_key: ${CEREBRAS_API_KEY}
      supported_models:
        - llama-3.3-70b
        - llama3.1-8b
      stream_options_support: false
      supports_tools: true
```

| Field | Value |
|---|---|
| `base_url` | `https://api.cerebras.ai/v1` |
| `stream_options_support` | `false` |
| `supports_tools` | `true` |

**Notes:** Cerebras throughput-tier pricing and embeddings endpoint require a native adapter
. The pricing entries cover the standard per-token rates for chat completions only.

---

## Token accounting — read this before trusting a cost figure

Every `openai_compat` instance is billed under one generic declaration, regardless of which backend
it points at:

| Axis | Assumed semantics |
|---|---|
| Cache | `prompt_tokens` **contains** `prompt_tokens_details.cached_tokens` |
| Reasoning | reasoning tokens are charged **beside** the reported completion total |

**The reasoning assumption is known to be wrong for some backends, and is deliberately left in
place.** Where a backend reports its reasoning count *inside* `completion_tokens` — as OpenAI and
Anthropic both do — that subset is charged twice: once inside the completion total at the output
rate, and once again at the reasoning rate. Cost is overstated on reasoning-heavy requests to those
backends.

This is not an oversight. Speaking the OpenAI wire format proves what fields a backend emits, not
how it counts them, and a third-party backend's counting is not covered by any first-party
reference. Adopting OpenAI's declaration for every compat instance would be an unverified billing
change applied to backends nobody has measured — trading a known overstatement for an unknown one.

The declaration will be made per backend as captured payloads establish what each actually reports.
Native OpenAI and Azure deployments already have theirs and are **not** affected by this — see
`providers/openai.md` and `providers/azure.md`. If you are routing OpenAI or Azure traffic through
`openai_compat` rather than through their own adapters, you are opting into this overstatement.

The cache assumption is the safer of the two: a backend that reports its cache buckets disjointly
would have those tokens subtracted from a prompt total that never contained them, which understates
cost. No configured backend below is known to do this.

### Cache writes are echoed, not priced

If a backend reports `prompt_tokens_details.cache_write_tokens`, that field is passed through to
your client verbatim — it is an OpenAI-standard field your backend chose to send. **Nothing is
billed from it.** No cache-write quantity is accounted, cost and cost status are unaffected, and
nothing is carved out of the prompt total for it.

The gateway does not derive `cache_creation_input_tokens` from `cache_write_tokens` on this adapter.
If your backend sends `cache_creation_input_tokens` of its own, that value passes through
untouched — but it is the backend's number, not a quantity OxiGate accounted or billed, and no
gateway cost figure is derived from it.

Same reason as above: the wire format proves the field exists, not what the backend charges for it
or which cache duration it belongs to. Native OpenAI and Azure deployments do price it — see
`providers/openai.md` and `providers/azure.md` — because their vendors document a single supported
duration and a rate for it. No such reference covers "any backend that accepts this schema".

The written tokens are not lost — they stay inside `prompt_tokens` and are charged at the plain
input rate like any other prompt token. What is missing is the cache-write premium: if your backend
bills writes above its input rate, gateway cost figures understate that difference, and so does the
budget counter.

### What is not affected

`X-Oxigate-Input-Tokens` and `X-Oxigate-Output-Tokens` report the backend's own figures verbatim on
every instance. Only what is charged at each rate is subject to the assumptions above.

---

## Feature / behaviour table

| Feature | Behaviour |
|---|---|
| **Parsing** | Partial — only `model` and `max_tokens` are inspected for routing and budget pre-flight. The full request body is re-serialized from the deserialized `ChatRequest` and forwarded verbatim. |
| **Streaming** | Supported. Raw bytes forwarded; carry-buffer state machine reassembles SSE lines split across chunk boundaries. |
| **Cost signal timing** | End-of-stream — `usage` is scanned on every forwarded chunk; the last received value is authoritative. If absent, the request is finalized as **cost-unavailable**: a terminal `oxigate.usage` event and a spend row are still written, both carrying zeros, and one `request accounting anomaly` `WARN` reports `provider-usage-missing`. Those zeros mean the cost is *unknown*, not that the request was free. `stream_options_support: false` (the default) disables **injection**, not parsing — the adapter still scans every chunk, so a backend that sends `usage` unprompted is costed normally. It is a backend that sends none, un-asked, that hits this on every streamed request. |
| **Cache token breakdown** | Read only. `prompt_tokens_details.cached_tokens` is normalized to `cache_read_input_tokens` and priced at the tier's `cache_read_multiplier`. `cache_write_tokens` is echoed but never accounted, and `cache_creation_input_tokens` is never derived from it — see [Cache writes are echoed, not priced](#cache-writes-are-echoed-not-priced). The adapter does not parse a per-duration cache-write breakdown. |
| **Budget enforcement posture** | Pre-flight enforcement only (spend-based `HardCapLayer`). `max_tokens`-based projection will be added later. Mid-stream termination is not possible because usage arrives at or after stream end. |
| **Tool use** | Opt-in per instance via `supports_tools: true` (default: `false`). Set this for providers that implement the OpenAI tools spec. Affects the `/v1/models` response and future capability-aware routing filters. See [Response parsing and error handling](#response-parsing-and-error-handling) for how choice parse failures are handled. |

---

## `stream_options_support` opt-in

By default, `stream_options` is **not** injected into forwarded requests. Injecting it on
providers that do not recognise the field causes a 400 error.

Set `stream_options_support: true` only for providers known to support
`stream_options.include_usage: true` in their streaming responses:

| Provider | `stream_options` supported | Notes |
|---|---|---|
| OpenRouter | Yes (`stream_options_support: true`) | Normalises upstream providers; final chunk carries `usage` |
| DeepSeek | Unknown — probe before enabling | May vary by model |
| Kimi (Moonshot) | Unknown | — |
| Qwen | Unknown | — |

When `stream_options_support: false` (the default), the adapter still scans every chunk
for a `usage` field — some providers emit it without being asked. End-of-stream accounting
works if the provider emits usage spontaneously.

---

## Response parsing and error handling

OxiGate deserializes each choice in the upstream response into its internal `Choice` type.
If any choice fails to parse (e.g. the provider sends an unrecognised field shape), the
**entire request fails** with a serialization error rather than returning a partial response
with fewer choices. This is intentional: silently delivering fewer choices than the upstream
produced is a FinOps audit hazard — a truncated response looks identical to a complete one.

If a compat provider consistently triggers parse errors, check that the provider's wire
format matches the OpenAI chat completions spec. Providers with non-standard choice shapes
require a dedicated adapter, not `openai_compat[]`.

---

## Routing: `FallbackOnly` vs `Primary`

| `supported_models` config | `ProviderKind` | Effect |
|---|---|---|
| Omitted (`null`) | `FallbackOnly` | Excluded from normal model-based routing; weight defaults to 0.0. Reachable only as an explicit fallback target. |
| `[model-a, model-b]` | `Primary` | Participates in routing for those models; competes with other providers. |
| `[]` (empty list) | **config-time error** | Rejected at startup — an empty list produces no selectable models. |

**Why `FallbackOnly` by default?** Adding an unknown compat instance beside `openai` must
not silently route `gpt-4o` traffic to the wrong provider. Explicit `supported_models` is
the opt-in for primary routing.

---

## Migration from `upstream_url`

`upstream_url` has been removed. Migrate any config that used it:

**Before (deprecated — no longer works):**

```yaml
upstream_url: https://api.deepseek.com
```

**After:**

```yaml
providers:
  openai_compat:
    - name: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_KEY}
```

**Routing config** (`weights`, `fallbacks`) that previously referenced `"passthrough"` must
be updated to use the new instance name:

```yaml
# Before
routing:
  weights:
    passthrough: 1.0
fallbacks:
  - provider: openai
    targets: [{provider: passthrough}]

# After
routing:
  weights:
    deepseek: 1.0
fallbacks:
  - provider: openai
    targets: [{provider: deepseek}]
```

**Keyless providers** (e.g. local inference servers) — omit `api_key` entirely:

```yaml
providers:
  openai_compat:
    - name: local-llm
      base_url: http://localhost:11434
      # No api_key — no Authorization header will be sent
```

**Ollama** — Ollama uses NDJSON streaming, not SSE. It is **not** an `openai_compat`
instance and will not work with this adapter. Ollama support is tracked as (separate
wire format adapter).

## Arbitrary field passthrough via `req.extra`

Any JSON fields in the incoming request that are not part of the standard `ChatRequest`
schema (model, messages, temperature, max\_tokens, stream, tools, etc.) are captured in
`req.extra` and serialized verbatim into the outbound request body. Provider-specific
extensions work automatically without gateway changes:

| Provider | Example field | Effect |
|---|---|---|
| OpenRouter | `transforms`, `route`, `provider.order` | Forwarded unchanged |
| DeepSeek | `frequency_penalty`, `top_p` | Forwarded unchanged |
| Kimi / Qwen | Any vendor-specific key | Forwarded unchanged |
| Any | Any unknown JSON key | Forwarded unchanged |

No config required. Fields flow through because the adapter re-serializes the full
`ChatRequest` (including `extra`) as the outbound body. The gateway never strips or
validates provider-specific fields — it is the operator's responsibility to send only
fields the target provider accepts.
