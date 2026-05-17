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
        - deepseek-chat
        - deepseek-reasoner
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
     -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Hi"}]}'
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
        - deepseek-chat
        - deepseek-reasoner
      stream_options_support: false
      supports_tools: true
```

| Field | Value |
|---|---|
| `base_url` | `https://api.deepseek.com` |
| `stream_options_support` | `false` — probe before enabling; may vary by model |
| `supports_tools` | `true` |

**Notes:** `deepseek-reasoner` (DeepSeek-R1) carries a `cache_read_multiplier: 0.1` — cached
prompt tokens cost 10 % of the standard input rate. Usage fields follow the OpenAI schema; no
Anthropic-style `cache_creation_input_tokens` is available.

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

## Feature / behaviour table

| Feature | Behaviour |
|---|---|
| **Parsing** | Partial — only `model` and `max_tokens` are inspected for routing and budget pre-flight. The full request body is re-serialized from the deserialized `ChatRequest` and forwarded verbatim. |
| **Streaming** | Supported. Raw bytes forwarded; carry-buffer state machine reassembles SSE lines split across chunk boundaries. |
| **Cost signal timing** | End-of-stream — `usage` is scanned on every forwarded chunk; the last received value is authoritative. If absent, cost is zero for that request and a `WARN` is emitted. |
| **Cache token breakdown** | Not available — no provider in this category exposes Anthropic-style cache fields. `cache_read_input_tokens` and `cache_creation_input_tokens` will always be zero. |
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
