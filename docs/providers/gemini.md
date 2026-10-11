# Gemini Provider Implementation Notes

**Provider:** Google Gemini / Vertex AI  
**Adapter:** `src/providers/gemini/`  

---

## Tool Use

OpenAI `tools[]` / `tool_choice` are translated to Gemini `tools[].function_declarations` and `tool_config`.

| OpenAI `tool_choice` | Gemini `tool_config.function_calling_config` |
|----------------------|----------------------------------------------|
| absent | not sent |
| `"auto"` | `{"mode":"AUTO"}` |
| `"required"` or `"any"` | `{"mode":"ANY"}` |
| `"none"` | tools and tool_config both omitted |
| `{"type":"function","function":{"name":"X"}}` | `{"mode":"ANY","allowed_function_names":["X"]}` |

**G4 rule:** named-function `tool_choice` keeps all `function_declarations[]` and adds `allowed_function_names` as a filter — Gemini does not accept a reduced tool list with mode=ANY.

### Tool count limit

Maximum 128 tools per request. Exceeding this returns HTTP 400 `tool_count_exceeded`.

### Streaming tool calls

`Part::FunctionCall` parts in Gemini stream chunks are emitted as complete tool call deltas (OpenAI streaming format) with a stable `call_{request_id}_{idx}` ID.

---

## Tool message validation (F4)

When a `Role::Tool` message is present in the request, OxiGate validates that:

1. `tool_call_id` is non-null and non-empty — a missing or empty field is rejected with HTTP 400
   before the request reaches Gemini.
2. The `tool_call_id` matches a `tool_calls[].id` declared in an earlier assistant message
   **within the same request**. An orphaned ID is also rejected with HTTP 400.

**Before this change** (prior to): the gateway forwarded the malformed request to Gemini,
which returned a cryptic 4xx. **After this change**: the gateway rejects immediately.

**Error body shape** (HTTP 400):

```json
{
  "error": {
    "message": "tool_call_id 'X' has no matching prior assistant tool_call in this request; include the full conversation history (assistant message with tool_calls[])",
    "type": "invalid_request_error",
    "code": "invalid_request_error"
  }
}
```

Operators relying on Gemini's own error response for orphaned tool IDs must update client code to
include the full conversation history per the
[OpenAI multi-turn tool use spec](https://platform.openai.com/docs/guides/function-calling).

---

---

## Token accounting

Gemini's two usage axes point in **opposite** directions, and this is the one provider where the
reasoning axis differs from OpenAI and Anthropic. The gateway declares both once, in
`src/providers/gemini/translate.rs`.

| Axis | Semantics | Source |
|---|---|---|
| Cache | `promptTokenCount` **contains** `cachedContentTokenCount` | `ai.google.dev/api/generate-content` — `promptTokenCount` is "the total effective prompt size meaning this includes the number of tokens in the cached content" |
| Reasoning | `candidatesTokenCount` **excludes** `thoughtsTokenCount` | same page — `totalTokenCount` is "prompt + thoughts + response candidates", so thoughts sit outside the candidates count |

Two consequences follow, and they pull in different directions:

- **Cached tokens are carved out of the reported prompt** before the remainder is charged at the
  full input rate, then charged once at the cache rate. They are also counted **once**, not twice,
  when selecting the long-context pricing tier — double-counting them could push a request into a
  higher-priced tier it does not belong in.
- **Thought tokens are charged beside the candidates total, not carved out of it.** Applying the
  OpenAI or Anthropic treatment here would subtract a quantity that was never included, producing
  an *undercharge*. This asymmetry is why the accounting is declared per provider rather than as
  one global rule.

### Cache-read rate

The applicable Gemini and Vertex entries in the bundled pricing snapshot carry model-specific
`cache_read_multiplier` values, so cached tokens are charged at each model's own discounted rate
rather than at the full input rate. Operators overriding pricing can set `cache_read_multiplier`
per tier.

An entry that carries no multiplier — an operator override that omits it, or a model the snapshot
does not price for cache reads — charges cached tokens at 1.0× the tier's input rate. A **positive**
cached quantity priced that way reports a cost status of `rate-fallback`, so a missing discount is
visible rather than silent; a reported zero has nothing to misprice and does not degrade the status.

### Usage-field audit

Every member of the usage object Google documents for `generateContent` and
`streamGenerateContent`, and what the gateway does with it. AI Studio and Vertex share one
projection and one accounting declaration; their schemas differ by one member each.

- **Audited objects:** `UsageMetadata` and the `ModalityTokenCount` it references.
- **AI Studio (`v1beta`):** discovery document
  `https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta`, `revision`
  `20260910`. Google serves only the current revision, so the two schemas were captured at that
  revision and are identified by the SHA-256
  `1589a3bf7a5b7863ae0655d36da88be6b95050ff1183188f60843899b7a8531e` of
  `jq -S '{UsageMetadata: .schemas.UsageMetadata, ModalityTokenCount: .schemas.ModalityTokenCount}'`
  over the downloaded document.
- **Vertex (`v1`):** `googleapis/google-api-go-client` @
  `dfa9e13aa3e07a6283a6497f744ba5c4785a54fc`, `aiplatform/v1/aiplatform-api.json` (discovery
  `revision` `20260822`), schemas `GoogleCloudAiplatformV1GenerateContentResponseUsageMetadata` and
  `GoogleCloudAiplatformV1ModalityTokenCount`.
- Accessed 2026-09-11.

**Not read** means the member plays no part in cost, cost status, the spend row or the budget
counter.

| Member | `v1beta` | Vertex `v1` | Gateway use |
|---|---|---|---|
| `promptTokenCount` | ✓ | ✓ | `prompt_tokens`. Charged at the input rate after `cachedContentTokenCount` is carved out |
| `cachedContentTokenCount` | ✓ | ✓ | `cache_read_input_tokens`. Charged at the tier's `cache_read_multiplier`, and counted once when the tier is selected |
| `candidatesTokenCount` | ✓ | ✓ | `completion_tokens`. Charged whole at the output rate |
| `thoughtsTokenCount` | ✓ | ✓ | `completion_tokens_details.reasoning_tokens`. Charged at the thinking rate **beside** the candidates total |
| `totalTokenCount` | ✓ | ✓ | `total_tokens`. Reported, not priced |
| `toolUsePromptTokenCount` | ✓ | ✓ | Not read. `v1beta` describes it as the tokens "present in tool-use prompt(s)"; Vertex as the tokens "in the results from tool executions, which are provided back to the model as input", and lists it in `totalTokenCount` as an addend separate from `promptTokenCount`. An ordinary function result a client sends back — translated to a `functionResponse` part — is such a result, so the member can appear today. Whether Google populates it for function results, whether those tokens are outside `promptTokenCount`, and how they are billed are not established; until they are, the gateway neither charges nor subtracts it |
| `toolUsePromptTokensDetails[]` | ✓ | ✓ | Not read. The per-modality breakdown of `toolUsePromptTokenCount` |
| `promptTokensDetails[]` | ✓ | ✓ | Not read. The per-modality breakdown of `promptTokenCount`. Some modalities are priced differently, but the request translation forwards text parts only, so every prompt token it sends is text |
| `cacheTokensDetails[]` | ✓ | ✓ | Not read. The per-modality breakdown of `cachedContentTokenCount`; text only, for the same reason |
| `candidatesTokensDetails[]` | ✓ | ✓ | Not read. The per-modality breakdown of `candidatesTokenCount`. No bundled Gemini entry is an image- or audio-output model |
| `…Details[].modality` | ✓ | ✓ | Not read. `TEXT`, `IMAGE`, `VIDEO`, `AUDIO`, `DOCUMENT` or `MODALITY_UNSPECIFIED` |
| `…Details[].tokenCount` | ✓ | ✓ | Not read. The count for that modality |
| `serviceTier` | ✓ | — | Not read. The request translation selects no tier, so a request runs at the default, which the schema describes as standard |
| `trafficType` | — | ✓ | Not read. Reports how Vertex served the request — on demand, priority, flex, off-peak or provisioned throughput. The adapter selects none of these, and every request is priced at the catalogue's per-token rate; a request served from a provisioned-throughput commitment is priced the same way |

---

## Embeddings

The Gemini adapter supports `POST /v1/embeddings` with automatic single/batch dispatch.

### Supported embedding models

> **Known gap:** no Gemini embedding model is in the adapter's default model list. Google's
> current embedding model, `gemini-embedding-2`, is priced in the bundled snapshot, but the
> adapter does not yet speak its protocol: it sends a `taskType` the model does not accept, and it
> reads token usage from `embedding.statistics.tokenCount`, whereas the model reports it in
> `usageMetadata.promptTokenCount`. A request that succeeded would therefore be recorded at zero
> cost. Do not add the model through `supported_models` until this is fixed.

| Model | Dimensions | Max input tokens |
|-------|-----------|-----------------|
| `gemini-embedding-2` (not yet supported) | 3072 | 8192 |

### Single vs batch dispatch (API-key mode)

| Input count | Gemini API call |
|------------|-----------------|
| 1 | `embedContent` (lower latency) |
| > 1 | `batchEmbedContents` (single round-trip, max 100 items per Google docs) |

The Vertex AI arm always uses `predict` with `instances[]` for any input count.

### `embed_api_version` config field

Operators can override the API version segment for API-key mode:

```yaml
providers:
  gemini:
    embed_api_version: "v1beta"  # default: "v1"
```

- Applies to API-key arm only. Vertex always uses `/v1/`.
- Must not be empty or contain whitespace (validated at startup).
- Hot-reload class: **A** (requires provider restart on change).
- Default (`None`): `/v1/models/{model}:{endpoint}`.
- Override example: `v1beta` → `/v1beta/models/{model}:{endpoint}`.

### Token count parsing

Per-element `statistics.tokenCount` is extracted from each embedding response element and summed. A `WARN` log is emitted when `tokenCount` is absent in a response element; `0` is used in that case.

### Task type

All requests use `taskType: "RETRIEVAL_DOCUMENT"` (constant `GEMINI_DEFAULT_TASK_TYPE`).

---

## Changelog

| Date | Change |
|------|--------|
| 2026-10-10 | The health probe checks the configured `default_model` (default `gemini-3.8-flash`) instead of a fixed model name |
| 2026-10-10 | `gemini-2.5-pro`, `gemini-2.5-flash` and `gemini-2.5-flash-lite` leave the default model list: Vertex AI retires them on 2026-10-20 (replacements: `gemini-3.8-flash`, `gemini-3.1-flash-lite`). They stay priced, so a Gemini API deployment that still uses them can name them in `supported_models`. `supported_models` replaces the default list rather than extending it: list every other Gemini model you still want too, and add new releases yourself |
| 2026-10-10 | **Upgrade note.** `text-embedding-004`, `text-multilingual-embedding-002` and `gemini-embedding-exp-03-07` are no longer in the bundled pricing snapshot. Google has shut the first and last down on the Gemini API; Vertex serves `text-embedding-004` and `text-multilingual-embedding-002` until 2027-04-01. If your `supported_models` still names either, their requests are recorded at zero cost (reported as cost-unavailable) unless you add a `pricing.overrides` entry for them. Their replacement, `gemini-embedding-2`, is priced but not yet supported by the embeddings path — see *Supported embedding models* |
| 2026-09-03 | Model-specific cache-read multipliers imported into the bundled pricing snapshot. Cached prompt tokens are charged at their discounted rate instead of the full input rate |
| 2026-08-23 | Cached prompt tokens no longer charged twice, and no longer double-counted for tier selection. Cost drops sharply on cached prompts |
| 2026-05-09 | batchEmbedContents, embed_api_version, tokenCount parsing, EmbeddingCapabilities |
| 2026-05-06 | F4: gateway-level validation for missing/empty/orphaned `tool_call_id` |
| 2026-05-05 | Tool use translation |
