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

## Embeddings

The Gemini adapter supports `POST /v1/embeddings` with automatic single/batch dispatch.

### Supported embedding models

| Model | Dimensions | Max input tokens |
|-------|-----------|-----------------|
| `text-embedding-004` | 768 | 2048 |
| `gemini-embedding-exp-03-07` | 3072 | 2048 |
| `text-multilingual-embedding-002` | 768 | 2048 |

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
| 2026-05-09 | batchEmbedContents, embed_api_version, tokenCount parsing, EmbeddingCapabilities |
| 2026-05-06 | F4: gateway-level validation for missing/empty/orphaned `tool_call_id` |
| 2026-05-05 | Tool use translation |
