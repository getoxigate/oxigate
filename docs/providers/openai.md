# OpenAI Provider Implementation Notes

**Provider:** OpenAI  
**Adapter:** `src/providers/openai/`  

---

## Embeddings

The OpenAI adapter supports `POST /v1/embeddings` via the `embeddings()` method.

### Supported models

| Model | Dimensions | Input token limit |
|-------|-----------|-------------------|
| `text-embedding-3-small` | 512, 1536 | 8191 |
| `text-embedding-3-large` | 256, 1024, 3072 | 8191 |
| `text-embedding-ada-002` | 1536 | 8191 |

### Configuration

No extra config required beyond `providers.openai.api_key`. The `supported_models` list in YAML also governs which embedding models the adapter declares (operators may extend it).

### Request forwarding

The request body (`model`, `input`, `dimensions`, `encoding_format`) is forwarded verbatim to `https://api.openai.com/v1/embeddings` (or `api_base_url/v1/embeddings` when overridden).

### Token normalisation

OpenAI's embedding response may return `prompt_tokens: 0` with `total_tokens > 0` on some API versions. The adapter backfills `prompt_tokens` from `total_tokens` in that case so downstream cost tracking is accurate.

### Cost headers

Cost headers (`X-Oxigate-Request-Cost`, `X-Oxigate-Input-Tokens`, `X-Oxigate-Output-Tokens`) are injected on every successful response. `X-Oxigate-Output-Tokens` is always `0` for embeddings.

---

## Reasoning models (o-series)

| Feature | Behaviour |
|---------|-----------|
| `max_tokens` | Converted to `max_completion_tokens` |
| `system` role | Converted to `developer` role |
| `temperature` / `top_p` | Stripped for o1-series; forwarded for o3/o4-series |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-09 | embeddings() impl, EmbeddingCapabilities, cost headers for /v1/embeddings |
| 2026-05-05 | Initial OpenAI adapter |
