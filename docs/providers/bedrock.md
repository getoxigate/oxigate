# Bedrock Provider Implementation Notes

**Provider:** AWS Bedrock Converse API  
**Adapter:** `src/providers/bedrock/`  

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
| 2026-05-05 | Non-streaming tool use translation; streaming guard |
