# Smoke Tests

Canonical verification commands for OxiGate.

Extend this doc as new features add verification steps (DB, Redis, Prometheus scrape, etc.).

**PostgreSQL:** Local and Docker runs require PostgreSQL and Redis. The `pgcrypto` extension must be available in PostgreSQL for migrations.

---

## 1. docker-compose up (recommended)

**Why:** Runs gateway + PostgreSQL + Redis in one command. Uses explicit `oxigate` network so containers resolve each other by name. No manual setup of Postgres/Redis on the host.

**Provider and auth keys via `.env`:** `docker-compose.yml` uses `env_file: .env` (optional — compose starts without it). Provider keys and auth config go in `.env`, not in the compose file. Never add `VAR: ${VAR:-}` in compose for optional provider sections — an empty string is still set and causes figment validation failure for that provider.

```bash
# One-time human setup (not automated):
#   cp -n .env.example .env   — only if .env does not already exist
#   Then uncomment and fill in the providers you want, e.g.:
#     OXIGATE__PROVIDERS__GEMINI__MODE=api
#     OXIGATE__PROVIDERS__GEMINI__API_KEY=your-key
#   OXIGATE__AUTH__KEY is optional — leave commented for bypass mode.
#   The curl commands below assume bypass mode (no Bearer token required).

# 1. Build and start full stack
docker compose up -d --build

# 2. Health check (gateway on 8080) — expect 200
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health | grep -x 200
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health/ready | grep -x 200
```

The `docker-compose.yml` wires Postgres, Redis, and gateway with `depends_on` and healthchecks. DB/Redis URLs are hard-coded in the compose `environment` block so they always resolve inside the network. All other config (provider keys, auth) comes from `.env`.

---

## 1b. docker-compose teardown

**When to run:** After all smoke sections are complete. Do not run between sections — the stack must stay up for sections 5 through 18.

```bash
# 1. Capture final gateway logs before stopping
docker compose logs --tail=20 gateway

# 2. Stop and remove containers (volumes are preserved — data survives for next run)
docker compose down
```

---

## 2. Local (cargo)

**Why:** Exercises the built binary on your host instead of in Docker. Useful for debugging, profiling, or when iterating on Rust code. Build has no external deps; the server requires Postgres + Redis for migrations and runtime.

**Order:** (1) Build, (2) Start Postgres + Redis, (3) Start server.

```bash
# Option A — manual docker run (human only; skip if section 1 stack is already up):
#   Fails if ports 5432/6379 are already bound (e.g. section 1 is running).
#   docker run -d --name oxi-pg -p 5432:5432 -e POSTGRES_USER=oxigate -e POSTGRES_PASSWORD=changeme -e POSTGRES_DB=oxigate postgres:16-alpine
#   docker run -d --name oxi-redis -p 6379:6379 redis:7-alpine
#   On re-run: docker start oxi-pg oxi-redis (if stopped) or docker rm -f oxi-pg oxi-redis (to recreate)

# 1. Build (no Postgres/Redis required)
cargo build --release

# 2. Start Postgres + Redis (no-op if already running from section 1)
docker compose up -d postgres redis

# Verify both are up:
docker ps | grep -E 'oxi-pg|oxi-redis|postgres|redis'

# 3. Start server (must have Postgres + Redis running; otherwise startup fails or exits)
# To debug startup: drop the trailing " &" to run in foreground and see errors directly.
OXIGATE__SERVER__PORT=19999 OXIGATE__DATABASE__URL=postgres://oxigate:changeme@localhost:5432/oxigate OXIGATE__REDIS__URL=redis://localhost:6379 ./target/release/oxigate --config config/oxigate.yaml & echo $! > /tmp/oxigate-smoke.pid
sleep 3

# 4. Health checks (expect 200; if "Connection refused", server didn't start — check steps 2 and 3)
curl -s -o /dev/null -w "%{http_code}" http://localhost:19999/health | grep -x 200
curl -s -o /dev/null -w "%{http_code}" http://localhost:19999/health/ready | grep -x 200

# 5. Unknown route → 404 (bypass mode — no auth.key configured)
#    If OXIGATE__AUTH__KEY is set, add: -H "Authorization: Bearer <your-token>"
curl -s http://localhost:19999/v1/nonexistent

# 6. Graceful shutdown (SIGTERM → exit 0)
kill -TERM $(cat /tmp/oxigate-smoke.pid) 2>/dev/null; sleep 2; kill -0 $(cat /tmp/oxigate-smoke.pid) 2>/dev/null || echo "Exit: 0"
# Expected: "Exit: 0" — process is gone, confirming clean shutdown

# 7. Lint/test gate
cargo xtask check
```

---

## 5. Provider manual tests (Gemini)

**Why:** Verifies the Gemini adapter against live Google APIs. Run after the gateway is up (docker compose or local). Requires real API credentials; not suitable for CI without secrets.

**Prerequisites:** Gateway running (port 8080 by default; local runs may use a different port). `jq` installed (`apt install jq` / `brew install jq`). Without Gemini configured, requests return 503 (no provider handles the model). Restart the gateway after changing config.

**Env vars:**

| Var | Who uses it | Purpose |
|-----|-------------|---------|
| `OXIGATE__PROVIDERS__GEMINI__MODE` | Gateway (startup) | Set to `api` to enable Gemini API mode |
| `OXIGATE__PROVIDERS__GEMINI__API_KEY` | Gateway (startup) | Your Google API key; get from https://aistudio.google.com/apikey |
| `OXIGATE__AUTH__KEY` | Gateway (startup) | Bearer token the gateway will enforce on `/v1/*`. When absent, auth is bypassed (dev/CI). When set, curl must pass a matching value. |
| `OXIGATE_API_KEY` | curl (client) | Value sent as `Authorization: Bearer $OXIGATE_API_KEY`. Must match `OXIGATE__AUTH__KEY` if that is set; any value works in bypass mode. |

Gateway vars: put in `.env` (docker-compose loads it automatically via `env_file`). Only set the vars for providers you actually want — leaving a provider var unset means that provider section is absent and no validation runs for it. Client var: set `OXIGATE_API_KEY` in your shell or in `.env.smoke-runner` (must match `OXIGATE__AUTH__KEY` if auth is configured; any value works in bypass mode). Vertex mode: use `vertex_service_account_json` in YAML (or path env) instead of the API key.

```bash
# Client auth (human setup — not automated):
#   export OXIGATE_API_KEY=test
#   Or set OXIGATE_API_KEY in .env.smoke-runner for automated runs.
#   In bypass mode (no OXIGATE__AUTH__KEY on gateway) any value works.

# 1. Non-streaming chat (Gemini API mode)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Say hello"}]}' | jq .
# Expected: {"choices":[{"message":{"content":"Hello..."}}], "usage":{...}}

# 2. Streaming chat
curl -sN -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Count to 3"}],"stream":true}' \
  | grep -E 'data: \[DONE\]|event: oxigate\.usage'
# Expected: `data: [DONE]` followed by `event: oxigate.usage` — the provider's terminator,
#           then the gateway's own terminal event. The second line is the one that matters:
#           it is only emitted once the request has been priced and its spend row scheduled.

# 3. Function calling
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"What is the weather in London?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}]}' | jq .choices[0].message.tool_calls
# Expected: [{"function":{"name":"get_weather","arguments":"..."}]

# 4. Embeddings — single input
curl -s -X POST http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-004","input":"Hello world"}' | jq '.data[0].embedding | length'
# Expected: 768 (text-embedding-004 dimension)

# 4a. Embeddings — batch input (batchEmbedContents,)
curl -s -X POST http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-004","input":["Hello world","Goodbye world"]}' | jq '{count: (.data | length), dim: (.data[0].embedding | length)}'
# Expected: {"count":2,"dim":768}

# 4b. Embedding cost headers
curl -s -D - -o /dev/null -X POST http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-004","input":"Hello world"}' | grep -i "X-Oxigate"
# Expected: X-Oxigate-Request-Cost: 0.000... (non-zero), X-Oxigate-Output-Tokens: 0

# 5. Cost headers present (-D - dumps headers to stdout; -o /dev/null discards body)
curl -s -D - -o /dev/null -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Hi"}]}' | grep -i X-Oxigate-Request-Cost
# Expected: X-Oxigate-Request-Cost: 0.000... (non-zero)

# 6. Invalid model → clean error (not panic). response headers include all four cost headers.
curl -s -D - -o /dev/null -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-does-not-exist","messages":[{"role":"user","content":"Hi"}]}' | grep -E 'X-Oxigate-Request-Cost|X-Oxigate-Input-Tokens|X-Oxigate-Output-Tokens|X-Oxigate-Model-Used'
# Expected: X-Oxigate-Request-Cost: 0.000000, X-Oxigate-Input-Tokens: 0, X-Oxigate-Output-Tokens: 0, X-Oxigate-Model-Used: gemini-does-not-exist
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-does-not-exist","messages":[{"role":"user","content":"Hi"}]}' | jq .error
# Expected: {"code":404,"message":"model not found: gemini-does-not-exist"}

# 7. Streaming thinking tokens (Gemini 2.5 Pro — thinking is always on)
# Note: Flash uses a dynamic thinking budget and may not emit completion_tokens_details
# for simple prompts. Use Pro here for a deterministic assertion.
curl -sN -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-pro","messages":[{"role":"user","content":"Solve: 17 * 23"}],"stream":true}' \
  | grep '"completion_tokens_details"'
# Expected: data: {...,"usage":{"completion_tokens_details":{"reasoning_tokens":N},...}}

# 8. Non-streaming thinking tokens (Gemini 2.5)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-pro","messages":[{"role":"user","content":"Solve: 17 * 23"}]}' \
  | jq '.usage.completion_tokens_details'
# Expected: {"reasoning_tokens": N}
```

---

## 6. Provider manual tests (OpenAI)

**Why:** Verifies the OpenAI adapter against live OpenAI API. Run after the gateway is up (docker compose or local). Requires real API credentials; not suitable for CI without secrets.

**Prerequisites:** Gateway running (port 8080 by default). `jq` installed (`apt install jq` / `brew install jq`). Without OpenAI configured, requests to `gpt-*` models return 503. Restart the gateway after changing config.

**Env vars:**

| Var | Who uses it | Purpose |
|-----|-------------|---------|
| `OXIGATE__PROVIDERS__OPENAI__API_KEY` | Gateway (startup) | Your OpenAI API key; get from https://platform.openai.com/api-keys |
| `OXIGATE__AUTH__KEY` | Gateway (startup) | Bearer token the gateway enforces on `/v1/*`. Absent = bypass (dev). |
| `OXIGATE_API_KEY` | curl (client) | Value sent as `Authorization: Bearer $OXIGATE_API_KEY`. Must match `OXIGATE__AUTH__KEY` if set. |

Gateway vars: put in `.env` (docker-compose loads it automatically via `env_file`). Only set the vars for providers you actually want. Client var: set `OXIGATE_API_KEY` in your shell or in `.env.smoke-runner` (must match `OXIGATE__AUTH__KEY` if auth is configured; any value works in bypass mode).

```bash
# Client auth (human setup — not automated):
#   export OXIGATE_API_KEY=test
#   Or set OXIGATE_API_KEY in .env.smoke-runner for automated runs.
#   In bypass mode (no OXIGATE__AUTH__KEY on gateway) any value works.

# 1. Non-streaming chat
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Say hello"}]}' | jq .
# Expected: {"choices":[{"message":{"content":"Hello..."}}], "usage":{...}}

# 2. Streaming chat
curl -sN -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Count to 3"}],"stream":true}' \
  | grep -E 'data: \[DONE\]|event: oxigate\.usage'
# Expected: `data: [DONE]` followed by `event: oxigate.usage` — the provider's terminator,
#           then the gateway's own terminal event. The second line is the one that matters:
#           it is only emitted once the request has been priced and its spend row scheduled.

# 3. Cost headers present
curl -s -D - -o /dev/null -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}' | grep -i X-Oxigate-Request-Cost
# Expected: X-Oxigate-Request-Cost: 0.000... (non-zero)

# 4. Invalid model → clean error (not panic)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-does-not-exist","messages":[{"role":"user","content":"Hi"}]}' | jq .error
# Expected: {"code":404,"message":"model not found: ..."}

# 5. Reasoning model (o3) — optional; requires o3 access
# curl -s -X POST http://localhost:8080/v1/chat/completions \
#   -H "Authorization: Bearer $OXIGATE_API_KEY" \
#   -H "Content-Type: application/json" \
#   -d '{"model":"o3-mini","messages":[{"role":"user","content":"Solve: 17 * 23"}]}' \
#   | jq '.usage.completion_tokens_details'
# Expected: {"reasoning_tokens": N} when model supports reasoning

# 6. OpenAI embeddings
curl -s -X POST http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-3-small","input":"Hello world"}' | jq '{dim: (.data[0].embedding | length), model: .model}'
# Expected: {"dim":1536,"model":"text-embedding-3-small"}

# 6a. OpenAI embeddings with dimensions param
curl -s -X POST http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-3-small","input":"Hello world","dimensions":512}' | jq '.data[0].embedding | length'
# Expected: 512
```

---

## 7. Provider manual tests (Anthropic)

**Why:** Verifies the Anthropic Claude adapter against live Anthropic API. Run after the gateway is up (docker compose or local). Requires real API credentials; not suitable for CI without secrets.

**Prerequisites:** Gateway running (port 8080 by default). `jq` installed (`apt install jq` / `brew install jq`). Without Anthropic configured, requests to `claude-*` models return 503. Restart the gateway after changing config. Anthropic requires `max_tokens` on every request; the adapter uses `default_max_tokens` (4096) when the request omits it.

**Env vars:**

| Var | Who uses it | Purpose |
|-----|-------------|---------|
| `OXIGATE__PROVIDERS__ANTHROPIC__API_KEY` | Gateway (startup) | Your Anthropic API key; get from https://console.anthropic.com/ |
| `OXIGATE__AUTH__KEY` | Gateway (startup) | Bearer token the gateway enforces on `/v1/*`. Absent = bypass (dev). |
| `OXIGATE_API_KEY` | curl (client) | Value sent as `Authorization: Bearer $OXIGATE_API_KEY`. Must match `OXIGATE__AUTH__KEY` if set. |

Gateway vars: put in `.env` (docker-compose loads it automatically via `env_file`). Only set the vars for providers you actually want. Client var: set `OXIGATE_API_KEY` in your shell or in `.env.smoke-runner` (must match `OXIGATE__AUTH__KEY` if auth is configured; any value works in bypass mode).

```bash
# Client auth (human setup — not automated):
#   export OXIGATE_API_KEY=test
#   Or set OXIGATE_API_KEY in .env.smoke-runner for automated runs.
#   In bypass mode (no OXIGATE__AUTH__KEY on gateway) any value works.

# 1. Non-streaming chat
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Say hello"}]}' | jq .
# Expected: {"choices":[{"message":{"content":"Hello..."}}], "usage":{...}}

# 2. Streaming chat
curl -sN -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Count to 3"}],"stream":true}' \
  | grep -E 'data: \[DONE\]|event: oxigate\.usage'
# Expected: `data: [DONE]` followed by `event: oxigate.usage` — the provider's terminator,
#           then the gateway's own terminal event. The second line is the one that matters:
#           it is only emitted once the request has been priced and its spend row scheduled.

# 3. Tool use (function calling)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"What is the weather in London?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}]}' | jq .choices[0].message.tool_calls
# Expected: [{"function":{"name":"get_weather","arguments":"..."}}]

# 4. Cache tokens surfaced (informational — not a pass/fail assertion)
# Anthropic includes cache_creation_input_tokens / cache_read_input_tokens only when
# caching activates. Fields will be null when absent — that is expected behaviour.
# This step confirms the adapter passes the fields through without crashing.
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Hi"}]}' | jq '.usage | {cache_creation_input_tokens, cache_read_input_tokens}'
# Expected: object with both keys present (values may be null or integer)

# 5. Cost headers present
curl -s -D - -o /dev/null -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Hi"}]}' | grep -i X-Oxigate-Request-Cost
# Expected: X-Oxigate-Request-Cost: 0.000... (non-zero)

# 6. Invalid model → clean error (not panic)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-does-not-exist","messages":[{"role":"user","content":"Hi"}]}' | jq .error
# Expected: {"code":404,"message":"model not found: ..."}
```

---

## 10. Spend tracking

**Why:** Confirms a `spend_records` row is written after every completed request and that
the structured cost log line appears in gateway output with `org_id`.

**Prerequisites:** Gateway running with Postgres + Redis via docker-compose (section 1) — `docker compose logs` is only available for compose-managed stacks, not local cargo runs. Run at least one provider test (sections 5, 6, or 7) first to generate a request.

```bash
# Verify the structured cost log line appears in the deployed gateway's stdout.
# Proves the tracing-subscriber is attached on the released binary path —
# a wiring concern not visible to in-process integration tests.
docker compose logs --tail=100 gateway 2>&1 | grep '"chat_completion_cost"'
# Expected: JSON log lines containing: request_id, org_id, identity_id, cost_usd, latency_ms
```

`spend_records` row insertion, Redis counter increment, and Redis TTL are covered by
`tests/integration/spend_writer.rs` and verified as side effects of any provider call (sections 5, 6, 7, 17, 18).


---

## 10a. Spend query API

**Why:** Verifies the three read endpoints aggregate `spend_records` correctly and enforce
tenant isolation.

**Prerequisites:** Gateway running with Postgres + Redis. `jq` installed. Run at least one chat completion first so spend rows exist (or seed directly via `psql`).

```bash
# Client auth (human setup — not automated):
#   export OXIGATE_API_KEY=test
#   Or set OXIGATE_API_KEY in .env.smoke-runner for automated runs.
#   In bypass mode (no OXIGATE__AUTH__KEY on gateway) any value works.

# Daily spend — last 30 days (default window)
curl -s http://localhost:8080/v1/spend/daily \
  -H "Authorization: Bearer $OXIGATE_API_KEY" | jq .
# Expected: {"data":[{"date":"YYYY-MM-DD","cost_nano_usd":<int>},...]}

# Daily spend — explicit range
curl -s "http://localhost:8080/v1/spend/daily?from=2025-01-01&to=2025-01-31" \
  -H "Authorization: Bearer $OXIGATE_API_KEY" | jq .

# Spend by provider
curl -s http://localhost:8080/v1/spend/providers \
  -H "Authorization: Bearer $OXIGATE_API_KEY" | jq .
# Expected: {"data":[{"dimension":"openai","cost_nano_usd":<int>},...]}

# Spend by model
curl -s http://localhost:8080/v1/spend/models \
  -H "Authorization: Bearer $OXIGATE_API_KEY" | jq .
# Expected: {"data":[{"dimension":"gpt-4.1","cost_nano_usd":<int>},...]}

# Invalid date format → 400
curl -s "http://localhost:8080/v1/spend/daily?from=not-a-date" \
  -H "Authorization: Bearer $OXIGATE_API_KEY" | jq .error
# Expected: "invalid date format: not-a-date"

# Range > 365 days → 400
curl -s "http://localhost:8080/v1/spend/daily?from=2020-01-01&to=2021-12-31" \
  -H "Authorization: Bearer $OXIGATE_API_KEY" | jq .error
# Expected: "invalid date range: range must not exceed 365 days"
```

---

## 10b. Streaming spend when the client stops at `[DONE]`

**Why:** Every mainstream OpenAI SDK stops iterating a stream at `data: [DONE]` and closes the
response there. It never reads to end of stream. A gateway that only finalizes accounting at
EOF therefore loses the spend row for exactly the clients that matter most — silently, with a
200 and a complete-looking response. This section proves the row is written at the provider's
terminal chunk instead, so it does not depend on the client reading anything after it.

The other streaming steps in this document cannot prove that: `curl` drains to EOF, so they
pass either way. The proof needs an upstream that **holds the connection open past `[DONE]`**,
making EOF unreachable inside the client's lifetime.

**Prerequisites:** Postgres + Redis running, `python3`, and the official `openai` package
(`pip install openai`). Ports 18080/18081 must be free. No provider key and no real spend
required — the upstream is a local mock.

```bash
# ── 1. Mock upstream: OpenAI-shaped SSE that holds the socket open for 20s after [DONE] ──
cat > /tmp/oxigate_done_mock.py <<'EOF'
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHUNKS = [
    'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n',
    'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop"}]}\n\n',
    'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}\n\n',
    'data: [DONE]\n\n',
]

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for c in CHUNKS:
            b = c.encode()
            self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n")
            self.wfile.flush()
            time.sleep(0.05)
        time.sleep(20)          # the load-bearing part: EOF is unreachable to the client
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
    def log_message(self, *a): pass

ThreadingHTTPServer(("127.0.0.1", 18080), H).serve_forever()
EOF
python3 /tmp/oxigate_done_mock.py &
MOCK_PID=$!

# ── 2. Gateway config pointed at the mock ─────────────────────────────────
cat > /tmp/oxigate_done_test.yaml <<'EOF'
server:
  port: 18081
  host: "127.0.0.1"
database:
  url: "postgres://oxigate:changeme@localhost:5432/oxigate"
redis:
  url: "redis://localhost:6379"
log_level: "info"
providers:
  openai:
    api_key: "not-validated-by-the-mock"
    api_base_url: "http://127.0.0.1:18080"
EOF

cargo run --release --bin oxigate -- --config /tmp/oxigate_done_test.yaml &
GW_PID=$!
until curl -sf http://127.0.0.1:18081/health/ready >/dev/null; do sleep 1; done

# ── 3. Client: the real SDK, which stops at [DONE] ───────────────────────
python3 - <<'EOF'
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:18081/v1", api_key="unused")
n = 0
with client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
) as stream:
    for _ in stream:
        n += 1
print(f"SDK stopped after {n} chunks without reading to end of stream")
EOF

# ── 4. The assertion — a spend row exists while the upstream is still open ──
psql "postgres://oxigate:changeme@localhost:5432/oxigate" -t -c \
  "SELECT count(*) FROM spend_records
   WHERE model = 'gpt-4o-mini' AND prompt_tokens = 11 AND completion_tokens = 7
     AND created_at > NOW() - INTERVAL '1 minute';"
# Expected: 1
# A 0 here is the regression: accounting waited for an EOF the client never reached.

# ── 5. Teardown ───────────────────────────────────────────────────────────
kill $GW_PID $MOCK_PID 2>/dev/null
rm -f /tmp/oxigate_done_mock.py /tmp/oxigate_done_test.yaml
```

Run step 4 **before** the mock's 20-second hold expires. If it has expired, the run proves
nothing either way — the stream reached EOF, which is the path this test exists to bypass.

---

## 11a. GlobalSafetyLayer — instance-wide cap

Community feature. Blocks all `/v1/*` requests with 429 when aggregate instance spend exceeds
`budget.global_safety_cap_usd`. Zero overhead when cap is not configured.

**Prerequisites:** Postgres + Redis running (section 1 or section 2). `redis-cli` installed. This section starts its own local gateway instance — stop any other locally-running gateway first to avoid port conflicts.

```bash
# 1. Start gateway with global safety cap enabled
OXIGATE__BUDGET__GLOBAL_SAFETY_CAP_USD=10.0 \
OXIGATE__DATABASE__URL=postgres://oxigate:changeme@localhost:5432/oxigate \
OXIGATE__REDIS__URL=redis://localhost:6379 \
cargo run -- --config config/oxigate.yaml & echo $! > /tmp/oxigate-smoke.pid
sleep 3

# 2. Seed global spend above cap in Redis
redis-cli SET "oxigate:global:spend" 10000000001

# 3. Any /v1/* request should return 429 with budget cap header
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/v1/models | grep -x 429
# Expected: 429 (body: {"error":"global_budget_cap_exceeded"}, header: X-Oxigate-Budget-Cap: global)

# 4. Verify header and body with verbose output
curl -sv http://localhost:8080/v1/models 2>&1 | grep -E "< HTTP|X-Oxigate-Budget-Cap|global_budget"

# 5. Reset spend below cap and verify pass-through
redis-cli SET "oxigate:global:spend" 9999999999
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/v1/models | grep -x 200
# Expected: 200

# 6. Verify SIGHUP reloads the cap (human only — requires manual config edit before signalling)
#   In your editor: update OXIGATE__BUDGET__GLOBAL_SAFETY_CAP_USD in config/oxigate.yaml
#   Then send SIGHUP: kill -HUP $(cat /tmp/oxigate-smoke.pid)
#   Logs should show: "Class A reload: applying config, pricing, auth, and provider"

# 7. Teardown
kill -TERM $(cat /tmp/oxigate-smoke.pid) 2>/dev/null; sleep 1
```

---

## 12. Structured JSON logging

**Why:** Verifies log output contract and runtime log-level hot-reload behavior for operations.

**Prerequisites:** Postgres + Redis running (section 1 or section 2). No other gateway on port 8080 — this section starts its own.

```bash
# 1. Start gateway and capture logs to a file
RUST_LOG=info OXIGATE__DATABASE__URL=postgres://oxigate:changeme@localhost:5432/oxigate \
OXIGATE__REDIS__URL=redis://localhost:6379 \
cargo run -- --config config/oxigate.yaml > /tmp/oxigate-logging.log 2>&1 & echo $! > /tmp/oxigate-smoke.pid

# Wait for startup logs to appear
sleep 3

# 2. Validate startup log lines are JSON and include required keys
python3 - <<'PY'
import json
from pathlib import Path
lines = [ln for ln in Path("/tmp/oxigate-logging.log").read_text().splitlines() if ln.strip()]
assert lines, "log file is empty — server may still be starting"
for ln in lines[:10]:
    event = json.loads(ln)
    for key in ("timestamp", "level", "target", "message"):
        assert key in event, f"missing {key} in {event}"
print("JSON field contract: OK")
PY

# 3. Verify SIGHUP applies log_level change without restart (human only — requires config edit)
#   In your editor: change log_level in config/oxigate.yaml (e.g. warn -> info)
#   Then send SIGHUP: kill -HUP $(cat /tmp/oxigate-smoke.pid)

# 4. Confirm reload logs are present in captured output (human only — run after step 3)
#   grep -E '"log level updated"|"SIGHUP received"' /tmp/oxigate-logging.log

# 5. Cleanup
kill -TERM $(cat /tmp/oxigate-smoke.pid) 2>/dev/null; sleep 1
```

Expected:
- JSON parse succeeds and required top-level keys are present.
- `SIGHUP` path logs reload activity and applies the new level without process restart (manual step).

---

## 13. Observability

**Why:** Verifies a structured `"request completed"` log event is emitted for every request with required metadata fields.

**Prerequisites:** Postgres + Redis running (section 1 or section 2). No other gateway on port 8080 — this section starts its own. At least one provider configured in `.env` — the curl in step 2 uses `gemini-2.5-flash` by default; substitute any configured model if Gemini is not available.

```bash
# 1. Start gateway and capture logs
RUST_LOG=info OXIGATE__DATABASE__URL=postgres://oxigate:changeme@localhost:5432/oxigate \
OXIGATE__REDIS__URL=redis://localhost:6379 \
cargo run -- --config config/oxigate.yaml > /tmp/oxigate-observability.log 2>&1 & echo $! > /tmp/oxigate-smoke.pid

# Wait for gateway to be ready
sleep 3

# 2. Send a test request — substitute model if Gemini is not configured
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Hi"}]}' | jq .

# 3. Verify "request completed" span log exists with required fields
python3 - <<'PY'
import json
from pathlib import Path
lines = [l for l in Path("/tmp/oxigate-observability.log").read_text().splitlines() if l.strip()]
assert lines, "log file is empty — server may still be starting"
events = [json.loads(l) for l in lines]
completed = [e for e in events if e.get("message") == "request completed"]
assert completed, "No 'request completed' log event found"
span = completed[0].get("span", {})
required = ["request_id", "method", "path", "provider", "model_family",
            "status_code", "duration_ms", "cost_usd", "prompt_tokens", "completion_tokens"]
missing = [f for f in required if f not in span]
assert not missing, f"Missing span fields: {missing}"
# PII check: no Authorization header values in span
for k, v in span.items():
    assert "sk-" not in str(v), f"Possible API key in span field {k}"
print("Request span: OK")
PY

# 4. Cleanup
kill -TERM $(cat /tmp/oxigate-smoke.pid) 2>/dev/null; sleep 1
```

---

## 14. Load balancing strategies
 
**Why:** Verifies routing strategy config loading and basic routing behavior.
Algorithm correctness (cooldown, retry_after calculation) is covered by integration
tests (`cargo nextest run --test integration routing`).
 
**Cost:** ~$0.03–$0.04 per run (15 requests total with gpt-4o/claude-3-haiku).
 
**Prerequisites:** Gateway running with ≥2 providers configured.

Routing strategy config-load logging is covered by `tests/integration/routing.rs` startup
assertions; the smoke focuses on live distribution behaviour, which is the unique value-add
against real upstreams.

1. Verify weighted distribution (WeightedRandom, cost: ~$0.02)
Configure provider_a weight=9.0, provider_b weight=1.0 in YAML
Send 10 requests — expect ~90/10 split (±15% noise for small sample)

```bash
# OXIGATE_API_KEY — set in .env.smoke-runner; leave unset for bypass mode.
for i in {1..10}; do
  curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H "Authorization: Bearer $OXIGATE_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}' > /dev/null
done

# Check provider distribution in spend_records (fails if no rows written)
docker compose exec -T postgres psql -U oxigate oxigate \
  -c "SELECT provider, COUNT(*) FROM spend_records 
      WHERE created_at > NOW() - INTERVAL '1 minute' 
      GROUP BY provider;" | grep '([1-9]'
```
Expected: ~9 provider_a, ~1 provider_b (±15% noise)

2. Verify zero-weight provider exclusion (cost: ~$0.01)
Configure provider_a weight=0.0, provider_b weight=1.0
All requests should route to provider_b
```bash
for i in {1..4}; do
  curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H "Authorization: Bearer $OXIGATE_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}' > /dev/null
done

# Verify only provider_b was used (fails if no rows written)
docker compose exec -T postgres psql -U oxigate oxigate \
  -c "SELECT DISTINCT provider FROM spend_records 
      WHERE created_at > NOW() - INTERVAL '1 minute';" | grep '([1-9]'
```
Expected: 4 provider_b (provider_a never selected)

3. Verify LowestCost strategy selects cheapest provider (cost: ~$0.002)
Configure provider_a (cheap model) and provider_b (expensive model)
Requests should route to provider_a

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-3-haiku-20240307","messages":[{"role":"user","content":"Hi"}]}' > /dev/null

docker compose exec -T postgres psql -U oxigate oxigate \
  -c "SELECT provider FROM spend_records 
      WHERE created_at > NOW() - INTERVAL '1 minute' 
      ORDER BY created_at DESC LIMIT 1;" | grep '([1-9]'
```

Expected: "provider_a" (lower cost per pricing DB)


## 15. Prometheus metrics

**Why:** Verifies the `/metrics` scrape endpoint is reachable, unauthenticated, returns valid
Prometheus text, and exposes the required baseline and fallback/retry metric families.

**Prerequisites:** Gateway running (section 1 docker-compose or section 2 local). Run at least one provider
request first (sections 5, 6, or 7) to populate counters — a cold gateway still returns 200 with
zero-valued metrics, but request counters will only appear after traffic flows.

```bash
# 1. Endpoint reachable and returns 200 — proves /metrics route registered in deployed binary.
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/metrics | grep -x 200
# Expected: 200

# 2. Auth bypass (human only — restart local gateway with OXIGATE__AUTH__KEY=<token> set first; see section 2):
#    curl -s -o /dev/null -w "%{http_code}" http://localhost:19999/metrics | grep -x 200
#    Expected: 200 (not 401 — /metrics bypasses auth layer)
#    curl -s -o /dev/null -w "%{http_code}" http://localhost:19999/v1/nonexistent | grep -x 401
#    Expected: 401 — /v1/* is protected
```

Prom text format, baseline metric families, and fallback metric names are covered by
`tests/integration/prometheus_metrics.rs::test_metrics_output_contains_required_metric_families`.

---

## 16. LiteLLM proxy pattern (via OpenAI adapter)

**Why:** Verifies that OxiGate can proxy requests through a LiteLLM instance using the OpenAI
adapter's `api_base_url` override — giving access to all 100+ LiteLLM-supported providers without
any additional Rust code. Confirms that OxiGate's token counting, cost headers, and budget
enforcement remain accurate when LiteLLM sits between OxiGate and the upstream provider.

> **FinOps accuracy note:** OxiGate reads `usage` from LiteLLM's response body. LiteLLM passes
> through provider-reported counts for most OpenAI-compatible providers. For providers where
> LiteLLM uses tiktoken-based estimation, token counts may drift from what the provider bills.
> Validate `X-Oxigate-Cost-*` headers against your provider dashboard for the first few days after
> onboarding a new provider via LiteLLM.

**Prerequisites:** Postgres + Redis running (section 2). LiteLLM installed (`pip install litellm`). A
provider API key for any LiteLLM-supported provider (example below uses Groq — free tier).

```bash
# ── 1. Human setup — configure before running automated steps ─────────────
# Set GROQ_API_KEY in .env.smoke-runner or export in your shell before running.
# Add to .env (docker-compose) or export before starting the gateway:
#   OXIGATE__PROVIDERS__OPENAI__API_BASE_URL=http://localhost:4000
#   OXIGATE__PROVIDERS__OPENAI__API_KEY=proxy
# Client auth: set OXIGATE_API_KEY=test in .env.smoke-runner or your shell.
# Then restart: docker compose up -d  or  see section 2 for local restart.

# ── 2. Start LiteLLM proxy (GROQ_API_KEY must be in environment) ──────────
litellm --model groq/llama-3.1-8b-instant --port 4000 & echo $! > /tmp/litellm-smoke.pid
sleep 3

# ── 3. Non-streaming — verify response and cost headers ────────────────────
curl -s -D - -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"groq/llama-3.1-8b-instant","messages":[{"role":"user","content":"Say hi"}]}' \
  | tee /tmp/litellm_smoke.txt
# Expected: HTTP 200; OpenAI-format body; X-Oxigate-Cost-* headers with non-zero values

grep -i "x-oxigate" /tmp/litellm_smoke.txt

# ── 4. Streaming ───────────────────────────────────────────────────────────
curl -s -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"groq/llama-3.1-8b-instant","messages":[{"role":"user","content":"Count to 3"}],"stream":true}' \
  | grep 'data: \[DONE\]'
# Expected: SSE chunks followed by data: [DONE]

# ── 5. FinOps accuracy spot-check (human only) ────────────────────────────
# Compare X-Oxigate-Cost-Input-Tokens (from /tmp/litellm_smoke.txt) against
# LiteLLM logs for the same request. They should match exactly for Groq
# (provider-reported passthrough). No automated assertion.

# ── 6. Teardown ───────────────────────────────────────────────────────────
kill -TERM $(cat /tmp/litellm-smoke.pid) 2>/dev/null; sleep 1
```

---

## 17. AWS Bedrock adapter

**Why:** Verifies SigV4 signing, EventStream streaming, Claude model routing, and cost headers

**Prerequisites:** AWS credentials with `bedrock:InvokeModel`, `bedrock:Converse`, and
`bedrock:ConverseStream` permissions for `anthropic.*` model IDs in the configured region.

```bash
# Gateway env vars (human setup — not automated):
#   Put in .env (docker-compose) or export in shell before starting the gateway.
#   AWS_ACCESS_KEY_ID=AKIA...
#   AWS_SECRET_ACCESS_KEY=...
#   AWS_DEFAULT_REGION=us-east-1
#   OXIGATE__PROVIDERS__BEDROCK__REGION=us-east-1
# Client auth: set OXIGATE_API_KEY=test in .env.smoke-runner or your shell.
#   In bypass mode (no OXIGATE__AUTH__KEY on gateway) any value works.

# ── Non-streaming ──────────────────────────────────────────────────────────
curl -s -D - -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic.claude-3-5-sonnet-20241022-v2:0","messages":[{"role":"user","content":"Say hi"}]}' \
  | tee /tmp/bedrock_smoke.txt
# Expected: HTTP 200; OpenAI-format JSON; X-Oxigate-Cost-* headers with non-zero values

grep -i "x-oxigate" /tmp/bedrock_smoke.txt

# ── Streaming ──────────────────────────────────────────────────────────────
curl -s -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic.claude-3-5-sonnet-20241022-v2:0","messages":[{"role":"user","content":"Count to 3"}],"stream":true}' \
  | grep 'data: \[DONE\]'
# Expected: OpenAI SSE chunks followed by data: [DONE]
```

Unknown-model-prefix dispatch is OxiGate-internal logic and is covered by
`tests/integration/providers/bedrock.rs` + adapter unknown-prefix handling.

---

## 18. Azure OpenAI adapter

**Why:** Verifies deployment-based URL construction, `api-key` header auth, `stream_options` injection for streaming, and cost header accuracy for Azure-deployed models.

**Prerequisites:** An Azure OpenAI resource with a deployed model (e.g. `gpt-4o`). Use `api_version: "2024-10-21"`.

```bash
# Gateway env vars (human setup — not automated):
#   Put in .env (docker-compose) or export in shell before starting the gateway.
#   AZURE_ENDPOINT=https://my-resource.openai.azure.com
#   AZURE_DEPLOYMENT=gpt-4o
#   AZURE_API_KEY=...
# Client auth: set OXIGATE_API_KEY=test in .env.smoke-runner or your shell.
#   In bypass mode (no OXIGATE__AUTH__KEY on gateway) any value works.

# ── Non-streaming ──────────────────────────────────────────────────────────
curl -s -D - -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Say hi"}]}' \
  | tee /tmp/azure_smoke.txt
# Expected: HTTP 200; OpenAI-format JSON; X-Oxigate-Cost-* headers with non-zero values

grep -i "x-oxigate" /tmp/azure_smoke.txt

# ── Streaming ──────────────────────────────────────────────────────────────
curl -s -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Count to 3"}],"stream":true}' \
  | grep 'data: \[DONE\]'
# Expected: OpenAI SSE chunks; final chunk contains usage with non-zero token counts; data: [DONE]
```

Config validation (empty / unsafe `deployment_name`) is covered by
`src/config.rs::azure_empty_deployment_name_rejected` and adjacent tests.

---

## Workarounds (if it didn't work)

| Symptom | Workaround |
|---------|------------|
| **"Connection refused"** on curl | Server didn't start. Ensure Postgres + Redis are running before starting the gateway. Run the gateway without `&` to see startup errors. |
| **"Temporary failure in name resolution"** in gateway logs | Gateway can't resolve `postgres` or `redis`. Ensure `docker-compose.yml` uses explicit `networks: - oxigate` (all services on same network). Run `docker compose down` then `docker compose up -d --build` to recreate on the network. |
| **"migration N was previously applied but has been modified"** | SQLx checksum mismatch: DB was migrated with different migration files. Reset the DB: `docker compose down -v` then `docker compose up -d --build`. The `-v` removes volumes so migrations run fresh. |
| **"Container name already in use"** (Option A) | Containers `oxi-pg` or `oxi-redis` exist from a prior run. If running: proceed to step 3. If stopped: `docker start oxi-pg oxi-redis`. To recreate: `docker rm -f oxi-pg oxi-redis` then re-run the `docker run` commands. |
| **Port already allocated** (5432, 6379, 8080) | Another process or container holds the port. Stop conflicting containers (`docker compose down` or `docker rm -f oxi-pg oxi-redis`) or use a different port. |
| **curl returns nothing** | Run `curl -v` to see HTTP status and errors. Check `docker compose ps` and `docker compose logs gateway` — gateway may be crash-looping. |
| **`providers.X.api_key is required`** at startup | A provider section is being declared with an empty key. Do not use `VAR: ${VAR:-}` in `docker-compose.yml` for optional provider vars — an empty string counts as "declared". Put provider keys in `.env` only; unset vars are not passed to the container. |
| **`unknown variant: found \`\``** at startup | `OXIGATE__PROVIDERS__GEMINI__MODE` is set to an empty string. Either set it to `api` or `vertex` in `.env`, or remove it entirely to skip Gemini. |
| **`curl` returns empty body** | Gateway is crash-looping (config error). Run `docker compose logs` to see the startup error, fix `.env`, then `docker compose up -d` (no rebuild needed for config-only changes). |
| **"connection refused"** (Gemini tests without Gemini config) | No providers are configured. Configure Gemini: add `OXIGATE__PROVIDERS__GEMINI__MODE=api` and `OXIGATE__PROVIDERS__GEMINI__API_KEY=...` to `.env`, then restart. |
| **"connection refused"** (OpenAI tests without OpenAI config) | No OpenAI provider is configured. Add `OXIGATE__PROVIDERS__OPENAI__API_KEY=...` to `.env`, then restart. |
| **"gemini: connection refused"** or **"gemini: timeout"** | Gemini is configured but Google API is unreachable (network, firewall, or invalid endpoint). Check `GOOGLE_API_KEY` and network access to generativelanguage.googleapis.com. |
| **"openai: connection refused"** or **"openai: timeout"** | OpenAI is configured but OpenAI API is unreachable. Check `OXIGATE__PROVIDERS__OPENAI__API_KEY` and network access to api.openai.com. |
| **"connection refused"** (Anthropic tests without Anthropic config) | No Anthropic provider is configured. Add `OXIGATE__PROVIDERS__ANTHROPIC__API_KEY=...` to `.env`, then restart. |
| **"anthropic: connection refused"** or **"anthropic: timeout"** | Anthropic is configured but API is unreachable. Check `OXIGATE__PROVIDERS__ANTHROPIC__API_KEY` and network access to api.anthropic.com. |
| **401 Unauthorized on /v1/*** | `OXIGATE__AUTH__KEY` is set on the gateway and the curl command is missing or using the wrong Bearer token. Set `export OXIGATE_API_KEY=<same-value-as-OXIGATE__AUTH__KEY>` and add `-H "Authorization: Bearer $OXIGATE_API_KEY"` to your curl. Health routes (`/health`, `/health/ready`) never require auth. |
| **401 Unauthorized on /health** | Auth layer is incorrectly applied to health routes — Health routes must be on the top-level Router, not the `/v1/` sub-router. |
| **startup says python-bridge / feature mismatch** | Expected when YAML enables `providers.python_bridge` but the binary was built without `--features python-bridge`. Rebuild with `cargo build --release --features python-bridge` or remove/disable the bridge section. |
| **503 Python bridge unavailable** | LiteLLM missing from `venv_path`, wrong venv, or import failure. Install `litellm` in that venv; confirm `GROQ_API_KEY` for live Groq calls. |
| **Migration checksum mismatch after squash** | Run `docker compose down -v` then `docker compose up -d --build`. Migration 0001 was replaced with a new squashed file — existing DB state is incompatible with the new checksum. |
| **X-Oxigate-Request-Cost missing** (step 5) | Run without grep to see full response: `curl -s -D - -o /dev/null -X POST ...` — if 401, check `OXIGATE_API_KEY` matches `OXIGATE__AUTH__KEY`; if 200, headers should include X-Oxigate-Request-Cost. |
