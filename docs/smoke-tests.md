# Smoke Tests

Canonical verification commands for OxiGate.

Extend this doc as new features add verification steps (DB, Redis, Prometheus scrape, etc.).

**PostgreSQL:** Local and Docker runs require PostgreSQL and Redis. The `pgcrypto` extension must be available in PostgreSQL for migrations.

---

## 1. docker-compose (recommended)

**Why:** Runs gateway + PostgreSQL + Redis in one command. Uses explicit `oxigate` network so containers resolve each other by name. No manual setup of Postgres/Redis on the host.

**Provider and auth keys via `.env`:** `docker-compose.yml` uses `env_file: .env` (optional — compose starts without it). Provider keys and auth config go in `.env`, not in the compose file. Never add `VAR: ${VAR:-}` in compose for optional provider sections — an empty string is still set and causes figment validation failure for that provider.

```bash
# 1. Create .env from the example and fill in your keys
cp .env.example .env
# Edit .env — uncomment and set the providers you want, e.g.:
#   OXIGATE__PROVIDERS__GEMINI__MODE=api
#   OXIGATE__PROVIDERS__GEMINI__API_KEY=your-key
#   OXIGATE__AUTH__KEY=your-secret-token   # optional; omit for bypass mode

# 2. Build and start full stack
docker compose up -d --build

# 3. Health check (gateway on 8080) — expect JSON like {"status":"ok"}
curl -s http://localhost:8080/health
curl -s http://localhost:8080/health/ready

# 4. Graceful shutdown
docker compose down
docker compose logs gateway 2>&1 | tail -10
```

The `docker-compose.yml` wires Postgres, Redis, and gateway with `depends_on` and healthchecks. DB/Redis URLs are hard-coded in the compose `environment` block so they always resolve inside the network. All other config (provider keys, auth) comes from `.env`.

---

## 2. Local (cargo)

**Why:** Exercises the built binary on your host instead of in Docker. Useful for debugging, profiling, or when iterating on Rust code. Build has no external deps; the server requires Postgres + Redis for migrations and runtime.

**Order:** (1) Build, (2) Start Postgres + Redis, (3) Start server.

```bash
# 1. Build (no Postgres/Redis required)
cargo build --release

# 2. Start Postgres + Redis — pick one option, BEFORE starting the server

### Option A — manual docker run:
# If already ran before, either: docker start oxi-pg oxi-redis  (if stopped)  OR  docker rm -f oxi-pg oxi-redis  (to recreate)
docker run -d --name oxi-pg -p 5432:5432 -e POSTGRES_USER=oxigate -e POSTGRES_PASSWORD=changeme -e POSTGRES_DB=oxigate postgres:16-alpine
docker run -d --name oxi-redis -p 6379:6379 redis:7-alpine

### Option B — docker compose (postgres + redis only; gateway runs locally):
docker compose up -d postgres redis

# Verify both are up:
docker ps | grep -E 'oxi-pg|oxi-redis|postgres|redis'

# 3. Start server (must have Postgres + Redis running; otherwise startup fails or exits)
OXIGATE__SERVER__PORT=19999 OXIGATE__DATABASE__URL=postgres://oxigate:changeme@localhost:5432/oxigate OXIGATE__REDIS__URL=redis://localhost:6379 ./target/release/oxigate --config config/oxigate.yaml &
# If you see nothing, run without & to see errors: drop the trailing " &"

# 4. Health checks (expect JSON; if "Connection refused", server didn't start — check step 2 & 3)
curl -s http://localhost:19999/health
curl -s http://localhost:19999/health/ready

# 5. Unknown route → 404 (bypass mode — no auth.key configured)
#    If OXIGATE__AUTH__KEY is set, add: -H "Authorization: Bearer <your-token>"
curl -s http://localhost:19999/v1/nonexistent

# 6. Graceful shutdown (SIGTERM → exit 0). %1 = first background job
kill -TERM %1 2>/dev/null; wait %1 2>/dev/null; echo "Exit: $?"
# Expected: "Terminated"; "Exit: 0" = success

# 7. Lint/test gate
cargo xtask check
```

---

## 5. Provider manual tests (Gemini)

**Why:** Verifies the Gemini adapter against live Google APIs. Run after the gateway is up (docker compose or local). Requires real API credentials; not suitable for CI without secrets.

**Prerequisites:** Gateway running (port 8080 by default; local runs may use a different port). Without Gemini configured, requests return 503 (no provider handles the model). The gateway returns 503 if no configured provider handles the requested model. Restart the gateway after changing config.

**Env vars:**

| Var | Who uses it | Purpose |
|-----|-------------|---------|
| `OXIGATE__PROVIDERS__GEMINI__MODE` | Gateway (startup) | Set to `api` to enable Gemini API mode |
| `OXIGATE__PROVIDERS__GEMINI__API_KEY` | Gateway (startup) | Your Google API key; get from https://aistudio.google.com/apikey |
| `OXIGATE__AUTH__KEY` | Gateway + curl client | Bearer token the gateway enforces on `/v1/*`. Choose any value, set it in `.env`, and `export` it in your shell so curl sends it as `Authorization: Bearer $OXIGATE__AUTH__KEY`. When unset, auth is bypassed (dev/CI). |

Gateway vars: put in `.env` (docker-compose loads it automatically via `env_file`). Only set the vars for providers you actually want — leaving a provider var unset means that provider section is absent and no validation runs for it. Before running curl, `export OXIGATE__AUTH__KEY=<your-token>` in your shell so it is sent as the Bearer token. Vertex mode: use `vertex_service_account_json` in YAML (or path env) instead of the API key.

```bash
# Client auth (curl needs this):
export OXIGATE__AUTH__KEY=test

# 1. Non-streaming chat (Gemini API mode)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Say hello"}]}' | jq .
# Expected: {"choices":[{"message":{"content":"Hello..."}}], "usage":{...}}

# 2. Streaming chat
curl -sN -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Count to 3"}],"stream":true}'
# Expected: data: {"choices":[{"delta":{"content":"..."}}]} lines, ending with data: [DONE]

# 3. Function calling
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"What is the weather in London?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}]}' | jq .choices[0].message.tool_calls
# Expected: [{"function":{"name":"get_weather","arguments":"..."}]

# 4. Embeddings — single input
curl -s -X POST http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-004","input":"Hello world"}' | jq '.data[0].embedding | length'
# Expected: 768 (text-embedding-004 dimension)

# 4a. Embeddings — batch input (batchEmbedContents,)
curl -s -X POST http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-004","input":["Hello world","Goodbye world"]}' | jq '{count: (.data | length), dim: (.data[0].embedding | length)}'
# Expected: {"count":2,"dim":768}

# 4b. Embedding cost headers
curl -s -D - -o /dev/null -X POST http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-004","input":"Hello world"}' | grep -i "X-Oxigate"
# Expected: X-Oxigate-Request-Cost: 0.000... (non-zero), X-Oxigate-Output-Tokens: 0

# 5. Cost headers present (-D - dumps headers to stdout; -o /dev/null discards body)
curl -s -D - -o /dev/null -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Hi"}]}' | grep -i X-Oxigate-Request-Cost
# Expected: X-Oxigate-Request-Cost: 0.000... (non-zero)

# 6. Invalid model → clean error (not panic). response headers include all four cost headers.
curl -s -D - -o /dev/null -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-does-not-exist","messages":[{"role":"user","content":"Hi"}]}' | grep -E 'X-Oxigate-Request-Cost|X-Oxigate-Input-Tokens|X-Oxigate-Output-Tokens|X-Oxigate-Model-Used'
# Expected: X-Oxigate-Request-Cost: 0.000000, X-Oxigate-Input-Tokens: 0, X-Oxigate-Output-Tokens: 0, X-Oxigate-Model-Used: gemini-does-not-exist
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-does-not-exist","messages":[{"role":"user","content":"Hi"}]}' | jq .error
# Expected: {"code":404,"message":"model not found: gemini-does-not-exist"}

# 7. Streaming thinking tokens (Gemini 2.5)
curl -sN -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Solve: 17 * 23"}],"stream":true}' \
  | grep '"completion_tokens_details"'
# Expected: data: {...,"usage":{"completion_tokens_details":{"reasoning_tokens":N},...}}

# 8. Non-streaming thinking tokens (Gemini 2.5)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-pro","messages":[{"role":"user","content":"Solve: 17 * 23"}]}' \
  | jq '.usage.completion_tokens_details'
# Expected: {"reasoning_tokens": N}
```

---

## 6. Provider manual tests (OpenAI)

**Why:** Verifies the OpenAI adapter against live OpenAI API. Run after the gateway is up (docker compose or local). Requires real API credentials; not suitable for CI without secrets.

**Prerequisites:** Gateway running (port 8080 by default). Without OpenAI configured, requests to `gpt-*` models return 503. The gateway returns 503 if no configured provider handles the requested model. Restart the gateway after changing config.

**Env vars:**

| Var | Who uses it | Purpose |
|-----|-------------|---------|
| `OXIGATE__PROVIDERS__OPENAI__API_KEY` | Gateway (startup) | Your OpenAI API key; get from https://platform.openai.com/api-keys |
| `OXIGATE__AUTH__KEY` | Gateway + curl client | Bearer token the gateway enforces on `/v1/*`. Choose any value, set it in `.env`, and `export` it in your shell so curl sends it as `Authorization: Bearer $OXIGATE__AUTH__KEY`. When unset, auth is bypassed (dev/CI). |

Gateway vars: put in `.env` (docker-compose loads it automatically via `env_file`). Only set the vars for providers you actually want. Before running curl, `export OXIGATE__AUTH__KEY=<your-token>` in your shell so it is sent as the Bearer token.

```bash
# Client auth (curl needs this):
export OXIGATE__AUTH__KEY=test

# 1. Non-streaming chat
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Say hello"}]}' | jq .
# Expected: {"choices":[{"message":{"content":"Hello..."}}], "usage":{...}}

# 2. Streaming chat
curl -sN -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Count to 3"}],"stream":true}'
# Expected: data: {"choices":[{"delta":{"content":"..."}}]} lines, ending with data: [DONE]

# 3. Cost headers present
curl -s -D - -o /dev/null -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}' | grep -i X-Oxigate-Request-Cost
# Expected: X-Oxigate-Request-Cost: 0.000... (non-zero)

# 4. Invalid model → clean error (not panic)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-does-not-exist","messages":[{"role":"user","content":"Hi"}]}' | jq .error
# Expected: {"code":404,"message":"model not found: ..."}

# 5. Reasoning model (o3) — optional; requires o3 access
# curl -s -X POST http://localhost:8080/v1/chat/completions \
#   -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
#   -H "Content-Type: application/json" \
#   -d '{"model":"o3-mini","messages":[{"role":"user","content":"Solve: 17 * 23"}]}' \
#   | jq '.usage.completion_tokens_details'
# Expected: {"reasoning_tokens": N} when model supports reasoning

# 6. OpenAI embeddings
curl -s -X POST http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-3-small","input":"Hello world"}' | jq '{dim: (.data[0].embedding | length), model: .model}'
# Expected: {"dim":1536,"model":"text-embedding-3-small"}

# 6a. OpenAI embeddings with dimensions param
curl -s -X POST http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-3-small","input":"Hello world","dimensions":512}' | jq '.data[0].embedding | length'
# Expected: 512
```

---

## 7. Provider manual tests (Anthropic)

**Why:** Verifies the Anthropic Claude adapter against live Anthropic API. Run after the gateway is up (docker compose or local). Requires real API credentials; not suitable for CI without secrets..

**Prerequisites:** Gateway running (port 8080 by default). Without Anthropic configured, requests to `claude-*` models return 503. The gateway returns 503 if no configured provider handles the requested model. Restart the gateway after changing config. Anthropic requires `max_tokens` on every request; the adapter uses `default_max_tokens` (4096) when the request omits it.

**Env vars:**

| Var | Who uses it | Purpose |
|-----|-------------|---------|
| `OXIGATE__PROVIDERS__ANTHROPIC__API_KEY` | Gateway (startup) | Your Anthropic API key; get from https://console.anthropic.com/ |
| `OXIGATE__AUTH__KEY` | Gateway + curl client | Bearer token the gateway enforces on `/v1/*`. Choose any value, set it in `.env`, and `export` it in your shell so curl sends it as `Authorization: Bearer $OXIGATE__AUTH__KEY`. When unset, auth is bypassed (dev/CI). |

Gateway vars: put in `.env` (docker-compose loads it automatically via `env_file`). Only set the vars for providers you actually want. Before running curl, `export OXIGATE__AUTH__KEY=<your-token>` in your shell so it is sent as the Bearer token.

```bash
# Client auth (curl needs this):
export OXIGATE__AUTH__KEY=test

# 1. Non-streaming chat
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Say hello"}]}' | jq .
# Expected: {"choices":[{"message":{"content":"Hello..."}}], "usage":{...}}

# 2. Streaming chat
curl -sN -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Count to 3"}],"stream":true}'
# Expected: data: {"choices":[{"delta":{"content":"..."}}]} lines, ending with data: [DONE]

# 3. Tool use (function calling)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"What is the weather in London?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}]}' | jq .choices[0].message.tool_calls
# Expected: [{"function":{"name":"get_weather","arguments":"..."}}]

# 4. Cache tokens surfaced (when Claude uses prompt caching)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Hi"}]}' | jq '.usage | {cache_creation_input_tokens, cache_read_input_tokens}'
# Expected: Fields present (possibly 0) when model uses caching; omit when absent

# 5. Cost headers present
curl -s -D - -o /dev/null -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Hi"}]}' | grep -i X-Oxigate-Request-Cost
# Expected: X-Oxigate-Request-Cost: 0.000... (non-zero)

# 6. Invalid model → clean error (not panic)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-does-not-exist","messages":[{"role":"user","content":"Hi"}]}' | jq .error
# Expected: {"code":404,"message":"model not found: ..."}
```

---

## 10. Spend tracking

**Why:** Confirms a `spend_records` row is written after every completed request and that
the structured cost log line appears in gateway output with `org_id`.

**Prerequisites:** Gateway running with Postgres + Redis (docker-compose or local). Run any
provider manual test (§5–7) first to generate a request, then verify below.

```bash
# Verify the structured cost log line appears in the deployed gateway's stdout.
# Proves the tracing-subscriber is attached on the released binary path —
# a wiring concern not visible to in-process integration tests.
docker compose logs gateway 2>&1 | grep '"chat_completion_cost"' | tail -3
# Expected: JSON log lines containing: request_id, org_id, identity_id, cost_usd, latency_ms
```

`spend_records` row insertion, Redis counter increment, and Redis TTL are covered by
`tests/integration/spend_writer.rs` and verified as side effects of any provider call (§5–§7, §17, §18).


---

## 10a. Spend query API

**Why:** Verifies the three read endpoints aggregate `spend_records` correctly and enforce
tenant isolation.

**Prerequisites:** Gateway running with Postgres + Redis. Run at least one chat completion
first so spend rows exist (or seed directly via `psql`).

```bash
# Daily spend — last 30 days (default window)
curl -s http://localhost:8080/v1/spend/daily \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" | jq .
# Expected: {"data":[{"date":"YYYY-MM-DD","cost_nano_usd":<int>},...]}

# Daily spend — explicit range
curl -s "http://localhost:8080/v1/spend/daily?from=2025-01-01&to=2025-01-31" \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" | jq .

# Spend by provider
curl -s http://localhost:8080/v1/spend/providers \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" | jq .
# Expected: {"data":[{"dimension":"openai","cost_nano_usd":<int>},...]}

# Spend by model
curl -s http://localhost:8080/v1/spend/models \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" | jq .
# Expected: {"data":[{"dimension":"gpt-4.1","cost_nano_usd":<int>},...]}

# Invalid date format → 400
curl -s "http://localhost:8080/v1/spend/daily?from=not-a-date" \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY"
# Expected: {"error":"invalid date format: not-a-date"}

# Range > 365 days → 400
curl -s "http://localhost:8080/v1/spend/daily?from=2020-01-01&to=2021-12-31" \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY"
# Expected: {"error":"invalid date range: range must not exceed 365 days"}
```

---

## 11a. GlobalSafetyLayer — instance-wide cap

Community feature. Blocks all `/v1/*` requests with 429 when aggregate instance spend exceeds
`budget.global_safety_cap_usd`. Zero overhead when cap is not configured.

```bash
# 1. Start gateway with global safety cap enabled
OXIGATE__BUDGET__GLOBAL_SAFETY_CAP_USD=10.0 cargo run -- --config config/oxigate.yaml

# 2. Seed global spend above cap in Redis
redis-cli SET "oxigate:global:spend" 10000000001

# 3. Any /v1/* request should return 429 with budget cap header
curl -s -w "\n%{http_code}" http://localhost:8080/v1/models
# Expected: 429
# Expected header: X-Oxigate-Budget-Cap: global
# Expected body: {"error":"global_budget_cap_exceeded"}

# 4. Verify with verbose output
curl -sv http://localhost:8080/v1/models 2>&1 | grep -E "< HTTP|X-Oxigate-Budget-Cap|global_budget"

# 5. Reset spend below cap and verify pass-through
redis-cli SET "oxigate:global:spend" 9999999999
curl -s -w "\n%{http_code}" http://localhost:8080/v1/models
# Expected: 200

# 6. Verify SIGHUP reloads the cap (new value takes effect without restart)
# In a second terminal: update OXIGATE__BUDGET__GLOBAL_SAFETY_CAP_USD and send SIGHUP
kill -HUP $(pgrep oxigate)
# Logs should show: "Class A reload: applying config, pricing, auth, and provider"
```

---

## 12. Structured JSON logging

**Why:** Verifies log output contract and runtime log-level hot-reload behavior for operations.

**Prerequisites:** Gateway running (local or docker). For local runs, keep Postgres + Redis up per §2.

```bash
# 1. Start gateway and capture logs to a file
RUST_LOG= OXIGATE__DATABASE__URL=postgres://oxigate:changeme@localhost:5432/oxigate \
OXIGATE__REDIS__URL=redis://localhost:6379 \
cargo run -- --config config/oxigate.yaml > /tmp/oxigate-logging.log 2>&1 &

# 2. Validate startup log lines are JSON and include required keys
python3 - <<'PY'
import json
from pathlib import Path
lines = [ln for ln in Path("/tmp/oxigate-logging.log").read_text().splitlines() if ln.strip()]
for ln in lines[:10]:
    event = json.loads(ln)
    for key in ("timestamp", "level", "target", "message"):
        assert key in event, f"missing {key} in {event}"
print("JSON field contract: OK")
PY

# 3. Verify SIGHUP applies log_level change without restart
#    - update log_level in your config (e.g. warn -> info)
#    - send SIGHUP to running process
kill -HUP $(pgrep -f "oxigate --config")

# 4. Confirm reload logs are present in captured output
grep -E '"log level updated"|"SIGHUP received' /tmp/oxigate-logging.log | tail -5

# 5. Cleanup
kill -TERM $(pgrep -f "oxigate --config")
```

Expected:
- JSON parse succeeds and required top-level keys are present.
- `SIGHUP` path logs reload activity and applies the new level without process restart.

---

## 13. Observability


**Why:** Verifies a structured `"request completed"` log event is emitted for every request with required metadata fields.

```bash
# 1. Start gateway and capture logs
RUST_LOG=info OXIGATE__DATABASE__URL=postgres://oxigate:changeme@localhost:5432/oxigate \
OXIGATE__REDIS__URL=redis://localhost:6379 \
cargo run -- --config config/oxigate.yaml > /tmp/oxigate-observability.log 2>&1 &

# 2. Send a test request (any provider)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Hi"}]}'

# 3. Verify "request completed" span log exists with required fields
python3 - <<'PY'
import json
from pathlib import Path
lines = [l for l in Path("/tmp/oxigate-observability.log").read_text().splitlines() if l.strip()]
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
kill -TERM $(pgrep -f "oxigate --config")
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
for i in {1..10}; do
  curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}' > /dev/null
done

# Check provider distribution in spend_records
docker compose exec postgres psql -U oxigate oxigate \
  -c "SELECT provider_name, COUNT(*) FROM spend_records 
      WHERE created_at > NOW() - INTERVAL '1 minute' 
      GROUP BY provider_name;"
```
Expected: ~9 provider_a, ~1 provider_b (±15% noise)

2. Verify zero-weight provider exclusion (cost: ~$0.01)
Configure provider_a weight=0.0, provider_b weight=1.0
All requests should route to provider_b
```bash
for i in {1..4}; do
  curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}' > /dev/null
done

# Verify only provider_b was used
docker compose exec postgres psql -U oxigate oxigate \
  -c "SELECT DISTINCT provider_name FROM spend_records 
      WHERE created_at > NOW() - INTERVAL '1 minute';"
```
Expected: 4 provider_b (provider_a never selected)

3. Verify LowestCost strategy selects cheapest provider (cost: ~$0.002)
Configure provider_a (cheap model) and provider_b (expensive model)
Requests should route to provider_a

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-3-haiku-20240307","messages":[{"role":"user","content":"Hi"}]}' > /dev/null

docker compose exec postgres psql -U oxigate oxigate \
  -c "SELECT provider_name FROM spend_records 
      WHERE created_at > NOW() - INTERVAL '1 minute' 
      ORDER BY created_at DESC LIMIT 1;"
```

Expected: "provider_a" (lower cost per pricing DB)


## 15. Prometheus metrics

**Why:** Verifies the `/metrics` scrape endpoint is reachable, unauthenticated, returns valid
Prometheus text, and exposes the required baseline and fallback/retry metric families.

**Prerequisites:** Gateway running (§1 docker-compose or §2 local). Run at least one provider
request first (§5, §6, or §7) to populate counters — a cold gateway still returns 200 with
zero-valued metrics, but request counters will only appear after traffic flows.

```bash
# 1. Endpoint reachable and returns 200 — proves /metrics route registered in deployed binary.
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/metrics
# Expected: 200

# 2. Auth bypass: /metrics returns 200 even when auth is configured for /v1/*
#    Proves /metrics is on the outer router, before the auth layer — a deployment-wiring concern
#    (run this after restarting with OXIGATE__AUTH__KEY=<token> set — see §2 for the restart pattern).
curl -s -o /dev/null -w "%{http_code}" http://localhost:19999/metrics
# Expected: 200 (not 401)
curl -s -o /dev/null -w "%{http_code}" http://localhost:19999/v1/nonexistent
# Expected: 401 — /v1/* is protected
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

**Prerequisites:** Postgres + Redis running (§2). LiteLLM installed (`pip install litellm`). A
provider API key for any LiteLLM-supported provider (example below uses Groq — free tier).

```bash
# ── 1. Start LiteLLM proxy ─────────────────────────────────────────────────
export GROQ_API_KEY="your-groq-key"
litellm --model groq/llama-3.1-8b-instant --port 4000

# ── 2. Configure OxiGate to point at LiteLLM ──────────────────────────────
export OXIGATE__PROVIDERS__OPENAI__API_BASE_URL=http://localhost:4000
export OXIGATE__PROVIDERS__OPENAI__API_KEY=proxy
export OXIGATE__AUTH__KEY=test

# ── 3. Non-streaming — verify response and cost headers ────────────────────
curl -s -D - -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"groq/llama-3.1-8b-instant","messages":[{"role":"user","content":"Say hi"}]}' \
  | tee /tmp/litellm_smoke.txt
# Expected: HTTP 200; OpenAI-format body; X-Oxigate-Cost-* headers with non-zero values

grep -i "x-oxigate" /tmp/litellm_smoke.txt

# ── 4. Streaming ───────────────────────────────────────────────────────────
curl -s -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"groq/llama-3.1-8b-instant","messages":[{"role":"user","content":"Count to 3"}],"stream":true}'
# Expected: SSE chunks followed by data: [DONE]

# ── 5. FinOps accuracy spot-check ─────────────────────────────────────────
# Compare X-Oxigate-Cost-Input-Tokens against LiteLLM logs for the same request.
# They should match exactly for Groq (provider-reported passthrough).
```

---

## 17. AWS Bedrock adapter

**Why:** Verifies SigV4 signing, EventStream streaming, Claude model routing, and cost headers

**Prerequisites:** AWS credentials with `bedrock:InvokeModel`, `bedrock:Converse`, and
`bedrock:ConverseStream` permissions for `anthropic.*` model IDs in the configured region.

```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
export OXIGATE__PROVIDERS__BEDROCK__REGION=us-east-1
export OXIGATE__AUTH__KEY=test

# ── Non-streaming ──────────────────────────────────────────────────────────
curl -s -D - -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic.claude-3-5-sonnet-20241022-v2:0","messages":[{"role":"user","content":"Say hi"}]}' \
  | tee /tmp/bedrock_smoke.txt
# Expected: HTTP 200; OpenAI-format JSON; X-Oxigate-Cost-* headers with non-zero values

# ── Streaming ──────────────────────────────────────────────────────────────
curl -s -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic.claude-3-5-sonnet-20241022-v2:0","messages":[{"role":"user","content":"Count to 3"}],"stream":true}'
# Expected: OpenAI SSE chunks followed by data: [DONE]
```

Unknown-model-prefix dispatch is OxiGate-internal logic and is covered by
`tests/integration/providers/bedrock.rs` + adapter unknown-prefix handling.

---

## 18. Azure OpenAI adapter

**Why:** Verifies deployment-based URL construction, `api-key` header auth, `stream_options` injection,

**Prerequisites:** An Azure OpenAI resource with a deployed model (e.g. `gpt-4o`). Use `api_version: "2024-10-21"`.

```bash
export AZURE_ENDPOINT="https://my-resource.openai.azure.com"
export AZURE_DEPLOYMENT="gpt-4o"
export AZURE_API_KEY="..."
export OXIGATE__AUTH__KEY=test

# ── Non-streaming ──────────────────────────────────────────────────────────
curl -s -D - -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Say hi"}]}' \
  | tee /tmp/azure_smoke.txt
# Expected: HTTP 200; OpenAI-format JSON; X-Oxigate-Cost-* headers with non-zero values

# ── Streaming ──────────────────────────────────────────────────────────────
curl -s -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Count to 3"}],"stream":true}'
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
| **401 Unauthorized on /v1/*** | `OXIGATE__AUTH__KEY` is set on the gateway and the curl command is missing or using the wrong Bearer token. Set `export OXIGATE__AUTH__KEY=<your-token>` (the value configured on the gateway) and add `-H "Authorization: Bearer $OXIGATE__AUTH__KEY"` to your curl. Health routes (`/health`, `/health/ready`) never require auth. |
| **401 Unauthorized on /health** | Auth layer is incorrectly applied to health routes — Health routes must be on the top-level Router, not the `/v1/` sub-router. |
| **startup says python-bridge / feature mismatch** | Expected when YAML enables `providers.python_bridge` but the binary was built without `--features python-bridge`. Rebuild with `cargo build --release --features python-bridge` or remove/disable the bridge section. |
| **503 Python bridge unavailable** | LiteLLM missing from `venv_path`, wrong venv, or import failure. Install `litellm` in that venv; confirm `GROQ_API_KEY` for live Groq calls. |
| **Migration checksum mismatch after squash** | Run `docker compose down -v` then `docker compose up -d --build`. Migration 0001 was replaced with a new squashed file — existing DB state is incompatible with the new checksum. |
| **X-Oxigate-Request-Cost missing** (step 5) | Run without grep to see full response: `curl -s -D - -o /dev/null -X POST ...` — if 401, check the exported `OXIGATE__AUTH__KEY` matches the value configured on the gateway; if 200, headers should include X-Oxigate-Request-Cost. |
