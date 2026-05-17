# OxiGate

**FinOps for LLMs.** A high-performance, open-source LLM gateway written in Rust — with hard budget enforcement, cost attribution, and spend analytics built into the request path.

Drop-in OpenAI-compatible proxy. Route to any provider, track every dollar, enforce budgets before they blow.

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              OxiGate Gateway            │
                        │                                         │
 OpenAI-compatible ────►│  Auth → Budget → Cost → Tag → Route     │────► OpenAI
 client (any SDK)       │                                         │────► Anthropic
                        │  GlobalSafetyLayer (instance-wide cap)  │────► Gemini
                        │  HardCapLayer     (per-identity cap)    │────► Bedrock
                        │  BudgetLayer      (soft alerts)         │────► Azure OpenAI
                        │  TaggerLayer      (team / project)      │────► any OpenAI-compat
                        │  CostLayer        (nano-USD accounting) │
                        │  RouterLayer      (strategy + fallback) │
                        │                                         │
                        │  PostgreSQL ◄── spend_records           │
                        │  Redis      ◄── atomic spend counters   │
                        └─────────────────────────────────────────┘
```

Tower middleware pipeline. Every request passes through auth, budget check, cost
accounting, tagging, and routing — in microseconds.

---

## Provider status

| Provider | Chat | Streaming | Tool use | Embeddings | Cost tracking |
|---|---|---|---|---|---|
| OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | ✅ | n/a | ✅ (cache pricing) |
| Google Gemini / Vertex | ✅ | ✅ | ✅ | ✅ | ✅ (tiered + thinking) |
| AWS Bedrock (Converse) | ✅ | ✅ | ✅ | planned | ✅ (per-region) |
| Azure OpenAI | ✅ | ✅ | planned | planned | ✅ |
| OpenAI-compatible (DeepSeek, Groq, Together AI, Mistral, …) | ✅ | ✅ | varies | — | ✅ |

## Feature status

| Feature | Status |
|---|---|
| Hard + soft budget caps (per-identity) | ✅ |
| Instance-wide safety cap (GlobalSafetyLayer) | ✅ |
| Per-team and per-tag budget enforcement | ✅ |
| Budget reset — daily / weekly / monthly | ✅ |
| Progressive alerts (80 / 90 / 100%) | ✅ |
| Response cost headers (`X-Oxigate-*`) | ✅ |
| Spend tracking — Redis counters + PostgreSQL audit | ✅ |
| Spend query API (`GET /v1/spend/*`) | ✅ |
| Request tagging (`X-OxiGate-Team`, `X-OxiGate-Project`) | ✅ |
| Prometheus metrics (`/metrics`) | ✅ |
| Structured JSON logging | ✅ |
| Load balancing — weighted-random, rate-limit-aware, lowest-cost | ✅ |
| Fallback + retry with exponential backoff | ✅ |
| Config-based auth (single Bearer token) | ✅ |
| YAML config + env var overrides + SIGHUP hot-reload | ✅ |
| Health checks (`/health/ready`, `/health/live`) | ✅ |
| Per-key API key management | planned |
| Rate limiting (RPM / TPM) | planned |
| Plugin system (Rust `.so` + Python) | planned |
| OpenTelemetry OTLP export | planned (Pro) |

---

## Quickstart

### Docker Compose (recommended)

Starts the gateway, PostgreSQL, and Redis in one command.

```bash
git clone https://github.com/getoxigate/oxigate.git
cd oxigate

# Copy the example env (provider keys are optional — gateway starts without them
# and returns 503 for model requests until at least one provider is configured)
cp .env.example .env
# edit .env — uncomment and set the providers you want, e.g.:
#   OXIGATE__PROVIDERS__OPENAI__API_KEY=sk-...

# Build the image and start the full stack
docker compose up -d --build
```

The gateway is now listening on `http://localhost:8080`.

> **Tip:** To wipe all data and start fresh (e.g. after a config change that affects the DB
> schema), run `docker compose down -v` before `docker compose up -d --build`. The `-v` flag
> removes the Postgres volume so migrations run from scratch.

Test it:

```bash
curl http://localhost:8080/health/ready
# {"status":"ok"}

# Export the token you set as OXIGATE__AUTH__KEY in .env so curl can send it:
export OXIGATE__AUTH__KEY=your-secret-token

curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE__AUTH__KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

Cost headers appear on every response:

```
X-Oxigate-Request-Cost: 0.000003
X-Oxigate-Input-Tokens: 10
X-Oxigate-Output-Tokens: 25
X-Oxigate-Model-Used: gpt-4o-mini
```

### Build from source

Requires Rust stable, PostgreSQL, and Redis.

```bash
git clone https://github.com/getoxigate/oxigate.git
cd oxigate

cp .env.example .env
# edit .env — set provider keys; DB/Redis URLs default to localhost (see config/oxigate.yaml)

# Start the gateway — database migrations run automatically on startup
cargo run --release --bin oxigate -- --config config/oxigate.yaml
```

---

## Configuration

The gateway is configured via `config/oxigate.yaml` with environment variable overrides.

```yaml
server:
  port: 8080

auth:
  key: "your-secret-token"            # or set OXIGATE__AUTH__KEY

providers:
  openai:
    - name: openai-main
      api_key: "${OPENAI_API_KEY}"

budget:
  global_safety_cap_usd: 100.0        # instance-wide hard stop
```

See [`config/oxigate.yaml`](config/oxigate.yaml) for the full annotated reference and
[`docs/`](docs/) for provider-specific guides.

---

## Community and safety

Contribution workflow and the Contributor License Agreement (CLA) process are documented in [`CONTRIBUTING.md`](CONTRIBUTING.md). Report vulnerabilities privately using [`SECURITY.md`](SECURITY.md) — please do **not** file security reports as public GitHub issues.

Operational smoke checks (`docker compose`, local `cargo`, live provider spot tests) live in [`docs/smoke-tests.md`](docs/smoke-tests.md).

---

## License

Licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).

---

[oxigate.com](https://oxigate.com) · [info@oxigate.com](mailto:info@oxigate.com)
