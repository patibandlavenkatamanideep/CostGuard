# CostGuard

> **Open-source LLM evaluation proxy.** Unlike LiteLLM, Helicone, or Portkey, CostGuard's reliability layer is grounded in [RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench) — 1,412+ empirical runs across 39 tasks and 12 models. You get RDAB-calibrated validity scoring on every proxy call, plus retries, circuit breakers, cost tracking, and alerting.
>
> CostGuard is the runtime layer of [The Evaluation Stack](https://github.com/patibandlavenkatamanideep/RealDataAgentBench): [RDAB](https://github.com/patibandlavenkatamanideep/RealDataAgentBench) (benchmark methodology) → CostGuard (runtime enforcement) → [Tether](https://github.com/patibandlavenkatamanideep/Tether) (trace capture). See [How This Fits With RDAB and Tether](#how-this-fits-with-rdab-and-tether) for the full architecture.

[![CI/CD](https://github.com/patibandlavenkatamanideep/CostGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/patibandlavenkatamanideep/CostGuard/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Powered by RDAB](https://img.shields.io/badge/Evaluation-RealDataAgentBench-7c3aed)](https://github.com/patibandlavenkatamanideep/RealDataAgentBench)
---

## Demo

<video src="https://github.com/user-attachments/assets/f82997cd-c2f8-4298-b716-4b7810ced97c" controls width="100%"></video>

---

## What Is CostGuard?

**RDAB-calibrated validity scoring on every proxy call** — that's the differentiator. Most LLM proxies gate on latency and error codes. CostGuard gates on response quality using [RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench) empirical data: 1,412+ runs across 39 tasks and 12 frontier models.

CostGuard is a self-hostable LLM proxy with two headline capabilities: `/proxy` — a fast reliability filter in your hot path (~1ms validity overhead) — and `/evaluate` — full RDAB dataset benchmarking that returns four-dimensional scorecards and cost-weighted model rankings. Add `/replay` to close the loop: replay captured Tether production traces against any alternate model and get a 95% bootstrap CI on the quality delta.

- **Dataset benchmarking** — powered by [RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench). Upload any CSV/Parquet and get a cost-weighted model ranking grounded in real benchmark data. The key RDAB finding: every frontier model scores ≤0.25 on statistical validity even when scoring 0.83–1.00 on correctness — correct-looking output ≠ sound reasoning.
- **Real-time response filtering** — every `/proxy` call is scored with a RDAB-calibrated heuristic validator. Responses below your threshold are rejected automatically.
- **Automatic fallback** — on rejection, CostGuard retries with the next model in your fallback chain.
- **Exact cost tracking** — per-call token accounting at $0.000001 precision across 12 models and 5 providers.
- **Alerting** — validity drops, cost spikes, high failure rates, circuit breaker events, and consecutive rejections routed to Slack or any webhook.
- **Per-provider circuit breakers** — stops hammering a failing provider during an outage; state persists across restarts.

**Who this is for:** teams running LangGraph, CrewAI, or custom LLM agents who want reliability guarantees backed by actual benchmark data, not just latency-based health checks.

---

## How This Fits With RDAB and Tether

CostGuard is the runtime layer of The Evaluation Stack. Each project has a distinct job:

- **[RealDataAgentBench (RDAB)](https://github.com/patibandlavenkatamanideep/RealDataAgentBench)** — the benchmark methodology. 39 tasks, 4-dimensional scoring (correctness, code quality, efficiency, statistical validity), 1,412+ runs. Produces the empirical scorecards that CostGuard uses.
- **CostGuard (this repo)** — the runtime enforcement layer. Applies RDAB-calibrated scoring in the `/proxy` hot path, runs full RDAB evaluations via `/evaluate`, and adds circuit breakers, alerting, and Prometheus observability.
- **[Tether](https://github.com/patibandlavenkatamanideep/Tether)** — the trace capture layer (separate repo). Wraps OpenAI/Anthropic clients and persists every production call to SQLite. CostGuard's `POST /replay` endpoint reads those traces directly and replays them against any alternate model, returning RDAB quality deltas with a 95% bootstrap CI.

```
Production agent traffic
        │
        ▼
  ┌──────────┐   captures every call   ┌──────────────────────┐
  │  Tether  │ ──────────────────────► │ SQLite trace store   │
  └──────────┘                         └──────────┬───────────┘
                                                   │ POST /replay  ← live
                                                   ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  CostGuard /replay  →  RDAB quality delta  →  bootstrap CI │
  │  cost savings estimate against any alternate model          │
  └─────────────────────────────────────────────────────────────┘
```

Together, the stack enables something no single repo does alone: replay-based cost-routing recommendations with RDAB statistical confidence intervals against your actual production traffic — not synthetic benchmarks.

---

## Replay Production Traces

This is the unique cross-project capability. [Tether](https://github.com/patibandlavenkatamanideep/Tether) wraps your OpenAI client and captures every production call to SQLite. `POST /replay` reads that database directly and replays every prompt against any alternate model — no Tether package dependency on this side.

```bash
# 1. Capture 25 calls with Tether (see Tether README for setup)
# 2. Pass the db path + run_id to CostGuard:
curl -X POST http://localhost:8000/replay \
  -H "Content-Type: application/json" \
  -d '{
    "tether_db_path": "/tmp/tether.db",
    "run_id": "<uuid from TetheredOpenAI.run_id>",
    "alternate_model": "gpt-4.1-mini",
    "n_bootstrap_samples": 1000
  }'
```

```json
{
  "primary_model": "gpt-4o-mini",
  "alternate_model": "gpt-4.1-mini",
  "n_calls": 25,
  "delta": -0.0060,
  "ci_low": -0.0312,
  "ci_high": 0.0192,
  "primary_cost_usd": 0.00048312,
  "alternate_cost_usd": 0.00031205,
  "savings_per_call_usd": 0.00000684
}
```

A CI that straddles 0 means the quality difference is not statistically significant — the cheaper model is a safe swap. See [`scripts/demo_replay.py`](scripts/demo_replay.py) for a full end-to-end demo.

---

## How Validity Scoring Works (Be Honest With Yourself)

CostGuard has **two validity modes** — understanding the difference matters:

| Mode | Endpoint | How It Works | Latency |
|------|----------|-------------|---------|
| **Heuristic** | `POST /proxy` | RDAB-calibrated keyword scorer (~1ms) | ~1ms overhead |
| **Full RDAB** | `POST /evaluate` | Actual RDAB agent evaluation with dataset-grounded questions | 15s–3min |

The `/proxy` endpoint uses a fast heuristic scorer: it rewards statistical markers (p-values, confidence intervals, uncertainty quantification) and penalizes failure-mode phrases ("I cannot", "I don't know", error tracebacks, empty outputs). It is **not** a full LLM evaluation — it's a practical pre-filter you can run synchronously on every call without adding meaningful latency.

**What it catches:** broken responses, refusals, empty output, obvious errors. **What it misses:** fluent, confident, statistically unsound analysis — the most common failure mode in RDAB, and the one that matters most. A model generating plausible-sounding confidence intervals with the wrong methodology will typically pass the heuristic filter at any threshold.

The `/evaluate` endpoint runs actual [RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench) evaluations grounded in your uploaded dataset, returning four-dimensional RDAB scorecards from 1,412+ benchmark runs across 12 models. This is the right tool for catching the statistical validity gap.

If you need true response quality assurance: use `/evaluate` for batch benchmarking and use `/proxy` as a fast sanity filter in your hot path. They solve different problems.

---

## Run It Yourself

A read-only live demo runs at [costguard-production-3afa.up.railway.app](https://costguard-production-3afa.up.railway.app/) — no API keys stored, evaluation uses simulation mode. For full live-mode access with your own keys and data, self-host using the instructions below.

---

## Self-Host in One Command

```bash
git clone https://github.com/patibandlavenkatamanideep/CostGuard.git && cd CostGuard
cp .env.example .env   # add at least one provider API key
docker compose up
```

- Dashboard → **http://localhost:8501**
- API + Proxy → **http://localhost:8000**
- API Docs → **http://localhost:8000/docs**
- Prometheus Metrics → **http://localhost:8000/metrics**

With Grafana monitoring:
```bash
docker compose --profile monitoring up
# Grafana → http://localhost:3000 (admin / costguard)
```

---

## Architecture

```
Your Agent / LangGraph / CrewAI
        │
        ▼
┌─────────────────────────────────────────────┐
│          CostGuard Middleware Stack          │
│                                             │
│  RequestID → RateLimit → Security →         │
│  Prometheus (every request)                 │
└──────────────┬──────────────────────────────┘
               │
      ┌────────▼────────────────────────────┐
      │         POST /proxy                  │
      │  1. Circuit breaker check            │
      │  2. LLM call (30s timeout)           │
      │  3. Heuristic validity score (~1ms)  │
      │  4. Reject + fallback if score < T   │
      │  5. Cost calculation                 │
      │  6. Async alert checks               │
      │  7. SQLite log (thread pool)         │
      └────────────────────────────────────-─┘
               │
      ┌────────▼────────────────────────────┐
      │      Per-Provider Circuit Breaker   │
      │  CLOSED → (5 failures) → OPEN       │
      │  OPEN   → (60s timeout) → HALF_OPEN │
      │  HALF_OPEN → (2 successes) → CLOSED │
      └────────────────────────────────────-┘
               │
      ┌────────▼────────────────────────────┐
      │     LLM Providers                   │
      │  anthropic | openai | groq |        │
      │  google | xai                       │
      └────────────────────────────────────-┘

      ┌─────────────────────────────────────┐
      │         POST /evaluate              │
      │  Full RDAB benchmarking pipeline    │
      │  (dataset upload → question gen →   │
      │   agent eval → cost-weighted rank)  │
      └────────────────────────────────────-┘
```

---

## The Proxy — HTTP Guard Layer

CostGuard is an **HTTP proxy**, not an SDK wrapper. Using it requires replacing direct LLM SDK calls with HTTP calls to the `/proxy` endpoint — typically a one-site change if your agent centralizes its LLM calls. Cross-provider tool-call format translation is not handled: if your agent uses function calling, falling back mid-session from Claude to GPT-4.1 will produce format mismatches. `/proxy` fallback is designed for text completion tasks.

### Before (no reliability layer)
```python
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-...")
response = client.messages.create(model="claude-sonnet-4-6", ...)
# No validity check. No cost tracking. No fallback.
```

### After (with CostGuard)
```python
import httpx

response = httpx.post("http://costguard:8000/proxy", json={
    "model_id": "claude-sonnet-4-6",
    "prompt": "Analyze Q3 revenue trends and compute 95% confidence intervals.",
    "reject_threshold": 0.30,
    "fallback_models": ["gpt-4.1", "gemini-2.5-flash"],
}).json()

print(response["content"])           # the LLM's response
print(response["accepted"])          # True / False
print(response["validity_score"])    # heuristic scorecard
print(response["cost_usd"])          # exact cost for this call
print(response["fallback_used"])     # True if primary was rejected
```

### Proxy Response Schema
```json
{
  "call_id": "a3f9e1b2c4d5",
  "model_id": "claude-sonnet-4-6",
  "provider": "anthropic",
  "content": "The 95% confidence interval for Q3 revenue is...",
  "accepted": true,
  "rejection_reason": null,
  "fallback_used": false,
  "validity_score": {
    "rdab_score": 0.742,
    "correctness": 0.75,
    "code_quality": 0.70,
    "efficiency": 0.85,
    "stat_validity": 0.45,
    "simulated": true
  },
  "latency_ms": 843.2,
  "input_tokens": 1247,
  "output_tokens": 312,
  "cost_usd": 0.00000851,
  "attempts": 1,
  "circuit_breaker_state": "closed"
}
```

> **Note:** `validity_score.simulated: true` indicates the score came from the heuristic proxy scorer, not a full RDAB evaluation. This is expected for `/proxy` — the fast path.

---

## Dataset Benchmarking (POST /evaluate)

Upload any CSV or Parquet file. CostGuard generates dataset-grounded questions, runs them through all available models using RealDataAgentBench, and returns a ranked recommendation with exact cost estimates.

**Two modes:**
- **Simulation mode** (no API keys): returns calibrated scores from 1,412+ RDAB benchmark runs. Deterministic — same file always produces the same ranking.
- **Live mode** (with API keys): runs real RDAB agent evaluations against your actual dataset.

```bash
curl -X POST http://localhost:8000/evaluate \
  -F "file=@my_data.csv" \
  -F "task_description=Analyze customer churn patterns" \
  -F "num_questions=5"
```

---

## Alerting

Six alert types, all configurable via environment variables.

| Alert Type | Trigger | Default Threshold |
|-----------|---------|------------------|
| `ValidityThreshold` | Response validity below threshold | 0.25 |
| `CostSpike` | Single call cost > N× rolling average | 3× |
| `HighFailureRate` | >N% of recent calls failed | 20% |
| `ConsecutiveLowValidity` | N consecutive rejections from same model | 3 |
| `CircuitBreakerOpen` | Provider circuit breaker opened | — |
| `RateLimit` | 429 response from provider | — |

Alerts fire to console always. Add channels via environment variables:

```bash
# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../...

# Generic webhook (PagerDuty, OpsGenie, custom)
COSTGUARD_ALERT_WEBHOOK_URL=https://your-webhook.example.com/alerts
```

---

## Monitoring

Prometheus `/metrics` endpoint with Grafana dashboard included.

| Metric | Type | Description |
|--------|------|-------------|
| `costguard_proxy_requests_total` | Counter | Proxy calls by model, provider, status |
| `costguard_proxy_latency_seconds` | Histogram | LLM call latency per model |
| `costguard_proxy_rejections_total` | Counter | Responses rejected below threshold |
| `costguard_proxy_fallbacks_total` | Counter | Fallbacks triggered |
| `costguard_eval_requests_total` | Counter | Dataset evaluations by mode and status |
| `costguard_circuit_breaker_open` | Gauge | 1 if circuit breaker open for a provider |
| `costguard_alerts_fired_total` | Counter | Alerts fired by type and channel |
| `costguard_api_request_duration_seconds` | Histogram | API request latency |

---

## RDAB Scoring Methodology

CostGuard uses [RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench) — **1,412+ evaluation runs across 39 tasks and 12 models** — for the `/evaluate` endpoint.

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| **Correctness** | 50% | Answer accuracy vs ground truth (fuzzy-matched, ±15% tolerance) |
| **Code Quality** | 20% | Vectorised operations, naming conventions, no magic numbers |
| **Efficiency** | 15% | Token + step budget adherence |
| **Stat Validity** | 15% | Reports p-values, confidence intervals, avoids overconfident claims |

**Key RDAB Benchmark Findings (1,412+ runs · 39 tasks · 12 models):**
- **GPT-4.1** = top composite score at $0.013/task — best quality-per-dollar
- **Gemini 2.5 Flash** = cheapest at $0.0015/task; only 20.6% below top score
- **Stat validity gap**: model average 55.8% vs human expert baseline 81.3%

### Ranking Formula
```
composite = rdab_score × 0.75 + cost_score × 0.25
cost_score = 1 − sqrt(model_cost / max_cost_in_cohort)
```

---

## Supported Models (12 RDAB-benchmarked)

| Model | Provider | Tier | Input $/1K | Context |
|-------|----------|------|-----------|---------|
| Claude Sonnet 4.6 | Anthropic | Premium | $0.003 | 200K |
| Claude Opus 4.6 | Anthropic | Premium | $0.015 | 200K |
| Claude Haiku 4.5 | Anthropic | Economy | $0.00025 | 200K |
| **GPT-4.1** | OpenAI | Premium | **$0.002** | 1M |
| GPT-4.1 mini | OpenAI | Balanced | $0.0004 | 1M |
| GPT-4.1 nano | OpenAI | Economy | $0.0001 | 1M |
| GPT-4o | OpenAI | Premium | $0.0025 | 128K |
| GPT-4o mini | OpenAI | Balanced | $0.00015 | 128K |
| GPT-5 | OpenAI | Premium | $0.015 | 128K |
| **Gemini 2.5 Flash** | Google | Economy | **$0.000075** | 1M |
| Llama 3.3 70B (Groq) | Groq | Balanced | $0.00059 | 128K |
| Grok-3 mini | xAI | Balanced | $0.0003 | 131K |

---

## Quickstart (Local, No Docker)

```bash
git clone https://github.com/patibandlavenkatamanideep/CostGuard.git
cd CostGuard
cp .env.example .env    # add at least one provider API key
pip install -e .
./scripts/dev.sh
```

- Dashboard: **http://localhost:8501**
- API Docs: **http://localhost:8000/docs**

---

## Deploy

### Environment variables (all platforms)

| Variable | Required | Notes |
|---|---|---|
| `SECRET_KEY` | Yes | `openssl rand -hex 32` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GROQ_API_KEY` | At least one for Live Mode | Omit for Simulation Mode |
| `COSTGUARD_DB_PATH` | Recommended | Set to a persistent volume path (see per-platform notes) |
| `SLACK_WEBHOOK_URL` | No | Enables Slack alerting |
| `COSTGUARD_STATE_BACKEND` | No | `sqlite` (default) or `none` |

---

### Option 1 — Railway (recommended — always-on, persistent storage)

Config is already in [`railway.json`](railway.json). Connect your GitHub repo in the Railway dashboard and deploy in one click.

```bash
# After deploying, add secrets in Railway Dashboard → Variables
SECRET_KEY=<openssl rand -hex 32>
OPENAI_API_KEY=sk-...   # at least one provider
COSTGUARD_DB_PATH=/data/costguard_history.db
```

The live demo runs on Railway at [costguard-production-3afa.up.railway.app](https://costguard-production-3afa.up.railway.app/).

---

For Koyeb and Hugging Face Spaces deployment, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

### Option 2 — Self-host with Docker Compose

```bash
git clone https://github.com/patibandlavenkatamanideep/CostGuard.git
cd CostGuard
cp .env.example .env    # add SECRET_KEY and at least one provider key
docker compose up -d    # SQLite stored in named volume costguard-data

# With monitoring (Prometheus + Grafana on :3000)
docker compose --profile monitoring up -d
```

SQLite persists in the `costguard-data` Docker named volume across restarts and image updates.

---

### Platform comparison

| Platform | RAM | Sleep? | Persistent disk? | Config needed |
|----------|-----|--------|-----------------|---------------|
| **Railway** | 512 MB+ | No | Yes | `railway.json` ✅ ready |
| **Self-host** | Unlimited | No | Yes | `.env` only |

More options (Koyeb, Hugging Face Spaces) in [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Project Structure

```
costguard/
├── backend/
│   ├── main.py            # FastAPI app — routes, middleware wiring
│   ├── proxy.py           # LLM proxy + heuristic validator + auto-fallback
│   ├── alerting.py        # Alert engine (6 types, Slack + webhook channels)
│   ├── metrics.py         # Prometheus metrics + OpenTelemetry setup
│   ├── middleware.py      # RequestID + LRU-bounded RateLimit + SecurityHeaders
│   ├── circuit_breaker.py # Per-provider circuit breaker (CLOSED/OPEN/HALF_OPEN)
│   ├── config.py          # Pydantic settings (env var management)
│   ├── models.py          # Request/response schemas
│   └── logger.py          # Structured logging (loguru)
├── evaluation/
│   ├── engine.py          # RDAB evaluation orchestrator (live + simulation)
│   ├── observability.py   # SQLite logging + drift detection (WAL mode)
│   ├── data_loader.py     # CSV/Parquet ingestion (multi-encoding robust)
│   ├── pricing.py         # 12-model pricing catalogue
│   ├── tether_reader.py   # Read-only Tether SQLite reader (zero Tether dependency)
│   ├── question_generator.py
│   └── token_counter.py
├── frontend/
│   └── app.py             # Streamlit dashboard
├── deploy/
│   ├── prometheus.yml     # Prometheus scrape config
│   └── grafana/           # Grafana dashboard + datasource provisioning
├── tests/
│   ├── test_evaluation.py # Evaluation + pricing + data loader tests
│   ├── test_proxy.py      # Proxy + CB + alerting + retry + persistence tests (73 tests)
│   └── locustfile.py      # Load test — finds RPS ceiling (locust -f tests/locustfile.py)
├── scripts/
│   ├── demo_replay.py     # End-to-end Tether→CostGuard demo (25 calls, prints Exhibit A)
│   ├── dev.sh
│   └── start.sh
├── .dockerignore
├── docker-compose.yml     # Named volume + optional monitoring profile
├── Dockerfile             # Multi-stage build (builder + non-root runtime)
├── railway.json           # Railway deployment config
└── pyproject.toml
```

---

## API Reference

Full docs at `/docs` (Swagger) and `/redoc`.

### POST `/proxy` — Real-time LLM guard
```bash
curl -X POST http://localhost:8000/proxy \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt-4.1",
    "prompt": "Analyze revenue trends",
    "reject_threshold": 0.30,
    "fallback_models": ["claude-sonnet-4-6"]
  }'
```

### POST `/evaluate` — Full RDAB dataset benchmarking
```bash
curl -X POST http://localhost:8000/evaluate \
  -F "file=@my_data.csv" \
  -F "task_description=Analyze churn patterns"
```

### POST `/replay` — Replay a Tether run against an alternate model

See [Replay Production Traces](#replay-production-traces) above for the full example and response schema.

### GET `/health` — Deep health check
```bash
curl http://localhost:8000/health
# {"status":"ok","db_ok":true,"rdab_available":true,"circuit_breakers":{...}}
```

### GET `/metrics` — Prometheus scrape
```bash
curl http://localhost:8000/metrics
```

### GET `/proxy/status` — Circuit breaker states
```bash
curl http://localhost:8000/proxy/status
```

---

## Development

```bash
pytest tests/ -v
ruff check . && ruff format .
mypy backend/ evaluation/
```

---

## Known Limitations

- **Proxy validity scoring is heuristic-only** — the `/proxy` fast path rewards statistical keywords and penalizes failure phrases. It does not catch fluent-but-statistically-unsound responses. Use `/evaluate` for real quality assessment.
- **`reject_threshold` is uncalibrated** — 0.30 is a reasonable default for catching broken responses; it is not empirically derived from a precision/recall curve. Calibrate to your workload before treating it as a quality gate.
- **Tool-call format is not translated across providers** — CostGuard passes your prompt as-is to each provider. Fallback from Claude to GPT-4.1 or Gemini in an agent loop using function calling will produce format mismatches. `/proxy` fallback works reliably for text completion tasks only.
- **Rate limit state is in-memory** — IP-based rate limit buckets reset on server restart. Fine for most deployments.
- **SQLite for single-node persistence** — appropriate for self-hosted single-node deployments. Circuit breaker and alerting state survives process restarts via the `runtime_state` SQLite table. For multi-node or high-throughput deployments, migrate the state store to Redis and replace `observability.py` with PostgreSQL + asyncpg connection pool. Set `COSTGUARD_STATE_BACKEND=none` to disable persistence entirely.

---

## Engineering Checklist

| Component | Status | Notes |
|-----------|--------|-------|
| LLM proxy with auto-reject + fallback | ✅ Complete | `POST /proxy` — RDAB-calibrated heuristic scorer |
| Per-provider circuit breaker | ✅ Complete | CLOSED/OPEN/HALF_OPEN; state persists across restarts |
| 6 alert types with cooldown | ✅ Complete | Slack + generic webhook; cooldowns persist across restarts |
| Tether replay (`POST /replay`) | ✅ Complete | Reads Tether SQLite, replays against alternate model, 95% bootstrap CI |
| Prometheus metrics (13 metrics) + Grafana | ✅ Complete | Auto-provisioned dashboard via `--profile monitoring` |
| OpenTelemetry traces | ✅ Complete | Opt-in via `OTEL_EXPORTER_OTLP_ENDPOINT` |
| CI security scanning | ✅ Complete | Bandit + Trivy + pip-audit in GitHub Actions |
| Proxy unit tests (73 total) | ✅ Complete | CB, alerting, persistence, retry, middleware, scorer |
| Replay unit tests (17 total) | ✅ Complete | tether_reader (5), bootstrap CI (5), endpoint (7) |
| Load test | ✅ Complete | `locust -f tests/locustfile.py` — finds RPS ceiling |
| Multi-replica CB/alert state | ⚠️ Known limitation | Requires Redis for shared state across replicas |
| Observability store (multi-node) | ⚠️ Known limitation | SQLite single-node only; replace with PostgreSQL + asyncpg |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md). Report vulnerabilities via [GitHub Security Advisories](https://github.com/patibandlavenkatamanideep/CostGuard/security/advisories/new) — do not open public issues.

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Venkata Manideep Patibandla](https://venkatamanideep.com) · [LinkedIn](https://linkedin.com/in/manideep-analytics) · [GitHub](https://github.com/patibandlavenkatamanideep)

Part of The Evaluation Stack: [RDAB](https://github.com/patibandlavenkatamanideep/RealDataAgentBench) · [CostGuard](https://github.com/patibandlavenkatamanideep/CostGuard) · [Tether](https://github.com/patibandlavenkatamanideep/Tether)
