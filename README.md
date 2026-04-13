# CostGuard

> **Stop guessing which LLM to use. CostGuard benchmarks 15 models against your actual data in under 15 seconds — and tells you exactly what it will cost.**

[![CI/CD](https://github.com/patibandlavenkatamanideep/CostGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/patibandlavenkatamanideep/CostGuard/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Powered by RDAB](https://img.shields.io/badge/Evaluation-RealDataAgentBench-7c3aed)](https://github.com/patibandlavenkatamanideep/RealDataAgentBench)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20Now%20%E2%86%92-brightgreen)](https://costguard.up.railway.app)

---

<p align="center">
  <a href="https://costguard.up.railway.app">
    <img src="https://img.shields.io/badge/%E2%9A%A1%20Try%20it%20now%20%E2%80%94%20no%20sign%20up%2C%20no%20API%20keys-costguard.up.railway.app-6d28d9?style=for-the-badge&logo=railway&logoColor=white" alt="Try CostGuard Now" />
  </a>
</p>

<p align="center"><strong>No API keys needed for Simulation Mode. Upload a file, get results instantly.</strong></p>

---

## What is CostGuard?

Most teams pick an LLM and stick with it — often overpaying by 10–20× for tasks a cheaper model handles just as well. CostGuard fixes that.

Upload any CSV or Parquet file. CostGuard runs your data through **[RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench)** — a 4-dimensional evaluation harness — across 15 major LLMs, then surfaces:

- **Best model recommendation** ranked by RDAB score + exact cost
- **Per-run cost estimate** down to $0.000001 precision
- **One-click copyable config** — paste directly into your project
- **Radar chart** comparing Correctness · Code Quality · Efficiency · Stat Validity

No account. No data stored. Results in under 15 seconds.

---

## Architecture

```mermaid
graph TB
    User([User Browser]) -->|Upload CSV/Parquet| ST[Streamlit Dashboard<br/>:8501]
    ST -->|POST /evaluate| API[FastAPI Backend<br/>:8000]

    API --> DL[Data Loader<br/>CSV · Parquet · Validation]
    DL --> QG[Question Generator<br/>Auto-generates from schema]
    QG --> EE[CostGuard Engine<br/>Parallel model evaluation]

    EE -->|Dynamic TaskSchema| RDAB[RealDataAgentBench<br/>harness.Agent + CompositeScorer]

    RDAB --> OAI[OpenAI<br/>GPT-5 · 4.1 · 4o]
    RDAB --> ANT[Anthropic<br/>Claude Sonnet · Haiku]
    RDAB --> GGL[Google<br/>Gemini 2.5 Pro · Flash]
    RDAB --> GRQ[Groq<br/>Llama 3.3 · Mixtral]
    RDAB --> XAI[xAI<br/>Grok-3 · Grok-3 mini]

    RDAB --> SC[RDAB CompositeScorer<br/>Correctness · Code · Efficiency · StatVal]
    SC --> RK[Ranker<br/>60% RDAB + 40% Cost]
    RK -->|EvalResponse| API
    API -->|JSON| ST
    ST -->|Radar + Scatter + Table + Config| User

    style ST fill:#667eea,color:#fff
    style API fill:#764ba2,color:#fff
    style RDAB fill:#11998e,color:#fff
    style EE fill:#059669,color:#fff
```

---

## Supported Models

| Model | Provider | Tier | Input $/1K | RDAB Note |
|-------|----------|------|-----------|-----------|
| Claude Sonnet 4.6 | Anthropic | Premium | $0.003 | RDAB default model |
| Claude Opus 4.6 | Anthropic | Premium | $0.015 | Highest capability |
| Claude Haiku 4.5 | Anthropic | Economy | $0.00025 | Token-inefficient (RDAB finding) |
| **GPT-4.1** | OpenAI | Premium | **$0.002** | **RDAB cost-performance leader** |
| GPT-4.1 mini | OpenAI | Balanced | $0.0004 | Fast, 1M context |
| GPT-4.1 nano | OpenAI | Economy | $0.0001 | Ultra-cheap |
| GPT-4o | OpenAI | Premium | $0.0025 | Proven reliability |
| GPT-4o mini | OpenAI | Balanced | $0.00015 | Structured output |
| GPT-5 | OpenAI | Premium | $0.015 | Max capability, 16× GPT-4.1 cost |
| **Gemini 2.5 Flash** | Google | Economy | **$0.000075** | **Cheapest overall** |
| Gemini 2.5 Pro | Google | Premium | $0.00125 | 2M context |
| Llama 3.3 70B (Groq) | Groq | Balanced | $0.00059 | Best on modeling tasks (RDAB) |
| Mixtral 8x7B (Groq) | Groq | Economy | $0.00024 | Ultra-fast MoE |
| Grok-3 | xAI | Premium | $0.003 | sklearn blind spot (RDAB finding) |
| Grok-3 mini | xAI | Balanced | $0.0003 | Fast, cheap |

---

## Quickstart (Local)

### 1. Clone & configure

```bash
git clone https://github.com/patibandlavenkatamanideep/CostGuard.git
cd CostGuard

cp .env.example .env
# Optional: add API keys for Live Mode. Leave blank for Simulation Mode.
```

### 2. Install & run

```bash
pip install -e .
./scripts/dev.sh
```

Open:
- Dashboard: **http://localhost:8501**
- API Docs: **http://localhost:8000/docs**

### 3. Docker

```bash
cp .env.example .env
docker compose up
```

---

## Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/patibandlavenkatamanideep/CostGuard)

**Exact steps:**

```bash
# 1. Install the Railway CLI
npm install -g @railway/cli

# 2. Authenticate
railway login

# 3. Link or create a project
railway init

# 4. Deploy (Railway reads railway.json automatically)
railway up

# 5. Get your public URL
railway open
```

**Environment variables** (all optional — app fully works in Simulation Mode without any keys):

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Enable Claude models in Live Mode |
| `OPENAI_API_KEY` | Enable GPT models in Live Mode |
| `GROQ_API_KEY` | Enable Llama / Mixtral via Groq |
| `GEMINI_API_KEY` | Enable Gemini 2.5 Pro / Flash |
| `XAI_API_KEY` | Enable Grok-3 / Grok-3 mini |

> The container runs FastAPI (internal, port 8000) + Streamlit (public, `$PORT`) side-by-side via `scripts/start.sh`.

---

## Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/patibandlavenkatamanideep/CostGuard)

```bash
# 1. Fork this repo on GitHub

# 2. Go to https://dashboard.render.com → New → Web Service
#    Connect your fork — Render auto-detects render.yaml

# 3. Add API keys in the Environment tab (all optional)

# 4. Click Deploy — live in ~3 minutes
```

---

## API Reference

Auto-documented at `/docs` (Swagger) and `/redoc`.

### POST `/evaluate`

```bash
curl -X POST https://costguard.up.railway.app/evaluate \
  -F "file=@my_data.csv" \
  -F "task_description=Analyze customer churn patterns" \
  -F "num_questions=5"
```

**Response:**
```json
{
  "eval_id": "a3f9e1b2",
  "status": "completed",
  "dataset_stats": { "rows": 5000, "columns": 12 },
  "recommended_model": {
    "model_id": "gpt-4o-mini",
    "display_name": "GPT-4o mini",
    "accuracy_score": 0.87,
    "estimated_total_cost_usd": 0.000423,
    "latency_ms": 612
  },
  "recommendation_reason": "GPT-4o mini achieves the best balance...",
  "copyable_config": "{ \"model\": \"gpt-4o-mini\", ... }"
}
```

### GET `/health`
```bash
curl https://costguard.up.railway.app/health
```

### GET `/models`
```bash
curl https://costguard.up.railway.app/models
```

---

## RDAB Scoring Methodology

CostGuard uses [RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench) as its evaluation engine, scoring each model across **4 dimensions**:

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| **Correctness** | 50% | Answer accuracy vs ground truth (fuzzy-matched, ±15% tolerance) |
| **Code Quality** | 20% | Vectorised operations, naming conventions, no magic numbers |
| **Efficiency** | 15% | Token + step budget adherence (Easy: 20K tokens, Medium: 50K) |
| **Stat Validity** | 15% | Reports p-values, confidence intervals, avoids p-hacking |

### Key RDAB Benchmark Findings (163 runs)

- **GPT-4.1** = best cost-performance ratio ($0.038/task vs GPT-5's $0.596)
- **Gemini 2.5 Flash** = cheapest per RDAB score ($0.000075/1K input)
- **Llama 3.3-70B (Groq)** = outperforms on modeling tasks
- **Claude Haiku** = consumed 608K tokens vs GPT-4.1's 30K on the same task
- **Universal weakness**: all models score ~0.25 on stat_validity

---

## Business Impact

| Use Case | Without CostGuard | With CostGuard |
|----------|------------------|----------------|
| Model selection | 2–3 days of testing | **15 seconds** |
| Cost budgeting | Guesswork | Exact per-run estimates |
| Over-provisioning | ~60% of teams use GPT-4o for tasks GPT-4.1 handles at 20% of the cost | Right-sized model every time |
| Onboarding | Engineers research models manually | Copy one config block |

### Example savings (from RDAB benchmark data)

| Scenario | Old choice | CostGuard pick | Savings |
|----------|-----------|----------------|---------|
| Structured data analysis | GPT-4o: $0.0025/1K | GPT-4.1: $0.002/1K | **20% cheaper, same quality** |
| Budget inference at scale | GPT-4o: $0.0025/1K | GPT-4o-mini: $0.00015/1K | **94% cheaper**, <5% accuracy drop |
| Cheapest viable option | Claude Sonnet: $0.003/1K | Gemini 2.5 Flash: $0.000075/1K | **97.5% cheaper** |
| Worst over-provisioning (RDAB) | Claude Haiku: 608K tokens | GPT-4.1: 30K tokens | **Same task — 20× fewer tokens** |

> GPT-5 costs $0.015/1K input — 200× more than Gemini 2.5 Flash. CostGuard tells you when that premium is justified.

---

## Project Structure

```
costguard/
├── backend/
│   ├── main.py          # FastAPI app, routes, middleware
│   ├── config.py        # Pydantic settings, env vars
│   ├── models.py        # Request/response schemas
│   └── logger.py        # Structured logging
├── evaluation/
│   ├── engine.py        # Core evaluation orchestrator
│   ├── data_loader.py   # CSV/Parquet ingestion & validation
│   ├── pricing.py       # Model pricing catalogue
│   ├── question_generator.py  # RDAB-style question generation
│   └── token_counter.py # Token estimation
├── frontend/
│   └── app.py           # Streamlit dashboard
├── tests/
│   └── test_evaluation.py
├── scripts/
│   ├── dev.sh           # Local dev startup
│   └── start.sh         # Production single-container startup
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── railway.json
├── render.yaml
└── pyproject.toml
```

---

## Development

```bash
# Run tests
pytest tests/ -v

# Integration tests (requires running server)
pytest tests/ -m integration

# Lint & format
ruff check .
ruff format .

# Type check
mypy backend/ evaluation/
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Security

See [SECURITY.md](SECURITY.md). Report vulnerabilities to security@costguard.dev.

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built with FastAPI, Streamlit, and RealDataAgentBench. The best model is the one that fits your data — and your budget.*
