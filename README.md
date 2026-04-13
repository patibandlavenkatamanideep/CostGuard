# CostGuard

> **Instantly find the best LLM for your data — with exact cost estimates.**

[![CI/CD](https://github.com/patibandlavenkatamanideep/CostGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/patibandlavenkatamanideep/CostGuard/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Powered by RDAB](https://img.shields.io/badge/Evaluation-RealDataAgentBench-7c3aed)](https://github.com/patibandlavenkatamanideep/RealDataAgentBench)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20Now-brightgreen)](https://costguard.up.railway.app)

---

## What is CostGuard?

Upload any CSV or Parquet file → CostGuard benchmarks **14 major LLMs** against your actual data using **[RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench)** as the evaluation engine, and returns:

-  **Best model recommendation** with 4-dimensional RDAB scoring
-  **Exact cost estimate** per run (down to $0.000001)
-  **One-click copyable config** — paste directly into your project
-  **Radar chart** comparing Correctness · Code Quality · Efficiency · Stat Validity

No account required. No data stored. Works in under 15 seconds.

---

## Live Demo

**[costguard.up.railway.app](https://costguard.up.railway.app)**

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

All 5 providers natively supported by RealDataAgentBench.

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
git clone https://github.com/your-org/costguard.git
cd costguard

cp .env.example .env
# Edit .env and add your API keys (at least one of OPENAI_API_KEY or ANTHROPIC_API_KEY)
```

### 2. Install & run

```bash
pip install -e .
./scripts/dev.sh
```

Then open:
- Dashboard: **http://localhost:8501**
- API Docs: **http://localhost:8000/docs**

### 3. Docker (production)

```bash
cp .env.example .env  # fill in your keys
docker compose up
```

---

## Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template)

1. Fork this repo
2. Create a new Railway project → **Deploy from GitHub repo**
3. Add environment variables from `.env.example` in Railway's Variables tab
4. Railway auto-detects `railway.json` and deploys both services

---

## Deploy to Render

```bash
# Install Render CLI
npm install -g @render-oss/cli

render deploy
```

Or click: [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/your-org/costguard)

---

## API Reference

The FastAPI backend is fully documented at `/docs` (Swagger UI) and `/redoc`.

### POST `/evaluate`

```bash
curl -X POST http://localhost:8000/evaluate \
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
curl http://localhost:8000/health
```

### GET `/models`
```bash
curl http://localhost:8000/models
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
- **Gemini 2.5 Flash** = best cost-per-RDAB-score (cheapest at $0.000075/1K input)
- **Llama 3.3-70B (Groq)** = outperforms on modeling tasks
- **Claude Haiku** = consumed 608K tokens vs GPT-4.1's 30K on same task
- **Universal weakness**: All models score ~0.25 on stat_validity

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
│   ├── question_generator.py  # RealDataAgentBench-style Q generation
│   └── token_counter.py # Token estimation
├── frontend/
│   └── app.py           # Streamlit dashboard
├── tests/
│   └── test_evaluation.py
├── scripts/
│   └── dev.sh           # Local dev startup
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── railway.json
├── render.yaml
└── pyproject.toml
```

---

## Business Impact

| Use Case | Without CostGuard | With CostGuard |
|----------|------------------|----------------|
| Model selection | 2–3 days of testing | 15 seconds |
| Cost budgeting | Guesswork | Exact per-run estimates |
| Over-provisioning | ~60% of teams use GPT-4 for tasks GPT-4o-mini handles | Right-sized model every time |
| Onboarding | Engineers research models manually | Copy one config block |

> **Real-world savings:** Switching from GPT-4o to GPT-4o-mini for appropriate tasks saves **83–94%** on LLM costs with <5% accuracy impact for structured data tasks.

---

## Development

```bash
# Run tests
pytest tests/ -v

# Run integration tests (requires running server)
pytest tests/ -m integration

# Lint
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

See [SECURITY.md](SECURITY.md). To report a vulnerability, email security@costguard.dev.

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built with FastAPI, Streamlit, and the philosophy that the best tool is the one you'll actually use.*
