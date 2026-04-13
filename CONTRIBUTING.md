# Contributing to CostGuard

Thank you for your interest in contributing!

## Quick Start

```bash
git clone https://github.com/your-org/costguard.git
cd costguard
pip install -e ".[dev]"
cp .env.example .env
```

## Workflow

1. Fork the repo and create a feature branch: `git checkout -b feat/your-feature`
2. Make your changes with tests
3. Run `ruff check . && pytest tests/` — both must pass
4. Open a pull request against `main`

## Adding a New Model Provider

1. Add pricing data to `evaluation/pricing.py`
2. Add a `_call_<provider>` function in `evaluation/engine.py`
3. Add the provider to the dispatch in `_call_model()`
4. Add the env var to `.env.example` and `backend/config.py`
5. Add tests in `tests/test_evaluation.py`

## Code Style

- `ruff` for linting and formatting (`ruff check . && ruff format .`)
- Type hints on all public functions
- Docstrings on all public modules and classes

## Commit Convention

```
feat: add Together AI provider
fix: handle empty parquet files gracefully
docs: update model pricing for Q2 2025
```
