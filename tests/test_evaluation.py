"""Tests for the CostGuard evaluation engine — RDAB-powered."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from evaluation.data_loader import (
    DataLoadError,
    compute_stats,
    dataframe_to_prompt_text,
    load_bytes,
    sample_dataframe,
)
from evaluation.pricing import MODELS, get_model, get_models_for_providers
from evaluation.question_generator import generate_questions
from evaluation.token_counter import count_tokens, estimate_batch_tokens


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv_bytes() -> bytes:
    df = pd.DataFrame({
        "id": range(100),
        "name": [f"Item_{i}" for i in range(100)],
        "value": [i * 1.5 for i in range(100)],
        "category": [["A", "B", "C", "D"][i % 4] for i in range(100)],
    })
    return df.to_csv(index=False).encode()


@pytest.fixture
def sample_parquet_bytes() -> bytes:
    df = pd.DataFrame({"x": range(50), "y": [i**2 for i in range(50)]})
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "revenue": [100, 200, 300, 400, 500],
        "region": ["N", "S", "E", "W", "N"],
        "active": [True, False, True, True, False],
    })


# ─── Data Loader Tests ───────────────────────────────────────────────────────

class TestDataLoader:
    def test_load_csv_bytes(self, sample_csv_bytes):
        df = load_bytes(sample_csv_bytes, "test.csv")
        assert len(df) == 100
        assert "value" in df.columns

    def test_load_parquet_bytes(self, sample_parquet_bytes):
        df = load_bytes(sample_parquet_bytes, "test.parquet")
        assert len(df) == 50
        assert "x" in df.columns

    def test_load_unsupported_format(self, sample_csv_bytes):
        with pytest.raises(DataLoadError, match="Unsupported file format"):
            load_bytes(sample_csv_bytes, "test.xlsx")

    def test_load_empty_file(self):
        empty_csv = b"col1,col2\n"
        with pytest.raises(DataLoadError, match="no data rows"):
            load_bytes(empty_csv, "empty.csv")

    def test_compute_stats(self, sample_df):
        stats = compute_stats(sample_df, "test.csv", file_size_bytes=1024)
        assert stats.rows == 5
        assert stats.columns == 3
        assert stats.file_format == "CSV"
        assert 0 <= stats.missing_pct <= 100

    def test_sample_dataframe_small(self, sample_df):
        result = sample_dataframe(sample_df, max_rows=100)
        assert len(result) == len(sample_df)

    def test_sample_dataframe_large(self):
        big_df = pd.DataFrame({"x": range(1000)})
        result = sample_dataframe(big_df, max_rows=100)
        assert len(result) == 100

    def test_dataframe_to_prompt_text(self, sample_df):
        text = dataframe_to_prompt_text(sample_df)
        assert "revenue" in text
        assert "5 rows" in text


# ─── Pricing Tests ───────────────────────────────────────────────────────────

class TestPricing:
    def test_all_models_present(self):
        # We now have 15+ models including RDAB-supported ones
        assert len(MODELS) >= 12

    def test_rdab_default_model_present(self):
        """claude-sonnet-4-6 is RDAB's default model."""
        m = get_model("claude-sonnet-4-6")
        assert m is not None
        assert m.rdab_alias == "sonnet"

    def test_gpt41_present(self):
        """GPT-4.1 is RDAB's cost-performance leader."""
        m = get_model("gpt-4.1")
        assert m is not None
        assert m.rdab_alias == "gpt4.1"

    def test_gemini_flash_cheapest(self):
        """Gemini 2.5 Flash is the cheapest model per RDAB findings."""
        m = get_model("gemini-2.5-flash")
        assert m is not None
        all_costs = [model.input_per_1k for model in MODELS.values()]
        assert m.input_per_1k == min(all_costs)

    def test_all_models_have_rdab_alias(self):
        for model_id, m in MODELS.items():
            assert m.rdab_alias, f"Model {model_id} missing rdab_alias"

    def test_get_model_unknown(self):
        assert get_model("nonexistent-model-xyz") is None

    def test_cost_estimate(self):
        m = get_model("gpt-4.1-mini")
        assert m is not None
        cost = m.estimate_cost(1000, 200)
        assert cost > 0
        assert cost < 0.01  # Should be very cheap

    def test_get_models_for_providers_openai(self):
        openai_models = get_models_for_providers(["openai"])
        assert all(m.provider == "openai" for m in openai_models)

    def test_get_models_for_providers_anthropic(self):
        models = get_models_for_providers(["anthropic"])
        assert all(m.provider == "anthropic" for m in models)

    def test_get_models_no_providers_returns_all(self):
        all_models = get_models_for_providers([])
        assert len(all_models) == len(MODELS)

    def test_all_providers_covered(self):
        providers = {m.provider for m in MODELS.values()}
        assert providers == {"anthropic", "openai", "groq", "xai", "google"}


# ─── Question Generator Tests ────────────────────────────────────────────────

class TestQuestionGenerator:
    def test_generates_correct_count(self, sample_df):
        questions = generate_questions(sample_df, num_questions=5)
        assert len(questions) == 5

    def test_generates_strings(self, sample_df):
        questions = generate_questions(sample_df, num_questions=3)
        assert all(isinstance(q, str) and len(q) > 5 for q in questions)

    def test_deterministic_with_seed(self, sample_df):
        q1 = generate_questions(sample_df, num_questions=5, seed=42)
        q2 = generate_questions(sample_df, num_questions=5, seed=42)
        assert q1 == q2

    def test_different_seeds_differ(self, sample_df):
        q1 = generate_questions(sample_df, num_questions=5, seed=1)
        q2 = generate_questions(sample_df, num_questions=5, seed=99)
        assert q1 != q2


# ─── Token Counter Tests ─────────────────────────────────────────────────────

class TestTokenCounter:
    def test_count_tokens_returns_positive(self):
        tokens = count_tokens("Hello, world!")
        assert tokens > 0

    def test_longer_text_more_tokens(self):
        short = count_tokens("Hi")
        long = count_tokens("This is a much longer sentence with many more words in it.")
        assert long > short

    def test_estimate_batch(self):
        inp, out = estimate_batch_tokens(
            system_prompt="You are a helpful analyst.",
            data_text="col1,col2\n1,2\n3,4",
            questions=["What is the sum of col1?", "What is the max of col2?"],
        )
        assert inp > 0
        assert out > 0


# ─── RDAB Simulation Tests ───────────────────────────────────────────────────

class TestRDABSimulation:
    """Tests for the deterministic simulation fallback (no API keys needed)."""

    def test_simulation_returns_scorecard(self):
        from evaluation.engine import _simulate_scorecard
        pricing = get_model("gpt-4.1")
        assert pricing is not None
        sc, latency = _simulate_scorecard(pricing)
        assert 0 <= sc.rdab_score <= 1
        assert sc.simulated is True
        assert latency > 0

    def test_simulation_deterministic(self):
        from evaluation.engine import _simulate_scorecard
        pricing = get_model("claude-sonnet-4-6")
        assert pricing is not None
        sc1, _ = _simulate_scorecard(pricing, seed=42)
        sc2, _ = _simulate_scorecard(pricing, seed=42)
        assert sc1.rdab_score == sc2.rdab_score

    def test_premium_beats_economy_on_correctness(self):
        """Premium models should score higher on correctness than economy."""
        from evaluation.engine import _simulate_scorecard
        premium = get_model("gpt-4.1")
        economy = get_model("gemini-2.5-flash")
        assert premium and economy
        sc_p, _ = _simulate_scorecard(premium)
        sc_e, _ = _simulate_scorecard(economy)
        assert sc_p.correctness > sc_e.correctness

    def test_economy_better_efficiency(self):
        """Economy models should use fewer tokens → higher efficiency score."""
        from evaluation.engine import _simulate_scorecard
        premium = get_model("claude-sonnet-4-6")
        economy = get_model("claude-haiku-4-5-20251001")
        assert premium and economy
        sc_p, _ = _simulate_scorecard(premium)
        sc_e, _ = _simulate_scorecard(economy)
        assert sc_e.efficiency > sc_p.efficiency

    def test_stat_validity_universally_low(self):
        """RDAB finding: all models score ~0.25 on stat_validity."""
        from evaluation.engine import _simulate_scorecard
        for model_id in ["gpt-4.1", "claude-sonnet-4-6", "gemini-2.5-flash"]:
            pricing = get_model(model_id)
            assert pricing is not None
            sc, _ = _simulate_scorecard(pricing)
            assert sc.stat_validity < 0.45, (
                f"{model_id} stat_validity={sc.stat_validity:.3f} exceeds expected cap"
            )

    def test_all_models_get_valid_scores(self):
        """Every model in the catalogue should produce valid simulation scores."""
        from evaluation.engine import _simulate_scorecard
        for model_id, pricing in MODELS.items():
            sc, latency = _simulate_scorecard(pricing)
            assert 0 <= sc.rdab_score <= 1, f"{model_id} rdab_score out of range"
            assert 0 <= sc.correctness <= 1
            assert 0 <= sc.code_quality <= 1
            assert 0 <= sc.efficiency <= 1
            assert 0 <= sc.stat_validity <= 1
            assert latency > 0


# ─── API Integration Tests (require running server) ──────────────────────────

@pytest.mark.integration
class TestAPIIntegration:
    """These tests require a running FastAPI server. Run with: pytest -m integration"""

    @pytest.fixture(autouse=True)
    def client(self):
        import httpx
        self.client = httpx.Client(base_url="http://localhost:8000", timeout=60)
        yield
        self.client.close()

    def test_health_check(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "rdab_available" in body

    def test_models_endpoint(self):
        resp = self.client.get("/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 12

    def test_evaluate_csv_returns_rdab_scores(self, sample_csv_bytes):
        resp = self.client.post(
            "/evaluate",
            files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
            data={"task_description": "Test RDAB scoring", "num_questions": "3"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "recommended_model" in body
        rec = body["recommended_model"]
        sc = rec["rdab_scorecard"]
        assert 0 <= sc["rdab_score"] <= 1
        assert 0 <= sc["correctness"] <= 1
        assert 0 <= sc["code_quality"] <= 1
        assert 0 <= sc["efficiency"] <= 1
        assert 0 <= sc["stat_validity"] <= 1
