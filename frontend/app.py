"""CostGuard — Streamlit Dashboard"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.models import EvalMode, EvalResponse, SessionKeys  # noqa: E402
from evaluation.engine import run_evaluation  # noqa: E402
from evaluation.observability import (  # noqa: E402
    get_model_averages,
    get_recent_drift_events,
    get_recent_evaluations,
    get_total_eval_count,
)


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CostGuard — LLM Cost Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design system ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #f8fafc;
    color: #0f172a;
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #312e81 0%, #4f46e5 55%, #7c3aed 100%);
    border-radius: 16px;
    padding: 2.25rem 2rem 2rem;
    margin-bottom: 1.75rem;
    color: #fff;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-icon  { font-size: 2.4rem; line-height: 1; margin-bottom: 0.6rem; }
.hero-title { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.5px; margin: 0 0 0.4rem; }
.hero-sub   { font-size: 0.95rem; color: rgba(255,255,255,.8); margin: 0 0 1.25rem; max-width: 520px; line-height: 1.55; }
.hero-pills { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.hero-pill  {
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.25);
    border-radius: 999px;
    padding: 0.22rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ── Cards ── */
.card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}

/* ── Steps (how it works) ── */
.steps { display: flex; gap: 1rem; margin: 0.25rem 0 0; flex-wrap: wrap; }
.step  { flex: 1; min-width: 110px; display: flex; gap: 0.55rem; align-items: flex-start; }
.step-num {
    width: 22px; height: 22px; min-width: 22px;
    background: #ede9fe; color: #6366f1;
    border-radius: 50%; font-size: 0.68rem; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    margin-top: 1px;
}
.step-label { font-size: 0.8rem; font-weight: 600; color: #1e293b; display: block; }
.step-desc  { font-size: 0.75rem; color: #64748b; }

/* ── Mode badge ── */
.mode-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.76rem; font-weight: 600;
    padding: 0.28rem 0.85rem; border-radius: 999px;
    margin-bottom: 1.25rem;
}
.mode-live { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.mode-sim  { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }

/* ── Result Hero ── */
.result-hero {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.75rem 2rem 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.rh-top {
    display: flex; align-items: flex-start; gap: 1rem;
    margin-bottom: 1.5rem;
}
.rh-trophy { font-size: 2.4rem; line-height: 1; margin-top: 0.1rem; }
.rh-eyebrow {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .1em; color: #6366f1; margin-bottom: 0.3rem;
}
.rh-model {
    font-size: 2rem; font-weight: 800; color: #0f172a;
    letter-spacing: -0.5px; line-height: 1.15; margin-bottom: 0.3rem;
}
.rh-meta {
    font-size: 0.76rem; font-weight: 600; color: #64748b;
}
.rh-stats {
    display: flex; gap: 1rem; flex-wrap: wrap;
    border-top: 1px solid #f1f5f9; padding-top: 1.25rem;
}
.rh-stat {
    flex: 1; min-width: 140px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
.rh-stat-green  { background: #f0fdf4; border-color: #bbf7d0; }
.rh-stat-purple { background: #f5f3ff; border-color: #ddd6fe; }
.rh-stat-label {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: #94a3b8; margin-bottom: 0.35rem;
}
.rh-stat-value {
    font-size: 1.8rem; font-weight: 800; color: #0f172a;
    line-height: 1.1; margin-bottom: 0.2rem;
}
.rh-stat-green  .rh-stat-value { color: #15803d; }
.rh-stat-purple .rh-stat-value { color: #5b21b6; }
.rh-stat-sub { font-size: 0.72rem; color: #64748b; }

/* ── 4-Dimension Scorecard ── */
.scorecard {
    display: flex; gap: 1rem; flex-wrap: wrap; margin: 0 0 1.5rem;
}
.sc-item {
    flex: 1; min-width: 140px;
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    position: relative; overflow: hidden;
}
.sc-bar-bg {
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 4px; background: #f1f5f9;
}
.sc-bar-fill {
    position: absolute; bottom: 0; left: 0;
    height: 4px; background: #6366f1; border-radius: 0 2px 2px 0;
    transition: width .4s;
}
.sc-label  { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: #94a3b8; margin-bottom: 0.4rem; }
.sc-value  { font-size: 1.6rem; font-weight: 800; color: #0f172a; line-height: 1; margin-bottom: 0.2rem; }
.sc-weight { font-size: 0.7rem; color: #94a3b8; }

/* ── Section heading ── */
.sh {
    display: flex; align-items: center; gap: 0.5rem;
    border-bottom: 2px solid #f1f5f9;
    padding-bottom: 0.5rem;
    margin: 1.75rem 0 1rem;
}
.sh h3 { font-size: 0.95rem; font-weight: 700; color: #1e293b; margin: 0; }
.sh-icon { font-size: 1rem; }

/* ── Tier badge ── */
.tier { display: inline-block; font-size: 0.67rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; }
.t-premium  { background: #ede9fe; color: #5b21b6; }
.t-balanced { background: #dbeafe; color: #1d4ed8; }
.t-economy  { background: #dcfce7; color: #166534; }

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.65rem 1.5rem;
    box-shadow: 0 2px 8px rgba(99,102,241,.3);
    transition: all .15s;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(99,102,241,.45) !important;
    transform: translateY(-1px);
}
.stButton > button[kind="secondary"] {
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px;
    font-size: 0.8rem; font-weight: 600;
    color: #475569 !important;
    background: #fff !important;
    transition: all .12s;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #6366f1 !important;
    color: #6366f1 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #c7d2fe;
    border-radius: 10px;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color: #6366f1; }

/* ── Sidebar ── */
.sb-brand { font-size: 1.1rem; font-weight: 800; color: #4f46e5; margin: 0; }
.sb-tagline { font-size: 0.71rem; color: #94a3b8; margin: 0 0 1.25rem; }
.sb-section { font-size: 0.67rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: #94a3b8; margin: 1rem 0 0.4rem; }

/* ── Empty state ── */
.empty {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #94a3b8;
}
.empty-icon  { font-size: 3rem; margin-bottom: 0.75rem; }
.empty-title { font-size: 1rem; font-weight: 700; color: #64748b; margin-bottom: 0.35rem; }
.empty-sub   { font-size: 0.82rem; }

/* ── Config block ── */
.config-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 0.5rem;
}
.config-title { font-size: 0.85rem; font-weight: 700; color: #1e293b; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.77rem;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #f1f5f9;
    margin-top: 2rem;
}
.footer a { color: #6366f1; text-decoration: none; font-weight: 600; }
.footer a:hover { color: #4f46e5; text-decoration: underline; }

/* ── Observability ── */
.drift-alert {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-left: 4px solid #f97316;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.75rem;
}
.drift-alert-title { font-size: 0.82rem; font-weight: 700; color: #9a3412; margin-bottom: 0.2rem; }
.drift-alert-body  { font-size: 0.78rem; color: #7c2d12; line-height: 1.5; }
.obs-stat {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
    flex: 1;
    min-width: 100px;
}
.obs-stat-label { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: #94a3b8; }
.obs-stat-value { font-size: 1.3rem; font-weight: 800; color: #0f172a; }
.obs-row { display: flex; gap: 0.8rem; margin: 0.8rem 0 1.25rem; flex-wrap: wrap; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_cost(usd: float) -> str:
    if usd == 0:
        return "$0.00"
    if usd < 0.0001:
        return f"${usd * 1_000_000:.2f} µ"
    if usd < 0.01:
        return f"${usd * 1000:.3f} m"
    return f"${usd:.5f}"


def _run_evaluation_sync(
    file_bytes: bytes,
    filename: str,
    task_description: str,
    num_questions: int,
    session_keys: SessionKeys,
) -> dict:
    raw = asyncio.run(
        run_evaluation(
            file_content=file_bytes,
            filename=filename,
            file_size_bytes=len(file_bytes),
            task_description=task_description,
            num_questions=num_questions,
            session_keys=session_keys,
        )
    )
    return EvalResponse(**raw).model_dump(mode="json")


def _make_sample_ecommerce() -> bytes:
    df = pd.DataFrame({
        "customer_id":     range(1, 101),
        "age":             [25 + i % 45 for i in range(100)],
        "annual_spend_usd":[500 + (i * 137 % 9500) for i in range(100)],
        "region":          [["North", "South", "East", "West"][i % 4] for i in range(100)],
        "churn":           [i % 5 == 0 for i in range(100)],
    })
    return df.to_csv(index=False).encode()


def _make_sample_sales() -> bytes:
    df = pd.DataFrame({
        "deal_id":   range(1, 51),
        "value_usd": [10_000 + i * 4321 % 200_000 for i in range(50)],
        "stage":     [["Discovery", "Proposal", "Negotiation", "Closed"][i % 4] for i in range(50)],
        "rep":       [f"Rep_{i % 8}" for i in range(50)],
        "days_open": [5 + i % 90 for i in range(50)],
    })
    return df.to_csv(index=False).encode()


def _make_sample_products() -> bytes:
    df = pd.DataFrame({
        "product":  [f"SKU-{i:04d}" for i in range(80)],
        "category": [["Electronics", "Apparel", "Home", "Sports"][i % 4] for i in range(80)],
        "price":    [9.99 + i * 7.77 % 500 for i in range(80)],
        "rating":   [3.0 + (i % 20) / 10 for i in range(80)],
        "reviews":  [i * 13 % 2000 for i in range(80)],
    })
    return df.to_csv(index=False).encode()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sb-brand">🛡️ CostGuard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sb-tagline">LLM Cost Intelligence Platform</p>', unsafe_allow_html=True)

    st.markdown('<p class="sb-section">Task Description</p>', unsafe_allow_html=True)
    task_description = st.text_area(
        "Task",
        value="Analyze this dataset and answer questions about it.",
        height=80,
        help="Describe what you want to do with the data — e.g. 'Predict churn', 'Find outliers'.",
        label_visibility="collapsed",
    )

    st.markdown('<p class="sb-section">Evaluation Depth</p>', unsafe_allow_html=True)
    num_questions = st.slider(
        "Questions",
        min_value=1, max_value=10, value=5,
        help="More questions → more accurate benchmark, slightly longer runtime.",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown('<p class="sb-section">API Keys — Live Mode (optional)</p>', unsafe_allow_html=True)
    st.caption("Keys are used only for this session and never stored.")

    with st.expander("Enter API keys", expanded=False):
        anthropic_key = st.text_input("Anthropic (Claude)", type="password", placeholder="sk-ant-...")
        openai_key    = st.text_input("OpenAI (GPT)",        type="password", placeholder="sk-...")
        groq_key      = st.text_input("Groq (Llama)",        type="password", placeholder="gsk_...")
        gemini_key    = st.text_input("Google (Gemini)",     type="password", placeholder="AIza...")
        xai_key       = st.text_input("xAI (Grok)",          type="password", placeholder="xai-...")
        if st.button("Clear all keys", use_container_width=True):
            st.rerun()

    session_keys = SessionKeys(
        anthropic_api_key=anthropic_key or None,
        openai_api_key=openai_key or None,
        groq_api_key=groq_key or None,
        xai_api_key=xai_key or None,
        gemini_api_key=gemini_key or None,
    )
    any_key = session_keys.has_any_key()

    if any_key:
        live_names = [n for n, v in [
            ("Anthropic", anthropic_key), ("OpenAI", openai_key),
            ("Groq", groq_key), ("Google", gemini_key), ("xAI", xai_key),
        ] if v and v.strip()]
        st.success(f"Live Mode active — {', '.join(live_names)}")
    else:
        st.info("Simulation Mode — no keys required")

    st.divider()
    st.markdown('<p class="sb-section">Models Covered</p>', unsafe_allow_html=True)
    st.caption("Claude Sonnet · Opus · Haiku  \nGPT-5 · 4.1 · 4o · 4o-mini  \nGemini 2.5 Pro · Flash  \nLlama 3.3 70B · Mixtral  \nGrok-3 · Grok-3 mini")


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">🛡️</div>
  <h1 class="hero-title">CostGuard</h1>
  <p class="hero-sub">
    Upload any dataset and instantly benchmark 10 models on your actual data.
    Get the best model recommendation with exact per-run cost estimates — no guesswork.
  </p>
  <div class="hero-pills">
    <span class="hero-pill">10 Benchmarked Models</span>
    <span class="hero-pill">4-Dimensional RDAB Scoring</span>
    <span class="hero-pill">No Sign-up</span>
    <span class="hero-pill">Simulation Mode — No API Keys Needed</span>
    <span class="hero-pill">🔒 Data processed in memory — never stored</span>
    <span class="hero-pill">🐳 Self-host with Docker in one command</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Upload + How it works ────────────────────────────────────────────────────
col_up, col_how = st.columns([3, 2], gap="large")

with col_up:
    st.markdown("##### Upload your dataset")
    uploaded_file = st.file_uploader(
        "CSV or Parquet — up to 50 MB",
        type=["csv", "parquet"],
        label_visibility="collapsed",
    )

    st.markdown("##### Or try a sample")
    s1, s2, s3 = st.columns(3)
    if s1.button("🛒  E-commerce", use_container_width=True):
        st.session_state["sample_file"] = ("ecommerce_customers.csv", _make_sample_ecommerce())
        st.session_state.pop("result", None)
    if s2.button("📊  Sales Pipeline", use_container_width=True):
        st.session_state["sample_file"] = ("sales_pipeline.csv", _make_sample_sales())
        st.session_state.pop("result", None)
    if s3.button("📦  Product Metrics", use_container_width=True):
        st.session_state["sample_file"] = ("product_metrics.csv", _make_sample_products())
        st.session_state.pop("result", None)

with col_how:
    st.markdown("##### How it works")
    st.markdown("""
<div class="steps">
  <div class="step">
    <div class="step-num">1</div>
    <div><span class="step-label">Upload</span><span class="step-desc">CSV or Parquet file</span></div>
  </div>
  <div class="step">
    <div class="step-num">2</div>
    <div><span class="step-label">Benchmark</span><span class="step-desc">10 models on your data</span></div>
  </div>
  <div class="step">
    <div class="step-num">3</div>
    <div><span class="step-label">Compare</span><span class="step-desc">Score · cost · latency</span></div>
  </div>
  <div class="step">
    <div class="step-num">4</div>
    <div><span class="step-label">Deploy</span><span class="step-desc">Copy the ready config</span></div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.caption(
        "⏱ **Simulation mode:** ~5–15 seconds  ·  "
        "**Live mode (with API keys):** 1–3 minutes  \n"
        "🔒 Your file is processed in memory and never written to disk or stored."
    )


# ─── Active file + run button ─────────────────────────────────────────────────
active_file: tuple[str, bytes] | None = None
if uploaded_file is not None:
    active_file = (uploaded_file.name, uploaded_file.read())
elif "sample_file" in st.session_state:
    active_file = st.session_state["sample_file"]

if active_file:
    filename, file_bytes = active_file
    size_kb = len(file_bytes) / 1024
    size_str = f"{size_kb / 1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
    st.caption(f"📄 **{filename}** — {size_str}")

    mode_label = (
        "Running live benchmark across 10 models — this takes 1–3 minutes…"
        if any_key
        else "Running simulation benchmark across 10 models — usually under 15 seconds…"
    )
    if st.button("⚡  Analyze & Recommend", type="primary", use_container_width=True):
        with st.spinner(mode_label):
            try:
                data = _run_evaluation_sync(
                    file_bytes=file_bytes,
                    filename=filename,
                    task_description=task_description,
                    num_questions=num_questions,
                    session_keys=session_keys,
                )
                st.session_state["result"] = data
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                st.stop()
else:
    st.markdown("""
<div class="empty">
  <div class="empty-icon">📂</div>
  <div class="empty-title">No dataset loaded yet</div>
  <div class="empty-sub">Upload a CSV or Parquet file above, or click one of the sample datasets to get started.</div>
</div>
""", unsafe_allow_html=True)


# ─── Results ──────────────────────────────────────────────────────────────────
if result := st.session_state.get("result"):
    rec   = result["recommended_model"]
    stats = result["dataset_stats"]
    sc    = rec["rdab_scorecard"]
    mode  = result.get("eval_mode", "simulation")
    live_providers = result.get("live_providers", [])

    # ── Compute savings vs most expensive alternative ────────────────────────
    _others = [r for r in result["results"] if r["model_id"] != rec["model_id"]]
    _priciest = max(_others, key=lambda r: r["estimated_total_cost_usd"]) if _others else None
    if _priciest and _priciest["estimated_total_cost_usd"] > 0:
        _savings_pct = (1 - rec["estimated_total_cost_usd"] / _priciest["estimated_total_cost_usd"]) * 100
        _savings_label = f"vs {_priciest['display_name']}"
    else:
        _savings_pct = 0.0
        _savings_label = "vs alternatives"

    _mode_pill = (
        f'<span class="mode-badge mode-live">● Live — {", ".join(live_providers)}</span>'
        if mode == "live"
        else '<span class="mode-badge mode-sim">◎ Simulation — calibrated RDAB scores</span>'
    )

    # ── Result hero ──────────────────────────────────────────────────────────
    st.markdown(_mode_pill, unsafe_allow_html=True)
    st.markdown(f"""
<div class="result-hero">
  <div class="rh-top">
    <div class="rh-trophy">🏆</div>
    <div>
      <div class="rh-eyebrow">Best Model for Your Data</div>
      <div class="rh-model">{rec['display_name']}</div>
      <div class="rh-meta">{rec['provider'].upper()} &nbsp;·&nbsp; {rec['tier'].upper()} TIER &nbsp;·&nbsp; {rec['latency_ms']:.0f} ms avg latency</div>
    </div>
  </div>
  <div class="rh-stats">
    <div class="rh-stat rh-stat-purple">
      <div class="rh-stat-label">RDAB Score</div>
      <div class="rh-stat-value">{sc['rdab_score']:.0%}</div>
      <div class="rh-stat-sub">Composite benchmark (10 models)</div>
    </div>
    <div class="rh-stat">
      <div class="rh-stat-label">Estimated Cost</div>
      <div class="rh-stat-value">{_fmt_cost(rec['estimated_total_cost_usd'])}</div>
      <div class="rh-stat-sub">per run · {rec['estimated_tokens_input']:,} input tokens</div>
    </div>
    <div class="rh-stat rh-stat-green">
      <div class="rh-stat-label">Potential Savings</div>
      <div class="rh-stat-value">{_savings_pct:.0f}%</div>
      <div class="rh-stat-sub">{_savings_label}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 4-Dimension Scorecard ────────────────────────────────────────────────
    _sim_note = " · Simulated" if sc.get("simulated") else " · Live benchmark"
    st.markdown(
        f'<div class="sh"><span class="sh-icon">🎯</span>'
        f'<h3>4-Dimension RDAB Scorecard{_sim_note}</h3></div>',
        unsafe_allow_html=True,
    )
    _dims = [
        ("Correctness",  sc["correctness"],  "50% weight — answer accuracy"),
        ("Code Quality", sc["code_quality"], "20% weight — vectorisation & naming"),
        ("Efficiency",   sc["efficiency"],   "15% weight — token & step budget"),
        ("Stat Validity",sc["stat_validity"],"15% weight — p-values & rigour"),
    ]
    _sc_items = ""
    for _name, _val, _hint in _dims:
        _bar_pct = int(_val * 100)
        _sc_items += f"""
  <div class="sc-item">
    <div class="sc-label">{_name}</div>
    <div class="sc-value">{_val:.0%}</div>
    <div class="sc-weight">{_hint}</div>
    <div class="sc-bar-bg"><div class="sc-bar-fill" style="width:{_bar_pct}%"></div></div>
  </div>"""
    st.markdown(f'<div class="scorecard">{_sc_items}</div>', unsafe_allow_html=True)

    # ── Dataset overview ─────────────────────────────────────────────────────
    with st.expander("📂  Dataset Overview", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Rows",      f"{stats['rows']:,}")
        d2.metric("Columns",   stats["columns"])
        d3.metric("Missing",   f"{stats['missing_pct']:.1f}%")
        d4.metric("File Size", f"{stats['file_size_kb']:.0f} KB")
        st.caption(f"Format: **{stats['file_format']}**  ·  Columns: {', '.join(stats['column_names'][:12])}")

    # ── Model comparison charts ──────────────────────────────────────────────
    st.markdown('<div class="sh"><span class="sh-icon">📈</span><h3>All Models — Comparison</h3></div>', unsafe_allow_html=True)

    results_raw = result["results"]
    for r in results_raw:
        sc_r = r.get("rdab_scorecard", {})
        r["rdab_score"]   = sc_r.get("rdab_score", 0)
        r["correctness"]  = sc_r.get("correctness", 0)
        r["code_quality"] = sc_r.get("code_quality", 0)
        r["efficiency"]   = sc_r.get("efficiency", 0)
        r["stat_validity"]= sc_r.get("stat_validity", 0)
        r["simulated"]    = sc_r.get("simulated", True)
    df_models = pd.DataFrame(results_raw)

    TIER_COLORS = {"premium": "#6366f1", "balanced": "#3b82f6", "economy": "#10b981"}
    CHART_BG    = "#f8fafc"
    GRID_COLOR  = "#e2e8f0"

    tab1, tab2, tab3, tab4 = st.tabs([
        "Score vs Cost",
        "Radar — Top 5",
        "Latency",
        "Full Table",
    ])



    with tab1:
        fig = px.scatter(
            df_models,
            x="estimated_total_cost_usd",
            y="rdab_score",
            text="display_name",
            color="tier",
            size=[30] * len(df_models),
            color_discrete_map=TIER_COLORS,
            labels={
                "estimated_total_cost_usd": "Estimated Cost per Run (USD)",
                "rdab_score": "RDAB Score",
                "tier": "Tier",
            },
            title="RDAB Score vs Estimated Cost — top-left corner is best value",
        )
        fig.update_traces(
            textposition="top center",
            textfont=dict(size=12, color="#1e293b", family="Inter"),
        )
        fig.add_annotation(
            x=rec["estimated_total_cost_usd"],
            y=rec["rdab_scorecard"]["rdab_score"],
            text="  ★ Recommended",
            showarrow=True, arrowhead=2, arrowcolor="#6366f1",
            font={"color": "#6366f1", "size": 12, "family": "Inter"},
            bgcolor="rgba(99,102,241,0.10)", bordercolor="#6366f1", borderwidth=1, borderpad=4,
        )
        fig.update_layout(
            height=440,
            plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
            font=dict(family="Inter", color="#1e293b"),
            title=dict(font=dict(size=13, color="#1e293b")),
            xaxis=dict(
                gridcolor=GRID_COLOR, zeroline=False,
                title_font=dict(size=12, color="#1e293b"),
                tickfont=dict(size=11, color="#475569"),
            ),
            yaxis=dict(
                gridcolor=GRID_COLOR, zeroline=False,
                title_font=dict(size=12, color="#1e293b"),
                tickfont=dict(size=11, color="#475569"),
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11, color="#1e293b"),
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top5   = df_models.nlargest(5, "rdab_score")
        cats   = ["Correctness", "Code Quality", "Efficiency", "Stat Validity"]
        colors = ["#6366f1", "#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
        radar  = go.Figure()
        for i, (_, row) in enumerate(top5.iterrows()):
            vals = [row["correctness"], row["code_quality"], row["efficiency"], row["stat_validity"]]
            radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                name=row["display_name"],
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.18,
                line_width=2,
            ))
        radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont_size=9, gridcolor=GRID_COLOR),
                angularaxis=dict(tickfont_size=11),
                bgcolor=CHART_BG,
            ),
            title=dict(text="4-Dimensional Scorecard — Top 5 Models", font_size=13),
            height=480,
            paper_bgcolor=CHART_BG,
            font_family="Inter",
            legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(radar, use_container_width=True)

        top5_tbl = top5[["display_name", "rdab_score", "correctness", "code_quality", "efficiency", "stat_validity"]].copy()
        top5_tbl.columns = ["Model", "RDAB", "Correctness", "Code Quality", "Efficiency", "Stat Validity"]
        for col in top5_tbl.columns[1:]:
            top5_tbl[col] = top5_tbl[col].map("{:.1%}".format)
        st.dataframe(top5_tbl, use_container_width=True, hide_index=True)

    with tab3:
        fig2 = px.bar(
            df_models.sort_values("latency_ms"),
            x="display_name", y="latency_ms",
            color="tier",
            color_discrete_map=TIER_COLORS,
            title="Estimated Response Latency per Model",
            labels={"latency_ms": "Latency (ms)", "display_name": ""},
            text="latency_ms",
        )
        fig2.update_traces(texttemplate="%{text:.0f} ms", textposition="outside")
        fig2.update_layout(
            height=420,
            plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
            font_family="Inter",
            xaxis=dict(tickangle=-35, gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR),
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=80),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.info(
            "**Note on scores:** Some models may have completed fewer evaluation questions than others "
            "(e.g. due to API rate limits or missing keys in Simulation mode). "
            "Scores for those models are preliminary and marked as simulated. "
            "Provide API keys for Live Mode to get full benchmark runs.",
            icon="ℹ️",
        )
        tbl = df_models[[
            "display_name", "provider", "tier",
            "rdab_score", "correctness", "code_quality", "efficiency", "stat_validity",
            "latency_ms", "estimated_total_cost_usd",
            "input_cost_per_1k", "output_cost_per_1k",
        ]].copy()
        tbl.columns = [
            "Model", "Provider", "Tier",
            "RDAB", "Correctness", "Code Quality", "Efficiency", "Stat Validity",
            "Latency (ms)", "Cost/Run ($)", "Input $/1K", "Output $/1K",
        ]
        for col in ["RDAB", "Correctness", "Code Quality", "Efficiency", "Stat Validity"]:
            tbl[col] = tbl[col].map("{:.1%}".format)
        tbl["Latency (ms)"] = tbl["Latency (ms)"].map("{:.0f}".format)
        tbl["Cost/Run ($)"] = tbl["Cost/Run ($)"].map("{:.6f}".format)
        st.dataframe(
            tbl,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Provider": st.column_config.TextColumn(width="small"),
                "Tier":     st.column_config.TextColumn(width="small"),
            },
        )

    # ── Recommended config ───────────────────────────────────────────────────
    st.markdown(
        '<div class="sh"><span class="sh-icon">⚙️</span><h3>Recommended Config — Ready to Copy</h3></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Drop this into your project to start using **{rec['display_name']}** immediately.")
    st.code(result["copyable_config"], language="json")

    with st.expander(f"Why {rec['display_name']}? — Strengths & Limitations", expanded=False):
        col_s, col_l = st.columns(2)
        with col_s:
            st.markdown("**Strengths**")
            for s in rec["strengths"]:
                st.markdown(f"- {s}")
        with col_l:
            st.markdown("**Limitations**")
            for lim in rec["limitations"]:
                st.markdown(f"- {lim}")


# ─── History & Alerts ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="sh"><span class="sh-icon">📜</span>'
    '<h3>History &amp; Alerts</h3></div>',
    unsafe_allow_html=True,
)
st.caption("Every evaluation is logged locally. Score drift (>10% drop from average) triggers an alert.")

_hist_tab, _drift_tab, _avg_tab = st.tabs(["Recent Evaluations", "Drift Events", "Model Averages"])

with _hist_tab:
    try:
        _evals = get_recent_evaluations(limit=50)
        _total = get_total_eval_count()

        import datetime as _dt
        # Stats strip
        _simulated_count = sum(1 for e in _evals if e.get("simulated", 1))
        st.markdown(
            f'<div class="obs-row">'
            f'<div class="obs-stat"><div class="obs-stat-label">Total Evals</div>'
            f'<div class="obs-stat-value">{_total}</div></div>'
            f'<div class="obs-stat"><div class="obs-stat-label">Showing</div>'
            f'<div class="obs-stat-value">{len(_evals)}</div></div>'
            f'<div class="obs-stat"><div class="obs-stat-label">Simulated</div>'
            f'<div class="obs-stat-value">{_simulated_count}</div></div>'
            f'<div class="obs-stat"><div class="obs-stat-label">Live</div>'
            f'<div class="obs-stat-value">{len(_evals) - _simulated_count}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if _evals:
            import pandas as _pd2
            _df_hist = _pd2.DataFrame(_evals)
            _df_hist["time"] = _df_hist["timestamp"].apply(
                lambda ts: _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            )
            _df_hist["rdab_score"] = _df_hist["rdab_score"].map("{:.3f}".format)
            _df_hist["cost_usd"]   = _df_hist["cost_usd"].map("{:.6f}".format)
            _df_hist["mode"]       = _df_hist.apply(
                lambda r: "live" if r.get("simulated", 1) == 0 else "sim", axis=1
            )
            _show_cols = ["time", "eval_id", "recommended_model", "rdab_score",
                          "cost_usd", "mode", "dataset_hash"]
            _df_hist = _df_hist[[c for c in _show_cols if c in _df_hist.columns]]
            _df_hist.columns = [c.replace("_", " ").title() for c in _df_hist.columns]
            st.dataframe(_df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("No evaluations logged yet. Run your first analysis above!")
    except Exception as _e:
        st.info(f"History unavailable: {_e}")

with _drift_tab:
    try:
        _drifts = get_recent_drift_events(limit=20)
        if _drifts:
            import datetime as _dt2
            for _d in _drifts:
                _ts = _dt2.datetime.fromtimestamp(_d["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                st.markdown(
                    f'<div class="drift-alert">'
                    f'<div class="drift-alert-title">Score Drift — {_d["model_id"]}</div>'
                    f'<div class="drift-alert-body">'
                    f'Detected at {_ts} &nbsp;·&nbsp; '
                    f'Current: <strong>{_d["current_score"]:.3f}</strong> &nbsp;·&nbsp; '
                    f'Historical avg: <strong>{_d["historical_avg"]:.3f}</strong> &nbsp;·&nbsp; '
                    f'Drop: <strong>{_d["drop_pct"]:.1f}%</strong> &nbsp;·&nbsp; '
                    f'Eval ID: <code>{_d["eval_id"]}</code>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No drift events recorded. All model scores are stable.")

        _webhook = bool(os.getenv("SLACK_WEBHOOK_URL", "").strip())
        if _webhook:
            st.caption("Slack alerts: **enabled** (SLACK_WEBHOOK_URL is set)")
        else:
            st.caption("Slack alerts: disabled — set `SLACK_WEBHOOK_URL` in your environment to enable.")
    except Exception as _e:
        st.info(f"Drift data unavailable: {_e}")

with _avg_tab:
    try:
        _avgs = get_model_averages()
        if _avgs:
            import pandas as _pd3
            import datetime as _dt3
            _df_avg = _pd3.DataFrame(_avgs)
            _df_avg["last_seen"] = _df_avg["last_seen"].apply(
                lambda ts: _dt3.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            )
            for _col in ["avg_rdab", "avg_correctness", "avg_code_quality",
                         "avg_efficiency", "avg_stat_validity"]:
                if _col in _df_avg.columns:
                    _df_avg[_col] = _df_avg[_col].map("{:.3f}".format)
            if "avg_cost" in _df_avg.columns:
                _df_avg["avg_cost"] = _df_avg["avg_cost"].map("{:.6f}".format)
            _df_avg.columns = [c.replace("avg_", "").replace("_", " ").title()
                               for c in _df_avg.columns]
            st.dataframe(_df_avg, use_container_width=True, hide_index=True)
            st.caption("Only models with ≥ 2 logged runs are shown.")
        else:
            st.info("Run at least 2 evaluations to see per-model averages.")
    except Exception as _e:
        st.info(f"Model averages unavailable: {_e}")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  CostGuard &nbsp;·&nbsp;
  Powered by <a href="https://github.com/patibandlavenkatamanideep/RealDataAgentBench" target="_blank">RealDataAgentBench</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/patibandlavenkatamanideep/CostGuard" target="_blank">GitHub</a>
  &nbsp;·&nbsp; MIT License
</div>
""", unsafe_allow_html=True)


def run() -> None:
    import os
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", __file__,
         "--server.port", os.getenv("STREAMLIT_PORT", "8501"),
         "--server.address", "0.0.0.0"],
        check=True,
    )
