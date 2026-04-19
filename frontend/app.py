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

# ─── Design system — 2-colour palette: #111827 (dark) + #4f46e5 (indigo) ──────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*,*::before,*::after{box-sizing:border-box}

html,body,[data-testid="stAppViewContainer"]{
    font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif !important;
    background:#f9fafb; color:#111827;
}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e5e7eb}
#MainMenu,footer,header{visibility:hidden}

/* Hero */
.hero{background:#111827;border-radius:12px;padding:2rem;margin-bottom:1.5rem;color:#fff}
.hero-title{font-size:1.75rem;font-weight:800;letter-spacing:-.5px;margin:0 0 .4rem;color:#fff}
.hero-sub{font-size:.88rem;color:#d1d5db;margin:0 0 1.25rem;line-height:1.6;max-width:540px}
.hero-pills{display:flex;gap:.4rem;flex-wrap:wrap}
.hero-pill{background:transparent;border:1px solid #4b5563;border-radius:999px;padding:.18rem .65rem;font-size:.7rem;font-weight:500;color:#d1d5db}

/* Mode badge */
.mode-badge{display:inline-flex;align-items:center;gap:.35rem;font-size:.74rem;font-weight:600;padding:.22rem .75rem;border-radius:999px;margin-bottom:1rem}
.mode-live{background:#eef2ff;color:#4338ca;border:1px solid #c7d2fe}
.mode-sim{background:#f3f4f6;color:#6b7280;border:1px solid #e5e7eb}

/* Result hero */
.result-hero{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:1.75rem;margin-bottom:1.5rem}
.rh-eyebrow{font-size:.63rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#4f46e5;margin-bottom:.25rem}
.rh-model{font-size:1.875rem;font-weight:800;color:#111827;line-height:1.15;margin-bottom:.2rem}
.rh-meta{font-size:.74rem;color:#6b7280;font-weight:500;margin-bottom:1.25rem}
.rh-stats{display:flex;gap:.875rem;flex-wrap:wrap;border-top:1px solid #f3f4f6;padding-top:1.25rem}
.rh-stat{flex:1;min-width:130px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:1rem 1.25rem}
.rh-stat-label{font-size:.63rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#6b7280;margin-bottom:.3rem}
.rh-stat-value{font-size:1.75rem;font-weight:800;color:#4f46e5;line-height:1.1;margin-bottom:.15rem}
.rh-stat-sub{font-size:.7rem;color:#4b5563}

/* Scorecard */
.scorecard{display:flex;gap:.875rem;flex-wrap:wrap;margin:0 0 1.5rem}
.sc-item{flex:1;min-width:130px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:1rem 1.25rem;position:relative;overflow:hidden}
.sc-label{font-size:.63rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#6b7280;margin-bottom:.35rem}
.sc-value{font-size:1.5rem;font-weight:800;color:#111827;line-height:1;margin-bottom:.15rem}
.sc-weight{font-size:.68rem;color:#6b7280}
.sc-bar-bg{position:absolute;bottom:0;left:0;right:0;height:3px;background:#f3f4f6}
.sc-bar-fill{position:absolute;bottom:0;left:0;height:3px;background:#4f46e5}

/* Section heading */
.sh{display:flex;align-items:center;gap:.5rem;border-bottom:1.5px solid #e5e7eb;padding-bottom:.5rem;margin:1.75rem 0 1rem}
.sh h3{font-size:.95rem;font-weight:700;color:#111827;margin:0}

/* Steps */
.steps{display:flex;gap:1rem;flex-wrap:wrap}
.step{flex:1;min-width:100px;display:flex;gap:.5rem;align-items:flex-start}
.step-num{width:20px;height:20px;min-width:20px;background:#eef2ff;color:#4f46e5;border-radius:50%;font-size:.63rem;font-weight:800;display:flex;align-items:center;justify-content:center;margin-top:1px}
.step-label{font-size:.8rem;font-weight:600;color:#111827;display:block}
.step-desc{font-size:.73rem;color:#6b7280}

/* Buttons */
.stButton>button[kind="primary"]{background:#4f46e5 !important;color:#fff !important;border:none !important;border-radius:8px;font-weight:700;font-size:.95rem}
.stButton>button[kind="primary"]:hover{background:#4338ca !important}
.stButton>button[kind="secondary"]{border:1px solid #e5e7eb !important;border-radius:8px;font-size:.8rem;font-weight:600;color:#374151 !important;background:#fff !important}
.stButton>button[kind="secondary"]:hover{border-color:#4f46e5 !important;color:#4f46e5 !important}

/* File uploader */
[data-testid="stFileUploader"]{border:1.5px dashed #d1d5db;border-radius:8px}
[data-testid="stFileUploader"]:hover{border-color:#4f46e5}

/* Sidebar */
.sb-brand{font-size:1rem;font-weight:800;color:#111827;margin:0}
.sb-tagline{font-size:.7rem;color:#6b7280;margin:0 0 1.25rem}
.sb-section{font-size:.63rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#6b7280;margin:1rem 0 .35rem}
.sb-mode-live{background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;padding:.6rem .875rem;margin-bottom:.5rem;font-size:.78rem;font-weight:600;color:#3730a3}
.sb-mode-sim{background:#f3f4f6;border:1px solid #e5e7eb;border-radius:8px;padding:.6rem .875rem;margin-bottom:.5rem;font-size:.78rem;color:#6b7280}
.sb-provider{display:flex;align-items:center;justify-content:space-between;margin-bottom:.15rem}
.sb-provider-label{font-size:.78rem;font-weight:600;color:#374151}
.sb-provider-link{font-size:.68rem;color:#4f46e5;text-decoration:none;font-weight:600}
.sb-provider-link:hover{text-decoration:underline}
.sb-dot-live{display:inline-block;width:7px;height:7px;background:#4f46e5;border-radius:50%;margin-right:.35rem;vertical-align:middle}
.sb-dot-sim{display:inline-block;width:7px;height:7px;background:#d1d5db;border-radius:50%;margin-right:.35rem;vertical-align:middle}
.sb-live-summary{font-size:.72rem;color:#4338ca;font-weight:600;margin:.5rem 0 0}
.sb-sim-summary{font-size:.72rem;color:#6b7280;margin:.5rem 0 0}

/* Empty state */
.empty{text-align:center;padding:2.5rem 1rem}
.empty-icon{font-size:2.5rem;margin-bottom:.75rem}
.empty-title{font-size:1rem;font-weight:700;color:#374151;margin-bottom:.3rem}
.empty-sub{font-size:.82rem;color:#6b7280}

/* Confidence */
.conf-high{color:#4f46e5;font-weight:800}
.conf-mid{color:#374151;font-weight:800}
.conf-low{color:#6b7280;font-weight:800}

/* Output text */
.output-text{font-size:.84rem;color:#374151;line-height:1.65;white-space:pre-wrap}

/* Case study */
.casestudy{background:#111827;border-radius:12px;padding:2rem;margin:1.5rem 0}
.cs-eyebrow{font-size:.63rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:#a5b4fc;margin-bottom:.5rem}
.cs-title{font-size:1.4rem;font-weight:800;color:#fff;margin-bottom:.5rem;letter-spacing:-.3px}
.cs-body{font-size:.86rem;color:#d1d5db;line-height:1.65;margin-bottom:1.25rem}
.cs-stats{display:flex;gap:.875rem;flex-wrap:wrap}
.cs-stat{flex:1;min-width:110px;background:rgba(255,255,255,.06);border:1px solid #374151;border-radius:8px;padding:.85rem 1rem}
.cs-stat-val{font-size:1.5rem;font-weight:800;color:#a5b4fc;line-height:1;margin-bottom:.15rem}
.cs-stat-lbl{font-size:.68rem;color:#9ca3af}

/* Spend & drift alerts — single indigo-tinted style */
.spend-alert{background:#eef2ff;border:1px solid #c7d2fe;border-left:3px solid #4f46e5;border-radius:8px;padding:.85rem 1rem;margin-bottom:.65rem}
.spend-alert-title{font-size:.8rem;font-weight:700;color:#3730a3;margin-bottom:.15rem}
.spend-alert-body{font-size:.77rem;color:#4338ca;line-height:1.5}
.spend-ok{background:#f9fafb;border:1px solid #e5e7eb;border-left:3px solid #4f46e5;border-radius:8px;padding:.85rem 1rem;margin-bottom:.65rem}
.spend-ok-body{font-size:.8rem;color:#374151;line-height:1.5}
.drift-alert{background:#f9fafb;border:1px solid #e5e7eb;border-left:3px solid #111827;border-radius:8px;padding:.85rem 1rem;margin-bottom:.65rem}
.drift-alert-title{font-size:.8rem;font-weight:700;color:#111827;margin-bottom:.15rem}
.drift-alert-body{font-size:.77rem;color:#4b5563;line-height:1.5}

/* Obs stats */
.obs-stat{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:.75rem 1rem;text-align:center;flex:1;min-width:90px}
.obs-stat-label{font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#6b7280}
.obs-stat-value{font-size:1.25rem;font-weight:800;color:#111827}
.obs-row{display:flex;gap:.75rem;margin:.75rem 0 1.25rem;flex-wrap:wrap}

/* Footer */
.footer{text-align:center;color:#6b7280;font-size:.75rem;padding:1.5rem 0 .5rem;border-top:1px solid #f3f4f6;margin-top:2rem}
.footer a{color:#4f46e5;text-decoration:none;font-weight:600}
.footer a:hover{text-decoration:underline}
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

# Key state: use session_state so Clear button actually works
_KEY_FIELDS = ["sb_anthropic", "sb_openai", "sb_groq", "sb_gemini", "sb_xai"]
for _k in _KEY_FIELDS:
    if _k not in st.session_state:
        st.session_state[_k] = ""

with st.sidebar:
    st.markdown('<p class="sb-brand">CostGuard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sb-tagline">LLM Cost Intelligence</p>', unsafe_allow_html=True)

    # ── Mode indicator ────────────────────────────────────────────────────────
    _any_key_ss = any(st.session_state[k].strip() for k in _KEY_FIELDS)
    if _any_key_ss:
        _live_providers_ss = [n for n, k in [
            ("Anthropic", "sb_anthropic"), ("OpenAI", "sb_openai"),
            ("Groq", "sb_groq"), ("Google", "sb_gemini"), ("xAI", "sb_xai"),
        ] if st.session_state[k].strip()]
        st.markdown(
            f'<div class="sb-mode-live">● Live Mode — {", ".join(_live_providers_ss)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sb-mode-sim">◎ Simulation Mode — no API keys entered</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Task config ───────────────────────────────────────────────────────────
    st.markdown('<p class="sb-section">Task Description</p>', unsafe_allow_html=True)
    task_description = st.text_area(
        "Task",
        value="Analyze this dataset and answer questions about it.",
        height=68,
        help="Describe what you want to do — e.g. 'Predict churn', 'Find outliers'.",
        label_visibility="collapsed",
    )

    st.markdown('<p class="sb-section">Evaluation Depth</p>', unsafe_allow_html=True)
    num_questions = st.slider(
        "Questions", min_value=1, max_value=10, value=5,
        help="More questions = more accurate benchmark, slightly longer runtime.",
        label_visibility="collapsed",
    )

    st.divider()

    # ── API Keys — always visible ─────────────────────────────────────────────
    st.markdown('<p class="sb-section">API Keys — enables Live Mode</p>', unsafe_allow_html=True)
    st.caption("Session-only. Never stored. One key is enough — the rest fall back to simulation.")

    _providers = [
        ("Anthropic", "sb_anthropic", "sk-ant-...", "https://console.anthropic.com/settings/keys",  "Claude models"),
        ("OpenAI",    "sb_openai",    "sk-...",      "https://platform.openai.com/api-keys",         "GPT models"),
        ("Groq",      "sb_groq",      "gsk_...",     "https://console.groq.com/keys",                "Llama models"),
        ("Google",    "sb_gemini",    "AIza...",     "https://aistudio.google.com/apikey",           "Gemini models"),
        ("xAI",       "sb_xai",       "xai-...",     "https://console.x.ai/",                        "Grok models"),
    ]

    for _pname, _pkey, _placeholder, _plink, _pmodels in _providers:
        _is_set = bool(st.session_state[_pkey].strip())
        _dot = '<span class="sb-dot-live"></span>' if _is_set else '<span class="sb-dot-sim"></span>'
        st.markdown(
            f'<div class="sb-provider">'
            f'<span class="sb-provider-label">{_dot}{_pname}</span>'
            f'<a class="sb-provider-link" href="{_plink}" target="_blank">Get key →</a>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.text_input(
            f"{_pname} key",
            type="password",
            placeholder=_placeholder,
            key=_pkey,
            label_visibility="collapsed",
            help=f"{_pmodels} · {_plink}",
        )

    if st.button("Clear all keys", use_container_width=True):
        for _k in _KEY_FIELDS:
            st.session_state[_k] = ""
        st.rerun()

    # ── Build SessionKeys from state ──────────────────────────────────────────
    session_keys = SessionKeys(
        anthropic_api_key=st.session_state["sb_anthropic"].strip() or None,
        openai_api_key=st.session_state["sb_openai"].strip() or None,
        groq_api_key=st.session_state["sb_groq"].strip() or None,
        gemini_api_key=st.session_state["sb_gemini"].strip() or None,
        xai_api_key=st.session_state["sb_xai"].strip() or None,
    )
    any_key = session_keys.has_any_key()

    st.divider()
    st.markdown('<p class="sb-section">Models Covered</p>', unsafe_allow_html=True)
    st.caption("Claude Sonnet · Opus · Haiku\nGPT-5 · 4.1 · 4o · 4o-mini\nGemini 2.5 Pro · Flash\nLlama 3.3 70B · Grok-3")


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">CostGuard</div>
  <p class="hero-sub">
    Benchmark 12 LLMs on your actual data. Get a model recommendation with exact cost estimates — no guesswork, no sign-up.
  </p>
  <div class="hero-pills">
    <span class="hero-pill">12 Models</span>
    <span class="hero-pill">4-Dimensional RDAB Scoring</span>
    <span class="hero-pill">Simulation — No API Keys Needed</span>
    <span class="hero-pill">Data never stored</span>
    <span class="hero-pill">Self-host with Docker</span>
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
    <div><span class="step-label">Benchmark</span><span class="step-desc">12 models on your data</span></div>
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
    st.caption(f"**{filename}** — {size_str}")

    if not any_key:
        st.markdown("""
<div style="background:#eef2ff;border:1px solid #c7d2fe;border-left:3px solid #4f46e5;border-radius:8px;padding:.75rem 1rem;margin:.5rem 0;font-size:.82rem;color:#3730a3">
  <strong>Simulation Mode active.</strong>
  Scores are from the RDAB benchmark leaderboard — not from live inference on your data.<br>
  Enter at least one API key in the sidebar to run real models on your actual file.
</div>""", unsafe_allow_html=True)

    mode_label = (
        "Running live benchmark across 12 models — this takes 1–3 minutes…"
        if any_key
        else "Running simulation benchmark across 12 models — usually under 15 seconds…"
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

    _conf = rec.get("confidence_score", 0.5)
    _conf_expl = rec.get("confidence_explanation", "")
    _conf_cls = "conf-high" if _conf >= 0.70 else ("conf-mid" if _conf >= 0.50 else "conf-low")

    _mode_pill = (
        f'<span class="mode-badge mode-live">● Live — {", ".join(live_providers)}</span>'
        if mode == "live"
        else '<span class="mode-badge mode-sim">◎ Simulation — calibrated RDAB scores</span>'
    )

    # ── Result hero ──────────────────────────────────────────────────────────
    st.markdown(_mode_pill, unsafe_allow_html=True)
    st.markdown(f"""
<div class="result-hero">
  <div class="rh-eyebrow">Recommended Model</div>
  <div class="rh-model">{rec['display_name']}</div>
  <div class="rh-meta">{rec['provider'].upper()} &nbsp;·&nbsp; {rec['tier'].upper()} TIER &nbsp;·&nbsp; {rec['latency_ms']:.0f} ms avg latency</div>
  <div class="rh-stats">
    <div class="rh-stat">
      <div class="rh-stat-label">RDAB Score</div>
      <div class="rh-stat-value">{sc['rdab_score']:.0%}</div>
      <div class="rh-stat-sub">Composite benchmark — 12 models</div>
    </div>
    <div class="rh-stat">
      <div class="rh-stat-label">Cost per Run</div>
      <div class="rh-stat-value">{_fmt_cost(rec['estimated_total_cost_usd'])}</div>
      <div class="rh-stat-sub">{rec['estimated_tokens_input']:,} input tokens</div>
    </div>
    <div class="rh-stat">
      <div class="rh-stat-label">Potential Savings</div>
      <div class="rh-stat-value">{_savings_pct:.0f}%</div>
      <div class="rh-stat-sub">{_savings_label}</div>
    </div>
    <div class="rh-stat">
      <div class="rh-stat-label">Confidence</div>
      <div class="rh-stat-value"><span class="{_conf_cls}">{_conf:.0%}</span></div>
      <div class="rh-stat-sub">{_conf_expl}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 4-Dimension Scorecard ────────────────────────────────────────────────
    _sim_note = " · Simulated" if sc.get("simulated") else " · Live"
    st.markdown(
        f'<div class="sh"><h3>RDAB Scorecard{_sim_note}</h3></div>',
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
    st.info(
        "**Why is Stat Validity universally low?** "
        "This dimension measures whether a model reports uncertainty correctly — "
        "citing p-values, confidence intervals, and avoiding overconfident claims. "
        "All 12 models average **55.8%** here — versus a human expert baseline of **81.3%** "
        "on the same 5 tasks. That 25-point gap is a real model capability difference, "
        "confirmed across 276 runs in the RDAB benchmark: "
        "LLMs compute accurately but rarely add statistical rigour unprompted. "
        "It does **not** affect the model's ability to answer your data questions correctly — "
        "and it's exactly the kind of gap a cost-optimisation tool should surface.",
        icon="ℹ️",
    )

    # ── Dataset overview ─────────────────────────────────────────────────────
    with st.expander("📂  Dataset Overview", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Rows",      f"{stats['rows']:,}")
        d2.metric("Columns",   stats["columns"])
        d3.metric("Missing",   f"{stats['missing_pct']:.1f}%")
        d4.metric("File Size", f"{stats['file_size_kb']:.0f} KB")
        st.caption(f"Format: **{stats['file_format']}**  ·  Columns: {', '.join(stats['column_names'][:12])}")

    # ── Model comparison charts ──────────────────────────────────────────────
    st.markdown('<div class="sh"><h3>All Models — Comparison</h3></div>', unsafe_allow_html=True)

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

    TIER_COLORS = {"premium": "#4f46e5", "balanced": "#818cf8", "economy": "#c7d2fe"}
    CHART_BG    = "#ffffff"
    GRID_COLOR  = "#f3f4f6"

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Score vs Cost",
        "Radar — Top 5",
        "Latency",
        "Full Table",
        "Model Outputs",
    ])



    with tab1:
        # Clip zero costs so log scale doesn't break; use a realistic floor ($0.000001)
        _COST_FLOOR = 1e-6
        df_plot = df_models.copy()
        df_plot["plot_cost"] = df_plot["estimated_total_cost_usd"].clip(lower=_COST_FLOOR)
        _rec_cost  = max(rec["estimated_total_cost_usd"], _COST_FLOOR)
        _rec_score = rec["rdab_scorecard"]["rdab_score"]

        fig = px.scatter(
            df_plot,
            x="plot_cost",
            y="rdab_score",
            text="display_name",
            color="tier",
            size=[30] * len(df_plot),
            color_discrete_map=TIER_COLORS,
            labels={
                "plot_cost": "Cost per Run (USD) — log scale",
                "rdab_score": "RDAB Score",
                "tier": "Tier",
            },
            title="RDAB Score vs Estimated Cost — top-left corner is best value",
        )
        fig.update_traces(
            textposition="top center",
            textfont=dict(size=12, color="#111827", family="Inter"),
        )
        # Explicit star marker so the recommended model is always visible
        # even if the annotation arrow lands slightly off due to floating-point.
        fig.add_trace(go.Scatter(
            x=[_rec_cost],
            y=[_rec_score],
            mode="markers+text",
            marker=dict(symbol="star", size=20, color="#6366f1",
                        line=dict(color="#ffffff", width=1.5)),
            text=["★ Recommended"],
            textposition="top center",
            textfont=dict(size=12, color="#6366f1", family="Inter"),
            name="Recommended",
            showlegend=True,
        ))
        fig.update_xaxes(
            type="log",
            title="Cost per Run (USD) — log scale",
            gridcolor=GRID_COLOR, zeroline=False,
            title_font=dict(size=12, color="#111827"),
            tickfont=dict(size=11, color="#6b7280"),
            tickvals=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            ticktext=["$0.000001", "$0.00001", "$0.0001", "$0.001", "$0.01", "$0.10"],
        )
        fig.update_layout(
            height=460,
            plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
            font=dict(family="Inter", color="#111827"),
            title=dict(font=dict(size=13, color="#111827")),
            yaxis=dict(
                gridcolor=GRID_COLOR, zeroline=False,
                title_font=dict(size=12, color="#111827"),
                tickfont=dict(size=11, color="#6b7280"),
                tickformat=".0%",
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11, color="#111827"),
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top5   = df_models.nlargest(5, "rdab_score")
        cats   = ["Correctness", "Code Quality", "Efficiency", "Stat Validity"]
        colors = ["#4f46e5", "#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe"]
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
                opacity=0.35,
                line_width=2.5,
            ))
        radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    tickfont=dict(size=10, color="#6b7280"),
                    gridcolor=GRID_COLOR,
                    tickformat=".0%",
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color="#111827"),
                ),
                bgcolor=CHART_BG,
            ),
            title=dict(text="4-Dimensional Scorecard — Top 5 Models", font=dict(size=13, color="#111827")),
            height=480,
            paper_bgcolor=CHART_BG,
            font=dict(family="Inter", color="#111827"),
            legend=dict(
                orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
                font=dict(size=11, color="#111827"),
            ),
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
        fig2.update_traces(
            texttemplate="%{text:.0f} ms",
            textposition="outside",
            textfont=dict(size=11, color="#111827"),
        )
        fig2.update_layout(
            height=420,
            plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
            font=dict(family="Inter", color="#111827"),
            title=dict(font=dict(size=13, color="#111827")),
            xaxis=dict(
                tickangle=-35, gridcolor=GRID_COLOR,
                tickfont=dict(size=11, color="#111827"),
                title_font=dict(size=12, color="#111827"),
            ),
            yaxis=dict(
                gridcolor=GRID_COLOR,
                tickfont=dict(size=11, color="#6b7280"),
                title_font=dict(size=12, color="#111827"),
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=100),
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
            "simulated",
        ]].copy()
        tbl["Mode"] = tbl["simulated"].map(lambda s: "Simulation" if s else "Live")
        tbl = tbl.drop(columns=["simulated"])
        tbl.columns = [
            "Model", "Provider", "Tier",
            "RDAB", "Correctness", "Code Quality", "Efficiency", "Stat Validity",
            "Latency (ms)", "Cost/Run ($)", "Input $/1K", "Output $/1K", "Mode",
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
                "Mode":     st.column_config.TextColumn(width="small"),
            },
        )

    with tab5:
        _questions = result.get("questions_asked", [])
        _live_results = [r for r in result["results"] if r.get("actual_output")]
        if not _live_results:
            st.markdown("""
<div class="empty">
  <div class="empty-icon">💬</div>
  <div class="empty-title">No actual outputs available</div>
  <div class="empty-sub">
    This evaluation ran in <strong>Simulation Mode</strong> — no models were called, so there are no real responses to show.<br><br>
    Add at least one provider API key in the sidebar and re-run to see what each model actually says about your data.
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="sh"><h3>What each model said — {len(_live_results)} live model{"s" if len(_live_results) != 1 else ""}</h3></div>',
                unsafe_allow_html=True,
            )
            if _questions:
                with st.expander("Questions sent to every model", expanded=False):
                    for _qi, _q in enumerate(_questions, 1):
                        st.markdown(f"**{_qi}.** {_q}")

            _sorted_live = sorted(_live_results, key=lambda r: r.get("rdab_scorecard", {}).get("rdab_score", 0), reverse=True)
            for _lr in _sorted_live:
                _lr_sc = _lr.get("rdab_scorecard", {})
                _lr_conf = _lr.get("confidence_score", 0.5)
                _lr_conf_cls = "conf-high" if _lr_conf >= 0.70 else ("conf-mid" if _lr_conf >= 0.50 else "conf-low")
                with st.expander(
                    f"{_lr['display_name']}  ·  RDAB {_lr_sc.get('rdab_score', 0):.0%}  ·  {_fmt_cost(_lr['estimated_total_cost_usd'])}/run",
                    expanded=(_lr["model_id"] == rec["model_id"]),
                ):
                    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                    _mc1.metric("RDAB Score", f"{_lr_sc.get('rdab_score', 0):.0%}")
                    _mc2.metric("Correctness", f"{_lr_sc.get('correctness', 0):.0%}")
                    _mc3.metric("Cost/Run", _fmt_cost(_lr["estimated_total_cost_usd"]))
                    _mc4.metric("Confidence", f"{_lr_conf:.0%}")
                    st.markdown("**Model response:**")
                    st.markdown(
                        f'<div class="output-text">{_lr["actual_output"]}</div>',
                        unsafe_allow_html=True,
                    )
                    if _lr["model_id"] == rec["model_id"]:
                        st.success("★ This is the recommended model", icon="✅")

    # ── Recommended config ───────────────────────────────────────────────────
    st.markdown(
        '<div class="sh"><h3>Recommended Config</h3></div>',
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


# ─── Case Study ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="casestudy">
  <div class="cs-eyebrow">Real Case Study · CostGuard on CostGuard</div>
  <div class="cs-title">GPT-4.1 over GPT-5 — same quality, 87% cheaper</div>
  <div class="cs-body">
    We ran <strong>276 RDAB evaluations</strong> across 23 data analysis tasks and 12 models — the same benchmark
    that powers every recommendation in this tool. CostGuard recommended <strong>GPT-4.1</strong> over <strong>GPT-5</strong>
    for structured data analysis. GPT-4.1 scored <strong>88% RDAB</strong> (vs 79% for GPT-5)
    and cost <strong>$0.0140 per run</strong> versus <strong>$0.1053 for GPT-5</strong>.<br><br>
    A team making <strong>10,000 calls/month</strong> saves <strong>$912/month</strong> — 87% cost reduction —
    while actually getting <em>better</em> answers. GPT-5's extra cost buys no measurable quality gain
    on tabular data analysis tasks.
  </div>
  <div class="cs-stats">
    <div class="cs-stat">
      <div class="cs-stat-val">276</div>
      <div class="cs-stat-lbl">RDAB evaluation runs</div>
    </div>
    <div class="cs-stat">
      <div class="cs-stat-val">87%</div>
      <div class="cs-stat-lbl">cost reduction per run</div>
    </div>
    <div class="cs-stat">
      <div class="cs-stat-val">+9pt</div>
      <div class="cs-stat-lbl">RDAB score improvement</div>
    </div>
    <div class="cs-stat">
      <div class="cs-stat-val">$912</div>
      <div class="cs-stat-lbl">saved/month at 10K calls</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── History & Alerts ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="sh">'
    '<h3>History &amp; Alerts</h3></div>',
    unsafe_allow_html=True,
)
st.caption("Every evaluation is logged locally. Score drift (>10% drop from average) triggers an alert.")

_hist_tab, _drift_tab, _avg_tab, _spend_tab = st.tabs(["Recent Evaluations", "Drift Events", "Model Averages", "Spend Monitor"])

with _hist_tab:
    try:
        _evals = get_recent_evaluations(limit=50)
        _total = get_total_eval_count()

        import datetime as _dt
        _simulated_count = sum(1 for e in _evals if e.get("simulated", 1))
        _live_count = len(_evals) - _simulated_count
        if _live_count == 0 and _total > 0:
            st.info(
                "All logged evaluations used **Simulation Mode**. "
                "Add at least one provider API key and run a live benchmark "
                "to populate the Live column — recommended before public launch.",
                icon="💡",
            )
        st.markdown(
            f'<div class="obs-row">'
            f'<div class="obs-stat"><div class="obs-stat-label">Total Evals</div>'
            f'<div class="obs-stat-value">{_total}</div></div>'
            f'<div class="obs-stat"><div class="obs-stat-label">Showing</div>'
            f'<div class="obs-stat-value">{len(_evals)}</div></div>'
            f'<div class="obs-stat"><div class="obs-stat-label">Simulated</div>'
            f'<div class="obs-stat-value">{_simulated_count}</div></div>'
            f'<div class="obs-stat"><div class="obs-stat-label">Live</div>'
            f'<div class="obs-stat-value">{_live_count}</div></div>'
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

with _spend_tab:
    st.markdown("#### Spend Monitor — Project your monthly LLM cost")
    st.caption(
        "Enter your current model and usage volume. CostGuard projects your monthly spend "
        "and alerts you when a cheaper alternative scored equally well on this benchmark."
    )

    from evaluation.pricing import MODELS as _ALL_MODELS  # noqa: E402

    _model_ids = sorted(_ALL_MODELS.keys())
    _model_display = {mid: _ALL_MODELS[mid].display_name for mid in _model_ids}

    _sm_col1, _sm_col2 = st.columns([2, 1])
    with _sm_col1:
        _current_model_id = st.selectbox(
            "Your current model",
            options=_model_ids,
            format_func=lambda mid: _model_display[mid],
            index=_model_ids.index("gpt-5") if "gpt-5" in _model_ids else 0,
            key="spend_model",
        )
    with _sm_col2:
        _daily_calls = st.number_input(
            "Calls per day",
            min_value=1, max_value=10_000_000, value=1000, step=100,
            key="spend_calls",
        )

    _avg_input_tokens  = st.slider("Avg input tokens per call",  min_value=100, max_value=50000, value=2000, step=100, key="spend_in_tok")
    _avg_output_tokens = st.slider("Avg output tokens per call", min_value=50,  max_value=10000, value=512,  step=50,  key="spend_out_tok")

    _current_pricing = _ALL_MODELS[_current_model_id]
    _cost_per_call = _current_pricing.estimate_cost(_avg_input_tokens, _avg_output_tokens)
    _monthly_cost  = _cost_per_call * _daily_calls * 30

    _sm_c1, _sm_c2, _sm_c3 = st.columns(3)
    _sm_c1.metric("Cost per call", f"${_cost_per_call:.5f}")
    _sm_c2.metric("Daily spend",   f"${_cost_per_call * _daily_calls:.2f}")
    _sm_c3.metric("Monthly spend", f"${_monthly_cost:,.2f}")

    st.markdown("---")
    st.markdown("**Cheaper alternatives from this benchmark:**")

    _alternatives_found = False
    for _alt_id, _alt_p in sorted(_ALL_MODELS.items(), key=lambda x: x[1].estimate_cost(_avg_input_tokens, _avg_output_tokens)):
        if _alt_id == _current_model_id:
            continue
        _alt_cost_per_call = _alt_p.estimate_cost(_avg_input_tokens, _avg_output_tokens)
        if _alt_cost_per_call >= _cost_per_call:
            continue
        _alt_monthly = _alt_cost_per_call * _daily_calls * 30
        _savings_mo  = _monthly_cost - _alt_monthly
        _savings_pct_s = (1 - _alt_cost_per_call / _cost_per_call) * 100 if _cost_per_call > 0 else 0

        # Check if eval result has a score for this model to judge quality parity
        _eval_result = st.session_state.get("result")
        _alt_rdab = None
        _curr_rdab = None
        if _eval_result:
            for _mr in _eval_result.get("results", []):
                if _mr["model_id"] == _alt_id:
                    _alt_rdab = _mr.get("rdab_scorecard", {}).get("rdab_score")
                if _mr["model_id"] == _current_model_id:
                    _curr_rdab = _mr.get("rdab_scorecard", {}).get("rdab_score")

        _quality_note = ""
        if _alt_rdab is not None and _curr_rdab is not None:
            _rdab_diff = (_alt_rdab - _curr_rdab) * 100
            if _rdab_diff >= 0:
                _quality_note = f" · RDAB score is {_rdab_diff:.1f}pt **better** ✓"
            elif _rdab_diff >= -5:
                _quality_note = f" · RDAB score within {abs(_rdab_diff):.1f}pt (acceptable trade-off)"
            else:
                _quality_note = f" · RDAB score is {abs(_rdab_diff):.1f}pt lower — review before switching"

        if _savings_mo >= 1.0:
            _alert_cls = "spend-alert" if (_alt_rdab is None or (_curr_rdab is not None and _alt_rdab < _curr_rdab - 0.05)) else "spend-ok"
            _icon = "⚠️ Switch opportunity" if _alert_cls == "spend-alert" else "✅ Recommended switch"
            st.markdown(
                f'<div class="{_alert_cls}">'
                f'<div class="spend-alert-title">{_icon} → {_alt_p.display_name}</div>'
                f'<div class="spend-alert-body">'
                f'<strong>${_savings_mo:,.0f}/month</strong> savings ({_savings_pct_s:.0f}% cheaper)'
                f' at {_daily_calls:,} calls/day{_quality_note}. '
                f'Current: <strong>${_monthly_cost:,.0f}/mo</strong> → New: <strong>${_alt_monthly:,.0f}/mo</strong>.'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            _alternatives_found = True
            if _alternatives_found and list(_ALL_MODELS.keys()).index(_alt_id) >= 3:
                break  # show top 3 alternatives max

    if not _alternatives_found:
        st.markdown(
            '<div class="spend-ok"><div class="spend-ok-body">'
            '✅ You\'re already using one of the most cost-effective models for this call volume. '
            'Run an evaluation with your data to verify quality.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "Savings are projected based on token pricing only. "
        "Run a full evaluation with your API keys to validate quality before switching."
    )


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  CostGuard &nbsp;·&nbsp;
  Powered by <a href="https://github.com/patibandlavenkatamanideep/RealDataAgentBench" target="_blank">RealDataAgentBench</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/patibandlavenkatamanideep/CostGuard" target="_blank">GitHub</a>
  &nbsp;·&nbsp; MIT License
  &nbsp;·&nbsp; <a href="/Privacy_Policy" target="_self">Privacy Policy</a>
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
