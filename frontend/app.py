"""
CostGuard — Streamlit Dashboard

Calls the evaluation engine directly (no HTTP server required).
Works in Simulation Mode with zero API keys and in Live Mode with any
provider key entered in the sidebar.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on the path when running via `streamlit run frontend/app.py`
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.models import EvalMode, EvalResponse, SessionKeys  # noqa: E402
from evaluation.engine import run_evaluation  # noqa: E402


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CostGuard — LLM Cost Estimator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Minimal Professional CSS ─────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* ---- Global ---- */
    [data-testid="stAppViewContainer"] { background: #f8fafc; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }

    /* ---- Hide Streamlit chrome ---- */
    #MainMenu, footer, header { visibility: hidden; }

    /* ---- Header ---- */
    .cg-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .cg-header .cg-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #1a202c;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .cg-header .cg-sub {
        font-size: 0.9rem;
        color: #64748b;
        margin: 0.1rem 0 0 0;
    }

    /* ---- Mode pill ---- */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 1rem;
    }
    .pill-live { background: #dcfce7; color: #166534; }
    .pill-sim  { background: #fef9c3; color: #854d0e; }

    /* ---- Recommendation card ---- */
    .rec-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #6366f1;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .rec-card h3 {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a202c;
        margin: 0 0 0.4rem 0;
    }
    .rec-card p { color: #475569; font-size: 0.9rem; margin: 0; line-height: 1.6; }

    /* ---- Tier badges ---- */
    .badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 9px;
        border-radius: 999px;
    }
    .badge-premium  { background: #ede9fe; color: #5b21b6; }
    .badge-balanced { background: #dbeafe; color: #1e40af; }
    .badge-economy  { background: #dcfce7; color: #14532d; }

    /* ---- Primary button ---- */
    .stButton > button[kind="primary"] {
        background: #6366f1;
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.25rem;
        transition: background 0.15s;
    }
    .stButton > button[kind="primary"]:hover { background: #4f46e5; }

    /* ---- Subtle section label ---- */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="cg-header">
    <div>
        <p class="cg-title">🛡️ CostGuard</p>
        <p class="cg-sub">Upload a dataset → instantly find the best LLM with exact cost estimates</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_cost(usd: float) -> str:
    if usd < 0.0001:
        return f"${usd * 1e6:.2f} µ"
    if usd < 0.01:
        return f"${usd * 1000:.3f} m"
    return f"${usd:.4f}"


def _run_evaluation_sync(
    file_bytes: bytes,
    filename: str,
    task_description: str,
    num_questions: int,
    session_keys: SessionKeys,
) -> dict:
    """
    Call the async evaluation engine synchronously.
    Returns a plain dict by serializing through EvalResponse.
    """
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
    # Normalize: convert Pydantic models + enums → plain JSON-compatible dicts
    return EvalResponse(**raw).model_dump(mode="json")


def _make_sample_ecommerce() -> bytes:
    df = pd.DataFrame(
        {
            "customer_id": range(1, 101),
            "age": [25 + i % 45 for i in range(100)],
            "annual_spend_usd": [500 + (i * 137 % 9500) for i in range(100)],
            "region": [["North", "South", "East", "West"][i % 4] for i in range(100)],
            "churn": [i % 5 == 0 for i in range(100)],
        }
    )
    return df.to_csv(index=False).encode()


def _make_sample_sales() -> bytes:
    df = pd.DataFrame(
        {
            "deal_id": range(1, 51),
            "value_usd": [10_000 + i * 4321 % 200_000 for i in range(50)],
            "stage": [["Discovery", "Proposal", "Negotiation", "Closed"][i % 4] for i in range(50)],
            "rep": [f"Rep_{i % 8}" for i in range(50)],
            "days_open": [5 + i % 90 for i in range(50)],
        }
    )
    return df.to_csv(index=False).encode()


def _make_sample_products() -> bytes:
    df = pd.DataFrame(
        {
            "product": [f"SKU-{i:04d}" for i in range(80)],
            "category": [["Electronics", "Apparel", "Home", "Sports"][i % 4] for i in range(80)],
            "price": [9.99 + i * 7.77 % 500 for i in range(80)],
            "rating": [3.0 + (i % 20) / 10 for i in range(80)],
            "reviews": [i * 13 % 2000 for i in range(80)],
        }
    )
    return df.to_csv(index=False).encode()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    st.markdown('<p class="section-label">Evaluation</p>', unsafe_allow_html=True)
    task_description = st.text_area(
        "Task description",
        value="Analyze this dataset and answer questions about it.",
        height=80,
        help="Describe what you plan to do with this data.",
        label_visibility="collapsed",
    )
    num_questions = st.slider(
        "Evaluation depth",
        min_value=1, max_value=10, value=5,
        help="Number of benchmark questions — more = more accurate, slightly slower.",
    )

    st.divider()

    st.markdown('<p class="section-label">API Keys — optional (Live Mode)</p>', unsafe_allow_html=True)
    st.caption("Keys stay in this browser session and are never stored or logged.")

    with st.expander("Enter API keys", expanded=False):
        anthropic_key = st.text_input("Anthropic (Claude)", type="password", placeholder="sk-ant-...")
        openai_key    = st.text_input("OpenAI (GPT)",       type="password", placeholder="sk-...")
        groq_key      = st.text_input("Groq (Llama)",       type="password", placeholder="gsk_...")
        gemini_key    = st.text_input("Google (Gemini)",    type="password", placeholder="AIza...")
        xai_key       = st.text_input("xAI (Grok)",         type="password", placeholder="xai-...")
        if st.button("Clear keys", use_container_width=True):
            st.rerun()

    session_keys = SessionKeys(
        anthropic_api_key=anthropic_key or None,
        openai_api_key=openai_key or None,
        groq_api_key=groq_key or None,
        xai_api_key=xai_key or None,
        gemini_api_key=gemini_key or None,
    )
    any_key = session_keys.has_any_key()

    st.divider()
    if any_key:
        live_names = [
            name for name, val in [
                ("Anthropic", anthropic_key), ("OpenAI", openai_key),
                ("Groq", groq_key), ("Google", gemini_key), ("xAI", xai_key),
            ]
            if val and val.strip()
        ]
        st.success(f"Live Mode — {', '.join(live_names)}")
    else:
        st.info("Simulation Mode — no API keys needed")

    st.divider()
    st.caption("**Models covered:** Claude Sonnet/Opus/Haiku · GPT-5/4.1/4o · Gemini 2.5 · Llama 3.3 70B · Mixtral · Grok-3")


# ─── Mode Pill ────────────────────────────────────────────────────────────────
if any_key:
    st.markdown('<span class="pill pill-live">● Live Mode</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="pill pill-sim">● Simulation Mode — no API keys needed</span>', unsafe_allow_html=True)


# ─── Upload + Samples ─────────────────────────────────────────────────────────
col_up, col_how = st.columns([3, 2], gap="large")

with col_up:
    st.markdown("**Upload your dataset**")
    uploaded_file = st.file_uploader(
        "CSV or Parquet — up to 50 MB",
        type=["csv", "parquet"],
        label_visibility="collapsed",
    )

    st.markdown("**Or try a sample:**")
    s1, s2, s3 = st.columns(3)
    if s1.button("E-commerce", use_container_width=True):
        st.session_state["sample_file"] = ("ecommerce_customers.csv", _make_sample_ecommerce())
        st.session_state.pop("result", None)
    if s2.button("Sales pipeline", use_container_width=True):
        st.session_state["sample_file"] = ("sales_pipeline.csv", _make_sample_sales())
        st.session_state.pop("result", None)
    if s3.button("Product metrics", use_container_width=True):
        st.session_state["sample_file"] = ("product_metrics.csv", _make_sample_products())
        st.session_state.pop("result", None)

with col_how:
    st.markdown("**How it works**")
    st.markdown(
        "1. **Upload** a CSV or Parquet file\n"
        "2. *(Optional)* Add API keys for Live Mode\n"
        "3. Click **Analyze** — all LLMs benchmarked instantly\n"
        "4. Copy the recommended config"
    )


# ─── Resolve active file (upload > sample) ────────────────────────────────────
active_file: tuple[str, bytes] | None = None
if uploaded_file is not None:
    active_file = (uploaded_file.name, uploaded_file.read())
elif "sample_file" in st.session_state:
    active_file = st.session_state["sample_file"]


# ─── Run Evaluation ───────────────────────────────────────────────────────────
if active_file:
    filename, file_bytes = active_file
    st.caption(f"Loaded **{filename}** — {len(file_bytes) / 1024:.1f} KB")

    if st.button("⚡ Analyze & Get Recommendations", type="primary", use_container_width=True):
        label = "Running live benchmark..." if any_key else "Running simulation benchmark..."
        with st.spinner(label):
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
    st.info("Upload a CSV or Parquet file, or pick a sample dataset above.")


# ─── Results ──────────────────────────────────────────────────────────────────
if result := st.session_state.get("result"):
    rec   = result["recommended_model"]
    stats = result["dataset_stats"]
    sc    = rec["rdab_scorecard"]
    mode  = result.get("eval_mode", "simulation")
    live_providers = result.get("live_providers", [])

    st.divider()

    # ── Mode pill (result-level) ──────────────────────────────────────────────
    if mode == "live":
        st.markdown(
            f'<span class="pill pill-live">● Live — {", ".join(live_providers)}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="pill pill-sim">● Simulation — calibrated RDAB benchmark scores</span>',
            unsafe_allow_html=True,
        )

    # ── Recommendation card ───────────────────────────────────────────────────
    st.markdown(
        f"""
<div class="rec-card">
    <h3>✅ Recommended: {rec['display_name']}</h3>
    <p>{result['recommendation_reason']}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Key metrics ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RDAB Score",    f"{sc['rdab_score']:.1%}")
    m2.metric("Est. Cost/Run", _fmt_cost(rec["estimated_total_cost_usd"]))
    m3.metric("Latency",       f"{rec['latency_ms']:.0f} ms")
    m4.metric("Eval Time",     f"{result['total_eval_duration_s']:.1f} s")

    # ── RDAB scorecard breakdown ──────────────────────────────────────────────
    st.markdown("#### Score Breakdown")
    sim_note = " *(simulated)*" if sc.get("simulated") else ""
    st.caption(
        f"RealDataAgentBench{sim_note} · Correctness 50% · Code Quality 20% · Efficiency 15% · Stat Validity 15%"
    )
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Correctness",  f"{sc['correctness']:.1%}")
    d2.metric("Code Quality", f"{sc['code_quality']:.1%}")
    d3.metric("Efficiency",   f"{sc['efficiency']:.1%}")
    d4.metric("Stat Validity", f"{sc['stat_validity']:.1%}")

    # ── Dataset overview ──────────────────────────────────────────────────────
    with st.expander("Dataset overview", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",      f"{stats['rows']:,}")
        c2.metric("Columns",   stats["columns"])
        c3.metric("Missing",   f"{stats['missing_pct']:.1f}%")
        c4.metric("File Size", f"{stats['file_size_kb']:.1f} KB")
        st.caption(f"Format: {stats['file_format']} · Columns: {', '.join(stats['column_names'][:12])}")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("### Model Comparison")

    # Flatten scorecard fields into a flat DataFrame
    results_raw = result["results"]
    for r in results_raw:
        sc_r = r.get("rdab_scorecard", {})
        r["rdab_score"]    = sc_r.get("rdab_score", 0)
        r["correctness"]   = sc_r.get("correctness", 0)
        r["code_quality"]  = sc_r.get("code_quality", 0)
        r["efficiency"]    = sc_r.get("efficiency", 0)
        r["stat_validity"] = sc_r.get("stat_validity", 0)
        r["simulated"]     = sc_r.get("simulated", True)

    df_models = pd.DataFrame(results_raw)

    tab1, tab2, tab3, tab4 = st.tabs(["Score vs Cost", "Radar (Top 5)", "Latency", "Full Table"])

    TIER_COLORS = {"premium": "#6366f1", "balanced": "#3b82f6", "economy": "#10b981"}

    with tab1:
        fig = px.scatter(
            df_models,
            x="estimated_total_cost_usd",
            y="rdab_score",
            text="display_name",
            color="tier",
            size=[28] * len(df_models),
            color_discrete_map=TIER_COLORS,
            labels={
                "estimated_total_cost_usd": "Estimated Cost (USD)",
                "rdab_score": "RDAB Score",
            },
            title="RDAB Score vs Cost  ·  top-left = best value",
        )
        fig.update_traces(textposition="top center")
        fig.add_annotation(
            x=rec["estimated_total_cost_usd"],
            y=rec["rdab_scorecard"]["rdab_score"],
            text="✦ Recommended",
            showarrow=True, arrowhead=2,
            font={"color": "#6366f1", "size": 12},
        )
        fig.update_layout(height=420, plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top5 = df_models.nlargest(5, "rdab_score")
        cats = ["Correctness", "Code Quality", "Efficiency", "Stat Validity"]
        radar_fig = go.Figure()
        colors = ["#6366f1", "#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
        for i, (_, row) in enumerate(top5.iterrows()):
            vals = [row["correctness"], row["code_quality"], row["efficiency"], row["stat_validity"]]
            radar_fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                name=row["display_name"],
                line_color=colors[i % len(colors)],
                opacity=0.7,
            ))
        radar_fig.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            title="4-Dimensional Scorecard — Top 5 Models",
            height=460,
            paper_bgcolor="#f8fafc",
        )
        st.plotly_chart(radar_fig, use_container_width=True)

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
            title="Estimated Latency per Model",
            labels={"latency_ms": "Latency (ms)", "display_name": "Model"},
        )
        fig2.update_layout(height=380, plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc")
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
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
        tbl["Latency (ms)"]  = tbl["Latency (ms)"].map("{:.0f}".format)
        tbl["Cost/Run ($)"]  = tbl["Cost/Run ($)"].map("{:.6f}".format)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── Recommended config ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Recommended Config")
    st.caption(f"Drop this into your project to start using **{rec['display_name']}** immediately.")
    config_text = result["copyable_config"]
    st.code(config_text, language="json")

    with st.expander(f"Why {rec['display_name']}? — Full analysis", expanded=False):
        col_s, col_l = st.columns(2)
        with col_s:
            st.markdown("**Strengths**")
            for s in rec["strengths"]:
                st.markdown(f"- {s}")
        with col_l:
            st.markdown("**Limitations**")
            for lim in rec["limitations"]:
                st.markdown(f"- {lim}")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
<div style="text-align:center; color:#94a3b8; font-size:0.8rem; padding-bottom:1rem;">
    Built with FastAPI · Streamlit · Powered by
    <a href="https://github.com/patibandlavenkatamanideep/RealDataAgentBench" target="_blank"
       style="color:#6366f1;">RealDataAgentBench</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/patibandlavenkatamanideep/CostGuard" target="_blank"
       style="color:#6366f1;">GitHub</a>
</div>
""",
    unsafe_allow_html=True,
)


def run() -> None:
    import os
    import subprocess

    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", __file__,
            "--server.port", os.getenv("STREAMLIT_PORT", "8501"),
            "--server.address", "0.0.0.0",
        ],
        check=True,
    )
