"""
CostGuard — Streamlit Dashboard
Upload a CSV/Parquet → get instant LLM cost estimates and recommendations.
"""

from __future__ import annotations

import json
import os
import time

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CostGuard — LLM Cost Estimator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Hero gradient header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero-header h1 { font-size: 2.8rem; font-weight: 800; margin: 0; }
    .hero-header p  { font-size: 1.15rem; opacity: 0.9; margin-top: 0.5rem; }

    /* Recommendation card */
    .rec-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .rec-card h2 { font-size: 1.6rem; font-weight: 700; margin: 0; }
    .rec-card .subtitle { opacity: 0.85; margin-top: 0.3rem; }

    /* Metric card */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card .metric-value { font-size: 1.8rem; font-weight: 700; color: #2d3748; }
    .metric-card .metric-label { font-size: 0.85rem; color: #718096; margin-top: 0.25rem; }

    /* Copy button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Table styling */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: #f7f8ff;
    }

    /* Status badge */
    .badge-premium { color: #6b21a8; background: #f3e8ff; padding: 2px 10px; border-radius: 99px; font-size: 0.8rem; font-weight: 600; }
    .badge-balanced { color: #1d4ed8; background: #dbeafe; padding: 2px 10px; border-radius: 99px; font-size: 0.8rem; font-weight: 600; }
    .badge-economy { color: #065f46; background: #d1fae5; padding: 2px 10px; border-radius: 99px; font-size: 0.8rem; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-header">
    <h1> CostGuard</h1>
    <p>Upload your data → instantly find the best LLM with exact cost estimates</p>
    <p style="font-size:0.9rem; opacity:0.75;">Supports CSV · Parquet · Up to 50MB · Free to use</p>
</div>
""",
    unsafe_allow_html=True,
)


# ─── Helper: API client ───────────────────────────────────────────────────────

def call_evaluate(
    file_bytes: bytes,
    filename: str,
    task_description: str,
    num_questions: int,
) -> dict:
    with httpx.Client(base_url=API_BASE, timeout=120) as client:
        response = client.post(
            "/evaluate",
            files={"file": (filename, file_bytes, _mime(filename))},
            data={
                "task_description": task_description,
                "num_questions": str(num_questions),
            },
        )
    if response.status_code != 200:
        err = response.json()
        raise RuntimeError(err.get("detail", err.get("error", "Unknown error")))
    return response.json()


def _mime(filename: str) -> str:
    return "text/csv" if filename.lower().endswith(".csv") else "application/octet-stream"


def _fmt_cost(usd: float) -> str:
    if usd < 0.0001:
        return f"${usd*1e6:.2f}µ"
    if usd < 0.01:
        return f"${usd*1000:.3f}m"
    return f"${usd:.4f}"


def _tier_badge(tier: str) -> str:
    return f'<span class="badge-{tier}">{tier.upper()}</span>'


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    task_description = st.text_area(
        "Task description",
        value="Analyze this dataset and answer questions about it.",
        height=100,
        help="Describe what you plan to do with this data. More specific = better recommendations.",
    )
    num_questions = st.slider(
        "Evaluation depth",
        min_value=1,
        max_value=10,
        value=5,
        help="More questions = more accurate recommendation, but slower.",
    )
    st.divider()
    st.markdown("**Supported models:**")
    st.caption("GPT-4o, GPT-4o mini, Claude 3.5 Sonnet, Claude 3.5 Haiku, Gemini 1.5 Pro/Flash, Llama 3.1 70B, Mixtral 8x7B")


# ─── Upload Section ───────────────────────────────────────────────────────────
col_upload, col_info = st.columns([2, 1], gap="large")

with col_upload:
    st.markdown("### Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Drop a CSV or Parquet file here",
        type=["csv", "parquet"],
        help="Max 50MB. Your data never leaves this session.",
        label_visibility="collapsed",
    )

with col_info:
    st.markdown("### How it works")
    st.markdown(
        """
1. **Upload** your CSV or Parquet
2. **CostGuard** benchmarks all major LLMs against your actual data
3. **Get** the best model + exact cost per run
4. **Copy** the config with one click
        """,
        unsafe_allow_html=False,
    )

# ─── Sample datasets ──────────────────────────────────────────────────────────
st.markdown("**No data? Try a sample:**")
c1, c2, c3 = st.columns(3)

def _make_sample_csv() -> bytes:
    df = pd.DataFrame({
        "customer_id": range(1, 101),
        "age": [25 + i % 45 for i in range(100)],
        "annual_spend_usd": [500 + (i * 137 % 9500) for i in range(100)],
        "region": [["North", "South", "East", "West"][i % 4] for i in range(100)],
        "churn": [i % 5 == 0 for i in range(100)],
    })
    return df.to_csv(index=False).encode()

if c1.button("E-commerce sample"):
    st.session_state["sample_file"] = ("ecommerce_customers.csv", _make_sample_csv())

if c2.button("Sales pipeline sample"):
    df = pd.DataFrame({
        "deal_id": range(1, 51),
        "value_usd": [10_000 + i * 4321 % 200_000 for i in range(50)],
        "stage": [["Discovery", "Proposal", "Negotiation", "Closed"][i % 4] for i in range(50)],
        "rep": [f"Rep_{i % 8}" for i in range(50)],
        "days_open": [5 + i % 90 for i in range(50)],
    })
    st.session_state["sample_file"] = ("sales_pipeline.csv", df.to_csv(index=False).encode())

if c3.button("Product metrics sample"):
    df = pd.DataFrame({
        "product": [f"SKU-{i:04d}" for i in range(80)],
        "category": [["Electronics", "Apparel", "Home", "Sports"][i % 4] for i in range(80)],
        "price": [9.99 + i * 7.77 % 500 for i in range(80)],
        "rating": [3.0 + (i % 20) / 10 for i in range(80)],
        "reviews": [i * 13 % 2000 for i in range(80)],
    })
    st.session_state["sample_file"] = ("product_metrics.csv", df.to_csv(index=False).encode())


# Resolve file: uploaded > sample
active_file: tuple[str, bytes] | None = None
if uploaded_file is not None:
    active_file = (uploaded_file.name, uploaded_file.read())
elif "sample_file" in st.session_state:
    active_file = st.session_state["sample_file"]


# ─── Evaluation ───────────────────────────────────────────────────────────────
if active_file:
    filename, file_bytes = active_file
    st.success(f"Loaded **{filename}** ({len(file_bytes)/1024:.1f} KB)")

    run_btn = st.button("Analyze & Get Recommendations", type="primary", use_container_width=True)

    if run_btn or st.session_state.get("auto_run"):
        st.session_state["auto_run"] = False

        with st.spinner("Running benchmark evaluation across all LLMs..."):
            t0 = time.time()
            try:
                data = call_evaluate(file_bytes, filename, task_description, num_questions)
                st.session_state["result"] = data
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                st.stop()

if "sample_file" in st.session_state and not st.session_state.get("result"):
    st.session_state["auto_run"] = True


# ─── Results Section ──────────────────────────────────────────────────────────
if result := st.session_state.get("result"):
    rec = result["recommended_model"]
    stats = result["dataset_stats"]

    # ── Recommendation banner ──────────────────────────────────────────────
    st.markdown(
        f"""
<div class="rec-card">
    <h2> Best LLM for Your Data: {rec['display_name']}</h2>
    <div class="subtitle">{result['recommendation_reason']}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Key metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy Score", f"{rec['accuracy_score']:.1%}")
    m2.metric("Avg Latency", f"{rec['latency_ms']:.0f} ms")
    m3.metric("Est. Cost / Run", _fmt_cost(rec["estimated_total_cost_usd"]))
    m4.metric("Eval Duration", f"{result['total_eval_duration_s']:.1f}s")

    st.divider()

    # ── Dataset stats ─────────────────────────────────────────────────────
    with st.expander("Dataset Overview", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Rows", f"{stats['rows']:,}")
        s2.metric("Columns", stats["columns"])
        s3.metric("Missing %", f"{stats['missing_pct']:.1f}%")
        s4.metric("File Size", f"{stats['file_size_kb']:.1f} KB")
        st.caption(f"Format: {stats['file_format']} | Columns: {', '.join(stats['column_names'][:10])}")

    # ── Charts ────────────────────────────────────────────────────────────
    st.markdown("### Model Comparison")
    tab1, tab2, tab3 = st.tabs(["Accuracy vs Cost", "Latency", "Detailed Table"])

    models_df = pd.DataFrame(result["results"])

    with tab1:
        fig = px.scatter(
            models_df,
            x="estimated_total_cost_usd",
            y="accuracy_score",
            text="display_name",
            color="tier",
            size=[30] * len(models_df),
            color_discrete_map={
                "premium": "#7c3aed",
                "balanced": "#2563eb",
                "economy": "#059669",
            },
            labels={
                "estimated_total_cost_usd": "Estimated Cost (USD)",
                "accuracy_score": "Accuracy Score",
            },
            title="Accuracy vs Cost — top-left is best",
        )
        fig.update_traces(textposition="top center")
        fig.add_annotation(
            x=rec["estimated_total_cost_usd"],
            y=rec["accuracy_score"],
            text=" Recommended",
            showarrow=True,
            arrowhead=2,
            font={"color": "#059669", "size": 13},
        )
        fig.update_layout(showlegend=True, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = px.bar(
            models_df.sort_values("latency_ms"),
            x="display_name",
            y="latency_ms",
            color="tier",
            color_discrete_map={
                "premium": "#7c3aed",
                "balanced": "#2563eb",
                "economy": "#059669",
            },
            title="Average Latency per Model",
            labels={"latency_ms": "Latency (ms)", "display_name": "Model"},
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        display_df = models_df[
            [
                "display_name",
                "provider",
                "tier",
                "accuracy_score",
                "latency_ms",
                "estimated_total_cost_usd",
                "input_cost_per_1k",
                "output_cost_per_1k",
            ]
        ].copy()
        display_df.columns = [
            "Model",
            "Provider",
            "Tier",
            "Accuracy",
            "Latency (ms)",
            "Cost / Run ($)",
            "Input $/1K",
            "Output $/1K",
        ]
        display_df["Accuracy"] = display_df["Accuracy"].map("{:.1%}".format)
        display_df["Latency (ms)"] = display_df["Latency (ms)"].map("{:.0f}".format)
        display_df["Cost / Run ($)"] = display_df["Cost / Run ($)"].map("{:.6f}".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Copy config ───────────────────────────────────────────────────────
    st.markdown("### One-Click Config")
    st.caption(
        f"Drop this into your project to start using **{rec['display_name']}** immediately."
    )

    config_text = result["copyable_config"]

    col_code, col_btn = st.columns([4, 1])
    with col_code:
        st.code(config_text, language="json")
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Copy Config"):
            st.toast("Config copied to clipboard!", icon="")
            # Inject JS clipboard copy
            st.markdown(
                f"""<script>
                navigator.clipboard.writeText({json.dumps(config_text)});
                </script>""",
                unsafe_allow_html=True,
            )

    # ── Strengths + limitations of recommended model ───────────────────────
    with st.expander(f"Why {rec['display_name']}? Full analysis", expanded=False):
        col_s, col_l = st.columns(2)
        with col_s:
            st.markdown("**Strengths**")
            for s in rec["strengths"]:
                st.markdown(f"- {s}")
        with col_l:
            st.markdown("**Limitations**")
            for l in rec["limitations"]:
                st.markdown(f"- {l}")


else:
    if active_file is None:
        st.info("Upload a CSV or Parquet file above to get started, or try one of the sample datasets.")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
<div style="text-align:center; color:#718096; font-size:0.85rem;">
    Built with FastAPI + Streamlit ·
    <a href="https://github.com/your-org/costguard" target="_blank">GitHub</a> ·
    <a href="http://localhost:8000/docs" target="_blank">API Docs</a>
</div>
""",
    unsafe_allow_html=True,
)


def run() -> None:
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", __file__,
         "--server.port", os.getenv("STREAMLIT_PORT", "8501"),
         "--server.address", "0.0.0.0"],
        check=True,
    )
