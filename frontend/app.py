"""
CostGuard — Streamlit Dashboard
Upload a CSV/Parquet → get instant LLM cost estimates and recommendations.

Phase 2 additions:
  - Settings sidebar: optional API key entry (session-only, never stored)
  - Mode badge: "Live Mode" vs "Simulation Mode"
  - 4-dimensional radar chart for top 5 models
  - "Copy Recommended Config" button
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
    initial_sidebar_state="expanded",
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
        margin-bottom: 1rem;
        text-align: center;
    }
    .hero-header h1 { font-size: 2.8rem; font-weight: 800; margin: 0; }
    .hero-header p  { font-size: 1.15rem; opacity: 0.9; margin-top: 0.5rem; }

    /* Mode badges */
    .mode-live {
        display: inline-block;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.45rem 1.2rem;
        border-radius: 99px;
        margin-bottom: 1rem;
    }
    .mode-sim {
        display: inline-block;
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: #333;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.45rem 1.2rem;
        border-radius: 99px;
        margin-bottom: 1rem;
    }

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

    /* Primary button override */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
        transition: opacity 0.2s;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.88; }

    /* Table styling */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* Status badge */
    .badge-premium  { color: #6b21a8; background: #f3e8ff; padding: 2px 10px; border-radius: 99px; font-size: 0.8rem; font-weight: 600; }
    .badge-balanced { color: #1d4ed8; background: #dbeafe; padding: 2px 10px; border-radius: 99px; font-size: 0.8rem; font-weight: 600; }
    .badge-economy  { color: #065f46; background: #d1fae5; padding: 2px 10px; border-radius: 99px; font-size: 0.8rem; font-weight: 600; }

    /* Key input in sidebar */
    .key-input-label { font-size: 0.78rem; color: #718096; margin-bottom: 2px; }

    /* Big CTA banner */
    .cta-banner {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-radius: 14px;
        padding: 1.4rem 2rem;
        color: white;
        text-align: center;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 18px rgba(5,150,105,0.25);
    }
    .cta-banner h2 {
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.3px;
    }
    .cta-banner p  { font-size: 0.95rem; opacity: 0.9; margin: 0; }
    .sim-ok-banner {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 0.7rem 1.2rem;
        color: #92400e;
        font-size: 0.88rem;
        margin-bottom: 0.75rem;
    }
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
    api_keys: dict[str, str],
) -> dict:
    """Call /evaluate with optional session API keys passed as form fields."""
    form_data: dict[str, str] = {
        "task_description": task_description,
        "num_questions": str(num_questions),
    }
    # Merge non-empty keys (never send empty strings)
    for field, value in api_keys.items():
        if value and value.strip():
            form_data[field] = value.strip()

    with httpx.Client(base_url=API_BASE, timeout=120) as client:
        response = client.post(
            "/evaluate",
            files={"file": (filename, file_bytes, _mime(filename))},
            data=form_data,
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
    st.markdown("## Settings")

    # ── Evaluation options ─────────────────────────────────────────────────
    st.markdown("### Evaluation")
    task_description = st.text_area(
        "Task description",
        value="Analyze this dataset and answer questions about it.",
        height=90,
        help="Describe what you plan to do with this data. More specific = better recommendations.",
    )
    num_questions = st.slider(
        "Evaluation depth (questions)",
        min_value=1,
        max_value=10,
        value=5,
        help="More questions = more accurate recommendation, but slower.",
    )

    st.divider()

    # ── API Keys ───────────────────────────────────────────────────────────
    st.markdown("### API Keys (optional)")
    st.caption(
        "Enter keys to run **Live Mode** — real LLM agents evaluated against your data. "
        "Keys are kept in this browser session only and are **never stored or logged**."
    )

    with st.expander("Enter API keys", expanded=False):
        anthropic_key = st.text_input(
            "Anthropic (Claude)",
            type="password",
            placeholder="sk-ant-...",
            help="Enables Claude Sonnet 4.6, Opus 4.6, Haiku 4.5",
        )
        openai_key = st.text_input(
            "OpenAI (GPT)",
            type="password",
            placeholder="sk-...",
            help="Enables GPT-4.1, GPT-4o, GPT-4.1 mini/nano",
        )
        groq_key = st.text_input(
            "Groq (Llama / Mixtral)",
            type="password",
            placeholder="gsk_...",
            help="Enables Llama 3.3 70B, Llama 3.1 70B, Mixtral 8x7B via Groq",
        )
        gemini_key = st.text_input(
            "Google (Gemini)",
            type="password",
            placeholder="AIza...",
            help="Enables Gemini 2.5 Pro and Flash",
        )
        xai_key = st.text_input(
            "xAI (Grok)",
            type="password",
            placeholder="xai-...",
            help="Enables Grok-3 and Grok-3 mini",
        )
        if st.button("Clear all keys", use_container_width=True):
            for k in ("anthropic_key", "openai_key", "groq_key", "gemini_key", "xai_key"):
                st.session_state.pop(k, None)
            st.rerun()

    # Persist keys in session_state for the duration of the browser session
    st.session_state["_keys"] = {
        "anthropic_api_key": anthropic_key,
        "openai_api_key": openai_key,
        "groq_api_key": groq_key,
        "xai_api_key": xai_key,
        "gemini_api_key": gemini_key,
    }

    any_key = any(v and v.strip() for v in st.session_state["_keys"].values())

    st.divider()
    if any_key:
        providers_entered = [
            p for p, k in [
                ("Anthropic", anthropic_key), ("OpenAI", openai_key),
                ("Groq", groq_key), ("Google", gemini_key), ("xAI", xai_key),
            ] if k and k.strip()
        ]
        st.success(f"Live Mode enabled — {', '.join(providers_entered)}")
    else:
        st.info("No keys entered — Simulation Mode active")

    st.divider()
    st.markdown("**Supported models:**")
    st.caption(
        "Claude Sonnet/Opus/Haiku · GPT-5/4.1/4o · "
        "Gemini 2.5 Pro/Flash · Llama 3.3 70B · Mixtral · Grok-3"
    )


# ─── Mode Banner ──────────────────────────────────────────────────────────────
keys_in_session = st.session_state.get("_keys", {})
any_key = any(v and v.strip() for v in keys_in_session.values())

if any_key:
    st.markdown(
        '<div class="mode-live">Live Mode — running real LLM agents against your data</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="mode-sim">'
        'Simulation Mode — using calibrated RDAB benchmark scores '
        '(add API keys in the sidebar for live evaluation)'
        '</div>',
        unsafe_allow_html=True,
    )


# ─── Big CTA Banner ───────────────────────────────────────────────────────────
st.markdown(
    """
<div class="cta-banner">
    <h2>⚡ Upload your dataset and get a model recommendation in &lt;15 seconds</h2>
    <p>No account required &nbsp;·&nbsp; No data stored &nbsp;·&nbsp; Works instantly in Simulation Mode — no API keys needed</p>
</div>
<div class="sim-ok-banner">
    🟡 <strong>No API keys? No problem.</strong>&nbsp;
    Simulation Mode uses calibrated RDAB benchmark data to recommend the best model for your data —
    fully functional without any API keys.
</div>
""",
    unsafe_allow_html=True,
)

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
2. *(Optional)* Add API keys in the sidebar for **Live Mode**
3. **CostGuard** benchmarks all major LLMs against your data
4. **Get** the best model + exact cost per run
5. **Copy** the config with one click
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

    run_btn = st.button(
        "Analyze & Get Recommendations",
        type="primary",
        use_container_width=True,
    )

    if run_btn or st.session_state.get("auto_run"):
        st.session_state["auto_run"] = False

        spinner_label = (
            "Running live RDAB agents across all LLMs..."
            if any_key
            else "Running simulation benchmark across all LLMs..."
        )
        with st.spinner(spinner_label):
            try:
                data = call_evaluate(
                    file_bytes,
                    filename,
                    task_description,
                    num_questions,
                    st.session_state.get("_keys", {}),
                )
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
    eval_mode = result.get("eval_mode", "simulation")
    live_providers = result.get("live_providers", [])

    # ── Eval mode indicator ────────────────────────────────────────────────
    if eval_mode == "live":
        st.markdown(
            f'<div class="mode-live">'
            f'Live Mode — benchmarked against {", ".join(live_providers)}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="mode-sim">'
            'Simulation Mode (no keys provided) — '
            'scores based on calibrated RDAB historical data'
            '</div>',
            unsafe_allow_html=True,
        )

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
    sc = rec["rdab_scorecard"]
    m1.metric("RDAB Score", f"{sc['rdab_score']:.1%}", help="RealDataAgentBench composite score")
    m2.metric("Avg Latency", f"{rec['latency_ms']:.0f} ms")
    m3.metric("Est. Cost / Run", _fmt_cost(rec["estimated_total_cost_usd"]))
    m4.metric("Eval Duration", f"{result['total_eval_duration_s']:.1f}s")

    # ── RDAB 4-dimensional scorecard ─────────────────────────────────────
    st.markdown("#### RDAB Score Breakdown")
    sim_warn = (
        " *(simulation — add API keys in the sidebar for live evaluation)*"
        if sc.get("simulated")
        else ""
    )
    st.caption(
        f"Powered by RealDataAgentBench{sim_warn}. "
        "Scoring weights: Correctness 50% · Code Quality 20% · Efficiency 15% · Stat Validity 15%"
    )
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Correctness", f"{sc['correctness']:.1%}", help="Answer accuracy vs ground truth")
    d2.metric("Code Quality", f"{sc['code_quality']:.1%}", help="Vectorisation, naming, magic numbers")
    d3.metric("Efficiency", f"{sc['efficiency']:.1%}", help="Token + step budget adherence")
    d4.metric("Stat Validity", f"{sc['stat_validity']:.1%}", help="Statistical rigour in answers")

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
    tab1, tab2, tab3, tab4 = st.tabs(
        ["RDAB Score vs Cost", "4-D Radar (Top 5)", "Latency", "Detailed Table"]
    )

    # Flatten RDAB scorecard fields into the dataframe
    results_raw = result["results"]
    for r in results_raw:
        sc_r = r.get("rdab_scorecard", {})
        r["rdab_score"]    = sc_r.get("rdab_score", 0)
        r["correctness"]   = sc_r.get("correctness", 0)
        r["code_quality"]  = sc_r.get("code_quality", 0)
        r["efficiency"]    = sc_r.get("efficiency", 0)
        r["stat_validity"] = sc_r.get("stat_validity", 0)
        r["simulated"]     = sc_r.get("simulated", True)

    models_df = pd.DataFrame(results_raw)

    with tab1:
        fig = px.scatter(
            models_df,
            x="estimated_total_cost_usd",
            y="rdab_score",
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
                "rdab_score": "RDAB Composite Score",
            },
            title="RDAB Score vs Cost — top-left is best",
        )
        fig.update_traces(textposition="top center")
        fig.add_annotation(
            x=rec["estimated_total_cost_usd"],
            y=rec["rdab_scorecard"]["rdab_score"],
            text=" Recommended",
            showarrow=True,
            arrowhead=2,
            font={"color": "#059669", "size": 13},
        )
        fig.update_layout(showlegend=True, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Radar chart for top 5 models by RDAB score
        top5 = models_df.nlargest(5, "rdab_score")
        categories = ["Correctness", "Code Quality", "Efficiency", "Stat Validity"]
        radar_fig = go.Figure()
        colors = ["#7c3aed", "#2563eb", "#059669", "#d97706", "#dc2626"]
        for i, (_, row) in enumerate(top5.iterrows()):
            vals = [row["correctness"], row["code_quality"], row["efficiency"], row["stat_validity"]]
            vals_closed = vals + [vals[0]]
            radar_fig.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=categories + [categories[0]],
                fill="toself",
                name=row["display_name"],
                line_color=colors[i % len(colors)],
                opacity=0.75,
            ))
        radar_fig.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            showlegend=True,
            title="RDAB 4-Dimensional Scorecard — Top 5 Models",
            height=500,
        )
        st.plotly_chart(radar_fig, use_container_width=True)

        # Scorecard table for top 5
        st.markdown("**Top 5 Models — Score Details**")
        top5_display = top5[[
            "display_name", "rdab_score", "correctness", "code_quality",
            "efficiency", "stat_validity", "simulated",
        ]].copy()
        top5_display.columns = [
            "Model", "RDAB Score", "Correctness", "Code Quality",
            "Efficiency", "Stat Validity", "Simulated",
        ]
        for col in ["RDAB Score", "Correctness", "Code Quality", "Efficiency", "Stat Validity"]:
            top5_display[col] = top5_display[col].map("{:.1%}".format)
        st.dataframe(top5_display, use_container_width=True, hide_index=True)
        st.caption(
            "Universal stat_validity weakness (~0.25) is a known RDAB finding — "
            "no model consistently reports uncertainty correctly."
        )

    with tab3:
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

    with tab4:
        display_df = models_df[[
            "display_name", "provider", "tier",
            "rdab_score", "correctness", "code_quality", "efficiency", "stat_validity",
            "latency_ms", "estimated_total_cost_usd",
            "input_cost_per_1k", "output_cost_per_1k", "simulated",
        ]].copy()
        display_df.columns = [
            "Model", "Provider", "Tier",
            "RDAB Score", "Correctness", "Code Quality", "Efficiency", "Stat Validity",
            "Latency (ms)", "Cost / Run ($)",
            "Input $/1K", "Output $/1K", "Simulated",
        ]
        for col in ["RDAB Score", "Correctness", "Code Quality", "Efficiency", "Stat Validity"]:
            display_df[col] = display_df[col].map("{:.1%}".format)
        display_df["Latency (ms)"] = display_df["Latency (ms)"].map("{:.0f}".format)
        display_df["Cost / Run ($)"] = display_df["Cost / Run ($)"].map("{:.6f}".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Copy Recommended Config ────────────────────────────────────────────
    st.divider()
    st.markdown("### Recommended Config")
    st.caption(
        f"Drop this into your project to start using **{rec['display_name']}** immediately."
    )

    config_text = result["copyable_config"]

    # st.code renders a native copy button (clipboard icon top-right of the block)
    st.code(config_text, language="json")

    col_copy, col_spacer = st.columns([1, 3])
    with col_copy:
        if st.button("Copy Recommended Config", type="primary", use_container_width=True):
            # Inject clipboard write via component HTML (works in most browsers)
            st.markdown(
                f"""
                <script>
                (function() {{
                    const text = {json.dumps(config_text)};
                    if (navigator.clipboard && window.isSecureContext) {{
                        navigator.clipboard.writeText(text);
                    }} else {{
                        const ta = document.createElement('textarea');
                        ta.value = text;
                        document.body.appendChild(ta);
                        ta.select();
                        document.execCommand('copy');
                        document.body.removeChild(ta);
                    }}
                }})();
                </script>
                """,
                unsafe_allow_html=True,
            )
            st.toast("Config copied to clipboard!", icon="")

    # ── Strengths + limitations of recommended model ───────────────────────
    with st.expander(f"Why {rec['display_name']}? Full analysis", expanded=False):
        col_s, col_l = st.columns(2)
        with col_s:
            st.markdown("**Strengths**")
            for s in rec["strengths"]:
                st.markdown(f"- {s}")
        with col_l:
            st.markdown("**Limitations**")
            for lim in rec["limitations"]:
                st.markdown(f"- {lim}")


else:
    if active_file is None:
        st.info(
            "Upload a CSV or Parquet file above to get started, "
            "or try one of the sample datasets."
        )


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"""
<div style="text-align:center; color:#718096; font-size:0.85rem;">
    Built with FastAPI + Streamlit · Powered by
    <a href="https://github.com/patibandlavenkatamanideep/RealDataAgentBench" target="_blank">RealDataAgentBench</a> ·
    <a href="https://github.com/patibandlavenkatamanideep/CostGuard" target="_blank">GitHub</a> ·
    <a href="{API_BASE}/docs" target="_blank">API Docs</a>
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
