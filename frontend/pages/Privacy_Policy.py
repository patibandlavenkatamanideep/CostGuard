"""CostGuard — Privacy Policy"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

st.set_page_config(
    page_title="Privacy Policy — CostGuard",
    page_icon="🛡️",
    layout="wide",
)

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

.pp-header{background:#111827;border-radius:12px;padding:2rem;margin-bottom:1.75rem}
.pp-header h1{font-size:1.75rem;font-weight:800;color:#fff;margin:0 0 .3rem;letter-spacing:-.4px}
.pp-header p{font-size:.82rem;color:#9ca3af;margin:0}

.pp-summary{background:#eef2ff;border:1px solid #c7d2fe;border-left:3px solid #4f46e5;border-radius:8px;padding:1rem 1.25rem;margin-bottom:1.5rem;font-size:.85rem;color:#3730a3;line-height:1.6}
.pp-summary strong{color:#111827}

.pp-section{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:1.5rem;margin-bottom:1rem}
.pp-section h2{font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af;margin:0 0 .875rem;border-bottom:1px solid #f3f4f6;padding-bottom:.5rem}
.pp-section p,.pp-section li{font-size:.85rem;color:#374151;line-height:1.7;margin:.35rem 0}
.pp-section ul{padding-left:1.25rem;margin:.5rem 0}
.pp-section a{color:#4f46e5;text-decoration:none;font-weight:600}
.pp-section a:hover{text-decoration:underline}
.pp-section code{background:#f3f4f6;border-radius:4px;padding:.1rem .35rem;font-size:.8rem;color:#111827}

.footer{text-align:center;color:#9ca3af;font-size:.75rem;padding:1.5rem 0 .5rem;border-top:1px solid #f3f4f6;margin-top:2rem}
.footer a{color:#4f46e5;text-decoration:none;font-weight:600}
.footer a:hover{text-decoration:underline}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pp-header">
  <h1>Privacy Policy</h1>
  <p>CostGuard · Effective April 2026</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pp-summary">
  <strong>Short version:</strong> Your uploaded data is never stored on our servers.
  It is processed in memory for the duration of a single evaluation and discarded immediately after.
  We do not sell, share, or retain any business data you upload.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
<div class="pp-section">
  <h2>What we collect</h2>
  <p><strong>Uploaded files (CSV / Parquet)</strong></p>
  <ul>
    <li>Processed in memory only — never written to disk or stored.</li>
    <li>Discarded as soon as the evaluation completes.</li>
  </ul>
  <p><strong>API keys</strong></p>
  <ul>
    <li>Held in browser session state only — not in cookies, not on the server.</li>
    <li>Used only to call the LLM provider on your behalf for that session.</li>
    <li>Cleared when your tab closes or you click "Clear all keys".</li>
  </ul>
  <p><strong>Evaluation metadata (logged locally)</strong></p>
  <ul>
    <li>A summary row is saved to a local SQLite file on <em>your</em> server.</li>
    <li>Contains: eval ID, timestamp, recommended model, RDAB score, cost estimate, and a one-way hash of dataset shape (row/column counts + hashed column names).</li>
    <li>No row values, no cell contents, no raw column names are ever stored.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="pp-section">
  <h2>What we do not collect</h2>
  <ul>
    <li>No user accounts, emails, or names.</li>
    <li>No payment information.</li>
    <li>No IP addresses in application logs.</li>
    <li>No cookies or persistent tracking.</li>
    <li>No content from uploaded files — only structural metadata.</li>
    <li>No LLM responses after the evaluation completes.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="pp-section">
  <h2>Self-hosting</h2>
  <p>
    CostGuard is fully open-source (MIT License).
    When self-hosted via Docker or Railway, all data stays entirely within your own infrastructure.
    No telemetry or usage data is sent back to the CostGuard project.
  </p>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div class="pp-section">
  <h2>Third-party LLM providers</h2>
  <p>
    In Live Mode, a sample of your data (up to 500 rows as a text prompt) is sent to the LLM provider
    whose API key you entered. Each provider's own privacy policy applies to those calls.
  </p>
  <ul>
    <li><a href="https://www.anthropic.com/privacy" target="_blank">Anthropic</a></li>
    <li><a href="https://openai.com/policies/privacy-policy" target="_blank">OpenAI</a></li>
    <li><a href="https://policies.google.com/privacy" target="_blank">Google Gemini</a></li>
    <li><a href="https://groq.com/privacy-policy/" target="_blank">Groq</a></li>
    <li><a href="https://x.ai/privacy-policy" target="_blank">xAI (Grok)</a></li>
  </ul>
  <p>
    If your dataset contains PII, confidential data, or regulated information (HIPAA, GDPR),
    anonymise or redact it before uploading.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="pp-section">
  <h2>Data retention</h2>
  <ul>
    <li><strong>Uploaded files:</strong> zero retention — discarded immediately.</li>
    <li><strong>API keys:</strong> zero retention — cleared when the session ends.</li>
    <li><strong>Evaluation metadata:</strong> stored in local SQLite at <code>COSTGUARD_DB_PATH</code> until you delete it. Default: <code>/tmp/costguard_history.db</code>.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="pp-section">
  <h2>Your rights</h2>
  <p>
    Because we do not collect personal data, there is nothing to access, correct, or delete on our side.
    All evaluation metadata is on your own server — delete the SQLite file to remove it.
  </p>
  <p>
    Questions? Open an issue on
    <a href="https://github.com/patibandlavenkatamanideep/CostGuard" target="_blank">GitHub</a>.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="pp-section">
  <h2>Policy changes</h2>
  <p>
    Material changes will be noted with an updated effective date at the top of this page.
    The current version is always in the repository at
    <code>frontend/pages/Privacy_Policy.py</code>.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  CostGuard &nbsp;·&nbsp;
  <a href="/" target="_self">← Back to Home</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/patibandlavenkatamanideep/CostGuard" target="_blank">GitHub</a>
  &nbsp;·&nbsp; MIT License
</div>
""", unsafe_allow_html=True)
