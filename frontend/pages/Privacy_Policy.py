"""CostGuard — Privacy Policy page."""

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
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #f8fafc; color: #0f172a;
}
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.pp-hero { background: linear-gradient(135deg, #312e81 0%, #4f46e5 100%); border-radius: 16px; padding: 2rem; color: #fff; margin-bottom: 2rem; }
.pp-hero h1 { font-size: 2rem; font-weight: 800; margin: 0 0 0.4rem; }
.pp-hero p  { font-size: 0.9rem; color: rgba(255,255,255,.8); margin: 0; }
.pp-section { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1.25rem; }
.pp-section h2 { font-size: 1.1rem; font-weight: 700; color: #0f172a; margin: 0 0 1rem; }
.pp-section p, .pp-section li { font-size: 0.88rem; color: #334155; line-height: 1.7; }
.pp-section ul { padding-left: 1.25rem; margin: 0.5rem 0; }
.highlight { background: #f0fdf4; border-left: 3px solid #22c55e; padding: 0.75rem 1rem; border-radius: 6px; margin: 0.75rem 0; font-size: 0.88rem; color: #166534; }
.footer { text-align: center; color: #94a3b8; font-size: 0.77rem; padding: 1.5rem 0 0.5rem; border-top: 1px solid #f1f5f9; margin-top: 2rem; }
.footer a { color: #6366f1; text-decoration: none; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pp-hero">
  <h1>🛡️ Privacy Policy</h1>
  <p>CostGuard — LLM Cost Intelligence Platform &nbsp;·&nbsp; Effective date: April 2026</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pp-section">
  <h2>Summary — what matters most</h2>
  <div class="highlight">
    Your uploaded data is <strong>never stored on our servers</strong>.
    It is processed entirely in memory for the duration of a single evaluation request and discarded immediately after.
    We do not sell, share, or retain any business data you upload.
  </div>
  <p>
    CostGuard is a benchmarking tool, not a data platform.
    Everything below describes in detail how we handle the minimal information we do collect.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pp-section">
  <h2>1. What data we collect</h2>
  <p><strong>Data you upload (CSV / Parquet files)</strong></p>
  <ul>
    <li>Processed entirely in memory during your session.</li>
    <li>Never written to disk, never stored in a database, never transmitted to third parties.</li>
    <li>Discarded as soon as the evaluation completes or the session ends.</li>
  </ul>
  <p><strong>API keys you enter</strong></p>
  <ul>
    <li>Stored only in your browser session state (Streamlit session, not a cookie or local storage).</li>
    <li>Never written to server logs, environment variables, or any database.</li>
    <li>Used only to make calls to the respective LLM provider (OpenAI, Anthropic, Google, Groq, xAI) on your behalf during that session.</li>
    <li>Cleared automatically when your browser tab closes or you click "Clear all keys".</li>
  </ul>
  <p><strong>Evaluation metadata (logged locally)</strong></p>
  <ul>
    <li>When an evaluation completes, CostGuard saves a summary row to a local SQLite database
        (<code>/tmp/costguard_history.db</code> by default, configurable via <code>COSTGUARD_DB_PATH</code>).</li>
    <li>This row contains: evaluation ID, timestamp, recommended model ID, RDAB score, cost estimate,
        dataset fingerprint (a one-way hash of row count + column count + column names), and evaluation mode.</li>
    <li>Column names are hashed (SHA-256 prefix) before logging. Raw column names, row values, and any cell contents are never stored.</li>
    <li>This data lives on <em>your</em> server (or Railway volume) — it is not transmitted to Anthropic, CostGuard, or any third party.</li>
  </ul>
  <p><strong>Drift alerts (optional)</strong></p>
  <ul>
    <li>If you configure a Slack webhook URL, a brief alert is sent to your Slack workspace when a score drops more than 10% from the historical average. The alert contains model ID, scores, and evaluation ID — no user data, no uploaded file content.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pp-section">
  <h2>2. What we do NOT collect</h2>
  <ul>
    <li>No user accounts, emails, or names.</li>
    <li>No payment information (CostGuard is free and open-source).</li>
    <li>No IP addresses in application logs (server access logs may vary by hosting provider).</li>
    <li>No cookies or persistent tracking of any kind.</li>
    <li>No content from your uploaded files — only structural metadata (row/column counts, column name hashes).</li>
    <li>No LLM responses are retained after the evaluation completes.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pp-section">
  <h2>3. Third-party LLM providers</h2>
  <p>
    In Live Mode, CostGuard calls LLM provider APIs using <em>your</em> API keys.
    Your data (specifically, a sample of up to 500 rows formatted as a text prompt) is sent to those providers.
    Each provider's privacy policy applies to those calls:
  </p>
  <ul>
    <li><strong>Anthropic</strong> — <a href="https://www.anthropic.com/privacy" target="_blank">anthropic.com/privacy</a></li>
    <li><strong>OpenAI</strong> — <a href="https://openai.com/policies/privacy-policy" target="_blank">openai.com/policies/privacy-policy</a></li>
    <li><strong>Google Gemini</strong> — <a href="https://policies.google.com/privacy" target="_blank">policies.google.com/privacy</a></li>
    <li><strong>Groq</strong> — <a href="https://groq.com/privacy-policy/" target="_blank">groq.com/privacy-policy</a></li>
    <li><strong>xAI (Grok)</strong> — <a href="https://x.ai/privacy-policy" target="_blank">x.ai/privacy-policy</a></li>
  </ul>
  <p>
    CostGuard does not control how these providers handle data sent via API.
    If your dataset contains personally identifiable information (PII), confidential business data,
    or regulated information (HIPAA, GDPR), we strongly recommend anonymising or redacting it before uploading.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pp-section">
  <h2>4. Data retention</h2>
  <ul>
    <li><strong>Uploaded files:</strong> zero retention — discarded immediately after evaluation.</li>
    <li><strong>API keys:</strong> zero retention — cleared when the session ends.</li>
    <li><strong>Evaluation metadata:</strong> stored in local SQLite until you delete the file. Default path: <code>/tmp/costguard_history.db</code> (ephemeral on Railway; set <code>COSTGUARD_DB_PATH</code> to a persistent volume for durability).</li>
    <li><strong>Server access logs:</strong> dependent on your hosting provider (Railway, Docker, etc.). CostGuard itself does not write access logs.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pp-section">
  <h2>5. Self-hosting</h2>
  <p>
    CostGuard is fully open-source (MIT License).
    If you self-host using Docker or Railway, all data stays entirely within your own infrastructure.
    No telemetry, analytics, or usage data is sent back to the CostGuard project.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pp-section">
  <h2>6. Your rights</h2>
  <p>
    Because we do not collect personal data, there is nothing to access, correct, or delete on our side.
    Any evaluation metadata is stored on your own server and is fully under your control.
    To delete it: remove or clear the SQLite file at <code>COSTGUARD_DB_PATH</code>.
  </p>
  <p>
    If you have questions or concerns about this policy, open an issue on
    <a href="https://github.com/patibandlavenkatamanideep/CostGuard" target="_blank">GitHub</a>.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pp-section">
  <h2>7. Changes to this policy</h2>
  <p>
    If this policy changes materially (e.g., if we introduce accounts or telemetry),
    we will update this page and note the effective date at the top.
    The current version is always available in the repository at
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
