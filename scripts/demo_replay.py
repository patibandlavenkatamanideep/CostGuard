"""
Day 5 Demo: Tether → CostGuard /replay end-to-end.

Runs 25 gpt-4o-mini calls through TetheredOpenAI, then submits the captured
trace to CostGuard's /replay endpoint to compare against gpt-4.1-mini and
print the cost-savings + quality-delta report.

Usage:
    # Terminal 1: start CostGuard
    cd /path/to/CostGuard
    OPENAI_API_KEY=sk-... costguard-api

    # Terminal 2: run the demo
    cd /path/to/CostGuard
    OPENAI_API_KEY=sk-... python scripts/demo_replay.py

Requirements:
    - OPENAI_API_KEY set in the environment
    - Tether installed (pip install -e /path/to/Tether)
    - CostGuard running on localhost:8000 (or set COSTGUARD_URL)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

TETHER_DB = Path("/tmp/demo_tether.db")
COSTGUARD_URL = os.getenv("COSTGUARD_URL", "http://localhost:8000")
PRIMARY_MODEL = "gpt-4o-mini"
ALTERNATE_MODEL = "gpt-4.1-mini"
N_CALLS = 25
N_BOOTSTRAP = 1000

# ── 25 short, realistic prompts ───────────────────────────────────────────────
PROMPTS = [
    "What is the difference between precision and recall in machine learning?",
    "Explain gradient descent in two sentences.",
    "What is a transformer architecture and why does it matter?",
    "Name three advantages of using SQLite over a full database server.",
    "What does 'statistically significant' mean in plain English?",
    "Describe the bias-variance tradeoff concisely.",
    "What is the curse of dimensionality?",
    "How does a confusion matrix help evaluate a classifier?",
    "What is the purpose of a validation set in model training?",
    "Explain what overfitting is and how to detect it.",
    "What is the difference between L1 and L2 regularization?",
    "Why is feature scaling important for gradient-based algorithms?",
    "What is k-fold cross-validation and when do you use it?",
    "Explain what p-hacking is and why it is a problem.",
    "What is the purpose of a bootstrap confidence interval?",
    "Describe the difference between a parameter and a hyperparameter.",
    "What is a ROC curve and what does AUC measure?",
    "Why do we use logarithms in cross-entropy loss?",
    "What is the difference between bagging and boosting?",
    "Explain what a t-test assumes about the data.",
    "What is the central limit theorem in one sentence?",
    "What does it mean for a model to be calibrated?",
    "How does early stopping prevent overfitting?",
    "What is the difference between supervised and self-supervised learning?",
    "Why is reproducibility important in ML experiments?",
]


async def capture_traces() -> str:
    """Run N_CALLS through TetheredOpenAI and return the run_id."""
    try:
        from openai import OpenAI
        from tether import SQLiteStorage, TetheredOpenAI
    except ImportError as exc:
        print(f"[demo] Import error: {exc}")
        print("[demo] Install Tether: pip install -e /path/to/Tether")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[demo] OPENAI_API_KEY not set — aborting.")
        sys.exit(1)

    TETHER_DB.unlink(missing_ok=True)  # fresh run each time

    storage = SQLiteStorage(TETHER_DB)
    await storage.initialize()

    tethered = TetheredOpenAI(
        client=OpenAI(api_key=api_key),
        storage=storage,
        run_name="demo_replay_run",
    )

    print(f"[demo] Capturing {N_CALLS} calls with {PRIMARY_MODEL}...")
    t0 = time.perf_counter()

    for i, prompt in enumerate(PROMPTS[:N_CALLS], start=1):
        resp = tethered.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,
        )
        tokens = resp.usage.total_tokens if resp.usage else "?"
        print(f"  [{i:02d}/{N_CALLS}] {tokens} tokens — {prompt[:55]}...")

    elapsed = time.perf_counter() - t0
    run_id = str(tethered.run_id)
    print(f"[demo] Captured {N_CALLS} steps in {elapsed:.1f}s  run_id={run_id}")
    print(f"[demo] Database: {TETHER_DB}")

    steps = await storage.get_steps_for_run(tethered.run_id)
    total_cost = sum(s.cost_usd or 0 for s in steps)
    print(f"[demo] Primary cost: ${float(total_cost):.6f}")

    await storage.close()
    return run_id


def submit_replay(run_id: str) -> dict:
    """POST /replay to CostGuard and return the parsed JSON response."""
    try:
        import httpx
    except ImportError:
        print("[demo] httpx not found — install it: pip install httpx")
        sys.exit(1)

    payload = {
        "tether_db_path": str(TETHER_DB.resolve()),
        "run_id": run_id,
        "alternate_model": ALTERNATE_MODEL,
        "n_bootstrap_samples": N_BOOTSTRAP,
    }
    print(f"\n[demo] POSTing to {COSTGUARD_URL}/replay ...")
    print(f"[demo] Payload: {json.dumps(payload, indent=2)}")

    with httpx.Client(timeout=300.0) as client:
        resp = client.post(f"{COSTGUARD_URL}/replay", json=payload)

    if resp.status_code != 200:
        print(f"[demo] ERROR {resp.status_code}: {resp.text}")
        sys.exit(1)

    return resp.json()


def print_report(result: dict) -> None:
    """Pretty-print the replay result as the demo's Exhibit A."""
    print("\n" + "═" * 60)
    print("  REPLAY REPORT")
    print("═" * 60)
    print(f"  Primary model   : {result['primary_model']}")
    print(f"  Alternate model : {result['alternate_model']}")
    print(f"  Calls replayed  : {result['n_calls']}")
    print()
    print(f"  Quality scores (RDAB heuristic):")
    print(f"    Primary   : {result['primary_mean_score']:.4f}")
    print(f"    Alternate : {result['alternate_mean_score']:.4f}")
    print(f"    Delta     : {result['delta']:+.4f}  (95% CI [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}])")

    ci_lo, ci_hi = result["ci_low"], result["ci_high"]
    if ci_lo > 0:
        verdict = "✓ Alternate is statistically BETTER on quality."
    elif ci_hi < 0:
        verdict = "✗ Alternate is statistically WORSE on quality."
    else:
        verdict = "~ Quality difference is NOT statistically significant (CI straddles 0)."
    print(f"    Verdict   : {verdict}")

    print()
    primary_usd = result["primary_cost_usd"]
    alternate_usd = result["alternate_cost_usd"]
    savings_per = result["savings_per_call_usd"]
    savings_total = savings_per * result["n_calls"]
    pct = (savings_total / primary_usd * 100) if primary_usd > 0 else 0.0
    print(f"  Cost (this run):")
    print(f"    Primary   : ${primary_usd:.6f}")
    print(f"    Alternate : ${alternate_usd:.6f}")
    print(f"    Savings   : ${savings_total:.6f} total  (${savings_per:.8f}/call, {pct:.1f}%)")
    print("═" * 60)
    print()
    print("Full JSON response:")
    print(json.dumps(result, indent=2))


async def main() -> None:
    run_id = await capture_traces()
    result = submit_replay(run_id)
    print_report(result)


if __name__ == "__main__":
    asyncio.run(main())
