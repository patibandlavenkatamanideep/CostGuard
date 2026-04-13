"""
Generates realistic evaluation questions from a dataset.
These questions drive the RealDataAgentBench-style evaluation:
models are judged on how accurately they answer data questions.
"""

from __future__ import annotations

import random

import pandas as pd


def generate_questions(df: pd.DataFrame, num_questions: int = 5, seed: int = 42) -> list[str]:
    """
    Generate a diverse set of analytical questions about the dataset.
    Questions span: aggregation, filtering, correlation, anomaly detection,
    and natural-language description tasks.
    """
    rng = random.Random(seed)
    questions: list[str] = []

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols = df.columns.tolist()

    # ── Template pool ──────────────────────────────────────────────────────
    templates: list[str] = []

    # Aggregation questions
    for col in num_cols[:3]:
        templates += [
            f"What is the average value of '{col}' across all rows?",
            f"What is the maximum value of '{col}', and in which row does it appear?",
            f"How many rows have a '{col}' value above the median?",
        ]

    # Categorical distribution
    for col in cat_cols[:2]:
        templates += [
            f"What are the top 3 most frequent values in the '{col}' column?",
            f"How many unique values does the '{col}' column contain?",
        ]

    # Missing data
    templates.append("Which column has the highest percentage of missing values?")
    templates.append("How many rows contain at least one missing value?")

    # Correlation (if multiple numeric cols)
    if len(num_cols) >= 2:
        c1, c2 = num_cols[0], num_cols[1]
        templates.append(f"Is there a correlation between '{c1}' and '{c2}'? Describe it.")

    # Summary
    templates.append("Provide a one-paragraph executive summary of this dataset.")
    templates.append("What are the 3 most interesting patterns or anomalies in this data?")

    # General
    templates += [
        "How many rows and columns does this dataset have?",
        "What data types are present in this dataset?",
        "Which columns might be useful as features for a machine learning model?",
    ]

    # Sample without replacement (or with if not enough)
    n = min(num_questions, len(templates))
    selected = rng.sample(templates, n)
    if num_questions > len(templates):
        # Fill remainder with repeats from general questions
        extra = rng.choices(templates, k=num_questions - len(templates))
        selected = selected + extra

    return selected[:num_questions]
