"""
Data ingestion layer.
Loads CSV and Parquet files, validates them, computes dataset statistics,
and produces a representative sample for evaluation.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from backend.logger import logger
from backend.models import DatasetStats

if TYPE_CHECKING:
    pass


SUPPORTED_EXTENSIONS = {".csv", ".parquet"}
MAX_SAMPLE_ROWS = int(os.getenv("EVAL_MAX_ROWS", "500"))
SAMPLE_SEED = int(os.getenv("EVAL_SAMPLE_SEED", "42"))


class DataLoadError(Exception):
    """Raised when a file cannot be loaded or is invalid."""


def load_file(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV or Parquet file into a DataFrame.
    Raises DataLoadError on unsupported format or corrupt data.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    try:
        if suffix == ".csv":
            for encoding in ("utf-8", "latin-1", "cp1252", "utf-8-sig"):
                try:
                    df = pd.read_csv(path, low_memory=False, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise DataLoadError(f"Could not decode '{path.name}' with any supported encoding.")
        else:
            df = pd.read_parquet(path)
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"Failed to parse file '{path.name}': {exc}") from exc

    if df.empty:
        raise DataLoadError("The uploaded file contains no data rows.")

    logger.info(f"Loaded '{path.name}': {len(df)} rows × {len(df.columns)} cols")
    return df


def load_bytes(content: bytes, filename: str) -> pd.DataFrame:
    """Load a file from raw bytes (used in API file upload handler)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(f"Unsupported file format: '{suffix}'")

    buf = io.BytesIO(content)
    try:
        if suffix == ".csv":
            for encoding in ("utf-8", "latin-1", "cp1252", "utf-8-sig"):
                try:
                    buf.seek(0)
                    df = pd.read_csv(buf, low_memory=False, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise DataLoadError(f"Could not decode '{filename}' with any supported encoding (utf-8, latin-1, cp1252).")
        else:
            df = pd.read_parquet(buf)
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"Failed to parse '{filename}': {exc}") from exc

    if df.empty:
        raise DataLoadError("Uploaded file contains no data rows.")

    return df


def compute_stats(df: pd.DataFrame, filename: str, file_size_bytes: int) -> DatasetStats:
    """Compute a rich statistics summary for the dataset."""
    missing_pct = float(df.isnull().mean().mean() * 100)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return DatasetStats(
        rows=len(df),
        columns=len(df.columns),
        column_names=list(df.columns),
        dtypes=dtypes,
        missing_pct=round(missing_pct, 2),
        file_size_kb=round(file_size_bytes / 1024, 2),
        file_format=Path(filename).suffix.lstrip(".").upper(),
    )


def sample_dataframe(df: pd.DataFrame, max_rows: int = MAX_SAMPLE_ROWS) -> pd.DataFrame:
    """Return a representative sample of the DataFrame for evaluation."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=SAMPLE_SEED)


def dataframe_to_prompt_text(df: pd.DataFrame, max_rows: int = 50) -> str:
    """
    Convert a DataFrame sample to a compact text representation
    suitable for inclusion in an LLM prompt.
    """
    sample = df.head(max_rows)
    schema_lines = [
        f"  - {col} ({dtype})" for col, dtype in df.dtypes.items()
    ]
    schema_text = "\n".join(schema_lines)

    stats_parts = []
    for col in df.select_dtypes(include="number").columns[:5]:
        s = df[col].describe()
        stats_parts.append(
            f"  {col}: min={s['min']:.2f}, max={s['max']:.2f}, "
            f"mean={s['mean']:.2f}, nulls={df[col].isnull().sum()}"
        )
    stats_text = "\n".join(stats_parts) if stats_parts else "  (no numeric columns)"

    csv_preview = sample.to_csv(index=False)

    return (
        f"Dataset Shape: {len(df)} rows × {len(df.columns)} columns\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Numeric Statistics:\n{stats_text}\n\n"
        f"Data Preview (first {min(max_rows, len(df))} rows):\n{csv_preview}"
    )
