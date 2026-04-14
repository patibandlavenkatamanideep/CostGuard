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


def _read_csv_robust(file_obj: io.IOBase, filename: str) -> pd.DataFrame:
    """
    Read a CSV tolerating the four most common real-world problems:
    1. Non-UTF-8 encoding  (BX-Books, European datasets)
    2. Non-comma separator (BX-Books uses ';', some exports use tab/pipe)
    3. Bad / extra fields  (C error: Expected N fields, saw M)
    4. BOM prefix on first column name

    Strategy:
    - Read the raw bytes once so seek() is never needed.
    - Try (encoding, sep) pairs; sep=None lets pandas sniff the delimiter.
    - Guarantee success via a final latin-1 + errors='replace' fallback that
      cannot raise UnicodeDecodeError on any byte sequence.
    """
    file_obj.seek(0)
    raw: bytes = file_obj.read()

    # Encodings to try in order.  latin-1 maps every byte 0x00-0xFF so it
    # should always succeed at decoding; it's listed early to catch latin/cp
    # files before the BOM-aware utf-8-sig pass.
    encodings = ("utf-8-sig", "utf-8", "latin-1", "cp1252")
    last_exc: Exception | None = None

    for encoding in encodings:
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                encoding=encoding,
                sep=None,              # auto-sniff: comma, semicolon, tab, pipe …
                on_bad_lines="skip",   # skip rows with wrong field count
                engine="python",       # required for sep=None and more lenient overall
            )
            logger.info(f"Loaded '{filename}' with encoding={encoding}, sep=auto-sniffed")
            return df
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            last_exc = exc
            continue

    # Final guaranteed fallback: latin-1 decodes every byte; 'replace' handles
    # any residual codec error.  This path should be unreachable in practice.
    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            encoding="latin-1",
            encoding_errors="replace",
            sep=None,
            on_bad_lines="skip",
            engine="python",
        )
        logger.warning(f"Loaded '{filename}' via encoding fallback (latin-1/replace)")
        return df
    except Exception as exc:
        last_exc = exc

    raise DataLoadError(
        f"Could not parse '{filename}' with any encoding/separator combination. "
        f"Last error: {last_exc}"
    )


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
            with open(path, "rb") as fh:
                df = _read_csv_robust(fh, path.name)
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
            df = _read_csv_robust(buf, filename)
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
