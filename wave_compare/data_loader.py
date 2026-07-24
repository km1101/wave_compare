"""
data_loader.py
---------------
Reads the raw wave-buoy export CSV and turns it into a clean, long-format
DataFrame: one row per (timestamp, platform), with a `platform` column
already mapped to a human-readable name.

Designed to be source-agnostic: as long as a CSV has MotusID (or AWSID),
Year/Month/Day/Hour/Minute, and numeric parameter columns, it will load.
This also leaves room to add a second DataSource later (e.g. a model
hindcast) that doesn't necessarily share the same raw schema, as long as
it's adapted into the same long-format shape (timestamp, platform, params...).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from .config import ID_COLUMNS, TIME_COLUMNS, load_platform_mapping

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when the source file can't be parsed into the expected shape."""


def _build_timestamp(df: pd.DataFrame) -> pd.Series:
    missing = [c for c in TIME_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError(f"Missing time columns required to build a timestamp: {missing}")
    parts = df[TIME_COLUMNS].apply(pd.to_numeric, errors="coerce")
    ts = pd.to_datetime(
        dict(
            year=parts["Year"],
            month=parts["Month"],
            day=parts["Day"],
            hour=parts["Hour"],
            minute=parts["Minute"],
        ),
        errors="coerce",
    )
    return ts


def _pick_id_column(df: pd.DataFrame, prefer: Optional[str] = None) -> str:
    if prefer and prefer in df.columns:
        return prefer
    for c in ID_COLUMNS:
        if c in df.columns:
            return c
    raise DataLoadError(f"No platform id column found. Looked for: {ID_COLUMNS}")


def load_wave_csv(
    path: Union[str, Path],
    platform_mapping: Optional[Dict[str, str]] = None,
    id_column: Optional[str] = None,
    extra_columns: Optional[Iterable[str]] = None,
    drop_all_null_rows: bool = True,
) -> pd.DataFrame:
    """
    Load a raw wave-buoy CSV into a tidy long-format DataFrame.

    Parameters
    ----------
    path : path to the CSV file
    platform_mapping : {raw_id: platform_name}. If None, loads from
        config/platform_mapping.json (or the built-in default).
    id_column : force a specific id column (e.g. "MotusID"). Auto-detected
        if not given (prefers MotusID, falls back to AWSID).
    extra_columns : any additional raw columns to carry through untouched
        (e.g. Lat_Low/Long_Low) beyond the standard parameter catalog.
    drop_all_null_rows : drop rows where every parameter column is NaN.

    Returns
    -------
    DataFrame with columns: timestamp, platform_id, platform, raw_id_column,
    and all remaining numeric columns from the source file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        raise DataLoadError(f"{path} contains no rows.")

    id_col = _pick_id_column(df, prefer=id_column)
    mapping = platform_mapping if platform_mapping is not None else load_platform_mapping()

    df["timestamp"] = _build_timestamp(df)
    n_bad_ts = df["timestamp"].isna().sum()
    if n_bad_ts:
        logger.warning("Dropping %d rows with unparseable timestamps.", n_bad_ts)
    df = df.dropna(subset=["timestamp"]).copy()

    df["platform_id"] = df[id_col].astype("Int64").astype(str)
    df["platform"] = df["platform_id"].map(mapping).fillna(
        "Platform " + df["platform_id"]
    )

    # Keep: timestamp/platform metadata + every numeric column not used for
    # identification/time bookkeeping. This naturally includes the full
    # parameter catalog (H13, Tz, Hmax, Tmax, Hswell, Tswell, ...) plus
    # anything else numeric in the file, so nothing is artificially excluded.
    drop_cols = set(TIME_COLUMNS) | set(ID_COLUMNS) | {"Format ID", "platform_id_raw"}
    keep_extra = set(extra_columns) if extra_columns else set()

    numeric_cols = [
        c for c in df.columns
        if c not in drop_cols
        and c not in ("timestamp", "platform_id", "platform")
        and (pd.api.types.is_numeric_dtype(df[c]) or c in keep_extra)
    ]

    tidy = df[["timestamp", "platform_id", "platform"] + numeric_cols].copy()
    tidy = tidy.sort_values(["platform", "timestamp"]).reset_index(drop=True)

    if drop_all_null_rows:
        before = len(tidy)
        tidy = tidy.dropna(subset=numeric_cols, how="all").reset_index(drop=True)
        dropped = before - len(tidy)
        if dropped:
            logger.info("Dropped %d fully-empty parameter rows.", dropped)

    return tidy


def list_available_parameters(df: pd.DataFrame) -> List[str]:
    """Numeric columns in a tidy frame that are usable as comparison parameters."""
    exclude = {"timestamp", "platform_id", "platform"}
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

def list_platforms(df: pd.DataFrame) -> List[str]:
    return sorted(df["platform"].dropna().unique().tolist())


def filter_date_range(
    df: pd.DataFrame, start=None, end=None
) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out[out["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["timestamp"] <= pd.Timestamp(end)]
    return out
