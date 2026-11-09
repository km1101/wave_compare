from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
import numpy as np
import pandas as pd
from .config import ID_COLUMNS, TIME_COLUMNS, load_platform_mapping

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    pass


def _build_timestamp(df: pd.DataFrame) -> pd.Series:
    missing = [c for c in TIME_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError(f"Missing time columns: {missing}")
    parts = df[TIME_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return pd.to_datetime(
        dict(year=parts["Year"], month=parts["Month"], day=parts["Day"],
             hour=parts["Hour"], minute=parts["Minute"]), errors="coerce")


def _pick_id_column(df: pd.DataFrame, prefer: Optional[str] = None) -> str:
    if prefer and prefer in df.columns:
        return prefer
    for c in ID_COLUMNS:
        if c in df.columns:
            return c
    raise DataLoadError(f"No platform id column found. Looked for: {ID_COLUMNS}")


def load_wave_csv(path, platform_mapping=None, id_column=None,
                  extra_columns=None, drop_all_null_rows=True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise DataLoadError(f"{path} contains no rows.")

    id_col = _pick_id_column(df, prefer=id_column)
    mapping = platform_mapping if platform_mapping is not None else load_platform_mapping()

    # Build timestamp without fragmentation warning
    ts = _build_timestamp(df)
    n_bad = ts.isna().sum()
    if n_bad:
        logger.warning("Dropping %d rows with unparseable timestamps.", n_bad)
    good = ts.notna()
    df = df[good].copy()
    df["timestamp"] = ts[good].values

    df["platform_id"] = df[id_col].astype("Int64").astype(str)
    df["platform"] = df["platform_id"].map(mapping).fillna("Platform " + df["platform_id"])

    drop_cols = set(TIME_COLUMNS) | set(ID_COLUMNS) | {"Format ID"}
    keep_extra = set(extra_columns) if extra_columns else set()
    numeric_cols = [
        c for c in df.columns
        if c not in drop_cols and c not in ("timestamp", "platform_id", "platform")
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


def list_available_parameters(df: pd.DataFrame):
    exclude = {"timestamp", "platform_id", "platform"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def list_platforms(df: pd.DataFrame):
    return sorted(df["platform"].dropna().unique().tolist())


def filter_date_range(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out[out["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["timestamp"] <= pd.Timestamp(end)]
    return out
