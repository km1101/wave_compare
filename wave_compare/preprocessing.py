"""
preprocessing.py
-----------------
Turns the tidy long-format DataFrame into an aligned wide-format pair (or
N-way set) ready for statistical comparison, plus optional outlier removal
and resampling helpers.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def align_platforms(
    df: pd.DataFrame,
    platforms: Sequence[str],
    parameter: str,
    tolerance: str = "30min",
) -> pd.DataFrame:
    """
    Align two (or more) platforms' readings of `parameter` onto a common
    timestamp index using nearest-neighbor matching within `tolerance`.

    Returns a wide DataFrame: timestamp, <platform_1>, <platform_2>, ...
    Rows where any platform is missing are NOT dropped here (kept as NaN);
    drop them downstream with .dropna() when a complete pair is required.
    """
    if len(platforms) < 2:
        raise ValueError("Need at least two platforms to align.")

    frames = []
    for p in platforms:
        sub = (
            df[df["platform"] == p][["timestamp", parameter]]
            .dropna(subset=[parameter])
            .sort_values("timestamp")
            .rename(columns={parameter: p})
        )
        frames.append(sub)

    base = frames[0]
    for nxt in frames[1:]:
        base = pd.merge_asof(
            base.sort_values("timestamp"),
            nxt.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(tolerance),
        )

    return base.sort_values("timestamp").reset_index(drop=True)


def drop_incomplete(df_wide: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Drop rows with any NaN among the given (or all non-timestamp) columns."""
    cols = columns or [c for c in df_wide.columns if c != "timestamp"]
    return df_wide.dropna(subset=cols).reset_index(drop=True)


def remove_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: float = 3.0,
    window: int = 11,
) -> pd.Series:
    """
    Flag-and-NaN outliers in a single series. Returns a copy with outliers
    replaced by NaN (caller decides whether to drop or interpolate).

    method:
        "iqr"     - classic Tukey fence, threshold = IQR multiplier (default 1.5x3=mild->use 1.5 typically;
                    here `threshold` multiplies the IQR directly)
        "zscore"  - |z| > threshold flagged
        "hampel"  - rolling median + MAD, good for spiky buoy data
    """
    s = series.copy()

    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
        mask = (s < lo) | (s > hi)

    elif method == "zscore":
        mu, sigma = s.mean(), s.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            mask = pd.Series(False, index=s.index)
        else:
            z = (s - mu) / sigma
            mask = z.abs() > threshold

    elif method == "hampel":
        med = s.rolling(window, center=True, min_periods=1).median()
        diff = (s - med).abs()
        mad = diff.rolling(window, center=True, min_periods=1).median()
        # 1.4826 scales MAD to be comparable to a standard deviation under normality
        mask = diff > threshold * 1.4826 * mad

    else:
        raise ValueError(f"Unknown outlier method: {method}")

    s[mask] = np.nan
    return s


def resample_platform_series(
    df: pd.DataFrame,
    platform: str,
    parameter: str,
    freq: str = "1h",
    agg: str = "mean",
) -> pd.Series:
    """Resample one platform's parameter to a regular frequency."""
    sub = df[df["platform"] == platform].set_index("timestamp")[parameter].sort_index()
    return sub.resample(freq).agg(agg)


def compute_difference(df_wide: pd.DataFrame, ref: str, comp: str) -> pd.DataFrame:
    """Add Difference (comp-ref) and PercentDiff columns to an aligned wide frame."""
    out = df_wide.copy()
    out["Difference"] = out[comp] - out[ref]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["PercentDiff"] = np.where(
            out[ref] != 0, 100 * out["Difference"] / out[ref], np.nan
        )
    return out
