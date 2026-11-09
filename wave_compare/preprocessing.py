from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np
import pandas as pd


def align_platforms(df, platforms, parameter, tolerance="30min"):
    if len(platforms) < 2:
        raise ValueError("Need at least two platforms to align.")
    frames = []
    for p in platforms:
        sub = (df[df["platform"] == p][["timestamp", parameter]]
               .dropna(subset=[parameter]).sort_values("timestamp")
               .rename(columns={parameter: p}))
        frames.append(sub)
    base = frames[0]
    for nxt in frames[1:]:
        base = pd.merge_asof(
            base.sort_values("timestamp"), nxt.sort_values("timestamp"),
            on="timestamp", direction="nearest",
            tolerance=pd.Timedelta(tolerance))
    return base.sort_values("timestamp").reset_index(drop=True)


def drop_incomplete(df_wide, columns=None):
    cols = columns or [c for c in df_wide.columns if c != "timestamp"]
    return df_wide.dropna(subset=cols).reset_index(drop=True)


def remove_outliers(series, method="iqr", threshold=3.0, window=11):
    s = series.copy()
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        mask = (s < q1 - threshold * iqr) | (s > q3 + threshold * iqr)
    elif method == "zscore":
        mu, sigma = s.mean(), s.std(ddof=0)
        mask = (((s - mu) / sigma).abs() > threshold) if sigma > 0 else pd.Series(False, index=s.index)
    elif method == "hampel":
        med = s.rolling(window, center=True, min_periods=1).median()
        diff = (s - med).abs()
        mad = diff.rolling(window, center=True, min_periods=1).median()
        mask = diff > threshold * 1.4826 * mad
    else:
        raise ValueError(f"Unknown outlier method: {method}")
    s[mask] = np.nan
    return s


def resample_platform_series(df, platform, parameter, freq="1h", agg="mean"):
    sub = df[df["platform"] == platform].set_index("timestamp")[parameter].sort_index()
    return sub.resample(freq).agg(agg)


def compute_difference(df_wide, ref, comp):
    out = df_wide.copy()
    out["Difference"] = out[comp] - out[ref]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["PercentDiff"] = np.where(out[ref] != 0, 100 * out["Difference"] / out[ref], np.nan)
    return out
