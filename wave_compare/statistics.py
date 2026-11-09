"""
statistics.py
-------------
Pairwise comparison statistics between two platforms for a given parameter.
All functions take already-aligned (same-length, paired) numpy arrays /
pandas Series — use preprocessing.align_platforms() + dropna() first.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from ._pandas_compact import freq as _freq


@dataclass
class ComparisonStats:
    n: int
    ref_mean: float
    comp_mean: float
    bias: float            # mean(comp - ref)
    mae: float              # mean absolute error
    rmse: float
    scatter_index: float    # RMSE / mean(ref) — dimensionless, common in wave validation
    pearson_r: float
    spearman_r: float
    r_squared: float
    slope: float
    intercept: float
    loa_lower: float        # Bland-Altman lower limit of agreement
    loa_upper: float        # Bland-Altman upper limit of agreement
    loa_bias: float         # Bland-Altman mean difference
    loa_sd: float           # SD of differences

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_comparison_stats(ref: pd.Series, comp: pd.Series) -> ComparisonStats:
    """
    ref / comp: paired, equal-length series with no NaNs (caller's job to
    align + drop incomplete rows first — see preprocessing.align_platforms).
    `ref` is the baseline platform comp is measured against (just affects
    which side bias/SI are signed/normalised on — swap if needed).
    """
    ref = np.asarray(ref, dtype=float)
    comp = np.asarray(comp, dtype=float)

    if len(ref) != len(comp):
        raise ValueError("ref and comp must be the same length (already aligned).")
    if len(ref) < 2:
        raise ValueError("Need at least 2 paired points to compute statistics.")

    diff = comp - ref
    n = len(ref)

    bias = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ref_mean = float(np.mean(ref))
    scatter_index = float(rmse / ref_mean) if ref_mean != 0 else float("nan")

    pearson_r, _ = sp_stats.pearsonr(ref, comp) if np.std(ref) > 0 and np.std(comp) > 0 else (float("nan"), None)
    spearman_r, _ = sp_stats.spearmanr(ref, comp) if np.std(ref) > 0 and np.std(comp) > 0 else (float("nan"), None)

    if np.unique(ref).size > 1 and np.unique(comp).size > 1:
        try:
            slope, intercept, r_value, _, _ = sp_stats.linregress(ref, comp)
            r_squared = float(r_value ** 2)
        except ValueError:
            slope = intercept = r_squared = float("nan")
    else:
        slope = intercept = r_squared = float("nan")

    loa_bias = float(np.mean(diff))
    loa_sd = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    loa_lower = loa_bias - 1.96 * loa_sd
    loa_upper = loa_bias + 1.96 * loa_sd

    return ComparisonStats(
        n=n,
        ref_mean=ref_mean,
        comp_mean=float(np.mean(comp)),
        bias=bias,
        mae=mae,
        rmse=rmse,
        scatter_index=scatter_index,
        pearson_r=float(pearson_r),
        spearman_r=float(spearman_r),
        r_squared=r_squared,
        slope=float(slope),
        intercept=float(intercept),
        loa_lower=float(loa_lower),
        loa_upper=float(loa_upper),
        loa_bias=loa_bias,
        loa_sd=loa_sd,
    )


def descriptive_stats(series: pd.Series) -> Dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {k: float("nan") for k in
                ["n", "mean", "std", "min", "p25", "median", "p75", "max", "skew", "kurtosis"]}
    return {
        "n": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
    }


def time_binned_stats(
    df_wide: pd.DataFrame, ref: str, comp: str, freq: str = "1d"
) -> pd.DataFrame:
    """
    Compute ComparisonStats per time bin (e.g. daily/weekly), useful for
    spotting periods of poor agreement. Returns a DataFrame indexed by bin.
    """
    out_rows = []
    grouped = df_wide.set_index("timestamp").groupby(pd.Grouper(freq=freq))
    for bin_start, chunk in grouped:
        chunk = chunk.dropna(subset=[ref, comp])
        if len(chunk) < 2:
            continue
        stats_obj = compute_comparison_stats(chunk[ref], chunk[comp])
        row = {"bin_start": bin_start, **stats_obj.to_dict()}
        out_rows.append(row)
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Rolling / time-based statistics
# ---------------------------------------------------------------------------

def rolling_stats(series: pd.Series, window: str = "6h") -> pd.DataFrame:
    """Rolling mean and standard deviation over a time window.

    ``series`` must be indexed by timestamp.

    NaN handling
    ------------
    ``rolling().std()`` returns NaN for any sub-window that contains only
    one observation (variance is undefined for n=1). Those NaN values in
    the SD band break Plotly's ``fill="tonexty"`` shading — the filled area
    simply does not render for the gap. We fill them with 0 so the band
    collapses to the mean line at sparse points rather than disappearing.
    This is visually honest: a window too narrow to have a spread is shown
    as zero spread, not as a missing segment.
    """
    roll = series.rolling(_freq(window), min_periods=1)
    mean = roll.mean()
    std = roll.std().fillna(0)   # replace NaN SD with 0 to keep fill="tonexty" continuous
    return pd.DataFrame({"rolling_mean": mean, "rolling_std": std})


def periodic_stats(df: pd.DataFrame, parameter: str, freq: str = "ME") -> pd.DataFrame:
    """Per-period (monthly by default) descriptive stats for each platform.

    Parameters
    ----------
    df : tidy long-format DataFrame with columns timestamp, platform, <parameter>
    parameter : column to summarise
    freq : pandas offset alias for the period bin — "ME" (month-end, pandas ≥2.2)
           is the default; the _freq shim is applied so "M" also works on
           older pandas. Any other resample-compatible alias is accepted.

    Returns
    -------
    DataFrame with columns: period (bin label), platform, n, mean, std,
    min, p25, median, p75, max.
    """
    if parameter not in df.columns:
        return pd.DataFrame()

    rows = []
    for platform, grp in df.groupby("platform"):
        series = (
            grp[["timestamp", parameter]]
            .dropna(subset=[parameter])
            .set_index("timestamp")[parameter]
            .sort_index()
        )
        if series.empty:
            continue
        resampled = series.resample(_freq(freq))
        for period, chunk in resampled:
            chunk = chunk.dropna()
            if chunk.empty:
                continue
            rows.append({
                "period": period,
                "platform": platform,
                "n": int(chunk.count()),
                "mean": float(chunk.mean()),
                "std": float(chunk.std()) if len(chunk) > 1 else 0.0,
                "min": float(chunk.min()),
                "p25": float(chunk.quantile(0.25)),
                "median": float(chunk.median()),
                "p75": float(chunk.quantile(0.75)),
                "max": float(chunk.max()),
            })
    return pd.DataFrame(rows)
