"""
plotting.py
-----------
Plotly figure builders for the wave-platform comparison dashboard.
Every function returns a go.Figure — nothing is rendered directly, so
these can be reused outside Streamlit (e.g. exported to PNG/HTML).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .statistics import ComparisonStats
from . import statistics as stats_mod
from .config import describe_param

TEMPLATE = "plotly_white"
_TEMPLATE = TEMPLATE   # alias used by some functions


def _unit_label(parameter: str) -> str:
    """Return 'Label (unit)' for a known parameter, or just the column name."""
    info = describe_param(parameter)
    if info.label and info.unit:
        return f"{info.label} ({info.unit})"
    if info.label:
        return info.label
    return parameter


def time_series_overlay(
    df_wide: pd.DataFrame, ref: str, comp: str, label: str, unit: str = ""
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_wide["timestamp"], y=df_wide[ref], name=ref, mode="lines+markers", marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=df_wide["timestamp"], y=df_wide[comp], name=comp, mode="lines+markers", marker=dict(size=4)))
    fig.update_layout(
        title=f"{label} — Time Series Comparison",
        xaxis_title="Time",
        yaxis_title=f"{label} ({unit})" if unit else label,
        template=TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def difference_series(df_wide: pd.DataFrame, ref: str, comp: str, label: str, unit: str = "") -> go.Figure:
    diff = df_wide[comp] - df_wide[ref]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_wide["timestamp"], y=diff, mode="lines", name="Difference", line=dict(color="#d62728")))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=diff.mean(), line_dash="dot", line_color="orange",
                  annotation_text=f"mean={diff.mean():.3f}")
    fig.update_layout(
        title=f"{label} — Difference ({comp} − {ref})",
        xaxis_title="Time",
        yaxis_title=f"Δ {label} ({unit})" if unit else f"Δ {label}",
        template=TEMPLATE,
    )
    return fig


def rolling_mean_difference(
    df_wide: pd.DataFrame,
    ref: str,
    comp: str,
    label: str,
    unit: str = "",
    window: str = "1D",
) -> go.Figure:
    """Rolling mean ± SD of (comp − ref).

    The ±SD shaded band uses Plotly's fill="tonexty" technique, which
    requires BOTH band traces to be free of NaN breaks.  rolling().std()
    returns NaN for any sub-window with only one observation, which causes
    the fill to disappear silently for shorter windows (1D, 3D).

    The fix lives in statistics.rolling_stats(): NaN SD values are replaced
    with 0 (min_periods=1 already keeps the mean continuous), so the band
    collapses to the mean line at sparse points rather than breaking.
    """
    diff = df_wide[comp] - df_wide[ref]
    tmp = pd.DataFrame({
        "timestamp": df_wide["timestamp"],
        "diff": diff,
    })
    tmp = tmp.sort_values("timestamp")

    rolling = stats_mod.rolling_stats(
        tmp.set_index("timestamp")["diff"],
        window=window,
    ).reset_index()

    upper = rolling["rolling_mean"] + rolling["rolling_std"]
    lower = rolling["rolling_mean"] - rolling["rolling_std"]

    fig = go.Figure()

    # Upper band boundary (invisible line — just sets the ceiling for fill)
    fig.add_trace(go.Scatter(
        x=rolling["timestamp"],
        y=upper,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Lower band boundary — filled back to the upper trace
    fig.add_trace(go.Scatter(
        x=rolling["timestamp"],
        y=lower,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(214,39,40,0.2)",
        name="±1 SD",
    ))

    # Rolling mean line on top
    fig.add_trace(go.Scatter(
        x=rolling["timestamp"],
        y=rolling["rolling_mean"],
        mode="lines",
        name=f"{window} rolling mean",
        line=dict(color="#d62728", width=2),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"{label} — Rolling Mean Difference ± SD ({comp} − {ref}, {window})",
        xaxis_title="Time",
        yaxis_title=f"Δ {label} ({unit})" if unit else f"Δ {label}",
        template=TEMPLATE,
        hovermode="x unified",
    )
    return fig


def scatter_with_fit(df_wide: pd.DataFrame, ref: str, comp: str, stats_obj: ComparisonStats, label: str, unit: str = "") -> go.Figure:
    x = df_wide[ref]
    y = df_wide[comp]
    lims = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Observations",
                              marker=dict(size=5, opacity=0.55)))
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="1:1 line",
                              line=dict(color="gray", dash="dash")))
    fit_y = [stats_obj.slope * v + stats_obj.intercept for v in lims]
    fig.add_trace(go.Scatter(x=lims, y=fit_y, mode="lines", name="Linear fit",
                              line=dict(color="#1f77b4")))

    annotation = (
        f"N={stats_obj.n}<br>"
        f"R²={stats_obj.r_squared:.3f}<br>"
        f"Bias={stats_obj.bias:.3f} {unit}<br>"
        f"RMSE={stats_obj.rmse:.3f} {unit}<br>"
        f"SI={stats_obj.scatter_index:.3f}"
    )
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper", showarrow=False,
        text=annotation, align="left", bordercolor="gray", borderwidth=1,
        bgcolor="white", opacity=0.9,
    )
    fig.update_layout(
        title=f"{label} — Scatter ({comp} vs {ref})",
        xaxis_title=f"{ref} ({unit})" if unit else ref,
        yaxis_title=f"{comp} ({unit})" if unit else comp,
        template=TEMPLATE,
    )
    return fig


def bland_altman(df_wide: pd.DataFrame, ref: str, comp: str, stats_obj: ComparisonStats, label: str, unit: str = "") -> go.Figure:
    mean_vals = (df_wide[ref] + df_wide[comp]) / 2
    diff_vals = df_wide[comp] - df_wide[ref]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_vals, y=diff_vals, mode="markers", name="Pairs",
                              marker=dict(size=5, opacity=0.55)))
    fig.add_hline(y=stats_obj.loa_bias, line_color="orange",
                  annotation_text=f"bias={stats_obj.loa_bias:.3f}")
    fig.add_hline(y=stats_obj.loa_upper, line_dash="dash", line_color="firebrick",
                  annotation_text=f"+1.96 SD={stats_obj.loa_upper:.3f}")
    fig.add_hline(y=stats_obj.loa_lower, line_dash="dash", line_color="firebrick",
                  annotation_text=f"-1.96 SD={stats_obj.loa_lower:.3f}")
    fig.update_layout(
        title=f"{label} — Bland–Altman (Limits of Agreement)",
        xaxis_title=f"Mean of {ref} & {comp} ({unit})" if unit else "Mean of both platforms",
        yaxis_title=f"Difference ({comp} − {ref}) ({unit})" if unit else "Difference",
        template=TEMPLATE,
    )
    return fig


def distribution_compare(df_wide: pd.DataFrame, ref: str, comp: str, label: str, unit: str = "", kind: str = "histogram") -> go.Figure:
    fig = go.Figure()
    if kind == "histogram":
        fig.add_trace(go.Histogram(x=df_wide[ref], name=ref, opacity=0.6, nbinsx=40))
        fig.add_trace(go.Histogram(x=df_wide[comp], name=comp, opacity=0.6, nbinsx=40))
        fig.update_layout(barmode="overlay")
    elif kind == "box":
        fig.add_trace(go.Box(y=df_wide[ref], name=ref))
        fig.add_trace(go.Box(y=df_wide[comp], name=comp))
    elif kind == "violin":
        fig.add_trace(go.Violin(y=df_wide[ref], name=ref, box_visible=True, meanline_visible=True))
        fig.add_trace(go.Violin(y=df_wide[comp], name=comp, box_visible=True, meanline_visible=True))
    else:
        raise ValueError(f"Unknown distribution plot kind: {kind}")

    fig.update_layout(
        title=f"{label} — Distribution Comparison ({kind})",
        yaxis_title=f"{label} ({unit})" if unit and kind != "histogram" else label,
        xaxis_title=f"{label} ({unit})" if unit and kind == "histogram" else None,
        template=TEMPLATE,
    )
    return fig


def rolling_agreement(
    time_binned_df: pd.DataFrame, metric: str = "rmse", label: str = "", unit: str = ""
) -> go.Figure:
    """Plot how a chosen metric (rmse/bias/scatter_index/...) evolves over time bins."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=time_binned_df["bin_start"], y=time_binned_df[metric], name=metric.upper()))
    fig.update_layout(
        title=f"{label} — {metric.upper()} by Period",
        xaxis_title="Period",
        yaxis_title=f"{metric.upper()} ({unit})" if unit else metric.upper(),
        template=TEMPLATE,
    )
    return fig


def multi_param_summary_table(rows: list, label_col: str = "Parameter") -> go.Figure:
    """rows: list of dicts with comparable summary stats per parameter."""
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color="#1f3b57", font=dict(color="white"), align="left"),
        cells=dict(values=[df[c] for c in df.columns], align="left"),
    )])
    fig.update_layout(title="Multi-Parameter Summary", template=TEMPLATE)
    return fig


def monthly_summary_plot(df: pd.DataFrame, parameter: str) -> go.Figure:
    """Monthly mean per platform with min–max whiskers, as an error-bar plot.

    Uses ``periodic_stats`` with freq="ME" (month-end) to aggregate each
    platform's readings into monthly bins, then plots mean ± min/max range
    as error bars so over- or under-reading in a given month is immediately
    visible.
    """
    monthly = stats_mod.periodic_stats(df, parameter, freq="ME")
    if monthly.empty:
        return go.Figure()

    fig = go.Figure()
    for platform, sub in monthly.groupby("platform"):
        sub = sub.sort_values("period")
        fig.add_trace(go.Scatter(
            x=sub["period"],
            y=sub["mean"],
            mode="lines+markers",
            name=platform,
            error_y=dict(
                type="data",
                array=(sub["max"] - sub["mean"]).tolist(),
                arrayminus=(sub["mean"] - sub["min"]).tolist(),
                visible=True,
                thickness=1.5,
                width=4,
            ),
        ))

    fig.update_layout(
        template=_TEMPLATE,
        title=f"{_unit_label(parameter)} — Monthly summary (mean + min–max range)",
        xaxis_title="Month",
        yaxis_title=_unit_label(parameter),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig
