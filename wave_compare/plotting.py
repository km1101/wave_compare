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

TEMPLATE = "plotly_white"


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
