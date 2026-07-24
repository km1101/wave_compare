"""
export.py
---------
Small helpers to export comparison results: stats tables to CSV/Excel,
figures to PNG/HTML, for reporting outside the dashboard itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import plotly.graph_objects as go


def stats_to_dataframe(stats_dict: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([stats_dict])


def export_stats_csv(stats_dict: Dict[str, float], path: str) -> Path:
    df = stats_to_dataframe(stats_dict)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def export_stats_excel(tables: Dict[str, pd.DataFrame], path: str) -> Path:
    """tables: {sheet_name: dataframe}"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet, df in tables.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    return out


def export_figure(fig: go.Figure, path: str, scale: int = 2) -> Path:
    """Export a Plotly figure to .png/.svg/.html depending on extension."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".html":
        fig.write_html(out, include_plotlyjs="cdn")
    else:
        fig.write_image(out, scale=scale)
    return out
