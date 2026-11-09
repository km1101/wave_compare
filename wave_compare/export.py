from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd
import plotly.graph_objects as go


def stats_to_dataframe(stats_dict): return pd.DataFrame([stats_dict])


def export_stats_csv(stats_dict, path):
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    stats_to_dataframe(stats_dict).to_csv(out, index=False)
    return out


def export_stats_excel(tables: Dict[str, pd.DataFrame], path: str) -> Path:
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet, df in tables.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    return out


def export_figure(fig: go.Figure, path: str, scale: int = 2) -> Path:
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".html":
        fig.write_html(out, include_plotlyjs="cdn")
    else:
        fig.write_image(out, scale=scale)
    return out
