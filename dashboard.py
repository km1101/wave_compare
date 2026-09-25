"""
dashboard.py
------------
Streamlit dashboard for comparing wave-buoy platforms.

Run with:
    streamlit run dashboard.py

Features
--------
- Upload your own CSV, or use the bundled data/wave.csv
- Platform name mapping editable in the sidebar (and persisted to
  config/platform_mapping.json), or upload your own mapping CSV/JSON
- Pick any 2+ platforms and any of the available numeric parameters
  (not limited to H13/Tz — every numeric column in the file is offered)
- Date range filter, outlier filtering, alignment tolerance
- Time series overlay, difference plot, scatter + regression, Bland-Altman,
  distribution comparison, period-binned agreement metrics
- Multi-parameter summary table across all selected parameters at once
- Export stats (CSV/Excel) and figures (PNG/HTML)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wave_compare.config import (
    PARAMETER_CATALOG, DEFAULT_PARAMETERS, describe_param,
    load_platform_mapping, save_platform_mapping,
)
from wave_compare.data_loader import load_wave_csv, list_available_parameters, list_platforms, filter_date_range
from wave_compare.preprocessing import align_platforms, drop_incomplete, remove_outliers, compute_difference
from wave_compare.statistics import compute_comparison_stats, descriptive_stats, time_binned_stats
from wave_compare import plotting
from wave_compare.export import export_stats_excel

st.set_page_config(page_title="Wave Platform Comparison", layout="wide", page_icon="🌊")

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CSV = DATA_DIR / "wave_data.csv"
EXPORT_DIR = Path(__file__).resolve().parent / "exports"


# --------------------------------------------------------------------------
# Sidebar: data + platform mapping
# --------------------------------------------------------------------------
st.sidebar.title("🌊 Wave Platform Comparison")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Data source")
uploaded_csv = st.sidebar.file_uploader("Upload wave data CSV (optional)", type=["csv"])

st.sidebar.subheader("2. Platform mapping")
mapping_mode = st.sidebar.radio(
    "Mapping source", ["Saved config", "Edit manually", "Upload mapping file"], index=0
)

if mapping_mode == "Saved config":
    platform_mapping = load_platform_mapping()
elif mapping_mode == "Upload mapping file":
    map_file = st.sidebar.file_uploader("Mapping CSV or JSON (id,name)", type=["csv", "json"], key="mapfile")
    if map_file is not None:
        tmp_path = EXPORT_DIR / f"_uploaded_mapping{Path(map_file.name).suffix}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(map_file.getvalue())
        platform_mapping = load_platform_mapping(str(tmp_path))
    else:
        platform_mapping = load_platform_mapping()
else:  # Edit manually
    base_mapping = load_platform_mapping()
    raw_text = st.sidebar.text_area(
        "id = name (one per line)",
        value="\n".join(f"{k} = {v}" for k, v in base_mapping.items()),
        height=100,
    )
    platform_mapping = {}
    for line in raw_text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            platform_mapping[k.strip()] = v.strip()
    if st.sidebar.button("💾 Save mapping"):
        save_platform_mapping(platform_mapping)
        st.sidebar.success("Saved to config/platform_mapping.json")

with st.sidebar.expander("Current mapping"):
    st.json(platform_mapping)


# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load(path_or_bytes, mapping, is_upload: bool):
    if is_upload:
        tmp = EXPORT_DIR / "_uploaded_data.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(path_or_bytes)
        return load_wave_csv(tmp, platform_mapping=mapping)
    return load_wave_csv(path_or_bytes, platform_mapping=mapping)


try:
    if uploaded_csv is not None:
        df = _load(uploaded_csv.getvalue(), platform_mapping, True)
        st.sidebar.success(f"Loaded {uploaded_csv.name}")
    else:
        if not DEFAULT_CSV.exists():
            st.error("No data uploaded and no bundled data/wave.csv found.")
            st.stop()
        df = _load(str(DEFAULT_CSV), platform_mapping, False)
        st.sidebar.info("Using bundled data/wave.csv")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

platforms_available = list_platforms(df)
params_available = list_available_parameters(df)

if len(platforms_available) < 2:
    st.warning(f"Only one platform found ({platforms_available}) — need at least two to compare.")
    st.stop()


# --------------------------------------------------------------------------
# Sidebar: selection controls
# --------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("3. Comparison setup")

platforms_selected = st.sidebar.multiselect(
    "Platforms to compare", platforms_available,
    default=platforms_available[:2],
)

default_params = [p for p in DEFAULT_PARAMETERS if p in params_available] or params_available[:3]
params_selected = st.sidebar.multiselect(
    "Parameters", params_available, default=default_params,
    format_func=lambda c: f"{c} — {describe_param(c).label}",
)

date_min, date_max = df["timestamp"].min(), df["timestamp"].max()
date_range = st.sidebar.date_input(
    "Date range", value=(date_min.date(), date_max.date()),
    min_value=date_min.date(), max_value=date_max.date(),
)

tolerance = st.sidebar.select_slider(
    "Alignment tolerance", options=["5min", "15min", "30min", "1h", "2h", "3h"], value="30min"
)

st.sidebar.markdown("**Outlier filtering**")
outlier_on = st.sidebar.checkbox("Enable outlier removal", value=False)
outlier_method = st.sidebar.selectbox("Method", ["iqr", "zscore", "hampel"], disabled=not outlier_on)
outlier_threshold = st.sidebar.slider("Threshold", 1.0, 6.0, 3.0, 0.5, disabled=not outlier_on)

if len(platforms_selected) < 2 or not params_selected:
    st.info("⬅️ Select at least 2 platforms and 1 parameter in the sidebar to begin.")
    st.stop()

if len(date_range) == 2:
    df = filter_date_range(df, start=date_range[0], end=date_range[1])


# --------------------------------------------------------------------------
# Main: tabs
# --------------------------------------------------------------------------
st.title("Wave Platform Comparison Dashboard")
st.caption(
    f"Comparing **{', '.join(platforms_selected)}** across "
    f"**{len(params_selected)}** parameter(s), {df['timestamp'].min():%Y-%m-%d} → {df['timestamp'].max():%Y-%m-%d}"
)

tab_overview, tab_detail, tab_distrib, tab_period, tab_export = st.tabs(
    ["📊 Multi-Parameter Overview", "🔍 Parameter Deep-Dive", "📈 Distributions", "🗓️ Period Agreement", "⬇️ Export"]
)

ref_platform = platforms_selected[0]
other_platforms = platforms_selected[1:]

# Pre-compute aligned data + stats for every parameter, ref vs each other platform
all_results = {}  # {(param, other): {"wide":df, "stats":obj}}
for param in params_selected:
    for other in other_platforms:
        wide = align_platforms(df, [ref_platform, other], param, tolerance=tolerance)
        if outlier_on:
            wide[ref_platform] = remove_outliers(wide[ref_platform], method=outlier_method, threshold=outlier_threshold)
            wide[other] = remove_outliers(wide[other], method=outlier_method, threshold=outlier_threshold)
        wide = drop_incomplete(wide, [ref_platform, other])
        if len(wide) >= 2:
            stats_obj = compute_comparison_stats(wide[ref_platform], wide[other])
            all_results[(param, other)] = {"wide": wide, "stats": stats_obj}

# ---------------- Overview tab ----------------
with tab_overview:
    st.subheader(f"Summary — all parameters, reference = {ref_platform}")
    for other in other_platforms:
        rows = []
        for param in params_selected:
            res = all_results.get((param, other))
            info = describe_param(param)
            if res is None:
                rows.append({"Parameter": f"{param} ({info.label})", "N": 0, "Bias": None, "RMSE": None,
                             "Scatter Index": None, "R²": None, "Pearson r": None})
                continue
            s = res["stats"]
            rows.append({
                "Parameter": f"{param} ({info.label})",
                "Unit": info.unit,
                "N": s.n,
                "Bias": round(s.bias, 4),
                "MAE": round(s.mae, 4),
                "RMSE": round(s.rmse, 4),
                "Scatter Index": round(s.scatter_index, 4),
                "R²": round(s.r_squared, 4),
                "Pearson r": round(s.pearson_r, 4),
            })
        st.markdown(f"**{ref_platform} vs {other}**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------- Deep dive tab ----------------
with tab_detail:
    c1, c2 = st.columns(2)
    with c1:
        param = st.selectbox("Parameter", params_selected, format_func=lambda c: f"{c} — {describe_param(c).label}")
    with c2:
        other = st.selectbox("Compare against", other_platforms, key="detail_other")

    res = all_results.get((param, other))
    info = describe_param(param)

    if res is None:
        st.warning("No overlapping data for this parameter/platform pair within the chosen tolerance.")
    else:
        wide = res["wide"]
        s = res["stats"]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("N pairs", s.n)
        m2.metric("Bias", f"{s.bias:.3f} {info.unit}")
        m3.metric("RMSE", f"{s.rmse:.3f} {info.unit}")
        m4.metric("Scatter Index", f"{s.scatter_index:.3f}")
        m5.metric("R²", f"{s.r_squared:.3f}")

        st.plotly_chart(plotting.time_series_overlay(wide, ref_platform, other, info.label, info.unit), use_container_width=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(plotting.scatter_with_fit(wide, ref_platform, other, s, info.label, info.unit), use_container_width=True)
        with cc2:
            st.plotly_chart(plotting.bland_altman(wide, ref_platform, other, s, info.label, info.unit), use_container_width=True)
        st.plotly_chart(plotting.difference_series(wide, ref_platform, other, info.label, info.unit), use_container_width=True)

        with st.expander("Full statistics"):
            st.json(s.to_dict())

# ---------------- Distributions tab ----------------
with tab_distrib:
    c1, c2, c3 = st.columns(3)
    with c1:
        param_d = st.selectbox("Parameter", params_selected, format_func=lambda c: f"{c} — {describe_param(c).label}", key="dist_param")
    with c2:
        other_d = st.selectbox("Compare against", other_platforms, key="dist_other")
    with c3:
        kind = st.selectbox("Plot type", ["histogram", "box", "violin"])

    res = all_results.get((param_d, other_d))
    info = describe_param(param_d)
    if res is None:
        st.warning("No overlapping data for this parameter/platform pair.")
    else:
        wide = res["wide"]
        st.plotly_chart(plotting.distribution_compare(wide, ref_platform, other_d, info.label, info.unit, kind=kind), use_container_width=True)
        dc1, dc2 = st.columns(2)
        dc1.markdown(f"**{ref_platform} descriptive stats**")
        dc1.dataframe(pd.DataFrame([descriptive_stats(wide[ref_platform])]), hide_index=True)
        dc2.markdown(f"**{other_d} descriptive stats**")
        dc2.dataframe(pd.DataFrame([descriptive_stats(wide[other_d])]), hide_index=True)

# ---------------- Period agreement tab ----------------
with tab_period:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        param_p = st.selectbox("Parameter", params_selected, format_func=lambda c: f"{c} — {describe_param(c).label}", key="period_param")
    with c2:
        other_p = st.selectbox("Compare against", other_platforms, key="period_other")
    with c3:
        freq = st.selectbox("Bin size", ["6h", "1D", "3D", "1W"], index=1)
    with c4:
        metric = st.selectbox("Metric", ["rmse", "bias", "mae", "scatter_index", "r_squared"])

    res = all_results.get((param_p, other_p))
    info = describe_param(param_p)
    if res is None:
        st.warning("No overlapping data for this parameter/platform pair.")
    else:
        binned = time_binned_stats(res["wide"], ref_platform, other_p, freq=freq)
        if binned.empty:
            st.info("Not enough points per bin to compute statistics — try a larger bin size.")
        else:
            st.plotly_chart(plotting.rolling_agreement(binned, metric=metric, label=info.label, unit=info.unit), use_container_width=True)
            st.dataframe(binned, use_container_width=True, hide_index=True)

# ---------------- Export tab ----------------
with tab_export:
    st.subheader("Export results")
    st.write("Export the multi-parameter summary table (for the currently selected reference + comparison platform) to Excel.")
    other_e = st.selectbox("Comparison platform to export", other_platforms, key="export_other")

    if st.button("Generate Excel report"):
        tables = {}
        summary_rows = []
        for param in params_selected:
            res = all_results.get((param, other_e))
            if res is None:
                continue
            s = res["stats"]
            info = describe_param(param)
            summary_rows.append({"parameter": param, "label": info.label, "unit": info.unit, **s.to_dict()})
            tables[f"{param}_pairs"] = res["wide"]
        tables["summary"] = pd.DataFrame(summary_rows)

        out_path = EXPORT_DIR / f"comparison_{ref_platform}_vs_{other_e}.xlsx"
        export_stats_excel(tables, str(out_path))
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download Excel report", f, file_name=out_path.name)
        st.success(f"Report generated: {out_path.name}")

st.sidebar.markdown("---")
st.sidebar.caption("Add more platforms or parameters any time — nothing here is hard-coded to a specific buoy pair or wave variable.")
