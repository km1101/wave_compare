# Wave Platform Comparison Dashboard

Interactive tool for comparing wave measurements between two or more
platforms/buoys (e.g. TH2 vs Mobilis), built on the actual data format
exported from the Motus wave Sensor buoy CSV (`MotusID`,
`Year/Month/Day/Hour/Minute`, `H13`, `Tz`, `Hmax`, `Tmax`, `Hswell`,
`Tswell`, `Hm0`, `Tp`, `Tm02`, directional parameters, etc).




## Quick start

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).


## Project layout
```
wave_compare/
├── dashboard.py              # Streamlit app (entry point)
├── requirements.txt
├── config/
│   └── platform_mapping.json # editable id -> platform name map
├── data/
│   └── wave.csv              # your uploaded data (replace/upload your own)
├── exports/                  # generated reports land here
└── wave_compare/             # importable Python package (reusable outside the dashboard)
    ├── config.py             # platform mapping + parameter catalog
    ├── data_loader.py        # CSV -> tidy long-format DataFrame
    ├── preprocessing.py      # alignment, outlier removal, resampling
    ├── statistics.py         # bias/RMSE/SI/R²/Bland-Altman/etc
    ├── plotting.py           # Plotly figure builders
    └── export.py             # CSV/Excel/PNG export helpers
```
By default it loads `data/wave.csv` (the file you supplied). Use the
sidebar to upload a different CSV at any time — no code changes needed.

## Platform mapping

Platform IDs (`MotusID`/`AWSID`) are mapped to human-readable names via
`config/platform_mapping.json`:

```json
{
  "133": "TH2",
  "136": "Mobilis"
}
```

You can edit this file directly, edit the mapping in the sidebar
("Edit manually" → Save), or upload your own mapping file (CSV with
`id,name` columns, or JSON) — the dashboard isn't hard-coded to these two
platforms or IDs, so new data with unique ID can be uploaded.

## What "significant wave" means here

**every numeric column in the CSV is selectable** — including `Hmax`,
`Tmax`, `Hswell`, `Tswell`, `Hwind`, `Twind`, `Hm0`, `Tp`, `Tm02`,
`Hcrest`, `Htrough`, and the directional spread/peak/mean direction
columns. Add or remove parameters from the catalog in
`wave_compare/config.py` (`PARAMETER_CATALOG`) to change labels/units, or
just pick any other column straight from the sidebar — unrecognised
columns still work, they just get a generic label.

## Dashboard layout

- **Multi-Parameter Overview** — one summary table per comparison platform
  (bias, MAE, RMSE, scatter index, R², Pearson r) across every selected
  parameter at once.
- **Parameter Deep-Dive** — for one parameter/platform pair: time series
  overlay, scatter + 1:1 line + regression fit, Bland–Altman limits of
  agreement, and a difference-over-time plot, plus the full stats as JSON.
- **Distributions** — histogram / box / violin comparison plus descriptive
  stats (mean, std, skew, kurtosis, quartiles) for each platform.
- **Period Agreement** — how RMSE/bias/MAE/scatter index/R² evolve when
  binned into 6-hourly/daily/3-day/weekly windows, useful for spotting
  when two platforms disagree (e.g. during storms, sensor outages).
- **Export** — generates a multi-sheet Excel workbook (one sheet per
  parameter's paired data + a summary sheet) for the selected comparison.

Sidebar controls also let you: filter by date range, set the timestamp
alignment tolerance for matching readings between platforms (nearest
match within N minutes/hours), and optionally remove outliers (IQR,
z-score, or Hampel filter — good for spiky buoy spikes).





## Extending

- **More platforms**: add another `id: name` pair to the mapping; the
  multiselect and pairwise comparisons (reference = first platform
  selected) pick it up automatically.
- **More parameters**: any numeric column works out of the box; add an
  entry to `PARAMETER_CATALOG` in `wave_compare/config.py` to give it a
  friendly label/unit/category.
- **A model/hindcast as a third "platform"**: adapt its output into the
  same long-format shape (`timestamp`, `platform`, parameter columns) and
  concatenate it onto the loaded DataFrame before passing it to the rest
  of the pipeline — nothing else needs to change.


The `wave_compare` package has no Streamlit dependency itself — every
function returns plain DataFrames/dataclasses/Plotly figures, so you can
reuse the same statistics/plotting code on a different UI later.

