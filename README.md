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

The `wave_compare` package has no Streamlit dependency itself — every
function returns plain DataFrames/dataclasses/Plotly figures, so you can
reuse the same statistics/plotting code on a different UI later.

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
