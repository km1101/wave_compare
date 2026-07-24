"""
config.py
---------
Central configuration for the wave-platform comparison tool.

Two things live here:

1. Platform mapping: which raw ID column value (e.g. MotusID) corresponds to
   which human-readable platform name (e.g. "TH2", "Mobilis"). This is
   intentionally NOT hard-coded as a frozen dict — it can be supplied as:
     - a Python dict literal (DEFAULT_PLATFORM_MAPPING below)
     - a JSON file (config/platform_mapping.json)
     - a two-column CSV file (id,name)
   so new platforms/buoys can be added without touching code.

2. Parameter catalog: every numeric wave/met column we know how to compare,
   with a friendly label, unit, and category. The catalog is intentionally
   broad (not just Hsig/Tsig) - any numeric column in the source file can
   also be added on the fly, the catalog just gives nice labels/units to
   the ones we recognise.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# --------------------------------------------------------------------------
# Platform (buoy/station) ID -> name mapping
# --------------------------------------------------------------------------

# Fallback mapping, used only if no config file is found and nothing is
# passed explicitly to load_platform_mapping().
DEFAULT_PLATFORM_MAPPING: Dict[str, str] = {
    "133": "TH2",
    "136": "Mobilis",
}

PLATFORM_MAPPING_JSON = Path(__file__).resolve().parent.parent / "config" / "platform_mapping.json"


def load_platform_mapping(source: Optional[str] = None) -> Dict[str, str]:
    """
    Load a {raw_id (str) -> platform_name (str)} mapping.

    source:
        None            -> try config/platform_mapping.json, else fall back
                            to DEFAULT_PLATFORM_MAPPING.
        path to *.json  -> {"133": "TH2", "136": "Mobilis"}
        path to *.csv   -> two columns, header "id,name" (or no header,
                            first two columns used)
        dict            -> used directly (keys coerced to str)

    Returns a plain dict with string keys so it matches a stringified
    MotusID/AWSID column.
    """
    if source is None:
        if PLATFORM_MAPPING_JSON.exists():
            return load_platform_mapping(str(PLATFORM_MAPPING_JSON))
        return dict(DEFAULT_PLATFORM_MAPPING)

    if isinstance(source, dict):
        return {str(k): str(v) for k, v in source.items()}

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Platform mapping file not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            raw = json.load(f)
        return {str(k): str(v) for k, v in raw.items()}

    if path.suffix.lower() == ".csv":
        mapping: Dict[str, str] = {}
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return mapping
        start = 0
        # skip header row if it looks like one
        if rows[0][0].strip().lower() in ("id", "platform_id", "motusid", "awsid"):
            start = 1
        for row in rows[start:]:
            if len(row) < 2:
                continue
            mapping[str(row[0]).strip()] = str(row[1]).strip()
        return mapping

    raise ValueError(f"Unsupported platform mapping file type: {path.suffix}")


def save_platform_mapping(mapping: Dict[str, str], path: Optional[str] = None) -> Path:
    """Persist a mapping dict back to config/platform_mapping.json (or a custom path)."""
    out = Path(path) if path else PLATFORM_MAPPING_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(mapping, f, indent=2)
    return out


# --------------------------------------------------------------------------
# Parameter catalog
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamInfo:
    column: str          # raw column name in the source CSV
    label: str           # human-readable label
    unit: str            # display unit
    category: str        # grouping for the UI (height / period / direction / other)
    canonical: str = ""  # canonical role, e.g. "Hsig", "Tsig" (optional, for defaults)


# NOTE: "Hsig"/"Tsig" as literally-named columns don't exist in this data
# source. Per spec, H13 (significant wave height, average of highest 1/3)
# and Tz (mean zero-crossing period) are used as the significant-wave
# stand-ins, but the catalog is deliberately broad so any numeric column
# can be compared, not just these two.
PARAMETER_CATALOG: Dict[str, ParamInfo] = {
    "H13":        ParamInfo("H13", "Significant Wave Height (H1/3)", "m", "height", canonical="Hsig"),
    "Tz":         ParamInfo("Tz", "Mean Zero-Crossing Period (Tz)", "s", "period", canonical="Tsig"),
    "Hm0":        ParamInfo("Hm0", "Spectral Significant Wave Height (Hm0)", "m", "height"),
    "Hmax":       ParamInfo("Hmax", "Maximum Wave Height", "m", "height"),
    "Tmax":       ParamInfo("Tmax", "Period of Maximum Wave", "s", "period"),
    "Tp":         ParamInfo("Tp", "Peak Period", "s", "period"),
    "Tm02":       ParamInfo("Tm02", "Mean Period (Tm02)", "s", "period"),
    "Hcrest":     ParamInfo("Hcrest", "Maximum Crest Height", "m", "height"),
    "Htrough":    ParamInfo("Htrough", "Maximum Trough Depth", "m", "height"),
    "Hswell":     ParamInfo("Hswell", "Swell Wave Height", "m", "height"),
    "Tswell":     ParamInfo("Tswell", "Swell Period", "s", "period"),
    "Hwind":      ParamInfo("Hwind", "Wind-Sea Wave Height", "m", "height"),
    "Twind":      ParamInfo("Twind", "Wind-Sea Period", "s", "period"),
    "WPDir":      ParamInfo("WPDir", "Peak Wave Direction", "deg", "direction"),
    "WMDir":      ParamInfo("WMDir", "Mean Wave Direction", "deg", "direction"),
    "WPDirSwell": ParamInfo("WPDirSwell", "Swell Peak Direction", "deg", "direction"),
    "WPDirWind":  ParamInfo("WPDirWind", "Wind-Sea Peak Direction", "deg", "direction"),
    "WMSpr":      ParamInfo("WMSpr", "Mean Directional Spread", "deg", "other"),
}

# Default pair offered first in the UI / reports (matches the spec's
# "significant wave" request: H13 + Tz), can be overridden by the user.
DEFAULT_PARAMETERS = ["H13","Hm0", "Hmax","Hwind","Tp","Tz","Tm02","Tmax","WPDir","Hswell","Htrough","Hcrest", "Tswell","WMSpr"]

# Columns that identify the platform + build the timestamp - never offered
# as comparison "parameters" themselves.
ID_COLUMNS = ["MotusID", "AWSID"]
TIME_COLUMNS = ["Year", "Month", "Day", "Hour", "Minute"]


def describe_param(column: str) -> ParamInfo:
    """Return a ParamInfo for any column, inventing a generic label if unknown."""
    if column in PARAMETER_CATALOG:
        return PARAMETER_CATALOG[column]
    return ParamInfo(column, column, "", "other")
