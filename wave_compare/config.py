"""
config.py — platform mapping + parameter catalog.
"""
from __future__ import annotations
import csv, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

DEFAULT_PLATFORM_MAPPING: Dict[str, str] = {"133": "TH2", "136": "Mobilis"}
PLATFORM_MAPPING_JSON = Path(__file__).resolve().parent.parent / "config" / "platform_mapping.json"


def load_platform_mapping(source: Optional[str] = None) -> Dict[str, str]:
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
        with open(path) as f:
            raw = json.load(f)
        return {str(k): str(v) for k, v in raw.items()}
    if path.suffix.lower() == ".csv":
        mapping: Dict[str, str] = {}
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return mapping
        start = 0
        if rows[0][0].strip().lower() in ("id", "platform_id", "motusid", "awsid"):
            start = 1
        for row in rows[start:]:
            if len(row) >= 2:
                mapping[str(row[0]).strip()] = str(row[1]).strip()
        return mapping
    raise ValueError(f"Unsupported mapping file type: {path.suffix}")


def save_platform_mapping(mapping: Dict[str, str], path: Optional[str] = None) -> Path:
    out = Path(path) if path else PLATFORM_MAPPING_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(mapping, f, indent=2)
    return out


@dataclass(frozen=True)
class ParamInfo:
    column: str
    label: str
    unit: str
    category: str
    canonical: str = ""


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

DEFAULT_PARAMETERS = ["H13", "Tz", "Hmax", "Tmax", "Hswell", "Tswell","Hwind", "Twind","WPDir","WMDir","WPDirSwell","WPDirWind","WMSpr"]
ID_COLUMNS = ["MotusID", "AWSID"]
TIME_COLUMNS = ["Year", "Month", "Day", "Hour", "Minute"]


def describe_param(column: str) -> ParamInfo:
    if column in PARAMETER_CATALOG:
        return PARAMETER_CATALOG[column]
    return ParamInfo(column, column, "", "other")
