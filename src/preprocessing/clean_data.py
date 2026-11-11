"""Cleaning rules for Dataset A (NASA POWER) and Dataset B (archived forecasts).

The same `clean_forecasts` function must be applied to live forecasts at
inference time, so it only uses fixed physical rules (no fitted values).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"

FORECAST_FEATURES = [
    "forecast_shortwave_radiation", "forecast_temperature_mean", "forecast_temperature_max",
    "forecast_temperature_min", "forecast_humidity_mean", "forecast_cloud_cover_mean",
    "forecast_precipitation_sum", "forecast_wind_speed_mean",
]

# physical bounds: (lower, upper); values outside are clipped
FORECAST_BOUNDS = {
    "forecast_shortwave_radiation": (0.0, None),
    "forecast_humidity_mean": (0.0, 100.0),
    "forecast_cloud_cover_mean": (0.0, 100.0),
    "forecast_precipitation_sum": (0.0, None),   # NWP output can dip slightly < 0
    "forecast_wind_speed_mean": (0.0, None),
}
NASA_TARGET = "ALLSKY_SFC_SW_DWN"
NASA_TARGET_RANGE = (0.0, 10.0)  # kWh/m^2/day; TOA at these latitudes < ~11


def load_nasa(path: Path = PROCESSED / "nasa_solar_history.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_forecasts(path: Path = PROCESSED / "archived_forecasts.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["target_date", "forecast_issue_date"])


def clean_forecasts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clip to physical bounds, drop incomplete/duplicate rows.
    Returns (clean_df, report)."""
    report = {"rows_in": len(df)}
    df = df.copy()
    clipped = {}
    for col, (lo, hi) in FORECAST_BOUNDS.items():
        if col not in df:
            continue
        before = df[col].copy()
        df[col] = df[col].clip(lower=lo, upper=hi)
        clipped[col] = int((before.notna() & (before != df[col])).sum())
    report["values_clipped"] = clipped

    keys = ["region_id", "target_date", "lead_days"]
    dup = df.duplicated(keys, keep="first")
    report["duplicates_dropped"] = int(dup.sum())
    df = df[~dup]

    incomplete = df[FORECAST_FEATURES].isna().any(axis=1)
    if "hours_complete" in df:
        incomplete |= df["hours_complete"] < 24
    report["incomplete_rows_dropped"] = int(incomplete.sum())
    report["incomplete_dates"] = sorted(df.loc[incomplete, "target_date"].dt.strftime("%Y-%m-%d").unique())
    df = df[~incomplete]
    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def clean_nasa(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop missing/out-of-range targets and duplicate (region, date) rows."""
    report = {"rows_in": len(df)}
    dup = df.duplicated(["region_id", "date"], keep="first")
    df = df[~dup]
    lo, hi = NASA_TARGET_RANGE
    bad = df[NASA_TARGET].isna() | (df[NASA_TARGET] < lo) | (df[NASA_TARGET] > hi)
    report.update(duplicates_dropped=int(dup.sum()), invalid_target_dropped=int(bad.sum()))
    df = df[~bad]
    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report
