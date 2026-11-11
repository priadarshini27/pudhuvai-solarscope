"""Module 4: build Dataset C (solar_model_training.csv).

Feature construction is split into functions that need no target, so that
`build_features` can be reused unchanged by the live prediction engine.
Fitted transforms (one-hot encoding, scaling) belong in the model pipeline
(Module 6) and are fitted on the training split only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clean_data import FORECAST_FEATURES, NASA_TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "config" / "pipeline.json"

SEASONAL_FEATURES = ["doy_sin", "doy_cos"]
CLIMATOLOGY_FEATURES = ["climatology_solar_radiation", "clear_sky_radiation"]
FEATURES = FORECAST_FEATURES + ["lead_days"] + SEASONAL_FEATURES + CLIMATOLOGY_FEATURES + ["region_id"]
TARGET = "actual_solar_radiation"
RESIDUAL_TARGET = "target_residual"   # actual - forecast shortwave
SPLITS = ("train", "validation", "test")


def load_config(path: Path = CONFIG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _doy365(dates: pd.Series) -> pd.Series:
    """Day of year 1..365 (31 Dec of leap years folded onto 365)."""
    return dates.dt.day_of_year.clip(upper=365)


def seasonal_features(dates: pd.Series) -> pd.DataFrame:
    angle = 2 * np.pi * dates.dt.day_of_year / 365.25
    return pd.DataFrame({"doy_sin": np.sin(angle), "doy_cos": np.cos(angle)}, index=dates.index)


def build_climatology(nasa: pd.DataFrame, start_year: int, end_year: int,
                      window: int = 15) -> pd.DataFrame:
    """Region x day-of-year mean of NASA all-sky and clear-sky radiation from
    years [start_year, end_year], smoothed with a circular +/- `window` day
    moving average. Returns columns [region_id, doy, climatology_solar_radiation,
    clear_sky_radiation, n_years]."""
    d = nasa[(nasa["date"].dt.year >= start_year) & (nasa["date"].dt.year <= end_year)].copy()
    d["doy"] = _doy365(d["date"])
    out = []
    for region, g in d.groupby("region_id"):
        daily = g.groupby("doy").agg(allsky=(NASA_TARGET, "mean"),
                                     clrsky=("CLRSKY_SFC_SW_DWN", "mean"),
                                     n_years=(NASA_TARGET, "count")).reindex(range(1, 366))
        k = 2 * window + 1
        # circular smoothing: pad both ends with the opposite end of the year
        padded = pd.concat([daily.iloc[-window:], daily, daily.iloc[:window]])
        smooth = padded[["allsky", "clrsky"]].rolling(k, center=True, min_periods=1).mean()
        smooth = smooth.iloc[window:-window]
        out.append(pd.DataFrame({
            "region_id": region, "doy": daily.index,
            "climatology_solar_radiation": smooth["allsky"].to_numpy(),
            "clear_sky_radiation": smooth["clrsky"].to_numpy(),
            "n_years": daily["n_years"].to_numpy(),
        }))
    return pd.concat(out, ignore_index=True)


def add_climatology(df: pd.DataFrame, clim: pd.DataFrame, date_col: str = "target_date") -> pd.DataFrame:
    df = df.assign(doy=_doy365(df[date_col]))
    out = df.merge(clim[["region_id", "doy", *CLIMATOLOGY_FEATURES]], on=["region_id", "doy"],
                   how="left")
    return out.drop(columns="doy")


def build_features(forecasts: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """Cleaned daily forecasts -> model feature table (no target needed).
    Used for both training data and live inference."""
    df = forecasts.copy()
    df[SEASONAL_FEATURES] = seasonal_features(df["target_date"])
    df = add_climatology(df, clim)
    if df[CLIMATOLOGY_FEATURES].isna().any().any():
        raise ValueError("climatology lookup failed for some rows (unknown region?)")
    return df


def assign_split(dates: pd.Series, splits: dict) -> pd.Series:
    """Map each target date to train/validation/test (NaN if outside all)."""
    out = pd.Series(pd.NA, index=dates.index, dtype="object")
    for name in SPLITS:
        lo, hi = (pd.Timestamp(x) for x in splits[name])
        out[(dates >= lo) & (dates <= hi)] = name
    return out


def build_training_table(forecasts: pd.DataFrame, nasa: pd.DataFrame, clim: pd.DataFrame,
                         splits: dict) -> pd.DataFrame:
    """Join cleaned forecasts to cleaned NASA targets and add features/split."""
    feats = build_features(forecasts, clim)
    target = nasa[["region_id", "date", NASA_TARGET]].rename(
        columns={"date": "target_date", NASA_TARGET: TARGET})
    df = feats.merge(target, on=["region_id", "target_date"], how="inner")
    df[RESIDUAL_TARGET] = df[TARGET] - df["forecast_shortwave_radiation"]
    df["split"] = assign_split(df["target_date"], splits)
    df = df[df["split"].notna()]
    meta = ["target_date", "forecast_issue_date", "region_id", "lead_days", "forecast_model",
            "latitude", "longitude", "grid_latitude", "grid_longitude", "split"]
    cols = meta + [c for c in FEATURES if c not in meta] + [TARGET, RESIDUAL_TARGET]
    return df[cols].sort_values(["target_date", "region_id", "lead_days"]).reset_index(drop=True)
