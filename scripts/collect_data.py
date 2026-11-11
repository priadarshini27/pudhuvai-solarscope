"""Modules 2 & 3: build Dataset A (nasa_solar_history.csv) and Dataset B
(archived_forecasts.csv) from config/pipeline.json and config/locations.json.

Usage:  .venv/Scripts/python scripts/collect_data.py
Raw API responses are cached under data/raw/; processed CSVs, retrieval
metadata and a data-quality summary go to data/processed/.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_collection import nasa_power  # noqa: E402
from src.data_collection.openmeteo_archive import (  # noqa: E402
    DEFAULT_CACHE, aggregate_daily, fetch_range, load_locations)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("collect")

PROCESSED = ROOT / "data" / "processed"
FORECAST_COLUMNS = ["forecast_issue_date", "target_date", "lead_days", "region_id",
                    "latitude", "longitude", "grid_latitude", "grid_longitude",
                    "forecast_model", "forecast_shortwave_radiation",
                    "forecast_temperature_mean", "forecast_temperature_max",
                    "forecast_temperature_min", "forecast_humidity_mean",
                    "forecast_cloud_cover_mean", "forecast_precipitation_sum",
                    "forecast_wind_speed_mean", "hours_complete"]


def main():
    cfg = json.loads((ROOT / "config" / "pipeline.json").read_text(encoding="utf-8"))
    locations = load_locations()
    session = requests.Session()
    session.headers["User-Agent"] = "pudhuvai-solarscope-data-collection"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    yesterday = date.today() - timedelta(days=1)

    # ---- Dataset A: NASA POWER
    n = cfg["nasa"]
    nasa = nasa_power.build_history(locations, n["history_start"], yesterday.isoformat(),
                                    n["parameters"], time_standard=n["time_standard"],
                                    community=n["community"], session=session)
    # drop the trailing days not yet published (all-NaN target at the end)
    last_valid = nasa.dropna(subset=["ALLSKY_SFC_SW_DWN"])["date"].max()
    nasa = nasa[nasa["date"] <= last_valid]
    nasa.to_csv(PROCESSED / "nasa_solar_history.csv", index=False, date_format="%Y-%m-%d")

    # ---- Dataset B: archived forecasts
    model = cfg["forecast_model"]
    hourly, grid = fetch_range(locations, model, cfg["forecast_variables"], cfg["lead_days"],
                               date.fromisoformat(cfg["archive_start"]), yesterday,
                               cache_dir=DEFAULT_CACHE / model, session=session)
    daily = aggregate_daily(hourly, cfg["local_day"]["utc_offset"])
    g = grid.set_index("region_id")
    daily["latitude"] = daily["region_id"].map(g["requested_lat"])
    daily["longitude"] = daily["region_id"].map(g["requested_lon"])
    daily["grid_latitude"] = daily["region_id"].map(g["grid_lat"])
    daily["grid_longitude"] = daily["region_id"].map(g["grid_lon"])
    # previous_dayN is defined per hour (predicted N x 24 h before each hour),
    # so the nominal issue date of the daily aggregate is target_date - N.
    daily["forecast_issue_date"] = daily["target_date"] - pd.to_timedelta(daily["lead_days"], unit="D")
    # drop the partial first/last local days produced by the UTC->IST shift
    first = pd.Timestamp(cfg["archive_start"]) + pd.Timedelta(days=1)
    daily = daily[(daily["target_date"] >= first) & (daily["target_date"] <= pd.Timestamp(yesterday))]
    daily = daily[FORECAST_COLUMNS]
    daily.to_csv(PROCESSED / "archived_forecasts.csv", index=False, date_format="%Y-%m-%d")

    # ---- metadata + quality summary
    complete = daily["hours_complete"] == 24
    meta = {
        "retrieved_utc": pd.Timestamp.utcnow().isoformat(),
        "nasa_power": {"endpoint": nasa_power.BASE_URL, "time_standard": n["time_standard"],
                       "community": n["community"], "parameters": n["parameters"],
                       "date_range": [str(nasa["date"].min().date()), str(last_valid.date())],
                       "rows": len(nasa),
                       "missing_target_rows": int(nasa["ALLSKY_SFC_SW_DWN"].isna().sum())},
        "open_meteo": {"endpoint": "https://previous-runs-api.open-meteo.com/v1/forecast",
                       "model": model, "variables": cfg["forecast_variables"],
                       "lead_days": cfg["lead_days"],
                       "lead_definition": "hourly value predicted N x 24 h before valid time "
                                          "(_previous_dayN); daily aggregate over IST day",
                       "local_utc_offset": cfg["local_day"]["utc_offset"],
                       "date_range": [str(daily["target_date"].min().date()),
                                      str(daily["target_date"].max().date())],
                       "rows": len(daily), "complete_rows": int(complete.sum()),
                       "incomplete_target_dates": sorted({str(d.date()) for d in
                                                          daily.loc[~complete, "target_date"]}),
                       "grid_cells": grid.to_dict(orient="records")},
    }
    (PROCESSED / "collection_metadata.json").write_text(json.dumps(meta, indent=2),
                                                        encoding="utf-8")
    log.info("NASA rows %d (%s..%s), missing target %d", len(nasa), meta["nasa_power"]["date_range"][0],
             meta["nasa_power"]["date_range"][1], meta["nasa_power"]["missing_target_rows"])
    log.info("forecast rows %d, complete %d, incomplete dates %s", len(daily), complete.sum(),
             meta["open_meteo"]["incomplete_target_dates"])


if __name__ == "__main__":
    main()
