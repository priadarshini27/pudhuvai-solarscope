"""Client for the Open-Meteo Previous Runs API (archived forecasts).

`<variable>_previous_dayN` is the value that was predicted N x 24 hours before
the valid time (see https://open-meteo.com/en/docs/previous-runs-api).

All requests use timezone=GMT so that hourly timestamps (and any daily
aggregation done downstream) are in UTC.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = PROJECT_ROOT / "data" / "raw" / "openmeteo_archived_forecasts"
DEFAULT_LOCATIONS = PROJECT_ROOT / "config" / "locations.json"


class OpenMeteoError(RuntimeError):
    """The API returned an error payload (e.g. invalid model or variable)."""


def load_locations(path: Path = DEFAULT_LOCATIONS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["regions"]


def lead_variable(variable: str, lead: int) -> str:
    # lead 0 (latest run) is the plain variable; the API silently ignores
    # "_previous_day0".
    return variable if lead == 0 else f"{variable}_previous_day{lead}"


def _cache_path(cache_dir: Path, params: dict) -> Path:
    key = json.dumps(params, sort_keys=True)
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return cache_dir / f"{params['start_date']}_{params['end_date']}_{digest}.json"


def _get(params: dict, cache_dir: Path, session: requests.Session,
         retries: int = 5, pause: float = 1.0) -> list[dict]:
    """GET with on-disk JSON cache, retries and backoff. Returns a list of
    per-location payloads (the API returns a bare object for one location)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, params)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload["response"]

    delay = 5.0
    for attempt in range(1, retries + 1):
        try:
            r = session.get(BASE_URL, params=params, timeout=120)
        except requests.RequestException as exc:
            log.warning("request failed (%s), attempt %d/%d", exc, attempt, retries)
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code == 429 or r.status_code >= 500:
            log.warning("HTTP %d, backing off %.0fs (attempt %d/%d)",
                        r.status_code, delay, attempt, retries)
            time.sleep(delay)
            delay *= 2
            continue
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            raise OpenMeteoError(data.get("reason", "unknown error"))
        data = data if isinstance(data, list) else [data]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"request": {"url": BASE_URL, "params": params,
                                   "retrieved_utc": pd.Timestamp.utcnow().isoformat()},
                       "response": data}, f)
        time.sleep(pause)  # be gentle with the free API
        return data
    raise OpenMeteoError(f"giving up after {retries} attempts: {params}")


def fetch_previous_runs(locations: list[dict], models: Iterable[str],
                        variables: Iterable[str], leads: Iterable[int],
                        start_date: str, end_date: str,
                        cache_dir: Path = DEFAULT_CACHE,
                        session: requests.Session | None = None,
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch hourly archived forecasts for all locations in one request.

    Returns
    -------
    hourly : long DataFrame [time, region_id, model, variable, lead, value]
             with `time` as tz-naive UTC timestamps.
    grid   : DataFrame [region_id, requested_lat, requested_lon, grid_lat,
             grid_lon, elevation] describing the cell the API snapped to.
    """
    models, variables, leads = list(models), list(variables), list(leads)
    session = session or requests.Session()
    hourly_vars = [lead_variable(v, l) for v in variables for l in leads]
    params = {
        "latitude": ",".join(str(loc["latitude"]) for loc in locations),
        "longitude": ",".join(str(loc["longitude"]) for loc in locations),
        "hourly": ",".join(hourly_vars),
        "models": ",".join(models),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "GMT",
    }
    payloads = _get(params, cache_dir, session)
    if len(payloads) != len(locations):
        raise OpenMeteoError(f"expected {len(locations)} locations, got {len(payloads)}")

    frames, grid_rows = [], []
    for loc, p in zip(locations, payloads):
        grid_rows.append({
            "region_id": loc["region_id"],
            "requested_lat": loc["latitude"], "requested_lon": loc["longitude"],
            "grid_lat": p.get("latitude"), "grid_lon": p.get("longitude"),
            "elevation": p.get("elevation"),
        })
        h = p.get("hourly", {})
        times = pd.to_datetime(h.get("time", []))
        for model in models:
            for v in variables:
                for l in leads:
                    # with several models the API suffixes the model name;
                    # with a single model it may not.
                    name = lead_variable(v, l)
                    col = h.get(f"{name}_{model}", h.get(name) if len(models) == 1 else None)
                    if col is None:
                        col = [None] * len(times)
                    frames.append(pd.DataFrame({
                        "time": times, "region_id": loc["region_id"], "model": model,
                        "variable": v, "lead": l,
                        "value": pd.to_numeric(pd.Series(col), errors="coerce"),
                    }))
    hourly = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return hourly, pd.DataFrame(grid_rows)


def probe_models(candidates: Iterable[str], location: dict, date: str = "2025-06-01",
                 cache_dir: Path = DEFAULT_CACHE,
                 session: requests.Session | None = None) -> tuple[list[str], dict]:
    """Return (valid_models, {rejected_model: reason}) using a 1-day request each."""
    valid, rejected = [], {}
    session = session or requests.Session()
    for m in candidates:
        try:
            fetch_previous_runs([location], [m], ["shortwave_radiation"], [1],
                                date, date, cache_dir=cache_dir, session=session)
            valid.append(m)
        except OpenMeteoError as exc:
            rejected[m] = str(exc)
    return valid, rejected


def daily_completeness(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per UTC date: number of non-null hourly values (complete == 24)."""
    df = hourly.assign(date=hourly["time"].dt.floor("D"))
    out = (df.groupby(["region_id", "model", "variable", "lead", "date"])["value"]
             .agg(n_valid="count", n_hours="size").reset_index())
    out["complete"] = (out["n_valid"] == 24) & (out["n_hours"] == 24)
    return out


def fetch_range(locations: list[dict], model: str, variables: list[str], leads: list[int],
                start: date, end: date, chunk_days: int = 92,
                cache_dir: Path = DEFAULT_CACHE,
                session: requests.Session | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """fetch_previous_runs over a long range in fixed-size chunks (cache-friendly)."""
    session = session or requests.Session()
    frames, grid = [], None
    s = start
    while s <= end:
        e = min(s + timedelta(days=chunk_days - 1), end)
        hourly, grid = fetch_previous_runs(locations, [model], variables, leads,
                                           s.isoformat(), e.isoformat(),
                                           cache_dir=cache_dir, session=session)
        frames.append(hourly)
        log.info("fetched %s %s..%s", model, s, e)
        s = e + timedelta(days=1)
    hourly = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["time", "region_id", "model", "variable", "lead"])
    return hourly, grid


# variable -> {output column: aggregation}
DAILY_AGGREGATIONS = {
    "shortwave_radiation": {"forecast_shortwave_radiation": "sum_kwh"},
    "temperature_2m": {"forecast_temperature_mean": "mean",
                       "forecast_temperature_max": "max",
                       "forecast_temperature_min": "min"},
    "relative_humidity_2m": {"forecast_humidity_mean": "mean"},
    "cloud_cover": {"forecast_cloud_cover_mean": "mean"},
    "precipitation": {"forecast_precipitation_sum": "sum"},
    "wind_speed_10m": {"forecast_wind_speed_mean": "mean"},
}


def aggregate_daily(hourly: pd.DataFrame, utc_offset: str = "+05:30") -> pd.DataFrame:
    """Hourly archived forecasts (UTC) -> one row per region/model/lead/local day.

    * Hours are shifted by `utc_offset` and grouped into local calendar days.
      Open-Meteo hourly radiation is the mean over the preceding hour, so the
      sum of 24 hourly W/m^2 values / 1000 = kWh/m^2/day.
    * `hours_complete` is the minimum number of non-null hours across all
      variables; aggregates are NaN unless all 24 hours are present.
    * Units: temperature degC, humidity %, cloud cover %, precipitation mm,
      wind speed km/h (Open-Meteo defaults).
    """
    sign = -1 if utc_offset.startswith("-") else 1
    hh, mm = utc_offset.lstrip("+-").split(":")
    offset = sign * pd.Timedelta(hours=int(hh), minutes=int(mm))
    df = hourly.assign(target_date=(hourly["time"] + offset).dt.floor("D"))
    keys = ["region_id", "model", "lead", "target_date"]

    g = df.groupby(keys + ["variable"])["value"]
    stats = g.agg(["mean", "max", "min", "sum", "count", "size"]).reset_index()
    out = stats[keys].drop_duplicates().set_index(keys)
    counts = stats.pivot_table(index=keys, columns="variable", values="count")
    sizes = stats.pivot_table(index=keys, columns="variable", values="size")
    complete = (counts == 24) & (sizes == 24)

    for var, cols in DAILY_AGGREGATIONS.items():
        if var not in counts.columns:
            continue
        sub = stats[stats["variable"] == var].set_index(keys)
        ok = complete[var].reindex(sub.index).fillna(False)
        for col, how in cols.items():
            vals = sub["sum"] / 1000.0 if how == "sum_kwh" else sub[how]
            out[col] = vals.where(ok)
    out["hours_complete"] = counts.min(axis=1).astype(int)
    out = out.reset_index().rename(columns={"lead": "lead_days", "model": "forecast_model"})
    return out.sort_values(["region_id", "target_date", "lead_days"]).reset_index(drop=True)
