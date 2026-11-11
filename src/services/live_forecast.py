"""Module 7: live Open-Meteo Forecast API client.

Requests the same hourly variables, model and UTC timestamps as the archived
training data, so that `openmeteo_archive.aggregate_daily` can turn the live
response into daily features exactly as it did for training.

Lead-time note: archived `_previous_dayN` values are each predicted exactly
N x 24 h before their valid hour. A live "lead 1" day (tomorrow, IST) is taken
from the latest run, i.e. predicted roughly 18-48 h ahead. The two are close
but not identical; see doc section 6.2.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests

log = logging.getLogger(__name__)


class LiveForecastError(RuntimeError):
    """The live forecast could not be obtained or is unusable."""


_CACHE: dict[tuple, tuple[float, object]] = {}


def _cached(key: tuple, max_age_s: float, fn):
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < max_age_s:
        return hit[1]
    value = fn()
    _CACHE[key] = (time.time(), value)
    return value


def clear_cache():
    _CACHE.clear()


def _get_json(url: str, params: dict | None, session: requests.Session, retries: int = 3,
              timeout: float = 30.0):
    delay = 2.0
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
            else:
                data = r.json()
                if isinstance(data, dict) and data.get("error"):
                    raise LiveForecastError(f"Open-Meteo rejected the request: {data.get('reason')}")
                r.raise_for_status()
                return data
        except (requests.RequestException, ValueError) as exc:
            last = str(exc)
        log.warning("live request failed (%s), attempt %d/%d", last, attempt, retries)
        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    raise LiveForecastError(f"Open-Meteo forecast service unavailable ({last}). Please try again later.")


def fetch_live_hourly(locations: list[dict], model: str, variables: list[str], endpoint: str,
                      past_days: int = 1, forecast_days: int = 5, cache_minutes: float = 30,
                      session: requests.Session | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Latest-run hourly forecast for all locations (one request).

    Returns (hourly, grid) in the same long format as
    `openmeteo_archive.fetch_previous_runs`, with lead = 0.
    """
    session = session or requests.Session()
    params = {
        "latitude": ",".join(str(l["latitude"]) for l in locations),
        "longitude": ",".join(str(l["longitude"]) for l in locations),
        "hourly": ",".join(variables), "models": model, "timezone": "GMT",
        "past_days": past_days, "forecast_days": forecast_days,
    }
    key = ("hourly", endpoint, tuple(sorted(params.items())))
    payload = _cached(key, cache_minutes * 60, lambda: _get_json(endpoint, params, session))
    payloads = payload if isinstance(payload, list) else [payload]
    if len(payloads) != len(locations):
        raise LiveForecastError(f"expected {len(locations)} locations in the response, got {len(payloads)}")

    frames, grid = [], []
    for loc, p in zip(locations, payloads):
        h = p.get("hourly") or {}
        if "time" not in h:
            raise LiveForecastError(f"no hourly data for {loc['name']}")
        times = pd.to_datetime(h["time"])
        for v in variables:
            col = h.get(v, h.get(f"{v}_{model}"))
            if col is None:
                raise LiveForecastError(f"variable '{v}' missing from the live forecast for {loc['name']}")
            frames.append(pd.DataFrame({"time": times, "region_id": loc["region_id"], "model": model,
                                        "variable": v, "lead": 0,
                                        "value": pd.to_numeric(pd.Series(col), errors="coerce")}))
        grid.append({"region_id": loc["region_id"], "grid_lat": p.get("latitude"),
                     "grid_lon": p.get("longitude"), "elevation": p.get("elevation")})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(grid)


def fetch_run_metadata(model: str, endpoint_template: str, cache_minutes: float = 30,
                       session: requests.Session | None = None) -> dict:
    """Initialisation/availability time of the latest model run (UTC)."""
    session = session or requests.Session()
    url = endpoint_template.format(model=model)
    meta = _cached(("meta", url), cache_minutes * 60, lambda: _get_json(url, None, session))

    def ts(key):
        v = meta.get(key)
        return datetime.fromtimestamp(v, tz=timezone.utc) if v else None

    return {"run_initialisation_utc": ts("last_run_initialisation_time"),
            "run_availability_utc": ts("last_run_availability_time")}
