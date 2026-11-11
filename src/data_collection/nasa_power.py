"""Client for the NASA POWER Daily Point API.

Note: the daily product is available with two time standards. In Nov 2026 the
UTC series lagged ~10 weeks behind real time, while the LST (local solar
time) series lagged ~1 week. Which one the project uses is decided from the
archive audit (results/archive_audit/AUDIT_REPORT.md).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
FILL_VALUE = -999.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = PROJECT_ROOT / "data" / "raw" / "nasa_solar_history"


def fetch_daily(lat: float, lon: float, start: str, end: str,
                params: list[str] | None = None, time_standard: str = "LST",
                community: str = "RE", cache_dir: Path = DEFAULT_CACHE,
                use_cache: bool = True, session: requests.Session | None = None,
                retries: int = 4) -> pd.DataFrame:
    """Daily values for one point. `start`/`end` as YYYY-MM-DD.

    Returns a DataFrame indexed by date with one column per parameter;
    fill values (-999) are converted to NaN.
    """
    params = params or ["ALLSKY_SFC_SW_DWN"]
    session = session or requests.Session()
    s, e = start.replace("-", ""), end.replace("-", "")
    query = {"parameters": ",".join(params), "community": community,
             "latitude": lat, "longitude": lon, "start": s, "end": e,
             "format": "JSON", "time-standard": time_standard}

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{lat}_{lon}_{s}_{e}_{time_standard}_{'-'.join(params)}.json"
    if use_cache and path.exists():
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)["response"]
    else:
        delay = 5.0
        for attempt in range(1, retries + 1):
            try:
                r = session.get(BASE_URL, params=query, timeout=120)
                r.raise_for_status()
                payload = r.json()
                break
            except (requests.RequestException, ValueError) as exc:
                log.warning("NASA POWER request failed (%s), attempt %d/%d",
                            exc, attempt, retries)
                time.sleep(delay)
                delay *= 2
        else:
            raise RuntimeError(f"NASA POWER request failed: {query}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"request": {"url": BASE_URL, "params": query,
                                   "retrieved_utc": pd.Timestamp.utcnow().isoformat()},
                       "response": payload}, f)

    data = payload["properties"]["parameter"]
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    return df.replace(FILL_VALUE, float("nan")).astype(float)


def last_valid_date(df: pd.DataFrame, column: str = "ALLSKY_SFC_SW_DWN"):
    s = df[column].dropna()
    return s.index.max() if not s.empty else None


def build_history(locations: list[dict], start: str, end: str, params: list[str],
                  time_standard: str = "LST", community: str = "RE",
                  session: requests.Session | None = None) -> pd.DataFrame:
    """Dataset A: one row per region and date, fill values as NaN."""
    session = session or requests.Session()
    frames = []
    for loc in locations:
        df = fetch_daily(loc["latitude"], loc["longitude"], start, end, params=params,
                         time_standard=time_standard, community=community,
                         session=session)
        df = df.reset_index().assign(region_id=loc["region_id"],
                                     latitude=loc["latitude"], longitude=loc["longitude"])
        frames.append(df)
        log.info("NASA POWER %s: %d days", loc["region_id"], len(df))
    out = pd.concat(frames, ignore_index=True)
    return out[["date", "region_id", "latitude", "longitude", *params]]
