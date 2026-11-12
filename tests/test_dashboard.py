"""Offline smoke tests of the Streamlit dashboard (Modules 8 and 9).

The live Open-Meteo calls are replaced with synthetic hourly data dated
relative to today, so the tests need no network and work on any date.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.predict import SolarPredictor  # noqa: E402
from src.services.live_forecast import LiveForecastError  # noqa: E402

APP = str(ROOT / "src" / "dashboard" / "app.py")
GRID = {"puducherry": (12.0, 79.75), "karaikal": (11.0, 79.75), "mahe": (11.75, 75.5), "yanam": (16.75, 82.25)}


def synthetic(self, locations):
    start = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)
    times = pd.date_range(start, periods=24 * 6, freq="h")
    hour = times.hour.to_numpy()
    values = {"shortwave_radiation": np.clip(800 * np.sin(np.pi * (hour - 0.5) / 12), 0, None) * (hour <= 12),
              "cloud_cover": np.full(len(times), 40.0), "temperature_2m": 23.5 + 6 * np.sin(np.pi * hour / 24),
              "relative_humidity_2m": np.full(len(times), 75.0), "precipitation": np.zeros(len(times)),
              "wind_speed_10m": np.full(len(times), 10.0)}
    frames = [pd.DataFrame({"time": times, "region_id": l["region_id"], "model": "ecmwf_ifs025", "variable": v,
                            "lead": 0, "value": vals.astype(float)}) for l in locations for v, vals in values.items()]
    grid = pd.DataFrame([{"region_id": l["region_id"], "grid_lat": GRID[l["region_id"]][0],
                          "grid_lon": GRID[l["region_id"]][1], "elevation": 10} for l in locations])
    return pd.concat(frames, ignore_index=True), grid


def fresh_meta(self):
    return {"run_initialisation_utc": datetime.now(timezone.utc) - timedelta(hours=3)}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(SolarPredictor, "_default_fetch_hourly", synthetic)
    monkeypatch.setattr(SolarPredictor, "_default_fetch_meta", fresh_meta)
    monkeypatch.setattr(SolarPredictor, "log_predictions", staticmethod(lambda result, path: None))
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def run_app():
    return AppTest.from_file(APP, default_timeout=120).run()


def rooftop_kwh(at):
    """Daily kWh values shown in the rooftop estimator, e.g. '**11.1** kWh' -> '11.1'."""
    return [m.value.split("**")[1] if "**" in m.value else m.value
            for m in at.markdown if m.value.endswith(" kWh") or m.value == "—"]


def test_overview_renders_all_regions():
    at = run_app()
    assert not at.exception and not at.error
    assert [h.value for h in at.subheader] == ["Tomorrow at a glance"]
    assert [t.label for t in at.tabs] == ["Compare regions", "Historical patterns", "Forecast accuracy", "About the data"]


def test_region_and_taluk_selection_and_calculator():
    at = run_app()
    at.selectbox(key="region_select").select("karaikal").run()
    assert at.subheader[0].value == "Karaikal: next three days"
    energies = rooftop_kwh(at)
    assert len(energies) == 3 and all(v != "—" for v in energies)

    at.number_input(key="capacity_kw").set_value(6.0).run()
    doubled = [float(v) for v in rooftop_kwh(at)]
    assert doubled == pytest.approx([2 * float(v) for v in energies], abs=0.11)   # rounding to 0.1

    at.selectbox(key="taluk_select").select("karaikal:thirunallar").run()
    assert at.subheader[0].value.startswith("Thirunallar taluk")
    assert any("share one forecast" in i.value for i in at.info)
    assert not at.exception


def test_api_failure_shows_error_not_old_values(monkeypatch):
    def fail(self, locations):
        raise LiveForecastError("Open-Meteo forecast service unavailable (timeout)")
    monkeypatch.setattr(SolarPredictor, "_default_fetch_hourly", fail)
    at = run_app()
    assert not at.exception
    assert any("Live forecast unavailable" in e.value for e in at.error)
    assert not at.metric and "Tomorrow at a glance" not in [h.value for h in at.subheader]
