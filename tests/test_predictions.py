import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.evaluate import regression_metrics  # noqa: E402
from src.models.train import PerRegionRegressor, ResidualRegressor  # noqa: E402


def frame():
    return pd.DataFrame({"region_id": ["a", "a", "b", "b"],
                         "forecast_shortwave_radiation": [5.0, 6.0, 4.0, 3.0],
                         "x": [1.0, 2.0, 3.0, 4.0]})


def test_residual_regressor_adds_forecast_back():
    X = frame()
    y = X["forecast_shortwave_radiation"] - 0.5            # constant over-forecast
    m = ResidualRegressor(DummyRegressor(strategy="mean")).fit(X, y)
    assert np.allclose(m.predict(X), y)


def test_per_region_regressor_routes_rows():
    X = frame()
    y = np.where(X["region_id"] == "a", 10.0, 20.0)
    est = PerRegionRegressor(DummyRegressor(strategy="mean")).fit(X, y)
    assert np.allclose(est.predict(X.iloc[::-1]), y[::-1])
    with pytest.raises(ValueError):
        est.predict(X.assign(region_id="c"))


def test_per_region_models_are_independent():
    X = frame()[["region_id", "x"]]
    y = np.where(X["region_id"] == "a", 2 * X["x"], -X["x"])

    class XOnly(LinearRegression):
        def fit(self, X, y):
            return super().fit(X[["x"]], y)

        def predict(self, X):
            return super().predict(X[["x"]])

    est = PerRegionRegressor(XOnly()).fit(X, y)
    assert np.allclose(est.predict(X), y)


def test_regression_metrics():
    m = regression_metrics([1, 2, 3], [2, 2, 4])
    assert m["mae"] == pytest.approx(2 / 3) and m["bias"] == pytest.approx(2 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(2 / 3))


# ---------------------------------------------------------------- live engine (offline)
from datetime import datetime, timezone  # noqa: E402

from src.data_collection.openmeteo_archive import aggregate_daily  # noqa: E402
from src.models.predict import SolarPredictor, categorise  # noqa: E402
from src.preprocessing.build_training_data import build_features  # noqa: E402
from src.preprocessing.clean_data import clean_forecasts  # noqa: E402
from src.services.live_forecast import LiveForecastError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 11, 6, 0, tzinfo=timezone.utc)          # 11:30 IST
GRID = {"puducherry": (12.0, 79.75), "karaikal": (11.0, 79.75), "mahe": (11.75, 75.5), "yanam": (16.75, 82.25)}


def synthetic_hourly(locations, drop_hours=0, region_to_break=None):
    times = pd.date_range("2026-11-10T00:00", "2026-11-15T23:00", freq="h")
    # daylight roughly 00:30-12:30 UTC in India; peak ~800 W/m2
    hour = times.hour.to_numpy()
    sw = np.clip(800 * np.sin(np.pi * (hour - 0.5) / 12), 0, None) * (hour <= 12)
    base = {"shortwave_radiation": sw, "cloud_cover": np.full(len(times), 40.0),
            "temperature_2m": 23.5 + 6 * np.sin(np.pi * hour / 24), "relative_humidity_2m": np.full(len(times), 75.0),
            "precipitation": np.zeros(len(times)), "wind_speed_10m": np.full(len(times), 10.0)}
    frames = []
    for loc in locations:
        for v, vals in base.items():
            vals = vals.astype(float).copy()
            if loc["region_id"] == region_to_break and v == "shortwave_radiation" and drop_hours:
                vals[(times >= pd.Timestamp("2026-11-12")) & (times < pd.Timestamp("2026-11-12") + pd.Timedelta(hours=drop_hours))] = np.nan
            frames.append(pd.DataFrame({"time": times, "region_id": loc["region_id"], "model": "ecmwf_ifs025",
                                        "variable": v, "lead": 0, "value": vals}))
    grid = pd.DataFrame([{"region_id": l["region_id"], "grid_lat": GRID[l["region_id"]][0],
                          "grid_lon": GRID[l["region_id"]][1], "elevation": 10} for l in locations])
    return pd.concat(frames, ignore_index=True), grid


def fresh_meta():
    return {"run_initialisation_utc": datetime(2026, 11, 11, 0, 0, tzinfo=timezone.utc)}


def test_live_engine_three_days_all_regions():
    hourly_holder = {}

    def fetch(locs):
        hourly_holder["h"] = synthetic_hourly(locs)
        return hourly_holder["h"]

    res = SolarPredictor(ROOT, fetch_hourly=fetch, fetch_meta=fresh_meta).predict(now=NOW)
    p = res.predictions
    assert len(p) == 12 and (p["status"] == "ok").all()
    assert sorted(p["target_date"].astype(str).unique()) == ["2026-11-12", "2026-11-13", "2026-11-14"]
    assert set(p["lead_days"]) == {1, 2, 3}
    assert p["ml_prediction"].notna().all() and p["category"].notna().all()
    assert not any("forecast run" in w for w in res.warnings)


def test_live_features_match_training_pipeline():
    """Train/inference consistency: the live path must give the same prediction
    as running the training transformations on the same hourly data."""
    pred = SolarPredictor(ROOT, fetch_hourly=synthetic_hourly, fetch_meta=fresh_meta)
    res = pred.predict(["puducherry"], now=NOW)
    hourly, _ = synthetic_hourly([pred.locations["puducherry"]])
    daily = aggregate_daily(hourly, "+05:30")
    daily = daily[daily["target_date"] == "2026-11-13"].assign(lead_days=2)
    feats = build_features(clean_forecasts(daily)[0], pred.climatology)
    expected = pred.model.predict(feats)[0]
    got = res.predictions.set_index("lead_days").loc[2, "ml_prediction"]
    assert got == pytest.approx(expected)
    assert res.predictions.set_index("lead_days").loc[2, "raw_forecast"] == pytest.approx(
        feats["forecast_shortwave_radiation"].iloc[0])


def test_incomplete_day_is_not_predicted():
    fetch = lambda locs: synthetic_hourly(locs, drop_hours=6, region_to_break="mahe")  # noqa: E731
    res = SolarPredictor(ROOT, fetch_hourly=fetch, fetch_meta=fresh_meta).predict(now=NOW)
    p = res.predictions.set_index(["region_id", "lead_days"])
    assert p.loc[("mahe", 1), "status"] == "unavailable" and np.isnan(p.loc[("mahe", 1), "ml_prediction"])
    assert p.loc[("puducherry", 1), "status"] == "ok"
    assert any("Mahe" in w and "incomplete" in w for w in res.warnings)


def test_stale_run_and_grid_mismatch_warn():
    def fetch(locs):
        h, g = synthetic_hourly(locs)
        g.loc[g["region_id"] == "yanam", "grid_lat"] = 17.0
        return h, g
    stale = lambda: {"run_initialisation_utc": datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)}  # noqa: E731
    res = SolarPredictor(ROOT, fetch_hourly=fetch, fetch_meta=stale).predict(now=NOW)
    assert any("forecast run" in w and "old" in w for w in res.warnings)
    assert any("Yanam" in w and "grid cell" in w for w in res.warnings)


def test_api_failure_raises_clear_error():
    def fail(locs):
        raise LiveForecastError("Open-Meteo forecast service unavailable (timeout)")
    with pytest.raises(LiveForecastError, match="unavailable"):
        SolarPredictor(ROOT, fetch_hourly=fail, fetch_meta=fresh_meta).predict(now=NOW)


def test_meta_failure_is_non_fatal():
    def meta_fail():
        raise LiveForecastError("metadata down")
    res = SolarPredictor(ROOT, fetch_hourly=synthetic_hourly, fetch_meta=meta_fail).predict(now=NOW)
    assert len(res.predictions) == 12 and any("run time" in w for w in res.warnings)


def test_categorise_thresholds():
    assert categorise(4.99, 5.0, 6.0) == "low"
    assert categorise(5.0, 5.0, 6.0) == "moderate"
    assert categorise(6.0, 5.0, 6.0) == "high"
    assert categorise(float("nan"), 5.0, 6.0) is None


def test_drift_check_flags_unusual_model_inputs_only():
    def fetch(locs):
        h, g = synthetic_hourly(locs)
        h.loc[h["variable"] == "temperature_2m", "value"] += 12      # implausibly hot
        h.loc[h["variable"] == "cloud_cover", "value"] = 100.0        # monitored only
        return h, g
    res = SolarPredictor(ROOT, fetch_hourly=fetch, fetch_meta=fresh_meta).predict(["puducherry"], now=NOW)
    p = res.predictions
    assert (p["status"] == "check_inputs").all()
    assert p["notes"].str.contains("forecast_temperature").all()
    assert not p["notes"].str.contains("unusual inputs: .*cloud_cover", regex=True).any()
