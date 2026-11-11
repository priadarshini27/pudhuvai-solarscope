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
