"""Module 6: model definitions for solar-radiation prediction.

Every model is a scikit-learn compatible estimator that takes the feature
DataFrame produced by `src.preprocessing.build_training_data.build_features`
and predicts NASA daily solar radiation (kWh/m^2/day). Preprocessing (one-hot
encoding, scaling, interactions) lives inside the pipeline, so it is fitted on
the training rows only and saved together with the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.preprocessing.build_training_data import CLIMATOLOGY_FEATURES, SEASONAL_FEATURES
from src.preprocessing.clean_data import FORECAST_FEATURES

REGIONS = ["karaikal", "mahe", "puducherry", "yanam"]
FORECAST_COL = "forecast_shortwave_radiation"
INTERACTION_BASE = ["forecast_shortwave_radiation", "forecast_cloud_cover_mean"]

# feature sets for the regional ablation (doc section 10.4)
FEATURE_SETS = {
    # no region identity and no region-specific climatology
    "location_blind": {"numeric": FORECAST_FEATURES + ["lead_days"] + SEASONAL_FEATURES,
                       "categorical": []},
    # primary approach (section 7.6)
    "region_aware": {"numeric": FORECAST_FEATURES + ["lead_days"] + SEASONAL_FEATURES
                     + CLIMATOLOGY_FEATURES, "categorical": ["region_id"]},
}


class RegionInteractions(BaseEstimator, TransformerMixin):
    """Adds `<col>_x_<region>` columns (col value if row is that region, else 0),
    so a linear model can learn a region-specific slope (doc section 7.6)."""

    def __init__(self, columns=tuple(INTERACTION_BASE), regions=tuple(REGIONS)):
        self.columns = columns
        self.regions = regions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            for r in self.regions:
                X[f"{col}_x_{r}"] = np.where(X["region_id"] == r, X[col], 0.0)
        return X

    def get_feature_names_out(self, input_features=None):
        return None


def _preprocessor(numeric, categorical, scale: bool):
    transformers = [("num", StandardScaler() if scale else "passthrough", numeric)]
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="error", sparse_output=False),
                             categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_model(kind: str, feature_set: str = "region_aware", exclude: tuple = (), **params) -> Pipeline:
    """kind in {'linear', 'random_forest', 'xgboost'}; `exclude` drops features
    (e.g. inputs whose archived/live behaviour has drifted)."""
    fs = FEATURE_SETS[feature_set]
    numeric = [c for c in fs["numeric"] if c not in exclude]
    categorical = list(fs["categorical"])
    inter = tuple(c for c in INTERACTION_BASE if c not in exclude)
    steps = []
    if kind == "linear":
        if categorical:
            steps.append(("interactions", RegionInteractions(columns=inter)))
            numeric = numeric + [f"{c}_x_{r}" for c in inter for r in REGIONS]
        steps.append(("prep", _preprocessor(numeric, categorical, scale=True)))
        steps.append(("model", Ridge(**{"alpha": 1.0, **params})))
    elif kind == "random_forest":
        steps.append(("prep", _preprocessor(numeric, categorical, scale=False)))
        steps.append(("model", RandomForestRegressor(**{"n_estimators": 400, "n_jobs": -1,
                                                        "random_state": 42, **params})))
    elif kind == "xgboost":
        steps.append(("prep", _preprocessor(numeric, categorical, scale=False)))
        steps.append(("model", XGBRegressor(**{"n_estimators": 400, "learning_rate": 0.05,
                                               "max_depth": 4, "subsample": 0.8,
                                               "colsample_bytree": 0.8, "n_jobs": -1,
                                               "random_state": 42, **params})))
    else:
        raise ValueError(kind)
    return Pipeline(steps)


class ResidualRegressor(BaseEstimator, RegressorMixin):
    """Learns the correction (target - raw forecast) and adds it back, so the
    model is a region-aware forecast corrector (doc section 7.6)."""

    def __init__(self, estimator, offset_col: str = FORECAST_COL):
        self.estimator = estimator
        self.offset_col = offset_col

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y) - X[self.offset_col].to_numpy())
        return self

    def predict(self, X):
        return self.estimator_.predict(X) + X[self.offset_col].to_numpy()


class PerRegionRegressor(BaseEstimator, RegressorMixin):
    """One independent clone of `estimator` per region (ablation variant 3)."""

    def __init__(self, estimator, region_col: str = "region_id"):
        self.estimator = estimator
        self.region_col = region_col

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.models_ = {r: clone(self.estimator).fit(g, y.loc[g.index])
                        for r, g in X.groupby(self.region_col)}
        return self

    def predict(self, X):
        out = pd.Series(np.nan, index=X.index)
        for r, g in X.groupby(self.region_col):
            if r not in self.models_:
                raise ValueError(f"no model for region {r!r}")
            out.loc[g.index] = self.models_[r].predict(g)
        return out.to_numpy()


def build_estimator(kind: str, variant: str = "region_aware", target: str = "residual",
                    exclude: tuple = (), **params):
    """variant in {'location_blind', 'region_aware', 'per_region'};
    target in {'direct', 'residual'}."""
    if variant == "per_region":
        # inside one region, region_id/climatology identity is constant; keep
        # climatology (it still varies by day of year) but drop region one-hot
        base = make_model(kind, "region_aware", exclude=exclude, **params)
        base = _drop_region_categorical(base, kind, exclude)
        est = PerRegionRegressor(base)
    else:
        est = make_model(kind, variant, exclude=exclude, **params)
    return ResidualRegressor(est) if target == "residual" else est


def _drop_region_categorical(pipe: Pipeline, kind: str, exclude: tuple = ()) -> Pipeline:
    numeric = [c for c in FEATURE_SETS["region_aware"]["numeric"] if c not in exclude]
    steps = [(n, s) for n, s in pipe.steps if n != "interactions"]
    steps = [(n, _preprocessor(numeric, [], scale=(kind == "linear")))
             if n == "prep" else (n, s) for n, s in steps]
    return Pipeline(steps)
