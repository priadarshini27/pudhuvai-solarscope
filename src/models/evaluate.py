"""Evaluation metrics (kWh/m^2/day) shared by baselines and ML models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {"n": int(len(y_true)), "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())), "bias": float(err.mean()),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")}


def grouped_metrics(df: pd.DataFrame, y_true: str, y_pred: str,
                    by: list[str] | None = None) -> pd.DataFrame:
    """Metrics overall (by=None) or per group, as a tidy DataFrame."""
    if not by:
        return pd.DataFrame([regression_metrics(df[y_true], df[y_pred])])
    rows = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **regression_metrics(g[y_true], g[y_pred])})
    return pd.DataFrame(rows)


def evaluation_table(df: pd.DataFrame, y_true: str, predictions: dict[str, str],
                     split_col: str = "split") -> pd.DataFrame:
    """Long table: model x split x breakdown (overall / lead / region)."""
    out = []
    for name, col in predictions.items():
        for split, g in df.groupby(split_col):
            for scope, by in (("overall", None), ("lead", ["lead_days"]), ("region", ["region_id"])):
                m = grouped_metrics(g, y_true, col, by)
                m.insert(0, "group", "all" if by is None else m.pop(by[0]).astype(str))
                m.insert(0, "scope", scope)
                m.insert(0, "split", split)
                m.insert(0, "model", name)
                out.append(m)
    return pd.concat(out, ignore_index=True)
