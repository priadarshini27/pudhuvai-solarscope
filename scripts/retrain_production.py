"""Refit the evaluated model configuration on ALL labelled data for deployment.

Reads the configuration chosen in notebooks/04_model_experiments.ipynb
(models/evaluated/model_metadata.json) and refits it - same model, regional
variant, target, hyperparameters and excluded features - on every forecast
day that has a NASA label, ignoring the train/validation/test split.

Writes:
  models/solar_model.joblib        deployed pipeline (loaded by SolarPredictor)
  models/model_metadata.json       metadata; accuracy = evaluated model's test score
  models/training_reference.csv    inputs the model was fitted on (live drift checks)

Periodic retraining:  scripts/collect_data.py  ->  scripts/retrain_production.py
Usage:  .venv/Scripts/python scripts/retrain_production.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.evaluate import regression_metrics  # noqa: E402
from src.models.train import build_estimator, model_input_features  # noqa: E402
from src.preprocessing.build_training_data import FEATURES, TARGET, build_features, load_config  # noqa: E402
from src.preprocessing.clean_data import (NASA_TARGET, clean_forecasts, clean_nasa,  # noqa: E402
                                          load_forecasts, load_nasa)

MODELS = ROOT / "models"
EVALUATED = MODELS / "evaluated"


def labelled_table() -> pd.DataFrame:
    """Every cleaned forecast row joined to its NASA label (no split filter)."""
    clim = pd.read_csv(ROOT / "data" / "processed" / "climatology.csv")
    fc, _ = clean_forecasts(load_forecasts())
    nasa, _ = clean_nasa(load_nasa())
    feats = build_features(fc, clim)
    target = nasa[["region_id", "date", NASA_TARGET]].rename(columns={"date": "target_date", NASA_TARGET: TARGET})
    return feats.merge(target, on=["region_id", "target_date"], how="inner").sort_values(
        ["target_date", "region_id", "lead_days"]).reset_index(drop=True)


def main():
    cfg = load_config()
    ev = json.loads((EVALUATED / "model_metadata.json").read_text(encoding="utf-8"))
    if ev["forecast_model"] != cfg["forecast_model"]:
        raise SystemExit("evaluated model used a different forecast model than config/pipeline.json")

    data = labelled_table()
    exclude = tuple(ev.get("excluded_features", {}))
    model = build_estimator(ev["model"], ev["variant"], ev["target"], exclude=exclude, **ev["params"])
    model.fit(data, data[TARGET])

    # sanity checks: in-sample fit (optimistic, NOT an accuracy estimate) and
    # agreement with the evaluated model on the most recent 60 days
    fit = regression_metrics(data[TARGET], model.predict(data))
    recent = data[data["target_date"] > data["target_date"].max() - pd.Timedelta(days=60)]
    evaluated = joblib.load(EVALUATED / "solar_model.joblib")
    shift = float(np.mean(model.predict(recent) - evaluated.predict(recent)))
    by_year = (data.assign(err=model.predict(data) - data[TARGET])
               .groupby(data["target_date"].dt.year)["err"].mean().round(3).to_dict())

    meta = {
        **{k: ev[k] for k in ("model", "variant", "target", "params", "excluded_features",
                              "forecast_model", "lead_days", "local_day_utc_offset", "nasa_time_standard",
                              "climatology_file", "selection_rule")},
        "features": model_input_features(ev["variant"], exclude),
        "version": f"{ev['version']}-production",
        "created": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "trained_on": {"from": str(data["target_date"].min().date()), "to": str(data["target_date"].max().date()),
                       "rows": len(data), "note": "all labelled rows (train + validation + test periods)"},
        "accuracy": {"source": "test score of the evaluated model (models/evaluated), trained up to "
                               f"{ev['trained_on']['to']}; the production refit has no held-out test",
                     "test": ev["test"]},
        "in_sample_fit": {k: round(v, 4) for k, v in fit.items()},
        "in_sample_bias_by_year": by_year,
        "mean_change_vs_evaluated_last_60_days": round(shift, 4),
        "evaluated_model": "models/evaluated/solar_model.joblib",
        "training_reference_file": "models/training_reference.csv",
    }
    est = model.estimator_ if hasattr(model, "estimator_") else model
    if hasattr(est, "named_steps"):
        fitted_inputs = list(est.named_steps["prep"].get_feature_names_out())
        expected = {c for c in meta["features"] if c != "region_id"}
        used_base = {c for c in fitted_inputs if c in FEATURES}
        assert used_base == expected, f"metadata features {sorted(expected)} != fitted inputs {sorted(used_base)}"
    joblib.dump(model, MODELS / "solar_model.joblib")
    (MODELS / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    ref_cols = ["target_date", "region_id", "lead_days", "grid_latitude", "grid_longitude",
                *[c for c in data.columns if c.startswith("forecast_") and c != "forecast_model"
                  and c != "forecast_issue_date"]]
    data[ref_cols].to_csv(MODELS / "training_reference.csv", index=False, date_format="%Y-%m-%d")

    print(f"Production model: {meta['model']} / {meta['variant']} / {meta['target']} {meta['params']}")
    print(f"Excluded features: {list(exclude)}")
    print(f"Trained on {meta['trained_on']['from']} -> {meta['trained_on']['to']} ({len(data)} rows)")
    print(f"In-sample fit (optimistic): MAE {fit['mae']:.3f}, bias {fit['bias']:+.3f}; by year {by_year}")
    print(f"Mean change vs evaluated model over the last 60 days: {shift:+.3f} kWh/m2/day")
    print(f"Reported accuracy (evaluated model, test year): MAE {ev['test']['ml_model']['mae']:.3f}")


if __name__ == "__main__":
    main()
