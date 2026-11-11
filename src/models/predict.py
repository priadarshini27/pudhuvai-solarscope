"""Module 7: three-day solar-radiation predictions from the live forecast.

The live response goes through exactly the training transformations:
    aggregate_daily -> clean_forecasts -> build_features -> saved pipeline
and every prediction carries the raw forecast, a category, data-quality
status and warnings, so the dashboard never shows stale or doubtful values as
if they were fresh.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_collection.openmeteo_archive import aggregate_daily, load_locations
from src.preprocessing.build_training_data import build_features, load_config
from src.preprocessing.clean_data import FORECAST_FEATURES, clean_forecasts
from src.services import live_forecast
from src.services.live_forecast import LiveForecastError

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
IST = timezone(timedelta(hours=5, minutes=30))
MODEL_NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost", "linear": "Linear (ridge) regression"}


@dataclass
class PredictionResult:
    predictions: pd.DataFrame          # one row per region x target date
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def categorise(value: float, low_below: float, high_from: float) -> str | None:
    if value is None or not np.isfinite(value):
        return None
    return "low" if value < low_below else ("high" if value >= high_from else "moderate")


class SolarPredictor:
    """Loads the saved model and produces live three-day predictions.

    `fetch_hourly` / `fetch_meta` can be injected (tests, offline demos);
    by default they call the live Open-Meteo services.
    """

    def __init__(self, root: Path = ROOT, fetch_hourly=None, fetch_meta=None):
        self.root = root
        self.cfg = load_config(root / "config" / "pipeline.json")
        self.live = self.cfg["live"]
        self.locations = {l["region_id"]: l for l in load_locations(root / "config" / "locations.json")}

        model_path = root / "models" / "solar_model.joblib"
        meta_path = root / "models" / "model_metadata.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError("deployed model not found: run scripts/retrain_production.py "
                                    "(after notebooks/04_model_experiments.ipynb)")
        self.model = joblib.load(model_path)
        self.model_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.model_meta["forecast_model"] != self.cfg["forecast_model"]:
            raise ValueError("model was trained on a different forecast model than config/pipeline.json")

        self.climatology = pd.read_csv(root / self.model_meta["climatology_file"])
        # inputs the deployed model was fitted on (for drift checks and grid cells)
        ref_file = root / self.model_meta.get("training_reference_file", "data/processed/solar_model_training.csv")
        ref = pd.read_csv(ref_file, parse_dates=["target_date"])
        t = self.model_meta["trained_on"]
        self.reference = ref[(ref["target_date"] >= t["from"]) & (ref["target_date"] <= t["to"])]
        self.training_grid = self.reference.groupby("region_id")[["grid_latitude", "grid_longitude"]].first()

        self._fetch_hourly = fetch_hourly or self._default_fetch_hourly
        self._fetch_meta = fetch_meta or self._default_fetch_meta

    # ------------------------------------------------------------- fetching
    def _default_fetch_hourly(self, locations):
        return live_forecast.fetch_live_hourly(
            locations, self.cfg["forecast_model"], self.cfg["forecast_variables"],
            self.live["forecast_endpoint"], self.live["past_days"], self.live["forecast_days"],
            self.live["cache_minutes"])

    def _default_fetch_meta(self):
        return live_forecast.fetch_run_metadata(self.cfg["forecast_model"],
                                                self.live["run_metadata_endpoint"],
                                                self.live["cache_minutes"])

    # ------------------------------------------------------------- checks
    def _model_age_warning(self, today: date) -> str | None:
        trained_to = date.fromisoformat(self.model_meta["trained_on"]["to"])
        age = (today - trained_to).days
        if age > self.live["model_max_age_days"]:
            return (f"The model was trained on data up to {trained_to} ({age} days ago); "
                    "it should be retrained with newer data.")
        return None

    def _drift_flags(self, row: pd.Series) -> list[str]:
        """Inputs far outside the training distribution for this region and season."""
        d = self.live["drift"]
        ref = self.reference[self.reference["region_id"] == row["region_id"]]
        doy = row["target_date"].day_of_year
        dist = np.abs(ref["target_date"].dt.day_of_year - doy)
        ref = ref[np.minimum(dist, 365 - dist) <= d["window_days"]]
        if len(ref) < 20:
            return []
        flags = []
        for col in FORECAST_FEATURES:
            sd = ref[col].std()
            if sd and np.isfinite(sd):
                z = (row[col] - ref[col].mean()) / sd
                if abs(z) > d["z_warn"]:
                    flags.append(f"{col} (z={z:+.1f})")
        return flags

    # ------------------------------------------------------------- predict
    def predict(self, region_ids: list[str] | None = None, now: datetime | None = None,
                log_path: Path | None = None) -> PredictionResult:
        now = (now or datetime.now(timezone.utc)).astimezone(IST)
        today = now.date()
        region_ids = region_ids or list(self.locations)
        unknown = set(region_ids) - set(self.locations)
        if unknown:
            raise ValueError(f"unknown region(s): {sorted(unknown)}")
        locations = [self.locations[r] for r in region_ids]
        warnings: list[str] = []

        # 1. live hourly forecast (raises LiveForecastError on failure)
        hourly, grid = self._fetch_hourly(locations)

        # 2. same grid cell as training?
        for _, g in grid.iterrows():
            tg = self.training_grid.loc[g["region_id"]]
            if not (np.isclose(g["grid_lat"], tg["grid_latitude"]) and np.isclose(g["grid_lon"], tg["grid_longitude"])):
                warnings.append(f"{self.locations[g['region_id']]['name']}: live grid cell "
                                f"({g['grid_lat']}, {g['grid_lon']}) differs from training "
                                f"({tg['grid_latitude']}, {tg['grid_longitude']}).")

        # 3. forecast run freshness (non-fatal if the metadata endpoint fails)
        run_meta = {}
        try:
            run_meta = self._fetch_meta()
            init = run_meta.get("run_initialisation_utc")
            if init is not None:
                age_h = (now - init).total_seconds() / 3600
                if age_h > self.live["max_run_age_hours"]:
                    warnings.append(f"The latest forecast run is {age_h:.0f} h old (initialised "
                                    f"{init:%Y-%m-%d %H:%M} UTC); predictions may be out of date.")
        except LiveForecastError as exc:
            warnings.append(f"Could not check forecast run time: {exc}")

        w = self._model_age_warning(today)
        if w:
            warnings.append(w)

        # 4. same daily aggregation, cleaning and features as training
        daily = aggregate_daily(hourly, self.cfg["local_day"]["utc_offset"])
        targets = [today + timedelta(days=k) for k in self.cfg["lead_days"]]
        daily = daily[daily["target_date"].dt.date.isin(targets)].copy()
        daily["lead_days"] = (daily["target_date"].dt.date - today).apply(lambda d: d.days)
        clean, _ = clean_forecasts(daily)
        feats = build_features(clean, self.climatology) if len(clean) else clean

        cat = self.cfg["categories"]
        rows = []
        for rid in region_ids:
            for k, t in zip(self.cfg["lead_days"], targets):
                base = {"region_id": rid, "region": self.locations[rid]["name"], "target_date": t, "lead_days": k}
                hit = feats[(feats["region_id"] == rid) & (feats["target_date"].dt.date == t)] if len(feats) else feats
                if hit.empty:
                    rows.append({**base, "ml_prediction": np.nan, "raw_forecast": np.nan, "category": None,
                                 "status": "unavailable", "notes": "forecast incomplete for this day"})
                    warnings.append(f"{base['region']} {t}: forecast incomplete, no prediction made.")
                    continue
                r = hit.iloc[0]
                pred = float(self.model.predict(hit)[0])
                all_flags = self._drift_flags(r)
                # only inputs the model uses make a prediction doubtful;
                # monitor-only inputs (e.g. drifted cloud cover) are just noted
                used = set(self.model_meta["features"])
                flags = [f for f in all_flags if f.split(" ")[0] in used]
                monitored = [f for f in all_flags if f not in flags]
                notes = []
                if flags:
                    notes.append("unusual inputs: " + ", ".join(flags))
                if monitored:
                    notes.append("monitored (not used by model): " + ", ".join(monitored))
                rows.append({**base, "ml_prediction": pred, "raw_forecast": float(r["forecast_shortwave_radiation"]),
                             "category": categorise(pred, cat["low_below"], cat["high_from"]),
                             "status": "check_inputs" if flags else "ok", "notes": "; ".join(notes)})
                if flags:
                    warnings.append(f"{base['region']} {t}: unusual forecast inputs ({', '.join(flags)}).")

        preds = pd.DataFrame(rows)
        meta = {
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "issued_for_ist_date": today.isoformat(),
            "forecast_model": self.cfg["forecast_model"],
            "run_initialisation_utc": run_meta.get("run_initialisation_utc").isoformat()
            if run_meta.get("run_initialisation_utc") else None,
            "model_version": self.model_meta.get("version"),
            "model": f"{self.model_meta['model']} / {self.model_meta['variant']} / {self.model_meta['target']}",
            "model_name": MODEL_NAMES.get(self.model_meta["model"], self.model_meta["model"]),
            "grid": grid.to_dict(orient="records"),
        }
        result = PredictionResult(preds, warnings, meta)
        if log_path:
            self.log_predictions(result, log_path)
        return result

    @staticmethod
    def log_predictions(result: PredictionResult, path: Path):
        """Append predictions + metadata for reproducibility and later evaluation."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = result.predictions.assign(**{k: v for k, v in result.meta.items() if k != "grid"})
        rec.to_csv(path, mode="a", header=not path.exists(), index=False)
