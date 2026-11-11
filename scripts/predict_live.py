"""Print live three-day solar-radiation predictions for the four regions and
append them to data/predictions/prediction_log.csv.

Usage:  .venv/Scripts/python scripts/predict_live.py [region_id ...]
"""
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.predict import SolarPredictor  # noqa: E402
from src.services.live_forecast import LiveForecastError  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def main():
    regions = sys.argv[1:] or None
    try:
        result = SolarPredictor().predict(regions, log_path=ROOT / "data" / "predictions" / "prediction_log.csv")
    except LiveForecastError as exc:
        print(f"Live forecast unavailable: {exc}")
        sys.exit(1)
    m = result.meta
    print(f"Issued {m['issued_for_ist_date']} (IST) | forecast model {m['forecast_model']} | "
          f"run initialised {m['run_initialisation_utc']} | ML model {m['model']} ({m['model_version']})\n")
    with pd.option_context("display.width", 200, "display.max_colwidth", 80,
                           "display.float_format", "{:.2f}".format):
        print(result.predictions.drop(columns="region_id").to_string(index=False))
    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(" -", w)
    print("\nUnits: kWh/m²/day. Estimates from a gridded model, not local measurements.")


if __name__ == "__main__":
    main()
