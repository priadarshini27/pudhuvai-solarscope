import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_collection.openmeteo_archive import aggregate_daily, lead_variable  # noqa: E402


def make_hourly(start_utc: str, hours: int, values: dict[str, np.ndarray], lead: int = 1,
                region: str = "puducherry", model: str = "m") -> pd.DataFrame:
    times = pd.date_range(start_utc, periods=hours, freq="h")
    frames = [pd.DataFrame({"time": times, "region_id": region, "model": model,
                            "variable": var, "lead": lead, "value": vals})
              for var, vals in values.items()]
    return pd.concat(frames, ignore_index=True)


def test_lead_variable_naming():
    assert lead_variable("shortwave_radiation", 0) == "shortwave_radiation"
    assert lead_variable("shortwave_radiation", 2) == "shortwave_radiation_previous_day2"


def test_shortwave_sum_converted_to_kwh_and_local_day():
    # IST day 2025-01-02 = UTC 2025-01-01 18:30 .. 2025-01-02 18:30.
    # Hourly stamps are whole UTC hours, so the IST day holds UTC 19:00..18:00.
    sw = np.zeros(72)
    sw[24 + 4: 24 + 12] = 500.0          # 8 sunny hours on UTC 2025-01-02 (IST morning-afternoon)
    hourly = make_hourly("2025-01-01T00:00", 72, {"shortwave_radiation": sw})
    daily = aggregate_daily(hourly, "+05:30").set_index("target_date")
    assert daily.loc["2025-01-02", "forecast_shortwave_radiation"] == pytest.approx(4.0)
    assert daily.loc["2025-01-02", "hours_complete"] == 24


def test_temperature_stats_and_units_untouched():
    t = np.r_[np.full(24, 25.0), np.linspace(20, 35, 24), np.full(24, 25.0)]
    # IST day D holds UTC stamps 19:00 (D-1) .. 18:00 (D); the 2nd block of 24
    # (UTC 2025-01-01T19:00 ..) is exactly IST 2025-01-02
    hourly = make_hourly("2024-12-31T19:00", 72, {"temperature_2m": t})
    daily = aggregate_daily(hourly, "+05:30").set_index("target_date")
    row = daily.loc["2025-01-02"]
    assert row["forecast_temperature_max"] == pytest.approx(35.0)
    assert row["forecast_temperature_min"] == pytest.approx(20.0)
    assert row["forecast_temperature_mean"] == pytest.approx(27.5)


def test_incomplete_day_gives_nan_not_partial_sum():
    sw = np.full(48, 100.0)
    sw[30] = np.nan
    hourly = make_hourly("2025-01-01T00:00", 48, {"shortwave_radiation": sw,
                                                   "cloud_cover": np.full(48, 50.0)})
    daily = aggregate_daily(hourly, "+00:00").set_index("target_date")
    assert np.isnan(daily.loc["2025-01-02", "forecast_shortwave_radiation"])
    assert daily.loc["2025-01-02", "hours_complete"] == 23
    # other complete variables on that day are kept
    assert daily.loc["2025-01-02", "forecast_cloud_cover_mean"] == pytest.approx(50.0)
    assert daily.loc["2025-01-01", "forecast_shortwave_radiation"] == pytest.approx(2.4)


def test_leads_kept_separate():
    a = make_hourly("2025-01-01T00:00", 24, {"precipitation": np.full(24, 1.0)}, lead=1)
    b = make_hourly("2025-01-01T00:00", 24, {"precipitation": np.full(24, 2.0)}, lead=3)
    daily = aggregate_daily(pd.concat([a, b]), "+00:00").set_index("lead_days")
    assert daily.loc[1, "forecast_precipitation_sum"] == pytest.approx(24.0)
    assert daily.loc[3, "forecast_precipitation_sum"] == pytest.approx(48.0)
