import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.rooftop import estimate_daily_energy, validate_inputs  # noqa: E402

CFG = json.loads((ROOT / "config" / "pipeline.json").read_text(encoding="utf-8"))["rooftop"]


def test_formula():
    # 5 kWh/m2/day (5 peak sun hours) x 3 kWp x 0.75 = 11.25 kWh
    assert estimate_daily_energy(5.0, 3.0, 0.75, CFG) == pytest.approx(11.25)
    assert estimate_daily_energy(0.0, 3.0, 0.75, CFG) == 0.0


def test_scales_linearly():
    a = estimate_daily_energy(4.2, 2.0, 0.8, CFG)
    assert estimate_daily_energy(4.2, 4.0, 0.8, CFG) == pytest.approx(2 * a)
    assert estimate_daily_energy(8.4, 2.0, 0.8, CFG) == pytest.approx(2 * a)


def test_missing_radiation_gives_none():
    assert estimate_daily_energy(None, 3.0, 0.75, CFG) is None
    assert estimate_daily_energy(float("nan"), 3.0, 0.75, CFG) is None


@pytest.mark.parametrize("cap,pr", [(0, 0.75), (-1, 0.75), (1e6, 0.75), (math.inf, 0.75),
                                    (3, 0.2), (3, 1.2), (3, float("nan"))])
def test_rejects_invalid_inputs(cap, pr):
    with pytest.raises(ValueError):
        validate_inputs(cap, pr, CFG)


def test_rejects_negative_radiation():
    with pytest.raises(ValueError):
        estimate_daily_energy(-1.0, 3.0, 0.75, CFG)


def test_defaults_are_within_bounds():
    validate_inputs(CFG["default_capacity_kw"], CFG["default_performance_ratio"], CFG)
