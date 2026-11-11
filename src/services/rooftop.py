"""Module 9: simplified rooftop solar-energy estimate.

    energy (kWh/day) = radiation (kWh/m²/day) × capacity (kWp) × performance ratio

A panel's kWp rating is defined at 1 kW/m² irradiance, so daily radiation in
kWh/m²/day equals "peak sun hours", and capacity × peak sun hours is the ideal
output. The performance ratio (PR) lumps together real-world losses
(temperature, wiring, inverter, soiling, mismatch, minor shading).

This is an indicative, educational estimate, not a PV-system simulation: it
uses horizontal-surface radiation and ignores tilt, orientation, shading
geometry and module/inverter specifics.
"""
from __future__ import annotations

import math


def validate_inputs(capacity_kw: float, performance_ratio: float, cfg: dict | None = None) -> None:
    lo_c, hi_c = (cfg or {}).get("min_capacity_kw", 0.1), (cfg or {}).get("max_capacity_kw", 1000.0)
    lo_pr, hi_pr = (cfg or {}).get("min_performance_ratio", 0.3), (cfg or {}).get("max_performance_ratio", 1.0)
    if not (isinstance(capacity_kw, (int, float)) and math.isfinite(capacity_kw) and lo_c <= capacity_kw <= hi_c):
        raise ValueError(f"system capacity must be between {lo_c} and {hi_c} kW")
    if not (isinstance(performance_ratio, (int, float)) and math.isfinite(performance_ratio)
            and lo_pr <= performance_ratio <= hi_pr):
        raise ValueError(f"performance ratio must be between {lo_pr} and {hi_pr}")


def estimate_daily_energy(radiation_kwh_m2: float | None, capacity_kw: float, performance_ratio: float,
                          cfg: dict | None = None) -> float | None:
    """Indicative daily energy in kWh, or None when no radiation estimate exists."""
    validate_inputs(capacity_kw, performance_ratio, cfg)
    if radiation_kwh_m2 is None or not math.isfinite(radiation_kwh_m2):
        return None
    if radiation_kwh_m2 < 0:
        raise ValueError("radiation cannot be negative")
    return radiation_kwh_m2 * capacity_kw * performance_ratio
