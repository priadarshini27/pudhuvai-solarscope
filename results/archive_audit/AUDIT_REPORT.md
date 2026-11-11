# Open-Meteo archive coverage audit

Generated 2026-11-11 by `scripts/audit_archive_coverage.py`. Scan range 2018-01-01 → 2026-11-10.

A day is **usable** for a region and model when all 24 UTC hours are present for every variable (shortwave_radiation, cloud_cover, temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m) at every lead (1, 2, 3). **Continuous start** is the first month after which every month is ≥95% usable for that region.

## 1. Model overview (window common to all four regions)

| model | continuous_start | last_usable | max_gap_days | unusable_days | overlap_end | overlap_days | lead1_mean_mae | lead1_mean_bias | lead1_mean_corr | lead_day1_differs_from_day0 |
|---|---|---|---|---|---|---|---|---|---|---|
| best_match | 2024-02-01 | 2026-11-10 | 0 | 0 | 2026-11-03 | 1007 | 0.63 | 0.08 | 0.81 | 0.93 |
| icon_seamless | 2024-02-01 | 2026-11-10 | 4 | 4 | 2026-11-03 | 1007 | 0.63 | -0.08 | 0.82 | 0.94 |
| gfs_seamless | 2024-02-01 | 2026-11-10 | 2 | 2 | 2026-11-03 | 1007 | 1.05 | 0.50 | 0.68 | 0.91 |
| ecmwf_ifs025 | 2024-04-01 | 2026-11-10 | 6 | 6 | 2026-11-03 | 947 | 0.64 | 0.33 | 0.81 | 0.91 |
| gem_global | 2026-07-01 | 2026-11-10 | 0 | 0 | 2026-11-03 | 126 | 0.89 | 0.40 | 0.62 | 0.93 |
| meteofrance_arpege_world | 2026-09-01 | 2026-11-10 | 0 | 0 | 2026-11-03 | 64 | 0.81 | 0.36 | 0.72 | 0.90 |

`lead_day1_differs_from_day0` = share of daylight hours where the day-1 archived value differs from the latest run. Values near 0 would mean the archive is not returning genuine older forecasts.

## 2. Coverage per region and model

| region_id | model | first_any_usable | continuous_start | last_usable | usable_days_since_start | unusable_days_since_start | longest_gap_days |
|---|---|---|---|---|---|---|---|
| karaikal | best_match | 2024-01-22 | 2024-02-01 | 2026-11-10 | 1014 | 0 | 0 |
| mahe | best_match | 2024-01-22 | 2024-02-01 | 2026-11-10 | 1014 | 0 | 0 |
| puducherry | best_match | 2024-01-22 | 2024-02-01 | 2026-11-10 | 1014 | 0 | 0 |
| yanam | best_match | 2024-01-22 | 2024-02-01 | 2026-11-10 | 1014 | 0 | 0 |
| karaikal | ecmwf_ifs025 | 2024-03-09 | 2024-04-01 | 2026-11-10 | 948 | 6 | 6 |
| mahe | ecmwf_ifs025 | 2024-03-09 | 2024-04-01 | 2026-11-10 | 948 | 6 | 6 |
| puducherry | ecmwf_ifs025 | 2024-03-09 | 2024-04-01 | 2026-11-10 | 948 | 6 | 6 |
| yanam | ecmwf_ifs025 | 2024-03-09 | 2024-04-01 | 2026-11-10 | 948 | 6 | 6 |
| karaikal | gem_global | 2026-06-01 | 2026-07-01 | 2026-11-10 | 133 | 0 | 0 |
| mahe | gem_global | 2026-06-01 | 2026-07-01 | 2026-11-10 | 133 | 0 | 0 |
| puducherry | gem_global | 2026-06-01 | 2026-07-01 | 2026-11-10 | 133 | 0 | 0 |
| yanam | gem_global | 2026-06-01 | 2026-07-01 | 2026-11-10 | 133 | 0 | 0 |
| karaikal | gfs_seamless | 2024-01-22 | 2024-02-01 | 2026-11-10 | 1012 | 2 | 2 |
| mahe | gfs_seamless | 2024-01-22 | 2024-02-01 | 2026-11-10 | 1012 | 2 | 2 |
| puducherry | gfs_seamless | 2024-01-22 | 2024-02-01 | 2026-11-10 | 1012 | 2 | 2 |
| yanam | gfs_seamless | 2024-01-22 | 2024-02-01 | 2026-11-10 | 1012 | 2 | 2 |
| karaikal | icon_seamless | 2024-01-22 | 2024-02-01 | 2026-11-10 | 1010 | 4 | 4 |
| mahe | icon_seamless | 2024-01-22 | 2024-02-01 | 2026-11-10 | 1010 | 4 | 4 |
| puducherry | icon_seamless | 2024-01-22 | 2024-02-01 | 2026-11-10 | 1010 | 4 | 4 |
| yanam | icon_seamless | 2024-01-22 | 2024-02-01 | 2026-11-10 | 1010 | 4 | 4 |
| karaikal | meteofrance_arpege_world | 2026-08-10 | 2026-09-01 | 2026-11-10 | 71 | 0 | 0 |
| mahe | meteofrance_arpege_world | 2026-08-10 | 2026-09-01 | 2026-11-10 | 71 | 0 | 0 |
| puducherry | meteofrance_arpege_world | 2026-08-10 | 2026-09-01 | 2026-11-10 | 71 | 0 | 0 |
| yanam | meteofrance_arpege_world | 2026-08-10 | 2026-09-01 | 2026-11-10 | 71 | 0 | 0 |

## 3. Grid cells used by Open-Meteo

| region_id | requested_lat | requested_lon | grid_lat | grid_lon | elevation |
|---|---|---|---|---|---|
| puducherry | 11.93 | 79.80 | 11.92 | 79.83 | 10.00 |
| karaikal | 10.92 | 79.82 | 10.93 | 79.80 | 4.00 |
| mahe | 11.70 | 75.54 | 11.70 | 75.59 | 11.00 |
| yanam | 16.73 | 82.21 | 16.70 | 82.18 | 9.00 |

## 4. NASA POWER label availability

| region_id | nasa_last_valid_LST | nasa_last_valid_UTC | nasa_first_valid_LST |
|---|---|---|---|
| puducherry | 2026-11-03 | 2026-08-31 | 2001-01-01 |
| karaikal | 2026-11-03 | 2026-08-31 | 2001-01-01 |
| mahe | 2026-11-03 | 2026-08-31 | 2001-01-01 |
| yanam | 2026-11-03 | 2026-08-31 | 2001-01-01 |

Difference between NASA daily totals using LST vs UTC day definitions:

| region_id | n_days | mean_abs_diff_kwh | max_abs_diff_kwh | mean_lst |
|---|---|---|---|---|
| puducherry | 943 | 0.000 | 0.170 | 5.327 |
| karaikal | 943 | 0.000 | 0.260 | 5.359 |
| mahe | 943 | 0.001 | 0.516 | 5.106 |
| yanam | 943 | 0.000 | 0.171 | 4.898 |

## 5. Raw forecast skill (forecast shortwave vs NASA LST, kWh/m²/day)

Indicative only: computed over the whole overlap, not a held-out test.

| model | region_id | MAE lead 1 | MAE lead 2 | MAE lead 3 |
|---|---|---|---|---|
| best_match | karaikal | 0.61 | 0.63 | 0.67 |
| best_match | mahe | 0.64 | 0.67 | 0.71 |
| best_match | puducherry | 0.59 | 0.62 | 0.67 |
| best_match | yanam | 0.68 | 0.75 | 0.81 |
| ecmwf_ifs025 | karaikal | 0.59 | 0.64 | 0.69 |
| ecmwf_ifs025 | mahe | 0.62 | 0.68 | 0.71 |
| ecmwf_ifs025 | puducherry | 0.61 | 0.64 | 0.69 |
| ecmwf_ifs025 | yanam | 0.73 | 0.80 | 0.84 |
| gem_global | karaikal | 0.75 | 0.75 | 0.91 |
| gem_global | mahe | 0.92 | 1.01 | 1.14 |
| gem_global | puducherry | 0.79 | 0.79 | 0.86 |
| gem_global | yanam | 1.10 | 1.15 | 1.30 |
| gfs_seamless | karaikal | 0.98 | 1.02 | 1.09 |
| gfs_seamless | mahe | 1.06 | 1.14 | 1.17 |
| gfs_seamless | puducherry | 1.01 | 1.04 | 1.07 |
| gfs_seamless | yanam | 1.17 | 1.22 | 1.28 |
| icon_seamless | karaikal | 0.58 | 0.62 | 0.65 |
| icon_seamless | mahe | 0.71 | 0.72 | 0.79 |
| icon_seamless | puducherry | 0.57 | 0.60 | 0.64 |
| icon_seamless | yanam | 0.68 | 0.73 | 0.80 |
| meteofrance_arpege_world | karaikal | 0.69 | 4.40 | 5.22 |
| meteofrance_arpege_world | mahe | 0.89 | 4.30 | 5.14 |
| meteofrance_arpege_world | puducherry | 0.72 | 4.18 | 5.05 |
| meteofrance_arpege_world | yanam | 0.92 | 3.79 | 4.33 |

## 6. Recommendation and proposed split

- **Pinned forecast model:** `ecmwf_ifs025` (usable 2024-04-01 → 2026-11-03, 947 days; mean lead-1 MAE 0.64).
- **train:** 2024-04-01 → 2025-05-03 (398 days ≈ 4,776 rows = days × 4 regions × 3 leads)
- **validation:** 2025-05-04 → 2025-11-03 (184 days ≈ 2,208 rows = days × 4 regions × 3 leads)
- **test:** 2025-11-04 → 2026-11-03 (365 days ≈ 4,380 rows = days × 4 regions × 3 leads)
- Test spans 12 months so both monsoons are represented for every region; validation 6 months; the remainder is training.

## 7. Caveats

- `best_match` excluded from the recommendation: selects the underlying model automatically; may change over time.
- `icon_seamless` excluded from the recommendation: relies on DWD ICON Global, which Open-Meteo lists as discontinued 2026-11-30.
- Unusable (gap) days inside the window are excluded from training/evaluation, not interpolated.
- `_previous_dayN` is defined per **hour** (value predicted N×24 h before that hour). A daily 'lead 1' total is therefore assembled from several runs. The live inference features must be built to mimic this (decide in preprocessing).
- Local-day aggregation used here: UTC + 05:30. Both UTC and IST day boundaries fall at night.
