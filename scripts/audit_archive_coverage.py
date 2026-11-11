"""Audit how far back the Open-Meteo Previous Runs archive goes for the four
Puducherry UT regions, and how it overlaps with NASA POWER labels.

Stages
  0. validate candidate model identifiers, record API grid cells
  1. coarse scan 2018 -> today: first 7 days of every month, key variables
  2. detailed scan: every day, all 6 variables x leads 1-3, for viable models
  3. NASA POWER label availability (LST vs UTC) and their difference
  4. quick lead-1/2/3 skill of raw forecast shortwave vs NASA
  5. AUDIT_REPORT.md with a proposed chronological split

Usage:  .venv/Scripts/python scripts/audit_archive_coverage.py
Raw responses are cached, so re-runs make no new API calls.
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_collection import nasa_power  # noqa: E402
from src.data_collection.openmeteo_archive import (  # noqa: E402
    DEFAULT_CACHE, OpenMeteoError, daily_completeness, fetch_previous_runs,
    load_locations, probe_models)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("audit")

OUT = ROOT / "results" / "archive_audit"
CACHE = DEFAULT_CACHE / "audit"

CANDIDATE_MODELS = [
    "best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "jma_gsm",
    "gem_global", "ukmo_global_deterministic_10km", "meteofrance_arpege_world",
]
VARIABLES = ["shortwave_radiation", "cloud_cover", "temperature_2m",
             "relative_humidity_2m", "precipitation", "wind_speed_10m"]
LEADS = [1, 2, 3]
COARSE_VARS = ["shortwave_radiation", "cloud_cover", "temperature_2m"]
COARSE_LEADS = [1, 3]
SCAN_START = date(2018, 1, 1)
TODAY = date.today()
LAST_DAY = TODAY - timedelta(days=1)
LOCAL_OFFSET = pd.Timedelta(hours=5, minutes=30)  # IST, for local-day aggregation
COVERAGE_OK = 0.95
NOT_PINNABLE = {
    "best_match": "selects the underlying model automatically; may change over time",
    "icon_seamless": "relies on DWD ICON Global, which Open-Meteo lists as discontinued 2026-11-30",
}
NASA_CLIM_START = "2001-01-01"


def month_starts(start: date, end: date):
    d = date(start.year, start.month, 1)
    while d <= end:
        yield d
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)


# ---------------------------------------------------------------- stage 0
def stage0_models(locations, session):
    valid, rejected = probe_models(CANDIDATE_MODELS, locations[0], cache_dir=CACHE,
                                   session=session)
    for m, why in rejected.items():
        log.warning("model %s rejected: %s", m, why)
    _, grid = fetch_previous_runs(locations, valid[:1], ["shortwave_radiation"], [1],
                                  "2025-06-01", "2025-06-01", cache_dir=CACHE,
                                  session=session)
    return valid, rejected, grid


# ---------------------------------------------------------------- stage 1
def stage1_coarse(locations, models, session) -> pd.DataFrame:
    rows = []
    for ms in month_starts(SCAN_START, LAST_DAY):
        end = min(ms + timedelta(days=6), LAST_DAY)
        hourly, _ = fetch_previous_runs(locations, models, COARSE_VARS, COARSE_LEADS,
                                        ms.isoformat(), end.isoformat(),
                                        cache_dir=CACHE, session=session)
        comp = daily_completeness(hourly)
        g = comp.groupby(["region_id", "model", "variable", "lead"])["complete"].mean()
        for key, frac in g.items():
            rows.append(dict(zip(["region_id", "model", "variable", "lead"], key),
                             month=ms.isoformat()[:7], coverage=frac))
        log.info("coarse %s done", ms.isoformat()[:7])
    return pd.DataFrame(rows)


def coarse_start(coarse: pd.DataFrame) -> dict[str, date | None]:
    """Per model: first sampled month from which ALL regions/vars/leads stay
    >= COVERAGE_OK in every later sampled month (current month excluded)."""
    cur = LAST_DAY.isoformat()[:7]
    worst = (coarse[coarse["month"] < cur]
             .groupby(["model", "month"])["coverage"].min().unstack("month"))
    out = {}
    for model, row in worst.iterrows():
        bad = row[row < COVERAGE_OK]
        if bad.empty:
            out[model] = date.fromisoformat(row.index.min() + "-01")
            continue
        later = row[row.index > bad.index.max()]
        out[model] = date.fromisoformat(later.index.min() + "-01") if not later.empty else None
    return out


# ---------------------------------------------------------------- stage 2
def stage2_detail(locations, model, start, session) -> pd.DataFrame:
    """Daily completeness for all variables/leads, in ~quarterly chunks."""
    frames, hourly_sw = [], []
    s = start
    while s <= LAST_DAY:
        e = min(s + timedelta(days=91), LAST_DAY)
        hourly, _ = fetch_previous_runs(locations, [model], VARIABLES, LEADS,
                                        s.isoformat(), e.isoformat(),
                                        cache_dir=CACHE, session=session)
        frames.append(daily_completeness(hourly))
        hourly_sw.append(hourly[hourly["variable"] == "shortwave_radiation"])
        log.info("detail %s %s..%s done", model, s, e)
        s = e + timedelta(days=1)
    return pd.concat(frames, ignore_index=True), pd.concat(hourly_sw, ignore_index=True)


def usable_days(comp: pd.DataFrame) -> pd.DataFrame:
    """A (region, date) is usable when every variable x lead is complete."""
    u = comp.groupby(["region_id", "model", "date"])["complete"].all().reset_index()
    return u.rename(columns={"complete": "usable"})


def summarise(usable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (region, model), g in usable.groupby(["region_id", "model"]):
        g = g.sort_values("date").set_index("date")["usable"]
        monthly = g.resample("MS").mean()
        monthly = monthly[monthly.index < pd.Timestamp(LAST_DAY.replace(day=1))]
        # first month that is >= COVERAGE_OK usable; isolated later outages
        # are reported as gaps (and dropped downstream) rather than
        # resetting the start date.
        good = monthly[monthly >= COVERAGE_OK]
        start = good.index.min() if not good.empty else pd.NaT
        after = g[g.index >= start] if pd.notna(start) else g.iloc[0:0]
        # longest run of unusable days after the continuous start
        runs = (~after).astype(int).groupby((after != after.shift()).cumsum()).sum()
        rows.append({
            "region_id": region, "model": model,
            "first_any_usable": g[g].index.min(),
            "continuous_start": start,
            "last_usable": g[g].index.max(),
            "usable_days_since_start": int(after.sum()),
            "unusable_days_since_start": int((~after).sum()),
            "longest_gap_days": int(runs.max()) if len(runs) else 0,
        })
    return pd.DataFrame(rows)


def daily_local_kwh(hourly_sw: pd.DataFrame) -> pd.DataFrame:
    """Hourly W/m^2 (mean over preceding hour) -> local-day kWh/m^2/day.
    Shifting UTC by +05:30 puts the day boundary at local midnight (night)."""
    df = hourly_sw.assign(date=(hourly_sw["time"] + LOCAL_OFFSET).dt.floor("D"))
    g = df.groupby(["region_id", "model", "lead", "date"])["value"]
    out = g.agg(total="sum", n="count").reset_index()
    out = out[out["n"] == 24]
    out["forecast_kwh"] = out.pop("total") / 1000.0
    return out.drop(columns="n")


def lead_sanity(locations, model, session) -> float:
    """Fraction of daylight hours where previous_day1 != previous_day0
    (should be well above 0 if genuine archived forecasts are returned)."""
    d0 = LAST_DAY - timedelta(days=30)
    hourly, _ = fetch_previous_runs(locations, [model], ["shortwave_radiation"], [0, 1],
                                    d0.isoformat(), (d0 + timedelta(days=6)).isoformat(),
                                    cache_dir=CACHE, session=session)
    p = hourly.pivot_table(index=["region_id", "time"], columns="lead", values="value")
    p = p[(p[0] > 0) | (p[1] > 0)].dropna()
    return float((p[0] != p[1]).mean()) if len(p) else float("nan")


# ---------------------------------------------------------------- stage 3
def stage3_nasa(locations, detail_start: date, session):
    lst, utc, latest = {}, {}, []
    for loc in locations:
        rid, lat, lon = loc["region_id"], loc["latitude"], loc["longitude"]
        # long LST record (also reused later for climatology)
        lst[rid] = nasa_power.fetch_daily(lat, lon, NASA_CLIM_START, LAST_DAY.isoformat(),
                                          time_standard="LST", session=session)
        utc[rid] = nasa_power.fetch_daily(lat, lon, detail_start.isoformat(),
                                          LAST_DAY.isoformat(), time_standard="UTC",
                                          session=session)
        latest.append({"region_id": rid,
                       "nasa_last_valid_LST": nasa_power.last_valid_date(lst[rid]),
                       "nasa_last_valid_UTC": nasa_power.last_valid_date(utc[rid]),
                       "nasa_first_valid_LST": lst[rid]["ALLSKY_SFC_SW_DWN"].dropna().index.min()})
    diffs = []
    for rid in lst:
        j = lst[rid].join(utc[rid], how="inner", lsuffix="_lst", rsuffix="_utc").dropna()
        d = j["ALLSKY_SFC_SW_DWN_lst"] - j["ALLSKY_SFC_SW_DWN_utc"]
        diffs.append({"region_id": rid, "n_days": len(d),
                      "mean_abs_diff_kwh": d.abs().mean(), "max_abs_diff_kwh": d.abs().max(),
                      "mean_lst": j["ALLSKY_SFC_SW_DWN_lst"].mean()})
    return lst, pd.DataFrame(latest), pd.DataFrame(diffs)


# ---------------------------------------------------------------- stage 4
def stage4_skill(fc_daily: pd.DataFrame, nasa_lst: dict) -> pd.DataFrame:
    nasa = pd.concat([df["ALLSKY_SFC_SW_DWN"].rename("nasa_kwh").to_frame()
                      .assign(region_id=rid).reset_index() for rid, df in nasa_lst.items()])
    j = fc_daily.merge(nasa, on=["region_id", "date"]).dropna()
    rows = []
    for (model, region, lead), g in j.groupby(["model", "region_id", "lead"]):
        err = g["forecast_kwh"] - g["nasa_kwh"]
        rows.append({"model": model, "region_id": region, "lead": lead, "n": len(g),
                     "mae": err.abs().mean(), "bias": err.mean(),
                     "rmse": float(np.sqrt((err ** 2).mean())),
                     "corr": g["forecast_kwh"].corr(g["nasa_kwh"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stage 5
def propose_split(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    test_start = (end - pd.DateOffset(months=12) + pd.Timedelta(days=1)).normalize()
    val_start = test_start - pd.DateOffset(months=6)
    return {"train": (start, val_start - pd.Timedelta(days=1)),
            "validation": (val_start, test_start - pd.Timedelta(days=1)),
            "test": (test_start, end)}


def fmt(d) -> str:
    return "—" if d is None or pd.isna(d) else pd.Timestamp(d).date().isoformat()


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (pd.Timestamp, date)) or (c.endswith(("start", "usable", "LST", "UTC"))
                                                       and not isinstance(v, (int, float, str))):
                cells.append(fmt(v))
            elif isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "pudhuvai-solarscope-archive-audit"
    locations = load_locations()

    valid, rejected, grid = stage0_models(locations, session)
    log.info("valid models: %s", valid)
    grid.to_csv(OUT / "grid_cells.csv", index=False)

    coarse = stage1_coarse(locations, valid, session)
    coarse.to_csv(OUT / "coverage_by_month.csv", index=False)
    starts = coarse_start(coarse)
    log.info("coarse continuous starts: %s", starts)
    viable = {m: s for m, s in starts.items() if s is not None}

    usable_all, summaries, fc_daily, sanity = [], [], [], {}
    for model, s in viable.items():
        # begin one month before the coarse estimate to catch the exact day
        s0 = max(SCAN_START, (pd.Timestamp(s) - pd.DateOffset(months=1)).date())
        try:
            comp, hourly_sw = stage2_detail(locations, model, s0, session)
            sanity[model] = lead_sanity(locations, model, session)
        except OpenMeteoError as exc:
            log.warning("detail scan failed for %s: %s", model, exc)
            continue
        u = usable_days(comp)
        usable_all.append(u)
        summaries.append(summarise(u))
        fc_daily.append(daily_local_kwh(hourly_sw))
    usable = pd.concat(usable_all, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    fc_daily = pd.concat(fc_daily, ignore_index=True)
    usable.to_csv(OUT / "usable_days.csv", index=False)

    detail_start = min(viable.values())
    nasa_lst, nasa_latest, nasa_diff = stage3_nasa(locations, detail_start, session)
    nasa_latest.to_csv(OUT / "nasa_latency.csv", index=False)
    nasa_diff.to_csv(OUT / "nasa_lst_vs_utc.csv", index=False)

    skill = stage4_skill(fc_daily, nasa_lst)
    skill.to_csv(OUT / "raw_forecast_skill.csv", index=False)

    # per model: window common to all four regions
    nasa_end = pd.Timestamp(nasa_latest["nasa_last_valid_LST"].min())
    per_model = (summary.groupby("model")
                 .agg(continuous_start=("continuous_start", "max"),
                      last_usable=("last_usable", "min"),
                      max_gap_days=("longest_gap_days", "max"),
                      unusable_days=("unusable_days_since_start", "max"))
                 .reset_index())
    per_model["overlap_end"] = per_model["last_usable"].clip(upper=nasa_end)
    per_model["overlap_days"] = (per_model["overlap_end"] - per_model["continuous_start"]).dt.days + 1
    lead1 = (skill[skill["lead"] == 1].groupby("model")[["mae", "bias", "corr"]].mean()
             .add_prefix("lead1_mean_").reset_index())
    per_model = per_model.merge(lead1, on="model", how="left")
    per_model["lead_day1_differs_from_day0"] = per_model["model"].map(sanity)
    per_model = per_model.sort_values(["overlap_days", "lead1_mean_mae"], ascending=[False, True])
    summary = summary.merge(per_model[["model"]], on="model")
    summary.to_csv(OUT / "coverage_summary.csv", index=False)
    per_model.to_csv(OUT / "model_overview.csv", index=False)

    # recommended: >= 2 years of overlap, best lead-1 MAE, excluding models
    # whose identity/continuity is not stable for live inference
    pinned = per_model[~per_model["model"].isin(NOT_PINNABLE)
                       & (per_model["overlap_days"] >= 730)]
    rec = (pinned.sort_values("lead1_mean_mae").iloc[0] if not pinned.empty
           else per_model.iloc[0])
    split = propose_split(rec["continuous_start"], rec["overlap_end"])
    n_days = {k: (e - s).days + 1 for k, (s, e) in split.items()}

    write_report(rejected, grid, per_model, summary, nasa_latest, nasa_diff, skill,
                 rec, split, n_days)
    log.info("report written to %s", OUT / "AUDIT_REPORT.md")


def write_report(rejected, grid, per_model, summary, nasa_latest, nasa_diff, skill,
                 rec, split, n_days):
    L = []
    L.append(f"# Open-Meteo archive coverage audit\n\nGenerated {date.today().isoformat()} "
             f"by `scripts/audit_archive_coverage.py`. Scan range {SCAN_START} → {LAST_DAY}.\n")
    L.append("A day is **usable** for a region and model when all 24 UTC hours are "
             f"present for every variable ({', '.join(VARIABLES)}) at every lead (1, 2, 3). "
             f"**Continuous start** is the first month after which every month is ≥{COVERAGE_OK:.0%} "
             "usable for that region.\n")
    if rejected:
        L.append("Rejected model identifiers: " +
                 ", ".join(f"`{m}` ({r})" for m, r in rejected.items()) + "\n")

    L.append("## 1. Model overview (window common to all four regions)\n")
    L.append(md_table(per_model))
    L.append("\n`lead_day1_differs_from_day0` = share of daylight hours where the day-1 archived "
             "value differs from the latest run. Values near 0 would mean the archive is not "
             "returning genuine older forecasts.\n")

    L.append("## 2. Coverage per region and model\n")
    L.append(md_table(summary))

    L.append("\n## 3. Grid cells used by Open-Meteo\n")
    L.append(md_table(grid))

    L.append("\n## 4. NASA POWER label availability\n")
    L.append(md_table(nasa_latest))
    L.append("\nDifference between NASA daily totals using LST vs UTC day definitions:\n")
    L.append(md_table(nasa_diff, "{:.3f}"))

    L.append("\n## 5. Raw forecast skill (forecast shortwave vs NASA LST, kWh/m²/day)\n")
    L.append("Indicative only: computed over the whole overlap, not a held-out test.\n")
    piv = skill.pivot_table(index=["model", "region_id"], columns="lead", values="mae").reset_index()
    piv.columns = [f"MAE lead {c}" if isinstance(c, (int, np.integer)) else c for c in piv.columns]
    L.append(md_table(piv))

    L.append("\n## 6. Recommendation and proposed split\n")
    L.append(f"- **Pinned forecast model:** `{rec['model']}` "
             f"(usable {fmt(rec['continuous_start'])} → {fmt(rec['overlap_end'])}, "
             f"{int(rec['overlap_days'])} days; mean lead-1 MAE {rec['lead1_mean_mae']:.2f}).")
    for k, (s, e) in split.items():
        L.append(f"- **{k}:** {fmt(s)} → {fmt(e)} ({n_days[k]} days ≈ "
                 f"{n_days[k] * 4 * 3:,} rows = days × 4 regions × 3 leads)")
    L.append("- Test spans 12 months so both monsoons are represented for every region; "
             "validation 6 months; the remainder is training.")
    L.append("\n## 7. Caveats\n")
    for m, why in NOT_PINNABLE.items():
        L.append(f"- `{m}` excluded from the recommendation: {why}.")
    L.append("- Unusable (gap) days inside the window are excluded from training/evaluation, "
             "not interpolated.")
    L.append("- `_previous_dayN` is defined per **hour** (value predicted N×24 h before that hour). "
             "A daily 'lead 1' total is therefore assembled from several runs. The live inference "
             "features must be built to mimic this (decide in preprocessing).")
    L.append("- Local-day aggregation used here: UTC + 05:30. Both UTC and IST day boundaries fall at night.")
    (OUT / "AUDIT_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
