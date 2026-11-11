"""Module 8: Pudhuvai SolarScope dashboard.

Run:  .venv/Scripts/streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from src.geospatial.map_data import (CATEGORY_COLOR, CATEGORY_LABEL, build_map,  # noqa: E402
                                     load_layer, region_from_click, taluks_of)
from src.models.predict import SolarPredictor  # noqa: E402
from src.services import live_forecast  # noqa: E402
from src.services.live_forecast import LiveForecastError  # noqa: E402
from src.viz import GRID, INK, INK_2, NEUTRAL, REGION_COLOR, REGION_LABEL, REGION_ORDER, SURFACE  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
CATEGORY_ICON = {"low": "☁️", "moderate": "⛅", "high": "☀️", None: "⚠️"}
UNIT = "kWh/m²/day"

st.set_page_config(page_title="Pudhuvai SolarScope", page_icon="☀️", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner=False)
def get_predictor() -> SolarPredictor:
    return SolarPredictor()


@st.cache_data(ttl=30 * 60, show_spinner="Fetching the latest Open-Meteo forecast…")
def get_predictions():
    res = get_predictor().predict(log_path=ROOT / "data" / "predictions" / "prediction_log.csv")
    return res.predictions, res.warnings, res.meta


@st.cache_data(show_spinner=False)
def load_static():
    nasa = pd.read_csv(ROOT / "data" / "processed" / "nasa_solar_history.csv", parse_dates=["date"])
    test = pd.read_csv(ROOT / "results" / "models" / "test_metrics_v2_no_cloud_cover.csv")
    meta = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "config" / "pipeline.json").read_text(encoding="utf-8"))
    return nasa, test, meta, cfg


def style_fig(fig: go.Figure, height: int = 340, ytitle: str = UNIT) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=78, b=10), paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, font=dict(color=INK_2, size=12),
                      title=dict(font=dict(color=INK, size=14), y=0.98, yanchor="top", x=0.01),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
                      hoverlabel=dict(bgcolor="white"), yaxis_title=ytitle)
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7")
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def fmt_ist(iso: str | None) -> str:
    if not iso:
        return "unknown"
    return datetime.fromisoformat(iso).astimezone(IST).strftime("%d %b %Y, %H:%M IST")


# ------------------------------------------------------------------ state
ss = st.session_state
# selectbox values use sentinels ("all" / "region") because Streamlit shows a
# None option as an empty placeholder
ss.setdefault("region_select", "all")
ss.setdefault("taluk_select", "region")
ss.setdefault("last_click", None)
# a map click is applied here, before the selectbox widgets are drawn
if "pending_selection" in ss:
    rid, tid = ss.pop("pending_selection")
    ss.region_select, ss.taluk_select = rid or "all", tid or "region"

nasa, test_metrics, model_meta, cfg = load_static()
regions = {f["properties"]["region_id"]: f["properties"] for f in load_layer("regions")["features"]}

error = None
try:
    preds, warnings, meta = get_predictions()
except LiveForecastError as exc:
    preds, warnings, meta, error = pd.DataFrame(), [], {}, str(exc)

tomorrow = preds[preds["lead_days"] == 1].set_index("region_id") if len(preds) else pd.DataFrame()
categories = tomorrow["category"].to_dict() if len(tomorrow) else {}
tomorrow_vals = tomorrow["ml_prediction"].dropna().to_dict() if len(tomorrow) else {}

# ------------------------------------------------------------------ header
head_l, head_r = st.columns([4, 1])
with head_l:
    st.title("☀️ Pudhuvai SolarScope")
    st.caption("Three-day solar-radiation outlook for the four regions of the Union Territory of "
               "Puducherry: an ML-corrected weather forecast, for education and decision support.")
with head_r:
    st.write("")
    if st.button("↻ Refresh forecast", width="stretch"):
        get_predictions.clear()
        live_forecast.clear_cache()
        st.rerun()

if error:
    st.error(f"**Live forecast unavailable.** {error} No predictions are shown rather than out-of-date "
             "values. Historical data below is still available.", icon="🚫")

# ------------------------------------------------------------------ selectors + map
col_map, col_panel = st.columns([3, 2], gap="large")

with col_map:
    s1, s2 = st.columns(2)

    def _region_changed():
        ss.taluk_select = "region"

    s1.selectbox("Region", ["all", *REGION_ORDER], key="region_select", on_change=_region_changed,
                 format_func=lambda r: "All regions (overview)" if r == "all" else REGION_LABEL[r])
    sel_region = None if ss.region_select == "all" else ss.region_select
    taluk_names = {t["subdivision_id"]: t["subdivision_name"] for t in taluks_of(sel_region)} if sel_region else {}
    if ss.taluk_select not in ["region", *taluk_names]:
        ss.taluk_select = "region"
    s2.selectbox("Taluk", ["region", *taluk_names], key="taluk_select", disabled=not sel_region,
                 format_func=lambda t: "Whole region" if t == "region" else taluk_names[t])
    sel_taluk = None if ss.taluk_select == "region" else ss.taluk_select

    fmap = build_map(sel_region, categories, tomorrow_vals, sel_taluk)
    out = st_folium(fmap, height=470, use_container_width=True, key=f"map_{sel_region}",
                    returned_objects=["last_active_drawing", "last_object_clicked"])
    sig = json.dumps([((out or {}).get("last_active_drawing") or {}).get("properties"),
                      (out or {}).get("last_object_clicked")], default=str)
    if out and sig != ss.last_click:
        ss.last_click = sig
        rid, tid = region_from_click(out, sel_region)
        if rid and (rid, tid) != (sel_region, sel_taluk):
            if rid != sel_region:
                tid = None if sel_region is None else tid
            ss.pending_selection = (rid, tid)
            st.rerun()
    st.caption("Click a region marker or outline to open it; in a region, click a taluk. "
               "Colour shows tomorrow's availability category.")

# ------------------------------------------------------------------ forecast panel
with col_panel:
    if error:
        st.info("Forecast cards will appear here once the live forecast is available.")
    elif sel_region is None:
        st.subheader("Tomorrow at a glance")
        for rid in REGION_ORDER:
            if rid not in tomorrow.index:
                continue
            r = tomorrow.loc[rid]
            with st.container(border=True):
                a, b = st.columns([3, 2])
                a.markdown(f"**{REGION_LABEL[rid]}**  \n{pd.Timestamp(r['target_date']):%a %d %b}")
                if pd.notna(r["ml_prediction"]):
                    b.markdown(f"### {r['ml_prediction']:.1f}")
                    b.caption(f"{CATEGORY_ICON[r['category']]} {CATEGORY_LABEL[r['category']]} · {UNIT}")
                else:
                    b.markdown("**Unavailable**")
        st.caption("Select a region for the full three-day outlook.")
    else:
        rid = sel_region
        name = REGION_LABEL[rid]
        if sel_taluk:
            st.subheader(f"{taluk_names[sel_taluk]} taluk · {name}")
            st.info(f"All taluks of {name} share one forecast and one NASA grid cell (the region's "
                    "representative point), so these are the region's values, not a separate local "
                    "estimate for this taluk.", icon="ℹ️")
        else:
            st.subheader(f"{name}: next three days")
        rows = preds[preds["region_id"] == rid].sort_values("lead_days")
        for _, r in rows.iterrows():
            with st.container(border=True):
                a, b = st.columns([3, 2])
                a.markdown(f"**{pd.Timestamp(r['target_date']):%A %d %b}**  \nDay {r['lead_days']} ahead")
                if pd.notna(r["ml_prediction"]):
                    b.markdown(f"### {r['ml_prediction']:.2f}")
                    b.caption(UNIT)
                    a.markdown(f"{CATEGORY_ICON[r['category']]} **{CATEGORY_LABEL[r['category']]}** availability  \n"
                               f"Raw ECMWF forecast: {r['raw_forecast']:.2f}")
                    if r["status"] != "ok":
                        a.warning(f"Check inputs: {r['notes']}", icon="⚠️")
                else:
                    b.markdown("**Unavailable**")
                    a.caption(r["notes"])
        t = regions[rid]
        st.caption(f"Representative point {t['representative_lat']:.2f}°N, {t['representative_lon']:.2f}°E · "
                   f"NASA cell {t['nasa_solar_cell']} · taluks: {', '.join(t['taluks'])}")

if not error and meta:
    st.caption(f"Forecast model **{meta['forecast_model']}**, run initialised {fmt_ist(meta['run_initialisation_utc'])} · "
               f"retrieved {fmt_ist(meta['retrieved_utc'])} · ML model {meta['model']} ({meta['model_version']}), "
               f"trained to {model_meta['trained_on']['to']}")
for w in warnings:
    st.warning(w, icon="⚠️")

st.divider()

# ------------------------------------------------------------------ tabs
focus = sel_region
tab_cmp, tab_hist, tab_acc, tab_about = st.tabs(["Compare regions", "Historical patterns", "Forecast accuracy", "About the data"])

with tab_cmp:
    if len(preds):
        fig = go.Figure()
        for rid in REGION_ORDER:
            d = preds[preds["region_id"] == rid].sort_values("lead_days")
            fig.add_bar(name=REGION_LABEL[rid], x=[f"{pd.Timestamp(x):%a %d %b}" for x in d["target_date"]],
                        y=d["ml_prediction"], marker_color=REGION_COLOR[rid], marker_line_color=SURFACE,
                        marker_line_width=2, customdata=d[["raw_forecast", "category"]].to_numpy(),
                        hovertemplate="%{x}<br>" + REGION_LABEL[rid] + ": %{y:.2f} " + UNIT +
                                      "<br>raw forecast %{customdata[0]:.2f}<br>%{customdata[1]}<extra></extra>")
        fig.add_hline(y=cfg["categories"]["low_below"], line_dash="dot", line_color=NEUTRAL,
                      annotation_text="low / moderate", annotation_position="top left")
        fig.add_hline(y=cfg["categories"]["high_from"], line_dash="dot", line_color=NEUTRAL,
                      annotation_text="moderate / high", annotation_position="top left")
        st.plotly_chart(style_fig(fig.update_layout(barmode="group", title="Predicted solar radiation by region and day")),
                        width="stretch")
        tbl = preds.assign(region=preds["region_id"].map(REGION_LABEL)).pivot(
            index="region", columns="target_date", values="ml_prediction")
        tbl = tbl.reindex([REGION_LABEL[r] for r in REGION_ORDER])[sorted(tbl.columns)]
        tbl.columns = [f"{pd.Timestamp(c):%a %d %b}" for c in tbl.columns]
        st.dataframe(tbl.round(2), width="stretch")
    else:
        st.info("No live predictions to compare.")

with tab_hist:
    hist = nasa[nasa["date"].dt.year <= 2023].assign(month=lambda d: d["date"].dt.month)
    monthly = hist.groupby(["month", "region_id"])["ALLSKY_SFC_SW_DWN"].mean().unstack()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure()
    for rid in REGION_ORDER:
        sel = focus is None or rid == focus
        fig.add_scatter(x=months, y=monthly[rid], name=REGION_LABEL[rid], mode="lines+markers",
                        line=dict(color=REGION_COLOR[rid], width=3 if sel else 1.5),
                        opacity=1 if sel else 0.35, hovertemplate="%{x}: %{y:.2f} " + UNIT + "<extra>" + REGION_LABEL[rid] + "</extra>")
    st.plotly_chart(style_fig(fig.update_layout(title="Average daily solar radiation by month, 2001–2023 (NASA POWER)")),
                    width="stretch")
    st.caption("Each region has its own low-sun season: Puducherry and Karaikal in Nov–Dec (north-east monsoon), "
               "Mahe in Jun–Jul (south-west monsoon), Yanam from June to December.")

    rid = focus or "puducherry"
    recent = nasa[(nasa["region_id"] == rid) & (nasa["date"] > nasa["date"].max() - pd.Timedelta(days=365))]
    clim = pd.read_csv(ROOT / "data" / "processed" / "climatology.csv")
    clim = clim[clim["region_id"] == rid].set_index("doy")["climatology_solar_radiation"]
    fig = go.Figure()
    fig.add_scatter(x=recent["date"], y=recent["ALLSKY_SFC_SW_DWN"], name="Daily value", mode="lines",
                    line=dict(color="#86b6ef", width=1), hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra></extra>")
    fig.add_scatter(x=recent["date"], y=recent["ALLSKY_SFC_SW_DWN"].rolling(30, center=True, min_periods=10).mean(),
                    name="30-day average", mode="lines", line=dict(color=REGION_COLOR["puducherry"], width=2.5),
                    hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra>30-day average</extra>")
    fig.add_scatter(x=recent["date"], y=recent["date"].dt.day_of_year.clip(upper=365).map(clim), name="2001–2023 normal",
                    mode="lines", line=dict(color=NEUTRAL, width=2, dash="dash"),
                    hovertemplate="%{x|%d %b}: %{y:.2f}<extra>normal</extra>")
    st.plotly_chart(style_fig(fig.update_layout(title=f"{REGION_LABEL[rid]}: last 12 months of NASA POWER data")),
                    width="stretch")
    if focus is None:
        st.caption("Showing Puducherry; select a region to change.")

with tab_acc:
    st.markdown("Accuracy on a **held-out test year** (Nov 2025 – Nov 2026), measured with the same configuration trained "
                "only on earlier data, compared with the raw ECMWF forecast. Lower is better.")
    t = test_metrics[test_metrics["model"].isin(["ml_model", "raw_forecast"])]
    c1, c2 = st.columns(2)
    for col, scope, title, labeller in [(c1, "lead", "Mean absolute error by days ahead", lambda g: f"Day {g}"),
                                        (c2, "region", "Mean absolute error by region", lambda g: REGION_LABEL.get(g, g))]:
        d = t[t["scope"] == scope]
        fig = go.Figure()
        for model, colr, label in [("ml_model", REGION_COLOR["puducherry"], "SolarScope ML model"),
                                   ("raw_forecast", REGION_COLOR["karaikal"], "Raw ECMWF forecast")]:
            dd = d[d["model"] == model]
            fig.add_bar(x=[labeller(g) for g in dd["group"]], y=dd["mae"], name=label, marker_color=colr,
                        marker_line_color=SURFACE, marker_line_width=2, text=dd["mae"].round(2), textposition="outside",
                        hovertemplate="%{x}: MAE %{y:.3f}<extra>" + label + "</extra>")
        col.plotly_chart(style_fig(fig.update_layout(barmode="group", title=title), 320, "MAE (" + UNIT + ")"),
                         width="stretch")
    o = t[t["scope"] == "overall"].set_index("model")
    st.caption(f"Overall test MAE {o.loc['ml_model', 'mae']:.2f} vs {o.loc['raw_forecast', 'mae']:.2f} for the raw forecast "
               f"({100 * (1 - o.loc['ml_model', 'mae'] / o.loc['raw_forecast', 'mae']):.0f}% lower). Errors are largest in the monsoon seasons.")

with tab_about:
    st.markdown(f"""
**What is predicted.** Daily total solar radiation reaching a horizontal surface (global horizontal irradiation) in **{UNIT}** for the next three days.
The prediction is an ML correction of the ECMWF IFS 0.25° forecast (via Open-Meteo), trained against NASA POWER satellite estimates.

**Availability categories** are a project-defined rule, not an official classification:
low < {cfg['categories']['low_below']}, moderate {cfg['categories']['low_below']}–{cfg['categories']['high_from']},
high ≥ {cfg['categories']['high_from']} {UNIT} (terciles of NASA daily values 2001–2023, all four regions).

**Spatial resolution.** Each region is represented by one point. NASA POWER solar data is on a ~1° grid (~110 km) and the forecast on a 0.25° grid,
so **all taluks of a region share the same values**. These are not street- or rooftop-level measurements.

**Limitations.** The model's typical error is about ±0.6 {UNIT} (larger in the monsoon). Forecast inputs can change behaviour when the
weather model is upgraded; the app checks inputs against the training data and warns when they look unusual. Estimates are indicative only
and do not guarantee the output of any solar installation. The Yanam outline in the boundary data covers ~13 km², less than the
officially cited ~30 km²; this affects only the map, not the predictions.

**Data sources and attribution**
- Solar radiation history and training labels: [NASA POWER](https://power.larc.nasa.gov/) (NASA Langley Research Center POWER Project).
- Weather forecasts: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0), ECMWF IFS model. Open-Meteo's free API is for non-commercial use.
- Boundaries: {load_layer('regions')['metadata']['attribution']}.
- Base map: © OpenStreetMap contributors.
""")

st.caption("Pudhuvai SolarScope is an educational decision-support tool. Predictions are estimates from gridded data and models, "
           "not local measurements, and do not guarantee rooftop electricity generation.")
