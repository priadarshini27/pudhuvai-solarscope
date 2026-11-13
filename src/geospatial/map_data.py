"""Module 1: geographic layers and the interactive Leaflet (folium) map."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import folium
from branca.element import MacroElement, Template

ROOT = Path(__file__).resolve().parents[2]
GEO = ROOT / "data" / "geo"

# ordinal low/moderate/high: cloud haze -> gold -> full-sun orange
CATEGORY_COLOR = {"low": "#C1ADBC", "moderate": "#FBB946", "high": "#F49005", None: "#D9C7B2"}
CATEGORY_LABEL = {"low": "Low", "moderate": "Moderate", "high": "High", None: "Unavailable"}
OUTLINE, MUTED = "#773701", "#8C7B6B"


@lru_cache(maxsize=None)
def load_layer(name: str) -> dict:
    """'regions' -> puducherry_regions.geojson, 'taluks' -> administrative_subdivisions.geojson"""
    file = {"regions": "puducherry_regions.geojson", "taluks": "administrative_subdivisions.geojson"}[name]
    path = GEO / file
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/prepare_geo.py")
    return json.loads(path.read_text(encoding="utf-8"))


def taluks_of(region_id: str) -> list[dict]:
    return [f["properties"] for f in load_layer("taluks")["features"]
            if f["properties"]["region_id"] == region_id]


def _bounds(features: list[dict]) -> list[list[float]]:
    xs, ys = [], []

    def walk(c):
        if isinstance(c[0], (int, float)):
            xs.append(c[0]); ys.append(c[1])
        else:
            for x in c:
                walk(x)
    for f in features:
        walk(f["geometry"]["coordinates"])
    return [[min(ys), min(xs)], [max(ys), max(xs)]]


def legend_html(thresholds: tuple[float, float] | None, show_unavailable: bool) -> str:
    """Category legend: swatch + label + kWh/m²/day range (when thresholds are given)."""
    if thresholds:
        lo, hi = thresholds
        ranges = {"low": f"&lt; {lo:g}", "moderate": f"{lo:g}–{hi:g}", "high": f"≥ {hi:g}", None: "no forecast"}
    else:
        ranges = {k: "" for k in CATEGORY_LABEL}
    keys = ["high", "moderate", "low"] + ([None] if show_unavailable else [])
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin-top:3px">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{CATEGORY_COLOR[k]};'
        f'border:1px solid {OUTLINE};flex:none"></span>'
        f'<span>{CATEGORY_LABEL[k]}</span>'
        f'<span style="margin-left:auto;padding-left:8px;color:{MUTED}">{ranges[k]}</span></div>'
        for k in keys)
    return (f'<div style="background:#FFFAF3;color:#2B1A0B;font:12px/1.3 sans-serif;padding:6px 9px;'
            f'border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);min-width:150px">'
            f'<div style="font-weight:600">Tomorrow\'s solar availability</div>'
            f'<div style="color:{MUTED};font-size:11px">kWh/m²/day</div>{rows}</div>')


class _Legend(MacroElement):
    """Legend as a Leaflet control, so it stays on the map in every view."""
    _template = Template("""
        {% macro script(this, kwargs) %}
        var {{ this.get_name() }} = L.control({position: "bottomright"});
        {{ this.get_name() }}.onAdd = function () {
            var div = L.DomUtil.create("div", "solar-legend");
            div.innerHTML = {{ this.html|tojson }};
            return div;
        };
        {{ this.get_name() }}.addTo({{ this._parent.get_name() }});
        {% endmacro %}
    """)

    def __init__(self, html: str):
        super().__init__()
        self._name = "Legend"
        self.html = html


def build_map(selected_region: str | None, categories: dict[str, str | None],
              tomorrow_values: dict[str, float], selected_taluk: str | None = None,
              thresholds: tuple[float, float] | None = None) -> folium.Map:
    """Overview of the four regions (markers + outlines), or a zoomed region view
    with its taluks. Clicking a feature returns its properties via st_folium.
    `thresholds` = (low_below, high_from) adds the value ranges to the legend."""
    regions = load_layer("regions")["features"]
    m = folium.Map(tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer("OpenStreetMap", name="Base map", control=False, opacity=0.7,
                     attr="© OpenStreetMap contributors | Boundaries: OpenStreetMap, "
                          "Karaikal taluks via geoBoundaries / LGD (ODbL 1.0)").add_to(m)
    # smaller credit line, so it fits on one line and leaves the scale bar visible
    m.get_root().header.add_child(folium.Element(
        "<style>.leaflet-control-attribution{font-size:9px;line-height:1.3;}</style>"))

    def region_style(feature):
        rid = feature["properties"]["region_id"]
        sel = rid == selected_region
        # in a region view the taluks carry the fill; the region is outline only
        return {"fillColor": CATEGORY_COLOR[categories.get(rid)],
                "fillOpacity": 0.0 if sel else (0.25 if selected_region else 0.55),
                "color": OUTLINE, "weight": 2.5 if sel else 1}

    folium.GeoJson(
        load_layer("regions"), name="Regions", style_function=region_style,
        highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.7},
        tooltip=folium.GeoJsonTooltip(fields=["region_name", "area_km2"], aliases=["Region", "Area (km²)"]),
    ).add_to(m)

    if selected_region:
        feats = [f for f in load_layer("taluks")["features"] if f["properties"]["region_id"] == selected_region]
        folium.GeoJson(
            {"type": "FeatureCollection", "features": feats}, name="Taluks",
            style_function=lambda f: (
                {"fillColor": CATEGORY_COLOR[categories.get(selected_region)], "fillOpacity": 0.8,
                 "color": "#2B1A0B", "weight": 3}
                if f["properties"]["subdivision_id"] == selected_taluk else
                {"fillColor": CATEGORY_COLOR[categories.get(selected_region)],
                 "fillOpacity": 0.15 if selected_taluk else 0.45, "color": OUTLINE, "weight": 1, "dashArray": "4 3"}),
            highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.75},
            tooltip=folium.GeoJsonTooltip(fields=["subdivision_name", "region_name", "area_km2"],
                                          aliases=["Taluk", "Region", "Area (km²)"]),
        ).add_to(m)
        m.fit_bounds(_bounds(feats), padding=(20, 20))
    else:
        # regions are tiny at UT scale: add a labelled marker for each
        for f in regions:
            p = f["properties"]
            rid = p["region_id"]
            val = tomorrow_values.get(rid)
            label = f"{p['region_name']}: {val:.1f}" if val is not None else p["region_name"]
            folium.CircleMarker(
                location=[p["representative_lat"], p["representative_lon"]], radius=11,
                color="#FFFAF3", weight=2, fill=True, fill_color=CATEGORY_COLOR[categories.get(rid)], fill_opacity=0.95,
                tooltip=f"{label} kWh/m²/day tomorrow ({CATEGORY_LABEL[categories.get(rid)]}) – click to open",
            ).add_to(m)
            folium.map.Marker(
                [p["representative_lat"], p["representative_lon"]],
                icon=folium.DivIcon(html=f'<div style="font:600 12px sans-serif;color:#2B1A0B;white-space:nowrap;'
                                         f'transform:translate(14px,-8px);text-shadow:0 0 3px #fff,0 0 3px #fff">{label}</div>'),
            ).add_to(m)
        m.fit_bounds(_bounds(regions), padding=(40, 40))
    shown = [selected_region] if selected_region else [f["properties"]["region_id"] for f in regions]
    m.add_child(_Legend(legend_html(thresholds, any(categories.get(r) is None for r in shown))))
    return m


def region_from_click(result: dict | None, current_region: str | None) -> tuple[str | None, str | None]:
    """Translate an st_folium return value into (region_id, subdivision_id)."""
    if not result:
        return None, None
    drawing = result.get("last_active_drawing") or {}
    props = drawing.get("properties") or {}
    if "subdivision_id" in props:
        return props["region_id"], props["subdivision_id"]
    if "region_id" in props:
        return props["region_id"], None
    click = result.get("last_object_clicked")
    if click and current_region is None:
        # marker click: nearest representative point
        regions = [f["properties"] for f in load_layer("regions")["features"]]
        nearest = min(regions, key=lambda p: (p["representative_lat"] - click["lat"]) ** 2
                      + (p["representative_lon"] - click["lng"]) ** 2)
        return nearest["region_id"], None
    return None, None
