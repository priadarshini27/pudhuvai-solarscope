import json
import sys
from pathlib import Path

import pytest
from shapely.geometry import Point, shape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geospatial.map_data import build_map, legend_html, load_layer, region_from_click, taluks_of  # noqa: E402

LOCATIONS = {l["region_id"]: l for l in json.loads((ROOT / "config" / "locations.json").read_text(encoding="utf-8"))["regions"]}
EXPECTED_TALUKS = {"puducherry": {"Puducherry", "Ozhukarai", "Villianur", "Bahour"},
                   "karaikal": {"Karaikal", "Thirunallar"}, "mahe": {"Mahe"}, "yanam": {"Yanam"}}


def test_regions_layer_has_four_regions_with_points_inside():
    feats = load_layer("regions")["features"]
    assert {f["properties"]["region_id"] for f in feats} == set(EXPECTED_TALUKS)
    for f in feats:
        loc = LOCATIONS[f["properties"]["region_id"]]
        assert shape(f["geometry"]).contains(Point(loc["longitude"], loc["latitude"]))


def test_taluks_are_the_official_eight_and_inside_their_region():
    regions = {f["properties"]["region_id"]: shape(f["geometry"]) for f in load_layer("regions")["features"]}
    for rid, names in EXPECTED_TALUKS.items():
        assert {t["subdivision_name"] for t in taluks_of(rid)} == names
    for f in load_layer("taluks")["features"]:
        g = shape(f["geometry"])
        assert regions[f["properties"]["region_id"]].buffer(1e-6).contains(g.representative_point())
        assert f["properties"]["shares_grid_cell_with"] is not None


def test_area_close_to_official_for_main_regions():
    area = {f["properties"]["region_id"]: f["properties"]["area_km2"] for f in load_layer("regions")["features"]}
    assert area["puducherry"] == pytest.approx(293, rel=0.05)
    assert area["karaikal"] == pytest.approx(161, rel=0.05)
    assert area["mahe"] == pytest.approx(9, rel=0.15)
    assert area["yanam"] == pytest.approx(30, rel=0.15)


def test_karaikal_taluks_tile_the_region():
    region = next(shape(f["geometry"]) for f in load_layer("regions")["features"]
                  if f["properties"]["region_id"] == "karaikal")
    taluks = [shape(f["geometry"]) for f in load_layer("taluks")["features"]
              if f["properties"]["region_id"] == "karaikal"]
    assert taluks[0].intersection(taluks[1]).area < 1e-3 * region.area
    union = taluks[0].union(taluks[1])
    assert union.symmetric_difference(region).area < 1e-3 * region.area


def test_click_mapping():
    taluk = {"last_active_drawing": {"properties": {"region_id": "karaikal", "subdivision_id": "karaikal:thirunallar"}}}
    assert region_from_click(taluk, "karaikal") == ("karaikal", "karaikal:thirunallar")
    region = {"last_active_drawing": {"properties": {"region_id": "mahe"}}}
    assert region_from_click(region, None) == ("mahe", None)
    marker = {"last_active_drawing": None, "last_object_clicked": {"lat": 16.7, "lng": 82.2}}
    assert region_from_click(marker, None) == ("yanam", None)
    assert region_from_click(None, None) == (None, None)


def test_build_map_overview_and_region():
    html = build_map(None, {"mahe": "high"}, {"mahe": 6.2}).get_root().render()
    assert "Mahe: 6.2" in html and "Boundaries: OpenStreetMap" in html
    html = build_map("puducherry", {"puducherry": "low"}, {}, "puducherry:ozhukarai").get_root().render()
    assert "Ozhukarai" in html


def test_map_legend_shows_categories_and_ranges():
    html = legend_html((5.0, 6.0), show_unavailable=False)
    for text in ["solar availability", "High", "Moderate", "Low", "≥ 6", "5–6", "&lt; 5"]:
        assert text in html, text
    assert "Unavailable" not in html
    assert "Unavailable" in legend_html((5.0, 6.0), show_unavailable=True)
    # attached to the map; a region without a forecast adds the Unavailable row
    full = {"puducherry": "high", "karaikal": "moderate", "mahe": "low", "yanam": "low"}
    assert "L.control" in build_map(None, full, {}, thresholds=(5.0, 6.0)).get_root().render()
    assert "Unavailable" not in build_map(None, full, {}).get_root().render()
    assert "Unavailable" in build_map(None, {"puducherry": "high"}, {}).get_root().render()
