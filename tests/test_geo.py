import json
import sys
from pathlib import Path

import pytest
from shapely.geometry import Point, shape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geospatial.map_data import build_map, load_layer, region_from_click, taluks_of  # noqa: E402

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
    assert "Mahe: 6.2" in html and "geoBoundaries" in html
    html = build_map("puducherry", {"puducherry": "low"}, {}, "puducherry:ozhukarai").get_root().render()
    assert "Ozhukarai" in html
