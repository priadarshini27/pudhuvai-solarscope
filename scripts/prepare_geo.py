"""Module 1: build the UT region and taluk boundary layers for the map.

Source: geoBoundaries gbOpen IND ADM3 (sub-districts = taluks), release commit
9469f09, derived from the Local Government Directory (lgdirectory.gov.in) by
Pathways Data Pvt. Ltd.; licence ODbL 1.0 (attribution + share-alike).

Region outlines are built by merging each region's official taluks rather than
using the geoBoundaries ADM2 (district) layer, because that layer places
Thirunallar taluk (Karaikal region) inside its "Puducherry" district polygon.

Usage:  .venv/Scripts/python scripts/prepare_geo.py
        (expects data/raw/geo/geoBoundaries-IND-ADM3.geojson)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "geo" / "geoBoundaries-IND-ADM3.geojson"
OUT = ROOT / "data" / "geo"
SOURCE_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/"
              "IND/ADM3/geoBoundaries-IND-ADM3.geojson")

# official taluks of the UT (shapeName in the source -> region, display name)
TALUKS = {
    "Puducherry Taluk": ("puducherry", "Puducherry"),
    "Ozhukarai Taluk": ("puducherry", "Ozhukarai"),
    "Villianur Taluk": ("puducherry", "Villianur"),
    "Bahour Taluk": ("puducherry", "Bahour"),
    "Karaikal Taluk": ("karaikal", "Karaikal"),
    "Thirunallar Taluk": ("karaikal", "Thirunallar"),
    "Mahe Taluk": ("mahe", "Mahe"),
    "Yanam Taluk": ("yanam", "Yanam"),
}
SIMPLIFY_DEG = 0.0002          # ~20 m; keeps tiny Mahe/Yanam shapes intact
ATTRIBUTION = ("Boundaries: geoBoundaries (gbOpen IND ADM3), from the Local Government Directory "
               "via Pathways Data Pvt. Ltd., ODbL 1.0")


def area_km2(geom) -> float:
    """Approximate area using a local equirectangular projection."""
    lat = geom.centroid.y
    kx, ky = 111.32 * math.cos(math.radians(lat)), 110.57
    from shapely.affinity import scale
    return scale(geom, xfact=kx, yfact=ky, origin=(0, 0)).area


def nasa_cell(lat: float, lon: float) -> str:
    return f"{math.floor(lat)}°N {math.floor(lon)}°E"


def main():
    if not RAW.exists():
        sys.exit(f"missing {RAW}; download it from {SOURCE_URL}")
    locations = {l["region_id"]: l for l in json.loads((ROOT / "config" / "locations.json")
                                                        .read_text(encoding="utf-8"))["regions"]}
    src = json.loads(RAW.read_text(encoding="utf-8"))

    found = {}
    for f in src["features"]:
        name = f["properties"]["shapeName"]
        if name in TALUKS:
            if name in found:
                sys.exit(f"duplicate feature for {name}")
            found[name] = (shape(f["geometry"]), f["properties"]["shapeID"])
    missing = set(TALUKS) - set(found)
    if missing:
        sys.exit(f"taluks not found in source: {sorted(missing)}")

    taluk_features, by_region = [], {}
    for src_name, (geom, shape_id) in found.items():
        region_id, name = TALUKS[src_name]
        geom = geom.buffer(0).simplify(SIMPLIFY_DEG, preserve_topology=True)
        by_region.setdefault(region_id, []).append((name, geom))
        c = geom.representative_point()
        taluk_features.append({"type": "Feature", "geometry": mapping(geom), "properties": {
            "subdivision_id": f"{region_id}:{name.lower()}", "subdivision_name": name,
            "subdivision_type": "taluk", "region_id": region_id,
            "region_name": locations[region_id]["name"], "area_km2": round(area_km2(geom), 1),
            "nasa_solar_cell_of_taluk": nasa_cell(c.y, c.x),
            "source_shape_id": shape_id,
        }})

    region_features, checks = [], []
    for region_id, parts in by_region.items():
        loc = locations[region_id]
        geom = unary_union([g for _, g in parts])
        pt = Point(loc["longitude"], loc["latitude"])
        dist_km = 0.0 if geom.contains(pt) else geom.distance(pt) * 111
        names = sorted(n for n, _ in parts)
        region_features.append({"type": "Feature", "geometry": mapping(geom), "properties": {
            "region_id": region_id, "region_name": loc["name"], "taluks": names,
            "area_km2": round(area_km2(geom), 1),
            "representative_lat": loc["latitude"], "representative_lon": loc["longitude"],
            "representative_point_inside": bool(geom.contains(pt)),
            "nasa_solar_cell": nasa_cell(loc["latitude"], loc["longitude"]),
        }})
        checks.append((loc["name"], names, round(area_km2(geom), 1), geom.contains(pt), round(dist_km, 2)))

    # every taluk is served by its region's representative point: record that
    # all taluks of a region share one forecast/NASA grid cell (doc Module 1)
    for f in taluk_features:
        p = f["properties"]
        others = [n for n, _ in by_region[p["region_id"]] if n != p["subdivision_name"]]
        p["data_source_point"] = f"{p['region_name']} representative point"
        p["shares_grid_cell_with"] = others
        if p["nasa_solar_cell_of_taluk"] != nasa_cell(locations[p["region_id"]]["latitude"],
                                                       locations[p["region_id"]]["longitude"]):
            p["note"] = "taluk lies in a different NASA 1° cell than the region's representative point"

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"source": SOURCE_URL, "licence": "ODbL 1.0", "attribution": ATTRIBUTION,
            "processing": ["selected the 8 official UT taluks by name",
                           f"geometry repaired (buffer 0) and simplified ({SIMPLIFY_DEG} deg)",
                           "region outlines = union of their taluks (ADM2 layer not used: "
                           "it puts Thirunallar taluk inside its 'Puducherry' district)"]}
    for name, feats in [("puducherry_regions.geojson", region_features),
                        ("administrative_subdivisions.geojson", taluk_features)]:
        (OUT / name).write_text(json.dumps({"type": "FeatureCollection", "metadata": meta,
                                            "features": feats}, separators=(",", ":")), encoding="utf-8")
    (OUT / "SOURCE.md").write_text(
        "# Boundary data\n\n"
        f"- Source: {SOURCE_URL}\n- Licence: ODbL 1.0. These derived files are also ODbL 1.0.\n"
        f"- Attribution: {ATTRIBUTION}\n- Processing: " + "; ".join(meta["processing"]) + "\n"
        "- Built by `scripts/prepare_geo.py`.\n", encoding="utf-8")

    print(f"{'region':12}{'area km2':>9}  rep. point inside?  taluks")
    for name, names, area, inside, dist in checks:
        print(f"{name:12}{area:9.1f}  {'yes' if inside else f'NO ({dist} km away)':18}  {', '.join(names)}")
    for f in taluk_features:
        if "note" in f["properties"]:
            print("note:", f["properties"]["subdivision_name"], "-", f["properties"]["note"])
    print("written:", ", ".join(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
