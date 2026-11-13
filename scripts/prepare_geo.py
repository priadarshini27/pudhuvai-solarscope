"""Module 1: build the UT region and taluk boundary layers for the map.

Sources
- OpenStreetMap administrative relations (© OpenStreetMap contributors, ODbL 1.0),
  fetched via the Nominatim lookup API: the six taluks of Puducherry, Mahe and
  Yanam, and the Karaikal district outline. These are the same boundaries the
  OSM base tiles draw, and they match the officially cited areas.
- geoBoundaries gbOpen IND ADM3 (release 9469f09, from the Local Government
  Directory via Pathways Data Pvt. Ltd., ODbL 1.0), used only to split Karaikal
  district into its Karaikal and Thirunallar taluks, which OSM does not map.
  The two taluks are clipped to the OSM district and the uncovered slivers are
  given to the taluk they share the longest edge with.

Region outlines are the union of their taluks. The OSM "Puducherry" district
relation (5335580) is not used, because it also contains Mahe and Yanam.

Usage:  .venv/Scripts/python scripts/prepare_geo.py [--fetch]
        (expects data/raw/geo/osm_puducherry.geojson, downloaded by --fetch, and
         data/raw/geo/geoBoundaries-IND-ADM3.geojson)
"""
from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "geo"
RAW_GB = RAW_DIR / "geoBoundaries-IND-ADM3.geojson"
RAW_OSM = RAW_DIR / "osm_puducherry.geojson"
OUT = ROOT / "data" / "geo"
GB_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/"
          "IND/ADM3/geoBoundaries-IND-ADM3.geojson")

# OSM relation id -> (region, display name); taluks mapped in OSM
OSM_TALUKS = {
    10088305: ("puducherry", "Puducherry"),
    10088306: ("puducherry", "Ozhukarai"),
    10088304: ("puducherry", "Villianur"),
    10088307: ("puducherry", "Bahour"),
    10088317: ("mahe", "Mahe"),
    10088316: ("yanam", "Yanam"),
}
OSM_KARAIKAL = 1819590
# Karaikal district split, taken from geoBoundaries (shapeName -> display name)
GB_KARAIKAL_TALUKS = {"Karaikal Taluk": "Karaikal", "Thirunallar Taluk": "Thirunallar"}

OSM_URL = ("https://nominatim.openstreetmap.org/lookup?format=geojson&polygon_geojson=1&osm_ids="
           + ",".join(f"R{i}" for i in [*OSM_TALUKS, OSM_KARAIKAL]))
SIMPLIFY_DEG = 0.0002          # ~20 m; keeps tiny Mahe/Yanam shapes intact
ATTRIBUTION = ("Boundaries: © OpenStreetMap contributors; Karaikal/Thirunallar split from geoBoundaries "
               "(gbOpen IND ADM3, Local Government Directory via Pathways Data Pvt. Ltd.); ODbL 1.0")


def area_km2(geom) -> float:
    """Approximate area using a local equirectangular projection."""
    lat = geom.centroid.y
    kx, ky = 111.32 * math.cos(math.radians(lat)), 110.57
    from shapely.affinity import scale
    return scale(geom, xfact=kx, yfact=ky, origin=(0, 0)).area


def nasa_cell(lat: float, lon: float) -> str:
    return f"{math.floor(lat)}°N {math.floor(lon)}°E"


def main_cell(geom) -> str:
    """NASA 1° cell holding the largest share of the geometry's area."""
    from shapely.geometry import box
    x0, y0, x1, y1 = geom.bounds
    cells = [(geom.intersection(box(x, y, x + 1, y + 1)).area, y, x)
             for x in range(math.floor(x0), math.floor(x1) + 1)
             for y in range(math.floor(y0), math.floor(y1) + 1)]
    _, y, x = max(cells)
    return nasa_cell(y, x)


def fetch_osm():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(OSM_URL, headers={"User-Agent": "boundary-prep-script/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        RAW_OSM.write_bytes(r.read())
    print("downloaded", RAW_OSM)


def load_osm() -> dict[int, object]:
    feats = json.loads(RAW_OSM.read_text(encoding="utf-8"))["features"]
    geoms = {f["properties"]["osm_id"]: shape(f["geometry"]).buffer(0)
             for f in feats if f["properties"].get("osm_type") == "relation"}
    missing = {*OSM_TALUKS, OSM_KARAIKAL} - set(geoms)
    if missing:
        sys.exit(f"OSM relations missing from {RAW_OSM}: {sorted(missing)}; rerun with --fetch")
    return geoms


def split_karaikal(district) -> list[tuple[str, object, str]]:
    """Clip the geoBoundaries Karaikal taluks to the OSM district; assign slivers."""
    found = {}
    for f in json.loads(RAW_GB.read_text(encoding="utf-8"))["features"]:
        name = f["properties"]["shapeName"]
        if name in GB_KARAIKAL_TALUKS:
            if name in found:
                sys.exit(f"duplicate feature for {name}")
            found[name] = (shape(f["geometry"]).buffer(0), f["properties"]["shapeID"])
    if set(found) != set(GB_KARAIKAL_TALUKS):
        sys.exit(f"taluks not found in geoBoundaries: {sorted(set(GB_KARAIKAL_TALUKS) - set(found))}")

    # resolve any overlap between the two source taluks first
    (n1, (g1, _)), (n2, (g2, _)) = found.items()
    parts = {n1: g1.intersection(district), n2: g2.difference(g1).intersection(district)}
    gaps = district.difference(unary_union(list(parts.values())))
    for gap in getattr(gaps, "geoms", [gaps]):
        if gap.is_empty or gap.area == 0:
            continue
        # longest shared edge wins; nearest taluk if it touches neither
        edge = {n: gap.boundary.intersection(g.buffer(1e-7)).length for n, g in parts.items()}
        best = max(edge, key=edge.get) if max(edge.values()) > 0 else min(parts, key=lambda n: parts[n].distance(gap))
        parts[best] = unary_union([parts[best], gap])
    return [(GB_KARAIKAL_TALUKS[n], parts[n].buffer(0),
             f"geoBoundaries:{found[n][1]}+osm:relation/{OSM_KARAIKAL}") for n in parts]


def main():
    if "--fetch" in sys.argv:
        fetch_osm()
    if not RAW_OSM.exists():
        sys.exit(f"missing {RAW_OSM}; run with --fetch (downloads {OSM_URL})")
    if not RAW_GB.exists():
        sys.exit(f"missing {RAW_GB}; download it from {GB_URL}")
    locations = {l["region_id"]: l for l in json.loads((ROOT / "config" / "locations.json")
                                                        .read_text(encoding="utf-8"))["regions"]}
    osm = load_osm()

    sources = [(region_id, name, osm[rel_id], f"osm:relation/{rel_id}")
               for rel_id, (region_id, name) in OSM_TALUKS.items()]
    sources += [("karaikal", name, geom, sid) for name, geom, sid in split_karaikal(osm[OSM_KARAIKAL])]

    taluk_features, by_region = [], {}
    for region_id, name, geom, shape_id in sources:
        geom = geom.buffer(0).simplify(SIMPLIFY_DEG, preserve_topology=True)
        by_region.setdefault(region_id, []).append((name, geom))
        taluk_features.append({"type": "Feature", "geometry": mapping(geom), "properties": {
            "subdivision_id": f"{region_id}:{name.lower()}", "subdivision_name": name,
            "subdivision_type": "taluk", "region_id": region_id,
            "region_name": locations[region_id]["name"], "area_km2": round(area_km2(geom), 1),
            "nasa_solar_cell_of_taluk": main_cell(geom),
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
    meta = {"source": [OSM_URL, GB_URL], "licence": "ODbL 1.0", "attribution": ATTRIBUTION,
            "processing": ["Puducherry, Ozhukarai, Villianur, Bahour, Mahe and Yanam taluks = OSM taluk relations",
                           f"Karaikal and Thirunallar = geoBoundaries ADM3 taluks clipped to the OSM Karaikal "
                           f"district (relation {OSM_KARAIKAL}), gaps given to the taluk sharing the longest edge",
                           f"geometry repaired (buffer 0) and simplified ({SIMPLIFY_DEG} deg)",
                           "region outlines = union of their taluks (OSM relation 5335580 not used: "
                           "it puts Mahe and Yanam inside its 'Puducherry' district)"]}
    for name, feats in [("puducherry_regions.geojson", region_features),
                        ("administrative_subdivisions.geojson", taluk_features)]:
        (OUT / name).write_text(json.dumps({"type": "FeatureCollection", "metadata": meta,
                                            "features": feats}, separators=(",", ":")), encoding="utf-8")
    (OUT / "SOURCE.md").write_text(
        "# Boundary data\n\n"
        f"- Sources: OpenStreetMap relations via Nominatim ({OSM_URL}); geoBoundaries {GB_URL}\n"
        "- Licence: ODbL 1.0. These derived files are also ODbL 1.0.\n"
        f"- Attribution: {ATTRIBUTION}\n- Processing: " + "; ".join(meta["processing"]) + "\n"
        "- Built by `scripts/prepare_geo.py`.\n", encoding="utf-8")

    print(f"{'region':12}{'area km2':>9}  rep. point inside?  taluks")
    for name, names, area, inside, dist in checks:
        print(f"{name:12}{area:9.1f}  {'yes' if inside else f'NO ({dist} km away)':18}  {', '.join(names)}")
    for f in taluk_features:
        p = f["properties"]
        print(f"  {p['subdivision_name']:12}{p['area_km2']:7.1f} km2  {p['source_shape_id']}"
              + (f"  note: {p['note']}" if "note" in p else ""))
    print("written:", ", ".join(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
