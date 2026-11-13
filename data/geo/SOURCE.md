# Boundary data

- Sources: OpenStreetMap relations via Nominatim (https://nominatim.openstreetmap.org/lookup?format=geojson&polygon_geojson=1&osm_ids=R10088305,R10088306,R10088304,R10088307,R10088317,R10088316,R1819590); geoBoundaries https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND/ADM3/geoBoundaries-IND-ADM3.geojson
- Licence: ODbL 1.0. These derived files are also ODbL 1.0.
- Attribution: Boundaries: © OpenStreetMap contributors; Karaikal/Thirunallar split from geoBoundaries (gbOpen IND ADM3, Local Government Directory via Pathways Data Pvt. Ltd.); ODbL 1.0
- Processing: Puducherry, Ozhukarai, Villianur, Bahour, Mahe and Yanam taluks = OSM taluk relations; Karaikal and Thirunallar = geoBoundaries ADM3 taluks clipped to the OSM Karaikal district (relation 1819590), gaps given to the taluk sharing the longest edge; geometry repaired (buffer 0) and simplified (0.0002 deg); region outlines = union of their taluks (OSM relation 5335580 not used: it puts Mahe and Yanam inside its 'Puducherry' district)
- Built by `scripts/prepare_geo.py`.
