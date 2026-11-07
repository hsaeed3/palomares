#!/usr/bin/env bash
# dev7 work package D (data). Idempotent; downloads cached under data/raw/dev7, data/cache/p7.
set -euo pipefail
cd "$(dirname "$0")"
nice -n5 python3 01_osm_network.py        # OSM network, junctions, termini, scope polygons
nice -n5 python3 02_fetch_lidar.py        # stream 2021 Alameda EPT -> reduced nodes (no raw LAZ kept)
nice -n5 python3 03_dtm.py                # DTM 0.5/1 m, DSM, CHM, gdist, intensity, cover
nice -n5 python3 04_centerlines.py        # refined centerlines + cross-sections per road
nice -n5 python3 05_features.py           # features, gaps, trees, buildings, canyon profile
nice -n5 python3 07_mapillary_index.py    # Mapillary index (needs MAPILLARY_TOKEN or MAPILARY_TOKEN)
nice -n5 python3 06_viewer.py             # reports/dev7/viewer.html
nice -n5 python3 08_report.py             # reports/dev7/data_report.md
