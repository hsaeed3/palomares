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
# package M: Mapillary survey (needs the token; downloads only the sampled thumb_1024 images)
nice -n5 python3 09_mapillary_features.py # Mapillary map features -> mapillary_features.csv
nice -n5 python3 10_mapillary_images.py   # sampled frames -> data/raw/dev7/mapillary/img (resumable)
nice -n5 python3 12_catalog_dev7.py       # dev7 asset types in config/asset_catalog.yaml
nice -n5 python3 11_mapillary_ledger.py   # ledger_parts/dev7_*.jsonl + reports/dev7/mapillary_attribution.csv
nice -n5 python3 ../p5/ledger_merge.py    # -> data/scene/ledger.jsonl (Palomares) + ledger_dev7.jsonl
nice -n5 python3 13_survey_report.py      # reports/dev7/survey.html
