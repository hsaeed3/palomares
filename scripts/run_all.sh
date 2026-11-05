#!/usr/bin/env bash
# Rebuild Phase 1 from scratch (needs network). Each step is idempotent and caches downloads under data/raw, data/cache.
set -euo pipefail
cd "$(dirname "$0")"
python3 01_osm_centerline.py      # OSM ways -> joined S->N line, fixes config/origin.json (once)
python3 01b_osm_context.py        # OSM context (junctions, driveways, streams, bridges)
python3 02_fetch_lidar.py         # USGS 3DEP EPT (EPSG:3857) -> 150 m corridor, reprojected to UTM 10N
python3 03_chunk_points.py        # re-bin into 250 m station chunks
python3 04_dtm.py                 # DTMs 0.5 m / 1 m, DSM, CHM, ground-distance raster
python3 05a_straight_rasters.py   # road-aligned diagnostic rasters along OSM
python3 05b_fetch_imagery.py      # USGS imagery 1 m + 3DEP 10 m backdrop DEM
python3 06_refine_centerline.py   # snap centerline to the real road surface
python3 07_cross_sections.py      # per-metre width/edge environment -> centerline.csv/geojson
python3 07b_surface_fit.py        # high-precision z/camber/grade from raw ground returns
python3 08_features.py            # features, gaps, markings, trees
python3 09_viewer.py              # reports/viewer.html
python3 10_report.py              # reports/phase1_report.md
python3 11_pick_corners.py        # assets/corners.json
node 12_screens.js                # assets/*.jpg (headless Chromium + SwiftShader)
