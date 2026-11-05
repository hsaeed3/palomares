#!/usr/bin/env bash
# Rebuild Phase 2 from the Phase 1 data/derived package (+ CC0 texture download).
set -euo pipefail
cd "$(dirname "$0")"
python3 19_gopro_calibrate.py      # GoPro spring frames -> dry/wet split palette + foliage crops
python3 20_fetch_textures.py      # ambientCG CC0 textures -> data/raw/textures
python3 21_build_geometry.py      # road, shoulders, terrain CDT, backdrop, trees, barriers, lines, kerbs, locators
python3 21b_track_dressing.py      # track_* dressing: gantries, timing huts, summit bleacher, corner barriers
python3 22_build_kn5.py           # kn5 files + AC folder (build/palomares_road)
python3 23_qa.py                  # independent QA -> reports/qa_phase2.md
python3 24_visual_export.py       # round-trip kn5 -> render scene
node 24_visual_render.js road_summit
node 24_visual_render.js track_summit
python3 26_contact_sheet.py       # reports/visual_check.html
python3 25_package.py             # dist/palomares_road_vX.Y.Z.zip
