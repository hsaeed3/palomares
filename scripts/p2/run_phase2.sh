#!/usr/bin/env bash
# Rebuild Phase 2 from the Phase 1 data/derived package (+ CC0 texture download).
set -euo pipefail
cd "$(dirname "$0")"
# Steps (see build.py FULL): 19 calibrate, 20 textures, 21 geometry, 21b dressing, 22 kn5, 23 QA,
# 24 visual export + renders, 26 contact sheet, 25 package.
exec python3 build.py full "$@"
