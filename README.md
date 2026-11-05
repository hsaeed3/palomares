# Palomares Road → Assetto Corsa hill climb

Status: **Phase 2 in progress** (drivable SR-84 → summit, layouts `road_summit` / `track_summit`). Textures and terrain are calibrated against the spring GoPro frames in `assets/gopro/`. See `CLAUDE.md` for the plan.

- Report: [`reports/phase1_report.md`](reports/phase1_report.md)
- 3D viewer: `reports/viewer.html` (single file. Open it in a desktop browser; it loads three.js from jsDelivr.)
- Corner screenshots: [`assets/`](assets/README.md)
- Data package: [`data/README.md`](data/README.md)
- Rebuild: `make all` (network required)

Attribution: road geometry seed © OpenStreetMap contributors (ODbL). LiDAR, DEM and imagery: USGS 3DEP / The National Map (public domain). Textures: ambientCG (CC0).

## Build the installable track (zip for Content Manager)

**Prebuilt:** `dist/palomares_road_v<VERSION>.zip` is committed via Git LFS. Download it from the GitHub file page ("Download raw file"), or run `git lfs pull` after cloning. Without LFS, a plain clone only contains a small pointer file.

To build it yourself:

```bash
pip install -r requirements.txt
make package          # ~10-15 min on 4 cores (21_build_geometry dominates); 16 GB RAM; needs internet for ambientCG CC0 textures
```

`make package` runs `scripts/p2/build.py`: a header, `[k/6] step...` lines, per-section timings inside the geometry
step, a counter per kn5 file written, and a summary table with the zip path and size. If a step fails, the runner names
it and exits non-zero. Colours and progress bars appear only on a terminal (set `NO_COLOR=1` to turn them off).
`pip install tqdm` gives nicer bars, but it is optional. `make phase2` (`scripts/p2/run_phase2.sh`) uses the same runner with QA and renders.

GPU: there is no `make package-gpu`. The build time is in the constrained-Delaunay terrain triangulation
(`triangle`, CPU-only, single-threaded) and in serial Python geometry code. Batched raster sampling on CUDA would save
well under a second, so torch is not used.

This writes `dist/palomares_road_v<VERSION>.zip`, with `palomares_road/` at the top level of the archive. Content Manager expects that layout.
`make phase2` does the same plus QA (`reports/qa_phase2.md`) and the render/visual checks (needs Node + Playwright Chromium).

The build needs Linux or WSL with Python 3.11+. The Phase 1 data it uses (`data/derived/`) is in the repo, so no LiDAR download is needed.

### Install

- **Content Manager:** drag the zip onto the CM window (or *Content → Install*, or `Ctrl+V` the path), then confirm **Install**. CM reads `palomares_road/` from the archive.
- **Manual:** unzip and copy `palomares_road/` into `<Steam>/steamapps/common/assettocorsa/content/tracks/`.
- **RAR:** not needed, because CM installs `.zip`, `.rar` and `.7z` alike. If you want one anyway, re-pack the folder: `rar a palomares_road.rar palomares_road/` (keep the folder at the top level).

Then pick **Palomares Road** → layout `road_summit` (real road) or `track_summit` (kerbs, no centre line). Use Practice or Hotlap. More details are in [`docs/phase2/INSTALL.md`](docs/phase2/INSTALL.md).
