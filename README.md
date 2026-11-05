# Palomares Road → Assetto Corsa hill climb

Status: **Phase 2 in progress** (drivable SR-84 → summit, layouts `road_summit` / `track_summit`). Textures and terrain are calibrated against the spring GoPro frames in `assets/gopro/`. See `CLAUDE.md` for the plan.

- Report: [`reports/phase1_report.md`](reports/phase1_report.md)
- 3D viewer: `reports/viewer.html` (single file. Open it in a desktop browser; it loads three.js from jsDelivr.)
- Corner screenshots: [`assets/`](assets/README.md)
- Data package: [`data/README.md`](data/README.md)
- Rebuild: `make all` (network required)

Attribution: road geometry seed © OpenStreetMap contributors (ODbL). LiDAR, DEM and imagery: USGS 3DEP / The National Map (public domain). Textures: ambientCG (CC0).

## Build the installable track (zip for Content Manager)

The zip is not committed (`dist/*.zip` is gitignored), so build it once:

```bash
pip install -r requirements.txt
make package          # ~5 min, 16 GB RAM is enough; needs internet for the CC0 textures (ambientCG)
```

This writes `dist/palomares_road_v<VERSION>.zip`, with `palomares_road/` at the top level of the archive. Content Manager expects that layout.
`make phase2` does the same plus QA (`reports/qa_phase2.md`) and the render/visual checks (needs Node + Playwright Chromium).

The build needs Linux or WSL with Python 3.11+. The Phase 1 data it uses (`data/derived/`) is in the repo, so no LiDAR download is needed.

### Install

- **Content Manager:** drag the zip onto the CM window (or *Content → Install*, or `Ctrl+V` the path), then confirm **Install**. CM reads `palomares_road/` from the archive.
- **Manual:** unzip and copy `palomares_road/` into `<Steam>/steamapps/common/assettocorsa/content/tracks/`.
- **RAR:** not needed, because CM installs `.zip`, `.rar` and `.7z` alike. If you want one anyway, re-pack the folder: `rar a palomares_road.rar palomares_road/` (keep the folder at the top level).

Then pick **Palomares Road** → layout `road_summit` (real road) or `track_summit` (kerbs, no centre line). Use Practice or Hotlap. More details are in [`docs/phase2/INSTALL.md`](docs/phase2/INSTALL.md).
