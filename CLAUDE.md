# Palomares Road → Assetto Corsa Hill Climb — Agent Instructions

You are building a drivable Assetto Corsa (AC) track of **Palomares Road** (Alameda County, CA), from its southern end at **Niles Canyon Road (SR-84)** to its northern end in **Castro Valley** (near I-580 / Eden Canyon Rd — confirm the exact termini in OpenStreetMap). It is a point-to-point **hill climb** running south → north.

The goal is not survey-grade 1:1. It's a road that **drives like the real one**: correct layout, elevation, camber and width, with cliffs, cut banks, drop-offs, walls and guardrails roughly where they really are. The road runs under heavy tree canopy, so **airborne LiDAR ground returns are the primary geometry source**, not map imagery.

You work entirely in this cloud environment. You cannot run Assetto Corsa. The user drives each build on their own PC and sends feedback.

---

## 0. Environment requirements (for the user setting this up)

**Recommended machine:** 8+ vCPU, 32 GB RAM, 150 GB disk. A GPU is optional and not needed.

**Network allowlist** (HTTPS):

| Purpose | Hosts |
|---|---|
| USGS 3DEP LiDAR (EPT point clouds on AWS) | `usgs-lidar-public.s3.amazonaws.com`, `s3-us-west-2.amazonaws.com`, `s3.us-west-2.amazonaws.com` |
| USGS LiDAR tiles / DEMs / metadata | `tnmaccess.nationalmap.gov`, `prd-tnm.s3.amazonaws.com`, `rockyweb.usgs.gov`, `apps.nationalmap.gov` |
| NOAA mirror of the same LiDAR (fallback) | `noaa-nos-coastal-lidar-pds.s3.amazonaws.com`, `coast.noaa.gov` |
| OpenStreetMap (centerline, road tags) | `overpass-api.de`, `overpass.kumi.systems`, `api.openstreetmap.org`, `nominatim.openstreetmap.org` |
| Aerial imagery (NAIP / USGS, public domain) | `basemap.nationalmap.gov`, `planetarycomputer.microsoft.com`, `naipeuwest.blob.core.windows.net` |
| Street-level reference photos (open license) | `graph.mapillary.com`, `*.fbcdn.net` (Mapillary image CDN) — needs a free Mapillary token |
| CC0 textures (asphalt, rock, dirt, bark, foliage) | `ambientcg.com`, `api.polyhaven.com`, `dl.polyhaven.org` |
| Python / conda packages | `pypi.org`, `files.pythonhosted.org`, `conda.anaconda.org`, `repo.anaconda.com`, `micro.mamba.pm` |
| Blender (headless binary) | `download.blender.org`, `mirrors.ocf.berkeley.edu` |
| GitHub (AC tooling, kn5 exporters, reference code) | `github.com`, `api.github.com`, `codeload.github.com`, `objects.githubusercontent.com`, `raw.githubusercontent.com` |
| AC modding docs/reference (read-only) | `assettocorsamods.net`, `www.overtake.gg`, `www.gtplanet.net` |

**Secrets (optional):** `MAPILLARY_TOKEN`. Do **not** use Google Street View or Google Photorealistic 3D Tiles for anything that ends up in the track. Their terms forbid extracting assets from them. If the user provides them, use them only as visual reference.

---

## 1. Tooling

Set up once, and record exact versions in `docs/ENVIRONMENT.md`:

- **micromamba / conda-forge env** with: `pdal`, `python-pdal`, `gdal`, `pyproj`, `laspy[lazrs]`, `numpy`, `scipy`, `shapely`, `geopandas`, `rasterio`, `trimesh`, `pillow`, `matplotlib`, `requests`, `osmnx` (or raw Overpass queries).
- **Blender 4.x headless** (official Linux tarball, run as `blender -b -P script.py`). Do all modeling through scripts so every build is reproducible.
- **KN5 export:** evaluate the open-source Blender → kn5 exporters on GitHub first (e.g. `moppius/blender-assetto-corsa-tools`, and the "AC Tools" / `ac-track-tools` Blender add-ons). If none works with your Blender version, write a minimal kn5 writer in Python. The format is well documented by open-source kn5 readers on GitHub. Either way, **write a kn5 reader/validator** and round-trip every kn5 you produce: parse it back and check mesh counts, vertex/index limits, materials and textures.
- **Never use the Windows-only Kunos ksEditor.** The whole pipeline must run headless on Linux.

---

## 2. Coordinate system & data rules

- **Working CRS: UTM Zone 10N, meters (EPSG:26910 or EPSG:6339), NAVD88 heights.**
- The USGS EPT point clouds on AWS are stored in **Web Mercator (EPSG:3857)**. At this latitude its "meters" are about 27% too large. **Always reproject to UTM before measuring anything.** The county source data is in CA State Plane Zone 3, **US survey feet**, so convert units carefully.
- Pick a **local origin at the midpoint of the full road** and keep it fixed for every phase, so all layouts and later extensions line up. AC uses floats, so a central origin keeps coordinates under about 8 km and avoids precision jitter.
- Store all intermediate data in a stable `data/` layout with a `data/README.md`. Phases 2 and 3 must reuse the Phase 1 data. Do not re-derive it.
- Main source: **2019–2021 USGS Alameda County LiDAR (QL1/QL0)**. Check which quality level covers each part of the corridor and record it. Use the 2006 Alameda LiDAR only as a fallback.

---

## 3. Phases

Work in this order. **Stop at the end of each phase**, deliver, and wait for the user's go-ahead.

### Phase 1 — "Port" the full road (data + model, not yet a game track)

Covers the **entire** road from Niles Canyon Rd to Castro Valley.

1. **Corridor:** get Palomares Road from OSM (all ways, ordered, joined into one polyline). Buffer it by about 150 m for terrain and about 40 m for detailed work.
2. **LiDAR:** clip the point cloud to the corridor, in chunks to fit in memory. Keep classes 2 (ground), 3–5 (vegetation), 6 (buildings), 9 (water), 17 (bridge deck) and intensity.
3. **Bare-earth DTM:** 0.25–0.5 m within the 40 m road buffer, 1–2 m out to 150 m. Fill gaps under dense canopy by TIN interpolation and **flag every gap longer than 5 m along the road** in the report.
4. **Refined centerline:** OSM is only a few meters accurate. Snap the centerline to the actual road using the DTM (flat, smooth strip; break lines at the edges) and LiDAR **intensity**, because lane paint and fresh asphalt show up clearly. Resample to 1 m spacing, smoothed so curvature stays continuous.
5. **Cross-section per meter:** road width (left/right edge offsets), camber/crossfall, and shoulder type. Detect what lies just past each edge: rising cut bank or rock wall, drop-off, ditch, flat shoulder.
6. **Features:** extract or flag guardrails, retaining walls, bridges/culverts (class 17, creek crossings), driveways, junctions and signs where you can. Use the canopy height model (DSM minus DTM) to get **tree positions, heights and crown sizes** within about 60 m of the road.
7. **Road markings:** map where the real road has center lines and edge lines, and what type (double yellow, broken, none). Use intensity and Mapillary photos.
8. **Find the summit:** the highest point on the centerline. It is the finish line for Phase 2.

**Phase 1 deliverables:**
- `reports/phase1_report.md`:
  - total length and summit location/elevation
  - elevation profile chart
  - grade statistics
  - width range
  - LiDAR coverage/quality map
  - list of gaps and uncertain sections, with distance markers
- A **standalone HTML 3D viewer** (three.js, single file) of the corridor: terrain, road ribbon and tree points, with distance markers. The user will review it.
- `data/` package: centerline (GeoJSON + CSV with distance, x, y, z, width, camber), DTMs (GeoTIFF), features (GeoJSON).

### Phase 2 — Drivable hill climb: Niles Canyon → summit (two versions)

Build one AC track with two layouts covering **start at SR-84 → finish at the summit**:

| Layout folder | Description |
|---|---|
| `road_summit` | The road as it really is: real width, real center/edge lines, real road edges, no kerbs. |
| `track_summit` | Same geometry and width, **no center line**, race kerbs on corner apexes and exits, edge lines optional. |

Build steps:

1. **Road mesh:** sweep a clean road surface along the refined centerline using the measured width and camber. Use about 0.5–1 m across and 1–2 m along, finer in tight corners. Don't use the raw LiDAR mesh as the road surface, because it drives badly. Add slight, low-amplitude surface noise for realism. Do not add noise that causes force-feedback chatter.
2. **Terrain & edges:** blend the DTM into the road edges with no gaps or steps. Model cut banks, rock walls, drop-offs and ditches from the cross-section data. Add collision walls where the real road has walls or cliffs, so the car can't climb a vertical face or fall through.
3. **Barriers:** guardrails and retaining walls as simple modular pieces at the Phase 1 positions.
4. **Kerbs (track layout only):** place them automatically where curvature exceeds a threshold, covering apex plus exit, using the standard red/white AC kerb profile. Write the placement rules and parameters to `config/kerbs.yaml` so the user can tune them.
5. **Vegetation:** trees at their real LiDAR positions and heights. Use a few species-appropriate models (oak/bay/eucalyptus style) as alpha-tested cards or low-poly meshes, with LOD and visibility distances. The canopy tunnel look matters, but keep frame rate in check: target under 1,500 draw calls.
6. **Materials:** AC shaders (`ksPerPixel`, `ksPerPixelMultiMap` for the road, alpha-tested shaders for foliage), CC0 textures only. The road texture carries the markings for each layout.
7. **AC integration** (check every convention against a known-good community track or the docs before relying on it):
   - Multi-layout track folder `palomares_road/`: `models_<layout>.ini`, `<layout>/data/` (`surfaces.ini`, `cameras.ini`, `lighting.ini`, etc.) and `ui/<layout>/` (`ui_track.json`, `preview.png`, `outline.png`) for each layout. Shared kn5 files go in the root.
   - Physical surfaces via object naming (`1ROAD…`, `1KERB…`, `1GRASS…`, `1WALL…`, etc.) defined in `surfaces.ini`, including friction, damping, IS_VALID_TRACK and kerb vibration.
   - Point-to-point timing: A-to-B start/finish gate objects (`AC_AB_START_L/R`, `AC_AB_FINISH_L/R`), plus `AC_START_0`, `AC_PIT_0`, `AC_HOTLAP_START_0` placed just before the start gate, facing uphill.
   - `map.png`, track length and description in `ui_track.json`. Include attribution for OSM (ODbL), USGS (public domain) and any Mapillary/CC-BY sources.
   - **kn5 limits:** meshes use 16-bit indices, so split every mesh to **65,535 vertices or fewer**. Chunk the road and terrain about every 200–300 m.
   - Optional: a CSP `ext_config.ini` for nicer grass and trees, but the track must work without CSP.
8. **Package:** `dist/palomares_road_vX.Y.zip` containing the full `palomares_road/` folder, ready to drop into `assettocorsa/content/tracks/`. Include `CHANGELOG.md` and `INSTALL.md`.

### Phase 3 — Complete the full road (only after the user says Phase 2 feels right)

Extend from the summit down into Castro Valley with the same methods and the same origin. Add layouts `road_full` and `track_full` (start SR-84, finish at the northern terminus). Keep the `_summit` layouts working by sharing the same kn5 chunks, with the finish gates moved. Re-run every QA check over the full length.

---

## 4. QA (run before every delivery)

Ideally a separate QA subagent runs these, one that didn't build the meshes:

- **Geometry vs. source:** road surface height against the LiDAR ground points along the centerline (report mean and max deviation; investigate anything over 0.3 m). Layout against the refined centerline (over 0.5 m is an error).
- **Drivability:** no vertical step over 2 cm and no hole anywhere in the drivable or collision surface. Check for normals pointing the wrong way, and for spikes in grade or camber change per meter. Report the max grade, which should be plausible.
- **Visual check:** use headless Blender to render driver's-eye views every 250 m. Put them side by side with the nearest Mapillary photo where one exists, in a `reports/visual_check.html` contact sheet.
- **kn5 validation:** round-trip parse, vertex/index limits, every texture referenced and present, every physics object name matching `surfaces.ini`.
- **Package check:** the folder structure matches a known-good multi-layout track, and every layout has complete `data/` and `ui/` folders.

---

## 5. Working with the user

- The user tests on their own PC. Give them a **short feedback template** with each build: distance marker or landmark, layout, what feels wrong (bump, grip, width, wall missing or misplaced, visual issue), and a screenshot if possible.
- Each iteration: apply the fixes, bump the version, update the changelog, re-run QA and re-deliver.
- Report problems plainly (data gaps, uncertain sections, conventions you couldn't verify), with the distance marker where each one occurs.
- **Subagents:** parallelize independent work. For example: data/terrain, road and cross-sections, vegetation/environment, AC integration/packaging, and a separate QA agent. Data must be finished before the road work starts, and the road before AC integration. Use one shared `data/` folder and one Blender build script per module.

## 6. Repo layout

```
config/      # kerbs.yaml, materials.yaml, layout definitions, origin
scripts/     # 01_fetch … 09_package — numbered, re-runnable, idempotent
blender/     # headless build scripts per module
data/        # raw/ (gitignored), derived/ (centerline, DTMs, features)
reports/     # phase reports, viewer, visual checks
dist/        # packaged builds
docs/        # ENVIRONMENT.md, decisions, AC conventions verified
```

The whole pipeline must rebuild from scratch with `make all` (or `scripts/run_all.sh`) given network access.

## 7. Strict rules for every session and agent

- **Commit messages:** never include a session URL (e.g. `Claude-Session: https://claude.ai/code/...`) or any other session link in commit messages, co-author trailers, PR bodies or repo files. The only allowed trailer is `Co-Authored-By: Claude <noreply@anthropic.com>`.
- **Runtime budget:** before starting any command, estimate how long it will take. Anything expected to run over 10 minutes must be split into resumable batches (each under 10 minutes) or reduced in scope. Do only the work the current deliverable needs, e.g. the 8 litmus frames, not all 219, and widen the scope only after the small run is verified.
- **Slowdown watch:** the coordinator runs `scripts/ops/watchdog.sh`. It fires when any pipeline process runs longer than 10 min, free RAM drops below 2 GB, or more than 2 heavy steps (21, 21d, 22, 03_dtm, calib register) run at once. When it fires, the coordinator re-plans: it pauses, splits or re-orders the work. Agents check for running heavy steps (`ps`) before starting their own, and use the quick path (`build.py quick`) and `view.py` before any full build.
