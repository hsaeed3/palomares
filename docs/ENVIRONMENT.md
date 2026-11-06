# Environment (Phase 1)

- Cloud container: 4 vCPU, 15 GB RAM, Linux 6.18, Python 3.13.18, Node 22.23.4
- Python packages: see `requirements.txt` (pip, not conda). Exact versions: numpy 2.5.3, scipy 1.18.1, laspy 2.7.0 + lazrs 0.8.2, pyproj 3.8.0, shapely 2.2.0, rasterio 1.5.2, pillow 12.3.0, matplotlib 3.11.2
- Playwright 1.63.0 (global node module) with the pre-installed Chromium. WebGL comes from SwiftShader (`--use-angle=swiftshader`).
- Blender: not needed in Phase 1. It will be installed headless for Phase 2.

## Deviations from CLAUDE.md §1, with reasons
- **No PDAL.** USGS EPT is read directly: octree hierarchy JSON plus LAZ nodes decoded with laspy/lazrs (`scripts/02_fetch_lidar.py`). This avoids a conda stack and gives the same points.
- **pip instead of micromamba.** All needed wheels exist on PyPI for py3.13.
- **Overpass was unreliable** (connection resets through the proxy), so OSM data comes from the main API (`/api/0.6/way/{id}/full`, `/api/0.6/map` bboxes).

## Data source facts verified
- EPT sets `CA_AlamedaCo_{1,2,3}_2021` = project CA_AlamedaCounty_2021_B21, **QL1**, Riegl VQ-1560ii. Work unit 1 was flown 2021-06-21 to 07-02, and unit 2 from 2019-07-02. NVA 4.04 cm and VVA 10.46 cm at 95% (USGS WP report).
- EPT horizontal is EPSG:3857 and Z is **metres** NAVD88. The source delivery is EPSG:6420/6360 (ft), but the EPT Z values match USGS EPQS to within ±0.17 m.
- Vegetation is **not** classified in this delivery (classes 1, 2, 7, 9, 17, 18 and 20 only).

## Phase 2 additions
- No Blender. All geometry is built in Python (`scripts/p2/`) and written with our own kn5 writer, which is reproducible and headless. Renders use three.js in headless Chromium (SwiftShader) from the **round-tripped** kn5 data.
- Extra pip packages: `triangle` (Shewchuk constrained Delaunay), `pyyaml`.
- Mapillary: the token is configured, but per the user Palomares Rd has no Mapillary coverage, so the visual check has no street-photo column. Drop-in slot: `reports/visual/reference/ref_s<station>.png`.
