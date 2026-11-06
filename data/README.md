# data/

| Path | Content | In git |
|---|---|---|
| `raw/` | OSM way/context downloads, EPT metadata & hierarchy, CC0 textures (ambientCG) | no |
| `cache/` | per-EPT-node clipped points (`pts/`), 250 m station chunks (`chunks/`), debug arrays | no |
| `derived/` | Phase 1 products below (reused by Phases 2–3, never re-derived) | yes |

## derived/
All rasters are GeoTIFF in **EPSG:26910 (UTM 10N, m)**, with heights in **NAVD88 (GEOID18) m**. Local origin is in `config/origin.json`: local = UTM − (x0, y0).

| File | Description |
|---|---|
| `centerline.csv` | Refined centerline every 1 m, S→N. Columns: `s_m, x_local, y_local, z_navd88, x_utm, y_utm, lon, lat, width_m, edge_left_m, edge_right_m, crossfall_left, crossfall_right, grade, curvature_1pm, env_left, env_right, ground_cover, paint_center, paint_left, paint_right`. Left/right are relative to the uphill (S→N) direction. Crossfall is dz/dd with d positive to the left, so for a crowned road `crossfall_left < 0` and `crossfall_right > 0`. |
| `centerline.geojson` | Same line in WGS84 lon/lat/z |
| `osm_centerline_*.{csv,geojson}` | Original joined OSM line (ODbL) |
| `dtm_r050_b40.tif` | Bare-earth DTM, 0.5 m, within 40 m of road (TIN of class 2 + 17) |
| `dtm_r100_b150.tif` | Bare-earth DTM, 1 m, within 150 m |
| `dsm_r100_b150.tif`, `chm_r100_b150.tif` | First-surface model and canopy height (DSM − DTM), 1 m |
| `gdist_r050_b40.tif` | Distance from each cell to the nearest ground return (interpolation confidence) |
| `ortho_r100.tif` | USGS imagery (public domain), 1 m RGB over corridor bbox |
| `backdrop_dem_r10.tif`, `backdrop_ortho_r05.tif` | 3DEP 10 m DEM and 5 m imagery, bbox + 2 km (horizon) |
| `features.geojson` | junction, driveway, stream_crossing, bridge_osm, bridge_lidar_class17, guardrail_candidate, steep_face, steep_drop, lidar_gap/sparse, sign, summit (all carry station `s_m` or `s_start_m/s_end_m`) |
| `trees.csv`, `trees.geojson` | 9,067 trees ≥ 4 m within 60 m of the road: position, ground z, height, crown radius, station/offset |
| `gaps.csv` | Ground-return gaps > 5 m along the centerline (none after counting bridge-deck returns) |
| `markings.csv` | Per-50 m paint-intensity scores and center/edge line classification |
| `epqs_check.csv` | Centerline z vs USGS EPQS 1 m DEM at 12 stations |
