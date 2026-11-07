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

## dev6 measured scenery (scripts/p6, run 01 → 07 in order; 09 = review plots)

| File | Description |
|---|---|
| `derived/naip2022_r060.tif` | NAIP 2022-05-19, 0.6 m, RGB+NIR, within 170 m of the road (public domain, USDA FSA via Planetary Computer) |
| `derived/osm_buildings_barriers.geojson` | OSM building footprints (26) near the road (ODbL); no barrier ways exist in OSM here |
| `derived/buildings.geojson` | LiDAR-measured buildings: footprint, rectangle parts, roof form/eave/ridge, NAIP roof colour, NDVI, OSM IoU |
| `derived/fences_detected.geojson` | LiDAR line detections: fence/wall/hedge (segment detector) and guardrail (lateral-profile detector) |
| `derived/understory.tif` | 1 m: band 1 = share of returns 0.3–3 m above ground, band 2 = NAIP NDVI |
| `derived/understory.geojson` | verge hedge rows (0.3–4 m beyond the edge) and bank shrubs (4–12 m) derived from understory.tif |
| `derived/mapillary_images.geojson` | Mapillary images within 40 m (CC-BY-SA 4.0); thumbnails in `cache/p6/mapillary/` |
| `cache/p6/nodes/` | reduced EPT nodes (not in git) |

## dev7 data package D (scripts/p7, run `scripts/p7/run_all.sh`)

`derived/dev7/` covers the Niles town streets, Niles Canyon Rd (SR-84, Mission Blvd → Sunol stop) and the Palomares connector. It uses the same CRS and fixed origin as Phase 1. See `reports/dev7/data_report.md`.

| File | Description |
|---|---|
| `network.geojson`, `junctions.geojson` | OSM drivable edges split at junctions (tags: lanes, width, surface, oneway, bridge, …, role) and graph nodes (degree, roads, signal/stop control) — ODbL |
| `osm_centerline_<road>.csv`, `osm_centerlines.geojson` | ordered OSM line per named road (local, UTM, lon/lat, node id) |
| `centerline_<road>.csv/.geojson` | LiDAR-refined 1 m centerline + cross-sections (Phase-1 columns, plus curb_left/right, rise_*_5m, gdist_m, canopy_frac, bridge17, snap_offset_m) |
| `centerlines_summary.json`, `termini.json`, `scope.json/.geojson` | per-road stats; confirmed termini; town / 40 m detail / 150 m terrain polygons (WKT local) |
| `dtm_r050_b40.tif`, `dtm_r100_b150.tif`, `dsm_r100_b150.tif`, `chm_r100_b150.tif`, `gdist_r050_b40.tif`, `int_r050_b40.tif`, `cover_r100_b150.tif` | rasters (EPSG:26910, NAVD88) |
| `features.geojson`, `gaps.csv` | junctions, signals/stops, bridges (OSM + class 17), rail underpass / crossings, culverts, walls/cuts, drops, guardrail candidates, curbs, LiDAR gaps, canyon high/low points |
| `osm_features.geojson` | parking, rail lines/platforms/stations, buildings, barriers, waterways, POIs (ODbL) |
| `trees.csv/.geojson`, `buildings.geojson` | CHM tree tops ≥ 4 m within 60 m; OSM footprints with LiDAR height |
| `mapillary_index.csv`, `mapillary_coverage.csv` | Mapillary images within 40 m (id, lon/lat, compass, captured_at, road, s, offset) — CC-BY-SA 4.0, index only |
