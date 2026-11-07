# Install — Palomares Road (Assetto Corsa)

0. Build the zip if you only have the repo: `make package` (see README).
1. **Content Manager:** drag `palomares_road_vX.Y.Z.zip` onto CM and click Install, then skip to step 3. **Manual:** unzip it.
2. Copy the `palomares_road/` folder into `<Steam>/steamapps/common/assettocorsa/content/tracks/`.
3. In Content Manager (or the AC launcher) pick **Palomares Road**, then a layout:
   - **road_summit**: the real road. Real width and center/edge lines, no kerbs.
   - **track_summit**: same road, no center line, red/white kerbs on apexes and exits.
4. Mode: **Practice** or **Hotlap**. The run is point-to-point: start just above SR-84 (Niles Canyon Rd), finish at the summit (about 7.4 km, +329 m).

Content Patch (CSP) is not required. If timing doesn't start or stop under vanilla AC, try with CSP (A-B timing support varies; see CHANGELOG known issues).

The first load takes a while: about 100 MB of uncompressed PNG textures and geometry.

## Data sources and attribution

- Road centerline and some building outlines: © OpenStreetMap contributors, ODbL (openstreetmap.org/copyright).
- Terrain, buildings (footprints, roof shapes), fence/rail lines and understorey: USGS 3DEP LiDAR, Alameda County 2021 (public domain).
- Aerial imagery and roof colours: USGS imagery and USDA FSA NAIP 2022 (public domain).
- Street-level reference images: © Mapillary contributors, CC-BY-SA 4.0 (mapillary.com). Used as reference only; no Mapillary pixels are in the track.
- Textures: ambientCG / Poly Haven (CC0) and procedural.
