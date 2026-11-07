# Changelog

## 0.2.0-dev6 (palomares_road_dev6)

**Test build.** It was shipped early at the user's request. The new FFB-safety gate (qa_ffb_safety) still reports 215 findings in this build; they are listed below and the fix is in progress for dev6.1.

- **Collision:** follows AC conventions now. Barriers, fences and the forest line each get one continuous, invisible, closed slab at least 1 m deep, with flared ends. Trees, shrubs and small props have no collision, so there are no trunk-hit FFB spikes. Buildings always collide.
- **Shader cache (NVIDIA DXCache popup):** every texture is now DDS with mipmaps and power-of-two sizes. Terrain materials are merged into atlases, about 191 down to about 13. Tangents are fixed. Clear `%LOCALAPPDATA%\NVIDIA\DXCache` once before testing.
- **Light through hills:** backdrop hills now cast shadows.
- **Trees:** cards made from photographic CC0 leaf cut-outs, with more and smaller clusters. Conifers and bare winter sycamores are added. A far-zone scatter (60-150 m) is driven by LiDAR canopy height and NAIP.
- **Grass:** spring-green terrain base. GrassFX tall grass plus wildflowers.
- **Road:** crack seal is baked into the asphalt, and the random decals in mid-road are removed. Edge lines are solid. Near-road terrain follows the LiDAR.
- **Racing line:** resampled at 1 m, so it is a continuous AC line instead of sparse arrows.
- **Measured placement:** LiDAR buildings, NAIP-derived understorey, and new litmus camera calibration.

**Known issues (FFB gate findings, distance from SR-84):**
- **Building colliders:** zero-thickness boxes at s 14,300-15,700 (Castro Valley end).
- **Kerbs:** about 27 crease warnings over 25 deg, in both directions.
- **Road aprons:** creases of 9-10 deg against the 8 deg limit, at 4 spots.
- **Barrier walls:** 8 road-face creases of 31-37 deg.
- If you feel a sharp FFB hit anywhere, note the distance marker. Keep FFB gain moderate on this build.

## 0.2.0-dev5

**SAFETY WARNING: do not drive this build with strong force feedback or a direct-drive wheel.** Collisions with
trees, barriers, fences and terrain can produce violent force-feedback spikes and pass-through. dev6 rebuilds all
collision geometry to standard AC conventions. Until then, lower FFB gain substantially or use a gamepad.

- New GoPro-registered workflow: segmentation of all 219 frames, per-frame camera poses (44 pinned by eye),
  a scene ledger (634 roadside objects) and a fast photo-vs-render scorer.
- 118-type procedural asset catalog (bridges, houses, fences, signs, shrubs, props) placed from the ledger.
- Near-road terrain now follows LiDAR (banks rise at the edge, small mounds kept); frame-3 yard flattened.
- Solid white edge lines everywhere (LiDAR had missed long stretches).
- Litmus: 8 fixed frames rendered from the registered poses (reports/litmus.html).
- Known: one ranch-wire fence at s 9303-9345 has a collision gap; ~a dozen frames have weak poses
  (26, 62-64, 66, 70-72, 76, 106, 138, 178, 180, 186, 187); many assets and textures still differ from the photos.

## 0.1.0 — first drivable build (Phase 2)
- Layouts `road_summit` and `track_summit`: SR-84 → summit (s = 20 m → 7392 m, 7372 m timed), road modelled 250 m past the finish as run-off.
- Road surface swept along the LiDAR-refined centerline: 13 vertices across, 1 m along, measured width and per-side crossfall (lightly smoothed), ≤4 mm long-wave surface noise.
- 0.8 m dirt shoulders, then bare-earth terrain from the 0.5 m / 1 m LiDAR DTM (constrained Delaunay, stitched exactly to the shoulder edge, no gaps). Rock texture on slopes over ~49°.
- 10 m DEM horizon textured with USGS imagery.
- 5,169 LiDAR trees (≥ 4 m, real positions/heights/crown sizes) as alpha-tested crossed cards, 3 procedural species styles (oak / bay / eucalyptus by shape).
- Guardrails at the Phase 1 guardrail candidates (posts + W-beam, with a collision strip), invisible collision walls behind steep cuts, end walls.
- Paint: double yellow center line and white edge lines from the Phase 1 marking map (road layout); edge lines only (track layout).
- Kerbs auto-placed from curvature (`config/kerbs.yaml`): 44 kerbs.

### Known issues / unverified
- **Art direction: the target season is SPRING** (the user's GoPro reference frames: lush green hills, fresh lime oak/alder growth, darker bay), not summer. The USGS backdrop ortho and LiDAR-era imagery are summer-gold; 22_build_kn5.py grades the backdrop toward the GoPro spring-grass palette. Any remaining golden/brown summer cue is a bug.
- **Spawn direction unverified.** Locators use the AC-Tools convention (Y up, −Z front). If you spawn facing downhill, report it: it's a one-line flip.
- **A-B timing** uses `AC_AB_START_L/R` / `AC_AB_FINISH_L/R`. Not verified in vanilla AC; CSP is known to support it.
- Guardrail positions come from low-confidence LiDAR candidates. Some may be missing or extra.
- Tree cards are procedural placeholders, not photo textures.
- Textures are PNG inside the kn5 (DDS would load faster).
