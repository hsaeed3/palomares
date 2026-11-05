# Changelog

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
