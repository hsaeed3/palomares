# dev7 decisions

## D7-1: Keep the fixed local origin even though the extents reach about 10 km (work package D, 2026-11-07)

**Facts** (from `scripts/p7/01_osm_network.py` and `data/derived/dev7/scope.json`, local = UTM 10N − `config/origin.json`):

| Area | Max radial distance | Max abs(x) | Max abs(y) |
|---|---|---|---|
| Niles town polygon (Niles Blvd + 230 m) | 8.12 km | 1.21 km | 8.12 km |
| 40 m road detail area | 10.14 km | 8.18 km | 8.18 km |
| 150 m terrain area | 10.14 km | 8.19 km | 8.19 km |
| Niles Canyon Rd centerline | 5.6–9.76 km | | |
| Sunol stub (Paloma Rd, Pleasanton-Sunol Rd) | up to 10.10 km | | |

The farthest drivable point is the Paloma Rd stub east of Sunol, at 10.1 km. Niles town sits about 8.0–8.1 km south of the origin.

**Decision:** keep the origin unchanged. Rationale:

- The Palomares layouts must not move (PLAN.md), and every dev7 layout lives in the same AC track folder, so it shares one coordinate frame.
- float32 spacing is what matters, and it applies per axis, not radially. No coordinate exceeds 8.19 km on either axis. Up to 8192 m the float32 ULP is 0.49 mm, and from 8192 m to 16384 m it is 0.98 mm. Vertex and physics positions therefore stay at or below 1 mm resolution everywhere. That is about 20× below the 2 cm step QA threshold, so it causes no visible jitter or FFB effect.
- Moving the origin would change every existing kn5, the AI lines and the cameras of the 12 shipped layouts for no measurable gain.

**Guard for later packages:** if a future extension pushes any axis beyond 16.38 km (ULP 1.95 mm), revisit this decision. Do not add terrain beyond about 1 km past Sunol without updating this entry. Package G should still report `max(|x|, |y|)` per kn5 in QA.
