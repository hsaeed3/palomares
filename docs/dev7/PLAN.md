# dev7: Niles + Niles Canyon extension (free roam)

This is the user's request. dev6 continues in parallel and is released as its own build.

## Scope

- **Free-roam map.** It starts at the block **before Devout Coffee on Niles Blvd**, in the Niles district of Fremont, near the nursery and homes. Confirm the exact spot from Mapillary and OSM. It contains:
  - the whole Niles "town area": Niles Blvd main street, Devout Coffee, the parking lots, and the railroad and train (Niles depot, Niles Canyon Railway, UP/ACE tracks)
  - our **full Palomares Road**
  - the **complete Niles Canyon Road (SR-84)**, from the **Mission Blvd intersection** to the **traffic light / stop sign in Sunol**
- **Niles Canyon "hill climb" layouts**, forward and reverse, each as a road (standard) and track version.
- Real roadside features must be present, e.g. the **flashing warning signs before the railroad underpass** on Niles Canyon Rd, plus the bridges, rail crossings and underpasses.
- **Street view must come from Mapillary** for this asset pack. The env var is `MAPILARY_TOKEN` (one L); read either spelling. Imagery is CC-BY-SA, so attribution is required.

## New layouts (added to the existing 12)

| folder | what |
|---|---|
| `freeroam` | whole network, no timing gates, spawn on Niles Blvd before Devout Coffee |
| `canyon_road`, `canyon_track` | Niles Canyon Rd hill climb, Mission Blvd → Sunol |
| `canyon_rev_road`, `canyon_rev_track` | Sunol → Mission Blvd |

Same conventions as Palomares:
- The track version keeps the road texture and the yellow lines, and only adds kerbs and tasteful race furniture.
- Point-to-point AB gates.

## Rules carried over

- **Data:** LiDAR (USGS 2021 Alameda EPT, the same 3 sets) is the geometry source. OSM supplies the network topology. NAIP and Mapillary supply types and positions. GoPro frames exist only for Palomares.
- **CRS and origin:** UTM10N. The existing local origin (config/origin.json) is kept fixed so the Palomares layouts don't move. If the new extents exceed about 8–10 km from the origin, record the precision decision in docs/dev7/DECISIONS.md.
- **Trademarks:** no real logos or brand trademarks. "Devout Coffee" appears as a generic coffee-shop storefront with neutral signage (the shop is present, the logo isn't). The railroad uses generic period rolling stock, not branded liveries.
- **Props, trees and collision:** use whatever dev6 finalises, namely CSP and Content-Manager-native or CC0 assets, the collision conventions in docs/AC_CONVENTIONS.md, and the material/texture budget (the NVIDIA cache issue). Nothing is hand-made from scratch where a proven asset or CSP feature exists.
- **FFB safety:** the dev6 qa_ffb_safety gate applies to everything.

## Work packages

1. **D: data** (now). Covers the OSM network, LiDAR fetch, DTMs, refined centerlines for every road in scope, cross-sections, features, the Mapillary image index and the corridor reports.
2. **M: Mapillary survey** (now). Positions and types of every singular asset in town and canyon, written as ledger rows with `src=mapillary`.
3. **G: multi-road geometry** (after D and the dev6 prop decision). Junctions, intersections, parking lots, rail, and the town surfaces.
4. **A: assets** (after M and the dev6 prop decision). Town buildings, the train/rail set, the flashing signs, the underpass and the bridges.
5. **I: AC integration** (after G and A). The 5 new layouts, free-roam spawns, AI lines and packaging.
6. **QA** (separate agent). Everything in CLAUDE.md §4, plus qa_ffb_safety.
