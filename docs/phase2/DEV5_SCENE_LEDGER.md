# dev5: GoPro-registered scene ledger + fast per-frame scoring

## Why

Up to dev4, every roadside fix needed a full rebuild (21 → 22 → QA, which takes tens of minutes) and a render that often hung. Placements came from reading the frames by eye, with stations only accurate to ±40–50 m.

dev5 changes this:

1. Every GoPro frame gets a **registered camera pose** in the local track frame.
2. Every frame gets a **semantic segmentation**, plus our own refinements.
3. All singular roadside content lives in one **scene ledger**. This covers signs, markings, props, fences, bridges, houses, shrubs, mounds and tree-density edits.
4. A **fast scorer** renders *class-ID images* of the ledger plus cached geometry from each frame's pose. It compares them with the segmentation in seconds, with no kn5 build.

The full build reads the ledger only when packaging.

Segmentation is used for analysis only. No pixels from it, and no model weights, ship in the track.

## Data layout (all under `data/scene/`)

| path | what | writer |
|---|---|---|
| `poses.json` | per frame: `{frame, s, lateral, height, yaw, pitch, roll, fov_v, lens, conf, score}` in local coords (x_local, y_local, z NAVD88) | registration (step R) |
| `seg/NNN.png` | uint8 class-ID mask at 1021×743 (half res), classes in `seg_classes.json` | segmentation (step S) |
| `seg/NNN_conf.png` | uint8 confidence 0–255 | S |
| `seg_classes.json` | id → name, using the shared class list below | S |
| `ledger.jsonl` | one object per line (schema below) | ledger agents |
| `config/asset_catalog.yaml` | every asset type the ledger may reference | asset agent |
| `cache/` | cached render inputs: road/terrain/trees per 200 m chunk as npz | scorer |
| `scores/NNN.json` | per-class IoU / precision / recall, plus per-instance hit or miss | scorer |
| `scores/summary.csv` | one row per frame | scorer |

## Shared class list (ids are fixed)

```
0 unknown   1 sky        2 road       3 marking_yellow  4 marking_white  5 shoulder_dirt
6 terrain_grass 7 terrain_bare 8 rock_cut  9 vegetation_tree 10 shrub     11 trunk
12 sign_face 13 sign_back  14 pole      15 guardrail  16 fence      17 wall
18 building  19 bridge     20 vehicle(ignore) 21 person(ignore) 22 water 23 mailbox_prop
24 other_prop
```

## Ledger schema (`ledger.jsonl`)

```json
{"id":"sign_0042","cls":"sign","asset":"sign.curve_left","s":1234.5,"side":"R","off":1.8,
 "dz":0.0,"yaw":0.0,"scale":1.0,"s1":null,"params":{},"frames":[22,23],
 "conf":"high","src":"seg|survey|lidar|manual","status":"placed|todo|rejected","note":""}
```

- `s`: station in metres (centerline.csv, 0 at SR-84).
- `side`: L/R relative to northbound.
- `off`: metres beyond the **pavement edge** (road.outer). It is negative only for objects on the pavement, such as markings.
- `yaw`: degrees relative to the facing default for that class. A sign defaults to facing traffic in its `params.dir` (nb or sb).
- Linear objects (fence, guardrail, wall, marking, hedge, tree_density) use `s` to `s1`.
- `cls` must be one of: `sign, post, pole, mailbox, guardrail, fence, wall, bridge, culvert, building, driveway, gate, shrub, hedge, rock, debris, prop, marking, mound, ditch, tree_density, tree_add, tree_remove`.
- `mound` / `ditch`: terrain edits applied to the DTM before meshing. `params: {radius, height, length}`, Gaussian or capsule shape, and never within 0.3 m of the pavement.
- `tree_density`: span multiplier on the shrub/understorey density, plus an optional `canopy_fill` value.
- `asset`: must exist in `config/asset_catalog.yaml`.

## Iteration loop for agents (no kn5 build)

```
python3 scripts/p5/score.py --frames 20-30          # seconds: renders class-ID view + scores
python3 scripts/p5/score.py --frames 22 --overlay   # writes reports/dev5/overlay/022.jpg (photo | seg | render | diff)
python3 scripts/p5/ledger_lint.py                   # schema, catalog ids, overlaps, clearance from pavement
```

Only `make package` (21 → 22 → QA → 25) consumes the ledger for the real track.

## How agents use the scorer and poses (scripts/p5)

| script | does | typical time |
|---|---|---|
| `scene.py` | shared: class ids, camera/lens model (same as `27_litmus_fit.py`), chunk cache, class-ID renderer (numpy projection + PIL painter's order, pinhole; photo/seg undistorted to it) | — |
| `score.py` | render + compare one or more frames, write `scores/NNN.json`, `scores/summary.csv`, optional overlay | ~0.5–1 s / frame (512×372), `--all` ≈ 1.5 min on 3 workers |
| `register.py` | pose per frame → `poses.json` | ~9 s / frame, all 219 ≈ 11 min on 3 workers |
| `ledger_lint.py` | schema / catalog / overlap / clearance checks | < 1 s |

- **Cache**: `data/scene/cache/chunk_KKK.npz` (triangles + class per 200 m of station) is rebuilt automatically from
  `build/geometry.pkl` when that file changes (≈5 s); `--rebuild-cache` forces it. The ledger is *not* cached: it is
  read fresh on every run, so editing `ledger.jsonl` and re-running `score.py --frames N --overlay` is the whole loop.
- **Ledger proxies**: each object is a box (`proxy: [w, d, h]`, `z0`, `class` from `config/asset_catalog.yaml` if
  present, else `scene.DEFAULT_PROXY`), at `s`, `side`, `off` beyond the pavement edge (`geom.Road.outer`); linear
  objects get a box every 2 m from `s` to `s1`. Signs get a post. `mound/ditch/tree_density/marking/driveway` have no proxy.
- **Reading a score**: open the overlay first (photo | seg | render, then photo with render edges in magenta and the
  disagreement map: green agree, red disagree, yellow = road in photo only, magenta = road in render only). The
  numbers are secondary. `groups` holds merged IoUs (`vegetation` = tree+shrub+trunk, `terrain` = grass+bare+rock);
  per-class values are noisy. Per ledger instance: `px` = visible pixels, `match` = fraction of them the seg labels
  with a compatible class, `found` = px ≥ 6 and match ≥ 0.4, `occluded` = fewer than 6 pixels visible.
- **Seg**: `data/scene/seg/NNN.png` is used when it exists, else an HSV stand-in (`"seg": "hsv"` in the json); do not
  judge content against the stand-in.
- **Poses**: `score.py` uses `poses.json` and falls back to the old `gopro_stations_*.csv`. If a frame looks wrong in
  the overlay, decide the station by eye (landmarks, bridges, junctions) and pin it:
  `python3 scripts/p5/register.py --set 104=5560` (stored in `data/scene/pose_pins.json`; only the rest of the
  pose is refit within ±15 m). Low `conf` (< 0.3) or `flag_moved_30m` frames are the ones to check by eye first.
- CPU: everything runs at `nice 5` with at most 3 processes.

## dev7 rows: the `road` field (package M, 2026-11-07)

dev7 rows (Niles town, Niles Canyon Rd, Sunol) add a **`road`** field: a centerline id from `data/derived/dev7/centerline_<road>.csv`, such as `niles_canyon_road` (s=0 at Mission Blvd, 10729 at the Sunol stop) or `niles_blvd` (s=0 at the Sullivan Underpass end).

- `s` / `s1` are stations on **that** road. `side` is L/R relative to increasing s on that road. `off` is measured beyond that road's pavement edge (`edge_left_m` / `edge_right_m`). `dir` nb/sb means with / against increasing s.
- Rows without `road` are Palomares rows, exactly as before.
- `scripts/p5/ledger_merge.py` writes rows with `road` to `data/scene/ledger_dev7.jsonl` and everything else to `ledger.jsonl`, so the Palomares build never sees them.
- Lint dev7 rows with `ledger_lint.py --ledger data/scene/ledger_dev7.jsonl`. It checks s against the road's length.
- Extra fields:
  - `mly`: the Mapillary image ids the row was judged from (attribution in `reports/dev7/mapillary_attribution.csv`).
  - `params.mly_feature`: Mapillary map-feature ids.
- `src` may also be `mapillary`, `osm` or `naip`.
- dev7 asset types carry `dev7: true` in `config/asset_catalog.yaml` and have no generator yet. `assets.py --yaml` keeps them.
