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
