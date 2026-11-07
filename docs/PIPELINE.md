# Fast pipeline (dev7 plan, section P)

The full build (`scripts/p2/build.py package|full`: 19 → 21 → 21b → 21c → 21e → 21f → 21d → 22 → QA → 25) is for
**releases only**. For day-to-day work, follow the loop below: **edit → view → quick build → verify**. Each step takes
seconds, or under a minute.

Measured on the 4-core / 15 GB dev machine, with other agents' jobs running alongside:

| what | command | time |
|---|---|---|
| full 22 (all kn5, textures, AC folder) | `build.py package` step 22 | **500 s** (+ 21: several min, 21d: 364 s) |
| seed the quick dir (once per release build) | `build.py quick-seed` | ~510 s (copy + one full 22) |
| one ledger prop edit → updated kn5 | `build.py quick` | **22 s** (21e 18 s, collision 1 s, kn5 3 s) |
| kn5 only, nothing changed | `build.py quick --no-21e` | 5–10 s |
| far-zone trees regenerate → trees kn5 | `21f` + `quick --no-21e` | 10 s + 5 s |
| asset library, nothing changed | `assetlib.py` | 4 s |
| asset library, all 121 types forced | `assetlib.py --force` | 100 s |
| view one asset | `view.py asset <type>` | ~2 s |
| view one 200 m chunk | `view.py chunk <s0>` | ~7 s |

## 1. Build dirs: `build/` vs `build_quick/`

- `build/` belongs to the full build and to the other agents (21d / FFB-safety work). **Quick tools never write
  to it.** `quick.py` and `quick_collision.py` refuse to run on it.
- `build_quick/` is a working copy. `build.py quick-seed` copies the 21-stage outputs of `build/` (geometry.pkl and
  atlases) under build/'s lock, saves the seed's 21d collision meshes, and runs **one** full 22 there. That full 22
  run also writes `build_quick/kn5_cache/`:
  - `state.pkl`: every material and texture after albedo calibration and the terrain macro atlases
  - `terrain_uv.pkl`: the final terrain UVs and materials (22 remaps them)
  - `kn5_inputs.json`: an input hash per kn5 file (mesh arrays and flags, dummies, materials, texture bytes, the
    `kn5.py` source)
  - `ledger_chunks.json`: a hash of the ledger rows in each 200 m chunk
- Every Phase 2 step accepts `PALOMARES_BUILD_DIR=<dir>` (`geom.BUILD`), so any single step can be re-run on the
  copy, e.g. `PALOMARES_BUILD_DIR=build_quick python3 scripts/p2/21f_far_vegetation.py`.

## 2. Edit

| you change | then run |
|---|---|
| a ledger row (`data/scene/ledger.jsonl`: move/add/remove a prop, terrain or tree edit) | `build.py quick` |
| an asset generator (`scripts/p5/assets.py`, `trees3d.py`, `foliage_photo.py`) | `assetlib.py <type>`, `view.py asset <type>`, then `build.py quick` (ledger types) |
| far-zone parameters (`21f_far_vegetation.py`) | `PALOMARES_BUILD_DIR=build_quick python3 scripts/p2/21f_far_vegetation.py` + `build.py quick --no-21e` |
| road, terrain, near trees, 21b/21c, 21d, textures in 22 | not covered by quick: run that step on `build_quick/`, then `quick --no-21e`; for textures or 21d, run a full build |

## 3. View

```
python3 scripts/p2/view.py asset shrub.coyote_brush          # 4 angles + a GoPro frame where the type is seen
python3 scripts/p2/view.py asset tree3d.oak --frame 120      # pick the reference frame
python3 scripts/p2/view.py chunk 9300 [--len 200] [--dir build_quick]
```

The views are written to `reports/view/<name>.jpg` and `.html` (gitignored). The chunk view shows two driver's-eye
views (right lane, uphill), an overview, and the model rendered from the nearest calibrated GoPro pose
(`data/scene/poses.json`) beside that frame (`assets/gopro/NNN.png`). It falls back to a local Mapillary image if
one has been fetched. The renderer is numpy + PIL: pinhole projection, painter's sort, Lambert shading, colour
sampled from the real diffuse textures, alpha-tested texels dropped. The chunk view reads the materials and
textures exactly as quick 22 writes them (`kn5_quick.load_state`).
UV convention: mesh v is top-relative on the unflipped tile sheet. The stored PNG/DDS is flipped top-bottom, and
`kn5.py` writes -v, so a texel row is `1 - v`. The viewer follows this, so what it shows is what AC samples.

## 4. Quick build

```
python3 scripts/p2/build.py quick                 # chunks with changed ledger rows are detected and listed
python3 scripts/p2/build.py quick --s 9300-9500   # force 21e, report that range
python3 scripts/p2/build.py quick --no-21e        # geometry.pkl already updated: kn5 only
```

Steps:
1. Diff the ledger per 200 m chunk against the last run.
2. Run 21e on `build_quick/geometry.pkl`. It is idempotent and takes about 18 s for all 9,300 rows.
3. `quick_collision.py restore`: 21e re-adds raw per-prop 1WALL boxes, which are dropped, and the seed's 21d
   collision is put back.
4. Run 22 with `PALOMARES_KN5_QUICK=1`. The texture section is skipped and the cache is loaded. The 21-stage atlases
   (ledger, roadside, dressing, race) are re-read. The same kn5-writing code as the full run is executed, but a kn5
   is rewritten only when its input hash changed.
5. Round-trip each rewritten kn5 with `kn5.read_kn5`: vertex limit and textures present.

The finished track folder is in `build_quick/palomares_road/`.

### What quick does not do (use a full build for a release)
- **Collision of edited props.** 21d (terrain passes plus the shared 1WALL slab rebuild) takes 6 minutes and is
  not idempotent: re-running it on a processed pkl changed road, terrain and kerb kn5. Quick keeps the collision
  of the seed, so a newly added wall-type prop gets no collision until the next full build. Visual changes are
  exact.
- The AC folder ini, ui and map files, the AI lines, QA, renders and packaging.
- Texture changes in 22 (the cache is stale). Re-seed or run a full build.
- If terrain topology changed (21 re-run), quick 22 stops with "terrain topology changed": re-seed.

## 5. Verify

- **Quick = full.** After seeding, `quick` (21e + collision restore + quick 22) on an unchanged ledger produced
  **all 17 kn5 byte-identical** (sha1) to the full 22 output. A forced rewrite of all 17 in quick mode was also
  byte-identical. After moving one prop (post_s0055 at s 9470, +2 m), only `palomares_roadside.kn5` was rewritten,
  in 22 s. Reverting the edit restored the original bytes.
- `view.py chunk <s0>` on the changed range. Compare with the GoPro frame.
- For a release: the full build (`build.py package`), then `qa_ffb_safety.py` and the QA in CLAUDE.md section 4.

## Asset library: `assets/<type>/`

`python3 scripts/p2/assetlib.py [types…] [--force] [--list]` builds every type whose **source hash** changed and
skips the rest. The hash covers the generator sources, the catalog entry, and the bytes of the texture used.

| file | content |
|---|---|
| `manifest.json` | type, cls, source_hash, sources, parts (mesh, texture, material, alpha), LODs, verts/tris, GoPro frames, build time |
| `lod0_<part>.npz`, `lod1_<part>.npz` | kn5-ready arrays P, N, UV, T (local frame, metres, z up, base at origin) |
| `collision.json` | collision proxy (generator primitives, or a trunk cylinder / bounding box), with the 21d convention note |
| `preview.png` | 4-angle render, plus a GoPro frame where the type is seen |
| `assets/_textures/*.dds` | shared DXT1/DXT5 mipmapped textures (`ledger.dds`, `tree_trunk.dds`, `tree_leaves.dds`) |
| `assets/_index.json` | one line per type |

There are 121 types: `tree3d.{oak,bay,euc,syc,con,bare}` (trees3d prototypes; LOD1 is the simplified prototype 21
uses beyond 18 m from the road edge) and the 115 geometry-producing ids of `scripts/p5/assets.py` CATALOG
(= `config/asset_catalog.yaml`). The ids `tree.*` are the ledger's tree_add cards, not the 3D species. Tree
textures come from `build_quick/kn5_cache` (22 makes them from the `foliage_photo.py` crops), so seed first. The
`.npz` and `.dds` files are gitignored (rebuilt in about 100 s); manifests, collision and previews are tracked.
**Not done yet:** 21 and 21e still call the generators directly rather than loading `assets/`. The library is the
prepared, hashed and previewed form used for judging and editing assets.

## Two-zone vegetation

- **Near zone (0–60 m, i.e. all of `trees.csv`):** exact LiDAR position, height and crown, from 21 as before.
- **Far zone (60–150 m):** `21f_far_vegetation.py` (in the full build after 21e, before 21d).
  - Tree positions are local maxima of the 3 m-smoothed CHM (`chm_r100_b150.tif`, at least 4 m high), with NAIP
    2022 NDVI of at least 0.05, and outside every LiDAR crown.
  - Height comes from the CHM. Crown radius is 0.32 h, clipped to 2.5–9 m.
  - Species: oak for the less green third (savanna), bay for the rest, euc for tall crowns over 24 m, con for some
    12–24 m crowns.
  - A per-position hash sets the prototype variant (leaf-colour column), ±12 % scale, yaw and lean.
  - Meshes are trees3d LOD-1 prototypes, merged per 500 m per material into the **shared** `tree_trunk` /
    `tree_leaves` materials. `tree_leaves` is already the one ksTree replacement in `ext_config.ini`. No shadows,
    lod_out 1,400 m, no collision.
- **Result on the dev6 build:** 12,226 far trees (bay 6,424, oak 4,078, con 1,140, euc 584), for **+80 draw calls**.
  The worst layout goes from 1,153 to about **1,233, under the 1,300 budget**. 21f runs in about 10 s.

## dev7 assets (A) and Niles Canyon kerbs (K)

- `scripts/p5/assets_dev7.py` registers the 59 dev7 catalog types (+ `rail.track`) into `assets.CATALOG` with
  material **ledger7** (own 4096² atlas, `assets/_textures/ledger7.dds`) and small emissive parts
  (`beacon_a/b`, `xing_a/b`, `sig_r/y/g`, `lamp_glow`). `generate_parts(aid, L, params)` returns `{material: MB}`;
  `prefill_atlas(rows)` must run on all `ledger_dev7.jsonl` rows before the atlas is written (sign faces and shop
  names are painted lazily). `csp_snippet()` gives the ext_config lines that make the beacons / crossing lights
  wig-wag (CSP `CONDITION` with `FLASHING_FREQUENCY`, as in acc-extension-config `tracks/shibuya.ini`) and the
  lamp globes glow at night. Without CSP the lenses are plain coloured.
- Sources: `scripts/p5/tex_dev7.py --fetch` (ambientCG + Poly Haven CC0, into `data/raw/dev7/`). Every manifest
  records `sources` and `licence`.
- `python3 scripts/p2/assetlib.py --dev7 [types]` builds them (about 5 s per type incl. the textured preview;
  run in batches of about 10 to stay under 2 minutes). `view.py asset <type>` uses a textured renderer for dev7
  types and shows the ledger row's Mapillary images. `python3 scripts/p7k/assets_sheet.py` writes
  `reports/dev7/assets.html`.
- Frames: bridges over the road span along local y centred on the anchor (yaw to the railway bearing);
  `building.town_generic` geometry is relative to the footprint centroid (place there with yaw 0);
  storefronts face -y (road side) with frontage `s1 - s`.
- Kerbs: `python3 scripts/p7k/canyon_kerbs.py` (about 6 s) places Niles Canyon apex/exit kerbs, green run-off
  and yellow sausage kerbs from the racing line (cache `data/scene/cache/racing_line_niles_canyon_road.pkl`),
  writes `data/scene/kerbs_dev7/{placements.json,<layout>.npz}` and exits 1 if any vertex lies inside any dev7
  road's `outer` (the on-pavement lint) or a crease exceeds 25 deg. Rules in `config/kerbs.yaml: niles_canyon`.
  `scripts/p7k/view_kerbs.py <layout>` renders one run-off.
- Palomares kerb paint: `config/kerbs.yaml: style.palomares` (red / yellow / green blocks); `race_tex.kerb_atlas`
  and `kerbs.py` take the block count from there. Geometry unchanged.
