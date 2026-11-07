# Trees and spring grass (dev6): research, decision, implementation

User feedback (dev5): grass in game is golden-brown summer, not the green spring of the GoPro frames; tree leaves read
completely flat. The painted procedural leaf cards (scripts/p2/trees3d.py + 22's `_leaf_cell`) were the root cause
for the trees; a khaki terrain macro colour was the root cause for the grass.

## 1. Research

### CSP tree system (`[TREES]`)
Source: CSP docs "Track trees" (written for CSP 0.1.79), https://cup.acstuff.club/docs/csp/tracks/trees
- `ext_config.ini`: `[TREES] LIST_0 = trees/trees.txt`; list at `content/tracks/<track>/extension/trees/trees.txt`.
- Line types: `tree: my_tree.bin; 208.4, -1, -53` (also `pos=..; angle=..; size=..; width=..; color=..`) and
  `configure: size variance = 0.8, 1.2` (keys: size/width/angle/color/brightness variance, fake shadow, seed).
- Alignment: `SURFACE_MESHES` / `SURFACE_MATERIALS` under `[TREES]`. Seasons via `[CONDITION_...]` + `SEASON_AUTUMN_n`.
- Tree models are **`.bin` archives produced only by Content Manager's "Tree converter"** (Content / Tools / Creator
  tools) from an FBX with LOD A/B/C meshes (~10k / 6k / 3k tris), diffuse + normal per material. The docs list no
  CSP-shipped tree library.
- Ambient occlusion / shadows need a **VAO bakery v17+ bake** that writes `compiled_trees.bin`
  (`[TREES] COMPILED_LIST = compiled_trees.bin`), recompiled whenever terrain changes.
- `[LIGHTING] ENABLE_TREES_LIGHTING`, `[WIND] TREES_NORMALIZATION`, `[BASIC] SUPPORTS_WIND`: general options page
  https://github.com/ac-custom-shaders-patch/acc-extension-config/wiki/Tracks-%E2%80%93-General-options
- acc-extension-config (`config/tracks/*.ini`, cloned 2026-11-07): **no track config uses `[TREES]`**; community
  configs instead switch existing mesh-tree materials to `ksTree` (`[SHADER_REPLACEMENT_...] SHADER = ksTree`,
  e.g. ks_nordschleife.ini) and use `[ALIGN_TREES_...] TREE_MATERIALS = shader:ksTree` (la_canyons.ini,
  ks_nordschleife.ini).

**Verdict:** the CSP tree system cannot be produced headless on Linux (closed `.bin` format, Windows CM converter,
Windows VAO bakery) and does nothing without CSP. Not used. If the user wants it later, the path is: we export
FBX tree LODs + `trees.txt` from our LiDAR positions, the user runs CM's tree converter once on Windows.

### Non-CSP fallback on good community tracks
Mesh trees in the SpeedTree style: a wood skeleton (tapered tubes) plus many small (0.5-2 m) alpha-tested cards, each
a *photograph* of a leafy twig, crossed/scattered through the crown volume; vertex normals bent out from the crown
centre so lighting rolls off like a sphere instead of per-card. CSP then treats them as foliage via `ksTree`.
Our geometry already followed this pattern (trees3d.py); what made it look flat was (a) the texture: flat
polygon "leaves" drawn with PIL, no sky gaps or self-shadowing; (b) far too few, too large cards (2.4-4 m).

### CC0 3D tree sources (Poly Haven, api.polyhaven.com, CC0)
Tree models: tree_small_02 (4.6 M polys), island_tree_01/02/03 (1.8-4.8 M), jacaranda_tree (0.3 M),
fir_tree_01 (7.9 M), pine_tree_01 (17 M), fir/pine saplings, searsia_lucida/burchellii (0.8-1.4 M),
shrub_01..04, quiver trees, dead trunks. All far beyond a track budget (target <1 k tris per tree near), and none
is a California oak/bay. Useful parts are their **leaf sprite sheets** (colour + alpha):
- searsia_lucida `Diffuse/Alpha`: small rounded leathery leaves -> **coast live oak**
- island_tree_02 `leaves_diff/alpha`: lanceolate glossy leaves -> **California bay**, narrowed -> **eucalyptus**,
  broadened + lime -> **sycamore**, tiny/narrow -> **cypress** scale foliage
- rejected after inspection: tree_small_02 leaves (baked grid pattern), fir_sapling twigs (UV-island padding,
  no clean sprites), jacaranda (pinnate fern-like, wrong species).
Kenney / Quaternius: not on the network allowlist. Blender is not installed here (Sapling Tree Gen would need it and
still needs leaf textures; it would give us the same thing trees3d.py does).

### GrassFX colour
Source: https://github.com/ac-custom-shaders-patch/acc-extension-config/wiki/Tracks-%E2%80%93-Grass-FX
- Blades whose red and blue channels are black (pure green) **take the colour of the surface under them**; green
  channel = brightness/AO. Non-pure-green texels (flowers) keep their own colour.
- The CSP-shipped atlas `grass_fx/highlands.dds` (8 x 3 grid, ~150 configs use it) is pure-green blades in row 1,
  tall blades + yellow/white/blue/red flower tufts, a purple-flower bush and seed heads in rows 2-3.
- So in-game grass colour == our terrain colour. Measured: the terrain macro maps averaged sRGB (111,115,75)
  (G-R ~ +4: khaki / dry-grass), because terrain_macro.py weighted the summer-ortho-dark corridor to leaf litter.
  The GoPro spring grass measures sRGB ~(107,125,60) (frames 020, 170, 209, 215, open patches).
- `COLOR_FACTOR` (in our old config) is not a documented key: removed.

## 2. Decision
1. **Trees:** keep our LiDAR-placed mesh trees (they work with and without CSP), replace the painted leaf cards
   with **photographic leaf-cluster cards composed from CC0 Poly Haven leaf cut-outs**
   (`scripts/p2/foliage_photo.py`): real leaves laid along a twig, painter-ordered with depth darkening
   (self-shadowing), ragged outline, sky gaps, hard alpha + colour dilation for clean mips. ~2.5x more cards at
   about half the size (oak 170 cards of 1.3-2.0 m near, 40 far), normals bent out from the crown centre.
   Added conifers (Monterey cypress/pine: conical crown, straight leader) and winter-bare sycamores
   (twig cards, visible limbs) at GoPro frames 206-211.
2. **CSP:** one `[SHADER_REPLACEMENT_0FOLIAGE]` for `tree_leaves, shrub` -> `ksTree` (wind, translucency, tree
   lighting). One material for all leaves, one for all bark (shader-cache size; no per-species materials).
3. **Grass:** green the terrain macro (spring bias, green understorey over litter) so the surface colour GrassFX
   copies is spring green; use `grass_fx/highlands.dds` with tall-grass and wildflower groups (yellow mustard /
   buttercup, blue lupine, white, sparse purple) on open ground, fewer on banks, none on litter.

## 3. Without CSP
Vanilla AC: same trees with `ksPerPixelAT` (alpha test 0.5), photographic clusters and bent normals; no wind, no
GrassFX (terrain texture alone carries the green spring colour).
