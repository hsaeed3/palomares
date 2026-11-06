# CSP / Pure: what changed and what to check in game

**Why nothing happened before:** the track shipped with no `extension/ext_config.ini`. Every tree used
plain `ksPerPixelAT`, which CSP renders as an ordinary alpha-tested surface. Without a ksTree mapping
there is no foliage translucency or tree wind, and GrassFX/RainFX have no materials to work on.

**Now:** `palomares_road/extension/ext_config.ini` (CSP only; vanilla AC is unchanged):
- tree_oak/bay/euc and shrub -> `ksTree`, double-sided, alpha ref 0.5, diffuse raised (0.35 -> 0.5/0.55) so sun reads on leaves
- GrassFX on ground, bank, litter; road, shoulder, rock, kerbs and paint occlude it
- RainFX: puddles on road_asphalt and shoulder_dirt, soaking asphalt/dirt/litter/rock, paint as lines
- `[WIND] DYNAMIC_FLAGS=AUTO`, `[LIGHTING] SPECULAR_MULT=1 LIT_MULT=1`

## Check in game (CSP + Pure on)
1. In Content Manager > Settings > Custom Shaders Patch > Tracks, confirm the config is "loaded from track folder".
2. Low sun (morning/evening) under the canopy at any wooded section: leaves glow when backlit, and light shafts
   show through gaps (needs Pure god rays / CSP volumetric lights enabled).
3. Trees cast dappled shadows on the road; tree cards are not black or invisible from either side.
4. With wind > 10 km/h, foliage sways gently, with no flicker or popping.
5. GrassFX tufts on verges and banks, none poking through asphalt or kerbs.
6. Rain weather: asphalt darkens and shows puddles, lines stay visible, no dirt "lakes" on the shoulder.
7. Turn CSP off: the track must look exactly as before.
Report: whether trees look too bright or too dark (we tune PROP ksDiffuse), and any card that went
fully transparent (we tune ksAlphaRef).

## v-next: canopy blocks all light / midday blow-out
- **Solid canopy shadows:** every alpha-tested material had `ksAlphaRef = 0`. With a threshold of 0 every texel passes
  the alpha test, so the shadow pass rendered whole quads (forum report: alpha ref drives the AT shadow cut-out,
  https://assettocorsamods.net/goto/post?id=12493). Now `ksAlphaRef = 0.5` in the kn5 (works without CSP) and again in ext_config.
- Tree card normals had an upward bias of +0.8, which made N·L flat and removed backlight. It is now +0.3 (21).
  Shrubs don't cast shadows.
- **Albedo:** the textures were graded to GoPro exposure (asphalt ~0.33 linear). `albedo_calibrate()` in 22 rescales
  each texture to the target in `config/materials.yaml`. Before/after values are in build/albedo_calibration.json.
- ksAmbient/ksDiffuse: base 0.5/0.5, asphalt 0.5/0.6 (spec 0.15, fresnel max 0.15), trees 0.5/0.5.

Tune in game: albedo targets in config/materials.yaml (asphalt 0.13; raise if shade looks dead, lower if midday is
blown out), ksAlphaRef (0.5; raise for more holes in the crowns), and the tree PROP_1 ksDiffuse in ext_config.

### Night: road decals glowed under headlights (fixed in dev3)
`road_decal` (crack-seal and patch overlays) was listed in RainFX `LINES_MATERIALS`, so CSP treated it as retroreflective road paint, and it had a higher specular than the asphalt. It now sits in `PUDDLES`/`SOAKING` with the asphalt, with specular 0.06 / exp 20. Only `paint_*` count as lines.

## dev-next: terrain macro layer + GrassFX that actually spawns

**Why GrassFX showed nothing:** GrassFX only spawns where the *surface colour* passes its mask, and the defaults expect
green-ish texels with luminance 0.02-0.35 (`MASK_MAIN_THRESHOLD 0.5`, `MASK_RED_THRESHOLD 0.05`, `MASK_MIN/MAX_LUMINANCE
0.02/0.35`; CSP docs "Tracks - Grass FX", and e.g. ks_highlands `MASK_MAX_LUMINANCE 0.105`). Our albedo-calibrated textures sit at
sRGB ~0.4-0.5 and litter/bank are olive-brown, so almost nothing passed. Also, `ground/bank/litter` are reused by the shoulder
strips, so grass would have grown on the shoulder.
No special shader is needed (GrassFX works on any listed material; `USE_MULTILAYER_MASK` / `USE_MULTIMAP_DETAIL_MASK` are optional
extras for ksMultilayer txMask / multimap diffuse alpha). `ORIGINAL_GRASS_MATERIALS =` is left empty on purpose (we have no
ksGrass meshes; an empty value means "hide nothing").

**Now (ext_config.ini, scripts/p2/ext_config.py):**
- `GRASS_MATERIALS` = only terrain `ter_ground_NN, ter_bank_NN, ter_litter_NN`. Masks opened (`MASK_MAIN_THRESHOLD -1`,
  `MASK_MIN/MAX_LUMINANCE -1/1`); density/shape come from configurations: A ground = tall wild spring grass (`SHAPE_SIZE 2.6`,
  untidy), B bank = medium (1.9), slightly patchy, C litter = short sparse tufts (1.4). Blade colour is taken from the surface.
- `OCCLUDING_MATERIALS` = road, fresh asphalt, decals, shoulder strips (ground/bank/litter/shoulder_dirt/road_edge), rock, kerb,
  runoff and all paint, so nothing grows on the road, shoulder, kerbs, rock or markings.
- No track-level tree/bush instancing exists in CSP configs (checked acc-extension-config); trees stay our own meshes.

**Terrain no longer tiles (scripts/p2/terrain_macro.py, 21 terrain material/UV, 22):** grass/bank/litter are
`ksPerPixelMultiMap` per 250 m chunk: `txDiffuse` = unique 1024 px macro map over the chunk (USGS ortho regraded to the spring
GoPro palette: open -> grass, canopy -> litter, plus fbm at 6/20/60 m), alpha 0; `txDetail` = the class texture at ~5-6 m via
`detailUVMultiplier`. AC multiplies detail into diffuse where diffuse alpha is 0 (Kunos car workflow), so mean albedo is kept.
Rock: multimap with a 20 m base tile x 4.65 m detail. Physics names `1GRASS_t*` unchanged; no extra draw calls.

**dev-next-2 fixes (terrain looked pale/flat, still no grass):**
- The macro `txDiffuse` used to be a near-grey *multiplier* (~sRGB 204/206/199) with all the colour in `txDetail`. Anything
  that reads only the diffuse (GrassFX colour/mask sampling, distant mips, the offline renderer) saw pale grey. Now
  `txDiffuse` carries the real spring colour (ortho re-graded to the GoPro palette, `MACRO_VAR` 0.6 of its variation) / 0.8, and
  `txDetail` = class texture x 0.8 / macro reference colour (near neutral: grass a bit greener, bank/litter browner).
  Final colour = macro x detail = same calibrated mean albedo as before. Physics names, UVs, draw calls unchanged.
- GrassFX adjustments now set `USE_MULTIMAP_DETAIL_MASK = 0` and `USE_MULTILAYER_MASK = 0`: our multimap diffuse alpha is 0
  everywhere (= "detail on" for AC), and must never be read as a grass-density mask (alpha 0 = no grass).
  Mask syntax checked against acc-extension-config (`highforce.ini` uses the same opened `-1/0/-1/1` masks,
  `sakina_hill.ini`/`ks_highlands.ini` the `GRASS_FX_ADJUSTMENT_n: MATERIALS / MAP = A` form).
- Offline renderer (24_visual_render.js, 24e): forces alpha 1 on multimap terrain; the alpha-0 macro was writing a=0 into the
  premultiplied canvas, which is what washed the terrain out to near white in reports/visual/edges/f030.jpg.

### In-game GrassFX check (exact)
1. Install the new zip (delete the old `palomares_road*` folder first). Check
   `content/tracks/palomares_road/extension/ext_config.ini` exists and its `[GRASS_FX]` `GRASS_MATERIALS` line lists `ter_ground_00 ...`.
2. Content Manager > Settings > Custom Shaders Patch > **Grass FX**: tick *Active*; set *Grass quality/range* above minimum;
   leave *Use for tracks without config* irrelevant (we ship a config). Also in CSP > *General*: track configs must not be
   disabled (no "Ignore track configs").
3. In a session, CSP's in-game app *Shaders Patch Settings / Grass FX* should show the module active; if CSP's
   *Developer > Track config* shows "loaded from track folder", the config was read.
4. Expect at s 1750, 3300, 5300 (open south slopes): dense knee-high untidy spring grass either side from the shoulder edge out;
   s 9000 / 12500 (canopy): short sparse tufts on the litter verge, medium tufts on cut banks; none on rock faces.
5. Never: grass on asphalt, shoulder strip, kerbs, decals, paint, rock. Report the distance marker if seen.
6. Still no grass anywhere: set `MASK_MAIN_THRESHOLD = -1` in B and C too (only A is fully open); if then still none, report the
   CSP version and a screenshot of the Grass FX settings page (Pure does not replace GrassFX).
7. Too tall/dense: lower SHAPE_SIZE in A/B/C; too uniform: raise MASK_MAIN_THRESHOLD in B/C towards 0.2-0.5.
8. Terrain without CSP: no visible 6 m repeat from the driver's seat; colour patches follow the real meadows/canopy.
