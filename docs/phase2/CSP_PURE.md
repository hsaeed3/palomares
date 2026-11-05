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
