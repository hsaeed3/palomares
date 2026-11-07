# Camera model for the reference frames (assets/gopro/NNN.png)

## What the frames are

These are not raw GoPro frames. Each one is a screenshot of a **360-panorama web viewer**: you can see `···`, collapse and close buttons top-right and an `N` compass bottom-right. The viewer re-projects the panorama **rectilinearly**, so poles and fences stay straight. That means **there is no fisheye or GoPro SuperView distortion to model**. The screenshots are hand-cropped, and every frame has a different size (1986–2066 × 1400–1496 px; only one is 2042×1486). So the principal point is taken as each crop's centre.

## Model (`config/camera.json`)

```
x = w/2 + dcx + f·u·(1 + k1·r²)     y = h/2 + dcy − f·v·(1 + k1·r²)     u,v = camera-plane tangents, r² = u²+v²
f = 956.6 px (screenshot px, shared by all frames)   dcx = dcy = 0   k1 = 0
→ vfov 75.0° at h = 1468, hfov 93.7° at w = 2042.   Render: pinhole, vfov 75°.
mount height 2.10 m (registration bound 1.95–2.25 m)
```

Old model: `f_rel 0.8, lam 1.0` meant pinhole f = 0.8·h ≈ 1175 px, vfov 64°. The old model was also applied to the seg as if it were 2042×1486, ignoring the per-crop size.

## Calibration (`scripts/p5/calib.py`)

**Features.** From `data/scene/seg`, resampled to each screenshot's true size (half resolution): the yellow line (class 3), the white lines (class 4), and the road boundary. The UI corners are masked.

**Model geometry.** The track's own painted-line triangles (scorer cache, densified) and the pavement edge (`geom.Road.outer`).

**Cost.** Symmetric chamfer (model→photo and photo→model), truncated at 25 half-res px, summed over yellow, white and 0.7×edge.

**Per-frame 6-DoF fit.** A station/yaw scan (±15 m for pinned frames, ±30 m otherwise), then Powell. The start is the dev5 pose.

**Joint focal scan** over 40 frames spread along the road, with height free (1.0–2.6 m):

| hfov | 70 | 80 | 87 | 95 | 110 |
|---|---|---|---|---|---|
| mean cost | 32.55 | 32.77 | 32.78 | 33.02 | 36.76 |
| median fitted h (m) | 1.80 | 1.81 | 1.99 | 2.13 | 2.34 |

- **f is weakly observable from lines.** The cost is flat from 70° to 95° because height and pitch compensate.
- **I picked f by eye** on overlays (frames 47, 137, 197): near-field edge lines align better at about 95° than at 70°. That agrees with the viewer's default 75° vertical field (≈ 94° horizontal). It is a judgement, not a sharp minimum.
- **k1:** with poses fixed, ±0.03 changes the cost by under 0.1, so k1 = 0. This is consistent with a rectilinear viewer.
- **dcx/dcy:** any offset is absorbed by yaw/pitch, so it is left at 0.
- **The skyline was tried as a focal cue and rejected.** Near tree crowns dominate it (residuals of 120 px or more).

**Height.** On 12 frames at f = 956.6, forcing h to 1.5–1.6 m made every frame worse: mean cost 38.3 against 33.6 with h free, where the median was 2.08 m. With a pinhole camera, the road's width in the image compared with its depth below the horizon depends only on road width ÷ camera height. So h ≈ 2.1 m holds *given the modelled road width*. Either the camera really was on a roof mast (plausible for a 360 rig), or the model road is about 25 % too wide. The lines alone cannot tell these apart. See "Open issues".

## Per-frame residuals (litmus frames)

- **cost:** the calib cost.
- **fwd:** mean model→photo distance, full-res px, averaged over the yellow, white and edge features.
- **Before:** dev5 pose with the old lens.
- **After:** edge re-registration with the new K.

| frame | s_old | cost before | fwd px before | cost after | fwd px after | h after | Δs (m) |
|---|---|---|---|---|---|---|---|
| 021 | 1312 | 35.9 | 27 | 30.7 | 22 | 2.25 | -11.2 |
| 065 | 3715 | 42.3 | 31 | 38.4 | 29 | 2.17 | -5.4 |
| 078 | 3975 | 45.5 | 40 | 38.5 | 31 | 1.97 | -14.7 |
| 121 | 7640 | 36.4 | 25 | 25.1 | 16 | 2.25 | +3.4 |
| 132 | 8625 | 36.8 | 27 | 31.7 | 25 | 2.10 | +3.9 |
| 158 | 10466 | 37.4 | 30 | 29.6 | 24 | 2.22 | +3.8 |
| 189 | 13188 | 38.3 | 28 | 31.0 | 23 | 2.05 | -5.4 |
| 201 | 13938 | 42.2 | 35 | 30.6 | 22 | 2.14 | -1.7 |

The 40-frame calibration set had a "before" mean cost of 41.7; at hfov 95 with free height it was 33.0. The px numbers include seg noise and far road beyond the model window. Near-field lines in the overlays are within roughly 10–20 px.

## Consumers

- `scripts/p5/scene.py`: `CAMERA`, `cam_map`, and `undistort(..., src_size=)`. The seg is mapped with its screenshot's true size. `LENS` is kept as the equivalent pinhole for `register.py`.
- `scripts/p5/score.py` and `register.py`: pass `src_size`.
- `scripts/p2/27_litmus_poses.py`: fov, size and lens are read from `config/camera.json`.
- `27_litmus.js`: unchanged; it renders a pinhole at `cfg.fov`.
- `28_litmus_sheet.py`: resamples the photo with `scene.undistort` and adds a 50/50 blend panel.

## State and how to continue

- `data/scene/poses.json` now holds **edge-registered poses only for the 8 litmus frames** (`reg: calib_edge`). The other 211 frames still have dev5 poses, which were fitted with the old lens. The pre-change file is backed up as `data/scene/poses_pre_camera.json`.
- To refit the rest, run batches of about 30 frames. Each batch takes about 40 s per frame per worker, so about 10 minutes with 2 workers. The runs are resumable through `<out>.part.jsonl`:
  `nice -n10 python3 scripts/p5/calib.py register --frames <30 ids> --workers 2 --hbox 1.95 2.25 --out data/scene/calib_bN.json`
  then `python3 scripts/p5/calib.py commit --out data/scene/calib_bN.json`.
- `scripts/p5/register.py` (the IoU registrar) was not re-run.

## Open issues

- **065:** the model's left edge line sits about 90 px right of the photo's. This is a lateral, width or edge geometry problem at s ≈ 3710 (a pinned frame).
- **021:** the near-left edge is about 40 px off, the far curve shape differs, and h sits at the 2.25 m bound.
- **Road width or camera height ambiguity.** At h 1.5–1.6 m the model road projects about 25 % too wide. The road's real width, or the edge-line offsets in the build, should be checked against the LiDAR before trusting h = 2.1.
- `build/geometry.pkl` changed during this work (another agent). The cache and the litmus were rebuilt from the 05:23 geometry; the 40-frame scan used the older cache.

## Update: asphalt-edge fix and switch to a 70° view

- **Edge-feature fix.** calib.py's edge feature used `Road.outer` (asphalt plus dirt shoulder, median 8.86 m across). It now uses the asphalt half-widths `Road.el`/`Road.er` through `scene.asphalt_edges()` (median 6.99 m, which matches `width_m` and the DTM flat strip of 6.62 m). The scorer was already correct: class 2 stops at the asphalt edge and the shoulder is class 5.
- **Height after the fix.** With height free (1.2–2.3 m), frame 021 → 2.30 m and frame 065 → 2.07 m (`data/scene/calib_h2.json`). The model's paint is normal: the yellow is at ±0.09 m and the white lines are 0.3–0.45 m inside the asphalt edge, about 5.7 m apart.
- **12-frame focal scan** (height free 1.2–2.6, frames 10 22 32 55 73 101 128 137 150 169 191 214):

| hfov | mean cost | median height |
|---|---|---|
| 60° | 30.15 | 2.01 m |
| 70° | 32.06 | 1.88 m |
| 94° | 33.01 | 2.05 m |

- **Current `config/camera.json`:** 70° was the only view that lowered both cost and height, so it is now in use: f = 1458.14 px, render vfov 53.44°, height 1.88 m (bounds 1.6–2.2).
- **Litmus at 70°** (`data/scene/calib_litmus.json`): mean cost 28.57 (31.95 at 94°). Heights: 021 2.20, 065 1.82, 078 1.91, 121 2.20, 132 2.06, 158 2.20, 189 2.20, 201 2.16. Five of the eight sit at the 2.2 m bound.
- **By eye:** 121, 132, 189 and 201 align. 021, 065, 078 and 158 still show left-edge and far-curve offsets. The result is not clearly better than at 94°.
- **Still open:** focal length and height are only weakly separable with line features. The batches have not been run.
