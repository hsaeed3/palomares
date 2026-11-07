"""Litmus test, step 1 (dev5): pick the litmus frames from the registered poses data/scene/poses.json and write
config/litmus_poses.json (three.js/AC coords [x_local, z, -y_local], same as 24_visual_export.py) for 27_litmus.js
and 28_litmus_sheet.py.  Replaces the old fit of 27_litmus_fit.py (kept for its lens/undistort helpers).

Selection (stable, so later builds compare like for like): frames with conf >= 0.6, not in the known-misaligned
list, s < S_MAX-30; the road 0..S_MAX is split into N equal station bins and one frame per bin is drawn with
numpy default_rng(SEED).  Pass --frames to pin a list explicitly.
Usage: python3 27_litmus_poses.py [--n 8] [--seed 5] [--frames 3 40 ...]
"""
import argparse, importlib.util, json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fit", pathlib.Path(__file__).with_name("27_litmus_fit.py"))
fit = importlib.util.module_from_spec(spec); spec.loader.exec_module(fit)
MISALIGNED = {26, 62, 63, 64, 66, 70, 71, 72, 76, 106, 138, 178, 180, 186, 187}
MIN_CONF = 0.6

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--frames", type=int, nargs="*"); a = ap.parse_args()
    src = json.load(open(ROOT / "data/scene/poses.json")); L = json.load(open(ROOT / "config/camera.json"))
    smax = float(fit.S[-1])
    elig = [p for p in src["poses"] if p["conf"] >= MIN_CONF and p["frame"] not in MISALIGNED and p["s"] < smax - 30]
    if a.frames:
        chosen = [p for p in src["poses"] if p["frame"] in set(a.frames)]
    else:
        rng = np.random.default_rng(a.seed); chosen = []
        for b in range(a.n):
            lo, hi = smax * b / a.n, smax * (b + 1) / a.n
            cand = [p for p in elig if lo <= p["s"] < hi]
            if cand: chosen.append(cand[int(rng.integers(len(cand)))])
    out = []
    for p in sorted(chosen, key=lambda p: p["frame"]):
        q = [p["s"], p["lateral"], p["height"], np.radians(p["yaw"]), np.radians(p["pitch"]), np.radians(p["roll"])]
        eye, look, up = fit.to_three(q)
        out.append(dict(frame=p["frame"], s=p["s"], lateral_m=p["lateral"], height_m=p["height"], yaw_rel_deg=p["yaw"],
                        pitch_deg=p["pitch"], roll_deg=p["roll"], eye=eye, look=look, up=up, fov=L["render_vfov_deg"],
                        conf=p["conf"], score=p["score"], road_iou=p["road_iou"], pinned=p["pinned"]))
    json.dump(dict(note="three.js/AC coords [x_local, z, -y_local]; fov = vertical deg of the pinhole render; photo 2042x1486 is "
                        "resampled to that pinhole by 28_litmus_sheet.py with config/camera.json (docs/phase2/CAMERA.md). Poses from data/scene/poses.json (dev5, "
                        "Mask2Former-registered, some pinned by eye); conf/score/road_iou copied from there.",
                   source="data/scene/poses.json", lens=dict(model=L["model"], f_px=L["f_px"], dcx=L["dcx"], dcy=L["dcy"],
                   k1=L["k1"], height_m=L["height_m"]), fov=L["render_vfov_deg"], aspect=L["ref_size"][0] / L["ref_size"][1], size=L["ref_size"], seed=a.seed,
                   n=a.n, min_conf=MIN_CONF, excluded=sorted(MISALIGNED), n_eligible=len(elig), poses=out),
              open(ROOT / "config/litmus_poses.json", "w"), indent=1)
    print("wrote config/litmus_poses.json:", [(p["frame"], round(p["s"]), p["conf"]) for p in out])
