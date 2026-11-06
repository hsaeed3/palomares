"""dev5 per-frame camera pose registration -> data/scene/poses.json

  python3 scripts/p5/register.py                 # all frames (3 workers), then monotone pass
  python3 scripts/p5/register.py --frames 99-110 # refit some frames, merged into poses.json
  python3 scripts/p5/register.py --frames none   # only re-run the monotone pass
  python3 scripts/p5/register.py --set 104=5560  # manual station pin (by eye) for a frame; refits only pose around it

Method (kept simple on purpose):
 1. prior s0 per frame = old data/derived/gopro_stations_*.csv guess (landmark anchors in 27_litmus_fit.ANCH pinned);
    search window +-40 m (csv conf high), +-100 (med), +-150 (low).
 2. candidates: coarse road-ribbon IoU grid (27_litmus_fit.score) over the window; best per 25 m bin, top 4 bins +
    the csv pose; each refined by 27's Nelder-Mead on (s, lateral, height, yaw, pitch, roll) within +-12 m.
 3. each candidate judged by the full class-ID render (scene.render, 256x186) vs seg on stable classes
    (sky/horizon silhouette from DTM + backdrop, road, vegetation, terrain) + the road/yellow fit term
    - prior penalty (0.002 per m beyond 10 m from the csv guess: a 50 m move needs +0.08 score); the best wins.  This is what breaks the "every straight looks alike" aliasing.
 4. monotone pass: weighted isotonic regression of s over frame number; frames > 20 m off it are refit inside
    [prev, next].  Manual pins (data/scene/pose_pins.json) override everything.
 conf = 0..1 from road IoU, full-render score and the margin to the runner-up candidate.
Lens (f_rel, lam) and FOV are the global fit in config/litmus_poses.json (fixed).
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as SC
import score as SCR

ROOT = SC.ROOT
_spec = importlib.util.spec_from_file_location("fit27", ROOT / "scripts/p2/27_litmus_fit.py")
F = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(F)
F.S_MAX = SC.S_MAX - 5                     # whole road
CAM = SC.LENS
RW, RH = 256, 186
PINS = SC.SCENE / "pose_pins.json"
OUTP = SC.SCENE / "poses.json"
ANCHOR_FRAMES = dict(F.ANCH)


def seg_for_fit(seg):
    """27-style (road mask, yellow distance) at 256x186 in the distorted photo from a class-id seg"""
    from scipy import ndimage
    s = np.asarray(Image.fromarray(seg).resize((F.W, F.H), Image.NEAREST))
    m = np.isin(s, [2, 3, 4]); m[: int(F.H * 0.25)] = False
    yd = ndimage.distance_transform_edt(~ndimage.binary_dilation(s == 3, iterations=1))
    return m, yd


def full_score(p, seg_u, valid):
    ren, _ = SC.render(p, RW, RH, ahead=300)
    sc = []
    for ids in ([1], [2, 3, 4], [9, 10, 11], [5, 6, 7, 8]):
        a = np.isin(ren, ids) & valid; b = np.isin(seg_u, ids) & valid
        u = (a | b).sum()
        if u > 50: sc.append((a & b).sum() / u)
    return float(np.mean(sc)) if sc else 0.0


def register_frame(f, prior, win, pin=None, lo=None, hi=None):
    t0 = time.time()
    seg, src = SCR.load_seg(f)
    ph = seg_for_fit(seg)
    seg_u, valid = SC.undistort(seg, order=0, ow=RW, oh=RH)
    valid &= ~np.isin(seg_u, list(SC.IGNORE))
    s0 = float(pin if pin is not None else prior["s"])
    if pin is not None: win = 15
    a = max(s0 - win, 7, lo if lo is not None else -1e9); b = min(s0 + win, SC.S_MAX - 10, hi if hi is not None else 1e9)
    if b <= a: a, b = max(min(a, b) - 5, 7), max(a, b) + 5
    grid = []
    for s in np.arange(a, b + 1e-6, 3.0):
        for lat in (-0.5, 0.5, 1.5):
            for yaw in (-0.10, 0, 0.10):
                q = np.array([s, lat, 1.45, yaw, 0.06, 0.0]); grid.append((F.score(q, CAM, ph)[2], q))
    best = {}
    for v, q in grid:
        k = int(q[0] // 25)
        if k not in best or v < best[k][0]: best[k] = (v, q)
    starts = [q for _, q in sorted(best.values(), key=lambda t: t[0])[:4]]
    pc = SCR.pvec(prior)
    if a <= pc[0] <= b: starts.append(pc)
    cands = []
    for q0 in starts:
        r = F.fit(f, CAM, float(q0[0]), win=12, ph=ph)
        q = np.array(r["p"])
        q[0] = np.clip(q[0], a, b)
        fs = full_score(q, seg_u, valid)
        tot = fs + 0.5 * r["iou"] - 0.25 * min(r["yellow_res_px"] / 480, 1) - 2e-3 * max(abs(q[0] - s0) - 10, 0)
        cands.append(dict(p=q, iou=r["iou"], yres=r["yellow_res_px"], full=fs, tot=tot))
    cands.sort(key=lambda c: -c["tot"]); c = cands[0]
    alt = next((d for d in cands[1:] if abs(d["p"][0] - c["p"][0]) > 15), None)
    margin = c["tot"] - alt["tot"] if alt else 0.2
    conf = float(np.clip(0.4 * (c["iou"] - 0.6) / 0.3 + 0.3 * (c["full"] - 0.3) / 0.4 + 0.3 * margin / 0.1, 0, 1))
    s, lat, h, yaw, pit, rol = c["p"]
    return dict(frame=f, s=round(float(s), 1), lateral=round(float(lat), 2), height=round(float(h), 2),
                yaw=round(float(np.degrees(yaw)), 2), pitch=round(float(np.degrees(pit)), 2), roll=round(float(np.degrees(rol)), 2),
                fov_v=SC.FOV, lens=dict(f_rel=CAM[0], lam=CAM[1]), conf=round(conf, 3), score=round(c["full"], 4),
                road_iou=round(float(c["iou"]), 3), yellow_res_px=round(float(c["yres"]), 1), margin=round(float(margin), 3),
                alt_s=round(float(alt["p"][0]), 1) if alt else None, s_old=prior["s"], seg=src, pinned=pin is not None,
                t_s=round(time.time() - t0, 1))


def _job(args):
    try: os.nice(5)
    except OSError: pass
    return register_frame(*args)


def isotonic(y, w):
    """weighted pool-adjacent-violators, non-decreasing"""
    blocks = []
    for yi, wi in zip(y, w):
        blocks.append([yi * wi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop(); blocks[-1][0] += b[0]; blocks[-1][1] += b[1]; blocks[-1][2] += b[2]
    out = []
    for sw, ww, n in blocks: out += [sw / ww] * n
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames"); ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    try: os.nice(5)
    except OSError: pass
    SC.build_cache()
    pins = json.load(open(PINS)) if PINS.exists() else {}
    for kv in a.set:
        k, v = kv.split("="); pins[str(int(k))] = float(v)
    if a.set: json.dump(pins, open(PINS, "w"), indent=1)
    old = {}
    import csv, glob
    for fn in sorted(glob.glob(str(ROOT / "data/derived/gopro_stations_*.csv"))):
        for r in csv.DictReader(open(fn)):
            old[int(r["frame"])] = dict(s=float(r["s_m"]), lateral=float(r["lateral_m"]), height=float(r["height_m"]),
                                        yaw=float(r["yaw_rel_deg"]), pitch=float(r["pitch_deg"]), roll=float(r["roll_deg"]), conf=r["confidence"])
    for f, s in ANCHOR_FRAMES.items():
        if f in old: old[f]["conf"] = "high"
    prev = {r["frame"]: r for r in json.load(open(OUTP))["poses"]} if OUTP.exists() else {}
    frames = [] if a.frames == "none" else SCR.parse_frames(a.frames) if a.frames else sorted(old)
    frames = sorted(set(frames) | {int(k) for k in a.set and [kv.split("=")[0] for kv in a.set]})
    W = dict(high=40, med=100, low=150)
    t0 = time.time()
    jobs = [(f, old[f], W.get(old[f]["conf"], 150), pins.get(str(f))) for f in frames if f in old]
    import multiprocessing as mp
    if a.frames == "none": jobs = []
    with mp.Pool(min(a.workers, 3)) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    for r in res: prev[r["frame"]] = r
    # monotone pass over all known frames
    fr = sorted(prev); s = np.array([prev[f]["s"] for f in fr])
    w = np.array([0.05 + prev[f]["conf"] + (5 if prev[f].get("pinned") else 0) for f in fr])
    iso = isotonic(s, w)
    bad = [i for i in range(len(fr)) if abs(s[i] - iso[i]) > 20 and not prev[fr[i]].get("pinned")]
    if bad:
        print("monotone refit:", [fr[i] for i in bad], flush=True)
        good = [i for i in range(len(fr)) if i not in bad]
        jobs = []
        for i in bad:
            lo = max([s[j] for j in good if j < i], default=0.0); hi = min([s[j] for j in good if j > i], default=SC.S_MAX)
            jobs.append((fr[i], old[fr[i]], 150, None, lo, hi))
        with mp.Pool(min(a.workers, 3)) as pool:
            for r in pool.map(_job, jobs, chunksize=1):
                r["monotone_refit"] = True; r["conf"] = round(r["conf"] * 0.7, 3); prev[r["frame"]] = r
    # last resort: frames still out of order are clamped onto the isotonic fit (pose otherwise kept, conf capped)
    fr = sorted(prev); s = np.array([prev[f]["s"] for f in fr])
    w = np.array([0.05 + prev[f]["conf"] + (5 if prev[f].get("pinned") else 0) for f in fr]); iso = isotonic(s, w)
    for i, f in enumerate(fr):
        if abs(s[i] - iso[i]) > 1 and not prev[f].get("pinned"):
            prev[f].update(s_fit=prev[f].get("s_fit", prev[f]["s"]), s=round(float(iso[i]), 1), forced_monotone=True,
                           conf=round(min(prev[f]["conf"], 0.15), 3))
    for f, r in prev.items():
        r["flag_moved_30m"] = abs(r["s"] - r["s_old"]) > 30
    out = [prev[f] for f in sorted(prev)]
    viol = [out[i]["frame"] for i in range(1, len(out)) if out[i]["s"] < out[i - 1]["s"] - 1]
    json.dump(dict(note="local coords; pose relative to centerline tangent at s (lateral>0 = right of centreline, as "
                        "27_litmus_fit.cam_basis); yaw/pitch/roll deg; fov_v = pinhole render fov; lens r=tan(lam*theta)/lam, "
                        "f=f_rel*height. conf 0..1; score = mean IoU sky/road/vegetation/terrain of the 256x186 render vs seg.",
                   lens=dict(f_rel=CAM[0], lam=CAM[1]), fov_v=SC.FOV, monotone_violations=viol, poses=out), open(OUTP, "w"), indent=1)
    c = np.array([r["conf"] for r in out])
    print(f"wrote {OUTP} ({len(out)} frames) in {time.time() - t0:.0f}s; conf quartiles {np.percentile(c, [10, 25, 50, 75, 90]).round(2)}; "
          f"moved>30m {sum(r['flag_moved_30m'] for r in out)}; monotone violations {viol}")


if __name__ == "__main__":
    main()
