"""Camera calibration + edge-based re-registration (see docs/phase2/CAMERA.md).

The "GoPro" frames are screenshots of a 360-panorama viewer (UI buttons top-right, compass bottom-right): each is a
RECTILINEAR (pinhole) reprojection, hand-cropped (sizes 1986..2066 x 1400..1496), so the principal point is the crop
centre plus a global offset and the focal length (viewer zoom) is shared, in screenshot pixels.
Model: x = cx + f*u/z*(1 + k1*r^2), r^2 = (u^2+v^2)/z^2;  cx = w/2 + dcx, cy = h/2 + dcy.
Features (from data/scene/seg, mapped to screenshot px): yellow line, white line, road(+marking) boundary.
Model geometry: cached class-3/4 triangles (the track's own painted lines) and the pavement edge (geom.Road.outer).
Cost: symmetric chamfer (truncated 25 px at half resolution).

  python3 scripts/p5/calib.py calib  [--frames ...]   # joint fit of f, dcx, dcy, k1 (+ per-frame poses)
  python3 scripts/p5/calib.py register [--frames ...] # per-frame 6-DoF with fixed intrinsics -> data/scene/poses_cam.json
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image
from scipy import ndimage, optimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as SC

ROOT = SC.ROOT
CAMF = ROOT / "config/camera.json"
TR = 25.0
HLO, HHI = 1.0, 2.6


def feats(f):
    im = Image.open(ROOT / f"assets/gopro/{f:03d}.png"); w, h = im.size
    seg = np.asarray(Image.open(SC.SCENE / f"seg/{f:03d}.png").resize((w // 2, h // 2), Image.NEAREST))
    H, W = seg.shape
    mask_ui = np.zeros_like(seg, bool); mask_ui[: int(H * .09), int(W * .85):] = True; mask_ui[int(H * .9):, int(W * .9):] = True
    road = np.isin(seg, [2, 3, 4]); road = ndimage.binary_opening(road, iterations=2)
    road[: H // 3] = False
    bnd = road & ~ndimage.binary_erosion(road, iterations=1); bnd[-3:] = False; bnd[:, :2] = False; bnd[:, -2:] = False
    out = dict(w=w, h=h, W=W, H=H)
    for k, m in (("y", seg == 3), ("wl", seg == 4), ("e", bnd)):
        m = m & ~mask_ui; m[: H // 3] = False
        out[k + "_dt"] = ndimage.distance_transform_edt(~m).astype(np.float32)
        yy, xx = np.nonzero(m); sel = np.random.default_rng(0).permutation(len(xx))[:1500]
        out[k + "_px"] = np.c_[xx[sel], yy[sel]].astype(np.float32)
    return out


_GEO = {}
def geo(s):
    k = int(s // SC.CHUNK)
    if k not in _GEO:
        Vs, Cs = [], []
        for kk in (k, k + 1):
            V, C = SC.chunk(kk); Vs.append(V); Cs.append(C)
        V = np.concatenate(Vs); C = np.concatenate(Cs)
        P = {}
        for c, n in ((3, "y"), (4, "wl")):
            T = V[C == c].astype(np.float64)
            t = np.linspace(0, 1, 9)[:, None, None]
            P[n] = np.concatenate([(T[:, a] * (1 - t) + T[:, b] * t).reshape(-1, 3) for a, b in ((0, 1), (1, 2), (2, 0))])
            P[n] = np.unique(np.round(P[n], 1), axis=0)
        es, eL, eR = SC.pavement_edges()
        ss = np.arange(max(k * SC.CHUNK - 40, 0), min((k + 2) * SC.CHUNK + 40, SC.S_MAX), 0.5)
        x, y, z = (np.interp(ss, SC.S, a) for a in (SC.X, SC.Y, SC.Z)); i = np.round(ss).astype(int).clip(0, len(SC.S) - 1)
        L = np.interp(ss, es, eL); R = np.interp(ss, es, eR)
        P["e"] = np.concatenate([np.c_[x - SC.TY[i] * L, y + SC.TX[i] * L, z], np.c_[x + SC.TY[i] * R, y - SC.TX[i] * R, z]])
        _GEO[k] = P
    return _GEO[k]


def project(p, K, pts, w, h):
    f, dcx, dcy, k1 = K
    eye, fwd, rt, up = SC.cam_basis(p)
    d = pts - eye; z = d @ fwd; u = d @ rt / np.maximum(z, 1e-3); v = d @ up / np.maximum(z, 1e-3)
    r2 = u * u + v * v; g = 1 + k1 * r2
    return np.c_[w / 2 + dcx + f * u * g, h / 2 + dcy - f * v * g], z


def cost(p, K, F, detail=False):
    P = geo(p[0]); tot, res = 0.0, {}
    for k, wgt in (("y", 1.0), ("wl", 1.0), ("e", 0.7)):
        X = P[k]; eye = SC.cam_basis(p)[0]
        near = np.abs(X[:, 0] - eye[0]) + np.abs(X[:, 1] - eye[1]) < 120
        q, z = project(p, K, X[near], F["w"], F["h"]); q = q / 2
        ok = (z > 3) & (z < 80) & (q[:, 0] >= 0) & (q[:, 0] < F["W"] - 1) & (q[:, 1] >= F["H"] / 3) & (q[:, 1] < F["H"] - 1)
        q = q[ok]
        npx = len(F[k + "_px"])
        if npx < 30 and len(q) < 30: continue
        if len(q) < 10:
            fw, bw = TR, TR
        else:
            fw = np.minimum(F[k + "_dt"][q[:, 1].astype(int), q[:, 0].astype(int)], TR).mean() if npx >= 30 else TR
            # reverse: photo feature pixels -> nearest projected model point
            m = np.zeros((F["H"], F["W"]), bool); m[q[:, 1].astype(int), q[:, 0].astype(int)] = True
            m = ndimage.binary_dilation(m, iterations=1)
            dt = ndimage.distance_transform_edt(~m)
            px = F[k + "_px"].astype(int); bw = np.minimum(dt[px[:, 1], px[:, 0]], TR).mean() if npx >= 30 else fw
        c = 0.5 * (fw + bw); res[k] = (float(fw) * 2, float(bw) * 2)   # in full screenshot px
        tot += wgt * c
    return (tot, res) if detail else tot


LO = np.array([-1e9, -2.5, HLO, -0.4, -0.25, -0.12]); HI = np.array([1e9, 3.5, HHI, 0.4, 0.25, 0.12])


def fit_pose(p0, K, F, sw=15.0, hbox=(HLO, HHI), iters=2):
    lo = LO.copy(); hi = HI.copy(); lo[0] = p0[0] - sw; hi[0] = p0[0] + sw; lo[2], hi[2] = hbox
    fn = lambda q: cost(np.clip(q, lo, hi), K, F) + 0.02 * max(abs(q[0] - p0[0]) - 5, 0)
    best = (fn(p0), np.array(p0, float))
    # coarse station / yaw scan first (aliasing), then Powell
    for ds in np.arange(-sw, sw + 1e-6, 3.0):
        for dyaw in (-0.08, 0, 0.08):
            q = best[1].copy() if ds == -sw and dyaw == -0.08 else np.array(p0, float)
            q = np.array(p0, float); q[0] += ds; q[3] += dyaw; v = fn(q)
            if v < best[0]: best = (v, q)
    q = best[1]
    for _ in range(iters):
        o = optimize.minimize(fn, q, method="Powell", options=dict(xtol=1e-3, ftol=1e-3, maxfev=1500,
                              direc=np.diag([2, .3, .1, .03, .02, .02])))
        q = np.clip(o.x, lo, hi)
    return q, cost(q, K, F)


def load_poses():
    return {r["frame"]: r for r in json.load(open(SC.SCENE / "poses.json"))["poses"]}


def pvec(r):
    return np.array([r["s"], r["lateral"], r["height"], *np.radians([r["yaw"], r["pitch"], r["roll"]])])


def _fitjob(a):
    os.nice(5) if hasattr(os, "nice") else None
    f, p0, K, sw, hb = a
    F = feats(f); q, c = fit_pose(p0, K, F, sw, hb, iters=1)
    return f, q, c, cost(q, K, F, True)[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--K", type=float, nargs=4); ap.add_argument("--hbox", type=float, nargs=2, default=[HLO, HHI])
    ap.add_argument("--out"); ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    os.nice(5)
    import multiprocessing as mp
    P = load_poses()
    frames = a.frames or sorted(P)
    K = tuple(a.K) if a.K else tuple(json.load(open(CAMF))["K"]) if CAMF.exists() else (1021 / np.tan(np.radians(40.65)), 0, 0, 0)
    pins = json.load(open(SC.SCENE / "pose_pins.json"))
    jobs = [(f, pvec(P[f]), K, 15.0 if str(f) in pins else 30.0, tuple(a.hbox)) for f in frames if f in P]
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        R = pool.map(_fitjob, jobs, chunksize=1)
    out = {f: dict(p=q.tolist(), cost=float(c), res=r) for f, q, c, r in R}
    json.dump(dict(K=K, hbox=a.hbox, frames=out), open(a.out, "w"), indent=1)
    cs = [v["cost"] for v in out.values()]
    print(f"K={K} n={len(cs)} cost mean {np.mean(cs):.2f} median {np.median(cs):.2f}  h median "
          f"{np.median([v['p'][2] for v in out.values()]):.2f}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()


def overlay(f, p, K, path, extra=None):
    """photo (half res) + projected model yellow (red), white (cyan), pavement edge (magenta); seg features faint"""
    from PIL import ImageDraw
    F = feats(f); im = Image.open(ROOT / f"assets/gopro/{f:03d}.png").convert("RGB").resize((F["W"], F["H"]))
    d = ImageDraw.Draw(im); P = geo(p[0]); eye = SC.cam_basis(p)[0]
    for k, col in (("y", (255, 0, 0)), ("wl", (0, 255, 255)), ("e", (255, 0, 255))):
        X = P[k]; near = np.abs(X[:, 0] - eye[0]) + np.abs(X[:, 1] - eye[1]) < 200
        q, z = project(p, K, X[near], F["w"], F["h"]); q = q[(z > 2) & (z < 150)] / 2
        for x, y in q:
            d.point((x, y), fill=col)
    im.save(path, quality=88)
