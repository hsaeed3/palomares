"""Litmus test, step 1: fit GoPro frame -> camera pose on the centerline geometry.

Frame -> station: landmark anchors (survey A/B + features.geojson) interpolated, then per frame a
coarse search over station + Nelder-Mead over (s, lateral, height, yaw, pitch, roll) maximising
IoU of the projected road ribbon vs. the road segmented in the photo, plus a yellow-centre-line term.
FOV is estimated once over several frames and then fixed.

Writes config/litmus_poses.json in the AC/three.js coords used by 24_visual_export.py
(three = [x_local, z, -y_local]).  Usage: python3 27_litmus_fit.py [--n 8] [--seed 7]
"""
import argparse, csv, json, pathlib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, optimize

ROOT = pathlib.Path(__file__).resolve().parents[2]
W, H = 256, 186            # analysis resolution (photo is 2042x1486)
S_MAX = 7392.0             # Phase 2 summit
# frame -> station anchors (frame 1 = SR-84; 54 first bridge s~3509 (LiDAR cl.17); 91 bridge_osm + junction 4388;
# 137/138 bridges 9193/9307; 171 bridge 11344; 219 Palo Verde Rd 15586)
ANCH = [(1, 0.0), (54, 3509.0), (91, 4388.0), (137, 9199.0), (138, 9313.0), (171, 11352.0), (219, 15586.0)]

rows = list(csv.DictReader(open(ROOT / "data/derived/centerline.csv")))
CL = np.array([[float(r[k]) for k in ("s_m", "x_local", "y_local", "z_navd88", "edge_left_m", "edge_right_m")] for r in rows])
S, X, Y, Z, EL, ER = CL.T
EL = EL - 0.5; ER = ER - 0.5   # pavement edge -> painted edge line (shoulder litter is not segmented as road)
TX, TY = np.gradient(X), np.gradient(Y); n = np.hypot(TX, TY); TX /= n; TY /= n


def anchor_s(f):
    fa, sa = zip(*ANCH); return float(np.interp(f, fa, sa))


def seg_photo(f):
    a = np.asarray(Image.open(ROOT / f"assets/gopro/{f:03d}.png").convert("RGB").resize((W, H)), float) / 255
    mx, mn = a.max(-1), a.min(-1); sat = (mx - mn) / (mx + 1e-6); v = mx
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    yellow = (sat > 0.30) & (r > 0.40) & (g > 0.30) & (b < 0.6 * g) & (r > g * 0.9)
    white = (sat < 0.18) & (v > 0.75)
    grey = (sat < 0.14) & (v > 0.10) & (v < 0.80)
    m = (grey | yellow | white)
    m[: int(H * 0.30)] = False
    m[int(H * 0.92):, int(W * 0.9):] = False; m[: int(H * 0.08), int(W * 0.85):] = False
    hue = np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360
    yel_soft = (hue > 35) & (hue < 65) & (sat > 0.42) & (v > 0.5)
    yel_line = (hue > 30) & (hue < 62) & (sat > 0.22) & (v > 0.55)
    m = ndimage.binary_opening(m, iterations=1) | ndimage.binary_dilation(yel_soft, iterations=1)
    m[: int(H * 0.30)] = False
    m = ndimage.binary_closing(m, iterations=1)
    lab, nl = ndimage.label(m)
    seeds = lab[H - 6:H - 1, int(W * .25):int(W * .75)].ravel(); seeds = seeds[seeds > 0]
    if len(seeds):
        m = lab == np.bincount(seeds).argmax()
    m = ndimage.binary_fill_holes(np.vstack([m, np.ones((1, W), bool)]))[:-1]
    m = ndimage.binary_closing(m, structure=np.ones((1, 9)))
    ydist = ndimage.distance_transform_edt(~ndimage.binary_dilation(yel_line & m, iterations=1))
    return m, ydist


def cam_basis(p):
    s, lat, h, yaw, pit, rol = p
    i = int(np.clip(round(s), 0, len(S) - 2))
    tx, ty = TX[i], TY[i]
    eye = np.array([np.interp(s, S, X) + ty * lat, np.interp(s, S, Y) - tx * lat, np.interp(s, S, Z) + h])
    hd = np.arctan2(ty, tx) + yaw
    fwd = np.array([np.cos(hd) * np.cos(pit), np.sin(hd) * np.cos(pit), np.sin(pit)])
    rt = np.array([np.sin(hd), -np.cos(hd), 0.0]); up = np.cross(rt, fwd)
    c, sn = np.cos(rol), np.sin(rol); rt, up = c * rt + sn * up, -sn * rt + c * up
    return eye, fwd, rt, up


def project(p, cam, pts):
    """cam = (f_rel, lam): focal / image height and the 'fov-model' fisheye factor
    (r_d = tan(lam*theta)/lam; lam=1 pinhole, lam->0 equidistant).  GoPro wide is strongly barrel-distorted."""
    eye, fwd, rt, up = cam_basis(p)
    d = pts - eye; zc = d @ fwd
    u, w = d @ rt, d @ up
    th = np.arctan2(np.hypot(u, w), zc)                     # angle off axis (works behind the plane too)
    f, lam = cam[0] * H, cam[1]
    rd = np.tan(np.minimum(lam * th, 1.45)) / lam
    ph = np.arctan2(w, u)
    return np.stack([W / 2 + f * rd * np.cos(ph), H / 2 - f * rd * np.sin(ph)], -1), np.where(th < np.radians(80), zc, -1)


def render(p, cam):
    i0 = int(max(p[0] - 6, 0)); i1 = int(min(p[0] + 160, len(S) - 1))
    sl = slice(i0, i1)
    L = np.stack([X[sl] - TY[sl] * EL[sl], Y[sl] + TX[sl] * EL[sl], Z[sl]], -1)
    R = np.stack([X[sl] + TY[sl] * ER[sl], Y[sl] - TX[sl] * ER[sl], Z[sl]], -1)
    C = np.stack([X[sl], Y[sl], Z[sl] + 0.01], -1)
    pl, zl = project(p, cam, L); pr, zr = project(p, cam, R); pc, zc = project(p, cam, C)
    im = Image.new("L", (W, H)); dr = ImageDraw.Draw(im)
    for k in range(len(L) - 1):
        if min(zl[k], zr[k], zl[k + 1], zr[k + 1]) > 0.3:
            dr.polygon([tuple(pl[k]), tuple(pr[k]), tuple(pr[k + 1]), tuple(pl[k + 1])], fill=1)
    ok = (zc > 2.0) & (zc < 35) & (pc[:, 0] >= 0) & (pc[:, 0] < W) & (pc[:, 1] >= 0) & (pc[:, 1] < H)
    return np.asarray(im, bool), pc[ok]


def score(p, cam, ph):
    m, yd = ph
    r, pc = render(p, cam)
    iou = (r & m).sum() / max((r | m).sum(), 1)
    ych = np.mean(np.minimum(yd[pc[:, 1].astype(int), pc[:, 0].astype(int)], 60)) / 60 if len(pc) > 3 else 3
    return iou, ych, -(iou - 0.5 * ych)


LO = np.array([-1e9, -0.8, 1.0, -0.30, -0.20, -0.07]); HI = np.array([1e9, 3.2, 1.7, 0.30, 0.10, 0.07])


def fit(f, cam, s0, win=None, ph=None):
    """win=None: search the whole Phase-2 road; frame-number prior only as a weak tie-break (0.03 per km)"""
    ph = ph or seg_photo(f)
    best = []
    lo, hi = (7, S_MAX - 10) if win is None else (max(s0 - win, 7), min(s0 + win, len(S) - 170))
    for s in np.arange(lo, hi, 4.0 if win else 6.0):
        for lat in (0.3, 1.5):
            for yaw in (-0.12, 0, 0.12):
                p = np.array([s, lat, 1.35, yaw, -0.05, 0.0]); best.append((score(p, cam, ph)[2] + 0.03e-3 * abs(s - s0), p))
    best.sort(key=lambda t: t[0]); res = []
    for _, p0 in best[:4]:
        fn = lambda q: score(np.clip(q, LO, HI), cam, ph)[2] + 0.01 * abs(q[1] - 1.5) + 0.05 * abs(q[2] - 1.35)
        o = optimize.minimize(fn, p0, method="Nelder-Mead",
                              options=dict(initial_simplex=np.vstack([p0, p0 + np.diag([4, .5, .15, .05, .03, .03])]), maxiter=500, xatol=1e-3, fatol=1e-4))
        q = np.clip(o.x, LO, HI); res.append((score(q, cam, ph), q))
    res.sort(key=lambda t: t[0][2])
    (iou, ych, _), q = res[0]
    return dict(frame=f, p=q.tolist(), iou=float(iou), yellow_res_px=float(ych * 60 * 2042 / W))


def to_three(p):
    eye, fwd, rt, up = cam_basis(p)
    t = lambda v: [float(v[0]), float(v[2]), float(-v[1])]
    return t(eye), t(eye + fwd * 20), t(up)


def edge_angles(cam, w=2042, h=1486):
    """half-angles (deg) to the photo's top edge and side edge"""
    f = cam[0] * h; lam = cam[1]
    a = lambda r: np.degrees(np.arctan(lam * r / f) / lam)
    return a(h / 2), a(w / 2)


def undistort(img, cam, vfov, out_wh):
    """resample the fisheye photo to a pinhole view with vertical fov `vfov` (deg), size out_wh"""
    a = np.asarray(img, np.float32); h, w = a.shape[:2]; ow, oh = out_wh
    fo = (oh / 2) / np.tan(np.radians(vfov) / 2)
    yy, xx = np.mgrid[0:oh, 0:ow]; u = (xx + .5 - ow / 2) / fo; v = (yy + .5 - oh / 2) / fo
    ru = np.hypot(u, v); th = np.arctan(ru); f = cam[0] * h; lam = cam[1]
    rd = f * np.tan(lam * th) / lam; sc = np.where(ru > 0, rd / np.maximum(ru, 1e-9), f)
    sx = w / 2 + u * sc - .5; sy = h / 2 + v * sc - .5
    out = np.stack([ndimage.map_coordinates(a[..., c], [sy, sx], order=1, cval=0) for c in range(3)], -1)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-iou", type=float, default=0.80); ap.add_argument("--max-yellow", type=float, default=40.0)
    ap.add_argument("--cam", type=float, nargs=2, help="skip calibration: f_rel lam"); a = ap.parse_args()
    # frames are hand-picked screenshots (irregular spacing): anchors are only a prior; bridge at s~9199 is frame 137,
    # so frames 4..125 may lie below the summit; the fitted station decides (must be < S_MAX - 30)
    elig = list(range(4, 126))
    rng = np.random.default_rng(a.seed)
    if a.cam:
        CAM = tuple(a.cam); calib = None
    else:
        # lens: one global (f_rel, lam), grid search on 6 calibration frames, then fixed for all frames
        cal = sorted(rng.choice(elig, 6, replace=False).tolist()); phs = {f: seg_photo(f) for f in cal}
        loc = {f: fit(f, (0.7, 1.0), anchor_s(f), ph=phs[f])["p"][0] for f in cal}; print("calib stations", loc, flush=True)
        grid = []
        for lam in (1.0, 0.8, 0.6):
            for fr in (0.55, 0.65, 0.75, 0.85, 0.95):
                rs = [fit(f, (fr, lam), loc[f], win=60, ph=phs[f]) for f in cal]
                sc = np.mean(sorted(r["iou"] - 0.5 * r["yellow_res_px"] / 480 for r in rs)[1:])
                grid.append((sc, fr, lam)); print(f"f_rel {fr} lam {lam}: score {sc:.3f} IoU {np.mean([r['iou'] for r in rs]):.3f}", flush=True)
        _, fr, lam = max(grid)
        o = optimize.minimize(lambda q: -np.mean(sorted(r["iou"] - 0.5 * r["yellow_res_px"] / 480 for r in
                              [fit(f, (q[0], q[1]), loc[f], win=40, ph=phs[f]) for f in cal])[1:]),
                              [fr, lam], method="Nelder-Mead", options=dict(initial_simplex=[[fr, lam], [fr + .05, lam], [fr, lam + .1]], maxiter=14))
        CAM = (float(o.x[0]), float(min(max(o.x[1], 0.05), 1.0))); calib = dict(frames=cal, grid=[list(map(float, g)) for g in grid])
    va, ha = edge_angles(CAM); print("lens f_rel=%.3f lam=%.3f -> photo FOV %.1f deg vertical, %.1f deg horizontal" % (CAM[0], CAM[1], 2 * va, 2 * ha))
    VFOV = round(2 * va, 1)   # pinhole render fov; the photo is undistorted to this (corners letterboxed)
    order = rng.permutation(elig).tolist(); chosen, rejected = [], []
    for f in order:
        if len(chosen) == a.n: break
        r = fit(f, CAM, anchor_s(f)); print(f, round(anchor_s(f)), round(r["p"][0], 1), round(r["iou"], 3), round(r["yellow_res_px"], 1), flush=True)
        if r["iou"] >= a.min_iou and r["yellow_res_px"] <= a.max_yellow and r["p"][0] < S_MAX - 30: chosen.append(r)
        else: rejected.append(dict(frame=f, s=round(r["p"][0], 1), iou=round(r["iou"], 3), yellow_res_px=round(r["yellow_res_px"], 1)))
    out = []
    for r in sorted(chosen, key=lambda r: r["frame"]):
        s, lat, h, yaw, pit, rol = r["p"]; eye, look, up = to_three(r["p"])
        out.append(dict(frame=r["frame"], s=round(s, 1), s_anchor=round(anchor_s(r["frame"]), 1), lateral_m=round(lat, 2), height_m=round(h, 2),
                        yaw_rel_deg=round(np.degrees(yaw), 2), pitch_deg=round(np.degrees(pit), 2), roll_deg=round(np.degrees(rol), 2),
                        eye=eye, look=look, up=up, fov=VFOV, iou=round(r["iou"], 3), yellow_res_px=round(r["yellow_res_px"], 1)))
    json.dump(dict(note="three.js/AC coords [x_local, z, -y_local]; fov = vertical deg of the pinhole render; photo 2042x1486 is "
                        "undistorted to that pinhole by 28_litmus_sheet.py using lens (f_rel=focal/height, lam: r=tan(lam*theta)/lam); "
                        "fit = IoU of road mask (analysis 256x186) + yellow centre-line residual (mean, full-res px, capped 480)",
                   lens=dict(f_rel=round(CAM[0], 4), lam=round(CAM[1], 4), vfov_photo=round(2 * va, 1), hfov_photo=round(2 * ha, 1)),
                   fov=VFOV, aspect=2042 / 1486, size=[2042, 1486], seed=a.seed, min_iou=a.min_iou, max_yellow_px=a.max_yellow,
                   calibration=calib, rejected=rejected, poses=out),
              open(ROOT / "config/litmus_poses.json", "w"), indent=1)
    print("wrote config/litmus_poses.json", len(out), "poses; rejected", [r["frame"] for r in rejected])
