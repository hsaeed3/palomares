#!/usr/bin/env python3
"""dev7 G drivability QA on the BUILT kn5 of one network road (or several): steps, holes, grade, overlaps.

    PALOMARES_BUILD_DIR=build_dev7 python3 scripts/p7g/qa_net.py niles_canyon_road [--kn5 'palomares_niles_ncr*.kn5']

Physics surfaces (1ROAD / 1GRASS / 1KERB / 1SIDEWALK / 1RAIL ...) are ray-cast from above on a road-frame grid
(every 0.5 m along, 0.25 m across, out to the shoulder + 3 m).  Reported:
  holes     : samples with no physics surface (inside the drivable band = asphalt + shoulder)
  steps     : |dz| between neighbouring samples beyond what the local slope explains (> 2 cm)
  overlaps  : two physics surfaces under one sample more than 2 cm apart (double surfaces, e.g. Palomares/NCR)
  grade     : centreline grade of the BUILT surface (25 m and 5 m windows), max camber change per metre
  open edges: welded open boundary edges of the physics meshes inside the drivable band
Writes reports/dev7/qa_net_<road>.json, exit 1 on holes / steps / overlaps.
"""
import argparse, json, sys, pathlib, collections
import numpy as np
from scipy.spatial import cKDTree
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0] / "p2"))
import network as NW
from geom import ROOT, BUILD
from kn5 import read_kn5

ap = argparse.ArgumentParser(); ap.add_argument("road"); ap.add_argument("--kn5", default=None)
ap.add_argument("--s", default=None); ap.add_argument("--tag", default=None); ap.add_argument("--extra", default="", help="more kn5 globs (comma) e.g. Palomares")
A = ap.parse_args()
ROADS = A.road.split(",")
TAG = {"niles_canyon_road": "ncr"}.get(ROADS[0], ROADS[0])
TRK = BUILD / "palomares_road"
files = sorted(TRK.glob(A.kn5 or f"palomares_niles_{TAG}*.kn5"))
for g in filter(None, A.extra.split(",")):
    files += sorted(pathlib.Path(g).parent.glob(pathlib.Path(g).name))
P_all, T_all, owner = [], [], []
k0 = 0
for f in files:
    for m in read_kn5(str(f))["meshes"]:
        nm = m["name"]
        if not nm[:1].isdigit() or "WALL" in nm:
            continue
        P = np.c_[m["pos"][:, 0], -m["pos"][:, 2], m["pos"][:, 1]].astype(float)
        P_all.append(P); T_all.append(m["idx"].reshape(-1, 3) + k0); owner += [f"{f.name}:{nm}"] * (len(m["idx"]) // 3); k0 += len(P)
P = np.concatenate(P_all); T = np.concatenate(T_all); owner = np.array(owner)
print(f"{len(files)} kn5, {len(T)} physics tris")
# ---- 2D grid hash of triangles
CELL = 2.0
tmin = P[T][:, :, :2].min(1); tmax = P[T][:, :, :2].max(1)
c0 = np.floor(tmin / CELL).astype(int); c1 = np.floor(tmax / CELL).astype(int)
bins = collections.defaultdict(list)
for t in range(len(T)):
    for cx in range(c0[t, 0], c1[t, 0] + 1):
        for cy in range(c0[t, 1], c1[t, 1] + 1):
            bins[(cx, cy)].append(t)


def cast(xy):
    """all hits (z, tri) per sample: returns list of arrays"""
    out = [[] for _ in range(len(xy))]
    key = np.floor(xy / CELL).astype(int)
    order = collections.defaultdict(list)
    for i, k in enumerate(map(tuple, key)):
        order[k].append(i)
    for k, idx in order.items():
        tl = bins.get(k)
        if not tl:
            continue
        tl = np.array(tl); idx = np.array(idx)
        a, b, c = P[T[tl, 0]], P[T[tl, 1]], P[T[tl, 2]]
        q = xy[idx][:, None, :]
        v0 = (b - a)[None, :, :2]; v1 = (c - a)[None, :, :2]; v2 = q - a[None, :, :2]
        den = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        u = (v2[..., 0] * v1[..., 1] - v1[..., 0] * v2[..., 1]) / den
        v = (v0[..., 0] * v2[..., 1] - v2[..., 0] * v0[..., 1]) / den
        inside = (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
        z = a[None, :, 2] + u * (b - a)[None, :, 2] + v * (c - a)[None, :, 2]
        for r, cidx in zip(*np.nonzero(inside)):
            out[idx[r]].append((z[r, cidx], tl[cidx]))
    return out



def check(rname, s_arg):
    global road, N, TAG
    road = NW.load_road(rname); N = len(road.s); TAG = {'niles_canyon_road': 'ncr'}.get(rname, rname)
    A.s = s_arg
    i0, i1 = 2, N - 3
    _bj = BUILD / "dev7" / f"{TAG}_build.json"
    if not A.s and _bj.exists() and "clusters_clip" in json.load(open(_bj)).get("profile", {}):
        _c = json.load(open(_bj))["profile"]["clusters_clip"]; A.s = f"{_c[0] + 2}-{_c[1] - 2}"   # 21n's built range
    if A.s:
        a_, b_ = (float(v) for v in A.s.split("-")); i0, i1 = int(max(a_, 2)), int(min(b_, N - 3))
    st = np.arange(i0, i1, 0.5)
    ii = np.round(st).astype(int)
    SH = 0.6
    res = dict(road=rname, files=[f.name for f in files], s=[float(st[0]), float(st[-1])])
    holes, steps, overl, kinks, breaks = [], [], [], [], []
    ztop_cl = np.full(len(st), np.nan); zl2 = np.full(len(st), np.nan); zr2 = np.full(len(st), np.nan)
    BLK = 400
    for b0 in range(0, len(st), BLK):
        sl = slice(b0, b0 + BLK); I_ = ii[sl]
        W = np.maximum(road.el[I_], road.er[I_]) + SH
        du = np.arange(-1, 1.0001, 0.25 / 12)           # fraction of half-band; ~0.25 m at 12 m
        D = du[None] * W[:, None]
        # band per side
        band = np.where(D >= 0, road.el[I_, None] + SH, road.er[I_, None] + SH)
        keep = np.abs(D) <= band - 0.05
        fr = st[sl] - I_
        X = road.x[I_, None] + road.tx[I_, None] * fr[:, None] + road.nx[I_, None] * D
        Y = road.y[I_, None] + road.ty[I_, None] * fr[:, None] + road.ny[I_, None] * D
        xy = np.c_[X.ravel(), Y.ravel()]
        hits = cast(xy)
        Z = np.full(len(xy), np.nan)
        for k, h in enumerate(hits):
            if not keep.ravel()[k] or not h:
                continue
            zs = np.array([z for z, _ in h])
            top = zs.max(); Z[k] = top
            lower = zs[zs < top - 0.02]
            if len(lower) and (top - lower.max()) < 0.6:
                overl.append((float(st[sl][k // D.shape[1]]), float(D.ravel()[k]), float(top - lower.max()), owner[[t for z, t in h if z == top][0]]))
        Z = Z.reshape(D.shape)
        miss = keep & ~np.isfinite(Z)
        for r, c in zip(*np.nonzero(miss)):
            holes.append((float(st[sl][r]), float(D[r, c])))
        # steps = discontinuities: a neighbour jump bigger than 2 cm beyond what the adjacent slopes explain
        #   jump_i = |dz_i| - max(|dz_{i-1}|, |dz_{i+1}|) * (h_i / h_nb)   (a slope break is not a step)
        for axis in (0, 1):
            dz = np.diff(Z, axis=axis)
            a_ = np.abs(dz)
            nb = np.full_like(a_, np.nan)
            if axis == 0:
                nb[1:-1] = np.maximum(a_[:-2], a_[2:])      # both neighbours required (band ends are not steps)
            else:
                nb[:, 1:-1] = np.maximum(a_[:, :-2], a_[:, 2:])
            jump = a_ - nb
            for r, c in zip(*np.nonzero(np.isfinite(jump) & (jump > 0.02))):
                # candidate: re-sample the cell at 24 sub-steps (1-2 cm); a real step shows as > 2 cm between two
                # sub-samples, a slope break (asphalt crossfall -> shoulder rise) does not
                r2, c2 = (r + 1, c) if axis == 0 else (r, c + 1)
                t = np.linspace(0, 1, 25)[:, None]
                q = (1 - t) * np.array([X[r, c], Y[r, c]]) + t * np.array([X[r2, c2], Y[r2, c2]])
                zz = np.array([max((z for z, _ in h), default=np.nan) for h in cast(q)])
                jmp = np.nanmax(np.abs(np.diff(zz))) if np.isfinite(zz).sum() > 1 else 0.0
                if jmp > 0.02:
                    steps.append((float(st[sl][r]), float(D[r, c]), float(jmp), axis))
                else:
                    breaks.append(float(jump[r, c]))
        # kinks (info only): second difference along s > 2 cm on the ASPHALT
        d2 = np.abs(Z[2:] - 2 * Z[1:-1] + Z[:-2])
        asp = np.abs(D[1:-1]) < np.where(D[1:-1] >= 0, road.el[I_[1:-1], None], road.er[I_[1:-1], None]) - 0.1
        for r, c in zip(*np.nonzero((d2 > 0.02) & asp)):
            kinks.append((float(st[sl][r + 1]), float(D[r + 1, c]), float(d2[r, c])))
        cc = D.shape[1] // 2
        ztop_cl[sl] = Z[:, cc]
        jl = np.argmin(np.abs(D - 2.0), axis=1); jr = np.argmin(np.abs(D + 2.0), axis=1)
        zl2[sl] = Z[np.arange(len(I_)), jl]; zr2[sl] = Z[np.arange(len(I_)), jr]
    # grade / camber of the built surface
    ok = np.isfinite(ztop_cl)
    g5 = np.gradient(np.convolve(ztop_cl, np.ones(10) / 10, "same"), 0.5)
    w25 = min(50, max(len(st) // 4, 1))
    zc = np.convolve(np.where(ok, ztop_cl, np.nanmean(ztop_cl)), np.ones(w25) / w25, "same")
    g25 = np.gradient(zc, 0.5)[w25:-w25]
    cam = (zl2 - zr2) / 4.0; dcam = np.abs(np.diff(cam[::2]))
    res.update(holes=len(holes), holes_list=holes[:50], steps=len(steps),
               steps_list=sorted(steps, key=lambda t: -t[2])[:50],
               step_max_m=max([t[2] for t in steps], default=0.0),
               slope_breaks_checked=len(breaks),
               asphalt_kinks=len(kinks), asphalt_kinks_list=sorted(kinks, key=lambda t: -t[2])[:30],
               overlaps=len(overl), overlaps_list=overl[:50],
               grade_max_pct_25m=float(np.nanmax(np.abs(g25)) * 100), grade_max_pct_5m=float(np.nanmax(np.abs(g5[10:-10] if len(g5) > 30 else g5)) * 100),
               grade_max_s=float(st[w25:-w25][np.nanargmax(np.abs(g25))]),
               camber_change_max_pct_per_m=float(np.nanmax(dcam) * 100) if len(dcam) else 0.0)
    # open edges of the physics mesh union inside the band (welded at 1 mm)
    _, inv = np.unique(np.round(P, 3), axis=0, return_inverse=True); W_ = inv.ravel()[T]
    E = np.sort(np.r_[W_[:, [0, 1]], W_[:, [1, 2]], W_[:, [2, 0]]], 1)
    Eu, cnt = np.unique(E, axis=0, return_counts=True)
    Pw = np.zeros((inv.max() + 1, 3)); Pw[inv.ravel()] = P
    op = Eu[cnt == 1]; mid = Pw[op].mean(1)
    kd = cKDTree(np.c_[road.x, road.y]); dmid, j = kd.query(mid[:, :2])
    off = (mid[:, 0] - road.x[j]) * road.nx[j] + (mid[:, 1] - road.y[j]) * road.ny[j]
    hw = np.where(off > 0, road.el[j], road.er[j]) + SH
    inb = (np.abs(off) < hw - 0.05) & (j >= i0) & (j <= i1) & (dmid < hw + 1)
    from geom import load_centerline, CFG
    _pal = load_centerline(CFG["build_end_m"] + 150); _dp, _ = cKDTree(np.c_[_pal["x_local"], _pal["y_local"]]).query(mid[:, :2])
    pj = inb & (_dp < 152)        # inside the Palomares terrain footprint: stitched by the junction step, reported apart
    res["open_edges_palomares_junction"] = int(pj.sum()); inb = inb & ~pj
    res["open_edges_in_band"] = int(inb.sum()); res["open_edges_in_band_s"] = sorted(set(int(road.s[x]) for x in j[inb]))[:50]
    out = ROOT / "reports" / "dev7" / f"qa_net_{rname}.json"; out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print({k: v for k, v in res.items() if not k.endswith("_list") and k != "files"})
    return res

ap_cluster = None
results = {}
_cl = None
for _cn in ("town", "sunol"):
    _f = BUILD / "dev7" / f"{_cn}_build.json"
    if _f.exists():
        _cl = (_cl or {}); _cl.update(json.load(open(_f))["log"].get("roads", {}))
for rn in ROADS:
    s_arg = A.s
    if s_arg is None and _cl and rn in _cl and rn != "niles_canyon_road":
        s_arg = f"{_cl[rn][0]}-{_cl[rn][1]}"
    results[rn] = check(rn, s_arg)
bad = {k: v for k, v in results.items() if v["holes"] or v["steps"] or v["overlaps"] or v["open_edges_in_band"]}
if len(ROADS) > 1:
    json.dump({k: {kk: vv for kk, vv in v.items() if not kk.endswith("_list")} for k, v in results.items()},
              open(ROOT / "reports" / "dev7" / f"qa_net_{A.tag or 'set'}.json", "w"), indent=1)
    print(f"{len(ROADS)} roads, failing: {sorted(bad)}")
sys.exit(1 if bad else 0)
