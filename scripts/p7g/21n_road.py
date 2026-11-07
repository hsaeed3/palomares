#!/usr/bin/env python3
"""dev7 G, step 21n: one network road -> its own kn5 files (palomares_niles_<tag>_*.kn5), Palomares kn5 untouched.

    PALOMARES_BUILD_DIR=build_dev7 nice python3 scripts/p7g/21n_road.py niles_canyon_road [--s 0-2000]

Same construction as scripts/p2/21_build_geometry.py, reused pieces: geom.Road (+ network.load_road profile fixes),
geom.Raster, grid_tris, vertex_normals, collision_conv.slab, kn5.write_kn5, and the Palomares road/terrain
materials + DDS textures (read back from the built palomares_road.kn5, so the look matches).
  road     : 1ROAD ribbon (1 m rows, ~0.8 m columns), crown at the centreline, 250 m chunks
  shoulder : 1GRASS strips edge -> edge + 0.6 m, slope-limited toward the DTM (bridge decks: 1ROAD deck strip)
  terrain  : 1GRASS constrained Delaunay of the dev7 DTM, hole = road+shoulder outline (shared boundary vertices =>
             no crack, no step). Points inside the Palomares terrain footprint are left to palomares_terrain.kn5.
  bridges  : OSM bridge spans -> 1WALL parapet slabs (collision_conv.slab, 1.3 m, 0.6 m deep), visual parapets
Outputs: $BUILD/palomares_road/palomares_niles_<tag>.kn5 (road+shoulders+walls) and _terrain.kn5,
$BUILD/dev7/<tag>_build.json (timings, counts, profile fixes).  Runtime NCR full length: see the json.
"""
import argparse, json, math, sys, time, pathlib, re
import numpy as np, triangle
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0] / "p2"))
import network as NW
from geom import ROOT, BUILD, CFG, DEV7, Raster, Road, grid_tris, vertex_normals, load_centerline, build_lock
import collision_conv as CC
from kn5 import Mesh, Material, write_kn5, read_kn5, ac_from_local

ap = argparse.ArgumentParser()
ap.add_argument("road"); ap.add_argument("--s", default=None, help="station range a-b (small tests)")
ap.add_argument("--tag", default=None)
A = ap.parse_args()
assert BUILD.resolve() != (ROOT / "build").resolve(), "dev7 G writes to its own build dir (PALOMARES_BUILD_DIR=build_dev7)"
build_lock("dev7_g")
T0 = time.time(); TIM = {}
tick = lambda k: TIM.__setitem__(k, round(time.time() - T0 - sum(TIM.values()), 1))
TAG = A.tag or {"niles_canyon_road": "ncr"}.get(A.road, A.road)
LOG = {}
road = NW.load_road(A.road, LOG)
N = len(road.s)
i0, i1 = 0, N - 1
if A.s:
    a, b = (float(v) for v in A.s.split("-")); i0, i1 = int(max(a, 0)), int(min(b, road.s[-1]))
I = np.arange(i0, i1 + 1)
DTM = Raster(DEV7 / "dtm_r100_b150.tif"); DTM5 = Raster(DEV7 / "dtm_r050_b40.tif")
dtm = lambda x, y: np.where(np.isfinite(DTM5.sample(x, y)), DTM5.sample(x, y), DTM.sample(x, y))
feat = json.load(open(DEV7 / "features.geojson"))["features"]
bridges = [(f["properties"]["s_start_m"], f["properties"]["s_end_m"]) for f in feat
           if f["properties"]["kind"] == "bridge_osm" and f["properties"].get("road") == A.road]
on_bridge = np.zeros(N, bool)
for a, b in bridges:
    on_bridge[int(a):int(b) + 1] = True
tick("load")

# ------------------------------------------------------------------ road ribbon (asphalt) + shoulders
NC = 15                                   # columns across the asphalt (odd: centre column = crown)
u = np.linspace(-1, 1, NC)
D = np.where(u[None] >= 0, u[None] * road.el[I, None], u[None] * road.er[I, None])   # (n, NC), d<0 right
# columns run right -> left (d increasing = left): grid_tris convention
ZR = road.surface_z(I[:, None], D)
XR, YR = road.xy(I[:, None], D)
SH = 0.6
_w = np.clip(gaussian_filter1d(on_bridge.astype(float), 5.0) * 2, 0, 1); wbr = np.maximum(_w * _w * (3 - 2 * _w), on_bridge)[I]   # deck rows exactly flush
sh_cols = []
for side in (-1, 1):                     # shoulder: edge, mid, outer
    e = np.where(side > 0, road.el[I], road.er[I]); ze = road.surface_z(I, side * e)
    cols = []
    for f in (0.5, 1.0):
        d = side * (e + f * SH); x, y = road.xy(I, d)
        zt = dtm(x, y); zt = np.where(np.isfinite(zt), zt, ze)
        lim = f * SH * 0.30
        # along-s smoothing of the shoulder rise (DTM speckle gave 10-20 cm row-to-row steps on the shoulder)
        rise = gaussian_filter1d(np.clip(zt - ze, -lim, lim), 6.0, mode="nearest")
        z = ze + (1 - wbr) * rise                       # deck: flush with the asphalt, blended over ~6 m
        cols.append((x, y, z))
    sh_cols.append(cols)
tick("ribbon")

groups = {"road": [], "terrain": [], "walls": []}


def add(g, name, mat, P, T, UV, **fl):
    groups[g].append(dict(name=name, mat=mat, P=P, T=T, UV=UV, N=vertex_normals(P, T), **fl))


CH = 250
for c0 in range(0, len(I) - 1, CH):
    sl = slice(c0, min(c0 + CH + 1, len(I)))
    n = sl.stop - sl.start
    P = np.c_[XR[sl].ravel(), YR[sl].ravel(), ZR[sl].ravel()]
    UV = np.c_[0.5 + D[sl].ravel() / 12, (road.s[I][sl][:, None].repeat(NC, 1)).ravel() / 12]
    cid = int(road.s[I][sl.start] // CH)
    add("road", f"1ROAD_{TAG}_{cid:02d}", "road_asphalt", P, grid_tris(n, NC), UV)
    for k, side in enumerate((-1, 1)):
        e = np.where(side > 0, road.el[I][sl], road.er[I][sl])
        x0, y0 = XR[sl][:, -1 if side > 0 else 0], YR[sl][:, -1 if side > 0 else 0]
        z0 = ZR[sl][:, -1 if side > 0 else 0]
        cols = [(x0, y0, z0)] + [(c[0][sl], c[1][sl], c[2][sl]) for c in sh_cols[k]]
        if side < 0:
            cols = cols[::-1]                # right side: outer -> edge (columns right -> left)
        Ps = np.stack([np.c_[c[0], c[1], c[2]] for c in cols], 1).reshape(-1, 3)
        UVs = np.c_[np.tile(np.arange(3) / 2.0, n), road.s[I][sl].repeat(3) / 4]
        Ts = grid_tris(n, 3)
        br = on_bridge[I][sl]
        rows = (Ts.min(1) // 3)
        deck = br[rows] & br[np.minimum(rows + 1, n - 1)]
        sname = "L" if side > 0 else "R"
        if (~deck).any():
            add("road", f"1GRASS_{TAG}_sh{sname}{cid:02d}", "shoulder_dirt", Ps, Ts[~deck], UVs)
        if deck.any():
            add("road", f"1ROAD_{TAG}_deck{sname}{cid:02d}", "road_asphalt", Ps, Ts[deck], UVs * [1 / 6, 3])
tick("road_mesh")

# ------------------------------------------------------------------ terrain (constrained Delaunay, like 21)
Lx, Ly, Lz = sh_cols[1][1]; Rx, Ry, Rz = sh_cols[0][1]
bound = np.concatenate([np.c_[Rx, Ry], np.c_[Lx[::-1], Ly[::-1]]]); bz = np.concatenate([Rz, Lz[::-1]])
# Palomares footprint (its terrain kn5 covers it): skip
pal = load_centerline(CFG["build_end_m"] + 150)
pkd = cKDTree(np.c_[pal["x_local"], pal["y_local"]])
rkd = cKDTree(np.c_[road.x[I], road.y[I]])
xmin, ymin = bound.min(0) - 150; xmax, ymax = bound.max(0) + 150
pts = []
for step, dmin, dmax in ((1.0, -1.0, 16), (2.0, 16, 50), (4.0, 50, 146)):
    gx = np.arange(math.floor(xmin), xmax, step); gy = np.arange(math.floor(ymin), ymax, step)
    for y0 in range(0, len(gy), 300):
        X, Y = np.meshgrid(gx, gy[y0:y0 + 300]); X = X.ravel(); Y = Y.ravel()
        dist, j = rkd.query(np.c_[X, Y], workers=2, distance_upper_bound=dmax + 20)
        ok = np.isfinite(dist); X, Y, dist, j = X[ok], Y[ok], dist[ok], j[ok]
        jj = I[np.clip(j, 0, len(I) - 1)]
        off = (X - road.x[jj]) * road.nx[jj] + (Y - road.y[jj]) * road.ny[jj]
        hw = np.where(off > 0, road.el[jj], road.er[jj]) + SH
        de = np.abs(off) - hw
        m = (de >= dmin) & (de < dmax) & (dist < hw + dmax) & (j > 0) & (j < len(I) - 1)
        pts.append(np.c_[X[m], Y[m]])
pts = np.concatenate(pts)
# ---- stitch to the shipped Palomares road/terrain (scripts/p7g/stitch.py): cut copies for the dev7 layouts
CUT = 4.0
dpal_road, _ = pkd.query(np.c_[road.x[I], road.y[I]])
STITCH = bool((dpal_road < 160).any())
SEGV = np.zeros((0, 3)); SEGE = np.zeros((0, 2), int); KEPT = ROADCUT = None


def in_cut(xy):
    dd, j_ = rkd.query(xy, workers=2); jj_ = I[j_]
    of = (xy[:, 0] - road.x[jj_]) * road.nx[jj_] + (xy[:, 1] - road.y[jj_]) * road.ny[jj_]
    h_ = np.where(of > 0, road.el[jj_], road.er[jj_]) + SH
    return (np.abs(of) < h_ + CUT) & (dd < h_ + CUT + 2) & (j_ > 0) & (j_ < len(I) - 1)


if STITCH:
    import stitch as STI
    snap = BUILD / "dev7" / "src"
    kept_all, cut_road = [], []
    for nm in ("palomares_terrain", "palomares_road", "palomares_lines_road"):
        src_ = STI.snapshot(ROOT / f"build/palomares_road/{nm}.kn5", snap)
        (kp, kt), (rp, rt) = STI.cut_copy(src_, BUILD / "palomares_road" / f"{nm}_dev7.kn5", in_cut, LOG.setdefault("stitch", {}))
        if nm != "palomares_lines_road":
            kept_all.append((kp, kt))
        if len(rt):
            cut_road.append((rp, rt))
    off_ = np.cumsum([0] + [len(p_) for p_, _ in kept_all])[:-1]
    KP = np.concatenate([p_ for p_, _ in kept_all]); KT = np.concatenate([t_ + o for (_, t_), o in zip(kept_all, off_)])
    # only the kept geometry near this road matters (speed): triangles within 170 m
    dk, _ = rkd.query(KP[KT].mean(1)[:, :2], workers=2); KT = KT[dk < 170]
    KEPT = STI.TriGrid(KP, KT)
    if cut_road:
        off_ = np.cumsum([0] + [len(p_) for p_, _ in cut_road])[:-1]
        ROADCUT = STI.TriGrid(np.concatenate([p_ for p_, _ in cut_road]), np.concatenate([t_ + o for (_, t_), o in zip(cut_road, off_)]))
    SEGV, SEGE = STI.open_boundary(KP, KT)
    # keep boundary edges near this road's terrain area (the far outer loop of the Palomares terrain stays open there)
    de_, _ = rkd.query(SEGV[:, :2], workers=2); okv = de_ < 146
    oke = okv[SEGE].all(1); used_, inv_ = np.unique(SEGE[oke], return_inverse=True)
    SEGV, SEGE = SEGV[used_], inv_.reshape(-1, 2)
    _, kt_ = KEPT.locate(pts); pts = pts[kt_ < 0]
    if len(SEGV):
        dsv, _ = cKDTree(SEGV[:, :2]).query(pts, workers=2); pts = pts[dsv > 0.35]
    LOG["stitch"]["boundary_vertices"] = int(len(SEGV)); LOG["stitch"]["boundary_edges"] = int(len(SEGE))
else:
    dpal, _ = pkd.query(pts, workers=2); pts = pts[dpal > 152]
# off the ribbon: > 0.35 m from the outline and outside the road+shoulder band
db, _ = cKDTree(bound).query(pts, workers=2)
_, j = rkd.query(pts, workers=2); jj = I[j]
off = (pts[:, 0] - road.x[jj]) * road.nx[jj] + (pts[:, 1] - road.y[jj]) * road.ny[jj]
hw = np.where(off > 0, road.el[jj], road.er[jj]) + SH
keep = (db > 0.35) & (np.abs(off) > hw + 0.3)
pts, off, hw, jj = pts[keep], off[keep], hw[keep], jj[keep]
pz = dtm(pts[:, 0], pts[:, 1])
# near the edge: terrain within 6 m of the shoulder is slope-limited (<= 40 deg) from the shoulder outer z, so the
# CDT never folds right at the drivable edge (qa_ffb_safety TERR_CREASE); on bridge spans the deck edge drops freely
de = np.abs(off) - hw
zo = np.where(off > 0, sh_cols[1][1][2][jj - i0], sh_cols[0][1][2][jj - i0])   # actual shoulder outer z
lim = 0.35 * np.clip(de, 0.05, 3.0) + 0.84 * np.maximum(de - 3.0, 0)   # ~19 deg in the first 3 m, then 40 deg
near = (de < 6.0) & ~on_bridge[jj]
pz = np.where(near, zo + np.clip(pz - zo, -lim, lim), pz)
# under / beside bridge decks: no terrain within 4 m of the deck edge (the abutment skirt starts beyond)
okb = ~(on_bridge[jj] & (de < 4.0))
pts, pz = pts[okb], pz[okb]
pts, pz = pts[np.isfinite(pz)], pz[np.isfinite(pz)]
nb = len(bound); segs = np.c_[np.arange(nb), (np.arange(nb) + 1) % nb]
if len(SEGV):                                   # fixed vertices of the kept Palomares geometry (exact positions)
    dsb, _ = cKDTree(bound).query(SEGV[:, :2]); assert dsb.min() > 0.2, "Palomares boundary touches the road outline"
segs = np.r_[segs, SEGE + nb + len(pts)]
V = np.concatenate([bound, pts, SEGV[:, :2]]); VZ = np.concatenate([bz, pz, SEGV[:, 2]])
NFIX = nb; FIXED = np.zeros(len(V), bool); FIXED[:nb] = True; FIXED[nb + len(pts):] = True
mid = len(I) // 2
tri = triangle.triangulate(dict(vertices=V, segments=segs, holes=[[road.x[I[mid]], road.y[I[mid]]]]), "pc")
TV, TT = tri["vertices"], tri["triangles"]
assert len(TV) == len(V), "triangle inserted Steiner points"
P3 = np.c_[TV, VZ]
e = np.stack([np.linalg.norm(P3[TT[:, a], :2] - P3[TT[:, b], :2], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], 1).max(1)
TT = TT[e < 9.0]
MOUTH = np.zeros(len(TT), bool)
if KEPT is not None:
    cen = P3[TT].mean(1)[:, :2]
    _, kt_ = KEPT.locate(cen); TT = TT[kt_ < 0]; cen = cen[kt_ < 0]
    if ROADCUT is not None:
        _, rt_ = ROADCUT.locate(cen); MOUTH = rt_ >= 0
    # membrane: vertices of the mouth and within 4 m of the fixed Palomares boundary relax toward their neighbours
    mv = np.zeros(len(P3), bool); mv[np.unique(TT[MOUTH])] = True
    if len(SEGV):
        dsv, _ = cKDTree(SEGV[:, :2]).query(P3[:, :2]); mv |= dsv < 4.0
    mv &= ~FIXED
    nb_ = np.r_[TT[:, [0, 1]], TT[:, [1, 2]], TT[:, [2, 0]]]
    for _it in range(60):
        sm = np.zeros(len(P3)); ct = np.zeros(len(P3))
        np.add.at(sm, nb_[:, 0], P3[nb_[:, 1], 2]); np.add.at(ct, nb_[:, 0], 1)
        np.add.at(sm, nb_[:, 1], P3[nb_[:, 0], 2]); np.add.at(ct, nb_[:, 1], 1)
        P3[mv, 2] = sm[mv] / np.maximum(ct[mv], 1)
    LOG["stitch"]["membrane_vertices"] = int(mv.sum()); LOG["stitch"]["mouth_tris"] = int(MOUTH.sum())
cr = np.cross(P3[TT[:, 1]] - P3[TT[:, 0]], P3[TT[:, 2]] - P3[TT[:, 0]])[:, 2]
TT[cr < 0] = TT[cr < 0][:, [0, 2, 1]]
# crease relaxation near the drivable edge (qa_ffb_safety: 1GRASS creases <= 45 deg within 3 m): vertices of
# folded pairs (both faces < 40 deg, crease > 40 deg) move toward their neighbours' mean; outline vertices fixed
dedge, _ = cKDTree(bound).query(P3[:, :2], workers=2)
nbr = np.r_[TT[:, [0, 1]], TT[:, [1, 2]], TT[:, [2, 0]]]
nrelax = 0
for _it in range(12):
    fn_ = np.cross(P3[TT[:, 1]] - P3[TT[:, 0]], P3[TT[:, 2]] - P3[TT[:, 0]]); fn_ /= np.linalg.norm(fn_, axis=1, keepdims=True)
    te = np.r_[np.sort(TT[:, [0, 1]], 1), np.sort(TT[:, [1, 2]], 1), np.sort(TT[:, [2, 0]], 1)]; tid = np.tile(np.arange(len(TT)), 3)
    o = np.lexsort((te[:, 1], te[:, 0])); te, tid = te[o], tid[o]
    same = (te[1:] == te[:-1]).all(1); ta, tb = tid[:-1][same], tid[1:][same]
    ang = np.degrees(np.arccos(np.clip((fn_[ta] * fn_[tb]).sum(1), -1, 1)))
    c40 = np.cos(np.radians(40))
    near_t = dedge[TT].min(1) < 4.0
    bad = (ang > 40) & (fn_[ta, 2] > c40) & (fn_[tb, 2] > c40) & near_t[ta] & near_t[tb]
    if not bad.any():
        break
    vb = np.unique(TT[np.r_[ta[bad], tb[bad]]]); vb = vb[~FIXED[vb]]
    sm = np.zeros(len(P3)); ct = np.zeros(len(P3))
    np.add.at(sm, nbr[:, 0], P3[nbr[:, 1], 2]); np.add.at(ct, nbr[:, 0], 1)
    np.add.at(sm, nbr[:, 1], P3[nbr[:, 0], 2]); np.add.at(ct, nbr[:, 1], 1)
    P3[vb, 2] = 0.5 * P3[vb, 2] + 0.5 * sm[vb] / np.maximum(ct[vb], 1)
    nrelax += len(vb)
LOG["terrain_relaxed_vertices"] = int(nrelax)
tick("terrain_cdt")
# material by slope (ground / bank / rock), chunk by nearest station (250 m)
fn = np.cross(P3[TT[:, 1]] - P3[TT[:, 0]], P3[TT[:, 2]] - P3[TT[:, 0]]); fn /= np.linalg.norm(fn, axis=1, keepdims=True)
slope = np.degrees(np.arccos(np.clip(fn[:, 2], -1, 1)))
cls = np.where(slope > 49, "rock", np.where(slope > 25, "bank", "ground"))
if MOUTH.any():                                  # junction mouth: asphalt continuing the cut Palomares road
    Tm = TT[MOUTH]; used, inv = np.unique(Tm, return_inverse=True); Pm = P3[used]
    add("road", f"1ROAD_{TAG}_mouth", "road_asphalt", Pm, inv.reshape(-1, 3), Pm[:, :2] / 12.0)
    TT, cls = TT[~MOUTH], cls[~MOUTH]
    # NCR shoulder triangles inside the cut Palomares asphalt become asphalt too
    for m in list(groups["road"]):
        if m["name"].startswith("1GRASS") and ROADCUT is not None:
            _, rt_ = ROADCUT.locate(m["P"][m["T"]].mean(1)[:, :2]); ins = rt_ >= 0
            if ins.any():
                add("road", m["name"].replace("1GRASS", "1ROAD") + "m", "road_asphalt", m["P"], m["T"][ins], m["P"][:, :2] / 12.0)
                m["T"] = m["T"][~ins]
_, jt = rkd.query(P3[TT].mean(1)[:, :2], workers=2); cht = (road.s[I[jt]] // 1000).astype(int)
for c in np.unique(cht):
    for k in ("ground", "bank", "rock"):
        sel = (cht == c) & (cls == k)
        if not sel.any():
            continue
        Tk = TT[sel]; used, inv = np.unique(Tk, return_inverse=True)
        Pk = P3[used]
        add("terrain", f"1GRASS_{TAG}_t{k[0]}{c:02d}", k, Pk, inv.reshape(-1, 3), Pk[:, :2] / 8.0)
tick("terrain_mesh")

# ------------------------------------------------------------------ bridge parapets (1WALL slabs + visual)
cl = {"s_m": road.s, "x_local": road.x, "y_local": road.y, "z_navd88": road.z,
      "edge_left_m": road.el, "edge_right_m": road.er}
F = CC.Frame(cl)
for bi, (a, b) in enumerate(bridges):
    a, b = max(a, road.s[i0]), min(b, road.s[i1])
    if b - a < 4:
        continue
    s = np.arange(a, b + 0.01, 2.0)
    ii = np.clip(np.round(s).astype(int), 0, N - 1)
    for side, nm in ((1, "L"), (-1, "R")):
        d = np.where(side > 0, road.el[ii], road.er[ii]) + SH + 0.05
        zb = road.surface_z(ii, side * d, noise=False) - 0.3; zt = zb + 1.3 + 0.3
        P, T, Nn = CC.slab(F, s, d, zb, zt, side, depth=np.full(len(s), 0.6))
        groups["walls"].append(dict(name=f"1WALL_{TAG}_br{bi}{nm}", mat="invisible", P=P, T=T, N=Nn,
                                    UV=np.zeros((len(P), 2)), visible=True, renderable=False, cast_shadows=False))
        Pv, Tv, Nv = CC.slab(F, s, d, zb + 0.3, zb + 0.3 + 0.95, side, depth=np.full(len(s), 0.35))
        groups["walls"].append(dict(name=f"{TAG}_parapet{bi}{nm}", mat="bank", P=Pv, T=Tv, N=Nv, UV=Pv[:, [0, 2]] / 3))
tick("bridges")

# ------------------------------------------------------------------ kn5 (materials + DDS from the Palomares road kn5)
# material source: a snapshot of a built Palomares road kn5 (build/ is rewritten during full builds)
SRC = BUILD / "dev7" / "material_src_palomares_road.kn5"
if not SRC.exists():
    import shutil
    for cand in (ROOT / "build/palomares_road/palomares_road.kn5", ROOT / "build_quick/palomares_road/palomares_road.kn5"):
        if cand.exists():
            SRC.parent.mkdir(parents=True, exist_ok=True); shutil.copy(cand, SRC); break
src = read_kn5(str(SRC))
SM = {m["name"]: m for m in src["materials"]}
mats, texs = [], {}
for nm in ("road_asphalt", "shoulder_dirt", "ground", "bank"):
    m = SM[nm]; props = {p: ((v[3], v[4], v[5]) if p == "ksEmissive" else v[0]) for p, v in m["props"].items()}
    mats.append(Material(nm, m["shader"], props, dict(m["textures"]), blend=m.get("blend", 0), alpha_tested=m.get("alpha_tested", False)))
    for t in m["textures"].values():
        texs[t] = src["textures"][t]
mats.append(Material("rock", "ksPerPixel", None, {"txDiffuse": "bank_d.dds"}))
mats.append(Material("invisible", "ksPerPixel", None, {"txDiffuse": "dirt_d.dds"}))
OUT = BUILD / "palomares_road"; OUT.mkdir(parents=True, exist_ok=True)
MERGE_M = 1000.0     # draw calls: one mesh per (physics key, material, renderable) per 1 km (kn5.split_mesh keeps <= 65k verts)


def merge(lst):
    """chunk merging: concatenate meshes sharing (physics prefix, material, flags, 1 km bin); walls stay separate
    (invisible colliders cost no draw call and qa_ffb_safety checks each slab as its own component)."""
    out, bins = [], {}
    for m in lst:
        nm = m["name"]
        if nm.startswith("1WALL"):
            out.append(m); continue
        key = re.match(r"^(\d?[A-Z]*)", nm).group(1) or "vis"
        km = int(np.median(np.asarray(rkd.query(m["P"][::max(len(m["P"]) // 50, 1), :2])[1])) + i0) if len(m["P"]) else 0
        km = int(road.s[min(km, N - 1)] // MERGE_M)
        fl = (m.get("visible", True), m.get("renderable", True), m.get("cast_shadows", True))
        bins.setdefault((key, m["mat"], fl, km), []).append(m)
    for (key, mat, fl, km), ms in sorted(bins.items(), key=lambda kv: str(kv[0])):
        off = np.cumsum([0] + [len(m["P"]) for m in ms])[:-1]
        nm = f"{key}_{TAG}_{mat}_{km:02d}" if key != "vis" else f"{TAG}_{mat}_{km:02d}"
        out.append(dict(name=nm, mat=mat, P=np.concatenate([m["P"] for m in ms]), T=np.concatenate([m["T"] + o for m, o in zip(ms, off)]),
                        UV=np.concatenate([m["UV"] for m in ms]), N=np.concatenate([m["N"] for m in ms]),
                        visible=fl[0], renderable=fl[1], cast_shadows=fl[2]))
    return out


for g in groups:
    groups[g] = merge(groups[g])
files = {}
for g, fname in (("road", f"palomares_niles_{TAG}.kn5"), ("walls", f"palomares_niles_{TAG}.kn5"), ("terrain", f"palomares_niles_{TAG}_terrain.kn5")):
    files.setdefault(fname, []).extend(groups[g])
summary = {}
for fname, lst in files.items():
    nodes = []
    for m in lst:
        Pa = ac_from_local(m["P"]); Na = ac_from_local(m["N"])
        nodes.append(Mesh(m["name"], Pa, Na, m["UV"], m["T"].reshape(-1), m["mat"],
                          cast_shadows=m.get("cast_shadows", True), visible=m.get("visible", True),
                          renderable=m.get("renderable", True)))
    used = {n.material for n in nodes}
    ms = [m for m in mats if m.name in used]
    tx = {t: texs[t] for m in ms for t in m.textures.values() if t in texs}
    write_kn5(str(OUT / fname), ms, tx, nodes)
    k = read_kn5(str(OUT / fname))                              # round trip
    vmax = max(len(m["pos"]) for m in k["meshes"])
    assert vmax <= 65535 and all(t in k["textures"] for mm in k["materials"] for t in mm["textures"].values())
    summary[fname] = dict(meshes=len(k["meshes"]), vmax=vmax, tris=int(sum(len(m["idx"]) // 3 for m in k["meshes"])),
                          draw_calls=sum(1 for m in k["meshes"] if m["renderable"] and m["visible"]))
tick("kn5")
xy = np.concatenate([m["P"][:, :2] for g in groups.values() for m in g])
rep = dict(road=A.road, tag=TAG, s=[float(road.s[i0]), float(road.s[i1])], timings_s=TIM, total_s=round(time.time() - T0, 1),
           files=summary, profile=LOG, bridges=bridges, max_abs_xy_m=float(np.abs(xy).max()),
           max_grade_pct=float(np.abs(np.gradient(road.z[I]) / np.gradient(road.s[I])).max() * 100))
(BUILD / "dev7").mkdir(exist_ok=True)
json.dump(rep, open(BUILD / "dev7" / f"{TAG}_build.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
