#!/usr/bin/env python3
"""dev7 G, step 21t: a town cluster (Niles core, Sunol) as ONE constrained-Delaunay surface -> palomares_niles_<c>*.kn5

    PALOMARES_BUILD_DIR=build_dev7 nice python3 scripts/p7g/21t_cluster.py niles

Why one surface: a town grid has a junction every ~100 m; separate ribbons would overlap or leave gaps at every
corner. Here every road, junction fillet, parking lot, curb, sidewalk and the terrain are regions of one CDT, so
neighbouring surfaces share vertices (no crack, no step) and junction heights come from one blended field.
  * roads     : every dev7 road whose centreline enters the cluster (network.load_road, profile fixes), clipped to it
  * height    : composite road field, lowest priority first (narrow -> wide, NCR last): z = lerp(z, z_r, a_r),
                a_r = 1 on road r, smoothstep to 0 at BLEND m past its edge / past its clipped end -> junctions and
                parking lots meet the major road's surface over BLEND m, no step by construction
  * curbs     : data/derived/dev7 curb_left/right heights (> CURB_MIN) -> rolled curb (0 -> h over CURB_W) and a
                SIDEWALK_W sidewalk at +1.5 % crossfall (physics key SIDEWALK, config/dev7/surfaces_dev7.ini)
  * parking   : OSM parking polygons, 3 m-smoothed DTM pulled to the adjacent roads (1ROAD)
  * terrain   : dev7 DTM, blended from the hardscape edge over 4 m; stitched to the fixed NCR kn5 (21n) through its
                open boundary vertices (stitch.py), so the canyon road continues without a crack
Outputs $BUILD/palomares_road/palomares_niles_<c>.kn5 (+ _terrain.kn5), $BUILD/dev7/<c>_build.json.
"""
import argparse, json, math, sys, time, pathlib, re
import numpy as np, triangle, shapely
import shapely.geometry as sg
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d, gaussian_filter
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0] / "p2"))
import network as NW, stitch as STI
from geom import ROOT, BUILD, DEV7, Raster, vertex_normals, build_lock
from kn5 import Mesh, Material, write_kn5, read_kn5, ac_from_local

ap = argparse.ArgumentParser(); ap.add_argument("cluster"); A = ap.parse_args()
assert BUILD.resolve() != (ROOT / "build").resolve()
build_lock("dev7_g")
T0 = time.time(); TIM = {}; LOG = {}
def tick(k): TIM[k] = round(time.time() - T0 - sum(TIM.values()), 1)
CL = A.cluster; TAG = {"niles": "town", "sunol": "sunol"}[CL]
BLEND, CURB_MIN, CURB_W, SIDEWALK_W = 16.0, 0.08, 0.30, 1.9
POLY = NW.cluster_polygon(CL); shapely.prepare(POLY)
DTM = Raster(DEV7 / "dtm_r100_b150.tif"); DTM5 = Raster(DEV7 / "dtm_r050_b40.tif")
def dtm(x, y):
    a = DTM5.sample(x, y); return np.where(np.isfinite(a), a, DTM.sample(x, y))

# ------------------------------------------------------------------ roads in the cluster
summ = json.load(open(DEV7 / "centerlines_summary.json"))
roads = {}
for name in summ:
    if name == "palomares_road":                # shipped Palomares
        continue
    c = NW.load_centerline(None, DEV7 / f"centerline_{name}.csv")
    ins = shapely.contains_xy(POLY.buffer(5), c["x_local"], c["y_local"])
    if ins.sum() < 5:
        continue
    r = NW.load_road(name, LOG)
    idx = np.flatnonzero(ins)
    r.lo, r.hi = int(idx.min()), int(idx.max())
    r.kd = cKDTree(np.c_[r.x, r.y])
    cl_ = np.nan_to_num(r.raw.get("curb_left", np.zeros(len(r.s)))); cr_ = np.nan_to_num(r.raw.get("curb_right", np.zeros(len(r.s))))
    r.curb_l = gaussian_filter1d(np.where(cl_ > CURB_MIN, np.clip(cl_, 0, 0.2), 0), 2.0)
    r.curb_r = gaussian_filter1d(np.where(cr_ > CURB_MIN, np.clip(cr_, 0, 0.2), 0), 2.0)
    if name != "niles_canyon_road":        # town crossfall: curb/gutter returns read as 6-10 % crowns; streets are 2-4 %
        r.cl = np.clip(r.cl, -0.04, 0.04); r.cr = np.clip(r.cr, -0.04, 0.04)
    r.prio = 1e6 if name == "niles_canyon_road" else float(np.median(r.el + r.er))
    roads[name] = r
order = sorted(roads, key=lambda n: roads[n].prio)
LOG["roads"] = {n: [float(roads[n].s[roads[n].lo]), float(roads[n].s[roads[n].hi])] for n in order}
print("roads", len(roads), order[-5:])
tick("roads")


def road_poly(r):
    i = np.arange(r.lo, r.hi + 1)
    L = np.stack(r.xy(i, r.el[i]), 1); R = np.stack(r.xy(i, -r.er[i]), 1)
    return sg.Polygon(np.r_[R, L[::-1]]).buffer(0)


def side_bands(r, side, d0, d1, mask):
    """polygons beyond the edge on one side where mask (per station) holds, offsets d0..d1 past the edge"""
    out = []
    i = np.arange(r.lo, r.hi + 1); m = mask[i]
    for run in np.split(i[m], np.flatnonzero(np.diff(i[m]) > 1) + 1):
        if len(run) < 3:
            continue
        e = r.el[run] if side > 0 else r.er[run]
        A_ = np.stack(r.xy(run, side * (e + d0)), 1); B_ = np.stack(r.xy(run, side * (e + d1)), 1)
        out.append(sg.Polygon(np.r_[A_, B_[::-1]]).buffer(0))
    return out


# ------------------------------------------------------------------ fixed neighbours (NCR kn5 from 21n): kept as is
fixed = []
for f in sorted((BUILD / "palomares_road").glob("palomares_niles_ncr*.kn5")):   # fixed road kn5s only (not assets/kerbs/timing)
    if f.name.startswith(f"palomares_niles_{TAG}"):
        continue
    for m in read_kn5(str(f))["meshes"]:
        if m["name"][:1].isdigit() and "WALL" not in m["name"]:
            P = STI.enu(m["pos"]); T = m["idx"].reshape(-1, 3)
            near = shapely.contains_xy(POLY.buffer(160), P[:, 0], P[:, 1])
            if near.any():
                fixed.append((P, T[near[T].any(1)]))
KEPT = None; SEGV = np.zeros((0, 3)); SEGE = np.zeros((0, 2), int)
if fixed:
    off_ = np.cumsum([0] + [len(p) for p, _ in fixed])[:-1]
    KP = np.concatenate([p for p, _ in fixed]); KT = np.concatenate([t + o for (_, t), o in zip(fixed, off_)])
    KEPT = STI.TriGrid(KP, KT)
    SEGV, SEGE = STI.open_boundary(KP, KT)
    okv = shapely.contains_xy(POLY.buffer(1.0), SEGV[:, 0], SEGV[:, 1])
    oke = okv[SEGE].all(1); used, inv = np.unique(SEGE[oke], return_inverse=True)
    SEGV, SEGE = SEGV[used], inv.reshape(-1, 2)
    FIXPOLY = shapely.union_all([sg.Polygon(KP[t]) for t in KT if shapely.contains_xy(POLY.buffer(5), *KP[t].mean(0)[:2])]).buffer(0.01) \
        if len(KT) < 400000 else None
LOG["fixed_boundary_vertices"] = int(len(SEGV))
tick("fixed")

# ------------------------------------------------------------------ regions
pave = shapely.union_all([road_poly(roads[n]) for n in order]); CARR = pave.buffer(-0.3); shapely.prepare(CARR)   # carriageways
park = []
for f in json.load(open(DEV7 / "osm_features.geojson"))["features"]:
    if f["properties"]["kind"] == "parking" and f["geometry"]["type"] == "Polygon":
        g = NW._to_local(sg.shape(f["geometry"])).buffer(0)
        if g.intersects(POLY):
            park.append(g)
PARK = shapely.union_all(park) if park else sg.Polygon()
pave = pave.union(PARK).buffer(3.0).buffer(-3.0)              # corner fillets (curb returns ~3 m)
walk = []
for n in order:
    r = roads[n]
    for side, h in ((1, r.curb_l), (-1, r.curb_r)):
        walk += side_bands(r, side, 0.0, CURB_W + SIDEWALK_W, h > 0.03)
WALK = shapely.union_all(walk).difference(pave) if walk else sg.Polygon()
DOM = POLY
if KEPT is not None and FIXPOLY is not None:
    DOM = POLY.difference(FIXPOLY)
pave = pave.intersection(DOM).buffer(0).simplify(0.05)
def _clean(g, amin=0.5, wmin=0.4):
    # drop slivers (curb bands vs fillet edges): opening by wmin/2, then parts under amin m2
    g = g.buffer(-wmin / 2).buffer(wmin / 2)
    return shapely.union_all([q for q in shapely.get_parts(g) if q.area >= amin]) if not g.is_empty else g
WALK = _clean(WALK.intersection(DOM).difference(pave)).difference(pave).buffer(0).simplify(0.05)
PARK = PARK.intersection(pave)
# snap region borders to a 0.2 m precision grid: borders that nearly coincide become identical (no slivers between
# almost-parallel boundary lines)
pave = shapely.set_precision(pave, 0.2); WALK = shapely.set_precision(WALK.difference(pave), 0.2)
WALK = shapely.union_all([q for q in shapely.get_parts(WALK) if q.area >= 0.5]) if not WALK.is_empty else WALK
for g in (pave, WALK, PARK, DOM):
    shapely.prepare(g)
LOG["areas_m2"] = dict(pavement=round(pave.area), sidewalk=round(WALK.area), parking=round(PARK.area))
tick("regions")

# ------------------------------------------------------------------ CDT input: noded region boundaries + points
def densify(lines, step):
    pts, segs, k = [], [], 0
    for ln in lines:
        c = np.asarray(ln.coords)[:, :2]
        if len(c) < 2:
            continue
        L = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
        if L[-1] < 0.05:
            continue
        t = np.linspace(0, L[-1], max(int(L[-1] / step) + 1, 2))
        q = np.c_[np.interp(t, L, c[:, 0]), np.interp(t, L, c[:, 1])]
        pts.append(q); segs.append(np.c_[np.arange(len(q) - 1), np.arange(1, len(q))] + k); k += len(q)
    return np.concatenate(pts), np.concatenate(segs)


bl = shapely.node(shapely.union_all([pave.boundary, WALK.boundary, DOM.boundary]))
lines = [g for g in shapely.get_parts(shapely.line_merge(bl))]
BV, BE = densify(lines, 0.75)
# weld coincident densified ends (mm)
key = np.round(BV, 3); U, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
BV = BV[np.unique(inv, return_index=True)[1]] if False else U; BE = inv[BE]; BE = BE[BE[:, 0] != BE[:, 1]]
# keep the fixed (NCR) boundary away from new boundary points
if len(SEGV):
    dfx, _ = cKDTree(SEGV[:, :2]).query(BV)
    drop = dfx < 0.3
    remap = -np.ones(len(BV), int); remap[~drop] = np.arange((~drop).sum()); BE = remap[BE]; BE = BE[(BE >= 0).all(1)]; BV = BV[~drop]
xmin, ymin, xmax, ymax = DOM.bounds
pts = []
hard = pave.union(WALK); shapely.prepare(hard)
HB, _ = densify([g for g in shapely.get_parts(hard.boundary)], 0.5); hkd = cKDTree(HB)
for step in (1.0, 2.0, 4.0):
    gx = np.arange(math.floor(xmin), xmax, step); gy = np.arange(math.floor(ymin), ymax, step)
    X, Y = np.meshgrid(gx, gy); X = X.ravel(); Y = Y.ravel()
    ok = shapely.contains_xy(DOM, X, Y)
    X, Y = X[ok], Y[ok]
    dh, _ = hkd.query(np.c_[X, Y], workers=2)
    inh = shapely.contains_xy(hard, X, Y)
    band = {1.0: inh | (dh < 12), 2.0: ~inh & (dh >= 12) & (dh < 40), 4.0: ~inh & (dh >= 40)}[step]
    pts.append(np.c_[X[band], Y[band]])
pts = np.concatenate(pts)
db, _ = cKDTree(BV).query(pts, workers=2); pts = pts[db > 0.3]
_bl = shapely.union_all(lines); shapely.prepare(_bl)
pts = pts[~shapely.dwithin(_bl, shapely.points(pts), 0.25)]      # not on / next to a border segment (collinear tris)
if KEPT is not None:
    _, kt = KEPT.locate(pts); pts = pts[kt < 0]
    if len(SEGV):
        dsv, _ = cKDTree(SEGV[:, :2]).query(pts); pts = pts[dsv > 0.35]
# no two input vertices closer than 5 cm (slivers / degenerate triangles otherwise)
_t = cKDTree(BV); _pr = _t.query_pairs(0.15, output_type="ndarray")
if len(_pr):
    rep = np.arange(len(BV))
    for a_, b_ in _pr[np.argsort(_pr[:, 0])]:
        rep[max(a_, b_)] = rep[min(a_, b_)]
    keepv = rep == np.arange(len(BV)); newi = np.cumsum(keepv) - 1
    BE = newi[rep[BE]]; BE = BE[BE[:, 0] != BE[:, 1]]; BE = np.unique(np.sort(BE, 1), axis=0); BV = BV[keepv]
db2, _ = cKDTree(BV).query(pts); pts = pts[db2 > 0.3]
LOG["vertex_merge_pairs"] = int(len(_pr))
nB = len(BV); nP = len(pts)
V = np.r_[BV, pts, SEGV[:, :2]]
S = np.r_[BE, SEGE + nB + nP]
tri = triangle.triangulate(dict(vertices=V, segments=S), "pc")      # c: keep the hull; DOM / KEPT filter below
TV, TT = tri["vertices"], tri["triangles"]
FIX = np.zeros(len(TV), bool); FIX[nB + nP:nB + nP + len(SEGV)] = True
# weld vertices at the same position (1 mm): Steiner points can coincide with input points
_k = np.round(TV, 3); _u, _first, _inv = np.unique(_k, axis=0, return_index=True, return_inverse=True); _inv = _inv.ravel()
if len(_u) < len(TV):
    _rep = _first[_inv]; FIXw = FIX.copy()
    np.logical_or.at(FIXw, _rep, FIX); FIX = FIXw
    TT = _rep[TT]; TT = TT[(TT[:, 0] != TT[:, 1]) & (TT[:, 1] != TT[:, 2]) & (TT[:, 0] != TT[:, 2])]
LOG["welded_duplicates"] = int(len(TV) - len(_u))
assert np.allclose(TV[nB + nP:nB + nP + len(SEGV)], SEGV[:, :2]), "fixed vertices moved"
cen = TV[TT].mean(1)
keep = shapely.contains_xy(DOM, cen[:, 0], cen[:, 1])
if KEPT is not None:
    _, kt = KEPT.locate(cen); keep &= kt < 0
TT = TT[keep]; cen = cen[keep]
_A = TV[TT]; _ar = 0.5 * np.abs((_A[:, 1, 0] - _A[:, 0, 0]) * (_A[:, 2, 1] - _A[:, 0, 1]) - (_A[:, 1, 1] - _A[:, 0, 1]) * (_A[:, 2, 0] - _A[:, 0, 0]))
TT = TT[_ar > 1e-3]; cen = TV[TT].mean(1)          # near-collinear caps of 3 consecutive border points (< 10 cm2; float32 at 8 km makes them degenerate): dropped
# collapse slivers / degenerate triangles (height < 3 cm): merge the shortest edge's movable vertex into the other;
# keeps the surface watertight (no holes), fixed NCR vertices never move
# slivers (height < 4 cm, mostly where a region border meets the grid): flip their longest edge when the quad is
# convex (Lawson flip; ignores the constraint there, which only separates regions), never an edge of the NCR boundary
def _flip_slivers(TT, TV, FIX, passes=4):
    nf = 0
    for _ in range(passes):
        A = TV[TT]; e = [A[:, 1] - A[:, 0], A[:, 2] - A[:, 1], A[:, 0] - A[:, 2]]
        L = np.stack([np.hypot(*x.T) for x in e], 1)
        ar = 0.5 * np.abs(e[0][:, 0] * (-e[2][:, 1]) - e[0][:, 1] * (-e[2][:, 0]))
        bad = np.flatnonzero(2 * ar / np.maximum(L.max(1), 1e-9) < 0.04)
        if not len(bad):
            break
        emap = {}
        for t, tri_ in enumerate(TT):
            for k in range(3):
                a, b = tri_[k], tri_[(k + 1) % 3]
                emap.setdefault((min(a, b), max(a, b)), []).append(t)
        used = np.zeros(len(TT), bool)
        for t in bad:
            if used[t]:
                continue
            k = int(np.argmax(L[t])); a, b, c = TT[t, k], TT[t, (k + 1) % 3], TT[t, (k + 2) % 3]
            if FIX[a] and FIX[b]:
                continue
            nb_ = [u for u in emap.get((min(a, b), max(a, b)), []) if u != t]
            if len(nb_) != 1 or used[nb_[0]]:
                continue
            u = nb_[0]; d = [v for v in TT[u] if v not in (a, b)][0]
            # convex quad a-c-b-d: c and d on opposite sides of a-b, a and b on opposite sides of c-d
            def side(p, q, r):
                return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
            if side(TV[c], TV[d], TV[a]) * side(TV[c], TV[d], TV[b]) >= 0:
                continue
            TT[t] = [c, a, d]; TT[u] = [d, b, c]; used[t] = used[u] = True; nf += 1
    return TT, nf
_nf = 0   # flips disabled: they produced folds; region borders are snapped instead (set_precision)
LOG["sliver_flips"] = int(_nf)
for _it in range(0):     # (sliver collapse retired: quality CDT instead; collapse made overlapping triangles)
    A3 = np.c_[TV, np.zeros(len(TV))][TT]; e0 = A3[:, 1] - A3[:, 0]; e1 = A3[:, 2] - A3[:, 1]; e2 = A3[:, 0] - A3[:, 2]
    L3 = np.stack([np.linalg.norm(e[:, :2], axis=1) for e in (e0, e1, e2)], 1)
    ar = 0.5 * np.abs(e0[:, 0] * (-e2[:, 1]) - e0[:, 1] * (-e2[:, 0]))
    bad = np.flatnonzero(2 * ar / np.maximum(L3.max(1), 1e-9) < 0.06)
    if not len(bad):
        break
    rep = np.arange(len(TV)); done = np.zeros(len(TV), bool)
    for t in bad:
        k = int(np.argmin(L3[t])); a_, b_ = TT[t, k], TT[t, (k + 1) % 3]
        if L3[t, k] > 0.6:                    # long thin needle: collapse its apex onto the opposite edge's end
            a_, b_ = TT[t, (k + 2) % 3], TT[t, k]
        if done[a_] or done[b_]:
            continue
        if FIX[b_] and FIX[a_]:
            continue
        src_, dst_ = (b_, a_) if not FIX[b_] else (a_, b_)
        rep[src_] = dst_; done[src_] = done[dst_] = True
    TT = rep[TT]; TT = TT[(TT[:, 0] != TT[:, 1]) & (TT[:, 1] != TT[:, 2]) & (TT[:, 0] != TT[:, 2])]
LOG["sliver_passes"] = 0
LOG["cdt"] = dict(vertices=int(len(TV)), tris=int(len(TT)), steiner=int(len(TV) - len(V)))
tick("cdt")

# ------------------------------------------------------------------ heights
def composite(xy):
    z = gaussian_dtm(xy)
    for n in order:
        r = roads[n]
        d, j = r.kd.query(xy, workers=2)
        j = np.clip(j, 0, len(r.s) - 1)
        off = (xy[:, 0] - r.x[j]) * r.nx[j] + (xy[:, 1] - r.y[j]) * r.ny[j]
        along = (xy[:, 0] - r.x[j]) * r.tx[j] + (xy[:, 1] - r.y[j]) * r.ty[j]
        hw = np.where(off > 0, r.el[j], r.er[j])
        zr = r.surface_z(j, np.clip(off, -r.er[j], r.el[j]), noise=False)
        out_side = np.maximum(np.abs(off) - hw, 0)
        past = np.where(j <= 0, np.maximum(-along, 0), 0) + np.where(j >= len(r.s) - 1, np.maximum(along, 0), 0)
        # blend length grows with the height difference to bridge (<= ~5 % extra grade): a 2 m step between two
        # streets (Limelight Terrace end vs Cinematic Common) is spread over 40 m, not 16 m
        Lb = np.maximum(BLEND, np.abs(zr - z) / 0.05)
        a = np.clip(1 - np.hypot(out_side, past) / Lb, 0, 1); a = a * a * (3 - 2 * a)
        z = (1 - a) * z + a * zr
    return z


_gd = {}
def gaussian_dtm(xy):
    """3 m-smoothed DTM (parking lots, fillets far from any road)"""
    if "r" not in _gd:
        r = Raster(DEV7 / "dtm_r100_b150.tif"); a = r.a.copy(); m = np.isfinite(a)
        a[~m] = np.nanmean(a); s_ = gaussian_filter(a, 3.0); w_ = gaussian_filter(m.astype(float), 3.0)
        r.a = np.where(m, s_ / np.maximum(w_, 1e-6) * 0 + s_, np.nan); _gd["r"] = r
    return _gd["r"].sample(xy[:, 0], xy[:, 1])


Z = np.full(len(TV), np.nan)
# vertex classes from the triangle classes (centroid test), so every pavement triangle has pavement heights at all
# three corners (a point test with a tolerance left some corners on terrain / sidewalk heights -> folds)
_c2 = TV[TT].mean(1)
TPAVE = shapely.contains_xy(pave, _c2[:, 0], _c2[:, 1]); TWALK = shapely.contains_xy(WALK, _c2[:, 0], _c2[:, 1]) & ~TPAVE
inP = np.zeros(len(TV), bool); inP[TT[TPAVE].ravel()] = True
inW = np.zeros(len(TV), bool); inW[TT[TWALK].ravel()] = True; inW &= ~inP
# gap slivers between pavement and sidewalk (all three corners already hardscape): sidewalk/curb, not terrain
_hs = (inP | inW)[TT].all(1) & ~TPAVE & ~TWALK
# (thin gap fillers along a border, near-degenerate; as curb they carry no carriageway crease limit)
TWALK = TWALK | _hs
LOG["gap_slivers_reclassed"] = int(_hs.sum())
zc = composite(TV)
# fold-targeted smoothing of the composite field over the pavement triangles (junction blends, crowns meeting at
# an angle): qa_ffb_safety ROAD crease <= 8 deg; fixed NCR vertices stay
_tp = TT[TPAVE]
_mov = np.zeros(len(TV), bool); _mov[np.unique(_tp)] = True; _mov &= ~FIX
_e = np.r_[_tp[:, [0, 1]], _tp[:, [1, 2]], _tp[:, [2, 0]]]
_tp_all = _tp
for _it in range(400):
    if _it % 50 == 0:                 # full re-check every 50 passes (smoothing can create folds elsewhere)
        _tp = _tp_all; _e = np.r_[_tp[:, [0, 1]], _tp[:, [1, 2]], _tp[:, [2, 0]]]
    elif _it % 50 == 1 and len(vb):   # in between only the neighbourhood of the folds is re-checked
        _near = cKDTree(TV[vb]).query(TV[:, :2] if TV.shape[1] == 2 else TV, distance_upper_bound=6.0)[0] < 6.0
        _tp = _tp_all[_near[_tp_all].any(1)]; _e = np.r_[_tp[:, [0, 1]], _tp[:, [1, 2]], _tp[:, [2, 0]]]
    P_ = np.c_[TV, zc]; f_ = np.cross(P_[_tp[:, 1]] - P_[_tp[:, 0]], P_[_tp[:, 2]] - P_[_tp[:, 0]]); f_ /= np.linalg.norm(f_, axis=1, keepdims=True)
    te = np.r_[np.sort(_tp[:, [0, 1]], 1), np.sort(_tp[:, [1, 2]], 1), np.sort(_tp[:, [2, 0]], 1)]; tid = np.tile(np.arange(len(_tp)), 3)
    o = np.lexsort((te[:, 1], te[:, 0])); te, tid = te[o], tid[o]
    same = (te[1:] == te[:-1]).all(1); ta, tb = tid[:-1][same], tid[1:][same]
    bad = np.degrees(np.arccos(np.clip((f_[ta] * f_[tb]).sum(1), -1, 1))) > 4.0
    vb = np.unique(_tp[np.r_[ta[bad], tb[bad]]]); vb = vb[_mov[vb]]
    if not len(vb):
        break
    sm = np.zeros(len(TV)); ct = np.zeros(len(TV))
    np.add.at(sm, _e[:, 0], zc[_e[:, 1]]); np.add.at(ct, _e[:, 0], 1); np.add.at(sm, _e[:, 1], zc[_e[:, 0]]); np.add.at(ct, _e[:, 1], 1)
    zc[vb] = 0.5 * zc[vb] + 0.5 * sm[vb] / np.maximum(ct[vb], 1)
LOG["pave_fold_iters"] = _it; LOG["pave_fold_left"] = int(len(vb))
Z[inP] = zc[inP]
# sidewalk: curb ramp + crossfall from the pavement edge, curb height from the nearest road side
PB, _ = densify([g for g in shapely.get_parts(pave.boundary)], 0.25); pkd = cKDTree(PB)
dp, _ = pkd.query(TV[inW]); q = TV[inW]
hq = np.zeros(len(q)); best = np.full(len(q), np.inf)
for n in order:
    r = roads[n]; d, j = r.kd.query(q); j = np.clip(j, 0, len(r.s) - 1)
    off = (q[:, 0] - r.x[j]) * r.nx[j] + (q[:, 1] - r.y[j]) * r.ny[j]
    h_ = np.where(off > 0, r.curb_l[j], r.curb_r[j]); dd = np.abs(np.abs(off) - np.where(off > 0, r.el[j], r.er[j]))
    b = dd < best; hq[b] = h_[b]; best[b] = dd[b]
ramp = np.clip(dp / CURB_W, 0, 1); ramp = ramp * ramp * (3 - 2 * ramp)
Z[inW] = zc[inW] + hq * ramp + 0.015 * np.maximum(dp - CURB_W, 0)
# terrain: DTM blended from the nearest hardscape vertex over 4 m
inT = ~inP & ~inW & ~FIX
hv = np.flatnonzero(inP | inW); hk = cKDTree(TV[hv])
dt, jt = hk.query(TV[inT]); zt = dtm(TV[inT, 0], TV[inT, 1]); zt = np.where(np.isfinite(zt), zt, Z[hv][jt])
w = np.clip(dt / 4.0, 0, 1); w = w * w * (3 - 2 * w)
Z[inT] = (1 - w) * Z[hv][jt] + w * zt
Z[FIX] = SEGV[:, 2]
# terrain next to the fixed NCR boundary: blend to it over 4 m
if len(SEGV):
    df, jf = cKDTree(SEGV[:, :2]).query(TV[inT]); wf = np.clip(df / 4.0, 0, 1); wf = wf * wf * (3 - 2 * wf)
    Z[inT] = (1 - wf) * SEGV[jf, 2] + wf * Z[inT]
# pavement next to the fixed NCR asphalt end: the composite (NCR prio) already matches it to noise (~4 mm)
# terrain folds next to the hardscape (qa_ffb_safety: 1GRASS creases <= 45 deg within 3 m): relax terrain vertices of
# folded pairs (both faces < 40 deg, crease > 38 deg) within 5 m of a hardscape vertex
_tt = TT[~(TPAVE | TWALK)]; _ne = np.r_[_tt[:, [0, 1]], _tt[:, [1, 2]], _tt[:, [2, 0]]]
_dh = np.full(len(TV), np.inf); _dh[inT] = dt
for _it in range(40):
    P_ = np.c_[TV, Z]; f_ = np.cross(P_[_tt[:, 1]] - P_[_tt[:, 0]], P_[_tt[:, 2]] - P_[_tt[:, 0]]); f_ /= np.linalg.norm(f_, axis=1, keepdims=True) + 1e-12
    te = np.r_[np.sort(_tt[:, [0, 1]], 1), np.sort(_tt[:, [1, 2]], 1), np.sort(_tt[:, [2, 0]], 1)]; tid = np.tile(np.arange(len(_tt)), 3)
    o = np.lexsort((te[:, 1], te[:, 0])); te, tid = te[o], tid[o]
    same = (te[1:] == te[:-1]).all(1); ta, tb = tid[:-1][same], tid[1:][same]
    c40 = np.cos(np.radians(40))
    _an = np.degrees(np.arccos(np.clip((f_[ta] * f_[tb]).sum(1), -1, 1)))
    bad = ((_an > 38) & (f_[ta, 2] > c40) & (f_[tb, 2] > c40)) | (_an > 95)      # + fold-overs (spikes) at any slope
    vb = np.unique(_tt[np.r_[ta[bad], tb[bad]]]); vb = vb[inT[vb] & (_dh[vb] < 5.0)]
    if not len(vb):
        break
    sm = np.zeros(len(TV)); ct = np.zeros(len(TV))
    np.add.at(sm, _ne[:, 0], Z[_ne[:, 1]]); np.add.at(ct, _ne[:, 0], 1); np.add.at(sm, _ne[:, 1], Z[_ne[:, 0]]); np.add.at(ct, _ne[:, 1], 1)
    Z[vb] = 0.5 * Z[vb] + 0.5 * sm[vb] / np.maximum(ct[vb], 1)
LOG["terrain_fold_iters"] = _it
# terrain slivers (height < 3 cm): put a movable apex on the line of its long edge (a sliver cannot fold then)
for _p in range(3):
    A = np.c_[TV, Z][TT]; L3 = np.stack([np.linalg.norm(A[:, (k + 1) % 3, :2] - A[:, k, :2], axis=1) for k in range(3)], 1)
    ar2 = np.abs((A[:, 1, 0] - A[:, 0, 0]) * (A[:, 2, 1] - A[:, 0, 1]) - (A[:, 1, 1] - A[:, 0, 1]) * (A[:, 2, 0] - A[:, 0, 0]))
    h_ = ar2 / np.maximum(L3.max(1), 1e-9)
    sl = np.flatnonzero(h_ < 0.03)
    k_ = np.argmax(L3[sl], 1)
    a_, b_, c_ = TT[sl, k_], TT[sl, (k_ + 1) % 3], TT[sl, (k_ + 2) % 3]
    mvc = inT[c_] & ~FIX[c_]
    t_ = np.clip(((TV[c_] - TV[a_]) * (TV[b_] - TV[a_])).sum(1) / np.maximum(((TV[b_] - TV[a_]) ** 2).sum(1), 1e-12), 0, 1)
    Z[c_[mvc]] = (Z[a_] + t_ * (Z[b_] - Z[a_]))[mvc]
LOG["terrain_slivers_flattened"] = int(mvc.sum())
assert np.isfinite(Z).all(), "unset heights"
P3 = np.c_[TV, Z]
cr = np.cross(P3[TT[:, 1]] - P3[TT[:, 0]], P3[TT[:, 2]] - P3[TT[:, 0]])[:, 2]
LOG["inverted_tris"] = int((cr <= 0).sum())
TT = TT[cr > 0]     # triangle output is CCW; a clockwise one can only be a fold-over: the area is covered by its neighbours
tick("heights")

# ------------------------------------------------------------------ classes -> meshes (1 km tiles, merged)
cen = P3[TT].mean(1)
cp, cw = TPAVE, TWALK
fn = np.cross(P3[TT[:, 1]] - P3[TT[:, 0]], P3[TT[:, 2]] - P3[TT[:, 0]]); fn /= np.linalg.norm(fn, axis=1, keepdims=True)
slope = np.degrees(np.arccos(np.clip(fn[:, 2], -1, 1)))
carr = shapely.contains_xy(CARR, cen[:, 0], cen[:, 1])
cls = np.where(cp & ~carr, "apron", np.where(cp, "asphalt", np.where(cw, "walk", np.where(slope > 49, "rock", np.where(slope > 25, "bank", "ground")))))
LOG["class_tris"] = {k: int((cls == k).sum()) for k in np.unique(cls)}; print(LOG["class_tris"], "pts", nP, "pave pts", int(shapely.contains_xy(pave, pts[:, 0], pts[:, 1]).sum()))
tile = (np.floor(cen[:, 0] / 1000).astype(int) * 100 + np.floor(cen[:, 1] / 1000).astype(int)) % 10000
# apron = paved but off every carriageway (junction fillets, parking lots): qa_ffb_safety apron crease limit
NAME = {"asphalt": ("1ROAD", "road_asphalt"), "apron": ("1ROAD_AP", "road_asphalt"), "walk": ("1SIDEWALK", "sidewalk"), "ground": ("1GRASS", "ground"),
        "bank": ("1GRASS", "bank"), "rock": ("1GRASS", "rock")}
groups = {"road": [], "terrain": []}
for t in np.unique(tile):
    for k, (key, mat) in NAME.items():
        sel = (tile == t) & (cls == k)
        if not sel.any():
            continue
        Tk = TT[sel]; used, inv = np.unique(Tk, return_inverse=True); Pk = P3[used]; Tk = inv.reshape(-1, 3)
        g = "road" if k in ("asphalt", "apron", "walk") else "terrain"
        groups[g].append(dict(name=f"{key}_{TAG}_{mat}_{len(groups[g]):02d}", mat=mat, P=Pk, T=Tk,
                              UV=Pk[:, :2] / (12.0 if k == "asphalt" else 4.0 if k == "walk" else 8.0), N=vertex_normals(Pk, Tk)))
tick("meshes")

# ------------------------------------------------------------------ kn5
SRC = BUILD / "dev7" / "material_src_palomares_road.kn5"
src = read_kn5(str(SRC)); SM = {m["name"]: m for m in src["materials"]}
mats, texs = [], {}
for nm in ("road_asphalt", "ground", "bank"):
    m = SM[nm]; props = {p: ((v[3], v[4], v[5]) if p == "ksEmissive" else v[0]) for p, v in m["props"].items()}
    mats.append(Material(nm, m["shader"], props, dict(m["textures"]))); texs.update({t: src["textures"][t] for t in m["textures"].values()})
mats.append(Material("rock", "ksPerPixel", None, {"txDiffuse": "bank_d.dds"}))
# sidewalk: the fresh-asphalt diffuse brightened by the material (no concrete texture in the shipped set yet)
mats.append(Material("sidewalk", "ksPerPixel", {"ksAmbient": 0.75, "ksDiffuse": 0.85, "ksSpecular": 0.05, "ksSpecularEXP": 8.0,
                                                 "ksEmissive": (0, 0, 0), "ksAlphaRef": 0.0}, {"txDiffuse": "asphalt_fresh_d.dds"}))
texs["asphalt_fresh_d.dds"] = src["textures"]["asphalt_fresh_d.dds"]
OUT = BUILD / "palomares_road"
summary = {}
for g, fname in (("road", f"palomares_niles_{TAG}.kn5"), ("terrain", f"palomares_niles_{TAG}_terrain.kn5")):
    nodes = [Mesh(m["name"], ac_from_local(m["P"]), ac_from_local(m["N"]), m["UV"], m["T"].reshape(-1), m["mat"]) for m in groups[g]]
    used = {n.material for n in nodes}; ms = [m for m in mats if m.name in used]
    write_kn5(str(OUT / fname), ms, {t: texs[t] for m in ms for t in m.textures.values()}, nodes)
    k = read_kn5(str(OUT / fname)); vmax = max(len(m["pos"]) for m in k["meshes"])
    assert vmax <= 65535 and all(t in k["textures"] for mm in k["materials"] for t in mm["textures"].values())
    summary[fname] = dict(meshes=len(k["meshes"]), vmax=vmax, tris=int(sum(len(m["idx"]) // 3 for m in k["meshes"])),
                          draw_calls=sum(1 for m in k["meshes"] if m["renderable"] and m["visible"]))
tick("kn5")
rep = dict(cluster=CL, tag=TAG, timings_s=TIM, total_s=round(time.time() - T0, 1), files=summary, log=LOG)
json.dump(rep, open(BUILD / "dev7" / f"{TAG}_build.json", "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "log"}, indent=1)); print(LOG.get("areas_m2"), LOG.get("cdt"))
