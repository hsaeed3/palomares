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
BLEND, CURB_MIN, CURB_W, SIDEWALK_W = 8.0, 0.08, 0.30, 1.9
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
pave = shapely.union_all([road_poly(roads[n]) for n in order])
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
        walk += side_bands(r, side, -0.05, CURB_W + SIDEWALK_W, h > 0.03)
WALK = shapely.union_all(walk).difference(pave) if walk else sg.Polygon()
DOM = POLY
if KEPT is not None and FIXPOLY is not None:
    DOM = POLY.difference(FIXPOLY)
pave = pave.intersection(DOM).buffer(0); WALK = WALK.intersection(DOM).buffer(0); PARK = PARK.intersection(pave)
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
if KEPT is not None:
    _, kt = KEPT.locate(pts); pts = pts[kt < 0]
    if len(SEGV):
        dsv, _ = cKDTree(SEGV[:, :2]).query(pts); pts = pts[dsv > 0.35]
nB = len(BV); nP = len(pts)
V = np.r_[BV, pts, SEGV[:, :2]]
S = np.r_[BE, SEGE + nB + nP]
tri = triangle.triangulate(dict(vertices=V, segments=S), "pc")      # c: keep the hull; DOM / KEPT filter below
TV, TT = tri["vertices"], tri["triangles"]
FIX = np.zeros(len(TV), bool); FIX[nB + nP:nB + nP + len(SEGV)] = True
assert np.allclose(TV[nB + nP:nB + nP + len(SEGV)], SEGV[:, :2]), "fixed vertices moved"
cen = TV[TT].mean(1)
keep = shapely.contains_xy(DOM, cen[:, 0], cen[:, 1])
if KEPT is not None:
    _, kt = KEPT.locate(cen); keep &= kt < 0
TT = TT[keep]; cen = cen[keep]
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
        a = np.clip(1 - np.hypot(out_side, past) / BLEND, 0, 1); a = a * a * (3 - 2 * a)
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
inP = shapely.contains_xy(pave.buffer(0.02), TV[:, 0], TV[:, 1])
inW = shapely.contains_xy(WALK.buffer(0.02), TV[:, 0], TV[:, 1]) & ~inP
zc = composite(TV)
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
assert np.isfinite(Z).all(), "unset heights"
P3 = np.c_[TV, Z]
cr = np.cross(P3[TT[:, 1]] - P3[TT[:, 0]], P3[TT[:, 2]] - P3[TT[:, 0]])[:, 2]
TT[cr < 0] = TT[cr < 0][:, [0, 2, 1]]
tick("heights")

# ------------------------------------------------------------------ classes -> meshes (1 km tiles, merged)
cen = P3[TT].mean(1)
cp = shapely.contains_xy(pave, cen[:, 0], cen[:, 1]); cw = shapely.contains_xy(WALK, cen[:, 0], cen[:, 1]) & ~cp
fn = np.cross(P3[TT[:, 1]] - P3[TT[:, 0]], P3[TT[:, 2]] - P3[TT[:, 0]]); fn /= np.linalg.norm(fn, axis=1, keepdims=True)
slope = np.degrees(np.arccos(np.clip(fn[:, 2], -1, 1)))
cls = np.where(cp, "asphalt", np.where(cw, "walk", np.where(slope > 49, "rock", np.where(slope > 25, "bank", "ground"))))
LOG["class_tris"] = {k: int((cls == k).sum()) for k in np.unique(cls)}; print(LOG["class_tris"], "pts", nP, "pave pts", int(shapely.contains_xy(pave, pts[:, 0], pts[:, 1]).sum()))
tile = (np.floor(cen[:, 0] / 1000).astype(int) * 100 + np.floor(cen[:, 1] / 1000).astype(int)) % 10000
NAME = {"asphalt": ("1ROAD", "road_asphalt"), "walk": ("1SIDEWALK", "sidewalk"), "ground": ("1GRASS", "ground"),
        "bank": ("1GRASS", "bank"), "rock": ("1GRASS", "rock")}
groups = {"road": [], "terrain": []}
for t in np.unique(tile):
    for k, (key, mat) in NAME.items():
        sel = (tile == t) & (cls == k)
        if not sel.any():
            continue
        Tk = TT[sel]; used, inv = np.unique(Tk, return_inverse=True); Pk = P3[used]; Tk = inv.reshape(-1, 3)
        g = "road" if k in ("asphalt", "walk") else "terrain"
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
