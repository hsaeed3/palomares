#!/usr/bin/env python3
"""P (dev7 plan "Vegetation, two zones"), step 21f: procedural FAR-zone trees into <build>/geometry.pkl.

Near zone (unchanged, 21_build_geometry.py): every LiDAR tree of data/derived/trees.csv, i.e. within NEAR_M (60 m) of
the centreline, at its exact position, height and crown radius.
Far zone (this step): NEAR_M .. FAR_M (the 150 m LiDAR buffer), scattered from the canopy-height model
(data/derived/chm_r100_b150.tif) and NAIP 2022 greenness (data/derived/naip2022_r060.tif, NDVI from bands 4/1):
  * candidate = local maximum of the 3 m-smoothed CHM in a 2*R_PEAK+1 m window, CHM >= MIN_H, NDVI >= MIN_NDVI,
    and not inside the crown of a LiDAR tree (near zone or its fringe);
  * height = CHM at the peak; crown radius = clip(0.32 * h, 2.5, 9) (the median ratio of trees.csv);
  * species: euc if h > 24 and the crown is narrow, oak where greenness is moderate (open oak savanna), bay in dense
    green canopy, con for tall narrow non-euc crowns; same classifier spirit as 21 (no LiDAR crown shape out here);
  * per-instance hash -> prototype variant (4 per species, each with its own leaf-colour column), yaw, +-12 % scale,
    lean; i.e. automatic species / scale / rotation / colour variance, fully deterministic.
Meshes: trees3d.py LOD-1 prototypes, merged per FAR_CHUNK (500 m) per material using the SAME shared materials as the
near trees (tree_trunk, tree_leaves = ksTree in ext_config.ini), so the far zone costs 2 draw calls per 500 m (~62),
no shadows, lod_out LOD_OUT. Appended to G["trees"] as far_* (-> palomares_trees.kn5). No collision (convention).
Idempotent: previous far_* meshes are dropped first. Writes reports/veg_far.json (counts, species, draw calls).
"""
import csv, json, math, pickle, sys, pathlib, time
import numpy as np, rasterio
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import Road, Raster, ROOT, DER, BUILD, CFG
from geom import build_lock; build_lock()
import trees3d as T3

NEAR_M, FAR_M = 60.0, 150.0
MIN_H, MIN_NDVI, R_PEAK = 4.0, 0.05, 3
FAR_CHUNK, LOD_OUT = 500.0, 1400.0
MAX_V = 60000

t0 = time.time()
geo = pickle.load(open(BUILD / "geometry.pkl", "rb"))
G, meta = geo["groups"], geo["meta"]
S_END = float(meta["s_end"])
road = Road(CFG["build_end_m"])
KD = cKDTree(np.c_[road.x, road.y])
o = json.load(open(ROOT / "config/origin.json")); X0, Y0 = o["x0"], o["y0"]

# ---- candidates from the CHM
with rasterio.open(DER / "chm_r100_b150.tif") as r:
    chm = r.read(1).astype(np.float32); tf = r.transform
    if r.nodata is not None:
        chm[chm == r.nodata] = 0
chm = np.nan_to_num(chm, nan=0.0)
sm = uniform_filter(chm, 3)
peak = (sm == maximum_filter(sm, 2 * R_PEAK + 1)) & (sm >= MIN_H)
rr, cc = np.nonzero(peak)
xu = tf.c + (cc + 0.5) * tf.a; yu = tf.f + (rr + 0.5) * tf.e
xl, yl = xu - X0, yu - Y0
d, j = KD.query(np.c_[xl, yl])
keep = (d >= NEAR_M) & (d <= FAR_M) & (road.s[j] <= S_END + 60)
rr, cc, xl, yl, xu, yu, d, j = (a[keep] for a in (rr, cc, xl, yl, xu, yu, d, j))
h = chm[rr, cc].astype(float)

# ---- NAIP greenness (NDVI) at the peaks
with rasterio.open(DER / "naip2022_r060.tif") as r:
    rows, cols = rasterio.transform.rowcol(r.transform, xu, yu)
    rows = np.clip(np.asarray(rows), 0, r.height - 1); cols = np.clip(np.asarray(cols), 0, r.width - 1)
    win = rasterio.windows.Window(int(cols.min()), int(rows.min()), int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1))
    red = r.read(1, window=win).astype(float); nir = r.read(4, window=win).astype(float)
ri, ci = rows - int(rows.min()), cols - int(cols.min())
ndvi = (nir[ri, ci] - red[ri, ci]) / np.maximum(nir[ri, ci] + red[ri, ci], 1.0)
ok = ndvi >= MIN_NDVI

# ---- not inside a LiDAR tree crown (near zone + its fringe)
tr = list(csv.DictReader(open(DER / "trees.csv")))
txy = np.array([[float(t["x_local"]), float(t["y_local"])] for t in tr]); tcr = np.array([float(t["crown_radius_m"]) for t in tr])
if len(txy):
    dn, jn = cKDTree(txy).query(np.c_[xl, yl])
    ok &= dn > np.maximum(tcr[jn], 2.5)
xl, yl, h, ndvi, d, j = (a[ok] for a in (xl, yl, h, ndvi, d, j))
cr = np.clip(0.32 * h, 2.5, 9.0)
dtm = Raster(DER / "dtm_r100_b150.tif")
z = dtm.sample(xl, yl)
ok = np.isfinite(z)
xl, yl, h, ndvi, d, j, cr, z = (a[ok] for a in (xl, yl, h, ndvi, d, j, cr, z))
n = len(xl)
hsh = ((np.round(xl * 10).astype(np.int64) * 73856093) ^ (np.round(yl * 10).astype(np.int64) * 19349663)) % 1000003 / 1000003.0

# ---- species (deterministic)
# oak savanna = the less green third of the far canopy (open crowns over dry grass), bay/laurel in the dense green rest
sp = np.where(ndvi < (np.percentile(ndvi, 35) if n else 0), "oak", "bay").astype("<U8")
sp = np.where((h > 24) & (hsh < 0.6), "euc", sp)
sp = np.where((h > 12) & (h <= 24) & (sp == "bay") & (hsh > 0.8), "con", sp)

# ---- meshes
G["trees"] = [m for m in G.get("trees", []) if not m["name"].startswith("far_")]
PROTOS = {(s, v): T3.make_proto(s, v, 1) for s in set(sp) for v in range(T3.N_PROTO)}
acc = {}
for k in range(n):
    s_ = str(sp[k]); v = int(hsh[k] * 997) % T3.N_PROTO
    sc = 0.88 + 0.24 * ((hsh[k] * 7919) % 1.0)
    yaw = 2 * math.pi * ((hsh[k] * 104729) % 1.0)
    a = 2 * math.pi * ((hsh[k] * 1299709) % 1.0)
    inst = T3.instance(PROTOS[(s_, v)], xl[k], yl[k], z[k] - 0.2, h[k] * sc, cr[k] * sc, s_, yaw, np.array([math.cos(a), math.sin(a)]), 0.04 * ((hsh[k] * 31) % 1.0))
    c = int(road.s[j[k]] // FAR_CHUNK)
    for key, mat in (("bark", "tree_trunk"), ("leaf", "tree_leaves")):
        acc.setdefault((c, mat), []).append(inst[key])
ndraw = 0
for (c, mat), parts in sorted(acc.items()):
    cur, nv, part = [], 0, 0
    def flush():
        global ndraw, part
        if not cur:
            return
        off = np.cumsum([0] + [len(p[0]) for p in cur])[:-1]
        P = np.concatenate([p[0] for p in cur]); Nn = np.concatenate([p[1] for p in cur]); UV = np.concatenate([p[2] for p in cur])
        T = np.concatenate([p[3] + o_ for p, o_ in zip(cur, off)])
        G["trees"].append(dict(name=f"far_{mat}_{c:02d}_{part}", material=mat, P=P, T=T, UV=UV, N=Nn, cast_shadows=False,
                               transparent=False, lod_out=LOD_OUT))
        ndraw += 1; part += 1
    for p in parts:
        if cur and nv + len(p[0]) > MAX_V:
            flush(); cur, nv = [], 0
        cur.append(p); nv += len(p[0])
    flush()

pickle.dump(geo, open(BUILD / "geometry.pkl", "wb"))
rep = dict(near_zone_m=NEAR_M, far_zone_m=[NEAR_M, FAR_M], far_trees=int(n), species={str(s): int((sp == s).sum()) for s in sorted(set(sp))},
           height_m=[round(float(np.min(h)), 1), round(float(np.median(h)), 1), round(float(np.max(h)), 1)] if n else None,
           draw_calls=ndraw, far_chunk_m=FAR_CHUNK, lod_out_m=LOD_OUT, near_lidar_trees=len(tr), seconds=round(time.time() - t0, 1))
out = (BUILD / "veg_far.json") if BUILD.name != "build" else (ROOT / "reports" / "veg_far.json")
out.write_text(json.dumps(rep, indent=1))
print(f"21f far vegetation: {n} trees {rep['species']}, {ndraw} meshes (draw calls), {rep['seconds']} s -> {out}")
