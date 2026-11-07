#!/usr/bin/env python3
"""dev7 D-03: rasters from the streamed EPT nodes (data/cache/p7/nodes), same method as Phase-1 scripts/04_dtm.py
(Delaunay-TIN linear interpolation via matplotlib's trapezoid-map tri finder of class 2 ground + class 17 bridge deck), but tiled on a fixed grid instead of
along one road, so the whole town/canyon network is covered. All GeoTIFF EPSG:26910, NAVD88 m:
  dtm_r050_b40.tif    0.5 m bare earth inside the 40 m detail area (roads, parking, town rail)
  dtm_r100_b150.tif   1 m bare earth inside the 150 m terrain area (+ the 0.65 km^2 town polygon)
  gdist_r050_b40.tif  0.5 m distance to nearest ground/bridge return (TIN-gap diagnostic)
  int_r050_b40.tif    0.5 m mean ground intensity (paint / asphalt), NaN where no returns
  dsm_r100_b150.tif   1 m first-surface (max z, all non-noise classes)
  chm_r100_b150.tif   1 m canopy/building height = DSM - DTM (>= 0)
  cover_r100_b150.tif 1 m band1 first-return count, band2 ground-return count, band3 bridge(17) count
Run: nice -n5 python3 scripts/p7/03_dtm.py
"""
import glob, json
import numpy as np, rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import wkt
from scipy.spatial import cKDTree, Delaunay
import matplotlib.tri as mtri
from common7 import *

g = json.load(open(CACHE7 / "grid.json")); GX0, GY1, GW, GH = g["x0_local"], g["y1_local"], g["w"], g["h"]
sc = json.load(open(DER7 / "scope.json"))
DET, TER = wkt.loads(sc["detail_40m"]), wkt.loads(sc["terrain_150m"])
GA = CACHE7 / "ground_all.npz"
files = sorted(glob.glob(str(CACHE7 / "nodes" / "*.npz")))
dsm = np.full(GH * GW, -1e9, np.float32); nf = np.zeros(GH * GW, np.uint16); ng = np.zeros(GH * GW, np.uint16)
cc = np.zeros(32, np.int64)
if GA.exists():
    A = dict(np.load(GA)); gx, gy, gz, gi, gc = A["x"], A["y"], A["z"], A["i"], A["c"]
for k, f in enumerate(files):
    d = np.load(f)
    np.maximum.at(dsm, d["dsm_cell"], d["dsm_z"]); np.add.at(nf, d["f_cell"], d["f_n"]); np.add.at(ng, d["g_cell"], d["g_n"])
    cc += d["ccount"]
if not GA.exists():
    parts = {k: [] for k in ("x", "y", "z", "i", "c")}
    for f in files:
        d = np.load(f)
        for k, s in (("x", "gx"), ("y", "gy"), ("z", "gz"), ("i", "gi"), ("c", "gc")): parts[k].append(d[s])
    A = {k: np.concatenate(v) for k, v in parts.items()}
    # drop exact duplicates between overlapping datasets (1 cm)
    key = np.round(A["x"] * 100).astype(np.int64) * 10_000_000 + np.round(A["y"] * 100).astype(np.int64)
    _, u = np.unique(key, return_index=True); A = {k: v[u] for k, v in A.items()}
    np.savez(GA, **A); gx, gy, gz, gi, gc = A["x"], A["y"], A["z"], A["i"], A["c"]
print("ground pts", len(gx), "classes", {i: int(v) for i, v in enumerate(cc) if v}, flush=True)
json.dump({"class_counts": {str(i): int(v) for i, v in enumerate(cc) if v}, "ground_pts_dedup": int(len(gx))}, open(CACHE7 / "class_counts.json", "w"))
dsm = dsm.reshape(GH, GW); dsm[dsm < -1e8] = np.nan
TF1 = from_origin(GX0 + X0, GY1 + Y0, 1.0, 1.0); TF05 = from_origin(GX0 + X0, GY1 + Y0, 0.5, 0.5)
H2, W2 = GH * 2, GW * 2
m1 = rasterize([(TER, 1)], out_shape=(GH, GW), transform=from_origin(GX0, GY1, 1, 1), dtype="uint8").astype(bool)
m05 = rasterize([(DET, 1)], out_shape=(H2, W2), transform=from_origin(GX0, GY1, .5, .5), dtype="uint8").astype(bool)
dtm1 = np.full((GH, GW), np.nan, np.float32); dtm05 = np.full((H2, W2), np.nan, np.float32)
gd05 = np.full((H2, W2), np.nan, np.float32)
# intensity per 0.5 m cell (ground class 2 only)
c05 = np.floor((gx - GX0) * 2).astype(np.int64); r05 = np.floor((GY1 - gy) * 2).astype(np.int64)
ok = (c05 >= 0) & (c05 < W2) & (r05 >= 0) & (r05 < H2) & (gc == 2)
cid = r05[ok] * W2 + c05[ok]
isum = np.bincount(cid, weights=gi[ok].astype(np.float64), minlength=H2 * W2)
icnt = np.bincount(cid, minlength=H2 * W2)
inten = np.where(icnt > 0, isum / np.maximum(icnt, 1), np.nan).astype(np.float32).reshape(H2, W2)
inten[~m05] = np.nan; del isum, icnt, cid
b17 = np.zeros(GH * GW, np.uint16)
c1 = np.floor(gx - GX0).astype(np.int64); r1 = np.floor(GY1 - gy).astype(np.int64)
o17 = (gc == 17) & (c1 >= 0) & (c1 < GW) & (r1 >= 0) & (r1 < GH)
np.add.at(b17, r1[o17] * GW + c1[o17], 1)
# tiles
T = TILE; HALO = 25.0
tc = np.floor((gx - GX0) / T).astype(np.int32); tr = np.floor((GY1 - gy) / T).astype(np.int32)
NTC, NTR = int(np.ceil(GW / T)), int(np.ceil(GH / T))
order = np.lexsort((tc, tr)); keyv = tr[order].astype(np.int64) * NTC + tc[order]
starts = np.searchsorted(keyv, np.arange(NTR * NTC)); ends = np.searchsorted(keyv, np.arange(NTR * NTC), side="right")
tree = cKDTree(np.c_[gx, gy])
for R in range(NTR):
    for C in range(NTC):
        r0, c0 = int(R * T), int(C * T); r1_, c1_ = min(GH, int((R + 1) * T)), min(GW, int((C + 1) * T))
        sub1 = m1[r0:r1_, c0:c1_]
        if not sub1.any(): continue
        ids = np.concatenate([order[starts[k]:ends[k]] for rr in (R - 1, R, R + 1) for cc_ in (C - 1, C, C + 1)
                              if 0 <= rr < NTR and 0 <= cc_ < NTC for k in [rr * NTC + cc_]])
        if len(ids) < 10: continue
        x0, x1 = GX0 + c0 - HALO, GX0 + c1_ + HALO; y1, y0 = GY1 - r0 + HALO, GY1 - r1_ - HALO
        ids = ids[(gx[ids] > x0) & (gx[ids] < x1) & (gy[ids] > y0) & (gy[ids] < y1)]
        if len(ids) < 10: continue
        P = np.c_[gx[ids], gy[ids]].astype(np.float64)
        TRI = mtri.Triangulation(P[:, 0], P[:, 1], Delaunay(P).simplices)
        _I = mtri.LinearTriInterpolator(TRI, gz[ids].astype(np.float64))
        tin = lambda q: np.ma.filled(_I(q[:, 0], q[:, 1]).astype(np.float64), np.nan)
        rr, cc2 = np.nonzero(sub1)
        dtm1[r0 + rr, c0 + cc2] = tin(np.c_[GX0 + c0 + cc2 + .5, GY1 - (r0 + rr) - .5])
        sub05 = m05[2 * r0:2 * r1_, 2 * c0:2 * c1_]
        if sub05.any():
            rr, cc2 = np.nonzero(sub05)
            q = np.c_[GX0 + (2 * c0 + cc2 + .5) / 2, GY1 - (2 * r0 + rr + .5) / 2]
            dtm05[2 * r0 + rr, 2 * c0 + cc2] = tin(q)
            gd05[2 * r0 + rr, 2 * c0 + cc2] = tree.query(q, workers=2)[0]
    print("tile row", R, "/", NTR, flush=True)
chm = np.where(np.isfinite(dtm1), np.clip(np.where(np.isfinite(dsm), dsm - dtm1, 0), 0, None), np.nan).astype(np.float32)
dsm = np.where(np.isfinite(dtm1), np.where(np.isfinite(dsm), np.maximum(dsm, dtm1), dtm1), np.nan).astype(np.float32)


def wr(name, arr, tf, dtype="float32", nodata=np.nan):
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(DER7 / f"{name}.tif", "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0], dtype=dtype,
                       crs=UTM, transform=tf, nodata=nodata, compress="deflate", predictor=3 if dtype == "float32" else 2, tiled=True, BIGTIFF="IF_SAFER") as w:
        w.write(arr.astype(dtype))
    print(name, arr.shape, flush=True)


wr("dtm_r050_b40", dtm05, TF05); wr("gdist_r050_b40", gd05, TF05); wr("int_r050_b40", inten, TF05)
wr("dtm_r100_b150", dtm1, TF1); wr("dsm_r100_b150", dsm, TF1); wr("chm_r100_b150", chm, TF1)
wr("cover_r100_b150", np.stack([nf.reshape(GH, GW), ng.reshape(GH, GW), b17.reshape(GH, GW)]), TF1, "uint16", 0)
print("done; dtm05 cells", int(np.isfinite(dtm05).sum()), "dtm1 cells", int(np.isfinite(dtm1).sum()))
