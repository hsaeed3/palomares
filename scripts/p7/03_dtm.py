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
import glob, json, os, gc as GC
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
# Class 17 (bridge deck) is only drivable ground where a *road* bridge is (OSM highway bridge, 6 m buffer). Every other
# class-17 return is an overhead structure (UP / Niles Canyon Railway decks over SR-84, Niles Blvd, Sullivan Underpass,
# rail bridges over the creek) -> dropped so the TIN runs underneath. Rail-bridge buffers (15 m) win over road ones.
import shapely
_hb, _rb = [], []
for e in json.load(open(DER7 / "osm_features.geojson"))["features"]:
    q = e["properties"]
    if "bridge" in q["kind"] and e["geometry"]["type"] == "LineString" and (q["kind"].startswith("rail_") or q["kind"].startswith("highway_")):
        c = np.array(e["geometry"]["coordinates"]); x_, y_ = ll2loc(c[:, 0], c[:, 1])
        L_ = shapely.LineString(np.c_[x_, y_])
        if q["kind"].startswith("rail_"): _rb.append(L_.buffer(15.0))
        elif q.get("highway") not in ("footway", "path", "cycleway", "steps"): _hb.append(L_.buffer(6.0))
ROADB = shapely.union_all(_hb) if _hb else shapely.Polygon()
if _rb: ROADB = ROADB.difference(shapely.union_all(_rb))
drop = np.zeros(len(gx), bool)
m17 = np.flatnonzero(gc == 17); drop[m17] = ~shapely.contains_xy(ROADB, gx[m17].astype(float), gy[m17].astype(float))
gx, gy, gz, gi, gc = gx[~drop], gy[~drop], gz[~drop], gi[~drop], gc[~drop]
print("dropped non-road-bridge class-17 pts", int(drop.sum()))
print("ground pts", len(gx), "classes", {i: int(v) for i, v in enumerate(cc) if v}, flush=True)
json.dump({"class_counts": {str(i): int(v) for i, v in enumerate(cc) if v}, "ground_pts_dedup": int(len(gx)), "non_road_bridge_17_dropped": int(drop.sum())}, open(CACHE7 / "class_counts.json", "w"))
dsm = dsm.reshape(GH, GW); dsm[dsm < -1e8] = np.nan
TF1 = from_origin(GX0 + X0, GY1 + Y0, 1.0, 1.0); TF05 = from_origin(GX0 + X0, GY1 + Y0, 0.5, 0.5)
H2, W2 = GH * 2, GW * 2
m1 = rasterize([(TER, 1)], out_shape=(GH, GW), transform=from_origin(GX0, GY1, 1, 1), dtype="uint8").astype(bool)
m05 = rasterize([(DET, 1)], out_shape=(H2, W2), transform=from_origin(GX0, GY1, .5, .5), dtype="uint8").astype(bool)
dtm1 = np.full((GH, GW), np.nan, np.float32)
from rasterio.windows import Window
TMP = {k: rasterio.open(CACHE7 / f"tmp_{k}.tif", "w", driver="GTiff", height=H2, width=W2, count=1, dtype="float32", crs=UTM,
                        transform=TF05, nodata=np.nan, tiled=True, BIGTIFF="YES") for k in ("dtm_r050_b40", "gdist_r050_b40", "int_r050_b40")}
b17 = np.zeros(GH * GW, np.uint16)
c1 = np.floor(gx - GX0).astype(np.int64); r1 = np.floor(GY1 - gy).astype(np.int64)
o17 = (gc == 17) & (c1 >= 0) & (c1 < GW) & (r1 >= 0) & (r1 < GH)
np.add.at(b17, r1[o17] * GW + c1[o17], 1)
# tiles
DBG = os.environ.get('DBG')
T = 100.0; HALO = 15.0   # small tiles keep Delaunay + trapezoid map < ~0.5 GB
tc = np.floor((gx - GX0) / T).astype(np.int32); tr = np.floor((GY1 - gy) / T).astype(np.int32)
NTC, NTR = int(np.ceil(GW / T)), int(np.ceil(GH / T))
order = np.lexsort((tc, tr)); keyv = tr[order].astype(np.int64) * NTC + tc[order]
starts = np.searchsorted(keyv, np.arange(NTR * NTC)); ends = np.searchsorted(keyv, np.arange(NTR * NTC), side="right"); del keyv
del c1, r1, o17, tc, tr
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
        if DBG: print("tile", R, C, len(ids), "rss MB", int(open('/proc/self/statm').read().split()[1]) * 4096 // 2**20, flush=True)
        TRI = mtri.Triangulation(P[:, 0], P[:, 1], Delaunay(P).simplices)
        _I = mtri.LinearTriInterpolator(TRI, gz[ids].astype(np.float64))
        tin = lambda q: np.ma.filled(_I(q[:, 0], q[:, 1]).astype(np.float64), np.nan)
        rr, cc2 = np.nonzero(sub1)
        dtm1[r0 + rr, c0 + cc2] = tin(np.c_[GX0 + c0 + cc2 + .5, GY1 - (r0 + rr) - .5])
        sub05 = m05[2 * r0:2 * r1_, 2 * c0:2 * c1_]
        if sub05.any():
            rr, cc2 = np.nonzero(sub05)
            q = np.c_[GX0 + (2 * c0 + cc2 + .5) / 2, GY1 - (2 * r0 + rr + .5) / 2]
            hh, ww = sub05.shape; win = Window(2 * c0, 2 * r0, ww, hh)
            blk = np.full((hh, ww), np.nan, np.float32); blk[rr, cc2] = tin(q); TMP["dtm_r050_b40"].write(blk, 1, window=win)
            blk = np.full((hh, ww), np.nan, np.float32); blk[rr, cc2] = cKDTree(P).query(q, workers=2)[0]; TMP["gdist_r050_b40"].write(blk, 1, window=win)
            # mean ground (class 2) intensity per 0.5 m cell
            ii = ids[gc[ids] == 2]
            ci = np.floor((gx[ii] - GX0) * 2).astype(np.int64) - 2 * c0; ri = np.floor((GY1 - gy[ii]) * 2).astype(np.int64) - 2 * r0
            o_ = (ci >= 0) & (ci < ww) & (ri >= 0) & (ri < hh)
            lid = ri[o_] * ww + ci[o_]
            isum = np.bincount(lid, weights=gi[ii][o_].astype(np.float64), minlength=hh * ww); icnt = np.bincount(lid, minlength=hh * ww)
            blk = np.where(icnt > 0, isum / np.maximum(icnt, 1), np.nan).reshape(hh, ww).astype(np.float32)
            TMP["int_r050_b40"].write(np.where(sub05, blk, np.nan).astype(np.float32), 1, window=win)
        del tin, _I, TRI; GC.collect()          # matplotlib.tri keeps reference cycles (~100 MB/tile otherwise)
    if R % 5 == 0: print("tile row", R, "/", NTR, flush=True)
chm = np.where(np.isfinite(dtm1), np.clip(np.where(np.isfinite(dsm), dsm - dtm1, 0), 0, None), np.nan).astype(np.float32)
dsm = np.where(np.isfinite(dtm1), np.where(np.isfinite(dsm), np.maximum(dsm, dtm1), dtm1), np.nan).astype(np.float32)


def wr(name, arr, tf, dtype="float32", nodata=np.nan):
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(DER7 / f"{name}.tif", "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0], dtype=dtype,
                       crs=UTM, transform=tf, nodata=nodata, compress="deflate", predictor=3 if dtype == "float32" else 2, tiled=True, BIGTIFF="IF_SAFER") as w:
        w.write(arr.astype(dtype))
    print(name, arr.shape, flush=True)


for k, ds in TMP.items():
    ds.close()
    with rasterio.open(CACHE7 / f"tmp_{k}.tif") as src:
        prof = src.profile; prof.update(compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
        with rasterio.open(DER7 / f"{k}.tif", "w", **prof) as dst:
            for _, w_ in src.block_windows(1): dst.write(src.read(1, window=w_), 1, window=w_)
    (CACHE7 / f"tmp_{k}.tif").unlink(); print(k, flush=True)
wr("dtm_r100_b150", dtm1, TF1); wr("dsm_r100_b150", dsm, TF1); wr("chm_r100_b150", chm, TF1)
wr("cover_r100_b150", np.stack([nf.reshape(GH, GW), ng.reshape(GH, GW), b17.reshape(GH, GW)]), TF1, "uint16", 0)
print("done; dtm1 cells", int(np.isfinite(dtm1).sum()))
