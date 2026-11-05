"""Bare-earth DTM by TIN (Delaunay linear) interpolation of class-2 ground returns, per 250 m chunk.
 - dtm_r050_b40.tif : 0.5 m, within 40 m of the OSM line
 - dtm_r100_b150.tif: 1.0 m, within 150 m
 - dsm_r100_b150.tif / chm_r100_b150.tif : 1 m first-surface and canopy height (DSM-DTM)
 - gdist_r050_b40.tif: distance (m) from each 0.5 m cell to the nearest ground return (gap diagnostic)
All rasters: EPSG:26910, NAVD88 metres."""
import json, numpy as np, rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree, Delaunay
from scipy.interpolate import LinearNDInterpolator
from common import *

o = origin(); idx = json.load(open(CACHE / "chunks" / "index.json"))
CH = idx["chunk_m"]
allx, ally = [], []
for k in range(idx["n"]):
    d = np.load(CACHE / "chunks" / f"chunk_{k:03d}.npz")
    allx.append([d["x"].min(), d["x"].max()]); ally.append([d["y"].min(), d["y"].max()])
X0 = np.floor(np.min(allx)) - 2; X1 = np.ceil(np.max(allx)) + 2
Y0 = np.floor(np.min(ally)) - 2; Y1 = np.ceil(np.max(ally)) + 2

def grid(res):
    nx, ny = int((X1 - X0) / res), int((Y1 - Y0) / res)
    return nx, ny, from_origin(o["x0"] + X0, o["y0"] + Y1, res, res)

rasters = {}
for name, res in [("dtm_r050_b40", .5), ("gdist_r050_b40", .5), ("dtm_r100_b150", 1.), ("dsm_r100_b150", 1.), ("chm_r100_b150", 1.)]:
    nx, ny, tf = grid(res)
    rasters[name] = (np.full((ny, nx), np.nan, np.float32), res, tf)

def load(k):
    if 0 <= k < idx["n"]:
        return np.load(CACHE / "chunks" / f"chunk_{k:03d}.npz")

for k in range(idx["n"]):
    ds = [load(j) for j in (k - 1, k, k + 1)]; ds = [d for d in ds if d is not None]
    cat = {f: np.concatenate([d[f] for d in ds]) for f in ("x", "y", "z", "c", "s", "d")}
    s0, s1 = k * CH, (k + 1) * CH
    if k == idx["n"] - 1: s1 = 1e9
    if k == 0: s0 = -1e9
    g = (cat["c"] == 2) | (cat["c"] == 17)          # bridge deck counts as drivable "ground"
    gx, gy, gz = cat["x"][g].astype(np.float64), cat["y"][g].astype(np.float64), cat["z"][g].astype(np.float64)
    # thin duplicate xy
    tin = LinearNDInterpolator(Delaunay(np.c_[gx, gy]), gz)
    gtree = cKDTree(np.c_[gx, gy])
    own = (cat["s"] >= s0) & (cat["s"] < s1)
    # station-ownership of cells: nearest point's station decides (avoids double writes across chunks)
    ownt = cKDTree(np.c_[cat["x"], cat["y"]])
    for name, (arr, res, tf) in rasters.items():
        buf = 40 if "b40" in name else 150
        xs = cat["x"][own]; ys = cat["y"][own]
        if len(xs) == 0: continue
        c0 = int((xs.min() - X0) / res) - 1; c1 = int((xs.max() - X0) / res) + 2
        r0 = int((Y1 - ys.max()) / res) - 1; r1 = int((Y1 - ys.min()) / res) + 2
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        cx = X0 + (cc + .5) * res; cy = Y1 - (rr + .5) * res
        q = np.c_[cx.ravel(), cy.ravel()]
        dn, jn = ownt.query(q, distance_upper_bound=3 * res + 2)
        ok = np.isfinite(dn)
        jn = np.where(ok, jn, 0)
        ok &= own[jn] & (np.abs(cat["d"][jn]) <= buf)
        if not ok.any(): continue
        qq = q[ok]; R = rr.ravel()[ok]; C = cc.ravel()[ok]
        if name.startswith("dtm"):
            arr[R, C] = tin(qq)
        elif name.startswith("gdist"):
            arr[R, C] = gtree.query(qq)[0]
        elif name.startswith("dsm"):
            m = own & ~np.isin(cat["c"], (7, 18))
            pc_ = np.floor((cat["x"][m] - X0) / res).astype(int); pr_ = np.floor((Y1 - cat["y"][m]) / res).astype(int)
            tmp = np.full(arr.shape[0] * arr.shape[1], -np.inf, np.float32)
            np.maximum.at(tmp, pr_.astype(np.int64) * arr.shape[1] + pc_, cat["z"][m])
            v = tmp[R.astype(np.int64) * arr.shape[1] + C]
            arr[R, C] = np.where(np.isfinite(v), v, np.nan)
    print("chunk", k, "ground", g.sum(), flush=True)

dsm, dtm1 = rasters["dsm_r100_b150"][0], rasters["dtm_r100_b150"][0]
chm = rasters["chm_r100_b150"][0]
chm[:] = np.where(np.isfinite(dtm1), np.clip(np.where(np.isfinite(dsm), dsm - dtm1, 0), 0, None), np.nan)
for name, (arr, res, tf) in rasters.items():
    with rasterio.open(DER / f"{name}.tif", "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=UTM, transform=tf, nodata=np.nan, compress="deflate", predictor=3, tiled=True) as w:
        w.write(arr, 1)
    print(name, arr.shape, np.isfinite(arr).sum())
