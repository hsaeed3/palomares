#!/usr/bin/env python3
"""dev6 step 05: understorey density raster from LiDAR returns 0.3-3 m above ground (class 1; this delivery has no
veg classes) / first returns, on the dtm_r100_b150 1 m grid, within 70 m of the road; band 2 = NAIP NDVI (0.6->1 m).
-> data/derived/understory.tif (float32: 1 = low-veg return fraction, 2 = NDVI). Idempotent."""
import sys, glob, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import DER, CACHE
import rasterio
from rasterio.warp import reproject, Resampling
R = rasterio.open(DER / "dtm_r100_b150.tif"); H, W = R.shape; INV = ~R.transform
low = np.zeros(H * W, np.float32); first = np.zeros(H * W, np.float32)
for f in glob.glob(str(CACHE / "p6/nodes/*.npz")):
    d = np.load(f)
    first[d["fc"]] += d["fn"]
    m = (d["lh"] >= 0.3) & (d["lh"] <= 3.0)
    c, r = INV * (d["lx"][m], d["ly"][m]); k = np.clip(r.astype(int), 0, H - 1) * W + np.clip(c.astype(int), 0, W - 1)
    np.add.at(low, k, 1)
frac = np.where(first > 0, np.minimum(low / np.maximum(first, 1), 1), np.nan).reshape(H, W).astype(np.float32)
N = rasterio.open(DER / "naip2022_r060.tif"); a = N.read([1, 4]).astype(np.float32)
nd = np.where(a.sum(0) > 0, (a[1] - a[0]) / (a[1] + a[0] + 1e-6), np.nan).astype(np.float32)
ndv = np.full((H, W), np.nan, np.float32)
reproject(nd, ndv, src_transform=N.transform, src_crs=N.crs, dst_transform=R.transform, dst_crs=R.crs, resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
p = R.profile; p.update(count=2, dtype="float32", compress="deflate", nodata=np.nan, tiled=True)
with rasterio.open(DER / "understory.tif", "w", **p) as o: o.write(frac, 1); o.write(ndv, 2)
v = np.isfinite(frac); print("cells", v.sum(), "shrubby(>0.15)", int((frac[v] > 0.15).sum()), "median frac", float(np.nanmedian(frac)))
