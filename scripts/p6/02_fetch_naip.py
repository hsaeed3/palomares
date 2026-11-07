#!/usr/bin/env python3
"""dev6 step 02: NAIP 2022-05 (0.6 m, RGB+NIR, public domain, USDA FSA) from Microsoft Planetary Computer, mosaicked
and warped to UTM 10N on a 0.6 m grid aligned to the corridor bbox of dtm_r100_b150. Pixels farther than 170 m from
the centreline are zeroed. -> data/derived/naip2022_r060.tif (4 bands uint8, R G B NIR). Idempotent."""
import sys, pathlib, requests, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import DER
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import from_bounds, Window
from scipy.spatial import cKDTree

OUT = DER / "naip2022_r060.tif"
if OUT.exists():
    print("exists", OUT); sys.exit(0)
ref = rasterio.open(DER / "dtm_r100_b150.tif"); b = ref.bounds; RES = 0.6
Wd, Ht = int((b.right - b.left) / RES), int((b.top - b.bottom) / RES)
tr = from_origin(b.left, b.top, RES, RES)
st = requests.post("https://planetarycomputer.microsoft.com/api/stac/v1/search",
                   json={"collections": ["naip"], "bbox": [-122.03, 37.59, -121.94, 37.70], "limit": 100,
                         "datetime": "2022-01-01/2022-12-31"}).json()["features"]
tok = requests.get("https://planetarycomputer.microsoft.com/api/sas/v1/token/naip").json()["token"]
urls = [f["assets"]["image"]["href"] + "?" + tok for f in st]
print(len(urls), "NAIP 2022 tiles:", [f["id"] for f in st])
cl = np.genfromtxt(DER / "centerline.csv", delimiter=",", names=True); KD = cKDTree(np.c_[cl["x_utm"], cl["y_utm"]][::2])
prof = dict(driver="GTiff", width=Wd, height=Ht, count=4, dtype="uint8", crs="EPSG:26910", transform=tr, tiled=True,
            blockxsize=512, blockysize=512, compress="deflate", predictor=2, nodata=0, BIGTIFF="IF_SAFER")
tmp = OUT.with_suffix(".tmp.tif")
with rasterio.Env(GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=3, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
    with rasterio.open(tmp, "w+", **prof) as dst:
        for u in urls:
            with rasterio.open("/vsicurl/" + u) as src, WarpedVRT(src, crs="EPSG:26910", transform=tr, width=Wd, height=Ht,
                                                                   resampling=Resampling.bilinear, nodata=0) as v:
                # footprint of this tile in the dst grid
                from rasterio.warp import transform_bounds
                l, bt, r, t = transform_bounds(src.crs, "EPSG:26910", *src.bounds)
                w0 = from_bounds(max(l, b.left), max(bt, b.bottom), min(r, b.right), min(t, b.top), tr).round_offsets().round_lengths()
                for r0 in range(int(w0.row_off), int(w0.row_off + w0.height), 1024):
                    for c0 in range(int(w0.col_off), int(w0.col_off + w0.width), 1024):
                        win = Window(c0, r0, min(1024, w0.col_off + w0.width - c0), min(1024, w0.row_off + w0.height - r0))
                        xs = tr.c + (c0 + np.array([0, win.width, 0, win.width, win.width / 2])) * RES
                        ys = tr.f - (r0 + np.array([0, 0, win.height, win.height, win.height / 2])) * RES
                        if KD.query(np.c_[xs, ys])[0].min() > 170 + 450: continue
                        a = v.read(window=win)
                        if not a.any(): continue
                        cc, rr = np.meshgrid(c0 + np.arange(win.width) + 0.5, r0 + np.arange(win.height) + 0.5)
                        far = KD.query(np.c_[tr.c + cc.ravel() * RES, tr.f - rr.ravel() * RES], distance_upper_bound=170)[0] > 170
                        a[:, far.reshape(cc.shape)] = 0
                        old = dst.read(window=win); m = a.any(0) & ~old.any(0); old[:, m] = a[:, m]
                        dst.write(old, window=win)
                print("  tile done", u.split("/")[-1][:40], flush=True)
tmp.rename(OUT); print("wrote", OUT)
