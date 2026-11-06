"""Public-domain USGS imagery (USGSImageryOnly, NAIP-based) and a 3DEP 10 m backdrop DEM, all in UTM 10N.
 - ortho_r100.tif     : 1 m RGB over the corridor bbox
 - backdrop_dem_r10.tif / backdrop_ortho_r05.tif : bbox + 2 km (distant hills for viewer / Phase 2 horizon)"""
import io, numpy as np, rasterio, requests, time
from rasterio.transform import from_origin
from PIL import Image
from common import *
IMG = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/export"
DEM = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
with rasterio.open(DER / "dtm_r100_b150.tif") as r: b = r.bounds

def get(url, p):
    for t in range(5):
        try:
            q = requests.get(url, params=p, timeout=180); q.raise_for_status(); return q.content
        except Exception as e: print("retry", e); time.sleep(2 ** t)
    raise RuntimeError(url)

def mosaic(x0, y0, x1, y1, res, kind, tile=2000):
    W, H = int((x1 - x0) / res), int((y1 - y0) / res)
    out = np.zeros((3, H, W), np.uint8) if kind == "img" else np.full((H, W), np.nan, np.float32)
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            w, h = min(tile, W - c0), min(tile, H - r0)
            bb = f"{x0 + c0*res},{y1 - (r0+h)*res},{x0 + (c0+w)*res},{y1 - r0*res}"
            if kind == "img":
                a = np.array(Image.open(io.BytesIO(get(IMG, dict(bbox=bb, bboxSR=26910, imageSR=26910, size=f"{w},{h}", format="jpg", f="image")))).convert("RGB"))
                out[:, r0:r0 + h, c0:c0 + w] = a.transpose(2, 0, 1)
            else:
                with rasterio.MemoryFile(get(DEM, dict(bbox=bb, bboxSR=26910, imageSR=26910, size=f"{w},{h}", format="tiff", pixelType="F32",
                                                       interpolation="RSP_BilinearInterpolation", f="image"))) as m:
                    out[r0:r0 + h, c0:c0 + w] = m.open().read(1)
    return out, from_origin(x0, y1, res, res)

def save(name, a, tf):
    if a.ndim == 2: a = a[None]
    prof = dict(driver="GTiff", height=a.shape[1], width=a.shape[2], count=a.shape[0], dtype=a.dtype, crs=UTM, transform=tf, tiled=True)
    prof.update(dict(compress="jpeg", photometric="ycbcr", jpeg_quality=88) if a.dtype == np.uint8 else dict(compress="deflate", predictor=3, nodata=np.nan))
    with rasterio.open(DER / name, "w", **prof) as w: w.write(a)
    print(name, a.shape)

if not (DER / "ortho_r100.tif").exists():
    save("ortho_r100.tif", *mosaic(b.left, b.bottom, b.right, b.top, 1.0, "img"))
B = (b.left - 2000, b.bottom - 2000, b.right + 2000, b.top + 2000)
if not (DER / "backdrop_dem_r10.tif").exists():
    save("backdrop_dem_r10.tif", *mosaic(*B, 10.0, "dem"))
if not (DER / "backdrop_ortho_r05.tif").exists():
    save("backdrop_ortho_r05.tif", *mosaic(*B, 5.0, "img"))
