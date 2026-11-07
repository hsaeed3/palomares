#!/usr/bin/env python3
"""dev6 step 01: re-stream the 2021 USGS Alameda 3DEP EPT (same sets/CRS rules as scripts/02_fetch_lidar.py) and keep
only what the measured-scenery steps need. Nothing raw is cached on disk; each EPT node is downloaded, reprojected
EPSG:3857 -> UTM 10N, reduced, and written to data/cache/p6/nodes/<ds>_<key>.npz (idempotent: done nodes are skipped).

Kept per node (UTM metres, NAVD88 z, HAG = z - dtm_r100_b150):
  bld  : class 6 within 160 m of the refined centerline         (x, y, z, rn, nr)
  low  : class 1/3/4/5 with 0.25 <= HAG <= 3.5 within 70 m        (x, y, z, hag, c, i)
  cnt  : 1 m grid (dtm_r100 grid) index + count of all first returns / ground returns within 70 m
Run: nice -n5 python3 scripts/p6/01_fetch_lidar_nonground.py [--workers 3]
"""
import sys, io, json, pathlib, argparse, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import DER, CACHE, RAW, T_UTM_WM, T_WM_UTM, EPT_BASE, EPT_SETS
import requests, laspy, rasterio
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from shapely.geometry import LineString, box
from shapely.ops import transform as shp_tf
from shapely.prepared import prep
from concurrent.futures import ThreadPoolExecutor

OUT = CACHE / "p6" / "nodes"; OUT.mkdir(parents=True, exist_ok=True)
META = CACHE / "p6" / "ept"; META.mkdir(parents=True, exist_ok=True)
BLD_BUF, LOW_BUF = 160.0, 70.0
cl = np.genfromtxt(DER / "centerline.csv", delimiter=",", names=True)
CLXY = np.c_[cl["x_utm"], cl["y_utm"]]
TREE = cKDTree(CLXY[::2])
corr = LineString(CLXY).buffer(BLD_BUF + 5)
pc = prep(shp_tf(lambda x, y, z=None: T_UTM_WM.transform(x, y), corr))
R = rasterio.open(DER / "dtm_r100_b150.tif"); DTM = R.read(1); INV = ~R.transform; H, W = DTM.shape
S = requests.Session()


def get(url, f=None):
    if f is not None and f.exists():
        return f.read_bytes()
    err = None
    for k in range(6):
        try:
            r = S.get(url, timeout=180); r.raise_for_status()
            if f is not None: f.write_bytes(r.content)
            return r.content
        except Exception as e:
            err = e; time.sleep(2 + 3 * k)
    raise err


def nodes_for(ds):
    meta = json.loads(get(f"{EPT_BASE}/{ds}/ept.json", META / f"ept_{ds}.json")); b = meta["bounds"]; out = []
    def walk(hf):
        h = json.loads(get(f"{EPT_BASE}/{ds}/ept-hierarchy/{hf}.json", META / f"{ds}_h_{hf}.json")); st = [hf]
        while st:
            key = st.pop()
            if key not in h: continue
            d, X, Y, Z = map(int, key.split("-")); w = (b[3] - b[0]) / 2 ** d; x0, y0 = b[0] + X * w, b[1] + Y * w
            if not pc.intersects(box(x0, y0, x0 + w, y0 + w)): continue
            if h[key] == -1 and key != hf: walk(key); continue
            if h[key] > 0: out.append(key)
            st += [f"{d+1}-{2*X+a}-{2*Y+c}-{2*Z+e}" for a in (0, 1) for c in (0, 1) for e in (0, 1)]
    walk("0-0-0-0"); return out


def hag_of(x, y, z):
    c, r = INV * (x, y)
    g = map_coordinates(DTM, [r - 0.5, c - 0.5], order=1, mode="nearest", cval=np.nan)
    return z - g, np.clip(r.astype(int), 0, H - 1) * W + np.clip(c.astype(int), 0, W - 1)


def proc(job):
    ds, key = job; fo = OUT / f"{ds}_{key}.npz"
    if fo.exists(): return 0
    las = laspy.read(io.BytesIO(get(f"{EPT_BASE}/{ds}/ept-data/{key}.laz")))
    x, y = T_WM_UTM.transform(np.asarray(las.x), np.asarray(las.y)); z = np.asarray(las.z, np.float64)
    c = np.asarray(las.classification); rn = np.asarray(las.return_number); nr = np.asarray(las.number_of_returns)
    dd = TREE.query(np.c_[x, y], distance_upper_bound=BLD_BUF)[0]
    o = {}
    mb = (c == 6) & (dd <= BLD_BUF)
    o.update(bx=x[mb], by=y[mb], bz=z[mb].astype(np.float32), brn=rn[mb].astype(np.uint8), bnr=nr[mb].astype(np.uint8))
    near = dd <= LOW_BUF
    hag, cell = hag_of(x[near], y[near], z[near]); cn = c[near]
    ml = np.isin(cn, (1, 3, 4, 5)) & (hag >= 0.25) & (hag <= 3.5)
    xi, yi = x[near][ml], y[near][ml]
    o.update(lx=xi, ly=yi, lz=z[near][ml].astype(np.float32), lh=hag[ml].astype(np.float32), lc=cn[ml].astype(np.uint8),
             li=np.asarray(las.intensity)[near][ml].astype(np.uint16))
    first = rn[near] == 1
    u, k = np.unique(cell[first], return_counts=True); o.update(fc=u.astype(np.int64), fn=k.astype(np.uint32))
    u, k = np.unique(cell[cn == 2], return_counts=True); o.update(gc=u.astype(np.int64), gn=k.astype(np.uint32))
    np.savez_compressed(fo, **o); return len(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    jf = CACHE / "p6" / "jobs.json"
    if jf.exists(): jobs = json.load(open(jf))
    else:
        jobs = []
        for ds in EPT_SETS:
            ks = nodes_for(ds); print(ds, len(ks), "nodes", flush=True); jobs += [(ds, k) for k in ks]
        json.dump(jobs, open(jf, "w"))
    todo = [j for j in jobs if not (OUT / f"{j[0]}_{j[1]}.npz").exists()]
    print(len(jobs), "nodes,", len(todo), "to do", flush=True); n = 0; t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for k, _ in enumerate(ex.map(proc, todo), 1):
            if k % 50 == 0: print(f"{k}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    print("done")
