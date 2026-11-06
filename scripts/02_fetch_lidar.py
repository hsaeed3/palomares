"""Clip USGS 3DEP EPT (Web Mercator) point clouds to the 150 m road corridor.
Traverses the EPT octree hierarchy, downloads intersecting LAZ nodes (cached in data/raw/ept/),
reprojects to UTM 10N and stores per-tile compact npz in data/cache/pts/."""
import json, io, sys, requests, numpy as np, laspy
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import LineString, box
from shapely.ops import transform as shp_tf
from shapely.prepared import prep
from common import *

BUF = 150.0
KEEP = {1, 2, 3, 4, 5, 6, 7, 9, 17, 18}   # 1 (unclassified) & 7/18 kept for diagnostics/noise masking
cl = np.loadtxt(DER / "osm_centerline_utm.csv", delimiter=",", skiprows=1)
corr = LineString(cl[:, 1:3]).buffer(BUF)
corr_wm = shp_tf(lambda x, y, z=None: T_UTM_WM.transform(x, y), corr.buffer(5))
pc = prep(corr_wm)
from scipy.spatial import cKDTree
_ls = LineString(cl[:, 1:3]); _d = np.arange(0, _ls.length, 2.0)
TREE = cKDTree(np.array([_ls.interpolate(t).coords[0] for t in _d]))
S = requests.Session()
(RAW / "ept").mkdir(exist_ok=True); (CACHE / "pts").mkdir(exist_ok=True)

def get(url, f=None):
    if f is not None and f.exists():
        return f.read_bytes()
    for i in range(5):
        try:
            r = S.get(url, timeout=120); r.raise_for_status()
            if f is not None: f.write_bytes(r.content)
            return r.content
        except Exception as e:
            err = e
    raise err

def nodes_for(ds):
    meta = json.loads(get(f"{EPT_BASE}/{ds}/ept.json", RAW / f"ept_{ds}.json"))
    b = meta["bounds"]
    out = []
    def walk(hfile):
        h = json.loads(get(f"{EPT_BASE}/{ds}/ept-hierarchy/{hfile}.json", RAW / "ept" / f"{ds}_h_{hfile}.json"))
        stack = [hfile]
        while stack:
            key = stack.pop()
            if key not in h: continue
            d, X, Y, Z = map(int, key.split("-"))
            w = (b[3] - b[0]) / 2 ** d
            x0, y0 = b[0] + X * w, b[1] + Y * w
            if not pc.intersects(box(x0, y0, x0 + w, y0 + w)): continue
            if h[key] == -1 and key != hfile:
                walk(key); continue
            if h[key] > 0: out.append(key)
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        stack.append(f"{d+1}-{2*X+dx}-{2*Y+dy}-{2*Z+dz}")
    walk("0-0-0-0")
    return out

def proc(args):
    ds, key = args
    fo = CACHE / "pts" / f"{ds}_{key}.npz"
    if fo.exists(): return fo
    raw = get(f"{EPT_BASE}/{ds}/ept-data/{key}.laz")
    las = laspy.read(io.BytesIO(raw))
    x, y = T_WM_UTM.transform(np.asarray(las.x), np.asarray(las.y))
    c = np.asarray(las.classification)
    m = np.isin(c, list(KEEP))
    m[m] = TREE.query(np.c_[x[m], y[m]], distance_upper_bound=BUF)[0] <= BUF
    np.savez(fo, x=x[m].astype(np.float64), y=y[m], z=np.asarray(las.z)[m].astype(np.float32),
             i=np.asarray(las.intensity)[m].astype(np.uint16), c=c[m].astype(np.uint8),
             rn=np.asarray(las.return_number)[m].astype(np.uint8), nr=np.asarray(las.number_of_returns)[m].astype(np.uint8))
    return fo

if __name__ == "__main__":
    jobs = []
    for ds in EPT_SETS:
        ks = nodes_for(ds); print(ds, len(ks), "nodes", flush=True)
        jobs += [(ds, k) for k in ks]
    json.dump(jobs, open(CACHE / "ept_jobs.json", "w"))
    done = 0
    with ThreadPoolExecutor(12) as ex:
        for f in ex.map(proc, jobs):
            done += 1
            if done % 100 == 0: print(done, "/", len(jobs), flush=True)
    print("done", len(jobs))
