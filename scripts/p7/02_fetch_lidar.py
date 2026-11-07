#!/usr/bin/env python3
"""dev7 D-02: stream the USGS 3DEP 2021 Alameda EPT (CA_AlamedaCo_{1,2,3}_2021, EPSG:3857) over the dev7 150 m
terrain area (data/derived/dev7/scope.json). Raw LAZ is never stored; each EPT node is downloaded, reprojected
3857 -> UTM 10N (-> fixed local origin), reduced and written to data/cache/p7/nodes/<ds>_<key>.npz (idempotent).

Kept per node:
  ground  : class 2 (ground) + 17 (bridge deck) inside the terrain area: x,y (local, f64->f32), z, intensity, class
  dsm     : per 1 m cell of the global grid (data/cache/p7/grid.json) max z of all non-noise returns (not 7/18)
  first   : per 1 m cell count of first returns, and of ground returns (coverage/penetration diagnostics)
  ccount  : class histogram
Run: nice -n5 python3 scripts/p7/02_fetch_lidar.py [--workers 4]
"""
import io, json, time, argparse
import numpy as np, requests, laspy, shapely
from shapely import wkt
from shapely.geometry import box
from shapely.ops import transform as shp_tf
from shapely.prepared import prep
from concurrent.futures import ThreadPoolExecutor
from common7 import *

OUT = CACHE7 / "nodes"; OUT.mkdir(parents=True, exist_ok=True)
META = CACHE7 / "ept"; META.mkdir(parents=True, exist_ok=True)
sc = json.load(open(DER7 / "scope.json"))
TER = wkt.loads(sc["terrain_150m"]); shapely.prepare(TER)
b = TER.bounds
GX0, GY1 = np.floor(b[0]) - 4, np.ceil(b[3]) + 4
GW, GH = int(np.ceil(b[2] + 4 - GX0)), int(np.ceil(GY1 - (b[1] - 4)))
json.dump({"x0_local": GX0, "y1_local": GY1, "w": GW, "h": GH, "res": 1.0}, open(CACHE7 / "grid.json", "w"))
pc = prep(shp_tf(lambda x, y, z=None: T_UTM_WM.transform(np.asarray(x) + X0, np.asarray(y) + Y0), TER.buffer(3)))
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
    meta = json.loads(get(f"{EPT_BASE}/{ds}/ept.json", META / f"ept_{ds}.json")); bb = meta["bounds"]; out = []
    def walk(hf):
        h = json.loads(get(f"{EPT_BASE}/{ds}/ept-hierarchy/{hf}.json", META / f"{ds}_h_{hf}.json")); st = [hf]
        while st:
            key = st.pop()
            if key not in h: continue
            d, X, Y, Z = map(int, key.split("-")); w = (bb[3] - bb[0]) / 2 ** d; x0, y0 = bb[0] + X * w, bb[1] + Y * w
            if not pc.intersects(box(x0, y0, x0 + w, y0 + w)): continue
            if h[key] == -1 and key != hf: walk(key); continue
            if h[key] > 0: out.append((key, h[key]))
            st += [f"{d+1}-{2*X+a}-{2*Y+c}-{2*Z+e}" for a in (0, 1) for c in (0, 1) for e in (0, 1)]
    walk("0-0-0-0"); return out


def proc(job):
    ds, key = job; fo = OUT / f"{ds}_{key}.npz"
    if fo.exists(): return 0
    las = laspy.read(io.BytesIO(get(f"{EPT_BASE}/{ds}/ept-data/{key}.laz")))
    ux, uy = T_WM_UTM.transform(np.asarray(las.x), np.asarray(las.y)); x = ux - X0; y = uy - Y0
    z = np.asarray(las.z, np.float64); c = np.asarray(las.classification); rn = np.asarray(las.return_number)
    inside = shapely.contains_xy(TER, x, y)
    cc = np.bincount(c[inside], minlength=32)
    g = inside & ((c == 2) | (c == 17))
    col = np.floor(x - GX0).astype(np.int64); row = np.floor(GY1 - y).astype(np.int64)
    ok = inside & (col >= 0) & (col < GW) & (row >= 0) & (row < GH)
    cell = row * GW + col
    v = ok & ~np.isin(c, (7, 18))
    u, inv = np.unique(cell[v], return_inverse=True)
    zmax = np.full(len(u), -1e9); np.maximum.at(zmax, inv, z[v])
    f = ok & (rn == 1); uf, nf = np.unique(cell[f], return_counts=True)
    gg = ok & (c == 2); ug, ng = np.unique(cell[gg], return_counts=True)
    np.savez(fo, gx=x[g].astype(np.float32), gy=y[g].astype(np.float32), gz=z[g].astype(np.float32),
             gi=np.asarray(las.intensity)[g].astype(np.uint16), gc=c[g].astype(np.uint8),
             dsm_cell=u.astype(np.int64), dsm_z=zmax.astype(np.float32), f_cell=uf, f_n=nf.astype(np.uint16),
             g_cell=ug, g_n=ng.astype(np.uint16), ccount=cc)
    return int(g.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    jf = CACHE7 / "ept_jobs.json"
    if jf.exists():
        jobs = [tuple(j) for j in json.load(open(jf))]
    else:
        jobs = []
        for ds in EPT_SETS:
            ks = nodes_for(ds); print(ds, len(ks), "nodes", sum(n for _, n in ks), "pts", flush=True)
            jobs += [(ds, k) for k, _ in ks]
        json.dump(jobs, open(jf, "w"))
    todo = [j for j in jobs if not (OUT / f"{j[0]}_{j[1]}.npz").exists()]
    print(len(jobs), "jobs,", len(todo), "to do", flush=True)
    done = 0; t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for _ in ex.map(proc, todo):
            done += 1
            if done % 50 == 0: print(done, "/", len(todo), f"{time.time() - t0:.0f}s", flush=True)
    print("done")
