"""Re-bin corridor points into 250 m station chunks along the OSM line (local coords, float32).
Adds station s and signed lateral offset d (left of travel direction S->N is +)."""
import glob, json, numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from common import *

CH = 250.0
o = origin()
cl = np.loadtxt(DER / "osm_centerline_utm.csv", delimiter=",", skiprows=1)
ls = LineString(cl[:, 1:3])
ss = np.arange(0, ls.length, 1.0)
P = np.array([ls.interpolate(t).coords[0] for t in ss])
T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
tree = cKDTree(P)
nch = int(np.ceil(ls.length / CH))
out = CACHE / "chunks"; out.mkdir(exist_ok=True)
files = sorted(glob.glob(str(CACHE / "pts" / "*.npz")))
groups = {k: [] for k in range(nch)}
# process in batches to bound memory; flush to per-batch part files
def flush(tag):
    for k, lst in groups.items():
        if lst:
            np.savez(out / f"part_{k:03d}_{tag}.npz", **{f: np.concatenate([a[f] for a in lst]) for f in lst[0]})
            lst.clear()
for bi, f in enumerate(files):
    d = np.load(f)
    if len(d["c"]) == 0: continue
    _, j = tree.query(np.c_[d["x"], d["y"]])
    rel = np.c_[d["x"] - P[j, 0], d["y"] - P[j, 1]]
    along = (rel * T[j]).sum(1); lat = T[j, 0] * rel[:, 1] - T[j, 1] * rel[:, 0]
    s = ss[j] + along
    k = np.clip((s // CH).astype(int), 0, nch - 1)
    rec = dict(x=(d["x"] - o["x0"]).astype(np.float32), y=(d["y"] - o["y0"]).astype(np.float32), z=d["z"],
               i=d["i"], c=d["c"], rn=d["rn"], nr=d["nr"], s=s.astype(np.float32), d=lat.astype(np.float32))
    for kk in np.unique(k):
        m = k == kk
        groups[kk].append({f: v[m] for f, v in rec.items()})
    if bi % 400 == 399: flush(bi)
flush("last")
for k in range(nch):
    parts = sorted(out.glob(f"part_{k:03d}_*.npz"))
    if not parts: continue
    ds = [np.load(p) for p in parts]
    np.savez(out / f"chunk_{k:03d}.npz", **{f: np.concatenate([x[f] for x in ds]) for f in ds[0].files})
    for p in parts: p.unlink()
json.dump({"chunk_m": CH, "n": nch}, open(out / "index.json", "w"))
print("chunks", nch)
