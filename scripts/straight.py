"""Road-aligned ('straightened') rasters of LiDAR ground returns along any centerline."""
import json, numpy as np
from scipy.spatial import cKDTree
from common import *

def frame(xy, step=0.5):
    seg = np.hypot(*np.diff(xy, axis=0).T); s = np.r_[0, np.cumsum(seg)]
    ss = np.arange(0, s[-1], step)
    P = np.c_[np.interp(ss, s, xy[:, 0]), np.interp(ss, s, xy[:, 1])]
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
    return ss, P, T

def straighten(xy_local, DS=0.5, DD=0.25, DM=24.0, classes=(2, 17), extra=False):
    """xy_local: (N,2) centerline in local coords. Returns dict of (ns, nd) arrays."""
    ss, P, T = frame(xy_local, 0.5)
    tree = cKDTree(P)
    idx = json.load(open(CACHE / "chunks" / "index.json"))
    ns, nd = int(ss[-1] / DS) + 1, int(2 * DM / DD)
    acc = {k: np.zeros((ns, nd)) for k in ("n", "z", "zz", "i")}
    if extra:
        acc["veg"] = np.zeros((ns, nd)); acc["vegmax"] = np.zeros((ns, nd))
    for k in range(idx["n"]):
        d = np.load(CACHE / "chunks" / f"chunk_{k:03d}.npz")
        x, y = d["x"].astype(np.float64), d["y"].astype(np.float64)
        dist, j = tree.query(np.c_[x, y], distance_upper_bound=DM * 1.5)
        ok = np.isfinite(dist); j = np.where(ok, j, 0)
        rel = np.c_[x - P[j, 0], y - P[j, 1]]
        along = (rel * T[j]).sum(1); lat = T[j, 0] * rel[:, 1] - T[j, 1] * rel[:, 0]
        s = ss[j] + along
        ok &= (np.abs(lat) < DM) & (s >= 0) & (s < ns * DS)
        si = (s / DS).astype(int).clip(0, ns - 1); di = ((lat + DM) / DD).astype(int).clip(0, nd - 1)
        g = ok & np.isin(d["c"], classes)
        z = d["z"].astype(np.float64)
        np.add.at(acc["n"], (si[g], di[g]), 1); np.add.at(acc["z"], (si[g], di[g]), z[g])
        np.add.at(acc["zz"], (si[g], di[g]), z[g] ** 2); np.add.at(acc["i"], (si[g], di[g]), d["i"][g])
        if extra:
            v = ok & (d["c"] == 1)
            np.add.at(acc["veg"], (si[v], di[v]), 1)
            np.maximum.at(acc["vegmax"], (si[v], di[v]), z[v])
    n = acc["n"]; out = {"n": n.astype(np.uint16), "ds": DS, "dd": DD, "dm": DM}
    with np.errstate(invalid="ignore", divide="ignore"):
        out["z"] = np.where(n > 0, acc["z"] / n, np.nan).astype(np.float32)
        out["i"] = np.where(n > 0, acc["i"] / n, np.nan).astype(np.float32)
        out["zsd"] = np.where(n > 1, np.sqrt(np.maximum(acc["zz"] / n - (acc["z"] / n) ** 2, 0)), np.nan).astype(np.float32)
    if extra:
        out["veg"] = acc["veg"].astype(np.uint16); out["vegmax"] = np.where(acc["veg"] > 0, acc["vegmax"], np.nan).astype(np.float32)
    return out
