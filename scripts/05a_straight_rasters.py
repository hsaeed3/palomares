"""Road-aligned rasters in (s, d) space along the OSM line: s step 0.5 m, d step 0.25 m, |d|<=24 m.
Bands: ground z (mean), ground intensity (median-ish mean), ground count, nonground-low count (0.3-3 m above ground)."""
import json, numpy as np
from common import *
idx = json.load(open(CACHE / "chunks" / "index.json"))
DS, DD, DM = 0.5, 0.25, 24.0
L = idx["n"] * idx["chunk_m"]
ns, nd = int(L / DS) + 1, int(2 * DM / DD)
zs = np.zeros((ns, nd)); n = np.zeros((ns, nd)); it = np.zeros((ns, nd)); zmin = np.full((ns, nd), np.inf)
for k in range(idx["n"]):
    d = np.load(CACHE / "chunks" / f"chunk_{k:03d}.npz")
    m = (d["c"] == 2) | (d["c"] == 17)
    m &= np.abs(d["d"]) < DM
    si = np.clip((d["s"][m] / DS).astype(int), 0, ns - 1); di = ((d["d"][m] + DM) / DD).astype(int).clip(0, nd - 1)
    np.add.at(zs, (si, di), d["z"][m]); np.add.at(n, (si, di), 1); np.add.at(it, (si, di), d["i"][m])
np.savez_compressed(CACHE / "straight_ground.npz", z=(zs / np.maximum(n, 1)).astype(np.float32), n=n.astype(np.uint16),
                    i=(it / np.maximum(n, 1)).astype(np.float32), ds=DS, dd=DD, dm=DM)
print(ns, nd, (n > 0).mean())
