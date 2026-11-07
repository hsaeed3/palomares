"""Final safety net before kn5 writing: no visible or collision geometry may stand on the drivable asphalt.

Runs on build/geometry.pkl after every geometry step (21..21f, 21d). For every mesh that is not the road surface,
its paint, kerbs or terrain, triangles with a vertex inside the asphalt (|offset| < half-width) between 2 cm and
4.5 m above the road surface are dropped, and terrain vertices poking above the asphalt are pushed 3 cm below it.
Whatever the upstream data says, the asphalt stays clear. Prints what it removed, per 100 m, so the source data
can be fixed; qa_on_road.py verifies the written kn5 afterwards.
"""
import collections, pickle
import numpy as np
from scipy.spatial import cKDTree
from geom import Road, BUILD, CFG
from geom import build_lock; build_lock()

road = Road(CFG["build_end_m"]); N = len(road.s); KD = cKDTree(np.c_[road.x, road.y])
KEEP_GROUPS = ("road", "lines_road", "kerbs_nb", "kerbs_sb")          # drivable surface + paint + kerbs
geo = pickle.load(open(BUILD / "geometry.pkl", "rb")); G = geo["groups"]
LOG = collections.Counter(); TER = 0


def classify(P):
    d, j = KD.query(P[:, :2]); j = np.clip(j, 0, N - 1)
    off = (P[:, 0] - road.x[j]) * road.nx[j] + (P[:, 1] - road.y[j]) * road.ny[j]
    half = np.where(off > 0, road.el[j], road.er[j])
    zr = road.surface_z(j, off, noise=False)
    inside = (np.abs(off) < half - 0.02) & (d < half + 1.0) & (j > 2) & (j < N - 3)
    return inside, zr, j


for gname, lst in G.items():
    if gname in KEEP_GROUPS or not isinstance(lst, list):
        continue
    for m in lst:
        if not isinstance(m, dict) or "P" not in m or not len(m.get("T", [])):
            continue
        nm = m["name"]
        if nm.startswith(("1ROAD", "1KERB", "line_", "decal", "edge_")) or "AC_" in nm:
            continue
        P = np.asarray(m["P"], float); inside, zr, j = classify(P)
        if not inside.any():
            continue
        if gname == "terrain" or nm.startswith("1GRASS"):
            up = inside & (P[:, 2] > zr - 0.03)
            if up.any():
                P = P.copy(); P[up, 2] = zr[up] - 0.03; m["P"] = P; TER += int(up.sum())
            continue
        bad = inside & (P[:, 2] > zr + 0.02) & (P[:, 2] < zr + 4.5)
        if not bad.any():
            continue
        T = np.asarray(m["T"]); kill = bad[T].any(1)
        if nm.startswith("1WALL"):
            # collision: keep the shell closed -> slide the intruding vertices out to the asphalt edge (+5 cm)
            d_, jj = KD.query(P[:, :2]); jj = np.clip(jj, 0, N - 1)
            off = (P[:, 0] - road.x[jj]) * road.nx[jj] + (P[:, 1] - road.y[jj]) * road.ny[jj]
            half = np.where(off > 0, road.el[jj], road.er[jj]); sg = np.where(off >= 0, 1.0, -1.0)
            mv = bad; dd = (sg * (half + 0.05) - off)[mv]
            P = P.copy(); P[mv, 0] += dd * road.nx[jj[mv]]; P[mv, 1] += dd * road.ny[jj[mv]]; m["P"] = P
            for s in np.unique(road.s[j[bad]] // 100 * 100):
                LOG[(gname, nm.rstrip("0123456789_"), int(s))] += 1
            continue
        if kill.any():
            m["T"] = T[~kill]
            for s in np.unique(road.s[j[bad]] // 100 * 100):
                LOG[(gname, nm.rstrip("0123456789_"), int(s))] += 1

# ---- kerbs / run-off: drop any single kerb piece (connected component) that folds more than KERB_LIM.
# Folds sit where a kerb drapes over an uneven shoulder edge; a missing kerb is safe, a folded one is not.
KERB_LIM = 24.0
DROP = collections.Counter()


def drop_folded(m):
    P = np.asarray(m["P"], float); T = np.asarray(m["T"])
    if len(T) < 2:
        return
    key = np.round(P * 1000).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True); F = inv.ravel()[T]; nv = F.max() + 1
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    r_ = np.r_[F[:, 0], F[:, 1], F[:, 2]]; c_ = np.r_[F[:, 1], F[:, 2], F[:, 0]]
    _, lab = connected_components(coo_matrix((np.ones(len(r_)), (r_, c_)), shape=(nv, nv)), directed=False)
    E = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1); fid = np.tile(np.arange(len(F)), 3)
    o = np.lexsort((E[:, 1], E[:, 0])); E, fid = E[o], fid[o]; same = np.all(E[1:] == E[:-1], 1)
    Pn = np.zeros((nv, 3)); Pn[F.ravel()] = P[T.ravel()]
    n = np.cross(Pn[F[:, 1]] - Pn[F[:, 0]], Pn[F[:, 2]] - Pn[F[:, 0]]); ln_ = np.linalg.norm(n, axis=1)
    n /= np.maximum(ln_[:, None], 1e-12)
    a_, b_ = fid[:-1][same], fid[1:][same]
    ok = (ln_[a_] > 1e-6) & (ln_[b_] > 1e-6)
    ang = np.degrees(np.arccos(np.clip((n[a_] * n[b_]).sum(1), -1, 1)))
    hot = ok & (ang > KERB_LIM)
    if not hot.any():
        return
    badc = np.unique(lab[F[a_[hot], 0]]); kill = np.isin(lab[F[:, 0]], badc)
    m["T"] = T[~kill]; DROP[m["name"]] += int(len(badc))


for gname in ("kerbs_nb", "kerbs_sb"):
    for m in G.get(gname, []):
        if isinstance(m, dict) and m.get("name", "").startswith(("1KERB", "1RUNOFF")):
            drop_folded(m)
print(f"21g: folded kerb pieces dropped: {dict(DROP)}")

pickle.dump(geo, open(BUILD / "geometry.pkl", "wb"))
print(f"21g: terrain vertices pushed below asphalt: {TER}; meshes trimmed at {len(LOG)} (group, kind, 100 m) spots")
by = collections.Counter()
for (g, k, s), n in LOG.items():
    by[(g, k)] += 1
for (g, k), n in by.most_common(20):
    ss = sorted(s for (g2, k2, s) in LOG if (g2, k2) == (g, k))
    print(f"  {g}:{k} at {n} spots, s={ss[:8]}{'...' if len(ss) > 8 else ''}")
