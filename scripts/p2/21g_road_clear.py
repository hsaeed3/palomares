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

# ---- crease smoothing on drivable non-road surfaces (kerbs, run-off, shoulders): interior vertices of folded edges
# are relaxed in z towards their neighbours; boundary vertices (shared with road / terrain) never move, so no steps.
LIM = {"1KERB": 22.0, "1RUNOFF": 22.0, "1GRASS_sh": 40.0}
SM = collections.Counter()


def smooth_creases(m, lim):
    P = np.asarray(m["P"], float); T = np.asarray(m["T"])
    if len(T) < 2:
        return 0
    key = np.round(P * 1000).astype(np.int64)
    _, w, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
    Q = P[w].copy(); F = inv[T]
    E = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1); fid = np.r_[np.arange(len(F))] .repeat(1)
    fid = np.r_[np.arange(len(F)), np.arange(len(F)), np.arange(len(F))]
    o = np.lexsort((E[:, 1], E[:, 0])); E, fid = E[o], fid[o]
    same = np.all(E[1:] == E[:-1], 1)
    bnd = np.ones(len(Q), bool); inner = np.zeros(len(Q), bool)
    cnt = np.bincount(np.r_[E[:, 0], E[:, 1]], minlength=len(Q))
    pair_a, pair_b = fid[:-1][same], fid[1:][same]
    ein = E[:-1][same]
    inner[np.r_[ein[:, 0], ein[:, 1]]] = True
    # boundary vertex: touches an edge used once
    once = np.ones(len(E), bool); once[:-1] &= ~same; once[1:] &= ~same
    bnd[:] = False; bnd[np.r_[E[once, 0], E[once, 1]]] = True
    nb = [[] for _ in range(len(Q))]
    for a_, b_ in ein:
        nb[a_].append(b_); nb[b_].append(a_)
    moved = 0
    for it in range(60):
        n = np.cross(Q[F[:, 1]] - Q[F[:, 0]], Q[F[:, 2]] - Q[F[:, 0]]); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        ang = np.degrees(np.arccos(np.clip((n[pair_a] * n[pair_b]).sum(1), -1, 1)))
        hot = ang > lim
        if not hot.any():
            break
        vs = np.unique(np.r_[F[pair_a[hot]].ravel(), F[pair_b[hot]].ravel()])
        vs = vs[~bnd[vs]]
        if not len(vs):
            break
        for v in vs:
            if nb[v]:
                Q[v, 2] = 0.5 * Q[v, 2] + 0.5 * Q[nb[v], 2].mean()
        moved += len(vs)
    m["P"] = Q[inv]
    return moved


for gname, lst in G.items():
    if not isinstance(lst, list):
        continue
    for m in lst:
        if not isinstance(m, dict) or "P" not in m:
            continue
        for pre, lim in LIM.items():
            if m["name"].startswith(pre):
                k_ = smooth_creases(m, lim)
                if k_:
                    SM[m["name"]] += k_
print(f"21g: crease smoothing moved vertices in {len(SM)} meshes: {dict(SM.most_common(8))}")

pickle.dump(geo, open(BUILD / "geometry.pkl", "wb"))
print(f"21g: terrain vertices pushed below asphalt: {TER}; meshes trimmed at {len(LOG)} (group, kind, 100 m) spots")
by = collections.Counter()
for (g, k, s), n in LOG.items():
    by[(g, k)] += 1
for (g, k), n in by.most_common(20):
    ss = sorted(s for (g2, k2, s) in LOG if (g2, k2) == (g, k))
    print(f"  {g}:{k} at {n} spots, s={ss[:8]}{'...' if len(ss) > 8 else ''}")
