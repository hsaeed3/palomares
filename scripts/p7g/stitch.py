"""dev7 G: stitch a network road to FIXED existing kn5 geometry (the shipped Palomares road/terrain).

The shipped kn5 files are never modified.  For the dev7 layouts a cut COPY is written (`<name>_dev7.kn5`): every
triangle with a vertex inside the cut zone (road outline + margin) is removed.  The new road's terrain CDT then
uses the kept meshes' open boundary edges as constrained segments with their exact float32 vertices, so new and old
surfaces share vertices: no crack, no step.  Triangles of the new CDT that fall inside kept old geometry are dropped.
"""
import collections, pathlib, shutil, hashlib
import numpy as np
from kn5 import read_kn5, write_kn5, Material, Mesh


def enu(p):
    return np.c_[p[:, 0], -p[:, 2], p[:, 1]].astype(float)


class TriGrid:
    """2D bin hash of triangles for point location (z of the top hit, containment)."""
    def __init__(self, P, T, cell=2.0):
        self.P, self.T, self.cell = P, T, cell
        lo = np.floor(P[T][:, :, :2].min(1) / cell).astype(int); hi = np.floor(P[T][:, :, :2].max(1) / cell).astype(int)
        self.bins = collections.defaultdict(list)
        for t in range(len(T)):
            for cx in range(lo[t, 0], hi[t, 0] + 1):
                for cy in range(lo[t, 1], hi[t, 1] + 1):
                    self.bins[(cx, cy)].append(t)

    def locate(self, xy):
        """-> (z of the top surface or nan, triangle index or -1) per point"""
        xy = np.asarray(xy, float); z = np.full(len(xy), np.nan); tri = np.full(len(xy), -1)
        key = np.floor(xy / self.cell).astype(int)
        groups = collections.defaultdict(list)
        for i, k in enumerate(map(tuple, key)):
            groups[k].append(i)
        P, T = self.P, self.T
        for k, idx in groups.items():
            tl = self.bins.get(k)
            if not tl:
                continue
            tl = np.array(tl); idx = np.array(idx)
            a, b, c = P[T[tl, 0]], P[T[tl, 1]], P[T[tl, 2]]
            q = xy[idx][:, None, :]
            v0 = (b - a)[None, :, :2]; v1 = (c - a)[None, :, :2]; v2 = q - a[None, :, :2]
            den = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]; den = np.where(np.abs(den) < 1e-12, 1e-12, den)
            u = (v2[..., 0] * v1[..., 1] - v1[..., 0] * v2[..., 1]) / den
            v = (v0[..., 0] * v2[..., 1] - v2[..., 0] * v0[..., 1]) / den
            ins = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
            zz = np.where(ins, a[None, :, 2] + u * (b - a)[None, :, 2] + v * (c - a)[None, :, 2], -np.inf)
            best = zz.argmax(1); hit = ins.any(1)
            z[idx[hit]] = zz[hit, best[hit]]; tri[idx[hit]] = tl[best[hit]]
        return z, tri


def snapshot(src, dst_dir):
    """copy a shipped kn5 into the dev7 build (full Palomares builds rewrite build/ under us); keyed by content"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    d = dst_dir / src.name
    if src.exists() and (not d.exists() or d.stat().st_mtime < src.stat().st_mtime):
        shutil.copy(src, d)
    return d


def cut_copy(src_path, out_path, inside, log):
    """write src kn5 minus triangles with any vertex inside the zone; returns (kept physics P,T), (removed asphalt
    (1ROAD) P,T), (all removed physics P,T) in ENU"""
    k = read_kn5(str(src_path))
    mats = []
    for m in k["materials"]:
        props = {p: ((v[3], v[4], v[5]) if p == "ksEmissive" else v[0]) for p, v in m["props"].items()}
        mats.append(Material(m["name"], m["shader"], props, dict(m["textures"]), blend=m["blend"], alpha_tested=m["alpha_tested"]))
    nodes, keptP, keptT, roadP, roadT, nk, nr = [], [], [], [], [], 0, 0
    allP, allT, na = [], [], 0
    removed = 0
    for m in k["meshes"]:
        P = enu(m["pos"]); T = m["idx"].reshape(-1, 3)
        bad = inside(P[:, :2])[T].any(1)
        removed += int(bad.sum())
        nm = m["name"]; phys = nm[:1].isdigit() and "WALL" not in nm
        if phys and bad.any():
            allP.append(P); allT.append(T[bad] + na); na += len(P)
        if phys and bad.any() and nm.startswith("1ROAD"):
            roadP.append(P); roadT.append(T[bad] + nr); nr += len(P)
        Tk = T[~bad]
        if not len(Tk):
            continue
        if phys:
            keptP.append(P); keptT.append(Tk + nk); nk += len(P)
        nodes.append(Mesh(nm, m["pos"], m["nrm"], m["uv"], Tk.reshape(-1), mats[m["material"]].name,
                          cast_shadows=m["cast_shadows"], visible=m["visible"], transparent=m["transparent"],
                          renderable=m["renderable"], lod_in=m["lod_in"], lod_out=m["lod_out"], layer=m["layer"]))
    write_kn5(str(out_path), mats, k["textures"], nodes)
    log[out_path.name] = dict(src=str(src_path), removed_tris=removed)
    cat = lambda L, D: np.concatenate(L) if L else np.zeros((0, D))
    return (cat(keptP, 3), cat(keptT, 3).astype(int)), (cat(roadP, 3), cat(roadT, 3).astype(int)), (cat(allP, 3), cat(allT, 3).astype(int))


def open_boundary(P, T):
    """welded (exact float32 position) open edges -> unique vertices V (n,3) and edges E (m,2) into V"""
    key = np.round(P.astype(np.float32).astype(float), 4)
    U, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    W = inv[T]
    E = np.sort(np.r_[W[:, [0, 1]], W[:, [1, 2]], W[:, [2, 0]]], 1)
    Eu, cnt = np.unique(E, axis=0, return_counts=True)
    ob = Eu[cnt == 1]
    Vp = np.zeros((len(U), 3)); Vp[inv] = P
    used, e2 = np.unique(ob, return_inverse=True)
    return Vp[used], e2.reshape(-1, 2)
