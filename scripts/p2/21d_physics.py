#!/usr/bin/env python3
"""21d: physics post-process on build/geometry.pkl (runs after 21c, before 22).  See docs/AC_CONVENTIONS.md
"Physics meshes".

1. Terrain (1GRASS_t*): heights low-pass filtered (Taubin, no shrink) with full weight within NEAR m of the road
   edge, fading to 0 at FAR m; then crease/spike clamp: vertices of adjacent-triangle pairs whose normals differ by
   more than CREASE_DEG (or that stick out of their 1-ring by > SPIKE m) get extra local smoothing.  Vertices shared
   with the road group (shoulder outer edge = the stitch) are pinned, so the stitch stays exact.  Visual = physics
   (same mesh), so wheels never float above / sink below what is drawn.
2. Trees: an 8-sided closed prism collider (1WALL_tree_NN, one mesh per chunk, renderable=False) for every trunk
   within TREE_R m of the road edge, sized from the actual trunk mesh.  Any tree mesh with a physical (digit) name is
   renamed non-physical.
3. Walls: every 1WALL mesh gets its away-from-road faces moved WALL_DEPTH m away (collide.thicken_away), so a
   head-on hit can't push the car out through the back face of a thin box.
Writes reports/qa_physics.json (terrain crease/spike stats within 25 m of the road edge).
"""
import json, pickle, sys, pathlib
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import ROOT, BUILD
import collide as CO

NEAR, FAR = 30.0, 55.0
CREASE_DEG, SPIKE = 25.0, 0.12
QA_R = 25.0
TREE_R = 25.0

geo = pickle.load(open(BUILD / "geometry.pkl", "rb"))
G = geo["groups"]
TERRAIN_DONE = bool(geo["meta"].get("physics_21d_done"))  # terrain smoothing must run once per fresh 21 output
road_ms = [m for m in G["road"] if m["name"].startswith(("1ROAD", "1GRASS"))]
road_xy = np.concatenate([m["P"][:, :2] for m in G["road"] if m["name"].startswith("1ROAD")])
road_kd = cKDTree(road_xy)
edge_xy = np.concatenate([m["P"][:, :2] for m in road_ms])   # road + shoulders: distance to the drivable edge

# ------------------------------------------------------------------ 1. terrain
if TERRAIN_DONE:
    print("21d: terrain already smoothed on this geometry.pkl (rerun 21 for a fresh one) - skipping terrain step")
else:
    TM = [m for m in G["terrain"] if m["name"].startswith("1GRASS_t")]
    allP = np.concatenate([m["P"] for m in TM])
    keys = np.round(allP[:, :2], 3)
    uk, inv = np.unique(keys, axis=0, return_inverse=True); inv = inv.ravel()
    nU = len(uk)
    z = np.zeros(nU); z[inv] = allP[:, 2]
    off = np.cumsum([0] + [len(m["P"]) for m in TM])
    Tg = np.concatenate([inv[m["T"].reshape(-1, 3) + o] for m, o in zip(TM, off[:-1])])
    rk = {tuple(r) for r in np.round(np.concatenate([m["P"][:, :2] for m in road_ms]), 3)}
    pinned = np.array([tuple(r) in rk for r in uk])
    d_edge, _ = cKDTree(edge_xy).query(uk, workers=4)
    w = np.clip((FAR - d_edge) / (FAR - NEAR), 0, 1); w[pinned] = 0
    print(f"terrain: {nU} unique vertices, {len(Tg)} tris, {pinned.sum()} pinned (road stitch)")

    e = np.r_[Tg[:, [0, 1]], Tg[:, [1, 2]], Tg[:, [2, 0]]]
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nU, nU)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    Dinv = sp.diags(1 / deg)
    L = lambda zz: Dinv @ (A @ zz) - zz
    # boundary (holes) check: edges used once, not on the road stitch
    ee = np.sort(e, 1); u_e, cnt = np.unique(ee, axis=0, return_counts=True)
    open_e = u_e[(cnt == 1) & ~(pinned[u_e[:, 0]] & pinned[u_e[:, 1]])]
    open_near = int((d_edge[open_e].min(1) < QA_R).sum())


    def stats(zz):
        P3 = np.c_[uk, zz]
        fn = np.cross(P3[Tg[:, 1]] - P3[Tg[:, 0]], P3[Tg[:, 2]] - P3[Tg[:, 0]])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        near_t = d_edge[Tg].min(1) < QA_R
        # adjacent triangle pairs via shared edges
        te = np.r_[np.sort(Tg[:, [0, 1]], 1), np.sort(Tg[:, [1, 2]], 1), np.sort(Tg[:, [2, 0]], 1)]
        tid = np.tile(np.arange(len(Tg)), 3)
        o = np.lexsort((te[:, 1], te[:, 0])); te, tid = te[o], tid[o]
        same = (te[1:] == te[:-1]).all(1)
        a, b = tid[:-1][same], tid[1:][same]
        ang = np.degrees(np.arccos(np.clip((fn[a] * fn[b]).sum(1), -1, 1)))
        nearp = near_t[a] | near_t[b]
        spike = np.abs(L(zz))
        near_v = d_edge < QA_R
        return dict(pairs_near=int(nearp.sum()), max_crease_deg=float(ang[nearp].max()),
                    p99_crease_deg=float(np.percentile(ang[nearp], 99)),
                    n_crease_over=int((ang[nearp] > CREASE_DEG).sum()),
                    max_spike_m=float(spike[near_v & ~pinned].max()), n_spike_over=int((spike[near_v & ~pinned] > SPIKE).sum()),
                    max_slope_deg=float(np.degrees(np.arccos(np.abs(fn[near_t, 2]).min())))), ang, a, b


    before, *_ = stats(z)
    z0 = z.copy()
    for _ in range(10):                         # Taubin lambda/mu: low-pass without shrinking cut banks / drops
        z = z + 0.5 * w * L(z)
        z = z - 0.53 * w * L(z)
    for it in range(200):                        # crease / spike clamp (local)
        s, ang, a, b = stats(z)
        bad = np.zeros(nU, bool)
        hit = ang > CREASE_DEG
        bad[Tg[a[hit]].ravel()] = True; bad[Tg[b[hit]].ravel()] = True
        bad |= np.abs(L(z)) > SPIKE
        bad &= (d_edge < QA_R + 5) & ~pinned
        if not bad.any():
            break
        z[bad] += 0.8 * L(z)[bad]
    after, *_ = stats(z)
    dz = z - z0
    print("terrain before", before); print("terrain after ", after)
    print(f"terrain height change: mean |dz| {np.abs(dz).mean():.3f} m, max {np.abs(dz).max():.2f} m; open edges near road {open_near}")
    for m, o in zip(TM, off[:-1]):
        ii = inv[o:o + len(m["P"])]
        m["P"] = m["P"].copy(); m["P"][:, 2] = z[ii]
        P = m["P"]; T = m["T"].reshape(-1, 3)
        fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
        N = np.zeros_like(P)
        for k in range(3):
            np.add.at(N, T[:, k], fn)
        m["N"] = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)

# ------------------------------------------------------------------ 1b. fill terrain holes near the road
# Open terrain edges within QA_R of the drivable edge (the gaps between the terrain and the road END rows at
# s=0 / s=end): chains of open edges whose ends sit on the road stitch are closed along the road end row and
# triangulated into an extra terrain mesh 1GRASS_hole_fill (idempotent; same material as the neighbour).
def fill_holes():
    from shapely.geometry import Polygon, Point
    from matplotlib.tri import Triangulation
    G["terrain"] = [m for m in G["terrain"] if m["name"] != "1GRASS_hole_fill"]
    TM_ = [m for m in G["terrain"] if m["name"].startswith("1GRASS_t")]
    aP = np.concatenate([m["P"] for m in TM_]); ky = np.round(aP[:, :2], 3)
    uk_, inv_ = np.unique(ky, axis=0, return_inverse=True); inv_ = inv_.ravel()
    zz = np.zeros(len(uk_)); zz[inv_] = aP[:, 2]
    of = np.cumsum([0] + [len(m["P"]) for m in TM_])
    Tg_ = np.concatenate([inv_[m["T"].reshape(-1, 3) + o] for m, o in zip(TM_, of[:-1])])
    rP = np.concatenate([m["P"] for m in road_ms])
    rk_ = {tuple(r) for r in np.round(rP[:, :2], 3)}
    pin = np.array([tuple(r) in rk_ for r in uk_])
    de, _ = cKDTree(edge_xy).query(uk_, workers=4)
    ee_ = np.sort(np.r_[Tg_[:, [0, 1]], Tg_[:, [1, 2]], Tg_[:, [2, 0]]], 1)
    ue, cn = np.unique(ee_, axis=0, return_counts=True)
    op = ue[(cn == 1) & ~(pin[ue[:, 0]] & pin[ue[:, 1]])]
    op = op[de[op].min(1) < QA_R]
    if not len(op):
        return 0, 0
    adj = {}
    for a, b in op:
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    seen = set(); P_all, T_all = [], []; nfill = 0
    rkd = cKDTree(rP[:, :2])
    for start in [v for v in adj if len(adj[v]) == 1 and pin[v]] + list(adj):
        if start in seen:
            continue
        chain = [start]; seen.add(start); cur = start
        while True:
            nx_ = [w for w in adj[cur] if w not in seen]
            if not nx_:
                break
            cur = nx_[0]; chain.append(cur); seen.add(cur)
        if len(chain) < 3:
            continue
        pts = [np.r_[uk_[v], zz[v]] for v in chain]
        a2, b2 = uk_[chain[-1]], uk_[chain[0]]
        if pin[chain[0]] and pin[chain[-1]]:
            # close along the road end row: road/shoulder vertices on the segment end -> start
            seg = b2 - a2; L2 = float(seg @ seg)
            cand = rkd.query_ball_point((a2 + b2) / 2, np.sqrt(L2) / 2 + 0.05)
            q = rP[cand]; t = ((q[:, :2] - a2) @ seg) / L2
            dist = np.abs((q[:, 0] - a2[0]) * seg[1] - (q[:, 1] - a2[1]) * seg[0]) / np.sqrt(L2)
            q = q[(dist < 0.02) & (t > 1e-3) & (t < 1 - 1e-3)]; t = ((q[:, :2] - a2) @ seg) / L2
            _, uu = np.unique(np.round(q[:, :2], 3), axis=0, return_index=True); q, t = q[uu], t[uu]
            pts += [p_ for _, p_ in sorted(zip(t, list(q)), key=lambda z_: z_[0])]
        pts = np.array(pts)
        poly = Polygon(pts[:, :2])
        if not poly.is_valid or poly.area > 400:
            print(f"  hole near {pts[0, :2].round(1)}: polygon invalid/too big (area {poly.area:.0f} m2) - not filled"); continue
        tri = Triangulation(pts[:, 0], pts[:, 1])
        keep = [t_ for t_ in tri.triangles if poly.buffer(1e-4).contains(Point(pts[t_, :2].mean(0)))]
        T_ = np.array(keep)
        if not len(T_):
            continue
        cr = np.cross(pts[T_[:, 1]] - pts[T_[:, 0]], pts[T_[:, 2]] - pts[T_[:, 0]])[:, 2]
        T_[cr < 0] = T_[cr < 0][:, [0, 2, 1]]
        T_all.append(T_ + sum(len(p_) for p_ in P_all)); P_all.append(pts); nfill += 1
        print(f"  filled terrain hole near ({pts[0, 0]:.0f}, {pts[0, 1]:.0f}): {len(chain)} open-edge vertices, {len(T_)} tris, {poly.area:.1f} m2")
    if P_all:
        P = np.concatenate(P_all); T = np.concatenate(T_all)
        m0 = TM_[0]
        fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]); N = np.zeros_like(P)
        for k in range(3):
            np.add.at(N, T[:, k], fn)
        G["terrain"].append(dict(name="1GRASS_hole_fill", material=m0["material"], P=P, T=T, UV=P[:, :2] / 4.0,
                                 N=N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12),
                                 **{k: v for k, v in m0.items() if k not in ("name", "material", "P", "T", "UV", "N")}))
    return len(op), nfill


n_open, n_fill = fill_holes()
print(f"terrain holes: {n_open} open edges within {QA_R} m of the road, {n_fill} holes filled (1GRASS_hole_fill)")

# ------------------------------------------------------------------ 2. trees
renamed = 0
for m in G["trees"]:
    if m["name"][:1].isdigit() and not m["name"].startswith("1WALL_tree"):
        m["name"] = "v" + m["name"]; renamed += 1
G["trees"] = [m for m in G["trees"] if not m["name"].startswith("1WALL_tree")]   # idempotent
import csv
DER = ROOT / "data" / "derived"
tr = list(csv.DictReader(open(DER / "trees.csv")))
tx = np.array([float(t["x_local"]) for t in tr]); ty = np.array([float(t["y_local"]) for t in tr])
tz = np.array([float(t["ground_z"]) for t in tr])
bark = np.concatenate([m["P"] for m in G["trees"] if m["material"] == "tree_trunk"])
bkd = cKDTree(bark[:, :2])
d_tree, _ = cKDTree(edge_xy).query(np.c_[tx, ty], workers=4)
nchunks = int(geo["meta"]["nchunks"])
cl_s = None
colls = {}
ntree = 0
for k in np.nonzero(d_tree < TREE_R)[0]:
    idx = bkd.query_ball_point([tx[k], ty[k]], 2.0)
    if not idx:
        continue
    q = bark[idx]; q = q[q[:, 2] < q[:, 2].min() + 1.5]          # trunk base ring(s)
    c = q[:, :2].mean(0); r = float(np.clip(np.linalg.norm(q[:, :2] - c, axis=1).max(), 0.15, 1.2))
    zb = float(q[:, 2].min())
    _, j = road_kd.query(c)
    ch = min(int(j * nchunks // len(road_xy)), nchunks - 1)
    colls.setdefault(ch, CO.Collider(f"1WALL_tree_{ch:02d}")).add(*CO.prism(c[0], c[1], r, zb - 1.0, zb + 4.5))
    ntree += 1
G["trees"] = [m for m in G["trees"] if not m["name"].startswith("1WALL_tree")]  # idempotent
for ch, col in sorted(colls.items()):
    G["trees"] += col.meshes()
print(f"trees: {ntree} trunk colliders within {TREE_R} m in {len(colls)} meshes; {renamed} physical-named tree meshes renamed")

# ------------------------------------------------------------------ 3. walls
# dev4: 'away from the road' is measured from the CENTRELINE (1 m stations): with wide / paved shoulders the
# 1ROAD vertices sit on the fence line itself, which flipped the away direction and pushed whole boxes 0.3 m out
# (qa_collisions: chainlink s=3500, guardrail 3490, fences 13625-14385 uncovered)
from geom import load_centerline as _lc
_c = _lc(); CL_XY = np.c_[_c["x_local"], _c["y_local"]]
nw = nv = 0
for gn, ms in G.items():
    if gn == "trees":
        continue
    for m in ms:
        if m["name"].startswith("1WALL") and not m.get("_thick"):
            nv += CO.thicken_away(m, CL_XY); m.update(CO.WALL_FLAGS); m["_thick"] = True; nw += 1
print(f"walls: {nw} 1WALL meshes thickened to {CO.WALL_DEPTH} m away from the road ({nv} vertices moved)")

_prev = geo["meta"].get("physics", {})
geo["meta"]["physics"] = dict(terrain_before=_prev.get("terrain_before") if TERRAIN_DONE else before,
                              terrain_after=_prev.get("terrain_after") if TERRAIN_DONE else after,
                              open_edges_near=_prev.get("open_edges_near") if TERRAIN_DONE else open_near,
                              open_edges_near_filled=n_open, holes_filled=n_fill,
                              tree_colliders=ntree, walls_thickened=nw)
geo["meta"]["physics_21d_done"] = True
pickle.dump(geo, open(BUILD / "geometry.pkl", "wb"))
(ROOT / "reports").mkdir(exist_ok=True)
json.dump(geo["meta"]["physics"], open(ROOT / "reports" / "qa_physics.json", "w"), indent=1)
