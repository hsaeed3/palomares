#!/usr/bin/env python3
"""21d: physics post-process on build/geometry.pkl (runs after 21c, before 22).  See docs/AC_CONVENTIONS.md
"Physics meshes".

1. Terrain (1GRASS_t*): heights low-pass filtered (Taubin, no shrink) with full weight within NEAR m of the road
   edge, fading to 0 at FAR m; then crease/spike clamp: vertices of adjacent-triangle pairs whose normals differ by
   more than CREASE_DEG (or that stick out of their 1-ring by > SPIKE m) get extra local smoothing.  Vertices shared
   with the road group (shoulder outer edge = the stitch) are pinned, so the stitch stays exact.  Visual = physics
   (same mesh), so wheels never float above / sink below what is drawn.
2+3. Collision (dev6): collision_conv.rebuild() replaces every per-object 1WALL box with the community convention:
   continuous welded invisible slabs per side along barrier lines and the dense-forest line, closed boxes for
   buildings, no collision for trees / small props.  See docs/AC_CONVENTIONS.md "Collision convention (dev6)".
Writes reports/qa_physics.json (terrain crease/spike stats within 25 m of the road edge).
"""
import json, pickle, sys, pathlib
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import ROOT, BUILD
from geom import build_lock; build_lock()  # one build/ writer at a time
import collide as CO

NEAR, FAR = 30.0, 55.0
CREASE_DEG, SPIKE = 25.0, 0.12
# dev5: keep the LiDAR's near-road relief (cut-bank toes, mounds, ditches seen in the GoPro frames). Taubin runs at
# RELIEF_W strength out to RELIEF_R m from the edge (ramping to full at NEAR); the crease clamp only acts within
# CLAMP_R m (where a car actually runs off), and beyond it the spike clamp only removes outliers > SPIKE_FAR m.
RELIEF_R, RELIEF_W = 15.0, 0.4
CLAMP_R, SPIKE_FAR = 3.0, 0.30
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
    w = np.clip((FAR - d_edge) / (FAR - NEAR), 0, 1)
    w = w * (RELIEF_W + (1 - RELIEF_W) * np.clip((d_edge - RELIEF_R) / (NEAR - RELIEF_R), 0, 1)); w[pinned] = 0
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
        bad &= d_edge < CLAMP_R
        bad |= np.abs(L(z)) > np.where(d_edge < CLAMP_R, SPIKE, SPIKE_FAR)
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
        T_ = T_[np.abs(cr) > 5e-3]; cr = cr[np.abs(cr) > 5e-3]        # drop slivers along the collinear road end row (float32 in the kn5 can flip near-zero areas)
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

# ------------------------------------------------------------------ 1c. run-off crease clamp (dev6 FFB safety)
# Shoulder strips (1GRASS_sh*), paved aprons (1ROAD_AP*) and the near-road terrain are smoothed JOINTLY (welded by
# position) wherever two adjacent triangles fold by more than RUNOFF_CREASE deg (apron: APRON_CREASE): dev5 had
# 30-62 deg folds at the shoulder/ditch edges, i.e. a wheel leaving the road hit a sharp ridge.  The main 1ROAD
# surface and every vertex within KERB_KEEP m of it (kerbs are draped there) are pinned.  Heights only.
RUNOFF_CREASE, APRON_CREASE, RUNOFF_R, KERB_KEEP = 22.0, 6.0, 10.0, 0.9


def runoff_clamp():
    main_road = [m for m in G["road"] if m["name"].startswith("1ROAD") and not m["name"].startswith("1ROAD_AP")]
    rxy = np.concatenate([m["P"][:, :2] for m in main_road]); rkd = cKDTree(rxy)
    sel = [m for m in G["road"] if m["name"].startswith(("1GRASS_sh", "1ROAD_AP"))] + \
          [m for m in G["terrain"] if m["name"].startswith("1GRASS_t") or m["name"] == "1GRASS_hole_fill"]
    # restrict terrain meshes to the ones that come near the road
    sel = [m for m in sel if rkd.query(m["P"][:, :2], distance_upper_bound=RUNOFF_R)[0].min() < RUNOFF_R]
    aP = np.concatenate([m["P"] for m in sel]); off = np.cumsum([0] + [len(m["P"]) for m in sel])
    ky, inv = np.unique(np.round(aP[:, :2], 3), axis=0, return_inverse=True); inv = inv.ravel()
    nU = len(ky); z = np.zeros(nU); z[inv] = aP[:, 2]
    Tg = np.concatenate([inv[m["T"].reshape(-1, 3) + o] for m, o in zip(sel, off[:-1])])
    apron = np.concatenate([np.full(len(m["T"].reshape(-1, 3)), m["name"].startswith("1ROAD_AP")) for m in sel])
    dr, _ = rkd.query(ky, workers=4)
    rk = {tuple(r) for r in np.round(rxy, 3)}
    pin = np.array([tuple(r) in rk for r in ky]) | (dr < KERB_KEEP) | (dr > RUNOFF_R)
    e = np.r_[Tg[:, [0, 1]], Tg[:, [1, 2]], Tg[:, [2, 0]]]
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nU, nU)).tocsr(); A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1; Dinv = sp.diags(1 / deg)
    te = np.r_[np.sort(Tg[:, [0, 1]], 1), np.sort(Tg[:, [1, 2]], 1), np.sort(Tg[:, [2, 0]], 1)]
    tid = np.tile(np.arange(len(Tg)), 3); o = np.lexsort((te[:, 1], te[:, 0])); te, tid = te[o], tid[o]
    same = (te[1:] == te[:-1]).all(1); a, b = tid[:-1][same], tid[1:][same]
    lim = np.where(apron[a] | apron[b], APRON_CREASE, RUNOFF_CREASE)
    xy = ky

    def ang_(zz):
        P3 = np.c_[xy, zz]; fn = np.cross(P3[Tg[:, 1]] - P3[Tg[:, 0]], P3[Tg[:, 2]] - P3[Tg[:, 0]])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        return np.degrees(np.arccos(np.clip((fn[a] * fn[b]).sum(1), -1, 1)))
    z0 = z.copy(); ang = ang_(z); before = float(ang.max()); nb = int((ang > lim).sum())
    for it in range(400):
        hit = ang > lim
        bad = np.zeros(nU, bool); bad[Tg[a[hit]].ravel()] = True; bad[Tg[b[hit]].ravel()] = True; bad &= ~pin
        if not bad.any():
            break
        L = Dinv @ (A @ z) - z
        z[bad] += 0.6 * L[bad]
        ang = ang_(z)
    act = ~(pin[Tg[a]].all(1) & pin[Tg[b]].all(1))
    after = float(ang[act].max()) if act.any() else 0.0
    for m, o_ in zip(sel, off[:-1]):
        ii = inv[o_:o_ + len(m["P"])]
        if np.abs(z[ii] - m["P"][:, 2]).max() > 1e-6:
            m["P"] = m["P"].copy(); m["P"][:, 2] = z[ii]
            P = m["P"]; T = m["T"].reshape(-1, 3)
            fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]); N = np.zeros_like(P)
            for k in range(3):
                np.add.at(N, T[:, k], fn)
            m["N"] = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    r = dict(max_crease_before=round(before, 1), n_over_before=nb, max_crease_after_unpinned=round(after, 1),
             n_over_after=int(((ang > lim) & act).sum()), iters=it, max_dz=round(float(np.abs(z - z0).max()), 3),
             mean_dz_moved=round(float(np.abs(z - z0)[np.abs(z - z0) > 1e-6].mean()) if (np.abs(z - z0) > 1e-6).any() else 0.0, 3))
    print(f"run-off crease clamp: {r}")
    return r


runoff_stats = runoff_clamp()

# ------------------------------------------------------------------ 2+3. collision rebuild (dev6, collision_conv.py)
# The per-object boxes of 21b/21c/21e/race_assets and dev4's 4,213 trunk prisms are replaced by continuous welded
# slabs per side (barrier + forest line), simple closed boxes for buildings and nothing for props/trees.
renamed = 0
for m in G["trees"]:
    if m["name"][:1].isdigit() and not m["name"].startswith("1WALL"):
        m["name"] = "v" + m["name"]; renamed += 1
import csv, collision_conv as CC
from geom import load_centerline as _lc
DER = ROOT / "data" / "derived"
tr = list(csv.DictReader(open(DER / "trees.csv")))
trees = tuple(np.array([float(t[k]) for t in tr]) for k in ("x_local", "y_local", "ground_z"))
cstats = CC.rebuild(G, _lc(), trees)
ntree, nw = 0, cstats["new_meshes"]
print(f"trees: no per-trunk colliders (convention); {renamed} physical-named tree meshes renamed")

_prev = geo["meta"].get("physics", {})
geo["meta"]["physics"] = dict(terrain_before=_prev.get("terrain_before") if TERRAIN_DONE else before,
                              terrain_after=_prev.get("terrain_after") if TERRAIN_DONE else after,
                              open_edges_near=_prev.get("open_edges_near") if TERRAIN_DONE else open_near,
                              open_edges_near_filled=n_open, holes_filled=n_fill,
                              tree_colliders=0, collision=cstats, runoff_clamp=runoff_stats)
geo["meta"]["physics_21d_done"] = True
pickle.dump(geo, open(BUILD / "geometry.pkl", "wb"))
(ROOT / "reports").mkdir(exist_ok=True)
json.dump(geo["meta"]["physics"], open(ROOT / "reports" / "qa_physics.json", "w"), indent=1)
