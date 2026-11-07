"""v0.2.0-dev6 collision rebuild to the community/Kunos convention (see docs/AC_CONVENTIONS.md "Collision convention").

Called by 21d_physics.py after 21c / 21e / 21b / race_assets have produced their per-object 1WALL boxes.  Those boxes
(4,213 tree prisms, 0.16 m sign-pole boxes, per-panel fence boxes, overlapping shells pushed 1.2 m back by the old
thicken_away) are NOT shipped any more.  They are used only as a *description* of where solid things stand:

  * every 1WALL mesh is split into connected components (welded by position) and each component is classified in the
    road frame (station s, signed offset d from the refined centreline):
      linear   : long along the road (>= LIN_MIN_LEN m) and shallow across it  -> barrier line (fence, guardrail, wall,
                 cut bank, crowd barrier, catch fence)
      building : footprint >= BLD_MIN x BLD_MIN m                             -> ONE closed 8-vertex box
      across   : crosses the centreline (1WALL_start / 1WALL_end)            -> ONE closed box
      prop     : anything else (poles, signs, mailboxes, cones, bales, trunks) -> NO collision (Kunos/community default)
  * barrier lines are merged PER SIDE into a single offset function d_side(s) on a 2 m station grid (nearest-to-road
    face wins, so colliders never overlap or nest), the forest line (dense trees, data/derived/trees.csv) is folded in
    the same way, then split into runs where the offset jumps, smoothed, and swept as a welded closed slab:
    road-side face exactly at the barrier face, WALL_DEPTH m deep, ground-following bottom/top, flared ends (the
    front face turns away from the road over the last FLARE m, like a guardrail terminal) so nothing presents a
    square corner to a car sliding along it.
  * output: one invisible mesh per side per 250 m chunk: 1WALL_L_NN / 1WALL_R_NN, plus 1WALL_bld_NN, 1WALL_start,
    1WALL_end.  visible=True, renderable=False, cast_shadows=False, material "invisible".
"""
import collections
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

FLAGS = dict(visible=True, renderable=False, cast_shadows=False)
MIN_DEPTH = 0.5      # thinner than this is no wall (gate: qa_ffb_safety MIN_THICK)
STEP = 2.0            # m between slab stations
WALL_DEPTH = 1.0      # slab depth behind the road-side face (thick enough that no step can tunnel through it)
WALL_TOP = 1.6        # m above the local face bottom ground, at least
LIN_MIN_LEN = 3.0
LIN_MAX_DEPTH = 3.5
BLD_MIN = 1.8
JUMP = 0.6            # m offset jump between neighbouring 2 m stations that starts a new run (<= ~17 deg face turn)
GAP_BRIDGE = 2        # bridge gaps of <= this many stations (4 m: guardrail post gaps, panel joints)
FLARE, FLARE_D = 4.0, 0.8
CHUNK = 250.0
# forest line
FOREST_MAX = 14.0     # only trees within this of the drivable edge define a forest line
FOREST_MIN_GAP = 2.0  # never closer than this to the drivable edge (keeps the real verge / ditch usable)
FOREST_WIN, FOREST_N = 10.0, 4   # >= FOREST_N trunks within +-FOREST_WIN m along the road = "dense"


def components(P, T):
    key = np.round(P, 3)
    _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    Tw = inv[T]
    n = inv.max() + 1
    e = np.r_[Tw[:, [0, 1]], Tw[:, [1, 2]]]
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    nc, lab = connected_components(A, directed=False)
    vl = lab[inv]
    return [np.nonzero(vl == c)[0] for c in range(nc)]


class Frame:
    """road frame from the refined centreline (1 m stations)"""
    def __init__(self, cl):
        self.s = cl["s_m"]; self.x = cl["x_local"]; self.y = cl["y_local"]; self.z = cl["z_navd88"]
        tx, ty = np.gradient(self.x), np.gradient(self.y); n = np.hypot(tx, ty)
        self.tx, self.ty = tx / n, ty / n; self.nx, self.ny = -self.ty, self.tx
        self.el = np.clip(cl["edge_left_m"], 2.4, 7.0) + 0.8; self.er = np.clip(cl["edge_right_m"], 2.4, 7.0) + 0.8
        self.kd = cKDTree(np.c_[self.x, self.y])

    def sd(self, xy):
        _, j = self.kd.query(xy, workers=4)
        dx, dy = xy[:, 0] - self.x[j], xy[:, 1] - self.y[j]
        return self.s[j] + dx * self.tx[j] + dy * self.ty[j], dx * self.nx[j] + dy * self.ny[j], j

    def xy(self, s, d):
        i = np.clip(np.interp(s, self.s, np.arange(len(self.s))), 0, len(self.s) - 1)
        x = np.interp(i, np.arange(len(self.s)), self.x); y = np.interp(i, np.arange(len(self.s)), self.y)
        h = np.unwrap(np.arctan2(self.ty, self.tx)); hh = np.interp(i, np.arange(len(self.s)), h)
        return x - np.sin(hh) * d, y + np.cos(hh) * d


def slab(F, s, d, zb, zt, side, depth=None):
    """welded closed slab: road-side face at signed offset side*d(s); returns P, T, N (own vertices per face strip,
    positions shared exactly between strips => watertight)."""
    n = len(s)
    depth = np.full(len(s), WALL_DEPTH) if depth is None else depth
    fx, fy = F.xy(s, side * d); bx, by = F.xy(s, side * (d + depth))
    FB = np.c_[fx, fy, zb]; FT = np.c_[fx, fy, zt]; BT = np.c_[bx, by, zt]; BB = np.c_[bx, by, zb]
    P, T, N = [], [], []

    def strip(A, B, want):
        """quads between rows A (bottom/first) and B along the run, wound so the normal faces `want` (callable i->vec)"""
        k = sum(len(p) for p in P)
        P.append(np.r_[A, B]); i = np.arange(n - 1)
        t1 = np.c_[k + i, k + i + 1, k + n + i + 1]; t2 = np.c_[k + i, k + n + i + 1, k + n + i]
        tt = np.r_[t1, t2]
        Q = np.concatenate(P)
        fn = np.cross(Q[tt[:, 1]] - Q[tt[:, 0]], Q[tt[:, 2]] - Q[tt[:, 0]])
        w = want(np.r_[i, i])
        flip = (fn * w).sum(1) < 0
        tt[flip] = tt[flip][:, [0, 2, 1]]
        T.append(tt)
        nn = np.zeros((2 * n, 3)); ww = want(np.r_[np.arange(n), np.arange(n)])
        N.append(ww / (np.linalg.norm(ww, axis=1, keepdims=True) + 1e-12))

    tow = np.c_[fx - bx, fy - by, np.zeros(n)]; tow /= np.linalg.norm(tow, axis=1, keepdims=True) + 1e-12
    up = np.tile([0, 0, 1.0], (n, 1))
    strip(FB, FT, lambda i: tow[i])          # road-side face, normal toward the road
    strip(FT, BT, lambda i: up[i])           # top
    strip(BT, BB, lambda i: -tow[i])         # back
    strip(BB, FB, lambda i: -up[i])          # bottom
    # end caps (two tris each)
    for q, sg in ((0, -1.0), (n - 1, 1.0)):
        k = sum(len(p) for p in P)
        quad = np.array([FB[q], FT[q], BT[q], BB[q]]); P.append(quad)
        tt = np.array([[k, k + 1, k + 2], [k, k + 2, k + 3]])
        ax = np.array([F.xy(s[q] + 0.01 * sg, side * d[q])]).ravel() - np.array([fx[q], fy[q]])
        w = np.r_[ax / (np.linalg.norm(ax) + 1e-12), 0.0]
        fn = np.cross(quad[1] - quad[0], quad[2] - quad[0])
        if fn @ w < 0:
            tt = tt[:, [0, 2, 1]]
        T.append(tt); N.append(np.tile(w, (4, 1)))
    return np.concatenate(P), np.concatenate(T), np.concatenate(N)


def obb_ext(Pc):
    xy = Pc[:, :2] - Pc[:, :2].mean(0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    return float(np.ptp(xy @ vt[0])), float(np.ptp(xy @ vt[1]))


def obb_box(Pc):
    """closed 8-vertex oriented box around points (xy PCA), outward winding"""
    xy = Pc[:, :2]; c = xy.mean(0)
    u, s_, vt = np.linalg.svd(xy - c, full_matrices=False); ax = vt[0]; ay = np.array([-ax[1], ax[0]])
    a = (xy - c) @ ax; b = (xy - c) @ ay
    z0, z1 = Pc[:, 2].min(), Pc[:, 2].max()
    cs = [(a.min(), b.min()), (a.max(), b.min()), (a.max(), b.max()), (a.min(), b.max())]
    ring = np.array([c + p * ax + q * ay for p, q in cs])
    P = np.r_[np.c_[ring, np.full(4, z0)], np.c_[ring, np.full(4, z1)]]
    T = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    cen = P.mean(0)
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    fl = (fn * (P[T].mean(1) - cen)).sum(1) < 0
    T[fl] = T[fl][:, [0, 2, 1]]
    N = (P - cen); N /= np.linalg.norm(N, axis=1, keepdims=True)
    return P, T, N


def rebuild(G, cl, trees, log=print):
    """shared colliders (all groups but 'dressing') -> G['roadside']; track-layout-only dressing colliders ->
    G['dressing'] (only where the shared slab is not already in front, so the two never overlap)"""
    F = Frame(cl)
    shared = [g for g in G if g != "dressing"]
    out1, st1, near1 = _core(G, F, shared, trees, "1WALL", None)
    G.setdefault("roadside", []).extend(out1)
    out2, st2, _ = _core(G, F, [g for g in G if g == "dressing"], None, "1WALL_dressing", out1)
    G.setdefault("dressing", []).extend(out2)
    log(f"collision rebuild (shared): {st1}")
    log(f"collision rebuild (track dressing): {st2}")
    return dict(shared=st1, dressing=st2, new_meshes=st1["new_meshes"] + st2["new_meshes"])


def _core(G, F, groups, trees, prefix, block):
    s_max = float(F.s[-1]); ns = int(s_max // STEP) + 1
    grid = np.arange(ns) * STEP
    near = {+1: np.full(ns, np.inf), -1: np.full(ns, np.inf)}
    zlo = {+1: np.full(ns, np.nan), -1: np.full(ns, np.nan)}; zhi = {+1: np.full(ns, np.nan), -1: np.full(ns, np.nan)}
    src = {+1: np.zeros(ns, "U6"), -1: np.zeros(ns, "U6")}
    blds, across, stats = [], [], dict(linear=0, building=0, prop=0, across=0, old_meshes=0, old_tris=0)
    for gn in groups:
        ms = G[gn]
        keep = []
        for m in ms:
            if not m["name"].startswith("1WALL"):
                keep.append(m); continue
            stats["old_meshes"] += 1; P = np.asarray(m["P"], float); T = np.asarray(m["T"]).reshape(-1, 3)
            stats["old_tris"] += len(T)
            if m["name"].startswith("1WALL_tree"):
                stats["prop"] += 1; continue
            for comp in components(P, T):
                Pc = P[comp]
                s, d, _ = F.sd(Pc[:, :2])
                L = s.max() - s.min(); D = np.abs(d).max() - np.abs(d).min()
                ext = np.ptp(Pc[:, :2], axis=0)
                if m["name"].startswith(("1WALL_start", "1WALL_end")):
                    across.append((m["name"], Pc)); stats["across"] += 1
                elif d.min() < -1 and d.max() > 1 and np.abs(d).min() < 3:
                    stats["prop"] += 1          # overhead gantry / banner over the road: visual only
                elif L >= LIN_MIN_LEN and D <= LIN_MAX_DEPTH:
                    side = 1 if np.median(d) > 0 else -1
                    b = np.clip((s / STEP).round().astype(int), 0, ns - 1)
                    ad = np.abs(d)
                    for bi in np.unique(b):
                        sel = b == bi; dn = ad[sel].min()
                        if dn < near[side][bi]:
                            near[side][bi] = dn; src[side][bi] = "bar"
                        zlo[side][bi] = np.fmin(zlo[side][bi], Pc[sel, 2].min()); zhi[side][bi] = np.fmax(zhi[side][bi], Pc[sel, 2].max())
                    stats["linear"] += 1
                elif min(obb_ext(Pc)) >= BLD_MIN * 0.6 and max(obb_ext(Pc)) >= BLD_MIN:   # oriented extents (a thin
                    # diagonal panel has a large axis-aligned box but is still a prop)
                    blds.append(Pc); stats["building"] += 1
                else:
                    stats["prop"] += 1
        G[gn] = keep

    # buildings: one closed box each, skipped when it would sit on the drivable surface
    PB, TB, NB, k = [], [], [], 0
    # parts of one building (house + garage + porch boxes) overlap: merge overlapping footprints into ONE box so
    # colliders never nest or intersect
    for _pass in range(60):          # merge until stable (6 passes left 2 overlapping clusters in dev7)
      if blds:
          bb = np.array([[q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()] for q in blds])
          par = list(range(len(blds)))
          def fnd(i):
              while par[i] != i:
                  par[i] = par[par[i]]; i = par[i]
              return i
          kd = cKDTree((bb[:, :2] + bb[:, 2:]) / 2); rad = np.hypot(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1]).max()
          for i in range(len(blds)):
              for j in kd.query_ball_point((bb[i, :2] + bb[i, 2:]) / 2, rad + 0.2):
                  if j > i and bb[i, 0] < bb[j, 2] + 0.2 and bb[j, 0] < bb[i, 2] + 0.2 and bb[i, 1] < bb[j, 3] + 0.2 and bb[j, 1] < bb[i, 3] + 0.2:
                      par[fnd(i)] = fnd(j)
          cl_ = collections.defaultdict(list)
          for i in range(len(blds)):
              cl_[fnd(i)].append(blds[i])
          stats["building_clusters"] = len(cl_)
          n0 = len(blds); blds = [obb_box(np.concatenate(v))[0] for v in cl_.values()]
          if len(blds) == n0:
              break
    for Pc in blds:
        P, T, N = obb_box(Pc)
        s, d, j = F.sd(P[:, :2])
        edge = np.where(d > 0, F.el[j], F.er[j])
        if (np.abs(d) < edge).any():
            stats["building"] -= 1; stats["prop"] += 1; continue
        PB.append(P); TB.append(T + k); NB.append(N); k += 8
    bld_out = []
    for i0 in range(0, len(PB), 4000):
        P = np.concatenate(PB[i0:i0 + 4000]); T = np.concatenate([t - 8 * i0 for t in TB[i0:i0 + 4000]])
        bld_out.append(dict(name=f"{prefix}_bld_{i0 // 4000:02d}", material="invisible", P=P, T=T, UV=np.zeros((len(P), 2)),
                        N=np.concatenate(NB[i0:i0 + 4000]), **FLAGS))
    # occupancy (per side, per 2 m station: nearest / farthest |d|) of everything already placed: the shared
    # colliders (block) and this pass's buildings.  Slabs are cut back or dropped where they would overlap them.
    occ_lo = {+1: np.full(ns, np.inf), -1: np.full(ns, np.inf)}; occ_hi = {+1: np.full(ns, -np.inf), -1: np.full(ns, -np.inf)}
    def occupy(meshes):
        for m in meshes:
            for sP in sample_faces(m["P"], m["T"]):
                s_, d_, _ = F.sd(sP[:, :2]); bi = np.clip(np.round(s_ / STEP).astype(int), 0, ns - 1); sd_ = np.where(d_ >= 0, 1, -1)
                for sg in (1, -1):
                    w = sd_ == sg
                    np.minimum.at(occ_lo[sg], bi[w], np.abs(d_[w])); np.maximum.at(occ_hi[sg], bi[w], np.abs(d_[w]))
    if block is not None:
        occupy(block)
        # this pass's building boxes must not intersect the shared colliders either
        keepb = []
        for m in bld_out:
            P_, T_ = m["P"], m["T"]
            Pn, Tn, Nn = [], [], []
            for q in range(len(P_) // 8):
                bp = P_[8 * q:8 * q + 8]; s_, d_, _ = F.sd(sample_faces(bp, T_[:12] - T_[:12].min())[0][:, :2])
                bi = np.clip(np.round(s_ / STEP).astype(int), 0, ns - 1); sg = np.where(d_ >= 0, 1, -1); ad = np.abs(d_)
                hit = any(((occ_hi[g][bi[sg == g]] > ad[sg == g].min() - 0.05) & (occ_lo[g][bi[sg == g]] < ad[sg == g].max() + 0.05)).any()
                          for g in (1, -1) if (sg == g).any())
                if hit:
                    stats["building"] -= 1; stats["prop"] += 1; continue
                Pn.append(bp); Tn.append(T_[:12] - T_[:12].min() + 8 * len(Pn) - 8); Nn.append(m["N"][8 * q:8 * q + 8])
            if Pn:
                keepb.append(dict(m, P=np.concatenate(Pn), T=np.concatenate(Tn), N=np.concatenate(Nn), UV=np.zeros((8 * len(Pn), 2))))
        bld_out = keepb
    occupy(bld_out)
    for sg in (1, -1):   # widen by one station each way (the slab is sampled at stations, faces span between them)
        lo_, hi_ = occ_lo[sg].copy(), occ_hi[sg].copy()
        occ_lo[sg][1:] = np.minimum(occ_lo[sg][1:], lo_[:-1]); occ_lo[sg][:-1] = np.minimum(occ_lo[sg][:-1], lo_[1:])
        occ_hi[sg][1:] = np.maximum(occ_hi[sg][1:], hi_[:-1]); occ_hi[sg][:-1] = np.maximum(occ_hi[sg][:-1], hi_[1:])

    # forest line
    tx, ty, tz = trees if trees is not None else (np.zeros(0),) * 3
    ts, td, tj = F.sd(np.c_[tx, ty])
    nforest = 0
    for side in (+1, -1):
        edge = F.el if side > 0 else F.er
        ok = (np.sign(td) == side)
        gap = np.abs(td) - edge[tj]
        ok &= (gap > FOREST_MIN_GAP - 0.5) & (gap < FOREST_MAX)
        ss, dd, zz = ts[ok], np.abs(td[ok]) - 0.6, tz[ok]
        o = np.argsort(ss); ss, dd, zz = ss[o], dd[o], zz[o]
        lo = np.searchsorted(ss, grid - FOREST_WIN); hi = np.searchsorted(ss, grid + FOREST_WIN)
        ei = np.clip(np.interp(grid, F.s, np.arange(len(F.s))).astype(int), 0, len(F.s) - 1)
        for bi in np.nonzero(hi - lo >= FOREST_N)[0]:
            fd = max(float(np.percentile(dd[lo[bi]:hi[bi]], 25)), edge[ei[bi]] + FOREST_MIN_GAP)
            if fd < near[side][bi]:
                near[side][bi] = fd; src[side][bi] = "forest"
                zw = zz[lo[bi]:hi[bi]]; zcl = np.interp(grid[bi], F.s, F.z)
                zlo[side][bi] = min(zw.min(), zcl) - 2.0; zhi[side][bi] = max(zw.max(), zcl) + 2.5; nforest += 1

    # cut against occupancy: depth = room before the next collider behind; drop the station when < MIN_DEPTH
    # of room, or when something already occupies the space in front of / at the face
    depth = {}
    for sg in (1, -1):
        room = occ_lo[sg] - near[sg] - 0.05
        clash = (occ_hi[sg] > near[sg] - 0.05) & (occ_lo[sg] < near[sg] + WALL_DEPTH + 0.05)
        depth[sg] = np.where(clash, np.clip(room, 0, WALL_DEPTH), WALL_DEPTH)
        drop = clash & (room < MIN_DEPTH)
        near[sg][drop] = np.inf; stats["stations_cut"] = stats.get("stations_cut", 0) + int(drop.sum())
    # ground for stations without barrier z (forest): centreline z (bottom well below, top well above)
    zc = np.interp(grid, F.s, F.z)
    out, nslab, nrun = [], 0, 0
    acc = {}
    claimed = {+1: np.zeros(ns, bool), -1: np.zeros(ns, bool)}
    for side in (+1, -1):
        has = np.isfinite(near[side])
        # bridge short gaps between stations with similar offsets
        idx = np.nonzero(has)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= GAP_BRIDGE + 1 and abs(near[side][a] - near[side][b]) < JUMP and src[side][a] == src[side][b] \
                    and np.isinf(occ_lo[side][a + 1:b]).all():
                for k in range(a + 1, b):
                    t = (k - a) / (b - a); near[side][k] = near[side][a] * (1 - t) + near[side][b] * t; has[k] = True
                    src[side][k] = src[side][a]
        # runs
        i = 0
        while i < ns:
            if not has[i]:
                i += 1; continue
            j = i
            while j + 1 < ns and has[j + 1] and abs(near[side][j + 1] - near[side][j]) < JUMP and src[side][j + 1] == src[side][i]:
                j += 1
            r = np.arange(i, j + 1); i = j + 1
            d = near[side][r].copy()
            if len(r) >= 5:   # smooth but never move the face toward the road by more than 5 cm or away by > 15 cm
                ds = gaussian_filter1d(d, 1.5, mode="nearest"); d = np.clip(ds, d - 0.15, d + 0.05)
            lo_ = zlo[side][r]; hi_ = zhi[side][r]
            lo_ = np.where(np.isfinite(lo_), lo_, zc[r] - 3.0); hi_ = np.where(np.isfinite(hi_), hi_, zc[r] + 4.0)
            # flared ends: the slab continues FLARE m past each end of the barrier into free stations, turning away
            # from the road (guardrail-terminal shape) - the barrier itself stays fully covered, and a car sliding
            # along it never meets a square corner
            ne = int(round(FLARE / STEP))
            free = lambda k: 0 <= k < ns and not has[k] and not claimed[side][k] and not (
                occ_hi[side][k] > near[side][r[0] if k < r[0] else r[-1]] - 0.05 and occ_lo[side][k] < near[side][r[0] if k < r[0] else r[-1]] + FLARE_D + WALL_DEPTH + 0.05)
            pre, post = [], []
            while len(pre) < ne and free(r[0] - 1 - len(pre)):
                pre.append(r[0] - 1 - len(pre))
            while len(post) < ne and free(r[-1] + 1 + len(post)):
                post.append(r[-1] + 1 + len(post))
            pre = pre[::-1]
            rr = np.r_[np.array(pre, int), r, np.array(post, int)]
            if len(rr) < 2:
                continue
            claimed[side][rr] = True
            fl = np.r_[np.arange(len(pre), 0, -1), np.zeros(len(r)), np.arange(1, len(post) + 1)] / max(ne, 1)
            d = np.r_[np.full(len(pre), d[0]), d, np.full(len(post), d[-1])] + FLARE_D * fl ** 2
            lo_ = np.r_[np.full(len(pre), lo_[0]), lo_, np.full(len(post), lo_[-1])]
            hi_ = np.r_[np.full(len(pre), hi_[0]), hi_, np.full(len(post), hi_[-1])]
            dep = np.r_[np.full(len(pre), depth[side][r[0]]), depth[side][r], np.full(len(post), depth[side][r[-1]])]
            dep = np.minimum.reduce([dep, np.r_[dep[1:], dep[-1]], np.r_[dep[0], dep[:-1]]])   # faces span stations
            r = rr; s = grid[r].astype(float)
            lo_ = gaussian_filter1d(lo_, 1.0, mode="nearest") - 0.3
            hi_ = np.maximum(gaussian_filter1d(hi_, 1.0, mode="nearest"), lo_ + WALL_TOP + 0.3)
            P, T, N = slab(F, s, d, lo_, hi_, side, dep)
            ch = int(min(s.mean() // CHUNK, 999))
            acc.setdefault((side, ch), []).append((P, T, N)); nrun += 1
    for (side, ch), parts in sorted(acc.items()):
        Ps, Ts, Ns, k = [], [], [], 0
        for P, T, N in parts:
            Ps.append(P); Ts.append(T + k); Ns.append(N); k += len(P)
        P = np.concatenate(Ps); assert len(P) < 65000
        out.append(dict(name=f"{prefix}_{'L' if side > 0 else 'R'}_{ch:02d}", material="invisible", P=P, T=np.concatenate(Ts),
                        UV=np.zeros((len(P), 2)), N=np.concatenate(Ns), **FLAGS)); nslab += 1
    for nm, Pc in across:
        nm = "1WALL_start" if nm.startswith("1WALL_start") else "1WALL_end"
        # across-road end wall: rebuild as a 2 m deep closed box on the same footprint line
        s, d, j = F.sd(Pc[:, :2]); sm = s.mean(); back = -1 if nm == "1WALL_start" else 1
        q = np.array([sm, sm + back * 2.0]); dd = np.array([d.min() - 1, d.max() + 1])
        ring = np.array([F.xy(q[0], dd[0]), F.xy(q[0], dd[1]), F.xy(q[1], dd[1]), F.xy(q[1], dd[0])]).reshape(4, 2)
        z0, z1 = Pc[:, 2].min(), Pc[:, 2].max()
        P, T, N = obb_box(np.r_[np.c_[ring, np.full(4, z0)], np.c_[ring, np.full(4, z1)]])
        out.append(dict(name=nm, material="invisible", P=P, T=T, UV=np.zeros((8, 2)), N=N, **FLAGS))
    out = bld_out + out
    stats.update(slab_meshes=nslab, runs=nrun, forest_stations=nforest,
                 new_tris=int(sum(len(m["T"]) for m in out)), new_meshes=len(out))
    return out, stats, near


def sample_faces(P, T):
    """per connected component: barycentric samples of its faces (dense enough for 2 m station occupancy)"""
    P = np.asarray(P, float); T = np.asarray(T).reshape(-1, 3)
    bw = np.array([[u, v, 1 - u - v] for u in np.linspace(0, 1, 5) for v in np.linspace(0, 1, 5) if u + v <= 1 + 1e-9])
    return [np.einsum("kj,tjc->tkc", bw, P[T]).reshape(-1, 3)]
