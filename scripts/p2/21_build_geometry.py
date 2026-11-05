"""Phase 2: build all track geometry (local ENU metres, origin config/origin.json) -> build/geometry.pkl

Groups (each becomes one or more kn5 files in 22_build_kn5.py):
  road      : 1ROAD surface + 1GRASS shoulders, 250 m chunks              (shared)
  terrain   : 1GRASS ground / rock from constrained Delaunay of the DTM   (shared)
  backdrop  : non-physical 10 m DEM horizon with USGS imagery            (shared)
  trees     : alpha-tested crossed cards at LiDAR tree positions          (shared)
  barriers  : guardrails (visual + 1WALL), invisible cut-bank / end walls (shared)
  lines_road, lines_track : paint ribbons (per layout)
  kerbs     : 1KERB red/white (track layouts)
Also writes reports/visual_check_stations.csv.
"""
import csv, json, math, pickle, sys
import numpy as np, yaml, triangle
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d, label
from matplotlib.path import Path as MPath
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from geom import ROOT, DER, BUILD, CFG, SHOULDER_W, Road, Raster, load_centerline, vertex_normals, grid_tris

S_END = CFG["build_end_m"]
CHUNK = 250.0
rng = np.random.default_rng(7)
road = Road(S_END)
N = len(road.s)
dtm05 = Raster(DER / "dtm_r050_b40.tif")
dtm1 = Raster(DER / "dtm_r100_b150.tif")
full = load_centerline()
full_xy = np.c_[full["x_local"], full["y_local"]]
full_tree = cKDTree(full_xy)
G = {}  # group -> list of mesh dicts


def dtm(x, y):
    z = dtm05.sample(x, y)
    z2 = dtm1.sample(x, y)
    return np.where(np.isfinite(z), z, z2)


def add(group, name, material, P, T, UV, N_=None, **flags):
    P = np.asarray(P, float); T = np.asarray(T, np.int64).reshape(-1, 3)
    if len(T) == 0:
        return
    G.setdefault(group, []).append(dict(name=name, material=material, P=P, T=T, UV=np.asarray(UV, float),
                                        N=vertex_normals(P, T) if N_ is None else N_, **flags))


def chunk_of(i):
    return min(int(road.s[i] // CHUNK), int(road.s[-1] // CHUNK))


# ------------------------------------------------------------------ road + shoulders
fr = np.array([1, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6])
D_ROAD = np.concatenate([-road.er[:, None] * fr, np.zeros((N, 1)), road.el[:, None] * fr[::-1]], axis=1)  # N x 13, right->left
ii = np.repeat(np.arange(N)[:, None], D_ROAD.shape[1], 1)
RX, RY = road.xy(ii, D_ROAD)
RZ = road.surface_z(ii, D_ROAD)

# shoulder: edge -> edge+0.4 -> edge+0.8, outer z pulled toward DTM but limited
def shoulder(side):
    sgn = 1 if side == "L" else -1
    e = road.el if side == "L" else road.er
    col = -1 if side == "L" else 0
    ex, ey, ez = RX[:, col], RY[:, col], RZ[:, col]
    d_out = sgn * (e + SHOULDER_W)
    ox, oy = road.xy(np.arange(N), d_out)
    zt = dtm(ox, oy)
    zt = np.where(np.isfinite(zt), zt, ez - 0.05)
    dz = gaussian_filter1d(np.clip(zt - ez, -0.30, 0.35), 2, mode="nearest")
    oz = ez + dz
    mx, my = road.xy(np.arange(N), sgn * (e + SHOULDER_W / 2))
    mz = ez + 0.5 * dz - 0.02            # slight dished edge
    return np.stack([ex, mx, ox], 1), np.stack([ey, my, oy], 1), np.stack([ez, mz, oz], 1)

SLX, SLY, SLZ = shoulder("L")
SRX, SRY, SRZ = shoulder("R")

nchunks = chunk_of(N - 1) + 1
for c in range(nchunks):
    i0 = int(c * CHUNK); i1 = min(int((c + 1) * CHUNK), N - 1)
    rows = np.arange(i0, i1 + 1); nr = len(rows)
    P = np.stack([RX[rows], RY[rows], RZ[rows]], -1).reshape(-1, 3)
    UV = np.stack([D_ROAD[rows] / 3.5, np.repeat(road.s[rows, None], 13, 1) / 3.5], -1).reshape(-1, 2)
    add("road", f"1ROAD_{c:02d}", "road_asphalt", P, grid_tris(nr, 13), UV)
    for side, (X, Y, Z) in (("L", (SLX, SLY, SLZ)), ("R", (SRX[:, ::-1], SRY[:, ::-1], SRZ[:, ::-1]))):
        # columns must increase to the left: L = edge->out, R = out->edge
        P = np.stack([X[rows], Y[rows], Z[rows]], -1).reshape(-1, 3)
        UV = np.stack([np.tile(np.arange(3) * 0.2, nr), np.repeat(road.s[rows], 3) / 2.0], -1)
        add("road", f"1GRASS_sh{side}{c:02d}", "shoulder_dirt", P, grid_tris(nr, 3), UV)

# ------------------------------------------------------------------ terrain (constrained Delaunay)
bound = np.concatenate([np.c_[SRX[:, 2], SRY[:, 2]], np.c_[SLX[::-1, 2], SLY[::-1, 2]]])
bound_z = np.concatenate([SRZ[:, 2], SLZ[::-1, 2]])
nb = len(bound)
segs = np.c_[np.arange(nb), (np.arange(nb) + 1) % nb]
road_poly = MPath(bound)

xmin, ymin = full_xy[full["s_m"] <= S_END + 200].min(0) - 155
xmax, ymax = full_xy[full["s_m"] <= S_END + 200].max(0) + 155
pts = []
for step, dmin, dmax in ((1.0, 0, 16), (2.0, 16, 50), (4.0, 50, 148)):
    gx = np.arange(math.floor(xmin), xmax, step); gy = np.arange(math.floor(ymin), ymax, step)
    for y0 in range(0, len(gy), 400):
        X, Y = np.meshgrid(gx, gy[y0:y0 + 400]); X = X.ravel(); Y = Y.ravel()
        dist, j = full_tree.query(np.c_[X, Y], workers=4)
        hw = 0.5 * (full["width_m"][j])
        de = dist - hw
        m = (de >= dmin) & (de < dmax) & (full["s_m"][j] <= S_END + 150)
        pts.append(np.c_[X[m], Y[m]])
pts = np.concatenate(pts)
# remove grid points inside / too close to the built road polygon
inside = road_poly.contains_points(pts)
dmin_b, _ = cKDTree(bound).query(pts)
bt = cKDTree(np.c_[road.x, road.y]); dc, jc = bt.query(pts)
hw_b = np.maximum(road.el, road.er)[jc] + SHOULDER_W + 0.5
pts = pts[(~inside) & (dmin_b > 0.6) & ~(dc < hw_b)]
pz = dtm(pts[:, 0], pts[:, 1])
pts, pz = pts[np.isfinite(pz)], pz[np.isfinite(pz)]
print("terrain points", len(pts), "boundary", nb)
V = np.concatenate([bound, pts]); VZ = np.concatenate([bound_z, pz])
tri = triangle.triangulate(dict(vertices=V, segments=segs, holes=[[road.x[N // 2], road.y[N // 2]]]), "pc")
TV = tri["vertices"]; TT = tri["triangles"]
assert len(TV) == len(V), "triangle inserted Steiner points"
P3 = np.c_[TV, VZ]
e = np.stack([np.linalg.norm(P3[TT[:, a], :2] - P3[TT[:, b], :2], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], 1).max(1)
TT = TT[e < 9.0]
# orient CCW from above
cr = np.cross(P3[TT[:, 1]] - P3[TT[:, 0]], P3[TT[:, 2]] - P3[TT[:, 0]])[:, 2]
TT[cr < 0] = TT[cr < 0][:, [0, 2, 1]]
fn = np.cross(P3[TT[:, 1]] - P3[TT[:, 0]], P3[TT[:, 2]] - P3[TT[:, 0]])
cen = P3[TT].mean(1)
# rock vs ground from a 3 m-smoothed slope field (coherent patches, not per-triangle noise)
from scipy.ndimage import gaussian_filter
_a = np.nan_to_num(dtm1.a, nan=np.nanmean(dtm1.a)); _a = gaussian_filter(_a, 3)
_gy, _gx = np.gradient(_a, dtm1.t.e, dtm1.t.a)
_slope = Raster.__new__(Raster); _slope.__dict__.update(dtm1.__dict__); _slope.a = np.hypot(_gx, _gy)
_sl = _slope.sample(cen[:, 0], cen[:, 1])
nz = np.where(np.isfinite(_sl), 1 / np.sqrt(1 + np.nan_to_num(_sl) ** 2), 1.0)
_, jc = bt.query(cen[:, :2])
tch = np.minimum((road.s[jc] // CHUNK).astype(int), nchunks - 1)
for c in range(nchunks):
    for kind, sel in (("g", (tch == c) & (nz >= 0.66)), ("r", (tch == c) & (nz < 0.66))):
        T = TT[sel]
        if not len(T):
            continue
        used, inv = np.unique(T, return_inverse=True)
        P = P3[used]
        UV = np.c_[P[:, 0] / 6.0, P[:, 1] / 6.0] if kind == "g" else np.c_[(P[:, 0] + P[:, 1]) / 5.0, P[:, 2] / 5.0]
        add("terrain", f"1GRASS_t{kind}{c:02d}", "ground" if kind == "g" else "rock", P, inv.reshape(-1, 3), UV)
print("terrain tris", len(TT))

# ------------------------------------------------------------------ backdrop (non-physical)
bd = Raster(DER / "backdrop_dem_r10.tif")
H, W = bd.a.shape
rr, cc = np.meshgrid(np.arange(0, H, 2), np.arange(0, W, 2), indexing="ij")
X = bd.t.c + (cc + 0.5) * bd.t.a - bd.x0; Y = bd.t.f + (rr + 0.5) * bd.t.e - bd.y0
Z = bd.a[rr, cc] - 1.0
dist, _ = full_tree.query(np.c_[X.ravel(), Y.ravel()], workers=4)
nr_, nc_ = rr.shape
keep_v = (dist > 125).reshape(nr_, nc_) & np.isfinite(Z)
Tg = grid_tris(nr_, nc_)  # rows go south (row dir = -y), cols east -> fix orientation below
Pb = np.c_[X.ravel(), Y.ravel(), np.nan_to_num(Z.ravel())]
Tg = Tg[keep_v.ravel()[Tg].all(1)]
cr = np.cross(Pb[Tg[:, 1]] - Pb[Tg[:, 0]], Pb[Tg[:, 2]] - Pb[Tg[:, 0]])[:, 2]
Tg[cr < 0] = Tg[cr < 0][:, [0, 2, 1]]
UVb = np.c_[(cc.ravel() + 0.5) / W, (rr.ravel() + 0.5) / H]
# split into 4x4 tiles to stay under vertex limits
for ty in range(4):
    for tx in range(4):
        m = (rr.ravel()[Tg[:, 0]] * 4 // H == ty) & (cc.ravel()[Tg[:, 0]] * 4 // W == tx)
        T = Tg[m]
        if len(T):
            used, inv = np.unique(T, return_inverse=True)
            add("backdrop", f"backdrop_{ty}{tx}", "backdrop", Pb[used], inv.reshape(-1, 3), UVb[used], cast_shadows=False)

# ------------------------------------------------------------------ trees
tr = list(csv.DictReader(open(DER / "trees.csv")))
tx_, ty_, tz_, th_, tcr_, ts_ = (np.array([float(t[k]) for t in tr]) for k in ("x_local", "y_local", "ground_z", "height_m", "crown_radius_m", "s_m"))
d_tc, j_tc = bt.query(np.c_[tx_, ty_])
hw_t = np.maximum(road.el, road.er)[j_tc]
sel = (ts_ <= S_END + 60) & (d_tc > hw_t + 1.2)
print("trees used", sel.sum(), "of", len(tr))
ratio = tcr_ / np.maximum(th_, 1)
species = np.where((th_ > 24) & (ratio < 0.3), "euc", np.where(ratio > 0.42, "oak", "bay"))
tree_buckets = {}
for k in np.nonzero(sel)[0]:
    c = min(int(road.s[j_tc[k]] // CHUNK), nchunks - 1)
    tree_buckets.setdefault((c, species[k]), []).append(k)
for (c, sp), ks in sorted(tree_buckets.items()):
    P, T, UV = [], [], []
    for k in ks:
        h = th_[k]; w = min(max(tcr_[k] * 2.0, 2.5), h * 1.1)
        z0 = tz_[k] - 0.3
        a0 = rng.uniform(0, np.pi)
        for a in (a0, a0 + np.pi / 2):
            dx, dy = math.cos(a) * w / 2, math.sin(a) * w / 2
            q = [(tx_[k] - dx, ty_[k] - dy, z0), (tx_[k] + dx, ty_[k] + dy, z0), (tx_[k] + dx, ty_[k] + dy, z0 + h), (tx_[k] - dx, ty_[k] - dy, z0 + h)]
            for back in (False, True):
                b = len(P); P += q; UV += [(0, 0), (1, 0), (1, 1), (0, 1)]
                T += [(b, b + 1, b + 2), (b, b + 2, b + 3)] if not back else [(b, b + 2, b + 1), (b, b + 3, b + 2)]
    P = np.array(P); T = np.array(T)
    # upward-biased normals give softer, more natural lighting on cards
    Nn = vertex_normals(P, T); Nn[:, 2] += 0.8; Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    add("trees", f"tree_{sp}_{c:02d}", f"tree_{sp}", P, T, np.array(UV), N_=Nn, cast_shadows=True, transparent=False, lod_out=900.0)

# ------------------------------------------------------------------ barriers
feats = json.load(open(DER / "features.geojson"))["features"]
bars = []  # (s0, s1, side)
for f in feats:
    p = f["properties"]
    if p["kind"] == "guardrail_candidate" and p["s_start_m"] < S_END:
        bars.append((p["s_start_m"], min(p["s_end_m"], S_END - 1), p["side"]))
GR = []
for gi, (s0, s1, side) in enumerate(bars):
    i = np.arange(int(s0), int(s1) + 1)
    sgn = 1 if side == "left" else -1
    e = road.el[i] if side == "left" else road.er[i]
    d = sgn * (e + 0.45)
    x, y = road.xy(i, d)
    zb = road.surface_z(i, sgn * e, noise=False)  # posts stand at road-edge height
    GR.append(dict(s0=float(s0), s1=float(s1), side=side))
    # beam: double sided vertical strip 0.45..0.80 m
    P = np.concatenate([np.c_[x, y, zb + 0.45], np.c_[x, y, zb + 0.80]])
    n = len(i); a = np.arange(n - 1)
    T1 = np.c_[a, a + 1, a + n + 1]; T2 = np.c_[a, a + n + 1, a + n]
    T = np.concatenate([T1, T2, T1[:, [0, 2, 1]], T2[:, [0, 2, 1]]])
    UV = np.c_[np.r_[i - i[0], i - i[0]] / 4.0, np.r_[np.zeros(n), np.ones(n)]]
    # visual beam: separate vertices for the back face so normals don't cancel
    Tv = np.concatenate([T1, T2, T1[:, [0, 2, 1]] + 2 * n, T2[:, [0, 2, 1]] + 2 * n])
    add("barriers", f"guardrail_{gi:03d}", "guardrail", np.concatenate([P, P]), Tv, np.concatenate([UV, UV]))
    # physics: same strip but taller (0..1.0 m), invisible
    P = np.concatenate([np.c_[x, y, zb - 0.3], np.c_[x, y, zb + 1.0]])
    add("barriers", f"1WALL_gr{gi:03d}", "invisible", P, T, UV, visible=False, renderable=False, cast_shadows=False)
    # posts every ~1.9 m
    PP, PT = [], []
    for k in range(0, n, 2):
        cx, cy, cz = x[k], y[k], zb[k]
        b = len(PP)
        for dz in (-0.2, 0.75):
            for ox, oy in ((-0.07, -0.07), (0.07, -0.07), (0.07, 0.07), (-0.07, 0.07)):
                PP.append((cx + ox, cy + oy, cz + dz))
        for q in range(4):
            q1 = (q + 1) % 4
            PT += [(b + q, b + q1, b + 4 + q1), (b + q, b + 4 + q1, b + 4 + q)]
    PP = np.array(PP)
    add("barriers", f"grpost_{gi:03d}", "guardrail", PP, np.array(PT), PP[:, :2] * 2)

# invisible walls behind steep cuts (prevents climbing near-vertical faces)
WALLS = []
for side, env in (("left", road.env_l), ("right", road.env_r)):
    m = env == "wall_or_steep_cut"
    lab, nl = label(m)
    for l in range(1, nl + 1):
        idx = np.nonzero(lab == l)[0]
        if len(idx) < 8:
            continue
        sgn = 1 if side == "left" else -1
        e = road.el[idx] if side == "left" else road.er[idx]
        x, y = road.xy(idx, sgn * (e + SHOULDER_W + 1.2))
        zb = road.surface_z(idx, sgn * e, noise=False)
        P = np.concatenate([np.c_[x, y, zb - 1.0], np.c_[x, y, zb + 4.0]])
        n = len(idx); a = np.arange(n - 1)
        T1 = np.c_[a, a + 1, a + n + 1]; T2 = np.c_[a, a + n + 1, a + n]
        T = np.concatenate([T1, T2, T1[:, [0, 2, 1]], T2[:, [0, 2, 1]]])
        add("barriers", f"1WALL_cut{len(WALLS):03d}", "invisible", P, T, np.zeros((len(P), 2)), visible=False, renderable=False, cast_shadows=False)
        WALLS.append(dict(s0=float(road.s[idx[0]]), s1=float(road.s[idx[-1]]), side=side))

# end walls at s=0 (behind the pit spot) and at the end of the built road
for nm, i in (("start", 1), ("end", N - 2)):
    d = np.linspace(-road.er[i] - SHOULDER_W - 1, road.el[i] + SHOULDER_W + 1, 8)
    x, y = road.xy(np.full(8, i), d); zb = road.z[i]
    P = np.concatenate([np.c_[x, y, np.full(8, zb - 2)], np.c_[x, y, np.full(8, zb + 4)]])
    a = np.arange(7); T1 = np.c_[a, a + 1, a + 9]; T2 = np.c_[a, a + 9, a + 8]
    T = np.concatenate([T1, T2, T1[:, [0, 2, 1]], T2[:, [0, 2, 1]]])
    add("barriers", f"1WALL_{nm}", "invisible", P, T, np.zeros((16, 2)), visible=False, renderable=False, cast_shadows=False)

# ------------------------------------------------------------------ road markings
mk = list(csv.DictReader(open(DER / "markings.csv")))
def mk_at(s):
    for r in mk:
        if float(r["s_start_m"]) <= s < float(r["s_end_m"]):
            return r
    return mk[-1]

def ribbon(group, name, mat, i, d_center, width, dash=None):
    """paint strip along station indices i at offset d_center (array), lifted 6 mm."""
    if dash:
        on, period = dash
        i = i[((road.s[i] % period) < on)]
    if len(i) < 2:
        return
    # split into continuous runs
    runs = np.split(i, np.nonzero(np.diff(i) > 1)[0] + 1)
    P, T, UV = [], [], []
    for r in runs:
        if len(r) < 2:
            continue
        dc = d_center[r] if np.ndim(d_center) else np.full(len(r), d_center)
        dd = np.stack([dc - width / 2, dc + width / 2], 1)
        rr_ = np.repeat(r[:, None], 2, 1)
        x, y = road.xy(rr_, dd); z = road.surface_z(rr_, dd) + 0.006
        b = len(P)
        P += list(np.stack([x, y, z], -1).reshape(-1, 3)); UV += [(0, 0), (1, 0)] * len(r)
        Tq = grid_tris(len(r), 2, b); T += list(Tq)
    if T:
        add(group, name, mat, np.array(P), np.array(T), np.array(UV), cast_shadows=False)

idx_all = np.arange(2, N - 2)
ctype = np.array([mk_at(s)["assumed_center_type"] for s in road.s])
ledge = np.array([mk_at(s)["left_edge_line"] in ("likely", "possible") for s in road.s])
redge = np.array([mk_at(s)["right_edge_line"] in ("likely", "possible") for s in road.s])
for c in range(nchunks):
    ic = idx_all[(idx_all >= c * CHUNK) & (idx_all <= (c + 1) * CHUNK)]
    for grp in ("lines_road", "lines_track"):
        ribbon(grp, f"line_edgeL_{c:02d}", "paint_white", ic[ledge[ic]], road.el - 0.25, 0.10)
        ribbon(grp, f"line_edgeR_{c:02d}", "paint_white", ic[redge[ic]], -road.er + 0.25, 0.10)
    dy = ic[ctype[ic] == "double_yellow"]
    ribbon("lines_road", f"line_cdyL_{c:02d}", "paint_yellow", dy, 0.10, 0.10)
    ribbon("lines_road", f"line_cdyR_{c:02d}", "paint_yellow", dy, -0.10, 0.10)
    br = ic[np.isin(ctype[ic], ["broken_yellow", "broken"])]
    ribbon("lines_road", f"line_cbr_{c:02d}", "paint_yellow", br, 0.0, 0.10, dash=(3, 12))

# ------------------------------------------------------------------ kerbs (track layouts)
K = yaml.safe_load(open(ROOT / "config/kerbs.yaml"))
ks = gaussian_filter1d(road.k, K["sigma_m"])
spans = {"left": [], "right": []}
lab, nl = label(np.abs(ks) > K["curvature_threshold_1pm"])
for l in range(1, nl + 1):
    idx = np.nonzero(lab == l)[0]
    if len(idx) < K["min_corner_length_m"]:
        continue
    apex = idx[np.argmax(np.abs(ks[idx]))]
    inner = "left" if ks[apex] > 0 else "right"
    outer = "right" if inner == "left" else "left"
    spans[inner].append((road.s[apex] - K["apex_entry_m"], road.s[apex] + K["inside_exit_m"]))
    s0 = road.s[apex] + K["outside_exit_start_m"]
    spans[outer].append((s0, s0 + K["outside_exit_length_m"]))
KERBS = []
for side in spans:
    sp = sorted(spans[side]); merged = []
    for a, b in sp:
        if any(lo <= a <= hi or lo <= b <= hi for lo, hi in K["skip_s_ranges"]):
            continue
        if merged and a - merged[-1][1] < K["merge_gap_m"]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for a, b in merged:
        i = np.arange(max(int(a), 1), min(int(b), N - 2) + 1)
        if len(i) < 3:
            continue
        sgn = 1 if side == "left" else -1
        e = road.el[i] if side == "left" else road.er[i]
        prof_d = np.array([0.0, 0.15, 0.35, 0.55, 0.75, 1.0]) * K["width_m"]
        prof_h = np.array([0.003, 0.02, 1.0 * K["height_m"], 1.0 * K["height_m"], 0.6 * K["height_m"], 0.003])
        if side == "right":
            prof_d, prof_h = prof_d[::-1], prof_h[::-1]
        dd = sgn * (e[:, None] + prof_d[None, :])
        rr_ = np.repeat(i[:, None], len(prof_d), 1)
        x, y = road.xy(rr_, dd)
        # base: the shoulder surface (piecewise linear over its 3 columns) so the kerb's outer edge lands on the
        # shoulder/terrain seam with no step; inner edge = road edge
        SZ = SLZ if side == "left" else SRZ
        t = np.clip((np.abs(dd) - e[:, None]) / (SHOULDER_W / 2), 0, 2)
        z0, z1, z2 = SZ[rr_, 0], SZ[rr_, 1], SZ[rr_, 2]
        base = np.where(t <= 1, z0 + (z1 - z0) * t, z1 + (z2 - z1) * (t - 1))
        ramp = np.clip(np.minimum(np.arange(len(i)), np.arange(len(i))[::-1]) / 2.0, 0, 1)  # 2 m taper at kerb ends
        z = base + 0.003 + (prof_h[None, :] - 0.003) * ramp[:, None]
        P = np.stack([x, y, z], -1).reshape(-1, 3)
        UV = np.stack([np.tile(np.linspace(0, 1, len(prof_d)), (len(i), 1)), np.repeat(road.s[i, None], len(prof_d), 1) / (2 * K["stripe_length_m"])], -1).reshape(-1, 2)
        add("kerbs", f"1KERB_{side[0].upper()}{len(KERBS):03d}", "kerb", P, grid_tris(len(i), len(prof_d)), UV, cast_shadows=False)
        KERBS.append(dict(s0=float(road.s[i[0]]), s1=float(road.s[i[-1]]), side=side))
print("kerbs", len(KERBS))

# ------------------------------------------------------------------ timing / spawn locators
def frame(s, d=0.0, lift=0.0):
    i = int(round(s))
    x, y = road.xy(i, d)
    return dict(p=(float(x), float(y), float(road.surface_z(i, d, noise=False) + lift)),
                fwd=(float(road.tx[i]), float(road.ty[i]), float(road.tz) if hasattr(road, "tz") else 0.0))

loc = {}
for nm, s in (("AC_START_0", CFG["spawn_m"]), ("AC_HOTLAP_START_0", CFG["spawn_m"]), ("AC_PIT_0", CFG["pit_m"])):
    loc[nm] = frame(s, 0.0, 0.0)
for nm, s in (("AB_START", CFG["start_gate_m"]), ("AB_FINISH", CFG["finish_gate_m"])):
    i = int(round(s))
    loc[f"AC_{nm}_L"] = frame(s, road.el[i] + 1.0, 0.5)
    loc[f"AC_{nm}_R"] = frame(s, -road.er[i] - 1.0, 0.5)

BUILD.mkdir(exist_ok=True)
meta = dict(s_end=float(road.s[-1]), nchunks=nchunks, guardrails=GR, walls=WALLS, kerbs=KERBS, locators=loc,
            n_trees=int(sel.sum()))
pickle.dump(dict(groups=G, meta=meta), open(BUILD / "geometry.pkl", "wb"))

# stations for visual checks (every 250 m)
with open(ROOT / "reports/visual_check_stations.csv", "w") as f:
    w = csv.writer(f); w.writerow(["station_m", "lat", "lon", "heading_deg_true", "z_navd88"])
    for s in np.arange(0, CFG["finish_gate_m"] + 1, 250):
        i = int(s); hd = (math.degrees(math.atan2(road.tx[i], road.ty[i])) + 360) % 360  # grid north ~ true north (+/-0.7 deg)
        w.writerow([int(s), f"{full['lat'][i]:.7f}", f"{full['lon'][i]:.7f}", f"{hd:.0f}", f"{road.z[i]:.1f}"])
for g, ms in G.items():
    print(f"{g:12s} meshes={len(ms):4d} verts={sum(len(m['P']) for m in ms):8d} tris={sum(len(m['T']) for m in ms):8d}")
