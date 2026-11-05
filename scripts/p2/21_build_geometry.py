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
from progress import bar, Section
import time as _time
_T0 = _time.time()
def _tick(name):
    global _T0
    print(f"  - {name:28s} {_time.time() - _T0:6.1f}s", flush=True); _T0 = _time.time()

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

_tick("road + shoulders")
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
dcen, jc = bt.query(cen[:, :2])
tch = np.minimum((road.s[jc] // CHUNK).astype(int), nchunks - 1)
# surface classes, calibrated against the spring GoPro survey (data/derived/gopro_survey_*.json):
#   r rock      > ~49 deg
#   b cut bank  ~25-49 deg: mixed bare dirt / moss / new grass (most cut banks in the frames)
#   l litter    gentle, within 6 m of the road edge or under tree crowns (CHM > 5 m): oak/bay leaf litter
#   g grass     gentle and open: bright spring grass
_chm_r = Raster(DER / "chm_r100_b150.tif")
_chm = _chm_r.sample(cen[:, 0], cen[:, 1])
_edge = dcen - np.maximum(road.el, road.er)[jc]
_jit = np.sin(cen[:, 0] * 0.37) * np.cos(cen[:, 1] * 0.29) * 1.5  # break the straight 6 m contour
kind_t = np.where(nz < 0.66, "r", np.where(nz < 0.9, "b",
                  np.where((_edge < 6 + _jit) | (np.nan_to_num(_chm) > 5), "l", "g")))
TMAT = {"g": "ground", "r": "rock", "b": "bank", "l": "litter"}
for c in range(nchunks):
    for kind in "grbl":
        T = TT[(tch == c) & (kind_t == kind)]
        if not len(T):
            continue
        used, inv = np.unique(T, return_inverse=True)
        P = P3[used]
        UV = np.c_[P[:, 0] / 6.0, P[:, 1] / 6.0] if kind in "gl" else np.c_[(P[:, 0] + P[:, 1]) / 5.0, P[:, 2] / 5.0]
        add("terrain", f"1GRASS_t{kind}{c:02d}", TMAT[kind], P, inv.reshape(-1, 3), UV)
print("terrain classes", {k: int((kind_t == k).sum()) for k in "grbl"})
print("terrain tris", len(TT))

_tick("terrain")
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

_tick("backdrop")
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

_tick("trees")
# ------------------------------------------------------------------ understory shrubs (GoPro: dense green brush on every
# bank and drop-off between the trunks; LiDAR trees >= 4 m miss it). Non-physical crossed cards, 1 mesh per chunk.
srng = np.random.default_rng(11)
SH_P, SH_T, SH_UV, SH_C = [], [], [], []
_rx, _ry, _rnx, _rny = road.x.tolist(), road.y.tolist(), road.nx.tolist(), road.ny.tolist()  # scalar fast path
for i in bar(range(0, N, 2), "shrubs"):
    for side in (-1, 1):
        for _ in range(2):
            off = srng.uniform(2.2, 26.0) ** 1.0
            hw_i = road.el[i] if side > 0 else road.er[i]
            d = side * (hw_i + SHOULDER_W + off)
            x, y = _rx[i] + _rnx[i] * d, _ry[i] + _rny[i] * d
            x, y = x + srng.normal(0, 0.6), y + srng.normal(0, 0.6)
            sl = _slope.sample1(x, y)
            if np.isfinite(sl) and sl > 1.15:  # bare rock faces: no brush
                continue
            ch = _chm_r.sample1(x, y)
            if off > 8 and not (np.isfinite(ch) and ch > 3) and srng.uniform() < 0.6:  # thinner in the open
                continue
            z = dtm05.sample1(x, y)
            if not math.isfinite(z):
                z = dtm1.sample1(x, y)
            if not np.isfinite(z):
                continue
            h = srng.uniform(0.9, 2.6) * (1.3 if off > 6 else 1.0); w = h * srng.uniform(1.2, 1.9)
            a0 = srng.uniform(0, np.pi); u0 = 0.5 * srng.integers(0, 2)  # two shrub styles on one atlas
            for a in (a0, a0 + np.pi / 2):
                dx, dy = math.cos(a) * w / 2, math.sin(a) * w / 2
                b = len(SH_P)
                SH_P += [(x - dx, y - dy, z - 0.25), (x + dx, y + dy, z - 0.25), (x + dx, y + dy, z + h), (x - dx, y - dy, z + h)]
                SH_UV += [(u0, 0), (u0 + 0.5, 0), (u0 + 0.5, 1), (u0, 1)]
                SH_T += [(b, b + 1, b + 2), (b, b + 2, b + 3), (b, b + 2, b + 1), (b, b + 3, b + 2)]
                SH_C += [chunk_of(i)] * 4
SH_P = np.array(SH_P); SH_T = np.array(SH_T); SH_UV = np.array(SH_UV); SH_C = np.array(SH_C)
for c in range(nchunks):
    m = SH_C[SH_T[:, 0]] == c
    if not m.any():
        continue
    used, inv = np.unique(SH_T[m], return_inverse=True)
    P = SH_P[used]; T = inv.reshape(-1, 3)
    Nn = np.tile([0, 0, 1.0], (len(P), 1))
    add("trees", f"shrub_{c:02d}", "shrub", P, T, SH_UV[used], N_=Nn, cast_shadows=False, lod_out=250.0)
print("shrub cards", len(SH_T) // 4)

_tick("shrubs")
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

_tick("barriers + walls")
# ------------------------------------------------------------------ road markings
# MUTCD dimensions, worn-paint textures and RPMs: see scripts/p2/road_surface.py. Non-physical (names don't start
# with 1ROAD/1KERB...), lifted >= 2.3 cm, shared by road and track layouts (group lines_road).
import road_surface as RS
mk = list(csv.DictReader(open(DER / "markings.csv")))
def mk_at(s):
    for r in mk:
        if float(r["s_start_m"]) <= s < float(r["s_end_m"]):
            return r
    return mk[-1]

def surf_pt(sf, d):
    """road surface point at fractional station sf (index units, 1 m) and offset d, linear along s like the mesh"""
    sf = np.clip(np.asarray(sf, float), 0, N - 1.001); i0 = np.floor(sf).astype(int); f = sf - i0
    x0, y0 = road.xy(i0, d); x1, y1 = road.xy(i0 + 1, d)
    z0 = road.surface_z(i0, d); z1 = road.surface_z(i0 + 1, d)
    return np.stack([x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, z0 + (z1 - z0) * f], -1)

def ribbon(group, name, mat, i, d_center, width, dash=None, cols=(0.0, 1.0), vper=RS.LINE_PERIOD_M, lift=RS.LIFT["line"]):
    """paint strip along station indices i at offset d_center (array); cols = fractions across (include the crown
    when the strip straddles d=0); u across, v = s / vper."""
    if dash:
        on, period = dash
        i = i[((road.s[i] % period) < on)]
    if len(i) < 2:
        return
    runs = np.split(i, np.nonzero(np.diff(i) > 1)[0] + 1)
    cols = np.asarray(cols); nc = len(cols)
    P, T, UV = [], [], []
    for r in runs:
        if len(r) < 2:
            continue
        dc = d_center[r] if np.ndim(d_center) else np.full(len(r), d_center)
        dd = dc[:, None] - width / 2 + width * cols[None, :]
        rr_ = np.repeat(r[:, None], nc, 1)
        x, y = road.xy(rr_, dd); z = road.surface_z(rr_, dd) + lift
        b = len(P)
        P += list(np.stack([x, y, z], -1).reshape(-1, 3))
        UV += list(np.stack([np.tile(cols, len(r)), np.repeat(road.s[r] / vper, nc)], -1))
        T += list(grid_tris(len(r), nc, b))
    if T:
        add(group, name, mat, np.array(P), np.array(T), np.array(UV), cast_shadows=False)

idx_all = np.arange(2, N - 2)
ctype = np.array([mk_at(s)["assumed_center_type"] for s in road.s])
ledge = np.array([mk_at(s)["left_edge_line"] in ("likely", "possible") for s in road.s])
redge = np.array([mk_at(s)["right_edge_line"] in ("likely", "possible") for s in road.s])
EW = RS.EDGE_RIBBON
for c in range(nchunks):
    ic = idx_all[(idx_all >= c * CHUNK) & (idx_all <= (c + 1) * CHUNK)]
    # edge line centre 0.25 m inside the pavement edge (GoPro: ~20-30 cm of asphalt outside the line)
    ribbon("lines_road", f"line_edgeL_{c:02d}", "paint_white", ic[ledge[ic]], road.el - 0.25, EW)
    ribbon("lines_road", f"line_edgeR_{c:02d}", "paint_white", ic[redge[ic]], -road.er + 0.25, EW)
    dy = ic[ctype[ic] == "double_yellow"]
    ribbon("lines_road", f"line_cdy_{c:02d}", "paint_yellow_dbl", dy, 0.0, RS.DY_RIBBON, cols=(0.0, 0.5, 1.0))
    br = ic[np.isin(ctype[ic], ["broken_yellow", "broken"])]
    ribbon("lines_road", f"line_cbr_{c:02d}", "paint_yellow", br, 0.0, EW, dash=RS.BROKEN, cols=(0.0, 0.5, 1.0))

# raised pavement markers: yellow, every 24 m, one each side of the double yellow (GoPro frames 166-176;
# survey B section of the road, i.e. the northern half). Thin low frustums, non-physical, no shadows.
_svs, _, _ = RS.survey_condition(ROOT, road.s[-1])
_n_a = 110  # frames 1-110 = survey A, 111-219 = survey B
RPM_S0 = float(_svs[min(_n_a, len(_svs) - 1)]) if len(_svs) > _n_a else 0.0
def rpm_box(sf, d):
    c = surf_pt(sf, d); i = int(round(min(sf, N - 1)))
    t = np.array([road.tx[i], road.ty[i], 0]); n = np.array([road.nx[i], road.ny[i], 0])
    lo, hi = 0.05, 0.035; h0, h1 = RS.LIFT["rpm"] - 0.008, RS.LIFT["rpm"] + 0.012
    q = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    bot = [c + t * a * lo + n * b * lo + [0, 0, h0] for a, b in q]
    top = [c + t * a * hi + n * b * hi + [0, 0, h1] for a, b in q]
    P, T = [], []
    def quad(a, b, cc, dd):
        k = len(P); P.extend([a, b, cc, dd]); T.extend([(k, k + 1, k + 2), (k, k + 2, k + 3)])
    quad(*top)
    for j in range(4):
        quad(bot[j], bot[(j + 1) % 4], top[(j + 1) % 4], top[j])
    return P, T
for c in range(nchunks):
    P, T = [], []
    for s in np.arange(math.ceil(max(c * CHUNK, RPM_S0, 3) / RS.RPM_SPACING) * RS.RPM_SPACING, min((c + 1) * CHUNK, road.s[-1] - 3), RS.RPM_SPACING):
        i = int(s)
        if ctype[i] != "double_yellow":
            continue
        for d in (-(RS.DY_RIBBON / 2 + 0.07), RS.DY_RIBBON / 2 + 0.07):
            p, t = rpm_box(s, d); T += [tuple(np.array(x) + len(P)) for x in t]; P += p
    if T:
        P = np.array(P); UV = np.tile([[0.1, 0.4], [0.9, 0.4], [0.9, 0.6], [0.1, 0.6]], (len(P) // 4, 1))
        add("lines_road", f"rpm_{c:02d}", "rpm_yellow", P, np.array(T), UV, N_=None, cast_shadows=False)

# ------------------------------------------------------------------ asphalt decals + road UV atlas
# Road UV: u = 0.5 + d/12, v = s/12 (lateral features of the 12 m procedural asphalt atlas stay registered to the
# centre line). Decals: black crack-seal bands (longitudinal at the centre joint / wheel paths / edges, transverse),
# patch repairs (grey newer asphalt + fresh black), alligator cracking; density from the GoPro survey condition.
for m in G["road"]:
    if m["name"].startswith("1ROAD_"):
        c = int(m["name"].split("_")[1]); i0 = int(c * CHUNK); i1 = min(int((c + 1) * CHUNK), N - 1)
        rows = np.arange(i0, i1 + 1)
        m["UV"] = np.stack([0.5 + D_ROAD[rows] / RS.ATLAS_M, np.repeat(road.s[rows, None] / RS.ATLAS_M, 13, 1)], -1).reshape(-1, 2)
_ss, _scr, _spa = RS.survey_condition(ROOT, road.s[-1])
_scr = gaussian_filter1d(_scr, 3); _spa = gaussian_filter1d(_spa, 3)
drng = np.random.default_rng(1234)

def decal(acc, s0, s1, d0, d1, cell, lift, rot=False):
    """quad grid on the road surface between stations s0..s1 (m) and offsets d0..d1; rot: sprite v runs across"""
    nr = max(2, int(math.ceil((s1 - s0) / 1.0)) + 1)
    dl = [d0, d1] if not (d0 < 0 < d1) else [d0, 0.0, d1]
    dl = np.unique(np.concatenate([np.linspace(dl[k], dl[k + 1], max(2, int(math.ceil((dl[k + 1] - dl[k]) / 0.7)) + 1)) for k in range(len(dl) - 1)]))
    S_ = np.linspace(s0, s1, nr); Dg, Sg = np.meshgrid(dl, S_)
    p = surf_pt(Sg.ravel(), Dg.ravel()); p[:, 2] += lift
    u0, v0, u1, v1 = cell
    fu = ((Dg - d0) / (d1 - d0)).ravel(); fv = ((Sg - s0) / (s1 - s0)).ravel()
    if rot:
        fu, fv = fv, fu
    uv = np.stack([u0 + (u1 - u0) * fu, v0 + (v1 - v0) * fv], -1)
    k = acc["n"]; acc["P"].append(p); acc["UV"].append(uv); acc["T"].append(grid_tris(nr, len(dl), k)); acc["n"] = k + len(p)

for c in range(nchunks):
    acc = dict(P=[], UV=[], T=[], n=0)
    sa, sb = c * CHUNK + 3, min((c + 1) * CHUNK, road.s[-1] - 3)
    if sb - sa < 5:
        continue
    for side in (1, -1):
        e_of = lambda i: road.el[i] if side > 0 else road.er[i]
        # longitudinal crack seal: centre joint, wheel paths, near the edge; chained 6 m sprites with gaps
        for dbase, rate in ((0.30, 1 / 22), (0.80, 1 / 40), (2.30, 1 / 45), (None, 1 / 35)):
            s = sa + drng.uniform(0, 20)
            while s < sb - 2:
                i = int(s); crk = np.interp(s, _ss, _scr)
                if drng.random() < min(0.9, rate * 25 * (0.5 + 1.5 * crk)):
                    L = drng.uniform(4, 30); dd = (dbase if dbase is not None else e_of(i) - 0.55) + drng.normal(0, 0.08)
                    t = s
                    while t < min(s + L, sb - 1):
                        ln = min(RS.SEAL_CELL_M[1], sb - t)
                        if ln < 1.0:
                            break
                        di = side * dd
                        decal(acc, t, t + ln, di - 0.3, di + 0.3, RS.CELLS["seal"][drng.integers(8)], RS.LIFT["seal"])
                        t += ln; dd += drng.normal(0, 0.05)
                    s += L + drng.uniform(3, 25)
                else:
                    s += drng.uniform(8, 30)
        # patches
        s = sa + drng.uniform(0, 30)
        while s < sb - 3:
            pat = np.interp(s, _ss, _spa); i = int(s)
            if drng.random() < 0.25 + 0.6 * pat:
                L = drng.uniform(1.5, 7.0); W = drng.uniform(0.9, min(2.8, e_of(i) - 0.4))
                d_in = drng.uniform(0.25, max(0.3, e_of(i) - 0.35 - W))
                cell = RS.CELLS["patch"][3 if drng.random() < 0.3 else drng.integers(3)]
                a, b = sorted((side * d_in, side * (d_in + W)))
                decal(acc, s, min(s + L, sb), a, b, cell, RS.LIFT["patch"])
                s += L
            s += drng.uniform(25, 90) / (1 + 1.5 * pat)
        # alligator / block cracking in the wheel paths
        s = sa + drng.uniform(0, 40)
        while s < sb - 3:
            crk = np.interp(s, _ss, _scr)
            if drng.random() < 0.15 + 0.7 * crk:
                L = drng.uniform(1.8, 4.0); dc = side * drng.choice([0.8, 1.6, 2.3])
                decal(acc, s, min(s + L, sb), dc - L / 2, dc + L / 2, RS.CELLS["network"][drng.integers(4)], RS.LIFT["network"])
            s += drng.uniform(15, 60) / (1 + 2 * crk)
    # transverse crack seal (thermal cracks), lane or full width
    s = sa + drng.uniform(0, 15)
    while s < sb - 1:
        crk = np.interp(s, _ss, _scr); i = int(s)
        if drng.random() < 0.4 + 0.5 * crk:
            fullw = drng.random() < 0.4; side = drng.choice([1, -1])
            a, b = (-road.er[i] + 0.3, road.el[i] - 0.3) if fullw else ((0.1, road.el[i] - 0.3) if side > 0 else (-road.er[i] + 0.3, -0.1))
            # sprite is 6 m long: map its length across the road, 0.6 m along
            for k0 in np.arange(a, b - 0.5, RS.SEAL_CELL_M[1]):
                decal(acc, s - 0.3, s + 0.3, k0, min(k0 + RS.SEAL_CELL_M[1], b), RS.CELLS["seal"][drng.integers(8)], RS.LIFT["seal"] + 0.001, rot=True)
        s += drng.uniform(6, 30) / (1 + crk)
    if acc["P"]:
        add("lines_road", f"decal_{c:02d}", "road_decal", np.concatenate(acc["P"]), np.concatenate(acc["T"]), np.concatenate(acc["UV"]), cast_shadows=False)

_tick("markings + decals")
# ------------------------------------------------------------------ kerbs (track layouts)
K = yaml.safe_load(open(ROOT / "config/kerbs.yaml"))
ks = gaussian_filter1d(road.k, K["sigma_m"])
KERBS = []
for dname, sgn_dir in (("nb", 1), ("sb", -1)):
    spans = {"left": [], "right": []}
    lab, nl = label(np.abs(ks) > K["curvature_threshold_1pm"])
    for l in range(1, nl + 1):
        idx = np.nonzero(lab == l)[0]
        if len(idx) < K["min_corner_length_m"]:
            continue
        apex = idx[np.argmax(np.abs(ks[idx]))]
        inner = "left" if ks[apex] > 0 else "right"
        outer = "right" if inner == "left" else "left"
        sa = road.s[apex]
        if sgn_dir > 0:   # northbound (s increasing): exit lies at larger s
            spans[inner].append((sa - K["apex_entry_m"], sa + K["inside_exit_m"]))
            spans[outer].append((sa + K["outside_exit_start_m"], sa + K["outside_exit_start_m"] + K["outside_exit_length_m"]))
        else:             # southbound: corner turns the other way for the driver, but the inside/outside sides are the same
            spans[inner].append((sa - K["inside_exit_m"], sa + K["apex_entry_m"]))
            spans[outer].append((sa - K["outside_exit_start_m"] - K["outside_exit_length_m"], sa - K["outside_exit_start_m"]))
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
            add(f"kerbs_{dname}", f"1KERB_{dname.upper()}{side[0].upper()}{len(KERBS):03d}", "kerb", P, grid_tris(len(i), len(prof_d)), UV, cast_shadows=False)
            KERBS.append(dict(s0=float(road.s[i[0]]), s1=float(road.s[i[-1]]), side=side, dir=dname))
print("kerbs", len(KERBS))

# ------------------------------------------------------------------ timing / spawn locators
def frame(s, d=0.0, lift=0.0):
    i = int(round(s))
    x, y = road.xy(i, d)
    return dict(p=(float(x), float(y), float(road.surface_z(i, d, noise=False) + lift)),
                fwd=(float(road.tx[i]), float(road.ty[i]), float(road.tz) if hasattr(road, "tz") else 0.0))

def frame_dir(s, d, lift, sgn):
    f = frame(s, d, lift); f["fwd"] = tuple(sgn * v for v in f["fwd"]); return f


def layout_locators(s_from, s_to):
    """spawn/pit before the start gate, gates across the road, all facing the run direction"""
    sgn = 1 if s_to > s_from else -1
    L = {}
    for nm, back in (("AC_START_0", CFG["spawn_back_m"]), ("AC_HOTLAP_START_0", CFG["spawn_back_m"]), ("AC_PIT_0", CFG["pit_back_m"])):
        L[nm] = frame_dir(float(np.clip(s_from - sgn * back, 1, N - 2)), 0.0, 0.0, sgn)
    for nm, s_ in (("AB_START", s_from), ("AB_FINISH", s_to)):
        i = int(round(s_))
        # L/R are relative to the driver: for a southbound run the driver's left is the road's right
        dl, dr = (road.el[i] + 1.0, -road.er[i] - 1.0) if sgn > 0 else (-road.er[i] - 1.0, road.el[i] + 1.0)
        L[f"AC_{nm}_L"] = frame_dir(s_, dl, 0.5, sgn)
        L[f"AC_{nm}_R"] = frame_dir(s_, dr, 0.5, sgn)
    return L


LAYLOC = {}
for seg, sd in CFG["segments"].items():
    LAYLOC[seg] = dict(s_from=CFG[sd["from"]], s_to=CFG[sd["to"]], name=sd["name"],
                       loc=layout_locators(CFG[sd["from"]], CFG[sd["to"]]))
loc = LAYLOC["summit"]["loc"]  # legacy

_tick("kerbs + locators")
BUILD.mkdir(exist_ok=True)
meta = dict(segments=LAYLOC, s_end=float(road.s[-1]), nchunks=nchunks, guardrails=GR, walls=WALLS, kerbs=KERBS, locators=loc,
            n_trees=int(sel.sum()))
pickle.dump(dict(groups=G, meta=meta), open(BUILD / "geometry.pkl", "wb"))

_tick("write geometry.pkl")
# stations for visual checks (every 250 m)
with open(ROOT / "reports/visual_check_stations.csv", "w") as f:
    w = csv.writer(f); w.writerow(["station_m", "lat", "lon", "heading_deg_true", "z_navd88"])
    for s in np.arange(0, CFG["build_end_m"], 250):
        i = int(s); hd = (math.degrees(math.atan2(road.tx[i], road.ty[i])) + 360) % 360  # grid north ~ true north (+/-0.7 deg)
        w.writerow([int(s), f"{full['lat'][i]:.7f}", f"{full['lon'][i]:.7f}", f"{hd:.0f}", f"{road.z[i]:.1f}"])
for g, ms in G.items():
    print(f"{g:12s} meshes={len(ms):4d} verts={sum(len(m['P']) for m in ms):8d} tris={sum(len(m['T']) for m in ms):8d}")
