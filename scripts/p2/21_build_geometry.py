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

# shoulder: edge -> edge+w/2 -> edge+w, w = per-station width (edge_profile.py: road.sh_w_l/r, 0.15-2.6 m),
# outer z pulled toward DTM but limited; mid column dips by road.ditch_l/r (shallow V ditch / gutter)
def shoulder(side):
    sgn = 1 if side == "L" else -1
    e = road.el if side == "L" else road.er
    W_ = road.sh_w_l if side == "L" else road.sh_w_r
    DZ_ = road.ditch_l if side == "L" else road.ditch_r
    col = -1 if side == "L" else 0
    ex, ey, ez = RX[:, col], RY[:, col], RZ[:, col]
    d_out = sgn * (e + W_)
    ox, oy = road.xy(np.arange(N), d_out)
    zt = dtm(ox, oy)
    zt = np.where(np.isfinite(zt), zt, ez - 0.05)
    dz = gaussian_filter1d(np.clip(zt - ez, -0.30, 0.35), 2, mode="nearest")
    oz = ez + dz
    mx, my = road.xy(np.arange(N), sgn * (e + W_ / 2))
    mz = ez + 0.5 * dz - np.minimum(0.02, 0.1 * W_) - DZ_   # slight dished edge / ditch V
    return np.stack([ex, mx, ox], 1), np.stack([ey, my, oy], 1), np.stack([ez, mz, oz], 1)

SLX, SLY, SLZ = shoulder("L")
SRX, SRY, SRZ = shoulder("R")
EDGE_MAT = {"litter": "litter", "dirt": "shoulder_dirt", "grass": "ground", "ivy": "bank", "paved": "road_asphalt"}

nchunks = chunk_of(N - 1) + 1
for c in range(nchunks):
    i0 = int(c * CHUNK); i1 = min(int((c + 1) * CHUNK), N - 1)
    rows = np.arange(i0, i1 + 1); nr = len(rows)
    P = np.stack([RX[rows], RY[rows], RZ[rows]], -1).reshape(-1, 3)
    import road_surface as _RS   # road UV = 12 m asphalt atlas: u = 0.5 + d/12, v = s/12
    UV = np.stack([0.5 + D_ROAD[rows] / _RS.ATLAS_M, np.repeat(road.s[rows, None] / _RS.ATLAS_M, 13, 1)], -1).reshape(-1, 2)
    # freshly repaved spans (config/surface_overrides.yaml) -> separate 1ROADF_NN mesh / road_asphalt_fresh. The joint is
    # the station row at the span end (straight transverse); both meshes share that row's vertex positions, so no step.
    Tg_ = grid_tris(nr, 13); fq = road.fresh[rows][Tg_[:, 0] // 13]
    for nm_, mat_, Tm in ((f"1ROAD_{c:02d}", "road_asphalt", Tg_[~fq]), (f"1ROADF_{c:02d}", "road_asphalt_fresh", Tg_[fq])):
        if len(Tm):
            used, inv = np.unique(Tm, return_inverse=True)
            add("road", nm_, mat_, P[used], inv.reshape(-1, 3), UV[used])
    for side, (X, Y, Z) in (("L", (SLX, SLY, SLZ)), ("R", (SRX[:, ::-1], SRY[:, ::-1], SRZ[:, ::-1]))):
        # columns must increase to the left: L = edge->out, R = out->edge. One mesh per edge material
        # (1GRASS_sh{side}{mat}NN; paved aprons are 1ROADAP_{side}NN), quads take the material of their first row.
        P = np.stack([X[rows], Y[rows], Z[rows]], -1).reshape(-1, 3)
        Wc = (road.sh_w_l if side == "L" else road.sh_w_r)[rows]
        ucol = np.array([0, 0.5, 1.0]) if side == "L" else np.array([1.0, 0.5, 0])
        UV = np.stack([(ucol[None, :] * Wc[:, None]).ravel() / 2.0, np.repeat(road.s[rows], 3) / 2.0], -1)
        Tg_ = grid_tris(nr, 3); mrow = (road.sh_mat_l if side == "L" else road.sh_mat_r)[rows][Tg_[:, 0] // 3]
        for mname in EDGE_MAT:
            Tm = Tg_[mrow == mname]
            if len(Tm):
                used, inv = np.unique(Tm, return_inverse=True)
                nm_ = f"1ROADAP_{side}{c:02d}" if mname == "paved" else f"1GRASS_sh{side}{mname}{c:02d}"
                add("road", nm_, EDGE_MAT[mname], P[used], inv.reshape(-1, 3), UV[used])

# ragged asphalt edge: non-physical alpha-tested strip over the pavement edge (edge_<mat> rows of road_edge.png, built
# by edge_tex.py in 22). Columns: road.strip_in inside the edge (debris / wash-out on the asphalt) -> asphalt edge ->
# shoulder mid column (crumbled edge, missing chunks). v of the edge column is fixed at EDGE_V so the ragged mask lines up.
EDGE_ROW = {"litter": 0, "dirt": 1, "grass": 2, "ivy": 3}
EDGE_V = 0.55
for c in range(nchunks):
    i0 = int(c * CHUNK); i1 = min(int((c + 1) * CHUNK), N - 1)
    for side, sgn, X, Y, Z in (("L", 1, SLX, SLY, SLZ), ("R", -1, SRX, SRY, SRZ)):
        mat_ = road.sh_mat_l if side == "L" else road.sh_mat_r
        si_ = road.strip_in_l if side == "L" else road.strip_in_r
        e_ = road.el if side == "L" else road.er
        acc = dict(P=[], UV=[], T=[], n=0)
        rows_all = np.arange(i0, i1 + 1)
        brk = np.nonzero((mat_[rows_all[1:]] != mat_[rows_all[:-1]]) | (road.fresh[rows_all[1:]] != road.fresh[rows_all[:-1]]))[0] + 1
        for run in np.split(rows_all, brk):
            mname = mat_[run[0]]
            if mname not in EDGE_ROW or road.fresh[run[0]]:   # fresh pavement: crisp edge, no ragged strip
                continue
            run = np.r_[run, run[-1] + 1] if run[-1] + 1 <= i1 else run      # close the gap to the next run
            if len(run) < 2:
                continue
            di = sgn * (e_[run] - si_[run])
            xi, yi = road.xy(run, di); zi = road.surface_z(run, di) + 0.012
            cols = [(xi, yi, zi), (X[run, 0], Y[run, 0], Z[run, 0] + 0.012), (X[run, 1], Y[run, 1], Z[run, 1] + 0.010)]
            if side == "R":
                cols = cols[::-1]
            Pm = np.stack([np.stack(cc_, -1) for cc_ in cols], 1).reshape(-1, 3)
            r0 = EDGE_ROW[mname] / 4.0
            vv = np.array([0.02, EDGE_V, 0.98]) if side == "L" else np.array([0.98, EDGE_V, 0.02])
            UVm = np.stack([np.repeat(road.s[run] / 4.0, 3), np.tile(r0 + vv / 4.0, len(run))], -1)
            k_ = acc["n"]; acc["P"].append(Pm); acc["UV"].append(UVm); acc["T"].append(grid_tris(len(run), 3, k_)); acc["n"] = k_ + len(Pm)
        if acc["P"]:
            add("road", f"edge_{side}{c:02d}", "road_edge", np.concatenate(acc["P"]), np.concatenate(acc["T"]), np.concatenate(acc["UV"]), cast_shadows=False)

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
_sd_b = (pts[:, 0] - road.x[jc]) * road.nx[jc] + (pts[:, 1] - road.y[jc]) * road.ny[jc]
hw_b = road.outer(jc, np.where(_sd_b >= 0, 1, -1)) + 0.5     # per-side, per-station outer edge (variable shoulder)
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
# materials / UVs (macro colour layer, 22 terrain_macro()): grass / bank / litter use one top-down UV over a square
# per-chunk window (unique macro map in txDiffuse, tiled class detail in txDetail via detailUVMultiplier);
# material per (class, chunk) = "ter_<class>_<cc>". Rock keeps a slope-following projection, 2-scale tiled (22).
TERRAIN_MACRO = {}
for c in range(nchunks):
    Tc = TT[(tch == c) & (kind_t != "r")]
    if len(Tc):
        pc = P3[np.unique(Tc)]
        lo, hi = pc[:, :2].min(0) - 2.0, pc[:, :2].max(0) + 2.0
        TERRAIN_MACRO[c] = dict(x0=float(lo[0]), y0=float(lo[1]), size=float((hi - lo).max()))
    for kind in "grbl":
        T = TT[(tch == c) & (kind_t == kind)]
        if not len(T):
            continue
        used, inv = np.unique(T, return_inverse=True)
        P = P3[used]
        if kind == "r":
            UV = np.c_[(P[:, 0] + P[:, 1]) / 20.0, P[:, 2] / 20.0]   # 20 m base tile, detail x4.3 in 22
            add("terrain", f"1GRASS_t{kind}{c:02d}", "rock", P, inv.reshape(-1, 3), UV)
        else:
            mc = TERRAIN_MACRO[c]
            UV = np.c_[(P[:, 0] - mc["x0"]) / mc["size"], (P[:, 1] - mc["y0"]) / mc["size"]]
            add("terrain", f"1GRASS_t{kind}{c:02d}", f"ter_{TMAT[kind]}_{c:02d}", P, inv.reshape(-1, 3), UV,
                macro=dict(chunk=c, cls=TMAT[kind], **mc))
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
# ------------------------------------------------------------------ trees: procedural 3D (scripts/p2/trees3d.py)
# 4 prototypes per style (oak / bay / pale sycamore-alder / tall euc), instanced at the LiDAR tree positions, scaled to
# LiDAR height + crown radius, random yaw, slight lean (near-road oaks lean over the road). Merged per 250 m chunk per
# material (tree_trunk bark, tree_leaves clusters) and per LOD; split <= 60k vertices per mesh.
import trees3d as T3
tr = list(csv.DictReader(open(DER / "trees.csv")))
tx_, ty_, tz_, th_, tcr_, ts_ = (np.array([float(t[k]) for t in tr]) for k in ("x_local", "y_local", "ground_z", "height_m", "crown_radius_m", "s_m"))
d_tc, j_tc = bt.query(np.c_[tx_, ty_])
hw_t = np.maximum(road.el, road.er)[j_tc]
_sd_t = np.where((tx_ - road.x[j_tc]) * road.nx[j_tc] + (ty_ - road.y[j_tc]) * road.ny[j_tc] >= 0, 1, -1)
sw_t = road.sh_w(j_tc, _sd_t)                                  # per-station shoulder width on the tree's side
sel = (ts_ <= S_END + 60) & (d_tc > hw_t + sw_t + 1.3)  # trunk clear of road + shoulder + 0.5 m margin
# road clearance envelope: nothing of a tree below CLEAR_H above the road inside road + shoulder + 0.5 m
CLEAR_H, CLEAR_M = 5.5, 0.5
def env_violation(P):
    """bool per vertex: inside the road clearance envelope"""
    d, j = bt.query(P[:, :2])
    side = (P[:, 0] - road.x[j]) * road.nx[j] + (P[:, 1] - road.y[j]) * road.ny[j]
    lim = road.outer(j, np.where(side >= 0, 1, -1)) + CLEAR_M
    return (np.abs(side) < lim) & (d < lim + 1.0) & (P[:, 2] < road.z[j] + CLEAR_H), road.z[j] + CLEAR_H
def clear_instance(inst):
    """leaf quads touching the envelope are pruned; bark vertices in it are lifted to the clearance height
    (limbs arch up over the road like the real canopy tunnel)"""
    P, N, UV, T = inst["bark"]
    bad, zc = env_violation(P)
    if bad.any():
        P = P.copy(); P[bad, 2] = zc[bad] + 0.05; inst["bark"] = (P, N, UV, T)
    P, N, UV, T = inst["leaf"]
    if len(P):
        bad, _ = env_violation(P)
        if bad.any():
            keep = ~bad.reshape(-1, 4).any(1); nq = int(keep.sum())
            P = P.reshape(-1, 4, 3)[keep].reshape(-1, 3); N = N.reshape(-1, 4, 3)[keep].reshape(-1, 3)
            UV = UV.reshape(-1, 4, 2)[keep].reshape(-1, 2)
            T = (np.array([(0, 1, 2), (0, 2, 3), (0, 2, 1), (0, 3, 2)])[None] + 4 * np.arange(nq)[:, None, None]).reshape(-1, 3)
            inst["leaf"] = (P, N, UV, T)
    return inst
print("trees used", sel.sum(), "of", len(tr))
ratio = tcr_ / np.maximum(th_, 1)
species = np.where((th_ > 24) & (ratio < 0.3), "euc", np.where(ratio > 0.42, "oak", "bay"))
# pale sycamore/alder: the creek-bottom trees (lowest ground within ~300 m along the road) and half of the tall narrow ones
_zl = np.array([np.percentile(tz_[np.abs(ts_ - s) < 300], 25) if np.any(np.abs(ts_ - s) < 300) else -1e9 for s in ts_])
_h = (np.arange(len(tr)) * 2654435761) % 1000 / 1000.0  # stable per-tree hash
species = np.where(((species == "euc") & (_h < 0.5)) | ((species != "oak") & (tz_ < _zl) & (_h < 0.6)), "syc", species)
TREE_NEAR = 18.0   # m past the road edge: full prototype (twigs, all clusters); beyond: simpler LOD prototype
PROTOS = {(sp, v, lod): T3.make_proto(sp, v, lod) for sp in T3.SPECIES for v in range(T3.N_PROTO) for lod in (0, 1)}
trng = np.random.default_rng(21)
tree_acc = {}  # (chunk, lod, material) -> lists
for k in np.nonzero(sel)[0]:
    c = min(int(road.s[j_tc[k]] // CHUNK), nchunks - 1)
    sp = str(species[k]); lod = 0 if d_tc[k] - hw_t[k] < TREE_NEAR else 1
    pr = PROTOS[(sp, int(_h[k] * 997) % T3.N_PROTO, lod)]
    jr = j_tc[k]; to_road = np.array([road.x[jr] - tx_[k], road.y[jr] - ty_[k]]); to_road /= np.linalg.norm(to_road) + 1e-9
    if sp == "oak" and lod == 0:
        # roadside oaks reach out over the road, but only as far as keeps the trunk out of the envelope below 5.5 m
        lean_dir, lean = to_road, min(trng.uniform(0.05, 0.22), max(0.0, (d_tc[k] - hw_t[k] - sw_t[k] - CLEAR_M - 0.8) / CLEAR_H))
    else:
        a = trng.uniform(0, 2 * np.pi); lean_dir, lean = np.array([math.cos(a), math.sin(a)]), trng.uniform(0, 0.06)
    inst = clear_instance(T3.instance(pr, tx_[k], ty_[k], tz_[k], th_[k], tcr_[k], sp, trng.uniform(0, 2 * np.pi), lean_dir, lean))
    for key, mat in (("bark", "tree_trunk"), ("leaf", "tree_leaves")):
        tree_acc.setdefault((c, lod, mat), []).append(inst[key])
_tv = {}
for (c, lod, mat), parts in sorted(tree_acc.items()):
    n = 0; buf = []
    def _flush(buf, n):
        P = np.concatenate([p[0] for p in buf]); Nn = np.concatenate([p[1] for p in buf]); UV = np.concatenate([p[2] for p in buf])
        off = np.cumsum([0] + [len(p[0]) for p in buf[:-1]]); T = np.concatenate([p[3] + o for p, o in zip(buf, off)])
        add("trees", f"{mat}_{c:02d}_{lod}_{n}", mat, P, T, UV, N_=Nn, cast_shadows=True, transparent=False,
            lod_out=600.0 if lod == 0 else 1100.0)
        _tv[mat] = _tv.get(mat, 0) + len(P)
    cnt = 0
    for p in parts:
        if cnt + len(p[0]) > 60000:
            _flush(buf, n); n += 1; buf = []; cnt = 0
        buf.append(p); cnt += len(p[0])
    if buf: _flush(buf, n)
print("3D trees: vertices", _tv, "total", sum(_tv.values()), "| near", int(((d_tc - hw_t) < TREE_NEAR)[sel].sum()), "far", int(((d_tc - hw_t) >= TREE_NEAR)[sel].sum()),
      "| species", {s: int((species[sel] == s).sum()) for s in T3.SPECIES})
_tick("trees")
# ------------------------------------------------------------------ understory shrubs (GoPro: dense green brush on every
# bank and drop-off between the trunks; LiDAR trees >= 4 m miss it). Non-physical 3D clumps of leaf quads, 1 mesh per chunk.
srng = np.random.default_rng(11)
SH_P, SH_T, SH_UV, SH_C, SH_N = [], [], [], [], []
_rx, _ry, _rnx, _rny = road.x.tolist(), road.y.tolist(), road.nx.tolist(), road.ny.tolist()  # scalar fast path
for i in bar(range(0, N, 2), "shrubs"):
    for side in (-1, 1):
        for _ in range(2):
            off = 2.2 + 23.8 * srng.uniform() ** 1.8  # denser right at the road edge / on the banks
            hw_i = road.el[i] if side > 0 else road.er[i]
            d = side * (hw_i + road.sh_w(i, side) + off)
            x, y = _rx[i] + _rnx[i] * d, _ry[i] + _rny[i] * d
            x, y = x + srng.normal(0, 0.6), y + srng.normal(0, 0.6)
            sl = _slope.sample1(x, y)
            if np.isfinite(sl) and sl > 1.15:  # bare rock faces: no brush
                continue
            ch = _chm_r.sample1(x, y)
            if off > 8 and not (np.isfinite(ch) and ch > 3) and srng.uniform() < 0.6:  # thinner in the open
                continue
            if off > 8 and srng.uniform() < 0.25:  # far brush thinned (pays for the denser roadside clumps)
                continue
            z = dtm05.sample1(x, y)
            if not math.isfinite(z):
                z = dtm1.sample1(x, y)
            if not np.isfinite(z):
                continue
            h = srng.uniform(0.9, 2.6) * (1.3 if off > 6 else 1.0); w = h * srng.uniform(1.2, 1.9)
            u0 = 0.5 * srng.integers(0, 2)  # two shrub styles on one atlas
            # 3D clump of 4-7 leaf quads at varying heights (fern / poison oak low, toyon / coyote brush tall), spherical normals
            cP, cN, cUV, cT = T3.shrub_clump(x, y, z - 0.25, h, w, u0, srng, 6 if off < 4 else 5 if off < 8 else 3)
            _keep = ~env_violation(np.array(cP))[0].reshape(-1, 4).any(1)  # shrub quads never inside the road envelope
            cP = [q for qi, q in enumerate(cP) if _keep[qi // 4]]; cN = [q for qi, q in enumerate(cN) if _keep[qi // 4]]
            cUV = [q for qi, q in enumerate(cUV) if _keep[qi // 4]]
            cT = [tt for tt in cT[:4 * int(_keep.sum())]]
            b = len(SH_P)
            SH_P += cP; SH_N += cN; SH_UV += cUV; SH_T += [(t0 + b, t1 + b, t2 + b) for (t0, t1, t2) in cT]
            SH_C += [chunk_of(i)] * len(cP)
SH_P = np.array(SH_P); SH_T = np.array(SH_T); SH_UV = np.array(SH_UV); SH_C = np.array(SH_C); SH_N = np.array(SH_N)
for c in range(nchunks):
    m = SH_C[SH_T[:, 0]] == c
    if not m.any():
        continue
    used, inv = np.unique(SH_T[m], return_inverse=True)
    P = SH_P[used]; T = inv.reshape(-1, 3)
    Nn = SH_N[used]
    add("trees", f"shrub_{c:02d}", "shrub", P, T, SH_UV[used], N_=Nn, cast_shadows=False, lod_out=250.0)
print("shrub clumps", len(SH_T) // 16, "vertices", len(SH_P))

# build check: no tree / shrub vertex inside the road clearance envelope
_nviol = sum(int(env_violation(m["P"])[0].sum()) for m in G["trees"] if m["material"] in ("tree_trunk", "tree_leaves", "shrub"))
print("road clearance envelope (5.5 m over road + shoulder + 0.5 m): violating tree/shrub vertices =", _nviol)
assert _nviol == 0, f"{_nviol} tree/shrub vertices inside the road clearance envelope"
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
    add("barriers", f"1WALL_gr{gi:03d}", "invisible", P, T, UV, renderable=False, cast_shadows=False)
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
        x, y = road.xy(idx, sgn * (e + road.sh_w(idx, sgn) + 1.2))
        zb = road.surface_z(idx, sgn * e, noise=False)
        P = np.concatenate([np.c_[x, y, zb - 1.0], np.c_[x, y, zb + 4.0]])
        n = len(idx); a = np.arange(n - 1)
        T1 = np.c_[a, a + 1, a + n + 1]; T2 = np.c_[a, a + n + 1, a + n]
        T = np.concatenate([T1, T2, T1[:, [0, 2, 1]], T2[:, [0, 2, 1]]])
        add("barriers", f"1WALL_cut{len(WALLS):03d}", "invisible", P, T, np.zeros((len(P), 2)), renderable=False, cast_shadows=False)
        WALLS.append(dict(s0=float(road.s[idx[0]]), s1=float(road.s[idx[-1]]), side=side))

# end walls at s=0 (behind the pit spot) and at the end of the built road
for nm, i in (("start", 1), ("end", N - 2)):
    d = np.linspace(-road.outer(i, -1) - 1, road.outer(i, 1) + 1, 8)
    x, y = road.xy(np.full(8, i), d); zb = road.z[i]
    P = np.concatenate([np.c_[x, y, np.full(8, zb - 2)], np.c_[x, y, np.full(8, zb + 4)]])
    a = np.arange(7); T1 = np.c_[a, a + 1, a + 9]; T2 = np.c_[a, a + 9, a + 8]
    T = np.concatenate([T1, T2, T1[:, [0, 2, 1]], T2[:, [0, 2, 1]]])
    add("barriers", f"1WALL_{nm}", "invisible", P, T, np.zeros((16, 2)), renderable=False, cast_shadows=False)

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
    for fz, sfx in ((False, ""), (True, "f")):   # fresh pavement: crisp unworn paint (paint_*_fresh)
        icf = ic[road.fresh[ic] == fz]
        if fz and len(icf):                      # include the old row before each fresh run: no 1 m gap at the joint
            icf = np.intersect1d(np.union1d(icf, icf - 1), ic)
        mf = "_fresh" if fz else ""
        ribbon("lines_road", f"line_edgeL{sfx}_{c:02d}", "paint_white" + mf, icf[ledge[icf]], road.el - road.line_in_l, EW)   # inset 0.15-0.8 m (edge_profile.py)
        ribbon("lines_road", f"line_edgeR{sfx}_{c:02d}", "paint_white" + mf, icf[redge[icf]], -road.er + road.line_in_r, EW)
        dy = icf[ctype[icf] == "double_yellow"]
        ribbon("lines_road", f"line_cdy{sfx}_{c:02d}", "paint_yellow_dbl" + mf, dy, 0.0, RS.DY_RIBBON, cols=(0.0, 0.5, 1.0))
        br = icf[np.isin(ctype[icf], ["broken_yellow", "broken"])]
        ribbon("lines_road", f"line_cbr{sfx}_{c:02d}", "paint_yellow" + mf, br, 0.0, EW, dash=RS.BROKEN, cols=(0.0, 0.5, 1.0))

# raised pavement markers: yellow, every 24 m, one each side of the double yellow (GoPro frames 166-176;
# survey B section of the road, i.e. the northern half). Thin low frustums, non-physical, no shadows.
_svs, _, _ = RS.survey_condition(ROOT, road.s[-1])
_n_a = 110  # frames 1-110 = survey A, 111-219 = survey B
RPM_S0 = float(_svs[min(_n_a, len(_svs) - 1)]) if len(_svs) > _n_a else 0.0
_SSF = DER / "surface_spans.json"      # written by 21c_roadside_assets.py from the roadside survey (surface entries)
SURF_SPANS = json.load(open(_SSF)) if _SSF.exists() else []
RPM_SPANS = [(sp["s0"], sp["s1"]) for sp in SURF_SPANS if sp["kind"] == "rpm_present"]
import edge_profile as _EP
RPM_SPANS += _EP.fresh_spans()              # freshly repaved: new yellow RPMs throughout
SURF_SPANS = SURF_SPANS + [dict(kind="fresh_overlay", s0=a, s1=b) for a, b in _EP.fresh_spans()]
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
    for s in np.arange(math.ceil(max(c * CHUNK, 3) / RS.RPM_SPACING) * RS.RPM_SPACING, min((c + 1) * CHUNK, road.s[-1] - 3), RS.RPM_SPACING):
        i = int(s)
        if s < RPM_S0 and not any(a <= s <= b for a, b in RPM_SPANS):   # survey heuristic (north half) + observed rpm_present spans (21c)
            continue
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
# (1ROAD_/1ROADF_ UVs are already set in this atlas mapping when the road meshes are built)
_ss, _scr, _spa = RS.survey_condition(ROOT, road.s[-1])
_scr = gaussian_filter1d(_scr, 3); _spa = gaussian_filter1d(_spa, 3)
if SURF_SPANS:   # observed condition spans override the evenly-spaced survey heuristic (densify to 5 m so short spans register)
    _sd = np.arange(0, road.s[-1] + 5, 5.0); _scr = np.interp(_sd, _ss, _scr); _spa = np.interp(_sd, _ss, _spa); _ss = _sd
    for _sp in SURF_SPANS:
        _m = (_ss >= _sp["s0"] - 2.5) & (_ss <= _sp["s1"] + 2.5)
        if _sp["kind"] == "cracked_old":
            _scr[_m] = np.maximum(_scr[_m], 0.9)
        elif _sp["kind"] == "crack_sealed":
            _scr[_m] = np.clip(_scr[_m], 0.35, 0.55)
        elif _sp["kind"] == "patched":
            _spa[_m] = np.maximum(_spa[_m], 0.85)
        elif _sp["kind"] == "fresh_overlay":
            _scr[_m] = 0.0; _spa[_m] = 0.0
drng = np.random.default_rng(1234)

def decal(acc, s0, s1, d0, d1, cell, lift, rot=False):
    """quad grid on the road surface between stations s0..s1 (m) and offsets d0..d1; rot: sprite v runs across"""
    if road.fresh[int(np.clip(s0, 0, N - 1)):int(np.clip(s1, 0, N - 1)) + 1].any():
        return                                   # freshly repaved: no crack-seal / patch / crack decals
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
# apex / exit kerbs + painted run-off: scripts/p2/kerbs.py, parameters in config/kerbs.yaml
import kerbs as KB
K = yaml.safe_load(open(ROOT / "config/kerbs.yaml"))
KERBS, _corners = KB.build(road, K, RX, RY, RZ, {"left": (SLX, SLY, SLZ), "right": (SRX, SRY, SRZ)}, P3, TT, GR, add, chunk_of, {"left": road.sh_w_l, "right": road.sh_w_r})
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
