"""Phase 2: light hill-climb dressing for the track_* layouts -> group "dressing" in build/geometry.pkl

Gates (s 20 / 7392 / 15563): banner gantry over the road (>= 5.5 m clear), timing hut on the flattest
spot beside the gate, flag poles; a small bleacher near the summit gate if the DTM has a flat spot.
Corners: short runs of water-barrier blocks / tyre stacks in front of the worst steep drops
(features.geojson steep_drop ranked by curvature). Everything is in ONE textured mesh (atlas
build/dressing_atlas.png, material "dressing") plus ONE invisible collision mesh 1WALL_dressing.
Idempotent: replaces any existing "dressing" group. Run after 21_build_geometry.py.
"""
import json, pickle, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from geom import ROOT, DER, BUILD, SHOULDER_W, Road, Raster, vertex_normals

geo = pickle.load(open(BUILD / "geometry.pkl", "rb"))
G, meta = geo["groups"], geo["meta"]
G.pop("dressing", None)
road = Road(meta["s_end"])
N = len(road.s)
dtm05 = Raster(DER / "dtm_r050_b40.tif"); dtm1 = Raster(DER / "dtm_r100_b150.tif")
ctree = cKDTree(np.c_[road.x, road.y])
GATES = [20, 7392, 15563]


def dtm(x, y):
    z = dtm05.sample(x, y); z2 = dtm1.sample(x, y)
    return np.where(np.isfinite(z), z, z2)


# ------------------------------------------------------------------ atlas (4 x 4 tiles of 256 px, row 0 = banner)
A = Image.new("RGB", (1024, 1024), (120, 120, 120)); dr = ImageDraw.Draw(A)
def tile(c, r): return (c * 256, r * 256, c * 256 + 255, r * 256 + 255)
# banner: dark blue with checker ends and text
dr.rectangle((0, 0, 1023, 255), fill=(20, 40, 90))
for i in range(4):
    for j in range(8):
        col = (240, 240, 240) if (i + j) % 2 else (15, 15, 15)
        for x0 in (i * 32, 896 + i * 32):
            dr.rectangle((x0, j * 32, x0 + 31, j * 32 + 31), fill=col)
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 96)
except OSError:
    font = ImageFont.load_default()
dr.text((512, 128), "PALOMARES", fill=(255, 255, 255), font=font, anchor="mm")
UVT = {}
def reg(name, c, r, fill=None, fn=None):
    if fill: dr.rectangle(tile(c, r), fill=fill)
    if fn: fn(*tile(c, r))
    UVT[name] = ((c * 256 + 4) / 1024, (r * 256 + 4) / 1024, (c * 256 + 252) / 1024, (r * 256 + 252) / 1024)
UVT["banner"] = (2 / 1024, 2 / 1024, 1022 / 1024, 254 / 1024)
reg("metal", 0, 1, (150, 152, 156))
def stripes(x0, y0, x1, y1):
    for k in range(8): dr.rectangle((x0 + k * 32, y0, x0 + k * 32 + 31, y1), fill=(200, 30, 30) if k % 2 else (235, 235, 235))
reg("tent", 1, 1, fn=stripes)
def wood(x0, y0, x1, y1):
    for k in range(8): dr.rectangle((x0, y0 + k * 32, x1, y0 + k * 32 + 29), fill=(150 + 8 * (k % 3), 110, 70)); dr.line((x0, y0 + k * 32 + 30, x1, y0 + k * 32 + 30), fill=(70, 50, 30), width=2)
reg("wood", 2, 1, fn=wood)
def tyres(x0, y0, x1, y1):
    dr.rectangle((x0, y0, x1, y1), fill=(22, 22, 22))
    for k in range(4): dr.line((x0, y0 + k * 64 + 62, x1, y0 + k * 64 + 62), fill=(5, 5, 5), width=4)
    for k in range(4): dr.rectangle((x0, y0 + k * 64 + 26, x1, y0 + k * 64 + 36), fill=(225, 225, 225) if k % 2 else (22, 22, 22))
reg("tyre", 3, 1, fn=tyres)
reg("block_red", 0, 2, (205, 35, 30)); reg("block_white", 1, 2, (235, 235, 230))
reg("orange", 2, 2, (240, 120, 20)); reg("canvas", 3, 2, (230, 230, 225))
def flag(x0, y0, x1, y1):
    for i in range(8):
        for j in range(8): dr.rectangle((x0 + i * 32, y0 + j * 32, x0 + i * 32 + 31, y0 + j * 32 + 31), fill=(15, 15, 15) if (i + j) % 2 else (240, 240, 240))
reg("flag", 0, 3, fn=flag)
A.transpose(Image.FLIP_TOP_BOTTOM).save(BUILD / "dressing_atlas.png")   # kn5 v runs bottom-up; UVT uses top-down

# ------------------------------------------------------------------ geometry helpers
VP, VT, VU = [], [], []      # visual
CP, CT = [], []              # collision


def quad(P4, uvr, flip_v=False):
    u0, v0, u1, v1 = uvr
    if flip_v: v0, v1 = v1, v0
    o = sum(len(p) for p in VP)
    VP.append(np.asarray(P4, float)); VU.append(np.array([[u0, v1], [u1, v1], [u1, v0], [u0, v0]]))
    VT.append(np.array([[o, o + 1, o + 2], [o, o + 2, o + 3]]))


def box(c, fwd, L, W, z0, z1, uvr, coll=False, top=None):
    """box centred at xy c, long axis fwd (unit xy), length L (along fwd), width W, from z0 to z1.
    faces CCW outward. 'top' tile overrides the top/bottom tile."""
    f = np.array([fwd[0], fwd[1], 0.0]) / np.hypot(*fwd); l = np.array([-f[1], f[0], 0.0])
    c = np.array([c[0], c[1], 0.0])
    def p(a, b, z): return c + f * a * L / 2 + l * b * W / 2 + np.array([0, 0, z])
    B = [p(-1, -1, z0), p(1, -1, z0), p(1, 1, z0), p(-1, 1, z0)]
    U = [p(-1, -1, z1), p(1, -1, z1), p(1, 1, z1), p(-1, 1, z1)]
    faces = [[B[0], B[1], U[1], U[0]], [B[1], B[2], U[2], U[1]], [B[2], B[3], U[3], U[2]], [B[3], B[0], U[0], U[3]]]
    for q in faces: quad(q, uvr)
    quad([U[0], U[1], U[2], U[3]], top or uvr); quad([B[3], B[2], B[1], B[0]], top or uvr)
    if coll:
        o = sum(len(q) for q in CP); CP.append(np.array(B + U))
        T = np.array([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7], [4, 5, 6], [4, 6, 7]]) + o
        CT.append(T)


def strip_wall(x, y, z0, z1):
    """double-sided invisible collision strip along a polyline"""
    n = len(x); o = sum(len(q) for q in CP)
    CP.append(np.concatenate([np.c_[x, y, z0], np.c_[x, y, z1]]))
    a = np.arange(n - 1); T1 = np.c_[a, a + 1, a + n + 1]; T2 = np.c_[a, a + n + 1, a + n]
    CT.append(np.concatenate([T1, T2, T1[:, [0, 2, 1]], T2[:, [0, 2, 1]]]) + o)


def idx(s): return int(np.clip(np.searchsorted(road.s, s), 0, N - 1))
def tangent(i): return np.array([road.tx[i], road.ty[i]])
def normal(i): return np.array([road.nx[i], road.ny[i]])


def clear_of_road(pts, margin):
    """all points at least margin + local half width from ANY part of the centerline"""
    d, j = ctree.query(pts)
    half = np.maximum(road.el[j], road.er[j]) + SHOULDER_W
    return bool(np.all(d >= half + margin))


def footprint_ok(cx, cy, fwd, L, W, margin=1.5, maxrange=0.8):
    f = fwd / np.linalg.norm(fwd); l = np.array([-f[1], f[0]])
    g = np.array([[a, b] for a in np.linspace(-.5, .5, 5) for b in np.linspace(-.5, .5, 4)])
    pts = np.array([cx, cy]) + g[:, :1] * L * f + g[:, 1:] * W * l
    z = dtm(pts[:, 0], pts[:, 1])
    if not np.all(np.isfinite(z)) or not clear_of_road(pts, margin):
        return None
    return (z.max() - z.min(), z.min(), z.max()) if z.max() - z.min() <= maxrange else None


LOG = []
# ------------------------------------------------------------------ gates
for sg in GATES:
    i = idx(sg); t = tangent(i); n = normal(i)
    dl = road.el[i] + SHOULDER_W + 1.0; drr = road.er[i] + SHOULDER_W + 1.0
    zroad = max(road.surface_z(i, d, noise=False) for d in np.linspace(-road.er[i], road.el[i], 9))
    zb = zroad + 5.6                                        # beam underside: >= 5.5 m clear of the crown
    legs = []
    for d in (dl, -drr):
        x, y = road.xy(i, d); zg = float(dtm(x, y)) if np.isfinite(dtm(x, y)) else road.z[i]
        zg = min(zg, road.surface_z(i, d, noise=False))
        box((x, y), t, 0.5, 0.5, zg - 0.4, zb + 1.4, UVT["metal"], coll=True)
        legs.append((x, y))
    mid = np.mean(legs, 0); span = dl + drr + 0.5
    box(mid, n, span, 0.4, zb, zb + 0.25, UVT["metal"])                     # lower beam
    box(mid, n, span, 0.4, zb + 1.15, zb + 1.4, UVT["metal"])               # upper beam
    # banner panels both faces (slightly in front/behind the truss)
    hw = (dl + drr) / 2 - 0.3; lv = n / np.linalg.norm(n)
    for sgn in (1, -1):
        c = mid + sgn * t * 0.22
        a = c - lv * hw * sgn; b = c + lv * hw * sgn   # left-to-right as seen by the viewer facing it
        quad([[*a, zb + 0.25], [*b, zb + 0.25], [*b, zb + 1.15], [*a, zb + 1.15]], UVT["banner"])
    LOG.append(f"gantry s={sg} span {span:.1f} m clear {zb - zroad:.1f} m")
    # timing hut: flattest 3 x 2.4 m spot within s-30..s+10, either side, 2..10 m past the shoulder
    best = None
    for ds in range(-30, 11, 2):
        j = idx(sg + ds)
        for side in (1, -1):
            for off in np.arange(2.5, 10.1, 0.5):
                d = side * ((road.el[j] if side > 0 else road.er[j]) + SHOULDER_W + off)
                x, y = road.xy(j, d)
                r = footprint_ok(x, y, tangent(j), 3.0, 2.4)
                if r and abs(ds) > 3:
                    sc = r[0] + 0.05 * off + 0.01 * abs(ds)
                    if best is None or sc < best[0]:
                        best = (sc, x, y, j, side, ds, off, r)
    if best:
        _, x, y, j, side, ds, off, r = best
        box((x, y), tangent(j), 3.0, 2.4, r[1] - 0.3, r[2] + 2.3, UVT["canvas"], coll=True, top=UVT["metal"])
        box((x, y), tangent(j), 3.4, 2.8, r[2] + 2.3, r[2] + 2.5, UVT["tent"])
        LOG.append(f"  timing hut s={sg + ds} {'L' if side > 0 else 'R'} {off:.1f} m past shoulder")
    # two checker flag poles just outside the gantry legs
    for (x, y), d in zip(legs, (1, -1)):
        px, py = np.array([x, y]) + d * n * 0.9 - t * 1.0
        zg = dtm(px, py); zg = float(zg) if np.isfinite(zg) else road.z[i]
        box((px, py), t, 0.08, 0.08, zg - 0.3, zg + 4.0, UVT["metal"])
        quad([[px, py, zg + 3.0], [px - t[0] * 1.2, py - t[1] * 1.2, zg + 3.0],
              [px - t[0] * 1.2, py - t[1] * 1.2, zg + 3.9], [px, py, zg + 3.9]], UVT["flag"])
        quad([[px - t[0] * 1.2, py - t[1] * 1.2, zg + 3.0], [px, py, zg + 3.0],
              [px, py, zg + 3.9], [px - t[0] * 1.2, py - t[1] * 1.2, zg + 3.9]], UVT["flag"])

# ------------------------------------------------------------------ summit bleacher
best = None
for ds in range(-60, 61, 3):
    j = idx(7392 + ds)
    for side in (1, -1):
        for off in np.arange(3.0, 20.1, 1.0):
            d = side * ((road.el[j] if side > 0 else road.er[j]) + SHOULDER_W + off)
            x, y = road.xy(j, d)
            r = footprint_ok(x, y, tangent(j), 8.0, 3.2, margin=3.0, maxrange=1.2)
            if r:
                sc = r[0] + 0.04 * off + 0.004 * abs(ds)
                if best is None or sc < best[0]:
                    best = (sc, x, y, j, side, ds, off, r)
if best:
    _, x, y, j, side, ds, off, r = best
    t = tangent(j); n = normal(j) * side           # n points away from the road
    for k in range(3):                             # 3 tiers, back tier highest
        c = np.array([x, y]) + n * (k - 1) * 1.05
        box(c, t, 8.0, 1.05, r[1] - 0.4, r[2] + 0.45 * (k + 1), UVT["wood"], coll=True)
    LOG.append(f"bleacher s={7392 + ds} {'L' if side > 0 else 'R'} {off:.0f} m past shoulder (ground range {r[0]:.2f} m)")
else:
    LOG.append("bleacher: no flat spot near the summit gate, skipped")

# ------------------------------------------------------------------ corner barriers in front of steep drops
F = json.load(open(DER / "features.geojson"))["features"]
drops = [f["properties"] for f in F if f["properties"].get("kind") == "steep_drop"]
gr = meta["guardrails"]
cand = []
for p in drops:
    s0, s1, side = p["s_start_m"], p["s_end_m"], p["side"]
    if s1 > road.s[-1] - 5 or any(abs((s0 + s1) / 2 - g) < 120 for g in GATES):
        continue
    if any(g["side"] == side and g["s0"] - 5 < s1 and g["s1"] + 5 > s0 for g in gr):
        continue                                    # already guarded
    a, b = idx(s0 - 30), idx(s1 + 10)
    kk = road.k[a:b]
    outside = -kk.min() if side == "left" else kk.max()   # left drop is the outside of a RIGHT turn (k<0)
    kmax = float(np.abs(kk).max())
    cand.append((-(kmax + 0.5 * max(outside, 0)) * (1 + 0.02 * p["length_m"]), s0, s1, side, kmax))
cand.sort()
picked = []
for sc, s0, s1, side, kmax in cand:
    if len(picked) >= 8: break
    if kmax < 0.015: continue
    if all(abs(s0 - q[0]) > 400 for q in picked):
        picked.append((s0, s1, side, kmax))
picked.sort()
for r_i, (s0, s1, side, kmax) in enumerate(picked):
    sgn = 1 if side == "left" else -1
    sa, sb = s0 - 6, s1 + 4
    ss = np.arange(sa, sb + 0.01, 2.0 if r_i % 2 == 0 else 1.0)
    jj = np.array([idx(s) for s in ss])
    e = road.el[jj] if sgn > 0 else road.er[jj]
    d = sgn * (e + SHOULDER_W + 0.35)
    x, y = road.xy(jj, d)
    ze = road.surface_z(jj, sgn * (e + SHOULDER_W), noise=False)
    zg = dtm(x, y); zg = np.where(np.isfinite(zg), zg, ze)
    zbase = np.minimum(zg, ze) - 0.15
    for q in range(len(ss) - 1):
        cx, cy = (x[q] + x[q + 1]) / 2, (y[q] + y[q + 1]) / 2
        fwd = np.array([x[q + 1] - x[q], y[q + 1] - y[q]]); L = np.hypot(*fwd) * 0.96
        zt = max(ze[q], ze[q + 1])
        if r_i % 2 == 0:   # water-filled plastic blocks, alternating red/white
            box((cx, cy), fwd, L, 0.55, min(zbase[q], zbase[q + 1]), zt + 0.85, UVT["block_red" if q % 2 else "block_white"])
        else:              # tyre stacks (1 m pitch)
            box((cx, cy), fwd, L, 0.65, min(zbase[q], zbase[q + 1]), zt + 0.75, UVT["tyre"], top=UVT["tyre"])
    strip_wall(x, y, zbase - 0.3, np.maximum(ze, zg) + 1.1)
    LOG.append(f"{'water blocks' if r_i % 2 == 0 else 'tyre wall'} s={sa:.0f}-{sb:.0f} {side[0].upper()} (|k|max {kmax:.3f} 1/m, R~{1 / max(kmax, 1e-3):.0f} m)")

# ------------------------------------------------------------------ write
P = np.concatenate(VP); T = np.concatenate(VT); UV = np.concatenate(VU)
G["dressing"] = [dict(name="dressing", material="dressing", P=P, T=T, UV=UV, N=vertex_normals(P, T))]
Pc = np.concatenate(CP); Tc = np.concatenate(CT)
G["dressing"].append(dict(name="1WALL_dressing", material="invisible", P=Pc, T=Tc, UV=np.zeros((len(Pc), 2)),
                          N=vertex_normals(Pc, Tc), visible=False, renderable=False, cast_shadows=False))
assert len(P) < 65535 and len(Pc) < 65535, (len(P), len(Pc))
meta["dressing"] = LOG
pickle.dump(geo, open(BUILD / "geometry.pkl", "wb"))
print("\n".join(LOG)); print(f"dressing: {len(P)} verts visual, {len(Pc)} collision, 2 meshes")
