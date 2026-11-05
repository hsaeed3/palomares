#!/usr/bin/env python3
"""Phase 2, step 21c: real roadside assets (fences, walls, guardrails, poles, signs, buildings, cypress ...).

Run after 21b.  Loads build/geometry.pkl, builds group "roadside" (shared by every layout) from
data/derived/roadside_assets_*.json and writes the pkl back.  Idempotent: it never reads its own output as input
(old roadside group / 21's guardrail + wall meshes are dropped and regenerated every run from meta + the asset files).

 * ONE visual mesh per 250 m chunk (single atlas material => one draw call per chunk), lod_out 450 m
 * ONE collision mesh (1WALL_rs_NN) per chunk: closed thin boxes, >= 1.4 m tall (scripts/p2/collide.py)
 * replaces 21's guardrail visuals/collision and its 1WALL_cut*/start/end walls with robust ones;
   `remove_guardrail` spans (and data/derived/roadside_overrides.json "suppress") cut the guardrail_candidate
   runs of 21, real `guardrail_wbeam` entries replace the candidates they overlap
 * writes data/derived/surface_spans.json (merged `surface` entries; read by 21's decal + RPM code),
   build/roadside_atlas.png (read by 22), build/roadside_solid.json (read by qa_collisions.py)
"""
import glob, json, math, pickle, re, sys, pathlib, collections
import numpy as np
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import Road, Raster, ROOT, DER, BUILD, CFG, SHOULDER_W
from geom import build_lock; build_lock()  # one build/ writer at a time
import collide as CO
import roadside_tex as RX

CHUNK = 250.0
LOD_OUT = 450.0
FAR = 6.0               # fences / walls farther than this from the road edge get no collision
geo = pickle.load(open(BUILD / "geometry.pkl", "rb"))
G, meta = geo["groups"], geo["meta"]
S_END = float(meta["s_end"])
road = Road(CFG["build_end_m"]); N = len(road.s)
dtm05 = Raster(DER / "dtm_r050_b40.tif"); dtm1 = Raster(DER / "dtm_r100_b150.tif")
nchunks = int(S_END // CHUNK) + 1


def ground(x, y):
    z = dtm05.sample(x, y); z2 = dtm1.sample(x, y)
    return np.where(np.isfinite(z), z, z2)


# ================================================================== load assets
def load_assets():
    files = sorted(glob.glob(str(DER / "roadside_assets_*.json")))
    base, fills = [], []
    for f in files:
        d = json.load(open(f))
        if isinstance(d, dict):
            d = next((v for v in d.values() if isinstance(v, list)), [])
        for e in d:
            e["_src"] = pathlib.Path(f).name
            (fills if "fill" in f else base).append(e)
    if fills:   # fill entries replace the low-confidence entries they cover
        ffr = {fr for e in fills for fr in e.get("frames", [])}
        lo = min(e["s0"] for e in fills); hi = max(e["s1"] for e in fills)
        def covered(e):
            if e.get("confidence") != "low":
                return False
            fr = set(e.get("frames", []))
            return (fr and fr <= ffr) or (e["s0"] >= lo and e["s1"] <= hi)
        n0 = len(base); base = [e for e in base if not covered(e)]
        print(f"  fill: dropped {n0 - len(base)} low-confidence entries covered by fill files")
    seen, out = set(), []
    for e in base + fills:
        k = (e["type"], e.get("side"), round(e["s0"]), round(e["s1"]))
        if k in seen:
            continue
        seen.add(k); out.append(e)
    return out, [pathlib.Path(f).name for f in files]


ASSETS, FILES = load_assets()
print(f"21c: {len(ASSETS)} asset entries from {len(FILES)} files: {', '.join(FILES)}")
CNT = collections.Counter(e["type"] for e in ASSETS)


# ================================================================== surface spans (-> 21 decals / RPMs)
KINDS = ("cracked_old", "crack_sealed", "patched", "fresh_overlay", "rpm_present")


def surface_spans():
    raw = []
    for e in ASSETS:
        if e["type"] != "surface":
            continue
        a = e.get("attrs", {}) or {}
        vals = [str(v).lower() for k, v in a.items() if k in ("condition", "state", "kind", "surface", "type")] + [str(e.get("subtype", "")).lower()]
        for k in KINDS:
            if any(k in v for v in vals):
                raw.append((k, float(e["s0"]), float(e["s1"]), e.get("confidence", "")))
        rp = str(a.get("rpm_present", "")).lower()
        if rp in ("true", "yes", "1", "present"):
            raw.append(("rpm_present", float(e["s0"]), float(e["s1"]), e.get("confidence", "")))
    out = []
    for k in KINDS:
        iv = sorted((s0, s1) for kk, s0, s1, _ in raw if kk == k)
        m = []
        for s0, s1 in iv:
            if m and s0 <= m[-1][1] + 1.0:
                m[-1][1] = max(m[-1][1], s1)
            else:
                m.append([s0, s1])
        out += [dict(kind=k, s0=a, s1=b) for a, b in m]
    json.dump(out, open(DER / "surface_spans.json", "w"), indent=1)
    return out


SPANS = surface_spans()
print("  surface spans:", dict(collections.Counter(s["kind"] for s in SPANS)))


# ================================================================== geometry helpers
# Offsets in the survey are from the asphalt edge. Ground-referenced assets are kept at least on the shoulder's
# outer edge (road.sh_w_l/r, edge_profile.py) and never sunk into a ditch / drop: if the DTM at the requested offset
# is more than DROP_MAX below the road edge, the asset is pulled in toward the edge until the ground is within range
# (else it stands on the shoulder's outer lip). LOW_ASSETS collects any base still > DROP_MAX below the road edge.
DROP_MAX = 0.6
CLAMP_MAX_OFF = 6.0   # roadside furniture only; buildings / tree rows further out keep their surveyed offset
LOW_ASSETS = []


def clamp_off(i, sgn, e, off, zr):
    """per-sample offset from the asphalt edge (arrays): >= shoulder width, ground within DROP_MAX of the road edge"""
    i = np.atleast_1d(i); e = np.atleast_1d(e); zr = np.atleast_1d(zr)
    sw = road.sh_w(i, sgn); off = np.maximum(np.broadcast_to(np.asarray(off, float), i.shape), sw)
    fr = np.linspace(1.0, 0.0, 9)                                    # candidates: off -> shoulder edge
    cand = sw[:, None] + (off - sw)[:, None] * fr[None, :]
    x, y = road.xy(np.repeat(i[:, None], 9, 1), sgn * (e[:, None] + cand))
    zg = ground(x, y)
    ok = np.isfinite(zg) & (zg >= zr[:, None] - DROP_MAX)
    ok[:, -1] = True
    return cand[np.arange(len(i)), ok.argmax(1)]


class Pt:
    def __init__(self, s, side, off, ref="ground"):
        sgn = 1 if side == "left" else -1
        s = float(np.clip(s, 1, S_END - 1))
        e = float(np.interp(s, road.s, road.el if sgn > 0 else road.er))
        X, Y = float(np.interp(s, road.s, road.x)), float(np.interp(s, road.s, road.y))
        nx, ny = float(np.interp(s, road.s, road.nx)), float(np.interp(s, road.s, road.ny)); nn = math.hypot(nx, ny); nx, ny = nx / nn, ny / nn
        tx, ty = float(np.interp(s, road.s, road.tx)), float(np.interp(s, road.s, road.ty)); tn = math.hypot(tx, ty); tx, ty = tx / tn, ty / tn
        i = int(round(s)); i = min(max(i, 0), N - 1)
        zr = float(road.surface_z(i, sgn * e, noise=False))
        sw = float(road.sh_w(i, sgn))
        if ref != "road" and off <= CLAMP_MAX_OFF:
            off = float(clamp_off(i, sgn, e, off, zr)[0])
        d = sgn * (e + off)
        self.x, self.y = X + nx * d, Y + ny * d
        self.t = np.array([tx, ty]); self.out = np.array([nx * sgn, ny * sgn])      # outward = away from the road
        zs = float(road.surface_z(i, sgn * (e + min(off, sw)), noise=False))
        if ref == "road":
            self.z = zr
        else:
            zg = float(ground(self.x, self.y)); w = min(max((off - sw) / 1.2, 0), 1)
            self.z = zs * (1 - w) + zg * w if np.isfinite(zg) else zs
            if self.z < zr - DROP_MAX and off <= CLAMP_MAX_OFF:
                LOW_ASSETS.append(dict(s=round(s, 1), side=side, off=round(off, 2), drop=round(zr - self.z, 2)))
        self.zr = zr; self.s = s; self.e = e; self.sgn = sgn; self.i = i


class Line:
    """polyline parallel to the road edge (offset off from the edge) with terrain-following z"""
    def __init__(self, s0, s1, side, off, ref="ground", step=1.0):
        s0 = float(np.clip(s0, 1, S_END - 2)); s1 = float(np.clip(s1, s0 + 0.5, S_END - 1))
        n = max(2, int(math.ceil((s1 - s0) / step)) + 1); ss = np.linspace(s0, s1, n); sgn = 1 if side == "left" else -1
        e = np.interp(ss, road.s, road.el if sgn > 0 else road.er)
        nx, ny = np.interp(ss, road.s, road.nx), np.interp(ss, road.s, road.ny); nn = np.hypot(nx, ny); nx, ny = nx / nn, ny / nn
        idx = np.clip(np.round(ss).astype(int), 0, N - 1)
        zr = road.surface_z(idx, sgn * e, noise=False)
        sw = road.sh_w(idx, sgn)
        if ref != "road" and off <= CLAMP_MAX_OFF:   # per-sample clamp, smoothed so a fence line bends gently in toward the edge
            raw = clamp_off(idx, sgn, e, off, zr)
            from scipy.ndimage import minimum_filter1d
            sm = gaussian_filter1d(minimum_filter1d(raw, max(3, int(6 / step)) | 1, mode="nearest"), max(1.0, 2.5 / step), mode="nearest")
            offv = np.maximum(np.minimum(sm, raw), sw)                       # smooth, but never further out than the clamp
        else:
            offv = np.full(n, float(off))
        d = sgn * (e + offv)
        self.x = np.interp(ss, road.s, road.x) + nx * d; self.y = np.interp(ss, road.s, road.y) + ny * d
        zs = road.surface_z(idx, sgn * (e + np.minimum(offv, sw)), noise=False)
        if ref == "road":
            z = zr
        else:
            zg = ground(self.x, self.y); w = np.clip((offv - sw) / 1.2, 0, 1)
            z = np.where(np.isfinite(zg), zs * (1 - w) + zg * w, zs)
            lo = (z < zr - DROP_MAX) & (offv <= CLAMP_MAX_OFF)
            if lo.any():
                LOW_ASSETS.append(dict(s=round(float(ss[lo][0]), 1), side=side, off=round(float(offv[lo][0]), 2), drop=round(float((zr - z)[lo].max()), 2), n=int(lo.sum())))
        self.z = gaussian_filter1d(z, max(0.7, 1.5 / step), mode="nearest"); self.zr = gaussian_filter1d(zr, max(0.7, 1.5 / step), mode="nearest")
        self.ss, self.sgn, self.side = ss, sgn, side
        self.cum = np.r_[0, np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))]; self.L = float(self.cum[-1])

    def at(self, a, arr=None):
        a = np.clip(a, 0, self.L)
        return np.interp(a, self.cum, self.x), np.interp(a, self.cum, self.y), np.interp(a, self.cum, self.z if arr is None else arr)

    def poly(self, a, b, step=1.0):
        a_ = np.unique(np.r_[np.arange(a, b, step), b]); x, y, z = self.at(a_)
        return a_, x, y, z


def pieces(s0, s1):
    """split [s0, s1] at chunk boundaries"""
    out, a = [], s0
    while a < s1 - 1e-6:
        b = min(s1, (math.floor(a / CHUNK + 1e-9) + 1) * CHUNK)
        out.append((a, b)); a = b
    return out or [(s0, s1)]


class Acc:
    def __init__(self, c=0):
        self.P, self.T, self.UV, self.N, self.n = [], [], [], [], 0
        self.col = CO.Collider(f"1WALL_rs_{c:02d}")

    def quad(self, P4, rect, out=None, uspan=(0, 1), vspan=(0, 1)):
        P4 = np.asarray(P4, float)
        nrm = np.cross(P4[1] - P4[0], P4[3] - P4[0])
        if out is not None and np.dot(nrm, out) < 0:
            P4 = P4[[1, 0, 3, 2]]; uspan = (uspan[1], uspan[0]); nrm = -nrm
        ln = np.linalg.norm(nrm); nrm = nrm / (ln if ln > 1e-12 else 1)
        u0, v0, u1, v1 = rect
        ua = u0 + (u1 - u0) * uspan[0]; ub = u0 + (u1 - u0) * uspan[1]
        va = v1 - (v1 - v0) * vspan[0]; vb = v1 - (v1 - v0) * vspan[1]       # vspan 0 = bottom
        self.P.append(P4); self.UV.append(np.array([[ua, va], [ub, va], [ub, vb], [ua, vb]])); self.N.append(np.tile(nrm, (4, 1)))
        self.T.append(np.array([[0, 1, 2], [0, 2, 3]]) + self.n); self.n += 4

    def tri(self, P3, rect, uv3):
        P3 = np.asarray(P3, float); nrm = np.cross(P3[1] - P3[0], P3[2] - P3[0]); nrm /= (np.linalg.norm(nrm) + 1e-12)
        u0, v0, u1, v1 = rect
        self.P.append(P3); self.N.append(np.tile(nrm, (3, 1)))
        self.UV.append(np.array([[u0 + (u1 - u0) * a, v1 - (v1 - v0) * b] for a, b in uv3])); self.T.append(np.array([[0, 1, 2]]) + self.n); self.n += 3

    def quad2(self, P4, rect, **kw):
        """double sided (alpha-tested sheets): both windings, own normals"""
        P4 = np.asarray(P4, float)
        self.quad(P4, rect, **kw); self.quad(P4[[1, 0, 3, 2]], rect, **{**kw, "uspan": kw.get("uspan", (0, 1))[::-1]})

    def box(self, cx, cy, z0, z1, fwd, L, W, rect, rect_top=None):
        f = np.array(fwd, float); f = f / (np.hypot(*f) + 1e-12); l = np.array([-f[1], f[0]])
        c = np.array([cx, cy]); co = [c - f * L / 2 - l * W / 2, c + f * L / 2 - l * W / 2, c + f * L / 2 + l * W / 2, c - f * L / 2 + l * W / 2]
        for k in range(4):
            a, b = co[k], co[(k + 1) % 4]; mid = (a + b) / 2 - c
            self.quad([[*a, z0], [*b, z0], [*b, z1], [*a, z1]], rect, out=np.array([*mid, 0]))
        self.quad([[*co[0], z1], [*co[1], z1], [*co[2], z1], [*co[3], z1]], rect_top or rect, out=np.array([0, 0, 1.0]))

    def prism(self, cx, cy, z0, z1, r, sides, rect):
        a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        pts = [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in a]
        for k in range(sides):
            p, q = pts[k], pts[(k + 1) % sides]; mid = np.array([(p[0] + q[0]) / 2 - cx, (p[1] + q[1]) / 2 - cy, 0])
            self.quad([[*p, z0], [*q, z0], [*q, z1], [*p, z1]], rect, out=mid)

    def mesh(self, name):
        if not self.P:
            return None
        P = np.concatenate(self.P); T = np.concatenate(self.T); UV = np.concatenate(self.UV); Nn = np.concatenate(self.N)
        return dict(name=name, material="roadside", P=P, T=T, UV=UV, N=Nn, cast_shadows=False, lod_out=LOD_OUT)


class _Chunks(dict):
    def __missing__(self, c):
        self[c] = Acc(c); return self[c]


ACC = _Chunks()
SIGNCHK = []      # emitted sign-face quads, verified by qa_collisions.py (orientation / mirroring)
SOLID = []         # sidecar for qa_collisions: every item that must have collision
SKIPPED = collections.Counter()
ATLAS = RX.base_atlas(); R = ATLAS.rect


def chunk_of(s):
    return min(int(s // CHUNK), nchunks - 1)


def need_col(kind, line_or_pts, ref_z, far, s0, s1):
    SOLID.append(dict(type=kind, s0=float(s0), s1=float(s1), pts=[[round(float(a), 2), round(float(b), 2)] for a, b in line_or_pts],
                      z=[round(float(z), 2) for z in ref_z]))


# ------------------------------------------------------------------ elements
def strip_panel(acc, ln, a, b, z0, z1, rect, tw, th, double=True):
    """vertical sheet along the line between arclengths a..b; z0/z1 offsets from the line z (scalars).
    tw/th = world size of one texture repeat; quads follow the terrain and are <= 1.5 m long."""
    nseg = max(1, int(round((b - a) / tw))); rows = max(1, int(round((z1 - z0) / th)))
    for k in range(nseg):
        sa = a + (b - a) * k / nseg; sb = a + (b - a) * (k + 1) / nseg
        nq = max(1, int(math.ceil((sb - sa) / 1.5)))
        for j in range(nq):
            qa = sa + (sb - sa) * j / nq; qb = sa + (sb - sa) * (j + 1) / nq
            xa, ya, za = ln.at(qa); xb, yb, zb = ln.at(qb)
            for r in range(rows):
                f0, f1 = r / rows, (r + 1) / rows
                P4 = [[xa, ya, za + z0 + (z1 - z0) * f0], [xb, yb, zb + z0 + (z1 - z0) * f0], [xb, yb, zb + z0 + (z1 - z0) * f1], [xa, ya, za + z0 + (z1 - z0) * f1]]
                kw = dict(uspan=(j / nq, (j + 1) / nq))
                (acc.quad2 if double else acc.quad)(P4, rect, **kw)


def wall_body(acc, ln, a, b, z0, z1, thick, rect, tw, th, top=None):
    """solid wall of thickness `thick`: both faces + cap, textured from the same tile, ends closed"""
    ar, x, y, z = ln.poly(a, b, 1.2)
    if len(x) < 2:
        return
    t = np.gradient(np.c_[x, y], axis=0); t /= (np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-12); l = np.c_[-t[:, 1], t[:, 0]]
    rows = max(1, int(round((z1 - z0) / th)))
    for sgn in (1, -1):
        px = x + l[:, 0] * thick / 2 * sgn; py = y + l[:, 1] * thick / 2 * sgn
        for i in range(len(x) - 1):
            seg = math.hypot(px[i + 1] - px[i], py[i + 1] - py[i]); nu = max(1, int(round(seg / tw)))
            for r in range(rows):
                f0, f1 = r / rows, (r + 1) / rows
                P4 = [[px[i], py[i], z[i] + z0 + (z1 - z0) * f0], [px[i + 1], py[i + 1], z[i + 1] + z0 + (z1 - z0) * f0],
                      [px[i + 1], py[i + 1], z[i + 1] + z0 + (z1 - z0) * f1], [px[i], py[i], z[i] + z0 + (z1 - z0) * f1]]
                acc.quad(P4, rect, out=np.array([l[i, 0] * sgn, l[i, 1] * sgn, 0.0]), uspan=(0, min(1.0, seg / tw)))
    for i in range(len(x) - 1):
        acc.quad([[x[i] - l[i, 0] * thick / 2, y[i] - l[i, 1] * thick / 2, z[i] + z1], [x[i + 1] - l[i + 1, 0] * thick / 2, y[i + 1] - l[i + 1, 1] * thick / 2, z[i + 1] + z1],
                  [x[i + 1] + l[i + 1, 0] * thick / 2, y[i + 1] + l[i + 1, 1] * thick / 2, z[i + 1] + z1], [x[i] + l[i, 0] * thick / 2, y[i] + l[i, 1] * thick / 2, z[i] + z1]],
                 top or rect, out=np.array([0, 0, 1.0]), uspan=(0, 0.3), vspan=(0, 0.3))
    for q, sg in ((0, -1.0), (len(x) - 1, 1.0)):
        acc.quad([[x[q] - l[q, 0] * thick / 2, y[q] - l[q, 1] * thick / 2, z[q] + z0], [x[q] + l[q, 0] * thick / 2, y[q] + l[q, 1] * thick / 2, z[q] + z0],
                  [x[q] + l[q, 0] * thick / 2, y[q] + l[q, 1] * thick / 2, z[q] + z1], [x[q] - l[q, 0] * thick / 2, y[q] - l[q, 1] * thick / 2, z[q] + z1]],
                 rect, out=np.array([t[q, 0] * sg, t[q, 1] * sg, 0.0]), uspan=(0, 0.3))


def posts_on(acc, ln, a, b, sp, w, top, rect, below=0.35, ref_arr=None):
    n = max(1, int(round((b - a) / sp))); out = []
    for k in range(n + 1):
        aa = a + (b - a) * k / n; x, y, z = ln.at(aa)
        i = min(int(np.searchsorted(ln.cum, aa)), len(ln.cum) - 1)
        tx, ty = (ln.x[min(i + 1, len(ln.x) - 1)] - ln.x[max(i - 1, 0)]), (ln.y[min(i + 1, len(ln.y) - 1)] - ln.y[max(i - 1, 0)])
        acc.box(float(x), float(y), float(z) - below, float(z) + top, (tx, ty), w, w, rect)
        out.append((float(x), float(y), float(z)))
    return out


def collide_line(acc, ln, a, b, z0=-0.7, h=None, half=0.07, kind="", s0=0, s1=0, zarr=None, minh=CO.MIN_TOP):
    """closed thin box along the line a..b; top >= minh above the reference z"""
    ar, x, y, z = ln.poly(max(a - 0.5, 0.0), min(b + 0.5, ln.L), 1.0)   # dev4: 0.5 m past each end (end posts were uncovered)
    top = max(minh, h or 0)
    acc.col.ribbon(x, y, z + z0, z + top, half)
    need_col(kind, zip(x, y), z, False, s0, s1)


# ------------------------------------------------------------------ types
FENCE = {   # type: (post_rect, post_w, post_spacing, post_top, panel_rect, tile_w, tile_h, default_h, z0)
    "wood_split_rail_fence": ("wood_dark", 0.14, 3.0, None, "splitrail", 3.0, 1.0, 1.2, 0.0),
    "wood_board_fence": ("wood_dark", 0.12, 2.4, None, "boards", 2.4, 1.2, 1.4, 0.0),
    "white_rail_fence": ("wood_white", 0.13, 2.4, None, "whiterail", 2.4, 1.2, 1.3, 0.0),
    "ranch_wire_fence": ("wood_dark", 0.12, 4.0, None, "ranchwire", 1.5, 1.2, 1.2, 0.0),
    "chainlink_fence": ("metal", 0.07, 3.0, None, "chainlink", 0.64, 0.64, 1.8, 0.0),
    "solid_wood_privacy_fence": ("wood_dark", 0.14, 2.4, None, "privacy", 2.4, 1.8, 1.8, 0.0),
    "guardrail_cable": ("metal", 0.09, 4.0, None, "cable", 4.0, 0.8, 0.8, 0.2),
}
WALLS = {   # type: (rect, tile, thick, default_h, ref, cap)
    "retaining_wall_concrete": ("concrete", 1.6, 0.35, 1.2, "ground"),
    "retaining_wall_stone": ("stone", 1.6, 0.45, 1.0, "ground"),
    "retaining_wall_timber": ("timber", 2.4, 0.30, 1.2, "ground"),
    "bridge_parapet_stone": ("stone", 1.6, 0.40, 0.8, "road"),
    "bridge_parapet_concrete": ("concrete", 1.6, 0.30, 0.8, "road"),
    "culvert_headwall": ("concrete", 1.6, 0.35, 0.6, "ground"),
}


def textof(a):
    return " ".join(str(v) for k, v in (a or {}).items() if k in ("text", "type_detail", "shape", "legend", "note", "kind", "label")).lower()


def sides_of(e):
    s = e.get("side", "right")
    return ["left", "right"] if s in ("both", "center") else [s]


def do_fence(e, side):
    t = e["type"]; prect, pw, sp, ptop, panel, tw, th, hdef, z0 = FENCE[t]
    off = float(e.get("offset_m") or 1.0); h = float(e.get("height_m") or hdef)
    h = min(max(h, 0.6), 3.0)
    ref = "road" if t == "guardrail_cable" else "ground"
    s0, s1 = float(e["s0"]), float(e["s1"])
    if s1 - s0 < 3:
        s1 = s0 + 6
    ln = Line(s0, s1, side, off, ref=ref); far = off > FAR
    for (pa, pb) in pieces(s0, s1):
        acc = ACC[chunk_of(0.5 * (pa + pb))]; a = max(0, pa - s0); b = min(ln.L, (pb - s0) * ln.L / (s1 - s0))
        if b - a < 0.3:
            continue
        strip_panel(acc, ln, a, b, z0, h, R[panel], tw, th)
        posts_on(acc, ln, a, b, sp, pw, h + 0.05, R[prect])
        if not far:
            collide_line(acc, ln, a, b, kind=t, s0=pa, s1=pb, h=h)
        else:
            SKIPPED[t] += 1


def do_wall(e, side):
    t = e["type"]; rect, tw, thick, hdef, ref = WALLS[t]
    off = float(e.get("offset_m") or 0.3); h = min(max(float(e.get("height_m") or hdef), 0.3), 5.0)
    s0, s1 = float(e["s0"]), float(e["s1"])
    if s1 - s0 < 3:
        s0, s1 = s0 - 2, s1 + 2
    ln = Line(s0, s1, side, off + thick / 2 if ref == "ground" else off, ref=ref)
    far = off > FAR and not t.startswith("bridge")
    for (pa, pb) in pieces(s0, s1):
        acc = ACC[chunk_of(0.5 * (pa + pb))]; a = max(0, pa - s0); b = min(ln.L, (pb - s0) * ln.L / (s1 - s0))
        if b - a < 0.3:
            continue
        wall_body(acc, ln, a, b, -0.45, h, thick, R[rect], tw, tw)
        if not far:
            collide_line(acc, ln, a, b, kind=t, s0=pa, s1=pb, h=h, half=max(0.07, thick / 2 * 0.9))


def do_rockfall(e, side):
    off = float(e.get("offset_m") or 1.5); h = float(e.get("height_m") or 4)
    s0, s1 = float(e["s0"]), float(e["s1"]); ln = Line(s0, s1, side, off)
    back = min(max(h * 0.6, 2.0), 5.0); nb = int(back)
    for (pa, pb) in pieces(s0, s1):
        acc = ACC[chunk_of(0.5 * (pa + pb))]; a = max(0, pa - s0); b = min(ln.L, (pb - s0) * ln.L / (s1 - s0))
        if b - a < 0.3:
            continue
        ar, x, y, z = ln.poly(a, b, 1.5)
        sg = ln.sgn
        for i in range(len(x) - 1):
            rows = []
            for r in range(nb + 1):
                ax_ = np.array([x[i], x[i + 1]]) + np.array([road.nx[min(int(ln.ss[0] + ar[i]), N - 1)] * sg * r, road.nx[min(int(ln.ss[0] + ar[i + 1]), N - 1)] * sg * r])
                ay_ = np.array([y[i], y[i + 1]]) + np.array([road.ny[min(int(ln.ss[0] + ar[i]), N - 1)] * sg * r, road.ny[min(int(ln.ss[0] + ar[i + 1]), N - 1)] * sg * r])
                zz = ground(ax_, ay_); zz = np.where(np.isfinite(zz), zz, z[i:i + 2]) + 0.15 + 0.0
                rows.append((ax_, ay_, zz))
            for r in range(nb):
                A, B = rows[r], rows[r + 1]
                acc.quad2([[A[0][0], A[1][0], A[2][0]], [A[0][1], A[1][1], A[2][1]], [B[0][1], B[1][1], B[2][1]], [B[0][0], B[1][0], B[2][0]]], R["rockmesh"])
        collide_line(acc, ln, a, b, kind=e["type"], s0=pa, s1=pb, h=CO.MIN_TOP)


def do_guardrail(s0, s1, side, off, h, posts_wood, kind, conf=""):
    ln = Line(s0, s1, side, off, ref="road")
    for (pa, pb) in pieces(s0, s1):
        acc = ACC[chunk_of(0.5 * (pa + pb))]; a = max(0, pa - s0); b = min(ln.L, (pb - s0) * ln.L / max(s1 - s0, 1e-6))
        if b - a < 0.3:
            continue
        strip_panel(acc, ln, a, b, h - 0.36, h, R["wbeam"], 3.81, 0.36)
        posts_on(acc, ln, a, b, 1.905, 0.14 if posts_wood else 0.1, h - 0.05, R["wood_dark" if posts_wood else "metal"], below=0.5)
        collide_line(acc, ln, a, b, z0=-0.8, h=max(CO.MIN_TOP, h + 0.6), kind=kind, s0=pa, s1=pb)


def do_pole(e, side, poles):
    off = float(e.get("offset_m") or 3); H = float(e.get("height_m") or 10); s0, s1 = float(e["s0"]), float(e["s1"])
    sp = float((e.get("attrs") or {}).get("spacing_m") or 45)
    ss = [0.5 * (s0 + s1)] if s1 - s0 < 5 else list(np.arange(s0, s1 + 0.1, sp))
    for s in ss:
        p = Pt(s, side, off)
        acc = ACC[chunk_of(p.s)]
        acc.prism(p.x, p.y, p.z - 0.3, p.z + H, 0.14, 7, R["pole"])
        arm_z = p.z + H - 0.5; n = np.array([-p.t[1], p.t[0]])
        acc.box(p.x, p.y, arm_z, arm_z + 0.12, n, 2.2, 0.1, R["wood_dark"])
        for dz in (-0.9, 0.0, 0.9):                                              # insulators
            acc.box(p.x + n[0] * dz, p.y + n[1] * dz, arm_z + 0.12, arm_z + 0.3, n, 0.06, 0.06, R["white"])
        if off <= 8:
            acc.col.box(p.x, p.y, p.t, 0.34, 0.34, p.z - 0.6, p.z + 2.6)
            need_col("utility_pole", [(p.x, p.y)], [p.z], False, p.s, p.s)
        poles[side].append((p.s, p.x, p.y, p.z + H - 0.35, n))


def do_wires(poles):
    for side, lst in poles.items():
        lst.sort(key=lambda q: q[0])
        for A, B in zip(lst[:-1], lst[1:]):
            if B[0] - A[0] > 90:
                continue
            span = math.hypot(B[1] - A[1], B[2] - A[2]); sag = 0.0004 * span * span + 0.1
            for dz in (-0.9, 0.0, 0.9):
                u = np.linspace(0, 1, max(3, int(span / 4)))
                x = A[1] + A[4][0] * dz + (B[1] + B[4][0] * dz - A[1] - A[4][0] * dz) * u
                y = A[2] + A[4][1] * dz + (B[2] + B[4][1] * dz - A[2] - A[4][1] * dz) * u
                z = A[3] + (B[3] - A[3]) * u - sag * 4 * u * (1 - u)
                for k in range(len(u) - 1):
                    acc = ACC[chunk_of(A[0] + (B[0] - A[0]) * u[k])]
                    t = np.array([x[k + 1] - x[k], y[k + 1] - y[k]]); t /= np.hypot(*t) + 1e-9; w = np.array([-t[1], t[0]]) * 0.02
                    P4 = [[x[k] - w[0], y[k] - w[1], z[k]], [x[k + 1] - w[0], y[k + 1] - w[1], z[k + 1]], [x[k + 1] + w[0], y[k + 1] + w[1], z[k + 1]], [x[k] + w[0], y[k] + w[1], z[k]]]
                    acc.quad2(P4, R["black"])
                    P4 = [[x[k], y[k], z[k] - 0.02], [x[k + 1], y[k + 1], z[k + 1] - 0.02], [x[k + 1], y[k + 1], z[k + 1] + 0.02], [x[k], y[k], z[k] + 0.02]]
                    acc.quad2(P4, R["black"])


def do_sign(e, side):
    a = e.get("attrs") or {}
    txt = " ".join(str(a.get(k, "")) for k in ("text", "type_detail", "shape")).lower().strip() or str(a.get("note", "")).lower()
    off = float(e.get("offset_m") or 1.5); s = 0.5 * (float(e["s0"]) + float(e["s1"]))
    im, w, h, key = RX.sign_image(txt, None)
    p = Pt(s, side, off)
    # Approach direction: every readable sign faces northbound traffic (the survey drove S->N and only describes what a
    # northbound driver reads); signs that are only seen from behind face southbound traffic.
    nb = not key.startswith("back")
    # Turn direction = how the ROAD turns ahead of an approaching driver (signed curvature road.k, + = left for s increasing;
    # southbound drivers see the opposite sign).  Overrides the survey wording when the curve is real (> ~5 deg in 45 m).
    i0 = min(max(int(round(s)), 0), N - 1)
    dh = float(road.k[i0:i0 + 45].sum()) if nb else -float(road.k[max(i0 - 45, 0):i0 + 1].sum())
    cdir = ("left" if dh > 0 else "right") if abs(dh) > 0.09 else None
    if cdir and key.startswith(("chev", "dia_", "arrow")):
        im, w, h, key = RX.sign_image(txt, None, dirn_override=cdir)
    if im is None:
        return "skipped"
    if max(im.size) > 160:
        k_ = 160 / max(im.size); im = im.resize((max(8, int(im.size[0] * k_)), max(8, int(im.size[1] * k_))))
    rect = ATLAS.put("sign_" + key, im)
    acc = ACC[chunk_of(p.s)]
    ht = float(e.get("height_m") or 2.0)
    zb = min(max(ht - h, 0.9), 1.9) if h < 0.5 else min(max(ht - h, 1.2), 2.1)
    face = -p.t if nb else p.t                                                      # normal points AGAINST the approach direction
    face = face * 0.97 + (-p.out) * 0.2; face /= np.linalg.norm(face)             # slightly angled toward the road
    r = np.array([-face[1], face[0]])        # viewer's right: the viewer looks along -face (NOT along +face: that was the mirror bug)
    c = np.array([p.x, p.y]) + face * 0.05
    ztop = p.z + zb + h
    P4 = [[*(c - r * w / 2), p.z + zb], [*(c + r * w / 2), p.z + zb], [*(c + r * w / 2), ztop], [*(c - r * w / 2), ztop]]
    acc.quad(P4, rect, out=np.array([*face, 0]))
    SIGNCHK.append(dict(s=float(p.s), key=key, nb=bool(nb), t=[float(p.t[0]), float(p.t[1])], face=[float(face[0]), float(face[1])],
                        cdir=cdir, P=np.round(acc.P[-1], 3).tolist(), UV=np.round(acc.UV[-1], 5).tolist(), N=np.round(acc.N[-1][0], 3).tolist(),
                        rect=[float(v) for v in rect]))
    acc.quad(P4, R["signback"], out=np.array([*(-face), 0]))
    acc.box(p.x, p.y, p.z - 0.3, ztop - (0.0 if h < 0.5 else 0.05), p.t, 0.07, 0.07, R["metal"])
    if off <= 5:
        acc.col.box(p.x, p.y, p.t, 0.16, 0.16, p.z - 0.5, p.z + 1.6)
        need_col("road_sign", [(p.x, p.y)], [p.z], False, p.s, p.s)
    return key


def do_mailbox(e, side):
    off = float(e.get("offset_m") or 1.5); a = e.get("attrs") or {}
    n = int(a.get("count") or max(1, min(6, round((float(e["s1"]) - float(e["s0"])) / 0.5) + 1 if e["s1"] > e["s0"] else 2)))
    p = Pt(0.5 * (e["s0"] + e["s1"]), side, off); acc = ACC[chunk_of(p.s)]; hh = float(e.get("height_m") or 1.1)
    wd = n * 0.42
    acc.box(p.x, p.y, p.z - 0.3, p.z + hh - 0.1, p.t, 0.12, 0.12, R["wood_dark"])
    acc.box(p.x, p.y, p.z + hh - 0.1, p.z + hh - 0.06, p.t, wd, 0.2, R["wood_dark"])
    for k in range(n):
        o = (k - (n - 1) / 2) * 0.42
        acc.box(p.x + p.t[0] * o, p.y + p.t[1] * o, p.z + hh - 0.06, p.z + hh + 0.16, p.out, 0.5, 0.2, R["mailbox"])
    if off <= FAR:
        acc.col.box(p.x, p.y, p.t, wd, 0.5, p.z - 0.5, p.z + 1.5)
        need_col("mailbox_cluster", [(p.x, p.y)], [p.z], False, p.s, p.s)


def do_gate(e, side):
    a = e.get("attrs") or {}; mat = (str(a.get("material", "")) + " " + str(a.get("note", ""))).lower()
    off = float(e.get("offset_m") or 3); h = float(e.get("height_m") or 2.0); s0, s1 = float(e["s0"]), float(e["s1"])
    if s1 - s0 < 3:
        s0, s1 = 0.5 * (s0 + s1) - 2.2, 0.5 * (s0 + s1) + 2.2
    ln = Line(s0, s1, side, off); acc = ACC[chunk_of(0.5 * (s0 + s1))]
    pill = any(k in mat for k in ("pillar", "brick", "stone")) and "not pillar" not in mat
    leaf = "chainlink" if "chainlink" in mat else ("whiterail" if any(k in mat for k in ("pipe", "white", "horse", "rail")) else "gate_iron")
    ph = max(h, 1.2) + (0.15 if pill else 0)
    ends = []
    for q in (0, ln.L):
        x, y, z = ln.at(q); ends.append((x, y, z))
        t = np.array([ln.x[-1] - ln.x[0], ln.y[-1] - ln.y[0]]); t /= np.hypot(*t) + 1e-9
        if pill:
            acc.box(float(x), float(y), float(z) - 0.4, float(z) + ph, t, 0.55, 0.55, R["stone"], rect_top=R["concrete"])
        else:
            acc.box(float(x), float(y), float(z) - 0.4, float(z) + h, t, 0.14, 0.14, R["metal"])
    a_, b_ = (0.28, ln.L - 0.28) if pill else (0, ln.L)
    if leaf == "chainlink":
        strip_panel(acc, ln, a_, b_, 0.0, min(h, 2.0), R["chainlink"], 0.64, 0.64)
    else:
        strip_panel(acc, ln, a_, b_, 0.1, max(0.9, h - (0.4 if pill else 0.1)), R[leaf], 2.4, 1.2)
    if off <= FAR:
        for pa, pb in pieces(s0, s1):
            aa, bb = max(0, pa - s0), min(ln.L, (pb - s0) * ln.L / (s1 - s0))
            if bb - aa > 0.3:
                collide_line(ACC[chunk_of(0.5 * (pa + pb))], ln, aa, bb, kind="gate_estate", s0=pa, s1=pb, h=max(h, CO.MIN_TOP))
    return pill


def do_building(e, side):
    a = e.get("attrs") or {}; txt = textof(a)
    off = float(e.get("offset_m") or 12); H = float(e.get("height_m") or 5)
    m = re.search(r"(\d+)\s*x\s*(\d+)", txt)
    Ls = float(e["s1"]) - float(e["s0"]); L = float(m.group(1)) if m else max(Ls, 10.0); D = float(m.group(2)) if m else 9.0
    L = min(max(L, 6), 30); D = min(max(D, 5), 20)
    p = Pt(0.5 * (e["s0"] + e["s1"]), side, off + D / 2); acc = ACC[chunk_of(p.s)]
    t = p.t; n = p.out; c = np.array([p.x, p.y])
    cor = [c - t * L / 2 - n * D / 2, c + t * L / 2 - n * D / 2, c + t * L / 2 + n * D / 2, c - t * L / 2 + n * D / 2]
    zg = ground(np.array([q[0] for q in cor]), np.array([q[1] for q in cor])); zg = zg[np.isfinite(zg)]
    zf = float(np.mean(zg)) if len(zg) else p.z; zlow = float(np.min(zg)) - 0.4 if len(zg) else zf - 0.4
    eave = zf + max(3.0, H - 1.5); ridge = eave + 1.8
    wall = R["bldg%d" % (int(abs(hash(round(p.s))) % 2))]
    for k in range(4):
        A, B = cor[k], cor[(k + 1) % 4]; mid = (A + B) / 2 - c; ln_ = np.hypot(*(B - A)); nu = max(1, int(round(ln_ / 4.0)))
        for j in range(nu):
            a0 = A + (B - A) * j / nu; a1 = A + (B - A) * (j + 1) / nu
            acc.quad([[*a0, zlow], [*a1, zlow], [*a1, eave], [*a0, eave]], wall, out=np.array([*mid, 0]))
    ov = 0.5
    lo0 = cor[0] - t * ov - n * ov; lo1 = cor[3] - t * ov + n * ov; hi0 = cor[1] + t * ov - n * ov; hi1 = cor[2] + t * ov + n * ov
    # two roof planes (ridge along n through the middle), gable triangles on t-ends
    rm0 = (cor[0] + cor[3]) / 2 - t * ov; rm1 = (cor[1] + cor[2]) / 2 + t * ov
    acc.quad([[*lo0, eave - 0.1], [*rm0, ridge], [*rm1, ridge], [*hi0, eave - 0.1]], R["roof"], out=np.array([*(-n), 1.0]))
    acc.quad([[*rm0, ridge], [*lo1, eave - 0.1], [*hi1, eave - 0.1], [*rm1, ridge]], R["roof"], out=np.array([*n, 1.0]))
    gm0 = (cor[0] + cor[3]) / 2; gm1 = (cor[1] + cor[2]) / 2
    for (q0, q1, g, sg) in ((cor[0], cor[3], gm0, -1.0), (cor[1], cor[2], gm1, 1.0)):
        P3 = np.array([[*q0, eave], [*q1, eave], [*g, ridge]])
        if np.dot(np.cross(P3[1] - P3[0], P3[2] - P3[0]), np.array([*(t * sg), 0])) < 0:
            P3 = P3[[1, 0, 2]]
        acc.tri(P3, wall, [(0, 0), (1, 0), (0.5, 0.4)])
    if off <= 15:
        acc.col.box(c[0], c[1], t, L, D, zlow, eave + 1.0)
        need_col("building", [tuple(c - n * D / 2 + t * k) for k in np.linspace(-L / 2, L / 2, int(L) + 1)], [zf] * (int(L) + 1), False, p.s, p.s)


def do_cypress(e, side):
    a = e.get("attrs") or {}
    off = float(e.get("offset_m") or 4); h = float(e.get("height_m") or 14); s0, s1 = float(e["s0"]), float(e["s1"])
    cnt = int(a.get("count") or 0)
    sp = float(a.get("spacing_m") or (((s1 - s0) / max(cnt - 1, 1)) if cnt > 1 else 4.5))
    sp = min(max(sp, 2.0), 9.0)
    for s in np.arange(s0, s1 + 0.01, sp) if s1 > s0 else [s0]:
        p = Pt(float(s), side, off); acc = ACC[chunk_of(p.s)]
        hh = h * (0.9 + 0.2 * ((hash(round(s)) % 100) / 100)); w = max(2.4, hh * 0.2)
        for ang in (0.0, np.pi / 2):
            d = np.array([math.cos(ang), math.sin(ang)]) * w / 2
            acc.quad2([[p.x - d[0], p.y - d[1], p.z - 0.3], [p.x + d[0], p.y + d[1], p.z - 0.3], [p.x + d[0], p.y + d[1], p.z + hh], [p.x - d[0], p.y - d[1], p.z + hh]], R["cypress"])
        if off <= FAR:
            acc.col.box(p.x, p.y, p.t, 0.5, 0.5, p.z - 0.5, p.z + 2.5)
            need_col("cypress_row", [(p.x, p.y)], [p.z], False, p.s, p.s)


def do_delineator(e, side):
    off = float(e.get("offset_m") or 1.2); h = float(e.get("height_m") or 1.1); s0, s1 = float(e["s0"]), float(e["s1"])
    for s in (np.arange(s0, s1 + 0.01, 12.0) if s1 - s0 > 6 else [0.5 * (s0 + s1)]):
        p = Pt(float(s), side, off); acc = ACC[chunk_of(p.s)]
        acc.box(p.x, p.y, p.z - 0.2, p.z + h, p.t, 0.09, 0.09, R["delin"])


def do_apron(e, side):
    s0, s1 = float(e["s0"]), float(e["s1"]); depth = 4.5
    if s1 - s0 < 3:
        s0, s1 = s0 - 2.5, s1 + 2.5
    for (pa, pb) in pieces(s0, s1):
        acc = ACC[chunk_of(0.5 * (pa + pb))]
        ss = np.unique(np.r_[np.arange(pa, pb, 1.5), pb]); offs = np.linspace(-0.3, depth, 5)
        pts = np.array([[(p_.x, p_.y, p_.z + 0.07) for p_ in (Pt(s, side, o) for o in offs)] for s in ss])
        for i in range(len(ss) - 1):
            for j in range(len(offs) - 1):
                acc.quad([pts[i, j], pts[i + 1, j], pts[i + 1, j + 1], pts[i, j + 1]], R["apron"], out=np.array([0, 0, 1.0]))


def do_hump(e):
    s0, s1 = float(e["s0"]), float(e["s1"])
    if s1 - s0 < 2:
        s0, s1 = s0 - 1.5, s1 + 1.5
    acc = ACC[chunk_of(0.5 * (s0 + s1))]
    ss = np.linspace(s0, s1, max(2, int((s1 - s0) / 1.0) + 1)); idx = np.clip(np.round(ss).astype(int), 0, N - 1)
    dl = road.el[idx] - 0.7; dr = -road.er[idx] + 0.7
    ds = np.linspace(0, 1, 4)
    for k in range(len(ss) - 1):
        for j in range(3):
            P4 = []
            for (kk, dd) in ((k, j), (k + 1, j), (k + 1, j + 1), (k, j + 1)):
                d = dr[kk] + (dl[kk] - dr[kk]) * ds[dd]; ii = idx[kk]
                x, y = road.xy(ii, d); z = float(road.surface_z(ii, d, noise=False)) + 0.028
                P4.append([float(x), float(y), z])
            acc.quad(P4, R["hump"], out=np.array([0, 0, 1.0]), uspan=(j / 3, (j + 1) / 3), vspan=(k / (len(ss) - 1), (k + 1) / (len(ss) - 1)))


# ================================================================== main
# 0. drop previous output + 21's own guardrail/wall meshes (regenerated below)
old = G.get("barriers", [])
pat = re.compile(r"^(guardrail_\d+|grpost_\d+|1WALL_gr\d+|1WALL_cut\d+|1WALL_start|1WALL_end)$")
G["barriers"] = [m for m in old if not pat.match(m["name"])]
print(f"  barriers: dropped {len(old) - len(G['barriers'])} meshes of 21 (guardrails/posts/1WALL_gr/cut/start/end), regenerating")
G.pop("roadside", None)

# 1. guardrails: 21's candidates minus suppressed spans, plus real W-beams
supp = collections.defaultdict(list)       # side -> [(s0,s1)]
for e in ASSETS:
    if e["type"] == "remove_guardrail":
        for sd in sides_of(e) if e.get("side") not in (None, "center") else ["left", "right"]:
            supp[sd].append((float(e["s0"]), float(e["s1"])))
ovf = DER / "roadside_overrides.json"
if ovf.exists():
    for e in json.load(open(ovf)).get("suppress", []):
        for sd in ([e["side"]] if e.get("side") in ("left", "right") else ["left", "right"]):
            supp[sd].append((float(e["s0"]), float(e["s1"])))
real = collections.defaultdict(list)
for e in ASSETS:
    if e["type"] == "guardrail_wbeam":
        for sd in sides_of(e):
            real[sd].append((float(e["s0"]), float(e["s1"])))


def subtract(iv, cuts):
    out = [list(iv)]
    for c0, c1 in cuts:
        nxt = []
        for a, b in out:
            if c1 <= a or c0 >= b:
                nxt.append([a, b]); continue
            if c0 > a:
                nxt.append([a, c0])
            if c1 < b:
                nxt.append([c1, b])
        out = nxt
    return [(a, b) for a, b in out if b - a >= 4.0]


n_cand = n_kept = 0; len_cand = len_kept = 0.0
for g in meta["guardrails"]:
    n_cand += 1; len_cand += g["s1"] - g["s0"]
    for a, b in subtract((g["s0"], g["s1"]), supp[g["side"]] + real[g["side"]]):
        do_guardrail(a, b, g["side"], 0.45, 0.80, False, "guardrail_candidate"); n_kept += 1; len_kept += b - a
print(f"  guardrail candidates: {n_cand} runs / {len_cand:.0f} m -> {n_kept} pieces / {len_kept:.0f} m kept "
      f"({len_cand - len_kept:.0f} m suppressed by remove_guardrail or replaced by real W-beam)")
n_real = 0; len_real = 0.0
for e in ASSETS:
    if e["type"] == "guardrail_wbeam":
        a = e.get("attrs") or {}
        for sd in sides_of(e):
            do_guardrail(float(e["s0"]), float(e["s1"]), sd, float(e.get("offset_m") or 0.8), min(max(float(e.get("height_m") or 0.75), 0.6), 0.85),
                         "wood" in str(a.get("posts", "")).lower(), "guardrail_wbeam"); n_real += 1; len_real += e["s1"] - e["s0"]
print(f"  real guardrail_wbeam: {n_real} runs / {len_real:.0f} m")

# 2. cut-bank walls (invisible) and end walls: closed boxes
for wi, w in enumerate(meta["walls"]):
    sgn = 1 if w["side"] == "left" else -1
    idx = np.arange(int(round(w["s0"])), int(round(w["s1"])) + 1); idx = np.clip(idx, 0, N - 1)
    e_ = road.el[idx] if sgn > 0 else road.er[idx]
    x, y = road.xy(idx, sgn * (e_ + road.sh_w(idx, sgn) + 1.2)); zb = road.surface_z(idx, sgn * e_, noise=False)
    ACC[chunk_of(w["s0"])].col.ribbon(x, y, zb - 1.0, zb + 4.0, 0.1)
for nm, i in (("1WALL_start", 1), ("1WALL_end", N - 2)):
    c = CO.Collider(nm)
    d = np.linspace(-road.outer(i, -1) - 1.8, road.outer(i, 1) + 1.8, 9)
    x, y = road.xy(np.full(9, i), d); c.ribbon(x, y, road.z[i] - 2.0, road.z[i] + 4.0, 0.15)
    ENDM = G.setdefault("roadside", []); ENDM += c.meshes()

# 3. assets
poles = {"left": [], "right": []}; signs = collections.Counter()
for e in ASSETS:
    t = e["type"]
    try:
        for sd in sides_of(e):
            if t in FENCE:
                do_fence(e, sd)
            elif t in WALLS:
                do_wall(e, sd)
            elif t == "rockfall_mesh":
                do_rockfall(e, sd)
            elif t == "utility_pole":
                do_pole(e, sd, poles)
            elif t == "road_sign":
                signs[do_sign(e, sd)] += 1
            elif t == "mailbox_cluster":
                do_mailbox(e, sd)
            elif t == "gate_estate":
                do_gate(e, sd)
            elif t == "building":
                do_building(e, sd)
            elif t == "cypress_row":
                do_cypress(e, sd)
            elif t == "delineator_post":
                do_delineator(e, sd)
            elif t == "driveway_apron":
                do_apron(e, sd)
        if t == "hump_marking":
            do_hump(e)
    except Exception as ex:                       # one bad entry must not lose the rest
        print(f"  WARN {t} s={e.get('s0')}: {type(ex).__name__}: {ex}")
do_wires(poles)

# 4. emit
out = G.setdefault("roadside", [])
nv = 0
for c in sorted(ACC):
    a = ACC[c]; m = a.mesh(f"rs_{c:02d}")
    if m:
        out.append(m); nv += len(m["P"])
    for cm in a.col.meshes():
        out.append(cm)
print(f"  collision flags fixed on {CO.fix_wall_flags(G)} 1WALL meshes (visible=True, renderable=False)")
(BUILD / "roadside_atlas.png").write_bytes(ATLAS.png())
json.dump(SOLID, open(BUILD / "roadside_solid.json", "w")); json.dump(SIGNCHK, open(BUILD / "roadside_signs.json", "w"))
json.dump(LOW_ASSETS, open(BUILD / "roadside_low_assets.json", "w"), indent=0)
print(f"  check: ground-referenced asset samples with base > {DROP_MAX} m below the road edge: {len(LOW_ASSETS)}"
      + (f" (first: {LOW_ASSETS[:3]})" if LOW_ASSETS else ""))
meta["roadside"] = dict(files=FILES, counts=dict(CNT), n_visual_meshes=sum(1 for m in out if m["material"] == "roadside"), verts=nv,
                        guardrail_pieces=n_kept, signs=dict(signs), collision_items=len(SOLID), skipped_far=dict(SKIPPED))
pickle.dump(geo, open(BUILD / "geometry.pkl", "wb"))
print(f"21c: roadside visual meshes {meta['roadside']['n_visual_meshes']} ({nv} verts), collision meshes "
      f"{sum(1 for m in out if m['name'].startswith('1WALL'))}, solid items {len(SOLID)}, far-skipped {dict(SKIPPED)}")
print("  asset counts:", dict(CNT))
print("  signs:", dict(signs))
