"""Phase 2: hill-climb dressing for the track_* layouts -> groups "dressing" (shared), "dressing_nb",
"dressing_sb" (direction-specific: distance boards) in build/geometry.pkl.

All branding is FICTIONAL and drawn here (no real marks): "Palomares Hill Climb" + invented sponsors.
Everything is procedural (PIL), one RGBA atlas build/dressing_atlas.png (material "dressing", alpha-tested
for the catch fence), so each group is ONE visual mesh; collision is ONE invisible 1WALL_dressing mesh.

  gates (s 20 / 7392 / 15563): banner gantry (>= 5.5 m clear), start-light post, timing hut on the
      flattest spot, 2-4 sponsor hoardings set back on flat ground, checker flag poles
  summit: small bleacher behind a debris/catch fence
  corners: barriers in front of the worst unguarded steep drops, set back 0.6 m behind the shoulder,
      following the road edge and the ground:  tyre wall + blue conveyor-belt facing | TecPro-style
      blue/red blocks | concrete jersey (F-shape) barriers, rotating
  distance boards: 100 m / 50 m before the 8 hardest corners per run direction (outside of the corner)
  marshal posts: ~every 1.8 km at flat spots near hard corners: hut + yellow flag + post number
Idempotent: replaces the dressing groups. Run after 21_build_geometry.py.
"""
import json, pickle, sys
import numpy as np, yaml
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import gaussian_filter1d, label
from scipy.spatial import cKDTree
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from geom import ROOT, DER, BUILD, SHOULDER_W, Road, Raster, vertex_normals

geo = pickle.load(open(BUILD / "geometry.pkl", "rb"))
G, meta = geo["groups"], geo["meta"]
for g in ("dressing", "dressing_nb", "dressing_sb"):
    G.pop(g, None)
road = Road(meta["s_end"])
N = len(road.s)
dtm05 = Raster(DER / "dtm_r050_b40.tif"); dtm1 = Raster(DER / "dtm_r100_b150.tif")
ctree = cKDTree(np.c_[road.x, road.y])
GATES = [20, 7392, 15563]
KCFG = yaml.safe_load(open(ROOT / "config/kerbs.yaml"))
rng = np.random.default_rng(21)


def dtm(x, y):
    z = dtm05.sample(x, y); z2 = dtm1.sample(x, y)
    return np.where(np.isfinite(z), z, z2)


# ================================================================== atlas (2048 RGBA, 8 x 8 tiles of 256 px)
S = 2048; TS = 256
A = Image.new("RGBA", (S, S), (120, 120, 120, 255)); dr = ImageDraw.Draw(A)
def F(sz, bold=True):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", sz)
    except OSError:
        return ImageFont.load_default()
UVT = {}
def region(name, x0, y0, x1, y1, pad=3):
    UVT[name] = ((x0 + pad) / S, (y0 + pad) / S, (x1 - pad) / S, (y1 - pad) / S)
def tilebox(c, r, w=1, h=1): return (c * TS, r * TS, (c + w) * TS, (r + h) * TS)
def grain(box, amt=10, seed=0):
    x0, y0, x1, y1 = box
    a = np.asarray(A.crop(box), float)
    n = np.random.default_rng(seed).normal(0, amt, a.shape[:2])[..., None]
    a[..., :3] = (a[..., :3] + n).clip(0, 255)
    A.paste(Image.fromarray(a.astype(np.uint8), "RGBA"), (x0, y0))

NAVY, RED, WHITE, GOLD = (16, 30, 66), (200, 32, 36), (245, 245, 242), (232, 178, 48)

def logo_mark(cx, cy, r, fg, bg):
    """fictional Palomares Hill Climb mark: roundel with a stylised ridge line + switchback"""
    dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fg)
    dr.ellipse((cx - r * .82, cy - r * .82, cx + r * .82, cy + r * .82), fill=bg)
    pts = [(cx - r * .7, cy + r * .3), (cx - r * .3, cy - r * .35), (cx, cy + r * .05), (cx + r * .3, cy - r * .5), (cx + r * .7, cy + r * .3)]
    dr.polygon(pts + [(cx + r * .7, cy + r * .45), (cx - r * .7, cy + r * .45)], fill=fg)
    dr.line([(cx - r * .45, cy + r * .62), (cx + r * .2, cy + r * .5), (cx - r * .1, cy + r * .35)], fill=bg, width=max(2, int(r * .08)))

# --- row 0: main banner (full width)
x0, y0, x1, y1 = 0, 0, S, TS
for k in range(TS):
    c = tuple(int(a + (b - a) * k / TS) for a, b in zip((22, 44, 96), NAVY)); dr.line((0, k, S, k), fill=c)
dr.rectangle((0, 0, S, 14), fill=RED); dr.rectangle((0, TS - 14, S, TS), fill=RED)
for i in range(3):   # checker ends
    for j in range(7):
        for xs in (i * 32 + 16, S - 16 - (i + 1) * 32):
            dr.rectangle((xs, 22 + j * 30, xs + 31, 22 + j * 30 + 29), fill=WHITE if (i + j) % 2 else (12, 12, 12))
logo_mark(250, 128, 82, WHITE, NAVY)
dr.text((1060, 104), "PALOMARES HILL CLIMB", fill=WHITE, font=F(118), anchor="mm")
dr.text((1060, 196), "ALAMEDA COUNTY  ·  NILES CANYON  →  CASTRO VALLEY", fill=GOLD, font=F(38, False), anchor="mm")
region("banner", 0, 0, S, TS)

# --- row 1: four fictional sponsor boards (512 x 256)
def sponsor(i, bg, fg, name, sub, mark):
    x0 = i * 512; dr.rectangle((x0, TS, x0 + 511, 2 * TS - 1), fill=bg)
    dr.rectangle((x0 + 8, TS + 8, x0 + 503, 2 * TS - 9), outline=fg, width=4)
    mark(x0 + 92, TS + 128, fg, bg)
    dr.text((x0 + 312, TS + 108), name, fill=fg, font=F(58 if len(name) < 10 else 46), anchor="mm")
    dr.text((x0 + 312, TS + 172), sub, fill=fg, font=F(26, False), anchor="mm")
    region(f"sponsor{i}", x0, TS, x0 + 512, 2 * TS)
def m_drop(cx, cy, fg, bg):  # oil drop
    dr.polygon([(cx, cy - 70), (cx + 48, cy + 10), (cx - 48, cy + 10)], fill=fg); dr.ellipse((cx - 48, cy - 30, cx + 48, cy + 66), fill=fg)
    dr.ellipse((cx - 18, cy + 4, cx + 6, cy + 30), fill=bg)
def m_tyre(cx, cy, fg, bg):
    dr.ellipse((cx - 66, cy - 66, cx + 66, cy + 66), fill=fg); dr.ellipse((cx - 34, cy - 34, cx + 34, cy + 34), fill=bg)
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        dr.line((cx + 40 * np.cos(a), cy + 40 * np.sin(a), cx + 64 * np.cos(a), cy + 64 * np.sin(a)), fill=bg, width=6)
def m_cup(cx, cy, fg, bg):
    dr.rounded_rectangle((cx - 46, cy - 30, cx + 34, cy + 56), 14, fill=fg); dr.arc((cx + 14, cy - 12, cx + 66, cy + 36), -90, 90, fill=fg, width=12)
    for k in (-20, 0, 20): dr.arc((cx + k - 10, cy - 78, cx + k + 10, cy - 38), 90, 270, fill=fg, width=5)
def m_clock(cx, cy, fg, bg):
    dr.ellipse((cx - 64, cy - 64, cx + 64, cy + 64), outline=fg, width=12)
    dr.line((cx, cy, cx, cy - 44), fill=fg, width=10); dr.line((cx, cy, cx + 34, cy + 16), fill=fg, width=10)
    dr.rectangle((cx - 14, cy - 86, cx + 14, cy - 68), fill=fg)
sponsor(0, (18, 18, 20), GOLD, "SUNOL RIDGE", "MOTOR OIL · EST. 1931", m_drop)
sponsor(1, (230, 230, 226), (20, 20, 20), "CANYON TYRE", "GRIP FOR EVERY BEND", m_tyre)
sponsor(2, (92, 52, 30), (246, 232, 206), "EDEN VALLEY", "ROASTERS · CASTRO VALLEY", m_cup)
sponsor(3, (200, 32, 36), WHITE, "REDTAIL", "TIMING SYSTEMS", m_clock)

# --- row 2: barrier materials
def t_galv(b):
    dr.rectangle(b, fill=(160, 164, 168)); grain(b, 9, 1)
reg_row = 2
b = tilebox(0, 2); t_galv(b); region("metal", *b)
b = tilebox(1, 2)                                   # blue conveyor belt facing with bolt rows + seam
dr.rectangle(b, fill=(28, 70, 150)); grain(b, 7, 2)
for yb in (30, 128, 226):
    for xb in range(16, 256, 48): dr.ellipse((b[0] + xb - 5, b[1] + yb - 5, b[0] + xb + 5, b[1] + yb + 5), fill=(170, 175, 185))
dr.line((b[0], b[1] + 2, b[2], b[1] + 2), fill=(12, 30, 70), width=4)
for k in range(0, 256, 6): dr.line((b[0] + k, b[1] + 200, b[0] + k + 30, b[1] + 255), fill=(40, 40, 42), width=2)  # rubber scuffs
region("belt", *b)
b = tilebox(2, 2)                                   # tyre stack side: 4 tyres high, treads
dr.rectangle(b, fill=(18, 18, 18))
for k in range(4):
    y = b[1] + k * 64
    dr.rectangle((b[0], y + 6, b[2], y + 58), fill=(30, 30, 30))
    for xx in range(b[0], b[2], 10): dr.line((xx, y + 10, xx + 5, y + 54), fill=(12, 12, 12), width=3)
grain(b, 6, 3); region("tyre_side", *b)
b = tilebox(3, 2)                                   # tyre tops (looking down into the stack)
dr.rectangle(b, fill=(16, 16, 16))
for cx in (64, 192):
    for cy in (64, 192):
        dr.ellipse((b[0] + cx - 62, b[1] + cy - 62, b[0] + cx + 62, b[1] + cy + 62), fill=(34, 34, 34))
        dr.ellipse((b[0] + cx - 34, b[1] + cy - 34, b[0] + cx + 34, b[1] + cy + 34), fill=(6, 6, 6))
region("tyre_top", *b)
for ci, (nm, col) in enumerate((("tecpro_blue", (20, 66, 160)), ("tecpro_red", (196, 30, 34)))):
    b = tilebox(4 + ci, 2); dr.rectangle(b, fill=col)
    for yy in (b[1] + 18, b[3] - 30): dr.rectangle((b[0], yy, b[2], yy + 12), fill=tuple(int(c * .7) for c in col))   # moulded ribs
    logo_mark(b[0] + 128, b[1] + 104, 54, WHITE, col)
    dr.text((b[0] + 128, b[1] + 196), "PHC", fill=WHITE, font=F(40), anchor="mm")
    grain(b, 5, 4 + ci); region(nm, *b)
b = tilebox(6, 2); dr.rectangle(b, fill=(176, 172, 164)); grain(b, 12, 6)  # concrete
for k in range(5): dr.line((b[0] + rng.integers(0, 256), b[1], b[0] + rng.integers(0, 256), b[3]), fill=(150, 146, 140), width=1)
dr.rectangle((b[0], b[3] - 26, b[2], b[3]), fill=(128, 118, 100))             # ground stain at the foot
region("concrete", *b)
b = tilebox(7, 2)
for k in range(8):
    dr.rectangle((b[0], b[1] + k * 32, b[2], b[1] + k * 32 + 29), fill=(150 + 8 * (k % 3), 110, 70))
    dr.line((b[0], b[1] + k * 32 + 30, b[2], b[1] + k * 32 + 30), fill=(70, 50, 30), width=2)
grain(b, 8, 7); region("wood", *b)

# --- row 3: boards, marshal, flags
for ci, txt in enumerate(("100", "50")):
    b = tilebox(ci, 3); dr.rectangle(b, fill=WHITE)
    dr.rectangle((b[0] + 8, b[1] + 8, b[2] - 9, b[3] - 9), outline=(20, 20, 20), width=10)
    dr.text((b[0] + 128, b[1] + 132), txt, fill=(15, 15, 15), font=F(150 if len(txt) < 3 else 120), anchor="mm")
    region(f"board{txt}", *b)
b = tilebox(2, 3); dr.rectangle(b, fill=WHITE); dr.rectangle((b[0], b[1] + 90, b[2], b[1] + 150), fill=(240, 120, 20))
dr.text((b[0] + 128, b[1] + 120), "MARSHAL", fill=WHITE, font=F(40), anchor="mm"); grain(b, 4, 8); region("marshal", *b)
b = tilebox(3, 3); dr.rectangle(b, fill=(232, 232, 226)); grain(b, 6, 9); region("canvas", *b)
b = tilebox(4, 3); dr.rectangle(b, fill=(250, 214, 0)); region("flag_yellow", *b)
b = tilebox(5, 3)
for i in range(8):
    for j in range(8): dr.rectangle((b[0] + i * 32, b[1] + j * 32, b[0] + i * 32 + 31, b[1] + j * 32 + 31), fill=(15, 15, 15) if (i + j) % 2 else WHITE)
region("flag", *b)
b = tilebox(6, 3); dr.rectangle(b, fill=(46, 48, 52)); grain(b, 6, 10); region("dark", *b)
b = tilebox(7, 3)                                   # roof: red/white awning stripes
for k in range(8): dr.rectangle((b[0] + k * 32, b[1], b[0] + k * 32 + 31, b[3]), fill=RED if k % 2 else WHITE)
region("awning", *b)

# --- row 4: catch fence (alpha, 4 tiles = 1024 x 256 = one 4 x 1 m panel), start lights, number plates
b = tilebox(0, 4, 4, 1)
fence = np.zeros((TS, 4 * TS, 4), np.uint8)
yy, xx = np.mgrid[0:TS, 0:4 * TS]
mesh = (np.abs(((xx + yy) % 24) - 12) < 1.6) | (np.abs(((xx - yy) % 24) - 12) < 1.6)
fence[mesh] = (175, 178, 182, 255)
for xp in (0, 4 * TS - 10): fence[:, xp:xp + 10] = (110, 112, 116, 255)     # posts at the panel ends
fence[:6] = fence[-6:] = (110, 112, 116, 255); fence[TS // 2 - 2:TS // 2 + 2] = (140, 142, 146, 255)
A.paste(Image.fromarray(fence, "RGBA"), (b[0], b[1])); region("fence", *b, pad=1)
b = tilebox(4, 4); dr.rectangle(b, fill=(14, 14, 14))
for k, col in enumerate(((230, 30, 30), (230, 30, 30), (60, 20, 20), (30, 200, 60))):   # 2 red lit, amber off, green
    cy = b[1] + 34 + k * 62; dr.ellipse((b[0] + 98, cy - 26, b[0] + 158, cy + 26), fill=col)
region("lights", *b)
for k in range(12):                                 # marshal post numbers (row 5) and corner numbers T1..T12 (row 6)
    for row, txt, bg, fg in ((5, f"{k + 1}", (240, 120, 20), WHITE), (6, f"T{k + 1}", (16, 30, 66), WHITE)):
        cx0 = k * 168 + 4; cy0 = row * TS + 50
        bb = (cx0, cy0, cx0 + 156, cy0 + 156); dr.rectangle(bb, fill=bg); dr.rectangle((cx0 + 6, cy0 + 6, cx0 + 150, cy0 + 150), outline=fg, width=5)
        dr.text((cx0 + 78, cy0 + 80), txt, fill=fg, font=F(96 if len(txt) < 3 else 70), anchor="mm")
        region(("post" if row == 5 else "turn") + f"{k + 1}", *bb)
b7 = tilebox(0, 7); dr.rectangle(b7, fill=WHITE); dr.rectangle((8, 7 * TS + 8, TS - 9, 8 * TS - 9), outline=(20, 20, 20), width=10)
dr.text((128, 7 * TS + 132), "150", fill=(15, 15, 15), font=F(120), anchor="mm"); region("board150", *b7)
for c0, txt, bg in ((1, "START", (20, 120, 50)), (3, "FINISH", (15, 15, 15))):
    bb = tilebox(c0, 7, 2, 1); dr.rectangle(bb, fill=bg)
    if txt == "FINISH":
        for i in range(16):
            for j, y_ in enumerate((0, 224)): dr.rectangle((bb[0] + i * 32, bb[1] + y_, bb[0] + i * 32 + 31, bb[1] + y_ + 31), fill=WHITE if (i + j) % 2 else (15, 15, 15))
    dr.text((bb[0] + 256, bb[1] + 128), txt, fill=WHITE, font=F(120), anchor="mm"); region(txt.lower(), *bb)
A.transpose(Image.FLIP_TOP_BOTTOM).save(BUILD / "dressing_atlas.png")   # kn5 v runs bottom-up; UVT uses top-down

# ================================================================== geometry helpers
class Bucket:
    def __init__(self): self.P, self.T, self.U = [], [], []; self.n = 0
    def quad(self, P4, uvr):
        u0, v0, u1, v1 = uvr
        self.P.append(np.asarray(P4, float)); self.U.append(np.array([[u0, v1], [u1, v1], [u1, v0], [u0, v0]]))
        self.T.append(np.array([[0, 1, 2], [0, 2, 3]]) + self.n); self.n += 4
B = {"dressing": Bucket(), "dressing_nb": Bucket(), "dressing_sb": Bucket()}
CP, CT = [], []              # collision (shared)
cur = B["dressing"]


def quad(P4, uvr, out=None):
    """P4 = BL, BR, TR, TL seen from the front. With out=, flip so the face normal points along out."""
    P4 = np.asarray(P4, float)
    if out is not None:
        nrm = np.cross(P4[1] - P4[0], P4[3] - P4[0])
        if np.dot(nrm, out) < 0:
            P4 = P4[[1, 0, 3, 2]]
    cur.quad(P4, uvr)


def panel(c, z0, z1, look, w, uvr, back=None):
    """vertical panel centred at xy c, seen by a viewer looking along 'look'; optional back face"""
    look = np.asarray(look, float) / np.hypot(*look); R = np.array([look[1], -look[0]])
    a = np.asarray(c) - R * w / 2; b_ = np.asarray(c) + R * w / 2
    quad([[*a, z0], [*b_, z0], [*b_, z1], [*a, z1]], uvr)
    if back:
        quad([[*b_, z0], [*a, z0], [*a, z1], [*b_, z1]], back)


def box(c, fwd, L, W, z0, z1, uvr, coll=False, top=None, faces=None):
    """box centred at xy c, long axis fwd, length L, width W, z0..z1. faces: {'r','f','l','b'} side tiles
    (r = right of fwd, f = +fwd end)."""
    f = np.array([fwd[0], fwd[1], 0.0]) / np.hypot(*fwd); l = np.array([-f[1], f[0], 0.0])
    c = np.array([c[0], c[1], 0.0]); faces = faces or {}
    def p(a, b, z): return c + f * a * L / 2 + l * b * W / 2 + np.array([0, 0, z])
    Bv = [p(-1, -1, z0), p(1, -1, z0), p(1, 1, z0), p(-1, 1, z0)]
    U = [p(-1, -1, z1), p(1, -1, z1), p(1, 1, z1), p(-1, 1, z1)]
    for key, q, o in (("r", [Bv[0], Bv[1], U[1], U[0]], -l), ("f", [Bv[1], Bv[2], U[2], U[1]], f),
                      ("l", [Bv[2], Bv[3], U[3], U[2]], l), ("b", [Bv[3], Bv[0], U[0], U[3]], -f)):
        quad(q, faces.get(key, uvr), out=o)
    quad([U[0], U[1], U[2], U[3]], top or uvr, out=np.array([0, 0, 1.0]))
    if coll:
        o = sum(len(q) for q in CP); CP.append(np.array(Bv + U))
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
def edge(i, side): return road.el[i] if side > 0 else road.er[i]
def shw(i, side):
    """shoulder width at station i (variable per station if geom.Road provides sh_w_l / sh_w_r)"""
    if hasattr(road, "sh_w"):
        return float(np.asarray(road.sh_w(i, side)))
    a = getattr(road, "sh_w_l" if side > 0 else "sh_w_r", None)
    return float(a[i]) if a is not None else SHOULDER_W
def gz(x, y, fallback):
    z = dtm(x, y); return float(z) if np.isfinite(z) else float(fallback)


def clear_of_road(pts, margin):
    d, j = ctree.query(pts)
    half = np.maximum(road.outer(j, 1), road.outer(j, -1))   # per-station outer shoulder edge
    return bool(np.all(d >= half + margin))


def footprint_ok(cx, cy, fwd, L, W, margin=1.5, maxrange=0.8):
    f = fwd / np.linalg.norm(fwd); l = np.array([-f[1], f[0]])
    g = np.array([[a, b] for a in np.linspace(-.5, .5, 5) for b in np.linspace(-.5, .5, 4)])
    pts = np.array([cx, cy]) + g[:, :1] * L * f + g[:, 1:] * W * l
    z = dtm(pts[:, 0], pts[:, 1])
    if not np.all(np.isfinite(z)) or not clear_of_road(pts, margin):
        return None
    return (z.max() - z.min(), z.min(), z.max()) if z.max() - z.min() <= maxrange else None


def flat_spot(s0, s1, step, offs, L, W, sides=(1, -1), margin=1.5, maxrange=0.8, wo=0.05, ws=0.01, s_ref=None, avoid=()):
    best = None; s_ref = s0 if s_ref is None else s_ref
    for s in np.arange(s0, s1 + 0.01, step):
        j = idx(s)
        for side in sides:
            for off in offs:
                x, y = road.xy(j, side * (road.outer(j, side) + off))
                if any(np.hypot(x - ax, y - ay) < ar for ax, ay, ar in avoid):
                    continue
                r = footprint_ok(x, y, tangent(j), L, W, margin, maxrange)
                if r:
                    sc = r[0] + wo * off + ws * abs(s - s_ref)
                    if best is None or sc < best[0]:
                        best = (sc, x, y, j, side, s, off, r)
    return best


LOG = []
USED = []   # (x, y, r) footprints already taken
# ================================================================== gates
for gi, sg in enumerate(GATES):
    i = idx(sg); t = tangent(i); n = normal(i)
    dl = road.outer(i, 1) + 1.0; drr = road.outer(i, -1) + 1.0
    zroad = max(road.surface_z(i, d, noise=False) for d in np.linspace(-road.er[i], road.el[i], 9))
    zb = zroad + 5.6                                        # beam underside: >= 5.5 m clear of the crown
    legs = []
    for d in (dl, -drr):
        x, y = road.xy(i, d); zg = min(gz(x, y, road.z[i]), road.surface_z(i, d, noise=False))
        box((x, y), t, 0.45, 0.45, zg - 0.4, zb + 1.5, UVT["metal"], coll=True)
        box((x, y), t, 0.9, 0.9, zg - 0.1, zg + 0.12, UVT["concrete"])          # footing
        legs.append((x, y, zg)); USED.append((x, y, 2.0))
    mid = np.mean([l[:2] for l in legs], 0); span = dl + drr + 0.45
    box(mid, n, span, 0.35, zb, zb + 0.18, UVT["metal"])                     # truss chords
    box(mid, n, span, 0.35, zb + 1.32, zb + 1.5, UVT["metal"])
    hw = (dl + drr) / 2 - 0.25
    for sgn in (1, -1):                                     # banner both faces, readable from each direction
        panel(mid + sgn * t * 0.19, zb + 0.18, zb + 1.32, -sgn * t, 2 * hw, UVT["banner"])
    LOG.append(f"gantry s={sg} span {span:.1f} m clear {zb - zroad:.1f} m")
    # start lights on the left leg (both faces)
    x, y, zg = legs[0]
    for sgn in (1, -1):
        panel(np.array([x, y]) + sgn * t * 0.24, zg + 2.2, zg + 3.4, -sgn * t, 0.5, UVT["lights"])
    # timing hut
    best = flat_spot(sg - 30, sg + 10, 2, np.arange(2.5, 10.1, 0.5), 3.0, 2.4, s_ref=sg, avoid=USED)
    if best:
        _, x, y, j, side, s, off, r = best
        box((x, y), tangent(j), 3.0, 2.4, r[1] - 0.3, r[2] + 2.3, UVT["canvas"], coll=True, top=UVT["metal"],
            faces={"r" if side > 0 else "l": UVT["marshal"]})
        box((x, y), tangent(j), 3.4, 2.8, r[2] + 2.3, r[2] + 2.45, UVT["awning"])
        USED.append((x, y, 3.5)); LOG.append(f"  timing hut s={s:.0f} {'L' if side > 0 else 'R'} {off:.1f} m past shoulder")
    # sponsor hoardings (4 x 1 m on legs), set back 1.5-5 m, facing the road
    for k, (a_, b_) in enumerate(((sg - 45, sg - 12), (sg + 12, sg + 45))):
        best = flat_spot(max(a_, 3), min(b_, N - 3), 3, np.arange(1.5, 5.1, 0.5), 4.2, 0.6, maxrange=0.6, avoid=USED)
        if not best:
            continue
        _, x, y, j, side, s, off, r = best
        look = -side * normal(j)                            # viewer on the road looks outward toward the board
        sp = UVT[f"sponsor{(gi * 2 + k) % 4}"]
        panel((x, y), r[2] + 0.35, r[2] + 1.35, look, 4.0, sp, back=UVT["dark"])
        for e_ in (-1.85, 1.85):
            box(np.array([x, y]) + tangent(j) * e_, tangent(j), 0.08, 0.08, r[1] - 0.3, r[2] + 1.35, UVT["dark"])
        CP_x = np.array([x, y]) + np.outer([-2, 2], tangent(j))
        strip_wall(CP_x[:, 0], CP_x[:, 1], np.full(2, r[1] - 0.3), np.full(2, r[2] + 1.35))
        USED.append((x, y, 3.0)); LOG.append(f"  hoarding s={s:.0f} {'L' if side > 0 else 'R'} +{off:.1f} m")
    best = flat_spot(sg - 40, sg + 25, 3, np.arange(2.5, 6.1, 0.5), 12.0, 1.0, maxrange=0.8, s_ref=sg, avoid=USED)
    if best:                                                # spectator area: 3 x 4 m catch-fence panels, 2 m tall
        _, x, y, j, side, s, off, r = best
        t_ = tangent(j)
        for k in (-1, 0, 1):
            c = np.array([x, y]) + t_ * 4.0 * k; zz = gz(*c, r[1])
            for h in range(2):
                panel(c, zz - 0.1 + h, zz + 0.9 + h, side * normal(j), 4.0, UVT["fence"], back=UVT["fence"])
            box(c - t_ * 2.0, t_, 0.08, 0.08, zz - 0.3, zz + 2.0, UVT["dark"])
        ends = np.array([x, y]) + np.outer([-6, 6], t_)
        strip_wall(ends[:, 0], ends[:, 1], np.full(2, r[1] - 0.5), np.full(2, r[2] + 2.0))
        USED.append((x, y, 7.0)); LOG.append(f"  spectator fence s={s:.0f} {'L' if side > 0 else 'R'} +{off:.1f} m")
    for (x, y, zg), d in zip(legs, (1, -1)):                # checker flag poles just outside the legs
        p = np.array([x, y]) + d * n * 0.9 - t * 1.0
        zz = gz(*p, road.z[i])
        box(p, t, 0.08, 0.08, zz - 0.3, zz + 4.0, UVT["metal"])
        q = p - t * 1.2
        for P4 in ([[*p, zz + 3.0], [*q, zz + 3.0], [*q, zz + 3.9], [*p, zz + 3.9]],
                   [[*q, zz + 3.0], [*p, zz + 3.0], [*p, zz + 3.9], [*q, zz + 3.9]]):
            quad(P4, UVT["flag"])

# ================================================================== summit bleacher + catch fence
best = flat_spot(7392 - 60, 7392 + 60, 3, np.arange(4.0, 20.1, 1.0), 8.0, 3.2, margin=3.0, maxrange=1.2, wo=0.04, ws=0.004,
                 s_ref=7392, avoid=USED)
if best:
    _, x, y, j, side, s, off, r = best
    t = tangent(j); nn = normal(j) * side               # nn points away from the road
    for k in range(3):
        c = np.array([x, y]) + nn * (k - 1) * 1.05
        box(c, t, 8.0, 1.05, r[1] - 0.4, r[2] + 0.45 * (k + 1), UVT["wood"], coll=True)
    # debris fence between road and bleacher: 3 panels of 4 m, 3 m tall, double sided (alpha)
    fc = np.array([x, y]) - nn * 2.6
    for k in (-1, 0, 1):
        c = fc + t * 4.0 * k; zz = gz(*c, r[1])
        for h in range(3):                              # 4 x 1 m texture panels, 3 high
            panel(c, zz - 0.1 + h, zz + 0.9 + h, nn, 4.0, UVT["fence"], back=UVT["fence"])
        box(c - t * 2.0, t, 0.1, 0.1, zz - 0.3, zz + 3.0, UVT["dark"])
    box(fc + t * 6.0, t, 0.1, 0.1, gz(*(fc + t * 6.0), r[1]) - 0.3, gz(*(fc + t * 6.0), r[1]) + 3.0, UVT["dark"])
    ends = np.array([fc - t * 6, fc + t * 6]); strip_wall(ends[:, 0], ends[:, 1], np.full(2, r[1] - 0.5), np.full(2, r[2] + 3.0))
    USED.append((x, y, 8.0))
    LOG.append(f"bleacher + catch fence s={s:.0f} {'L' if side > 0 else 'R'} {off:.0f} m past shoulder (ground range {r[0]:.2f} m)")
else:
    LOG.append("bleacher: no flat spot near the summit gate, skipped")

# ================================================================== corner barriers in front of steep drops
Fe = json.load(open(DER / "features.geojson"))["features"]
drops = [f["properties"] for f in Fe if f["properties"].get("kind") == "steep_drop"]
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
    outside = -kk.min() if side == "left" else kk.max()
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
KINDS = ("tyre_belt", "tecpro", "jersey")
JERSEY = np.array([[0, 0], [0.06, 0.08], [0.17, 0.32], [0.2, 0.81], [0.4, 0.81], [0.43, 0.32], [0.54, 0.08], [0.6, 0]])
for r_i, (s0, s1, side, kmax) in enumerate(picked):
    kind = KINDS[r_i % 3]
    sgn = 1 if side == "left" else -1
    sa, sb = s0 - 6, s1 + 4
    pitch = {"tyre_belt": 1.0, "tecpro": 1.5, "jersey": 3.0}[kind]
    ss = np.arange(sa, sb + 0.01, pitch)
    jj = np.array([idx(s) for s in ss])
    e = road.el[jj] if sgn > 0 else road.er[jj]
    ze = road.surface_z(jj, sgn * (e + road.sh_w(jj, sgn)), noise=False)       # shoulder outer edge (barrier top reference)
    setback = 0.6
    dface = sgn * (e + road.sh_w(jj, sgn) + setback)                            # road-facing face line
    xf, yf = road.xy(jj, dface)
    zg = dtm(xf, yf); zg = np.where(np.isfinite(zg), zg, ze)
    zbase = np.minimum(zg, ze) - 0.12
    out = np.c_[sgn * road.nx[jj], sgn * road.ny[jj]]                  # away from the road
    for q in range(len(ss) - 1):
        pa = np.array([xf[q], yf[q]]); pb = np.array([xf[q + 1], yf[q + 1]])
        fwd = pb - pa; L = np.hypot(*fwd)
        if L < 0.2: continue
        o = (out[q] + out[q + 1]) / 2; o /= np.hypot(*o)
        zb0 = min(zbase[q], zbase[q + 1]); ztop = max(ze[q], ze[q + 1])
        rf = "r" if sgn > 0 else "l"                                    # face toward the road (fwd = +s)
        if kind == "tyre_belt":
            depth = 0.75; c = (pa + pb) / 2 + o * depth / 2
            box(c, fwd, L * 0.995, depth, zb0, ztop + 0.95, UVT["tyre_side"], top=UVT["tyre_top"], faces={rf: UVT["belt"]})
        elif kind == "tecpro":
            depth = 0.8; c = (pa + pb) / 2 + o * depth / 2
            tile = UVT["tecpro_blue" if q % 2 == 0 else "tecpro_red"]
            box(c, fwd, L - 0.03, depth, zb0, ztop + 1.0, tile, top=tile)
        else:                                                           # jersey: extruded F-shape, 3 m units
            Lu = L - 0.02; pa_ = pa + fwd / L * 0.01; pb_ = pa_ + fwd / L * Lu
            z0a, z0b = zbase[q] + 0.1, zbase[q + 1] + 0.1
            zta, ztb = max(ze[q], z0a), max(ze[q + 1], z0b)
            prof = JERSEY
            def P(pp, z0, k): return [*(pp + o * prof[k, 0]), z0 + prof[k, 1]]
            for k in range(len(prof) - 1):
                cen = np.r_[(pa_ + pb_) / 2 + o * 0.3, (zta + ztb) / 2 + 0.4]
                Q = [P(pa_, zta, k), P(pb_, ztb, k), P(pb_, ztb, k + 1), P(pa_, zta, k + 1)]
                quad(Q, UVT["concrete"], out=np.mean(Q, 0) - cen)
            # skirt from the jersey foot down to the ground (on falling ground)
            for k in (0, len(prof) - 1):
                Q = [P(pa_, z0a - 0.3, k), P(pb_, z0b - 0.3, k), P(pb_, ztb, k), P(pa_, zta, k)]
                quad(Q, UVT["concrete"], out=np.r_[(-o if k == 0 else o), 0])
    strip_wall(xf, yf, zbase - 0.3, np.maximum(ze, zg) + 1.1)
    LOG.append(f"{kind} s={sa:.0f}-{sb:.0f} {side[0].upper()} (|k|max {kmax:.3f} 1/m, R~{1 / max(kmax, 1e-3):.0f} m)")

# ================================================================== distance boards (direction-specific)
ks = gaussian_filter1d(road.k, KCFG["sigma_m"])
lab, nl = label(np.abs(ks) > KCFG["curvature_threshold_1pm"])
corners = []
for l in range(1, nl + 1):
    ix = np.nonzero(lab == l)[0]
    if len(ix) >= KCFG["min_corner_length_m"]:
        corners.append((float(np.abs(ks[ix]).max()), ix[0], ix[-1], 1 if ks[ix[np.argmax(np.abs(ks[ix]))]] > 0 else -1))
NB = 8
BOARDED = []
for dname, sd in (("nb", 1), ("sb", -1)):
    cur = B[f"dressing_{dname}"]; placed = []
    for kpk, i0, i1, inner in sorted(corners, reverse=True):
        if len(placed) >= NB: break
        entry = road.s[i0] if sd > 0 else road.s[i1]
        st = [entry - sd * 150, entry - sd * 100, entry - sd * 50]
        if min(st) < 40 or max(st) > road.s[-1] - 40 or any(abs(entry - p) < 300 for p in placed):
            continue
        # skip if another (sharp) corner sits between the board and the corner (boards would be misleading)
        if any(c[0] > 0.6 * kpk and ((sd > 0 and st[0] < road.s[c[2]] < entry - 5) or (sd < 0 and entry + 5 < road.s[c[1]] < st[0]))
               for c in corners):
            continue
        placed.append(entry)
        for s_b, txt in zip(st, ("150", "100", "50")):
            j = idx(s_b); runv = sd * tangent(j)
            best = None
            for side in (-inner, inner):                                    # prefer the outside of the corner
                for off in (0.9, 0.5):
                    x, y = road.xy(j, side * (road.outer(j, side) + off))
                    ze_ = road.surface_z(j, side * (road.outer(j, side)), noise=False)
                    zz = gz(x, y, ze_)
                    if abs(zz - ze_) < 0.8:
                        best = (x, y, zz); break
                if best: break
            if not best:
                side = -inner; x, y = road.xy(j, side * (road.outer(j, side) + 0.3))
                best = (x, y, road.surface_z(j, side * (road.outer(j, side)), noise=False) - 0.1)
            x, y, zz = best
            R_ = np.array([runv[1], -runv[0]])                          # breakaway style: two thin posts, notched
            for o_ in (-0.25, 0.25):
                box(np.array([x, y]) + R_ * o_, runv, 0.04, 0.04, zz - 0.4, zz + 0.25, UVT["dark"])
                box(np.array([x, y]) + R_ * o_, runv, 0.035, 0.035, zz + 0.27, zz + 1.75, UVT["metal"])
            panel(np.array([x, y]) - runv * 0.03, zz + 1.0, zz + 1.75, runv, 0.75, UVT[f"board{txt}"], back=UVT["metal"])
        BOARDED.append((entry, i0, i1, inner, best))
        LOG.append(f"  boards {dname} corner entry s={entry:.0f} (R~{1 / kpk:.0f} m)")
    # corner numbers T1.. in run order, small plate at the corner entry on the outside
    for tn, (entry, i0, i1, inner, _) in enumerate(sorted(BOARDED, key=lambda b_: sd * b_[0])):
        j = idx(entry); runv = sd * tangent(j); side = -inner
        x, y = road.xy(j, side * (edge(j, side) + shw(j, side) + 0.6))
        zz = gz(x, y, road.surface_z(j, side * edge(j, side), noise=False))
        ze_ = road.surface_z(j, side * (edge(j, side) + shw(j, side)), noise=False)
        if abs(zz - ze_) > 0.8: zz = ze_ - 0.1
        box((x, y), runv, 0.05, 0.05, zz - 0.4, zz + 1.3, UVT["dark"])
        panel(np.array([x, y]) - runv * 0.03, zz + 0.85, zz + 1.3, runv, 0.45, UVT[f"turn{tn + 1}"], back=UVT["metal"])
        LOG.append(f"  corner T{tn + 1} {dname} s={entry:.0f}")
    BOARDED.clear()
GATE_ROLE = {"nb": {20: "start", 7392: "finish", 15563: "finish"}, "sb": {15563: "start", 7392: "finish", 20: "finish"}}
for dname, sd in (("nb", 1), ("sb", -1)):
    cur = B[f"dressing_{dname}"]
    for g, role in GATE_ROLE[dname].items():
        s_b = g - sd * 6; j = idx(s_b); runv = sd * tangent(j)
        side = -sd                                                      # driver's right
        x, y = road.xy(j, side * (edge(j, side) + shw(j, side) + 0.9))
        ze_ = road.surface_z(j, side * (edge(j, side) + shw(j, side)), noise=False)
        zz = gz(x, y, ze_); zz = zz if abs(zz - ze_) < 0.8 else ze_ - 0.1
        for o_ in (-0.7, 0.7):
            box(np.array([x, y]) + np.array([runv[1], -runv[0]]) * o_, runv, 0.06, 0.06, zz - 0.4, zz + 2.0, UVT["dark"])
        panel(np.array([x, y]) - runv * 0.04, zz + 1.3, zz + 2.0, runv, 1.6, UVT[role], back=UVT["metal"])
        LOG.append(f"  {role.upper()} board {dname} s={s_b:.0f}")
cur = B["dressing"]

# ================================================================== marshal posts (shared)
zs = gaussian_filter1d(road.z, 8); vc = np.gradient(np.gradient(zs))         # vertical curvature (1/m)
crest_lab, ncr = label(vc < -0.0025)
crests = sorted(((float(-vc[ix].min()), ix[np.argmin(vc[ix])]) for ix in (np.nonzero(crest_lab == l)[0] for l in range(1, ncr + 1))), reverse=True)
targets = []
for v_, ic in crests:
    s_c = float(road.s[ic])
    if 300 < s_c < road.s[-1] - 300 and all(abs(s_c - t_) > 1000 for t_ in targets) and all(abs(s_c - g) > 300 for g in GATES):
        targets.append(s_c); LOG.append(f"  blind crest s={s_c:.0f} (vert. curvature {v_:.4f} 1/m)")
    if len(targets) >= 6: break
hard = sorted(corners, reverse=True)
for kpk, i0, i1, inner in hard:
    s_c = float(road.s[(i0 + i1) // 2])
    if 300 < s_c < road.s[-1] - 300 and all(abs(s_c - t_) > 1200 for t_ in targets) and len(targets) < 10 and all(abs(s_c - g) > 300 for g in GATES):
        targets.append(s_c)
targets.sort()
for k, s_c in enumerate(targets[:10]):
    best = flat_spot(s_c - 60, s_c + 60, 3, np.arange(2.0, 8.1, 1.0), 2.0, 2.0, maxrange=0.6, s_ref=s_c, avoid=USED)
    if not best:
        LOG.append(f"  marshal post near s={s_c:.0f}: no flat spot"); continue
    _, x, y, j, side, s, off, r = best
    t = tangent(j); towards = -side * normal(j)
    rf = "r" if side > 0 else "l"
    box((x, y), t, 1.4, 1.2, r[1] - 0.3, r[2] + 2.1, UVT["canvas"], coll=True, top=UVT["metal"], faces={rf: UVT["marshal"]})
    box((x, y), t, 1.7, 1.5, r[2] + 2.1, r[2] + 2.22, UVT["awning"])
    if f"post{k + 1}" in UVT:
        panel(np.array([x, y]) + towards * 0.62, r[2] + 1.0, r[2] + 1.5, -towards, 0.5, UVT[f"post{k + 1}"])
    fp = np.array([x, y]) + towards * 1.0 + t * 0.9; zz = gz(*fp, r[1])
    box(fp, t, 0.05, 0.05, zz - 0.3, zz + 2.6, UVT["dark"])
    q_ = fp + towards * 0.0 + t * 0.9
    for P4 in ([[*fp, zz + 1.9], [*q_, zz + 1.9], [*q_, zz + 2.55], [*fp, zz + 2.55]],
               [[*q_, zz + 1.9], [*fp, zz + 1.9], [*fp, zz + 2.55], [*q_, zz + 2.55]]):
        quad(P4, UVT["flag_yellow"])
    USED.append((x, y, 2.5)); LOG.append(f"  marshal post {k + 1} s={s:.0f} {'L' if side > 0 else 'R'} +{off:.0f} m")

# ================================================================== write
for g, bk in B.items():
    if not bk.P: continue
    P = np.concatenate(bk.P); T = np.concatenate(bk.T); UV = np.concatenate(bk.U)
    assert len(P) < 65535, (g, len(P))
    G[g] = [dict(name=g, material="dressing", P=P, T=T, UV=UV, N=vertex_normals(P, T))]
Pc = np.concatenate(CP); Tc = np.concatenate(CT)
G["dressing"].append(dict(name="1WALL_dressing", material="invisible", P=Pc, T=Tc, UV=np.zeros((len(Pc), 2)),
                          N=vertex_normals(Pc, Tc), visible=False, renderable=False, cast_shadows=False))
assert len(Pc) < 65535, len(Pc)
meta["dressing"] = LOG
pickle.dump(geo, open(BUILD / "geometry.pkl", "wb"))
print("\n".join(LOG))
print("dressing:", {g: sum(len(m["P"]) for m in G.get(g, [])) for g in B}, "verts;", len(Pc), "collision")
