"""dev5 asset catalog: procedural low-poly generators for every ledger asset type (config/asset_catalog.yaml).

Every texture is painted here or by scripts/p2/roadside_tex.py (procedural, licence-free; MUTCD sign layouts are
public-domain designs redrawn). ONE RGBA atlas + ONE alpha-tested material ("ledger") => one draw call per 200 m chunk.

Local frame of a generated asset: x = forward (along the road, northbound), y = outward (away from the road),
z = up, origin on the ground at the anchor point. Linear assets (fence, guardrail, wall, hedge, marking...) span
x in [0, L] and are bent along the road by 21e. Signs face -x (toward an approaching northbound driver) by default.

    gen(L_or_None, params, rng) -> MB   (MB.col: collision boxes / lines in the same local frame)
    python3 scripts/p5/assets.py --yaml      # rewrite config/asset_catalog.yaml from CATALOG
    python3 scripts/p5/assets.py --sheet     # contact sheet reports/dev5/catalog.html (+ thumbnails)
"""
import math, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/p2"))
import roadside_tex as RX                       # noqa: E402

UV_ATLAS = None


# ============================================================================= textures (atlas)
def _n(h, w, s, a=1.0, seed=None):
    return RX.noise(h, w, s, a)


def _tile(h, w, rgb, var=10, sig=(1.0, 6.0), aniso=None):
    return RX.solid_noise(h, w, rgb, var, sig, aniso)


def t_brick(rgb=(150, 72, 52)):
    w = h = 256; a = _tile(h, w, rgb, 16, (1.0, 3.0))
    for r in range(0, h, 16):
        a[r:r + 2, :, :3] = (190, 182, 170)
        off = 0 if (r // 16) % 2 else 16
        for c in range(off, w, 32):
            a[r:r + 16, c:c + 2, :3] = (190, 182, 170)
    return RX.rgba(a)


def t_picket():                                   # black steel estate fence: bars + 2 rails, alpha between
    w, h = 128, 128; a = np.zeros((h, w, 4)); col = (22, 22, 24, 255)
    for x in range(4, w, 16):
        a[6:, x:x + 4] = col; a[0:8, x - 1:x + 5] = col              # bar + finial
    a[14:20] = col; a[h - 18:h - 12] = col
    return RX.rgba(a)


def t_lattice():
    w = h = 128; im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = (128, 96, 66, 255)
    for k in range(-h, w + h, 24):
        d.line((k, 0, k + h, h), fill=c, width=6); d.line((k, h, k + h, 0), fill=c, width=6)
    d.rectangle((0, 0, w, 6), fill=c); d.rectangle((0, h - 7, w, h), fill=c)
    return im


def t_pipegate(rgb=(60, 92, 58)):                 # tubular ranch gate, 5 bars + diagonal brace
    w, h = 256, 128; im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = (*rgb, 255)
    for k in range(5):
        y = 6 + k * 28; d.line((0, y, w, y), fill=c, width=7)
    d.line((0, h - 6, w, 6), fill=c, width=6); d.rectangle((0, 0, 8, h), fill=c); d.rectangle((w - 9, 0, w, h), fill=c)
    return im


def t_corrugated(rgb):
    w, h = 256, 256; a = _tile(h, w, rgb, 8, (1.0, 10.0))
    x = np.arange(w); a[..., :3] *= (0.86 + 0.14 * np.sin(x / 3.0))[None, :, None]
    return RX.rgba(a)


def t_siding(rgb, windows=True, door=False):
    w = h = 256; a = _tile(h, w, rgb, 6, (1.0, 10.0))
    for y in range(0, h, 12):
        a[y:y + 2, :, :3] *= 0.82
    im = RX.rgba(a); d = ImageDraw.Draw(im)
    if windows:
        for x0 in (30, 160):
            d.rectangle((x0 - 5, 70, x0 + 71, 160), fill=(240, 238, 232, 255)); d.rectangle((x0, 75, x0 + 66, 155), fill=(48, 56, 64, 255))
            d.line((x0 + 33, 75, x0 + 33, 155), fill=(240, 238, 232, 255), width=4)
    if door:
        d.rectangle((100, 120, 150, h), fill=(96, 70, 52, 255))
    d.rectangle((0, h - 14, w, h), fill=(110, 104, 96, 255))
    return im


def t_garage_door():
    w = h = 128; a = _tile(h, w, (226, 224, 216), 5, (1.0, 6.0))
    for y in range(0, h, 26):
        a[y:y + 3, :, :3] *= 0.75
    return RX.rgba(a)


def t_tank(rgb=(30, 32, 30)):
    w, h = 256, 128; a = _tile(h, w, rgb, 6, (1.0, 6.0))
    for y in range(10, h, 28):
        a[y:y + 3, :, :3] *= 1.5
    return RX.rgba(a)


def t_rock(rgb=(124, 116, 104)):
    a = _tile(128, 128, rgb, 26, (1.0, 3.0, 10.0)); return RX.rgba(a)


def t_bark(rgb=(78, 64, 50)):
    return RX.rgba(_tile(128, 128, rgb, 18, (1.0, 4.0), aniso=(0.3, 5)))


def t_gravel():
    a = _tile(128, 128, (138, 132, 122), 30, (0.6, 1.5)); return RX.rgba(a)


def _blobs(w, h, cols, n, rmin, rmax, shape, rng, alpha_noise=True):
    """foliage card: n random leaf blobs coloured from cols, inside the silhouette shape(x,y)->bool (0..1 coords)"""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    k = 0
    while k < n:
        x, y = rng.uniform(0, 1), rng.uniform(0, 1)
        if not shape(x, y):
            continue
        r = rng.uniform(rmin, rmax); c = cols[rng.integers(len(cols))]
        j = rng.uniform(0.85, 1.15)
        col = tuple(int(min(255, v * j)) for v in c) + (255,)
        d.ellipse((x * w - r, y * h - r * 0.8, x * w + r, y * h + r * 0.8), fill=col); k += 1
    return im


def t_shrub(kind):
    rng = np.random.default_rng(hash(kind) % 2 ** 31)
    dome = lambda x, y: (x - 0.5) ** 2 / 0.22 + (y - 1.0) ** 2 / 0.95 < 1 and y > 0.08
    if kind == "coyote":       # dense dark olive mound (Baccharis pilularis)
        return _blobs(256, 192, [(58, 78, 38), (74, 94, 46), (46, 62, 32), (88, 104, 52)], 2600, 3, 7, dome, rng)
    if kind == "poisonoak":    # glossy green with reddish new leaves, open
        return _blobs(256, 192, [(70, 110, 40), (96, 130, 50), (128, 72, 40), (60, 96, 36)], 900, 5, 10, dome, rng)
    if kind == "blackberry":   # tangled dark green, arching canes
        im = _blobs(256, 192, [(44, 70, 34), (56, 86, 40), (36, 54, 28)], 1800, 4, 8, lambda x, y: y > 0.18 + 0.12 * math.sin(x * 9), rng)
        d = ImageDraw.Draw(im)
        for _ in range(30):
            x = rng.uniform(0, 256); d.arc((x - 60, 20, x + 60, 260), 200, 340, fill=(84, 60, 48, 255), width=2)
        return im
    if kind == "fern":         # bright green fronds radiating from the base
        im = Image.new("RGBA", (256, 160), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        for k in range(16):
            ang = math.radians(20 + 140 * k / 15); L = rng.uniform(90, 140)
            x1, y1 = 128 + math.cos(ang) * L, 160 - math.sin(ang) * L * 0.9
            for t in np.linspace(0.1, 1, 14):
                px, py = 128 + (x1 - 128) * t, 160 + (y1 - 160) * t; s = 14 * (1 - t) + 3
                d.line((px - s * math.sin(ang), py - s * math.cos(ang), px + s * math.sin(ang), py + s * math.cos(ang)), fill=(92, 146, 54, 255), width=3)
            d.line((128, 160, x1, y1), fill=(70, 110, 40, 255), width=2)
        return im
    if kind == "grass":        # tall spring grass clump: vertical yellow-green blades
        im = Image.new("RGBA", (192, 192), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        for _ in range(260):
            x0 = rng.normal(96, 34); h = rng.uniform(70, 185); lean = rng.normal(0, 22)
            c = [(118, 150, 60), (140, 162, 70), (96, 130, 50), (170, 170, 96)][rng.integers(4)]
            d.line((x0, 192, x0 + lean, 192 - h), fill=(*c, 255), width=2)
        return im
    if kind == "ivy":
        return _blobs(256, 128, [(40, 74, 34), (54, 92, 40), (30, 60, 28)], 1500, 4, 8, lambda x, y: y > 0.1, rng)
    if kind == "hedge":        # clipped privet / laurel face
        return _blobs(256, 192, [(48, 80, 38), (60, 94, 44), (40, 66, 32)], 3200, 3, 6, lambda x, y: True, rng)
    if kind == "oak":          # coast live oak crown card with trunk
        im = _blobs(256, 256, [(46, 66, 34), (58, 82, 40), (36, 52, 28), (70, 92, 46)], 2600, 4, 9,
                    lambda x, y: (x - 0.5) ** 2 / 0.24 + (y - 0.42) ** 2 / 0.14 < 1, rng)
        d = ImageDraw.Draw(im); d.polygon([(118, 256), (138, 256), (134, 150), (122, 150)], fill=(70, 58, 46, 255))
        return im
    if kind == "litter":       # leaf litter / branch debris patch (ground decal)
        im = _blobs(128, 128, [(118, 96, 64), (96, 78, 52), (140, 116, 76), (80, 66, 46)], 900, 2, 4,
                    lambda x, y: (x - .5) ** 2 + (y - .5) ** 2 < .2, rng)
        return im
    raise KeyError(kind)


def t_branches():
    rng = np.random.default_rng(5); im = Image.new("RGBA", (256, 128), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for _ in range(40):
        x = rng.uniform(0, 256); y = rng.uniform(40, 128); L = rng.uniform(30, 120); a = rng.normal(0, 0.5)
        d.line((x, y, x + L * math.cos(a), y - abs(L * math.sin(a)) * 0.5), fill=(96, 78, 60, 255), width=int(rng.integers(2, 6)))
    for _ in range(300):
        x, y = rng.uniform(0, 256), rng.uniform(50, 128); d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(112, 96, 60, 255))
    return im


def t_plaque(text, bg=RX.YEL, w=256, h=192):
    im, d = RX._canvas(w, h); d.rounded_rectangle((0, 0, w - 1, h - 1), 10, fill=RX.BLK); d.rounded_rectangle((6, 6, w - 7, h - 7), 8, fill=bg)
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        RX._txt(d, (w // 2, h // 2 + (i - (len(lines) - 1) / 2) * 70), ln, 64 if len(ln) < 6 else 46, RX.BLK)
    return im


def t_address(num="5240"):
    im, d = RX._canvas(160, 240); d.rectangle((0, 0, 159, 239), fill=RX.WHT); d.rectangle((4, 4, 155, 235), outline=RX.BLK, width=4)
    RX._txt(d, (80, 60), num, 40, RX.BLK); RX._txt(d, (80, 130), "FIRE", 26, RX.RED); RX._txt(d, (80, 170), "ACCESS", 26, RX.RED)
    return im


def t_cmp():
    w = h = 128; a = _tile(h, w, (150, 150, 144), 8, (1.0, 4.0)); y, x = np.mgrid[:h, :w]
    r = np.hypot(x - 64, y - 64); a[..., :3] *= (0.85 + 0.15 * np.sin(r / 2.0))[..., None]
    a[r < 40, :3] = (20, 20, 22)
    return RX.rgba(a)


class LAtlas(RX.Atlas):
    """same packing as roadside_tex.Atlas, on a 4096 x 2048 sheet (rects normalised per axis)"""
    W, H = 4096, 2048

    def __init__(self):
        import io as _io
        self.img = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0)); self.rect = {}; self.x = self.y = self.rowh = 0

    def put(self, name, im):
        if name in self.rect:
            return self.rect[name]
        w, h = im.size; P = RX.PAD
        if self.x + w + 2 * P > self.W:
            self.x, self.y, self.rowh = 0, self.y + self.rowh, 0
        assert self.y + h + 2 * P <= self.H, "ledger atlas full"
        x0, y0 = self.x + P, self.y + P
        a = np.pad(np.asarray(im.convert("RGBA")), ((P, P), (P, P), (0, 0)), mode="edge")
        self.img.paste(Image.fromarray(a), (self.x, self.y))
        self.rect[name] = (x0 / self.W, y0 / self.H, (x0 + w) / self.W, (y0 + h) / self.H)
        self.x += w + 2 * P; self.rowh = max(self.rowh, h + 2 * P)
        return self.rect[name]


def build_atlas():
    """roadside_tex base tiles (wood, stone, concrete, chainlink, rails, buildings, wbeam ...) + dev5 tiles"""
    B = RX.base_atlas(); A = LAtlas()
    for k, (u0, v0, u1, v1) in B.rect.items():
        S_ = B.img.size[0]
        A.put(k, B.img.crop((round(u0 * S_), round(v0 * S_), round(u1 * S_), round(v1 * S_))))
    rng = np.random.default_rng(3)
    extra = {
        "brick": t_brick(), "brick_tan": t_brick((176, 140, 104)), "picket": t_picket(), "lattice": t_lattice(),
        "pipegate": t_pipegate(), "pipegate_grey": t_pipegate((150, 152, 150)),
        "corr_red": t_corrugated((132, 44, 36)), "corr_grey": t_corrugated((150, 152, 150)), "corr_green": t_corrugated((70, 96, 72)),
        "siding_white": t_siding((226, 224, 216)), "siding_tan": t_siding((196, 172, 136)), "siding_grey": t_siding((150, 150, 146), door=True),
        "barn_wall": t_siding((128, 44, 36), windows=False), "trailer": t_siding((222, 222, 214)), "garage_door": t_garage_door(),
        "tank_black": t_tank(), "tank_green": t_tank((62, 86, 64)), "rock": t_rock(), "rock_dark": t_rock((92, 88, 82)),
        "bark": t_bark(), "stump_top": RX.rgba(_tile(64, 64, (150, 120, 84), 10, (1.0, 3.0))), "gravel": t_gravel(),
        "blue": RX.rgba(_tile(32, 32, (40, 70, 130), 4, (1.0,))), "green_box": RX.rgba(_tile(32, 32, (70, 90, 72), 4, (1.0,))),
        "galv": RX.rgba(_tile(64, 64, (168, 170, 166), 6, (1.0, 3.0))), "reflector": Image.new("RGBA", (16, 16), (240, 180, 20, 255)),
        "reflector_r": Image.new("RGBA", (16, 16), (200, 30, 30, 255)), "yellow": Image.new("RGBA", (16, 16), (232, 196, 30, 255)),
        "paint_white": Image.new("RGBA", (16, 16), (236, 236, 230, 255)), "cmp": t_cmp(), "branches": t_branches(),
        "plaque_15": t_plaque("15\nMPH"), "plaque_25": t_plaque("25\nMPH"), "address": t_address(),
        "plaque_end": t_plaque("END", RX.WHT), "trash_lid": RX.rgba(_tile(32, 32, (40, 90, 48), 4, (1.0,))),
    }
    for k in ("coyote", "poisonoak", "blackberry", "fern", "grass", "ivy", "hedge", "oak", "litter"):
        extra["veg_" + k] = t_shrub(k)
    for k, im in extra.items():
        A.put(k, im)
    return A


def atlas():
    global UV_ATLAS
    if UV_ATLAS is None:
        UV_ATLAS = build_atlas()
    return UV_ATLAS


def R(name):
    return atlas().rect[name]


# ============================================================================= mesh builder
class MB:
    """quads/tris in the local asset frame. P (n,3), UV (n,2), N (n,3), T (m,3). col: collision primitives.
    meta: free data (e.g. sign front-quad vertex index)."""
    def __init__(self):
        self.P, self.UV, self.N, self.T, self.n = [], [], [], [], 0
        self.col, self.meta = [], {}

    def quad(self, P4, rect, out=None, uspan=(0, 1), vspan=(0, 1)):
        P4 = np.asarray(P4, float); nrm = np.cross(P4[1] - P4[0], P4[3] - P4[0])
        if out is not None and np.dot(nrm, out) < 0:
            P4 = P4[[1, 0, 3, 2]]; uspan = (uspan[1], uspan[0]); nrm = -nrm
        nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
        u0, v0, u1, v1 = rect
        ua, ub = u0 + (u1 - u0) * uspan[0], u0 + (u1 - u0) * uspan[1]
        va, vb = v1 - (v1 - v0) * vspan[0], v1 - (v1 - v0) * vspan[1]
        self.P.append(P4); self.UV.append(np.array([[ua, va], [ub, va], [ub, vb], [ua, vb]])); self.N.append(np.tile(nrm, (4, 1)))
        self.T.append(np.array([[0, 1, 2], [0, 2, 3]]) + self.n); self.n += 4
        return self.n - 4

    def quad2(self, P4, rect, **kw):
        P4 = np.asarray(P4, float); k = self.quad(P4, rect, **kw)
        us = kw.pop("uspan", (0, 1)); self.quad(P4[[1, 0, 3, 2]], rect, uspan=us[::-1], **kw)
        return k

    def tri(self, P3, rect, uv3):
        P3 = np.asarray(P3, float); nrm = np.cross(P3[1] - P3[0], P3[2] - P3[0]); nrm /= np.linalg.norm(nrm) + 1e-12
        u0, v0, u1, v1 = rect
        self.P.append(P3); self.N.append(np.tile(nrm, (3, 1)))
        self.UV.append(np.array([[u0 + (u1 - u0) * a, v1 - (v1 - v0) * b] for a, b in uv3])); self.T.append(np.array([[0, 1, 2]]) + self.n); self.n += 3

    def box(self, cx, cy, z0, z1, L, W, rect, top=None, yaw=0.0, uscale=None):
        f = np.array([math.cos(yaw), math.sin(yaw)]); l = np.array([-f[1], f[0]]); c = np.array([cx, cy])
        co = [c - f * L / 2 - l * W / 2, c + f * L / 2 - l * W / 2, c + f * L / 2 + l * W / 2, c - f * L / 2 + l * W / 2]
        for k in range(4):
            a, b = co[k], co[(k + 1) % 4]; mid = (a + b) / 2 - c
            us = (0, min(1.0, np.hypot(*(b - a)) / uscale)) if uscale else (0, 1)
            vs = (0, min(1.0, (z1 - z0) / uscale)) if uscale else (0, 1)
            self.quad([[*a, z0], [*b, z0], [*b, z1], [*a, z1]], rect, out=np.array([*mid, 0]), uspan=us, vspan=vs)
        self.quad([[*co[0], z1], [*co[1], z1], [*co[2], z1], [*co[3], z1]], top or rect, out=np.array([0, 0, 1.0]))

    def cyl(self, cx, cy, z0, z1, r, sides, rect, top=None, r1=None):
        r1 = r if r1 is None else r1; a = np.linspace(0, 2 * np.pi, sides + 1)
        for k in range(sides):
            p0 = (cx + r * math.cos(a[k]), cy + r * math.sin(a[k])); p1 = (cx + r * math.cos(a[k + 1]), cy + r * math.sin(a[k + 1]))
            q0 = (cx + r1 * math.cos(a[k]), cy + r1 * math.sin(a[k])); q1 = (cx + r1 * math.cos(a[k + 1]), cy + r1 * math.sin(a[k + 1]))
            mid = np.array([math.cos((a[k] + a[k + 1]) / 2), math.sin((a[k] + a[k + 1]) / 2), 0])
            self.quad([[*p0, z0], [*p1, z0], [*q1, z1], [*q0, z1]], rect, out=mid, uspan=(k / sides, (k + 1) / sides))
        if top:
            for k in range(sides):
                self.tri([[cx, cy, z1], [cx + r1 * math.cos(a[k]), cy + r1 * math.sin(a[k]), z1], [cx + r1 * math.cos(a[k + 1]), cy + r1 * math.sin(a[k + 1]), z1]],
                         top, [(0.5, 0.5), (0.5 + 0.5 * math.cos(a[k]), 0.5 + 0.5 * math.sin(a[k])), (0.5 + 0.5 * math.cos(a[k + 1]), 0.5 + 0.5 * math.sin(a[k + 1]))])

    def hcyl(self, x0, x1, y, z, r, sides, rect):
        """horizontal cylinder along x (rails, logs, pipes)"""
        a = np.linspace(0, 2 * np.pi, sides + 1)
        for k in range(sides):
            c0, s0, c1, s1 = math.cos(a[k]), math.sin(a[k]), math.cos(a[k + 1]), math.sin(a[k + 1])
            self.quad([[x0, y + r * c0, z + r * s0], [x1, y + r * c0, z + r * s0], [x1, y + r * c1, z + r * s1], [x0, y + r * c1, z + r * s1]],
                      rect, out=np.array([0, (c0 + c1) / 2, (s0 + s1) / 2]))

    def card(self, cx, cy, w, h, rect, n=2, z0=0.0, rot=0.0):
        """n crossed vertical alpha cards centred at (cx, cy)"""
        for k in range(n):
            a = rot + math.pi * k / n; d = np.array([math.cos(a), math.sin(a)]) * w / 2
            self.quad2([[cx - d[0], cy - d[1], z0], [cx + d[0], cy + d[1], z0], [cx + d[0], cy + d[1], z0 + h], [cx - d[0], cy - d[1], z0 + h]], rect)

    def flat(self, x0, x1, y0, y1, z, rect, nx=1, ny=1):
        """horizontal patch subdivided nx*ny (so it can drape over terrain)"""
        for i in range(nx):
            for j in range(ny):
                a0, a1 = x0 + (x1 - x0) * i / nx, x0 + (x1 - x0) * (i + 1) / nx
                b0, b1 = y0 + (y1 - y0) * j / ny, y0 + (y1 - y0) * (j + 1) / ny
                self.quad([[a0, b0, z], [a1, b0, z], [a1, b1, z], [a0, b1, z]], rect, out=np.array([0, 0, 1.0]),
                          uspan=(i / nx, (i + 1) / nx), vspan=(j / ny, (j + 1) / ny))

    def sheet(self, x0, x1, y, z0, z1, rect, tw, double=True, seg=1.5):
        """vertical sheet along x at offset y, repeating the tile every tw metres; quads <= seg long"""
        nseg = max(1, int(round((x1 - x0) / tw)))
        for k in range(nseg):
            sa = x0 + (x1 - x0) * k / nseg; sb = x0 + (x1 - x0) * (k + 1) / nseg; nq = max(1, int(math.ceil((sb - sa) / seg)))
            for j in range(nq):
                qa = sa + (sb - sa) * j / nq; qb = sa + (sb - sa) * (j + 1) / nq
                P4 = [[qa, y, z0], [qb, y, z0], [qb, y, z1], [qa, y, z1]]
                (self.quad2 if double else self.quad)(P4, rect, uspan=(j / nq, (j + 1) / nq), **({} if double else {"out": np.array([0, -1.0, 0])}))

    def slab(self, x0, x1, y0, y1, z0, z1, rect, tw=1.6, seg=1.5, top=None):
        """solid wall along x between y0..y1 (inner y0 faces the road), segmented so it bends"""
        n = max(1, int(math.ceil((x1 - x0) / seg)))
        xs = np.linspace(x0, x1, n + 1)
        for i in range(n):
            a, b = xs[i], xs[i + 1]; us = (0, min(1.0, (b - a) / tw)); vs = (0, min(1.0, (z1 - z0) / tw))
            self.quad([[a, y0, z0], [b, y0, z0], [b, y0, z1], [a, y0, z1]], rect, out=np.array([0, -1.0, 0]), uspan=us, vspan=vs)
            self.quad([[a, y1, z0], [b, y1, z0], [b, y1, z1], [a, y1, z1]], rect, out=np.array([0, 1.0, 0]), uspan=us, vspan=vs)
            self.quad([[a, y0, z1], [b, y0, z1], [b, y1, z1], [a, y1, z1]], top or rect, out=np.array([0, 0, 1.0]), uspan=us, vspan=(0, 0.3))
        for x, s in ((x0, -1.0), (x1, 1.0)):
            self.quad([[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]], rect, out=np.array([s, 0, 0]), uspan=(0, 0.3))

    def rock(self, cx, cy, r, h, rect, rng, sub=1):
        """jittered low-poly boulder (octahedron subdivided once = 32 faces), base sunk 15 %"""
        V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
        F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
        for _ in range(sub):
            V2, F2, cache = list(V), [], {}
            def mid(a, b):
                k = (min(a, b), max(a, b))
                if k not in cache:
                    m = (V2[a] + V2[b]) / 2; V2.append(m / np.linalg.norm(m)); cache[k] = len(V2) - 1
                return cache[k]
            for a, b, c in F:
                ab, bc, ca = mid(a, b), mid(b, c), mid(c, a); F2 += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
            V, F = np.array(V2), F2
        V = V * (1 + rng.uniform(-0.18, 0.18, (len(V), 1)))
        V = V * np.array([r, r * rng.uniform(0.7, 1.0), h / 2]) + np.array([cx, cy, h / 2 * 0.7])
        for a, b, c in F:
            P3 = V[[a, b, c]]
            if np.dot(np.cross(P3[1] - P3[0], P3[2] - P3[0]), P3.mean(0) - np.array([cx, cy, h * 0.35])) < 0:
                P3 = P3[[0, 2, 1]]
            self.tri(P3, rect, [(0, 0), (1, 0), (0.5, 1)])

    # collision primitives (local frame): ('box', cx, cy, L, W, z0, z1, yaw) / ('line', y, z0, z1, half)
    def cbox(self, cx, cy, L, W, z0, z1, yaw=0.0):
        self.col.append(("box", cx, cy, L, W, z0, z1, yaw))

    def cline(self, y, z0, z1, half=0.07):
        self.col.append(("line", y, z0, z1, half))

    def arrays(self):
        if not self.P:
            return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 2)), np.zeros((0, 3))
        return np.concatenate(self.P), np.concatenate(self.T), np.concatenate(self.UV), np.concatenate(self.N)


# ============================================================================= generators
def _rng(p):
    return np.random.default_rng(int(p.get("seed", 7)))


# ---- signs (MUTCD-style, both faces: design front + galvanised back) ---------------------------------------------
SIGN_TEXT = {   # asset id -> default text for roadside_tex.sign_image (params.text overrides)
    "sign.curve_left": "yellow warning diamond curve left", "sign.curve_right": "yellow warning diamond curve right",
    "sign.chevron_left": "chevron left", "sign.chevron_right": "chevron right",
    "sign.arrow_left": "arrow left", "sign.arrow_right": "arrow right",
    "sign.speed_limit": "SPEED LIMIT 25", "sign.stop": "STOP", "sign.hump": "HUMP", "sign.slow": "diamond SLOW",
    "sign.warning_blank": "warning diamond", "sign.street_name": "street blade <PALOMARES RD>",
    "sign.narrow_bridge": "narrow bridge", "sign.object_marker": "object marker", "sign.mile_marker": "mile marker",
    "sign.mirror": "mirror", "sign.round_prohibition": "round", "sign.text_private": "private",
    "sign.blank_white": "white sign", "sign.back_diamond": "back of diamond", "sign.back_rect": "back of sign",
}
PLAQUES = {"sign.advisory_15": ("plaque_15", 0.6, 0.45), "sign.advisory_25": ("plaque_25", 0.6, 0.45),
           "sign.address_fire": ("address", 0.4, 0.6)}


def sign_design(asset, p):
    """-> (rect, w, h, key) of the sign face"""
    if asset in PLAQUES:
        k, w, h = PLAQUES[asset]; return R(k), w, h, k
    txt = p.get("text") or SIGN_TEXT.get(asset, "warning diamond")
    im, w, h, key = RX.sign_image(txt, None, dirn_override=p.get("_dir"))
    if im is None:
        im, w, h, key = RX.sign_image("back of sign", None)
    if max(im.size) > 160:
        k_ = 160 / max(im.size); im = im.resize((max(8, int(im.size[0] * k_)), max(8, int(im.size[1] * k_))))
    return atlas().put("sign_" + key, im), w, h, key


def g_sign(L, p, asset="sign.warning_blank"):
    m = MB(); rect, w, h, key = sign_design(asset, p)
    w *= float(p.get("scale", 1.0)); h *= float(p.get("scale", 1.0))
    top = float(p.get("height", 2.1 if h >= 0.5 else 2.0))
    zb = max(top - h, 0.9)
    # front faces -x (approaching NB driver looks along +x): viewer's right = -y... the driver looks along +x, so
    # his right is -y (y = outward is to his right only on the R side; 21e mirrors nothing, it rotates) -> BL at +y
    front = m.quad([[-0.04, w / 2, zb], [-0.04, -w / 2, zb], [-0.04, -w / 2, zb + h], [-0.04, w / 2, zb + h]], rect, out=np.array([-1.0, 0, 0]))
    m.meta.update(front=front, key=key)
    m.quad([[-0.03, -w / 2, zb], [-0.03, w / 2, zb], [-0.03, w / 2, zb + h], [-0.03, -w / 2, zb + h]], R("signback"), out=np.array([1.0, 0, 0]))
    sub = p.get("plaque")                                     # optional plaque below (e.g. "15" advisory)
    if sub:
        pr, pw, ph, _ = sign_design(sub, {})
        m.quad([[-0.04, pw / 2, zb - ph - 0.05], [-0.04, -pw / 2, zb - ph - 0.05], [-0.04, -pw / 2, zb - 0.05], [-0.04, pw / 2, zb - 0.05]], pr, out=np.array([-1.0, 0, 0]))
        m.quad([[-0.03, -pw / 2, zb - ph - 0.05], [-0.03, pw / 2, zb - ph - 0.05], [-0.03, pw / 2, zb - 0.05], [-0.03, -pw / 2, zb - 0.05]], R("signback"), out=np.array([1.0, 0, 0]))
    m.box(0.0, 0.0, -0.3, zb + h - 0.05, 0.06, 0.06, R("galv"))
    m.cbox(0, 0, 0.16, 0.16, -0.5, 1.6)
    return m


# ---- posts ---------------------------------------------------------------------------------------------------------
def g_delineator(L, p, col="paint_white"):
    m = MB(); h = float(p.get("height", 1.1))
    m.box(0, 0, -0.2, h, 0.08, 0.08, R("delin") if col == "paint_white" else R("yellow"))
    m.box(-0.045, 0, h - 0.3, h - 0.12, 0.01, 0.07, R("reflector"))
    return m


def g_hazard_marker(L, p):
    m = MB(); rect, w, h, _ = sign_design("sign.object_marker", {})
    m.box(0, 0, -0.3, 1.2, 0.06, 0.06, R("galv"))
    m.quad([[-0.04, w / 2, 0.6], [-0.04, -w / 2, 0.6], [-0.04, -w / 2, 0.6 + h], [-0.04, w / 2, 0.6 + h]], rect, out=np.array([-1.0, 0, 0]))
    m.quad([[-0.03, -w / 2, 0.6], [-0.03, w / 2, 0.6], [-0.03, w / 2, 0.6 + h], [-0.03, -w / 2, 0.6 + h]], R("signback"), out=np.array([1.0, 0, 0]))
    m.cbox(0, 0, 0.16, 0.16, -0.5, 1.6)
    return m


def g_bollard(L, p):
    m = MB(); m.box(0, 0, -0.3, 0.9, 0.15, 0.15, R("wood_dark")); m.cbox(0, 0, 0.2, 0.2, -0.5, 1.5); return m


def g_tpost(L, p):
    m = MB(); m.box(0, 0, -0.3, 1.3, 0.05, 0.04, R("corr_green")); m.box(0, 0, 1.2, 1.3, 0.06, 0.05, R("paint_white")); return m


# ---- poles ---------------------------------------------------------------------------------------------------------
def g_pole(L, p, transformer=False):
    m = MB(); H = float(p.get("height", 10.5))
    m.cyl(0, 0, -0.3, H, 0.15, 7, R("pole"), r1=0.12)
    arm = H - 0.5
    m.box(0, 0, arm, arm + 0.1, 0.1, 2.4, R("wood_dark"))
    for dy in (-1.0, 0.0, 1.0):
        m.cyl(0, dy, arm + 0.1, arm + 0.32, 0.04, 5, R("white"))
    if transformer:
        m.cyl(0.0, -0.35, H - 2.6, H - 1.5, 0.28, 8, R("galv"), top=R("galv"))
    m.box(0, 0, 1.6, 1.9, 0.02, 0.17, R("yellow"))           # tag band
    m.cbox(0, 0, 0.34, 0.34, -0.6, 2.6)
    m.meta["wire_z"] = arm + 0.15; m.meta["wire_dy"] = (-1.0, 0.0, 1.0)
    return m


def g_streetlight(L, p):
    m = MB(); H = 8.0
    m.cyl(0, 0, -0.3, H, 0.1, 6, R("galv"), r1=0.07)
    m.box(0, -0.9, H - 0.1, H, 0.12, 1.8, R("galv")); m.box(0, -1.8, H - 0.25, H - 0.05, 0.5, 0.25, R("galv"))
    m.cbox(0, 0, 0.3, 0.3, -0.6, 2.6)
    return m


# ---- mailboxes / small props ---------------------------------------------------------------------------------------
def g_mailbox(L, p, n=1):
    m = MB(); n = int(p.get("count", n)); hh = 1.1; wd = n * 0.42
    m.box(0, 0, -0.3, hh - 0.1, 0.1, 0.1, R("wood_dark"))
    m.box(0, 0, hh - 0.1, hh - 0.06, wd, 0.22, R("wood_dark"))
    rng = _rng(p)
    for k in range(n):
        x = (k - (n - 1) / 2) * 0.42
        m.box(x, 0, hh - 0.06, hh + 0.14, 0.2, 0.48, R(["mailbox", "galv", "black", "white"][int(rng.integers(4))]), yaw=0.0)
    m.cbox(0, 0, wd, 0.5, -0.5, 1.5)
    return m


def g_mailbox_brick(L, p):
    m = MB(); m.box(0, 0, -0.2, 1.3, 0.6, 0.6, R("brick"), top=R("concrete")); m.box(0, -0.31, 0.7, 0.95, 0.42, 0.02, R("black"))
    m.cbox(0, 0, 0.6, 0.6, -0.5, 1.6); return m


def g_cbu(L, p):
    m = MB(); m.box(0, 0, -0.1, 1.6, 0.9, 0.5, R("galv"), top=R("galv")); m.box(0, 0, -0.1, 0.1, 1.0, 0.6, R("concrete"))
    m.cbox(0, 0, 0.9, 0.5, -0.5, 1.7); return m


def g_bin(L, p):
    m = MB(); rng = _rng(p); n = int(p.get("count", 2))
    for k in range(n):
        x = (k - (n - 1) / 2) * 0.75; c = ["blue", "trash_lid", "black"][k % 3]
        m.box(x, 0, 0, 1.05, 0.6, 0.7, R(c), top=R(c))
    return m


def g_utility_box(L, p):
    m = MB(); m.box(0, 0, 0, 1.2, 0.9, 0.6, R("green_box")); m.cbox(0, 0, 0.9, 0.6, -0.5, 1.5); return m


def g_hydrant(L, p):
    m = MB(); m.cyl(0, 0, 0, 0.75, 0.13, 8, R("yellow"), top=R("yellow")); m.box(0, 0, 0.45, 0.55, 0.42, 0.1, R("yellow"))
    m.cbox(0, 0, 0.3, 0.3, -0.5, 1.4); return m


def g_hay(L, p):
    m = MB(); m.box(0, 0, 0, 0.45, 1.0, 0.5, R("veg_grass") if False else R("splitrail")); return m


# ---- guardrails ----------------------------------------------------------------------------------------------------
def g_wbeam(L, p, wood=False):
    m = MB(); h = float(p.get("height", 0.78))
    m.sheet(0, L, 0.0, h - 0.36, h, R("wbeam"), 3.81)
    n = max(1, int(round(L / 1.905)))
    for k in range(n + 1):
        m.box(L * k / n, 0.12, -0.5, h - 0.05, 0.14 if wood else 0.1, 0.14 if wood else 0.1, R("wood_dark" if wood else "metal"))
    m.cline(0.05, -0.8, max(1.4, h + 0.6))
    return m


def g_timber_rail(L, p):
    """wooden guard rail (frame ~46): square posts + two heavy timber rails"""
    m = MB(); n = max(1, int(round(L / 2.4)))
    for k in range(n + 1):
        m.box(L * k / n, 0.1, -0.4, 0.9, 0.2, 0.2, R("timber"))
    for z in (0.45, 0.78):
        m.slab(0, L, -0.02, 0.06, z - 0.13, z + 0.1, R("timber"), tw=2.4, seg=2.4)
    m.cline(0.05, -0.8, 1.4)
    return m


def g_cable_rail(L, p):
    m = MB(); n = max(1, int(round(L / 4.0)))
    for k in range(n + 1):
        m.box(L * k / n, 0.05, -0.3, 0.85, 0.08, 0.08, R("metal"))
    m.sheet(0, L, 0.0, 0.2, 0.8, R("cable"), 4.0); m.cline(0.05, -0.8, 1.4)
    return m


# ---- fences (sheet + posts; collider when close) ---------------------------------------------------------------
FENCE = {   # id: (panel tile, tile_w, height, post tile, post w, post spacing)
    "fence.chainlink": ("chainlink", 0.64, 1.8, "galv", 0.06, 3.0),
    "fence.split_rail": ("splitrail", 3.0, 1.2, "wood_dark", 0.14, 3.0),
    "fence.board_3rail": ("boards", 2.4, 1.3, "wood_dark", 0.12, 2.4),
    "fence.white_rail": ("whiterail", 2.4, 1.3, "wood_white", 0.13, 2.4),
    "fence.ranch_wire": ("ranchwire", 1.5, 1.2, "wood_dark", 0.11, 4.0),
    "fence.privacy": ("privacy", 2.4, 1.8, "wood_dark", 0.14, 2.4),
    "fence.iron_picket": ("picket", 1.6, 1.6, "black", 0.07, 2.0),
    "fence.lattice": ("lattice", 1.2, 1.0, "wood", 0.08, 1.8),
    "fence.pipe_rail": ("pipegate_grey", 3.0, 1.2, "galv", 0.08, 3.0),
}


def g_fence(L, p, kind="fence.chainlink"):
    tile, tw, hd, ptile, pw, sp = FENCE[kind]; h = float(p.get("height", hd)); m = MB()
    m.sheet(0, L, 0.0, 0.0 if kind != "fence.ranch_wire" else 0.05, h, R(tile), tw * h / hd if tile in ("chainlink",) else tw)
    n = max(1, int(round(L / sp)))
    for k in range(n + 1):
        m.box(L * k / n, 0.0, -0.35, h + 0.05, pw, pw, R(ptile))
    if kind == "fence.chainlink":
        m.hcyl(0, L, 0, h, 0.025, 4, R("galv"))
    m.cline(0.0, -0.7, max(1.4, h))
    return m


# ---- walls ---------------------------------------------------------------------------------------------------------
WALL = {"wall.retaining_concrete": ("concrete", 0.35, 1.2), "wall.retaining_stone": ("stone", 0.45, 1.0),
        "wall.timber_crib": ("timber", 0.30, 1.2), "wall.brick_low": ("brick", 0.3, 0.9), "wall.sackcrete": ("rock", 0.5, 1.0)}


def g_wall(L, p, kind="wall.retaining_concrete"):
    tile, th, hd = WALL[kind]; h = float(p.get("height", hd)); m = MB()
    m.slab(0, L, 0.0, th, -0.45, h, R(tile), tw=2.4 if tile == "timber" else 1.6)
    if kind == "wall.timber_crib":                              # crib headers sticking out
        for x in np.arange(0.6, L, 2.4):
            for z in np.arange(0.1, h, 0.45):
                m.box(x, -0.05, z, z + 0.2, 0.2, 0.15, R("timber"))
    m.cline(th / 2, -0.7, max(1.4, h), half=max(0.07, th * 0.45))
    return m


def g_rockfall_mesh(L, p):
    m = MB(); h = float(p.get("height", 4.0))
    m.sheet(0, L, 0.1, 0.0, h, R("rockmesh"), 2.0)
    for x in np.arange(0, L + 0.01, 6.0):
        m.box(x, 0.1, h - 0.2, h + 0.3, 0.1, 0.1, R("galv"))
    m.cline(0.1, -0.7, 1.4)
    return m


# ---- bridges / culverts --------------------------------------------------------------------------------------------
def g_parapet(L, p, tile="concrete", h=0.8, th=0.3, rail=False):
    m = MB(); h = float(p.get("height", h))
    m.slab(-0.5, L + 0.5, 0.0, th, -0.8, h, R(tile), top=R("concrete"))
    if rail:
        m.hcyl(-0.5, L + 0.5, th / 2, h + 0.3, 0.05, 5, R("galv"))
        for x in np.arange(-0.5, L + 0.51, 2.0):
            m.box(x, th / 2, h, h + 0.3, 0.06, 0.06, R("galv"))
    m.slab(-0.5, L + 0.5, -0.05, th + 0.3, -1.4, -0.8, R("concrete"))      # deck edge fascia below
    m.cline(th / 2, -0.8, max(1.4, h + 0.6), half=th * 0.45)
    return m


def g_bridge_timber(L, p):
    m = MB(); n = max(1, int(round(L / 2.0)))
    for k in range(n + 1):
        m.box(L * k / n, 0.1, -0.6, 1.0, 0.2, 0.2, R("timber"))
    m.slab(0, L, 0.0, 0.1, 0.6, 0.85, R("timber"), tw=2.4); m.slab(0, L, -0.1, 0.25, -0.9, -0.5, R("timber"))
    m.cline(0.05, -0.8, 1.4)
    return m


def g_bridge_steel(L, p):
    """steel-girder bridge edge: I-girder fascia + steel tube railing on posts"""
    m = MB()
    m.slab(-0.5, L + 0.5, -0.1, 0.25, -1.3, -0.15, R("rock_dark"), tw=2.0)
    m.slab(-0.5, L + 0.5, 0.0, 0.25, -0.15, 0.15, R("concrete"))
    for z in (0.55, 1.0):
        m.hcyl(-0.5, L + 0.5, 0.12, z, 0.05, 5, R("galv"))
    for x in np.arange(-0.5, L + 0.51, 2.0):
        m.box(x, 0.12, 0.15, 1.05, 0.08, 0.08, R("galv"))
    m.cline(0.12, -0.8, 1.4)
    return m


def g_headwall(L, p):
    m = MB(); w = max(L or 0, 4.0) if L else 4.0
    m.slab(0, w, 0.0, 0.35, -0.6, float(p.get("height", 0.6)), R("concrete"))
    m.cline(0.17, -0.7, 1.4)
    return m


def g_cmp(L, p):
    """corrugated metal pipe end poking out of the fill (point, opening faces the road)"""
    m = MB(); r = float(p.get("radius", 0.45))
    m.quad([[-r, -0.02, 0.0], [r, -0.02, 0.0], [r, -0.02, 2 * r], [-r, -0.02, 2 * r]], R("cmp"), out=np.array([0, -1.0, 0]))
    m.flat(-r - 0.6, r + 0.6, -0.8, 0.0, 0.02, R("rock_dark"))
    return m


def g_drop_inlet(L, p):
    m = MB(); m.box(0, 0.3, -0.3, 0.15, 1.0, 0.8, R("concrete"), top=R("black")); return m


# ---- buildings -----------------------------------------------------------------------------------------------------
def _house(m, L, D, eave, pitch, wall, roof, gable=None):
    """walls + gable roof, ridge along x, front (y=0) faces the road"""
    co = [(0 - L / 2, 0), (L / 2, 0), (L / 2, D), (-L / 2, D)]; c = np.array([0, D / 2])
    for k in range(4):
        a, b = np.array(co[k]), np.array(co[(k + 1) % 4]); mid = (a + b) / 2 - c; nu = max(1, int(round(np.hypot(*(b - a)) / 4.0)))
        for j in range(nu):
            a0 = a + (b - a) * j / nu; a1 = a + (b - a) * (j + 1) / nu
            m.quad([[*a0, -0.5], [*a1, -0.5], [*a1, eave], [*a0, eave]], wall, out=np.array([*mid, 0]))
    ridge = eave + pitch * D / 2; ov = 0.5
    m.quad([[-L / 2 - ov, -ov, eave - 0.1], [L / 2 + ov, -ov, eave - 0.1], [L / 2 + ov, D / 2, ridge], [-L / 2 - ov, D / 2, ridge]], roof, out=np.array([0, -1.0, 1.0]))
    m.quad([[-L / 2 - ov, D / 2, ridge], [L / 2 + ov, D / 2, ridge], [L / 2 + ov, D + ov, eave - 0.1], [-L / 2 - ov, D + ov, eave - 0.1]], roof, out=np.array([0, 1.0, 1.0]))
    for x, s in ((-L / 2, -1.0), (L / 2, 1.0)):
        P3 = np.array([[x, 0, eave], [x, D, eave], [x, D / 2, ridge]])
        if np.dot(np.cross(P3[1] - P3[0], P3[2] - P3[0]), [s, 0, 0]) < 0:
            P3 = P3[[1, 0, 2]]
        m.tri(P3, gable or wall, [(0, 0), (1, 0), (0.5, 0.4)])
    return ridge


def g_building(L, p, kind="building.ranch_house"):
    m = MB(); rng = _rng(p)
    dims = {"building.ranch_house": (16, 10, 3.0, 0.45, "siding_tan"), "building.house_two_storey": (12, 9, 5.8, 0.6, "siding_white"),
            "building.barn": (14, 10, 5.0, 0.9, "barn_wall"), "building.shed": (4, 3, 2.4, 0.4, "corr_green"),
            "building.garage": (7, 6.5, 2.8, 0.45, "siding_grey"), "building.stable": (20, 8, 3.0, 0.35, "wood")}
    Ld, Dd, eave, pitch, wall = dims[kind]
    Lx = float(p.get("L", Ld)); D = float(p.get("D", Dd)); eave = float(p.get("eave", eave))
    roof = R("corr_grey") if kind in ("building.barn", "building.shed", "building.stable") else R("roof")
    _house(m, Lx, D, eave, pitch, R(wall), roof)
    if kind == "building.garage":
        m.quad([[-2.6, -0.02, 0], [2.6, -0.02, 0], [2.6, -0.02, 2.3], [-2.6, -0.02, 2.3]], R("garage_door"), out=np.array([0, -1.0, 0]))
    if kind == "building.barn":
        m.quad([[-2.0, -0.02, 0], [2.0, -0.02, 0], [2.0, -0.02, 3.6], [-2.0, -0.02, 3.6]], R("wood_white"), out=np.array([0, -1.0, 0]))
    m.cbox(0, D / 2, Lx, D, -0.5, eave + 1.0)
    return m


def g_trailer(L, p):
    m = MB(); Lx, D = 14.0, 4.2
    m.box(0, D / 2, 0.6, 3.4, Lx, D, R("trailer"), top=R("galv"))
    m.box(0, D / 2, 0.0, 0.6, Lx - 1, D - 0.4, R("black"))
    m.cbox(0, D / 2, Lx, D, -0.5, 3.4); return m


def g_tank(L, p, kind="black"):
    m = MB(); r = float(p.get("radius", 1.8)); h = float(p.get("height", 3.2))
    m.cyl(0, r, 0, h, r, 12, R("tank_black" if kind == "black" else ("tank_green" if kind == "green" else "galv")), top=R("black"))
    m.cyl(0, r, h, h + 0.35, r, 12, R("black"), r1=0.2, top=R("black"))
    m.cbox(0, r, 2 * r, 2 * r, -0.5, h); return m


def g_carport(L, p):
    m = MB(); Lx, D, h = 6.0, 6.0, 2.6
    for x in (-Lx / 2, Lx / 2):
        for y in (0, D):
            m.box(x, y, 0, h, 0.1, 0.1, R("galv"))
    m.quad([[-Lx / 2 - 0.2, -0.2, h], [Lx / 2 + 0.2, -0.2, h], [Lx / 2 + 0.2, D + 0.2, h + 0.3], [-Lx / 2 - 0.2, D + 0.2, h + 0.3]], R("corr_grey"), out=np.array([0, 0, 1.0]))
    m.quad([[-Lx / 2 - 0.2, -0.2, h - 0.02], [-Lx / 2 - 0.2, D + 0.2, h + 0.28], [Lx / 2 + 0.2, D + 0.2, h + 0.28], [Lx / 2 + 0.2, -0.2, h - 0.02]], R("corr_grey"), out=np.array([0, 0, -1.0]))
    return m


# ---- driveways -----------------------------------------------------------------------------------------------------
def g_apron(L, p, tile="apron"):
    m = MB(); depth = float(p.get("depth", 4.5)); L = max(L or 6.0, 3.0); inset = float(p.get("inset", 0.3))
    m.flat(0, L, -inset, depth, 0.07, R(tile), nx=max(1, int(L / 1.5)), ny=4)
    m.meta["drape"] = True
    return m


# ---- gates ---------------------------------------------------------------------------------------------------------
def g_gate(L, p, kind="gate.estate_brick_iron"):
    m = MB(); L = max(L or 4.4, 2.0)
    if kind == "gate.estate_brick_iron":
        for x in (0, L):
            m.box(x, 0, -0.4, 2.3, 0.6, 0.6, R("brick"), top=R("concrete"))
            m.box(x, 0, 2.3, 2.45, 0.7, 0.7, R("concrete"))
        m.sheet(0.3, L - 0.3, 0, 0.1, 2.0, R("picket"), 1.6)
    elif kind == "gate.pipe_ranch":
        for x in (0, L):
            m.box(x, 0, -0.4, 1.4, 0.15, 0.15, R("wood_dark"))
        m.sheet(0.1, L - 0.1, 0, 0.1, 1.3, R("pipegate"), L)
    elif kind == "gate.chainlink_swing":
        for x in (0, L):
            m.box(x, 0, -0.4, 1.9, 0.08, 0.08, R("galv"))
        m.sheet(0.05, L - 0.05, 0, 0.0, 1.8, R("chainlink"), 0.64)
    else:   # gate.white_rail
        for x in (0, L):
            m.box(x, 0, -0.4, 1.5, 0.18, 0.18, R("wood_white"))
        m.sheet(0.1, L - 0.1, 0, 0.1, 1.3, R("whiterail"), 2.4)
    m.cline(0.0, -0.7, 1.4)
    return m


# ---- vegetation (alpha cards in the same atlas) ------------------------------------------------------------------
SHRUB = {"shrub.coyote_brush": ("veg_coyote", 2.2, 1.5), "shrub.poison_oak": ("veg_poisonoak", 1.6, 1.2),
         "shrub.blackberry": ("veg_blackberry", 3.0, 1.6), "shrub.fern": ("veg_fern", 1.2, 0.7),
         "shrub.grass_clump": ("veg_grass", 0.9, 0.9), "shrub.ivy": ("veg_ivy", 2.0, 0.5), "shrub.toyon": ("veg_hedge", 2.6, 2.6)}


def g_shrub(L, p, kind="shrub.coyote_brush"):
    tile, w, h = SHRUB[kind]; s = float(p.get("scale", 1.0)); m = MB(); rng = _rng(p)
    m.card(0, 0, w * s, h * s, R(tile), n=3, z0=-0.1, rot=rng.uniform(0, math.pi))
    return m


def g_shrub_line(L, p, kind="shrub.blackberry"):
    """hedge-like linear strip of shrub cards (blackberry thicket, ivy band, grass verge)"""
    tile, w, h = SHRUB[kind]; m = MB(); rng = _rng(p); dens = float(p.get("density", 1.0))
    for x in np.arange(0, L + 0.01, max(0.4, w * 0.6 / dens)):
        s = rng.uniform(0.75, 1.2); m.card(x + rng.uniform(-0.3, 0.3), rng.uniform(0.0, 0.8), w * s, h * s, R(tile), n=2, z0=-0.1, rot=rng.uniform(0, 3.14))
    return m


def g_hedge(L, p):
    """clipped hedge: box with leaf faces"""
    m = MB(); h = float(p.get("height", 1.8)); th = float(p.get("width", 1.2))
    m.slab(0, L, 0.0, th, -0.1, h, R("veg_hedge"), tw=2.0, seg=2.0)
    return m


def g_cypress_row(L, p):
    m = MB(); rng = _rng(p); h = float(p.get("height", 12.0)); sp = float(p.get("spacing", 3.5))
    for x in np.arange(0, (L or 0) + 0.01, sp):
        hh = h * rng.uniform(0.9, 1.1); w = max(2.2, hh * 0.2)
        m.card(x, 0.0, w, hh, R("cypress"), n=2, z0=-0.3)
        m.cbox(x, 0, 0.5, 0.5, -0.5, 2.5)
    return m


def g_tree_card(L, p):
    """tree_add: coast-live-oak crown card + collider trunk"""
    m = MB(); h = float(p.get("height", 9.0)); w = float(p.get("crown", h * 0.9))
    m.card(0, 0, w, h, R("veg_oak"), n=3, z0=-0.2)
    m.cbox(0, 0, 0.5, 0.5, -0.5, 2.5)
    return m


# ---- rocks / debris ------------------------------------------------------------------------------------------------
def g_rock(L, p, size=1.0):
    m = MB(); rng = _rng(p); s = float(p.get("scale", 1.0)) * size
    m.rock(0, 0, 0.6 * s, 0.9 * s, R("rock"), rng)
    if s >= 0.8:
        m.cbox(0, 0, 1.0 * s, 1.0 * s, -0.4, max(1.4, 0.8 * s))
    return m


def g_rock_cluster(L, p):
    m = MB(); rng = _rng(p); n = int(p.get("count", 4))
    for k in range(n):
        s = rng.uniform(0.4, 1.1); m.rock(rng.uniform(-1.5, 1.5), rng.uniform(0, 1.5), 0.6 * s, 0.8 * s, R("rock" if k % 2 else "rock_dark"), rng)
    m.cbox(0, 0.75, 3.0, 1.8, -0.4, 1.4)
    return m


def g_outcrop(L, p):
    m = MB(); rng = _rng(p); s = float(p.get("scale", 1.0))
    m.rock(0, 0.8 * s, 2.2 * s, 2.4 * s, R("rock_dark"), rng); m.rock(1.6 * s, 0.4 * s, 1.0 * s, 1.2 * s, R("rock"), rng)
    m.cbox(0.5 * s, 0.8 * s, 4.0 * s, 2.4 * s, -0.5, 2.0)
    return m


def g_log(L, p):
    m = MB(); Ll = float(p.get("length", 4.0)); r = float(p.get("radius", 0.25))
    m.hcyl(-Ll / 2, Ll / 2, 0.0, r * 0.8, r, 6, R("bark"))
    for x in (-Ll / 2, Ll / 2):
        m.quad([[x, -r, 0], [x, r, 0], [x, r, 1.8 * r], [x, -r, 1.8 * r]], R("stump_top"), out=np.array([1.0 if x > 0 else -1.0, 0, 0]))
    return m


def g_stump(L, p):
    m = MB(); r = float(p.get("radius", 0.35)); m.cyl(0, 0, -0.1, 0.45, r, 7, R("bark"), top=R("stump_top")); return m


def g_branch_pile(L, p):
    m = MB(); s = float(p.get("scale", 1.0)); rng = _rng(p)
    m.card(0, 0, 3.0 * s, 1.2 * s, R("branches"), n=3, z0=-0.05, rot=rng.uniform(0, 3.14)); return m


def g_litter(L, p):
    m = MB(); s = float(p.get("scale", 1.0)); m.flat(-1.5 * s, 1.5 * s, -1.5 * s, 1.5 * s, 0.03, R("veg_litter"), 2, 2); m.meta["drape"] = True
    return m


# ---- markings (on the pavement; off < 0) ---------------------------------------------------------------------------
def g_hump_marking(L, p):
    m = MB(); L = max(L or 3.0, 2.0); w = float(p.get("width", 5.0))
    m.flat(0, L, -w / 2, w / 2, 0.028, R("hump"), nx=max(1, int(L)), ny=3); m.meta["drape"] = "road"; return m


def g_stop_bar(L, p):
    m = MB(); w = float(p.get("width", 3.0)); m.flat(0, 0.45, -w / 2, w / 2, 0.02, R("paint_white"), 1, 3); m.meta["drape"] = "road"; return m


def g_none(L, p):
    return MB()


# ============================================================================= catalog
def _p(f, **kw):
    return lambda L, p: f(L, p, **kw)


#            id: (cls, gen, (L, W, H), proxy, collide, seg_class, linear, desc)
CATALOG = {}


def reg(aid, cls, gen, dims, proxy, collide, seg, linear=False, desc="", mats=("ledger",)):
    CATALOG[aid] = dict(id=aid, cls=cls, gen=gen, dims=list(dims), proxy=proxy, collide=collide, seg=seg, linear=linear, desc=desc, mats=list(mats))


for aid, txt in SIGN_TEXT.items():
    reg(aid, "sign", _p(g_sign, asset=aid), (0.06, 0.75, 2.1), dict(shape="plane", w=0.75, h=0.75, z0=1.35), True, 12,
        desc=f"MUTCD-style sign on a galvanised post ({txt}); front design + grey back")
for aid in PLAQUES:
    reg(aid, "sign", _p(g_sign, asset=aid), (0.06, 0.6, 1.8), dict(shape="plane", w=0.6, h=0.45, z0=1.3), True, 12, desc="plaque on its own post")
reg("post.delineator_white", "post", g_delineator, (0.08, 0.08, 1.1), dict(shape="box", w=0.08, d=0.08, h=1.1), False, 14, desc="white flexible delineator, amber reflector")
reg("post.delineator_yellow", "post", _p(g_delineator, col="yellow"), (0.08, 0.08, 1.1), dict(shape="box", w=0.08, d=0.08, h=1.1), False, 14, desc="yellow delineator")
reg("post.hazard_marker", "post", g_hazard_marker, (0.06, 0.3, 1.2), dict(shape="plane", w=0.3, h=0.6, z0=0.6), True, 12, desc="OM3 object marker on post")
reg("post.wood_bollard", "post", g_bollard, (0.15, 0.15, 0.9), dict(shape="box", w=0.15, d=0.15, h=0.9), True, 14)
reg("post.t_post", "post", g_tpost, (0.05, 0.05, 1.3), dict(shape="box", w=0.05, d=0.05, h=1.3), False, 14, desc="green steel T-post")
reg("pole.utility_wood", "pole", g_pole, (0.3, 2.4, 10.5), dict(shape="cylinder", r=0.15, h=10.5), True, 14, desc="wood utility pole, crossarm, 3 insulators; wires to the next pole")
reg("pole.utility_transformer", "pole", _p(g_pole, transformer=True), (0.3, 2.4, 10.5), dict(shape="cylinder", r=0.15, h=10.5), True, 14, desc="pole with can transformer")
reg("pole.streetlight", "pole", g_streetlight, (0.2, 1.8, 8.0), dict(shape="cylinder", r=0.1, h=8.0), True, 14)
reg("mailbox.single", "mailbox", _p(g_mailbox, n=1), (0.48, 0.42, 1.25), dict(shape="box", w=0.42, d=0.5, h=1.25), True, 23)
reg("mailbox.cluster", "mailbox", _p(g_mailbox, n=3), (0.48, 1.3, 1.25), dict(shape="box", w=1.3, d=0.5, h=1.25), True, 23, desc="row of rural mailboxes on a board")
reg("mailbox.brick_column", "mailbox", g_mailbox_brick, (0.6, 0.6, 1.3), dict(shape="box", w=0.6, d=0.6, h=1.3), True, 23)
reg("mailbox.cbu", "mailbox", g_cbu, (0.9, 0.5, 1.6), dict(shape="box", w=0.9, d=0.5, h=1.6), True, 23, desc="grey cluster box unit")
reg("prop.trash_bins", "prop", g_bin, (1.5, 0.7, 1.05), dict(shape="box", w=1.5, d=0.7, h=1.05), False, 24, desc="wheelie bins (blue/green/black)")
reg("prop.utility_box", "prop", g_utility_box, (0.9, 0.6, 1.2), dict(shape="box", w=0.9, d=0.6, h=1.2), True, 24)
reg("prop.hydrant", "prop", g_hydrant, (0.42, 0.3, 0.75), dict(shape="cylinder", r=0.15, h=0.75), True, 24)
reg("guardrail.wbeam_steel", "guardrail", g_wbeam, (1, 0.3, 0.78), dict(shape="plane", h=0.78), True, 15, True, "W-beam on steel posts")
reg("guardrail.wbeam_wood", "guardrail", _p(g_wbeam, wood=True), (1, 0.3, 0.78), dict(shape="plane", h=0.78), True, 15, True, "W-beam on wood posts")
reg("guardrail.timber_rail", "guardrail", g_timber_rail, (1, 0.3, 0.9), dict(shape="plane", h=0.9), True, 15, True, "two-rail timber guard rail")
reg("guardrail.cable", "guardrail", g_cable_rail, (1, 0.1, 0.85), dict(shape="plane", h=0.85), True, 15, True)
for aid, (tile, tw, hd, *_r) in FENCE.items():
    reg(aid, "fence", _p(g_fence, kind=aid), (1, 0.1, hd), dict(shape="plane", h=hd), True, 16, True, f"{tile} panels on posts")
for aid, (tile, th, hd) in WALL.items():
    reg(aid, "wall", _p(g_wall, kind=aid), (1, th, hd), dict(shape="box", d=th, h=hd), True, 17, True, f"{tile} wall")
reg("wall.rockfall_mesh", "wall", g_rockfall_mesh, (1, 0.1, 4.0), dict(shape="plane", h=4.0), True, 17, True, "draped rockfall mesh on the cut")
reg("bridge.parapet_concrete", "bridge", _p(g_parapet, tile="concrete"), (1, 0.3, 0.8), dict(shape="box", d=0.3, h=0.8), True, 19, True, "concrete slab bridge parapet")
reg("bridge.parapet_stone", "bridge", _p(g_parapet, tile="stone", th=0.4), (1, 0.4, 0.8), dict(shape="box", d=0.4, h=0.8), True, 19, True, "stone masonry parapet")
reg("bridge.parapet_rail", "bridge", _p(g_parapet, tile="concrete", h=0.6, rail=True), (1, 0.3, 0.9), dict(shape="box", d=0.3, h=0.9), True, 19, True, "low concrete parapet + steel tube rail")
reg("bridge.timber_rail", "bridge", g_bridge_timber, (1, 0.3, 1.0), dict(shape="plane", h=1.0), True, 19, True, "timber bridge rail on posts")
reg("bridge.steel_girder", "bridge", g_bridge_steel, (1, 0.3, 1.05), dict(shape="plane", h=1.05), True, 19, True, "steel girder fascia + tube railing")
reg("culvert.headwall", "culvert", g_headwall, (4, 0.35, 0.6), dict(shape="box", w=4, d=0.35, h=0.6), True, 17, True, "concrete culvert headwall")
reg("culvert.cmp_end", "culvert", g_cmp, (0.9, 0.1, 0.9), dict(shape="plane", w=0.9, h=0.9), False, 24, desc="corrugated pipe outlet + rip-rap")
reg("culvert.drop_inlet", "culvert", g_drop_inlet, (1.0, 0.8, 0.15), dict(shape="box", w=1, d=0.8, h=0.15), False, 24)
for aid in ("building.ranch_house", "building.house_two_storey", "building.barn", "building.shed", "building.garage", "building.stable"):
    reg(aid, "building", _p(g_building, kind=aid), (16, 10, 5), dict(shape="box", w=16, d=10, h=5), True, 18, desc=aid.split(".")[1].replace("_", " "))
reg("building.trailer", "building", g_trailer, (14, 4.2, 3.4), dict(shape="box", w=14, d=4.2, h=3.4), True, 18, desc="mobile home / trailer")
reg("building.water_tank", "building", _p(g_tank, kind="black"), (3.6, 3.6, 3.5), dict(shape="cylinder", r=1.8, h=3.5), True, 18, desc="black poly water tank")
reg("building.water_tank_green", "building", _p(g_tank, kind="green"), (3.6, 3.6, 3.5), dict(shape="cylinder", r=1.8, h=3.5), True, 18)
reg("building.water_tank_steel", "building", _p(g_tank, kind="steel"), (3.6, 3.6, 3.5), dict(shape="cylinder", r=1.8, h=3.5), True, 18)
reg("building.carport", "building", g_carport, (6, 6, 2.9), dict(shape="box", w=6, d=6, h=2.9), False, 18)
reg("driveway.asphalt", "driveway", g_apron, (6, 4.5, 0.07), dict(shape="plane_h", d=4.5), False, 2, True, "asphalt driveway apron")
reg("driveway.gravel", "driveway", _p(g_apron, tile="gravel"), (6, 4.5, 0.07), dict(shape="plane_h", d=4.5), False, 5, True, "gravel / dirt driveway apron")
for aid in ("gate.estate_brick_iron", "gate.pipe_ranch", "gate.chainlink_swing", "gate.white_rail"):
    reg(aid, "gate", _p(g_gate, kind=aid), (4.4, 0.6, 2.4), dict(shape="plane", h=2.0), True, 16, True, aid.split(".")[1].replace("_", " "))
for aid, (tile, w, h) in SHRUB.items():
    reg(aid, "shrub", _p(g_shrub, kind=aid), (w, w, h), dict(shape="cylinder", r=w / 2, h=h), False, 10, desc=tile[4:] + " card clump")
reg("hedge.blackberry_thicket", "hedge", _p(g_shrub_line, kind="shrub.blackberry"), (1, 2.0, 1.6), dict(shape="box", d=1.6, h=1.6), False, 10, True)
reg("hedge.ivy_band", "hedge", _p(g_shrub_line, kind="shrub.ivy"), (1, 1.5, 0.5), dict(shape="box", d=1.5, h=0.5), False, 10, True)
reg("hedge.grass_verge", "hedge", _p(g_shrub_line, kind="shrub.grass_clump"), (1, 1.0, 0.9), dict(shape="box", d=1.0, h=0.9), False, 6, True, "tall spring grass along the edge")
reg("hedge.coyote_brush", "hedge", _p(g_shrub_line, kind="shrub.coyote_brush"), (1, 2.0, 1.5), dict(shape="box", d=2.0, h=1.5), False, 10, True)
reg("hedge.clipped", "hedge", g_hedge, (1, 1.2, 1.8), dict(shape="box", d=1.2, h=1.8), False, 10, True, "clipped privet/laurel hedge")
reg("hedge.cypress_row", "hedge", g_cypress_row, (1, 2.4, 12), dict(shape="box", d=2.4, h=12), True, 9, True, "Italian cypress row")
reg("rock.boulder_s", "rock", _p(g_rock, size=0.5), (0.6, 0.5, 0.45), dict(shape="box", w=0.6, d=0.5, h=0.45), False, 8)
reg("rock.boulder_m", "rock", _p(g_rock, size=1.0), (1.2, 1.0, 0.9), dict(shape="box", w=1.2, d=1.0, h=0.9), True, 8)
reg("rock.boulder_l", "rock", _p(g_rock, size=1.8), (2.2, 1.8, 1.6), dict(shape="box", w=2.2, d=1.8, h=1.6), True, 8)
reg("rock.cluster", "rock", g_rock_cluster, (3.0, 1.8, 0.9), dict(shape="box", w=3, d=1.8, h=0.9), True, 8)
reg("rock.outcrop", "rock", g_outcrop, (4.0, 2.4, 2.4), dict(shape="box", w=4, d=2.4, h=2.4), True, 8)
reg("debris.log", "debris", g_log, (4.0, 0.5, 0.45), dict(shape="box", w=4, d=0.5, h=0.45), False, 24)
reg("debris.stump", "debris", g_stump, (0.7, 0.7, 0.45), dict(shape="cylinder", r=0.35, h=0.45), False, 11)
reg("debris.branch_pile", "debris", g_branch_pile, (3.0, 3.0, 1.2), dict(shape="box", w=3, d=3, h=1.2), False, 24)
reg("debris.leaf_litter", "debris", g_litter, (3, 3, 0.03), dict(shape="plane_h", d=3), False, 5)
reg("marking.hump", "marking", g_hump_marking, (3, 5, 0.03), dict(shape="plane_h", d=5), False, 4, True, "white speed-hump triangles")
reg("marking.stop_bar", "marking", g_stop_bar, (0.45, 3, 0.02), dict(shape="plane_h", d=3), False, 4)
reg("terrain.mound", "mound", g_none, (6, 6, 1), dict(shape="none"), False, 6, desc="DTM edit (gaussian/capsule)")
reg("terrain.ditch", "ditch", g_none, (6, 2, 0.5), dict(shape="none"), False, 5, desc="DTM edit (negative)")
reg("tree.density", "tree_density", g_none, (1, 1, 1), dict(shape="none"), False, 10, True, "shrub density multiplier / canopy fill")
reg("tree.oak", "tree_add", g_tree_card, (8, 8, 9), dict(shape="cylinder", r=4, h=9), True, 9, desc="extra coast live oak card")
reg("tree.remove", "tree_remove", g_none, (1, 1, 1), dict(shape="none"), False, 9, desc="remove LiDAR tree(s) within params.radius")

def generate(aid, L=None, params=None):
    e = CATALOG[aid]; p = dict(params or {})
    return e["gen"](L if e["linear"] else None, p)


# ============================================================================= yaml + contact sheet
def frames_by_asset():
    """frames where each asset type was surveyed (legacy roadside_assets via the seed mapping) + by-eye notes"""
    import json, glob
    sys.path.insert(0, str(HERE))
    from ledger_seed import legacy_to_rows
    out = {}
    for f in sorted(glob.glob(str(ROOT / "data/derived/roadside_assets_*.json"))):
        for e in json.load(open(f)):
            for r in legacy_to_rows(e):
                out.setdefault(r["asset"], set()).update(e.get("frames", []))
    for aid, fr in EYE_FRAMES.items():
        out.setdefault(aid, set()).update(fr)
    return {k: sorted(v) for k, v in out.items()}


EYE_FRAMES = {   # seen by eye in the GoPro frames (contact sheets of every 5th frame) but absent from the survey files
    "rock.boulder_l": [1], "rock.cluster": [1], "debris.branch_pile": [26, 31], "shrub.fern": [76, 81], "shrub.coyote_brush": [36, 46, 186],
    "hedge.blackberry_thicket": [61, 66], "fence.lattice": [191], "fence.iron_picket": [206], "gate.estate_brick_iron": [141],
    "gate.pipe_ranch": [136], "building.water_tank": [171], "prop.trash_bins": [56], "sign.advisory_15": [91],
    "guardrail.timber_rail": [56], "hedge.cypress_row": [211], "mailbox.brick_column": [201], "sign.address_fire": [141],
    "shrub.grass_clump": [166, 196], "shrub.ivy": [156], "hedge.clipped": [196], "debris.stump": [136], "tree.oak": [101, 131],
}


SEG_NAMES = {0: "unknown", 1: "sky", 2: "road", 3: "marking_yellow", 4: "marking_white", 5: "shoulder_dirt", 6: "terrain_grass",
             7: "terrain_bare", 8: "rock_cut", 9: "vegetation_tree", 10: "shrub", 11: "trunk", 12: "sign_face", 13: "sign_back",
             14: "pole", 15: "guardrail", 16: "fence", 17: "wall", 18: "building", 19: "bridge", 22: "water", 23: "mailbox_prop", 24: "other_prop"}


def write_yaml():
    import yaml
    fr = frames_by_asset()
    import re, inspect
    src = inspect.getsource(sys.modules[__name__])
    rows = []
    for aid, e in CATALOG.items():
        m = MB() if e["cls"] in ("mound", "ditch", "tree_density", "tree_remove") else generate(aid, 10.0 if e["linear"] else None, {})
        P, T, UV, N = m.arrays()
        gname = e["gen"].__name__
        if gname == "<lambda>":
            gname = {"sign": "g_sign", "fence": "g_fence", "wall": "g_wall", "shrub": "g_shrub", "gate": "g_gate", "building": "g_building"}.get(e["cls"], "g_?")
            if aid.startswith("hedge."):
                gname = "g_shrub_line"
            if aid.startswith(("bridge.parapet",)):
                gname = "g_parapet"
            if aid in ("mailbox.single", "mailbox.cluster"):
                gname = "g_mailbox"
            if aid.startswith("pole.utility"):
                gname = "g_pole"
            if aid.startswith("post.delineator"):
                gname = "g_delineator"
            if aid.startswith("guardrail.wbeam"):
                gname = "g_wbeam"
            if aid.startswith("rock.boulder"):
                gname = "g_rock"
            if aid.startswith("building.water_tank"):
                gname = "g_tank"
            if aid == "driveway.gravel":
                gname = "g_apron"
        px = dict(e["proxy"]); L_, W_, H_ = e["dims"]
        px.setdefault("w", L_); px.setdefault("d", px.get("r", W_ / 2) * 2 if px.get("shape") == "cylinder" else (0.1 if px.get("shape") == "plane" else W_)); px.setdefault("h", H_)
        z0 = px.pop("z0", 0.0)
        rows.append(dict(id=aid, cls=e["cls"], linear=e["linear"], dims_m=dict(zip(("L", "W", "H"), e["dims"])), proxy=px, z0=z0,
                         collision=bool(e["collide"]), seg_class=SEG_NAMES.get(e["seg"], "other_prop"), materials=e["mats"],
                         tris=int(len(T)) if not e["linear"] else None, tris_per_10m=int(len(T)) if e["linear"] else None,
                         frames=fr.get(aid, []), generator=f"scripts/p5/assets.py:{gname}", desc=e["desc"]))
    hdr = ("# dev5 asset catalog: every asset type the scene ledger (data/scene/ledger.jsonl) may reference.\n"
           "# GENERATED by `python3 scripts/p5/assets.py --yaml` from scripts/p5/assets.py CATALOG; edit there.\n"
           "# proxy: simple shape for the scorer (local frame x fwd, y out, z up; plane = vertical sheet facing the road,\n"
           "#   plane_h = flat on the ground, sign planes face -x).  materials: one atlas material 'ledger' (build/ledger_atlas.png).\n"
           "# seg_class: class name from data/scene/seg_classes.json; z0: proxy base above ground.  frames: GoPro frames where the type is seen (survey files + by eye).\n")
    (ROOT / "config/asset_catalog.yaml").write_text(hdr + yaml.safe_dump(dict(version=1, material="ledger", atlas="build/ledger_atlas.png",
                                                                                assets=rows), sort_keys=False, width=140))
    print("wrote config/asset_catalog.yaml:", len(rows), "types")


def contact_sheet():
    """one matplotlib thumbnail per asset (textured, painter's-algorithm polygons) -> reports/dev5/catalog.html"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import html, io, base64, collections
    A = atlas()
    for aid in CATALOG:                       # signs paint their faces into the atlas lazily: generate everything first
        generate(aid, 4.0 if CATALOG[aid]["linear"] else None, {})
    img = np.asarray(A.img.convert("RGBA")).astype(float) / 255; SH, SW = img.shape[:2]
    out = ROOT / "reports/dev5"; (out / "catalog").mkdir(parents=True, exist_ok=True)
    cells = []; by_cls = collections.defaultdict(list)
    for aid, e in CATALOG.items():
        if e["cls"] in ("mound", "ditch", "tree_density", "tree_remove"):
            continue
        L = {"guardrail": 8.0, "fence": 8.0, "wall": 8.0, "bridge": 8.0, "hedge": 10.0, "marking": 3.0, "driveway": 6.0, "gate": 4.4, "culvert": 4.0}.get(e["cls"], 6.0)
        m = generate(aid, L if e["linear"] else None, {})
        P, T, UV, N = m.arrays()
        if not len(T):
            continue
        # view: camera on the road side, looking at the asset front (-y side) from slightly left, like a driver
        az = math.radians(-55 if e["cls"] != "sign" else -140); el = math.radians(12)
        cx = P.mean(0)
        d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])     # toward the camera
        r = np.cross([0, 0, 1], d); r /= np.linalg.norm(r); u = np.cross(d, r)
        Q = P - cx; X, Y, Z = Q @ r, Q @ u, Q @ d
        # textured z-buffer rasteriser with alpha test (what the alpha-tested AC shader shows)
        RES = 240; ext = max(np.ptp(X), np.ptp(Y)) * 0.55 + 0.1
        sx = (X - X.mean()) / ext * RES / 2 + RES / 2; sy = RES / 2 - (Y - Y.mean()) / ext * RES / 2
        zb = np.full((RES, RES), -1e9); col = np.zeros((RES, RES, 3)); col[:] = (0.80, 0.86, 0.90)
        fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
        for k, t in enumerate(T):
            if np.dot(fn[k], d) <= 0:
                continue
            ax_, ay_ = sx[t], sy[t]; x0, x1 = int(max(0, ax_.min())), int(min(RES - 1, ax_.max()) + 1); y0, y1 = int(max(0, ay_.min())), int(min(RES - 1, ay_.max()) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            den = (ay_[1] - ay_[2]) * (ax_[0] - ax_[2]) + (ax_[2] - ax_[1]) * (ay_[0] - ay_[2])
            if abs(den) < 1e-9:
                continue
            w0 = ((ay_[1] - ay_[2]) * (gx - ax_[2]) + (ax_[2] - ax_[1]) * (gy - ay_[2])) / den
            w1 = ((ay_[2] - ay_[0]) * (gx - ax_[2]) + (ax_[0] - ax_[2]) * (gy - ay_[2])) / den; w2 = 1 - w0 - w1
            ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not ins.any():
                continue
            zz = w0 * Z[t[0]] + w1 * Z[t[1]] + w2 * Z[t[2]]
            uu = w0 * UV[t[0], 0] + w1 * UV[t[1], 0] + w2 * UV[t[2], 0]; vv = w0 * UV[t[0], 1] + w1 * UV[t[1], 1] + w2 * UV[t[2], 1]
            tx = img[np.clip((vv * SH).astype(int), 0, SH - 1), np.clip((uu * SW).astype(int), 0, SW - 1)]
            sub = zb[y0:y1, x0:x1]; ok = ins & (tx[..., 3] > 0.5) & (zz > sub)
            sh = 0.6 + 0.4 * max(0.0, float(np.dot(fn[k], light)))
            sub[ok] = zz[ok]; col[y0:y1, x0:x1][ok] = tx[..., :3][ok] * sh
        fig = plt.figure(figsize=(2.4, 2.4), dpi=80); axx = fig.add_axes([0, 0, 1, 1]); axx.imshow(col, interpolation="bilinear"); axx.axis("off")
        b = io.BytesIO(); fig.savefig(b, format="png"); plt.close(fig)
        (out / "catalog" / f"{aid}.png").write_bytes(b.getvalue())
        by_cls[e["cls"]].append((aid, len(T), base64.b64encode(b.getvalue()).decode(), e["desc"]))
    css = ("body{font:13px system-ui;margin:16px;background:#f4f4f2;color:#222}h2{margin:18px 0 6px}"
           ".g{display:flex;flex-wrap:wrap;gap:8px}.c{background:#fff;border:1px solid #ddd;padding:4px;width:200px}"
           ".c img{width:192px;height:192px}.c b{display:block;font-size:12px}.c i{color:#666;font-size:11px}"
           "@media (prefers-color-scheme:dark){body{background:#1b1b1b;color:#ddd}.c{background:#262626;border-color:#333}}")
    h = [f"<!doctype html><meta charset=utf-8><title>Asset Catalog</title><style>{css}</style><h1>dev5 asset catalog ({len(CATALOG)} types)</h1>"]
    for cls, lst in by_cls.items():
        h.append(f"<h2>{cls} ({len(lst)})</h2><div class=g>")
        for aid, nt, b64, desc in lst:
            h.append(f"<div class=c><img src='data:image/png;base64,{b64}'><b>{aid}</b><i>{nt} tris. {html.escape(desc)}</i></div>")
        h.append("</div>")
    (out / "catalog.html").write_text("".join(h))
    # also a single PNG grid for quick viewing
    ims = sorted((out / "catalog").glob("*.png")); cols_ = 10; rows_ = math.ceil(len(ims) / cols_)
    grid = Image.new("RGB", (cols_ * 192, rows_ * 206), (255, 255, 255)); dr = ImageDraw.Draw(grid)
    for i, f in enumerate(ims):
        t = Image.open(f).convert("RGB").resize((192, 192)); grid.paste(t, ((i % cols_) * 192, (i // cols_) * 206))
        dr.text(((i % cols_) * 192 + 2, (i // cols_) * 206 + 193), f.stem[:30], fill=(0, 0, 0))
    grid.save(out / "catalog_grid.jpg", quality=85)
    A.img.save(out / "ledger_atlas_preview.png")
    print("wrote", out / "catalog.html", len(ims), "thumbnails")


if __name__ == "__main__":
    if "--yaml" in sys.argv:
        write_yaml()
    if "--sheet" in sys.argv:
        contact_sheet()
