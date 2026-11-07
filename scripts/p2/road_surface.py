"""Procedural road surface + markings for Palomares Rd (CC0 Asphalt026C structure + procedural detail).

Shared by 21_build_geometry.py (UV mapping, decal placement, RPMs) and 22_build_kn5.py (textures).

Road UV ("atlas" mapping):  u = 0.5 + d / ATLAS_M   (d = lateral offset, + = left, centre line at u = 0.5)
                            v = s / ATLAS_M          (s = station)
so lateral features (centre grime band, wheel-track polish, oil drip in the lane centres) sit at fixed
offsets from the centre line, and the texture repeats only every ATLAS_M = 12 m in both directions.

Markings (US MUTCD 2009/2023, sec. 3A.05-3A.06, 3B.01, 3B.11): normal line width 4-6 in -> 0.10 m;
double line = two normal lines separated by a discernible space (4 in typical) -> 0.10 m gap;
broken line 10 ft dash / 30 ft gap -> 3.05 / 9.14 m (12.19 m period). Raised pavement markers (sec. 3B.11
/ Caltrans) on the centre line every 24 m (80 ft), yellow, flanking the double line.
Paint colours: median of yellow / white pixels in the freshest-marking GoPro frames 166-176.
"""
import json, math, re
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ATLAS_M = 12.0
LINE_W, DY_GAP = 0.10, 0.10
DY_RIBBON = 2 * LINE_W + DY_GAP + 2 * 0.03          # 3 cm feather margin each side
EDGE_RIBBON = LINE_W + 2 * 0.03
LINE_PERIOD_M = 8.0                                   # v period of the line textures (wear pattern)
BROKEN = (3.05, 12.19)
RPM_SPACING = 24.0
# lifts above the road surface. dev6: the only decal GEOMETRY left is the surveyed patches (alpha-TESTED ksPerPixelAT, depth
# write on, 1.0 cm lift); crack seal, hairlines and crazing are baked into the 12 m asphalt atlas (asphalt_atlas). The old
# alpha-blended no-depth-write decal layer sorted/z-fought in AC and showed its whole quads ("random textures in the road").
# decal atlas (2048 px, v measured from the top of the image). Cells are inset 1.5 px against bilinear bleed.
#   y   0..256 : 8 crack-seal ribbon cells (128x256 px = ribbon width x 0.5 m of crack) | 8 hairline-crack cells
#   y 256..1504: 3 patch tones (0 new/black, 1 mid, 2 old/grey), 416 px each: lane 1024x416 | strip 1024x192 + 2 small 512x224
#   y 1536..2048: 4 alligator crazing cells 512x512
_AT = 2048
_INS = 1.5


def _cell(x0, y0, w, h):
    return ((x0 + _INS) / _AT, (y0 + _INS) / _AT, (x0 + w - _INS) / _AT, (y0 + h - _INS) / _AT)


CELLS = {"seal": [_cell(k * 128, 0, 128, 256) for k in range(8)],
         "hair": [_cell(1024 + k * 128, 0, 128, 256) for k in range(8)],
         "network": [_cell(k * 512, 1536, 512, 512) for k in range(4)]}
PATCH_CELLS = []   # per tone: dict(lane=cell, strip=cell, small=[cell, cell]); nominal (long, short) metres below
for _t in range(3):
    _y = 256 + _t * 416
    PATCH_CELLS.append(dict(lane=_cell(0, _y, 1024, 416), strip=_cell(1024, _y, 1024, 192),
                            small=[_cell(1024, _y + 192, 512, 224), _cell(1536, _y + 192, 512, 224)]))
PATCH_ASPECT = dict(lane=1024 / 416, strip=1024 / 192, small=512 / 224)   # long side / short side in the cell
CELLS["patch"] = PATCH_CELLS
SEAL_CELL_M = (0.15, 0.5)      # ribbon width (m) x crack length per cell (m) at unit scale
LIFT = dict(patch=0.010, network=0.018, hair=0.019, seal=0.020, line=0.023, rpm=0.024)


def paint_colours(root):
    """median paint colour from the freshest GoPro frames (yellow: R>150,G>110,B<90; white: bright, neutral)"""
    try:
        Y, W = [], []
        for f in range(166, 177):
            p = root / f"assets/gopro/{f:03d}.png"
            if not p.exists():
                continue
            a = np.asarray(Image.open(p).convert("RGB"), float)[800:]
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            Y.append(a[(r > 150) & (g > 110) & (b < 90) & (r - b > 90)])
            W.append(a[(a.min(-1) > 185) & (a.max(-1) - a.min(-1) < 20)])
        y, w = np.median(np.concatenate(Y), 0), np.median(np.concatenate(W), 0)
        return tuple(int(v) for v in y), tuple(int(v) for v in w)
    except Exception:
        return (212, 167, 56), (220, 218, 212)


# ------------------------------------------------------------------ periodic noise helpers
def pnoise(shape, scale_px, seed, aniso=(1.0, 1.0)):
    """tileable band-limited noise (FFT), zero mean unit std. scale_px = feature size; aniso = (row, col) stretch"""
    r = np.random.default_rng(seed)
    H, W = shape
    fy = np.fft.fftfreq(H)[:, None] * aniso[0]; fx = np.fft.fftfreq(W)[None, :] * aniso[1]
    f = np.hypot(fy, fx)
    amp = np.exp(-(f * scale_px) ** 2 * 2.0)
    amp[0, 0] = 0
    n = np.real(np.fft.ifft2(np.fft.fft2(r.standard_normal(shape)) * amp))
    return (n - n.mean()) / (n.std() + 1e-9)


def _lines_wrapped(draw, pts, W, H, fill, width):
    for ox in (-W, 0, W):
        for oy in (-H, 0, H):
            draw.line([(x + ox, y + oy) for x, y in pts], fill=fill, width=width)


def _walk(r, x, y, ang, n, step, wander):
    pts = [(x, y)]
    for _ in range(n):
        ang += r.normal(0, wander)
        x += step * math.cos(ang); y += step * math.sin(ang)
        pts.append((x, y))
    return pts


def _lab_shift(img, dL):
    a = np.asarray(img, float)
    return a + dL * 2.55


# ------------------------------------------------------------------ asphalt atlas
def asphalt_atlas(base_rgb, base_nrm, base_rough, size=2048, seed=11):
    """base_*: graded CC0 Asphalt026C tile (PIL, any size). Returns (diffuse RGB, normal RGB, maps RGB) PIL images
    covering ATLAS_M x ATLAS_M, seamless both ways."""
    r = np.random.default_rng(seed)
    rep = 4                                                  # 3 m per base tile
    t = size // rep
    def tiled(img, flip):
        a = np.asarray(img.resize((t, t), Image.LANCZOS), float)
        A = np.tile(a, (rep, rep, 1))
        B = np.roll(np.tile(a[::-1, ::-1] if flip else a, (rep, rep, 1)), (t * 3 // 8, t * 5 // 8), (0, 1))
        return A, B
    A, B = tiled(base_rgb, True)
    m = (pnoise((size, size), 160, seed) > 0)[..., None]   # low-frequency mix mask breaks the 3 m repeat
    m = np.asarray(Image.fromarray((m[..., 0] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(24)), float)[..., None] / 255
    col = A * (1 - m) + B * m
    lum0 = col.mean(-1, keepdims=True)
    # lateral coordinate in metres from the centre line for every column; x = column
    d = (np.arange(size) / size - 0.5) * ATLAS_M           # u=0.5 -> centre line
    D = np.abs(d)[None, :]
    # large-scale tonal variation (aged/oxidised areas), very gentle
    col *= (1 + 0.035 * pnoise((size, size), 220, seed + 1))[..., None]
    # wheel-track polish: lighter, smoother, slightly streaky along the road
    streak = pnoise((size, size), 60, seed + 2, aniso=(0.15, 1.0))
    wt = np.exp(-((D - 0.75) / 0.28) ** 2) + np.exp(-((D - 2.35) / 0.30) ** 2)
    wt = wt * (0.75 + 0.25 * streak)
    col = col * (1 - 0.05 * wt[..., None]) + 0.07 * wt[..., None] * (lum0 + 25)
    # oil drip / tyre-rubber band in the lane centres: darker, spotty
    ob = np.exp(-((D - 1.55) / 0.30) ** 2)
    spots = np.clip(pnoise((size, size), 5, seed + 3) - 0.6, 0, None) + 0.4 * np.clip(pnoise((size, size), 24, seed + 4, (0.3, 1)) , 0, None)
    col *= (1 - 0.10 * (ob * (0.35 + spots))[..., None]).clip(0.7, 1)
    # faint grime band either side of the double yellow (traffic doesn't sweep the crown)
    gb = np.exp(-((D - 0.0) / 0.45) ** 2) * (0.8 + 0.2 * pnoise((size, size), 40, seed + 5))
    col *= (1 - 0.07 * gb[..., None])
    # fine cracks (thin, dark): longitudinal near the centre joint and wheel paths, plus transverse
    cr = Image.new("L", (size, size), 0); dr = ImageDraw.Draw(cr)
    px = size / ATLAS_M
    for dc in (-0.28, 0.28, -0.9, 0.9, -2.2, 2.3, -3.1, 3.0):
        for _ in range(2):
            x = (0.5 + (dc + r.normal(0, 0.08)) / ATLAS_M) * size; y = r.uniform(0, size)
            pts = _walk(r, x, y, math.pi / 2 + r.normal(0, 0.05), int(r.uniform(20, 70)), px * 0.08, 0.12)
            _lines_wrapped(dr, pts, size, size, int(r.uniform(90, 170)), 1 if r.random() < 0.7 else 2)
    for _ in range(9):                                       # transverse
        y = r.uniform(0, size); x = r.uniform(0, size)
        pts = _walk(r, x, y, r.choice([0, math.pi]) + r.normal(0, 0.1), int(r.uniform(15, 45)), px * 0.08, 0.18)
        _lines_wrapped(dr, pts, size, size, int(r.uniform(80, 150)), 1)
    crack = np.asarray(cr.filter(ImageFilter.GaussianBlur(0.6)), float)[..., None] / 255
    col = col * (1 - 0.55 * crack)
    # dev6: baked crack SEAL (was floating decal ribbons): broken, meandering 3-6 cm tar bands at the centre joint, wheel
    # paths and lane edges, plus a few transverse; dark, low-gloss (see maps below). Sparse: the real road is mostly plain.
    sl = Image.new("L", (size, size), 0); ds = ImageDraw.Draw(sl)
    for dc in (-0.33, 0.33, -0.95, 1.0, -2.35, 2.3, -3.2, 3.15):
        for _ in range(1 if abs(dc) > 3 else 2):
            if r.random() < 0.35:
                continue
            x = (0.5 + (dc + r.normal(0, 0.06)) / ATLAS_M) * size; y = r.uniform(0, size)
            pts = _walk(r, x, y, math.pi / 2 + r.normal(0, 0.02), int(r.uniform(25, 80)), px * 0.08, 0.025)
            _lines_wrapped(ds, pts, size, size, int(r.uniform(140, 200)), int(round(px * r.uniform(0.025, 0.04))))
    for _ in range(3):
        y = r.uniform(0, size); x = (0.5 + r.uniform(-3, 3) / ATLAS_M) * size
        pts = _walk(r, x, y, r.choice([0, math.pi]) + r.normal(0, 0.08), int(r.uniform(12, 30)), px * 0.08, 0.05)
        _lines_wrapped(ds, pts, size, size, int(r.uniform(130, 190)), int(round(px * 0.03)))
    seal = 0.8 * np.asarray(sl.filter(ImageFilter.GaussianBlur(1.0)), float)[..., None] / 255
    seal = seal * np.clip(0.85 + 0.3 * pnoise((size, size), 30, seed + 9), 0.5, 1.0)[..., None]
    col = col * (1 - seal) + seal * np.array([34.0, 34.0, 36.0])
    crack = np.maximum(crack, 0.6 * seal)
    # aggregate speckle (fine light/dark stones)
    sp = r.standard_normal((size, size))
    col += (np.where(sp > 2.6, 14, 0) - np.where(sp < -2.8, 12, 0))[..., None]
    diff = Image.fromarray(col.clip(0, 255).astype(np.uint8))
    # normal: blended base normals, flattened in the polished wheel tracks, crack grooves
    NA, NB = tiled(base_nrm, False)
    n = (NA * (1 - m) + NB * m) / 255 * 2 - 1
    flat = (1 - 0.5 * wt)[..., None]
    n[..., :2] *= flat
    gy, gx = np.gradient(crack[..., 0])
    n[..., 0] -= gx * 2.0; n[..., 1] += gy * 2.0
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nrm = Image.fromarray(((n + 1) / 2 * 255).clip(0, 255).astype(np.uint8))
    # maps (R spec, G gloss, B detail): base roughness; polished wheel tracks shinier, cracks matte
    RA, RB = tiled(base_rough.convert("RGB"), False)
    rough = ((RA * (1 - m) + RB * m)[..., 0] / 255)
    gloss = (1 - rough) * (1 + 0.6 * wt) * (1 - 0.6 * crack[..., 0])
    spec = 0.3 * (1 + 0.5 * wt)
    maps = np.stack([spec, gloss, np.zeros_like(gloss)], -1)
    maps = Image.fromarray((maps * 255).clip(0, 255).astype(np.uint8)).resize((1024, 1024), Image.LANCZOS)
    return diff, nrm, maps


# ------------------------------------------------------------------ markings
def line_tex(kind, rgb, asphalt_rgb, seed, W=128, H=2048, fresh=False):
    """RGBA, alpha-tested. u across the ribbon, v along (LINE_PERIOD_M). kind 'single' (EDGE_RIBBON) or 'double'
    (DY_RIBBON: line / gap / line). Worn paint: breakup holes concentrated in the wheel-crossed zones,
    feathered soft edges (paint colour blends to asphalt over the outer 1-2 cm), faint dirt on the paint."""
    r = np.random.default_rng(seed)
    width = DY_RIBBON if kind == "double" else EDGE_RIBBON
    x = (np.arange(W) + 0.5) / W * width - width / 2     # metres across ribbon
    if kind == "double":
        cl = [-(DY_GAP + LINE_W) / 2, (DY_GAP + LINE_W) / 2]
    else:
        cl = [0.0]
    edge = np.min([np.abs(x - c) for c in cl], 0) - LINE_W / 2   # <0 inside a line
    ragged = 0.004 * pnoise((H, W), 6, seed, aniso=(1, 0.3))     # +-4 mm ragged edge
    e = edge[None, :] + ragged
    cov = np.clip(0.5 - e / 0.008, 0, 1)                          # soft 8 mm feather
    wear = pnoise((H, W), 10, seed + 1) + 0.6 * pnoise((H, W), 40, seed + 2, aniso=(0.4, 1))
    holes = wear > (9.0 if fresh else 2.0)                        # ~2-3 % paint lost (fresh: none)
    alpha = (cov > 0.5) & ~holes
    # colour: paint, dirtied, edge pixels mixed toward asphalt
    tint = (1 - (0.02 if fresh else 0.06) * np.clip(pnoise((H, W), 18, seed + 3), -2, 2) - (0.0 if fresh else 0.04))[..., None]
    pr = np.array(rgb, float)[None, None] * tint
    a_rgb = np.array(asphalt_rgb, float)[None, None]
    mix = np.clip(cov, 0, 1)[..., None] ** 0.7
    thin = (wear > 1.3)[..., None] * (0.0 if fresh else 0.30)      # worn-thin paint shows asphalt through
    c = pr * mix * (1 - thin) + a_rgb * (1 - mix * (1 - thin))
    c += r.normal(0, 4, c.shape)
    out = np.dstack([c.clip(0, 255), alpha * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def rpm_tex(rgb):
    a = np.zeros((32, 32, 3), np.uint8); a[:] = rgb
    a[12:20, 4:28] = (235, 205, 90)   # retroreflective lens
    return Image.fromarray(a)


# ------------------------------------------------------------------ decal atlas
def _sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def decal_atlas(patch_rgb_fn=None, size=2048, seed=21):
    """RGBA atlas, straight (non-premultiplied) alpha for an alpha-BLENDED material: every texel carries a plausible RGB,
    alpha carries the shape with a soft texture-space ramp at all edges (see CELLS / PATCH_CELLS).
    - seal: thin black tar band (FWHM ~ 0.65 x ribbon width), smooth, mottled, soft edges, lighter oxidised fringe
    - hair: narrower, greyer, partial-opacity hairline crack
    - patch: rounded rectangle (wobbly SDF), feathered edge, grain from the road asphalt, subtle darker lip; 3 tones
    - network: alligator crazing (Voronoi cell edges, broken, patchy region) alpha only on the cracks."""
    r = np.random.default_rng(seed)
    S = size
    img = np.zeros((S, S, 4), np.float64); img[..., :3] = (60, 60, 62)

    def put(cell_px, rgb, a):
        x0, y0, w, h = cell_px
        img[y0:y0 + h, x0:x0 + w, :3] = rgb; img[y0:y0 + h, x0:x0 + w, 3] = a * 255

    # --- crack-seal ribbons / hairlines (tar; cell = ribbon cross-section x 0.5 m of crack, same centre line in every variant)
    for kind, x_off, core, amax, tar, edge_tar in (("seal", 0, 0.34, 0.93, (30, 30, 32), (74, 74, 76)),
                                                  ("hair", 1024, 0.18, 0.62, (46, 46, 48), (80, 80, 82))):
        for k in range(8):
            w, h = 128, 256
            xs = (np.arange(w) + 0.5) / w; vs = (np.arange(h) + 0.5) / h
            rr = np.abs(xs - 0.5)[None, :] * 2
            env = (np.sin(np.pi * vs) ** 2)[:, None]                       # variants only differ away from the cell ends
            wob = 1 + 0.16 * np.clip(pnoise((h, w), 34, seed + 10 * k + x_off, aniso=(1.0, 0.02)), -2.2, 2.2) * env
            re_ = rr / np.clip(wob, 0.72, 1.35)
            a = 1 - _sstep(core, 0.97, re_)
            if kind == "seal":
                mott = np.clip(0.9 + 0.07 * pnoise((h, w), 9, seed + 77 + k, aniso=(1.0, 0.1)), 0.7, 1.0)
            else:
                mott = np.clip(0.8 + 0.18 * pnoise((h, w), 12, seed + 88 + k, aniso=(1.0, 0.1)), 0.4, 1.0)
            a = a * mott * amax
            fr = _sstep(0.2, 0.95, re_)[..., None]
            tone = np.array(tar, float) * (1 + 0.06 * pnoise((h, w), 8, seed + 33 + k))[..., None]
            rgb = tone * (1 - fr) + np.array(edge_tar, float) * fr
            put((x_off + k * 128, 0, w, h), rgb, a)

    # --- patches
    grain = None
    if patch_rgb_fn is not None:
        try:
            g = np.asarray(patch_rgb_fn(0, 1024), float).mean(-1)
            grain = np.clip(g / g.mean(), 0.7, 1.35)
        except Exception:
            grain = None
    TONES = [((58, 58, 60), 0.9, False), ((86, 86, 87), 0.85, True), ((114, 114, 116), 0.8, True)]   # rgb, grain gain, tar seam
    for t, (trgb, gain, seam_on) in enumerate(TONES):
        y0 = 256 + t * 416
        for kind, (cx, cy, w, h) in (("lane", (0, y0, 1024, 416)), ("strip", (1024, y0, 1024, 192)),
                                     ("small0", (1024, y0 + 192, 512, 224)), ("small1", (1536, y0 + 192, 512, 224))):
            sd = seed + 300 + 17 * t + len(kind) * 3 + (7 if kind.endswith("1") else 0)
            yy, xx = np.mgrid[0:h, 0:w].astype(float); px, py = xx - w / 2 + 0.5, yy - h / 2 + 0.5
            m = 0.07 * min(w, h) + 22
            Rr = {"lane": 0.12, "strip": 0.28, "small0": 0.2, "small1": 0.2}[kind] * min(w, h) * r.uniform(0.8, 1.25)
            qx, qy = np.abs(px) - (w / 2 - m - Rr), np.abs(py) - (h / 2 - m - Rr)
            sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - Rr
            sc = min(w, h) / 416
            sdf = sdf + sc * (1.6 * np.clip(pnoise((h, w), 38, sd), -2.2, 2.2) + 5 * np.clip(pnoise((h, w), 110, sd + 1), -2.2, 2.2)
                              + 0.8 * pnoise((h, w), 4, sd + 2))
            F = 9.0 * sc + 2.5
            d_in = -sdf
            ae = _sstep(0, F, d_in)
            a = ae * (0.97 - 0.03 * np.clip(pnoise((h, w), 30, sd + 3), -1.5, 1.5))
            if grain is not None:
                gtile = np.roll(grain, 311 * t + 57 * len(kind), (0, 1))
                gtile = np.tile(gtile, (1, 2))[:h, :w]
            else:
                gtile = np.ones((h, w))
            col = trgb[0] * np.ones((h, w))
            col = col * (1 + gain * 0.55 * (gtile - 1)) * (1 + 0.045 * pnoise((h, w), 42, sd + 4) + 0.025 * pnoise((h, w), 70, sd + 5, aniso=(0.2, 1.0)))
            col = col + 4.0 * r.standard_normal((h, w)) * (0.6 if t == 0 else 1.0)
            ring = np.exp(-((d_in - 0.95 * F) / (0.5 * F)) ** 2)
            col = col * (1 - (0.10 if t == 0 else 0.17) * ring)                      # slightly darker lip just inside the edge
            rgb = np.stack([col, col, col * 1.01], -1)
            if seam_on:                                                                 # thin tar seal along the joint (broken)
                sm = np.exp(-((d_in - 0.35 * F) / (0.3 * F)) ** 2) * np.clip(0.9 + 0.8 * pnoise((h, w), 26, sd + 6), 0, 1) * 0.75
                rgb = rgb * (1 - sm[..., None]) + np.array((36, 36, 38.0)) * sm[..., None]
            bd = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
            a = a * _sstep(1, 6, bd)
            put((cx, cy, w, h), rgb.clip(0, 255), a)

    # --- alligator crazing (512 px ~ 2.6 m): jittered Voronoi, line = |d2 - d1| small; broken, patchy, soft region mask
    from scipy.spatial import cKDTree
    for k in range(4):
        w = h = 512
        cs = 21
        gx, gy = np.meshgrid(np.arange(-1, w // cs + 2), np.arange(-1, h // cs + 2))
        Pj = np.stack([(gx + r.uniform(0.1, 0.9, gx.shape)) * cs, (gy + r.uniform(0.1, 0.9, gy.shape)) * cs], -1).reshape(-1, 2)
        yy, xx = np.mgrid[0:h, 0:w]
        dd, _ = cKDTree(Pj).query(np.stack([xx.ravel(), yy.ravel()], -1), k=2)
        e = ((dd[:, 1] - dd[:, 0]) * 0.5).reshape(h, w)
        sd = seed + 500 + k
        line = np.clip(1 - e / 1.5, 0, 1) ** 1.3
        line = line * (pnoise((h, w), 16, sd) > -0.35)                                   # broken lines
        region = _sstep(-0.15, 0.55, pnoise((h, w), 90, sd + 1))
        bd = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
        region = region * _sstep(8, 70, bd)
        a = (0.62 * line + 0.07) * region * np.clip(0.75 + 0.3 * pnoise((h, w), 8, sd + 2), 0.3, 1)
        a = a * _sstep(1, 6, bd)
        rgb = np.dstack([np.full((h, w), v) for v in (34.0, 34.0, 36.0)])
        put((k * 512, 1536, w, h), rgb, a)
    return Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGBA")


# ------------------------------------------------------------------ decal placement (used by 21_build_geometry.py)
def load_surface_notes(root):
    """survey `surface` entries -> [dict(kind, s0, s1, side, text)] (rpm_present / fresh_overlay excluded)"""
    import glob
    out = []
    for f in sorted(glob.glob(str(root / "data/derived/roadside_assets_*.json"))):
        try:
            for a in json.load(open(f)):
                if a.get("type") != "surface":
                    continue
                at = a.get("attrs", {})
                kind = at.get("condition") or at.get("kind") or at.get("state") or ""
                if kind in ("rpm_present", "fresh_overlay", ""):
                    continue
                text = " ".join(str(v) for k, v in at.items() if k not in ("condition", "kind", "state", "present", "rpm", "rpm_present", "sealing", "puddle"))
                out.append(dict(kind=kind, s0=float(a["s0"]), s1=float(a["s1"]), side=a.get("side", "center"), text=text.lower()))
        except Exception:
            pass
    return out


# observed alligator crazing inside longer survey spans (note: "alligator cracking left shoulder 10250-10260, 10200")
ALLIGATOR_EXTRA = [dict(s0=10196.0, s1=10266.0, side=1, edge=True)]


class _Decals:
    def __init__(self, road, surf_pt, grid_tris, chunk, nchunks, spans, surv, notes, seed=1234):
        self.road, self.surf_pt, self.grid_tris, self.CH, self.nch = road, surf_pt, grid_tris, chunk, nchunks
        self.rng = np.random.default_rng(seed)
        self.N = len(road.s); self.S = float(road.s[-1])
        ss, scr, spa = surv
        sg = np.arange(self.N, dtype=float)
        cr = 0.10 + 0.30 * np.clip(np.interp(sg, ss, scr), 0, 1)       # background (survey frames, evenly spaced heuristic)
        pa = 0.04 + 0.10 * np.clip(np.interp(sg, ss, spa), 0, 1)
        fresh = np.array(road.fresh, bool).copy()
        for sp in spans:
            m = (sg >= sp["s0"]) & (sg <= sp["s1"])
            k = sp["kind"]
            if k == "cracked_old":
                cr[m] = np.maximum(cr[m], 0.85); pa[m] = np.maximum(pa[m], 0.12)
            elif k == "crack_sealed":
                cr[m] = np.maximum(cr[m], 0.55)
            elif k == "patched":
                pa[m] = np.maximum(pa[m], 0.35); cr[m] = np.maximum(cr[m], 0.35)
            elif k == "fresh_overlay":
                fresh |= m
        fr = np.convolve(fresh.astype(float), np.ones(3), "same") > 0          # 1 m margin around fresh pavement
        cr[fr] = 0; pa[fr] = 0
        self.cr, self.pa, self.fresh = cr, pa, fr
        self.notes = notes
        self.patches = []      # dict(s0,s1,d0,d1,tone)
        self.layers = {k: [] for k in ("patch", "network", "hair", "seal")}   # lists of (s_mid, P, UV, T)
        self.stats = dict(patch=0, network=0, hair=0, seal=0)

    # -- helpers
    def e(self, s):
        i = int(np.clip(s, 0, self.N - 1)); return float(self.road.el[i]), float(self.road.er[i])

    def fr_at(self, s):
        return bool(self.fresh[int(np.clip(s, 0, self.N - 1))])

    def is_fresh(self, s0, s1):
        return bool(self.fresh[int(np.clip(s0, 0, self.N - 1)):int(np.clip(s1, 0, self.N - 1)) + 1].any())

    def in_patch(self, s, d, m=0.2):
        return any(p["s0"] - m <= s <= p["s1"] + m and p["d0"] - m <= d <= p["d1"] + m for p in self.patches)

    def meander(self, n, step, amp, wl=(3.0, 11.0)):
        r = self.rng
        x = np.arange(n) * step; out = np.zeros(n)
        for _ in range(3):
            out += r.uniform(0.4, 1.0) * np.sin(2 * math.pi * x / r.uniform(*wl) + r.uniform(0, 6.3))
        return out / 3 * amp * 2

    def snoise(self, n, step, wl):
        x = np.arange(n) * step
        return 0.5 * (np.sin(2 * math.pi * x / wl + self.rng.uniform(0, 6.3)) + np.sin(2 * math.pi * x / (wl * 0.43) + self.rng.uniform(0, 6.3)))

    # -- geometry primitives
    def quad(self, layer, s0, s1, d0, d1, cell, lift, rot=False, flip=(False, False), shear=0.0):
        nr = max(2, int(math.ceil((s1 - s0) / 1.0)) + 1)
        dl = [d0, d1] if not (d0 < 0 < d1) else [d0, 0.0, d1]
        dl = np.unique(np.concatenate([np.linspace(dl[k], dl[k + 1], max(2, int(math.ceil((dl[k + 1] - dl[k]) / 0.6)) + 1)) for k in range(len(dl) - 1)]))
        S_ = np.linspace(s0, s1, nr); Dg, Sg = np.meshgrid(dl, S_)
        fu = ((Dg - d0) / (d1 - d0)).ravel(); fv = ((Sg - s0) / (s1 - s0)).ravel()
        if shear:
            Dg = Dg + shear * (Sg - (s0 + s1) / 2)
        p = self.surf_pt(Sg.ravel(), Dg.ravel()); p[:, 2] += lift
        if rot:
            fu, fv = fv, fu
        if flip[0]:
            fu = 1 - fu
        if flip[1]:
            fv = 1 - fv
        u0, v0, u1, v1 = cell
        uv = np.stack([u0 + (u1 - u0) * fu, v0 + (v1 - v0) * fv], -1)
        self.layers[layer].append(((s0 + s1) / 2, p, uv, self.grid_tris(nr, len(dl), 0)))
        self.stats[layer] += 1

    def ribbon(self, layer, sd, hw, lift, cells, step_skip=None):
        """sd (n,2) polyline in (station m, offset m); hw (n,) half widths (m); one cell variant per 0.4 m segment"""
        n = len(sd)
        if n < 2:
            return
        t = np.gradient(sd, axis=0); t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
        nv = np.stack([-t[:, 1], t[:, 0]], -1)
        hw = np.maximum(hw, 0.0015)[:, None]
        Lp, Rp = sd + nv * hw, sd - nv * hw
        k = np.arange(n - 1)
        a, b, c, d = Rp[k], Lp[k], Rp[k + 1], Lp[k + 1]                   # (R_i, L_i, R_i+1, L_i+1)
        verts = np.stack([a, b, c, d], 1).reshape(-1, 2)                    # 4 per segment
        keep = np.ones(len(k), bool)
        if step_skip is not None:
            keep = step_skip[k] & step_skip[k + 1]
        ci = self.rng.integers(len(cells), size=len(k))
        cu = np.array([cells[j] for j in ci])                                # (n-1, 4)
        uv = np.stack([np.stack([cu[:, 0], cu[:, 1]], -1), np.stack([cu[:, 2], cu[:, 1]], -1),
                       np.stack([cu[:, 0], cu[:, 3]], -1), np.stack([cu[:, 2], cu[:, 3]], -1)], 1).reshape(-1, 2)
        base = np.arange(len(k)) * 4
        # winding: CCW seen from +z in the (x = -d, y = s) plane
        ab = np.stack([-(b[:, 1] - a[:, 1]), b[:, 0] - a[:, 0]], -1); ac = np.stack([-(c[:, 1] - a[:, 1]), c[:, 0] - a[:, 0]], -1)
        ccw = (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]) > 0
        T1 = np.where(ccw[:, None], np.stack([base, base + 1, base + 2], -1), np.stack([base, base + 2, base + 1], -1))
        T2 = np.where(ccw[:, None], np.stack([base + 1, base + 3, base + 2], -1), np.stack([base + 1, base + 2, base + 3], -1))
        T = np.concatenate([T1[keep], T2[keep]])
        if len(T) == 0:
            return
        p = self.surf_pt(verts[:, 0], verts[:, 1]); p[:, 2] += lift
        self.layers[layer].append((float(sd[:, 0].mean()), p, uv, T))
        self.stats[layer] += 1

    # -- cracks
    def crack(self, s0, s1, d_fn, W, p_hair=0.25, taper=1.8, branch=0.3, along=True, hair=None):
        """crack along s (d_fn(s) -> offset) or, with along=False, d_fn is an (n,2) polyline (s, d)"""
        r = self.rng
        step = 0.4
        if along:
            n = int((s1 - s0) / step) + 1
            if n < 4:
                return
            ss = s0 + np.arange(n) * step
            sd = np.stack([ss, d_fn(ss)], -1)
        else:
            sd = d_fn; n = len(sd)
        ds = np.arange(n) * step; Ltot = ds[-1]
        tp = np.clip(np.minimum(ds, Ltot - ds) / taper, 0, 1); tp = tp * tp * (3 - 2 * tp)
        hair = (r.random() < p_hair) if hair is None else hair
        w = W * (0.5 if hair else 1.0) * (1 + 0.25 * self.snoise(n, step, r.uniform(5, 14))) * tp
        ok = np.array([(not self.fr_at(s)) and (not self.in_patch(s, d)) for s, d in sd]) & (w > 0.004)
        lay = "hair" if hair else "seal"
        self.ribbon(lay, sd, w / 2, LIFT[lay], CELLS[lay], step_skip=ok)
        if along and branch and r.random() < branch and n > 12 and not hair:
            j = int(r.integers(4, n - 4)); ang = r.choice([-1, 1]) * r.uniform(0.35, 0.9)
            m = int(r.uniform(1.0, 3.5) / step); pts = [sd[j]]; th = ang
            for _ in range(m):
                th += r.normal(0, 0.1); pts.append(pts[-1] + step * np.array([math.cos(th), math.sin(th)]))
            pts = np.array(pts)
            tb = np.clip(np.minimum(np.arange(m + 1), m - np.arange(m + 1)) / 2.5, 0, 1)
            el, er = self.e(pts[0, 0])
            okb = np.array([(-er + 0.3 < d < el - 0.3) and abs(d) > 0.26 and not self.in_patch(s, d) and not self.fr_at(s) for s, d in pts])
            self.ribbon("hair", pts, W * 0.22 * tb + 0.002, LIFT["hair"], CELLS["hair"], step_skip=okb)

    def transverse(self, s, d_a, d_b, W, hair=None):
        r = self.rng
        step = 0.4
        n = max(4, int(abs(d_b - d_a) / step) + 1)
        dd = np.linspace(d_a, d_b, n)
        slope = r.uniform(-0.25, 0.25); bow = r.uniform(-0.15, 0.15)
        x = (dd - d_a) / (d_b - d_a + 1e-9)
        ss = s + slope * (dd - d_a) + bow * (4 * x * (1 - x)) + self.meander(n, step, 0.025, (1.5, 5))
        self.crack(0, 0, np.stack([ss, dd], -1), W, along=False, taper=0.9, branch=0, hair=hair)

    # -- patches
    def add_patch(self, s0, s1, d0, d1, tone):
        r = self.rng
        s0, s1 = max(s0, 4), min(s1, self.S - 4)
        if s1 - s0 < 0.25 or self.is_fresh(s0 - 1, s1 + 1):
            return
        el, er = self.e((s0 + s1) / 2)
        d0, d1 = max(d0, -er + 0.35), min(d1, el - 0.35)
        if d1 - d0 < 0.25:
            return
        if -0.2 < d0 < 0.22 < d1:
            d0 = 0.22                              # never over the double yellow
        a, b = s1 - s0, d1 - d0
        rot = a >= b
        long_, short = (a, b) if rot else (b, a)
        asp = long_ / max(short, 1e-3)
        kind = "strip" if asp > 3.6 else "lane" if (asp > 1.8 or long_ > 1.7) else "small"
        pc = PATCH_CELLS[tone]
        cell = pc["small"][int(r.integers(2))] if kind == "small" else pc[kind]
        self.patches.append(dict(s0=s0, s1=s1, d0=d0, d1=d1, tone=tone))
        self.quad("patch", s0, s1, d0, d1, cell, LIFT["patch"], rot=rot, flip=(bool(r.random() < 0.5), bool(r.random() < 0.5)),
                  shear=float(r.normal(0, 0.012)) if long_ > 3 else 0.0)

    def patch_from_text(self, s0, s1, text):
        r = self.rng
        L = s1 - s0
        if "l-shaped" in text:                      # GoPro 164: black L patch, near part narrow, far part wider
            self.add_patch(s0 + 15, s0 + 24, 0.25, 2.9, 0)
            self.add_patch(s0 + 6, s0 + 15, 0.25, 1.7, 0)
            return
        if re.search(r"black|dark|fresh", text):
            tone = 0
        elif re.search(r"lighter|grey|gray|repair|two-tone", text):
            tone = 2
        else:
            tone = int(r.choice([0, 1, 1, 2]))
        nrep = int(np.clip(round(L / 22), 1, 3))
        for q in range(nrep):
            sc = s0 + (q + r.uniform(0.25, 0.75)) * L / nrep
            el, er = self.e(sc)
            if re.search(r"full-width|transverse|seam|band", text):
                ln = r.uniform(0.35, 0.7) if re.search(r"seam|band", text) else r.uniform(1.2, 3.5)
                self.add_patch(sc - ln / 2, sc + ln / 2, -er + 0.3, el - 0.3, tone)
                continue
            lf, rt = bool(re.search(r"left", text)), bool(re.search(r"right", text))
            side = 1 if lf and not rt else -1 if rt and not lf else int(r.choice([-1, 1]))
            e_s = el if side > 0 else er
            lc = r.uniform(3, 9)
            if "edge" in text:
                d_a, d_b = e_s - r.uniform(0.9, 1.5), e_s - 0.3
                lc = r.uniform(5, 14)
            elif "centre-left" in text or "center-left" in text:
                d_a, d_b = 0.3, 0.3 + r.uniform(1.2, 1.9); side = 1
            elif "outline" in text or "manhole" in text:
                d_a, d_b = 0.4, r.uniform(1.8, 2.4); lc = r.uniform(3, 5)
            else:
                wd = r.uniform(0.6, 2.4); d_a = r.uniform(0.3, max(0.31, e_s - 0.35 - wd)); d_b = d_a + wd
            a, b = sorted((side * d_a, side * d_b))
            self.add_patch(sc - lc / 2, sc + lc / 2, a, b, tone)

    def background_patches(self):
        r = self.rng
        for side in (1, -1):
            s = 8 + r.uniform(0, 100)
            while s < self.S - 8:
                pa = self.pa[int(s)]
                if pa < 0.02:
                    s += r.uniform(20, 50); continue
                el, er = self.e(s); e_s = el if side > 0 else er
                u = r.random()
                if u < 0.5:                                   # wheel-path / edge strip
                    c = r.choice([0.95, 2.3, e_s - 0.8]); wd = r.uniform(0.4, 0.85); lc = r.uniform(3, 12)
                    a, b = sorted((side * (c - wd / 2), side * (c + wd / 2)))
                    self.add_patch(s, s + lc, a, b, int(r.choice([0, 1, 1, 2])))
                elif u < 0.8:                                 # small pothole fill
                    wd = r.uniform(0.5, 1.2); lc = r.uniform(0.6, 1.5); c = r.uniform(0.9, max(1.0, e_s - 0.8))
                    a, b = sorted((side * (c - wd / 2), side * (c + wd / 2)))
                    self.add_patch(s, s + lc, a, b, int(r.choice([0, 1])))
                else:                                         # trench / transverse repair across one lane
                    wl = r.uniform(0.45, 0.85); a, b = (0.3, e_s - 0.3) if side > 0 else (-e_s + 0.3, -0.3)
                    self.add_patch(s, s + wl, a, b, int(r.choice([0, 1, 2])))
                s += r.uniform(0.7, 1.4) * 25 / (pa + 0.01)

    # -- alligator crazing: only where the survey says severe
    def alligator(self, s0, s1, side, edge=False):
        r = self.rng
        T = 2.6
        for s in np.arange(s0, s1, T * 0.92):
            sd_ = side if side else int(r.choice([-1, 1]))
            el, er = self.e(s); e_s = el if sd_ > 0 else er
            a_, b_ = ((e_s - 2.0, e_s - 0.3) if edge else (0.55, min(3.1, e_s - 0.3)))
            if b_ - a_ < 0.6:
                continue
            if self.is_fresh(s - 1, s + T + 1) or self.in_patch(s + T / 2, sd_ * (a_ + b_) / 2, 0.0):
                continue
            ds_ = sorted((sd_ * a_, sd_ * b_))
            self.quad("network", s, s + T, ds_[0], ds_[1], CELLS["network"][int(r.integers(4))], LIFT["network"],
                      rot=bool(r.random() < 0.5), flip=(bool(r.random() < 0.5), bool(r.random() < 0.5)))

    # -- driver
    def run(self, legacy=False):
        """dev6: geometry = surveyed patches only (note text). legacy=True restores the dev5 random background patches,
        crazing and crack-seal ribbons (now baked into the asphalt atlas)."""
        r = self.rng
        seen = []
        for nt in sorted(self.notes, key=lambda x: x["s0"]):     # 1) patches: surveyed (note text) + sparse background
            if re.search(r"\bwet\b", nt["text"]):
                continue
            is_p = nt["kind"] == "patched" or "patch" in nt["text"]
            if is_p and (nt["s1"] - nt["s0"]) <= 140 and not any(abs(nt["s0"] - a) < 12 and abs(nt["s1"] - b) < 12 for a, b in seen):
                seen.append((nt["s0"], nt["s1"]))
                self.patch_from_text(nt["s0"], nt["s1"], nt["text"])
        if not legacy:
            return
        self.background_patches()
        for nt in self.notes:                                     # 2) alligator crazing
            if re.search(r"alligator|fatigue|pothole", nt["text"]) and (nt["s1"] - nt["s0"]) <= 140:
                lf, rt = bool(re.search(r"left", nt["text"])), bool(re.search(r"right", nt["text"]))
                self.alligator(nt["s0"], nt["s1"], 1 if lf and not rt else -1 if rt and not lf else 0,
                               edge=bool(re.search(r"edge|shoulder|apron", nt["text"])))
        for ex in ALLIGATOR_EXTRA:
            self.alligator(ex["s0"], ex["s1"], ex["side"], ex["edge"])
        for side in (1, -1):                                      # 3) longitudinal crack seal
            for dbase, wgt in ((0.31, 0.7), (0.95, 0.8), (2.3, 0.6), (None, 0.45)):
                s = 4 + r.uniform(0, 25)
                while s < self.S - 6:
                    c = self.cr[int(s)] * wgt
                    if c < 0.04:
                        s += r.uniform(10, 25); continue
                    if r.random() < min(0.75, c * 0.8):
                        L = r.uniform(7, 12 + 45 * c)
                        e1 = s
                        while e1 < min(s + L, self.S - 5) and self.cr[int(e1)] * wgt >= 0.03:
                            e1 += 2.0
                        el, er = self.e(s); e_s = el if side > 0 else er
                        d0 = float(np.clip((dbase if dbase is not None else e_s - 0.6) + r.normal(0, 0.06), 0.3, e_s - 0.3))
                        amp = r.uniform(0.012, 0.04) if dbase == 0.31 else r.uniform(0.02, 0.07)
                        n_ = max(int((e1 - s) / 0.4) + 2, 3)
                        mean = self.meander(n_, 0.4, amp) + np.cumsum(r.normal(0, 0.004, n_))
                        dfn = (lambda ss, d0=d0, mean=mean, side=side, s=s: side * np.clip(d0 + np.interp(ss - s, np.arange(len(mean)) * 0.4, mean), 0.28, 9))
                        self.crack(s, e1, dfn, r.uniform(0.06, 0.1), p_hair=0.15 + 0.35 * self.cr[int(s)] * (wgt > 0.5))
                        s = e1 + r.uniform(8, 50) / (0.3 + c)
                    else:
                        s += r.uniform(12, 35)
        s = 6 + r.uniform(0, 30)                                  # 4) transverse thermal cracks
        while s < self.S - 6:
            c = self.cr[int(s)]
            if c > 0.05 and r.random() < c * 0.55:
                el, er = self.e(s); side = int(r.choice([-1, 1]))
                if r.random() < 0.3:
                    a, b = -er + 0.35, el - 0.35
                else:
                    far = r.uniform(0.55, 1.0) * ((el if side > 0 else er) - 0.35)
                    a, b = (0.32, far) if side > 0 else (-far, -0.32)
                self.transverse(s, a, b, r.uniform(0.05, 0.09))
            s += r.uniform(14, 55) / (0.25 + c)
        for nt in self.notes:                                     # 5) cracks named in the survey notes
            tx = nt["text"]
            if nt["kind"] in ("cracked_old", "crack_sealed") and not re.search(r"\bwet\b", tx):
                if "transverse" in tx:
                    m = re.search(r"~?(\d{4,5})", tx)
                    sc = float(m.group(1)) if m else (nt["s0"] + nt["s1"]) / 2
                    el, er = self.e(sc)
                    if not self.is_fresh(sc - 1, sc + 1):
                        self.transverse(sc, -er + 0.35, el - 0.35, 0.07, hair=False)
                if "longitudinal" in tx and (nt["s1"] - nt["s0"]) <= 100:
                    side = 1 if "left" in tx else -1 if "right" in tx else int(r.choice([-1, 1]))
                    s0_, s1_ = nt["s0"] + 3, nt["s1"] - 3
                    mean = self.meander(int((s1_ - s0_) / 0.4) + 3, 0.4, 0.05)
                    dd0 = 0.4 if "centre" in tx else 0.95
                    self.crack(s0_, s1_, lambda ss, mean=mean, side=side, s0_=s0_, dd0=dd0: side * (dd0 + np.interp(ss - s0_, np.arange(len(mean)) * 0.4, mean)),
                               0.085, p_hair=0.0, branch=0.5)

    def chunks(self):
        order = ("patch", "network", "hair", "seal")
        per = {}
        for lay in order:
            for sm, P, UV, T in self.layers[lay]:
                c = min(int(sm // self.CH), self.nch - 1)
                per.setdefault(c, {k: [] for k in order})[lay].append((P, UV, T))
        out = {}
        for c, dct in per.items():
            Ps, UVs, Ts, k = [], [], [], 0
            for lay in order:
                for P, UV, T in dct[lay]:
                    Ps.append(P); UVs.append(UV); Ts.append(T + k); k += len(P)
            out[c] = (np.concatenate(Ps), np.concatenate(Ts), np.concatenate(UVs))
        return out


def build_decals(road, surf_pt, grid_tris, chunk, nchunks, spans, surv, notes, seed=1234):
    """-> ({chunk: (P, T, UV)}, stats). Patches (lift 1.6 cm) < alligator (1.8) < hairlines (1.9) < crack seal (2.0)."""
    d = _Decals(road, surf_pt, grid_tris, chunk, nchunks, spans, surv, notes, seed)
    d.run()
    return d.chunks(), dict(d.stats, patch_rects=len(d.patches))


def survey_condition(root, s_total):
    """per-frame asphalt condition from the GoPro surveys, frames assumed evenly spaced along the road
    (frame 1 = SR-84, last frame = northern terminus; the frames carry no GPS). Returns (s array, cracked, patched)."""
    fr = []
    for f in ("A", "B"):
        try:
            fr += json.load(open(root / f"data/derived/gopro_survey_{f}.json"))
        except Exception:
            pass
    if not fr:
        return np.array([0.0, s_total]), np.zeros(2), np.zeros(2)
    fr.sort(key=lambda r: r["frame"])
    s = np.linspace(0, s_total, len(fr))
    cr = np.array([("crack" in r["asphalt"]) * 1.0 for r in fr])
    pa = np.array([("patch" in r["asphalt"] or "patch" in r.get("notes", "")) * 1.0 for r in fr])
    return s, cr, pa
