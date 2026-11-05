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
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ATLAS_M = 12.0
LINE_W, DY_GAP = 0.10, 0.10
DY_RIBBON = 2 * LINE_W + DY_GAP + 2 * 0.03          # 3 cm feather margin each side
EDGE_RIBBON = LINE_W + 2 * 0.03
LINE_PERIOD_M = 8.0                                   # v period of the line textures (wear pattern)
BROKEN = (3.05, 12.19)
RPM_SPACING = 24.0
# lifts above the road surface (>= 1.5 cm, layered so decals never z-fight each other)
LIFT = dict(patch=0.016, network=0.018, seal=0.020, line=0.023, rpm=0.024)

# decal atlas cells (u0, v0, u1, v1) in texture space (v measured from the top of the image)
CELLS = {"seal": [(k / 8, 0.0, (k + 1) / 8, 0.5) for k in range(8)],       # 0.6 m x 6 m longitudinal tar bands
         "patch": [(k / 4, 0.5, (k + 1) / 4, 0.75) for k in range(4)],     # newer-asphalt patches (3 grey, 1 black)
         "network": [(k / 4, 0.75, (k + 1) / 4, 1.0) for k in range(4)]}   # alligator / block cracking
SEAL_CELL_M = (0.6, 6.0)


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
def line_tex(kind, rgb, asphalt_rgb, seed, W=128, H=2048):
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
    holes = wear > 2.0                                            # ~2-3 % paint lost
    alpha = (cov > 0.5) & ~holes
    # colour: paint, dirtied, edge pixels mixed toward asphalt
    tint = (1 - 0.06 * np.clip(pnoise((H, W), 18, seed + 3), -2, 2) - 0.04)[..., None]
    pr = np.array(rgb, float)[None, None] * tint
    a_rgb = np.array(asphalt_rgb, float)[None, None]
    mix = np.clip(cov, 0, 1)[..., None] ** 0.7
    thin = (wear > 1.3)[..., None] * 0.30                          # worn-thin paint shows asphalt through
    c = pr * mix * (1 - thin) + a_rgb * (1 - mix * (1 - thin))
    c += r.normal(0, 4, c.shape)
    out = np.dstack([c.clip(0, 255), alpha * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def rpm_tex(rgb):
    a = np.zeros((32, 32, 3), np.uint8); a[:] = rgb
    a[12:20, 4:28] = (235, 205, 90)   # retroreflective lens
    return Image.fromarray(a)


# ------------------------------------------------------------------ decal atlas
def decal_atlas(patch_rgb_fn, tar_rgb=(40, 40, 42), size=2048, seed=21):
    """RGBA alpha-tested atlas: crack-seal bands, patches, crack networks (see CELLS)."""
    r = np.random.default_rng(seed)
    img = np.zeros((size, size, 4), np.float64)
    # --- seal strips: meandering tar band, 4-8 cm wide, smeared edges
    for k, (u0, v0, u1, v1) in enumerate(CELLS["seal"]):
        cw, ch = int((u1 - u0) * size), int((v1 - v0) * size)
        pxm_u = cw / SEAL_CELL_M[0]; pxm_v = ch / SEAL_CELL_M[1]
        m = Image.new("L", (cw, ch), 0); dr = ImageDraw.Draw(m)
        x = cw / 2; pts = []
        for y in np.linspace(-10, ch + 10, 120):
            x += r.normal(0, 0.012 * pxm_u); x = np.clip(x, cw * 0.3, cw * 0.7); pts.append((x, y))
        w = int(r.uniform(0.04, 0.08) * pxm_u)
        dr.line(pts, fill=255, width=w)
        if r.random() < 0.6:                                   # a branch
            j = int(r.uniform(20, 90)); bx, by = pts[j]
            br = _walk(r, bx, by, math.pi / 2 + r.choice([-0.5, 0.5]), 25, 0.08 * pxm_v, 0.15)
            k_ = next((q for q, (a, b) in enumerate(br) if not (12 < a < cw - 12 and 12 < b < ch - 12)), len(br))
            br = br[:k_]
            if len(br) > 1:
                dr.line(br, fill=255, width=max(4, w // 2))
        mm = np.asarray(m.filter(ImageFilter.GaussianBlur(2.5)), float) / 255
        mm = mm + 0.25 * pnoise((ch, cw), 4, seed + k) * (mm > 0.15)
        # strip ends taper so chained strips don't show seams
        yy = np.linspace(0, 1, ch)[:, None]
        mm *= np.clip(np.minimum(yy, 1 - yy) / 0.04, 0, 1)
        tone = np.array(tar_rgb, float) * (1 + 0.08 * pnoise((ch, cw), 8, seed + 50 + k))[..., None]
        cell = np.dstack([tone, (mm > 0.5) * 255.0])
        img[int(v0 * size):int(v0 * size) + ch, int(u0 * size):int(u0 * size) + cw] = cell
    # --- patches: newer asphalt, irregular rectangle with tar-sealed outline
    for k, (u0, v0, u1, v1) in enumerate(CELLS["patch"]):
        cw, ch = int((u1 - u0) * size), int((v1 - v0) * size)
        base = np.asarray(patch_rgb_fn(k, cw), float)[:ch, :cw]
        yy, xx = np.mgrid[0:ch, 0:cw] / np.array([ch, cw])[:, None, None]
        edge = np.minimum(np.minimum(xx, 1 - xx), np.minimum(yy, 1 - yy))
        e = edge + 0.012 * pnoise((ch, cw), 10, seed + 100 + k)
        alpha = e > 0.03
        outline = (e > 0.03) & (e < 0.05)
        c = base.copy()
        c[outline] = c[outline] * 0.45
        img[int(v0 * size):int(v0 * size) + ch, int(u0 * size):int(u0 * size) + cw] = np.dstack([c, alpha * 255.0])
    # --- crack networks (alligator / block): Voronoi-ish polygon edges, dark, alpha only on cracks
    for k, (u0, v0, u1, v1) in enumerate(CELLS["network"]):
        cw, ch = int((u1 - u0) * size), int((v1 - v0) * size)
        m = Image.new("L", (cw, ch), 0); dr = ImageDraw.Draw(m)
        n = int(r.uniform(40, 90))
        P = r.uniform(0.08, 0.92, (n, 2)) * [cw, ch]
        from scipy.spatial import Delaunay
        tri = Delaunay(P)
        for s in tri.simplices:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                if r.random() < 0.55:
                    pa, pb = P[s[a]], P[s[b]]
                    mid = (pa + pb) / 2 + r.normal(0, 4, 2)
                    dr.line([tuple(pa), tuple(mid), tuple(pb)], fill=255, width=int(r.choice([2, 2, 3])))
        mm = np.asarray(m.filter(ImageFilter.GaussianBlur(0.8)), float) / 255
        yy, xx = np.mgrid[0:ch, 0:cw] / np.array([ch, cw])[:, None, None]
        fade = np.clip((0.5 - np.hypot(xx - 0.5, yy - 0.5)) / 0.12, 0, 1)
        alpha = (mm * fade + 0.15 * pnoise((ch, cw), 6, seed + 200 + k)) > 0.45
        tone = np.array([48, 48, 48.0]) * np.ones((ch, cw, 1))
        img[int(v0 * size):int(v0 * size) + ch, int(u0 * size):int(u0 * size) + cw] = np.dstack([tone, alpha * 255.0])
    return Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGBA")


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
