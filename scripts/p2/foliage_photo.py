"""Photographic leaf-cluster cards from CC0 Poly Haven leaf sprite sheets (dev6, replaces the painted leaf polygons).

The SpeedTree / community-track recipe: a tree is many small (~1-1.8 m) alpha-tested cards, each a photo of a leafy
twig with real gaps, darker leaves underneath and lighter on top; vertex normals bent out from the crown centre
(scripts/p2/trees3d.py) so the crown shades as a volume. Here each card is composed from individual real leaves cut
out of the Poly Haven sheets (data/raw/textures/polyhaven, fetched by 20_fetch_textures.py), laid along a twig.

Atlas tree_leaves.png: 2048 x 2048, 4 columns x 5 rows (cell 512 x 409.6 px); row r covers v in [r/5, (r+1)/5]
(row 0 = bottom of the image): 0 oak, 1 bay, 2 euc, 3 syc (cols 0-1 leafy, cols 2-3 bare winter twigs), 4 conifer.
Columns 0..3 run old/dark/interior -> fresh/lime/outer (rows 0-2, 4).
"""
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
PH = ROOT / "data/raw/textures/polyhaven"
ROWS = 5
SRC = {"oak": ("searsia_lucida", "Diffuse", "Alpha"), "bay": ("island_tree_02", "leaves_diff", "leaves_alpha"),
       "euc": ("island_tree_02", "leaves_diff", "leaves_alpha"), "syc": ("island_tree_02", "leaves_diff", "leaves_alpha"),
       "con": ("island_tree_02", "leaves_diff", "leaves_alpha")}
# (tree_small_02 / fir_sapling sheets were checked and rejected: UV-island padding and baked grids, no clean sprites;
#  cypress sprays are built from narrowed, tiny bay-leaf cut-outs = scale-like foliage)
# per species: leaves per card, leaf length px (of a 512 cell ~ 1.4 m card), width squash, colour multiplier
# (sRGB gain to the GoPro spring crowns: live oak dark glossy olive, bay darker yellow-green, euc grey-blue-green,
# sycamore fresh lime, cypress dark blue-green)
LEAF = {"oak": dict(n=520, L=(22, 34), sq=1.0, tint=(0.95, 1.05, 0.78)),
        "bay": dict(n=300, L=(40, 60), sq=0.85, tint=(0.86, 0.98, 0.80)),
        "euc": dict(n=220, L=(55, 85), sq=0.40, tint=(0.92, 0.98, 0.98)),
        "syc": dict(n=200, L=(44, 64), sq=1.5, tint=(1.06, 1.18, 0.80)),
        "con": dict(n=1100, L=(12, 20), sq=0.35, tint=(0.70, 0.86, 0.80))}
COL_TINT = (np.array([0.70, 0.76, 0.72]), np.array([0.86, 0.92, 0.88]), np.array([1.0, 1.03, 0.96]), np.array([1.10, 1.18, 0.86]))
BARK = {"oak": (66, 60, 53), "bay": (104, 100, 92), "euc": (150, 138, 118), "syc": (170, 164, 150), "con": (80, 66, 54)}
_CACHE = {}


def sprites(kind, max_n=60):
    """individual leaf cut-outs (RGBA arrays, long axis vertical, stem at the bottom) from the Poly Haven sheet"""
    key = SRC[kind][0]
    if key in _CACHE:
        return _CACHE[key]
    aid, dn, an = SRC[kind]
    rgb = np.asarray(Image.open(PH / aid / f"{dn}.png").convert("RGB"))
    a = np.asarray(Image.open(PH / aid / f"{an}.png").convert("L"))
    if a.shape != rgb.shape[:2]:
        a = np.asarray(Image.fromarray(a).resize(rgb.shape[1::-1]))
    lab, n = ndimage.label(a > 128)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        m = lab[sl] == i + 1
        area = m.sum()
        fill = area / (h * w); c = rgb[sl][m].mean(0)
        # leaf-shaped (not a padded rectangle / UV island), green, not a pale highlight card
        if area < 1500 or not 0.3 < fill < 0.8 or max(h, w) > 4 * min(h, w) or c[1] < 0.95 * c[0] or c.mean() > 170:
            continue
        im = np.dstack([rgb[sl], np.where(m, a[sl], 0)])
        # align the principal axis vertically
        yy, xx = np.nonzero(m); cov = np.cov(np.vstack([xx, yy]))
        ev, evec = np.linalg.eigh(cov); vx, vy = evec[:, 1]
        ang = math.degrees(math.atan2(vx, vy))
        spr = Image.fromarray(im.astype(np.uint8), "RGBA").rotate(-ang, expand=True, resample=Image.BICUBIC)
        spr = spr.crop(spr.getbbox())
        out.append(spr)
    out.sort(key=lambda s: -s.size[0] * s.size[1])
    _CACHE[key] = out[:max_n]
    return _CACHE[key]


def _twig(d, r, x, y, ang, ln, wid, col, segs, out):
    for i in range(segs):
        ang += r.normal(0, 0.22)
        x1, y1 = x + math.sin(ang) * ln / segs, y - math.cos(ang) * ln / segs
        S_ = d.im.size[0]
        if math.hypot(x1 - S_ / 2, y1 - S_ / 2) > S_ * 0.46 and y1 < S_ * 0.9:   # stay inside the card
            return
        d.line([(x, y), (x1, y1)], fill=col, width=max(1, int(round(wid * (1 - 0.6 * i / segs)))))
        out.append((x1, y1, ang)); x, y = x1, y1


def cell(kind, seed, col, S=512, bare=False):
    """one leaf-cluster card (S x S RGBA): twig skeleton + real leaves along it, painter-ordered deep -> front with
    depth darkening (self-shadowing inside the cluster), ragged outline with sky gaps."""
    r = np.random.default_rng(9000 + seed)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bark = tuple(int(v * 0.85) for v in BARK[kind]) + (255,)
    pts = []
    # main twig from the bottom centre, 3-5 side twigs: leaves sit along them (never a disc of confetti)
    nmain = 2 if kind in ("oak", "syc") else 1
    for m in range(nmain if not bare else 3):
        _twig(d, r, S / 2 + r.normal(0, 25), S * 0.97, r.normal(0, 0.3), S * r.uniform(0.55, 0.7), 7 if bare else 5, bark, 7, pts)
    for _ in range(r.integers(5, 8) if not bare else 9):
        x, y, a = pts[r.integers(len(pts))]
        _twig(d, r, x, y, a + r.choice([-1, 1]) * r.uniform(0.5, 1.1), S * r.uniform(0.15, 0.3), 3 if not bare else 2.5, bark, 4, pts)
    if bare:  # winter sycamore: fine twigs and a few pale buds only
        for _ in range(40):
            x, y, a = pts[r.integers(len(pts))]
            _twig(d, r, x, y, a + r.normal(0, 0.9), S * r.uniform(0.05, 0.14), 1.5, bark, 2, [])
        for _ in range(14):
            x, y, _a = pts[r.integers(len(pts))]
            q = r.uniform(2, 4); d.ellipse([x - q, y - q, x + q, y + q], fill=(150, 170, 90, 255))
        return np.asarray(img)
    P = LEAF[kind]; spr = sprites(kind); tint = np.array(P["tint"]) * COL_TINT[col]
    n = P["n"]
    hole = ndimage.gaussian_filter(r.uniform(0, 1, (S // 8, S // 8)), 1.5)
    hole = np.asarray(Image.fromarray((hole * 255).astype(np.uint8)).resize((S, S), Image.BICUBIC)) > 150
    for i in range(n):
        x, y, a = pts[r.integers(len(pts))]
        x += r.normal(0, 22); y += r.normal(0, 22)
        if not (24 < x < S - 24 and 24 < y < S - 24) or hole[int(y), int(x)] or math.hypot(x - S / 2, y - S / 2) > S * 0.47:
            continue
        depth = i / n                                    # painter's order: first = deep inside the cluster
        L = r.uniform(*P["L"]) * (0.8 + 0.3 * depth)
        s = spr[r.integers(len(spr))]
        w = max(4, int(L * s.size[0] / s.size[1] * P["sq"])); h = max(4, int(L))
        lf = s.resize((w, h), Image.LANCZOS)
        # leaves splay out of the twig; euc and bay droop
        rot = math.degrees(a) + r.choice([-1, 1]) * r.uniform(25, 70) + (r.uniform(60, 150) * r.choice([-1, 1]) if kind in ("euc",) else 0)
        lf = lf.rotate(-rot + r.normal(0, 12), expand=True, resample=Image.BICUBIC)
        arr = np.asarray(lf).astype(float)
        shade = (0.55 + 0.5 * depth) * r.uniform(0.9, 1.1) * (1.08 if y < S * 0.45 else 1.0)
        hue = np.array([r.normal(1, 0.05), r.normal(1, 0.04), r.normal(1, 0.06)])
        arr[..., :3] = (arr[..., :3] * tint * hue * shade).clip(0, 255)
        lf = Image.fromarray(arr.astype(np.uint8), "RGBA")
        img.alpha_composite(lf, (int(x - lf.size[0] / 2), int(y - lf.size[1] / 2)))
    a = np.asarray(img).copy()
    a[..., 3] = np.where(a[..., 3] > 110, 255, 0)      # hard alpha for alpha test (no grey fringes in mips)
    return a


def leaf_atlas(N=2048):
    A = np.zeros((N, N, 4), np.uint8)
    ch = N / ROWS
    for ri, k in enumerate(("oak", "bay", "euc", "syc", "con")):
        y0 = int(round(N - (ri + 1) * ch)); y1 = int(round(N - ri * ch))
        for v in range(4):
            bare = (k == "syc" and v >= 2)
            c = cell(k, ri * 4 + v, v if k != "syc" else (2 if v == 0 else 3), bare=bare)
            c = np.asarray(Image.fromarray(c, "RGBA").resize((N // 4, y1 - y0), Image.LANCZOS)).copy()
            c[..., 3] = np.where(c[..., 3] > 127, 255, 0)
            # dilate colour into transparent texels so mips don't pull black fringes
            rgb = c[..., :3].astype(float); m = c[..., 3] > 0
            if m.any():
                idx = ndimage.distance_transform_edt(~m, return_distances=False, return_indices=True)
                rgb = rgb[idx[0], idx[1]]
            A[y0:y1, v * N // 4:(v + 1) * N // 4, :3] = rgb.astype(np.uint8)
            A[y0:y1, v * N // 4:(v + 1) * N // 4, 3] = c[..., 3]
    return Image.fromarray(A, "RGBA")


if __name__ == "__main__":
    import sys
    leaf_atlas().save(sys.argv[1] if len(sys.argv) > 1 else "/tmp/leaf_atlas.png")
