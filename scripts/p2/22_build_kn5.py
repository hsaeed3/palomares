"""Phase 2: textures + materials + kn5 files + AC track folder -> build/palomares_road/"""
import io, json, math, pickle, shutil, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from geom import ROOT, DER, BUILD, CFG
from geom import build_lock; build_lock()  # one build/ writer at a time
from kn5 import Material, Mesh, Dummy, write_kn5, ac_from_local

VERSION = (ROOT / "VERSION").read_text().strip()
geo = pickle.load(open(BUILD / "geometry.pkl", "rb"))
G, meta = geo["groups"], geo["meta"]
TRACK = BUILD / "palomares_road"
if TRACK.exists():
    shutil.rmtree(TRACK)
TRACK.mkdir(parents=True)
TEX = ROOT / "data/raw/textures"
rng = np.random.default_rng(3)


def png(img):
    b = io.BytesIO(); img.save(b, "PNG", optimize=True); return b.getvalue()


def jpg_tex(aid, kind, size=1024):
    return Image.open(TEX / aid / f"{aid}_1K-JPG_{kind}.jpg").convert("RGB").resize((size, size))


def maps_from_rough(aid, spec=0.25):
    r = np.asarray(Image.open(TEX / aid / f"{aid}_1K-JPG_Roughness.jpg").convert("L"), float) / 255
    a = np.stack([np.full_like(r, spec), 1 - r, np.zeros_like(r)], -1)
    return Image.fromarray((a * 255).astype(np.uint8))


GP = ROOT / "data/derived/gopro"
PAL = json.load(open(GP / "palette.json"))  # scripts/p2/19_gopro_calibrate.py (dry-road only, wet mode excluded)


def _lab(a):
    c = a / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _rgb(lab):
    fy = (lab[..., 0] + 16) / 116; fx = fy + lab[..., 1] / 500; fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    M = np.linalg.inv(np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]))
    c = np.clip(xyz @ M.T, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return (c * 255).clip(0, 255).astype(np.uint8)


def graded(aid, cls, size=1024, contrast=1.0, dL=0.0, chroma=1.0):
    """CC0 texture structure, colour statistics of the GoPro class (Lab mean/std transfer).
    dL/chroma: deliberate offsets (texture is lit by AC, photo is lit by an overcast sky)."""
    a = np.asarray(jpg_tex(aid, "Color", size), float); lab = _lab(a)
    mu, sd = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    tm, ts = np.array(PAL[cls]["lab_mean"]), np.array(PAL[cls]["lab_std"])
    ts = np.minimum(ts, sd * 2.5) * contrast  # class std includes lighting variance across frames; keep texture detail plausible
    out = (lab - mu) / sd * ts + tm
    out[..., 0] += dL; out[..., 1:] *= chroma
    return Image.fromarray(_rgb(out))


def _load_fol():
    """photo leaf crops, sky-contaminated ones dropped, each normalised to the common Lab mean so tiles don't show"""
    out = []
    for p in sorted(GP.glob("foliage_*.png")):
        a = np.asarray(Image.open(p).convert("RGB"), float)
        if ((a.min(-1) > 190)).mean() > 0.01:
            continue
        out.append(_lab(a))
    mu = np.mean([l.reshape(-1, 3).mean(0) for l in out], 0)
    res = []
    for l in out:
        m = l.reshape(-1, 3).mean(0); l = l - m + mu
        res.append(_rgb(l))
    return res


FOL = _load_fol()


def photo_fill(H, W, r, tint=None):
    """tile real leaf texture (spring GoPro crops) over an H x W canvas"""
    acc = np.zeros((H, W, 3)); wsum = np.full((H, W, 1), 1e-6)
    win = np.outer(np.hanning(256), np.hanning(256))[..., None] + 1e-3  # feathered overlap, no seams
    for y in range(-128, H, 128):
        for x in range(-128, W, 128):
            p = FOL[r.integers(len(FOL))].astype(float)
            if r.uniform() < 0.5: p = p[:, ::-1]
            y0, x0 = max(y, 0), max(x, 0); y1, x1 = min(y + 256, H), min(x + 256, W)
            acc[y0:y1, x0:x1] += (p * win)[y0 - y:y1 - y, x0 - x:x1 - x]; wsum[y0:y1, x0:x1] += win[y0 - y:y1 - y, x0 - x:x1 - x]
    out = (acc / wsum).clip(0, 255).astype(np.uint8)
    if tint is not None:
        out = (out.astype(float) * tint).clip(0, 255).astype(np.uint8)
    return out


TREE_VARIANTS = 4          # card variants per species, side by side in one atlas (u = (v + u) / 4); 21 picks one per tree
# card trunk geometry shared with 21's 3D trunks: fork height (fraction of card height) and trunk half-width (px of 512)
TREE_FORK = {"oak": 0.16, "bay": 0.0, "euc": 0.42, "syc": 0.24}
TREE_TW = {"oak": 26, "bay": 9, "euc": 12, "syc": 15}
# bark colours sampled from the spring GoPro trunks (wet-ish overcast): mean sRGB, mottle sRGB, moss/lichen sRGB, moss amount
BARK = {"oak": ((66, 60, 53), (92, 86, 76), (88, 98, 58), 0.45),      # coast live oak: dark grey-brown, mossy
        "bay": ((104, 100, 92), (128, 124, 114), (96, 104, 70), 0.20),  # bay laurel: grey stems
        "euc": ((150, 138, 118), (182, 172, 152), (120, 110, 90), 0.0),  # euc: tan/grey strips
        "syc": ((196, 192, 178), (150, 140, 118), (170, 178, 150), 0.15)}  # sycamore/alder: pale mottled white/grey/tan
BARK_ID = {"oak": "Bark007", "bay": "Bark014", "euc": "Bark012", "syc": "Bark014"}


def _vnoise(r, H, W, cell):
    g = r.uniform(0, 1, (max(2, H // cell + 2), max(2, W // cell + 2)))
    return np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), float) / 255


def bark_tex(kind, H=512, W=512, seed=0):
    """CC0 ambientCG bark relief, re-coloured to the GoPro trunk colours, with low-frequency mottle and moss/lichen"""
    r = np.random.default_rng(100 + seed)
    src = np.asarray(jpg_tex(BARK_ID[kind], "Color", 1024).convert("L").resize((W, H)), float)
    lum = (src - src.mean()) / (src.std() + 1e-6)
    base, mott, moss = (np.array(c, float) for c in BARK[kind][:3]); mamt = BARK[kind][3]
    m1 = _vnoise(r, H, W, 48)
    if kind == "syc":  # plane-tree patchwork: sharp-edged pale/tan/grey plates
        m1 = (m1 > 0.5).astype(float) * 0.8 + 0.2 * _vnoise(r, H, W, 12)
    col = base[None, None] * (1 - 0.6 * m1[..., None]) + mott[None, None] * 0.6 * m1[..., None]
    col = col * (1 + 0.22 * lum[..., None])
    if mamt > 0:  # moss in the crevices + lichen blotches
        mm = np.clip((_vnoise(r, H, W, 64) - (1 - mamt)) * 3, 0, 1) * np.clip(0.6 - 0.25 * lum, 0, 1)
        col = col * (1 - mm[..., None]) + moss[None, None] * (1 + 0.15 * lum[..., None]) * mm[..., None]
    return col.clip(0, 255)


def _branch(segs, tips, r, x, y, ang, ln, w, depth, kind):
    """recursive tapered, sinuous limb: appends (x0,y0,x1,y1,w0,w1) segments; ang 0 = up"""
    n = 6; wend = w * 0.62
    curl = r.uniform(-0.12, 0.12)
    for i in range(n):
        ang += curl + r.normal(0, 0.10)
        if kind == "oak" and depth <= 2:
            ang += 0.05 * np.sign(ang)  # oak limbs droop outward and low
        if abs(x - 256) > 200 and np.sign(math.sin(ang)) == np.sign(x - 256):
            ang *= 0.6  # steer back inside the card
        x1 = x + math.sin(ang) * ln / n; y1 = y - math.cos(ang) * ln / n
        w1 = w + (wend - w) * (i + 1) / n
        segs.append((x, y, x1, y1, w + (wend - w) * i / n, w1)); x, y = x1, y1
    if depth == 0 or wend < 1.2:
        tips.append((x, y)); return
    kids = 2 if r.uniform() < 0.7 else 3
    spread = {"oak": 0.75, "bay": 0.35, "euc": 0.45, "syc": 0.65}[kind]
    for k in range(kids):
        da = (k - (kids - 1) / 2) * spread + r.normal(0, 0.15)
        _branch(segs, tips, r, x, y, ang + da, ln * r.uniform(0.6, 0.8), wend * (0.8 if k == 0 else 0.62), depth - 1, kind)


def _tree_card(kind, seed, W=512, H=1024):
    r = np.random.default_rng(seed)
    segs, tips = [], []
    fork = TREE_FORK[kind]; tw = TREE_TW[kind]; yf = H * (1 - fork)
    if kind == "bay":  # multi-stem: 3-5 grey stems from the root crown
        for k in range(r.integers(3, 6)):
            _branch(segs, tips, r, W / 2 + r.uniform(-14, 14), H, r.normal(0, 0.18), H * r.uniform(0.32, 0.45), r.uniform(7, 11), 4, kind)
    else:
        segs.append((W / 2, H, W / 2, yf, tw, tw * 0.85))  # straight base (matches 21's 3D trunk)
        if kind == "euc":
            _branch(segs, tips, r, W / 2, yf, r.normal(0, 0.06), H * 0.30, tw * 0.85, 4, kind)
        else:
            nl = {"oak": r.integers(3, 5), "syc": r.integers(2, 4)}[kind]
            for k in range(nl):
                a = (k - (nl - 1) / 2) * (1.1 if kind == "oak" else 0.7) / max(nl - 1, 1) * 2 + r.normal(0, 0.15)
                ln = H * (0.30 if kind == "oak" else 0.34) * r.uniform(0.8, 1.1)
                _branch(segs, tips, r, W / 2, yf, a, ln, tw * 0.85 * r.uniform(0.7, 0.9), 4, kind)
    # wood: three-pass shading (shadow side, mid, light from upper left) -> reads as round limbs
    wm = Image.new("L", (W, H), 0); ws = Image.new("L", (W, H), 0)
    dm, ds = ImageDraw.Draw(wm), ImageDraw.Draw(ws)

    def quad(d, s, k, off, val):
        x0, y0, x1, y1, w0, w1 = s
        dx, dy = x1 - x0, y1 - y0; L_ = math.hypot(dx, dy) + 1e-6; nx, ny = -dy / L_, dx / L_
        w0, w1 = w0 * k, w1 * k; o0, o1 = off * s[4], off * s[5]
        p = [(x0 + nx * (w0 - o0), y0 + ny * (w0 - o0)), (x1 + nx * (w1 - o1), y1 + ny * (w1 - o1)),
             (x1 - nx * (w1 + o1), y1 - ny * (w1 + o1)), (x0 - nx * (w0 + o0), y0 - ny * (w0 + o0))]
        d.polygon(p, fill=val); d.ellipse([x1 - w1, y1 - w1, x1 + w1, y1 + w1], fill=val)
    for s in segs:
        quad(dm, s, 1.0, 0, 255); quad(ds, s, 1.0, 0, 70)
    for s in segs:
        quad(ds, s, 0.7, 0.25, 140)
    for s in segs:
        quad(ds, s, 0.35, 0.45, 215)
    wa = np.asarray(wm) > 128
    sh = np.asarray(ws.filter(ImageFilter.GaussianBlur(1.5)), float) / 160
    bt = bark_tex(kind, 512, 512, seed)
    bark = np.vstack([bt, bt])[:H, :W]  # bark tiles vertically along the card
    wood = (bark * sh[..., None]).clip(0, 255)
    # crown: leaf clusters on the branch tips + envelope fill; gaps let the limbs show through
    mask = Image.new("L", (W, H), 0); d = ImageDraw.Draw(mask)
    env = {"oak": (W / 2, H * 0.54, W * 0.49, H * 0.30), "bay": (W / 2, H * 0.40, W * 0.40, H * 0.36),
           "euc": (W / 2, H * 0.32, W * 0.34, H * 0.30), "syc": (W / 2, H * 0.38, W * 0.44, H * 0.32)}[kind]
    cx, cy, rx, ry = env
    nb = {"oak": 650, "bay": 700, "euc": 420, "syc": 380}[kind]
    for _ in range(nb):
        a = r.uniform(0, 2 * np.pi); q = math.sqrt(r.uniform(0, 1))
        x = cx + math.cos(a) * rx * q; y = cy + math.sin(a) * ry * q
        if kind == "oak":  # skirts reach down over the road at the sides
            y += max(0, abs(x - cx) / rx - 0.45) * H * 0.22
        s_ = r.uniform(10, 30) * (1.0 - 0.35 * q)
        d.ellipse([x - s_, y - s_ * 0.8, x + s_, y + s_ * 0.8], fill=255)
    for (x, y) in tips:
        s_ = r.uniform(20, 40); d.ellipse([x - s_, y - s_ * 0.8, x + s_, y + s_ * 0.8], fill=255)
    m = np.asarray(mask.filter(ImageFilter.GaussianBlur(3)), float) / 255
    noise = _vnoise(r, H, W, 4) * 0.6 + _vnoise(r, H, W, 24) * 0.4
    thr = {"oak": 0.60, "bay": 0.58, "euc": 0.63, "syc": 0.65}[kind]
    alpha = (m * (0.7 + 0.55 * noise)) > thr
    # porous canopy: clumped leaf masses with sky/sun holes (~15-30 % of the crown) and ragged edges, so light gets through
    holes = (_vnoise(r, H, W, 22) * 0.65 + _vnoise(r, H, W, 6) * 0.35) > {"oak": 0.60, "bay": 0.63, "euc": 0.58, "syc": 0.57}[kind]
    crown = m > 0.3
    alpha = alpha & ~(holes & crown)
    leaves = photo_fill(H, W, r).astype(float)
    if kind == "syc":
        leaves = leaves * np.array([1.08, 1.12, 0.85])  # fresh yellow-green spring sycamore leaf
    elif kind == "bay":
        leaves = leaves * np.array([0.9, 0.95, 0.9])
    yy = (np.arange(H)[:, None] - (cy - ry)) / (2 * ry)
    shade = np.clip(1.15 - 0.5 * yy, 0.5, 1.2)
    rgb = (leaves * shade[..., None]).clip(0, 255)
    # limbs inside the crown sit in leaf shadow; lower trunk is in front
    yi = np.arange(H)[:, None]
    in_shadow = np.clip(0.55 + 0.45 * (yi - (cy - ry)) / (2.2 * ry), 0.55, 1.0)
    wood = wood * np.where(m[..., None] > 0.3, in_shadow[..., None] * 0.85, 1.0)
    front_wood = wa & (yi > yf - H * 0.04)
    leaf = alpha & ~front_wood
    out = np.where(leaf[..., None], rgb, wood)
    a = (leaf | wa) * 255
    return np.dstack([out, a]).astype(np.uint8)


def foliage(kind, seed):
    """tree card atlas (RGBA, TREE_VARIANTS x 512 wide, 1024 high): per variant a recursively branched, tapered,
    shaded limb system with graded CC0 bark (moss/lichen on oak), photo-leaf crown (GoPro crops) with gaps."""
    cards = [_tree_card(kind, seed * 17 + v) for v in range(TREE_VARIANTS)]
    return Image.fromarray(np.hstack(cards), "RGBA")


LEAF_TINT = {"oak": (1.0, 1.08, 0.80), "bay": (0.80, 0.90, 0.82), "euc": (0.92, 0.98, 0.90), "syc": (1.08, 1.16, 0.76)}  # spring: lime new growth, darker bay
LEAF_SIZE = {"oak": (7, 15), "bay": (9, 20), "euc": (8, 22), "syc": (12, 26)}  # px of a 512 cell (~1.2 m clump)


def _leaf_cell(kind, seed, S=512):
    """one alpha-tested leaf cluster: individual photo-textured leaves in a ragged clump with sky holes + twigs"""
    r = np.random.default_rng(500 + seed)
    mask = Image.new("L", (S, S), 0); lum = Image.new("L", (S, S), 128)
    dm, dl = ImageDraw.Draw(mask), ImageDraw.Draw(lum)
    c = S / 2; R = S * 0.47
    # 3-6 sub-clumps -> clumped masses with gaps between them
    subs = [(c + r.normal(0, R * 0.35), c + r.normal(0, R * 0.35), R * r.uniform(0.45, 0.7)) for _ in range(r.integers(4, 7))]
    holes = _vnoise(r, S, S, 40) > 0.80  # ~10-20 % sky holes inside the clump
    lo, hi = LEAF_SIZE[kind]
    for _ in range(3000):
        sx, sy, sr = subs[r.integers(len(subs))]
        a = r.uniform(0, 2 * np.pi); q = math.sqrt(r.uniform())
        x, y = sx + math.cos(a) * sr * q, sy + math.sin(a) * sr * q
        if not (14 < x < S - 14 and 14 < y < S - 14) or math.hypot(x - c, y - c) > R or holes[int(y), int(x)]:
            continue
        L_ = r.uniform(lo, hi); W_ = L_ * r.uniform(0.35, 0.55); th = r.uniform(0, np.pi)
        ca, sa = math.cos(th), math.sin(th)
        pts = [(x + ca * L_ * t - sa * W_ * w, y + sa * L_ * t + ca * W_ * w) for t, w in ((-1, 0), (-0.3, 1), (0.5, 0.7), (1, 0), (0.5, -0.7), (-0.3, -1))]
        dm.polygon(pts, fill=255)
        dl.polygon(pts, fill=int(np.clip(128 + r.normal(0, 22) - 30 * q + (y < c) * 10, 40, 230)))
    # twigs inside the clump
    for _ in range(r.integers(2, 5)):
        x, y = c + r.normal(0, 30), S * 0.9; ang = r.normal(0, 0.5)
        for i in range(5):
            ang += r.normal(0, 0.25); x1, y1 = x + math.sin(ang) * 40, y - math.cos(ang) * 40
            dm.line([(x, y), (x1, y1)], fill=254, width=max(1, 5 - i)); x, y = x1, y1
    m = np.asarray(mask); l = np.asarray(lum, float) / 128
    leaves = photo_fill(S, S, r).astype(float) * np.array(LEAF_TINT[kind]) * l[..., None]
    twig = np.array(BARK[kind][0], float) * 0.8
    rgb = np.where((m == 254)[..., None], twig, leaves).clip(0, 255)
    return np.dstack([rgb, np.where(m > 128, 255, 0)]).astype(np.uint8)


COL_TINT = (np.array([0.74, 0.80, 0.76]), np.array([0.90, 0.94, 0.90]), np.array([1.0, 1.04, 0.96]), np.array([1.12, 1.22, 0.74]))


def leaf_atlas():
    """tree_leaves.png 2048x2048: 4x4 cells of 512; row = species (v bottom-up: oak, bay, euc, syc), col = variant"""
    A = np.zeros((2048, 2048, 4), np.uint8)
    for ri, k in enumerate(("oak", "bay", "euc", "syc")):
        for v in range(4):
            y0 = (3 - ri) * 512
            cell = _leaf_cell(k, ri * 4 + v).astype(float)
            cell[..., :3] *= COL_TINT[v]  # per-cluster tint via column: darker old leaves (0) .. lime new growth (3)
            A[y0:y0 + 512, v * 512:(v + 1) * 512] = cell.clip(0, 255).astype(np.uint8)
    return Image.fromarray(A, "RGBA")


def trunk_atlas():
    """tree_trunk.png: bark strips u in [i/4,(i+1)/4] = oak, bay, euc, syc; mirror-tiled so v wraps seamlessly"""
    cols = []
    for k in ("oak", "bay", "euc", "syc"):
        b = bark_tex(k, 256, 256, 7)
        cols.append(np.vstack([b, b[::-1]]))
    return Image.fromarray(np.hstack(cols).clip(0, 255).astype(np.uint8))


def bank_tex():
    """Ground068 structure: its mossy parts -> GoPro bank-grass colour, the rest -> GoPro bare-dirt colour"""
    src = np.asarray(jpg_tex("Ground068", "Color"), float); lab0 = _lab(src)
    green = np.clip((-lab0[..., 1] - 2) / 8, 0, 1)[..., None]
    green = np.asarray(Image.fromarray((green[..., 0] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), float)[..., None] / 255
    dirt = _lab(np.asarray(graded("Ground068", "dirt_cut", contrast=0.9), float))
    gr = _lab(np.asarray(graded("Ground068", "grass", contrast=0.9, chroma=1.3), float))
    return Image.fromarray(_rgb(dirt * (1 - green) + gr * green))


def shrub_atlas():
    """two understory shrub styles side by side (u 0-0.5, 0.5-1), photo leaves, alpha-tested"""
    r = np.random.default_rng(21); W, H = 512, 256
    mask = Image.new("L", (W, H), 0); d = ImageDraw.Draw(mask)
    for half in (0, 1):
        for _ in range(120):
            x = half * 256 + 128 + r.normal(0, 55); y = H - abs(r.normal(0, 85)) - 10
            s_ = r.uniform(10, 30); d.ellipse([x - s_, y - s_, x + s_, y + s_], fill=255)
    m = np.asarray(mask.filter(ImageFilter.GaussianBlur(2)), float) / 255
    noise = np.asarray(Image.fromarray(r.uniform(0, 255, (H // 3, W // 3)).astype(np.uint8)).resize((W, H), Image.BICUBIC), float) / 255
    alpha = (m * (0.7 + 0.6 * noise)) > 0.6
    alpha[:, 250:262] = False  # keep halves separate under mip filtering
    lv = photo_fill(H, W, r).astype(float)
    lv[:, 256:] *= np.array([1.08, 1.06, 0.9])  # second style: lighter, yellower (coyote brush / new growth)
    yy = np.arange(H)[:, None] / H
    lv *= np.clip(1.2 - 0.5 * yy, 0.6, 1.2)[..., None]
    return Image.fromarray(np.dstack([lv.clip(0, 255), alpha * 255]).astype(np.uint8), "RGBA")


def solid(rgb, size=16):
    return Image.new("RGB", (size, size), rgb)


def kerb_tex():
    """apex (ridged) | exit (flat) kerb atlas + normal map: scripts/p2/race_tex.py (procedural)"""
    import race_tex as RT
    return RT.kerb_atlas()


def guardrail_tex():
    import race_tex as RT
    return RT.guardrail_wbeam()


import rasterio
with rasterio.open(DER / "backdrop_ortho_r05.tif") as r:
    o = r.read()[:3].transpose(1, 2, 0)
with rasterio.open(DER / "backdrop_dem_r10.tif") as r:
    bd_shape = r.shape
# art direction = SPRING (GoPro frames): backdrop ortho is summer imagery -> grade golden/brown dry grass toward the GoPro
# spring-grass palette; dark tree canopy (low L / already green) is left alone
_ol = _lab(o.astype(float)); _gt = np.array(PAL["grass"]["lab_mean"])
_gold = np.clip((_ol[..., 1] + 6) / 10, 0, 1) * np.clip((_ol[..., 2] - 6) / 12, 0, 1) * np.clip((_ol[..., 0] - 28) / 15, 0, 1)
_ol[..., 1] += (_gt[1] - _ol[..., 1]) * _gold * 0.9; _ol[..., 2] += (_gt[2] - _ol[..., 2]) * _gold * 0.8
_ol[..., 0] -= _gold * np.clip(_ol[..., 0] - _gt[0], 0, None) * 0.5  # dry grass is brighter than lush grass
o = _rgb(_ol)
backdrop_img = Image.fromarray(o).resize((4096, int(4096 * o.shape[0] / o.shape[1]))).filter(ImageFilter.GaussianBlur(0.6))

def road_surface_tex():
    """12 m procedural asphalt atlas, worn-paint line textures, RPM and decal atlas (scripts/p2/road_surface.py)"""
    import road_surface as RS
    base = graded("Asphalt026C", "asphalt_dry", contrast=0.55, dL=-6)
    d, n, mp = RS.asphalt_atlas(base, jpg_tex("Asphalt026C", "NormalDX"), Image.open(TEX / "Asphalt026C" / "Asphalt026C_1K-JPG_Roughness.jpg"))
    yel, wht = RS.paint_colours(ROOT)
    asph = tuple(int(v) for v in np.asarray(base, float).reshape(-1, 3).mean(0))
    def patch_rgb(k, size):  # newer asphalt: darker, finer, less oxidised; k=3 fresh black patch
        g = graded("Asphalt026C", "asphalt_dry", 512, contrast=0.35 if k < 3 else 0.25, dL=(-11, -15, -8, -34)[k])
        return np.roll(np.asarray(g.resize((size, size)), float), 97 * k, (0, 1))
    return {"asphalt_d.png": png(d), "asphalt_n.png": png(n), "asphalt_maps.png": png(mp),
            "paint_white.png": png(RS.line_tex("single", wht, asph, 5)), "paint_yellow.png": png(RS.line_tex("single", yel, asph, 6)),
            "paint_yellow_dbl.png": png(RS.line_tex("double", yel, asph, 7)), "rpm.png": png(RS.rpm_tex(yel)),
            "road_decal.png": png(RS.decal_atlas(patch_rgb)),
            # freshly repaved span (config/surface_overrides.yaml): dark, fine uniform aggregate, crisp paint
            "asphalt_fresh_d.png": png(fresh_asphalt(asph)),
            "paint_white_fresh.png": png(RS.line_tex("single", wht, (24, 24, 25), 15, fresh=True)),
            "paint_yellow_fresh.png": png(RS.line_tex("single", yel, (24, 24, 25), 16, fresh=True)),
            "paint_yellow_dbl_fresh.png": png(RS.line_tex("double", yel, (24, 24, 25), 17, fresh=True))}


def fresh_asphalt(asph, size=2048, seed=41):
    """new black asphalt: the CC0 tile at low contrast, 3 m per tile (same 12 m atlas UVs as the old road), plus fine
    uniform aggregate speckle; no wear lanes, cracks or patches. Overall level set by albedo_calibrate (~0.07)."""
    g = np.asarray(graded("Asphalt026C", "asphalt_dry", 512, contrast=0.25, dL=-30).resize((size // 4, size // 4)), float)
    a = np.tile(g, (4, 4, 1))
    r = np.random.default_rng(seed)
    a += r.normal(0, 7, a.shape[:2])[..., None] * np.array([1, 1, 1.03])
    a = 0.85 * a + 0.15 * a.mean((0, 1))
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


T = {
    # dry-asphalt grade; AC lighting is brighter than the overcast photos, so pull L down a little
    "ground_d.png": png(graded("Grass004", "grass", contrast=0.8, dL=4, chroma=1.35)),  # vivid spring grass
    "rock_d.png": png(graded("Rock030", "dirt_cut", contrast=0.9, dL=-4, chroma=0.6)),
    "dirt_d.png": png(graded("Ground048", "litter", contrast=0.7)),          # road shoulder: oak leaf litter + gravel
    "litter_d.png": png(graded("Ground048", "litter", contrast=0.8, dL=-3)),
    "bank_d.png": png(bank_tex()),  # moss / new grass over bare dirt cut banks
    "shrub.png": png(shrub_atlas()),
    "backdrop_d.png": png(backdrop_img),
    # 3D trees (21 + trees3d.py): leaf-cluster atlas + bark strips. foliage() card atlases kept for reference, unused
    "tree_leaves.png": png(leaf_atlas()), "tree_trunk.png": png(trunk_atlas()),
    **dict(zip(("kerb_d.png", "kerb_n.png"), map(png, kerb_tex()))), "runoff_d.png": png(__import__("race_tex").runoff_tex()),
    "guardrail.png": png(guardrail_tex()), "black.png": png(solid((0, 0, 0))),
    **road_surface_tex(),
}

def albedo_calibrate(T, cfg=ROOT / "config" / "materials.yaml"):
    """Rescale photo-graded textures to plausible diffuse albedo (camera exposure != albedo; Pure/CSP blow
    out over-bright albedo at midday). Mean linear luminance of opaque texels -> target; hue/contrast kept."""
    import yaml
    tgt = (yaml.safe_load(open(cfg)) or {}).get("albedo", {})
    rep = {}
    for name, a in tgt.items():
        if name not in T:
            continue
        im = Image.open(io.BytesIO(T[name])); mode = im.mode
        arr = np.asarray(im.convert("RGBA")).astype(np.float32) / 255.0
        lin = np.where(arr[..., :3] <= 0.04045, arr[..., :3] / 12.92, ((arr[..., :3] + 0.055) / 1.055) ** 2.4)
        m = arr[..., 3] > 0.5
        if not m.any():
            continue
        Y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        y0 = float(Y[m].mean()); lin = np.clip(lin * (a / max(y0, 1e-4)), 0, 1)
        srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
        arr[..., :3] = srgb
        out = Image.fromarray((arr * 255 + 0.5).astype(np.uint8), "RGBA")
        T[name] = png(out if mode == "RGBA" else out.convert("RGB"))
        rep[name] = (round(y0, 3), a)
    (BUILD / "albedo_calibration.json").write_text(json.dumps(rep, indent=1))
    print("  albedo (before -> target):", ", ".join(f"{k} {v[0]}->{v[1]}" for k, v in rep.items()))


BASE = {"ksAmbient": 0.5, "ksDiffuse": 0.5, "ksSpecular": 0.0, "ksSpecularEXP": 1.0, "ksEmissive": (0, 0, 0), "ksAlphaRef": 0.0}
MATS = {
    "road_asphalt": Material("road_asphalt", "ksPerPixelMultiMap",
                             {**BASE, "ksAmbient": 0.5, "ksDiffuse": 0.6, "ksSpecular": 0.15, "ksSpecularEXP": 30.0, "fresnelC": 0.0, "fresnelEXP": 2.0, "fresnelMaxLevel": 0.15,
                              "isAdditive": 0.0, "useDetail": 0.0, "detailUVMultiplier": 1.0, "shadowBiasMult": 1.0},
                             {"txDiffuse": "asphalt_d.png", "txNormal": "asphalt_n.png", "txMaps": "asphalt_maps.png", "txDetail": "asphalt_d.png"}),
    "shoulder_dirt": Material("shoulder_dirt", "ksPerPixel", {**BASE, "ksAmbient": 0.5, "ksDiffuse": 0.55}, {"txDiffuse": "dirt_d.png"}),
    "ground": Material("ground", "ksPerPixel", dict(BASE), {"txDiffuse": "ground_d.png"}),
    "rock": Material("rock", "ksPerPixel", dict(BASE), {"txDiffuse": "rock_d.png"}),
    "litter": Material("litter", "ksPerPixel", {**BASE, "ksAmbient": 0.5, "ksDiffuse": 0.55}, {"txDiffuse": "litter_d.png"}),
    "bank": Material("bank", "ksPerPixel", dict(BASE), {"txDiffuse": "bank_d.png"}),
    "shrub": Material("shrub", "ksPerPixelAT", {**BASE, "ksAmbient": 0.5, "ksDiffuse": 0.5, "ksAlphaRef": 0.5}, {"txDiffuse": "shrub.png"}, alpha_tested=True),
    "backdrop": Material("backdrop", "ksPerPixel", {**BASE, "ksAmbient": 0.5, "ksDiffuse": 0.35}, {"txDiffuse": "backdrop_d.png"}),
    # markings + decals: alpha-tested (worn breakup), non-physical, slightly glossier than asphalt
    "paint_white": Material("paint_white", "ksPerPixelAT", {**BASE, "ksAlphaRef": 0.5, "ksSpecular": 0.12, "ksSpecularEXP": 25.0}, {"txDiffuse": "paint_white.png"}, alpha_tested=True),
    "paint_yellow": Material("paint_yellow", "ksPerPixelAT", {**BASE, "ksAlphaRef": 0.5, "ksSpecular": 0.12, "ksSpecularEXP": 25.0}, {"txDiffuse": "paint_yellow.png"}, alpha_tested=True),
    "paint_yellow_dbl": Material("paint_yellow_dbl", "ksPerPixelAT", {**BASE, "ksAlphaRef": 0.5, "ksSpecular": 0.12, "ksSpecularEXP": 25.0}, {"txDiffuse": "paint_yellow_dbl.png"}, alpha_tested=True),
    "rpm_yellow": Material("rpm_yellow", "ksPerPixel", {**BASE, "ksSpecular": 0.5, "ksSpecularEXP": 60.0}, {"txDiffuse": "rpm.png"}),
    "road_decal": Material("road_decal", "ksPerPixelAT", {**BASE, "ksAlphaRef": 0.5, "ksSpecular": 0.06, "ksSpecularEXP": 20.0}, {"txDiffuse": "road_decal.png"}, alpha_tested=True),
    "kerb": Material("kerb", "ksPerPixelNM", {**BASE, "ksSpecular": 0.18, "ksSpecularEXP": 24.0}, {"txDiffuse": "kerb_d.png", "txNormal": "kerb_n.png"}),
    "runoff": Material("runoff", "ksPerPixel", {**BASE, "ksSpecular": 0.08, "ksSpecularEXP": 15.0}, {"txDiffuse": "runoff_d.png"}),
    "guardrail": Material("guardrail", "ksPerPixel", {**BASE, "ksSpecular": 0.4, "ksSpecularEXP": 40.0}, {"txDiffuse": "guardrail.png"}),
    "invisible": Material("invisible", "ksPerPixel", dict(BASE), {"txDiffuse": "black.png"}),
}
# track dressing (gantries, huts, bleacher, corner barriers): atlas written by 21b_track_dressing.py
T["dressing.png"] = (BUILD / "dressing_atlas.png").read_bytes()
# race event assets (race_assets.py, called from 21b): Poly Haven CC0 diffuse maps + procedural tiles in one atlas. Mixed
# light/dark tiles authored at plausible albedo, so (like roadside.png) NOT run through albedo_calibrate.
T["race_assets.png"] = (BUILD / "race_assets_atlas.png").read_bytes()
# roadside atlas (fences, walls, poles, signs, buildings, cypress): written by 21c_roadside_assets.py. It mixes white signs with dark
# poles, so it is NOT run through albedo_calibrate (a single mean-luminance target would wash out one or the other); tiles are
# authored at plausible albedo already.
T["roadside.png"] = (BUILD / "roadside_atlas.png").read_bytes()
albedo_calibrate(T)
# ragged asphalt-edge strip atlas (edge_tex.py), from the calibrated shoulder textures so strip and shoulder match
import edge_tex as _ET
T["road_edge.png"] = png(_ET.edge_atlas({k: Image.open(io.BytesIO(T[f])) for k, f in
                                         (("litter", "litter_d.png"), ("dirt", "dirt_d.png"), ("grass", "ground_d.png"), ("ivy", "bank_d.png"))}))
MATS["road_asphalt_fresh"] = Material("road_asphalt_fresh", "ksPerPixelMultiMap",
    {**MATS["road_asphalt"].props, "ksSpecular": 0.2, "ksSpecularEXP": 40.0, "fresnelMaxLevel": 0.18} if hasattr(MATS["road_asphalt"], "props") else
    {**BASE, "ksAmbient": 0.5, "ksDiffuse": 0.6, "ksSpecular": 0.2, "ksSpecularEXP": 40.0, "fresnelC": 0.0, "fresnelEXP": 2.0, "fresnelMaxLevel": 0.18,
     "isAdditive": 0.0, "useDetail": 0.0, "detailUVMultiplier": 1.0, "shadowBiasMult": 1.0},
    {"txDiffuse": "asphalt_fresh_d.png", "txNormal": "asphalt_n.png", "txMaps": "asphalt_maps.png", "txDetail": "asphalt_fresh_d.png"})
for _k in ("paint_white", "paint_yellow", "paint_yellow_dbl"):   # crisp, unworn, slightly glossier new paint
    MATS[_k + "_fresh"] = Material(_k + "_fresh", "ksPerPixelAT", {**BASE, "ksAlphaRef": 0.5, "ksSpecular": 0.16, "ksSpecularEXP": 30.0}, {"txDiffuse": _k + "_fresh.png"}, alpha_tested=True)
MATS["road_edge"] = Material("road_edge", "ksPerPixelAT", {**BASE, "ksAmbient": 0.5, "ksDiffuse": 0.55, "ksAlphaRef": 0.5}, {"txDiffuse": "road_edge.png"}, alpha_tested=True)
# RGBA atlas: opaque tiles + alpha-tested catch-fence mesh / flag cut-outs, one material = one draw call per kn5
MATS["dressing"] = Material("dressing", "ksPerPixelAT", {**BASE, "ksSpecular": 0.12, "ksSpecularEXP": 20.0}, {"txDiffuse": "dressing.png"}, alpha_tested=True)
MATS["race_assets"] = Material("race_assets", "ksPerPixelAT", {**BASE, "ksSpecular": 0.1, "ksSpecularEXP": 20.0, "ksAlphaRef": 0.5}, {"txDiffuse": "race_assets.png"}, alpha_tested=True)
MATS["roadside"] = Material("roadside", "ksPerPixelAT", {**BASE, "ksSpecular": 0.06, "ksSpecularEXP": 20.0, "ksAlphaRef": 0.5}, {"txDiffuse": "roadside.png"}, alpha_tested=True)
for sp in ("oak", "bay", "euc", "syc"):
    MATS[f"tree_{sp}"] = Material(f"tree_{sp}", "ksPerPixelAT", {**BASE, "ksAmbient": 0.5, "ksDiffuse": 0.5, "ksAlphaRef": 0.5}, {"txDiffuse": f"tree_{sp}.png"}, alpha_tested=True)
MATS["tree_trunk"] = Material("tree_trunk", "ksPerPixel", {**BASE, "ksAmbient": 0.5, "ksDiffuse": 0.45}, {"txDiffuse": "tree_trunk.png"})
MATS["tree_leaves"] = Material("tree_leaves", "ksPerPixelAT", {**BASE, "ksAmbient": 0.5, "ksDiffuse": 0.5, "ksAlphaRef": 0.5}, {"txDiffuse": "tree_leaves.png"}, alpha_tested=True)


# ---- terrain macro colour layer (scripts/p2/terrain_macro.py): per-chunk unique macro x tiled class detail (ksPerPixelMultiMap)
import terrain_macro as _TM
_TMP = {**BASE, "ksSpecular": 0.0, "ksSpecularEXP": 1.0, "fresnelC": 0.0, "fresnelEXP": 1.0, "fresnelMaxLevel": 0.0,
        "isAdditive": 0.0, "useDetail": 1.0, "detailUVMultiplier": 1.0, "shadowBiasMult": 1.0}
T["ter_flat_n.png"] = png(solid((128, 128, 255)))
T["ter_maps.png"] = png(solid((20, 0, 0)))
_ref = _TM.macro_ref(PAL, _rgb)
for _c, _f in (("ground", "ground_d.png"), ("bank", "bank_d.png"), ("litter", "litter_d.png")):
    T[f"ter_{_c}_det.png"] = png(_TM.detail_from(T[_f], _ref))
_rd, _rdet = _TM.rock_textures(T["rock_d.png"])
T["ter_rock_d.png"], T["ter_rock_det.png"] = png(_rd), png(_rdet)
MATS["rock"] = Material("rock", "ksPerPixelMultiMap", {**_TMP, "detailUVMultiplier": _TM.ROCK_DETAIL_MULT},
                        {"txDiffuse": "ter_rock_d.png", "txNormal": "ter_flat_n.png", "txMaps": "ter_maps.png", "txDetail": "ter_rock_det.png"})
_ortho = _TM.Ortho(DER / "ortho_r100.tif", json.load(open(ROOT / "config/origin.json")))
_tile = _TM.tile_gray_from(jpg_tex("Ground048", "Color"))
_macro = {}
for _m in G.get("terrain", []):
    mc = _m.get("macro")
    if not mc:
        continue
    c = mc["chunk"]
    if c not in _macro:
        _macro[c] = f"ter_macro_{c:02d}.png"
        T[_macro[c]] = png(_TM.macro_map(mc, _ortho, PAL, _lab, _rgb, _tile, seed=1000 + c))
    if _m["material"] not in MATS:
        MATS[_m["material"]] = Material(_m["material"], "ksPerPixelMultiMap",
                                        {**_TMP, "detailUVMultiplier": round(mc["size"] / _TM.DETAIL_PERIOD[mc["cls"]], 3)},
                                        {"txDiffuse": _macro[c], "txNormal": "ter_flat_n.png", "txMaps": "ter_maps.png",
                                         "txDetail": f"ter_{mc['cls']}_det.png"})
del _ortho
print(f"  terrain macro: {len(_macro)} chunk maps, {sum(k.startswith('ter_') for k in MATS)} terrain materials")


def merge_by(meshes, key):
    """merge visual meshes sharing (key(name), material) to cut draw calls"""
    out = {}
    for m in meshes:
        out.setdefault((key(m["name"]), m["material"]), []).append(m)
    res = []
    for (k, mat), ms_all in out.items():
        # dev4: keep every merged mesh under the 16-bit vertex limit (split greedily)
        groups_, cur, nv = [], [], 0
        for m in ms_all:
            if cur and nv + len(m["P"]) > 65000:
                groups_.append(cur); cur, nv = [], 0
            cur.append(m); nv += len(m["P"])
        groups_.append(cur)
        for gi_, ms in enumerate(groups_):
            k_ = k if gi_ == 0 else f"{k}_{gi_}"
            if len(ms) == 1:
                r = dict(ms[0]); r["name"] = ms[0]["name"] if len(ms_all) == 1 else k_; res.append(r); continue
            off = 0; P = []; Tt = []; UV = []; Nn = []
            for m in ms:
                P.append(m["P"]); Tt.append(m["T"] + off); UV.append(m["UV"]); Nn.append(m["N"]); off += len(m["P"])
            r = dict(ms[0]); r.update(name=k_, P=np.concatenate(P), T=np.concatenate(Tt), UV=np.concatenate(UV), N=np.concatenate(Nn))
            res.append(r)
    return res


def to_kn5_mesh(m):
    P = ac_from_local(m["P"]); Nn = ac_from_local(m["N"])
    return Mesh(m["name"], P, Nn, m["UV"], m["T"].reshape(-1), m["material"],
                cast_shadows=m.get("cast_shadows", True), visible=m.get("visible", True),
                transparent=m.get("transparent", False), renderable=m.get("renderable", True),
                lod_out=m.get("lod_out", 0.0))


def locator(name, l):
    p = np.array(l["p"]); f = np.array(l["fwd"]); f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0])
    # AC-space axes: Y up, local +Z = car forward, +X = car left (see docs/AC_CONVENTIONS.md, "Spawn / locator orientation")
    F = ac_from_local(f); U = ac_from_local(up)
    Z = F; Y = U; X = np.cross(Y, Z); X /= np.linalg.norm(X)  # rows X,Y,Z,T (row-vector matrix, translation in floats 12..14)
    t = ac_from_local(p)
    M = list(X) + [0] + list(Y) + [0] + list(Z) + [0] + list(t) + [1]
    return Dummy(name, [float(v) for v in M])


DRAWS = {}   # kn5 file -> renderable (non-physics) meshes = draw calls (before CSP shadow passes)


def write(fname, groups, extra_nodes=(), merge=None):
    ms = []
    for g in groups:
        gm = G.get(g, [])
        if merge and g in merge:
            gm = merge_by(gm, merge[g])
        ms += gm
    used = sorted({m["material"] for m in ms})
    mats = [MATS[u] for u in used]
    texn = sorted({t for u in used for t in MATS[u].textures.values()})
    nodes = [to_kn5_mesh(m) for m in ms] + list(extra_nodes)
    DRAWS[fname] = sum(-(-len(m["P"]) // 65535) for m in ms if m.get("renderable", True) and not m["name"].startswith("1WALL"))
    write_kn5(TRACK / fname, mats, {t: T[t] for t in texn}, nodes)
    write.n = getattr(write, "n", 0) + 1
    print(f"  [kn5 {write.n:2d}] {fname:28s} {len(nodes):4d} nodes  {(TRACK / fname).stat().st_size / 1e6:6.1f} MB")


gr_key = lambda n: "guardrail_all" if n.startswith(("guardrail_", "grpost_")) else n
def line_key(n):  # dev4: one mesh per 500 m (2 chunks) per material (was per chunk) - draw-call headroom
    t = n.rsplit("_", 1)[1]
    return f"lines_{int(t) // 2:02d}" if t.isdigit() else "lines_" + t
def road_key(n):  # visual edge strips L+R of 2 chunks -> one mesh; physical meshes keep their names
    import re as _re
    m_ = _re.match(r"^edge_[LR](\d+)$", n)
    return f"edge_{int(m_.group(1)) // 2:02d}" if m_ else n
def kerb_key(n):  # 1KERB_NB_07 -> 1KERB_NB_m01 (1 km); physical prefix 1KERB / 1RUNOFF kept
    a, d, c = n.split("_")
    return f"{a}_{d}_m{int(c) // 4:02d}" if c.isdigit() else n

write("palomares_road.kn5", ["road"], merge={"road": road_key})
write("palomares_terrain.kn5", ["terrain", "backdrop"])
write("palomares_trees.kn5", ["trees"])
write("palomares_barriers.kn5", ["barriers"], merge={"barriers": gr_key})
write("palomares_roadside.kn5", ["roadside"])     # 21c: guardrails, fences, walls, poles, signs, buildings + their 1WALL_rs_* collision
write("palomares_lines_road.kn5", ["lines_road"], merge={"lines_road": line_key})
# lines_track retired: track layouts keep the real markings (user request)
write("palomares_kerbs_nb.kn5", ["kerbs_nb"], merge={"kerbs_nb": kerb_key})
write("palomares_kerbs_sb.kn5", ["kerbs_sb"], merge={"kerbs_sb": kerb_key})
write("palomares_dressing.kn5", ["dressing"])
write("palomares_dressing_nb.kn5", ["dressing_nb"])   # direction-specific: distance boards face the run direction
write("palomares_dressing_sb.kn5", ["dressing_sb"])
SEG = meta["segments"]
for seg, sd in SEG.items():
    L = sd["loc"]
    write(f"palomares_timing_{seg}.kn5", [], extra_nodes=[locator(n, L[n]) for n in sorted(L)])

# ------------------------------------------------------------------ AC folder
SHARED = ["palomares_road.kn5", "palomares_terrain.kn5", "palomares_trees.kn5", "palomares_barriers.kn5", "palomares_roadside.kn5"]
STY = {"road": ("Road", "The road as it really is: real width, center and edge lines, no kerbs.", ["palomares_lines_road.kn5"]),
       "track": ("Track", "Same road surface and markings, plus red/white kerbs on apexes and exits and light circuit dressing at the gates.", ["palomares_lines_road.kn5"])}
LAYOUTS = {}
for seg, sd in SEG.items():
    sgn = 1 if sd["s_to"] > sd["s_from"] else -1
    for sty, (sname, sdesc, sfiles) in STY.items():
        files = SHARED + [f"palomares_timing_{seg}.kn5"] + sfiles + ([f"palomares_kerbs_{'nb' if sgn > 0 else 'sb'}.kn5", "palomares_dressing.kn5", f"palomares_dressing_{'nb' if sgn > 0 else 'sb'}.kn5"] if sty == "track" else [])
        LAYOUTS[f"{sty}_{seg}"] = dict(files=files, seg=seg, s_from=sd["s_from"], s_to=sd["s_to"], sgn=sgn,
                                       name=f"Palomares Road — {sd['name']} ({sname}) v{VERSION}",
                                       desc=f"Palomares Road, Alameda County CA: {sd['name']}, point to point. {sdesc}")
SURF = """[SURFACE_0]
KEY=ROAD
FRICTION=0.99
DAMPING=0
WAV=
WAV_PITCH=0
FF_EFFECT=NULL
DIRT_ADDITIVE=0
IS_VALID_TRACK=1
BLACK_FLAG=0
SIN_HEIGHT=0
SIN_LENGTH=0
IS_PITLANE=0
VIBRATION_GAIN=0
VIBRATION_LENGTH=0

[SURFACE_1]
KEY=KERB
FRICTION=0.93
DAMPING=0
WAV=kerb.wav
WAV_PITCH=1.3
FF_EFFECT=1
DIRT_ADDITIVE=0
IS_VALID_TRACK=1
BLACK_FLAG=0
SIN_HEIGHT=0
SIN_LENGTH=0
IS_PITLANE=0
VIBRATION_GAIN=1.0
VIBRATION_LENGTH=1.5

[SURFACE_2]
KEY=GRASS
FRICTION=0.60
DAMPING=0.1
WAV=grass.wav
WAV_PITCH=0
FF_EFFECT=NULL
DIRT_ADDITIVE=1
IS_VALID_TRACK=0
BLACK_FLAG=0
SIN_HEIGHT=0.03
SIN_LENGTH=0.5
IS_PITLANE=0
VIBRATION_GAIN=0.15
VIBRATION_LENGTH=0.5

[SURFACE_3]
KEY=WALL
FRICTION=0.40
DAMPING=0
WAV=
WAV_PITCH=0
FF_EFFECT=NULL
DIRT_ADDITIVE=0
IS_VALID_TRACK=0
BLACK_FLAG=0
SIN_HEIGHT=0
SIN_LENGTH=0
IS_PITLANE=0
VIBRATION_GAIN=0
VIBRATION_LENGTH=0

[SURFACE_4]
KEY=RUNOFF
FRICTION=0.82
DAMPING=0
WAV=
WAV_PITCH=0
FF_EFFECT=NULL
DIRT_ADDITIVE=0
IS_VALID_TRACK=0
BLACK_FLAG=0
SIN_HEIGHT=0
SIN_LENGTH=0
IS_PITLANE=0
VIBRATION_GAIN=0.05
VIBRATION_LENGTH=0.3
"""
LIGHT = "[LIGHTING]\nSUN_PITCH_ANGLE=45\nSUN_HEADING_ANGLE=200\n"

# map.png (top-down outline) + map.ini
road = G["road"]
cl = np.concatenate([m["P"][:, :2] for m in road if m["name"].startswith("1ROAD")])
import csv as _csv
rows = [r for r in _csv.DictReader(open(DER / "centerline.csv")) if float(r["s_m"]) <= meta["s_end"] + 1]
xy = np.array([[float(r["x_local"]), float(r["y_local"])] for r in rows])
acx, acz = xy[:, 0], -xy[:, 1]
SCALE = 2.0; MARGIN = 20
W = int((acx.max() - acx.min()) / SCALE + 2 * MARGIN); Hh = int((acz.max() - acz.min()) / SCALE + 2 * MARGIN)
XO = -acx.min() + MARGIN * SCALE; ZO = -acz.min() + MARGIN * SCALE
mp = Image.new("RGBA", (W, Hh), (0, 0, 0, 0)); d = ImageDraw.Draw(mp)
pts = [((x + XO) / SCALE, (z + ZO) / SCALE) for x, z in zip(acx[::3], acz[::3])]
d.line(pts, fill=(255, 255, 255, 140), width=6, joint="curve")
MAPINI = f"[PARAMETERS]\nWIDTH={W}\nHEIGHT={Hh}\nMARGIN={MARGIN}\nSCALE_FACTOR={SCALE}\nX_OFFSET={XO:.3f}\nZ_OFFSET={ZO:.3f}\nDRAWING_SIZE=10\nMAX_SIZE=1600\n"


def cams(L_):
    # simple replay cameras every ~400 m along the run, 8 m off the road, 4 m up; IN/OUT are run fractions
    out = ["[HEADER]\nVERSION=3\nCAMERA_COUNT={n}\nSET_NAME=Palomares\n"]
    k = 0
    a_, b_ = sorted((int(L_["s_from"]), int(L_["s_to"]))); runlen = b_ - a_
    for s in range(a_ + 80, b_ - 20, 400):
        fr_ = (s - a_) / runlen if L_["sgn"] > 0 else (b_ - s) / runlen
        r = rows[s]
        x, y, z = float(r["x_local"]), float(r["y_local"]), float(r["z_navd88"])
        tx = float(rows[s + 1]["x_local"]) - x; ty = float(rows[s + 1]["y_local"]) - y
        n = math.hypot(tx, ty); tx, ty = tx / n, ty / n
        cx, cy, cz = x + ty * 8, y - tx * 8, z + 4
        out.append(f"[CAMERA_{k}]\nNAME=cam{k}\nPOSITION={cx:.2f},{cz:.2f},{-cy:.2f}\nFORWARD={-ty:.3f},-0.2,{-tx:.3f}\nUP=0,1,0\nMIN_FOV=10\nMAX_FOV=50\nIN_POINT={max(0, fr_ - 200 / runlen):.4f}\nOUT_POINT={min(1, fr_ + 200 / runlen):.4f}\nSHADOW_SPLIT0=50\nSHADOW_SPLIT1=200\nSHADOW_SPLIT2=600\nNEAR_PLANE=1\nFAR_PLANE=5000\nMIN_EXPOSURE=0\nMAX_EXPOSURE=10000\nDOF_FACTOR=10\nDOF_RANGE=10000\nDOF_FOCUS=0\nDOF_MANUAL=0\nSPLINE=\nSPLINE_ROTATION=0\nFOV_GAMMA=0\nSPLINE_ANIMATION_LENGTH=0\nIS_FIXED=0\n")
        k += 1
    out[0] = out[0].format(n=k)
    return "\n".join(out)


ATTR = ("Data: centerline from OpenStreetMap contributors (ODbL, openstreetmap.org/copyright); "
        "terrain from USGS 3DEP LiDAR CA_AlamedaCounty_2021_B21 and USGS imagery (public domain); "
        "textures ambientCG (CC0). Fan-made, not affiliated with Alameda County.")
for lay, L_ in LAYOUTS.items():
    (TRACK / f"models_{lay}.ini").write_text("".join(f"[MODEL_{i}]\nFILE={f}\nPOSITION=0,0,0\nROTATION=0,0,0\n\n" for i, f in enumerate(L_["files"])))
    dd = TRACK / lay / "data"; dd.mkdir(parents=True)
    (dd / "surfaces.ini").write_text(SURF)
    (dd / "lighting.ini").write_text(LIGHT)
    (dd / "cameras.ini").write_text(cams(L_))
    (dd / "map.ini").write_text(MAPINI)
    (dd / "groove.ini").write_text("[HEADER]\nGROOVES_NUMBER=0\n")
    (dd / "audio_sources.ini").write_text("")
    # map: full road dim, this layout's run bright, start dot green / finish red
    mpl = mp.copy(); dl = ImageDraw.Draw(mpl)
    a_, b_ = sorted((int(L_["s_from"]), int(L_["s_to"])))
    dl.line(pts[a_ // 3:b_ // 3 + 1], fill=(255, 210, 60, 255), width=8, joint="curve")
    for s_, col in ((L_["s_from"], (40, 220, 60, 255)), (L_["s_to"], (230, 40, 40, 255))):
        px, pz = pts[min(int(s_) // 3, len(pts) - 1)]; dl.ellipse([px - 9, pz - 9, px + 9, pz + 9], fill=col)
    mpl.save(TRACK / lay / "map.png"); mp_l = mpl
    ui = TRACK / "ui" / lay; ui.mkdir(parents=True)
    json.dump({"name": L_["name"], "description": L_["desc"] + " " + ATTR, "tags": ["hillclimb", "point to point", "usa", "touge", "lidar"],
               "geotags": ["37.5978 N", "121.9469 W"], "country": "USA", "city": "Sunol / Castro Valley, CA",
               "length": f"{abs(L_['s_to'] - L_['s_from']):.0f}", "width": "5-8", "pitboxes": "1", "run": "point to point",
               "author": "palomares pipeline", "version": VERSION, "url": ""},
              open(ui / "ui_track.json", "w"), indent=2, ensure_ascii=False)
    # outline.png: same line on transparent 1000 px
    ol = mp_l.copy(); ol.thumbnail((1000, 1000)); ol.save(ui / "outline.png")
# AI / ideal line: <layout>/ai/fast_lane.ai from the racing line of the layout's run direction (ai_spline.py)
import ai_spline
from geom import Road as _Road
AI = ai_spline.export_layouts(TRACK, {lay: dict(s_from=L_["s_from"], s_to=L_["s_to"]) for lay, L_ in LAYOUTS.items()}, _Road(meta["s_end"]))
json.dump(AI, open(BUILD / "ai_lines.json", "w"), indent=1)
_dc = {lay: sum(DRAWS.get(f, 0) for f in L_["files"]) for lay, L_ in LAYOUTS.items()}
print("  draw calls (renderable meshes, per layout): max %d (%s), min %d; roadside kn5 = %d, budget 1500" % (
    max(_dc.values()), max(_dc, key=_dc.get), min(_dc.values()), DRAWS.get("palomares_roadside.kn5", 0)))
print("   by file:", {f: n for f, n in DRAWS.items() if n})
from ext_config import write_ext_config
print("  ext_config ->", write_ext_config(TRACK, MATS))
print("track folder ->", TRACK)
