"""Phase 2: textures + materials + kn5 files + AC track folder -> build/palomares_road/"""
import io, json, math, pickle, shutil, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from geom import ROOT, DER, BUILD, CFG
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


def trunk_atlas():
    """bark strips for 21's 3D trunks: u in [i/4,(i+1)/4] = oak, bay, euc, syc"""
    return Image.fromarray(np.hstack([bark_tex(k, 512, 256, 7) for k in ("oak", "bay", "euc", "syc")]).astype(np.uint8))


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
            "road_decal.png": png(RS.decal_atlas(patch_rgb))}


T = {
    # dry-asphalt grade; AC lighting is brighter than the overcast photos, so pull L down a little
    "ground_d.png": png(graded("Grass004", "grass", contrast=0.8, dL=4, chroma=1.35)),  # vivid spring grass
    "rock_d.png": png(graded("Rock030", "dirt_cut", contrast=0.9, dL=-4, chroma=0.6)),
    "dirt_d.png": png(graded("Ground048", "litter", contrast=0.7)),          # road shoulder: oak leaf litter + gravel
    "litter_d.png": png(graded("Ground048", "litter", contrast=0.8, dL=-3)),
    "bank_d.png": png(bank_tex()),  # moss / new grass over bare dirt cut banks
    "shrub.png": png(shrub_atlas()),
    "backdrop_d.png": png(backdrop_img),
    "tree_oak.png": png(foliage("oak", 1)), "tree_bay.png": png(foliage("bay", 2)), "tree_euc.png": png(foliage("euc", 3)),
    "tree_syc.png": png(foliage("syc", 4)), "tree_trunk.png": png(trunk_atlas()),
    **dict(zip(("kerb_d.png", "kerb_n.png"), map(png, kerb_tex()))), "runoff_d.png": png(__import__("race_tex").runoff_tex()),
    "guardrail.png": png(guardrail_tex()), "black.png": png(solid((0, 0, 0))),
    **road_surface_tex(),
}
BASE = {"ksAmbient": 0.45, "ksDiffuse": 0.45, "ksSpecular": 0.0, "ksSpecularEXP": 1.0, "ksEmissive": (0, 0, 0), "ksAlphaRef": 0.0}
MATS = {
    "road_asphalt": Material("road_asphalt", "ksPerPixelMultiMap",
                             {**BASE, "ksSpecular": 0.15, "ksSpecularEXP": 30.0, "fresnelC": 0.0, "fresnelEXP": 2.0, "fresnelMaxLevel": 0.15,
                              "isAdditive": 0.0, "useDetail": 0.0, "detailUVMultiplier": 1.0, "shadowBiasMult": 1.0},
                             {"txDiffuse": "asphalt_d.png", "txNormal": "asphalt_n.png", "txMaps": "asphalt_maps.png", "txDetail": "asphalt_d.png"}),
    "shoulder_dirt": Material("shoulder_dirt", "ksPerPixel", dict(BASE), {"txDiffuse": "dirt_d.png"}),
    "ground": Material("ground", "ksPerPixel", dict(BASE), {"txDiffuse": "ground_d.png"}),
    "rock": Material("rock", "ksPerPixel", dict(BASE), {"txDiffuse": "rock_d.png"}),
    "litter": Material("litter", "ksPerPixel", dict(BASE), {"txDiffuse": "litter_d.png"}),
    "bank": Material("bank", "ksPerPixel", dict(BASE), {"txDiffuse": "bank_d.png"}),
    "shrub": Material("shrub", "ksPerPixelAT", {**BASE, "ksAmbient": 0.55, "ksDiffuse": 0.35}, {"txDiffuse": "shrub.png"}, alpha_tested=True),
    "backdrop": Material("backdrop", "ksPerPixel", {**BASE, "ksAmbient": 0.5, "ksDiffuse": 0.35}, {"txDiffuse": "backdrop_d.png"}),
    # markings + decals: alpha-tested (worn breakup), non-physical, slightly glossier than asphalt
    "paint_white": Material("paint_white", "ksPerPixelAT", {**BASE, "ksSpecular": 0.12, "ksSpecularEXP": 25.0}, {"txDiffuse": "paint_white.png"}, alpha_tested=True),
    "paint_yellow": Material("paint_yellow", "ksPerPixelAT", {**BASE, "ksSpecular": 0.12, "ksSpecularEXP": 25.0}, {"txDiffuse": "paint_yellow.png"}, alpha_tested=True),
    "paint_yellow_dbl": Material("paint_yellow_dbl", "ksPerPixelAT", {**BASE, "ksSpecular": 0.12, "ksSpecularEXP": 25.0}, {"txDiffuse": "paint_yellow_dbl.png"}, alpha_tested=True),
    "rpm_yellow": Material("rpm_yellow", "ksPerPixel", {**BASE, "ksSpecular": 0.5, "ksSpecularEXP": 60.0}, {"txDiffuse": "rpm.png"}),
    "road_decal": Material("road_decal", "ksPerPixelAT", {**BASE, "ksSpecular": 0.2, "ksSpecularEXP": 35.0}, {"txDiffuse": "road_decal.png"}, alpha_tested=True),
    "kerb": Material("kerb", "ksPerPixelNM", {**BASE, "ksSpecular": 0.18, "ksSpecularEXP": 24.0}, {"txDiffuse": "kerb_d.png", "txNormal": "kerb_n.png"}),
    "runoff": Material("runoff", "ksPerPixel", {**BASE, "ksSpecular": 0.08, "ksSpecularEXP": 15.0}, {"txDiffuse": "runoff_d.png"}),
    "guardrail": Material("guardrail", "ksPerPixel", {**BASE, "ksSpecular": 0.4, "ksSpecularEXP": 40.0}, {"txDiffuse": "guardrail.png"}),
    "invisible": Material("invisible", "ksPerPixel", dict(BASE), {"txDiffuse": "black.png"}),
}
# track dressing (gantries, huts, bleacher, corner barriers): atlas written by 21b_track_dressing.py
T["dressing.png"] = (BUILD / "dressing_atlas.png").read_bytes()
# RGBA atlas: opaque tiles + alpha-tested catch-fence mesh / flag cut-outs, one material = one draw call per kn5
MATS["dressing"] = Material("dressing", "ksPerPixelAT", {**BASE, "ksSpecular": 0.12, "ksSpecularEXP": 20.0}, {"txDiffuse": "dressing.png"}, alpha_tested=True)
for sp in ("oak", "bay", "euc", "syc"):
    MATS[f"tree_{sp}"] = Material(f"tree_{sp}", "ksPerPixelAT", {**BASE, "ksAmbient": 0.55, "ksDiffuse": 0.35}, {"txDiffuse": f"tree_{sp}.png"}, alpha_tested=True)
MATS["tree_trunk"] = Material("tree_trunk", "ksPerPixel", {**BASE, "ksAmbient": 0.5, "ksDiffuse": 0.45}, {"txDiffuse": "tree_trunk.png"})


def merge_by(meshes, key):
    """merge visual meshes sharing (key(name), material) to cut draw calls"""
    out = {}
    for m in meshes:
        out.setdefault((key(m["name"]), m["material"]), []).append(m)
    res = []
    for (k, mat), ms in out.items():
        if len(ms) == 1:
            res.append(ms[0]); continue
        off = 0; P = []; Tt = []; UV = []; Nn = []
        for m in ms:
            P.append(m["P"]); Tt.append(m["T"] + off); UV.append(m["UV"]); Nn.append(m["N"]); off += len(m["P"])
        r = dict(ms[0]); r.update(name=k, P=np.concatenate(P), T=np.concatenate(Tt), UV=np.concatenate(UV), N=np.concatenate(Nn))
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
    # AC-space axes (AC-Tools convention: Y up, -Z front)
    F = ac_from_local(f); U = ac_from_local(up)
    Z = -F; Y = U; X = np.cross(Y, Z); X /= np.linalg.norm(X)
    t = ac_from_local(p)
    M = list(X) + [0] + list(Y) + [0] + list(Z) + [0] + list(t) + [1]
    return Dummy(name, [float(v) for v in M])


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
    write_kn5(TRACK / fname, mats, {t: T[t] for t in texn}, nodes)
    write.n = getattr(write, "n", 0) + 1
    print(f"  [kn5 {write.n:2d}] {fname:28s} {len(nodes):4d} nodes  {(TRACK / fname).stat().st_size / 1e6:6.1f} MB")


gr_key = lambda n: "guardrail_all" if n.startswith(("guardrail_", "grpost_")) else n
line_key = lambda n: "lines_" + n.rsplit("_", 1)[1]  # one mesh per chunk per colour

write("palomares_road.kn5", ["road"])
write("palomares_terrain.kn5", ["terrain", "backdrop"])
write("palomares_trees.kn5", ["trees"])
write("palomares_barriers.kn5", ["barriers"], merge={"barriers": gr_key})
write("palomares_lines_road.kn5", ["lines_road"], merge={"lines_road": line_key})
# lines_track retired: track layouts keep the real markings (user request)
write("palomares_kerbs_nb.kn5", ["kerbs_nb"])
write("palomares_kerbs_sb.kn5", ["kerbs_sb"])
write("palomares_dressing.kn5", ["dressing"])
write("palomares_dressing_nb.kn5", ["dressing_nb"])   # direction-specific: distance boards face the run direction
write("palomares_dressing_sb.kn5", ["dressing_sb"])
SEG = meta["segments"]
for seg, sd in SEG.items():
    L = sd["loc"]
    write(f"palomares_timing_{seg}.kn5", [], extra_nodes=[locator(n, L[n]) for n in sorted(L)])

# ------------------------------------------------------------------ AC folder
SHARED = ["palomares_road.kn5", "palomares_terrain.kn5", "palomares_trees.kn5", "palomares_barriers.kn5"]
STY = {"road": ("Road", "The road as it really is: real width, center and edge lines, no kerbs.", ["palomares_lines_road.kn5"]),
       "track": ("Track", "Same road surface and markings, plus red/white kerbs on apexes and exits and light circuit dressing at the gates.", ["palomares_lines_road.kn5"])}
LAYOUTS = {}
for seg, sd in SEG.items():
    sgn = 1 if sd["s_to"] > sd["s_from"] else -1
    for sty, (sname, sdesc, sfiles) in STY.items():
        files = SHARED + [f"palomares_timing_{seg}.kn5"] + sfiles + ([f"palomares_kerbs_{'nb' if sgn > 0 else 'sb'}.kn5", "palomares_dressing.kn5", f"palomares_dressing_{'nb' if sgn > 0 else 'sb'}.kn5"] if sty == "track" else [])
        LAYOUTS[f"{sty}_{seg}"] = dict(files=files, seg=seg, s_from=sd["s_from"], s_to=sd["s_to"], sgn=sgn,
                                       name=f"Palomares Road — {sd['name']} ({sname})",
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
from ext_config import write_ext_config
print("  ext_config ->", write_ext_config(TRACK, MATS))
print("track folder ->", TRACK)
