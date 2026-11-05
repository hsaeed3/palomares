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


def foliage(kind, seed):
    """tree card (RGBA 512x1024): photo-leaf crown (GoPro crops) in a clumped alpha envelope, dark oak-bark trunk.
    Shapes follow the frames: broad low crowns (oak), denser rounder (bay), tall open (euc/sycamore-ish)."""
    r = np.random.default_rng(seed)
    W, H = 512, 1024
    mask = Image.new("L", (W, H), 0); d = ImageDraw.Draw(mask)
    trunk_top = {"euc": 0.30, "oak": 0.22, "bay": 0.25}[kind]
    tw = {"euc": 14, "oak": 30, "bay": 20}[kind]
    bark = {"euc": (122, 114, 100), "oak": (74, 64, 56), "bay": (88, 78, 66)}[kind]
    if kind == "euc":
        cx, cy, rx, ry = W / 2, H * 0.40, W * 0.40, H * 0.38
    elif kind == "oak":
        cx, cy, rx, ry = W / 2, H * 0.44, W * 0.49, H * 0.40
    else:
        cx, cy, rx, ry = W / 2, H * 0.42, W * 0.45, H * 0.41
    blobs = []
    for _ in range(420):
        a = r.uniform(0, 2 * np.pi); q = math.sqrt(r.uniform(0, 1))
        x = cx + math.cos(a) * rx * q; y = cy + math.sin(a) * ry * q
        s_ = r.uniform(14, 46) * (1.0 - 0.4 * q)
        d.ellipse([x - s_, y - s_ * 0.85, x + s_, y + s_ * 0.85], fill=255); blobs.append((x, y, s_))
    m = np.asarray(mask.filter(ImageFilter.GaussianBlur(3)), float) / 255
    # ragged leafy edge: threshold against leaf-scale noise
    noise = np.asarray(Image.fromarray((r.uniform(0, 255, (H // 4, W // 4))).astype(np.uint8)).resize((W, H), Image.BICUBIC), float) / 255
    alpha = (m * (0.75 + 0.5 * noise)) > 0.62
    leaves = photo_fill(H, W, r).astype(float)
    # self-shadow: darker underneath and toward the trunk, lit crown top (overcast GoPro light)
    yy = (np.arange(H)[:, None] - (cy - ry)) / (2 * ry)
    shade = np.clip(1.15 - 0.55 * yy, 0.45, 1.2)
    rgb = (leaves * shade[..., None]).clip(0, 255)
    wood = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dt = ImageDraw.Draw(wood)
    y_top = H * (1 - trunk_top)
    dt.polygon([(W / 2 - tw, H), (W / 2 + tw, H), (W / 2 + tw * 0.5, y_top), (W / 2 - tw * 0.5, y_top)], fill=bark + (255,))
    for _ in range(7):
        x1 = cx + r.uniform(-rx, rx) * 0.75; y1 = cy + r.uniform(-0.2, 0.6) * ry
        dt.line([(W / 2, y_top + r.uniform(0, 40)), (x1, y1)], fill=bark + (255,), width=int(r.uniform(5, 11)))
    w_ = np.asarray(wood).astype(float); wd = w_[..., :3] * 0.8  # limbs sit in crown shadow
    lower = np.arange(H)[:, None] > cy + ry * 0.55  # trunk itself in front below the crown
    front = alpha & ~(lower & (w_[..., 3] > 0))
    rgb = np.where(front[..., None], rgb, np.where(np.arange(H)[:, None, None] > y_top, w_[..., :3], wd))
    im = Image.fromarray(np.dstack([rgb, (front | (w_[..., 3] > 0)) * 255]).astype(np.uint8), "RGBA")
    a_ = np.asarray(im).copy(); a_[..., 3] = np.where(a_[..., 3] > 128, 255, 0)
    return Image.fromarray(a_)


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
    im = Image.new("RGB", (64, 128)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 64, 64], fill=(200, 30, 30)); d.rectangle([0, 64, 64, 128], fill=(235, 235, 235))
    return im


def guardrail_tex():
    a = np.full((64, 256, 3), 165, np.uint8)
    a[:, :, :] = (np.asarray(jpg_tex("Asphalt026C", "Roughness", 256).resize((256, 64)))[..., :1] * 0.3 + 140).astype(np.uint8)
    a[20:24] -= 40; a[40:44] -= 40
    return Image.fromarray(a)


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
    "kerb.png": png(kerb_tex()), "guardrail.png": png(guardrail_tex()), "black.png": png(solid((0, 0, 0))),
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
    "kerb": Material("kerb", "ksPerPixel", {**BASE, "ksSpecular": 0.1, "ksSpecularEXP": 20.0}, {"txDiffuse": "kerb.png"}),
    "guardrail": Material("guardrail", "ksPerPixel", {**BASE, "ksSpecular": 0.4, "ksSpecularEXP": 40.0}, {"txDiffuse": "guardrail.png"}),
    "invisible": Material("invisible", "ksPerPixel", dict(BASE), {"txDiffuse": "black.png"}),
}
# track dressing (gantries, huts, bleacher, corner barriers): atlas written by 21b_track_dressing.py
T["dressing.png"] = (BUILD / "dressing_atlas.png").read_bytes()
MATS["dressing"] = Material("dressing", "ksPerPixel", {**BASE, "ksSpecular": 0.1, "ksSpecularEXP": 20.0}, {"txDiffuse": "dressing.png"})
for sp in ("oak", "bay", "euc"):
    MATS[f"tree_{sp}"] = Material(f"tree_{sp}", "ksPerPixelAT", {**BASE, "ksAmbient": 0.55, "ksDiffuse": 0.35}, {"txDiffuse": f"tree_{sp}.png"}, alpha_tested=True)


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
        files = SHARED + [f"palomares_timing_{seg}.kn5"] + sfiles + ([f"palomares_kerbs_{'nb' if sgn > 0 else 'sb'}.kn5", "palomares_dressing.kn5"] if sty == "track" else [])
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
print("track folder ->", TRACK)
