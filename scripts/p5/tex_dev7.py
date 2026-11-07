"""dev7 texture sources for the 'ledger7' atlas (scripts/p5/assets_dev7.py).

Licence rule (docs/dev7/PLAN.md, user decision): only CC0 / public-domain sources.
  * ambientCG (CC0)  : photo-derived material colour maps, fetched once into data/raw/dev7/cc0/<id>/ (CC0 1.0)
  * Poly Haven (CC0) : low-poly prop models (glTF + diffuse), data/raw/dev7/polyhaven/<id>/ (CC0 1.0)
  * MUTCD sign layouts: public-domain US federal designs, redrawn here with DejaVu Sans (free font, generic look)
  * business names    : plain text in DejaVu Sans Bold only, no logos / brand artwork / brand fonts
Everything else is painted procedurally here.

  python3 scripts/p5/tex_dev7.py --fetch      # download the CC0 sources (about 90 MB, about 1 minute)
"""
import json, math, pathlib, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CC0 = ROOT / "data/raw/dev7/cc0"
PH = ROOT / "data/raw/dev7/polyhaven"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_C = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"

# ambientCG ids (CC0) -> what they are used for
AMBIENTCG = {
    "Bricks066": "red brick (firehouse, brick storefronts)",
    "Concrete034": "smooth cast concrete (UP girder, abutments, platform)",
    "Concrete018": "weathered concrete (piers, jersey, parapets)",
    "PaintedPlaster017": "stucco / plaster facades (tinted)",
    "WoodSiding008": "painted lap / board siding (tinted)",
    "WoodSiding011": "board-and-batten siding (depots, false fronts, tinted)",
    "PaintedMetal004": "painted steel (truss, girders, coach body, tinted)",
    "Metal032": "galvanised steel (poles, arms, railings)",
    "Rust004": "rusted steel (rails, deck girder weathering)",
    "Rock030": "rock (cut faces, ashlar abutment base)",
    "PavingStones070": "cut-stone ashlar (stone abutment, parapet)",
}
POLYHAVEN = {
    "painted_wooden_bench": "prop.bench",
    "metal_trash_can": "prop.trash_can",
    "fire_hydrant": "(reference only, dev5 hydrant kept)",
    "planter_box_01": "(reference only)",
    "concrete_road_barrier": "wall.jersey_barrier",
}


def fetch():
    CC0.mkdir(parents=True, exist_ok=True); PH.mkdir(parents=True, exist_ok=True)
    for i in AMBIENTCG:
        z = CC0 / f"{i}.zip"
        if not (CC0 / i / f"{i}_1K-JPG_Color.jpg").exists():
            subprocess.run(["curl", "-sfL", "-m", "120", "-o", str(z), f"https://ambientcg.com/get?file={i}_1K-JPG.zip"], check=True)
            subprocess.run(["unzip", "-qo", str(z), "-d", str(CC0 / i), "*Color.jpg"], check=True)
    import urllib.request
    for m in POLYHAVEN:
        d = PH / m; d.mkdir(exist_ok=True)
        if (d / f"{m}.gltf").exists():
            continue
        fj = json.loads(urllib.request.urlopen(f"https://api.polyhaven.com/files/{m}").read())
        g = fj["gltf"]["1k"]["gltf"]
        subprocess.run(["curl", "-sfL", "-o", str(d / f"{m}.gltf"), g["url"]], check=True)
        for k, v in g["include"].items():
            if k.endswith(".bin") or "_diff_" in k:
                (d / k).parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["curl", "-sfL", "-o", str(d / k), v["url"]], check=True)


def F(sz, cond=False):
    return ImageFont.truetype(FONT_C if cond else FONT, int(sz))


def cc0(i, size=256, tint=None, crop=1.0, bright=1.0):
    """ambientCG colour map -> RGBA tile (optionally a centre crop fraction, grey-tinted)"""
    p = CC0 / i / f"{i}_1K-JPG_Color.jpg"
    if not p.exists():
        fetch()
    im = Image.open(p).convert("RGB")
    if crop < 1.0:
        w, h = im.size; cw, ch = int(w * crop), int(h * crop); im = im.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
    im = im.resize((size, size) if isinstance(size, int) else size, Image.LANCZOS)
    a = np.asarray(im, np.float32)
    if tint is not None:
        g = a.mean(2, keepdims=True); g = g / max(g.mean(), 1.0)
        a = np.clip(g * np.array(tint, np.float32)[None, None, :], 0, 255)
    a = np.clip(a * bright, 0, 255)
    return Image.fromarray(a.astype(np.uint8)).convert("RGBA")


def solid(rgb, size=32, var=6, seed=0):
    rng = np.random.default_rng(seed)
    a = np.clip(np.array(rgb, np.float32)[None, None, :] + rng.normal(0, var, (size, size, 1)), 0, 255).astype(np.uint8)
    return Image.fromarray(a).convert("RGBA")


def _shade(im, k):
    a = np.asarray(im.convert("RGBA"), np.float32); a[..., :3] *= k; return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------ facades (one bay = 3.5 m wide)
def glass(w, h, lit=False):
    im = Image.new("RGBA", (w, h)); d = ImageDraw.Draw(im)
    for y in range(h):
        t = y / max(h - 1, 1); c = (int(58 + 50 * (1 - t)), int(70 + 52 * (1 - t)), int(80 + 54 * (1 - t)), 255)
        d.line([(0, y), (w, y)], fill=c)
    d.line([(int(w * 0.15), 0), (int(w * 0.55), h)], fill=(150, 166, 178, 255), width=max(2, w // 18))   # reflection streak
    return im


def shop_bay(wall, frame=(40, 36, 32), door=False, size=(192, 192)):
    """ground-floor storefront bay: wall tile + display window (transom, kick plate) [+ door]"""
    w, h = size; im = wall.resize(size).copy(); d = ImageDraw.Draw(im)
    x0, x1, y0, y1 = int(w * 0.08), int(w * 0.92), int(h * 0.12), int(h * 0.97)
    d.rectangle((x0 - 4, y0 - 4, x1 + 4, y1), fill=frame + (255,))
    if door:
        dx0, dx1 = int(w * 0.62), int(w * 0.86)
        im.paste(glass(x1 - x0 - 8 - (x1 - dx0), int(h * 0.55)), (x0 + 4, y0 + int(h * 0.15)))
        im.paste(glass(dx1 - dx0 - 8, y1 - y0 - 20), (dx0 + 4, y0 + 14))
        d.rectangle((x0 + 4, y0 + int(h * 0.72), dx0 - 6, y1 - 3), fill=tuple(int(c * 1.4) for c in frame) + (255,))   # kick plate
    else:
        im.paste(glass(x1 - x0 - 8, int(h * 0.58)), (x0 + 4, y0 + int(h * 0.15)))
        d.rectangle((x0 + 4, y0 + int(h * 0.75), x1 - 4, y1 - 3), fill=tuple(int(c * 1.4) for c in frame) + (255,))
    im.paste(glass(x1 - x0 - 8, int(h * 0.1)), (x0 + 4, y0 + 2))           # transom
    return im


def upper_bay(wall, frame=(236, 232, 222), arched=False, n=2, size=(192, 192)):
    """upper storey bay (3.5 m x 3.2 m): n sash windows with sills [+ arched heads]"""
    w, h = size; im = wall.resize(size).copy(); d = ImageDraw.Draw(im)
    ww = int(w / (n * 1.9)); gap = (w - n * ww) // (n + 1)
    for k in range(n):
        x0 = gap + k * (ww + gap); y0, y1 = int(h * 0.22), int(h * 0.78)
        if arched:
            d.ellipse((x0 - 4, y0 - ww // 2 - 4, x0 + ww + 4, y0 + ww // 2 + 4), fill=frame + (255,))
        d.rectangle((x0 - 4, y0 - 4, x0 + ww + 4, y1 + 4), fill=frame + (255,))
        g = glass(ww, y1 - y0)
        if arched:
            mask = Image.new("L", g.size, 255); ImageDraw.Draw(mask).rectangle((0, 0, ww, 0), fill=255)
            im.paste(glass(ww, ww // 2), (x0, y0 - ww // 2), Image.new("L", (ww, ww // 2), 0))
            ImageDraw.Draw(im).pieslice((x0, y0 - ww // 2, x0 + ww, y0 + ww // 2), 180, 360, fill=(80, 92, 104, 255))
        im.paste(g, (x0, y0))
        d.line([(x0, (y0 + y1) // 2), (x0 + ww, (y0 + y1) // 2)], fill=frame + (255,), width=4)           # meeting rail
        d.rectangle((x0 - 8, y1 + 4, x0 + ww + 8, y1 + 10), fill=_dk(frame) + (255,))                      # sill
    return im


def _dk(c, k=0.8):
    return tuple(int(v * k) for v in c)


def bay_doors(wall, frame=(150, 40, 32), size=(192, 192), arch=True):
    """firehouse apparatus door (arched head, panelled red doors)"""
    w, h = size; im = wall.resize(size).copy(); d = ImageDraw.Draw(im)
    x0, x1, y0 = int(w * 0.1), int(w * 0.9), int(h * 0.12)
    if arch:
        d.pieslice((x0 - 6, y0 - 6, x1 + 6, y0 + (x1 - x0) // 2 + 6), 180, 360, fill=(208, 200, 186, 255))
        d.pieslice((x0, y0, x1, y0 + (x1 - x0) // 2), 180, 360, fill=(70, 80, 90, 255))
    yd = y0 + (x1 - x0) // 4
    d.rectangle((x0, yd, x1, h), fill=frame + (255,))
    for k in range(4):
        for j in range(3):
            px0 = x0 + 8 + k * (x1 - x0 - 16) // 4; py0 = yd + 10 + j * (h - yd - 14) // 3
            d.rectangle((px0, py0, px0 + (x1 - x0 - 16) // 4 - 8, py0 + (h - yd - 14) // 3 - 8), outline=_dk(frame, 0.6) + (255,), width=3)
    d.line([((x0 + x1) // 2, yd), ((x0 + x1) // 2, h)], fill=(30, 20, 18, 255), width=3)
    return im


def text_board(text, bg=(236, 230, 214), fg=(34, 34, 34), w=512, h=96, border=True):
    """plain business-name board (generic font, no logo)"""
    im = Image.new("RGBA", (w, h), bg + (255,)); d = ImageDraw.Draw(im)
    if border:
        d.rectangle((3, 3, w - 4, h - 4), outline=fg + (255,), width=3)
    sz = h * 0.62
    while sz > 8 and d.textlength(text, font=F(sz)) > w * 0.9:
        sz -= 1
    d.text((w / 2, h / 2), text, font=F(sz), fill=fg + (255,), anchor="mm")
    return im


def awning(rgb, w=128, h=64, stripes=True):
    im = Image.new("RGBA", (w, h), rgb + (255,)); d = ImageDraw.Draw(im)
    if stripes:
        for x in range(0, w, 16):
            d.rectangle((x, 0, x + 7, h), fill=(240, 236, 226, 255))
    for x in range(0, w, 16):                       # scalloped valance
        d.pieslice((x, h - 16, x + 16, h + 8), 0, 180, fill=_dk(rgb, 0.7) + (255,))
    return im


def roof_flat():
    return _shade(cc0("Concrete018", 128, tint=(92, 90, 86)), 1.0)


def shingles(rgb=(86, 78, 70)):
    im = Image.new("RGBA", (128, 128), rgb + (255,)); d = ImageDraw.Draw(im); rng = np.random.default_rng(5)
    for r in range(0, 128, 10):
        off = 8 if (r // 10) % 2 else 0
        for x in range(-16 + off, 128, 16):
            k = rng.uniform(0.8, 1.15); d.rectangle((x, r, x + 15, r + 9), fill=tuple(int(c * k) for c in rgb) + (255,), outline=_dk(rgb, 0.6) + (255,))
    return im


# ------------------------------------------------------------------ rail
def ties_strip():
    """ballast with ties, top view, 1 tile = 2.4 m of track (4 ties) x 3.6 m wide"""
    w, h = 256, 160
    im = _shade(cc0("Rock030", (w, h), tint=(128, 120, 110)), 0.95); d = ImageDraw.Draw(im)
    for k in range(4):
        x = int((k + 0.5) * w / 4)
        d.rectangle((x - 9, 18, x + 9, h - 18), fill=(70, 58, 46, 255), outline=(48, 40, 32, 255))
    for y in (int(h * 0.34), int(h * 0.66)):
        d.rectangle((0, y - 3, w, y + 3), fill=(110, 84, 62, 255)); d.line([(0, y), (w, y)], fill=(170, 160, 150, 255), width=2)
    return im


def coach_side(rgb=(46, 82, 56), w=512, h=128, freight=False):
    if freight:
        im = cc0("PaintedMetal004", (w, h), tint=rgb); d = ImageDraw.Draw(im)
        for x in range(8, w, 20):
            d.line([(x, 0), (x, h)], fill=_dk(rgb, 0.6) + (255,), width=3)
        d.rectangle((int(w * 0.42), 10, int(w * 0.58), h - 6), outline=(40, 30, 26, 255), width=4)      # sliding door
        return im
    im = cc0("PaintedMetal004", (w, h), tint=rgb); d = ImageDraw.Draw(im)
    d.rectangle((0, int(h * 0.82), w, h), fill=_dk(rgb, 0.55) + (255,))
    for k in range(12):
        x0 = int(w * 0.07 + k * w * 0.072); d.rectangle((x0, int(h * 0.22), x0 + int(w * 0.05), int(h * 0.58)), fill=(186, 178, 150, 255))
        im.paste(glass(int(w * 0.05) - 6, int(h * 0.36) - 6), (x0 + 3, int(h * 0.22) + 3))
    for x0 in (int(w * 0.005), int(w * 0.95)):
        d.rectangle((x0, int(h * 0.12), x0 + int(w * 0.04), int(h * 0.9)), fill=_dk(rgb, 0.75) + (255,))        # vestibule door
    return im


# ------------------------------------------------------------------ signals / beacons
def lens(rgb, lit=1.0, size=64):
    im = Image.new("RGBA", (size, size), (16, 16, 16, 255)); d = ImageDraw.Draw(im)
    c = tuple(int(v * (0.45 + 0.55 * lit)) for v in rgb)
    d.ellipse((4, 4, size - 5, size - 5), fill=c + (255,))
    d.ellipse((size * 0.22, size * 0.18, size * 0.5, size * 0.42), fill=tuple(min(255, int(v * 1.3) + 40) for v in c) + (255,))
    return im


def stripes(c1, c2, w=256, h=32, n=8, diag=True):
    im = Image.new("RGBA", (w, h), c1 + (255,)); d = ImageDraw.Draw(im); s = w // n
    for k in range(-1, n + 1):
        x = k * s
        d.polygon([(x, h), (x + s // 2, h), (x + s // 2 + (h if diag else 0), 0), (x + (h if diag else 0), 0)], fill=c2 + (255,))
    return im


def mesh_alpha(rgb=(236, 110, 24), w=128, h=64, cell=10, bar=3):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for x in range(0, w, cell):
        d.line([(x, 0), (x, h)], fill=rgb + (255,), width=bar)
    for y in range(0, h, cell // 2 + 2):
        d.line([(0, y), (w, y)], fill=rgb + (255,), width=bar - 1)
    return im


def pickets(w=128, h=64):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for x in range(4, w, 16):
        d.polygon([(x, h), (x + 9, h), (x + 9, 8), (x + 4, 1), (x, 8)], fill=(238, 238, 232, 255))
    for y in (int(h * 0.3), int(h * 0.75)):
        d.rectangle((0, y, w, y + 5), fill=(222, 222, 216, 255))
    return im


def chainlink(w=128, h=128):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for k in range(-h, w, 12):
        d.line([(k, 0), (k + h, h)], fill=(150, 154, 156, 255), width=2); d.line([(k + h, 0), (k, h)], fill=(150, 154, 156, 255), width=2)
    return im


def palm_card(w=128, h=256):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im); rng = np.random.default_rng(11)
    d.polygon([(w * 0.44, h), (w * 0.56, h), (w * 0.54, h * 0.28), (w * 0.46, h * 0.28)], fill=(118, 100, 80, 255))
    for y in range(int(h * 0.3), h, 7):
        d.line([(w * 0.44, y), (w * 0.56, y + 2)], fill=(92, 78, 62, 255), width=2)
    cx, cy = w / 2, h * 0.26
    for k in range(22):
        a = math.pi * (0.05 + 0.9 * k / 21) + rng.normal(0, 0.06); L = w * rng.uniform(0.42, 0.52)
        droop = rng.uniform(0.3, 0.7)
        pts = [(cx + L * t * math.cos(a), cy - L * t * math.sin(a) + droop * L * t * t) for t in np.linspace(0, 1, 8)]
        col = (int(rng.uniform(50, 80)), int(rng.uniform(96, 128)), int(rng.uniform(40, 60)), 255)
        d.line(pts, fill=col, width=6)
        for (x, y) in pts[2:]:
            d.line([(x, y), (x + rng.normal(0, 4), y + 8)], fill=col, width=2)
    return im


def crown_card(rgb=(82, 116, 60), w=192, h=192, seed=4):
    """generic broadleaf street-tree crown + trunk (plane tree)"""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im); rng = np.random.default_rng(seed)
    d.polygon([(w * 0.46, h), (w * 0.54, h), (w * 0.53, h * 0.45), (w * 0.47, h * 0.45)], fill=(150, 140, 120, 255))
    for _ in range(140):
        x, y = rng.normal(w / 2, w * 0.2), rng.normal(h * 0.38, h * 0.16); r = rng.uniform(5, 12); k = rng.uniform(0.7, 1.2)
        if abs(x - w / 2) < w * 0.47 and 4 < y < h * 0.72:
            d.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(min(255, c * k)) for c in rgb) + (255,))
    return im


# ------------------------------------------------------------------ polyhaven glTF loader (CC0 models)
def load_gltf(name, skip=("rust", "aged", "cap_0", "chain")):
    """-> list of (P (n,3) z-up metres, N, UV, T, diffuse PIL) for the clean variant meshes"""
    d = PH / name; g = json.loads((d / f"{name}.gltf").read_text())
    buf = (d / g["buffers"][0]["uri"]).read_bytes()
    CT = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}; NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    def acc(i):
        a = g["accessors"][i]; bv = g["bufferViews"][a["bufferView"]]; o = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        n = NC[a["type"]]; dt = np.dtype(CT[a["componentType"]])
        stride = bv.get("byteStride", n * dt.itemsize)
        raw = np.frombuffer(buf, np.uint8, count=stride * (a["count"] - 1) + n * dt.itemsize, offset=o)
        if stride == n * dt.itemsize:
            return raw.view(dt).reshape(a["count"], n)
        return np.stack([np.frombuffer(raw[k * stride:k * stride + n * dt.itemsize].tobytes(), dt) for k in range(a["count"])])
    def nmat(nd):
        if "matrix" in nd:
            return np.array(nd["matrix"]).reshape(4, 4).T
        M = np.eye(4); t = nd.get("translation", [0, 0, 0]); r = nd.get("rotation", [0, 0, 0, 1]); s = nd.get("scale", [1, 1, 1])
        x, y, z, w = r
        Rm = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        M[:3, :3] = Rm * np.array(s)[None, :]; M[:3, 3] = t; return M
    out = []
    def walk(ni, Mp):
        nd = g["nodes"][ni]; M = Mp @ nmat(nd)
        if "mesh" in nd and not any(s in (nd.get("name") or "") for s in skip):
            for pr in g["meshes"][nd["mesh"]]["primitives"]:
                at = pr["attributes"]; P = acc(at["POSITION"]).astype(float); N = acc(at["NORMAL"]).astype(float); UV = acc(at["TEXCOORD_0"]).astype(float)
                T = acc(pr["indices"]).reshape(-1, 3).astype(np.int64)
                P = (np.c_[P, np.ones(len(P))] @ M.T)[:, :3]; N = N @ M[:3, :3].T; N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
                mt = g["materials"][pr["material"]]; ti = mt["pbrMetallicRoughness"]["baseColorTexture"]["index"]
                tex = Image.open(d / g["images"][g["textures"][ti]["source"]]["uri"]).convert("RGBA")
                zup = lambda A: np.c_[A[:, 0], -A[:, 2], A[:, 1]]
                out.append((zup(P), zup(N), UV, T, tex))
        for c in nd.get("children", []):
            walk(c, M)
    for ni in g["scenes"][g.get("scene", 0)]["nodes"]:
        walk(ni, np.eye(4))
    return out


def decimate(P, N, UV, T, cell):
    """vertex clustering in (position, uv) space; keeps UV seams (cluster key includes the quantised uv)"""
    if cell <= 0:
        return P, N, UV, T
    key = np.c_[np.floor(P / cell).astype(np.int64), np.floor(UV / 0.02).astype(np.int64)]
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True); inv = inv.ravel()
    k = len(cnt)
    Pn = np.zeros((k, 3)); Nn = np.zeros((k, 3)); Un = np.zeros((k, 2))
    np.add.at(Pn, inv, P); np.add.at(Nn, inv, N); np.add.at(Un, inv, UV)
    Pn /= cnt[:, None]; Un /= cnt[:, None]; Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12
    T2 = inv[T]; ok = (T2[:, 0] != T2[:, 1]) & (T2[:, 1] != T2[:, 2]) & (T2[:, 0] != T2[:, 2])
    T2 = np.unique(np.sort(T2[ok], axis=1), axis=0, return_index=True)[1]
    T2 = inv[T][ok][np.sort(T2)]
    return Pn, Nn, Un, T2


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch(); print("fetched CC0 sources into", CC0, PH)
