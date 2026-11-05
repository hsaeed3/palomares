"""Standalone three.js viewer of the corridor -> reports/viewer.html
Terrain layers:
  1. backdrop  : 3DEP 10 m DEM resampled to 30 m, bbox + 2 km, draped with 10 m USGS imagery
  2. corridor  : 1 m LiDAR DTM resampled to 3 m within 150 m, vertices within (edge + 9 m) removed
  3. skirt     : road-aligned strip from the exact road edge out to ~11.5 m, 2 m along, sampled from the 0.5 m DTM,
                 so terrain meets the road edge with no gap, overlap or clipping
Corridor/skirt are draped with 2 m USGS imagery (public domain) modulated by CC0 detail textures (ambientCG):
dry grass on open ground, rock on steep faces, leaf litter under canopy. Trees: LiDAR positions/heights/crowns,
crown colour sampled from the imagery at the tree."""
import base64, csv, io, json, numpy as np, rasterio
from PIL import Image, ImageOps, ImageEnhance
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from common import *

o = origin()
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
def jpg_uri(img, q=82):
    bio = io.BytesIO(); img.save(bio, "JPEG", quality=q); return "data:image/jpeg;base64," + base64.b64encode(bio.getvalue()).decode()

CL = list(csv.DictReader(open(DER / "centerline.csv")))
col = lambda k: np.array([float(r[k]) for r in CL])
S, X, Y, Z = col("s_m"), col("x_local"), col("y_local"), col("z_navd88")
WL, WR, CFL, CFR, G, K = col("edge_left_m"), col("edge_right_m"), col("crossfall_left"), col("crossfall_right"), col("grade"), col("curvature_1pm")
T = np.gradient(np.c_[X, Y], axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
Nn = np.c_[-T[:, 1], T[:, 0]]                      # left normal
ctree = cKDTree(np.c_[X, Y])

def raster(name):
    with rasterio.open(DER / name) as r: return r.read(), r.transform
def sampler(A, tf):
    def f(x, y, order=1):
        c = (x + o["x0"] - tf.c) / tf.a - .5; r = (y + o["y0"] - tf.f) / tf.e - .5
        return map_coordinates(A, [r, c], order=order, mode="nearest")
    return f

dtm05, tf05 = raster("dtm_r050_b40.tif"); dtm05 = dtm05[0]
dtm1, tf1 = raster("dtm_r100_b150.tif"); dtm1 = dtm1[0]
z05 = sampler(np.nan_to_num(dtm05, nan=np.nanmean(dtm05)), tf05); valid05 = sampler(np.isfinite(dtm05).astype(np.float32), tf05)
z1 = sampler(np.nan_to_num(dtm1, nan=np.nanmean(dtm1)), tf1)

# ---------- skirt ----------
OFFS = np.array([0, .5, 1, 1.5, 2, 3, 4, 5.5, 7, 9, 11.5])
k = np.arange(0, len(S), 2)
R_ = 1 / np.maximum(np.abs(np.convolve(K, np.ones(9) / 9, "same")), 1e-4)
skirt = {}
for side, W, CF, nm in ((1, WL, CFL, "L"), (-1, WR, CFR, "R")):
    inner = np.sign(np.convolve(K, np.ones(9) / 9, "same")) == side
    lim = np.where(inner, np.clip(0.8 * R_ - W, 3, 11.5), 11.5)
    off = np.minimum(OFFS[None, :], lim[k, None])
    d = side * (W[k, None] + off)                          # signed lateral offset of each skirt vertex
    px = X[k, None] + Nn[k, 0, None] * d; py = Y[k, None] + Nn[k, 1, None] * d
    edge_z = Z[k] + CF[k] * side * W[k]                    # matches the road ribbon's edge height exactly
    tz = np.where(valid05(px, py) > .5, z05(px, py), z1(px, py))
    tz[:, 0] = edge_z - 0.02
    # first 1 m: blend from road edge to terrain to avoid a vertical lip at shoulders
    w = np.clip(off / 1.5, 0, 1); tz = edge_z[:, None] * (1 - w) + tz * w
    tz[:, 0] = edge_z - 0.02
    skirt[nm] = {"x": b64(px.astype(np.float32)), "y": b64(py.astype(np.float32)), "z": b64(tz.astype(np.float32))}
skirt["nc"] = len(OFFS); skirt["nr"] = int(len(k))

# ---------- corridor 3 m grid ----------
ST = 3
A = dtm1[::ST, ::ST]; rr, cc = np.nonzero(np.isfinite(A))
gx = tf1.c - o["x0"] + (cc + .5) * ST; gy = tf1.f - o["y0"] - (rr + .5) * ST
dist, j = ctree.query(np.c_[gx, gy])
lat = (np.c_[gx - X[j], gy - Y[j]] * Nn[j]).sum(1)
edge = np.where(lat > 0, WL[j], WR[j])
keep = np.abs(lat) > edge + 9.0
zz = A[rr, cc] - np.where(np.abs(lat) < edge + 14, 0.25, 0.0)
rr, cc, zz = rr[keep], cc[keep], zz[keep]
terrain = {"nx": int(A.shape[1]), "ny": int(A.shape[0]), "x0": tf1.c - o["x0"], "y0": tf1.f - o["y0"], "step": float(ST),
           "r": b64(rr.astype(np.uint16)), "c": b64(cc.astype(np.uint16)), "z": b64(np.round(zz * 10).astype(np.int16))}

# ---------- backdrop 30 m ----------
bd, tfb = raster("backdrop_dem_r10.tif"); bd = bd[0][::3, ::3]
br, bc = np.nonzero(np.isfinite(bd))
bx = tfb.c - o["x0"] + (bc + .5) * 30; by = tfb.f - o["y0"] - (br + .5) * 30
bdist, _ = ctree.query(np.c_[bx, by])
kb = bdist > 105                                            # corridor mesh (to 150 m) replaces the backdrop here
br, bc, bdist = br[kb], bc[kb], bdist[kb]
bz = bd[br, bc] - np.clip((230 - bdist) / 125, 0, 1) * 10.0  # tuck the backdrop edge well under the corridor mesh
backdrop = {"nx": int(bd.shape[1]), "ny": int(bd.shape[0]), "x0": tfb.c - o["x0"], "y0": tfb.f - o["y0"], "step": 30.0,
            "r": b64(br.astype(np.uint16)), "c": b64(bc.astype(np.uint16)), "z": b64(np.round(bz * 10).astype(np.int16))}

# ---------- imagery ----------
orth, tfo = raster("ortho_r100.tif")
TS = 2048
mask1 = np.isfinite(dtm1)
tiles = []
for r0 in range(0, orth.shape[1], TS):
    for c0 in range(0, orth.shape[2], TS):
        m = mask1[r0:r0 + TS, c0:c0 + TS]
        if not m.any(): continue
        a_ = orth[:, r0:r0 + TS, c0:c0 + TS].transpose(1, 2, 0).copy()
        a_[~m] = a_[m].mean(0).astype(np.uint8)        # flat fill outside corridor compresses to ~nothing
        h_, w_ = m.shape
        pad = np.zeros((TS, TS, 3), np.uint8); pad[:h_, :w_] = a_
        if h_ < TS: pad[h_:, :w_] = a_[-1:]
        if w_ < TS: pad[:, w_:] = pad[:, w_ - 1:w_]
        tiles.append({"uri": jpg_uri(Image.fromarray(pad), 80), "x0": tfo.c - o["x0"] + c0, "y1": tfo.f - o["y0"] - r0, "w": float(TS), "h": float(TS)})
ortho = {"tiles": tiles, "x0": tfo.c - o["x0"], "y1": tfo.f - o["y0"], "ts": float(TS), "ncol": int(np.ceil(orth.shape[2] / TS)),
         "index": {f"{int((t['y1'] - (tfo.f - o['y0'])) / -TS)}_{int((t['x0'] - (tfo.c - o['x0'])) / TS)}": i for i, t in enumerate(tiles)}}
bo, tfbo = raster("backdrop_ortho_r05.tif")
bimg = Image.fromarray(bo.transpose(1, 2, 0)).resize((bo.shape[2] // 2, bo.shape[1] // 2), Image.LANCZOS)
bortho = {"uri": jpg_uri(bimg, 78), "x0": tfbo.c - o["x0"], "y1": tfbo.f - o["y0"], "w": bo.shape[2] * tfbo.a, "h": bo.shape[1] * -tfbo.e}

# ---------- CC0 detail textures (ambientCG) ----------
TEX = RAW / "tex"
def tex(name, size=1024, gray=False, contrast=1.0):
    im = Image.open(TEX / name / f"{name}_1K-JPG_Color.jpg").convert("RGB").resize((size, size), Image.LANCZOS)
    if gray:
        im = ImageOps.grayscale(im); im = ImageEnhance.Contrast(im).enhance(contrast)
        a = np.array(im).astype(float); a = a / a.mean() * 128; im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB")
    return jpg_uri(im, 80)
textures = {"asphalt": tex("Asphalt031"), "grass": tex("Grass004", gray=True, contrast=1.2), "rock": tex("Rock051", gray=True),
            "litter": tex("Ground037", gray=True), "bark": tex("Bark014", 256), "leaf": tex("ScatteredLeaves009", 256, gray=True, contrast=1.4)}

def png_uri(im):
    bio = io.BytesIO(); im.save(bio, "PNG", optimize=True); return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode()
rng = np.random.default_rng(7)
def leaf_card(src, n=260, size=512, leaf_px=(34, 60), elong=1.0):
    col = Image.open(TEX / src / f"{src}_1K-PNG_Color.png").convert("RGB"); op = Image.open(TEX / src / f"{src}_1K-PNG_Opacity.png").convert("L")
    W_, H_ = col.size; cell = W_ // 3                    # LeafSet024 is a 3x3 leaf atlas
    leaves = [(col.crop((i*cell, j*cell, (i+1)*cell, (j+1)*cell)), op.crop((i*cell, j*cell, (i+1)*cell, (j+1)*cell))) for i in range(3) for j in range(3)]
    card = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for _ in range(n):
        r = np.sqrt(rng.random()) * size * 0.42; t = rng.random() * 2 * np.pi
        cx, cy = size / 2 + r * np.cos(t), size / 2 + r * np.sin(t) * 0.9
        c_, o_ = leaves[rng.integers(len(leaves))]; px = int(rng.uniform(*leaf_px))
        c_ = c_.resize((px, int(px * elong))); o_ = o_.resize((px, int(px * elong)))
        ang = rng.uniform(0, 360); c_ = c_.rotate(ang, expand=True); o_ = o_.rotate(ang, expand=True)
        shade = rng.uniform(0.55, 1.0); c_ = Image.eval(c_, lambda v: int(v * shade))
        rgba = c_.convert("RGBA"); rgba.putalpha(o_)
        card.alpha_composite(rgba, (int(cx - rgba.size[0] / 2), int(cy - rgba.size[1] / 2)))
    a_ = np.array(card).astype(float); g = a_[..., :3].mean(2, keepdims=True)
    a_[..., :3] = np.clip(g / max(g[a_[..., 3] > 100].mean(), 1) * 205, 0, 255)                # greyscale: colour comes from the imagery per tree
    # thin branch lines under the leaves
    return png_uri(Image.fromarray(a_.astype(np.uint8)))
def grass_card(srcs=("Foliage002", "Foliage001"), size=256):
    card = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for k in range(22):
        src = srcs[k % len(srcs)]
        col = Image.open(TEX / src / f"{src}_1K-PNG_Color.png").convert("RGB"); op = Image.open(TEX / src / f"{src}_1K-PNG_Opacity.png").convert("L")
        n = 4 if src == "Foliage002" else 9; cw = col.size[0] // n; i = rng.integers(n)
        c_ = col.crop((i*cw, 0, (i+1)*cw, col.size[1])); o_ = op.crop((i*cw, 0, (i+1)*cw, col.size[1]))
        hgt = int(rng.uniform(.55, 1.0) * size); wdt = max(4, int(cw * hgt / col.size[1]))
        c_ = c_.resize((wdt, hgt)); o_ = o_.resize((wdt, hgt))
        ang = rng.uniform(-14, 14); c_ = c_.rotate(ang, expand=True, center=(wdt/2, hgt)); o_ = o_.rotate(ang, expand=True, center=(wdt/2, hgt))
        rgba = c_.convert("RGBA"); rgba.putalpha(o_)
        x = int(rng.uniform(.1, .9) * size - rgba.size[0] / 2); card.alpha_composite(rgba, (max(0, x), size - rgba.size[1]))
    a_ = np.array(card).astype(float); g = a_[..., :3].mean(2, keepdims=True)
    a_[..., :3] = np.clip(g / max(g[a_[..., 3] > 100].mean(), 1) * 215, 0, 255)
    return png_uri(Image.fromarray(a_.astype(np.uint8)))
textures["leafcard"] = leaf_card("LeafSet024")
textures["grasscard"] = grass_card()

# ---------- roadside grass tufts on open ground (not under canopy, not on steep faces) ----------
chm, tfc = raster("chm_r100_b150.tif"); chm = np.nan_to_num(chm[0])
tuft = []
for side, W in ((1, WL), (-1, WR)):
    for i in range(0, len(S), 1):
        for off in rng.uniform(0.6, 7.0, 2):
            d = side * (W[i] + off); x = X[i] + Nn[i, 0] * d + rng.normal(0, .3); y = Y[i] + Nn[i, 1] * d + rng.normal(0, .3)
            tuft.append((x, y))
tuft = np.array(tuft)
ch = sampler(chm, tfc)(tuft[:, 0], tuft[:, 1], 0)
tz = np.where(valid05(tuft[:, 0], tuft[:, 1]) > .5, z05(tuft[:, 0], tuft[:, 1]), z1(tuft[:, 0], tuft[:, 1]))
e = 1.0; gxz = (z05(tuft[:, 0] + e, tuft[:, 1]) - z05(tuft[:, 0] - e, tuft[:, 1])) / 2; gyz = (z05(tuft[:, 0], tuft[:, 1] + e) - z05(tuft[:, 0], tuft[:, 1] - e)) / 2
okg = (ch < 2.0) & (np.hypot(gxz, gyz) < 0.8)
tuft, tz = tuft[okg], tz[okg]
tc = np.stack([map_coordinates(orth[c].astype(np.float32), [((tuft[:, 1] + o["y0"]) - tfo.f) / tfo.e - .5, ((tuft[:, 0] + o["x0"]) - tfo.c) / tfo.a - .5], order=1) for c in range(3)], 1)
grass = {"x": b64(tuft[:, 0].astype(np.float32)), "y": b64(tuft[:, 1].astype(np.float32)), "z": b64(tz.astype(np.float32)),
         "rgb": b64(np.clip(tc * 1.1, 0, 255).astype(np.uint8)), "n": int(len(tuft))}

# ---------- road ----------
road = {"x": b64(X[k].astype(np.float32)), "y": b64(Y[k].astype(np.float32)), "z": b64(Z[k].astype(np.float32)),
        "wl": b64(WL[k].astype(np.float32)), "wr": b64(WR[k].astype(np.float32)), "cfl": b64(CFL[k].astype(np.float32)),
        "cfr": b64(CFR[k].astype(np.float32)), "s": b64(S[k].astype(np.float32)), "g": b64(G[k].astype(np.float32))}

# ---------- trees (colour from imagery) ----------
Tr = np.loadtxt(DER / "trees.csv", delimiter=",", skiprows=1)
so = sampler(orth.astype(np.float32), tfo)
rgb = np.stack([np.mean([map_coordinates(orth[c].astype(np.float32), [((Tr[:, 4] + dy + o["y0"]) - tfo.f) / tfo.e - .5, ((Tr[:, 3] + dx + o["x0"]) - tfo.c) / tfo.a - .5], order=1)
                         for dx in (-1, 0, 1) for dy in (-1, 0, 1)], 0) for c in range(3)], 1)
# imagery crowns are sunlit from above; keep hue, normalise brightness to a plausible foliage range
lum = rgb.mean(1, keepdims=True); rgb = rgb / np.maximum(lum, 1) * np.clip(lum, 55, 120)
trees = {"x": b64(Tr[:, 3].astype(np.float32)), "y": b64(Tr[:, 4].astype(np.float32)), "z": b64(Tr[:, 5].astype(np.float32)),
         "h": b64(Tr[:, 6].astype(np.float32)), "cr": b64(Tr[:, 7].astype(np.float32)), "rgb": b64(np.clip(rgb, 0, 255).astype(np.uint8)), "n": int(len(Tr))}

F = json.load(open(DER / "features.geojson"))
feats = []
for f in F["features"]:
    p = f["properties"]; g = f["geometry"]
    if g["type"] == "Point": feats.append({"k": p["kind"], "s0": p["s_m"], "s1": p["s_m"], "side": p.get("side", ""), "off": p.get("offset_m", 0)})
    else: feats.append({"k": p["kind"], "s0": p["s_start_m"], "s1": p["s_end_m"], "side": p.get("side", "")})
si = int(np.argmax(Z))
meta = {"length": float(S[-1]), "summit_s": float(S[si]), "summit_z": float(Z[si]), "origin": o}
data = {"TERRAIN": terrain, "BACKDROP": backdrop, "SKIRT": skirt, "ORTHO": ortho, "BORTHO": bortho, "TEX": textures,
        "ROAD": road, "TREES": trees, "GRASS": grass, "FEATS": feats, "META": meta}
html = open(ROOT / "scripts" / "viewer_template.html").read()
html = html.replace("/*DATA*/", "".join(f"const {k}={json.dumps(v)};\n" for k, v in data.items()))
(REPORTS / "viewer.html").write_text(html)
print(f"viewer {len(html)/1e6:.1f} MB; corridor verts {len(rr)}, skirt verts {2*len(k)*len(OFFS)}, backdrop {len(br)}, trees {len(Tr)}, grass tufts {len(tuft)}, ortho tiles {len(tiles)}")
