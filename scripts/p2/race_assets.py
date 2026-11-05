"""Phase 2: realistic hill-climb event assets for the track_* layouts (called from 21b_track_dressing.py).

Adds meshes (material "race_assets", one RGBA atlas build/race_assets_atlas.png) to the shared "dressing" group,
merged into one visual mesh per 1 km of road (draw calls ~ +16), plus ONE collision mesh 1WALL_race (collide.py
closed boxes) for every solid prop within RA_COLL_DIST of the road edge.

Sources (see docs/phase2/CREDITS.md):
  * Poly Haven CC0 glTF models (api.polyhaven.com / dl.polyhaven.org, cached in data/raw/race_assets/polyhaven/):
    fire extinguisher, portable generator, oil drum, jerrycan, picnic table, gas bottle. Only the 1k diffuse maps
    are used, packed into the atlas (512 px tiles); UVs are all inside 0..1 so they remap 1:1.
  * OSM (ODbL) side-road geometry from api.openstreetmap.org (data/raw/race_assets/osm/side_ways.json) to know
    which side of the road each junction / driveway leaves and in which direction.
  * Everything else is modelled here at real dimensions (numpy): modular aluminium grandstand (stepped treads,
    seat planks, scaffold sub-frame, stairs, handrails, cantilever roof with fascia, back cladding), catch fence,
    steel crowd-control barriers, pop-up gazebos (scissor frame, hip roof, printed valance), portable toilets,
    enclosed race-car trailers, big square straw bales, traffic cones, Type III ROAD CLOSED barricades,
    scaffold TV camera tower. Textures are procedural (PIL); branding is fictional (same PHC mark as 21b).

Placement (all on DTM-checked flat-ish ground, clear of the road shoulder, trees and roadside buildings):
  s=20 (SR-84 start for nb): paddock (3 gazebos incl. TIMING, 2 trailers, toilets, generator, drums, picnic
      table) + covered grandstand behind catch fence
  s=7392 (summit finish): covered grandstand + catch fence (replaces 21b's box bleacher), TV tower, gazebo, toilets
  s=15563 (Castro Valley start for sb): small paddock (gazebo, trailer, toilets)
  marshal posts (from 21b): 2 extinguishers + a straw-bale wall between road and post
  3 spectator areas on the inside of hard corners: crowd-barrier line, shade gazebo, sign
  side roads / driveways (OSM): ROAD CLOSED barricade across junction mouths, cone pairs at driveways
"""
import io, json, math, pathlib, sys, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import collide as CO
from geom import ROOT, DER, BUILD, vertex_normals

RAW = ROOT / "data/raw/race_assets"
REPLACES_BLEACHER = True        # 21b skips its box bleacher; the summit gets the modelled grandstand instead
RA_COLL_DIST = 8.0              # solid props whose footprint is within this of the road edge get a 1WALL box
PH_MODELS = {                   # Poly Haven id -> (atlas tile name per material, real height m for sanity)
    "korean_fire_extinguisher_01": None, "portable_generator": None, "Barrel_01": None,
    "metal_jerrycan": None, "wooden_picnic_table": None, "propane_tank": None,
}

# ====================================================================== atlas
S = 2048; T5 = 512; T2 = 256
ATL = Image.new("RGBA", (S, S), (128, 128, 128, 255)); dr = ImageDraw.Draw(ATL)
UV = {}
NAVY, RED, WHITE, GOLD = (16, 30, 66), (200, 32, 36), (245, 245, 242), (232, 178, 48)


def F(sz, bold=True):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", sz)
    except OSError:
        return ImageFont.load_default()


def reg(name, x0, y0, x1, y1, pad=4):
    UV[name] = ((x0 + pad) / S, (y0 + pad) / S, (x1 - pad) / S, (y1 - pad) / S)


def grain(box, amt=8, seed=0, blur=0):
    a = np.asarray(ATL.crop(box), float)
    n = np.random.default_rng(seed).normal(0, amt, a.shape[:2])
    if blur:
        from scipy.ndimage import gaussian_filter
        n = gaussian_filter(n, blur) * (1 + blur)
    a[..., :3] = (a[..., :3] + n[..., None]).clip(0, 255)
    ATL.paste(Image.fromarray(a.astype(np.uint8), "RGBA"), box[:2])


def t2(c, r):                 # 256 tile in rows 4..7 (y >= 1024)
    return (c * T2, 1024 + (r - 4) * T2, (c + 1) * T2, 1024 + (r - 3) * T2)


def logo(cx, cy, r, fg, bg):
    dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fg)
    dr.ellipse((cx - r * .82, cy - r * .82, cx + r * .82, cy + r * .82), fill=bg)
    pts = [(cx - r * .7, cy + r * .3), (cx - r * .3, cy - r * .35), (cx, cy + r * .05), (cx + r * .3, cy - r * .5), (cx + r * .7, cy + r * .3)]
    dr.polygon(pts + [(cx + r * .7, cy + r * .45), (cx - r * .7, cy + r * .45)], fill=fg)


# --- rows 0-1: Poly Haven diffuse maps (512 tiles), filled by load_models()
PH_SLOTS = []                 # free 512 slots
for r in range(2):
    for c in range(4):
        PH_SLOTS.append((c * T5, r * T5))
SWATCH = []                   # 64 px flat-colour swatches for tiny sub-materials (glass, labels)

# --- row 4
b = t2(0, 4); dr.rectangle(b, fill=(178, 182, 186)); grain(b, 7, 1)          # brushed aluminium (streaks along u)
a = np.asarray(ATL.crop(b), float); a[..., :3] += np.random.default_rng(2).normal(0, 6, (1, 256, 1)); ATL.paste(Image.fromarray(a.clip(0, 255).astype(np.uint8)), b[:2])
reg("alu", *b)
b = t2(1, 4); dr.rectangle(b, fill=(140, 144, 146)); grain(b, 12, 3, blur=2)  # galvanised steel (spangle)
reg("galv", *b)
b = t2(2, 4); dr.rectangle(b, fill=(150, 154, 158))                           # seat plank: extruded alu, ribs along u
for k in range(0, 256, 16): dr.rectangle((b[0], b[1] + k, b[2], b[1] + k + 3), fill=(118, 122, 126))
grain(b, 5, 4); reg("seat", *b)
b = t2(3, 4); dr.rectangle(b, fill=(112, 114, 116))                           # deck tread: anti-slip alu plank
for k in range(0, 256, 8):
    for j in range(0, 256, 8): dr.point((b[0] + j + (k // 8) % 2 * 4, b[1] + k), fill=(160, 162, 164))
for k in range(0, 256, 64): dr.rectangle((b[0], b[1] + k, b[2], b[1] + k + 2), fill=(70, 72, 74))
grain(b, 5, 5); reg("tread", *b)
for ci, (nm, col) in enumerate((("canopy_white", (236, 236, 232)), ("canopy_red", (186, 28, 32)), ("canopy_navy", (24, 40, 84)))):
    b = t2(4 + ci, 4); dr.rectangle(b, fill=col)
    for k in range(0, 256, 85): dr.line((b[0] + k, b[1], b[0] + k, b[3]), fill=tuple(int(c * .9) for c in col), width=2)   # seams
    grain(b, 4, 6 + ci, blur=3); reg(nm, *b)
b = t2(7, 4); dr.rectangle(b, fill=(34, 34, 36)); grain(b, 6, 9); reg("rubber", *b)

# --- row 5: valances (printed gazebo skirts), banner, scrim, mesh fences
def valance(c, r, txt, bg=NAVY, fg=WHITE):
    b = t2(c, r); dr.rectangle(b, fill=bg); dr.rectangle((b[0], b[1] + 200, b[2], b[3]), fill=RED)
    logo(b[0] + 40, b[1] + 100, 30, fg, bg)
    dr.text((b[0] + 148, b[1] + 100), txt, fill=fg, font=F(34 if len(txt) < 8 else 26), anchor="mm")
    grain(b, 3, 10 + c); reg("val_" + txt.lower(), *b)
valance(0, 5, "TIMING"); valance(1, 5, "PADDOCK"); valance(2, 5, "OFFICIALS")
b = (3 * T2, 1280, 5 * T2, 1536)                                               # grandstand fascia (512 x 256)
dr.rectangle(b, fill=NAVY); dr.rectangle((b[0], b[1], b[2], b[1] + 18), fill=RED); dr.rectangle((b[0], b[3] - 18, b[2], b[3]), fill=RED)
logo(b[0] + 64, b[1] + 128, 48, WHITE, NAVY)
dr.text((b[0] + 290, b[1] + 104), "PALOMARES", fill=WHITE, font=F(64), anchor="mm")
dr.text((b[0] + 290, b[1] + 170), "HILL CLIMB  GRANDSTAND", fill=GOLD, font=F(28), anchor="mm")
reg("fascia", *b)
b = t2(5, 5); dr.rectangle(b, fill=(22, 34, 70))                               # back cladding scrim (woven)
for k in range(0, 256, 4): dr.line((b[0], b[1] + k, b[2], b[1] + k), fill=(28, 42, 82))
logo(b[0] + 128, b[1] + 118, 70, (200, 205, 220), (22, 34, 70)); grain(b, 4, 15); reg("scrim", *b)
# crowd-control barrier infill (alpha): vertical pickets on a 2.3 x 1.0 panel -> 256 px wide
b = t2(6, 5); arr = np.zeros((256, 256, 4), np.uint8)
for xb in np.linspace(6, 250, 22).astype(int): arr[:, xb - 2:xb + 2] = (168, 172, 176, 255)
ATL.paste(Image.fromarray(arr, "RGBA"), b[:2]); reg("pickets", *b, pad=1)
b = t2(7, 5); arr = np.zeros((256, 256, 4), np.uint8)                          # catch fence diamond mesh (alpha), 1 x 1 m
yy, xx = np.mgrid[0:256, 0:256]
mesh = (np.abs(((xx + yy) % 32) - 16) < 1.7) | (np.abs(((xx - yy) % 32) - 16) < 1.7)
arr[mesh] = (172, 176, 180, 255); ATL.paste(Image.fromarray(arr, "RGBA"), b[:2]); reg("mesh", *b, pad=1)

# --- row 6: straw, toilet, trailer, cone, barricade
rs = np.random.default_rng(31)
for ci, nm in enumerate(("straw_side", "straw_end")):
    b = t2(ci, 6); dr.rectangle(b, fill=(196, 168, 102))
    for _ in range(2600):                                                     # straw stalks: along u (side) / random (end)
        x = b[0] + rs.integers(0, 256); y = b[1] + rs.integers(0, 256)
        ang = rs.normal(0, 0.15) if ci == 0 else rs.uniform(0, np.pi)
        L = rs.uniform(8, 30) if ci == 0 else rs.uniform(3, 9)
        col = tuple(int(v) for v in np.array((214, 186, 118)) * rs.uniform(0.72, 1.12))
        dr.line((x, y, x + L * np.cos(ang), y + L * np.sin(ang)), fill=col, width=1)
    if ci == 0:
        for yb in (80, 176): dr.rectangle((b[0], b[1] + yb, b[2], b[1] + yb + 3), fill=(220, 120, 40))   # orange twine
    grain(b, 6, 40 + ci); reg(nm, *b)
b = t2(2, 6); dr.rectangle(b, fill=(30, 92, 168))                              # portable toilet wall: moulded ribs
for k in range(0, 256, 32): dr.rectangle((b[0] + k, b[1], b[0] + k + 6, b[3]), fill=(24, 78, 148))
grain(b, 4, 42); reg("loo_wall", *b)
b = t2(3, 6); dr.rectangle(b, fill=(30, 92, 168))                              # toilet door
dr.rectangle((b[0] + 28, b[1] + 10, b[2] - 28, b[3] - 4), fill=(34, 100, 178), outline=(18, 60, 120), width=4)
for k in range(6): dr.rectangle((b[0] + 70, b[1] + 22 + k * 9, b[0] + 186, b[1] + 26 + k * 9), fill=(14, 40, 86))   # vent
dr.rectangle((b[0] + 180, b[1] + 120, b[0] + 214, b[1] + 136), fill=(40, 170, 60))                                  # VACANT tab
dr.rectangle((b[0] + 182, b[1] + 146, b[0] + 192, b[1] + 176), fill=(200, 200, 204))                                # handle
grain(b, 3, 43); reg("loo_door", *b)
b = t2(4, 6); dr.rectangle(b, fill=(232, 232, 228))                            # trailer side: white skin, rivet lines, stripe
for k in range(0, 256, 42): dr.line((b[0] + k, b[1], b[0] + k, b[3]), fill=(206, 206, 204), width=2)
dr.rectangle((b[0], b[1] + 150, b[2], b[1] + 172), fill=NAVY); dr.rectangle((b[0], b[1] + 176, b[2], b[1] + 182), fill=RED)
dr.rectangle((b[0], b[3] - 14, b[2], b[3]), fill=(60, 62, 64)); grain(b, 3, 44); reg("trailer", *b)
b = t2(5, 6); dr.rectangle(b, fill=(226, 226, 222))                            # trailer rear ramp door
dr.rectangle((b[0] + 10, b[1] + 10, b[2] - 10, b[3] - 16), outline=(120, 122, 124), width=5)
for xh in (40, 216): dr.rectangle((b[0] + xh - 6, b[1] + 60, b[0] + xh + 6, b[1] + 200), fill=(150, 152, 154))
dr.rectangle((b[0] + 30, b[3] - 40, b[0] + 60, b[3] - 22), fill=(200, 30, 20)); dr.rectangle((b[2] - 60, b[3] - 40, b[2] - 30, b[3] - 22), fill=(200, 30, 20))
grain(b, 3, 45); reg("trailer_door", *b)
b = t2(6, 6); dr.rectangle(b, fill=(238, 90, 24))                              # traffic cone: u around, v top->bottom
for y0_, y1_ in ((60, 92), (128, 152)): dr.rectangle((b[0], b[1] + y0_, b[2], b[1] + y1_), fill=(236, 236, 236))
grain(b, 4, 46); reg("cone", *b)
b = t2(7, 6); _st = Image.new("RGBA", (256, 256), WHITE); _sd = ImageDraw.Draw(_st)      # Type III rail stripes (u = along)
for k in range(-256, 512, 64): _sd.polygon([(k, 256), (k + 32, 256), (k + 32 + 90, 0), (k + 90, 0)], fill=(240, 96, 20))
ATL.paste(_st, b[:2]); grain(b, 3, 47); reg("stripes", *b)

# --- row 7: signs, ply, camera, misc + swatches
def sign(c, txt, sub, bg, fg, w=1):
    b = (c * T2, 1792, (c + w) * T2, 2048); dr.rectangle(b, fill=bg)
    dr.rectangle((b[0] + 6, b[1] + 6, b[2] - 7, b[3] - 7), outline=fg, width=6)
    dr.text(((b[0] + b[2]) / 2, b[1] + 100), txt, fill=fg, font=F(54 if w > 1 else 36), anchor="mm")
    if sub: dr.text(((b[0] + b[2]) / 2, b[1] + 172), sub, fill=fg, font=F(24 if w > 1 else 18, False), anchor="mm")
    reg("sign_" + txt.split()[0].lower(), *b)
sign(0, "ROAD CLOSED", "EVENT IN PROGRESS", (246, 246, 240), (12, 12, 12), w=2)
sign(2, "SPECTATORS", "STAY BEHIND BARRIER", (250, 200, 0), (12, 12, 12))
sign(3, "PADDOCK", "COMPETITORS ONLY", (20, 120, 50), WHITE)
b = t2(4, 7); dr.rectangle(b, fill=(176, 140, 92))                             # plywood deck
for k in range(0, 256, 6): dr.line((b[0], b[1] + k + rs.integers(-2, 3), b[2], b[1] + k), fill=(160, 124, 80))
grain(b, 6, 50); reg("ply", *b)
b = t2(5, 7); dr.rectangle(b, fill=(28, 28, 30))                               # broadcast camera body
dr.rectangle((b[0] + 20, b[1] + 30, b[0] + 120, b[1] + 70), fill=(60, 62, 66)); dr.rectangle((b[0] + 150, b[1] + 40, b[0] + 230, b[1] + 60), fill=(200, 30, 20))
grain(b, 3, 51); reg("camera", *b)
b = t2(6, 7); dr.rectangle(b, fill=(244, 244, 240)); grain(b, 3, 52); reg("white", *b)
b = t2(7, 7)                                                                   # 16 x 64 px swatches (flat colours)
for k in range(16):
    SWATCH.append((b[0] + (k % 4) * 64, b[1] + (k // 4) * 64))


def swatch(name, rgb):
    x, y = SWATCH.pop(0); dr.rectangle((x, y, x + 63, y + 63), fill=tuple(int(v) for v in rgb)); reg(name, x, y, x + 64, y + 64, pad=12)
    return name


swatch("sw_dark", (40, 42, 44)); swatch("sw_yellow", (238, 196, 20)); swatch("sw_glass", (50, 60, 70)); swatch("sw_lens", (14, 14, 16))
swatch("sw_orange", (230, 100, 20)); swatch("sw_red", (190, 30, 30))


# ====================================================================== glTF (minimal loader: no trimesh in this env)
_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
PH_URL = "https://api.polyhaven.com/files/{}"
MODELS = {}


def _fetch_ph(aid):
    d = RAW / "polyhaven" / aid
    g = d / f"{aid}_1k.gltf"
    if g.exists():
        return g
    d.mkdir(parents=True, exist_ok=True)
    meta = json.load(urllib.request.urlopen(PH_URL.format(aid), timeout=60))["gltf"]["1k"]["gltf"]
    (d / g.name).write_bytes(urllib.request.urlopen(meta["url"], timeout=60).read())
    for rel, v in meta["include"].items():
        if rel.endswith(".bin") or "_diff_" in rel:
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(urllib.request.urlopen(v["url"], timeout=120).read())
    return g


def _quat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_model(aid):
    """-> dict(P, N, UV(atlas), T) in ENU metres (glTF +Y up -> +Z up), origin at the model's floor centre"""
    if aid in MODELS:
        return MODELS[aid]
    path = _fetch_ph(aid); g = json.load(open(path))
    bufs = [open(path.parent / b["uri"], "rb").read() for b in g["buffers"]]

    def acc(i):
        a = g["accessors"][i]; bv = g["bufferViews"][a["bufferView"]]
        n = _NC[a["type"]]; dt = np.dtype(_CT[a["componentType"]])
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0); st = bv.get("byteStride", 0) or n * dt.itemsize
        raw = np.frombuffer(bufs[bv["buffer"]], np.uint8, count=st * (a["count"] - 1) + n * dt.itemsize, offset=off)
        return np.lib.stride_tricks.as_strided(raw, (a["count"], n * dt.itemsize), (st, 1)).copy().view(dt).reshape(a["count"], n)

    tile_of = {}
    for mi, m in enumerate(g["materials"]):
        tex = m.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if tex is None:
            tile_of[mi] = swatch(f"sw_{aid}_{mi}", (128, 128, 128)); continue
        uri = g["images"][g["textures"][tex["index"]]["source"]]["uri"]
        im = Image.open(path.parent / uri).convert("RGB")
        nm = f"ph_{aid}_{mi}"
        used = sum(1 for p in (pp for mm in g["meshes"] for pp in mm["primitives"]) if p.get("material") == mi)
        nv = sum(g["accessors"][p["attributes"]["POSITION"]]["count"] for mm in g["meshes"] for p in mm["primitives"] if p.get("material") == mi)
        if nv < 400 or not PH_SLOTS:                     # tiny sub-material (glass, label): flat mean colour
            tile_of[mi] = swatch(nm, np.asarray(im.resize((8, 8)), float).reshape(-1, 3).mean(0))
        else:
            x0, y0 = PH_SLOTS.pop(0)
            ATL.paste(im.resize((T5, T5), Image.LANCZOS).convert("RGBA"), (x0, y0)); reg(nm, x0, y0, x0 + T5, y0 + T5, pad=0)
            # pad=0: models' UV islands reach the tile edge; bleeding is limited by keeping UVs inside 0..1
            tile_of[mi] = nm

    def node_mats(ni, M):
        nd = g["nodes"][ni]
        L = np.eye(4)
        if "matrix" in nd:
            L = np.array(nd["matrix"]).reshape(4, 4).T
        else:
            R = _quat(nd.get("rotation", [0, 0, 0, 1])) * np.array(nd.get("scale", [1, 1, 1]))
            L[:3, :3] = R; L[:3, 3] = nd.get("translation", [0, 0, 0])
        W = M @ L
        yield nd, W
        for c in nd.get("children", []):
            yield from node_mats(c, W)

    roots = g["scenes"][g.get("scene", 0)]["nodes"]
    P_, N_, U_, T_ = [], [], [], []; n = 0
    for r in roots:
        for nd, W in node_mats(r, np.eye(4)):
            if "mesh" not in nd:
                continue
            for p in g["meshes"][nd["mesh"]]["primitives"]:
                P = acc(p["attributes"]["POSITION"]).astype(float); Nn = acc(p["attributes"]["NORMAL"]).astype(float)
                uv = acc(p["attributes"]["TEXCOORD_0"]).astype(float).clip(0, 1)
                idx = acc(p["indices"]).reshape(-1, 3).astype(np.int64)
                P = P @ W[:3, :3].T + W[:3, 3]; Nn = Nn @ W[:3, :3].T
                u0, v0, u1, v1 = UV[tile_of[p.get("material", 0)]]
                P_.append(P); N_.append(Nn); U_.append(np.c_[u0 + uv[:, 0] * (u1 - u0), v0 + uv[:, 1] * (v1 - v0)])
                T_.append(idx + n); n += len(P)
    P = np.concatenate(P_); Nn = np.concatenate(N_)
    C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)          # glTF (x, y-up, z) -> ENU (x, -z, y)
    P = P @ C.T; Nn = Nn @ C.T; Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12
    c = (P.min(0) + P.max(0)) / 2; P -= [c[0], c[1], P[:, 2].min()]
    MODELS[aid] = dict(P=P, N=Nn, UV=np.concatenate(U_), T=np.concatenate(T_))
    return MODELS[aid]


# ====================================================================== mesh building (local prop frame: x along, y away from road, z up)
class MB:
    def __init__(self):
        self.P, self.N, self.U, self.T, self.n = [], [], [], [], 0

    def add(self, P, T, U, N=None):
        P = np.asarray(P, float); T = np.asarray(T, np.int64).reshape(-1, 3)
        if N is None:
            N = vertex_normals(P, T)
        self.P.append(P); self.T.append(T + self.n); self.U.append(np.asarray(U, float)); self.N.append(np.asarray(N, float)); self.n += len(P)

    def extend(self, other, origin=(0, 0, 0), xaxis=(1, 0), scale=1.0):
        if not other.P:
            return
        P, T, U, N = other.arrays()
        x = np.array([xaxis[0], xaxis[1], 0.0]); x /= np.linalg.norm(x); y = np.array([-x[1], x[0], 0.0]); z = np.array([0, 0, 1.0])
        R = np.c_[x, y, z]
        self.add(P * scale @ R.T + np.asarray(origin, float), T, U, N @ R.T)

    def arrays(self):
        return np.concatenate(self.P), np.concatenate(self.T), np.concatenate(self.U), np.concatenate(self.N)

    def quad(self, a, b, c, d, tile, out=None, uv=((0, 1), (1, 1), (1, 0), (0, 0)), two=False):
        """a b c d = BL BR TR TL seen from the front (local uv v=0 at the top)"""
        Q = np.array([a, b, c, d], float); uvq = np.array(uv, float)
        if out is not None:
            nrm = np.cross(Q[1] - Q[0], Q[3] - Q[0])
            if np.dot(nrm, out) < 0:
                Q = Q[[1, 0, 3, 2]]; uvq = uvq[[1, 0, 3, 2]]
        u0, v0, u1, v1 = UV[tile]
        U = np.c_[u0 + uvq[:, 0] * (u1 - u0), v0 + uvq[:, 1] * (v1 - v0)]
        nrm = np.cross(Q[1] - Q[0], Q[3] - Q[0]); nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
        self.add(Q, [[0, 1, 2], [0, 2, 3]], U, np.tile(nrm, (4, 1)))
        if two:
            self.add(Q[[1, 0, 3, 2]], [[0, 1, 2], [0, 2, 3]], U[[1, 0, 3, 2]], np.tile(-nrm, (4, 1)))

    def box(self, x0, x1, y0, y1, z0, z1, tile, faces=None, bottom=False, uvs=None):
        """axis-aligned box in the local frame. faces: {'-y','+y','-x','+x','top'}: tile name"""
        f = faces or {}
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        def Q(key, pts, o):
            self.quad(*pts, f.get(key, tile), out=np.array(o, float))
        Q("-y", [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0))
        Q("+y", [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)], (0, 1, 0))
        Q("-x", [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)], (-1, 0, 0))
        Q("+x", [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0))
        Q("top", [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1))
        if bottom:
            Q("bot", [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)], (0, 0, -1))

    def cyl(self, p0, p1, r, tile, n=6, r1=None, caps=False, vrange=(0, 1)):
        """tube from p0 to p1 (radius r -> r1). u around, v along (v=vrange[0] at p1 = top)"""
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); ax = p1 - p0; L = np.linalg.norm(ax)
        if L < 1e-6:
            return
        ax /= L; ref = np.array([0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(ax, ref); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
        r1 = r if r1 is None else r1
        a = np.linspace(0, 2 * np.pi, n + 1)
        ring = np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2
        P = np.concatenate([p0 + ring * r, p1 + ring * r1])
        Nr = np.concatenate([ring, ring]) + ax * (r - r1) / L
        u0, v0, u1, v1 = UV[tile]
        uu = np.r_[np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1)]
        vv = np.r_[np.full(n + 1, vrange[1]), np.full(n + 1, vrange[0])]
        U = np.c_[u0 + uu * (u1 - u0), v0 + vv * (v1 - v0)]
        k = np.arange(n); m = n + 1
        T = np.concatenate([np.c_[k, k + 1, k + 1 + m], np.c_[k, k + 1 + m, k + m]])
        # winding: outward normal = cross((k+1)-k, (k+m)-k) should align with ring
        tst = np.cross(P[1] - P[0], P[m] - P[0])
        if np.dot(tst, ring[0] + ring[1]) < 0:
            T = T[:, [0, 2, 1]]
        self.add(P, T, U, Nr / np.linalg.norm(Nr, axis=1, keepdims=True))
        if caps:
            for pc, rr, sg in ((p1, r1, 1), (p0, r, -1)):
                if rr <= 0:
                    continue
                Pc = np.r_[[pc], pc + ring[:-1] * rr]
                Tc = np.c_[np.zeros(n, int), 1 + k, 1 + (k + 1) % n]
                if np.dot(np.cross(Pc[Tc[0, 1]] - Pc[0], Pc[Tc[0, 2]] - Pc[0]), ax * sg) < 0:
                    Tc = Tc[:, [0, 2, 1]]
                cu = (u0 + u1) / 2; cv = v0 + 0.1 * (v1 - v0)
                self.add(Pc, Tc, np.tile([cu, cv], (n + 1, 1)), np.tile(ax * sg, (n + 1, 1)))

    def model(self, aid, origin=(0, 0, 0), yaw=0.0, scale=1.0):
        m = load_model(aid); c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        self.add(m["P"] * scale @ R.T + np.asarray(origin, float), m["T"], m["UV"], m["N"] @ R.T)


# ====================================================================== props (local frame, ground at z=0 unless gl() is given)
def grandstand(nrows=7, L=14.0, gl=lambda x, y: 0.0, roof=True):
    """modular aluminium grandstand: front edge y=0 (faces -y), rows rise along +y. gl(x, y): local ground z.
    Returns (MB, footprint (x0, x1, y0, y1), zbase)."""
    m = MB(); d, rise, h0 = 0.80, 0.40, 0.55
    xs = np.arange(-L / 2, L / 2 + 0.01, 2.0)
    H = [h0 + k * rise for k in range(nrows)]
    ytop = nrows * d; htop = H[-1]
    for k in range(nrows):
        y0, y1, hk = k * d, (k + 1) * d, H[k]
        m.box(-L / 2, L / 2, y0, y1, hk - 0.06, hk, "tread")                                  # deck plank
        if k > 0:
            m.quad((-L / 2, y0, H[k - 1]), (L / 2, y0, H[k - 1]), (L / 2, y0, hk - 0.06), (-L / 2, y0, hk - 0.06), "galv", out=(0, -1, 0))  # riser
        m.box(-L / 2 + 0.05, L / 2 - 0.05, y1 - 0.38, y1 - 0.08, hk + 0.40, hk + 0.45, "seat")  # bench seat
        for x in np.arange(-L / 2 + 0.6, L / 2, 1.6):
            m.box(x - 0.02, x + 0.02, y1 - 0.26, y1 - 0.2, hk, hk + 0.40, "alu")               # seat brackets
    for x in xs:                                                                              # scaffold sub-frame
        for k in range(nrows + 1):
            y = min(k * d, ytop) - (0.04 if k == nrows else -0.04)
            hz = H[min(k, nrows - 1)] - 0.06
            m.cyl((x, y, gl(x, y) - 0.15), (x, y, hz), 0.035, "galv", n=6)
            m.box(x - 0.1, x + 0.1, y - 0.1, y + 0.1, gl(x, y) - 0.15, gl(x, y) - 0.1, "galv")  # base plate
        m.cyl((x, 0.04, h0 - 0.1), (x, ytop - 0.04, htop - 0.1), 0.05, "galv", n=6)           # raker beam
        m.cyl((x, 0.04, gl(x, 0.04) + 0.2), (x, ytop - 0.04, htop - 0.3), 0.025, "galv", n=5)  # diagonal brace
    for x0_, x1_ in zip(xs[:-1], xs[1:]):                                                     # plan braces (X) in the back bay
        yb = ytop - 0.04; zb = gl((x0_ + x1_) / 2, yb)
        m.cyl((x0_, yb, zb + 0.2), (x1_, yb, htop - 0.4), 0.022, "galv", n=5)
        m.cyl((x1_, yb, zb + 0.2), (x0_, yb, htop - 0.4), 0.022, "galv", n=5)
    zf = min(gl(x, 0) for x in xs)                                                            # front skirt scrim (road-facing)
    m.quad((-L / 2, -0.02, zf - 0.1), (L / 2, -0.02, zf - 0.1), (L / 2, -0.02, h0 - 0.06), (-L / 2, -0.02, h0 - 0.06), "scrim", out=(0, -1, 0))
    zb = min(gl(x, ytop) for x in xs)                                                         # back cladding to rail height
    for x0_, x1_ in zip(xs[:-1], xs[1:]):
        m.quad((x1_, ytop + 0.02, zb - 0.1), (x0_, ytop + 0.02, zb - 0.1), (x0_, ytop + 0.02, htop + 1.1), (x1_, ytop + 0.02, htop + 1.1),
               "scrim", out=(0, 1, 0), two=True)
    for x in xs:                                                                              # rear handrail posts
        m.cyl((x, ytop - 0.05, htop), (x, ytop - 0.05, htop + 1.1), 0.025, "alu", n=5)
    for zr in (0.55, 1.1):
        m.cyl((-L / 2, ytop - 0.05, htop + zr), (L / 2, ytop - 0.05, htop + zr), 0.025, "alu", n=6)
    # front rail (row 0 edge) and side rails along the rake
    for x in np.arange(-L / 2, L / 2 + 0.01, 2.0):
        m.cyl((x, 0.05, h0), (x, 0.05, h0 + 1.0), 0.022, "alu", n=5)
    m.cyl((-L / 2, 0.05, h0 + 1.0), (L / 2, 0.05, h0 + 1.0), 0.025, "alu", n=6)
    m.cyl((-L / 2, 0.05, h0 + 0.5), (L / 2, 0.05, h0 + 0.5), 0.02, "alu", n=6)
    for xe in (-L / 2 + 0.03, L / 2 - 0.03):
        for k in range(nrows):
            m.cyl((xe, k * d + 0.4, H[k]), (xe, k * d + 0.4, H[k] + 1.0), 0.022, "alu", n=5)
        m.cyl((xe, 0.05, h0 + 1.0), (xe, ytop - 0.05, htop + 1.1), 0.025, "alu", n=6)
        # end cheek panel (sloped triangle-ish) of scrim below the treads
        zg = gl(xe, ytop / 2)
        m.quad((xe, 0.0, zg - 0.1), (xe, ytop, zg - 0.1), (xe, ytop, htop - 0.06), (xe, 0.0, h0 - 0.06), "scrim", out=(np.sign(xe), 0, 0), two=True)
    # access stairs at the +x end, 1.1 m wide, rising toward +y (half-row steps)
    sx0, sx1 = L / 2 + 0.05, L / 2 + 1.15
    nst = int(round((htop - 0.06) / 0.2)); run = 0.27
    ys0 = ytop - nst * run
    for k in range(nst):
        z = (k + 1) * (htop - 0.06) / nst; y = ys0 + k * run
        m.box(sx0, sx1, y, y + run + 0.02, z - 0.05, z, "tread")
    zs = gl((sx0 + sx1) / 2, ys0)
    for xx in (sx0, sx1):
        m.cyl((xx, ys0, zs), (xx, ytop, htop - 0.08), 0.05, "galv", n=6)                      # stringers
        m.cyl((xx, ys0 + 0.2, zs + 1.0), (xx, ytop, htop + 1.0), 0.022, "alu", n=5)            # handrails
        for t in np.linspace(0.15, 0.95, 4):
            yy_ = ys0 + t * (ytop - ys0); zz_ = zs + t * (htop - zs)
            m.cyl((xx, yy_, gl(xx, yy_) - 0.1), (xx, yy_, zz_), 0.03, "galv", n=5)
            m.cyl((xx, yy_, zz_), (xx, yy_, zz_ + 1.0), 0.02, "alu", n=5)
    m.box(sx0, sx1, ytop - 0.02, ytop + 0.9, htop - 0.06, htop, "tread")                       # landing
    if roof:                                                                                  # cantilever roof on rear masts
        zr_b, zr_f = htop + 2.7, htop + 2.45; yf = -0.6
        for x in np.arange(-L / 2, L / 2 + 0.01, 3.5):
            m.cyl((x, ytop + 0.12, gl(x, ytop + 0.12) - 0.1), (x, ytop + 0.12, zr_b + 0.1), 0.08, "galv", n=8)
            m.box(x - 0.06, x + 0.06, yf, ytop + 0.2, zr_f - 0.25, zr_b - 0.02, "galv")         # roof beam (rough wedge as box)
            m.cyl((x, ytop + 0.12, zr_b + 0.08), (x, yf + 0.4, zr_f + 0.05), 0.02, "galv", n=4)  # tie rod
        top = [(-L / 2 - 0.3, yf, zr_f), (L / 2 + 0.3, yf, zr_f), (L / 2 + 0.3, ytop + 0.3, zr_b), (-L / 2 - 0.3, ytop + 0.3, zr_b)]
        m.quad(top[0], top[1], top[2], top[3], "canopy_white", out=(0, 0, 1))
        m.quad(top[1], top[0], top[3], top[2], "canopy_navy", out=(0, 0, -1))
        nf = max(1, int(round(L / 6)))                                                        # fascia banners
        xs_f = np.linspace(-L / 2 - 0.3, L / 2 + 0.3, nf + 1)
        for a_, b_ in zip(xs_f[:-1], xs_f[1:]):
            m.quad((a_, yf, zr_f - 0.75), (b_, yf, zr_f - 0.75), (b_, yf, zr_f + 0.05), (a_, yf, zr_f + 0.05), "fascia", out=(0, -1, 0))
            m.quad((b_, yf + 0.01, zr_f - 0.75), (a_, yf + 0.01, zr_f - 0.75), (a_, yf + 0.01, zr_f + 0.05), (b_, yf + 0.01, zr_f + 0.05), "canopy_navy", out=(0, 1, 0))
    return m, (-L / 2, sx1, -0.6, ytop + 0.4), htop


def gazebo(L=3.0, W=3.0, canopy="canopy_white", val="val_paddock", back_wall=True, gl=lambda x, y: 0.0, table=True):
    """pop-up gazebo, centred, eaves 2.05 m, peak +0.85 m"""
    m = MB(); he = 2.05; hp = he + (0.85 if L <= 3.2 else 0.95)
    xs = [-L / 2, L / 2] if L <= 3.2 else [-L / 2, 0.0, L / 2]
    legs = [(x, y) for x in xs for y in (-W / 2, W / 2)]
    for x, y in legs:
        z0 = gl(x, y)
        m.box(x - 0.022, x + 0.022, y - 0.022, y + 0.022, z0 - 0.05, he, "alu")
        m.box(x - 0.14, x + 0.14, y - 0.14, y + 0.14, z0 - 0.02, z0 + 0.12, "sw_dark")        # leg weight
    for (xa, ya), (xb, yb) in [((xs[0], -W / 2), (xs[-1], -W / 2)), ((xs[0], W / 2), (xs[-1], W / 2)),
                               ((xs[0], -W / 2), (xs[0], W / 2)), ((xs[-1], -W / 2), (xs[-1], W / 2))]:
        n = max(2, int(round(max(abs(xb - xa), abs(yb - ya)) / 1.0)))                         # scissor truss
        for t0, t1 in zip(np.linspace(0, 1, n + 1)[:-1], np.linspace(0, 1, n + 1)[1:]):
            pa = np.array([xa + (xb - xa) * t0, ya + (yb - ya) * t0]); pb = np.array([xa + (xb - xa) * t1, ya + (yb - ya) * t1])
            m.cyl((*pa, he), (*pb, he - 0.3), 0.012, "alu", n=4); m.cyl((*pa, he - 0.3), (*pb, he), 0.012, "alu", n=4)
    # roof: hip (ridge along x for L > W) or pyramid
    rx = max(0.0, (L - W) / 2)
    c = [(-L / 2, -W / 2, he), (L / 2, -W / 2, he), (L / 2, W / 2, he), (-L / 2, W / 2, he)]
    r0, r1 = (-rx, 0, hp), (rx, 0, hp)
    def tri(a, b, cc):
        for P3, o in ((np.array([a, b, cc], float), 1), (np.array([b, a, cc], float), -1)):
            nrm = np.cross(P3[1] - P3[0], P3[2] - P3[0]); nrm /= np.linalg.norm(nrm) + 1e-12
            if (nrm[2] > 0) != (o > 0):
                P3 = P3[[1, 0, 2]]; nrm = -nrm
            u0, v0, u1, v1 = UV[canopy]
            m.add(P3, [[0, 1, 2]], [[u0, v1], [u1, v1], [(u0 + u1) / 2, v0]], np.tile(nrm, (3, 1)))
    if rx > 0:
        for a, b, p_, q_ in ((c[0], c[1], r1, r0), (c[2], c[3], r0, r1)):
            m.quad(a, b, p_, q_, canopy, out=(0, 0, 1)); m.quad(b, a, q_, p_, canopy, out=(0, 0, -1))
        tri(c[1], c[2], r1); tri(c[3], c[0], r0)
    else:
        for a, b in zip(c, c[1:] + c[:1]):
            tri(a, b, r0)
    # valance (printed skirt) all round, outward print + plain inside
    for (xa, ya), (xb, yb), o in (((-L / 2, -W / 2), (L / 2, -W / 2), (0, -1, 0)), ((L / 2, W / 2), (-L / 2, W / 2), (0, 1, 0)),
                                  ((L / 2, -W / 2), (L / 2, W / 2), (1, 0, 0)), ((-L / 2, W / 2), (-L / 2, -W / 2), (-1, 0, 0))):
        nseg = max(1, int(round(max(abs(xb - xa), abs(yb - ya)) / 3.0)))
        for t0, t1 in zip(np.linspace(0, 1, nseg + 1)[:-1], np.linspace(0, 1, nseg + 1)[1:]):
            a = (xa + (xb - xa) * t0, ya + (yb - ya) * t0); b_ = (xa + (xb - xa) * t1, ya + (yb - ya) * t1)
            m.quad((*a, he - 0.27), (*b_, he - 0.27), (*b_, he + 0.01), (*a, he + 0.01), val, out=o)
            m.quad((*b_, he - 0.27), (*a, he - 0.27), (*a, he + 0.01), (*b_, he + 0.01), canopy, out=tuple(-v for v in o))
    if back_wall:
        zb = min(gl(x, W / 2) for x in xs)
        m.quad((L / 2, W / 2, zb - 0.05), (-L / 2, W / 2, zb - 0.05), (-L / 2, W / 2, he - 0.27), (L / 2, W / 2, he - 0.27), canopy, out=(0, 1, 0), two=True)
    if table:                                                                                 # folding table inside
        zt = gl(0, 0.3)
        m.box(-0.9, 0.9, 0.0, 0.75, zt + 0.70, zt + 0.74, "white")
        for xx in (-0.85, 0.85):
            for yy in (0.05, 0.7):
                m.cyl((xx, yy, zt), (xx, yy, zt + 0.70), 0.015, "alu", n=4)
    return m


def crowd_barriers(pts, gz_):
    """steel pedestrian barriers (2.3 x 1.1 m panels) along a world polyline: frame tubes, picket infill, flat feet"""
    m = MB(); pts = np.asarray(pts, float)
    L = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    nseg = max(1, int(L[-1] // 2.32))
    for k in range(nseg):
        a = np.array([np.interp(k * 2.32 + 0.02, L, pts[:, i]) for i in range(2)])
        b = np.array([np.interp(k * 2.32 + 2.30, L, pts[:, i]) for i in range(2)])
        za, zb = gz_(*a), gz_(*b)
        A0, B0 = np.r_[a, za + 0.08], np.r_[b, zb + 0.08]; A1, B1 = np.r_[a, za + 1.1], np.r_[b, zb + 1.1]
        m.quad(A0 + [0, 0, 0.05], B0 + [0, 0, 0.05], B1 - [0, 0, 0.04], A1 - [0, 0, 0.04], "pickets", two=True)
        for p, q in ((A0, B0), (A1, B1), (A0, A1), (B0, B1)):
            m.cyl(p, q, 0.02, "galv", n=5)
        t = (b - a) / np.linalg.norm(b - a); nrm = np.array([-t[1], t[0]])
        for p, z in ((a + t * 0.1, za), (b - t * 0.1, zb)):                                # flat feet across the panel
            f0, f1 = p - nrm * 0.32, p + nrm * 0.32
            m.cyl(np.r_[f0, z + 0.03], np.r_[f1, z + 0.03], 0.025, "galv", n=4)
            m.cyl(np.r_[p, z + 0.03], np.r_[p, z + 0.1], 0.018, "galv", n=4)
    return m


def catch_fence(pts, gz_, h=3.5):
    """debris/catch fence: galvanised posts every <= 3 m, diamond mesh (alpha, both faces), top + mid cables"""
    m = MB(); pts = np.asarray(pts, float)
    L = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    ss = np.linspace(0, L[-1], max(2, int(np.ceil(L[-1] / 3.0)) + 1))
    P = np.c_[np.interp(ss, L, pts[:, 0]), np.interp(ss, L, pts[:, 1])]; Z = np.array([gz_(*p) for p in P])
    for p, z in zip(P, Z):
        m.cyl((*p, z - 0.3), (*p, z + h + 0.05), 0.045, "galv", n=6)
    for k in range(len(P) - 1):
        a, b = P[k], P[k + 1]; za, zb = Z[k], Z[k + 1]; w = np.hypot(*(b - a))
        for h0 in np.arange(0.0, h - 0.01, 1.0):
            h1 = min(h, h0 + 1.0)
            m.quad((*a, za + h0), (*b, zb + h0), (*b, zb + h1), (*a, za + h1), "mesh", two=True)
        for hc in (h, h / 2):
            m.cyl((*a, za + hc), (*b, zb + hc), 0.012, "galv", n=4)
    return m, P, Z


def portaloo(gl=lambda x, y: 0.0):
    m = MB(); z0 = min(gl(x, y) for x in (-0.58, 0.58) for y in (-0.58, 0.58))
    m.box(-0.6, 0.6, -0.6, 0.6, z0 - 0.1, z0 + 0.1, "sw_dark")                              # skid base
    m.box(-0.56, 0.56, -0.56, 0.56, z0 + 0.1, z0 + 2.2, "loo_wall", faces={"-y": "loo_door", "top": "white"})
    m.quad((-0.56, -0.56, z0 + 2.2), (0.56, -0.56, z0 + 2.2), (0.0, 0.0, z0 + 2.34), (0.0, 0.0, z0 + 2.34), "white", out=(0, -1, 1))
    m.quad((0.56, 0.56, z0 + 2.2), (-0.56, 0.56, z0 + 2.2), (0.0, 0.0, z0 + 2.34), (0.0, 0.0, z0 + 2.34), "white", out=(0, 1, 1))
    m.quad((0.56, -0.56, z0 + 2.2), (0.56, 0.56, z0 + 2.2), (0.0, 0.0, z0 + 2.34), (0.0, 0.0, z0 + 2.34), "white", out=(1, 0, 1))
    m.quad((-0.56, 0.56, z0 + 2.2), (-0.56, -0.56, z0 + 2.2), (0.0, 0.0, z0 + 2.34), (0.0, 0.0, z0 + 2.34), "white", out=(-1, 0, 1))
    m.cyl((0.4, 0.4, z0 + 2.2), (0.4, 0.4, z0 + 2.45), 0.04, "sw_dark", n=5)                 # vent stack
    return m


def trailer(gl=lambda x, y: 0.0):
    """enclosed single-car race trailer, 6.0 x 2.4 x 2.9 m; hitch toward +x, ramp door at -x"""
    m = MB(); z0 = min(gl(x, y) for x in (-3, 0, 3) for y in (-1.2, 1.2)); zb = z0 + 0.55
    m.box(-3.0, 3.0, -1.2, 1.2, zb, zb + 2.35, "trailer", faces={"-x": "trailer_door", "top": "white", "+x": "white"})
    m.box(-3.0, 3.0, -1.15, 1.15, zb - 0.12, zb, "sw_dark")                                  # chassis rails
    m.quad((3.0, -1.2, zb + 0.4), (3.0, 1.2, zb + 0.4), (3.6, 0.9, zb + 0.9), (3.6, -0.9, zb + 0.9), "white", out=(1, 0, 0.3))   # V-nose
    m.quad((3.0, 1.2, zb + 2.35), (3.0, -1.2, zb + 2.35), (3.6, -0.9, zb + 2.1), (3.6, 0.9, zb + 2.1), "white", out=(1, 0, 1))
    m.quad((3.6, -0.9, zb + 0.9), (3.6, 0.9, zb + 0.9), (3.6, 0.9, zb + 2.1), (3.6, -0.9, zb + 2.1), "white", out=(1, 0, 0))
    for sgn in (1, -1):
        m.quad((3.0, sgn * 1.2, zb + 0.4), (3.6, sgn * 0.9, zb + 0.9), (3.6, sgn * 0.9, zb + 2.1), (3.0, sgn * 1.2, zb + 2.35), "white", out=(0.3, sgn, 0))
        m.quad((3.0, sgn * 1.2, zb), (3.0, sgn * 1.2, zb + 0.4), (3.6, sgn * 0.9, zb + 0.9), (3.6, sgn * 0.9, zb + 0.9), "white", out=(0, sgn, 0))
    for xw in (-0.5, 0.5):                                                                    # tandem axle wheels + fenders
        for sgn in (1, -1):
            yw = sgn * 1.02
            m.cyl((xw, yw - 0.1 * sgn, z0 + 0.33), (xw, yw + 0.1 * sgn, z0 + 0.33), 0.33, "rubber", n=12, caps=True)
            m.cyl((xw, yw + 0.105 * sgn, z0 + 0.33), (xw, yw + 0.11 * sgn, z0 + 0.33), 0.18, "alu", n=10, caps=True)
    for sgn in (1, -1):
        m.box(-1.0, 1.0, min(sgn * 1.2, sgn * 1.32), max(sgn * 1.2, sgn * 1.32), z0 + 0.7, z0 + 0.74, "sw_dark")
        m.cyl((3.0, sgn * 0.9, zb - 0.08), (4.6, 0, z0 + 0.48), 0.04, "sw_dark", n=6)          # A-frame drawbar
    m.cyl((4.6, 0, z0 + 0.48), (4.75, 0, z0 + 0.48), 0.06, "galv", n=8, caps=True)             # coupling
    m.cyl((4.1, 0.15, z0), (4.1, 0.15, z0 + 0.6), 0.03, "galv", n=6)                           # jockey wheel post
    return m


def bale(m, cx, cy, yaw, z0, L=1.2, W=0.8, H=0.7):
    c, s = math.cos(yaw), math.sin(yaw)
    sub = MB(); sub.box(-L / 2, L / 2, -W / 2, W / 2, 0, H, "straw_side", faces={"-x": "straw_end", "+x": "straw_end", "top": "straw_side"})
    m.extend(sub, origin=(cx, cy, z0 - 0.04), xaxis=(c, s))


def cone(m, x, y, z):
    sub = MB(); sub.box(-0.19, 0.19, -0.19, 0.19, 0, 0.03, "sw_dark")
    sub.cyl((0, 0, 0.03), (0, 0, 0.72), 0.16, "cone", n=10, r1=0.025)
    m.extend(sub, origin=(x, y, z))


def barricade(L=2.4):
    """Type III barricade: 3 striped rails on two uprights with feet, ROAD CLOSED plate on top (x along, y thickness)"""
    m = MB()
    for zc in (0.45, 0.85, 1.25):
        m.box(-L / 2, L / 2, -0.06, -0.035, zc - 0.1, zc + 0.1, "stripes", faces={"+y": "stripes", "top": "white"})
    for x in (-L / 2 + 0.25, L / 2 - 0.25):
        m.box(x - 0.03, x + 0.03, -0.03, 0.03, 0.0, 1.45, "white")
        m.box(x - 0.04, x + 0.04, -0.45, 0.45, 0.0, 0.05, "sw_dark")                            # foot
    m.box(-0.75, 0.75, -0.08, -0.06, 1.42, 1.9, "sign_road", faces={"+y": "white"})
    return m


def tv_tower(gl=lambda x, y: 0.0, H=4.0):
    """2 x 2 m scaffold camera tower, platform at H, ladder on +x, camera on tripod facing -y"""
    m = MB(); c = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for x, y in c:
        m.cyl((x, y, gl(x, y) - 0.15), (x, y, H + 1.05), 0.03, "galv", n=6)
        m.box(x - 0.1, x + 0.1, y - 0.1, y + 0.1, gl(x, y) - 0.15, gl(x, y) - 0.05, "ply")
    for zl in np.arange(0.2, H + 1.06, 2.0).tolist() + [H, H + 0.5, H + 1.0]:
        for (xa, ya), (xb, yb) in zip(c, c[1:] + c[:1]):
            m.cyl((xa, ya, zl), (xb, yb, zl), 0.024, "galv", n=5)
    for (xa, ya), (xb, yb) in zip(c, c[1:] + c[:1]):
        m.cyl((xa, ya, 0.3), (xb, yb, H - 0.1), 0.02, "galv", n=4)
    m.box(-1.05, 1.05, -1.05, 1.05, H - 0.05, H, "ply")
    for yy in (-0.25, 0.25):                                                                  # ladder on the +x face
        m.cyl((1.12, yy, 0.0), (1.12, yy, H + 0.9), 0.02, "alu", n=4)
    for zr in np.arange(0.3, H, 0.3):
        m.cyl((1.12, -0.25, zr), (1.12, 0.25, zr), 0.014, "alu", n=4)
    for a in (0, 2.1, 4.2):                                                                   # tripod + camera
        m.cyl((0, 0, H + 1.05), (0.4 * math.cos(a), 0.4 * math.sin(a), H), 0.015, "sw_dark", n=4)
    m.box(-0.15, 0.15, -0.3, 0.2, H + 1.05, H + 1.35, "camera")
    m.cyl((0, -0.3, H + 1.2), (0, -0.62, H + 1.2), 0.07, "sw_lens", n=8, caps=True)
    m.box(-0.5, 0.5, -0.5, 0.5, H + 1.9, H + 1.94, "canopy_navy")                              # rain cover
    for x, y in ((-0.45, -0.45), (0.45, 0.45), (-0.45, 0.45), (0.45, -0.45)):
        m.cyl((x, y, H), (x, y, H + 1.9), 0.015, "alu", n=4)
    return m


# ====================================================================== placement
def build(ns):
    """called by 21b just before it writes geometry.pkl. ns = 21b's globals (road, helpers, USED, LOG, G)."""
    from scipy.spatial import cKDTree
    road, G, LOG, USED = ns["road"], ns["G"], ns["LOG"], ns["USED"]
    idx, tangent, normal, gz, footprint_ok = ns["idx"], ns["tangent"], ns["normal"], ns["gz"], ns["footprint_ok"]
    ctree = ns["ctree"]; GATES = ns["GATES"]; s_end = float(road.s[-1])
    RLOG = []
    # obstacles: LiDAR tree trunks + roadside buildings / walls / fences (21c, previous build) + 21b's USED discs
    tr = np.loadtxt(DER / "trees.csv", delimiter=",", skiprows=1, usecols=(3, 4))
    obst = [tr]
    rsj = BUILD / "roadside_solid.json"
    if rsj.exists():
        for it in json.load(open(rsj)):
            if it["type"] in ("building", "retaining_wall_stone", "retaining_wall_concrete", "retaining_wall_timber", "chainlink_fence",
                              "wood_board_fence", "solid_wood_privacy_fence", "gate_estate", "utility_pole", "mailbox_cluster", "road_sign"):
                obst.append(np.asarray(it["pts"], float)[:, :2])
    otree = cKDTree(np.concatenate(obst))
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    MINE, MINE_C = [], set()                      # my own footprints (shapely rectangles) / their USED discs

    def rect(x, y, t, L, W):
        f = np.asarray(t, float) / np.linalg.norm(t); l = np.array([-f[1], f[0]])
        return Polygon([np.array([x, y]) + a * L / 2 * f + b * W / 2 * l for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

    def road_gap(pts):
        d, j = ctree.query(np.atleast_2d(pts))
        return d - np.maximum(road.outer(j, 1), road.outer(j, -1))

    def spot(s0, s1, L, W, offs, sides=(1, -1), step=2.0, margin=1.5, maxrange=0.8, s_ref=None, wo=0.05, ws=0.01, clear=0.6, wz=0.0):
        best = None; s_ref = (s0 + s1) / 2 if s_ref is None else s_ref
        R = 0.5 * math.hypot(L, W)
        for s in np.arange(max(s0, 1.0), min(s1, s_end - 1.0) + 0.01, step):
            j = idx(s); t = tangent(j)
            for side in sides:
                for off in offs:
                    x, y = road.xy(j, side * (road.outer(j, side) + off + W / 2))
                    if any(math.hypot(x - a, y - b) < r + R * 0.8 for a, b, r in USED if (a, b, r) not in MINE_C):
                        continue
                    if MINE and rect(x, y, t, L + 2, W + 2).intersects(unary_union(MINE)):
                        continue
                    r = footprint_ok(x, y, t, L, W, margin, maxrange)
                    if not r:
                        continue
                    f = t / np.linalg.norm(t); l = np.array([-f[1], f[0]])
                    g = np.array([[a, b] for a in np.linspace(-.5, .5, 7) for b in np.linspace(-.5, .5, 4)])
                    pts = np.array([x, y]) + g[:, :1] * (L + 2 * clear) * f + g[:, 1:] * (W + 2 * clear) * l
                    if np.any(otree.query(pts)[0] < 0.8):
                        continue
                    # wz: prefer ground near road level (a stand perched on a cut bank on tall legs looks wrong)
                    zr_ = road.surface_z(j, side * road.outer(j, side), noise=False)
                    sc = r[0] + wo * off + ws * abs(s - s_ref) + wz * abs(0.5 * (r[1] + r[2]) - zr_)
                    if best is None or sc < best[0]:
                        best = (sc, x, y, j, side, s, off, r)
        return best

    CH = {}                                       # chunk key -> [MB, ...]  (one visual mesh each)

    def put(sub, x, y, z, xaxis, s):
        k = int(s // 1000); lst = CH.setdefault(k, [MB()])
        nv = sum(len(p) for p in sub.P)
        if lst[-1].n + nv > 60000:
            lst.append(MB())
        lst[-1].extend(sub, origin=(x, y, z), xaxis=xaxis)

    coll = CO.Collider("1WALL_race"); ncoll = [0]

    def solid(x, y, xaxis, L, W, z0, z1):
        """1WALL box if any part of the footprint is within RA_COLL_DIST of the road edge"""
        f = np.asarray(xaxis, float) / np.linalg.norm(xaxis); l = np.array([-f[1], f[0]])
        cs = np.array([[x, y]]) + np.array([[a * L / 2 * f + b * W / 2 * l][0] for a in (-1, 1) for b in (-1, 1)])
        if road_gap(cs).min() < RA_COLL_DIST:
            coll.box(x, y, f, L, W, z0, z1); ncoll[0] += 1

    def frame(j, side):
        a = side * normal(j); a = a / np.linalg.norm(a)
        return np.array([a[1], -a[0]]), a         # xaxis (along road), yaxis (away from road)

    def W2(x, y, xa, ya, lx, ly):
        return np.array([x, y]) + xa * lx + ya * ly

    def glf(x, y, xa, ya, zref):
        return lambda lx, ly: gz(*W2(x, y, xa, ya, lx, ly), zref) - zref

    SITES = []

    def site(best, R, L, W, label):
        _, x, y, j, side, s, off, r = best
        xa, ya = frame(j, side); MINE.append(rect(x, y, tangent(j), L, W)); MINE_C.add((x, y, R)); USED.append((x, y, R))
        SITES.append(dict(label=label, x=float(x), y=float(y), z=float(r[2]), s=float(s), side=int(side), off=float(off), L=L, W=W))
        return x, y, j, side, s, off, r, xa, ya

    def model_at(aid, x, y, s, yaw=0.0, zoff=0.0):
        sub = MB(); sub.model(aid, yaw=yaw); put(sub, x, y, gz(x, y, 0.0) + zoff - 0.01, (1.0, 0.0), s)

    def side_txt(side): return "L" if side > 0 else "R"

    # ------------------------------------------------------------------ grandstands (start area + summit finish)
    def grandstand_site(name, s0, s1, s_ref, nrows, L):
        W = nrows * 0.8 + 1.0
        best = spot(s0, s1, L + 2.4, W, np.arange(4.5, 22.1, 1.0), maxrange=1.6, margin=3.5, s_ref=s_ref, wo=0.04, ws=0.004, wz=0.8)
        if not best:
            best = spot(s0, s1, L + 2.4, W, np.arange(4.5, 22.1, 1.0), maxrange=2.2, margin=3.5, s_ref=s_ref, wo=0.04, ws=0.004, wz=0.8)
        if not best:
            RLOG.append(f"{name}: no spot for a grandstand"); return None
        x, y, j, side, s, off, r, xa, ya = site(best, 0.5 * math.hypot(L + 2.4, W) + 1.0, L + 2.4, W, f"grandstand_{name.split()[0].lower().replace('-', '')}")
        zref = r[2]; front = -W / 2 + 0.6                       # local y of the stand front (footprint centred)
        o = W2(x, y, xa, ya, -0.6, front)                         # stand origin: x centre of the seating, y front
        gl = glf(*o, xa, ya, zref)
        m, fp, htop = grandstand(nrows=nrows, L=L, gl=gl)
        put(m, *o, zref, xa, s)
        solid(*W2(*o, xa, ya, 0.6, (fp[3] + fp[2]) / 2), xa, L + 2.4, fp[3] - fp[2], zref - 1.0, zref + htop + 1.2)
        # catch fence between road and stand, 2.5 m in front of it
        fl = [W2(*o, xa, ya, lx, -2.6) for lx in np.linspace(-L / 2 - 2.0, L / 2 + 3.4, 8)]
        fm, P, Z = catch_fence(fl, lambda a, b: gz(a, b, zref), h=3.5)
        put(fm, 0, 0, 0, (1.0, 0.0), s)
        if road_gap(P).min() < RA_COLL_DIST:
            coll.ribbon(P[:, 0], P[:, 1], Z - 0.5, Z + 3.5, half=0.06); ncoll[0] += 1
        RLOG.append(f"{name}: grandstand {nrows} rows x {L:.0f} m (~{int(L / 0.5) * nrows} seats) s={s:.0f} {side_txt(side)} "
                    f"{off:.1f} m past shoulder, ground range {r[0]:.2f} m, + catch fence {L + 5.4:.0f} m")
        return x, y, j, side, s, xa, ya

    if REPLACES_BLEACHER:
        grandstand_site("summit finish", 7392 - 70, 7392 + 70, 7392, 7, 14.0)
    grandstand_site("SR-84 start", GATES[0] + 5, GATES[0] + 170, GATES[0] + 40, 6, 12.0)

    # ------------------------------------------------------------------ paddocks
    def paddock(name, s0, s1, s_ref, full=True):
        L, W = (26.0, 11.0) if full else (16.0, 8.0)
        best = spot(s0, s1, L, W, np.arange(2.5, 25.1, 1.0), maxrange=1.4, margin=2.5, s_ref=s_ref, wo=0.03, ws=0.003)
        if not best and full:
            return paddock(name, s0, s1, s_ref, full=False)
        if not best:
            RLOG.append(f"{name}: no paddock spot"); return
        x, y, j, side, s, off, r, xa, ya = site(best, 0.5 * math.hypot(L, W) + 1.0, L, W, f"paddock_{name.split()[0].lower().replace('-', '')}")
        zref = r[2]
        items = []
        # gazebo row nearest the road (fronts face the road), trailers behind, toilets at the far end
        gz_row = [(-L / 2 + 3.5, 3.0, "val_timing", "canopy_red"), (-L / 2 + 7.5, 3.0, "val_paddock", "canopy_white")]
        if full:
            gz_row.append((-L / 2 + 13.0, 6.0, "val_paddock", "canopy_navy"))
        for lx, gl_len, val, can in gz_row:
            ly = -W / 2 + 2.0
            o = W2(x, y, xa, ya, lx, ly)
            g = gazebo(L=gl_len, W=3.0, canopy=can, val=val, gl=glf(*o, xa, ya, zref))
            put(g, *o, zref, xa, s); items.append(f"gazebo {gl_len:.0f}x3 {val[4:]}")
        ntr = 2 if full else 1
        for k in range(ntr):
            lx = -L / 2 + 4.0 + k * 7.5; ly = W / 2 - 2.2
            o = W2(x, y, xa, ya, lx, ly)
            put(trailer(gl=glf(*o, xa, ya, zref)), *o, zref, xa, s); items.append("trailer")
            solid(*W2(x, y, xa, ya, lx + 0.3, ly), xa, 6.6, 2.4, zref - 1.0, zref + 2.9)
        nl = 3 if full else 2
        for k in range(nl):
            lx = L / 2 - 1.0; ly = -W / 2 + 1.0 + k * 1.3
            o = W2(x, y, xa, ya, lx, ly)
            sub = portaloo(gl=glf(*o, -ya, xa, zref))                       # doors face -x (along the paddock)
            put(sub, *o, zref, -ya, s)
            solid(*o, xa, 1.2, 1.2, zref - 1.0, zref + 2.3)
        items.append(f"{nl} toilets")
        # generator + fuel behind the timing tent, oil drums (bins), picnic table, gas bottle, extinguisher
        props = [("portable_generator", -L / 2 + 3.5, -W / 2 + 4.6, 0.3), ("Barrel_01", -L / 2 + 1.0, -W / 2 + 4.4, 0.0),
                 ("Barrel_01", L / 2 - 2.6, -W / 2 + 0.8, 1.0), ("korean_fire_extinguisher_01", -L / 2 + 5.4, -W / 2 + 0.4, 0.5)]
        if full:
            props += [("wooden_picnic_table", L / 2 - 5.0, W / 2 - 2.0, math.pi / 2), ("propane_tank", -L / 2 + 4.6, -W / 2 + 4.6, 0.0),
                      ("Barrel_01", L / 2 - 2.6, W / 2 - 0.8, 2.0), ("metal_jerrycan", -L / 2 + 2.6, -W / 2 + 4.9, 1.2)]
        for aid, lx, ly, yaw in props:
            p = W2(x, y, xa, ya, lx, ly)
            model_at(aid, *p, s, yaw=yaw + math.atan2(xa[1], xa[0]))
        sub = MB(); sub.box(-0.6, 0.6, -0.04, -0.02, 1.2, 1.8, "sign_paddock", faces={"+y": "white"})
        for xx in (-0.5, 0.5): sub.cyl((xx, 0, -0.3), (xx, 0, 1.8), 0.03, "galv", n=5)
        o = W2(x, y, xa, ya, -L / 2 - 0.5, -W / 2 + 0.3); put(sub, *o, gz(*o, zref), xa, s)
        RLOG.append(f"{name}: paddock {L:.0f}x{W:.0f} m s={s:.0f} {side_txt(side)} {off:.1f} m past shoulder "
                    f"(ground range {r[0]:.2f} m): " + ", ".join(items))

    paddock("SR-84 start", GATES[0] - 15, GATES[0] + 140, GATES[0] + 30)
    paddock("Castro Valley start (sb)", GATES[2] - 150, GATES[2] + 20, GATES[2] - 30, full=False)

    # summit: officials gazebo + toilets after the finish line (cars park up here), TV tower
    best = spot(7392 + 8, 7392 + 120, 8.0, 4.0, np.arange(2.5, 15.1, 1.0), maxrange=0.9, s_ref=7392 + 30)
    if best:
        x, y, j, side, s, off, r, xa, ya = site(best, 5.0, 8.0, 4.0, "summit_officials"); zref = r[2]
        o = W2(x, y, xa, ya, -2.0, 0.0); put(gazebo(3.0, 3.0, "canopy_white", "val_officials", gl=glf(*o, xa, ya, zref)), *o, zref, xa, s)
        for k in range(2):
            o = W2(x, y, xa, ya, 1.6 + k * 1.3, 0.6)
            put(portaloo(gl=glf(*o, ya * -1, xa, zref)), *o, zref, -ya, s); solid(*o, xa, 1.2, 1.2, zref - 1, zref + 2.3)
        model_at("korean_fire_extinguisher_01", *W2(x, y, xa, ya, -0.2, -1.6), s)
        RLOG.append(f"summit: officials gazebo + 2 toilets s={s:.0f} {side_txt(side)} {off:.1f} m past shoulder")
    best = spot(7392 - 40, 7392 + 60, 2.4, 2.4, np.arange(1.5, 10.1, 0.5), maxrange=0.9, s_ref=7392, margin=1.5)
    if best:
        x, y, j, side, s, off, r, xa, ya = site(best, 2.5, 2.4, 2.4, "summit_tvtower"); zref = r[2]
        put(tv_tower(gl=glf(x, y, xa, ya, zref)), x, y, zref, xa, s); solid(x, y, xa, 2.2, 2.2, zref - 1, zref + 5.0)
        RLOG.append(f"summit: TV camera tower s={s:.0f} {side_txt(side)} {off:.1f} m past shoulder")

    # ------------------------------------------------------------------ marshal posts (21b): extinguishers + bale wall
    import re
    for line in list(LOG):
        mm = re.match(r"\s*marshal post (\d+) s=(\d+) ([LR]) \+(\d+) m", line)
        if not mm:
            continue
        k, s, sd, off = int(mm[1]), float(mm[2]), 1 if mm[3] == "L" else -1, float(mm[4])
        j = idx(s); x, y = road.xy(j, sd * (road.outer(j, sd) + off)); xa, ya = frame(j, sd); z = gz(x, y, road.z[j])
        sub = MB()
        for e in (-1, 1):
            p = W2(x, y, xa, ya, e * 1.05, 0.25); model_at("korean_fire_extinguisher_01", *p, s, yaw=float(e))
        nb = 0
        if off >= 2.6:                            # bale wall between road and post
            for lx in (-1.25, 0.0, 1.25):
                p = W2(x, y, xa, ya, lx, -1.45)
                if road_gap(p)[0] > 0.5:
                    bale(sub, *(p - np.array([x, y])), math.atan2(xa[1], xa[0]), gz(*p, z) - z); nb += 1
            if nb:
                c = W2(x, y, xa, ya, 0, -1.45); solid(*c, xa, 3.7, 0.8, z - 1.0, z + 0.7)
        else:                                     # tight spot: bales at the up-road end
            for lx in (-1.6,):
                p = W2(x, y, xa, ya, lx, 0.0)
                bale(sub, *(p - np.array([x, y])), math.atan2(ya[1], ya[0]), gz(*p, z) - z); nb += 1
            solid(*W2(x, y, xa, ya, -1.6, 0), ya, 1.2, 0.8, z - 1.0, z + 0.7)
        put(sub, x, y, z, (1.0, 0.0), s)
        SITES.append(dict(label=f"marshal{k}_s{int(s):05d}", x=float(x), y=float(y), z=float(z), s=s, side=sd, off=off, L=2.0, W=2.0))
        RLOG.append(f"marshal post {k} s={s:.0f}: 2 extinguishers + {nb} straw bales")

    # ------------------------------------------------------------------ spectator areas on the inside of hard corners
    mpost = [float(m_[1]) for m_ in (re.match(r"\s*marshal post \d+ s=(\d+)", l) for l in LOG) if m_]
    nsp = 0; sp_s = []
    for kpk, i0, i1, inner in sorted(ns["corners"], reverse=True):
        if nsp >= 3:
            break
        s_c = float(road.s[(i0 + i1) // 2])
        if min(abs(s_c - g) for g in GATES) < 400 or any(abs(s_c - q) < 1500 for q in sp_s):
            continue
        best = spot(s_c - 40, s_c + 40, 14.0, 6.0, np.arange(3.0, 12.1, 1.0), sides=(inner,), maxrange=1.0, margin=2.5, s_ref=s_c, wo=0.08)
        if not best:
            continue
        _sd = "left" if best[4] > 0 else "right"           # dev4 safety QA: never where cars run wide (exit kerbs, either direction)
        if any(k_["type"] == "exit" and k_["side"] == _sd and k_["s0"] - 30 <= best[5] <= k_["s1"] + 30 for k_ in ns["meta"].get("kerbs", [])):
            RLOG.append(f"spectator area near s={best[5]:.0f}: skipped, run-wide zone of an exit"); continue
        x, y, j, side, s, off, r, xa, ya = site(best, 8.0, 14.0, 6.0, f"spectators_s{int(s_c):05d}"); zref = r[2]; sp_s.append(s_c); nsp += 1
        bl = [W2(x, y, xa, ya, -7.0, 1.0), W2(x, y, xa, ya, -7.0, -2.6), W2(x, y, xa, ya, 7.0, -2.6), W2(x, y, xa, ya, 7.0, 1.0)]
        cb = crowd_barriers(bl[:2], lambda a, b: gz(a, b, zref)); put(cb, 0, 0, 0, (1.0, 0.0), s)
        cb = crowd_barriers(bl[1:3], lambda a, b: gz(a, b, zref)); put(cb, 0, 0, 0, (1.0, 0.0), s)
        cb = crowd_barriers(bl[2:], lambda a, b: gz(a, b, zref)); put(cb, 0, 0, 0, (1.0, 0.0), s)
        P = np.array(bl[1:3]); Z = np.array([gz(*p, zref) for p in P])
        if road_gap(P).min() < RA_COLL_DIST:
            coll.ribbon(P[:, 0], P[:, 1], Z - 0.3, Z + 1.2, half=0.06); ncoll[0] += 1
        o = W2(x, y, xa, ya, 2.5, 1.2)
        put(gazebo(3.0, 3.0, "canopy_navy", "val_paddock", back_wall=False, table=False, gl=glf(*o, xa, ya, zref)), *o, zref, xa, s)
        sub = MB(); sub.box(-0.5, 0.5, -0.04, -0.02, 1.0, 2.0, "sign_spectators", faces={"+y": "white"})
        for xx in (-0.42, 0.42): sub.cyl((xx, 0, -0.3), (xx, 0, 2.0), 0.03, "galv", n=5)
        o = W2(x, y, xa, ya, -4.0, -2.9); put(sub, *o, gz(*o, zref), xa, s)
        model_at("Barrel_01", *W2(x, y, xa, ya, -5.5, 0.0), s)
        model_at("wooden_picnic_table", *W2(x, y, xa, ya, -2.0, 1.2), s, yaw=math.atan2(xa[1], xa[0]))
        RLOG.append(f"spectator area s={s:.0f} {side_txt(side)} (inside of R~{1 / kpk:.0f} m corner) {off:.1f} m past shoulder: "
                    "crowd barriers 14 m + returns, shade gazebo, sign")

    # ------------------------------------------------------------------ side roads / driveways: ROAD CLOSED / cones
    from pyproj import Transformer
    org = json.load(open(ROOT / "config/origin.json"))
    tf = Transformer.from_crs("EPSG:4326", org["crs"], always_xy=True)
    ways = json.load(open(RAW / "osm/side_ways.json")) if (RAW / "osm/side_ways.json").exists() else {}
    if not ways:
        RLOG.append("side roads: data/raw/race_assets/osm/side_ways.json missing (run fetch_side_ways()), skipped")
    Fe = json.load(open(DER / "features.geojson"))["features"]
    nbar = ncone = 0; done = []
    import glob as _gl
    GATED = [(float(e["s0"]), float(e["s1"]), e.get("side", "both")) for f_ in _gl.glob(str(DER / "roadside_assets_*.json"))
             for e in json.load(open(f_)) if any(k_ in e.get("type", "") for k_ in ("gate", "chainlink"))]
    for f in sorted(Fe, key=lambda f: f["properties"].get("s_m", 0)):
        p = f["properties"]
        if p.get("kind") not in ("junction", "driveway") or str(p.get("osm_way")) not in ways:
            continue
        s = float(p["s_m"])
        if s < 40 or s > s_end - 40 or min(abs(s - g) for g in GATES) < 50 or p.get("highway") in ("primary", "primary_link", "path", "footway"):
            continue
        w = ways[str(p["osm_way"])]
        ex, ey = tf.transform(*np.asarray(w["lonlat"]).T); P = np.c_[ex - org["x0"], ey - org["y0"]]
        j = idx(s); c = np.array([road.x[j], road.y[j]])
        k0 = int(np.argmin(np.hypot(*(P - c).T)))
        for br in (P[k0:], P[k0::-1]):
            if len(br) < 2:
                continue
            L_ = np.r_[0, np.cumsum(np.hypot(*np.diff(br, axis=0).T))]
            if L_[-1] < 10:
                continue
            q = np.array([np.interp(min(14.0, L_[-1]), L_, br[:, i]) for i in range(2)])
            d = q - c; d /= np.linalg.norm(d); side = 1 if np.dot(d, normal(j)) > 0 else -1
            if abs(np.dot(d, normal(j))) < 0.45:
                continue                                         # runs along the road (parallel lane): skip
            dist = road.outer(j, side) + (3.0 if p["kind"] == "junction" else 1.5)
            mouth = c + d * dist / abs(np.dot(d, normal(j)))
            if road_gap(mouth)[0] < 0.6 or any(np.hypot(*(mouth - m_)) < (12 if p["kind"] == "junction" else 60) for m_ in done):
                continue
            ze = road.surface_z(j, side * road.outer(j, side), noise=False); zm = gz(*mouth, ze)
            if abs(zm - ze) > 1.5:
                continue
            if any(math.hypot(mouth[0] - a, mouth[1] - b) < r for a, b, r in USED):
                continue
            sd_txt = "left" if side > 0 else "right"
            if p["kind"] == "junction" and any(a_ - 8 <= s <= b_ + 8 and g_ in (sd_txt, "both", "center") for a_, b_, g_ in GATED):
                RLOG.append(f"  barricade s={s:.0f} {side_txt(side)}: skipped, gate / fence already closes the mouth"); continue
            xa = np.array([d[1], -d[0]]); done.append(mouth)
            sub = MB()
            if p["kind"] == "junction":
                sub.extend(barricade(), origin=(0, 0, 0)); nbar += 1
                for e in (-1.9, 1.9):
                    cone(sub, e, -0.8, gz(*W2(*mouth, xa, d, e, -0.8), zm) - zm); ncone += 1
                solid(*mouth, xa, 2.4, 0.3, zm - 1.0, zm + 1.5)
            else:
                for e in (-1.2, 1.2):
                    cone(sub, e, 0.0, gz(*W2(*mouth, xa, d, e, 0.0), zm) - zm); ncone += 1
            put(sub, *mouth, zm, xa, s)
            if p["kind"] == "junction":
                SITES.append(dict(label=f"roadclosed_s{int(s):05d}", x=float(mouth[0]), y=float(mouth[1]), z=float(zm), s=s, side=side, off=3.0, L=2.4, W=1.0))
            RLOG.append(f"  {'ROAD CLOSED barricade' if p['kind'] == 'junction' else 'cones'} s={s:.0f} {side_txt(side)} "
                        f"({p.get('name') or p.get('highway')})")
    RLOG.append(f"side roads: {nbar} ROAD CLOSED barricades, {ncone} cones")

    # ------------------------------------------------------------------ write meshes into the shared dressing group
    out = []
    for k in sorted(CH):
        for n_, mb in enumerate(CH[k]):
            if not mb.P:
                continue
            P, T, U, N = mb.arrays()
            assert len(P) < 65535, (k, len(P))
            out.append(dict(name=f"race_assets_{k:02d}" + (f"_{n_}" if n_ else ""), material="race_assets", P=P, T=T, UV=U, N=N))
    G["dressing"] = [m for m in G.get("dressing", []) if m["material"] != "race_assets" and not m["name"].startswith("1WALL_race")]
    G["dressing"] += out + coll.meshes()
    ATL.transpose(Image.FLIP_TOP_BOTTOM).save(BUILD / "race_assets_atlas.png")   # same convention as 21b's atlas
    nv = sum(len(m["P"]) for m in out)
    RLOG.append(f"race assets: {len(out)} visual meshes (draw calls), {nv} vertices, {ncoll[0]} collision pieces")
    ns["meta"]["race_sites"] = SITES
    for l in RLOG:
        LOG.append("[race] " + l)
    print("\n".join("[race] " + l for l in RLOG))
    return out


def fetch_side_ways():
    """(re)download the OSM geometry of every junction/driveway way in features.geojson (api.openstreetmap.org)"""
    import time
    F_ = json.load(open(DER / "features.geojson"))["features"]
    ids = sorted({f["properties"]["osm_way"] for f in F_ if f["properties"].get("kind") in ("junction", "driveway")})
    out = {}
    for i in ids:
        d = json.load(urllib.request.urlopen(f"https://api.openstreetmap.org/api/0.6/way/{i}/full.json", timeout=30))
        nodes = {e["id"]: (e["lon"], e["lat"]) for e in d["elements"] if e["type"] == "node"}
        w = [e for e in d["elements"] if e["type"] == "way"][0]
        out[str(i)] = dict(tags=w.get("tags", {}), lonlat=[nodes[n] for n in w["nodes"]]); time.sleep(0.3)
    (RAW / "osm").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RAW / "osm/side_ways.json", "w"))


if __name__ == "__main__":
    if "fetch" in sys.argv:
        fetch_side_ways()
        for a in PH_MODELS:
            _fetch_ph(a)
