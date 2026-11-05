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


def foliage(kind, seed):
    """procedural tree card (RGBA 512x1024): trunk + clumped crown, alpha-tested"""
    r = np.random.default_rng(seed)
    W, H = 512, 1024
    im = Image.new("RGBA", (W, H), (60, 70, 40, 0)); d = ImageDraw.Draw(im)
    trunk_top = {"euc": 0.55, "oak": 0.35, "bay": 0.4}[kind]
    tw = {"euc": 18, "oak": 34, "bay": 22}[kind]
    bark = {"euc": (170, 160, 140), "oak": (70, 60, 50), "bay": (85, 75, 60)}[kind]
    d.polygon([(W / 2 - tw, H), (W / 2 + tw, H), (W / 2 + tw * 0.4, H * (1 - trunk_top)), (W / 2 - tw * 0.4, H * (1 - trunk_top))], fill=bark + (255,))
    base = {"euc": (95, 110, 75), "oak": (70, 85, 40), "bay": (50, 75, 40)}[kind]
    # crown envelope
    if kind == "euc":
        cx, cy, rx, ry = W / 2, H * 0.32, W * 0.30, H * 0.30
    elif kind == "oak":
        cx, cy, rx, ry = W / 2, H * 0.40, W * 0.47, H * 0.33
    else:
        cx, cy, rx, ry = W / 2, H * 0.38, W * 0.40, H * 0.36
    for _ in range(2600):
        a = r.uniform(0, 2 * np.pi); q = math.sqrt(r.uniform(0, 1))
        x = cx + math.cos(a) * rx * q; y = cy + math.sin(a) * ry * q
        s = r.uniform(6, 20)
        shade = 0.55 + 0.45 * (1 - (y - (cy - ry)) / (2 * ry)) * r.uniform(0.7, 1.1)
        col = tuple(int(np.clip(c * shade * r.uniform(0.85, 1.15), 0, 255)) for c in base)
        d.ellipse([x - s, y - s * 0.8, x + s, y + s * 0.8], fill=col + (255,))
    # branches visible through gaps
    for _ in range(10):
        y0 = H * (1 - trunk_top) * r.uniform(0.9, 1.0); x1 = cx + r.uniform(-rx, rx) * 0.8; y1 = cy + r.uniform(-ry, ry) * 0.5
        d.line([(W / 2, y0), (x1, y1)], fill=bark + (255,), width=4)
    a = np.asarray(im).copy()
    a[..., 3] = np.where(a[..., 3] > 128, 255, 0)
    return Image.fromarray(a)


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

T = {
    "asphalt_d.png": png(jpg_tex("Asphalt026C", "Color")),
    "asphalt_n.png": png(jpg_tex("Asphalt026C", "NormalDX")),
    "asphalt_maps.png": png(maps_from_rough("Asphalt026C", 0.3).resize((1024, 1024))),
    "ground_d.png": png(jpg_tex("Ground037", "Color")),
    "rock_d.png": png(jpg_tex("Rock030", "Color")),
    "dirt_d.png": png(jpg_tex("Ground054", "Color")),
    "backdrop_d.png": png(backdrop_img),
    "tree_oak.png": png(foliage("oak", 1)), "tree_bay.png": png(foliage("bay", 2)), "tree_euc.png": png(foliage("euc", 3)),
    "paint_white.png": png(solid((230, 230, 225))), "paint_yellow.png": png(solid((225, 180, 40))),
    "kerb.png": png(kerb_tex()), "guardrail.png": png(guardrail_tex()), "black.png": png(solid((0, 0, 0))),
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
    "backdrop": Material("backdrop", "ksPerPixel", {**BASE, "ksAmbient": 0.5, "ksDiffuse": 0.35}, {"txDiffuse": "backdrop_d.png"}),
    "paint_white": Material("paint_white", "ksPerPixel", dict(BASE), {"txDiffuse": "paint_white.png"}),
    "paint_yellow": Material("paint_yellow", "ksPerPixel", dict(BASE), {"txDiffuse": "paint_yellow.png"}),
    "kerb": Material("kerb", "ksPerPixel", {**BASE, "ksSpecular": 0.1, "ksSpecularEXP": 20.0}, {"txDiffuse": "kerb.png"}),
    "guardrail": Material("guardrail", "ksPerPixel", {**BASE, "ksSpecular": 0.4, "ksSpecularEXP": 40.0}, {"txDiffuse": "guardrail.png"}),
    "invisible": Material("invisible", "ksPerPixel", dict(BASE), {"txDiffuse": "black.png"}),
}
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
    print(f"{fname:28s} {len(nodes):4d} nodes  {(TRACK / fname).stat().st_size / 1e6:6.1f} MB")


gr_key = lambda n: "guardrail_all" if n.startswith(("guardrail_", "grpost_")) else n
line_key = lambda n: "lines_" + n.rsplit("_", 1)[1]  # one mesh per chunk per colour

write("palomares_road.kn5", ["road"])
write("palomares_terrain.kn5", ["terrain", "backdrop"])
write("palomares_trees.kn5", ["trees"])
write("palomares_barriers.kn5", ["barriers"], merge={"barriers": gr_key})
write("palomares_lines_road.kn5", ["lines_road"], merge={"lines_road": line_key})
write("palomares_lines_track.kn5", ["lines_track"], merge={"lines_track": line_key})
write("palomares_kerbs.kn5", ["kerbs"])
L = meta["locators"]
write("palomares_timing.kn5", [], extra_nodes=[locator(n, L[n]) for n in sorted(L)])

# ------------------------------------------------------------------ AC folder
SHARED = ["palomares_road.kn5", "palomares_terrain.kn5", "palomares_trees.kn5", "palomares_barriers.kn5", "palomares_timing.kn5"]
LAYOUTS = {
    "road_summit": dict(files=SHARED + ["palomares_lines_road.kn5"], name="Palomares Road — Hill Climb (Road)",
                        desc="Palomares Road, Alameda County CA, from Niles Canyon Rd (SR-84) up to the summit. The road as it really is: real width, center and edge lines, no kerbs."),
    "track_summit": dict(files=SHARED + ["palomares_lines_track.kn5", "palomares_kerbs.kn5"], name="Palomares Road — Hill Climb (Track)",
                         desc="Palomares Road from Niles Canyon Rd (SR-84) up to the summit, race-prepared: same geometry, no center line, kerbs on corner apexes and exits."),
}
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
rows = [r for r in _csv.DictReader(open(DER / "centerline.csv")) if float(r["s_m"]) <= meta["s_end"]]
xy = np.array([[float(r["x_local"]), float(r["y_local"])] for r in rows])
acx, acz = xy[:, 0], -xy[:, 1]
SCALE = 2.0; MARGIN = 20
W = int((acx.max() - acx.min()) / SCALE + 2 * MARGIN); Hh = int((acz.max() - acz.min()) / SCALE + 2 * MARGIN)
XO = -acx.min() + MARGIN * SCALE; ZO = -acz.min() + MARGIN * SCALE
mp = Image.new("RGBA", (W, Hh), (0, 0, 0, 0)); d = ImageDraw.Draw(mp)
pts = [((x + XO) / SCALE, (z + ZO) / SCALE) for x, z in zip(acx[::3], acz[::3])]
d.line(pts, fill=(255, 255, 255, 255), width=6, joint="curve")
MAPINI = f"[PARAMETERS]\nWIDTH={W}\nHEIGHT={Hh}\nMARGIN={MARGIN}\nSCALE_FACTOR={SCALE}\nX_OFFSET={XO:.3f}\nZ_OFFSET={ZO:.3f}\nDRAWING_SIZE=10\nMAX_SIZE=1600\n"


def cams():
    # simple replay cameras every ~400 m, 6 m off the right edge, 4 m up
    out = ["[HEADER]\nVERSION=3\nCAMERA_COUNT={n}\nSET_NAME=Palomares\n"]
    k = 0
    for s in range(100, int(CFG["finish_gate_m"]), 400):
        r = rows[s]
        x, y, z = float(r["x_local"]), float(r["y_local"]), float(r["z_navd88"])
        tx = float(rows[s + 1]["x_local"]) - x; ty = float(rows[s + 1]["y_local"]) - y
        n = math.hypot(tx, ty); tx, ty = tx / n, ty / n
        cx, cy, cz = x + ty * 8, y - tx * 8, z + 4
        out.append(f"[CAMERA_{k}]\nNAME=cam{k}\nPOSITION={cx:.2f},{cz:.2f},{-cy:.2f}\nFORWARD={-ty:.3f},-0.2,{-tx:.3f}\nUP=0,1,0\nMIN_FOV=10\nMAX_FOV=50\nIN_POINT={max(0, (s - 200) / meta['s_end']):.4f}\nOUT_POINT={min(1, (s + 200) / meta['s_end']):.4f}\nSHADOW_SPLIT0=50\nSHADOW_SPLIT1=200\nSHADOW_SPLIT2=600\nNEAR_PLANE=1\nFAR_PLANE=5000\nMIN_EXPOSURE=0\nMAX_EXPOSURE=10000\nDOF_FACTOR=10\nDOF_RANGE=10000\nDOF_FOCUS=0\nDOF_MANUAL=0\nSPLINE=\nSPLINE_ROTATION=0\nFOV_GAMMA=0\nSPLINE_ANIMATION_LENGTH=0\nIS_FIXED=0\n")
        k += 1
    out[0] = out[0].format(n=k)
    return "\n".join(out)


length_m = CFG["finish_gate_m"] - CFG["start_gate_m"]
ATTR = ("Data: centerline from OpenStreetMap contributors (ODbL, openstreetmap.org/copyright); "
        "terrain from USGS 3DEP LiDAR CA_AlamedaCounty_2021_B21 and USGS imagery (public domain); "
        "textures ambientCG (CC0). Fan-made, not affiliated with Alameda County.")
for lay, L_ in LAYOUTS.items():
    (TRACK / f"models_{lay}.ini").write_text("".join(f"[MODEL_{i}]\nFILE={f}\nPOSITION=0,0,0\nROTATION=0,0,0\n\n" for i, f in enumerate(L_["files"])))
    dd = TRACK / lay / "data"; dd.mkdir(parents=True)
    (dd / "surfaces.ini").write_text(SURF)
    (dd / "lighting.ini").write_text(LIGHT)
    (dd / "cameras.ini").write_text(cams())
    (dd / "map.ini").write_text(MAPINI)
    (dd / "groove.ini").write_text("[HEADER]\nGROOVES_NUMBER=0\n")
    (dd / "audio_sources.ini").write_text("")
    mp.save(TRACK / lay / "map.png")
    ui = TRACK / "ui" / lay; ui.mkdir(parents=True)
    json.dump({"name": L_["name"], "description": L_["desc"] + " " + ATTR, "tags": ["hillclimb", "point to point", "usa", "touge", "lidar"],
               "geotags": ["37.5978 N", "121.9469 W"], "country": "USA", "city": "Sunol / Castro Valley, CA",
               "length": f"{length_m:.0f}", "width": "5-8", "pitboxes": "1", "run": "clockwise",
               "author": "palomares pipeline", "version": VERSION, "url": ""},
              open(ui / "ui_track.json", "w"), indent=2, ensure_ascii=False)
    # outline.png: same line on transparent 1000 px
    ol = mp.copy(); ol.thumbnail((1000, 1000)); ol.save(ui / "outline.png")
print("track folder ->", TRACK)
