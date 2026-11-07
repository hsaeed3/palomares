"""dev7 asset types (config/asset_catalog.yaml entries with dev7: true), registered into scripts/p5/assets.py CATALOG.

Source rule (user, dev7): existing CC0 assets first.
  * Poly Haven CC0 models (glTF, decimated here): prop.bench, prop.trash_can
  * ambientCG CC0 colour maps: every facade, concrete, steel, rock and siding surface (scripts/p5/tex_dev7.py)
  * CSP-native features: blinking amber beacons / crossing lights via ext_config MATERIAL_ADJUSTMENT + CONDITION
    (FLASHING_FREQUENCY, pattern verified in the CSP acc-extension-config repo, config/tracks/shibuya.ini);
    ornamental lamp globes via an emissive material switched by CSP NIGHT_SMOOTH (csp_snippet()).
  * procedural geometry only where a model can't fit the measured dimensions: building boxes, bridge spans sized
    by span_m / clearance_m, rail track, sign masts and sign faces (public-domain MUTCD layouts).
Material: one atlas "ledger7" (assets/_textures/ledger7.dds) for everything + three tiny emissive materials
(beacon_a, beacon_b: alternating amber; xing_a, xing_b: alternating red; lamp_glow). Parts are returned by
generate_parts(aid, L, params) -> {material: MB}; AS.generate() returns the main "ledger7" part only.
Local frame as in assets.py: x along the road, y outward (away from the road), z up; signs face -x.
Bridges over the road: span along y (across the road) centred on the anchor; the integration step yaws them to the
railway's bearing. town_generic: params.footprint (scene-local xy) -> geometry relative to the footprint centroid.
Collision: buildings, walls, abutments, piers, platforms collide; small props (signs on posts excepted, as dev5) don't.
"""
import math, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "scripts/p2"))
import assets as AS            # noqa: E402
import tex_dev7 as TX          # noqa: E402

MB = AS.MB
MAT = "ledger7"
YEL, BLK, WHT, GRN, RED = (246, 204, 20), (14, 14, 14), (244, 244, 240), (0, 104, 62), (190, 24, 30)


class Atlas7(AS.LAtlas):
    W, H = 4096, 4096


_A = None


def atlas():
    global _A
    if _A is None:
        _A = Atlas7(); _base(_A)
    return _A


def R7(name):
    return atlas().rect[name]


def put(name, fn):
    A = atlas()
    return A.rect[name] if name in A.rect else A.put(name, fn())


SIDING = {"white": (232, 230, 222), "cream": (236, 222, 186), "beige": (214, 196, 164), "brown": (120, 84, 56), "red": (150, 52, 40),
          "green": (78, 104, 72), "yellow": (218, 176, 84), "grey": (160, 160, 156), "blue": (110, 132, 150)}


def _base(A):
    T = TX
    A.put("galv", T.cc0("Metal032", 64, tint=(168, 172, 172)))
    A.put("steel_blk", T.cc0("PaintedMetal004", 128, tint=(44, 46, 46)))
    A.put("steel_grn", T.cc0("PaintedMetal004", 128, tint=(54, 70, 58)))
    A.put("rust", T.cc0("Rust004", 128))
    A.put("concrete", T.cc0("Concrete034", 128, tint=(178, 174, 166)))
    A.put("concrete_old", T.cc0("Concrete018", 128, tint=(150, 146, 138)))
    A.put("ashlar", T.cc0("PavingStones070", 128, tint=(150, 136, 118)))
    A.put("rock", T.cc0("Rock030", 256, tint=(128, 118, 104)))
    A.put("roof_flat", T.roof_flat()); A.put("shingle", T.shingles()); A.put("shingle_red", T.shingles((120, 60, 46)))
    A.put("brick", T.cc0("Bricks066", 128, tint=(156, 78, 58), crop=0.5))
    for k, c in SIDING.items():
        A.put(f"plaster_{k}", T.cc0("PaintedPlaster017", 128, tint=c, crop=0.5))
        A.put(f"siding_{k}", T.cc0("WoodSiding008", 128, tint=c, crop=0.5))
        A.put(f"bb_{k}", T.cc0("WoodSiding011", 128, tint=c, crop=0.5))
    for nm, c in (("black", BLK), ("white", WHT), ("yellow", YEL), ("amber_off", (120, 86, 20)), ("orange", (232, 104, 24)),
                  ("dark", (34, 34, 36)), ("wood", (96, 74, 54)), ("tie", (70, 58, 46)), ("green_paint", (40, 120, 60))):
        A.put(nm, T.solid(c))
    A.put("ties", T.ties_strip()); A.put("chainlink", T.chainlink()); A.put("pickets", T.pickets()); A.put("mesh_orange", T.mesh_alpha())
    A.put("stripe_rw", T.stripes((236, 234, 228), RED, 256, 32)); A.put("stripe_yb", T.stripes(YEL, BLK, 256, 32))
    A.put("palm", T.palm_card()); A.put("plane", T.crown_card())
    A.put("veg", T.crown_card((70, 104, 54), 128, 128, 9))
    A.put("coach_green", T.coach_side()); A.put("boxcar", T.coach_side((112, 52, 38), freight=True))
    for k in ("bench", "trash_can", "jersey"):
        pass


# ============================================================================= sign faces (MUTCD, public domain)
def _cv(w, h):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); return im, ImageDraw.Draw(im)


def _dia(d, fill=YEL):
    d.polygon([(128, 0), (255, 128), (128, 255), (0, 128)], fill=BLK + (255,)); d.polygon([(128, 12), (243, 128), (128, 243), (12, 128)], fill=fill + (255,))


def _txt(d, xy, s, sz, col=BLK):
    d.text(xy, s, font=TX.F(sz), fill=col + (255,), anchor="mm")


def _fit(d, xy, s, sz, wmax, col=BLK):
    while sz > 8 and d.textlength(s, font=TX.F(sz)) > wmax:
        sz -= 1
    _txt(d, xy, s, sz, col)


def face(kind, p):
    """-> (key, PIL, w_m, h_m)"""
    t = (p.get("text") or "").upper()
    if kind in ("reverse_turn", "turn_left", "turn_right", "side_road_left", "side_road_right", "deer", "falling_rocks", "slippery",
                "signal_ahead", "stop_ahead", "bicycle_warning", "text_diamond", "clearance", "ped_crossing"):
        im, d = _cv(256, 256); _dia(d, (196, 222, 40) if kind == "ped_crossing" else YEL); B = BLK + (255,)
        s = -1 if (kind.endswith("left") or p.get("first", "L") == "L") else 1
        if kind == "reverse_turn":
            d.line([(128 - 22 * s, 206), (128 - 22 * s, 150), (128 + 22 * s, 120), (128 + 22 * s, 82)], fill=B, width=20)
            d.polygon([(128 + 22 * s - 30, 86), (128 + 22 * s + 30, 86), (128 + 22 * s, 44)], fill=B)
        elif kind.startswith("turn"):
            d.line([(128 - 20 * s, 200), (128 - 20 * s, 104), (128 + 40 * s, 104)], fill=B, width=22)
            d.polygon([(128 + 40 * s, 72), (128 + 40 * s, 136), (128 + 84 * s, 104)], fill=B)
        elif kind.startswith("side_road"):
            d.line([(128, 210), (128, 52)], fill=B, width=22); d.line([(128, 140), (128 + 64 * s, 100)], fill=B, width=20)
            d.polygon([(128, 30), (100, 70), (156, 70)], fill=B)
        elif kind == "deer":
            d.polygon([(70, 140), (150, 126), (176, 92), (194, 100), (172, 140), (176, 196), (164, 196), (156, 156), (100, 160), (92, 196), (80, 196)], fill=B)
            d.line([(178, 92), (170, 62), (186, 50)], fill=B, width=6)
        elif kind == "falling_rocks":
            d.polygon([(64, 200), (64, 80), (100, 200)], fill=B)
            for x, y, r in ((130, 120, 14), (160, 160, 18), (120, 186, 12), (186, 196, 10)):
                d.ellipse((x - r, y - r, x + r, y + r), fill=B)
        elif kind == "slippery":
            d.rounded_rectangle((96, 70, 160, 120), 10, fill=B)
            for k in (-1, 1):
                d.line([(128 + 18 * k, 130), (110 + 26 * k, 160), (140 + 26 * k, 190), (120 + 26 * k, 215)], fill=B, width=8)
        elif kind == "signal_ahead":
            d.rounded_rectangle((100, 46, 156, 210), 10, fill=B)
            for y, c in ((76, RED), (128, YEL), (180, (20, 160, 70))):
                d.ellipse((112, y - 16, 144, y + 16), fill=c + (255,))
        elif kind == "stop_ahead":
            r = 40; cx, cy = 128, 92
            d.polygon([(cx + r * math.cos(math.pi / 8 + k * math.pi / 4), cy + r * math.sin(math.pi / 8 + k * math.pi / 4)) for k in range(8)], fill=RED + (255,))
            d.line([(128, 220), (128, 150)], fill=B, width=20); d.polygon([(100, 156), (156, 156), (128, 132)], fill=B)
        elif kind == "bicycle_warning":
            for x in (88, 168):
                d.ellipse((x - 30, 140, x + 30, 200), outline=B, width=8)
            d.line([(88, 170), (118, 120), (152, 120), (168, 170), (128, 170), (118, 120)], fill=B, width=7)
        elif kind == "ped_crossing":
            d.ellipse((116, 52, 140, 76), fill=B); d.line([(128, 80), (124, 140), (104, 200)], fill=B, width=14); d.line([(124, 140), (150, 198)], fill=B, width=12)
            d.line([(104, 108), (150, 100)], fill=B, width=10)
        else:                                                     # text diamond / clearance
            lines = (t or ("14'-6\"" if kind == "clearance" else "WARNING")).split(" ")
            if kind == "clearance" and not t:
                d.polygon([(128, 40), (108, 70), (148, 70)], fill=B); d.polygon([(128, 216), (108, 186), (148, 186)], fill=B)
                _fit(d, (128, 128), p.get("clearance", "14'-6\""), 60, 170)
            else:
                for i, ln in enumerate(lines[:3]):
                    _fit(d, (128, 128 + (i - (len(lines[:3]) - 1) / 2) * 46), ln, 40, 150)
        key = f"{kind}_{s}_{t}"; return key, im, 0.9 if kind in ("text_diamond", "clearance") else 0.75, 0.9 if kind in ("text_diamond", "clearance") else 0.75
    if kind == "rr_advance":
        im, d = _cv(256, 256); d.ellipse((0, 0, 255, 255), fill=BLK + (255,)); d.ellipse((10, 10, 245, 245), fill=YEL + (255,))
        d.line([(50, 50), (206, 206)], fill=BLK + (255,), width=18); d.line([(50, 206), (206, 50)], fill=BLK + (255,), width=18)
        _txt(d, (60, 128), "R", 54); _txt(d, (196, 128), "R", 54); return "rr_adv", im, 0.9, 0.9
    if kind == "yield":
        im, d = _cv(256, 224); d.polygon([(0, 0), (255, 0), (128, 222)], fill=RED + (255,)); d.polygon([(44, 24), (211, 24), (128, 168)], fill=WHT + (255,))
        _txt(d, (128, 60), "YIELD", 34, RED); return "yield", im, 0.9, 0.78
    if kind == "do_not_enter":
        im, d = _cv(256, 256); d.ellipse((0, 0, 255, 255), fill=WHT + (255,)); d.ellipse((8, 8, 247, 247), fill=RED + (255,))
        d.rectangle((40, 110, 215, 146), fill=WHT + (255,)); _txt(d, (128, 70), "DO NOT", 34, WHT); _txt(d, (128, 186), "ENTER", 34, WHT)
        return "dne", im, 0.75, 0.75
    if kind == "one_way":
        im, d = _cv(256, 88); d.rectangle((0, 0, 255, 87), fill=BLK + (255,))
        d.polygon([(20, 34), (180, 34), (180, 14), (240, 44), (180, 74), (180, 54), (20, 54)], fill=WHT + (255,)); _txt(d, (96, 44), "ONE WAY", 20, BLK)
        return "oneway", im, 0.9, 0.3
    if kind == "no_parking":
        im, d = _cv(160, 256); d.rectangle((0, 0, 159, 255), fill=WHT + (255,)); d.rectangle((4, 4, 155, 251), outline=RED + (255,), width=4)
        d.ellipse((30, 30, 130, 130), outline=RED + (255,), width=10); _txt(d, (80, 80), "P", 64); d.line([(44, 44), (116, 116)], fill=RED + (255,), width=10)
        _txt(d, (80, 170), "ANY", 30, RED); _txt(d, (80, 210), "TIME", 30, RED); return "nopark", im, 0.45, 0.72
    if kind in ("guide_green", "city_limit"):
        txt = t or ("NILES CANYON RD" if kind == "guide_green" else "CITY LIMIT"); w = 512 if kind == "guide_green" else 384
        im, d = _cv(w, 160); d.rounded_rectangle((0, 0, w - 1, 159), 14, fill=GRN + (255,)); d.rounded_rectangle((7, 7, w - 8, 152), 10, outline=WHT + (255,), width=5)
        ws = txt.split(" "); half = max(1, len(ws) // 2) if len(txt) > 14 else len(ws)
        lines = [" ".join(ws[:half]), " ".join(ws[half:])] if half < len(ws) else [txt]
        for i, ln in enumerate(lines):
            _fit(d, (w / 2, 80 + (i - (len(lines) - 1) / 2) * 58), ln, 56, w * 0.86, WHT)
        return f"{kind}_{txt}", im, (2.4 if kind == "guide_green" else 1.2), (0.75 if kind == "guide_green" else 0.5)
    if kind == "route_shield":
        im, d = _cv(224, 256); d.polygon([(0, 30), (40, 0), (184, 0), (223, 30), (223, 150), (112, 255), (0, 150)], fill=WHT + (255,))
        d.polygon([(8, 34), (44, 8), (180, 8), (215, 34), (215, 146), (112, 244), (8, 146)], fill=GRN + (255,))
        _txt(d, (112, 50), "CALIFORNIA", 20, WHT); _txt(d, (112, 140), str(p.get("route", 84)), 96, WHT); return f"shield{p.get('route', 84)}", im, 0.6, 0.68
    if kind in ("text_white_rect", "warning_text_rect"):
        bg = YEL if kind == "warning_text_rect" else WHT; txt = t or ("WHEN FLASHING" if bg == YEL else "NO DUMPING")
        im, d = _cv(256, 192); d.rounded_rectangle((0, 0, 255, 191), 10, fill=BLK + (255,)); d.rounded_rectangle((6, 6, 249, 185), 8, fill=bg + (255,))
        ws = txt.split(" "); n = len(ws)
        for i, ln in enumerate(ws[:4]):
            _fit(d, (128, 96 + (i - (min(n, 4) - 1) / 2) * 42), ln, 38, 220)
        return f"{kind}_{txt}", im, 0.9 if bg == YEL else 0.6, 0.68 if bg == YEL else 0.45
    if kind == "advisory":
        im, d = _cv(192, 192); d.rounded_rectangle((0, 0, 191, 191), 10, fill=BLK + (255,)); d.rounded_rectangle((6, 6, 185, 185), 8, fill=YEL + (255,))
        _txt(d, (96, 80), str(p.get("advisory", 25)), 84); _txt(d, (96, 152), "MPH", 40); return f"adv{p.get('advisory', 25)}", im, 0.45, 0.45
    raise KeyError(kind)


def face_rect(kind, p):
    key, im, w, h = face(kind, p)
    return atlas().put("sf_" + key, im), w, h


def _panel(m, rect, w, h, xf, yc, zb):
    m.quad([[xf, yc + w / 2, zb], [xf, yc - w / 2, zb], [xf, yc - w / 2, zb + h], [xf, yc + w / 2, zb + h]], rect, out=np.array([-1.0, 0, 0]))
    m.quad([[xf + 0.01, yc - w / 2, zb], [xf + 0.01, yc + w / 2, zb], [xf + 0.01, yc + w / 2, zb + h], [xf + 0.01, yc - w / 2, zb + h]], R7("galv"), out=np.array([1.0, 0, 0]))


def g_sign7(L, p, kind="reverse_turn", posts=1, top=2.3):
    m = MB(); rect, w, h = face_rect(kind, p); s = float(p.get("scale", 1.0)); w, h = w * s, h * s
    zb = max(top - h, 1.5 if posts == 2 else 0.9) if h < 1 else 2.0
    _panel(m, rect, w, h, -0.04, 0.0, zb)
    ys = [0.0] if posts == 1 else [-w * 0.3, w * 0.3]
    for y in ys:
        m.box(0.0, y, -0.3, zb + h - 0.05, 0.06 if posts == 1 else 0.1, 0.06 if posts == 1 else 0.1, R7("galv"))
    if kind == "bicycle_warning":
        r2, w2, h2 = face_rect("text_white_rect", {"text": "SHARE THE ROAD"}); _panel(m, r2, 0.6, 0.45, -0.04, 0.0, zb - 0.5)
    m.cbox(0, 0, 0.16, 0.16, -0.5, 1.6)
    return m


def g_store_text(L, p):
    m = MB(); txt = (p.get("text") or p.get("sign_text") or "SHOP").upper()
    rect = atlas().put("tb_" + txt, TX.text_board(txt)); w = float(p.get("w", max(2.0, 0.32 * len(txt)))); h = w * 96 / 512 * 1.6
    m.box(0, -0.05, 0, h, w, 0.08, R7("dark")); _front(m, rect, -w / 2, w / 2, -0.1, 0, h)
    return m


def _front(m, rect, x0, x1, y, z0, z1):
    """facade quad facing the road (-y)"""
    m.quad([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], rect, out=np.array([0, -1.0, 0]))


# ============================================================================= flasher mast (CSP blinking beacons)
def g_flasher_mast(L, p, parts=None):
    """galvanised pole 6.4 m at the shoulder, cantilever arm over the lane (toward -y), warning panel (params.panel)
    + advisory plaque hung under the arm, 2 amber 12-inch beacons beside the panel (materials beacon_a / beacon_b
    alternate via CSP BLINK / BLINK_ALT). Real units: Mapillary 2514402612051254, 252911107521045."""
    m = MB(); H = float(p.get("height", 6.4)); arm = float(p.get("arm_m", 3.5))
    m.cyl(0, 0, -0.4, H, 0.11, 8, R7("galv"), r1=0.08, top=R7("galv"))
    m.box(0, 0, -0.4, 0.15, 0.5, 0.5, R7("concrete"))                                      # footing
    m.box(0, -arm / 2, H - 0.45, H - 0.33, 0.1, arm, R7("galv"))                         # arm
    m.quad2([[0, 0, H - 1.4], [0, -1.2, H - 0.4], [0, -1.2, H - 0.35], [0, 0, H - 1.3]], R7("galv"))   # strut
    kind = (p.get("panel") or "sign.reverse_turn").split(".")[-1]
    rect, w, h = face_rect(kind, p); w *= 1.2; h *= 1.2
    yc = -arm * 0.62; zb = H - 0.55 - h
    _panel(m, rect, w, h, -0.05, yc, zb)
    m.box(-0.02, yc, zb, H - 0.4, 0.04, 0.05, R7("galv"))
    if p.get("advisory"):
        r2, w2, h2 = face_rect("advisory", p); _panel(m, r2, w2 * 1.2, h2 * 1.2, -0.05, yc, zb - 0.62)
    nb = int(p.get("beacons", 2)); parts = {} if parts is None else parts
    for k in range(nb):
        y = yc + (w / 2 + 0.35) * (1 if k == 0 else -1)
        z = zb + h * 0.5
        m.box(0.0, y, z - 0.22, z + 0.22, 0.2, 0.42, R7("yellow"), top=R7("yellow"))      # housing
        m.box(-0.18, y, z + 0.17, z + 0.21, 0.2, 0.42, R7("yellow"))                        # visor
        m.box(0.0, y, z + 0.22, H - 0.4, 0.04, 0.04, R7("galv"))
        mat = "beacon_a" if k % 2 == 0 else "beacon_b"
        b = parts.setdefault(mat, MB())
        b.disc_y_x(-0.105, y, z, 0.15, 10) if hasattr(b, "disc_y_x") else _disc_x(b, -0.105, y, z, 0.15, 10)
    m.cbox(0, 0, 0.3, 0.3, -0.5, 2.0)                                                       # pole only (no collision overhead)
    m.parts = parts
    return m


def _disc_x(m, x, y, z, r, n, rect=(0.0, 0.0, 1.0, 1.0)):
    """disc in the y-z plane facing -x (lens)"""
    a = np.linspace(0, 2 * np.pi, n + 1)
    for k in range(n):
        P3 = np.array([[x, y, z], [x, y + r * math.cos(a[k]), z + r * math.sin(a[k])], [x, y + r * math.cos(a[k + 1]), z + r * math.sin(a[k + 1])]])
        if np.cross(P3[1] - P3[0], P3[2] - P3[0])[0] > 0:
            P3 = P3[[0, 2, 1]]
        m.tri(P3, rect, [(0.5, 0.5), (0.5 + 0.5 * math.cos(a[k]), 0.5 + 0.5 * math.sin(a[k])), (0.5 + 0.5 * math.cos(a[k + 1]), 0.5 + 0.5 * math.sin(a[k + 1]))])


# ============================================================================= signals, lamps
def g_traffic_signal(L, p):
    m = MB(); arm = float(p.get("arm_m", 6.0)); H = 6.5; parts = {}
    m.cyl(0, 0, -0.4, H, 0.16, 8, R7("galv"), r1=0.12, top=R7("galv"))
    m.cyl(0, -arm / 2, H - 0.6, H - 0.4, 0.07, 6, R7("galv")) if False else m.box(0, -arm / 2, H - 0.6, H - 0.45, 0.14, arm, R7("galv"))
    for y in [-arm * 0.55, -arm * 0.95][: int(p.get("heads", 2))]:
        z = H - 1.7
        m.box(0, y, z, z + 1.05, 0.25, 0.32, R7("steel_blk"), top=R7("steel_blk"))
        for j, mat in enumerate(("sig_r", "sig_y", "sig_g")):
            _disc_x(parts.setdefault(mat, MB()), -0.13, y, z + 0.86 - j * 0.33, 0.11, 10)
    m.cbox(0, 0, 0.36, 0.36, -0.5, 2.0); m.parts = parts
    return m


def g_lamp_ornamental(L, p):
    m = MB(); parts = {}
    m.cyl(0, 0, -0.3, 0.6, 0.16, 8, R7("steel_blk"), r1=0.1)
    m.cyl(0, 0, 0.6, 3.8, 0.06, 8, R7("steel_blk"), r1=0.05)
    m.box(0, -0.25, 2.6, 2.65, 0.03, 0.5, R7("steel_blk")); m.box(0, -0.25, 1.4, 1.45, 0.03, 0.5, R7("steel_blk"))   # banner brackets
    g = parts.setdefault("lamp_glow", MB())
    g.cyl(0, 0, 3.85, 4.35, 0.2, 10, (0.0, 0.0, 1.0, 1.0), r1=0.16)
    g.cyl(0, 0, 4.35, 4.5, 0.16, 10, (0.0, 0.0, 1.0, 1.0), r1=0.02, top=(0.0, 0.0, 1.0, 1.0))
    m.cyl(0, 0, 3.75, 3.85, 0.22, 10, R7("steel_blk"), top=R7("steel_blk"))
    m.cyl(0, 0, 4.48, 4.62, 0.05, 6, R7("steel_blk"), top=R7("steel_blk"))
    m.parts = parts
    return m


# ============================================================================= Poly Haven CC0 props
_PH = {}


def ph_mesh(name, cell, scale_to=None, skip=("rust", "aged", "cap_0", "chain")):
    """Poly Haven glTF -> MB part on the ledger7 atlas (diffuse packed at 256 px, UVs remapped into its rect)"""
    k = (name, cell, tuple(skip))
    if k not in _PH:
        out = MB(); A = atlas(); meshes = TX.load_gltf(name, skip)
        for i, (P, N, UV, T, tex) in enumerate(meshes):
            P, N, UV, T = TX.decimate(P, N, UV, T, cell)
            if not len(T):
                continue
            rect = A.put(f"ph_{name}_{i}", tex.convert("RGBA").resize((256, 256), Image.LANCZOS))
            u0, v0, u1, v1 = rect
            uv = np.clip(UV, 0, 1)                                       # glTF v from the image top == pipeline convention
            UVa = np.c_[u0 + (u1 - u0) * uv[:, 0], v0 + (v1 - v0) * uv[:, 1]]
            out.P.append(P); out.N.append(N); out.UV.append(UVa); out.T.append(T + out.n); out.n += len(P)
        _PH[k] = out
    src = _PH[k]; m = MB()
    m.P, m.N, m.UV, m.T, m.n = list(src.P), list(src.N), list(src.UV), list(src.T), src.n
    if scale_to:
        P = np.concatenate(m.P); lo, hi = P.min(0), P.max(0); s = scale_to / max(hi - lo)
        off = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        m.P = [(q - off) * s for q in m.P]
    else:
        P = np.concatenate(m.P); lo, hi = P.min(0), P.max(0); off = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        m.P = [q - off for q in m.P]
    return m


def g_bench(L, p):
    """Poly Haven modular_street_seating (CC0): the assembled 2.4 m slatted bench with back and arm rests
    (kit parts at x > 0.5 m: spare seat / connectors are dropped)"""
    return ph_mesh("modular_street_seating", 0.02 if not p.get("lod") else 0.06,
                   skip=("suspended_support_02", "connector", "seat_bench"))


def g_trash_can(L, p):
    return ph_mesh("metal_trash_can", 0.012 if not p.get("lod") else 0.04)


def g_jersey(L, p):
    """California K-rail (F-shape profile 0.6 m base, 0.81 m high) extruded along x in 1.5 m segments (joints),
    ambientCG Concrete018. The Poly Haven concrete_road_barrier (CC0) was evaluated: 61k tris per 1.5 m piece, still
    8k after decimation, too heavy for a linear wall (docs/dev7/ASSETS.md)."""
    m = MB(); L = L or 10.0
    prof = [(-0.30, 0.0), (-0.30, 0.08), (-0.20, 0.33), (-0.075, 0.81), (0.075, 0.81), (0.20, 0.33), (0.30, 0.08), (0.30, 0.0)]
    n = max(1, int(round(L / 1.5)))
    for k in range(n):
        a, b = L * k / n, L * (k + 1) / n - 0.01
        for (y0, z0), (y1, z1) in zip(prof[:-1], prof[1:]):
            nrm = np.array([0, (y0 + y1) / 2, (z0 + z1) / 2 - 0.3])
            m.quad([[a, y0, z0 - 0.05], [b, y0, z0 - 0.05], [b, y1, z1], [a, y1, z1]] if z0 == 0 else [[a, y0, z0], [b, y0, z0], [b, y1, z1], [a, y1, z1]],
                   R7("concrete_old"), out=nrm, vspan=(0, 0.5))
        for x, sx in ((a, -1.0), (b, 1.0)):
            q = [[x, -0.30, 0], [x, 0.30, 0], [x, 0.075, 0.81], [x, -0.075, 0.81]]
            m.quad(q, R7("concrete_old"), out=np.array([sx, 0, 0]), vspan=(0, 0.4))
    m.cline(0.0, -0.3, 0.9, half=0.3)
    return m


def g_planter(L, p):
    m = MB(); rng = np.random.default_rng(int(p.get("seed", 7)))
    m.cyl(0, 0, 0, 0.6, 0.45, 12, R7("concrete"), r1=0.42, top=R7("dark"))
    m.clump(0, 0, 0.8, 0.7, R7("veg"), rng, nv=2, ntilt=3, z0=0.5)
    return m


def g_water_barrier(L, p):
    m = MB(); n = int(p.get("count", 1))
    for k in range(n):
        x = k * 1.85
        m.box(x, 0, 0, 0.75, 1.8, 0.45, R7("orange"), top=R7("orange")); m.box(x, 0, 0.75, 0.82, 1.5, 0.3, R7("orange"), top=R7("orange"))
    return m


def g_crash_cushion(L, p):
    m = MB()
    for k in range(5):
        m.box(k * 0.6, 0, 0, 0.85, 0.55, 0.85, R7("yellow"), top=R7("yellow"))
    m.quad([[-0.32, -0.43, 0.1], [-0.32, 0.43, 0.1], [-0.32, 0.43, 0.8], [-0.32, -0.43, 0.8]], R7("stripe_yb"), out=np.array([-1.0, 0, 0]))
    return m


# ============================================================================= rail set
def g_rail_track(L, p):
    """standard gauge (1.435 m) track along x: ballast shoulder, ties every 0.6 m (LOD0) / tie texture strip (LOD1)"""
    m = MB(); L = L or 10.0; g = 1.435; hb = 0.3
    m.quad([[0, -2.2, -0.02], [L, -2.2, -0.02], [L, -1.6, hb], [0, -1.6, hb]], R7("rock"), out=np.array([0, -1.0, 1.0]))
    m.quad([[0, 1.6, hb], [L, 1.6, hb], [L, 2.2, -0.02], [0, 2.2, -0.02]], R7("rock"), out=np.array([0, 1.0, 1.0]))
    nseg = max(1, int(math.ceil(L / 2.4)))
    for k in range(nseg):
        a, b = L * k / nseg, L * (k + 1) / nseg
        m.quad([[a, -1.6, hb], [b, -1.6, hb], [b, 1.6, hb], [a, 1.6, hb]], R7("ties"), out=np.array([0, 0, 1.0]))
    if not p.get("lod"):
        for x in np.arange(0.3, L, 0.6):
            m.box(x, 0, hb, hb + 0.16, 0.23, 2.6, R7("tie"))
    for y in (-g / 2 - 0.035, g / 2 + 0.035):
        zt = hb + 0.16
        m.box(L / 2, y, zt, zt + 0.15, L, 0.07, R7("rust"), top=R7("galv"))
    return m


def g_crossing_signal(L, p, parts=None):
    """crossbuck mast with 2 red flashers (xing_a / xing_b alternate via CSP), bell, raised gate arm when params.gates"""
    m = MB(); parts = {} if parts is None else parts
    m.cyl(0, 0, -0.3, 4.3, 0.08, 8, R7("galv"), top=R7("galv")); m.box(0, 0, -0.3, 0.3, 0.6, 0.6, R7("concrete"))
    cb = put("crossbuck", lambda: _crossbuck())
    for a in (math.radians(45), math.radians(-45)):
        c, s = math.cos(a) * 0.62, math.sin(a) * 0.62
        m.quad2([[-0.06, -c, 3.6 - s - 0.05], [-0.06, c, 3.6 + s - 0.05], [-0.06, c, 3.6 + s + 0.05 + 0.15], [-0.06, -c, 3.6 - s + 0.15]], cb)
    m.box(-0.02, 0, 2.6, 2.68, 0.06, 1.3, R7("galv"))
    for k, y in enumerate((-0.5, 0.5)):
        m.cyl(-0.05, y, 2.36, 2.38, 0.3, 12, R7("black"))
        m.quad([[-0.1, y + 0.3, 2.07], [-0.1, y - 0.3, 2.07], [-0.1, y - 0.3, 2.67], [-0.1, y + 0.3, 2.67]], R7("black"), out=np.array([-1.0, 0, 0]))
        _disc_x(parts.setdefault("xing_a" if k == 0 else "xing_b", MB()), -0.13, y, 2.37, 0.15, 10)
    m.cyl(0.05, 0, 4.3, 4.45, 0.16, 8, R7("black"), r1=0.1, top=R7("black"))                # bell
    if p.get("gates", True):
        m.box(0.25, 0.25, 0.0, 1.2, 0.35, 0.35, R7("galv"), top=R7("galv"))                  # gate mechanism
        m.box(0.25, 0.25, 1.2, 1.2 + 7.0, 0.1, 0.1, R7("stripe_rw"))                         # arm, raised (free roam)
    m.parts = parts
    return m


def _crossbuck():
    im = Image.new("RGBA", (256, 40), WHT + (255,)); d = ImageDraw.Draw(im); d.rectangle((0, 0, 255, 39), outline=BLK + (255,), width=3)
    d.text((128, 20), "RAILROAD CROSSING", font=TX.F(20), fill=BLK + (255,), anchor="mm"); return im


def g_crossbuck(L, p):
    m = g_crossing_signal(L, {"gates": False}); m.parts = {}; return m


def g_platform(L, p):
    m = MB(); Lp = float(p.get("length_m", 60)); W = 4.0
    m.box(Lp / 2, W / 2, -0.4, 0.6, Lp, W, R7("concrete_old"), top=R7("concrete"), uscale=3.0)
    m.box(Lp / 2, 0.3, 0.6, 0.605, Lp, 0.6, R7("yellow"))
    m.cbox(Lp / 2, W / 2, Lp, W, -0.4, 0.6)
    return m


def g_rail_car(L, p):
    """generic heavyweight coach / boxcar, plain livery, no lettering: body box (CC0 painted-metal side texture),
    arched roof, 2 trucks with wheels; on rail height 1.06 m"""
    m = MB(); typ = p.get("type", "heavyweight_coach"); fr = "box" in typ or "freight" in typ
    Lc, W, z0 = (15.0, 3.1, 1.06) if fr else (24.0, 3.1, 1.06); Hb = 2.8 if fr else 2.6
    side = R7("boxcar" if fr else "coach_green")
    for y, o in ((-W / 2, -1.0), (W / 2, 1.0)):
        q = [[0, y, z0], [Lc, y, z0], [Lc, y, z0 + Hb], [0, y, z0 + Hb]]
        m.quad(q, side, out=np.array([0, o, 0]))
    for x, o in ((0, -1.0), (Lc, 1.0)):
        m.quad([[x, -W / 2, z0], [x, W / 2, z0], [x, W / 2, z0 + Hb], [x, -W / 2, z0 + Hb]], R7("steel_grn" if not fr else "rust"), out=np.array([o, 0, 0]))
    roof = R7("dark"); n = 6
    for k in range(n):
        a0, a1 = math.pi * k / n, math.pi * (k + 1) / n
        y0, y1 = -W / 2 * math.cos(a0), -W / 2 * math.cos(a1); h0, h1 = 0.45 * math.sin(a0), 0.45 * math.sin(a1)
        m.quad([[0, y0, z0 + Hb + h0], [Lc, y0, z0 + Hb + h0], [Lc, y1, z0 + Hb + h1], [0, y1, z0 + Hb + h1]], roof, out=np.array([0, -math.cos((a0 + a1) / 2), 1.0]))
    m.box(Lc / 2, 0, z0 - 0.15, z0, Lc, W - 0.2, R7("dark"))
    for xt in (2.6, Lc - 2.6):
        m.box(xt, 0, 0.45, z0 - 0.15, 2.6, 2.4, R7("black"))
        if not p.get("lod"):
            for dx in (-0.9, 0.9):
                for y in (-0.75, 0.75):
                    m.hcyl_y(xt + dx, y, 0.45, 0.45) if hasattr(m, "hcyl_y") else m.ycyl(xt + dx, 0.46, y - 0.08, y + 0.08, 0.46, 10, R7("rust"))
    m.cbox(Lc / 2, 0, Lc, W, 0, z0 + Hb + 0.4)
    return m


# ============================================================================= bridges / abutments
def _truss_plane(m, y0, y1, x, zb, ht, panels, rect, w=0.35):
    """Pratt truss in the plane x = const spanning y0..y1 (members as square boxes along y / z)"""
    ys = np.linspace(y0, y1, panels + 1)
    m.box(x, (y0 + y1) / 2, zb, zb + w * 1.6, w, y1 - y0, rect)                            # bottom chord
    m.box(x, (ys[1] + ys[-2]) / 2, zb + ht - w, zb + ht, w, ys[-2] - ys[1], rect)          # top chord
    for i, y in enumerate(ys):
        if 0 < i < panels:
            m.box(x, y, zb, zb + ht, w * 0.7, w * 0.7, rect)                                 # verticals
    def member(pa, pb):
        pa, pb = np.array(pa), np.array(pb); d = pb - pa; n = np.array([0, -d[2], d[1]]); n = n / np.linalg.norm(n) * w * 0.35
        for off in (-w * 0.3, w * 0.3):
            m.quad2([[x + off, *(pa[1:] - n[1:])], [x + off, *(pb[1:] - n[1:])], [x + off, *(pb[1:] + n[1:])], [x + off, *(pa[1:] + n[1:])]], rect)
    member((x, ys[0], zb), (x, ys[1], zb + ht)); member((x, ys[-1], zb), (x, ys[-2], zb + ht))     # end posts
    mid = panels // 2
    for i in range(1, panels - 1):
        if i < mid:
            member((x, ys[i], zb + ht), (x, ys[i + 1], zb))
        else:
            member((x, ys[i + 1], zb + ht), (x, ys[i], zb))


def _pier(m, x0, x1, y, zt, W=2.0, rect=None):
    m.box((x0 + x1) / 2, y, -1.0, zt, x1 - x0, W, rect or R7("concrete_old"), top=R7("concrete"), uscale=3.0)
    m.cbox((x0 + x1) / 2, y, x1 - x0, W, -1.0, zt)


def g_truss_through(L, p):
    """Niles Canyon Railway steel through-truss (Mapillary 256078143846347): Pratt truss span_m along y on concrete
    piers, deck at clearance_m, two truss planes 5 m apart (x = +-2.5), portal / top lateral struts, rails on deck"""
    m = MB(); span = float(p.get("span_m", 46)); cl = float(p.get("clearance_m", 4.6)); W = 5.0; ht = 7.5
    zd = cl + 0.9; y0, y1 = -span / 2, span / 2; rect = R7("steel_blk"); pan = max(4, int(round(span / 7.5)))
    m.box(0, 0, cl, zd, W, span, rect)                                                        # floor beams / stringers (underside)
    m.box(0, 0, zd, zd + 0.05, W - 0.6, span, R7("tie"))
    for x in (-W / 2, W / 2):
        _truss_plane(m, y0, y1, x, zd, ht, pan, rect)
    for yy in np.linspace(y0, y1, pan + 1)[1:-1]:
        m.box(0, yy, zd + ht - 0.3, zd + ht, W, 0.25, rect)
    for y in (-0.75, 0.75):
        m.box(0, y, zd + 0.05, zd + 0.2, 0.07, span, R7("rust")) if False else m.box(y, 0, zd + 0.05, zd + 0.2, 0.07, span, R7("rust"))
    for yy in (y0 - 1.0, y1 + 1.0):
        _pier(m, -W / 2 - 0.5, W / 2 + 0.5, yy, zd, 2.0)
    return m


def g_deck_girder(L, p):
    """black steel plate-girder span(s) over the road (Mission Blvd x2, Niles Canyon s 3555): params.span_m, count
    spans along y with concrete piers / abutments; girder depth span/12"""
    m = MB(); span = float(p.get("span_m", 25)); n = int(p.get("count", 1)); cl = float(p.get("clearance_m", 4.6)); W = 4.6
    dep = max(1.6, span / 12); tot = span * n; y0 = -tot / 2; zb = cl
    for k in range(n):
        a, b = y0 + k * span, y0 + (k + 1) * span
        for x in (-W / 2 + 0.3, W / 2 - 0.3):
            m.box(x, (a + b) / 2, zb, zb + dep, 0.12, b - a - 0.2, R7("steel_blk"))                    # web
            m.box(x, (a + b) / 2, zb + dep - 0.05, zb + dep, 0.5, b - a - 0.2, R7("steel_blk"))        # flanges
            m.box(x, (a + b) / 2, zb, zb + 0.05, 0.5, b - a - 0.2, R7("steel_blk"))
            for yy in np.arange(a + 1.0, b - 0.5, 1.6):
                m.box(x - math.copysign(0.08, x), yy, zb, zb + dep, 0.04, 0.12, R7("steel_blk"))        # stiffeners
        m.box(0, (a + b) / 2, zb + dep, zb + dep + 0.25, W, b - a, R7("tie"))
    for k in range(n + 1):
        yy = y0 + k * span
        _pier(m, -W / 2 - 0.6, W / 2 + 0.6, yy + (-1.0 if k == 0 else 1.0 if k == n else 0.0), zb, 2.0 if 0 < k < n else 2.4)
    for x in (-W / 2, W / 2):
        m.hcyl(-0.01, 0.01, 0, 0, 0.01, 3, R7("galv")) if False else m.box(x, 0, zb + dep + 0.25, zb + dep + 1.1, 0.06, tot, R7("galv"))
    return m


def g_concrete_girder(L, p):
    """UP Oakland Subdivision concrete girder underpass (Mapillary 245974991574265, NARROW SUBWAY): concrete deck
    span_m along y, concrete abutments with flared wing walls both sides, pipe railing on top"""
    m = MB(); span = float(p.get("span_m", 18)); cl = float(p.get("clearance_m", 4.3)); W = float(p.get("width_m", 8.0)); dep = 1.4
    zb = cl; y0, y1 = -span / 2, span / 2
    m.box(0, 0, zb, zb + dep, W, span + 2.0, R7("concrete"), top=R7("concrete_old"), uscale=3.0)
    for yy in (y0 - 1.0, y1 + 1.0):
        _pier(m, -W / 2, W / 2, yy, zb, 2.0, R7("concrete"))
        for sx in (-1, 1):                                                                          # wing walls
            x0 = sx * W / 2; y = yy + math.copysign(1.0, yy)
            m.slab(0, 6.0, 0, 0.5, -1.0, zb * 0.9, R7("concrete")) if False else None
            m.box(x0 + sx * 3.0, yy, -1.0, zb * 0.7, 6.0, 0.6, R7("concrete"), top=R7("concrete"), uscale=3.0)
            m.cbox(x0 + sx * 3.0, yy, 6.0, 0.6, -1.0, zb * 0.7)
    for x in (-W / 2 + 0.1, W / 2 - 0.1):
        for z in (zb + dep + 0.5, zb + dep + 1.0):
            m.box(x, 0, z, z + 0.06, 0.06, span + 2.0, R7("galv"))
        for yy in np.arange(y0 - 1.0, y1 + 1.01, 2.0):
            m.box(x, yy, zb + dep, zb + dep + 1.06, 0.06, 0.06, R7("galv"))
    return m


def g_abutment(L, p, stone=False):
    m = MB(); Lx, W, H = float(p.get("L", 6)), float(p.get("W", 3)), float(p.get("H", 6)); rect = R7("ashlar" if stone else "concrete_old")
    m.box(Lx / 2, W / 2, -1.0, H, Lx, W, rect, top=R7("concrete"), uscale=2.0)
    if p.get("arch_opening"):
        put("arch", _arch)
        m.quad([[Lx * 0.2, -0.01, 0], [Lx * 0.8, -0.01, 0], [Lx * 0.8, -0.01, H * 0.6], [Lx * 0.2, -0.01, H * 0.6]], R7("arch"), out=np.array([0, -1.0, 0]))
    m.cbox(Lx / 2, W / 2, Lx, W, -1.0, H)
    return m


def _arch():
    im = TX.cc0("Concrete018", 128, tint=(150, 146, 138)); d = ImageDraw.Draw(im)
    d.rectangle((14, 64, 114, 127), fill=(30, 30, 30, 255)); d.pieslice((14, 14, 114, 114), 180, 360, fill=(30, 30, 30, 255)); return im


def g_balustrade(L, p):
    """historic panelled concrete parapet (old Niles Canyon arch bridges) on a raised 1 m sidewalk"""
    m = MB(); L = L or 10.0
    m.slab(0, L, 0.0, 1.0, -0.3, 0.18, R7("concrete"), top=R7("concrete"))                 # raised sidewalk
    pan = put("balust", _balust)
    m.slab(0, L, 1.0, 1.3, 0.18, 1.1, pan, tw=2.0, top=R7("concrete"))
    m.cline(1.15, -0.3, 1.4, half=0.2)
    return m


def _balust():
    im = TX.cc0("Concrete034", 128, tint=(186, 182, 172)); d = ImageDraw.Draw(im)
    d.rectangle((14, 20, 114, 108), outline=(130, 126, 118, 255), width=6); return im


# ============================================================================= walls, fences
def g_masonry_parapet(L, p):
    m = MB(); L = L or 10.0; h = float(p.get("height_m", 0.9)); r = R7("ashlar" if p.get("material") == "stone" else "concrete_old")
    m.slab(0, L, 0.0, 0.45, -0.3, h, r, top=R7("concrete")); m.cline(0.22, -0.3, h + 0.5, half=0.22); return m


def g_rock_cut(L, p):
    m = MB(); L = L or 10.0; h = float(p.get("height_m", 10)); rng = np.random.default_rng(int(p.get("seed", 7)))
    nx = max(2, int(L / 1.5)); nz = max(2, int(h / 2.0))
    xs = np.linspace(0, L, nx + 1); zs = np.linspace(-0.3, h, nz + 1)
    Y = 0.15 * np.arange(nz + 1)[None, :] / nz * h * 0.12 + rng.uniform(0, 0.35, (nx + 1, nz + 1)); Y[:, 0] = 0
    for i in range(nx):
        for j in range(nz):
            q = [[xs[i], Y[i, j], zs[j]], [xs[i + 1], Y[i + 1, j], zs[j]], [xs[i + 1], Y[i + 1, j + 1], zs[j + 1]], [xs[i], Y[i, j + 1], zs[j + 1]]]
            m.quad(q, R7("rock"), out=np.array([0, -1.0, 0]), uspan=((i % 3) / 3, (i % 3 + 1) / 3), vspan=((j % 3) / 3, (j % 3 + 1) / 3))
    m.cline(0.3, -0.5, h, half=0.3); return m


def g_retaining_block(L, p):
    m = MB(); L = L or 10.0; h = float(p.get("height_m", 3.5))
    blk = put("splitface", lambda: _blocks())
    m.slab(0, L, 0.0, 0.4, -0.3, h, blk, tw=1.6, top=R7("concrete"))
    if p.get("fence_on_top"):
        m.sheet(0, L, 0.2, h, h + 1.8, R7("chainlink"), 3.0)
        for x in np.arange(0, L + 0.01, 3.0):
            m.cyl(x, 0.2, h, h + 1.85, 0.03, 5, R7("galv"))
    m.cline(0.2, -0.3, h, half=0.2); return m


def _blocks():
    im = TX.cc0("Concrete018", 128, tint=(196, 172, 136)); d = ImageDraw.Draw(im)
    for r in range(0, 128, 32):
        d.line([(0, r), (128, r)], fill=(150, 130, 100, 255), width=3)
        for x in range((r // 32) % 2 * 32, 128, 64):
            d.line([(x, r), (x, r + 32)], fill=(150, 130, 100, 255), width=3)
    return im


def g_screen_curved(L, p):
    m = MB(); L = L or 10.0; h = float(p.get("height_m", 3.0))
    m.sheet(0, L, 0.0, 0.0, h * 0.65, R7("chainlink"), 3.0)
    nseg = max(1, int(L / 3.0))
    for k in range(nseg):
        a, b = L * k / nseg, L * (k + 1) / nseg
        m.quad2([[a, 0, h * 0.65], [b, 0, h * 0.65], [b, -0.9, h], [a, -0.9, h]], R7("chainlink"))
    for x in np.arange(0, L + 0.01, 3.0):
        m.box(x, 0, 0, h * 0.65, 0.05, 0.05, R7("galv")); m.quad2([[x, 0, h * 0.65], [x + 0.05, 0, h * 0.65], [x + 0.05, -0.9, h], [x, -0.9, h]], R7("galv"))
    return m


def g_construction_mesh(L, p):
    m = MB(); L = L or 10.0
    m.sheet(0, L, 0.0, 0.0, 1.2, R7("mesh_orange"), 2.0)
    for x in np.arange(0, L + 0.01, 2.5):
        m.box(x, 0.02, -0.3, 1.3, 0.04, 0.04, R7("dark"))
    return m


def g_picket(L, p):
    m = MB(); L = L or 10.0; h = float(p.get("height_m", 1.0))
    m.sheet(0, L, 0.0, 0.0, h, R7("pickets"), 2.0)
    for x in np.arange(0, L + 0.01, 2.4):
        m.box(x, 0.03, -0.3, h + 0.05, 0.09, 0.09, R7("white"))
    return m


# ============================================================================= buildings (procedural boxes + CC0 facades)
def _wall_tile(facade, colour):
    c = colour if colour in SIDING else {"SP yellow/brown": "yellow"}.get(colour, None)
    if facade.startswith("brick"):
        return "brick"
    if facade.startswith("stucco") or facade in ("flat_parapet", "cafe_patio", "flat_awning"):
        return f"plaster_{c or 'beige'}"
    if facade in ("board_batten_gable", "sp_depot_board_batten"):
        return f"bb_{c or 'brown'}"
    return f"siding_{c or 'white'}"


_BAYS = {"shop": lambda c: TX.shop_bay(c), "shopd": lambda c: TX.shop_bay(c, door=True), "upper": lambda c: TX.upper_bay(c),
         "uppera": lambda c: TX.upper_bay(c, arched=True), "fire": lambda c: TX.bay_doors(c)}


def _crop(name):
    A = atlas(); u0, v0, u1, v1 = A.rect[name]
    return A.img.crop((round(u0 * A.W), round(v0 * A.H), round(u1 * A.W), round(v1 * A.H)))


def BT(kind, wall):
    """facade bay tile (lazy: only the combinations the ledger uses are painted)"""
    return put(f"{kind}_{wall}", lambda: _BAYS[kind](_crop(wall)).resize((160, 160), Image.LANCZOS))


def _bay_tiles(wall):
    return None


def _shop(m, x0, x1, D, H, storeys, facade, colour, sign, rng, lod=0, ground_tile=None):
    """one storefront volume, frontage x0..x1 at y=0, depth D, height H (flat roof, parapet / false front)"""
    wall = _wall_tile(facade, colour); _bay_tiles(wall); W = x1 - x0
    nb = max(1, int(round(W / 3.5))); bw = W / nb; zf = min(4.0, H / max(storeys, 1))
    for k in range(nb):
        a, b = x0 + k * bw, x0 + (k + 1) * bw
        gt = ground_tile or (("shopd", wall) if k == nb // 2 else ("shop", wall))
        if ground_tile and ground_tile[0] == "fire":          # firehouse: 2 apparatus doors in the middle, office bays around
            gt = ground_tile if k in (nb // 2 - 1, nb // 2) else (("shopd", wall) if k == 0 else ("upper", wall))
        _front(m, (BT(*gt) if isinstance(gt, tuple) else R7(gt)), a, b, 0, -0.3, zf)
        z = zf
        for s_ in range(1, storeys):
            _front(m, BT("uppera" if facade.endswith("arched") else "upper", wall), a, b, 0, z, z + zf); z += zf
        if H > z:
            _front(m, R7(wall), a, b, 0, z, H, ) if False else m.quad([[a, 0, z], [b, 0, z], [b, 0, H], [a, 0, H]], R7(wall), out=np.array([0, -1.0, 0]), vspan=(0, min(1, (H - z) / 3.5)))
    top = H + (1.4 if facade == "western_false_front" else 0.6)
    sw = sh = zs = None
    if sign and not lod:                                   # name board above the awning / door line, fascia grown to fit
        sw = min(W - 0.6, max(3.0, 0.42 * len(sign))); sh = sw * 72 / 384 * 1.5
        zs = zf + 0.5 if storeys == 1 else zf + 0.2
        if storeys == 1:
            top = max(top, zs + sh + 0.25)
    m.quad([[x0, 0, H], [x1, 0, H], [x1, 0, top], [x0, 0, top]], R7(wall), out=np.array([0, -1.0, 0]), vspan=(0, 0.4))   # parapet / false front
    m.quad([[x1, 0, H], [x0, 0, H], [x0, 0, top], [x1, 0, top]], R7(wall), out=np.array([0, 1.0, 0]), vspan=(0, 0.4))
    m.box((x0 + x1) / 2, -0.06, top, top + 0.12, W + 0.1, 0.18, R7("white" if wall != "brick" else "concrete"))     # cornice
    for x, s in ((x0, -1.0), (x1, 1.0)):
        for j in range(max(1, int(D / 4))):
            ya, yb = D * j / max(1, int(D / 4)), D * (j + 1) / max(1, int(D / 4))
            m.quad([[x, ya, -0.3], [x, yb, -0.3], [x, yb, H], [x, ya, H]], R7(wall), out=np.array([s, 0, 0]))
    m.quad([[x0, D, -0.3], [x1, D, -0.3], [x1, D, H], [x0, D, H]], R7(wall), out=np.array([0, 1.0, 0]))
    m.quad([[x0, 0, H], [x1, 0, H], [x1, D, H], [x0, D, H]], R7("roof_flat"), out=np.array([0, 0, 1.0]))
    if not lod:
        if facade in ("flat_awning", "cafe_patio", "western_false_front"):                            # awning / porch roof
            col = [(150, 40, 36), (40, 80, 60), (60, 60, 64), (120, 90, 50)][int(rng.integers(4))]
            aw = put(f"awn_{col}", lambda: TX.awning(col))
            m.quad([[x0 + 0.2, -1.6, zf - 0.3], [x1 - 0.2, -1.6, zf - 0.3], [x1 - 0.2, 0, zf + 0.4], [x0 + 0.2, 0, zf + 0.4]], aw, out=np.array([0, -1.0, 1.0]))
            m.quad([[x1 - 0.2, -1.6, zf - 0.3], [x0 + 0.2, -1.6, zf - 0.3], [x0 + 0.2, 0, zf + 0.4], [x1 - 0.2, 0, zf + 0.4]], aw, out=np.array([0, 1.0, -1.0]))
        if sign:
            txt = sign.upper(); tb = put("tb_" + txt, lambda: TX.text_board(txt))
            xm = (x0 + x1) / 2; m.box(xm, -0.06, zs, zs + sh, sw + 0.1, 0.08, R7("dark"))
            _front(m, tb, xm - sw / 2, xm + sw / 2, -0.11, zs, zs + sh)
    m.cbox((x0 + x1) / 2, D / 2, W, D, -0.5, H + 1.0)


def g_storefront(L, p):
    """Niles Blvd / Sunol storefront: params storeys, height_m, facade, colour, sign_text, frontage_m (s1 - s)"""
    m = MB(); rng = np.random.default_rng(int(p.get("seed", 7)))
    W = float(p.get("frontage_m", 15.0)); H = float(p.get("height_m", 8.0 if int(p.get("storeys", 1)) > 1 else 4.8)); D = float(p.get("depth_m", 15.0))
    fac = p.get("facade", "flat_parapet"); st = int(p.get("storeys", 1))
    if fac == "brick_firehouse":
        _shop(m, -W / 2, W / 2, D, H, st, "brick", None, p.get("sign_text"), rng, p.get("lod", 0), ground_tile=("fire", "brick"))
    else:
        _shop(m, -W / 2, W / 2, D, H, st, fac, p.get("colour"), p.get("sign_text") or p.get("text"), rng, p.get("lod", 0))
    return m


def g_storefront_row(L, p):
    """row of attached shops (params.shops: plain names), frontage split evenly, per-shop height / colour jitter"""
    m = MB(); rng = np.random.default_rng(int(p.get("seed", 7)))
    shops = p.get("shops") or ["SHOP", "SHOP"]; W = float(p.get("frontage_m", max(12.0, 8.0 * len(shops)))); n = len(shops)
    H0 = float(p.get("height_m", 5.0)); st = int(p.get("storeys", 1)); fac = p.get("facade", "western_false_front")
    cols = ["white", "cream", "beige", "green", "red", "yellow", "blue", "grey"]
    for k, nm in enumerate(shops):
        a, b = -W / 2 + W * k / n, -W / 2 + W * (k + 1) / n
        colour = p.get("colour") or cols[int(rng.integers(len(cols)))]
        _shop(m, a, b, float(p.get("depth_m", 15.0)), H0 * rng.uniform(0.92, 1.1), st, fac, colour, nm, rng, p.get("lod", 0))
    return m


def g_depot(L, p):
    """wooden railway depot (Niles Depot, SP freight station, Sunol depot): board-and-batten walls, wide gable roof on
    brackets, bay window; length frontage_m, colour SP yellow/brown -> mustard walls"""
    m = MB(); Lx = float(p.get("frontage_m", 25.0)); D = float(p.get("depth_m", 8.0)); st = int(p.get("storeys", 1))
    wall = _wall_tile("board_batten_gable", "yellow" if "yellow" in str(p.get("colour", "")) else p.get("colour", "brown")); _bay_tiles(wall)
    eave = 3.6 * st
    nb = max(1, int(round(Lx / 3.5)))
    for k in range(nb):
        a, b = -Lx / 2 + Lx * k / nb, -Lx / 2 + Lx * (k + 1) / nb
        _front(m, BT("shopd" if k % 3 == 1 else "upper", wall), a, b, 0, -0.3, 3.6)
        if st > 1:
            _front(m, BT("upper", wall), a, b, 0, 3.6, eave)
        m.quad([[b, D, -0.3], [a, D, -0.3], [a, D, eave], [b, D, eave]], BT("upper", wall), out=np.array([0, 1.0, 0]))
    for x, s in ((-Lx / 2, -1.0), (Lx / 2, 1.0)):
        m.quad([[x, 0, -0.3], [x, D, -0.3], [x, D, eave], [x, 0, eave]], R7(wall), out=np.array([s, 0, 0]))
    ridge = AS._house.__code__ and None
    ov = 1.6; pitch = 0.55; ridge = eave + pitch * D / 2
    sh = R7("shingle_red" if "brown" in str(p.get("colour", "")) else "shingle")
    m.quad([[-Lx / 2 - 0.5, -ov, eave - ov * pitch], [Lx / 2 + 0.5, -ov, eave - ov * pitch], [Lx / 2 + 0.5, D / 2, ridge], [-Lx / 2 - 0.5, D / 2, ridge]], sh, out=np.array([0, -1.0, 1.0]))
    m.quad([[-Lx / 2 - 0.5, D / 2, ridge], [Lx / 2 + 0.5, D / 2, ridge], [Lx / 2 + 0.5, D + ov, eave - ov * pitch], [-Lx / 2 - 0.5, D + ov, eave - ov * pitch]], sh, out=np.array([0, 1.0, 1.0]))
    for x, s in ((-Lx / 2, -1.0), (Lx / 2, 1.0)):
        P3 = np.array([[x, 0, eave], [x, D, eave], [x, D / 2, ridge]])
        if np.dot(np.cross(P3[1] - P3[0], P3[2] - P3[0]), [s, 0, 0]) < 0:
            P3 = P3[[1, 0, 2]]
        m.tri(P3, R7(wall), [(0, 0), (1, 0), (0.5, 0.5)])
    if not p.get("lod"):
        for x in np.linspace(-Lx / 2 + 1, Lx / 2 - 1, max(2, nb)):                                   # eave brackets
            m.quad2([[x, 0, eave - 1.0], [x, 0, eave - 0.2], [x, -1.2, eave - 0.2 - 0.6], [x, -0.1, eave - 1.0]], R7("wood"))
        if p.get("name"):
            nm = p["name"].upper().replace("SOUTHERN PACIFIC ", "").replace(" MUSEUM", "")
            tb = put("tb_" + nm, lambda: TX.text_board(nm, bg=(236, 226, 196))); sw = min(Lx * 0.4, 0.4 * len(nm))
            _front(m, tb, -sw / 2, sw / 2, -0.05, eave - 0.95, eave - 0.95 + sw * 0.19)
    m.cbox(0, D / 2, Lx, D, -0.5, eave + 1.5)
    return m


def g_platform_wrap(L, p):
    return g_platform(L, p)


def _poly_area(P):
    x, y = P[:, 0], P[:, 1]; return 0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1))


def g_town_generic(L, p):
    """extruded OSM footprint (params.footprint, scene-local xy) to the LiDAR p90 height (params.height_m), CC0 wall
    texture chosen by hash (stucco / siding / brick), window bays per 3.2 m storey, flat roof with parapet.
    Geometry is relative to the footprint centroid (place there with yaw 0)."""
    m = MB(); fp = np.asarray(p.get("footprint") or [[-6, -5], [6, -5], [6, 5], [-6, 5], [-6, -5]], float)
    if np.allclose(fp[0], fp[-1]):
        fp = fp[:-1]
    if _poly_area(fp) < 0:
        fp = fp[::-1]
    c = fp.mean(0); fp = fp - c
    H = float(np.clip(p.get("height_m") or 6.0, 2.6, 30.0)); seed = int(abs(hash(tuple(np.round(fp[0], 1)))) % 997) if p.get("footprint") else 3
    kinds = ["plaster_white", "plaster_beige", "plaster_cream", "siding_white", "siding_grey", "siding_blue", "brick", "siding_yellow"]
    wall = kinds[seed % len(kinds)]; _bay_tiles(wall)
    nst = max(1, int(H // 3.2))
    for k in range(len(fp)):
        a, b = fp[k], fp[(k + 1) % len(fp)]; e = b - a; Le = np.hypot(*e)
        if Le < 0.05:
            continue
        nrm = np.array([e[1], -e[0], 0.0]) / Le                                     # outward for CCW
        nb = max(1, int(round(Le / 3.5)))
        for j in range(nb):
            pa, pb = a + e * j / nb, a + e * (j + 1) / nb
            for s_ in range(nst):
                z0, z1 = s_ * H / nst, (s_ + 1) * H / nst
                tile = BT("upper", wall) if (Le / nb > 2.0 and not p.get("lod")) else R7(wall)
                m.quad([[*pa, z0 - (0.4 if s_ == 0 else 0)], [*pb, z0 - (0.4 if s_ == 0 else 0)], [*pb, z1], [*pa, z1]], tile, out=nrm)
    try:                                                                              # roof: ear-clip triangulation
        from matplotlib.tri import Triangulation
        from matplotlib.path import Path
        tr = Triangulation(fp[:, 0], fp[:, 1]); pth = Path(fp)
        for t in tr.triangles:
            if pth.contains_point(fp[t].mean(0)):
                P3 = np.c_[fp[t], np.full(3, H)]
                if np.cross(P3[1] - P3[0], P3[2] - P3[0])[2] < 0:
                    P3 = P3[[0, 2, 1]]
                u0, v0, u1, v1 = R7("roof_flat"); m.tri(P3, R7("roof_flat"), [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])
    except Exception:
        pass
    lo, hi = fp.min(0), fp.max(0)
    m.col.append(dict(shape="prism", footprint=np.round(fp, 2).tolist(), z0=-0.5, z1=H, note="building wall prism (1WALL)"))
    m.cbox((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[0] - lo[0], hi[1] - lo[1], -0.5, H)
    return m


# ============================================================================= ground decals / parking / trees
def g_parking(L, p):
    m = MB(); poly = np.asarray(p.get("polygon") or [[0, 0], [20, 0], [20, 10], [0, 10]], float)
    from matplotlib.tri import Triangulation
    from matplotlib.path import Path
    c = poly.mean(0); q = poly - c; tr = Triangulation(q[:, 0], q[:, 1]); pth = Path(q)
    for t in tr.triangles:
        if pth.contains_point(q[t].mean(0)):
            P3 = np.c_[q[t], np.full(3, 0.03)]
            if np.cross(P3[1] - P3[0], P3[2] - P3[0])[2] < 0:
                P3 = P3[[0, 2, 1]]
            m.tri(P3, R7("dark"), [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)])
    return m


def g_keep_clear(L, p):
    m = MB(); t = put("keepclear", lambda: _stencil(["KEEP", "CLEAR"]))
    m.quad([[0, -1.5, 0.02], [6, -1.5, 0.02], [6, 1.5, 0.02], [0, 1.5, 0.02]], t, out=np.array([0, 0, 1.0])); return m


def _stencil(lines):
    im = Image.new("RGBA", (256, 128), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        d.text((128, 34 + i * 60), ln, font=TX.F(54, cond=True), fill=(236, 236, 230, 255), anchor="mm")
    return im


def g_angle_parking(L, p):
    m = MB(); L = L or 10.0; a = math.radians(float(p.get("angle_deg", 45))); n = max(1, int(L / 2.7))
    for k in range(n + 1):
        x = k * L / n; dx, dy = 5.5 / math.tan(a), 5.5
        d = np.array([dx, dy]) / np.hypot(dx, dy); nn = np.array([-d[1], d[0]]) * 0.05
        p0 = np.array([x, 0.0]); p1 = p0 + np.array([dx, dy])
        m.quad([[*(p0 - nn), 0.02], [*(p1 - nn), 0.02], [*(p1 + nn), 0.02], [*(p0 + nn), 0.02]], R7("white"), out=np.array([0, 0, 1.0]))
    return m


def g_tree_row(L, p, palm=True):
    m = MB(); rng = np.random.default_rng(int(p.get("seed", 7)))
    n = int(p.get("count", 6)); sp = float(p.get("spacing_m", 9.0 if palm else 12.0))
    for k in range(n):
        h = (15 if palm else 14) * rng.uniform(0.85, 1.12); w = h * (0.5 if palm else 0.8)
        m.card(k * sp, 0, w, h, R7("palm" if palm else "plane"), n=3 if palm else 2, rot=rng.uniform(0, math.pi))
    return m


# ============================================================================= registry
P_ = AS._p
REG = []


def reg(aid, cls, gen, dims, proxy, collide, seg, linear=False, desc="", sources=("procedural",), licence="n/a (procedural, own work)", parts=()):
    AS.reg(aid, cls, gen, dims, proxy, collide, seg, linear, desc, mats=(MAT,) + tuple(parts))
    AS.CATALOG[aid].update(dev7=True, sources=list(sources), licence=licence)
    REG.append(aid)


CC0_TEX = "ambientCG CC0 1.0 (textures)"
MUTCD = ("MUTCD sign layout (US federal, public domain), redrawn",)
reg("sign.flasher_mast", "sign", g_flasher_mast, (0.3, 4.0, 6.4), dict(shape="plane", w=0.9, h=0.9, z0=4.5), True, 12,
    desc="galvanised pole + cantilever arm over the lane, warning panel (params.panel) + advisory plaque, 2 amber beacons blinking via CSP (beacon_a/beacon_b)",
    sources=MUTCD + ("ambientCG Metal032",), licence=CC0_TEX + " + public-domain MUTCD design", parts=("beacon_a", "beacon_b"))
for aid, kind, posts, w, h in (("sign.warning_text_rect", "warning_text_rect", 2, 1.2, 2.4), ("sign.reverse_turn", "reverse_turn", 1, .75, 2.1), ("sign.turn_left", "turn_left", 1, .75, 2.1),
                               ("sign.turn_right", "turn_right", 1, .75, 2.1), ("sign.side_road_left", "side_road_left", 1, .75, 2.1), ("sign.side_road_right", "side_road_right", 1, .75, 2.1),
                               ("sign.text_diamond", "text_diamond", 1, .9, 2.3), ("sign.clearance", "clearance", 1, .9, 2.3), ("sign.bicycle_warning", "bicycle_warning", 1, .75, 2.4),
                               ("sign.deer", "deer", 1, .75, 2.1), ("sign.falling_rocks", "falling_rocks", 1, .75, 2.1), ("sign.slippery", "slippery", 1, .75, 2.1),
                               ("sign.signal_ahead", "signal_ahead", 1, .75, 2.1), ("sign.stop_ahead", "stop_ahead", 1, .75, 2.1), ("sign.rr_advance", "rr_advance", 1, .9, 2.1),
                               ("sign.ped_crossing", "ped_crossing", 1, .75, 2.1), ("sign.no_parking", "no_parking", 1, .45, 2.1), ("sign.yield", "yield", 1, .9, 2.1),
                               ("sign.do_not_enter", "do_not_enter", 1, .75, 2.1), ("sign.one_way", "one_way", 1, .9, 2.1), ("sign.guide_green", "guide_green", 2, 2.5, 3.5),
                               ("sign.city_limit", "city_limit", 2, 1.2, 2.4), ("sign.route_shield", "route_shield", 1, .6, 2.3), ("sign.text_white_rect", "text_white_rect", 1, .6, 2.1)):
    reg(aid, "sign", P_(g_sign7, kind=kind, posts=posts, top=h if kind not in ("guide_green",) else 3.5), (0.06, w, h), dict(shape="plane", w=w, h=min(w, 0.9), z0=h - min(w, .9)), True, 12,
        desc=f"{kind.replace('_', ' ')} sign (MUTCD layout redrawn) on {'2 posts' if posts == 2 else 'a galvanised post'}", sources=MUTCD + ("ambientCG Metal032",), licence=CC0_TEX + " + public-domain MUTCD design")
reg("sign.store_text", "sign", g_store_text, (0.1, 3.0, 0.8), dict(shape="plane", w=3.0, h=0.8), False, 18, desc="plain business-name board (params.text, DejaVu Sans Bold, no logo)",
    sources=("DejaVu Sans Bold (free font)",), licence="own text, free font")
reg("pole.traffic_signal", "pole", g_traffic_signal, (0.4, 6.0, 6.5), dict(shape="cylinder", r=0.16, h=6.5), True, 14, desc="signal pole + mast arm + 3-section heads (sig_r/sig_y/sig_g emissive parts)",
    sources=("ambientCG Metal032", "ambientCG PaintedMetal004"), licence=CC0_TEX, parts=("sig_r", "sig_y", "sig_g"))
reg("pole.lamp_ornamental", "pole", g_lamp_ornamental, (0.3, 1.2, 4.6), dict(shape="cylinder", r=0.16, h=4.6), False, 14, desc="black acorn lamp post with banner brackets; globe = lamp_glow (CSP NIGHT_SMOOTH emissive)",
    sources=("ambientCG PaintedMetal004",), licence=CC0_TEX, parts=("lamp_glow",))
reg("prop.bench", "prop", g_bench, (1.8, 0.6, 0.9), dict(shape="box", w=1.8, d=0.6, h=0.9), False, 24, desc="Poly Haven modular_street_seating (CC0), assembled bench",
    sources=("Poly Haven modular_street_seating",), licence="Poly Haven CC0 1.0")
reg("prop.trash_can", "prop", g_trash_can, (0.6, 0.6, 1.0), dict(shape="cylinder", r=0.3, h=1.0), False, 24, desc="Poly Haven metal_trash_can (CC0), decimated",
    sources=("Poly Haven metal_trash_can",), licence="Poly Haven CC0 1.0")
reg("prop.planter", "prop", g_planter, (0.9, 0.9, 1.2), dict(shape="cylinder", r=0.45, h=0.7), False, 24, desc="round concrete planter (ambientCG concrete) with shrub card",
    sources=("ambientCG Concrete034",), licence=CC0_TEX)
reg("prop.water_barrier", "prop", g_water_barrier, (1.8, 0.5, 0.8), dict(shape="box", w=1.8, d=0.5, h=0.8), False, 24, desc="orange water-filled barrier (params.count)")
reg("prop.crash_cushion", "prop", g_crash_cushion, (3.0, 0.9, 0.9), dict(shape="box", w=3.0, d=0.9, h=0.9), False, 24, desc="yellow crash cushion, striped nose")
reg("prop.rail_crossing_signal", "prop", g_crossing_signal, (0.4, 2.0, 4.5), dict(shape="cylinder", r=0.1, h=4.5), False, 24,
    desc="crossbuck + 2 red flashers (xing_a/xing_b alternate via CSP) + bell, gate arm raised (params.gates)", sources=("ambientCG Metal032",), licence=CC0_TEX, parts=("xing_a", "xing_b"))
reg("prop.rail_crossbuck", "prop", g_crossbuck, (0.2, 1.2, 4.5), dict(shape="plane", w=1.2, h=0.4, z0=3.4), False, 24, desc="passive crossbuck on a post", sources=("ambientCG Metal032",), licence=CC0_TEX)
reg("prop.rail_platform", "prop", g_platform, (60, 4.0, 0.6), dict(shape="box", w=60, d=4, h=0.6), True, 24, desc="low station platform (params.length_m), yellow edge strip",
    sources=("ambientCG Concrete018/034",), licence=CC0_TEX)
reg("prop.rail_car_generic", "prop", g_rail_car, (24, 3.1, 4.3), dict(shape="box", w=24, d=3.1, h=4.3), True, 24, desc="generic heavyweight coach / boxcar (params.type), plain livery, no lettering",
    sources=("ambientCG PaintedMetal004", "ambientCG Rust004"), licence=CC0_TEX)
reg("rail.track", "rail", g_rail_track, (1, 4.4, 0.5), dict(shape="plane_h", w=4.4), False, 19, True, desc="standard-gauge track: ballast (ambientCG Rock030), ties, rails",
    sources=("ambientCG Rock030", "ambientCG Rust004"), licence=CC0_TEX)
reg("wall.jersey_barrier", "wall", g_jersey, (1, 0.6, 0.81), dict(shape="plane", h=0.81), True, 17, True, desc="K-rail F-shape profile, 1.5 m segments (Poly Haven model too heavy, see docs)",
    sources=("ambientCG Concrete018",), licence=CC0_TEX)
reg("wall.masonry_parapet", "wall", g_masonry_parapet, (1, 0.45, 0.9), dict(shape="plane", h=0.9), True, 17, True, desc="low parapet (params.material stone|concrete)",
    sources=("ambientCG PavingStones070 / Concrete018",), licence=CC0_TEX)
reg("wall.rock_cut_face", "wall", g_rock_cut, (1, 1.0, 10), dict(shape="plane", h=10), True, 8, True, desc="steep rock cut face (ambientCG Rock030), collision line at the foot",
    sources=("ambientCG Rock030",), licence=CC0_TEX)
reg("wall.retaining_block", "wall", g_retaining_block, (1, 0.4, 3.5), dict(shape="plane", h=3.5), True, 17, True, desc="tan split-face block wall (params.fence_on_top)",
    sources=("ambientCG Concrete018",), licence=CC0_TEX)
reg("fence.screen_curved", "fence", g_screen_curved, (1, 1.0, 3.0), dict(shape="plane", h=3.0), False, 16, True, desc="curved chain-link screen on a bridge")
reg("fence.construction_mesh", "fence", g_construction_mesh, (1, 0.05, 1.2), dict(shape="plane", h=1.2), False, 16, True, desc="orange construction mesh on T-posts")
reg("fence.picket", "fence", g_picket, (1, 0.05, 1.0), dict(shape="plane", h=1.0), False, 16, True, desc="white picket fence")
reg("bridge.parapet_balustrade", "bridge", g_balustrade, (1, 1.3, 1.1), dict(shape="plane", h=1.1), True, 19, True, desc="panelled concrete parapet on a raised sidewalk",
    sources=("ambientCG Concrete034",), licence=CC0_TEX)
reg("bridge.rail_truss_through", "bridge", g_truss_through, (5, 46, 13.0), dict(shape="box", w=5, d=46, h=13), True, 19,
    desc="steel through-truss railway span (params.span_m, clearance_m) across the road (span along y) on concrete piers",
    sources=("ambientCG PaintedMetal004", "ambientCG Concrete018"), licence=CC0_TEX)
reg("bridge.rail_deck_girder", "bridge", g_deck_girder, (4.6, 25, 7.0), dict(shape="box", w=4.6, d=25, h=7), True, 19, desc="steel plate-girder railway span(s) (params.span_m, count) on concrete piers",
    sources=("ambientCG PaintedMetal004", "ambientCG Concrete018"), licence=CC0_TEX)
reg("bridge.rail_concrete_girder", "bridge", g_concrete_girder, (8, 18, 7.0), dict(shape="box", w=8, d=18, h=7), True, 19, desc="UP concrete girder underpass, abutments, wing walls, pipe railing",
    sources=("ambientCG Concrete034",), licence=CC0_TEX)
reg("bridge.abutment_concrete", "bridge", g_abutment, (6, 3.0, 6.0), dict(shape="box", w=6, d=3, h=6), True, 19, desc="concrete abutment / pier (params.arch_opening)",
    sources=("ambientCG Concrete018",), licence=CC0_TEX)
reg("bridge.abutment_stone", "bridge", P_(g_abutment, stone=True), (6, 3.0, 6.0), dict(shape="box", w=6, d=3, h=6), True, 19, desc="cut-stone ashlar abutment",
    sources=("ambientCG PavingStones070",), licence=CC0_TEX)
reg("building.storefront", "building", g_storefront, (15, 15, 8), dict(shape="box", w=15, d=15, h=8), True, 18,
    desc="storefront box: CC0 facade (brick / stucco / siding by params.facade, colour), shop-window bays, sign_text board (DEVOUT etc.)",
    sources=("ambientCG Bricks066 / PaintedPlaster017 / WoodSiding008/011",), licence=CC0_TEX + "; own text")
reg("building.storefront_row", "building", g_storefront_row, (40, 15, 6), dict(shape="box", w=40, d=15, h=6), True, 18, desc="row of attached storefronts (params.shops, plain names)",
    sources=("ambientCG PaintedPlaster017 / WoodSiding008/011",), licence=CC0_TEX + "; own text")
reg("building.depot", "building", g_depot, (25, 8, 7), dict(shape="box", w=25, d=8, h=7), True, 18, desc="board-and-batten railway depot, wide-eave gable roof, name board",
    sources=("ambientCG WoodSiding011",), licence=CC0_TEX)
reg("building.town_generic", "building", g_town_generic, (12, 10, 6), dict(shape="box", w=12, d=10, h=6), True, 18,
    desc="OSM footprint extruded to the LiDAR p90 height (params.footprint, height_m), CC0 walls, window bays", sources=("ambientCG PaintedPlaster017 / WoodSiding008 / Bricks066", "OSM footprint (ODbL)", "USGS LiDAR height"),
    licence=CC0_TEX)
reg("driveway.parking_lot", "driveway", g_parking, (10, 20, 0.05), dict(shape="plane_h", w=20), False, 2, desc="parking lot decal from the OSM polygon (G builds the drivable surface)")
reg("marking.keep_clear", "marking", g_keep_clear, (6, 3.0, 0.02), dict(shape="plane_h", w=3), False, 4, desc="KEEP CLEAR stencil")
reg("marking.angle_parking", "marking", g_angle_parking, (1, 5.5, 0.02), dict(shape="plane_h", w=5.5), False, 4, True, desc="angled stall lines (params.angle_deg)")
reg("tree.palm_row", "tree_add", P_(g_tree_row, palm=True), (1, 6, 15), dict(shape="plane", h=15), False, 9, desc="row of date palm cards (params.count)")
reg("tree.plane_row", "tree_add", P_(g_tree_row, palm=False), (1, 10, 14), dict(shape="plane", h=14), False, 9, desc="row of plane street-tree cards (params.spacing_m)")


def generate_parts(aid, L=None, params=None):
    """{material: MB}: 'ledger7' main mesh + emissive parts (beacon_a/b, xing_a/b, sig_*, lamp_glow)"""
    e = AS.CATALOG[aid]; m = e["gen"](L if e["linear"] else None, dict(params or {}))
    out = {MAT: m}
    out.update(getattr(m, "parts", {}) or {})
    return out


EMISSIVE = {"beacon_a": (255, 176, 20), "beacon_b": (255, 176, 20), "xing_a": (230, 30, 24), "xing_b": (230, 30, 24),
            "sig_r": (220, 30, 24), "sig_y": (240, 170, 20), "sig_g": (30, 200, 110), "lamp_glow": (255, 236, 196)}


def emissive_tex(mat):
    return TX.lens(EMISSIVE[mat], 1.0, 64)


def csp_snippet():
    """ext_config.ini lines for the dev7 emissive materials (CSP; the track works without it: lenses stay unlit-coloured).
    Pattern verified against ac-custom-shaders-patch/acc-extension-config config/tracks/shibuya.ini."""
    return """
; ---- dev7 flashers (assets_dev7.py): amber wig-wag beacons, crossing lights, signal heads, lamp globes
[CONDITION_DEV7_0]
NAME=DEV7_BLINK
INPUT=ONE
DELAY=0
FLASHING_FREQUENCY=1
FLASHING_SMOOTHNESS=0.3
FLASHING_SKIP_OFF_STATE=0
FLASHING_NOISE_AMPLITUDE=0
FLASHING_MIN_VALUE=0.0
FLASHING_SYNCED=1

[CONDITION_DEV7_1]
NAME=DEV7_BLINK_ALT
INPUT=ONE
DELAY=0.5
FLASHING_FREQUENCY=-1
FLASHING_SMOOTHNESS=0.3
FLASHING_SKIP_OFF_STATE=0
FLASHING_NOISE_AMPLITUDE=0
FLASHING_MIN_VALUE=0.0
FLASHING_SYNCED=1

[MATERIAL_ADJUSTMENT_DEV7_0]
ACTIVE=1
MATERIALS=beacon_a, xing_a
KEY_0=ksEmissive
VALUE_0=60, 38, 4, 1
VALUE_0_OFF=0, 0, 0
CONDITION=DEV7_BLINK

[MATERIAL_ADJUSTMENT_DEV7_1]
ACTIVE=1
MATERIALS=beacon_b, xing_b
KEY_0=ksEmissive
VALUE_0=60, 38, 4, 1
VALUE_0_OFF=0, 0, 0
CONDITION=DEV7_BLINK_ALT

[MATERIAL_ADJUSTMENT_DEV7_2]
ACTIVE=1
MATERIALS=lamp_glow
KEY_0=ksEmissive
VALUE_0=8, 7, 5, 1
VALUE_0_OFF=0, 0, 0
CONDITION=NIGHT_SMOOTH
"""


def prefill_atlas(ledger_rows):
    """paint every text / sign face the ledger needs into the atlas (lazy tiles) before it is written"""
    for r in ledger_rows:
        if r["asset"] in AS.CATALOG and AS.CATALOG[r["asset"]].get("dev7"):
            try:
                generate_parts(r["asset"], 10.0 if AS.CATALOG[r["asset"]]["linear"] else None, ledger_params(r))
            except Exception as e:      # noqa: BLE001
                print("  prefill", r["id"], r["asset"], e)


def ledger_params(r):
    p = dict(r.get("params") or {})
    if r.get("s1") is not None:
        p.setdefault("frontage_m", abs(float(r["s1"]) - float(r["s"])))
    return p
