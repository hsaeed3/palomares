#!/usr/bin/env python3
"""dev7 I: Niles Canyon layouts in the dev7 track folder ($BUILD/palomares_road, never build/).

  PALOMARES_BUILD_DIR=build_dev7 python3 scripts/p7i/canyon_layouts.py      (about 20 s)

Layouts (docs/dev7/PLAN.md): canyon_road / canyon_track = Mission Blvd -> Sunol (nb, s increasing),
canyon_rev_road / canyon_rev_track = Sunol -> Mission Blvd (sb). Road = G's NCR road + terrain + our roadside
assets; track adds the RBR kerbs / run-off / sausage kerbs of that direction. Writes per layout:
models_<layout>.ini, palomares_niles_timing_<layout>.kn5 (AC_START_0, AC_PIT_0, AC_HOTLAP_START_0, AC_AB_START_L/R,
AC_AB_FINISH_L/R), <layout>/data/{surfaces,lighting,cameras,map,groove,audio_sources}.ini, <layout>/map.png,
<layout>/ai/fast_lane.ai (ai_spline from the racing line, round-trip checked), ui/<layout>/{ui_track.json,
preview.png, outline.png}; and the CSP flasher block in extension/ext_config.ini (between markers).
"""
import json, math, os, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/p2"), str(ROOT / "scripts/p5"), str(ROOT / "scripts/p7g")]
os.environ.setdefault("PALOMARES_BUILD_DIR", str(ROOT / "build_dev7"))
import network as NW, racing_line as RL, ai_spline                   # noqa: E402
from kn5 import Dummy, write_kn5, read_kn5, ac_from_local            # noqa: E402
from PIL import Image, ImageDraw                                     # noqa: E402

BUILD = pathlib.Path(os.environ["PALOMARES_BUILD_DIR"]); TRACK = BUILD / "palomares_road"
assert BUILD.resolve() != (ROOT / "build").resolve()
VERSION = (ROOT / "VERSION").read_text().strip() if (ROOT / "VERSION").exists() else "dev7"
ROADN = "niles_canyon_road"
road = NW.load_road(ROADN); N = len(road.s)
RL.CACHE = ROOT / "data/scene/cache/racing_line_niles_canyon_road.pkl"; RLD = RL.compute(road); C = RL.cfg()
S_A, S_B = 70.0, float(road.s[-1]) - 90.0            # clear of the Mission Blvd junction / the Sunol stop
# G (ae99896): dev7 layouts use the junction-cut Palomares copies, never the shipped palomares_terrain/road/lines_road.kn5
BASEF = ["palomares_road_dev7.kn5", "palomares_terrain_dev7.kn5", "palomares_lines_road_dev7.kn5",
         "palomares_niles_town.kn5", "palomares_niles_town_terrain.kn5",
         "palomares_niles_ncr.kn5", "palomares_niles_ncr_terrain.kn5", "palomares_niles_assets.kn5"]
LAYOUTS = {
    "canyon_road": dict(s_from=S_A, s_to=S_B, dir="nb", style="Road"),
    "canyon_track": dict(s_from=S_A, s_to=S_B, dir="nb", style="Track"),
    "canyon_rev_road": dict(s_from=S_B, s_to=S_A, dir="sb", style="Road"),
    "canyon_rev_track": dict(s_from=S_B, s_to=S_A, dir="sb", style="Track"),
}
ATTR = ("Data: OpenStreetMap contributors (ODbL); USGS 3DEP LiDAR CA_AlamedaCounty_2021 and USDA NAIP (public domain); "
        "street-level reference from Mapillary contributors (CC-BY-SA 4.0); textures and models ambientCG and Poly Haven (CC0). "
        "Business names are plain text only. Fan-made, not affiliated with Caltrans or Alameda County.")


def at(s):
    i = int(np.clip(round(s), 0, N - 1)); return i


def loc(name, i, d, fwd, up=0.0):
    p = np.array([road.x[i] + road.nx[i] * d, road.y[i] + road.ny[i] * d, road.surface_z(i, d, noise=False) + up])
    f = np.array([road.tx[i], road.ty[i], 0.0]) * fwd; f /= np.linalg.norm(f)
    U = ac_from_local(np.array([0, 0, 1.0])); Z = ac_from_local(f); X = np.cross(U, Z); X /= np.linalg.norm(X)
    t = ac_from_local(p)
    return Dummy(name, [float(v) for v in list(X) + [0] + list(U) + [0] + list(Z) + [0] + list(t) + [1]])


def timing(lay, L):
    sd = 1 if L["dir"] == "nb" else -1; ia, ib = at(L["s_from"]), at(L["s_to"])
    ispawn = at(L["s_from"] - sd * 12)
    # driver's left in run direction = station-left for nb, station-right for sb
    def lr(i):
        l = road.el[i] - 0.3; r = road.er[i] - 0.3
        return (l, -r) if sd > 0 else (-r, l)
    nodes = []
    for nm, i in (("AB_START", ia), ("AB_FINISH", ib)):
        dl, dr = lr(i); nodes += [loc(f"AC_{nm}_L", i, dl, sd, 0.5), loc(f"AC_{nm}_R", i, dr, sd, 0.5)]
    lane = -sd * min(1.6, road.er[ispawn] * 0.5) if sd > 0 else min(1.6, road.el[ispawn] * 0.5)   # right-hand lane
    for nm in ("AC_START_0", "AC_PIT_0", "AC_HOTLAP_START_0"):
        nodes.append(loc(nm, ispawn, lane, sd, 0.05))
    f = TRACK / f"palomares_niles_timing_{lay}.kn5"; write_kn5(f, [], {}, nodes)
    return f.name, ispawn


def mapimg():
    x, y = road.x, road.y; ax, az = x, -y
    SC, M = 4.0, 20
    W = int((ax.max() - ax.min()) / SC + 2 * M); H = int((az.max() - az.min()) / SC + 2 * M)
    XO = -ax.min() + M * SC; ZO = -az.min() + M * SC
    pts = [((a + XO) / SC, (b + ZO) / SC) for a, b in zip(ax[::4], az[::4])]
    ini = f"[PARAMETERS]\nWIDTH={W}\nHEIGHT={H}\nMARGIN={M}\nSCALE_FACTOR={SC}\nX_OFFSET={XO:.3f}\nZ_OFFSET={ZO:.3f}\nDRAWING_SIZE=10\nMAX_SIZE=1600\n"
    return W, H, pts, ini


def cams(L):
    sd = 1 if L["dir"] == "nb" else -1; a, b = sorted((L["s_from"], L["s_to"])); run = b - a; out = []; k = 0
    for s in np.arange(a + 80, b - 20, 450):
        i = at(s); fr = (s - a) / run if sd > 0 else (b - s) / run
        side = 9.0; p = np.array([road.x[i] - road.nx[i] * side, road.y[i] - road.ny[i] * side, road.z[i] + 4])
        fwd = np.array([road.nx[i], road.ny[i], -0.25]); fwd /= np.linalg.norm(fwd); P = ac_from_local(p); F = ac_from_local(fwd)
        out.append(f"[CAMERA_{k}]\nNAME=cam{k}\nPOSITION={P[0]:.2f},{P[1]:.2f},{P[2]:.2f}\nFORWARD={F[0]:.3f},{F[1]:.3f},{F[2]:.3f}\nUP=0,1,0\nMIN_FOV=10\nMAX_FOV=50\n"
                   f"IN_POINT={max(0, fr - 225 / run):.4f}\nOUT_POINT={min(1, fr + 225 / run):.4f}\nSHADOW_SPLIT0=50\nSHADOW_SPLIT1=200\nSHADOW_SPLIT2=600\nNEAR_PLANE=1\nFAR_PLANE=5000\n"
                   "MIN_EXPOSURE=0\nMAX_EXPOSURE=10000\nDOF_FACTOR=10\nDOF_RANGE=10000\nDOF_FOCUS=0\nDOF_MANUAL=0\nSPLINE=\nSPLINE_ROTATION=0\nFOV_GAMMA=0\nSPLINE_ANIMATION_LENGTH=0\nIS_FIXED=0\n")
        k += 1
    return f"[HEADER]\nVERSION=3\nCAMERA_COUNT={k}\nSET_NAME=NilesCanyon\n\n" + "\n".join(out)


def ext_config():
    import assets_dev7 as D7
    f = TRACK / "extension/ext_config.ini"; f.parent.mkdir(parents=True, exist_ok=True)
    txt = f.read_text() if f.exists() else "[BASIC]\nIGNORE_OTHER_CONFIGS=0\n"
    a, b = "; >>> dev7 flashers (scripts/p7i/canyon_layouts.py)", "; <<< dev7 flashers"
    if a in txt:
        txt = txt[:txt.index(a)] + txt[txt.index(b) + len(b) + 1:]
    f.write_text(txt.rstrip() + "\n\n" + a + D7.csp_snippet() + b + "\n")
    return str(f.relative_to(TRACK))


def main():
    surf = (ROOT / "build_dev7/palomares_road/freeroam/data/surfaces.ini")
    SURF = surf.read_text() if surf.exists() else (ROOT / "build/palomares_road/track_full/data/surfaces.ini").read_text()
    extra = (ROOT / "config/dev7/surfaces_dev7.ini").read_text()          # G: SIDEWALK (+ RAIL) after SURFACE_0..4
    if "KEY=SIDEWALK" not in SURF:
        SURF = SURF.rstrip() + "\n\n" + extra.strip() + "\n"
    W, H, pts, MAPINI = mapimg(); rep = {}
    missing = [f for f in BASEF if not (TRACK / f).exists()]
    assert not missing, f"missing kn5: {missing}"
    for lay, L in LAYOUTS.items():
        tk, ispawn = timing(lay, L)
        files = BASEF + [tk] + ([f"palomares_niles_kerbs_{L['dir']}.kn5"] if L["style"] == "Track" else [])
        (TRACK / f"models_{lay}.ini").write_text("".join(f"[MODEL_{i}]\nFILE={f}\nPOSITION=0,0,0\nROTATION=0,0,0\n\n" for i, f in enumerate(files)))
        dd = TRACK / lay / "data"; dd.mkdir(parents=True, exist_ok=True)
        (dd / "surfaces.ini").write_text(SURF); (dd / "lighting.ini").write_text("[LIGHTING]\nSUN_PITCH_ANGLE=45\nSUN_HEADING_ANGLE=200\n")
        (dd / "cameras.ini").write_text(cams(L)); (dd / "map.ini").write_text(MAPINI)
        (dd / "groove.ini").write_text("[HEADER]\nGROOVES_NUMBER=0\n"); (dd / "audio_sources.ini").write_text("")
        mp = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(mp)
        d.line(pts, fill=(255, 210, 60, 255), width=8, joint="curve")
        for s_, col in ((L["s_from"], (40, 220, 60, 255)), (L["s_to"], (230, 40, 40, 255))):
            px, pz = pts[min(int(s_) // 4, len(pts) - 1)]; d.ellipse([px - 9, pz - 9, px + 9, pz + 9], fill=col)
        mp.save(TRACK / lay / "map.png")
        ui = TRACK / "ui" / lay; ui.mkdir(parents=True, exist_ok=True)
        rev = L["dir"] == "sb"; name = f"Niles Canyon Rd — {'Sunol → Mission Blvd' if rev else 'Mission Blvd → Sunol'} ({L['style']}) v{VERSION}"
        desc = ("Niles Canyon Road (SR-84), Fremont to Sunol, point to point" + (" in reverse" if rev else "") + ". "
                + ("The road as it really is." if L["style"] == "Road" else "Same road and markings, plus red/white kerbs, green run-off and yellow sausage kerbs on corner exits.")
                + " " + ATTR)
        json.dump({"name": name, "description": desc, "tags": ["hillclimb", "point to point", "usa", "canyon", "lidar"], "geotags": ["37.5940 N", "121.9370 W"],
                   "country": "USA", "city": "Fremont / Sunol, CA", "length": f"{abs(L['s_to'] - L['s_from']):.0f}", "width": "7-12", "pitboxes": "1",
                   "run": "point to point", "author": "palomares pipeline", "version": VERSION, "url": ""}, open(ui / "ui_track.json", "w"), indent=2, ensure_ascii=False)
        ol = mp.copy(); ol.thumbnail((1000, 1000)); ol.save(ui / "outline.png")
        pv = Image.new("RGB", (355, 200), (24, 40, 30)); m2 = mp.copy(); m2.thumbnail((330, 180)); pv.paste(m2, ((355 - m2.width) // 2, (200 - m2.height) // 2), m2)
        ImageDraw.Draw(pv).text((8, 6), name[:48], fill=(240, 240, 240)); pv.save(ui / "preview.png")
        rep[lay] = dict(files=files, spawn_s=float(road.s[ispawn]))
    RL.load = lambda: RLD
    ai = ai_spline.export_layouts(TRACK, {k: dict(s_from=v["s_from"], s_to=v["s_to"]) for k, v in LAYOUTS.items()}, road)
    for k in rep:
        rep[k]["ai"] = ai[k]
    rep["ext_config"] = ext_config()
    (BUILD / "dev7").mkdir(exist_ok=True); (BUILD / "dev7/canyon_layouts.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1)[:2000])


if __name__ == "__main__":
    main()
