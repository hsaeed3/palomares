#!/usr/bin/env python3
"""dev7 I: freeroam / freeroam_track layouts (whole network, no timing gates) in $BUILD/palomares_road.

  PALOMARES_BUILD_DIR=build_dev7 python3 scripts/p7i/freeroam_layouts.py     (about 10 s)

Files: the shipped Palomares kn5 (trees, barriers, roadside; NOT palomares_road/terrain/lines_road: G's junction-cut
copies replace them), the dev7 cut copies, Niles town + NCR + Sunol clusters (road + terrain), NCR assets and town
assets; freeroam_track adds every kerb in its main-layout direction (Palomares nb = from the Niles Canyon start,
NCR nb = from the Mission Blvd start) and the Palomares GP dressing. The Palomares kn5 come from build/ at package
time (scripts/p7i/package_dev7.py); here they are only counted (read-only) for the draw-call budget.
Spawn: Niles Blvd one block before Devout Coffee (s 92 -> 60, eastbound right lane), AC_START_0..3 / AC_PIT_0..3 /
AC_HOTLAP_START_0, no AB gates. AI: fast_lane.ai along NCR Mission Blvd -> Sunol (one continuous line; joining it to
Palomares through the junction is not done).
"""
import json, os, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/p2"), str(ROOT / "scripts/p7i"), str(ROOT / "scripts/p7g")]
os.environ.setdefault("PALOMARES_BUILD_DIR", str(ROOT / "build_dev7"))
import canyon_layouts as CL                                   # noqa: E402  (NCR road, racing line, helpers)
import network as NW                                          # noqa: E402
from kn5 import Dummy, write_kn5, read_kn5, ac_from_local     # noqa: E402
import ai_spline, racing_line as RL                           # noqa: E402
from PIL import Image, ImageDraw                              # noqa: E402

TRACK = CL.TRACK; PAL = ROOT / "build/palomares_road"
PAL_SHARED = ["palomares_trees.kn5", "palomares_barriers.kn5", "palomares_roadside.kn5"]
DEV7 = ["palomares_road_dev7.kn5", "palomares_terrain_dev7.kn5", "palomares_lines_road_dev7.kn5",
        "palomares_niles_town.kn5", "palomares_niles_town_terrain.kn5", "palomares_niles_ncr.kn5", "palomares_niles_ncr_terrain.kn5",
        "palomares_niles_sunol.kn5", "palomares_niles_sunol_terrain.kn5", "palomares_niles_lines.kn5", "palomares_niles_assets.kn5", "palomares_niles_town_assets.kn5"]
TRACKX = ["palomares_kerbs_nb.kn5", "palomares_dressing.kn5", "palomares_dressing_nb.kn5", "palomares_niles_kerbs_nb.kn5"]
LAYOUTS = {"freeroam": [], "freeroam_track": TRACKX}
DRAW_BUDGET = 1500


def spawns():
    r = NW.load_road("niles_blvd"); nodes = []
    for k, s in enumerate((92, 84, 76, 68)):
        i = int(s); d = -min(1.7, r.er[i] * 0.5)                 # eastbound (s increasing) right lane
        p = np.array([r.x[i] + r.nx[i] * d, r.y[i] + r.ny[i] * d, r.surface_z(i, d, noise=False) + 0.05])
        f = np.array([r.tx[i], r.ty[i], 0.0]); U = ac_from_local(np.array([0, 0, 1.0])); Z = ac_from_local(f / np.linalg.norm(f))
        X = np.cross(U, Z); X /= np.linalg.norm(X); t = ac_from_local(p)
        M = [float(v) for v in list(X) + [0] + list(U) + [0] + list(Z) + [0] + list(t) + [1]]
        nodes += [Dummy(f"AC_START_{k}", M), Dummy(f"AC_PIT_{k}", M)] + ([Dummy("AC_HOTLAP_START_0", M)] if k == 0 else [])
    f = TRACK / "palomares_niles_timing_freeroam.kn5"; write_kn5(f, [], {}, nodes)
    return f.name


def draws(f):
    p = TRACK / f if (TRACK / f).exists() else PAL / f
    if not p.exists():
        return None
    return sum(-(-len(m["pos"]) // 65535) for m in read_kn5(str(p))["meshes"]
               if m.get("renderable", True) and m.get("visible", True) and not m["name"][:1].isdigit() and len(m["pos"]))


def main():
    tk = spawns(); SURF = (ROOT / "build_dev7/palomares_road/canyon_road/data/surfaces.ini").read_text(); rep = {}
    W, H, pts, MAPINI = CL.mapimg()
    for lay, extra in LAYOUTS.items():
        files = PAL_SHARED + DEV7 + [tk] + extra
        missing = [f for f in files if not (TRACK / f).exists() and not (PAL / f).exists()]
        dc = {f: draws(f) for f in files}; tot = sum(v for v in dc.values() if v)
        (TRACK / f"models_{lay}.ini").write_text("".join(f"[MODEL_{i}]\nFILE={f}\nPOSITION=0,0,0\nROTATION=0,0,0\n\n" for i, f in enumerate(files)))
        dd = TRACK / lay / "data"; dd.mkdir(parents=True, exist_ok=True)
        (dd / "surfaces.ini").write_text(SURF); (dd / "lighting.ini").write_text("[LIGHTING]\nSUN_PITCH_ANGLE=45\nSUN_HEADING_ANGLE=200\n")
        (dd / "cameras.ini").write_text(CL.cams(dict(s_from=CL.S_A, s_to=CL.S_B, dir="nb"))); (dd / "map.ini").write_text(MAPINI)
        (dd / "groove.ini").write_text("[HEADER]\nGROOVES_NUMBER=0\n"); (dd / "audio_sources.ini").write_text("")
        mp = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(mp).line(pts, fill=(255, 210, 60, 255), width=8, joint="curve"); mp.save(TRACK / lay / "map.png")
        ui = TRACK / "ui" / lay; ui.mkdir(parents=True, exist_ok=True)
        name = f"Free roam{' (Track)' if 'track' in lay else ''} — Niles, Niles Canyon, Palomares v{CL.VERSION}"
        desc = ("Free roam over the whole map: Niles town (Niles Blvd, the depot and the railroad), Niles Canyon Road from Mission Blvd to Sunol "
                "and the full Palomares Road. No timing. Spawn on Niles Blvd before Devout Coffee."
                + (" With every race kerb and the Palomares GP dressing." if "track" in lay else "") + " " + CL.ATTR)
        json.dump({"name": name, "description": desc, "tags": ["free roam", "usa", "canyon", "touge", "lidar"], "geotags": ["37.5760 N", "121.9740 W"],
                   "country": "USA", "city": "Fremont / Sunol, CA", "length": "37000", "width": "5-12", "pitboxes": "4", "run": "free roam",
                   "author": "palomares pipeline", "version": CL.VERSION, "url": ""}, open(ui / "ui_track.json", "w"), indent=2, ensure_ascii=False)
        ol = mp.copy(); ol.thumbnail((1000, 1000)); ol.save(ui / "outline.png")
        pv = Image.new("RGB", (355, 200), (24, 40, 30)); m2 = mp.copy(); m2.thumbnail((330, 180)); pv.paste(m2, ((355 - m2.width) // 2, (200 - m2.height) // 2), m2)
        ImageDraw.Draw(pv).text((8, 6), name[:48], fill=(240, 240, 240)); pv.save(ui / "preview.png")
        rep[lay] = dict(files=files, missing=missing, draw_calls=tot, per_kn5=dc, budget_ok=tot < DRAW_BUDGET)
    RL.load = lambda: CL.RLD
    ai = ai_spline.export_layouts(TRACK, {k: dict(s_from=CL.S_A, s_to=CL.S_B) for k in LAYOUTS}, CL.road)
    for k in rep:
        rep[k]["ai"] = ai[k]
    (CL.BUILD / "dev7/freeroam_layouts.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: dict(draw_calls=v["draw_calls"], missing=v["missing"], ai=v["ai"]["points"]) for k, v in rep.items()}, indent=1))


if __name__ == "__main__":
    main()
