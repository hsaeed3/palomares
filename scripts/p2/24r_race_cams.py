"""Close-up cameras for the race event assets (race_assets.py) -> build/vis/<layout>/scene_race.json.
Run after 24_visual_export.py <layout>; then `node scripts/p2/24r_race_render.js <layout>` (-> reports/visual/race_assets/)."""
import json, pickle, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from geom import BUILD, Road

lay = sys.argv[1] if len(sys.argv) > 1 else "track_full"
geo = pickle.load(open(BUILD / "geometry.pkl", "rb")); meta = geo["meta"]
road = Road(meta["s_end"]); N = len(road.s)


def ac(x, y, z): return [float(x), float(z), float(-y)]


def road_pt(s, d, h):
    i = int(np.clip(round(s), 0, N - 1)); x, y = road.xy(i, d)
    return ac(x, y, road.surface_z(i, d, noise=False) + h)


cams = []
for st in meta.get("race_sites", []):
    s, side, tgt = st["s"], st["side"], ac(st["x"], st["y"], st["z"] + 1.4)
    big = st["L"] > 10
    for sd, tag in ((1, "nb"), (-1, "sb")) if big else ((1, "nb"),):
        back = 35 if big else 16
        s_e = s - sd * back if 2 < s - sd * back < N - 2 else s + sd * back
        cams.append(dict(s=f"{st['label']}_{tag}", eye=road_pt(s_e, -side * 1.6, 1.15), look=tgt))     # driver's eye, other lane
    if big:                                                                                          # high 3/4 overview
        i = int(np.clip(round(s), 0, N - 1)); x, y = road.xy(i, -side * (road.outer(i, -side) + 2))
        cams.append(dict(s=f"{st['label']}_overview", eye=ac(x + 0, y, st["z"] + 9.0), look=ac(st["x"], st["y"], st["z"] + 1.0)))
    i = int(np.clip(round(s), 0, N - 1)); x, y = road.xy(i, side * (road.outer(i, side) + 0.5))          # close-up from the road edge
    d = np.array([st["x"] - x, st["y"] - y]); dist = np.hypot(*d); d /= max(dist, 1e-6)
    t = np.array([road.tx[i], road.ty[i]]); e = np.array([x, y]) - t * (6.0 if big else 3.0)
    cams.append(dict(s=f"{st['label']}_close", eye=ac(*e, max(st["z"], road.z[i]) + (2.6 if big else 1.6)), look=tgt))
out = BUILD / "vis" / lay / "scene_race.json"
sc = json.load(open(BUILD / "vis" / lay / "scene.json")); sc["cams"] = cams
json.dump(sc, open(out, "w"))
print(len(cams), "cams ->", out); print("\n".join(c["s"] for c in cams))
