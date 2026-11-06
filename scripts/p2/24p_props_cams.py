"""Close-up cameras for the kerb / race-prop check (track layouts) -> build/vis/<layout>/scene_props.json.
Run after 24_visual_export.py <layout>; then `node scripts/p2/24p_props_render.js <layout>`
(renders to reports/visual/track_props/<label>.jpg)."""
import json, pickle, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from geom import BUILD, Road

lay = sys.argv[1] if len(sys.argv) > 1 else "track_full"
geo = pickle.load(open(BUILD / "geometry.pkl", "rb")); meta = geo["meta"]
road = Road(meta["s_end"]); N = len(road.s)
sd = 1 if lay.endswith(("summit", "full", "north")) and not lay.endswith("_rev") else -1
dname = "nb" if sd > 0 else "sb"


def P(s, d, h):
    i = int(np.clip(round(s), 0, N - 1)); x, y = road.xy(i, d)
    return [float(x), float(road.surface_z(i, d, noise=False) + h), float(-y)]


def cam(label, s_eye, d_eye, h_eye, s_look, d_look, h_look=0.0):
    return dict(s=label, eye=P(s_eye, d_eye, h_eye), look=P(s_look, d_look, h_look))


cams = []
K = [k for k in meta["kerbs"] if k["dir"] == dname]
for typ in ("apex", "exit"):
    ks = sorted([k for k in K if k.get("type") == typ], key=lambda k: -(k["s1"] - k["s0"]))[:2]
    for k in ks:
        sgn = 1 if k["side"] == "left" else -1
        i = int((k["s0"] + k["s1"]) / 2); e = road.el[i] if sgn > 0 else road.er[i]
        s_in = k["s0"] if sd > 0 else k["s1"]
        # low driver's eye in the lane next to the kerb, 10 m before it, looking at its middle
        cams.append(cam(f"{typ}_{dname}_s{int(k['s0']):05d}_driver", s_in - sd * 10, sgn * (e - 1.4), 0.95, (k["s0"] + k["s1"]) / 2, sgn * (e + 0.3)))
        # very low close-up of the kerb's leading taper and the road-edge joint
        cams.append(cam(f"{typ}_{dname}_s{int(k['s0']):05d}_joint", s_in - sd * 2.5, sgn * (e - 0.6), 0.35, s_in + sd * 4, sgn * (e + 0.3)))
for line in meta["dressing"]:
    w = line.split()
    if w[0] in ("tyre_belt", "tecpro", "jersey"):
        a, b = map(float, w[1][2:].split("-")); sgn = 1 if w[2] == "L" else -1
        i = int((a + b) / 2); e = road.el[i] if sgn > 0 else road.er[i]
        cams.append(cam(f"barrier_{w[0]}_s{int(a):05d}", a - 12, -sgn * 1.0, 1.1, (a + b) / 2, sgn * (e + 1.2), 0.5))
    elif w[0] == "boards" and w[1] == dname and len([c for c in cams if c["s"].startswith("boards")]) < 2:
        s_e = float(w[4].split("=")[1])
        cams.append(cam(f"boards_{dname}_s{int(s_e):05d}", s_e - sd * 130, 0.0, 1.15, s_e - sd * 75, 0.0, 0.6))
    elif w[0] == "marshal" and len([c for c in cams if c["s"].startswith("marshal")]) < 2:
        s_m = float(w[3].split("=")[1]); sgn = 1 if w[4] == "L" else -1
        i = int(s_m); e = road.el[i] if sgn > 0 else road.er[i]
        cams.append(cam(f"marshal_s{int(s_m):05d}", s_m - sd * 18, -sgn * 1.0, 1.15, s_m, sgn * (e + 3.0), 0.8))
for g in (20, 7392, 15563):
    cams.append(cam(f"gate_s{g:05d}", g - sd * 30 if 30 < g < N - 30 else g + 30 * (1 if g < 100 else -1), -1.7, 1.15, g, 0.0, 3.5))
cams.append(cam("bleacher_s07431", 7400, -1.7, 1.2, 7431, -10.0, 1.0))
out = BUILD / "vis" / lay / "scene_props.json"
sc = json.load(open(BUILD / "vis" / lay / "scene.json")); sc["cams"] = cams
json.dump(sc, open(out, "w"))
print(len(cams), "cams ->", out); print("\n".join(c["s"] for c in cams))
