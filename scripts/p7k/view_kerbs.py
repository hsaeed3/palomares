#!/usr/bin/env python3
"""dev7 K: driver's-eye + overview render of the Niles Canyon kerb set around one run-off (reports/view/kerbs_<layout>_<s>.jpg)
  python3 scripts/p7k/view_kerbs.py [canyon_track] [--s 1234]"""
import json, math, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT / "scripts/p2"), str(ROOT / "scripts/p7g")]
import view, network as NW
lay = next((a for a in sys.argv[1:] if not a.startswith("--") and not a[0].isdigit()), "canyon_track")
z = np.load(ROOT / f"data/scene/kerbs_dev7/{lay}.npz"); rep = json.load(open(ROOT / "data/scene/kerbs_dev7/placements.json"))
dname = rep["layouts"][lay]; items = rep["items"][dname]
s_c = float(sys.argv[sys.argv.index("--s") + 1]) if "--s" in sys.argv else next((it["s0"] for it in items if it["type"] == "sausage"), items[0]["s0"])
road = NW.load_road(rep["road"]); i0 = int(np.searchsorted(road.s, s_c - 90)); i1 = int(np.searchsorted(road.s, s_c + 90))
COL = {"kerb_rbr": None, "kerb_sausage": (236, 200, 30), "runoff_green": (42, 140, 70)}
P, T, C = [], [], []; n = 0
ii = np.arange(i0, i1); d = np.linspace(-1, 1, 9)
for k, dd in enumerate(d):
    pass
off = np.stack([np.interp(d, [-1, 1], [-road.outer(i, "right"), road.outer(i, "left")]) for i in ii])
x, y = road.xy(ii[:, None], off); zz = road.surface_z(ii[:, None], off, noise=False)
P.append(np.stack([x, y, zz], -1).reshape(-1, 3)); from geom import grid_tris
Tr = grid_tris(len(ii), 9); T.append(Tr); C.append(np.tile((70, 70, 72), (len(Tr), 1))); n = len(P[0])
for key in [k[:-3] for k in z.files if k.endswith("__P")]:
    Pk = z[key + "__P"].astype(float); Tk = z[key + "__T"]; mat = str(z[key + "__mat"])
    j = np.searchsorted(road.s, 0)
    if not (np.hypot(Pk[:, 0] - road.x[(i0 + i1) // 2], Pk[:, 1] - road.y[(i0 + i1) // 2]).min() < 120):
        continue
    if COL[mat] is None:
        V = z[key + "__UV"][:, 1]; cc = np.where((np.floor(V[Tk].mean(1) * 2) % 2)[:, None] == 0, (198, 30, 34), (236, 234, 228))
    else:
        cc = np.tile(COL[mat], (len(Tk), 1))
    P.append(Pk); T.append(Tk + n); C.append(cc); n += len(Pk)
P = np.concatenate(P); T = np.concatenate(T); C = np.concatenate(C).astype(float)
ic = (i0 + i1) // 2; sd = 1 if dname == "nb" else -1; ie = ic - sd * 40
eye = np.array([road.x[ie], road.y[ie], road.z[ie] + 1.2]) - np.array([road.nx[ie], road.ny[ie], 0]) * 1.5
tiles = [("driver", view.render(P, T, C, eye, [road.x[ic], road.y[ic], road.z[ic]], fov=60)),
         ("overview", view.render(P, T, C, [road.x[ic] + 25, road.y[ic] - 25, road.z[ic] + 45], [road.x[ic], road.y[ic], road.z[ic]], fov=60))]
print(view.strip(tiles, view.OUT / f"kerbs_{lay}_{int(s_c)}.jpg", f"{lay} ({dname}) around s {s_c:.0f}: red/white kerbs, green run-off, yellow sausage"))
