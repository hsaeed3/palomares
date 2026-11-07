"""Hard gate: no visible non-road geometry may sit on the drivable asphalt.

Checks every renderable mesh in every kn5 of the built track (except the road surface, its paint lines and timing
gates) for vertices inside the asphalt (|offset| < half-width - 5 cm) between 2 cm and 4.5 m above the road.
Terrain is checked too: terrain must stay BELOW the road surface inside the asphalt (no terrain poking through).
Fast (< 1 min): reads the kn5 files only.  Writes reports/qa_on_road.json, exit 1 on any hit.

  python3 scripts/p2/qa_on_road.py [--track build/palomares_road]
"""
import argparse, json, sys
import numpy as np
from scipy.spatial import cKDTree
from geom import ROOT, BUILD, CFG, Road
from kn5 import read_kn5

ap = argparse.ArgumentParser(); ap.add_argument("--track", default=str(BUILD / "palomares_road"))
ap.add_argument("--road", default=None, help="dev7 network road (scripts/p7g/network.py); default Palomares")
ap.add_argument("--kn5", default="*.kn5", help="kn5 glob inside --track"); a = ap.parse_args()
import pathlib
TRK = pathlib.Path(a.track)
if a.road:
    sys.path.insert(0, str(ROOT / "scripts/p7g")); import network as _NW
    road = _NW.load_road(a.road)
else:
    road = Road(CFG["build_end_m"])
NS = len(road.s)
ckd = cKDTree(np.c_[road.x, road.y])
SKIP = ("_road.kn5", "lines", "timing")
hits = []
for f in sorted(p.name for p in TRK.glob(a.kn5)):
    if any(k in f for k in SKIP):
        continue
    for m in read_kn5(str(TRK / f))["meshes"]:
        if not len(m["pos"]) or m["name"][:1].isdigit() and "WALL" not in m["name"]:
            continue                                   # physics-only drivable meshes (1ROAD/1KERB/1GRASS) are fine
        P = np.asarray(m["pos"], float); p = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)
        d, j = ckd.query(p[:, :2])
        off = (p[:, 0] - road.x[j]) * road.nx[j] + (p[:, 1] - road.y[j]) * road.ny[j]
        half = np.where(off > 0, road.el[j], road.er[j])
        zr = road.surface_z(j, off, noise=False)
        ok_j = (j > 2) & (j < NS - 3) & (d < half + 1.0)
        inside = ok_j & (np.abs(off) < half - 0.05)
        if "terrain" in f:
            bad = inside & (p[:, 2] > zr + 0.02)       # terrain poking through the asphalt
        else:
            bad = inside & (p[:, 2] > zr + 0.02) & (p[:, 2] < zr + 4.5)
        used = np.zeros(len(p), bool); used[np.asarray(m["idx"], int).ravel()] = True
        bad &= used                                     # vertices no triangle references are invisible
        if bad.any():
            ss = sorted({int(road.s[x] // 25 * 25) for x in j[bad]})
            hits.append({"kn5": f, "mesh": m["name"], "verts": int(bad.sum()), "stations": ss[:20], "n_stations": len(ss)})
out = ROOT / "reports" / ("qa_on_road.json" if not a.road else f"dev7/qa_on_road_{a.road}.json")
json.dump({"hits": hits, "n": len(hits)}, open(out, "w"), indent=1)
for h in hits[:60]:
    print(f"ON ROAD {h['kn5']}:{h['mesh']} {h['verts']} verts at s={h['stations'][:6]}{'...' if h['n_stations'] > 6 else ''}")
print(f"qa_on_road: {len(hits)} meshes with geometry on the asphalt -> {out}")
sys.exit(1 if hits else 0)
