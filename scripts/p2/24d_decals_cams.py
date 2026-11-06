#!/usr/bin/env python3
"""Camera list for the decal / southbound-sign visual check (reports/visual/decals): GoPro frame stations + 3 southbound signs.
Writes build/vis/<layout>/cams_decals.json (read by 24d_decals_render.js, port 8871)."""
import csv, json, sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
lay = sys.argv[1] if len(sys.argv) > 1 else "road_summit"
rows = list(csv.DictReader(open(ROOT / "data/derived/centerline.csv")))
X = np.array([[float(r["x_local"]), float(r["y_local"]), float(r["z_navd88"])] for r in rows])
def tang(i):
    t = X[min(i + 2, len(X) - 1), :2] - X[max(i - 2, 0), :2]; return t / np.hypot(*t)
cams = []
gp = {}
for f in sorted((ROOT / "data/derived").glob("gopro_stations_*.csv")):
    for r in csv.DictReader(open(f)):
        gp[int(r["frame"])] = r
for fr in (4, 19, 30, 104, 164):
    r = gp[fr]; s = float(r["s_m"]); lat = float(r["lateral_m"]); hh = float(r["height_m"])
    i = int(round(s)); t = tang(i); n = np.array([-t[1], t[0]])
    e = X[i, :2] + n * lat; ra = X[min(i + 25, len(X) - 1)]
    la = ra[:2] + n * lat
    cams.append(dict(s=f"gp{fr:03d}_s{int(s):05d}", eye=[e[0], X[i, 2] + hh, -e[1]], look=[la[0], ra[2] + 0.5, -la[1]], fr=fr))
sg = [q for q in json.load(open(ROOT / "build/roadside_signs.json")) if not q["nb"] and not q["key"].startswith("back")]
pick = []
for kind in ("dia_", "chev", "mph", "stop"):
    c = [q for q in sg if q["key"].startswith(kind)]
    if kind == "chev" and c: c = [q for q in c if q["s"] > 7000] or c
    if c: pick.append(c[len(c) // 2] if kind != "stop" else c[0])
for q in pick[:3] if len(pick) >= 3 else pick:
    P = np.array(q["P"]); ctr = P.mean(0); s = q["s"]; i = min(int(round(s + 32)), len(X) - 1); t = tang(i); n = np.array([-t[1], t[0]])
    e = X[i, :2] + n * 1.7                                   # SB driver's lane = NB-left (+d), looking toward decreasing s
    cams.append(dict(s=f"sb_{q['key']}_s{int(s):05d}", eye=[e[0], X[i, 2] + 1.15, -e[1]], look=[ctr[0], ctr[2] + 0.1, -ctr[1]]))
json.dump(dict(cams=cams), open(ROOT / "build/vis" / lay / "cams_decals.json", "w"), indent=1)
print(len(cams), "cams:", [c["s"] for c in cams])
