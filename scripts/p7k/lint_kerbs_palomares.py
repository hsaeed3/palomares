#!/usr/bin/env python3
"""dev7 K lint, Palomares: kerb / run-off meshes may sit on the shoulder but never on the asphalt (|offset| >= el / er).
Reads the built kn5 (read-only):  python3 scripts/p7k/lint_kerbs_palomares.py [--track build/palomares_road]
Exit 1 on a hit; writes reports/dev7/lint_kerbs_palomares.json."""
import argparse, json, pathlib, sys
import numpy as np
from scipy.spatial import cKDTree
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts/p2"))
from geom import Road, CFG
from kn5 import read_kn5
ap = argparse.ArgumentParser(); ap.add_argument("--track", default=str(ROOT / "build/palomares_road")); ap.add_argument("--tol", type=float, default=0.02)
a = ap.parse_args()
road = Road(CFG["build_end_m"]); kd = cKDTree(np.c_[road.x, road.y]); hits = []; n = 0
for f in sorted(pathlib.Path(a.track).glob("palomares_kerbs_*.kn5")):
    for m in read_kn5(str(f))["meshes"]:
        if not len(m["pos"]):
            continue
        P = np.asarray(m["pos"], float); p = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)
        if "idx" in m and len(m["idx"]):
            p = p[np.unique(np.asarray(m["idx"]))]
        _, j = kd.query(p[:, :2]); n += len(p)
        off = (p[:, 0] - road.x[j]) * road.nx[j] + (p[:, 1] - road.y[j]) * road.ny[j]
        half = np.where(off > 0, road.el[j], road.er[j])
        pen = half - a.tol - np.abs(off)
        if (pen > 0).any():
            k = int(np.argmax(pen)); hits.append(dict(kn5=f.name, mesh=m["name"], s=float(road.s[j[k]]), n=int((pen > 0).sum()), penetration_m=round(float(pen[k]), 3)))
json.dump(dict(verts=n, hits=hits), open(ROOT / "reports/dev7/lint_kerbs_palomares.json", "w"), indent=1)
print(f"Palomares kerb lint: {n} verts, {len(hits)} meshes on the asphalt"); [print(" ", h) for h in hits[:20]]
sys.exit(1 if hits else 0)
