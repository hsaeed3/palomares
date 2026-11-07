"""Round-trip check + plots of every <layout>/ai/fast_lane.ai -> reports/racing_line/ai_check/
  summary.json : per layout spacing, length monotonicity, ids, extras ranges, height above the asphalt, plan jitter
  <layout>_cNN.png : 4 tightest corners, parsed points over the asphalt edges (each point = arrow anchor)"""
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import ROOT, BUILD, CFG, Road
from kn5 import ac_from_local
import ai_spline as A

OUT = ROOT / "reports/racing_line/ai_check"; OUT.mkdir(parents=True, exist_ok=True)
road = Road(CFG["build_end_m"]); I = np.arange(len(road.s))
xl, yl = road.xy(I, road.el); xr, yr = road.xy(I, -road.er)
inv = np.linalg.inv(ac_from_local(np.eye(3)))           # AC -> local (linear axis swap)
lays = sys.argv[1:] or sorted(p.parent.name for p in (BUILD / "palomares_road").glob("*/ai/fast_lane.ai"))
summ = {}
for lay in lays:
    r = A.read(BUILD / "palomares_road" / lay / "ai" / "fast_lane.ai")
    P = r["P"].astype(float) @ inv.T; e = r["extra"]
    d = np.linalg.norm(np.diff(r["P"].astype(float), axis=0), axis=1)
    # height above the noiseless asphalt at the nearest station / offset
    from scipy.spatial import cKDTree
    st = cKDTree(np.c_[road.x, road.y]).query(P[:, :2])[1]
    off = (P[:, 0] - road.x[st]) * road.nx[st] + (P[:, 1] - road.y[st]) * road.ny[st]
    hz = P[:, 2] - road.surface_z(st, off, noise=False)
    t = np.diff(P[:, :2], axis=0); h = np.unwrap(np.arctan2(t[:, 1], t[:, 0])); dh = np.diff(h)
    summ[lay] = dict(points=int(r["n"]), length_m=float(r["length"][-1]), spacing_min=float(d.min()), spacing_max=float(d.max()),
                     length_monotone=bool(np.all(np.diff(r["length"]) > 0)), length_vs_chords_m=float(abs(r["length"][-1] - d.sum())),
                     ids_ok=bool(np.array_equal(r["id"], np.arange(r["n"]))), extra_count_ok=len(e["speed"]) == r["n"],
                     height_above_asphalt_m=[float(np.percentile(hz, 1)), float(np.median(hz)), float(np.percentile(hz, 99))],
                     inside_edges=bool(np.all((off < road.el[st] + 0.05) & (off > -road.er[st] - 0.05))),
                     curvature_sign_flips_per_km=float((np.diff(np.sign(dh)) != 0).sum() / r["length"][-1] * 1000),
                     direction_values=sorted(set(np.unique(e["direction"]).tolist())),
                     speed_kmh=[float(e["speed"].min() * 3.6), float(e["speed"].max() * 3.6)],
                     side_l=[float(e["side_l"].min()), float(e["side_l"].max())], side_r=[float(e["side_r"].min()), float(e["side_r"].max())],
                     normal_y_min=float(e["ny"].min()), extra_length=[float(e["length"].min()), float(e["length"].max())])
    if lay.startswith("road_full"):
        k = 1 / np.maximum(e["radius"], 1); cand = np.argsort(-k)
        picks = []
        for c in cand:
            if all(abs(c - p) > 300 for p in picks): picks.append(int(c))
            if len(picks) == 4: break
        for j, c in enumerate(sorted(picks)):
            sl = slice(max(c - 60, 0), c + 60); fig, ax = plt.subplots(figsize=(7, 7))
            s0 = st[sl]; ss = slice(max(s0.min() - 5, 0), s0.max() + 5)
            ax.plot(xl[ss], yl[ss], "k-", lw=1.5, label="asphalt edges"); ax.plot(xr[ss], yr[ss], "k-", lw=1.5)
            ax.plot(road.x[ss], road.y[ss], color="0.7", lw=0.8, ls="--", label="centerline")
            ax.plot(P[sl, 0], P[sl, 1], "-", color="tab:red", lw=1, label="fast_lane.ai (parsed)")
            ax.plot(P[sl, 0], P[sl, 1], ".", color="tab:red", ms=3)
            ax.set_aspect("equal"); ax.legend(loc="best", fontsize=8)
            ax.set_title(f"{lay}: s~{road.s[st[c]]:.0f} m, R {e['radius'][c]:.0f} m, {e['speed'][c]*3.6:.0f} km/h, every point 1 m")
            fig.savefig(OUT / f"{lay}_c{j}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(lay, json.dumps(summ[lay]))
json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
