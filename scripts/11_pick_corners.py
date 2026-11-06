"""Pick the tightest, well-separated corners along the road for screenshots -> assets/corners.json"""
import csv, json, numpy as np
from common import *
CL = list(csv.DictReader(open(DER / "centerline.csv")))
S = np.array([float(r["s_m"]) for r in CL]); K = np.array([float(r["curvature_1pm"]) for r in CL]); Z = np.array([float(r["z_navd88"]) for r in CL])
k = np.convolve(K, np.ones(9) / 9, "same")
order = np.argsort(-np.abs(k)); picks = []
for i in order:
    if all(abs(S[i] - S[j]) > 600 for j in picks) and 60 < S[i] < S[-1] - 60: picks.append(i)
    if len(picks) == 12: break
picks = sorted(picks)
si = int(np.argmax(Z))
out = [{"name": f"corner_{n+1:02d}_s{int(S[i]):05d}", "s": float(S[i]), "radius_m": float(1 / abs(k[i])),
        "dir": "left" if k[i] > 0 else "right", "z": float(Z[i])} for n, i in enumerate(picks)]
out += [{"name": "start_sr84_s00000", "s": 30.0, "radius_m": None, "dir": "", "z": float(Z[30])},
        {"name": f"summit_s{int(S[si]):05d}", "s": float(S[si]), "radius_m": None, "dir": "", "z": float(Z[si])}]
json.dump(out, open(ASSETS / "corners.json", "w"), indent=1)
for c in out: print(c)
