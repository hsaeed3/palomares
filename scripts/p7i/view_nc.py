#!/usr/bin/env python3
"""dev7 I: driver's-eye render of the built Niles Canyon kn5s (road + assets + kerbs) at station s, beside the nearest
ledger row's Mapillary image.  python3 scripts/p7i/view_nc.py <s> [--dir nb|sb] -> reports/view/nc_<s>.jpg"""
import io, json, math, os, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT / "scripts/p2"), str(ROOT / "scripts/p7g")]
import view, network as NW, dds
from kn5 import read_kn5
from PIL import Image
s0 = float(sys.argv[1]); sb = "--dir" in sys.argv and sys.argv[sys.argv.index("--dir") + 1] == "sb"
TR = pathlib.Path(os.environ.get("PALOMARES_BUILD_DIR", ROOT / "build_dev7")) / "palomares_road"
RN = sys.argv[sys.argv.index("--road") + 1] if "--road" in sys.argv else "niles_canyon_road"
road = NW.load_road(RN); i = int(s0); sd = -1 if sb else 1
ie = int(np.clip(i - sd * 45, 0, len(road.s) - 1)); lane = 1.6 * (1 if sb else -1)
eye = np.array([road.x[ie] + road.nx[ie] * lane, road.y[ie] + road.ny[ie] * lane, road.z[ie] + 1.25])
tgt = np.array([road.x[i] + road.nx[i] * lane, road.y[i] + road.ny[i] * lane, road.z[i] + 1.0])
P, T, C = [], [], []; n = 0
for f in (["palomares_niles_ncr.kn5", "palomares_niles_ncr_terrain.kn5", "palomares_niles_assets.kn5", f"palomares_niles_kerbs_{'sb' if sb else 'nb'}.kn5"] if RN == "niles_canyon_road" else ["palomares_niles_town.kn5", "palomares_niles_town_terrain.kn5", "palomares_niles_lines.kn5", "palomares_niles_town_assets.kn5"]):
    k = read_kn5(str(TR / f)); tex = {}
    for m in k["meshes"]:
        if not m.get("renderable", True) or not len(m["pos"]):
            continue
        Pm = np.asarray(m["pos"], float); Pm = np.c_[Pm[:, 0], -Pm[:, 2], Pm[:, 1]]
        Tm = np.asarray(m["idx"]).reshape(-1, 3); c = Pm[Tm].mean(1)
        sel = np.hypot(c[:, 0] - tgt[0], c[:, 1] - tgt[1]) < 260
        if not sel.any():
            continue
        Tm = Tm[sel]; mat = k["materials"][m["material"]] if isinstance(m["material"], int) else next(mm for mm in k["materials"] if mm["name"] == m["material"])
        tn = mat["textures"].get("txDiffuse")
        if tn not in tex:
            tex[tn] = Image.open(io.BytesIO(dds.to_png(k["textures"][tn]))) if tn in k["textures"] else Image.new("RGBA", (4, 4), (128, 128, 128, 255))
        UV = np.asarray(m["uv"], float); UV = np.c_[UV[:, 0], -UV[:, 1]]
        col, keep = view.tri_colors(UV, Tm, tex[tn], mat.get("alpha_tested", False) or "AT" in mat.get("shader", ""))
        P.append(Pm); T.append(Tm[keep] + n); C.append(col[keep]); n += len(Pm)
P = np.concatenate(P); T = np.concatenate(T); C = np.concatenate(C).astype(float)
tiles = [("driver", view.render(P, T, C, eye, tgt, fov=60, far=400))]
rows = [json.loads(l) for l in open(ROOT / "data/scene/ledger_dev7.jsonl")]
r = min((r for r in rows if r["road"] == RN and r.get("mly")), key=lambda r: abs(r["s"] - s0))
q = ROOT / f"data/raw/dev7/mapillary/img/{r['mly'][0]}.jpg"
if q.exists():
    tiles.append((f"Mapillary {r['mly'][0]} ({r['asset']} s {r['s']:.0f})", Image.open(q)))
print(view.strip(tiles, view.OUT / f"{RN[:6]}_{int(s0)}{'_sb' if sb else ''}.jpg", f"Niles Canyon built kn5 at s {s0:.0f} ({'sb' if sb else 'nb'})"))
