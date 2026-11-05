"""Phase 2 visual check, step 1: read the built kn5 files BACK (round trip) and export a compact
scene for the headless three.js renderer -> build/vis/<layout>/scene.bin + scene.json + textures."""
import json, struct, sys, pathlib, shutil, csv
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import ROOT, BUILD
from kn5 import read_kn5

TRACK = BUILD / "palomares_road"
for lay in ("road_summit", "track_summit"):
    out = BUILD / "vis" / lay
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    files = [l.split("=", 1)[1].strip() for l in (TRACK / f"models_{lay}.ini").read_text().splitlines() if l.startswith("FILE=")]
    meshes, mats, blob = [], [], bytearray()
    for fn in files:
        k = read_kn5(TRACK / fn)
        base = len(mats)
        for m in k["materials"]:
            tex = m["textures"].get("txDiffuse")
            if tex and not (out / tex).exists():
                (out / tex).write_bytes(k["textures"][tex])
            mats.append(dict(name=m["name"], tex=tex, alpha=bool(m["alpha_tested"])))
        for m in k["meshes"]:
            if m["name"].startswith("1WALL") or m["material"] + base >= len(mats) or mats[m["material"] + base]["name"] == "invisible":
                continue
            pos = m["pos"].astype("<f4"); uv = m["uv"].astype("<f4"); nrm = m["nrm"].astype("<f4"); idx = m["idx"].astype("<u4")
            ent = dict(name=m["name"], mat=m["material"] + base, nv=len(pos), ni=len(idx), off=len(blob))
            for a in (pos, nrm, uv, idx):
                blob += a.tobytes()
            meshes.append(ent)
    (out / "scene.bin").write_bytes(bytes(blob))
    # camera stations from the centerline (driver's eye, right lane)
    rows = list(csv.DictReader(open(ROOT / "data/derived/centerline.csv")))
    cams = []
    for s in list(range(0, 7392, 250)) + [7385]:
        s = max(s, 14)
        r = rows[s]; ra = rows[s + 25]
        x, y, z = float(r["x_local"]), float(r["y_local"]), float(r["z_navd88"])
        tx, ty = float(rows[s + 1]["x_local"]) - x, float(rows[s + 1]["y_local"]) - y
        n = (tx * tx + ty * ty) ** .5; tx, ty = tx / n, ty / n
        ex, ey = x + ty * 1.7, y - tx * 1.7   # right lane
        lx, ly, lz = float(ra["x_local"]) + ty * 1.7, float(ra["y_local"]) - tx * 1.7, float(ra["z_navd88"]) + 0.8
        cams.append(dict(s=s, eye=[ex, z + 1.15, -ey], look=[lx, lz, -ly]))
    json.dump(dict(meshes=meshes, materials=mats, cams=cams), open(out / "scene.json", "w"))
    print(lay, len(meshes), "meshes", len(blob) / 1e6, "MB")
