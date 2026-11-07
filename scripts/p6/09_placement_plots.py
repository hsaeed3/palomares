#!/usr/bin/env python3
"""dev6 check-by-looking: top-down plot over NAIP 2022 for every 1 km of station.
reports/dev6/placement_NN.jpg : measured buildings (footprint, colour by form), detected fence/wall lines,
understorey shrubs/hedges, and the ledger rows (placed = solid marker, rejected = x).
Run: nice -n5 python3 scripts/p6/09_placement_plots.py [--km 3,4]"""
import sys, json, pathlib, argparse
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import DER, ROOT, origin
import rasterio, rasterio.windows
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

O = origin(); X0, Y0 = O["x0"], O["y0"]
OUT = ROOT / "reports/dev6"; OUT.mkdir(parents=True, exist_ok=True)
cl = np.genfromtxt(DER / "centerline.csv", delimiter=",", names=True)
FORMC = dict(flat="#00e5ff", shed="#ffd400", gable="#ff3b30", hip="#ff00d4", tank="#ffffff")
LC = dict(fence="#ffd400", wall="#ff7b00", guardrail="#00e5ff", hedge="#3cff3c")


def lj(p):
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def ledger_xy(r):
    import sys as _s; _s.path.insert(0, str(ROOT / "scripts/p5")); import scene
    x, y, z, h = scene.ledger_place(r); return x, y


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--km", default=None); a = ap.parse_args()
    B = json.load(open(DER / "buildings.geojson"))["features"] if (DER / "buildings.geojson").exists() else []
    F = json.load(open(DER / "fences_detected.geojson"))["features"] if (DER / "fences_detected.geojson").exists() else []
    U = json.load(open(DER / "understory.geojson"))["features"] if (DER / "understory.geojson").exists() else []
    L = lj(ROOT / "data/scene/ledger.jsonl")
    naip = rasterio.open(DER / "naip2022_r060.tif")
    kms = [int(k) for k in a.km.split(",")] if a.km else range(int(cl["s_m"][-1] // 1000) + 1)
    for km in kms:
        m = (cl["s_m"] >= km * 1000 - 20) & (cl["s_m"] <= km * 1000 + 1020)
        xs, ys = cl["x_utm"][m], cl["y_utm"][m]
        x0, x1, y0, y1 = xs.min() - 120, xs.max() + 120, ys.min() - 120, ys.max() + 120
        win = rasterio.windows.from_bounds(x0, y0, x1, y1, naip.transform)
        img = naip.read([1, 2, 3], window=win, boundless=True, fill_value=0, out_shape=(3, int(win.height / 1.5), int(win.width / 1.5)))
        fig, ax = plt.subplots(figsize=(12, 12 * (y1 - y0) / (x1 - x0) if (y1 - y0) < 2 * (x1 - x0) else 20), dpi=110)
        ax.imshow(np.moveaxis(img, 0, -1), extent=(x0, x1, y0, y1)); ax.plot(xs, ys, "w-", lw=0.6, alpha=0.7)
        for s in range(km * 1000, km * 1000 + 1001, 100):
            j = min(int(s), len(cl) - 1); ax.text(cl["x_utm"][j], cl["y_utm"][j], f"{s}", color="w", fontsize=7)
        for f in B:
            p = f["properties"]; c = np.asarray(f["geometry"]["coordinates"][0])
            if c[:, 0].max() < x0 or c[:, 0].min() > x1 or c[:, 1].max() < y0 or c[:, 1].min() > y1: continue
            ax.plot(c[:, 0], c[:, 1], "-", color=FORMC.get(p["form"], "w"), lw=1.2)
            ax.text(c[:, 0].mean(), c[:, 1].mean(), f'{p["ridge_h"]:.0f}', color="w", fontsize=5)
        for f in F:
            c = np.asarray(f["geometry"]["coordinates"])
            if c[:, 0].max() < x0 or c[:, 0].min() > x1 or c[:, 1].max() < y0 or c[:, 1].min() > y1: continue
            ax.plot(c[:, 0], c[:, 1], "-", color=LC.get(f["properties"].get("cls", "fence"), "y"), lw=1.6)
        for f in U:
            g = f["geometry"]; p = f["properties"]
            if g["type"] == "Point":
                x, y = g["coordinates"]
                if x0 < x < x1 and y0 < y < y1: ax.plot(x, y, "o", ms=1.5, color="#7CFC00" if p["cls"] == "shrub" else "#c08040", alpha=0.6)
            else:
                c = np.asarray(g["coordinates"])
                if x0 < c[0, 0] < x1 and y0 < c[0, 1] < y1: ax.plot(c[:, 0], c[:, 1], "-", color="#3cff3c", lw=2.5, alpha=0.6)
        for r in L:
            if not (km * 1000 - 20 <= r["s"] <= km * 1000 + 1020): continue
            if r["cls"] not in ("building", "fence", "wall", "guardrail", "hedge", "shrub", "gate"): continue
            try: x, y = ledger_xy(r)
            except Exception: continue
            x = x + X0; y = y + Y0; rj = r.get("status") == "rejected"; meas = r.get("src") in ("lidar", "ortho")
            col = "#ff4040" if rj else ("#40ff40" if meas else "#ff40ff")
            if len(x) > 1 and not rj: ax.plot(x, y, ":", color=col, lw=1.0)
            ax.plot(x[0], y[0], "x" if rj else ("s" if r["cls"] == "building" else "^"), color=col, ms=4)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"km {km}: buildings (cyan flat, yellow shed, red gable, magenta hip, white tank; label ridge h m), "
                     f"lines (yellow fence, orange wall, cyan rail, green hedge), ledger: green measured / magenta eyeballed / red x rejected", fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / f"placement_{km:02d}.jpg", dpi=110, pil_kwargs=dict(quality=82)); plt.close(fig)
        print("wrote", OUT / f"placement_{km:02d}.jpg")


if __name__ == "__main__":
    main()
