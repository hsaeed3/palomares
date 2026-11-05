"""reports/visual_check.html: driver's-eye renders (both layouts) every 250 m + USGS ortho crop for location.
Mapillary has no coverage on Palomares Rd (checked by the user), so no street photos column."""
import csv, base64, io, sys, pathlib, json
import numpy as np, rasterio
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import ROOT, DER
st = list(csv.DictReader(open(ROOT / "reports/visual_check_stations.csv")))
cl = list(csv.DictReader(open(DER / "centerline.csv")))
o = json.load(open(ROOT / "config/origin.json"))
ort = rasterio.open(DER / "ortho_r100.tif")

def crop(s):
    r = cl[s]; X, Y = float(r["x_utm"]), float(r["y_utm"])
    row, col = ort.index(X, Y)
    w = rasterio.windows.Window(col - 100, row - 100, 200, 200)
    a = ort.read([1, 2, 3], window=w, boundless=True).transpose(1, 2, 0)
    im = Image.fromarray(a.astype(np.uint8)).resize((240, 240))
    from PIL import ImageDraw
    ImageDraw.Draw(im).ellipse([114, 114, 126, 126], outline=(255, 0, 0), width=3)
    b = io.BytesIO(); im.save(b, "JPEG", quality=80); return base64.b64encode(b.getvalue()).decode()

rows = []
for r in st:
    s = max(int(r["station_m"]), 14)
    cells = [f"<td><b>{s} m</b><br>{r['z_navd88']} m<br><small>{r['lat']}, {r['lon']}<br>hdg {r['heading_deg_true']}°</small></td>"]
    for lay in ("road_summit", "track_summit"):
        cells.append(f"<td><img src='visual/{lay}/s{s:05d}.jpg' width=480></td>")
    ref = ROOT / "reports/visual/reference" / f"ref_s{int(r['station_m'])}.png"
    cells.append(f"<td><img src='visual/reference/{ref.name}' width=480></td>" if ref.exists() else "<td><small>no street-level photo<br>(no Mapillary coverage)</small></td>")
    cells.append(f"<td><img src='data:image/jpeg;base64,{crop(s)}'><br><small>USGS ortho, 200 m</small></td>")
    rows.append("<tr>" + "".join(cells) + "</tr>")
html = f"""<!doctype html><html><head><meta charset=utf-8><title>Palomares visual check</title>
<style>body{{font-family:sans-serif;background:#111;color:#ddd}}td{{vertical-align:top;padding:4px}}img{{border-radius:4px}}</style></head><body>
<h2>Phase 2 visual check — driver's eye every 250 m (renders of the round-tripped kn5)</h2>
<p>Columns: station · road_summit · track_summit · street reference (drop ref_s&lt;station&gt;.png into reports/visual/reference/) · USGS ortho.</p>
<table><tr><th>station</th><th>road_summit</th><th>track_summit</th><th>reference</th><th>ortho</th></tr>{''.join(rows)}</table></body></html>"""
(ROOT / "reports/visual_check.html").write_text(html)
print("wrote reports/visual_check.html", len(rows), "rows")
