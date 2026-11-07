"""dev6 trees + spring grass before/after sheet -> reports/dev6/trees_grass.html
GoPro photo | litmus render before (reports/dev6/trees_grass/before, saved before the change) | after (reports/litmus).
Images are copied as JPEG next to the page so it stays light."""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/dev6"; D = OUT / "trees_grass"; B = D / "before"; A = D / "after"
A.mkdir(parents=True, exist_ok=True)
LIT = ROOT / "reports/litmus"
rows = []
for g in sorted(B.glob("render_road_full_*.png")):
    f = g.stem.split("_")[-1]
    after = LIT / g.name
    if not after.exists():
        continue
    cells = []
    for tag, src in (("gopro", B / f"{f}.jpg"), ("before", g), ("after", after)):
        if not src.exists():
            cells.append(""); continue
        im = Image.open(src).convert("RGB")
        if tag == "gopro" and im.width > 1.9 * im.height * 16 / 9:   # litmus jpg = photo | render pair: keep the photo half
            im = im.crop((0, 0, im.width // 2, im.height))
        im.thumbnail((960, 540)); p = D / f"{tag}_{f}.jpg"; im.save(p, quality=85)
        cells.append(p.relative_to(OUT).as_posix())
    rows.append((f, cells))
html = ["<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Trees and grass</title><style>:root{--bg:#fff;--fg:#222}@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#ddd}}"
        "body{background:var(--bg);color:var(--fg);font:14px system-ui;margin:16px}table{border-collapse:collapse;width:100%}"
        "td{padding:3px;vertical-align:top}img{width:100%;display:block}</style></head><body>",
        "<h1>dev6 trees + spring grass</h1><p>GoPro frame | before (painted leaf cards, khaki terrain) | after (photographic "
        "CC0 leaf clusters, conifers / bare sycamores, spring-green terrain). three.js preview renders, not AC/CSP: no "
        "GrassFX, no ksTree translucency or wind.</p><table><tr><th>frame</th><th>GoPro</th><th>before</th><th>after</th></tr>"]
for f, c in rows:
    html.append(f"<tr><td>{f}</td>" + "".join(f"<td><img src='{x}' loading=lazy></td>" if x else "<td></td>" for x in c) + "</tr>")
html.append("</table></body></html>")
(OUT / "trees_grass.html").write_text("\n".join(html))
print("wrote", OUT / "trees_grass.html", len(rows), "rows")
