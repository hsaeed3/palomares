#!/usr/bin/env python3
"""dev7 A: contact sheet of the dev7 asset library -> reports/dev7/assets.html (previews = view.py renders beside
the ledger row's Mapillary images; sources / licences from assets/<type>/manifest.json)"""
import html, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts/p5"))
import assets_dev7 as D
k = json.load(open(ROOT / "data/scene/kerbs_dev7/placements.json"))
rows = []
for aid in D.REG:
    m = json.loads((ROOT / f"assets/{aid}/manifest.json").read_text())
    col = json.loads((ROOT / f"assets/{aid}/collision.json").read_text())
    rows.append(f"""<div class=c><h3>{html.escape(aid)}</h3><a href="../../assets/{aid}/preview.png"><img src="../../assets/{aid}/preview.png" loading=lazy></a>
<p>{html.escape(m.get('desc', ''))}</p><p><b>Sources:</b> {html.escape('; '.join(s for s in m.get('sources', []) if not s.startswith('licence')))}<br><b>Licence:</b> {html.escape(str(m.get('licence')))}</p>
<p>{m['tris']} tris LOD0, LODs {len(m['lods'])}, materials {', '.join(p['material'] for p in m['parts'])}; collision: {'yes' if col['collide'] else 'none (small prop)'};
ledger row {m.get('ledger_row')}, Mapillary {', '.join(m.get('mly', [])) or '-'}</p></div>""")
cnt = "".join(f"<tr><td>{l}</td>" + "".join(f"<td>{v}</td>" for v in c.values()) + "</tr>" for l, c in k["counts"].items())
page = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>dev7 Assets</title>
<style>:root{{--bg:#fff;--fg:#1c1c1c;--card:#f3f3f1}}@media (prefers-color-scheme:dark){{:root{{--bg:#161616;--fg:#e8e8e6;--card:#232323}}}}
body{{background:var(--bg);color:var(--fg);font:14px system-ui;margin:0 16px}}.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}}
.c{{background:var(--card);padding:8px;border-radius:6px}}.c img{{width:100%}}h3{{margin:4px 0;font-size:15px}}table{{border-collapse:collapse}}td,th{{border:1px solid #888;padding:3px 8px}}</style>
<h1>dev7 asset library ({len(D.REG)} types)</h1>
<p>Each card: 4 views from <code>view.py asset</code> (textured renderer) beside the Mapillary images (CC-BY-SA) of the ledger row the asset was sized from.
Rebuild: <code>python3 scripts/p2/assetlib.py --dev7</code>. Sources: ambientCG and Poly Haven (CC0 1.0), MUTCD layouts (public domain), own text in DejaVu Sans.</p>
<h2>Row renders</h2><p><img src="asset_row_devout.jpg" style="max-width:100%"><br><img src="asset_row_firehouse.jpg" style="max-width:100%"></p>
<h2>Niles Canyon kerbs</h2><table><tr><th>layout</th><th>apex</th><th>exit</th><th>run-off</th><th>sausage</th></tr>{cnt}</table>
<p>Max crease (deg): {json.dumps({a: round(b, 1) for a, b in k['max_crease_deg'].items()})}, limit 25. On-pavement lint hits: {len(k['on_pavement'])}.</p>
<p><img src="kerbs_canyon_track_1434.jpg" style="max-width:100%"></p>
<h2>Assets</h2><div class=g>{''.join(rows)}</div>"""
(ROOT / "reports/dev7/assets.html").write_text(page); print("wrote reports/dev7/assets.html")
