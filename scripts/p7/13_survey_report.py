"""dev7 M: reports/dev7/survey.html contact sheet (key Mapillary frames with the ledger rows placed near them, per-road and
per-type counts, coverage gaps). Thumbnails are embedded at 360 px, CC-BY-SA 4.0 with creator credit. < 15 s."""
import base64, collections, csv, html, io, json, pathlib
from PIL import Image
ROOT = pathlib.Path(__file__).resolve().parents[2]; REP = ROOT / "reports/dev7"
L = [json.loads(l) for l in open(ROOT / "data/scene/ledger_dev7.jsonl")]
IDX = {r["id"]: r for r in csv.DictReader(open(ROOT / "data/derived/dev7/mapillary_index.csv"))}
IMG = ROOT / "data/raw/dev7/mapillary/img"
KEY = ("sign.flasher_mast", "bridge.", "wall.", "building.storefront", "building.depot", "prop.rail", "fence.screen", "sign.text_diamond",
       "sign.city_limit", "driveway.parking_lot", "prop.flag_pole", "guardrail.wbeam_wood", "sign.route_shield")
cards, seen = [], set()
for r in sorted(L, key=lambda r: (r["road"] != "niles_canyon_road", r["road"], r["s"])):
    if r["src"] != "mapillary" or not r["asset"].startswith(KEY) or not r["mly"]: continue
    iid = next((i for i in r["mly"] if i not in seen and (IMG / f"{i}.jpg").exists()), None)
    if not iid: continue
    ncr = sum(c[1]["road"] == "niles_canyon_road" for c in cards)
    if r["road"] == "niles_canyon_road" and ncr >= 45: continue
    seen.add(iid); cards.append((iid, r))
    if len(cards) >= 75: break
def thumb(iid):
    im = Image.open(IMG / f"{iid}.jpg").convert("RGB"); im.thumbnail((360, 270)); b = io.BytesIO(); im.save(b, "JPEG", quality=70)
    return base64.b64encode(b.getvalue()).decode()
def near(road, s, w=60):
    return [x for x in L if x["road"] == road and x["src"] == "mapillary" and abs(x["s"] - s) <= w][:8]
roads = collections.Counter(r["road"] for r in L); srcs = collections.defaultdict(collections.Counter)
for r in L: srcs[r["road"]][r["src"]] += 1
types = collections.Counter(r["asset"] for r in L)
css = """:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--card:#f4f4f6;--acc:#b35c00}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141416;--fg:#eee;--mut:#aaa;--card:#222226;--acc:#ffb35c}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px}
h1{font-size:22px}h2{font-size:17px;margin-top:28px}table{border-collapse:collapse;font-size:13px}td,th{padding:3px 8px;border-bottom:1px solid var(--card);text-align:left}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}.c{background:var(--card);border-radius:8px;padding:8px}
.c img{width:100%;border-radius:4px}.c b{color:var(--acc)}.c ul{margin:4px 0 0 16px;padding:0;font-size:12px}.m{color:var(--mut);font-size:12px}
.wrap{overflow-x:auto}"""
h = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>dev7 Mapillary survey</title><style>{css}</style>",
     "<h1>dev7 Mapillary survey (package M)</h1>",
     f"<p>{len(L)} ledger rows in <code>data/scene/ledger_dev7.jsonl</code> (parts <code>dev7_mapillary.jsonl</code>, <code>dev7_town_lidar.jsonl</code>). "
     "Station <code>s</code> is metres along <code>data/derived/dev7/centerline_&lt;road&gt;.csv</code>; side L/R relative to increasing s; off = metres beyond the pavement edge. "
     "Images: Mapillary contributors, CC-BY-SA 4.0 (full list in <code>mapillary_attribution.csv</code>).</p>",
     "<h2>Rows per road</h2><div class=wrap><table><tr><th>road</th><th>rows</th><th>mapillary</th><th>osm</th><th>lidar</th></tr>"]
for rd, n in roads.most_common():
    h.append(f"<tr><td>{rd}</td><td>{n}</td><td>{srcs[rd]['mapillary']}</td><td>{srcs[rd]['osm']}</td><td>{srcs[rd]['lidar']}</td></tr>")
h.append("</table></div><h2>Rows per asset type</h2><div class=wrap><table><tr><th>asset</th><th>rows</th></tr>")
for a, n in types.most_common(): h.append(f"<tr><td>{a}</td><td>{n}</td></tr>")
h.append("</table></div><h2>Key frames and the rows placed from them</h2><div class=grid>")
for iid, r in cards:
    ix = IDX.get(iid, {})
    li = "".join(f"<li>{html.escape(x['asset'])} s={x['s']:.0f}{'' if x['s1'] is None else '-%.0f' % x['s1']} {x['side']} off {x['off']} <span class=m>({x['conf']})</span></li>" for x in near(r["road"], r["s"]))
    h.append(f"<div class=c><img loading=lazy src='data:image/jpeg;base64,{thumb(iid)}' alt=''><div><b>{html.escape(r['asset'])}</b> {r['road']} s={r['s']:.0f} {r['side']}</div>"
             f"<div class=m>{html.escape(r['note'][:160])}</div><ul>{li}</ul>"
             f"<div class=m>image {iid}, {ix.get('creator','')} {ix.get('captured_at','')[:10]}, CC-BY-SA 4.0</div></div>")
h.append("</div>")
h.append(open(ROOT / "reports/dev7/survey_gaps.html").read() if (ROOT / "reports/dev7/survey_gaps.html").exists() else "")
(REP / "survey.html").write_text("\n".join(h))
print("wrote reports/dev7/survey.html:", len(cards), "cards")
