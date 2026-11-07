#!/usr/bin/env python3
"""dev6 step 06: Mapillary street-level images along the corridor (CC-BY-SA 4.0, (c) Mapillary contributors).
Token: env MAPILLARY_TOKEN or MAPILARY_TOKEN (never printed). Image list from graph.mapillary.com /images by small
bboxes along the centreline; keeps images within 40 m of the road. Writes
  data/derived/mapillary_images.geojson  (id, captured_at, compass_angle, is_pano, creator, s, offset, sequence)
  data/cache/p6/mapillary/<id>.jpg       (thumb_1024, at most one per 10 m of station per direction; gitignored)
Idempotent: list cached in data/cache/p6/mapillary_list.json; downloaded images skipped."""
import os, sys, json, time, pathlib, numpy as np, requests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import DER, CACHE, T_LL_UTM
from scipy.spatial import cKDTree
TOK = os.environ.get("MAPILLARY_TOKEN") or os.environ.get("MAPILARY_TOKEN")
if not TOK: sys.exit("no Mapillary token (MAPILLARY_TOKEN / MAPILARY_TOKEN)")
H = {"Authorization": f"OAuth {TOK}"}
D = CACHE / "p6/mapillary"; D.mkdir(parents=True, exist_ok=True); LST = CACHE / "p6/mapillary_list.json"
cl = np.genfromtxt(DER / "centerline.csv", delimiter=",", names=True)
F = "id,geometry,computed_geometry,captured_at,compass_angle,computed_compass_angle,is_pano,sequence,creator,thumb_1024_url"
if LST.exists(): imgs = json.load(open(LST))
else:
    imgs = {}
    for j in range(0, len(cl), 120):
        la, lo = cl["lat"][j], cl["lon"][j]; d = 0.0012
        url = f"https://graph.mapillary.com/images?fields={F}&bbox={lo-d:.6f},{la-d:.6f},{lo+d:.6f},{la+d:.6f}&limit=2000"
        for k in range(4):
            try:
                r = requests.get(url, headers=H, timeout=60); r.raise_for_status(); break
            except Exception as e:
                time.sleep(3 + 5 * k); r = None
        if r is None: print("fail at", j); continue
        for im in r.json().get("data", []): imgs[im["id"]] = im
        print(j, len(imgs), flush=True)
    json.dump(imgs, open(LST, "w"))
KD = cKDTree(np.c_[cl["x_utm"], cl["y_utm"]]); TX = np.gradient(cl["x_utm"]); TY = np.gradient(cl["y_utm"])
feats, pick = [], {}
for im in imgs.values():
    g = (im.get("computed_geometry") or im["geometry"])["coordinates"]; x, y = T_LL_UTM.transform(*g)
    dd, j = KD.query([x, y])
    if dd > 40: continue
    off = float(-TY[j] * (x - cl["x_utm"][j]) + TX[j] * (y - cl["y_utm"][j]))
    ca = im.get("computed_compass_angle", im.get("compass_angle")) or 0.0
    hd = np.degrees(np.arctan2(TX[j], TY[j])) % 360; nb = abs((ca - hd + 180) % 360 - 180) < 90
    p = dict(id=im["id"], captured_at=time.strftime("%Y-%m-%d", time.gmtime(im["captured_at"] / 1000)), compass=round(ca, 1),
             is_pano=im.get("is_pano", False), creator=(im.get("creator") or {}).get("username", ""), sequence=im.get("sequence"),
             s=float(cl["s_m"][j]), offset=round(off, 1), dir="nb" if nb else "sb", x_utm=round(x, 2), y_utm=round(y, 2))
    feats.append(dict(type="Feature", geometry=dict(type="Point", coordinates=list(g)), properties=p))
    key = (int(p["s"] // 10), p["dir"])
    if key not in pick or p["captured_at"] > pick[key][0]["captured_at"]: pick[key] = (p, im.get("thumb_1024_url"))
json.dump(dict(type="FeatureCollection", features=feats, license="CC-BY-SA 4.0, Mapillary contributors"), open(DER / "mapillary_images.geojson", "w"))
n = 0
for p, u in pick.values():
    f = D / f"{p['id']}.jpg"
    if f.exists() or not u: continue
    try:
        r = requests.get(u, timeout=60); r.raise_for_status(); f.write_bytes(r.content); n += 1
    except Exception as e: print("dl fail", p["id"], type(e).__name__)
yrs = sorted({p["properties"]["captured_at"][:4] for p in feats})
cov = len({int(p["properties"]["s"] // 100) for p in feats})
print(f"{len(imgs)} listed, {len(feats)} within 40 m, {len(pick)} picked, {n} downloaded; years {yrs}; 100 m bins covered {cov}/{int(cl['s_m'][-1]//100)+1}")
