#!/usr/bin/env python3
"""dev7 D-07: Mapillary image index along every dev7 road (CC-BY-SA 4.0, (c) Mapillary contributors). No downloads.
Token: env MAPILLARY_TOKEN or MAPILARY_TOKEN (never printed). Lists images via graph.mapillary.com /images in small
bboxes tiled over the 40 m detail area; keeps images within 40 m of a dev7 road centerline.
Writes data/derived/dev7/mapillary_index.csv (id, lon, lat, compass_angle, captured_at, is_pano, sequence, creator,
road, s_m, offset_m, dir) and data/derived/dev7/mapillary_coverage.csv (per road per 100 m bin: n images, newest year).
List cache: data/cache/p7/mapillary_list.json (idempotent)."""
import os, sys, json, time, csv, glob
import numpy as np, requests
from shapely import wkt
from shapely.geometry import box
from scipy.spatial import cKDTree
from common7 import *

TOK = os.environ.get("MAPILLARY_TOKEN") or os.environ.get("MAPILARY_TOKEN")
if not TOK: sys.exit("no Mapillary token (MAPILLARY_TOKEN / MAPILARY_TOKEN)")
H = {"Authorization": f"OAuth {TOK}"}
F = "id,geometry,computed_geometry,captured_at,compass_angle,computed_compass_angle,is_pano,sequence,creator"
LST = CACHE7 / "mapillary_list.json"
det = wkt.loads(json.load(open(DER7 / "scope.json"))["detail_40m"])
imgs = json.load(open(LST)) if LST.exists() else {}
done_cells = set(imgs.pop("_cells", []))
b = det.bounds; STEP = 220.0
cells = [(x, y) for x in np.arange(b[0], b[2], STEP) for y in np.arange(b[1], b[3], STEP) if det.intersects(box(x, y, x + STEP, y + STEP))]


def query(x, y, s, depth=0):
    lo0, la0 = loc2ll(x, y); lo1, la1 = loc2ll(x + s, y + s)
    url = f"https://graph.mapillary.com/images?fields={F}&bbox={lo0:.6f},{la0:.6f},{lo1:.6f},{la1:.6f}&limit=2000"
    for k in range(5):
        try:
            r = requests.get(url, headers=H, timeout=90); r.raise_for_status(); data = r.json().get("data", []); break
        except Exception as e:
            time.sleep(3 + 5 * k); data = None
    if data is None: return False
    if len(data) >= 2000 and depth < 3:          # saturated -> split
        h = s / 2
        return all(query(x + i * h, y + j * h, h, depth + 1) for i in (0, 1) for j in (0, 1))
    for im in data: imgs[im["id"]] = im
    return True


for k, (x, y) in enumerate(cells):
    key = f"{x:.0f}_{y:.0f}"
    if key in done_cells: continue
    if query(x, y, STEP): done_cells.add(key)
    if k % 20 == 0:
        print(k, "/", len(cells), len(imgs), flush=True)
        json.dump(dict(imgs, _cells=sorted(done_cells)), open(LST, "w"))
json.dump(dict(imgs, _cells=sorted(done_cells)), open(LST, "w"))
print("cells", len(cells), "done", len(done_cells), "images listed", len(imgs))
# assign to roads (refined centerline if present, else OSM)
P, R, Sm, T = [], [], [], []
for f in sorted(glob.glob(str(DER7 / "osm_centerline_*.csv"))):
    road = f.split("osm_centerline_")[1][:-4]
    ref = DER7 / f"centerline_{road}.csv"
    if ref.exists():
        a = np.genfromtxt(ref, delimiter=",", names=True); xy = np.c_[a["x_local"], a["y_local"]]; s = a["s_m"]
    else:
        a = np.genfromtxt(f, delimiter=",", names=True); xy0 = np.c_[a["x_local"], a["y_local"]]
        s0 = a["s_m"]; s = np.arange(0, s0[-1], 1.0); xy = np.c_[np.interp(s, s0, xy0[:, 0]), np.interp(s, s0, xy0[:, 1])]
    t = np.gradient(xy, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1), 1e-9)[:, None]
    P.append(xy); R += [road] * len(xy); Sm.append(s); T.append(t)
P = np.vstack(P); Sm = np.concatenate(Sm); T = np.vstack(T); R = np.array(R)
KD = cKDTree(P)
rows = []
for im in imgs.values():
    g = (im.get("computed_geometry") or im["geometry"])["coordinates"]
    x, y = ll2loc(g[0], g[1]); dd, j = KD.query([float(x), float(y)])
    if dd > 40: continue
    off = float(-T[j, 1] * (x - P[j, 0]) + T[j, 0] * (y - P[j, 1]))
    ca = im.get("computed_compass_angle", im.get("compass_angle"))
    hd = np.degrees(np.arctan2(T[j, 0], T[j, 1])) % 360
    dirn = "" if ca is None else ("fwd" if abs((ca - hd + 180) % 360 - 180) < 90 else "rev")
    rows.append(dict(id=im["id"], lon=round(g[0], 7), lat=round(g[1], 7), compass_angle=None if ca is None else round(ca, 1),
                     captured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(im["captured_at"] / 1000)),
                     is_pano=int(bool(im.get("is_pano"))), sequence=im.get("sequence"), creator=(im.get("creator") or {}).get("username", ""),
                     road=R[j], s_m=round(float(Sm[j]), 1), offset_m=round(off, 1), dir=dirn, dist_m=round(float(dd), 1)))
rows.sort(key=lambda r: (r["road"], r["s_m"]))
with open(DER7 / "mapillary_index.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
cov = {}
for road in sorted(set(R)):
    L = Sm[R == road].max(); nb = int(L // 100) + 1
    cnt = np.zeros(nb, int); yr = np.zeros(nb, int)
    for r in rows:
        if r["road"] == road:
            i = int(r["s_m"] // 100); cnt[i] += 1; yr[i] = max(yr[i], int(r["captured_at"][:4]))
    cov[road] = (cnt, yr)
with open(DER7 / "mapillary_coverage.csv", "w") as fo:
    fo.write("road,bin_start_m,n_images,newest_year\n")
    for road, (c, y) in cov.items():
        for i in range(len(c)): fo.write(f"{road},{i*100},{c[i]},{y[i]}\n")
for road, (c, y) in cov.items():
    print(f"{road}: {sum(c)} imgs, 100 m bins covered {(c > 0).sum()}/{len(c)}, years {sorted(set(y[y > 0]))[:1]}..{max(y) if y.max() else '-'}")
print("total in index", len(rows))
