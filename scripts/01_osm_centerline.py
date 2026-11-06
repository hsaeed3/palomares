"""Fetch Palomares Road ways from OSM, join them into one S->N polyline (UTM 10N),
and fix the local origin at the midpoint of the full road."""
import json, requests, numpy as np
from common import *

WAYS = [1190239384, 116299847, 116299515, 79430605]   # main chain; spurs 6368528, 6368499 excluded
UA = {"User-Agent": "palomares-ac-track-pipeline"}

def get_way(w):
    f = RAW / f"way_{w}.json"
    if not f.exists():
        r = requests.get(f"https://api.openstreetmap.org/api/0.6/way/{w}/full.json", headers=UA, timeout=60)
        r.raise_for_status(); f.write_text(r.text)
    d = json.loads(f.read_text())
    nodes = {e["id"]: e for e in d["elements"] if e["type"] == "node"}
    way = [e for e in d["elements"] if e["type"] == "way"][0]
    return way, [(nodes[n]["lon"], nodes[n]["lat"]) for n in way["nodes"]], way["nodes"]

segs = [get_way(w) for w in WAYS]
# join: orient each way to continue from the previous end node
line, last, tags = [], None, []
w0, w1 = segs[0][2], segs[1][2]
if w0[0] in (w1[0], w1[-1]):           # orient first way so its end touches the second
    segs[0] = (segs[0][0], segs[0][1][::-1], w0[::-1])
for way, pts, nids in segs:
    if last is not None and nids[-1] == last:
        pts, nids = pts[::-1], nids[::-1]
    if last is not None:
        assert nids[0] == last, f"way {way['id']} does not connect"
        pts = pts[1:]
    line += pts; last = nids[-1]; tags.append({"id": way["id"], **way["tags"]})
ll = np.array(line)
# ensure south -> north
if ll[0, 1] > ll[-1, 1]:
    ll = ll[::-1]
x, y = T_LL_UTM.transform(ll[:, 0], ll[:, 1])
xy = np.c_[x, y]
seg = np.hypot(*np.diff(xy, axis=0).T); s = np.r_[0, np.cumsum(seg)]
L = s[-1]
mx, my = np.interp(L / 2, s, x), np.interp(L / 2, s, y)
o = {"crs": UTM, "x0": round(float(mx), 1), "y0": round(float(my), 1), "z0": 0.0,
     "note": "Local origin = midpoint (by length) of full OSM Palomares Rd. FIXED for all phases. local = utm - (x0,y0); z = NAVD88 m."}
if not (CONFIG / "origin.json").exists():
    json.dump(o, open(CONFIG / "origin.json", "w"), indent=2)
gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"source": "OSM (ODbL)", "ways": tags, "length_m": L},
      "geometry": {"type": "LineString", "coordinates": ll.tolist()}}]}
json.dump(gj, open(DER / "osm_centerline_wgs84.geojson", "w"))
np.savetxt(DER / "osm_centerline_utm.csv", np.c_[s, xy], delimiter=",", header="s,x_utm,y_utm", comments="", fmt="%.3f")
print(f"OSM length {L:.0f} m, {len(xy)} pts, origin {o['x0']},{o['y0']}  start {ll[0]} end {ll[-1]}")
