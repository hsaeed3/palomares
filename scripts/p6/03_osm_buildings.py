#!/usr/bin/env python3
"""dev6 step 03: OSM building footprints (ODbL) + barrier=fence/wall/guard_rail/hedge ways within 200 m of the road,
for cross-checking LiDAR products. -> data/cache/p6/osm_context.json (raw Overpass) and
data/derived/osm_buildings_barriers.geojson (UTM 10N). Idempotent."""
import sys, json, pathlib, requests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import DER, CACHE, T_LL_UTM
import numpy as np
cl = np.genfromtxt(DER / "centerline.csv", delimiter=",", names=True)
raw = CACHE / "p6" / "osm_context.json"; raw.parent.mkdir(parents=True, exist_ok=True)
if not raw.exists():
    pts = np.c_[cl["lat"], cl["lon"]][::40]
    poly = " ".join(f"{a:.6f} {b:.6f}" for a, b in pts)
    bb = f"{cl['lat'].min()-0.003:.5f},{cl['lon'].min()-0.003:.5f},{cl['lat'].max()+0.003:.5f},{cl['lon'].max()+0.003:.5f}"
    q = f"""[out:json][timeout:120];
(way["building"]({bb}); way["barrier"]({bb}););
out geom tags;"""
    import time
    for host in ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]:
        try:
            r = requests.post(host, data={"data": q}, timeout=200); r.raise_for_status(); raw.write_text(r.text); break
        except Exception as e:
            print("overpass fail", host, e); time.sleep(20)
if not raw.exists():   # Overpass unreachable: OSM editing API /map in small bbox tiles along the road
    import xml.etree.ElementTree as ET
    nodes, ways = {}, {}
    seen = set()
    for la, lo in zip(cl["lat"][::150], cl["lon"][::150]):
        k = (round(la / 0.006), round(lo / 0.006))
        for dk in [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]:
            kk = (k[0] + dk[0], k[1] + dk[1])
            if kk in seen: continue
            seen.add(kk); a0, b0 = kk[0] * 0.006 - 0.003, kk[1] * 0.006 - 0.003
            r = requests.get(f"https://api.openstreetmap.org/api/0.6/map?bbox={b0:.4f},{a0:.4f},{b0+0.006:.4f},{a0+0.006:.4f}", timeout=120)
            if r.status_code != 200: print("osm api", r.status_code, kk); continue
            x = ET.fromstring(r.content)
            for n in x.iter("node"): nodes[n.get("id")] = (float(n.get("lon")), float(n.get("lat")))
            for w in x.iter("way"):
                t = {g.get("k"): g.get("v") for g in w.iter("tag")}
                if "building" in t or "barrier" in t: ways[w.get("id")] = ([nd.get("ref") for nd in w.iter("nd")], t)
    els = [dict(type="way", id=int(i), tags=t, geometry=[dict(lon=nodes[r][0], lat=nodes[r][1]) for r in refs if r in nodes]) for i, (refs, t) in ways.items()]
    raw.write_text(json.dumps(dict(elements=els, source="api.openstreetmap.org/api/0.6/map")))
J = json.loads(raw.read_text()); feats = []
from scipy.spatial import cKDTree
KD = cKDTree(np.c_[cl["x_utm"], cl["y_utm"]][::2])
for el in J["elements"]:
    if "geometry" not in el: continue
    xy = [list(T_LL_UTM.transform(g["lon"], g["lat"])) for g in el["geometry"]]
    if KD.query(np.array(xy))[0].min() > 200: continue
    t = el.get("tags", {}); kind = "building" if "building" in t else t.get("barrier", t.get("man_made"))
    closed = xy[0] == xy[-1] and len(xy) > 3
    feats.append(dict(type="Feature", properties=dict(osm_id=el["id"], kind=kind, tags=t),
                      geometry=dict(type="Polygon", coordinates=[xy]) if (kind == "building" and closed) else dict(type="LineString", coordinates=xy)))
json.dump(dict(type="FeatureCollection", crs=dict(type="name", properties=dict(name="EPSG:26910")), features=feats),
          open(DER / "osm_buildings_barriers.geojson", "w"))
import collections; print(collections.Counter(f["properties"]["kind"] for f in feats))
