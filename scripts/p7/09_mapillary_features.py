"""dev7 M step 1: Mapillary map features (traffic signs + points) in the dev7 bbox, projected to road/station.
Raw: data/raw/dev7/mapillary/map_features.json (cached). Out: data/derived/dev7/mapillary_features.csv.  ~1-2 min."""
import json, csv, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from common7 import DER7, BBOX_LL
from mly import get, MLY, Roads

S, W, N, E = 37.570, -121.990, 37.600, -121.875
raw = MLY / "map_features.json"
if raw.exists():
    F = json.load(open(raw))
else:
    F = {}
    for lo in np.arange(W, E, 0.01):
        for la in np.arange(S, N, 0.01):
            bb = f"{lo:.4f},{la:.4f},{min(lo + 0.01, E):.4f},{min(la + 0.01, N):.4f}"
            for typ in ("trafficsign", "point"):
                js = get("map_features", fields="id,object_value,object_type,geometry,first_seen_at,last_seen_at", bbox=bb, object_type=typ, limit=2000)
                for f in js.get("data", []):
                    F[f["id"]] = f
            print(bb, len(F), flush=True)
    json.dump(F, open(raw, "w"))
R = Roads(); rows = []
for f in F.values():
    lon, lat = f["geometry"]["coordinates"]
    p = R.project_ll(lon, lat)
    if p is None or p["dist"] > 40:
        continue
    rows.append(dict(id=f["id"], object_type=f.get("object_type"), value=f["object_value"], lon=lon, lat=lat,
                     road=p["road"], s=p["s"], side=p["side"], lat_m=p["lat"], off=p["off"],
                     first_seen=f.get("first_seen_at", "")[:10], last_seen=f.get("last_seen_at", "")[:10]))
rows.sort(key=lambda r: (r["road"], r["s"]))
with open(DER7 / "mapillary_features.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(F), "features in bbox,", len(rows), "within 40 m of a dev7 road")
