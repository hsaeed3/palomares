"""dev7 shared paths/helpers (data package D). Reuses scripts/common.py (fixed origin, CRS, EPT sets)."""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import ROOT, RAW, DER, CACHE, REPORTS, CONFIG, UTM, T_LL_UTM, T_UTM_LL, T_UTM_WM, T_WM_UTM, EPT_BASE, EPT_SETS, origin  # noqa

RAW7 = RAW / "dev7"; DER7 = DER / "dev7"; CACHE7 = CACHE / "p7"; REP7 = REPORTS / "dev7"
for p in (RAW7, DER7, CACHE7, REP7):
    p.mkdir(parents=True, exist_ok=True)
O = origin(); X0, Y0 = O["x0"], O["y0"]
BBOX_LL = (37.565, -121.995, 37.610, -121.865)      # S, W, N, E  (Overpass order)
DETAIL_BUF, TERRAIN_BUF = 40.0, 150.0
TILE = 250.0          # LiDAR re-bin / DTM tile size (local m)


def ll2loc(lon, lat):
    x, y = T_LL_UTM.transform(lon, lat)
    return np.asarray(x) - X0, np.asarray(y) - Y0


def loc2ll(x, y):
    return T_UTM_LL.transform(np.asarray(x) + X0, np.asarray(y) + Y0)


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def dump_geojson(path, feats):
    json.dump({"type": "FeatureCollection", "features": feats}, open(path, "w"), separators=(",", ":"))


def feat(geom, **props):
    return {"type": "Feature", "geometry": geom, "properties": props}


def line_ll(xy, z=None):
    lon, lat = loc2ll(xy[:, 0], xy[:, 1])
    c = np.c_[np.round(lon, 7), np.round(lat, 7)] if z is None else np.c_[np.round(lon, 7), np.round(lat, 7), np.round(z, 2)]
    return {"type": "LineString", "coordinates": c.tolist()}
