"""dev7 package M helpers: Mapillary graph API (token from MAPILARY_TOKEN or MAPILLARY_TOKEN, never printed)
and projection of lon/lat onto the dev7 refined centerlines -> (road, s, side, off).

side: L/R relative to the direction of increasing station on that road (dev7 rows carry a `road` field).
off : metres beyond the pavement edge on that side (edge_left_m / edge_right_m of the centerline CSV)."""
import os, csv, json, time, glob, pathlib
import numpy as np
import requests
from scipy.spatial import cKDTree
from common7 import DER7, RAW7, ll2loc

MLY = RAW7 / "mapillary"; IMG = MLY / "img"
IMG.mkdir(parents=True, exist_ok=True)
GRAPH = "https://graph.mapillary.com"


def token():
    t = os.environ.get("MAPILARY_TOKEN") or os.environ.get("MAPILLARY_TOKEN")
    if not t:
        raise SystemExit("no MAPILARY_TOKEN / MAPILLARY_TOKEN in env")
    return t


def get(path, **params):
    params["access_token"] = token()
    for k in range(5):
        try:
            r = requests.get(f"{GRAPH}/{path}", params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            err = type(e).__name__
        time.sleep(2 + 3 * k)
    raise RuntimeError(f"graph {path}: {err}")          # message never contains the token


class Roads:
    def __init__(self):
        self.R = {}
        P, idx = [], []
        for f in sorted(glob.glob(str(DER7 / "centerline_*.csv"))):
            road = pathlib.Path(f).stem[len("centerline_"):]
            a = np.genfromtxt(f, delimiter=",", names=True, usecols=("s_m", "x_local", "y_local", "z_navd88", "edge_left_m", "edge_right_m"))
            self.R[road] = a
            P.append(np.c_[a["x_local"], a["y_local"]]); idx += [(road, i) for i in range(len(a))]
        self.P = np.vstack(P); self.idx = idx; self.T = cKDTree(self.P)

    def project(self, x, y, road=None, k=40):
        """nearest centerline point (optionally restricted to one road) -> dict(road, s, side, lat, off, z)"""
        if road:
            a = self.R[road]; dd2 = np.hypot(a["x_local"] - x, a["y_local"] - y)
            cand = [(float(dd2.min()), (road, int(dd2.argmin())))]
        else:
            d, ii = self.T.query([x, y], k=1)
            cand = [(float(d), self.idx[int(ii)])]
        for dd, (rd, j) in cand:
            a = self.R[rd]; j0, j1 = max(j - 1, 0), min(j + 1, len(a) - 1)
            tx, ty = a["x_local"][j1] - a["x_local"][j0], a["y_local"][j1] - a["y_local"][j0]
            n = np.hypot(tx, ty) or 1.0
            lat = ((x - a["x_local"][j]) * (-ty) + (y - a["y_local"][j]) * tx) / n     # + = left
            side = "L" if lat >= 0 else "R"
            edge = a["edge_left_m"][j] if side == "L" else a["edge_right_m"][j]
            return dict(road=rd, s=round(float(a["s_m"][j]), 1), side=side, lat=round(float(lat), 2),
                        off=round(float(abs(lat) - edge), 2), z=round(float(a["z_navd88"][j]), 2), dist=float(dd))
        return None

    def project_ll(self, lon, lat, road=None):
        x, y = ll2loc(lon, lat)
        return self.project(float(x), float(y), road)
