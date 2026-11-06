"""Download OSM map data (all features) in small bboxes along the corridor for junction/driveway/bridge/waterway context."""
import numpy as np, requests, time
from common import *
cl = np.loadtxt(DER / "osm_centerline_utm.csv", delimiter=",", skiprows=1)
out = RAW / "osm_ctx"; out.mkdir(exist_ok=True)
for i, s0 in enumerate(np.arange(0, cl[-1, 0], 1500)):
    f = out / f"ctx_{i:02d}.xml"
    if f.exists(): continue
    m = (cl[:, 0] >= s0 - 50) & (cl[:, 0] <= s0 + 1550)
    lon, lat = T_UTM_LL.transform(cl[m, 1], cl[m, 2])
    b = (lon.min() - .002, lat.min() - .0015, lon.max() + .002, lat.max() + .0015)
    for t in range(4):
        try:
            r = requests.get("https://api.openstreetmap.org/api/0.6/map", params={"bbox": ",".join(f"{v:.5f}" for v in b)},
                             headers={"User-Agent": "palomares-ac-track-pipeline"}, timeout=120)
            r.raise_for_status(); f.write_bytes(r.content); break
        except Exception as e:
            print("retry", i, e); time.sleep(2 ** t)
    print(i, f.stat().st_size if f.exists() else "FAILED")
