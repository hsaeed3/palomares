"""dev7 M step 2: choose and download the Mapillary images the survey needs (thumb_1024) -> data/raw/dev7/mapillary/img/<id>.jpg
Selection (deterministic): Niles Canyon Rd fwd every 75 m + rev every 150 m (2023 preferred, nearest the centerline),
every 25 m both dirs within +-150 m of NCR landmarks (underpasses, bridges, rail crossings, stops, signals, Palomares jct),
town roads every 20 m. Writes data/derived/dev7/mapillary_sample.csv. Resumable (skips files on disk). ~5-8 min first run."""
import csv, json, sys, pathlib, concurrent.futures as cf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import requests
from common7 import DER7
from mly import get, IMG

idx = list(csv.DictReader(open(DER7 / "mapillary_index.csv")))
F = json.load(open(DER7 / "features.geojson"))["features"]
LMK = sorted({float(f["properties"]["s"]) for f in F if f["properties"].get("road") == "niles_canyon_road" and f["properties"].get("s") is not None
              and f["properties"]["kind"] in ("rail_underpass", "road_over_rail", "rail_tunnel_under_road", "level_crossing", "stop",
                                              "traffic_signals", "stream_crossing_bridge")} | {3539.0, 10700.0})
for f in F:
    p = f["properties"]
    if p.get("road") == "niles_canyon_road" and p["kind"] == "bridge_osm" and p.get("s_start_m") is not None:
        LMK += [float(p["s_start_m"]), float(p["s_end_m"])]


def score(r):
    return (0 if r["captured_at"][:4] == "2023" else 1, abs(float(r["offset_m"])))


pick = {}
def take(rows, step, tag):
    b = {}
    for r in rows:
        k = int(float(r["s_m"]) // step)
        if k not in b or score(r) < score(b[k]):
            b[k] = r
    for r in b.values():
        pick.setdefault(r["id"], dict(r, why=tag))

ncr = [r for r in idx if r["road"] == "niles_canyon_road" and float(r["dist_m"]) < 15]
take([r for r in ncr if r["dir"] == "fwd"], 75, "ncr_fwd75")
take([r for r in ncr if r["dir"] == "rev"], 150, "ncr_rev150")
for s0 in LMK:
    near = [r for r in ncr if abs(float(r["s_m"]) - s0) <= 150]
    for d in ("fwd", "rev"):
        take([r for r in near if r["dir"] == d], 25, f"lmk{int(s0)}_{d}")
town = [r for r in idx if r["road"] != "niles_canyon_road" and float(r["dist_m"]) < 15]
for road in sorted({r["road"] for r in town}):
    for d in ("fwd", "rev"):
        take([r for r in town if r["road"] == road and r["dir"] == d], 20, f"town_{d}")
rows = sorted(pick.values(), key=lambda r: (r["road"], float(r["s_m"])))
print(len(rows), "images selected", flush=True)


def fetch(r):
    out = IMG / f"{r['id']}.jpg"
    if out.exists() and out.stat().st_size > 5000:
        return r["id"], True
    try:
        js = get(r["id"], fields="thumb_1024_url")
        b = requests.get(js["thumb_1024_url"], timeout=60).content
        if len(b) > 5000:
            out.write_bytes(b); return r["id"], True
    except Exception:
        pass
    return r["id"], False

ok = {}
with cf.ThreadPoolExecutor(4) as ex:
    for i, (iid, good) in enumerate(ex.map(fetch, rows)):
        ok[iid] = good
        if i % 50 == 0:
            print(i, flush=True)
with open(DER7 / "mapillary_sample.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]) + ["have"]); w.writeheader()
    for r in rows:
        w.writerow(dict(r, have=int(ok[r["id"]])))
print(sum(ok.values()), "/", len(rows), "on disk")
