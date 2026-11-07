#!/usr/bin/env python3
"""dev7 D-05: features per road + trees + buildings (Phase-1 scripts/08 equivalents for the dev7 network).
Writes in data/derived/dev7/:
  features.geojson   every item carries road + station (s_m or s_start_m/s_end_m):
     junction, signal/stop, bridge_osm, bridge_lidar_class17, rail_underpass (+DSM-over-road height), road_over_rail,
     level_crossing, culvert_candidate, stream_crossing_bridge, wall_or_steep_cut, steep_drop, guardrail_candidate
     (OSM barrier or >= 15 m of drop beside a rural road), curb runs (town), lidar_gap (> 5 m), canyon summit/low point
  gaps.csv           ground-return gaps > 5 m along any centerline
  trees.csv/.geojson CHM tree tops >= 4 m within 60 m of any dev7 road (x/y local+UTM, ground z, height, crown r)
  buildings.geojson  OSM footprints with LiDAR height (CHM p90 inside footprint) and ground z
  profile_niles_canyon_road.png  elevation profile + grade
Run: nice -n5 python3 scripts/p7/05_features.py
"""
import glob, json, csv
import numpy as np, rasterio
from rasterio.features import rasterize
from scipy.ndimage import maximum_filter, map_coordinates
from scipy.spatial import cKDTree
from shapely.geometry import shape, LineString, Point
from common7 import *

RS = {k: rasterio.open(DER7 / f"{k}.tif") for k in ("chm_r100_b150", "dsm_r100_b150", "dtm_r100_b150")}
CHM = RS["chm_r100_b150"].read(1); DSM = RS["dsm_r100_b150"].read(1); DTM = RS["dtm_r100_b150"].read(1)
INV = ~RS["chm_r100_b150"].transform; TF = RS["chm_r100_b150"].transform


def samp(a, x, y, order=0):
    c, r = INV * (np.asarray(x) + X0, np.asarray(y) + Y0)
    return map_coordinates(a, [np.asarray(r) - .5, np.asarray(c) - .5], order=order, mode="constant", cval=np.nan)


roads = {}
for f in sorted(glob.glob(str(DER7 / "centerline_*.csv"))):
    r = f.split("centerline_")[1][:-4]
    roads[r] = np.genfromtxt(f, delimiter=",", names=True, dtype=None, encoding=None)
ALLP = np.vstack([np.c_[a["x_local"], a["y_local"]] for a in roads.values()])
ALLR = np.concatenate([[r] * len(a) for r, a in roads.items()]); ALLS = np.concatenate([a["s_m"] for a in roads.values()])
KD = cKDTree(ALLP)
LS = {r: LineString(np.c_[a["x_local"], a["y_local"]]) for r, a in roads.items()}
RURAL = {"niles_canyon_road", "palomares_road", "palomares_link_1", "paloma_road", "pleasanton_sunol_road"}


def station(x, y, road=None, maxd=30):
    if road:
        p = Point(x, y); return road, LS[road].project(p), LS[road].distance(p)
    d, j = KD.query([x, y])
    return (ALLR[j], float(ALLS[j]), float(d)) if d < maxd else (None, None, float(d))


def P(x, y, **kw):
    lon, lat = loc2ll(x, y)
    return feat({"type": "Point", "coordinates": [round(float(lon), 7), round(float(lat), 7)]}, **kw)


def runs(mask, minlen=1):
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]: j += 1
            if j - i + 1 >= minlen: out.append((i, j))
            i = j + 1
        else: i += 1
    return out


F = []; gaps = []
for r, a in roads.items():
    s = a["s_m"]; xy = np.c_[a["x_local"], a["y_local"]]
    def seg(i, j, kind, **kw):
        k = (i + j) // 2
        F.append(P(xy[k, 0], xy[k, 1], kind=kind, road=r, s_start_m=round(float(s[i]), 1), s_end_m=round(float(s[j]), 1),
                   length_m=round(float(s[j] - s[i]), 1), **kw))
    for side, key in (("left", "env_left"), ("right", "env_right")):
        e = a[key]
        for i, j in runs(e == "wall_or_steep_cut", 8): seg(i, j, "wall_or_steep_cut", side=side, rise_5m_max=round(float(np.nanmax(a[f"rise_{side}_5m"][i:j + 1])), 2))
        for i, j in runs(np.isin(e, ["steep_drop"]), 8): seg(i, j, "steep_drop", side=side, rise_5m_min=round(float(np.nanmin(a[f"rise_{side}_5m"][i:j + 1])), 2))
        if r in RURAL:
            for i, j in runs(np.isin(e, ["steep_drop", "drop_off"]), 15): seg(i, j, "guardrail_candidate", side=side, src="lidar_drop")
        else:
            for i, j in runs(a[f"curb_{side}"] > 0.07, 10): seg(i, j, "curb", side=side, height_m=round(float(np.median(a[f"curb_{side}"][i:j + 1])), 2))
    for i, j in runs(a["bridge17"] > 0, 3): seg(i, j, "bridge_lidar_class17", n_pts=int(a["bridge17"][i:j + 1].sum()))
    g = np.nan_to_num(a["gdist_m"], nan=9.0)
    for i, j in runs(g > 1.0, 1):
        L = float(s[j] - s[i]) + 2 * float(np.max(g[i:j + 1]))
        if L > 5:
            k = (i + j) // 2
            row = dict(road=r, s_start_m=round(float(s[i]), 1), s_end_m=round(float(s[j]), 1), gap_len_m=round(L, 1),
                       max_gdist_m=round(float(g[i:j + 1].max()), 2), canopy_frac=round(float(np.nanmean(a["canopy_frac"][i:j + 1])), 2),
                       bridge17=int(a["bridge17"][i:j + 1].sum()), lon=float(a["lon"][k]), lat=float(a["lat"][k]))
            gaps.append(row); F.append(P(xy[k, 0], xy[k, 1], kind="lidar_gap", **{k_: v for k_, v in row.items() if k_ not in ("lon", "lat")}))
# OSM-derived items
of = json.load(open(DER7 / "osm_features.geojson"))["features"]
net = json.load(open(DER7 / "network.geojson"))["features"]
for e in net:
    p = e["properties"]
    if p.get("bridge") not in (None, "no") or p.get("tunnel") not in (None, "no"):
        c = np.array(e["geometry"]["coordinates"]); x, y = ll2loc(c[:, 0], c[:, 1])
        r0, s0, d0 = station(x[0], y[0]); r1, s1, d1 = station(x[-1], y[-1])
        if r0 and r0 == r1:
            F.append(P(np.mean(x), np.mean(y), kind="bridge_osm" if p.get("bridge") not in (None, "no") else "tunnel_osm", road=r0,
                       s_start_m=round(min(s0, s1), 1), s_end_m=round(max(s0, s1), 1), osm_way=p["way"], layer=p.get("layer"), name=p.get("name")))
dsm_hits = []
for e in of:
    p = e["properties"]; k = p["kind"]
    if e["geometry"]["type"] != "Point": continue
    x, y = map(float, ll2loc(*e["geometry"]["coordinates"]))
    if k in ("rail_underpass", "road_over_rail", "rail_tunnel_under_road", "level_crossing_or_unknown", "culvert_candidate", "stream_crossing_bridge",
             "traffic_signals", "stop", "give_way", "rail_level_crossing", "rail_crossing", "rail_station", "rail_halt", "crossing"):
        rr, ss, dd = station(x, y)
        if rr is None: continue
        extra = {}
        if k == "rail_underpass":
            a = roads[rr]; j = int(np.argmin(np.abs(a["s_m"] - ss)))
            zc = a["z_navd88"][j]; ds_ = samp(DSM, [x], [y])[0]
            win = (np.abs(a["s_m"] - ss) < 15)
            extra = dict(road_z=round(float(zc), 2), dsm_over_road_m=round(float(ds_ - zc), 2), note="DSM = top of rail structure; clearance ~ DSM - deck thickness")
        kk = {"level_crossing_or_unknown": "rail_road_crossing", "rail_level_crossing": "level_crossing", "rail_crossing": "rail_ped_crossing"}.get(k, k)
        F.append(P(x, y, kind=kk, road=rr, s_m=round(ss, 1), dist_m=round(dd, 1), **{q: v for q, v in p.items() if q not in ("kind",)}, **extra))
    elif k == "rail_station" or k == "rail_halt":
        pass
for e in of:
    p = e["properties"]
    if p["kind"] == "barrier" and p.get("barrier") in ("guard_rail", "wall", "retaining_wall", "jersey_barrier", "fence", "kerb"):
        c = np.array(e["geometry"]["coordinates"][0] if e["geometry"]["type"] == "Polygon" else e["geometry"]["coordinates"])
        x, y = ll2loc(c[:, 0], c[:, 1]); rr, ss, dd = station(float(np.mean(x)), float(np.mean(y)), maxd=25)
        if rr is None: continue
        r0, s0, _ = station(x[0], y[0], rr); r1, s1, _ = station(x[-1], y[-1], rr)
        F.append(P(np.mean(x), np.mean(y), kind="guardrail_candidate" if p["barrier"] in ("guard_rail", "jersey_barrier") else "barrier_osm",
                   barrier=p["barrier"], road=rr, s_start_m=round(min(s0, s1), 1), s_end_m=round(max(s0, s1), 1), dist_m=round(dd, 1), src="osm", osm_way=p["osm_way"]))
for j in json.load(open(DER7 / "junctions.geojson"))["features"]:
    q = j["properties"]
    if q["degree"] < 3: continue
    for rr in {ALLR[i] for i in KD.query_ball_point([q["x_local"], q["y_local"]], 6)}:
        _, ss, dd = station(q["x_local"], q["y_local"], rr)
        F.append(P(q["x_local"], q["y_local"], kind="junction", road=rr, s_m=round(ss, 1), roads=q["roads"], control=q["control"], osm_node=q["osm_node"]))
# canyon profile: summit / low point
a = roads["niles_canyon_road"]; z = a["z_navd88"]; s = a["s_m"]
for kind, j in (("canyon_high_point", int(np.nanargmax(z))), ("canyon_low_point", int(np.nanargmin(z)))):
    F.append(P(a["x_local"][j], a["y_local"][j], kind=kind, road="niles_canyon_road", s_m=round(float(s[j]), 1), z_navd88=round(float(z[j]), 2)))
dump_geojson(DER7 / "features.geojson", F)
with open(DER7 / "gaps.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=["road", "s_start_m", "s_end_m", "gap_len_m", "max_gdist_m", "canopy_frac", "bridge17", "lon", "lat"]); w.writeheader(); w.writerows(gaps)
# trees: CHM local maxima >= 4 m within 60 m of any road (not on buildings: excluded via OSM footprints)
blds = [shape(e["geometry"]) for e in of if e["properties"]["kind"].startswith("building") and e["geometry"]["type"] == "Polygon"]
bm = rasterize([(shape({"type": "Polygon", "coordinates": [np.c_[T_LL_UTM.transform(*np.array(b.exterior.coords).T)].tolist()]}).buffer(1.5), 1) for b in blds],
               out_shape=CHM.shape, transform=TF, dtype="uint8") if blds else np.zeros(CHM.shape, np.uint8)
ch = np.nan_to_num(CHM, nan=0)
mx = maximum_filter(ch, size=5)
rr, cc = np.nonzero((ch == mx) & (ch >= 4) & (bm == 0))
x = TF.c + (cc + .5) * TF.a - X0; y = TF.f + (rr + .5) * TF.e - Y0
d, j = KD.query(np.c_[x, y]); keep = d <= 60
rr, cc, x, y, d, j = rr[keep], cc[keep], x[keep], y[keep], d[keep], j[keep]
h = ch[rr, cc]
rad = np.zeros(len(h))
for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
    for step in range(1, 11):
        r2 = np.clip(rr + np.round(step * np.sin(ang)).astype(int), 0, ch.shape[0] - 1); c2 = np.clip(cc + np.round(step * np.cos(ang)).astype(int), 0, ch.shape[1] - 1)
        still = (ch[r2, c2] > 0.5 * h) & (rad >= step - 1 - 1e-9 + 0) if False else (ch[r2, c2] > 0.5 * h)
        rad += still / 8.0
rad = np.clip(rad, 1.0, 10.0)
gz = DTM[rr, cc]
with open(DER7 / "trees.csv", "w") as fo:
    fo.write("x_local,y_local,x_utm,y_utm,lon,lat,ground_z,height_m,crown_r_m,road,s_m,dist_m\n")
    lon, lat = loc2ll(x, y)
    for k in range(len(h)):
        fo.write(f"{x[k]:.2f},{y[k]:.2f},{x[k]+X0:.2f},{y[k]+Y0:.2f},{lon[k]:.7f},{lat[k]:.7f},{gz[k]:.2f},{h[k]:.1f},{rad[k]:.1f},{ALLR[j[k]]},{ALLS[j[k]]:.0f},{d[k]:.1f}\n")
dump_geojson(DER7 / "trees.geojson", [feat({"type": "Point", "coordinates": [round(float(lon[k]), 7), round(float(lat[k]), 7)]}, h=round(float(h[k]), 1), r=round(float(rad[k]), 1)) for k in range(len(h))])
# buildings with LiDAR height
bf = []
for e in of:
    if not e["properties"]["kind"].startswith("building") or e["geometry"]["type"] != "Polygon": continue
    c = np.array(e["geometry"]["coordinates"][0]); ux, uy = T_LL_UTM.transform(c[:, 0], c[:, 1])
    poly = shape({"type": "Polygon", "coordinates": [np.c_[ux, uy].tolist()]})
    m = rasterize([(poly, 1)], out_shape=CHM.shape, transform=TF, dtype="uint8").astype(bool) if poly.area > 4 else None
    if m is None or not m.any(): continue
    hv = CHM[m]; gv = DTM[m]
    if not np.isfinite(hv).any(): continue
    bf.append(feat(e["geometry"], **e["properties"], lidar_height_p90_m=round(float(np.nanpercentile(hv, 90)), 2),
                   ground_z=round(float(np.nanmin(gv)), 2) if np.isfinite(gv).any() else None, area_m2=round(poly.area, 1)))
dump_geojson(DER7 / "buildings.geojson", bf)
# profile plot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(s / 1000, z, lw=1, color="#1f5f8b"); ax[0].set_ylabel("z NAVD88 (m)"); ax[0].grid(alpha=.3)
for e in F:
    p = e["properties"]
    if p.get("road") == "niles_canyon_road" and p["kind"] in ("rail_underpass", "bridge_osm", "junction") and "s_m" in p or (p.get("road") == "niles_canyon_road" and p["kind"] == "bridge_osm"):
        sm_ = p.get("s_m", p.get("s_start_m")); ax[0].axvline(sm_ / 1000, color={"rail_underpass": "r", "bridge_osm": "g", "junction": "#bbb"}[p["kind"]], lw=.6, alpha=.7)
gr = np.convolve(a["grade"], np.ones(25) / 25, mode="same") * 100
ax[1].plot(s / 1000, gr, lw=.8, color="#8b3a1f"); ax[1].set_ylabel("grade % (25 m)"); ax[1].set_xlabel("km from Mission Blvd (W) to Sunol (E)"); ax[1].grid(alpha=.3)
ax[0].set_title("Niles Canyon Rd (SR-84) - LiDAR refined centerline (red: rail underpass, green: OSM bridge)")
plt.tight_layout(); plt.savefig(REP7 / "profile_niles_canyon_road.png", dpi=110)
from collections import Counter
print(Counter(e["properties"]["kind"] for e in F)); print("gaps", len(gaps), "trees", len(h), "buildings", len(bf))
