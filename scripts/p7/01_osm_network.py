#!/usr/bin/env python3
"""dev7 D-01: OSM road network for the Niles town area + Niles Canyon Rd (SR-84) + Palomares connector.
Fetches one Overpass bbox extract (cached in data/raw/dev7/osm_bbox.json; overpass-api.de -> kumi fallback),
selects the scope, builds a junction graph and writes (all ODbL):
  data/derived/dev7/network.geojson       edges split at junctions, with tags (lanes,width,surface,oneway,...)
  data/derived/dev7/junctions.geojson     graph nodes (degree, road names, control: signal/stop)
  data/derived/dev7/osm_centerline_<road>.csv / .geojson   ordered OSM line per named road (local+UTM+lon/lat)
  data/derived/dev7/osm_features.geojson  parking, rail lines/platforms/stations, buildings, barriers,
                                          bridges, tunnels, culvert candidates, signals/stops, level crossings
  data/derived/dev7/scope.json            WKT (local m) of town polygon, 40 m detail and 150 m terrain areas
  data/derived/dev7/termini.json          exact termini (Mission Blvd jct, Sunol stop, Niles Blvd start block)
Run: nice -n5 python3 scripts/p7/01_osm_network.py
"""
import json, heapq, collections, time
import numpy as np, requests
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import unary_union
from shapely import wkt
from common7 import *

UA = {"User-Agent": "palomares-ac-track-pipeline/dev7"}
S, W, N, E = BBOX_LL
Q = f"""[out:json][timeout:180];(
way["highway"]({S},{W},{N},{E}); way["railway"]({S},{W},{N},{E}); way["amenity"="parking"]({S},{W},{N},{E});
way["building"]({S},{W},{N},{E}); way["barrier"]({S},{W},{N},{E}); way["waterway"]({S},{W},{N},{E});
way["man_made"~"bridge|embankment"]({S},{W},{N},{E});
node["railway"]({S},{W},{N},{E}); node["highway"~"traffic_signals|stop|give_way|crossing"]({S},{W},{N},{E});
node["public_transport"]({S},{W},{N},{E}); node["amenity"]({S},{W},{N},{E}); node["shop"]({S},{W},{N},{E});
);out body;>;out skel qt;"""


def fetch():
    """OSM API /map in 0.02 deg tiles (complete raw data, all tags), merged; Overpass is used only if the API fails."""
    f = RAW7 / "osm_map_tiles.json"
    if f.exists():
        return json.load(open(f))
    el = {}
    lats = np.arange(S, N, 0.015); lons = np.arange(W, E, 0.02)
    for la in lats:
        for lo in lons:
            tf = RAW7 / f"map_{la:.3f}_{lo:.3f}.json"
            if not tf.exists():
                for k in range(4):
                    try:
                        r = requests.get("https://api.openstreetmap.org/api/0.6/map.json", headers=UA, timeout=180,
                                         params={"bbox": f"{lo:.4f},{la:.4f},{min(lo + .02, E):.4f},{min(la + .015, N):.4f}"})
                        r.raise_for_status(); tf.write_text(r.text); break
                    except Exception as e:
                        print("api fail", la, lo, e); time.sleep(5 + 5 * k)
            for e in json.load(open(tf))["elements"]:
                el[(e["type"], e["id"])] = e
            print("tile", la, lo, len(el), flush=True)
    d = {"elements": list(el.values()), "source": "api.openstreetmap.org/api/0.6/map (ODbL)"}
    f.write_text(json.dumps(d)); return d


D = fetch()
NODES = {e["id"]: e for e in D["elements"] if e["type"] == "node"}
WAYS = [e for e in D["elements"] if e["type"] == "way"]
XY = {}
for nid, n in NODES.items():
    XY[nid] = tuple(map(float, ll2loc(n["lon"], n["lat"])))
DRIVE = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link",
         "tertiary", "tertiary_link", "residential", "unclassified", "living_street", "service"}


def wline(w):
    return LineString([XY[n] for n in w["nodes"]])


def nearest_node(lon, lat, cand):
    x, y = ll2loc(lon, lat)
    return min(cand, key=lambda n: (XY[n][0] - x) ** 2 + (XY[n][1] - y) ** 2)


def graph_of(ways):
    G = collections.defaultdict(list)
    for w in ways:
        ns = w["nodes"]
        for a, b in zip(ns[:-1], ns[1:]):
            L = np.hypot(XY[a][0] - XY[b][0], XY[a][1] - XY[b][1])
            G[a].append((b, L, w["id"])); G[b].append((a, L, w["id"]))
    return G


def dijkstra(G, a, b=None):
    dist = {a: 0.0}; prev = {}; pq = [(0.0, a)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if u == b: break
        for v, L, _ in G[u]:
            if d + L < dist.get(v, 1e18):
                dist[v] = d + L; prev[v] = u; heapq.heappush(pq, (d + L, v))
    if b is None: return dist, prev
    path = [b]
    while path[-1] != a: path.append(prev[path[-1]])
    return path[::-1]


hw = [w for w in WAYS if w.get("tags", {}).get("highway") in DRIVE]
name = lambda w: w.get("tags", {}).get("name")
# ---------------- termini ----------------
ncr_w = [w for w in hw if name(w) == "Niles Canyon Road" and w["tags"]["highway"] == "primary"]
Gn = graph_of(ncr_w); ncr_nodes = set(Gn)
mission_n = nearest_node(-121.97156, 37.57660, ncr_nodes)     # SR-84 / Mission Blvd (Niles Blvd continues W)
sunol_n = nearest_node(-121.88224, 37.59289, ncr_nodes)       # SR-84 / Main St / Pleasanton-Sunol Rd, Sunol
ncr_path = dijkstra(Gn, mission_n, sunol_n)
nb_w = [w for w in hw if name(w) == "Niles Boulevard"]
Gb = graph_of(nb_w)
nb_east = nearest_node(-121.97169, 37.57648, set(Gb))           # Niles Blvd meets Mission Blvd / SR-84
nb_west = nearest_node(-121.99566, 37.58064, set(Gb))
nb_full = dijkstra(Gb, nb_west, nb_east)
# Devout Coffee from OSM
dev = [n for n in NODES.values() if n.get("tags", {}).get("name", "").lower().startswith("devout")]
dev = dev[0] if dev else {"lon": -121.9807257, "lat": 37.5772056, "id": 3424127445}
dxy = Point(ll2loc(dev["lon"], dev["lat"]))
nbl = LineString([XY[n] for n in nb_full])
s_dev = nbl.project(dxy)
# named, non-service road junctions along Niles Blvd
use = collections.defaultdict(set)
for w in hw:
    if w["tags"]["highway"] != "service" and name(w):
        for n in w["nodes"]: use[n].add(name(w))
jn = [(nbl.project(Point(XY[n])), n, sorted(use[n] - {"Niles Boulevard"})) for n in nb_full if len(use[n] - {"Niles Boulevard"})]
jn.sort()
west_of = [j for j in jn if j[0] < s_dev - 5]       # travelling W->E (from the nursery end) towards the town
blk_dev_w = west_of[-1]; blk_before_w = west_of[-2]
start_n = blk_before_w[1]
i0 = nb_full.index(start_n); nb_path = nb_full[i0:]
termini = {
    "niles_canyon_road": {
        "west_mission_blvd": {"osm_node": mission_n, "lonlat": [NODES[mission_n]["lon"], NODES[mission_n]["lat"]],
                              "desc": "SR-84 Niles Canyon Rd / Mission Blvd (SR-238) signalised junction; Niles Blvd continues W"},
        "east_sunol": {"osm_node": sunol_n, "lonlat": [NODES[sunol_n]["lon"], NODES[sunol_n]["lat"]],
                       "desc": "Sunol: SR-84 Niles Canyon Rd / Main St / Pleasanton-Sunol Rd all-way STOP (SR-84 continues E as Paloma Rd to I-680)"},
        "length_osm_m": round(LineString([XY[n] for n in ncr_path]).length, 1)},
    "niles_blvd_start": {"osm_node": start_n, "lonlat": [NODES[start_n]["lon"], NODES[start_n]["lat"]],
                         "cross_streets": blk_before_w[2],
                         "desc": f"start of the block before the Devout Coffee block (block W of {blk_dev_w[2]}), "
                                 f"{s_dev - blk_before_w[0]:.0f} m W of Devout Coffee along Niles Blvd",
                         "devout_coffee": {"osm_node": dev["id"], "lonlat": [dev["lon"], dev["lat"]], "s_along_niles_blvd_m": round(s_dev - blk_before_w[0], 1)},
                         "niles_blvd_end": {"osm_node": nb_east, "desc": "Niles Blvd / Mission Blvd / SR-84"}},
}
# ---------------- scope ----------------
nb_line = LineString([XY[n] for n in nb_path]); ncr_line = LineString([XY[n] for n in ncr_path])
rail_w = [w for w in WAYS if w.get("tags", {}).get("railway") in ("rail", "platform", "station", "light_rail", "disused", "abandoned", "narrow_gauge")]
stations = [n for n in NODES.values() if n.get("tags", {}).get("railway") in ("station", "halt")]
town = nb_line.buffer(230)
town = unary_union([town] + [Point(XY[s["id"]]).buffer(80) for s in stations if town.distance(Point(XY[s["id"]])) < 150])
town = town.simplify(5)
pal = np.genfromtxt(DER / "centerline.csv", delimiter=",", names=True, max_rows=400)
pal_xy = np.c_[pal["x_local"], pal["y_local"]]
pal_s = Point(pal_xy[0]); pal_stub = LineString(pal_xy[:300])
conn_zone = pal_s.buffer(120); sunol_zone = Point(XY[sunol_n]).buffer(90)
ncr_set = {w["id"] for w in ncr_w}
scope_ways = []
for w in hw:
    L = wline(w); t = w["tags"]; role = None
    if w["id"] in ncr_set and any(n in set(ncr_path) for n in w["nodes"]):
        role = "canyon"
    elif L.intersects(town):
        role = "town"
    elif L.intersects(conn_zone) and t["highway"] != "service":
        role = "palomares_connector"
    elif L.intersects(sunol_zone) and t["highway"] != "service":
        role = "sunol_stub"
    elif L.distance(ncr_line) < 25 and t["highway"] != "service" and name(w) in (None, "Niles Canyon Road", "Main Street", "Old Canyon Road"):
        role = "canyon_side"
    if role: scope_ways.append((w, role))
# trim town / stub ways to their zone (keep contiguous node runs inside it, plus one node beyond each end)
ZONE = {"town": town.buffer(15), "palomares_connector": pal_s.buffer(160), "sunol_stub": Point(XY[sunol_n]).buffer(160)}
trimmed = []
for w, role in scope_ways:
    if role not in ZONE: trimmed.append((w, role)); continue
    ns = w["nodes"]; inz = [ZONE[role].contains(Point(XY[n])) for n in ns]
    runs, cur = [], []
    for i, f in enumerate(inz):
        if f or (i + 1 < len(ns) and inz[i + 1]) or (i > 0 and inz[i - 1]): cur.append(ns[i])
        else:
            if len(cur) > 1: runs.append(cur)
            cur = []
    if len(cur) > 1: runs.append(cur)
    for k, r in enumerate(runs):
        trimmed.append((dict(w, nodes=r, trimmed=len(r) < len(ns)), role))
scope_ways = trimmed
# ---------------- network graph ----------------
cnt = collections.Counter(n for w, _ in scope_ways for n in w["nodes"])
ends = {n for w, _ in scope_ways for n in (w["nodes"][0], w["nodes"][-1])}
JN = {n for n in cnt if cnt[n] > 1} | ends
ctrl = {nid: n["tags"].get("highway") for nid, n in NODES.items() if n.get("tags", {}).get("highway") in ("traffic_signals", "stop", "give_way")}
edges = []
for w, role in scope_ways:
    ns = w["nodes"]; cut = [0] + [i for i in range(1, len(ns) - 1) if ns[i] in JN] + [len(ns) - 1]
    for a, b in zip(cut[:-1], cut[1:]):
        seg = ns[a:b + 1]; L = LineString([XY[n] for n in seg])
        t = w["tags"]
        edges.append(feat(line_ll(np.array(L.coords)), way=w["id"], u=seg[0], v=seg[-1], length_m=round(L.length, 1), role=role,
                          name=t.get("name"), ref=t.get("ref"), highway=t["highway"], service=t.get("service"),
                          lanes=t.get("lanes"), width=t.get("width"), surface=t.get("surface"), oneway=t.get("oneway"),
                          maxspeed=t.get("maxspeed"), bridge=t.get("bridge"), tunnel=t.get("tunnel"), layer=t.get("layer"),
                          sidewalk=t.get("sidewalk"), lit=t.get("lit")))
dump_geojson(DER7 / "network.geojson", edges)
jf = []
deg = collections.Counter()
for e in edges:
    deg[e["properties"]["u"]] += 1; deg[e["properties"]["v"]] += 1
for n in JN:
    names = sorted({name(w) or w["tags"]["highway"] for w, _ in scope_ways if n in w["nodes"]})
    near_ctrl = sorted({c for cid, c in ctrl.items() if np.hypot(XY[cid][0] - XY[n][0], XY[cid][1] - XY[n][1]) < 25})
    jf.append(feat({"type": "Point", "coordinates": [NODES[n]["lon"], NODES[n]["lat"]]}, osm_node=n, degree=deg[n],
                   x_local=round(XY[n][0], 2), y_local=round(XY[n][1], 2), roads=names, control=near_ctrl,
                   kind="dead_end" if deg[n] == 1 else "junction"))
dump_geojson(DER7 / "junctions.geojson", jf)
# ---------------- named-road centerlines ----------------
by_name = collections.defaultdict(list)
for w, role in scope_ways:
    t = w["tags"]
    if t["highway"] in ("service",) or not name(w): continue
    by_name[name(w)].append(w)
by_name["Palomares Road"] = [w for w, r in scope_ways if name(w) == "Palomares Road"]
roads = {}


def write_cl(key, nodes_seq, meta):
    xy = np.array([XY[n] for n in nodes_seq]); L = LineString(xy)
    ss = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    ux, uy = xy[:, 0] + X0, xy[:, 1] + Y0; lon, lat = loc2ll(xy[:, 0], xy[:, 1])
    np.savetxt(DER7 / f"osm_centerline_{key}.csv", np.c_[ss, xy, ux, uy, lon, lat, nodes_seq], delimiter=",",
               fmt=["%.2f", "%.3f", "%.3f", "%.3f", "%.3f", "%.7f", "%.7f", "%d"], header="s_m,x_local,y_local,x_utm,y_utm,lon,lat,osm_node", comments="")
    roads[key] = dict(meta, length_osm_m=round(L.length, 1), n_nodes=len(nodes_seq), start_node=int(nodes_seq[0]), end_node=int(nodes_seq[-1]))
    return feat(line_ll(xy), road=key, **meta, length_m=round(L.length, 1))


clf = [write_cl("niles_canyon_road", ncr_path, {"name": "Niles Canyon Road", "ref": "CA 84", "from": "Mission Blvd", "to": "Sunol stop (Main St)"}),
       write_cl("niles_blvd", nb_path, {"name": "Niles Boulevard", "from": "start block (W of Devout Coffee)", "to": "Mission Blvd / SR-84"})]
for nm, ws in sorted(by_name.items()):
    if nm in ("Niles Canyon Road", "Niles Boulevard"): continue
    G = graph_of(ws); seen = set(); comps = []
    for n0 in G:
        if n0 in seen: continue
        st = [n0]; c = set()
        while st:
            u = st.pop()
            if u in c: continue
            c.add(u); st += [v for v, _, _ in G[u]]
        seen |= c; comps.append(c)
    comps.sort(key=len, reverse=True)
    for k, c in enumerate(comps):
        a = next(iter(c)); dist, _ = dijkstra(G, a); a = max(dist, key=dist.get)
        dist, _ = dijkstra(G, a); b = max(dist, key=dist.get)
        if dist[b] < 30: continue
        path = dijkstra(G, a, b)
        # consistent direction: W->E / S->N on the dominant axis
        if XY[path[0]][0] + XY[path[0]][1] > XY[path[-1]][0] + XY[path[-1]][1]: path = path[::-1]
        key = slug(nm) + (f"_{k + 1}" if len(comps) > 1 and k > 0 else "")
        role = collections.Counter(r for w, r in scope_ways if any(w is x for x in ws)).most_common(1)[0][0]
        clf.append(write_cl(key, path, {"name": nm, "role": role}))
# unnamed SR-84 -> Palomares connector link(s)
link = [w for w, r in scope_ways if r == "palomares_connector" and w["tags"]["highway"].endswith("_link")]
for k, w in enumerate(link):
    clf.append(write_cl(f"palomares_link_{k + 1}", w["nodes"], {"name": "SR-84 / Palomares Rd link", "role": "palomares_connector", "way": w["id"]}))
dump_geojson(DER7 / "osm_centerlines.geojson", clf)
# ---------------- features ----------------
terrain_area = unary_union([town, ncr_line.buffer(TERRAIN_BUF)] + [wline(w).buffer(TERRAIN_BUF).intersection(conn_zone.buffer(200).union(sunol_zone.buffer(100))) for w, r in scope_ways if r in ("palomares_connector", "sunol_stub")])
ff = []
for w in WAYS:
    t = w.get("tags", {}); ns = w["nodes"]
    if not ns or any(n not in XY for n in ns): continue
    L = wline(w)
    if not L.intersects(terrain_area): continue
    closed = ns[0] == ns[-1] and len(ns) > 3
    geom = Polygon([XY[n] for n in ns]) if closed else L
    g = mapping(geom); xy = np.array(geom.exterior.coords if closed else geom.coords)
    lon, lat = loc2ll(xy[:, 0], xy[:, 1]); cc = np.c_[np.round(lon, 7), np.round(lat, 7)].tolist()
    gj = {"type": "Polygon", "coordinates": [cc]} if closed else {"type": "LineString", "coordinates": cc}
    kind = None
    if t.get("amenity") == "parking": kind = "parking"
    elif t.get("railway") in ("rail", "narrow_gauge", "light_rail", "disused", "abandoned", "subway"): kind = "rail_" + t["railway"]
    elif t.get("railway") == "platform" or t.get("public_transport") == "platform" and "railway" in t: kind = "rail_platform"
    elif "building" in t: kind = "building"
    elif "barrier" in t: kind = "barrier"
    elif "waterway" in t: kind = "waterway"
    elif t.get("highway") == "service" and t.get("service") == "parking_aisle": kind = "parking_aisle"
    elif t.get("man_made") == "bridge": kind = "bridge_area"
    if kind is None and not (t.get("bridge") or t.get("tunnel")): continue
    if kind is None: kind = "highway_" + t.get("highway", "other")
    if t.get("bridge") and t.get("bridge") != "no": kind += "|bridge"
    if t.get("tunnel") and t.get("tunnel") != "no": kind += "|tunnel"
    ff.append(feat(gj, kind=kind, osm_way=w["id"], **{k: v for k, v in t.items() if k in ("name", "operator", "railway", "building", "barrier", "waterway", "highway", "bridge", "tunnel", "layer", "parking", "surface", "capacity", "height", "building:levels", "service", "usage", "electrified", "gauge", "tracks", "material")}))
# culvert / stream crossing candidates: waterways crossing scope roads without a bridge
road_lines = [(wline(w), w) for w, r in scope_ways if w["tags"]["highway"] != "service"]
for w in WAYS:
    t = w.get("tags", {})
    if "waterway" not in t: continue
    WL = wline(w)
    for RL, rw in road_lines:
        if WL.crosses(RL) or WL.intersects(RL):
            p = WL.intersection(RL); p = p if p.geom_type == "Point" else p.centroid
            br = rw["tags"].get("bridge") not in (None, "no")
            lon, lat = loc2ll(p.x, p.y)
            ff.append(feat({"type": "Point", "coordinates": [round(float(lon), 7), round(float(lat), 7)]}, kind="stream_crossing_bridge" if br else "culvert_candidate",
                           waterway=t.get("waterway"), stream=t.get("name"), road=name(rw), road_way=rw["id"], tunnel=t.get("tunnel")))
# railway over road (underpasses) and level crossings
rl = [(wline(w), w) for w in WAYS if w.get("tags", {}).get("railway") in ("rail", "narrow_gauge", "subway")]
for RL, rw in road_lines:
    for L, w in rl:
        if L.intersects(RL):
            p = L.intersection(RL); p = p if p.geom_type == "Point" else p.centroid
            rb = w["tags"].get("bridge") not in (None, "no"); hb = rw["tags"].get("bridge") not in (None, "no")
            tun = w["tags"].get("tunnel") not in (None, "no")
            kind = "rail_underpass" if rb else ("road_over_rail" if hb else ("rail_tunnel_under_road" if tun else "level_crossing_or_unknown"))
            lon, lat = loc2ll(p.x, p.y)
            ff.append(feat({"type": "Point", "coordinates": [round(float(lon), 7), round(float(lat), 7)]}, kind=kind, road=name(rw), road_way=rw["id"],
                           rail=w["tags"].get("name"), rail_way=w["id"], operator=w["tags"].get("operator")))
for nid, n in NODES.items():
    t = n.get("tags", {})
    if not t: continue
    p = Point(XY[nid])
    if not terrain_area.contains(p): continue
    k = None
    if t.get("railway") in ("level_crossing", "crossing", "station", "halt", "signal", "buffer_stop", "switch"): k = "rail_" + t["railway"]
    elif t.get("highway") in ("traffic_signals", "stop", "give_way", "crossing"): k = t["highway"]
    elif t.get("amenity") or t.get("shop"): k = "poi"
    elif t.get("public_transport"): k = "bus_stop"
    if k:
        ff.append(feat({"type": "Point", "coordinates": [n["lon"], n["lat"]]}, kind=k, osm_node=nid, **{kk: v for kk, v in t.items() if kk in ("name", "amenity", "shop", "operator", "crossing", "railway", "highway", "direction", "crossing:barrier", "crossing:light")}))
dump_geojson(DER7 / "osm_features.geojson", ff)
# ---------------- scope polygons ----------------
road_geoms = [wline(w) for w, r in scope_ways]
park = [Polygon([XY[n] for n in w["nodes"]]) for w in WAYS if w.get("tags", {}).get("amenity") == "parking" and w["nodes"][0] == w["nodes"][-1] and town.buffer(50).intersects(wline(w))]
rail_town = [wline(w) for w in rail_w if wline(w).intersects(town)]
detail = unary_union([g.buffer(DETAIL_BUF) for g in road_geoms] + [p.buffer(10) for p in park] + [g.buffer(25) for g in rail_town]).simplify(2)
terrain = unary_union([terrain_area, detail.buffer(5), town]).simplify(4)
json.dump({"town": town.wkt, "detail_40m": detail.wkt, "terrain_150m": terrain.wkt, "crs": "local = EPSG:26910 - origin",
           "areas_km2": {"town": round(town.area / 1e6, 3), "detail": round(detail.area / 1e6, 3), "terrain": round(terrain.area / 1e6, 3)}},
          open(DER7 / "scope.json", "w"))
dump_geojson(DER7 / "scope.geojson", [feat({"type": "Polygon", "coordinates": [np.c_[loc2ll(*np.array(g.exterior.coords).T)].round(7).tolist()]}, area=k)
                                      for k, G in (("town", town), ("detail_40m", detail), ("terrain_150m", terrain))
                                      for g in (G.geoms if G.geom_type == "MultiPolygon" else [G])])
termini["palomares_south_end"] = {"lonlat": list(map(float, loc2ll(*pal_xy[0]))), "dist_to_nearest_ncr_m": round(ncr_line.distance(pal_s), 1),
                                  "ncr_station_m": round(ncr_line.project(pal_s), 1), "note": "Phase-1 refined centerline s=0"}
termini["roads"] = roads
json.dump(termini, open(DER7 / "termini.json", "w"), indent=1)
ext = np.array([XY[n] for w, _ in scope_ways for n in w["nodes"]])
r = np.hypot(ext[:, 0], ext[:, 1])
print(json.dumps({k: v for k, v in termini.items() if k != "roads"}, indent=1))
print("edges", len(edges), "junctions", len(jf), "features", len(ff), "roads", len(roads))
print("max dist from origin over scope roads: %.0f m" % r.max(), "areas", json.load(open(DER7 / "scope.json"))["areas_km2"])
for k, v in roads.items(): print(k, v["length_osm_m"], v.get("role"))
