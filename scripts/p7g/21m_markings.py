#!/usr/bin/env python3
"""dev7 G, step 21m: road markings for the network -> $BUILD/palomares_road/palomares_niles_lines.kn5

    PALOMARES_BUILD_DIR=build_dev7 nice python3 scripts/p7g/21m_markings.py

Same paint as Palomares (scripts/p2/road_surface.py: MUTCD widths, DY_RIBBON / EDGE_RIBBON, BROKEN, LIFT['line'];
materials + DDS textures read back from the shipped palomares_lines_road.kn5), rules from config/dev7/markings_dev7.yaml
and the OSM oneway / lanes tags. Heights come from the BUILT physics surfaces (21n / 21t kn5, located per vertex),
so the paint follows the composite town surface exactly; lift 2.3 cm. Non-physical, no shadows, merged per 1 km.
Lines stop where they enter the pavement of a higher-priority road (junctions); edge lines also break there.
"""
import json, sys, pathlib, time
import numpy as np, yaml, shapely
import shapely.geometry as sg
from scipy.spatial import cKDTree
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0] / "p2"))
import network as NW, stitch as STI
from geom import ROOT, BUILD, DEV7, vertex_normals, grid_tris, build_lock
from kn5 import Mesh, Material, write_kn5, read_kn5, ac_from_local
import road_surface as RS

build_lock("dev7_g")
T0 = time.time(); LOG = {}
CFG7 = yaml.safe_load(open(ROOT / "config/dev7/markings_dev7.yaml"))
TRK = BUILD / "palomares_road"
# built physics surfaces (tops)
Ps, Ts, k0 = [], [], 0
for f in sorted(TRK.glob("palomares_niles_*.kn5")):
    if not any(t in f.name for t in ("_ncr", "_town", "_sunol")):
        continue
    for m in read_kn5(str(f))["meshes"]:
        if m["name"][:1].isdigit() and "WALL" not in m["name"] and "SIDEWALK" not in m["name"]:
            P = STI.enu(m["pos"]); Ps.append(P); Ts.append(m["idx"].reshape(-1, 3) + k0); k0 += len(P)
SURF = STI.TriGrid(np.concatenate(Ps), np.concatenate(Ts))
# built station ranges
rng = {"niles_canyon_road": None}
for c in ("town", "sunol"):
    f = BUILD / "dev7" / f"{c}_build.json"
    if f.exists():
        for n, v in json.load(open(f))["log"]["roads"].items():
            if n != "niles_canyon_road":
                rng[n] = v
# OSM oneway / lanes per road (nearest network way to the road's middle)
net = json.load(open(DEV7 / "network.geojson"))["features"]
ways = [(NW._to_local(sg.shape(f["geometry"])), f["properties"]) for f in net]
roads = {}
for n in rng:
    r = NW.load_road(n)
    lo, hi = (0, len(r.s) - 1) if rng[n] is None else (int(rng[n][0]), int(min(rng[n][1], len(r.s) - 1)))
    r.lo, r.hi = lo, hi
    mid = sg.Point(r.x[(lo + hi) // 2], r.y[(lo + hi) // 2])
    wg, w = min(ways, key=lambda wp: wp[0].distance(mid))
    # OSM way direction = traffic direction of a oneway: does it run with the stations?
    a_, b_ = (lo + hi) // 2, min((lo + hi) // 2 + 10, hi)
    r.with_osm = wg.project(sg.Point(r.x[b_], r.y[b_])) >= wg.project(sg.Point(r.x[a_], r.y[a_]))
    r.oneway = w.get("oneway") == "yes"; r.lanes = int(w.get("lanes") or (1 if r.oneway else 2))
    r.prio = 1e6 if n == "niles_canyon_road" else float(np.median(r.el + r.er))
    i = np.arange(lo, hi + 1)
    r.poly = sg.Polygon(np.r_[np.stack(r.xy(i, -r.er[i]), 1), np.stack(r.xy(i, r.el[i]), 1)[::-1]]).buffer(0)
    roads[n] = r
LOG["roads"] = {n: dict(oneway=r.oneway, lanes=r.lanes, with_osm=bool(r.with_osm)) for n, r in roads.items()}

groups = {}


def ribbon(name, mat, r, i, dc, width, cols=(0.0, 1.0), ucols=None, dash=None, block=None):
    """paint strip along station indices i of road r, centre offset dc (array per station or scalar)"""
    if dash:
        on, period = dash; i = i[(r.s[i] % period) < on]
    if block is not None and len(i):
        dcv = dc[i] if np.ndim(dc) else np.full(len(i), dc)
        x, y = r.xy(i, dcv); i = i[~shapely.contains_xy(block, x, y)]
    if len(i) < 2:
        return 0
    cols = np.asarray(cols); ucols = np.asarray(ucols if ucols is not None else cols); nc = len(cols)
    n = 0
    for run in np.split(i, np.flatnonzero(np.diff(i) > 1) + 1):
        if len(run) < 2:
            continue
        d0 = dc[run] if np.ndim(dc) else np.full(len(run), dc)
        dd = d0[:, None] - width / 2 + width * cols[None, :]
        rr = np.repeat(run[:, None], nc, 1); x, y = r.xy(rr, dd)
        z, t = SURF.locate(np.c_[x.ravel(), y.ravel()])
        z = np.where(np.isfinite(z), z, r.surface_z(rr.ravel(), dd.ravel(), noise=False)) + RS.LIFT["line"]
        P = np.c_[x.ravel(), y.ravel(), z]; T = grid_tris(len(run), nc)
        UV = np.c_[np.tile(ucols, len(run)), np.repeat(r.s[run] / RS.LINE_PERIOD_M, nc)]
        km = int(r.s[run[len(run) // 2]] // 1000)
        g = groups.setdefault((mat, n_key(r.name, km)), dict(P=[], T=[], UV=[], k=0))
        g["P"].append(P); g["T"].append(T + g["k"]); g["UV"].append(UV); g["k"] += len(P); n += len(run)
    return n


def n_key(road, km):
    return f"{road[:12]}{km:02d}"


EW, DY = RS.EDGE_RIBBON, RS.DY_RIBBON
SY_U = (0.0, (0.03 + RS.LINE_W + 0.03) / DY)           # single yellow = first line of the double-yellow texture
counts = {}
for n, r in roads.items():
    rule = CFG7["roads"].get(n, {"type": CFG7.get("default", "none")})
    if rule.get("type", "none") == "none":
        continue
    i = np.arange(max(r.lo, 2), min(r.hi, len(r.s) - 3) + 1)
    higher = [o.poly for m, o in roads.items() if m != n and o.prio >= r.prio]
    block = shapely.union_all(higher).buffer(0.3) if higher else None
    if block is not None:
        shapely.prepare(block)
    c = 0
    inset = 0.25                                       # edge line centre inside the pavement edge
    if rule["type"] == "two_way":
        c += ribbon(n, "paint_yellow_dbl", r, i, 0.0, DY, cols=(0.0, 0.5, 1.0), block=block)
        if rule.get("edge") == "solid":
            eb = block.buffer(1.0) if block is not None else None
            c += ribbon(n, "paint_white", r, i, r.el - inset, EW, block=eb)
            c += ribbon(n, "paint_white", r, i, -r.er + inset, EW, block=eb)
    elif rule["type"] == "one_way":
        # yellow on the driver's left (median side), white on the right; traffic runs with the OSM way
        yl, wr = (r.el - inset, -r.er + inset) if r.with_osm else (-r.er + inset, r.el - inset)
        c += ribbon(n, "paint_yellow_dbl", r, i, yl, SY_U[1] * DY, ucols=SY_U, block=block)
        c += ribbon(n, "paint_white", r, i, wr, EW, block=block)
        W = r.el + r.er
        nl = int(max(1, min(r.lanes, round(float(np.median(W[i])) / 3.4))))     # turn-lane tags overcount; ~3.4 m lanes
        for k in range(1, nl):
            c += ribbon(n, "paint_white", r, i, -r.er + W * k / nl, EW, dash=RS.BROKEN, block=block)
    counts[n] = c
LOG["line_stations"] = counts
# kn5
src = read_kn5(str(ROOT / "build/palomares_road/palomares_lines_road.kn5")); SM = {m["name"]: m for m in src["materials"]}
mats, texs = [], {}
for nm in ("paint_white", "paint_yellow_dbl"):
    m = SM[nm]; props = {p: ((v[3], v[4], v[5]) if p == "ksEmissive" else v[0]) for p, v in m["props"].items()}
    mats.append(Material(nm, m["shader"], props, dict(m["textures"]), blend=m["blend"], alpha_tested=m["alpha_tested"]))
    texs.update({t: src["textures"][t] for t in m["textures"].values()})
# merge per material per 1 km bin of the whole network (draw calls)
bins = {}
for (mat, key), g in groups.items():
    P = np.concatenate(g["P"]); km = int(np.floor(P[:, 0].mean() / 1000)) * 100 + int(np.floor(P[:, 1].mean() / 1000))
    b = bins.setdefault((mat, km), dict(P=[], T=[], UV=[], k=0))
    b["P"].append(P); b["T"].append(np.concatenate(g["T"]) + b["k"]); b["UV"].append(np.concatenate(g["UV"])); b["k"] += len(P)
nodes = []
for j, ((mat, km), b) in enumerate(sorted(bins.items(), key=lambda kv: str(kv[0]))):
    P = np.concatenate(b["P"]); T = np.concatenate(b["T"]); UV = np.concatenate(b["UV"])
    nodes.append(Mesh(f"niles_line_{mat}_{j:02d}", ac_from_local(P), ac_from_local(vertex_normals(P, T)), UV, T.reshape(-1), mat,
                      cast_shadows=False))
out = TRK / "palomares_niles_lines.kn5"
write_kn5(str(out), mats, texs, nodes)
k = read_kn5(str(out))
assert max(len(m["pos"]) for m in k["meshes"]) <= 65535
rep = dict(total_s=round(time.time() - T0, 1), meshes=len(k["meshes"]), draw_calls=len(k["meshes"]),
           tris=int(sum(len(m["idx"]) // 3 for m in k["meshes"])), log=LOG)
json.dump(rep, open(BUILD / "dev7" / "lines_build.json", "w"), indent=1)
print(json.dumps({kk: v for kk, v in rep.items() if kk != "log"}), LOG["roads"], LOG["line_stations"])
