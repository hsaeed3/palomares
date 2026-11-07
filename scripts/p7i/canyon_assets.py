#!/usr/bin/env python3
"""dev7 I: Niles Canyon Rd ledger assets -> $BUILD/palomares_road/palomares_niles_assets.kn5 (+ kerbs kn5s).

  PALOMARES_BUILD_DIR=build_dev7 python3 scripts/p7i/canyon_assets.py [--town]   (about 30 s)
  --town: every other dev7 road (Niles / Sunol clusters, Mission Blvd) -> palomares_niles_town_assets.kn5

* rows of data/scene/ledger_dev7.jsonl with road == niles_canyon_road; dev5 types via assets.generate on the
  "ledger" atlas (material nc_ledger), dev7 types via assets_dev7.generate_parts (nc_ledger7 + emissive parts).
  The dev7 atlas is prefilled from ALL ledger_dev7 rows (lazy sign / name tiles) before it is written.
* placement follows 21e: point objects at road.outer + off on the row's side, x along the road (signs face the
  driver of params.dir), linear objects swept along the outer edge from s to s1, buildings with a footprint at
  its centroid (world-aligned), bridges over the road centred on the centreline with the span across the road.
* road clearance (21g rule): nothing between 2 cm and 4.5 m above the asphalt inside the asphalt + 0.3 m:
  small objects are pushed out along the normal, triangles of long ones are dropped (overhead spans stay).
* collision: generator boxes / lines -> 1WALL boxes -> collision_conv.rebuild (community convention: slabs per
  side, one box per building, small props none). Nothing of this module edits collision_conv.
* kerbs: data/scene/kerbs_dev7/<layout>.npz (scripts/p7k/canyon_kerbs.py) -> palomares_niles_kerbs_{nb,sb}.kn5.
"""
import json, math, os, pathlib, sys, time, zlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/p2"), str(ROOT / "scripts/p5"), str(ROOT / "scripts/p7g")]
os.environ.setdefault("PALOMARES_BUILD_DIR", str(ROOT / "build_dev7"))
import geom                                        # noqa: E402
from geom import Raster, DEV7, load_centerline     # noqa: E402
import network as NW                               # noqa: E402
import assets as AS                                # noqa: E402
import assets_dev7 as D7                           # noqa: E402
import collision_conv as CC                        # noqa: E402
from kn5 import Material, Mesh, write_kn5, read_kn5, ac_from_local   # noqa: E402
from scipy.spatial import cKDTree                  # noqa: E402

BUILD = pathlib.Path(os.environ["PALOMARES_BUILD_DIR"]); TRACK = BUILD / "palomares_road"
assert BUILD.resolve() != (ROOT / "build").resolve(), "never write to build/"
ROADN = "niles_canyon_road"; CHUNK = 1000.0; CLEAR = 0.30
BASE = {"ksAmbient": 0.5, "ksDiffuse": 0.5, "ksSpecular": 0.0, "ksSpecularEXP": 1.0, "ksEmissive": (0, 0, 0), "ksAlphaRef": 0.0}

GROUP = "town" if "--town" in sys.argv else "canyon"
ROADS = {}


def set_road(name):
    """switch the module's current road (placement frame of a ledger row)"""
    global road, N, KD
    if name not in ROADS:
        r = NW.load_road(name); ROADS[name] = (r, len(r.s), cKDTree(np.c_[r.x, r.y]))
    road, N, KD = ROADS[name]


set_road(ROADN)
# whole-network pavement for the clearance rule: nothing on ANY road's asphalt (town streets cross each other)
_pv = []
for _nm in NW.roles():
    try:
        _r = NW.load_road(_nm)
    except Exception:
        continue
    _n = len(_r.s); _e = np.zeros(_n, bool); _e[:3] = _e[-3:] = True
    _pv.append((_r.x, _r.y, _r.nx, _r.ny, _r.el, _r.er, _r.z, _r.cl, _r.cr, _e))
PV = [np.concatenate(a) for a in zip(*_pv)]; PKD = cKDTree(np.c_[PV[0], PV[1]])
dtm = Raster(DEV7 / "dtm_r050_b40.tif")
GROUND_FREE = ("building", "bridge", "marking", "driveway", "tree_add")


def ground(x, y):
    return dtm.sample(np.asarray(x, float), np.asarray(y, float))


def station_lateral(P):
    _, j = KD.query(P[:, :2]); off = (P[:, 0] - road.x[j]) * road.nx[j] + (P[:, 1] - road.y[j]) * road.ny[j]
    return road.s[j], off, j


def on_asphalt(P, clear=CLEAR):
    X, Y, NX, NY, EL, ER, Z, CL, CR, END = PV
    _, j = PKD.query(P[:, :2]); off = (P[:, 0] - X[j]) * NX[j] + (P[:, 1] - Y[j]) * NY[j]
    half = np.where(off > 0, EL[j], ER[j]); zr = Z[j] + np.where(off > 0, CL[j] * off, CR[j] * off)
    return (np.abs(off) < half + clear) & (P[:, 2] > zr + 0.02) & (P[:, 2] < zr + 4.5) & ~END[j], j, off, half


CLEARLOG = {"pushed": 0, "trimmed_tris": 0}


def clear_road(P, T):
    bad, j, off, half = on_asphalt(P)
    if not bad.any():
        return P, T
    if float(np.ptp(P[:, :2], 0).max()) < 12:
        side = np.sign(np.median(off)) or 1.0; k = j[bad][0]
        need = float(np.max(half[bad] + CLEAR - side * off[bad])) + 0.01
        Q = P.copy(); Q[:, 0] += side * need * PV[2][k]; Q[:, 1] += side * need * PV[3][k]
        if not on_asphalt(Q)[0].any():
            CLEARLOG["pushed"] += 1; return Q, T
    keep = ~bad[T].any(1); CLEARLOG["trimmed_tris"] += int((~keep).sum())
    return P, T[keep]


class Acc:
    def __init__(self):
        self.P, self.T, self.UV, self.N, self.n = [], [], [], [], 0

    def add(self, P, T, UV, Nn):
        if len(P) and len(T):
            P, T = clear_road(np.asarray(P, float), np.asarray(T))
            if len(T):
                self.P.append(P); self.T.append(T + self.n); self.UV.append(UV); self.N.append(Nn); self.n += len(P)


VIS = {}       # (material, chunk) -> Acc
WALLS = []     # 1WALL boxes (P, T) in world
CUR = {"cls": ""}  # class of the row being placed (buildings collide in one shared pass)
BLD_POLYS = []  # footprints of placed surveyed buildings (shapely)


def acc(mat, s):
    i = int(np.clip(round(s), 0, N - 1)); k = (mat, int(road.x[i] // CHUNK), int(road.y[i] // CHUNK))   # 1 km world grid
    return VIS.setdefault(k, Acc())


def wbox(X, Y, fw, L, Wd, z0, z1):
    fw = np.asarray(fw, float); fw = fw / (np.linalg.norm(fw) + 1e-12); fl = np.array([-fw[1], fw[0]]); c = np.array([X, Y])
    co = [c - fw * L / 2 - fl * Wd / 2, c + fw * L / 2 - fl * Wd / 2, c + fw * L / 2 + fl * Wd / 2, c - fw * L / 2 + fl * Wd / 2]
    P = np.array([[*q, z0] for q in co] + [[*q, z1] for q in co])
    T = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    WALLS.append((P, T, "__bld__" if CUR["cls"] == "building" else road.name))


def wribbon(Q, half, z0, z1):
    """closed welded shell along polyline Q (n,3): width 2*half, from Q.z+z0 to Q.z+z1 (one component per row)"""
    t = np.gradient(Q[:, :2], axis=0); t /= np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-12; nl = np.c_[-t[:, 1], t[:, 0]]
    n = len(Q); A = Q[:, :2] - nl * half; B = Q[:, :2] + nl * half
    P = np.r_[np.c_[A, Q[:, 2] + z0], np.c_[B, Q[:, 2] + z0], np.c_[B, Q[:, 2] + z1], np.c_[A, Q[:, 2] + z1]]
    T = []
    for k in range(n - 1):
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            p0, p1, q0, q1 = a * n + k, a * n + k + 1, b * n + k, b * n + k + 1
            T += [[p0, q0, q1], [p0, q1, p1]]
    for k, sgn in ((0, 1), (n - 1, -1)):
        c = [a * n + k for a in range(4)]
        T += ([[c[0], c[2], c[1]], [c[0], c[3], c[2]]] if sgn > 0 else [[c[0], c[1], c[2]], [c[0], c[2], c[3]]])
    WALLS.append((P, np.array(T), road.name))


def side_sgn(r):
    return 1 if r.get("side", "R") in ("L", "left") else -1


def parts_of(r, L=None):
    p = D7.ledger_params(r); p.setdefault("seed", zlib.crc32(r["id"].encode()) & 0xffff)
    e = AS.CATALOG[r["asset"]]
    if e.get("dev7"):
        ps = D7.generate_parts(r["asset"], L, p)
        return {("nc_ledger7" if k == D7.MAT else k): v for k, v in ps.items()}, ps[D7.MAT]
    m = AS.generate(r["asset"], L, p)
    return {"nc_ledger": m}, m


def frame(r):
    s = float(np.clip(r["s"], 1, road.s[-1] - 1)); i = int(round(s)); sg = side_sgn(r); off = float(r.get("off") or 0)
    t = np.array([road.tx[i], road.ty[i]]); n = np.array([road.nx[i], road.ny[i]])
    if r["cls"] == "bridge" and not AS.CATALOG[r["asset"]]["linear"] and r["asset"].startswith("bridge.rail"):
        d = 0.0
    else:
        d = sg * (road.outer(i, sg) + max(off, 0.0))
    x, y = road.x[i] + n[0] * d, road.y[i] + n[1] * d
    zs = float(road.surface_z(i, sg * road.outer(i, sg) if d else 0.0, noise=False)); zg = float(ground([x], [y])[0])
    w = min(max(off / 1.2, 0), 1) if d else 0.0
    z = (zs * (1 - w) + zg * w if np.isfinite(zg) else zs) + float(r.get("dz") or 0)
    return s, i, sg, t, n, x, y, z


_RX = ROOT / "reports/dev7/rail_crossings_ncr.json"
RAILX = json.load(open(_RX))["crossings"] if _RX.exists() else []


def rail_crossing(s):
    """G's OSM rail crossing nearest station s (prefers a bridge way), or None"""
    c = [x for x in RAILX if abs(x["ncr_s"] - s) < 60]
    return min(c, key=lambda x: (abs(x["ncr_s"] - s), "bridge" not in x["kind"])) if c else None


def snap_to_centreline(cx, ya):
    """OSM road and refined centreline differ by up to ~8 m (s 2061): move the crossing point along the rail
    direction ya until it lies on the refined centreline, so the span is centred on the real road"""
    p = np.array([cx["x_local"], cx["y_local"]], float)
    for _ in range(4):
        _, j = KD.query(p); nrm = np.array([road.nx[j], road.ny[j]])
        off = (p - [road.x[j], road.y[j]]) @ nrm; k = ya @ nrm
        if abs(k) < 0.2 or abs(off) < 0.05:
            break
        p = p - ya * off / k
    return float(p[0]), float(p[1])


def axes(r, t, n, sg):
    if r["cls"] == "bridge" and r["asset"].startswith("bridge.rail") and (cx := rail_crossing(float(r["s"]))):
        th = math.radians(cx["rail_yaw_enu_deg"]); ya = np.array([math.cos(th), math.sin(th)])   # span (local y) along the rail
        return np.array([ya[1], -ya[0]]), ya
    if r["cls"] == "sign":
        sb = (r.get("params") or {}).get("dir", "nb") == "sb"
        face = (t if sb else -t) * 0.97 - n * sg * 0.2; face /= np.linalg.norm(face); xa = -face
    elif r["cls"] == "bridge" and r["asset"].startswith("bridge.rail"):
        xa = t                                        # local y (span) across the road
    else:
        xa = t if sg > 0 else -t
    yaw = math.radians(float(r.get("yaw") or 0)); c, s_ = math.cos(yaw), math.sin(yaw)
    xa = np.array([c * xa[0] - s_ * xa[1], s_ * xa[0] + c * xa[1]])
    return xa, np.array([-xa[1], xa[0]])


def place_point(r):
    cat = AS.CATALOG[r["asset"]]; parts, main = parts_of(r)
    s, i, sg, t, n, x, y, z = frame(r)
    fp = (r.get("params") or {}).get("footprint")
    if r["asset"] == "building.town_generic":
        # surveyed storefront / depot rows replace the OSM box they stand on (no doubled buildings)
        from shapely.geometry import Polygon
        if (r.get("params") or {}).get("by_eye_row"):
            CLEARLOG["bld_dup_skipped"] = CLEARLOG.get("bld_dup_skipped", 0) + 1; return
        if fp and len(fp) >= 3:
            pg = Polygon(fp).buffer(0)
            if any(pg.intersection(q).area > 0.2 * max(pg.area, 1e-6) for q in BLD_POLYS):
                CLEARLOG["bld_dup_skipped"] = CLEARLOG.get("bld_dup_skipped", 0) + 1; return
    if r["asset"] == "building.town_generic" and fp:
        c = np.asarray(fp, float); c = (c[:-1] if np.allclose(c[0], c[-1]) else c).mean(0)
        zg = ground([c[0]], [c[1]])[0]; x, y = c; z = float(zg) if np.isfinite(zg) else z
        xa, ya = np.array([1.0, 0]), np.array([0, 1.0])
    else:
        xa, ya = axes(r, t, n, sg)
        if r["asset"].startswith("bridge.rail") and (cx := rail_crossing(float(r["s"]))):
            x, y = snap_to_centreline(cx, ya)                   # OSM crossing slid along the rail onto the refined centreline
    for mat, mb in parts.items():
        P, T, UV, Nn = mb.arrays()
        if not len(P):
            continue
        W = np.c_[x + P[:, 0] * xa[0] + P[:, 1] * ya[0], y + P[:, 0] * xa[1] + P[:, 1] * ya[1], z + P[:, 2]]
        if r["asset"] in ("building.storefront", "building.storefront_row", "building.depot") and mat == "nc_ledger7":
            from shapely.geometry import MultiPoint
            BLD_POLYS.append(MultiPoint(W[:, :2]).convex_hull)
        Nw = np.c_[Nn[:, 0] * xa[0] + Nn[:, 1] * ya[0], Nn[:, 0] * xa[1] + Nn[:, 1] * ya[1], Nn[:, 2]]
        acc(mat, s).add(W, T, UV, Nw)
    if cat["collide"]:
        for c in main.col:
            if isinstance(c, tuple) and c[0] == "box":
                _, cx, cy, L, Wd, z0, z1, yaw = c
                fw = xa * math.cos(yaw) + ya * math.sin(yaw)
                X = x + cx * xa[0] + cy * ya[0]; Y = y + cx * xa[1] + cy * ya[1]
                wbox(X, Y, fw, L, Wd, z + z0, z + z1)


def place_linear(r):
    s0 = float(r["s"]); s1 = float(r["s1"] if r.get("s1") is not None else s0 + 4.0)
    s0, s1 = sorted((float(np.clip(s0, 1, road.s[-1] - 2)), float(np.clip(s1, 1, road.s[-1] - 1))))
    s1 = max(s1, s0 + 1.0); sg = side_sgn(r); off = max(float(r.get("off") or 0), 0.0)
    ss = np.linspace(s0, s1, max(2, int(math.ceil(s1 - s0)) + 1)); idx = np.clip(np.round(ss).astype(int), 0, N - 1)
    d = sg * (road.outer(idx, np.full(len(idx), sg)) + off)
    lx = np.interp(ss, road.s, road.x) + road.nx[idx] * d; ly = np.interp(ss, road.s, road.y) + road.ny[idx] * d
    zs = road.surface_z(idx, sg * road.outer(idx, np.full(len(idx), sg)), noise=False); zg = ground(lx, ly)
    w = min(off / 1.2, 1.0); lz = np.where(np.isfinite(zg), zs * (1 - w) + zg * w, zs) + float(r.get("dz") or 0)
    cum = np.r_[0, np.cumsum(np.hypot(np.diff(lx), np.diff(ly)))]; L = float(cum[-1])
    tt = np.gradient(np.c_[lx, ly], axis=0); tt /= np.hypot(tt[:, 0], tt[:, 1])[:, None] + 1e-12
    parts, main = parts_of(r, L)

    def mp(P, Nn=None):
        a = np.clip(P[:, 0] if sg > 0 else L - P[:, 0], 0, L)
        X = np.interp(a, cum, lx); Y = np.interp(a, cum, ly); Z = np.interp(a, cum, lz)
        tx = np.interp(a, cum, tt[:, 0]); ty = np.interp(a, cum, tt[:, 1]); k = sg    # local x -> k t, y -> k n_left
        W = np.c_[X + k * (-ty) * P[:, 1], Y + k * tx * P[:, 1], Z + P[:, 2]]
        if Nn is None:
            return W
        return W, np.c_[k * (Nn[:, 0] * tx - Nn[:, 1] * ty), k * (Nn[:, 0] * ty + Nn[:, 1] * tx), Nn[:, 2]]
    for mat, mb in parts.items():
        P, T, UV, Nn = mb.arrays()
        if len(P):
            W, Nw = mp(P, Nn); acc(mat, s0).add(W, T, UV, Nw)
    if AS.CATALOG[r["asset"]]["collide"]:
        for c in main.col:
            if isinstance(c, tuple) and c[0] == "line":
                _, yl, z0, z1, half = c
                a = np.arange(0, L + 1e-6, 2.0)
                if len(a) < 2:
                    a = np.array([0, L])
                Q = mp(np.c_[a if sg > 0 else L - a, np.full(len(a), yl), np.zeros(len(a))])
                if len(Q) >= 2:
                    wribbon(Q, half, z0, z1)


def materials():
    M = {"nc_ledger": Material("nc_ledger", "ksPerPixelAT", {**BASE, "ksSpecular": 0.06, "ksSpecularEXP": 20.0, "ksAlphaRef": 0.5}, {"txDiffuse": "nc_ledger.png"}, alpha_tested=True),
         "nc_ledger7": Material("nc_ledger7", "ksPerPixelAT", {**BASE, "ksSpecular": 0.06, "ksSpecularEXP": 20.0, "ksAlphaRef": 0.5}, {"txDiffuse": "nc_ledger7.png"}, alpha_tested=True),
         "invisible": Material("invisible", "ksPerPixel", dict(BASE), {"txDiffuse": "nc_black.png"})}
    import io
    T = {"nc_ledger.png": AS.atlas().png(), "nc_ledger7.png": D7.atlas().png()}
    for mt in D7.EMISSIVE:
        b = io.BytesIO(); D7.emissive_tex(mt).save(b, "PNG"); T[f"{mt}.png"] = b.getvalue()
        M[mt] = Material(mt, "ksPerPixel", {**BASE, "ksAmbient": 0.7, "ksDiffuse": 0.7}, {"txDiffuse": f"{mt}.png"})
    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", (64, 64)).save(b, "PNG"); T["nc_black.png"] = b.getvalue()
    return M, T


def vnorm(N):
    N = np.asarray(N, float); return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


def main():
    t0 = time.time()
    rows = [json.loads(l) for l in open(ROOT / "data/scene/ledger_dev7.jsonl")]
    D7.prefill_atlas(rows)
    for aid, e in AS.CATALOG.items():                      # dev5 sign faces paint lazily too
        if not e.get("dev7"):
            try:
                AS.generate(aid, 10.0 if e["linear"] else None, {"seed": 7})
            except Exception:
                pass
    roads_ok = set(NW.roles())
    if GROUP == "canyon":
        rr = [r for r in rows if r["road"] == ROADN]
    else:                                     # town: every other dev7 road of the network (Palomares rows live in 21c/21e)
        rr = [r for r in rows if r["road"] not in (ROADN, "palomares_road") and r["road"] in roads_ok]
    rr = [r for r in rr if r.get("status", "placed") != "removed" and r["asset"] in AS.CATALOG]
    skipped = {}
    rr.sort(key=lambda r: r["asset"] not in ("building.storefront", "building.storefront_row", "building.depot"))
    for r in rr:
        e = AS.CATALOG[r["asset"]]
        try:
            set_road(r["road"]); CUR["cls"] = r["cls"]; (place_linear if e["linear"] else place_point)(r)
        except Exception as ex:      # noqa: BLE001
            skipped[r["id"]] = f"{r['asset']}: {ex}"
    tagp = "nc" if GROUP == "canyon" else "nt"
    meshes = []
    for (mat, cx_, cy_), a_ in sorted(VIS.items()):
        if not a_.P:
            continue
        P = np.concatenate(a_.P); T = np.concatenate(a_.T); UV = np.concatenate(a_.UV); Nn = vnorm(np.concatenate(a_.N))
        meshes.append(dict(name=f"{tagp}_{mat}_{cx_}_{cy_}", P=P, T=T, UV=UV, N=Nn, material=mat,
                           cast_shadows=mat in ("nc_ledger", "nc_ledger7"), lod_out=1200.0 if mat.startswith("nc_ledger") else 600.0))
    # G's walls (bridge parapets etc.) already collide: drop ours within 2.5 m of any of them
    gw = []
    for f in sorted(TRACK.glob("palomares_niles_*.kn5")):
        if "assets" in f.name or "kerbs" in f.name or "timing" in f.name:
            continue
        for m in read_kn5(str(f))["meshes"]:
            if m["name"].startswith("1WALL") and len(m["pos"]):
                Pm = np.asarray(m["pos"], float); gw.append(np.c_[Pm[:, 0], -Pm[:, 2]])
    if gw:
        GKD = cKDTree(np.concatenate(gw))
        WALLS[:] = [w for w in WALLS if GKD.query(w[0][:, :2])[0].min() > 2.5]
    WALLS[:] = [w for w in WALLS if not on_asphalt(w[0], 0.0)[0].any()]          # input boxes on any asphalt: none
    # collision per road frame (collision_conv classifies in the road frame), names tagged per road
    cols = []; st = {}
    for rn in sorted({w[2] for w in WALLS}):
        G = {"assets": [dict(name=f"1WALL_{tagp}_obj{k:04d}", P=P, T=T, N=np.zeros_like(P), UV=np.zeros((len(P), 2)), material="invisible")
                        for k, (P, T, rn_) in enumerate(WALLS) if rn_ == rn]}
        # buildings of every road in ONE pass (overlapping footprints merge into one box; frame-independent),
        # walls / fences per road frame
        cl = load_centerline(None, DEV7 / f"centerline_{ROADN if rn == '__bld__' else rn}.csv")
        st[rn] = CC.rebuild(G, cl, None, log=lambda *a: None)
        tag = "" if GROUP == "canyon" else ("bld_" if rn == "__bld__" else f"{rn[:10]}_")
        for m in (m for g in G.values() for m in g if m["name"].startswith("1WALL")):
            m["name"] = m["name"].replace("1WALL_", f"1WALL_{tag}", 1); cols.append(m)
    # clearance on collision too: drop whole collider components that touch any asphalt (trimming triangles would
    # leave open, zero-thickness shells: qa_ffb_safety watertight / thickness gates)
    for m in cols:
        P = np.asarray(m["P"], float); T = np.asarray(m["T"]).reshape(-1, 3); bad = on_asphalt(P, 0.0)[0]
        if not bad.any():
            continue
        keep = []
        for comp in CC.components(P, T):
            cs = set(int(v) for v in comp)
            if not bad[list(cs)].any():
                keep += [t for t in T if int(t[0]) in cs]
        CLEARLOG["col_components_dropped"] = CLEARLOG.get("col_components_dropped", 0) + 1
        m["T"] = np.asarray(keep).reshape(-1, 3)
    M, TX = materials()
    nodes = []
    for m in meshes:
        nodes.append(Mesh(m["name"], ac_from_local(m["P"]), ac_from_local(m["N"]), m["UV"], m["T"].reshape(-1), m["material"],
                          cast_shadows=m["cast_shadows"], lod_out=m["lod_out"]))
    for m in cols:
        if len(m["T"]):
            Nn = m.get("N"); Nn = vnorm(Nn) if Nn is not None and len(Nn) == len(m["P"]) and np.abs(Nn).sum() > 0 else np.tile([0, 0, 1.0], (len(m["P"]), 1))
            nodes.append(Mesh(m["name"], ac_from_local(m["P"]), ac_from_local(Nn), np.zeros((len(m["P"]), 2)), np.asarray(m["T"]).reshape(-1), "invisible",
                              cast_shadows=False, visible=True, renderable=False))
    used = sorted({n.material for n in nodes})
    TRACK.mkdir(parents=True, exist_ok=True)
    out = TRACK / ("palomares_niles_assets.kn5" if GROUP == "canyon" else "palomares_niles_town_assets.kn5")
    write_kn5(out, [M[u] for u in used], {t: TX[t] for t in sorted({v for u in used for v in M[u].textures.values()})}, nodes)
    rt = read_kn5(str(out)); vmax = max(len(m["pos"]) for m in rt["meshes"])
    rep = dict(rows=len(rr), skipped=skipped, clear=CLEARLOG, collision=st, kn5=out.name, meshes=len(rt["meshes"]), vmax=vmax,
               draw_calls=sum(1 for m in rt["meshes"] if m.get("renderable", True) and not m["name"].startswith("1")), secs=round(time.time() - t0, 1))
    if GROUP == "canyon":
        write_kerbs(rep)
    (BUILD / "dev7").mkdir(exist_ok=True); (BUILD / f"dev7/{GROUP}_assets.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "skipped"}, default=str), "skipped", len(skipped))


def write_kerbs(rep):
    """kerb meshes from canyon_kerbs.py npz -> palomares_niles_kerbs_nb/sb.kn5 (physics 1KERB_/1RUNOFF_, visible)"""
    import io, yaml, race_tex as RT
    from PIL import Image
    st = yaml.safe_load(open(ROOT / "config/kerbs.yaml"))["style"]["niles_canyon"]
    kd, kn = RT.kerb_atlas(blocks=[np.array(c, float) for c in st["blocks"]])
    png = lambda im: (lambda b: (im.save(b, "PNG"), b.getvalue())[1])(io.BytesIO())
    sau = Image.new("RGB", (64, 64), (236, 196, 24)); run = Image.new("RGB", (64, 64), tuple(st["runoff_rgb"]))
    TX = {"nc_kerb_d.png": png(kd), "nc_kerb_n.png": png(kn), "nc_sausage.png": png(sau), "nc_runoff.png": png(run)}
    M = {"kerb_rbr": Material("kerb_rbr", "ksPerPixelNM", {**BASE, "ksSpecular": 0.18, "ksSpecularEXP": 24.0}, {"txDiffuse": "nc_kerb_d.png", "txNormal": "nc_kerb_n.png"}),
         "kerb_sausage": Material("kerb_sausage", "ksPerPixel", {**BASE, "ksSpecular": 0.15, "ksSpecularEXP": 20.0}, {"txDiffuse": "nc_sausage.png"}),
         "runoff_green": Material("runoff_green", "ksPerPixel", {**BASE, "ksSpecular": 0.08, "ksSpecularEXP": 15.0}, {"txDiffuse": "nc_runoff.png"})}
    from geom import vertex_normals
    for lay, dname in (("canyon_track", "nb"), ("canyon_rev_track", "sb")):
        z = np.load(ROOT / f"data/scene/kerbs_dev7/{lay}.npz"); nodes = []
        for key in sorted({k[:-3] for k in z.files if k.endswith("__P")}):
            P = z[key + "__P"].astype(float); T = z[key + "__T"].astype(np.int64); UV = z[key + "__UV"].astype(float); mat = str(z[key + "__mat"])
            if mat == "kerb_rbr":
                UV = np.c_[0.51 + 0.48 * UV[:, 0], UV[:, 1]]             # exit half of the kerb atlas (flat paint)
            Nn = vertex_normals(P, T)
            nodes.append(Mesh(key, ac_from_local(P), ac_from_local(Nn), UV, T.reshape(-1), mat, cast_shadows=False))
        used = sorted({n.material for n in nodes}); f = TRACK / f"palomares_niles_kerbs_{dname}.kn5"
        write_kn5(f, [M[u] for u in used], {t: TX[t] for t in sorted({v for u in used for v in M[u].textures.values()})}, nodes)
        rep.setdefault("kerbs", {})[f.name] = len(nodes)


if __name__ == "__main__":
    with geom.build_lock("dev7_canyon_assets") if hasattr(geom, "build_lock") else open(os.devnull):
        main()
