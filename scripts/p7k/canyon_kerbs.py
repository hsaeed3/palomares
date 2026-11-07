#!/usr/bin/env python3
"""dev7 K: Niles Canyon track kerbs (Red-Bull-Ring style) from the AI racing line, plus the lint check.

  python3 scripts/p7k/canyon_kerbs.py            # place + mesh + lint  (about 10 s)
  python3 scripts/p7k/canyon_kerbs.py --lint     # lint only (also run by QA)

Rules (config/kerbs.yaml section niles_canyon; the Palomares rules are reused where they apply):
  * the racing line per run direction: scripts/p2/racing_line.py on G's road (scripts/p7g/network.py load_road),
    cached in data/scene/cache/racing_line_niles_canyon_road.pkl (never build/racing_line.pkl = Palomares)
  * apex + exit spans: kerbs.line_rows (same labelling as the Palomares track: inside touch while turning = apex,
    outside touch after an apex on the other side = exit), dropped at junctions / level crossings / bridges / drops
  * NOTHING ON THE PAVEMENT: every kerb, run-off and sausage kerb lies beyond road.outer (asphalt + paved shoulder)
    + clear_m. The lint asserts this per vertex against every dev7 road's pavement (not only Niles Canyon Rd).
  * run-off: green paint outside exit kerbs where the verge is flat on the DTM (allowed_env, max rise / step)
  * sausage kerb: yellow, low (height_m), half-cosine section (no sharp lip: < 15 deg slope, creases < 25 deg =
    qa_ffb_safety KERB_CREASE), tapered ends, at the outer edge of each run-off = the extended track limit.
Direction: canyon_track (and freeroam_track) = nb = from the Mission Blvd start; canyon_rev_track = sb.
Outputs data/scene/kerbs_dev7/placements.json and <layout>.npz (physics meshes 1KERB_/1RUNOFF_ with P, T, UV,
material; local scene frame z up) for the integration step (I). Collision: the meshes ARE the drivable surface
(1KERB / 1RUNOFF prefixes), no walls.
"""
import argparse, json, pathlib, sys, time
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/p2"), str(ROOT / "scripts/p7g")]
import network as NW                                   # noqa: E402
import racing_line as RL                               # noqa: E402
import kerbs as KB                                     # noqa: E402
from geom import Raster, DEV7, grid_tris               # noqa: E402

OUT = ROOT / "data/scene/kerbs_dev7"
KCFG = yaml.safe_load(open(ROOT / "config/kerbs.yaml"))
NC = KCFG["niles_canyon"]
ROAD = "niles_canyon_road"
LAYOUTS = {"canyon_track": "nb", "canyon_rev_track": "sb", "freeroam_track": "nb"}


def racing(road):
    RL.CACHE = ROOT / "data/scene/cache/racing_line_niles_canyon_road.pkl"
    RL.CACHE.parent.mkdir(parents=True, exist_ok=True)
    return RL.compute(road)


def exclusions(road):
    N = len(road.s); ex = {"left": np.zeros(N, bool), "right": np.zeros(N, bool)}
    skip = np.zeros(N, bool)
    for a, b in NC["skip_s_ranges"]:
        skip[(road.s >= a) & (road.s <= b)] = True
    for f in json.load(open(DEV7 / "features.geojson"))["features"]:
        p = f["properties"]
        if p.get("road") != ROAD or p["kind"] not in NC["clear_kinds"]:
            continue
        c = NC["clear_kinds"][p["kind"]]
        s0 = p.get("s_start_m", p.get("s_m", 0)); s1 = p.get("s_end_m", s0)
        m = (road.s >= s0 - c) & (road.s <= s1 + c)
        sd = p.get("side")
        for side in ("left", "right"):
            if sd in (None, "both", side):
                ex[side] |= m
    return skip, ex


def spans(mask, minlen):
    out = []
    for r in KB.runs(mask):
        if len(r) >= minlen:
            out.append((int(r[0]), int(r[-1])))
    return out


def place():
    road = NW.load_road(ROAD); D = racing(road); N = len(road.s)
    dtm = Raster(DEV7 / "dtm_r050_b40.tif")
    skip, ex = exclusions(road)
    env = {"left": road.env_l, "right": road.env_r}
    PL = dict(KCFG["placement"], **NC.get("placement", {}))
    res = {}
    for dname in ("nb", "sb"):
        Dd = D[dname]; sd = 1 if dname == "nb" else -1
        rowlab, wide, info = KB.line_rows(road, Dd, sd, PL, env, skip, ex)
        items = []
        for side, sg in (("left", 1), ("right", -1)):
            outer = road.outer(np.arange(N), side) + NC["clear_m"]
            for v, typ in ((1, "apex"), (2, "exit")):
                T = NC["types"][typ]
                for a, b in spans(rowlab[side] == v, NC["min_kerb_length_m"]):
                    items.append(dict(dir=dname, type=typ, side=side, i0=a, i1=b, s0=float(road.s[a]), s1=float(road.s[b]),
                                      off0=outer[a:b + 1].tolist(), width=T["width_m"]))
            # run-off + sausage outside exits where the line runs wide and the ground allows
            for it in [k for k in items if k["type"] == "exit" and k["side"] == side]:
                a = max(0, it["i0"] - NC["runoff"]["pre_m"]); b = min(N - 1, it["i1"] + NC["runoff"]["post_m"])
                ii = np.arange(a, b + 1)
                o0 = outer[ii] + NC["types"]["exit"]["width_m"]; o1 = o0 + NC["runoff"]["width_m"]
                ok = np.isin(env[side][ii], NC["runoff"]["allowed_env"]) & ~ex[side][ii] & ~skip[ii]
                zr = road.surface_z(ii, sg * road.outer(ii, side), noise=False)
                for f in np.linspace(0, 1, 5):
                    o = o0 + (o1 - o0) * f; x, y = road.xy(ii, sg * o); z = dtm.sample(x, y)
                    ok &= np.isfinite(z) & (np.abs(z - zr) < NC["runoff"]["max_rise_m"])
                for ra, rb in spans(ok, NC["runoff"]["min_length_m"]):
                    j = ii[ra:rb + 1]
                    items.append(dict(dir=dname, type="runoff", side=side, i0=int(j[0]), i1=int(j[-1]), s0=float(road.s[j[0]]), s1=float(road.s[j[-1]]),
                                      off0=(outer[j] + NC["types"]["exit"]["width_m"]).tolist(), width=NC["runoff"]["width_m"]))
                    L = len(j); c0, c1 = int(L * NC["sausage"]["from_frac"]), int(L * NC["sausage"]["to_frac"])
                    if c1 - c0 >= NC["sausage"]["min_length_m"]:
                        js = j[c0:c1]
                        items.append(dict(dir=dname, type="sausage", side=side, i0=int(js[0]), i1=int(js[-1]), s0=float(road.s[js[0]]), s1=float(road.s[js[-1]]),
                                          off0=(outer[js] + NC["types"]["exit"]["width_m"] + NC["runoff"]["width_m"] + NC["sausage"]["gap_m"]).tolist(),
                                          width=NC["sausage"]["width_m"]))
        res[dname] = dict(items=items, apexes=len(info["apexes"]), stats=info["stats"])
    return road, dtm, res


def profile(typ, n):
    t = np.linspace(0, 1, n)
    if typ == "sausage":
        h = NC["sausage"]["height_m"] * 0.5 * (1 - np.cos(2 * np.pi * t))          # half-cosine bump, zero slope at both feet
    elif typ == "runoff":
        h = np.full(n, NC["runoff"]["lift_m"])
    else:
        T = NC["types"][typ]; h = T["height_m"] * np.interp(t, T["profile_t"], T["profile_h"])
    return t, h


def mesh(road, dtm, it):
    """grid mesh on the DTM (beyond the pavement), height profile across, smoothstep taper of height at the ends"""
    i = np.arange(it["i0"], it["i1"] + 1); sg = 1 if it["side"] == "left" else -1
    ncol = {"sausage": 9, "runoff": 5}.get(it["type"], 7)
    t, h = profile(it["type"], ncol)
    off = np.asarray(it["off0"])[:, None] + t[None, :] * it["width"]
    x, y = road.xy(i[:, None], sg * off)
    z = dtm.sample(x, y)
    zr = road.surface_z(i, sg * road.outer(i, it["side"]), noise=False)[:, None]
    z = np.where(np.isfinite(z), z, zr)
    z = gaussian_filter1d(gaussian_filter1d(z, 2.0, axis=0, mode="nearest"), 0.7, axis=1, mode="nearest")   # smooth base (no DTM noise crease)
    L = len(i); tp = NC["taper_m"]; d = np.minimum(np.arange(L), L - 1 - np.arange(L)) / max(tp, 1e-6)
    tap = np.clip(d, 0, 1); tap = tap * tap * (3 - 2 * tap)
    z = z + 0.004 + h[None, :] * tap[:, None]
    if sg < 0:
        x, y, z = x[:, ::-1], y[:, ::-1], z[:, ::-1]; t = t[::-1]
    P = np.stack([x, y, z], -1).reshape(-1, 3)
    if it["type"] == "runoff":
        U = np.broadcast_to(t * it["width"] / NC["runoff"]["tex_m"], (L, ncol)); V = np.repeat((road.s[i] / NC["runoff"]["tex_m"])[:, None], ncol, 1)
    else:
        U = np.broadcast_to(t, (L, ncol)); nb = len(KCFG["style"]["niles_canyon"]["blocks"])
        V = np.repeat((road.s[i] / (nb * NC["types"].get(it["type"], NC["types"]["exit"]).get("stripe_length_m", 1.0)))[:, None], ncol, 1)
    UV = np.stack([U, V], -1).reshape(-1, 2)
    return P, grid_tris(L, ncol), UV


def crease_max(P, T):
    e1 = P[T[:, 1]] - P[T[:, 0]]; e2 = P[T[:, 2]] - P[T[:, 0]]; n = np.cross(e1, e2)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    ed = {}
    for k, tri in enumerate(T):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            ed.setdefault((min(a, b), max(a, b)), []).append(k)
    mx = 0.0
    for v in ed.values():
        if len(v) == 2:
            mx = max(mx, float(np.degrees(np.arccos(np.clip(n[v[0]] @ n[v[1]], -1, 1)))))
    return mx


def pavement():
    """(kd-tree of all dev7 road stations, per-station outer half-widths) for the on-pavement lint"""
    xs, ys, ol, orr, nx, ny = [], [], [], [], [], []
    for name in NW.roles():
        try:
            r = NW.load_road(name)
        except Exception:
            continue
        i = np.arange(len(r.s))
        xs.append(r.x); ys.append(r.y); nx.append(r.nx); ny.append(r.ny); ol.append(r.outer(i, "left")); orr.append(r.outer(i, "right"))
    c = lambda a: np.concatenate(a)
    X, Y = c(xs), c(ys)
    return cKDTree(np.c_[X, Y]), X, Y, c(nx), c(ny), c(ol), c(orr)


def lint(meshes, PV):
    """every vertex beyond every road's pavement (outer) - tolerance; returns list of violations"""
    kd, X, Y, NX, NY, OL, OR = PV; bad = []
    for name, (P, T, UV, typ, s0) in meshes.items():
        idxs = kd.query_ball_point(P[:, :2], r=40.0)
        worst = 0.0
        for k, js in enumerate(idxs):
            if not js:
                continue
            js = np.asarray(js); dx = P[k, 0] - X[js]; dy = P[k, 1] - Y[js]
            along = np.abs(dx * -NY[js] + dy * NX[js]); off = dx * NX[js] + dy * NY[js]
            near = along < 0.75                                   # stations whose cross-section contains this point
            if not near.any():
                continue
            js, off = js[near], off[near]
            lim = np.where(off > 0, OL[js], OR[js]) - NC["lint_tol_m"]
            pen = (lim - np.abs(off)).max()
            worst = max(worst, pen)
        if worst > 0:
            bad.append(dict(mesh=name, type=typ, s=s0, penetration_m=round(float(worst), 3)))
    return bad


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lint", action="store_true"); a = ap.parse_args()
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    road, dtm, res = place()
    meshes = {}; counts = {}; creases = {}
    for layout, dname in LAYOUTS.items():
        items = res[dname]["items"]; arr = {}
        c = {"apex": 0, "exit": 0, "runoff": 0, "sausage": 0}
        for n_, it in enumerate(items):
            P, T, UV = mesh(road, dtm, it)
            pref = {"apex": "1KERB", "exit": "1KERB", "sausage": "1KERB", "runoff": "1RUNOFF"}[it["type"]]
            mat = {"apex": "kerb_rbr", "exit": "kerb_rbr", "sausage": "kerb_sausage", "runoff": "runoff_green"}[it["type"]]
            nm = f"{pref}_NC{dname.upper()}_{it['type'][:3].upper()}_{n_:03d}"
            arr[nm + "__P"] = P.astype(np.float32); arr[nm + "__T"] = T.astype(np.int32); arr[nm + "__UV"] = UV.astype(np.float32); arr[nm + "__mat"] = np.array(mat)
            meshes[f"{layout}/{nm}"] = (P, T, UV, it["type"], it["s0"])
            c[it["type"]] += 1
            creases[it["type"]] = max(creases.get(it["type"], 0.0), crease_max(P, T))
        np.savez_compressed(OUT / f"{layout}.npz", **arr); counts[layout] = c
    PV = pavement(); bad = lint(meshes, PV)
    rep = dict(road=ROAD, layouts=LAYOUTS, counts=counts, max_crease_deg=creases, kerb_crease_limit_deg=25.0,
               apexes={d: res[d]["apexes"] for d in res}, stats={d: res[d]["stats"] for d in res}, on_pavement=bad,
               items={d: [{k: v for k, v in it.items() if k != "off0"} | {"off_in_m": round(float(np.min(it["off0"])), 2)} for it in res[d]["items"]] for d in res},
               built=time.strftime("%Y-%m-%d %H:%M:%S"), secs=round(time.time() - t0, 1))
    (OUT / "placements.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("counts", "max_crease_deg", "on_pavement", "secs")}, indent=1))
    if bad or any(v > 25.0 for v in creases.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
