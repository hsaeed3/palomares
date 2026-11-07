"""dev7 G: the road network as a set of geom.Road objects (one per data/derived/dev7 centerline).

Profile fixes applied here (before geom.Road smoothing), so every consumer sees the same z:
  * interpolated bridge decks (struct == bridge_interp): re-fitted as a cubic Hermite between validated abutment
    anchors (ground return, settled grade) with the outside slopes; fixes the 2.4 m jump at NCR s~4390-4404
  * grade limit (MAX_GRADE) on town streets: Limelight / Sierra Terrace end ramps are LiDAR ramps onto
    embankments/driveways, not road grade
"""
import sys, pathlib, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "p2"))
from geom import Road, load_centerline, DEV7

ROLE_MAX_GRADE = {"town": 0.12, "canyon": 0.12, "canyon_side": 0.12, "sunol_stub": 0.12, "palomares_connector": 0.16}


def roles():
    return {k: v for k, v in json.load(open(DEV7 / "centerlines_summary.json")).items()}


def _bridge_fix(s, z, struct, look=30):
    z = z.copy(); fixes = []
    b = struct == "bridge_interp"
    if not b.any():
        return z, fixes
    edges = np.flatnonzero(np.diff(np.r_[0, b.astype(int), 0]))
    for a, e in zip(edges[::2], edges[1::2]):          # deck stations [a, e)
        ia = max(a - 4, 1)
        # east/upper anchor: first station after the deck where the next 8 m are settled (|grade| < 3 %)
        ie = None
        for k in range(e, min(e + look, len(z) - 9)):
            if np.abs(np.diff(z[k:k + 9])).max() < 0.03:
                ie = k; break
        ie = ie if ie is not None else min(e + 4, len(z) - 2)
        gj = np.abs(np.diff(z[max(a - 3, 0):ie + 2])).max()
        if gj < 0.04:                                    # deck joins its approaches smoothly: keep it
            continue
        g0 = (z[ia] - z[max(ia - 15, 0)]) / max(ia - max(ia - 15, 0), 1)
        g1 = (z[min(ie + 15, len(z) - 1)] - z[ie]) / max(min(ie + 15, len(z) - 1) - ie, 1)
        g0, g1 = np.clip(g0, -0.03, 0.03), np.clip(g1, -0.03, 0.03)
        L = s[ie] - s[ia]; t = (s[ia:ie + 1] - s[ia]) / L
        h00, h10, h01, h11 = 2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t, -2*t**3 + 3*t**2, t**3 - t**2
        new = h00 * z[ia] + h10 * L * g0 + h01 * z[ie] + h11 * L * g1
        fixes.append(dict(s0=float(s[ia]), s1=float(s[ie]), dz_max=float(np.abs(new - z[ia:ie + 1]).max())))
        z[ia:ie + 1] = new
    return z, fixes


def _grade_limit(s, z, gmax, it=200):
    z = z.copy()
    for _ in range(it):
        g = np.diff(z) / np.diff(s)
        bad = np.flatnonzero(np.abs(g) > gmax)
        if not len(bad):
            break
        for i in bad:                       # pull both ends toward each other (local, keeps the mean)
            ex = (abs(z[i + 1] - z[i]) - gmax * (s[i + 1] - s[i])) / 2 * np.sign(z[i + 1] - z[i])
            z[i] += ex; z[i + 1] -= ex
    return z


def load_road(name, log=None):
    c = load_centerline(None, DEV7 / f"centerline_{name}.csv")
    role = roles()[name].get("role", "town") if "role" in roles()[name] else ("canyon" if name == "niles_canyon_road" else "town")
    z, fx = _bridge_fix(c["s_m"], c["z_navd88"], c.get("struct", np.array([""] * len(c["s_m"]))))
    z = _grade_limit(c["s_m"], z, ROLE_MAX_GRADE.get(role, 0.12))
    if name != "niles_canyon_road":          # town streets: vertical curves after the grade limit (no 0 -> 12 % kinks)
        from scipy.ndimage import gaussian_filter1d as _g
        z = _g(z, 4.0, mode="nearest")
    if log is not None:
        log[name] = dict(bridge_fixes=fx, dz_grade_limit_max=float(np.abs(z - c["z_navd88"]).max()))
    r = Road(None, name=name, z_override=z)
    # bridge decks: no ground returns on/under the deck, the cross-section crossfall there is the creek bank
    # (NCR decks read as a 10 % crown).  Use a 2 % crown on decks, blended over ~5 m at each end.
    from scipy.ndimage import gaussian_filter1d
    deck = (c.get("struct", np.array([""] * len(z))) != "") & np.char.startswith(c.get("struct", np.array([""] * len(z))).astype(str), "bridge")
    for f in json.load(open(DEV7 / "features.geojson"))["features"]:
        p = f["properties"]
        if p["kind"] == "bridge_osm" and p.get("road") == name:
            deck[int(p["s_start_m"]):int(p["s_end_m"]) + 1] = True
    if deck.any():
        w = np.clip(gaussian_filter1d(deck.astype(float), 8.0) * 2, 0, 1); w = np.maximum(w * w * (3 - 2 * w), deck)
        r.cl = (1 - w) * r.cl + w * -0.02; r.cr = (1 - w) * r.cr + w * 0.02
        if log is not None:
            log[name]["deck_crown_stations"] = int(deck.sum())
    return r


# ------------------------------------------------------------------ clusters (town cores built as one surface)
def _to_local(geom):
    from pyproj import Transformer
    import shapely.ops
    o = json.load(open(DEV7.parents[1] / "config/origin.json")) if (DEV7.parents[1] / "config/origin.json").exists() else None
    o = o or json.load(open(pathlib.Path(__file__).resolve().parents[2] / "config/origin.json"))
    tr = Transformer.from_crs(4326, 26910, always_xy=True)
    return shapely.ops.transform(lambda x, y, z=None: (np.asarray(tr.transform(x, y)[0]) - o["x0"], np.asarray(tr.transform(x, y)[1]) - o["y0"]), geom)


def cluster_polygon(name):
    """local-coordinate shapely polygon of a cluster: 'niles' = scope town polygon (Niles Blvd + 230 m),
    'sunol' = 140 m around the NCR east end (Main St / Pleasanton-Sunol Rd / Paloma Rd all-way stop)."""
    import shapely.geometry as sg
    if name == "niles":
        sc = json.load(open(DEV7 / "scope.geojson"))
        return _to_local(sg.shape([f for f in sc["features"] if f["properties"]["area"] == "town"][0]["geometry"]))
    if name == "sunol":
        c = load_centerline(None, DEV7 / "centerline_niles_canyon_road.csv")
        return sg.Point(c["x_local"][-1], c["y_local"][-1]).buffer(140.0, 32)
    raise KeyError(name)


CLUSTERS = ("niles", "sunol")
