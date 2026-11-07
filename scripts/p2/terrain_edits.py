"""Near-road terrain: DTM toe distances for the shoulder + per-span overrides (config/terrain_overrides.yaml).

toe(road) -> {"toe_l", "toe_r", "rise_l", "rise_r"}: per station, how far past the asphalt edge the LiDAR ground stays at
  road level (first offset where it rises > TOE_UP or falls < -TOE_DN below the edge, with overrides applied), and the
  signed height at 1.5 m past the edge. edge_profile.py clamps the shoulder width to the toe, so a cut bank that rises
  at the pavement edge starts there in the build (dev5 GoPro frames) instead of behind a flat 1-2 m shoulder.
  Cached in data/derived/edge_toe.npz (key: DTM mtimes, centreline, overrides file), because the 0.5 m DTM is ~1.4 GB.

apply(road, x, y, z) -> z: blends terrain heights toward the pavement-edge height in `flatten` spans: places where
  the ground changed after the LiDAR flight (e.g. frame 3's graded, fenced gravel yard at s 112-172 R).
  Row: {s0, s1, side: L|R, off0, off1, taper, dz, frames, note}. off = metres beyond the asphalt edge; full effect
  from off0 to off1 - taper, linear fade to 0 at off1 and over `taper` m past s0 / s1. dz = height relative to the
  asphalt edge (negative = a little below the road).
"""
import hashlib, json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DER = ROOT / "data/derived"
CFG_F = ROOT / "config/terrain_overrides.yaml"
OFFS = np.arange(0.0, 4.01, 0.25)
TOE_UP, TOE_DN = 0.22, 0.45


def overrides():
    import yaml
    d = (yaml.safe_load(open(CFG_F)) or {}) if CFG_F.exists() else {}
    return d.get("flatten") or []


def _weights(road, x, y):
    """per point: list of (weight, target z) for every flatten row that touches it"""
    rows = overrides()
    if not rows:
        return []
    from scipy.spatial import cKDTree
    kd = getattr(road, "_kd_xy", None)
    if kd is None:
        kd = road._kd_xy = cKDTree(np.c_[road.x, road.y])
    _, j = kd.query(np.c_[x, y])
    d = (x - road.x[j]) * road.nx[j] + (y - road.y[j]) * road.ny[j]
    s = road.s[j]
    out = []
    for r in rows:
        sgn = 1 if str(r["side"]).upper().startswith("L") else -1
        e = road.el[j] if sgn > 0 else road.er[j]
        off = sgn * d - e
        tp = max(float(r.get("taper", 4.0)), 0.5)
        o0, o1 = float(r.get("off0", 0.0)), float(r["off1"])
        wo = np.clip((o1 - off) / tp, 0, 1) * (off >= o0 - 0.5)
        ws = np.clip(1 - np.maximum(float(r["s0"]) - s, s - float(r["s1"])) / tp, 0, 1)
        w = wo * ws
        if not w.any():
            continue
        zt = road.surface_z(j, sgn * e, noise=False) + float(r.get("dz", 0.0))
        out.append((w, zt))
    return out


def apply(road, x, y, z):
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.array(z, float)
    for w, zt in _weights(road, x, y):
        z = np.where(np.isfinite(z), z + w * (zt - z), zt)
    return z


def _key(road):
    h = hashlib.sha1()
    for f in (DER / "dtm_r050_b40.tif", DER / "dtm_r100_b150.tif"):
        h.update(str(f.stat().st_mtime_ns).encode())
    h.update(np.round(np.c_[road.x, road.y, road.z, road.el, road.er], 3).tobytes())
    h.update(CFG_F.read_bytes() if CFG_F.exists() else b"")
    h.update(f"{OFFS.tobytes()}{TOE_UP}{TOE_DN}v1".encode())
    return h.hexdigest()


def toe(road):
    f = DER / "edge_toe.npz"; k = _key(road)
    if f.exists():
        d = np.load(f)
        if str(d["key"]) == k and len(d["toe_l"]) == len(road.s):
            return {n: d[n] for n in ("toe_l", "toe_r", "rise_l", "rise_r")}
    from geom import Raster
    r05, r1 = Raster(DER / "dtm_r050_b40.tif"), Raster(DER / "dtm_r100_b150.tif")
    N = len(road.s); I = np.repeat(np.arange(N), len(OFFS)); O = np.tile(OFFS, N)
    out = {"key": k}
    for sgn, nm in ((1, "l"), (-1, "r")):
        e = road.el if sgn > 0 else road.er
        x, y = road.xy(I, sgn * (e[I] + O))
        z = r05.sample(x, y); z = np.where(np.isfinite(z), z, r1.sample(x, y))
        z = apply(road, x, y, z)
        ze = road.surface_z(I, sgn * e[I], noise=False)
        rel = (z - ze).reshape(N, len(OFFS))
        rel = np.where(np.isfinite(rel), rel, 0.0)
        # height of the ground near the toe relative to road level just past the edge (the first 0.25 m can carry
        # the asphalt's own crossfall / LiDAR mixed pixels: use the lower of the first two columns as reference)
        ref = np.minimum(rel[:, 0], rel[:, 1])[:, None]
        up = (rel - ref) > TOE_UP; dn = (rel - ref) < -TOE_DN
        hit = up | dn; hit[:, 0] = False
        first = np.where(hit.any(1), hit.argmax(1), len(OFFS) - 1)
        out[f"toe_{nm}"] = OFFS[first]
        out[f"rise_{nm}"] = rel[:, int(np.searchsorted(OFFS, 1.5))] - ref[:, 0]
    np.savez(f, **out)
    return {n: out[n] for n in ("toe_l", "toe_r", "rise_l", "rise_r")}
