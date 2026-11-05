"""Assetto Corsa AI spline (fast_lane.ai, version 7) writer + reader, and the per-layout export.

Binary layout (little endian), as read by gro-ove/actools AcTools/AiFile/AiSpline.cs, AiPoint.cs,
AiPointExtra.cs, AiSplineGrid.cs:
  int32 version (=7), int32 pointCount, int32 lapTime (0), int32 sampleCount (0)
  pointCount x { float32 x, y, z (AC world: x, up, -north) ; float32 length (cumulative m) ; int32 id }
  int32 extraCount (= pointCount)
  extraCount x 18 float32 { speed, gas, brake, obsoleteLatG, radius, sideLeft, sideRight, camber, direction,
                            normal.xyz, length, forward.xyz, tag, grade }
  int32 hasGrid (0: AC builds its own lookup grid)
Units used here: speed m/s, radius m (unsigned, capped 1000), sideLeft/Right = m from the line to the driver's
left / right asphalt edge, camber/grade/direction in radians. lapTime/sampleCount 0 (= not a recording).
"""
import struct
import numpy as np

EXTRA = ("speed", "gas", "brake", "latg", "radius", "side_l", "side_r", "camber", "direction",
         "nx", "ny", "nz", "length", "fx", "fy", "fz", "tag", "grade")


def write(path, P, length, extra):
    """P: (n,3) AC world coords; length: (n,) cumulative; extra: dict of EXTRA arrays (missing -> 0)"""
    n = len(P)
    ex = np.zeros((n, 18), np.float32)
    for k, name in enumerate(EXTRA):
        if name in extra:
            ex[:, k] = extra[name]
    with open(path, "wb") as f:
        f.write(struct.pack("<4i", 7, n, 0, 0))
        rec = np.zeros(n, dtype=[("p", "<f4", 3), ("l", "<f4"), ("id", "<i4")])
        rec["p"] = P; rec["l"] = length; rec["id"] = np.arange(n)
        f.write(rec.tobytes())
        f.write(struct.pack("<i", n))
        f.write(ex.astype("<f4").tobytes())
        f.write(struct.pack("<i", 0))


def read(path):
    b = open(path, "rb").read()
    ver, n, lap, samp = struct.unpack_from("<4i", b, 0); o = 16
    assert ver == 7, ver
    rec = np.frombuffer(b, dtype=[("p", "<f4", 3), ("l", "<f4"), ("id", "<i4")], count=n, offset=o); o += 20 * n
    (ne,) = struct.unpack_from("<i", b, o); o += 4
    ex = np.frombuffer(b, "<f4", count=18 * ne, offset=o).reshape(ne, 18); o += 72 * ne
    (grid,) = struct.unpack_from("<i", b, o); o += 4
    assert o == len(b), (o, len(b))
    return dict(version=ver, n=n, lap_time=lap, sample_count=samp, P=rec["p"].copy(), length=rec["l"].copy(),
                id=rec["id"].copy(), extra={k: ex[:, i].copy() for i, k in enumerate(EXTRA)}, has_grid=grid)


def layout_spline(D, road, i_from, i_to, C, lift=0.0):
    """points (run order, 1 m) + extras for one layout, standing start at i_from"""
    import racing_line as RL
    from kn5 import ac_from_local
    v, ph, idx = RL.speed_profile(D, road, i_from, i_to, C)
    sd = 1 if i_to >= i_from else -1
    x, y, z = D["x"][idx], D["y"][idx], D["z"][idx] + lift
    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    L = np.r_[0, np.cumsum(seg)]
    fx, fy, fz = np.gradient(x), np.gradient(y), np.gradient(z)
    fn = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2); fx, fy, fz = fx / fn, fy / fn, fz / fn
    # driver's left = station-left for nb, station-right for sb
    ch = C["car_half_width_m"]
    side_l = (D["gap_l"] if sd > 0 else D["gap_r"])[idx] + ch
    side_r = (D["gap_r"] if sd > 0 else D["gap_l"])[idx] + ch
    slope_run = D["slope"][idx] * sd                    # dz per m toward the driver's left
    camber = np.arctan(slope_run)
    lx, ly = -fy, fx                                     # driver's left (plan)
    ln = np.hypot(lx, ly); lx, ly = lx / ln, ly / ln
    # surface normal = forward x left-with-camber
    Lv = np.stack([lx, ly, slope_run], -1); Fv = np.stack([fx, fy, fz], -1)
    Nv = np.cross(Fv, Lv); Nv /= np.linalg.norm(Nv, axis=1, keepdims=True)
    Nv[Nv[:, 2] < 0] *= -1
    k = np.abs(D["k"][idx]); radius = np.minimum(1 / np.maximum(k, 1e-3), 1000)
    dv = np.gradient(v)
    gas = np.where(ph == 1, 1.0, np.where(ph == 0, 0.5, 0.0))
    brake = np.where(ph == -1, np.clip(-dv * v / (C["g"] * C["brake_g"]) , 0, 1), 0.0)
    P = ac_from_local(np.stack([x, y, z], -1))
    F = ac_from_local(Fv); Nn = ac_from_local(Nv)
    direction = np.arctan2(F[:, 0], F[:, 2])
    extra = dict(speed=v, gas=gas, brake=brake, radius=radius, side_l=side_l, side_r=side_r, camber=camber,
                 direction=direction, nx=Nn[:, 0], ny=Nn[:, 1], nz=Nn[:, 2], length=L, fx=F[:, 0], fy=F[:, 1], fz=F[:, 2],
                 grade=np.arctan(np.gradient(z) / np.maximum(seg.mean(), 0.5)))
    return P, L, extra, idx


def export_layouts(track_dir, layouts, road, log=print):
    """layouts: {name: dict(s_from, s_to)}. Writes <track>/<layout>/ai/fast_lane.ai, validates by round trip."""
    import racing_line as RL
    C = RL.cfg(); R = RL.load()
    out = {}
    for lay, L_ in layouts.items():
        i0 = int(np.argmin(np.abs(road.s - L_["s_from"]))); i1 = int(np.argmin(np.abs(road.s - L_["s_to"])))
        sd = 1 if i1 > i0 else -1
        # cover the spawn (before the start gate) and a little past the finish so the AI never runs off the spline
        a = int(np.clip(i0 - sd * 30, 0, len(road.s) - 1)); b = int(np.clip(i1 + sd * 30, 0, len(road.s) - 1))
        D = R["nb" if sd > 0 else "sb"]
        P, Ln, ex, idx = layout_spline(D, road, a, b, C)
        p = track_dir / lay / "ai"; p.mkdir(parents=True, exist_ok=True)
        f = p / "fast_lane.ai"; write(f, P, Ln, ex)
        r = read(f)
        assert r["n"] == len(P) and np.allclose(r["P"], P, atol=1e-3) and np.allclose(r["extra"]["speed"], ex["speed"], atol=1e-4)
        assert np.all(np.diff(r["length"]) > 0) and np.all(r["extra"]["side_l"] > 0) and np.all(r["extra"]["side_r"] > 0)
        out[lay] = dict(points=int(r["n"]), length_m=float(r["length"][-1]), vmax_kmh=float(ex["speed"].max() * 3.6),
                        est_time_s=float(np.sum(np.diff(Ln) / np.maximum(ex["speed"][:-1], 1.0))), bytes=f.stat().st_size)
        log(f"  ai {lay}: {r['n']} pts, {r['length'][-1]:.0f} m, vmax {ex['speed'].max() * 3.6:.0f} km/h, "
            f"est. {out[lay]['est_time_s']:.0f} s (round trip ok)")
    return out
