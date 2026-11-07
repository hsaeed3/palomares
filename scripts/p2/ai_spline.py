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
left / right asphalt edge, camber/grade in radians, direction = turn side +-1, extra length ~7.7 (Kunos value;
meaning unknown, NOT cumulative), points every 1.0 m lifted 6 cm above the asphalt. See docs/AC_CONVENTIONS.md. lapTime/sampleCount 0 (= not a recording).
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
    assert o == len(b) or grid != 0, (o, len(b))   # a grid (Kunos files) follows when hasGrid != 0
    return dict(version=ver, n=n, lap_time=lap, sample_count=samp, P=rec["p"].copy(), length=rec["l"].copy(),
                id=rec["id"].copy(), extra={k: ex[:, i].copy() for i, k in enumerate(EXTRA)}, has_grid=grid)


LIFT_M = 0.06        # line drawn this far above the asphalt: clears the +4 mm surface noise, the +12 mm paint and the
                     # +10 mm patch decals, which otherwise hid most of it (only scattered arrows poked through)
STEP_M = 1.0         # uniform point spacing (Kunos files: ~1-1.5 m; the ideal line is drawn point to point)
KUNOS_EXTRA_LENGTH = 7.7   # extra[12] in Kunos files is ~7-9 m everywhere (sepang), NOT the cumulative length


def layout_spline(D, road, i_from, i_to, C, lift=LIFT_M, step=STEP_M):
    """points (run order, uniform `step` m) + extras for one layout, standing start at i_from"""
    import racing_line as RL
    from kn5 import ac_from_local
    from scipy.ndimage import gaussian_filter1d as gf
    v, ph, idx = RL.speed_profile(D, road, i_from, i_to, C)
    sd = 1 if i_to >= i_from else -1
    ch = C["car_half_width_m"]
    # per-station source arrays (run order). Light plan smoothing (sigma 1 station, < 3 cm even in the hairpins)
    # removes the ~1-2 cm station-to-station jitter of road.xy() that made the curvature flip sign ~575x per km.
    xs = gf(D["x"][idx], 1.0, mode="nearest"); ys = gf(D["y"][idx], 1.0, mode="nearest"); zs = D["z"][idx]
    src = dict(v=v, ph=ph.astype(float), k=np.abs(D["k"][idx]), slope=D["slope"][idx] * sd,
               gl=(D["gap_l"] if sd > 0 else D["gap_r"])[idx] + ch,       # driver's left = station-left for nb
               gr=(D["gap_r"] if sd > 0 else D["gap_l"])[idx] + ch, ks=np.sign(D["k"][idx]) * sd)
    seg = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2); L0 = np.r_[0, np.cumsum(seg)]
    n = int(np.floor(L0[-1] / step)) + 1
    Lu = np.arange(n) * step
    x, y, z = (np.interp(Lu, L0, a) for a in (xs, ys, zs))
    z = z + lift
    S = {k_: np.interp(Lu, L0, a_) for k_, a_ in src.items()}
    v = S["v"]; ph = np.rint(S["ph"])
    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    L = np.r_[0, np.cumsum(seg)]
    fx, fy, fz = np.gradient(x), np.gradient(y), np.gradient(z)
    fn = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2); fx, fy, fz = fx / fn, fy / fn, fz / fn
    slope_run = S["slope"]                              # dz per m toward the driver's left
    camber = np.arctan(slope_run)
    lx, ly = -fy, fx
    ln = np.hypot(lx, ly); lx, ly = lx / ln, ly / ln
    Lv = np.stack([lx, ly, slope_run], -1); Fv = np.stack([fx, fy, fz], -1)
    Nv = np.cross(Fv, Lv); Nv /= np.linalg.norm(Nv, axis=1, keepdims=True)
    Nv[Nv[:, 2] < 0] *= -1
    radius = np.minimum(1 / np.maximum(S["k"], 1e-3), 1000)
    dv = np.gradient(v) / step
    gas = np.where(ph == 1, 1.0, np.where(ph == 0, 0.5, 0.0))
    brake = np.where(ph == -1, np.clip(-dv * v / (C["g"] * C["brake_g"]), 0, 1), 0.0)
    P = ac_from_local(np.stack([x, y, z], -1))
    F = ac_from_local(Fv); Nn = ac_from_local(Nv)
    # extra[8] "direction" is the turn side, +-1 (Kunos files; AssettoServer FastLaneParser: signed camber =
    # camber * direction). Kunos sign: -1 where the AC-plan heading atan2(z, x) increases. Run-order curvature
    # ks > 0 = turning to the driver's left (local plan) = AC heading atan2(z, x) decreasing (AC z = -north) -> +1.
    turn = gf(S["ks"] * radius ** -1, 5.0, mode="nearest")
    direction = np.where(turn >= 0, 1.0, -1.0)
    extra = dict(speed=v, gas=gas, brake=brake, radius=radius, side_l=S["gl"], side_r=S["gr"], camber=camber,
                 direction=direction, nx=Nn[:, 0], ny=Nn[:, 1], nz=Nn[:, 2], length=np.full(n, KUNOS_EXTRA_LENGTH),
                 fx=F[:, 0], fy=F[:, 1], fz=F[:, 2], grade=np.arctan(np.gradient(z) / step))
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
        # (kept >= 6 m inside the road ends: the end-of-world 1WALL_start / 1WALL_end caps sit at s=0 / s=end)
        a = int(np.clip(i0 - sd * 30, 6, len(road.s) - 7)); b = int(np.clip(i1 + sd * 30, 6, len(road.s) - 7))
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
