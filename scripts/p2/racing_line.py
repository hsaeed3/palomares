"""Phase 2: optimal racing line + GG speed profile per run direction (config/racing_line.yaml).

  line   : car-centre lateral offset a(s) (+ = left looking north / s increasing) on the 1 m stations,
           bounded to the asphalt (road.el / road.er) minus car half-width + edge margin.
           Minimum-curvature QP (sparse bounded least squares), iterated with the speed profile: where the
           car is traction/power-limited (exits) curvature is weighted up, so the apex moves late and the
           exit straightens -> the line differs between nb and sb.
  speed  : point-mass GG model on the 3D line: lateral lat_g (+ camber bank, crest unloading), friction-
           ellipse braking brake_g and accel_g, power-limited acceleration, grade and drag. Forward/backward
           passes give phase per station: 1 accelerate, 0 cornering-limited, -1 brake.
  zones  : braking zones (onset, turn-in, apex, entry/min speed) ranked by kinetic energy scrubbed.

compute(road) -> {"nb": D, "sb": D} with arrays in STATION order (index = road station index) and is cached in
build/racing_line.pkl (keyed by a hash of the road edges/centerline and the config).
speed_profile(D, road, i_from, i_to) -> v, phase, idx  for one layout (standing start at i_from).
"""
import hashlib, pickle, pathlib
import numpy as np, yaml
from scipy.ndimage import gaussian_filter1d, label, binary_closing
from scipy import sparse

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "config/racing_line.yaml"
CACHE = ROOT / "build/racing_line.pkl"


def cfg():
    return yaml.safe_load(open(CFG_PATH))


def bounds(road, C):
    m = C["car_half_width_m"] + C["edge_margin_m"]
    hi = road.el - m; lo = -(road.er - m)
    narrow = hi - lo < C["min_centre_room_m"]
    mid = (road.el - road.er) / 2
    hi = np.where(narrow, mid + C["min_centre_room_m"] / 2, hi)
    lo = np.where(narrow, mid - C["min_centre_room_m"] / 2, lo)
    return lo, hi


def optimise(road, C, w=None):
    """minimum (weighted) curvature lateral offset on a step_m grid, interpolated to every station"""
    N = len(road.s); h = C["step_m"]
    g = np.unique(np.r_[np.arange(0, N, int(h)), N - 1]); M = len(g)
    cx, cy, nx, ny = road.x[g], road.y[g], road.nx[g], road.ny[g]
    lo, hi = bounds(road, C); lo, hi = lo[g], hi[g]
    ww = np.ones(M) if w is None else np.interp(road.s[g], road.s, w)
    r = np.arange(1, M - 1)
    sw = np.sqrt(ww[r]) / h ** 2
    rows, cols, vals = [], [], []
    for comp, (nn, cc) in enumerate(((nx, cx), (ny, cy))):
        base = comp * len(r)
        for k, coef in ((-1, 1.0), (0, -2.0), (1, 1.0)):
            rows.append(base + np.arange(len(r))); cols.append(r + k); vals.append(sw * coef * nn[r + k])
    b = np.r_[-(cx[:-2] - 2 * cx[1:-1] + cx[2:]) * sw, -(cy[:-2] - 2 * cy[1:-1] + cy[2:]) * sw]
    base = 2 * len(r); q = np.arange(M - 1); sm = np.sqrt(C["smooth_weight"]) / h
    rows += [base + q, base + q]; cols += [q, q + 1]; vals += [np.full(M - 1, -sm), np.full(M - 1, sm)]
    b = np.r_[b, np.zeros(M - 1)]
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(base + M - 1, M))
    a_g = box_qp(A, b, lo, hi)
    a = np.interp(road.s, road.s[g], a_g)
    lo1, hi1 = bounds(road, C)
    return np.clip(a, lo1, hi1)


def box_qp(A, b, lo, hi, iters=200):
    """min |A a - b|^2 s.t. lo <= a <= hi. Primal active set on the banded normal equations (exact sparse
    solves): clamp violated variables, release clamped ones whose gradient points inward, repeat."""
    from scipy.sparse.linalg import spsolve
    Q = (A.T @ A).tocsr(); q = A.T @ b; M = Q.shape[0]
    a = np.clip(spsolve(Q.tocsc(), q), lo, hi)
    atl = a <= lo; ath = a >= hi
    for it in range(iters):
        fr = ~(atl | ath)
        a[atl] = lo[atl]; a[ath] = hi[ath]
        if fr.any():
            Qff = Q[fr][:, fr].tocsc()
            rhs = q[fr] - Q[fr][:, ~fr] @ a[~fr]
            a[fr] = spsolve(Qff, rhs)
        vl = fr & (a < lo - 1e-9); vh = fr & (a > hi + 1e-9)
        if vl.any() or vh.any():
            atl |= vl; ath |= vh; continue
        gr = Q @ a - q
        rl = atl & (gr < -1e-10); rh = ath & (gr > 1e-10)
        if not (rl.any() or rh.any()):
            break
        atl &= ~rl; ath &= ~rh
    return np.clip(a, lo, hi)


def geometry(road, a):
    """3D line, signed plan curvature (+ left), camber at the line, vertical curvature"""
    x, y = road.xy(np.arange(len(road.s)), a)
    z = road.surface_z(np.arange(len(road.s)), a, noise=False)
    xs, ys = gaussian_filter1d(x, 1.5, mode="nearest"), gaussian_filter1d(y, 1.5, mode="nearest")
    dx, dy = np.gradient(xs), np.gradient(ys); ddx, ddy = np.gradient(dx), np.gradient(dy)
    k = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    k = gaussian_filter1d(k, 1.0, mode="nearest")
    slope = np.where(a > 0, road.cl, road.cr)                          # dz/dd at the line
    zs = gaussian_filter1d(z, 4, mode="nearest")
    ds = np.hypot(np.gradient(x), np.gradient(y))
    kz = np.gradient(np.gradient(zs) / ds) / ds
    return x, y, z, k, slope, kz


def speed_run(C, k, ds, grade, bank, kz, v0, v_prev=None):
    """arrays in RUN order. k: |curvature|; ds[i]: length i -> i+1; grade: dz/ds in run direction"""
    gg = C["g"]; n = len(k)
    load = np.ones(n)
    if C["use_vertical_curvature"] and v_prev is not None:
        load = np.clip(1 + v_prev ** 2 * kz / gg, 0.4, 1.6)
    bk = np.clip(bank, -0.1, 0.1) if C["use_camber"] else np.zeros(n)
    aymax = gg * np.maximum(C["lat_g"] * load + bk, 0.3)
    vlat = np.minimum(np.sqrt(aymax / np.maximum(np.abs(k), 1e-6)), C["vmax_ms"])
    sgrade = grade / np.sqrt(1 + grade ** 2)
    # backward (braking) pass
    vb = vlat.copy()
    for i in range(n - 2, -1, -1):
        v1 = vb[i + 1]
        ell = np.sqrt(max(0.0, 1 - (v1 * v1 * abs(k[i + 1]) / aymax[i + 1]) ** 2))
        ab = gg * C["brake_g"] * load[i] * ell + gg * sgrade[i] + C["drag_coeff_per_m"] * v1 * v1
        vb[i] = min(vb[i], np.sqrt(v1 * v1 + 2 * ds[i] * max(ab, 0.5)))
    # forward (accel) pass
    v = vb.copy(); v[0] = min(v0, vb[0]) if v0 is not None else vb[0]
    ph = np.zeros(n, np.int8)
    for i in range(n - 1):
        v1 = max(v[i], 0.5)
        ell = np.sqrt(max(0.0, 1 - (v1 * v1 * abs(k[i]) / aymax[i]) ** 2))
        aa = min(gg * C["accel_g"] * load[i] * ell, C["power_to_mass_w_per_kg"] / v1) - gg * sgrade[i] - C["drag_coeff_per_m"] * v1 * v1
        vf = np.sqrt(max(v[i] * v[i] + 2 * ds[i] * aa, 0.0))
        if vf < vb[i + 1] - 1e-3:
            v[i + 1] = vf; ph[i + 1] = 1
        else:
            v[i + 1] = vb[i + 1]
            ph[i + 1] = -1 if vb[i + 1] < vlat[i + 1] - 1e-3 else 0
    if v0 is not None and v[0] < vb[0] - 1e-3:
        ph[0] = 1
    return v, ph, vlat


def _run(D, road, i_from, i_to, C, v0):
    sd = 1 if i_to >= i_from else -1
    idx = np.arange(i_from, i_to + sd, sd)
    x, y, z = D["x"][idx], D["y"][idx], D["z"][idx]
    ds = np.r_[np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2), 1.0]
    grade = np.gradient(gaussian_filter1d(z, 3, mode="nearest")) / np.maximum(ds, 0.2)
    k = D["k"][idx]
    bank = -np.sign(k) * D["slope"][idx]
    # slope = dz/dd (d to the station-left). Station-order k>0 = curving toward +d; favourable bank = surface
    # falling toward the centre of curvature. Both signs flip together for sb, so the formula is direction-free.
    v = None
    for _ in range(2):
        v, ph, vlat = speed_run(C, k, ds, grade, bank, D["kz"][idx], v0, v)
    return v, ph, vlat, idx, ds


def speed_profile(D, road, i_from, i_to, C=None, v0="cfg"):
    C = C or cfg()
    v, ph, vlat, idx, ds = _run(D, road, i_from, i_to, C, C["start_speed_ms"] if v0 == "cfg" else v0)
    return v, ph, idx


def braking_zones(D, road, C):
    """zones on the full-direction profile (flying, run order): dict(onset, turn_in, apex, v_entry, v_apex, side)"""
    v = D["v_run"]; ph = D["ph_run"]; idx = D["idx_run"]; k = D["k"][idx] * D["sd"]   # + = run-left turn
    br = binary_closing(ph == -1, np.ones(9, bool))
    lab, n = label(br); Z = []
    for l in range(1, n + 1):
        r = np.nonzero(lab == l)[0]; b0, b1 = r[0], r[-1]
        e = b1
        while e + 1 < len(v) and ph[e + 1] != 1:                   # cornering-limited part after braking
            e += 1
        ap = b0 + int(np.argmin(v[b0:e + 1]))
        drop = (v[b0] - v[ap]) * 3.6
        if drop < C["brake_min_drop_kmh"]:
            continue
        kap = abs(k[ap]) if abs(k[ap]) > 1e-4 else np.abs(k[b0:e + 1]).max()
        ti = b0
        for j in range(b0, ap + 1):
            if abs(k[j]) > C["turn_in_frac"] * kap:
                ti = j; break
        side = "left" if k[ap] > 0 else "right"
        Z.append(dict(onset=int(idx[b0]), turn_in=int(idx[ti]), apex=int(idx[ap]), v_entry=float(v[b0]), v_apex=float(v[ap]),
                      energy=float(v[b0] ** 2 - v[ap] ** 2), side=side, s_onset=float(road.s[idx[b0]]),
                      s_turn_in=float(road.s[idx[ti]]), s_apex=float(road.s[idx[ap]])))
    return Z


def _key(road, C):
    h = hashlib.sha1()
    for arr in (road.x, road.y, road.z, road.el, road.er, road.cl, road.cr):
        h.update(np.ascontiguousarray(arr, np.float64).tobytes())
    h.update(repr(sorted(C.items())).encode())
    h.update(pathlib.Path(__file__).read_bytes())
    return h.hexdigest()


def compute(road, force=False):
    C = cfg(); key = _key(road, C)
    if CACHE.exists() and not force:
        try:
            d = pickle.load(open(CACHE, "rb"))
            if d.get("key") == key:
                return d["dirs"]
        except Exception:
            pass
    N = len(road.s); out = {}
    for dname, sd in (("nb", 1), ("sb", -1)):
        w = None; a = None
        for it in range(C["iterations"]):
            a = optimise(road, C, w)
            x, y, z, k, slope, kz = geometry(road, a)
            D = dict(a=a, x=x, y=y, z=z, k=k, slope=slope, kz=kz, sd=sd)
            i0, i1 = (0, N - 1) if sd > 0 else (N - 1, 0)
            v, ph, vlat, idx, ds = _run(D, road, i0, i1, C, None)          # flying profile over the whole road
            vs = np.empty(N); vs[idx] = v; phs = np.empty(N, np.int8); phs[idx] = ph
            w = 1 + C["exit_weight"] * gaussian_filter1d((phs == 1).astype(float) * (np.abs(k) > 0.004), 4)
        D.update(v=vs, phase=phs, v_run=v, ph_run=ph, idx_run=idx)
        D["gap_l"] = road.el - a - C["car_half_width_m"]                    # tyre -> asphalt edge, station-left
        D["gap_r"] = road.er + a - C["car_half_width_m"]
        D["zones"] = braking_zones(D, road, C)
        out[dname] = D
    CACHE.parent.mkdir(exist_ok=True)
    pickle.dump(dict(key=key, dirs=out), open(CACHE, "wb"))
    return out


def load():
    return pickle.load(open(CACHE, "rb"))["dirs"]


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from geom import Road, CFG
    t = time.time(); road = Road(CFG["build_end_m"])
    R = compute(road, force="--force" in sys.argv)
    for d, D in R.items():
        print(d, "time %.0fs" % (time.time() - t), "v max %.0f km/h  min %.0f km/h" % (D["v"].max() * 3.6, D["v"].min() * 3.6),
              "run time %.1f s" % np.sum(1 / np.maximum(D["v_run"], 1)), "zones", len(D["zones"]),
              "min gap L %.2f R %.2f" % (D["gap_l"].min(), D["gap_r"].min()))
