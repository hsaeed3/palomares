"""Phase 2: race kerbs + painted run-off for the track_* layouts (called from 21_build_geometry.py).

Kerb types (config/kerbs.yaml):
  apex : inside of the corner, entry -> just past the peak. Narrow, rounded-ridged rumble profile.
  exit : outside of the corner after the peak. Full shoulder width, low flat-topped profile, plus a
         painted run-off strip on the terrain beyond it where the verge is flat (never at drops / walls /
         guardrails).
Fit rules:
  * the kerb's inner column IS the road-edge vertex (RX/RY/RZ column 0 / 12): zero step, follows camber;
  * the base continues the road plane outward (or the shoulder where that is higher) + lift_m, and the
    outer column sinks 2 mm below the shoulder, so the kerb never floats and never z-fights the shoulder;
  * height AND width taper in with a smoothstep over taper_m at both ends;
  * built on the 1 m road rows, so it follows the corner exactly like the shoulder does.
Meshes are merged per (direction, 250 m chunk): 1KERB_<DIR>_<cc> and 1RUNOFF_<DIR>_<cc>.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, label
from scipy.spatial import cKDTree
from matplotlib.tri import Triangulation, LinearTriInterpolator


def smoothstep(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def runs(mask):
    lab, n = label(mask)
    return [np.nonzero(lab == l)[0] for l in range(1, n + 1)]


def close_gaps(lab_row, val, gap):
    """fill gaps shorter than gap rows between runs of the same value"""
    idx = np.nonzero(lab_row == val)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap and np.all(lab_row[a + 1:b] == 0):
            lab_row[a + 1:b] = val


class TerrainZ:
    """exact z of the built terrain mesh (P3/TT) near the given query points"""
    def __init__(self, P3, TT, q, radius=4.0):
        if not len(q):
            self.f = None; return
        near = cKDTree(q).query(P3[:, :2], distance_upper_bound=radius)[0] < np.inf
        sel = TT[near[TT].any(1)]
        used, inv = np.unique(sel, return_inverse=True)
        P = P3[used]
        self.f = LinearTriInterpolator(Triangulation(P[:, 0], P[:, 1], inv.reshape(-1, 3)), P[:, 2])

    def __call__(self, x, y):
        if self.f is None:
            return np.full(np.shape(x), np.nan)
        return np.ma.filled(self.f(x, y).astype(float), np.nan)


def build(road, K, RX, RY, RZ, SH, P3, TT, guardrails, add, chunk_of, shoulder_w):
    """SH = {"left": (SLX, SLY, SLZ), "right": (SRX, SRY, SRZ)} (columns edge -> mid -> out)."""
    N = len(road.s)
    ks = gaussian_filter1d(road.k, K["sigma_m"])
    T = K["types"]
    lab, nl = label(np.abs(ks) > K["curvature_threshold_1pm"])
    corners = []
    for l in range(1, nl + 1):
        idx = np.nonzero(lab == l)[0]
        if len(idx) >= K["min_corner_length_m"]:
            apex = idx[np.argmax(np.abs(ks[idx]))]
            corners.append((apex, "left" if ks[apex] > 0 else "right"))
    skip = np.zeros(N, bool)
    for lo, hi in K["skip_s_ranges"]:
        skip |= (road.s >= lo) & (road.s <= hi)
    gr_mask = {"left": np.zeros(N, bool), "right": np.zeros(N, bool)}
    for g in guardrails:
        gr_mask[g["side"]][(road.s >= g["s0"] - 3) & (road.s <= g["s1"] + 3)] = True
    env = {"left": road.env_l, "right": road.env_r}

    KERBS, acc = [], {}

    def push(key, name, mat, P, Tr, UV):
        a = acc.setdefault(key, dict(name=name, mat=mat, P=[], T=[], UV=[], n=0))
        a["P"].append(P); a["T"].append(Tr + a["n"]); a["UV"].append(UV); a["n"] += len(P)

    for dname, sd in (("nb", 1), ("sb", -1)):
        rowlab = {"left": np.zeros(N, np.int8), "right": np.zeros(N, np.int8)}   # 1 apex, 2 exit
        for apex, inner in corners:
            outer = "right" if inner == "left" else "left"
            sa = road.s[apex]
            if sd > 0:
                a_rng = (sa - K["apex_entry_m"], sa + K["inside_exit_m"])
                e_rng = (sa + K["outside_exit_start_m"], sa + K["outside_exit_start_m"] + K["outside_exit_length_m"])
            else:
                a_rng = (sa - K["inside_exit_m"], sa + K["apex_entry_m"])
                e_rng = (sa - K["outside_exit_start_m"] - K["outside_exit_length_m"], sa - K["outside_exit_start_m"])
            m = (road.s >= e_rng[0]) & (road.s <= e_rng[1]) & (rowlab[outer] == 0)
            rowlab[outer][m] = 2
            rowlab[inner][(road.s >= a_rng[0]) & (road.s <= a_rng[1])] = 1      # apex wins over an exit
        for side in ("left", "right"):
            rl = rowlab[side]; rl[skip] = 0; rl[:1] = 0; rl[-2:] = 0
            for v in (1, 2):
                close_gaps(rl, v, K["merge_gap_m"])
            sgn = 1 if side == "left" else -1
            col = -1 if side == "left" else 0
            SX, SY, SZ = SH[side]
            e_all = road.el if side == "left" else road.er
            # outward slope of the road plane at the edge (surface_z: z = z0 + cl*d (d>0), cr*d (d<0))
            slope_all = road.cl if side == "left" else -road.cr
            for v, typ in ((1, "apex"), (2, "exit")):
                P_ = T[typ]
                for i in runs(rl == v):
                    if len(i) < K["min_kerb_length_m"]:
                        continue
                    n = len(i); k = np.arange(n)
                    tap = smoothstep(np.minimum(k, n - 1 - k) / max(K["taper_m"], 1e-3))
                    w = P_["width_m"] * (K["taper_min_width_frac"] + (1 - K["taper_min_width_frac"]) * tap)
                    t = np.array(P_["profile_t"], float); hp = np.array(P_["profile_h"], float)
                    e = e_all[i]
                    if isinstance(shoulder_w, dict):   # kerb fits on the real shoulder (>= 0.35 m; it covers the edge strip)
                        w = np.minimum(w, np.maximum(shoulder_w[side][i], 0.35))
                    dd = e[:, None] + w[:, None] * t[None, :]                 # outward distance from centre
                    rr = np.repeat(i[:, None], len(t), 1)
                    x, y = road.xy(rr, sgn * dd)
                    # shoulder surface under each column (piecewise linear edge -> mid -> out)
                    # shoulder_w: scalar (legacy) or {"left": sh_w_l, "right": sh_w_r} per-station arrays (edge_profile.py);
                    # where the kerb is wider than the shoulder it continues on the shoulder's outer edge height (u clipped at 2)
                    sw_ = shoulder_w[side][i] if isinstance(shoulder_w, dict) else np.full(n, float(shoulder_w))
                    u = np.clip((dd - e[:, None]) / (sw_[:, None] / 2), 0, 2)
                    z0, z1, z2 = SZ[rr, 0], SZ[rr, 1], SZ[rr, 2]
                    zsh = np.where(u <= 1, z0 + (z1 - z0) * u, z1 + (z2 - z1) * (u - 1))
                    zpl = RZ[i, col][:, None] + slope_all[i][:, None] * (dd - e[:, None])
                    fade = 1 - smoothstep((t - 0.75) / 0.25)[None, :]
                    base = zsh + np.clip(zpl - zsh, 0, K["max_above_shoulder_m"]) * fade
                    lift = np.where(t > 0, K["lift_m"], 0.0)[None, :] * np.ones((n, 1))
                    z = base + lift + P_["height_m"] * hp[None, :] * tap[:, None]
                    z[:, -1] = zsh[:, -1] - 0.002                             # outer edge sinks into the shoulder
                    x[:, 0], y[:, 0], z[:, 0] = RX[i, col], RY[i, col], RZ[i, col]   # == road-edge vertex
                    uu = P_["u0"] + (P_["u1"] - P_["u0"]) * t
                    vv = road.s[i] / (2 * P_["stripe_length_m"])
                    if side == "right":                                        # columns must increase to the left
                        x, y, z, uu = x[:, ::-1], y[:, ::-1], z[:, ::-1], uu[::-1]
                    Pm = np.stack([x, y, z], -1).reshape(-1, 3)
                    UV = np.stack([np.broadcast_to(uu, (n, len(t))), np.repeat(vv[:, None], len(t), 1)], -1).reshape(-1, 2)
                    from geom import grid_tris
                    c = chunk_of(i[n // 2])
                    push((dname, "k", c), f"1KERB_{dname.upper()}_{c:02d}", "kerb", Pm, grid_tris(n, len(t)), UV)
                    KERBS.append(dict(s0=float(road.s[i[0]]), s1=float(road.s[i[-1]]), side=side, dir=dname, type=typ))
                    if typ == "exit" and K["runoff"]["enabled"]:
                        build_runoff(road, K["runoff"], i, side, sgn, SX, SY, SZ, env[side], gr_mask[side], dname, P3, TT, push, chunk_of)
    for (dname, kind, c), a in sorted(acc.items()):
        add(f"kerbs_{dname}", a["name"], a["mat"], np.concatenate(a["P"]), np.concatenate(a["T"]), np.concatenate(a["UV"]),
            cast_shadows=False)
    return KERBS, corners


def build_runoff(road, R, i, side, sgn, SX, SY, SZ, env, grm, dname, P3, TT, push, chunk_of):
    """painted strip on the terrain beyond the exit kerb; inner column == shoulder/terrain seam vertex"""
    from geom import grid_tris
    ok = np.isin(env[i], R["allowed_env"]) & ~grm[i]
    nc = R["columns"]
    t = np.linspace(0, 1, nc)
    for r in runs(ok):
        ii = i[r]
        if len(ii) < R["min_length_m"]:
            continue
        n = len(ii); k = np.arange(n)
        tap = smoothstep(np.minimum(k, n - 1 - k) / R["taper_m"])
        w = R["width_m"] * (0.15 + 0.85 * tap)
        ox, oy, oz = SX[ii, 2], SY[ii, 2], SZ[ii, 2]
        nx, ny = sgn * road.nx[ii], sgn * road.ny[ii]
        x = ox[:, None] + nx[:, None] * w[:, None] * t[None, :]
        y = oy[:, None] + ny[:, None] * w[:, None] * t[None, :]
        tz = TerrainZ(P3, TT, np.c_[x.ravel(), y.ravel()])
        zt = tz(x.ravel(), y.ravel()).reshape(n, nc)
        zt[:, 0] = oz
        if np.isnan(zt).any() or np.abs(np.diff(zt, axis=1)).max() > R["max_step_across_m"] or \
                np.abs(zt[:, -1] - zt[:, 0]).max() > R["max_rise_m"]:
            # too rough / not flat: keep only rows that pass
            bad = np.isnan(zt).any(1) | (np.abs(np.diff(zt, axis=1)).max(1) > R["max_step_across_m"]) | \
                  (np.abs(zt[:, -1] - zt[:, 0]) > R["max_rise_m"])
            if bad.mean() > 0.3:
                continue
            zt = np.where(np.isnan(zt), oz[:, None], zt)
        lift = np.interp(t, [0, 0.25, 1], [R["lift_inner_m"], R["lift_m"], R["lift_m"]])
        z = zt + lift[None, :]
        uu = t * w[:, None] / R["tex_m"]
        vv = np.repeat(road.s[ii, None], nc, 1) / R["tex_m"]
        if side == "right":
            x, y, z, uu = x[:, ::-1], y[:, ::-1], z[:, ::-1], uu[:, ::-1]
        P = np.stack([x, y, z], -1).reshape(-1, 3)
        UV = np.stack([uu, vv], -1).reshape(-1, 2)
        c = chunk_of(ii[n // 2])
        push((dname, "r", c), f"1RUNOFF_{dname.upper()}_{c:02d}", "runoff", P, grid_tris(n, nc), UV)
