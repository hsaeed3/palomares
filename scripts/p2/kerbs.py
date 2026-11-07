"""Phase 2: race kerbs + painted run-off for the track_* layouts (called from 21_build_geometry.py).

Placement (dev4) follows the racing line of each run direction (racing_line.py): apex kerbs where the line's
tyre touches the inside edge while turning toward it, exit kerbs where it runs out to the outside edge after
that apex; nothing where the line stays > placement.touch_m from the edge, at driveways/junctions, or on
forbidden verge types; run-off only where the line runs really wide and the ground is flat.
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
    T = K["types"]
    import racing_line as RLm
    RL = RLm.compute(road)                       # optimal line per run direction (config/racing_line.yaml)
    PL = K["placement"]
    skip = np.zeros(N, bool)
    for lo, hi in K["skip_s_ranges"]:
        skip |= (road.s >= lo) & (road.s <= hi)
    gr_mask = {"left": np.zeros(N, bool), "right": np.zeros(N, bool)}
    for g in guardrails:
        gr_mask[g["side"]][(road.s >= g["s0"] - 3) & (road.s <= g["s1"] + 3)] = True
    env = {"left": road.env_l, "right": road.env_r}
    excl = access_mask(road, PL)                 # driveways / junctions / aprons per side

    KERBS, acc, corners = [], {}, []
    LINE = {}

    def push(key, name, mat, P, Tr, UV):
        a = acc.setdefault(key, dict(name=name, mat=mat, P=[], T=[], UV=[], n=0))
        a["P"].append(P); a["T"].append(Tr + a["n"]); a["UV"].append(UV); a["n"] += len(P)

    for dname, sd in (("nb", 1), ("sb", -1)):
        D = RL[dname]
        rowlab, wide, LINE[dname] = line_rows(road, D, sd, PL, env, skip, excl)
        corners += [(int(a_), sd_, dname) for a_, sd_ in LINE[dname]["apexes"]]
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
            # dev6 FFB safety: no kerb where the shoulder itself is not a near-plane continuation of the road (rising
            # cut-bank toe, falling verge, ditch): a kerb draped there folds by 2-6 cm.  Kink = mid column vs the
            # edge-outer chord; grade = shoulder cross-slope minus the road plane's.
            _swa = shoulder_w[side] if isinstance(shoulder_w, dict) else np.full(N, float(shoulder_w))
            _kink = np.abs(SZ[:, 1] - (SZ[:, 0] + SZ[:, 2]) / 2)                    # ditch / hump inside the shoulder
            _grade = np.abs((SZ[:, 2] - SZ[:, 0]) / np.maximum(_swa, 0.1) - slope_all)  # shoulder cross-slope vs road plane
            _bad = (_kink > K.get("max_shoulder_kink_m", 0.02)) | (_grade > K.get("max_shoulder_grade", 0.15))
            rl[_bad] = 0
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
                        w = np.minimum(w, np.maximum(shoulder_w[side][i] - 0.02, 0.10))   # dev4: never overhang the shoulder onto terrain / cut banks (23_qa 3a)
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
                    # dev6: never drape a kerb INTO a shoulder ditch / dip (3-6 cm folds in build 2): the kerb rides the
                    # chord from the road edge to its outer edge wherever the shoulder sags below it
                    _ch = zsh[:, :1] + (zsh[:, -1:] - zsh[:, :1]) * ((dd - dd[:, :1]) / np.maximum(dd[:, -1:] - dd[:, :1], 1e-6))
                    zsh = np.maximum(zsh, _ch)
                    zpl = RZ[i, col][:, None] + slope_all[i][:, None] * (dd - e[:, None])
                    fade = 1 - smoothstep((t - 0.75) / 0.25)[None, :]
                    base = zsh + np.clip(zpl - zsh, 0, K["max_above_shoulder_m"]) * fade
                    lift = np.where(t > 0, K["lift_m"], 0.0)[None, :] * np.ones((n, 1))
                    # dev6 FFB safety: where the shoulder forces a narrow kerb, scale its height with the width so the
                    # steepest lip (about 3 x height / width with these profiles) stays under ~14 deg; dev5 squeezed
                    # the full 2.5 cm into 10 cm => 40-50 deg micro-ramps with 2-5 cm of relief
                    hmax = np.minimum(P_["height_m"], K.get("max_lip_slope", 0.08) * w)
                    # ... and where the shoulder falls away from the road plane (steep verge), the raised profile on top
                    # of the falling shoulder made 3-6 cm folds (dev6 gate): fade the profile out there (flat paint)
                    from scipy.ndimage import gaussian_filter1d as _g1
                    drop = (zpl - zsh).max(1)
                    hk = np.clip(_g1(1 - smoothstep((drop - K.get("max_shoulder_drop_m", 0.025)) / 0.02), 2.0, mode="nearest"), 0, 1)
                    z = base + lift + (hmax * hk)[:, None] * hp[None, :] * tap[:, None]
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
                    gap = (D["gap_l"] if side == "left" else D["gap_r"])[i]
                    KERBS.append(dict(s0=float(road.s[i[0]]), s1=float(road.s[i[-1]]), side=side, dir=dname, type=typ,
                                      line_gap_min_m=round(float(gap.min()), 2), v_kmh=round(float(D["v"][i].mean() * 3.6), 0)))
                    if typ == "exit" and K["runoff"]["enabled"]:
                        build_runoff(road, K["runoff"], i, side, sgn, SX, SY, SZ, env[side], gr_mask[side] | ~wide[side], dname, P3, TT, push, chunk_of)
    for (dname, kind, c), a in sorted(acc.items()):
        add(f"kerbs_{dname}", a["name"], a["mat"], np.concatenate(a["P"]), np.concatenate(a["T"]), np.concatenate(a["UV"]),
            cast_shadows=False)
    return KERBS, corners


def access_mask(road, PL):
    """stations (per side) where a kerb would cross a driveway / junction / apron mouth"""
    import json
    from geom import DER
    N = len(road.s); m = {"left": np.zeros(N, bool), "right": np.zeros(N, bool)}
    def mark(s0, s1, side):
        for sd_ in (("left", "right") if side == "both" else (side,)):
            m[sd_] |= (road.s >= s0) & (road.s <= s1)
    for f in json.load(open(DER / "features.geojson"))["features"]:
        p = f["properties"]
        if p["kind"] not in ("driveway", "junction") or "s_m" not in p:
            continue
        r = PL["junction_clear_m"] if p["kind"] == "junction" else PL["driveway_clear_m"]
        o = p.get("offset_m", 0) or 0
        mark(p["s_m"] - r, p["s_m"] + r, "left" if o > PL["side_offset_m"] else "right" if o < -PL["side_offset_m"] else "both")
    try:
        import edge_profile
        for s0, s1, side, _mat in edge_profile._aprons():
            mark(s0 - PL["apron_clear_m"], s1 + PL["apron_clear_m"], side)
    except Exception as e:   # survey files optional
        print("kerbs: no apron spans:", e)
    return m


def line_rows(road, D, sd, PL, env, skip, excl):
    """kerb row labels from the racing line (station order). 1 = apex (inside, line touches the inside edge
    while turning toward it), 2 = exit (outside, line runs out to the edge after an apex on the other side).
    Entry-side touches (turn-in from the outside) get no kerb. Returns rowlab, wide (run-off allowed), info."""
    N = len(road.s)
    k = gaussian_filter1d(D["k"], 2)            # + = curving toward station-left (both directions)
    gap = {"left": D["gap_l"], "right": D["gap_r"]}
    touch = {sd_: gap[sd_] < PL["touch_m"] for sd_ in gap}
    turning = {"left": k > PL["apex_min_k_1pm"], "right": k < -PL["apex_min_k_1pm"]}
    rowlab = {"left": np.zeros(N, np.int8), "right": np.zeros(N, np.int8)}
    apex_side = np.zeros(N, np.int8)             # +1 apex on left, -1 on right
    apexes = []
    for side, sg in (("left", 1), ("right", -1)):
        for r in runs(touch[side] & turning[side]):
            if len(r) < PL["min_touch_m"]:
                continue
            apex_side[r] = sg
            ia = r[np.argmin(gap[side][r])]; apexes.append((int(ia), side))
    # walk in run direction: an outside touch within exit_lookback_m after an apex on the OTHER side = exit
    order = np.arange(N) if sd > 0 else np.arange(N)[::-1]
    last_side, last_pos = 0, -1e9
    exitm = {"left": np.zeros(N, bool), "right": np.zeros(N, bool)}
    for n_, i in enumerate(order):
        if apex_side[i]:
            last_side, last_pos = apex_side[i], n_
        for side, sg in (("left", 1), ("right", -1)):
            if touch[side][i] and not apex_side[i] and last_side == -sg and n_ - last_pos <= PL["exit_lookback_m"]:
                exitm[side][i] = True
    pad = int(round(PL["pad_m"]))
    forbid = {"apex": PL["apex_forbid_env"], "exit": PL["exit_forbid_env"]}
    stats = dict(apex_rows=0, exit_rows=0, dropped_access=0, dropped_env=0)
    for side, sg in (("left", 1), ("right", -1)):
        for v, typ, msk in ((2, "exit", exitm[side]), (1, "apex", (apex_side == sg))):
            for r in runs(msk):
                a = max(r[0] - pad, 0); b = min(r[-1] + pad, N - 1)
                rr = np.arange(a, b + 1)
                bad_env = np.isin(env[side][rr], forbid[typ]); bad_acc = excl[side][rr]
                stats["dropped_env"] += int(bad_env.sum()); stats["dropped_access"] += int((bad_acc & ~bad_env).sum())
                ok = rr[~bad_env & ~bad_acc]
                if v == 2:
                    ok = ok[rowlab[side][ok] == 0]
                rowlab[side][ok] = v                         # apex (written last) wins over an exit
                stats[f"{typ}_rows"] += len(ok)
    wide = {side: gaussian_filter1d((gap[side] < PL["runoff_gap_m"]).astype(float), 3) > 0.05 for side in gap}
    for side in rowlab:
        rowlab[side][skip] = 0
    return rowlab, wide, dict(apexes=apexes, stats=stats)


def build_runoff(road, R, i, side, sgn, SX, SY, SZ, env, grm, dname, P3, TT, push, chunk_of):
    """painted strip on the terrain beyond the exit kerb; inner column == shoulder/terrain seam vertex"""
    from geom import grid_tris
    ok = np.isin(env[i], R["allowed_env"]) & ~grm[i]          # grm also masks rows where the line does not run wide
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
