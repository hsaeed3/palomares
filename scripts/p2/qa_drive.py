#!/usr/bin/env python3
"""Drive simulation + hill-climb safety QA on the ROUND-TRIPPED kn5 physics meshes (build/palomares_road).

For the full-length layouts of each style and direction (road_full, track_full = nb; *_full_rev = sb) a virtual
car follows the layout's own <layout>/ai/fast_lane.ai (read back with ai_spline.read) at the AI speed:
  1. wheel contact: left/right wheel tracks (+-0.8 m) sampled every 0.1 m on the highest drivable physics surface
     (1ROAD / 1KERB / 1GRASS / 1RUNOFF of that layout's kn5 files): holes (no surface), vertical steps > 2 cm,
     vertical acceleration of the contact path at line speed (tyre footprint low-pass).
  2. kerb ride (track layouts): a "kerb-using" line pushed so the inside/outside tyre runs 0.3 m onto the kerb
     wherever the line comes within placement.touch_m of the edge; vertical accel per kerb, plus a static
     cross-section survey of every kerb: height above the surface it sits on, ramp angle, outer-edge drop.
  3. walls: any 1WALL vertex / face sample within the car envelope (1.0 m either side of the line, 0-2 m high).
  4. props: renderable non-road meshes (dressing / roadside / barriers / trees) with geometry over the asphalt
     below 4.5 m.
  5. organiser review: unprotected drops on the outside of fast corners (line speed), boards / barriers in the
     run-wide zone after an exit, blind crests without a marshal post, gantries/huts clearance.
Writes reports/safety_qa.md and reports/safety_qa.json. Read-only on build products.
"""
import configparser, json, pickle, re, sys, time, pathlib
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, label
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import ROOT, BUILD, DER, CFG, Road
from kn5 import read_kn5
import ai_spline, racing_line as RLm
import yaml

T0 = time.time()
TRK = BUILD / "palomares_road"
STEP_LIM = 0.02
AZ_LIM_G = 3.0          # vertical accel of the contact path that would unsettle / launch a car (flag)
AZ_SIGMA = 3            # samples (0.3 m): tyre envelope + suspension low-pass of the contact path
ENV_HALF = 1.0
KC = yaml.safe_load(open(ROOT / "config/kerbs.yaml"))
road = Road(CFG["build_end_m"]); NS = len(road.s)
ckd = cKDTree(np.c_[road.x, road.y])


def to_local(p):
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1).astype(np.float64)


def station(xy):
    d, j = ckd.query(xy)
    off = (xy[:, 0] - road.x[j]) * road.nx[j] + (xy[:, 1] - road.y[j]) * road.ny[j]
    return j, off


KN = {}
def kn(f):
    if f not in KN:
        KN[f] = read_kn5(str(TRK / f))
    return KN[f]


def layout_files(lay):
    cp = configparser.ConfigParser(); cp.optionxform = str; cp.read(TRK / f"models_{lay}.ini")
    return [cp[s]["FILE"] for s in cp.sections()]


class Surf:
    """highest-surface sampler over many triangle meshes (vectorised barycentric on KD candidates)"""
    def __init__(self, items):
        P, T, own, names = [], [], [], []; off = 0
        for k, (nm, p, t) in enumerate(items):
            P.append(p); T.append(t + off); own.append(np.full(len(t), k)); names.append(nm); off += len(p)
        self.P = np.vstack(P); self.T = np.vstack(T); self.own = np.concatenate(own); self.names = names
        A, B, C = (self.P[self.T[:, j]] for j in range(3))
        self.cen = (A[:, :2] + B[:, :2] + C[:, :2]) / 3
        self.r = np.max([np.hypot(*(X[:, :2] - self.cen).T) for X in (A, B, C)], 0)
        self.tree = cKDTree(self.cen); self.rmax = float(self.r.max())
        self.A, self.B, self.C = A, B, C

    def sample(self, q, chunk=40000):
        z = np.full(len(q), np.nan); who = np.full(len(q), -1)
        for c0 in range(0, len(q), chunk):
            qq = q[c0:c0 + chunk]
            L = self.tree.query_ball_point(qq, self.rmax + 1e-3)
            cnt = np.array([len(l) for l in L]);
            if not cnt.sum():
                continue
            qi = np.repeat(np.arange(len(qq)), cnt); ti = np.concatenate([np.asarray(l, int) for l in L if l])
            A, B, C = self.A[ti], self.B[ti], self.C[ti]; p = qq[qi]
            v0 = B[:, :2] - A[:, :2]; v1 = C[:, :2] - A[:, :2]; v2 = p - A[:, :2]
            d = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            with np.errstate(all="ignore"):
                u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / d; v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / d
            ok = (np.abs(d) > 1e-12) & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
            zz = np.where(ok, A[:, 2] + u * (B[:, 2] - A[:, 2]) + v * (C[:, 2] - A[:, 2]), -np.inf)
            best = np.full(len(qq), -np.inf); np.maximum.at(best, qi, zz)
            hit = np.isfinite(best)
            z[c0:c0 + chunk][hit] = best[hit]
            # owner of the top surface
            top = ok & (zz >= best[qi] - 1e-9)
            w = np.full(len(qq), -1); w[qi[top]] = self.own[ti[top]]
            who[c0:c0 + chunk] = w
        return z, who


def phys_items(files, keys, near=None, r=8.0):
    """physics meshes of these surface keys; with near (KD tree of line points) only triangles within r m of it"""
    out = []
    for f in files:
        for m in kn(f)["meshes"]:
            mm = re.match(r"^\d+([A-Z]+)", m["name"])
            key = next((k_ for k_ in keys if mm and mm.group(1).startswith(k_)), None)   # AC: surface KEY is a name prefix (1ROADF_ = ROAD)
            if key and len(m["idx"]):
                p = to_local(m["pos"]); t = m["idx"].reshape(-1, 3)
                if near is not None:
                    c = (p[t[:, 0], :2] + p[t[:, 1], :2] + p[t[:, 2], :2]) / 3
                    t = t[near.query(c, distance_upper_bound=r)[0] < np.inf]
                    if not len(t):
                        continue
                out.append((f"{f}:{m['name']}", p, t))
    return out


def resample(P, v, ds=0.1):
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
    s = np.arange(0, seg[-1], ds)
    Q = np.stack([np.interp(s, seg, P[:, k]) for k in range(3)], 1)
    return s, Q, np.interp(s, seg, v)


def wheel_paths(Q, half=0.8):
    f = np.gradient(Q[:, :2], axis=0); f /= np.linalg.norm(f, axis=1, keepdims=True)
    l = np.c_[-f[:, 1], f[:, 0]]
    return {"L": Q[:, :2] + l * half, "R": Q[:, :2] - l * half}, l


def ranges(mask, s_of, minlen=1):
    lab, n = label(mask); out = []
    for k in range(1, n + 1):
        ix = np.nonzero(lab == k)[0]
        if len(ix) >= minlen:
            out.append((ix[0], ix[-1]))
    return out


F = []   # findings
def find(layout, s, side, sev, kind, msg, fix=""):
    F.append(dict(layout=layout, s=round(float(s), 1), side=side, severity=sev, kind=kind, msg=msg, fix=fix))


SUM = {}
RL = RLm.load(); RC = RLm.cfg()
for lay in ("road_full", "road_full_rev", "track_full", "track_full_rev"):
    files = layout_files(lay)
    sd = 1 if not lay.endswith("_rev") else -1; dname = "nb" if sd > 0 else "sb"
    ai = ai_spline.read(TRK / lay / "ai" / "fast_lane.ai")
    P = to_local(ai["P"]); s_ln, Q, V = resample(P, ai["extra"]["speed"])
    drv_items = phys_items(files, ("ROAD", "KERB", "GRASS", "RUNOFF"), cKDTree(Q[::5, :2]))
    S = Surf(drv_items); kind_of = np.array([next(k_ for k_ in ("RUNOFF", "ROAD", "KERB", "GRASS") if re.match(r"^\d+([A-Z]+)", n.split(":")[1]).group(1).startswith(k_)) for n in S.names])
    W, lft = wheel_paths(Q)
    st_line, _ = station(Q[:, :2])
    res = dict(points=len(Q), length_m=float(s_ln[-1]))
    for wn, wp in W.items():
        z, who = S.sample(wp)
        hole = np.isnan(z)
        for a, b in ranges(hole, s_ln):
            find(lay, road.s[st_line[a]], wn, "HIGH", "hole", f"no drivable surface under the {wn} wheel for {(b - a + 1) * 0.1:.1f} m")
        zf = np.where(hole, np.interp(np.arange(len(z)), np.nonzero(~hole)[0], z[~hole]), z)
        dz = np.diff(zf); step = np.abs(dz - median_filter(dz, 11, mode="nearest"))
        res[f"max_step_{wn}_m"] = float(step.max()); res[f"n_step_over_{wn}"] = int((step > STEP_LIM).sum())
        for a, b in ranges(step > STEP_LIM, s_ln):
            k = a + int(np.argmax(step[a:b + 1]))
            sk = kind_of[who[k]] if who[k] >= 0 else "?"; sk2 = kind_of[who[k + 1]] if who[k + 1] >= 0 else "?"
            find(lay, road.s[st_line[k]], wn, "HIGH" if step[k] > 0.04 else "MED", "step",
                 f"vertical step {step[k] * 100:.1f} cm on the line ({sk}->{sk2}) at {V[k] * 3.6:.0f} km/h")
        zs = gaussian_filter1d(zf, AZ_SIGMA)       # tyre envelope low-pass (mesh facets are not felt)
        az = np.gradient(np.gradient(zs, 0.1), 0.1) * V ** 2 / 9.81
        res[f"max_az_{wn}_g"] = float(np.abs(az).max()); res[f"p999_az_{wn}_g"] = float(np.percentile(np.abs(az), 99.9))
        bad = np.abs(az) > AZ_LIM_G
        for a, b in ranges(bad, s_ln):
            k = a + int(np.argmax(np.abs(az[a:b + 1])))
            find(lay, road.s[st_line[k]], wn, "MED", "vertical_accel", f"contact-path vertical accel {az[k]:+.1f} g at {V[k] * 3.6:.0f} km/h on {kind_of[who[k]] if who[k] >= 0 else '?'}")
    # ---- kerb ride (track layouts): push the car onto the kerbs where the line is near the edge
    if lay.startswith("track"):
        D = RL[dname]; j = st_line
        gl, gr_ = D["gap_l"][j], D["gap_r"][j]
        # station-left is the run-left for nb; shift in station terms
        near_l = gl < KC["placement"]["touch_m"]; near_r = gr_ < KC["placement"]["touch_m"]
        shift = np.where(near_l & (gl <= gr_), gl + 0.3, np.where(near_r, -(gr_ + 0.3), 0.0))
        shift = gaussian_filter1d(shift, 30)                       # 3 m blend in and out
        nrm = np.c_[road.nx[j], road.ny[j]]
        Qk = Q[:, :2] + nrm * shift[:, None]
        Wk, _ = wheel_paths(np.c_[Qk, Q[:, 2]])
        kmax = []
        for wn, wp in Wk.items():
            z, who = S.sample(wp); hole = np.isnan(z)
            zf = np.where(hole, np.interp(np.arange(len(z)), np.nonzero(~hole)[0], z[~hole]), z)
            on_k = (who >= 0) & (kind_of[np.maximum(who, 0)] == "KERB")
            dz = np.diff(zf); step = np.abs(dz - median_filter(dz, 11, mode="nearest"))
            az = np.gradient(np.gradient(gaussian_filter1d(zf, AZ_SIGMA), 0.1), 0.1) * V ** 2 / 9.81
            for a, b in ranges(on_k, s_ln, 5):
                a2, b2 = max(a - 10, 0), min(b + 10, len(az) - 1)
                k = a2 + int(np.argmax(np.abs(az[a2:b2 + 1])))
                kmax.append((float(abs(az[k])), float(step[a2:b2].max()), float(road.s[j[k]]), wn, float(V[k] * 3.6)))
        res["kerb_rides"] = len(kmax)
        if kmax:
            res["kerb_ride_max_az_g"] = max(k[0] for k in kmax); res["kerb_ride_max_step_m"] = max(k[1] for k in kmax)
            for azk, stk, sk, wn, vk in kmax:
                if azk > AZ_LIM_G or stk > STEP_LIM:
                    find(lay, sk, wn, "HIGH" if (azk > 4 or stk > 0.04) else "MED", "kerb_ride",
                         f"riding the kerb at {vk:.0f} km/h: vertical accel {azk:.1f} g, step {stk * 100:.1f} cm")
        # static kerb cross-sections: height above the surface beneath, ramp angle, outer drop
        KI = phys_items([f for f in files if "kerbs" in f], ("KERB",))
        if KI:
            K_ = Surf(KI); base = Surf([it for it in drv_items if "KERB" not in it[0]])
            kv = np.vstack([it[1] for it in KI]); jj, off = station(kv[:, :2])
            sts = np.unique(jj)[::2]
            prof = []
            for i in sts:
                side = 1 if np.median(off[jj == i]) > 0 else -1
                e = road.el[i] if side > 0 else road.er[i]
                dd = np.arange(e - 0.05, e + 1.0, 0.02)
                x, y = road.xy(np.full(len(dd), i), side * dd)
                zk, _ = K_.sample(np.c_[x, y]); zb, _ = base.sample(np.c_[x, y])
                m = np.isfinite(zk)
                if m.sum() < 3:
                    continue
                h = zk[m] - np.where(np.isfinite(zb[m]), zb[m], zk[m])
                ztop = np.fmax(zk, zb); g_ = np.gradient(ztop, 0.02)
                ang = np.degrees(np.arctan(np.nanmax(np.abs(g_[np.isfinite(g_)]))))
                rise_out = zk[m][-1] - zk[m].max()
                prof.append((float(h.max()), float(ang), float(-rise_out), float(road.s[i]), "left" if side > 0 else "right"))
            P_ = np.array([p[:3] for p in prof])
            res["kerb_profile"] = dict(n=len(prof), max_height_above_base_m=float(P_[:, 0].max()),
                                       p95_height_m=float(np.percentile(P_[:, 0], 95)), max_ramp_deg=float(P_[:, 1].max()),
                                       p95_ramp_deg=float(np.percentile(P_[:, 1], 95)), max_outer_drop_m=float(P_[:, 2].max()))
            worst = sorted(prof, key=lambda p: -p[0])[:3]
            for h, ang, drop, sk, side in worst:
                if h > 0.05:
                    find(lay, sk, side, "MED", "kerb_height", f"kerb top {h * 1000:.0f} mm above the surface below it (ramp {ang:.0f} deg, outer drop {drop * 1000:.0f} mm)")
    # ---- walls inside the car envelope
    WI = phys_items(files, ("WALL",), cKDTree(Q[::5, :2]), 6.0)
    pts = []; nm = []
    for name, p, t in WI:
        A, B, C = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
        for X in (A, (A + B) / 2, (B + C) / 2, (A + C) / 2, (A + B + C) / 3):
            pts.append(X); nm += [name] * len(X)
    WP = np.vstack(pts); nm = np.array(nm)
    lkd = cKDTree(Q[:, :2]); dW, k = lkd.query(WP[:, :2])
    zl = Q[k, 2]
    hit = (dW < ENV_HALF) & (WP[:, 2] > zl + 0.05) & (WP[:, 2] < zl + 2.0)
    endw = np.char.find(nm.astype(str), ":1WALL_start") >= 0
    endw |= np.char.find(nm.astype(str), ":1WALL_end") >= 0
    res["end_walls_touch_line"] = bool((hit & endw).any())
    hit &= ~endw                                   # end-of-world caps behind the spline ends (intentional)
    seen = set()
    for q in np.nonzero(hit)[0]:
        key = (nm[q], int(road.s[st_line[k[q]]] // 25))
        if key in seen: continue
        seen.add(key)
        jj_, off_ = station(WP[q:q + 1, :2])
        find(lay, road.s[jj_[0]], "left" if off_[0] > 0 else "right", "HIGH", "wall_in_envelope",
             f"{nm[q].split(':')[1]} {dW[q]:.2f} m from the line ({off_[0]:+.1f} m from the centreline)")
    res["walls_in_envelope"] = len(seen)
    SUM[lay] = res
    print(f"{lay}: {res}", flush=True)

# ---- props over the asphalt (renderable meshes, not road/lines/terrain/trees canopy above 4.5 m)
props = {}
for f in sorted(p.name for p in TRK.glob("*.kn5")):
    if any(k in f for k in ("_road.kn5", "terrain", "lines", "timing")):
        continue
    for m in kn(f)["meshes"]:
        if m["name"][:1].isdigit() or not len(m["pos"]):
            continue
        p = to_local(m["pos"]); j, off = station(p[:, :2])
        half = np.where(off > 0, road.el[j], road.er[j])
        zr = road.surface_z(j, off, noise=False)
        inside = (np.abs(off) < half - 0.05) & (p[:, 2] < zr + 4.5) & (p[:, 2] > zr + 0.10) & (j > 2) & (j < NS - 3)   # flat decals (<= 10 cm) are paint
        if inside.any():
            q = np.nonzero(inside)[0]
            for s0 in np.unique(road.s[j[q]] // 50):
                props.setdefault((f, m["name"]), []).append(s0 * 50)
for (f, n), ss in props.items():
    for s0 in ss[:5]:
        find(f.replace(".kn5", ""), s0, "-", "HIGH" if "tree" not in f else "MED", "prop_on_road", f"{n} has geometry over the asphalt below 4.5 m")

# ---- organiser review (geometry.pkl meta + racing line)
meta = pickle.load(open(BUILD / "geometry.pkl", "rb"))["meta"]
gr = meta.get("guardrails", []); KR = meta.get("kerbs", []); BB = meta.get("brake_boards", {}); LOGD = meta.get("dressing", [])
feats = [f["properties"] for f in json.load(open(DER / "features.geojson"))["features"]]
drops = [p for p in feats if p.get("kind") == "steep_drop"]
bar = []
for l in LOGD:
    m = re.match(r"(tyre_belt|tecpro|jersey) s=(\d+)-(\d+) ([LR])", l)
    if m: bar.append((m.group(1), float(m.group(2)), float(m.group(3)), "left" if m.group(4) == "L" else "right"))
for p in drops:
    s0, s1, side = p["s_start_m"], p["s_end_m"], p["side"]
    if s1 > road.s[-1] - 5: continue
    guarded = any(g["side"] == side and g["s0"] - 5 < s1 and g["s1"] + 5 > s0 for g in gr)
    barred = any(b[3] == side and b[1] - 5 < s1 and b[2] + 5 > s0 for b in bar)
    i = np.arange(int(max(s0 - 20, 0)), int(min(s1 + 20, NS - 1)))
    for dname in ("nb", "sb"):
        D = RL[dname]; v = D["v"][i] * 3.6
        gap = (D["gap_l"] if side == "left" else D["gap_r"])[i]
        k = D["k"][i]; outside = (k < -0.004) if side == "left" else (k > 0.004)
        risk = (v > 100) & (outside | (gap < 0.6))
        if risk.any() and not guarded:
            q = i[np.argmax(np.where(risk, v, 0))]
            sev = "HIGH" if (v[risk].max() > 130 and not barred) else ("INFO" if barred else "MED")
            find(f"track_* {dname}", road.s[q], side, sev, "unprotected_drop",
                 f"steep drop ({p.get('length_m', s1 - s0):.0f} m) on the outside at {v[risk].max():.0f} km/h, line gap {gap.min():.2f} m"
                 + (" — event barrier present" if barred else " — no guardrail / event barrier"),
                 "covered by 21b event barrier" if barred else "")
# boards in the run-wide zone after an exit (direction-specific)
for dname, boards in BB.items():
    D = RL[dname]
    for b in boards:
        for pl in b.get("placed", [dict(s=s_, off_m=0.9, side=b["outside"]) for s_ in b["boards"]]):
            s_b = pl["s"]; i = int(round(s_b)); side = pl["side"]
            if pl["off_m"] >= 3.0:
                continue                                   # set back beyond the 2 m run-off

            ex = [k for k in KR if k["dir"] == dname and k["type"] == "exit" and k["side"] == side and k["s0"] - 5 <= s_b <= k["s1"] + 25]
            gap = (D["gap_l"] if side == "left" else D["gap_r"])[max(i - 10, 0):i + 10].min()
            if ex or gap < 0.45:
                find(f"track_* {dname}", s_b, side, "MED", "board_in_runwide", f"brake board in the run-wide zone (exit kerb/line gap {gap:.2f} m)",
                     "21b moves boards out of exit/run-off zones")
# blind crests without a marshal post
posts = [float(re.search(r"s=(\d+)", l).group(1)) for l in LOGD if l.strip().startswith("marshal post") and "s=" in l and "no flat" not in l]
zs = gaussian_filter1d(road.z, 8); vc = np.gradient(np.gradient(zs))
lab, n = label(vc < -0.0025)
for l in range(1, n + 1):
    ix = np.nonzero(lab == l)[0]; ic = ix[np.argmin(vc[ix])]
    for dname in ("nb", "sb"):
        v = RL[dname]["v"][ic] * 3.6
        sight = np.sqrt(2 * 1.2 / max(-vc[ic], 1e-4)) * 2
        if v > 110 and sight < 120 and not any(abs(road.s[ic] - p) < 250 for p in posts):
            find(f"track_* {dname}", road.s[ic], "-", "LOW", "blind_crest", f"crest at {v:.0f} km/h, sight distance ~{sight:.0f} m, no marshal post within 250 m")
# gantries / huts clearance from the dressing log
for l in LOGD:
    m = re.match(r"gantry s=(\d+) span ([\d.]+) m clear ([\d.]+) m", l)
    if m and float(m.group(3)) < 5.5:
        find("track_*", float(m.group(1)), "-", "HIGH", "gantry_clearance", l)

# ---- report
FIXES = [
    "Kerbs (track_*): placed from the racing line of each direction (kerbs.py `line_rows`): apex only where the line's tyre is "
    "within 0.6 m of the inside edge while turning toward it, exit only after an apex on the other side; none at driveways (8 m), "
    "junctions (15 m), surveyed aprons, steep-drop insides or drop-off / steep-drop exits; run-off only where the line runs to <= 0.45 m "
    "of the edge and the ground is flat.",
    "Brake boards (21b): at 150/100/50 m before the turn-in of the 8 heaviest braking zones of the speed profile per direction; "
    "moved to >= 3.2 m past the shoulder (beyond the run-off) wherever they would sit in an exit / run-wide zone.",
    "Event barriers (21b): unprotected steep drops ranked by racing-line exposure (speed where the drop is on the outside of the "
    "line or the line runs within 0.6 m of it, either direction); slow / inside-only drops (< 70 km/h) are no longer barriered.",
    "Terrain holes (21d `fill_holes`): the open terrain edges near the road (gaps between terrain and the road end rows at s=0 "
    "and s=end) are closed along the road end row (1GRASS_hole_fill).",
]
sev_rank = {"HIGH": 0, "MED": 1, "LOW": 2, "INFO": 3}
F.sort(key=lambda f: (sev_rank[f["severity"]], f["kind"], f["layout"], f["s"]))
json.dump(dict(summary=SUM, findings=F, runtime_s=time.time() - T0), open(ROOT / "reports/safety_qa.json", "w"), indent=1)
cnt = {}
for f in F: cnt[(f["severity"], f["kind"])] = cnt.get((f["severity"], f["kind"]), 0) + 1
L = ["# Safety QA — drive simulation + hill-climb organiser review", "",
     f"Generated by `scripts/p2/qa_drive.py` on the round-tripped kn5 physics meshes in `build/palomares_road` "
     f"(fast_lane.ai of each layout read back). Runtime {time.time() - T0:.0f} s.", "",
     "## Simulation summary", "", "| layout | length m | max step L/R cm | steps >2 cm L/R | max contact az L/R g | kerb rides | kerb-ride max az g | walls in envelope |",
     "|---|---|---|---|---|---|---|---|"]
for lay, r in SUM.items():
    L.append(f"| {lay} | {r['length_m']:.0f} | {r['max_step_L_m'] * 100:.1f} / {r['max_step_R_m'] * 100:.1f} | {r['n_step_over_L']} / {r['n_step_over_R']} | "
             f"{r['max_az_L_g']:.1f} / {r['max_az_R_g']:.1f} | {r.get('kerb_rides', '-')} | {r.get('kerb_ride_max_az_g', float('nan')):.1f} | {r['walls_in_envelope']} |")
for lay, r in SUM.items():
    if "kerb_profile" in r:
        kp = r["kerb_profile"]
        L.append(f"\n{lay} kerb cross-sections ({kp['n']}): max height above base {kp['max_height_above_base_m'] * 1000:.0f} mm "
                 f"(p95 {kp['p95_height_m'] * 1000:.0f} mm), max ramp {kp['max_ramp_deg']:.0f} deg (p95 {kp['p95_ramp_deg']:.0f}), "
                 f"max outer-edge drop {kp['max_outer_drop_m'] * 1000:.0f} mm")
L += ["", "## Fixes applied in the pipeline (dev4)", ""] + [f"- {x}" for x in FIXES]
L += ["", "## Finding counts", "", "| severity | kind | n |", "|---|---|---|"] + [f"| {a} | {b} | {n} |" for (a, b), n in sorted(cnt.items(), key=lambda x: (sev_rank[x[0][0]], x[0][1]))]
L += ["", "## Findings", "", "| severity | kind | layout | station m | side | finding | fix |", "|---|---|---|---|---|---|---|"]
for f in F[:400]:
    L.append(f"| {f['severity']} | {f['kind']} | {f['layout']} | {f['s']:.0f} | {f['side']} | {f['msg']} | {f['fix']} |")
if len(F) > 400:
    L.append(f"\n({len(F) - 400} more in reports/safety_qa.json)")
(ROOT / "reports/safety_qa.md").write_text("\n".join(L) + "\n")
print(f"safety QA: {len(F)} findings", cnt, f"{time.time() - T0:.0f}s")
