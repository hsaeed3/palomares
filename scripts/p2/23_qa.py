#!/usr/bin/env python3
"""Phase 2 independent QA (CLAUDE.md section 4). Writes reports/qa_phase2.md + .json.
Re-runnable, read-only on build products."""
import sys, os, json, glob, re, struct, io, configparser, time
import numpy as np
from scipy.spatial import cKDTree
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "p2"))
from kn5 import read_kn5
import rasterio

T0 = time.time()
TRK = os.path.join(ROOT, "build", "palomares_road")
R = {"checks": {}}
def chk(name, ok, **kw):
    R["checks"][name] = dict(status="PASS" if (ok is True or (not isinstance(ok,str) and bool(ok))) else ("WARN" if ok == "WARN" else "FAIL"), **kw)

# ---------- load ----------
cl = np.genfromtxt(os.path.join(ROOT, "data/derived/centerline.csv"), delimiter=",", names=True, dtype=None, encoding="utf-8")
S = cl["s_m"].astype(float); CX = cl["x_local"].astype(float); CY = cl["y_local"].astype(float); CZ = cl["z_navd88"].astype(float)
EL = cl["edge_left_m"].astype(float); ER = cl["edge_right_m"].astype(float)
org = json.load(open(os.path.join(ROOT, "config/origin.json")))
BUILD_END, FINISH, START_GATE = 7642, 7392, 20
try:
    import yaml; ly = yaml.safe_load(open(os.path.join(ROOT, "config/layouts.yaml")))
    BUILD_END, FINISH, START_GATE = ly["build_end_m"], ly["finish_gate_m"], ly["start_gate_m"]
except Exception: pass
m_b = S <= BUILD_END
tx = np.gradient(CX); ty = np.gradient(CY); tn = np.hypot(tx, ty); tx /= tn; ty /= tn
nlx, nly = -ty, tx  # left normal

kn5 = {}
c1 = {}
for f in sorted(glob.glob(os.path.join(TRK, "*.kn5"))):
    fn = os.path.basename(f)
    try:
        kn5[fn] = read_kn5(f); c1[fn] = "ok"
    except Exception as e:
        c1[fn] = f"PARSE FAIL: {e}"

# ---------- 1. kn5 validation ----------
probs = []
def tex_ok(b):
    return b[:8] == b"\x89PNG\r\n\x1a\n" or b[:4] == b"DDS "
for fn, k in kn5.items():
    for tn_, b in k["textures"].items():
        if not tex_ok(b): probs.append(f"{fn}: texture {tn_} not PNG/DDS")
        if b[:8] == b"\x89PNG\r\n\x1a\n":
            try:
                from PIL import Image; Image.open(io.BytesIO(b)).verify()
            except Exception as e: probs.append(f"{fn}: texture {tn_} corrupt PNG ({e})")
    for mt in k["materials"]:
        for sn, t in mt["textures"].items():
            if t not in k["textures"]: probs.append(f"{fn}: material {mt['name']} {sn} -> missing {t}")
    for m in k["meshes"]:
        nv = len(m["pos"])
        if nv > 65535: probs.append(f"{fn}:{m['name']} {nv} verts")
        if len(m["idx"]) and m["idx"].max() >= nv: probs.append(f"{fn}:{m['name']} index out of range")
        if len(m["idx"]) % 3: probs.append(f"{fn}:{m['name']} index count not /3")
        for a in ("pos", "nrm", "uv"):
            if not np.isfinite(m[a]).all(): probs.append(f"{fn}:{m['name']} non-finite {a}")
        if m["material"] >= len(k["materials"]): probs.append(f"{fn}:{m['name']} bad material id")
        if len(m["idx"]) == 0: probs.append(f"{fn}:{m['name']} empty mesh")
        t = m["idx"].reshape(-1, 3)
        if len(t) and np.any((t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2]) | (t[:, 0] == t[:, 2])):
            probs.append(f"{fn}:{m['name']} degenerate (index) triangles: {int(np.sum((t[:,0]==t[:,1])|(t[:,1]==t[:,2])|(t[:,0]==t[:,2])))}")
for fn, v in c1.items():
    if v != "ok": probs.append(f"{fn}: {v}")

layouts = {}
for ini in sorted(glob.glob(os.path.join(TRK, "models_*.ini"))):
    lay = os.path.basename(ini)[7:-4]
    cp = configparser.ConfigParser(); cp.optionxform = str; cp.read(ini)
    layouts[lay] = [cp[s]["FILE"] for s in cp.sections()]
dc = {}
for lay, files in layouts.items():
    n = 0; nmat = set()
    for f in files:
        if f in kn5:
            n += len(kn5[f]["meshes"])
    dc[lay] = n
    if n >= 1500: probs.append(f"layout {lay}: draw calls {n} >= 1500")
chk("1_kn5_validation", not probs, files=c1, problems=probs[:200], n_problems=len(probs), draw_calls=dc,
    note="draw calls = all meshes in layout kn5s (reader does not expose visible flag; upper bound)")

# ---------- gather drivable geometry (local ENU: x=east, y=north, z=up) ----------
def to_local(p):  # AC (e,u,-n) -> local (e,n,u)
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1).astype(np.float64)
surf = {"ROAD": [], "GRASS": [], "KERB": [], "WALL": []}
for fn, k in kn5.items():
    for m in k["meshes"]:
        mm = re.match(r"^\d*([A-Z]+)", m["name"])
        if m["name"][:1].isdigit() and mm and mm.group(1) in surf:
            surf[mm.group(1)].append((fn, m["name"], to_local(m["pos"]), m["idx"].reshape(-1, 3), m["nrm"]))
# surface key check vs surfaces.ini
used = {k for k, v in surf.items() if v}

class TriSet:
    def __init__(self, items):
        P, T, own = [], [], []; off = 0
        for i, (fn, nm, p, t, n) in enumerate(items):
            P.append(p); T.append(t + off); own.append(np.full(len(t), i)); off += len(p)
        self.P = np.vstack(P); self.T = np.vstack(T); self.own = np.concatenate(own); self.items = items
        a, b, c = (self.P[self.T[:, j], :2] for j in range(3))
        self.cen = (a + b + c) / 3; self.rad = np.max([np.hypot(*(a - self.cen).T), np.hypot(*(b - self.cen).T), np.hypot(*(c - self.cen).T)], 0)
        self.tree = cKDTree(self.cen); self.rmax = self.rad.max()
    def sample(self, q, exclude_owner=None, eps=1e-6):
        """height of surface at 2-D points q; NaN if none. returns (z, tri)"""
        z = np.full(len(q), np.nan); tri = np.full(len(q), -1)
        cand = self.tree.query_ball_point(q, self.rmax + 1e-3)
        for i, cs in enumerate(cand):
            if not cs: continue
            cs = np.array(cs)
            if exclude_owner is not None: cs = cs[self.own[cs] != exclude_owner[i]]
            if not len(cs): continue
            A = self.P[self.T[cs, 0]]; B = self.P[self.T[cs, 1]]; C = self.P[self.T[cs, 2]]
            v0 = B[:, :2] - A[:, :2]; v1 = C[:, :2] - A[:, :2]; v2 = q[i] - A[:, :2]
            d = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            ok = np.abs(d) > 1e-12
            with np.errstate(all="ignore"):
                u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / d; v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / d
            ins = ok & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps)
            if ins.any():
                j = np.where(ins)[0]
                zz = A[j, 2] + u[j] * (B[j, 2] - A[j, 2]) + v[j] * (C[j, 2] - A[j, 2])
                k = np.argmax(zz); z[i] = zz[k]; tri[i] = cs[j[k]]
        return z, tri

road = TriSet(surf["ROAD"])
drv_items = surf["ROAD"] + surf["GRASS"] + surf["KERB"]
drv = TriSet(drv_items)

# ---------- 2. geometry vs source ----------
sel = np.where(m_b & (S >= 1) & (S <= BUILD_END - 1))[0]  # end stations sit on the mesh boundary
q = np.stack([CX[sel], CY[sel]], 1)
zr, _ = road.sample(q)
with rasterio.open(os.path.join(ROOT, "data/derived/dtm_r050_b40.tif")) as ds:
    zd = np.array([v[0] for v in ds.sample(list(zip(CX[sel] + org["x0"], CY[sel] + org["y0"])))], float)
    if ds.nodata is not None: zd[zd == ds.nodata] = np.nan
miss = np.isnan(zr)
dz_cl = zr - CZ[sel]; dz_dtm = zr - zd
bad_cl = sel[np.abs(dz_cl) > 0.3]; bad_dtm = sel[np.abs(dz_dtm) > 0.3]
def ranges(idx):
    idx = np.sort(np.asarray(idx)); out = []
    if not len(idx): return out
    st = pv = idx[0]
    for i in idx[1:]:
        if i - pv > 3: out.append(f"{int(S[st])}-{int(S[pv])}" if st != pv else f"{int(S[st])}"); st = i
        pv = i
    out.append(f"{int(S[st])}-{int(S[pv])}" if st != pv else f"{int(S[st])}")
    return out
geo = dict(n_samples=int(len(sel)), road_missing_at_centerline=ranges(sel[miss]), n_missing=int(miss.sum()),
           vs_centerline=dict(mean=float(np.nanmean(dz_cl)), mean_abs=float(np.nanmean(np.abs(dz_cl))), max_abs=float(np.nanmax(np.abs(dz_cl))),
                              at_s=float(S[sel][np.nanargmax(np.abs(dz_cl))]), n_over_0p3=int(len(bad_cl)), stations_over_0p3=ranges(bad_cl)),
           vs_dtm=dict(mean=float(np.nanmean(dz_dtm)), mean_abs=float(np.nanmean(np.abs(dz_dtm))), max_abs=float(np.nanmax(np.abs(dz_dtm))),
                       at_s=float(S[sel][np.nanargmax(np.abs(dz_dtm))]), n_over_0p3=int(len(bad_dtm)), stations_over_0p3=ranges(bad_dtm)[:60]))
# layout: road mesh lateral extents vs centerline edges, measured along each station's normal
offs = np.arange(-12, 12.01, 0.05)
st2 = sel[::5]
lw, rw = np.full(len(st2), np.nan), np.full(len(st2), np.nan)
for ii, i in enumerate(st2):
    pts = np.stack([CX[i] + offs * nlx[i], CY[i] + offs * nly[i]], 1)
    z, _ = road.sample(pts)
    on = ~np.isnan(z)
    if on.any():
        # contiguous run containing 0
        i0 = np.argmin(np.abs(offs))
        if on[i0]:
            a = i0
            while a > 0 and on[a - 1]: a -= 1
            b = i0
            while b < len(offs) - 1 and on[b + 1]: b += 1
            lw[ii] = offs[b]; rw[ii] = -offs[a]
cen_off = (lw - rw) / 2  # road mesh centre offset (left +) from refined centerline
exp_cen = (EL[st2] - ER[st2]) / 2
e_cen = cen_off - exp_cen
e_l = lw - EL[st2]; e_r = rw - ER[st2]
lay_err = st2[(np.abs(e_cen) > 0.5) | np.isnan(e_cen)]
lay_edge = st2[(np.abs(e_l) > 0.5) | (np.abs(e_r) > 0.5)]
geo["layout"] = dict(step_m=5, mesh_center_minus_expected=dict(mean=float(np.nanmean(e_cen)), max_abs=float(np.nanmax(np.abs(e_cen))),
                     at_s=float(S[st2][np.nanargmax(np.abs(e_cen))])),
                     n_center_err_over_0p5=int(len(lay_err)), stations_center_err=ranges(lay_err)[:60],
                     edge_err_max_abs_left=float(np.nanmax(np.abs(e_l))), edge_err_max_abs_right=float(np.nanmax(np.abs(e_r))),
                     n_edge_err_over_0p5=int(len(lay_edge)), stations_edge_err=ranges(lay_edge)[:60],
                     mesh_width_min=float(np.nanmin(lw + rw)), mesh_width_max=float(np.nanmax(lw + rw)))
g_ok = (geo["vs_centerline"]["n_over_0p3"] == 0 and geo["n_missing"] == 0)
chk("2a_height_vs_centerline", g_ok, **{k: geo[k] for k in ("n_samples", "n_missing", "road_missing_at_centerline", "vs_centerline")})
chk("2b_height_vs_dtm", True if geo["vs_dtm"]["n_over_0p3"] == 0 else "WARN", vs_dtm=geo["vs_dtm"],
    note="DTM deviations are expected where the road was smoothed; investigate clusters")
chk("2c_layout", len(lay_err) == 0, **geo["layout"])

# ---------- 3. drivability ----------
# 3a vertical steps: (i) coincident vertex xy with differing z across all drivable meshes;
# (ii) every boundary vertex of each drivable mesh vs other drivable meshes' surface at that point.
P = drv.P
key = np.round(P[:, :2] / 0.002).astype(np.int64)
_, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
zmin = np.full(inv.max() + 1, np.inf); zmax = np.full(inv.max() + 1, -np.inf)
np.minimum.at(zmin, inv, P[:, 2]); np.maximum.at(zmax, inv, P[:, 2])
cstep = zmax - zmin
def boundary_edges(T):
    E = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    return u[c == 1], u[c > 2]
# per-mesh boundary vertices vs other meshes
step_rec = []
vown = np.concatenate([np.full(len(it[2]), i) for i, it in enumerate(drv_items)])
bverts = []
for i, (fn, nm, p, t, n) in enumerate(drv_items):
    be, _ = boundary_edges(t)
    bverts.append(np.unique(be.ravel()) + int(np.sum([len(x[2]) for x in drv_items[:i]])))
bv = np.concatenate(bverts)
zo, to = drv.sample(P[bv, :2], exclude_owner=vown[bv], eps=1e-4)
d = np.abs(zo - P[bv, 2]); has = ~np.isnan(d)
def station_of(xy):
    tr = cKDTree(np.stack([CX, CY], 1)); dd, ii = tr.query(xy); return S[ii], dd
s_bv, dist_bv = station_of(P[bv, :2])
big = has & (d > 0.02)
steps = []
for j in np.argsort(-np.where(big, d, 0))[:int(big.sum())]:
    steps.append(dict(dz=round(float(d[j]), 3), signed_this_minus_other=round(float(P[bv[j],2]-zo[j]),3), s=float(s_bv[j]), off=round(float(dist_bv[j]), 1),
                      a=drv_items[vown[bv[j]]][1], b=drv_items[drv.own[to[j]]][1]))
# summarise by pair type
from collections import Counter, defaultdict
pair = defaultdict(lambda: [0, 0.0])
for st_ in steps:
    k_ = "/".join(sorted([re.sub(r"\d+$", "", st_["a"]), re.sub(r"\d+$", "", st_["b"])]))
    pair[k_][0] += 1; pair[k_][1] = max(pair[k_][1], st_["dz"])
# separate: steps within road surface (1ROAD-1ROAD) or road-adjacent (dist<=edge+1)
road_steps = [x for x in steps if x["a"].startswith("1ROAD") or x["b"].startswith("1ROAD") or x["a"].startswith("1KERB") or x["b"].startswith("1KERB")]
chk("3a_vertical_steps", len(steps) == 0,
    n_boundary_vertices_checked=int(len(bv)), n_with_overlap_surface=int(has.sum()), n_steps_over_2cm=int(len(steps)),
    max_step=float(d[has].max()) if has.any() else 0, coincident_vertex_max_dz=float(cstep.max()),
    n_coincident_over_2cm=int(np.sum(cstep > 0.02)),
    by_mesh_pair={k: dict(n=v[0], max_dz=round(v[1], 3)) for k, v in sorted(pair.items(), key=lambda kv: -kv[1][0])},
    road_or_kerb_steps_n=len(road_steps), worst=steps[:25], worst_road_or_kerb=road_steps[:25], road_vs_terrain=[x for x in steps if "1ROAD" in x["a"]+x["b"]][:30])

# 3b holes: boundary edges of the union not covered by another drivable surface and not on outer boundary
Pw = P.copy()
kq = np.round(P / np.array([0.005, 0.005, 0.02])).astype(np.int64)
_, winv = np.unique(np.round(P[:, :2] / 0.005).astype(np.int64), axis=0, return_inverse=True); winv = winv.ravel()
Tw = winv[drv.T]
Tw = Tw[(Tw[:, 0] != Tw[:, 1]) & (Tw[:, 1] != Tw[:, 2]) & (Tw[:, 0] != Tw[:, 2])]
be, nonman = boundary_edges(Tw)
rep = np.zeros(winv.max() + 1, int); rep[winv] = np.arange(len(P))
mid = (P[rep[be[:, 0]], :2] + P[rep[be[:, 1]], :2]) / 2
# probe slightly to both sides of the edge; a hole means: both sides uncovered? no - outer boundary has one side uncovered.
ev = P[rep[be[:, 1]], :2] - P[rep[be[:, 0]], :2]; el = np.hypot(*ev.T) + 1e-12
nrm2 = np.stack([-ev[:, 1], ev[:, 0]], 1) / el[:, None]
zA, _ = drv.sample(mid + 0.05 * nrm2); zB, _ = drv.sample(mid - 0.05 * nrm2)
open_side = np.isnan(zA) | np.isnan(zB)
s_e, d_e = station_of(mid)
hw = np.interp(s_e, S, np.maximum(EL, ER))
# inside corridor: within road half width + 3 m (road + shoulders) and not beyond built ends
inner = open_side & (d_e < hw + 3.0) & (s_e > 2) & (s_e < BUILD_END - 2)
# also any open edge within 30 m of road (terrain hole) - report separately
terr_near = open_side & (d_e < 30) & (s_e > 2) & (s_e < BUILD_END - 2)
hole_list = sorted(set(int(round(x)) for x in s_e[inner]))
chk("3b_holes", int(inner.sum()) == 0, n_boundary_edges_union=int(len(be)), n_open_boundary_edges=int(open_side.sum()),
    n_open_edges_within_road_plus3m=int(inner.sum()), stations_road_holes=ranges(np.searchsorted(S, hole_list))[:80],
    n_open_edges_within_30m=int(terr_near.sum()),
    stations_within_30m=ranges(np.searchsorted(S, sorted(set(int(round(x)) for x in s_e[terr_near])))) [:80],
    n_nonmanifold_edges=int(len(nonman)),
    note="T-junction gaps where another surface covers both sides are not counted (probe 5 cm either side)")
# 3c normals
nd = []
for it in surf["ROAD"] + surf["GRASS"] + surf["KERB"]:
    fn, nm, p, t, n = it
    A, B, C = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    fz = np.cross(B - A, C - A)[:, 2]  # local up component of geometric normal, CCW
    # AC/D3D: kn5 uses clockwise front faces in AC space; mirror (z=-north) flips handedness -> CCW in local
    vn = n[:, 1]
    nd.append((nm, int(np.sum(fz < 0)), int(np.sum(fz > 0)), int(np.sum(vn < 0)), len(t), float(vn.min())))
geo_down = sum(x[1] for x in nd); geo_up = sum(x[2] for x in nd); vdown = sum(x[3] for x in nd)
# winding consistency: majority orientation should be uniform
minority = min(geo_down, geo_up)
worst_n = sorted(nd, key=lambda x: -min(x[1], x[2]))[:10]
chk("3c_normals", vdown == 0 and minority == 0, vertex_normals_with_negative_up=vdown,
    triangles_winding_a=geo_down, triangles_winding_b=geo_up, minority_winding_triangles=minority,
    meshes_with_mixed_winding=[dict(mesh=x[0], a=x[1], b=x[2]) for x in worst_n if min(x[1], x[2]) > 0],
    min_vertex_normal_up=float(min(x[5] for x in nd)))
# 3d grade / crossfall spikes from the road mesh itself
ss = S[sel]; zz = zr.copy()
gr = np.gradient(zz, ss)
dgr = np.abs(np.diff(gr))
# crossfall from mesh: sample at +-2 m
zl, _ = road.sample(np.stack([CX[sel] + 2 * nlx[sel], CY[sel] + 2 * nly[sel]], 1))
zrr, _ = road.sample(np.stack([CX[sel] - 2 * nlx[sel], CY[sel] - 2 * nly[sel]], 1))
xf = (zl - zrr) / 4.0
dxf = np.abs(np.diff(xf))
sp_g = sel[1:][dgr > 0.01]; sp_x = sel[1:][dxf > 0.01]
# second difference of z (bumps) in mm
d2 = np.abs(zz[2:] - 2 * zz[1:-1] + zz[:-2])
chk("3d_grade_crossfall", len(sp_g) == 0 and len(sp_x) == 0 and np.nanmax(np.abs(gr)) < 0.25,
    max_grade=float(np.nanmax(np.abs(gr))), max_grade_at_s=float(ss[np.nanargmax(np.abs(gr))]),
    grade_p99=float(np.nanpercentile(np.abs(gr), 99)),
    max_grade_change_per_m=float(np.nanmax(dgr)), at_s=float(ss[1:][np.nanargmax(dgr)]),
    n_grade_change_over_1pct_per_m=int(len(sp_g)), stations_grade_spike=ranges(sp_g)[:60],
    max_crossfall=float(np.nanmax(np.abs(xf))), max_crossfall_at_s=float(ss[np.nanargmax(np.abs(xf))]),
    max_crossfall_change_per_m=float(np.nanmax(dxf)), crossfall_change_at_s=float(ss[1:][np.nanargmax(dxf)]),
    n_crossfall_change_over_1pct_per_m=int(len(sp_x)), stations_crossfall_spike=ranges(sp_x)[:60],
    max_z_second_diff_mm=float(np.nanmax(d2) * 1000), z2nd_at_s=float(ss[1:-1][np.nanargmax(d2)]))

# ---------- 4. package ----------
pk = []
uij = {}
for lay, files in layouts.items():
    for f in files:
        if not os.path.exists(os.path.join(TRK, f)): pk.append(f"{lay}: missing {f}")
    for f in ("data/surfaces.ini", "data/cameras.ini", "data/lighting.ini", "data/map.ini", "map.png"):
        if not os.path.exists(os.path.join(TRK, lay, f)): pk.append(f"{lay}: missing {f}")
    for f in ("ui_track.json", "outline.png"):
        if not os.path.exists(os.path.join(TRK, "ui", lay, f)): pk.append(f"{lay}: missing ui/{f}")
    if not os.path.exists(os.path.join(TRK, "ui", lay, "preview.png")): pk.append(f"NOTE {lay}: ui/preview.png missing")
    try:
        j = json.load(open(os.path.join(TRK, "ui", lay, "ui_track.json"), encoding="utf-8"))
        for kk in ("name", "description", "length"):
            if kk not in j: pk.append(f"{lay}: ui_track.json missing {kk}")
        uij[lay] = dict(name=j.get("name"), length=j.get("length"))
        L = float(re.sub(r"[^\d.]", "", str(j.get("length", "0"))) or 0)
        if abs(L - (FINISH - START_GATE)) > 30: pk.append(f"{lay}: length {L} vs gate-to-gate {FINISH-START_GATE}")
        for a in ("OpenStreetMap", "USGS"):
            if a not in j.get("description", ""): pk.append(f"{lay}: attribution {a} missing")
    except Exception as e: pk.append(f"{lay}: ui_track.json invalid ({e})")
    sp = os.path.join(TRK, lay, "data", "surfaces.ini")
    if os.path.exists(sp):
        keys = set(re.findall(r"^\s*KEY\s*=\s*(\S+)", open(sp).read(), re.M))
        lay_used = set()
        for f in files:
            for m in kn5.get(f, {}).get("meshes", []):
                mm = re.match(r"^\d+([A-Z]+)", m["name"])
                if mm: lay_used.add(mm.group(1))
        for u in lay_used - keys: pk.append(f"{lay}: surface key {u} used but not in surfaces.ini")
    # map.ini sanity
    mp = os.path.join(TRK, lay, "data", "map.ini")
    if os.path.exists(mp) and os.path.exists(os.path.join(TRK, lay, "map.png")):
        from PIL import Image
        im = Image.open(os.path.join(TRK, lay, "map.png"))
        txt = open(mp).read()
        wv = re.search(r"WIDTH\s*=\s*([\d.]+)", txt); hv = re.search(r"HEIGHT\s*=\s*([\d.]+)", txt)
        if wv and hv and (abs(float(wv.group(1)) - im.size[0]) > 1 or abs(float(hv.group(1)) - im.size[1]) > 1):
            pk.append(f"{lay}: map.ini WIDTH/HEIGHT {wv.group(1)}x{hv.group(1)} != map.png {im.size}")
# timing objects
dm = {}
for fn, k in kn5.items():
    for nm, m in k["dummies"]:
        if nm.startswith("AC_"): dm[nm] = (fn, np.array(m).reshape(4, 4))
need = ["AC_START_0", "AC_PIT_0", "AC_HOTLAP_START_0", "AC_AB_START_L", "AC_AB_START_R", "AC_AB_FINISH_L", "AC_AB_FINISH_R"]
tim = {}
ctree = cKDTree(np.stack([CX, CY], 1))
for n_ in need:
    if n_ not in dm: pk.append(f"timing object {n_} missing"); continue
    M = dm[n_][1]; pos = M[3, :3]; loc = np.array([pos[0], -pos[2], pos[1]])
    dd, ii = ctree.query(loc[:2])
    zs, _ = road.sample(loc[None, :2])
    t_ac = np.array([tx[ii], 0, -ty[ii]])
    tim[n_] = dict(s=float(S[ii]), lateral=round(float(dd), 2), height_above_road=round(float(loc[2] - zs[0]), 2) if not np.isnan(zs[0]) else None,
                   axisX_dot_tangent=round(float(M[0, :3] @ t_ac), 2), axisZ_dot_tangent=round(float(M[2, :3] @ t_ac), 2))
for side in ("START", "FINISH"):
    L_, R_ = f"AC_AB_{side}_L", f"AC_AB_{side}_R"
    if L_ in tim and R_ in tim:
        pl = dm[L_][1][3, :3]; pr = dm[R_][1][3, :3]
        a = np.array([pl[0], -pl[2]]); b = np.array([pr[0], -pr[2]])
        ii = ctree.query((a + b) / 2)[1]
        sl = (a - np.array([CX[ii], CY[ii]])) @ np.array([nlx[ii], nly[ii]]); sr = (b - np.array([CX[ii], CY[ii]])) @ np.array([nlx[ii], nly[ii]])
        tim[f"{side}_gate"] = dict(s=float(S[ii]), left_offset=round(float(sl), 2), right_offset=round(float(sr), 2), width=round(float(np.hypot(*(a - b))), 2),
                                   road_left=float(EL[ii]), road_right=float(ER[ii]))
        if not (sl > EL[ii] - 0.5 and sr < -ER[ii] + 0.5): pk.append(f"{side} gate does not straddle the road (L {sl:.1f}, R {sr:.1f}, road +{EL[ii]:.1f}/-{ER[ii]:.1f})")
        exp = START_GATE if side == "START" else FINISH
        if abs(S[ii] - exp) > 5: pk.append(f"{side} gate at s={S[ii]} expected {exp}")
for n_ in ("AC_START_0", "AC_HOTLAP_START_0", "AC_PIT_0"):
    if n_ in tim and "START_gate" in tim:
        if tim[n_]["s"] >= tim["START_gate"]["s"]: pk.append(f"{n_} at s={tim[n_]['s']} not before start gate")
        if tim[n_]["lateral"] > 6: pk.append(f"{n_} {tim[n_]['lateral']} m off centerline")
        h = tim[n_]["height_above_road"]
        if h is None or h < -0.1 or h > 2: pk.append(f"{n_} height above road {h}")
        if abs(tim[n_]["axisZ_dot_tangent"]) < 0.9: pk.append(f"{n_}: local Z axis not aligned with road tangent (dot={tim[n_]['axisZ_dot_tangent']}, X dot={tim[n_]['axisX_dot_tangent']}) - check facing")
hard = [p for p in pk if not p.startswith("NOTE")]
chk("4_package", not hard, problems=pk, ui=uij, timing=tim, layouts={k: v for k, v in layouts.items()})
R["runtime_s"] = round(time.time() - T0, 1)

# ---------- write ----------
os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
    return str(o)
json.dump(R, open(os.path.join(ROOT, "reports/qa_phase2.json"), "w"), indent=1, default=conv)
L = ["# Phase 2 QA report (independent)", "", f"Generated by scripts/p2/23_qa.py in {R['runtime_s']} s.", "",
     "| Check | Status |", "|---|---|"]
for k, v in R["checks"].items(): L.append(f"| {k} | **{v['status']}** |")
for k, v in R["checks"].items():
    L += ["", f"## {k}: {v['status']}", "", "```", json.dumps({a: b for a, b in v.items() if a != "status"}, indent=1, default=conv)[:6000], "```"]
open(os.path.join(ROOT, "reports/qa_phase2.md"), "w").write("\n".join(L) + "\n")
for k, v in R["checks"].items(): print(k, v["status"])
