#!/usr/bin/env python3
"""FFB-safety hard gate (dev6).  Reads the BUILT kn5 files back and fails (exit 1) on the patterns that make AC
collisions violent or leaky (docs/AC_CONVENTIONS.md "Collision convention"):

 physics (digit-prefixed) meshes
   * physical KEY not in surfaces.ini; NaN; non-unit normals; degenerate (< 1e-4 m2) or sliver triangles
 1WALL
   * any collider on trees (1WALL_tree*) or a small prop (component footprint < 1.5 m2 and not long)
   * shells that are not watertight, thinner than MIN_THICK or lower than MIN_H, edges shorter than 5 cm
   * road-facing creases > WALL_CREASE deg within NEAR m of the drivable edge
   * overlapping colliders (road-frame boxes intersecting by > 5 cm)
   * > MAX_WALL_TRIS_PER_KM physics wall triangles per km
 1ROAD / 1KERB / 1RUNOFF : adjacent-triangle creases > ROAD_CREASE / KERB_CREASE deg
 1GRASS terrain          : creases > TERR_CREASE deg within TERR_R m of the drivable edge
 visual                  : NaN, non-unit normals, tangents not perpendicular to normals
 textures                : must be DDS (DXT1/DXT5), power of two, mipmapped, <= 4096
 shadows                 : terrain + backdrop must cast shadows
Writes reports/qa_ffb_safety.json.
"""
import json, re, sys, pathlib, collections
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import ROOT, BUILD
from kn5 import read_kn5
import dds
import collision_conv as CC

TRACK = BUILD / "palomares_road"
MIN_THICK, MIN_H, NEAR = 0.5, 1.2, 8.0
WALL_CREASE, ROAD_CREASE, KERB_CREASE, TERR_CREASE, TERR_R = 30.0, 8.0, 25.0, 45.0, 3.0
MAX_WALL_TRIS_PER_KM = 6000
fails, warns, stats = [], [], collections.defaultdict(dict)
fail = lambda s: fails.append(s)

keys = set()
for f in TRACK.glob("*/data/surfaces.ini"):
    keys |= set(re.findall(r"^KEY=(\w+)", f.read_text(), re.M))
keys |= {"WALL", "ROAD", "GRASS", "KERB", "SAND"}


def enu(p):   # AC (x, up, -north) -> (east, north, up)
    return np.c_[p[:, 0], -p[:, 2], p[:, 1]]


def tri_geo(P, T):
    e1 = P[T[:, 1]] - P[T[:, 0]]; e2 = P[T[:, 2]] - P[T[:, 0]]; e3 = P[T[:, 2]] - P[T[:, 1]]
    fn = np.cross(e1, e2); a = np.linalg.norm(fn, axis=1)
    L = np.maximum.reduce([np.linalg.norm(e, axis=1) for e in (e1, e2, e3)])
    Lmin = np.minimum.reduce([np.linalg.norm(e, axis=1) for e in (e1, e2, e3)])
    return fn / (a[:, None] + 1e-12), a / 2, L, Lmin


def creases(P, T, fn):
    """angle (deg) between adjacent triangles sharing a welded edge, + the pair indices"""
    _, inv = np.unique(np.round(P, 3), axis=0, return_inverse=True); W = inv.ravel()[T]
    te = np.r_[np.sort(W[:, [0, 1]], 1), np.sort(W[:, [1, 2]], 1), np.sort(W[:, [2, 0]], 1)]
    tid = np.tile(np.arange(len(T)), 3)
    o = np.lexsort((te[:, 1], te[:, 0])); te, tid = te[o], tid[o]
    same = (te[1:] == te[:-1]).all(1); a, b = tid[:-1][same], tid[1:][same]
    return np.degrees(np.arccos(np.clip((fn[a] * fn[b]).sum(1), -1, 1))), a, b, W


K = {f.name: read_kn5(f) for f in sorted(TRACK.glob("*.kn5"))}
# drivable edge = 1ROAD + shoulder vertices
road = np.concatenate([enu(m["pos"])[:, :2] for m in K["palomares_road.kn5"]["meshes"] if m["name"][:1].isdigit()])
road_kd = cKDTree(road[::3])
from geom import load_centerline
F = CC.Frame(load_centerline())

wall_boxes = []
n_wall_tris = 0
for fn_, k in K.items():
    for name, b in k["textures"].items():
        i = dds.info(b)
        if i is None:
            fail(f"{fn_}: texture {name} is not DDS")
            continue
        fc, w, h, mips = i
        if fc not in ("DXT1", "DXT5") or w & (w - 1) or h & (h - 1) or max(w, h) > 4096 or mips < 2:
            fail(f"{fn_}: texture {name} {fc} {w}x{h} mips={mips}")
    for m in k["meshes"]:
        nm = m["name"]; P = enu(m["pos"].astype(float)); T = m["idx"].reshape(-1, 3); N = m["nrm"]
        if not (np.isfinite(P).all() and np.isfinite(N).all() and np.isfinite(m["tan"]).all()):
            fail(f"{fn_}:{nm} NaN/inf in vertex data"); continue
        if (np.abs(np.linalg.norm(N, axis=1) - 1) > 0.02).any():
            fail(f"{fn_}:{nm} non-unit normals")
        if (np.abs((N * m["tan"]).sum(1)) > 0.05).any():
            fail(f"{fn_}:{nm} tangent not perpendicular to normal")
        phys = nm[:1].isdigit()
        if not phys:
            if nm.startswith(("1", "2")):
                pass
            if fn_ == "palomares_terrain.kn5" and not m["cast_shadows"]:
                fail(f"{fn_}:{nm} terrain/backdrop does not cast shadows (light leaks through hills)")
            continue
        key = re.match(r"^\d+([A-Z]+)", nm)
        if not key or key.group(1) not in keys:
            fail(f"{fn_}:{nm} physical name with unknown KEY"); continue
        key = key.group(1)
        fnr, area, L, Lmin = tri_geo(P, T)
        if fn_ == "palomares_terrain.kn5" and not m["cast_shadows"]:
            fail(f"{fn_}:{nm} terrain does not cast shadows")
        if (area < 1e-4).any():
            fail(f"{fn_}:{nm} {(area < 1e-4).sum()} degenerate triangles")
        sl = (2 * area / np.maximum(L, 1e-9)) < 0.005 * L
        if key in ("WALL", "ROAD", "KERB", "RUNOFF") and sl.sum():
            (fail if key == "WALL" else warns.append)(f"{fn_}:{nm} {sl.sum()} sliver triangles")
        if key == "WALL":
            if nm.startswith("1WALL_tree"):
                fail(f"{fn_}:{nm} collider on trees"); continue
            n_wall_tris += len(T)
            if m["renderable"] or m["cast_shadows"]:
                fail(f"{fn_}:{nm} collider renderable/casting shadows")
            if (Lmin < 0.05).any():
                fail(f"{fn_}:{nm} {(Lmin < 0.05).sum()} wall triangles with an edge < 5 cm")
            ang, a, b, W = creases(P, T, fnr)
            # watertight: every welded edge used exactly twice
            e = np.sort(np.r_[W[:, [0, 1]], W[:, [1, 2]], W[:, [2, 0]]], 1)
            _, cnt = np.unique(e, axis=0, return_counts=True)
            if (cnt != 2).any():
                fail(f"{fn_}:{nm} not watertight ({(cnt != 2).sum()} open/non-manifold edges)")
            # road-face creases near the drivable edge
            cen = P[T].mean(1); dr, _ = road_kd.query(cen[:, :2])
            s, d, j = F.sd(cen[:, :2])
            tow = -np.sign(d)[:, None] * np.c_[F.nx[j], F.ny[j]]
            face = ((fnr[:, :2] * tow).sum(1) > 0.5) & (np.abs(fnr[:, 2]) < 0.3)
            pr = face[a] & face[b] & ((dr[a] < NEAR) | (dr[b] < NEAR))
            mx = float(ang[pr].max()) if pr.any() else 0.0
            stats["wall_face_crease_max_deg"][nm] = round(mx, 1)
            if mx > WALL_CREASE:
                fail(f"{fn_}:{nm} road-face crease {mx:.0f} deg within {NEAR} m of the edge")
            for comp in CC.components(P, T.copy()):
                Pc = P[comp]; sc, dc, _ = F.sd(Pc[:, :2])
                ext = np.ptp(Pc[:, :2], axis=0); hgt = np.ptp(Pc[:, 2])
                long_ = np.ptp(sc) >= CC.LIN_MIN_LEN
                u, sv, vt = np.linalg.svd(Pc[:, :2] - Pc[:, :2].mean(0), full_matrices=False)
                if not long_:
                    thick = float(np.ptp((Pc[:, :2] - Pc[:, :2].mean(0)) @ vt[1]))
                else:   # depth across the road, measured in windows along the run (median over the run)
                    ws = [np.ptp(np.abs(dc[w])) for q in np.linspace(sc.min(), sc.max(), 12) for w in [np.abs(sc - q) < 1.5] if w.sum() >= 2]
                    thick = float(np.median(ws)) if ws else 0.0
                if thick < MIN_THICK:
                    fail(f"{fn_}:{nm} wall thinner than {MIN_THICK} m ({thick:.2f}) at s={sc.mean():.0f}")
                if hgt < MIN_H:
                    fail(f"{fn_}:{nm} wall lower than {MIN_H} m at s={sc.mean():.0f}")
                if not long_ and ext[0] * ext[1] < 1.5 and not nm.startswith(("1WALL_start", "1WALL_end")):
                    fail(f"{fn_}:{nm} small-prop collider ({ext[0]:.1f}x{ext[1]:.1f} m) at s={sc.mean():.0f}")
                wall_boxes.append((fn_ + ":" + nm, sc.min(), sc.max(), np.sign(np.median(dc)), np.abs(dc).min(), np.abs(dc).max(), Pc[:, 2].min(), Pc[:, 2].max()))
        elif key in ("ROAD", "KERB", "RUNOFF", "GRASS"):
            lim = {"ROAD": ROAD_CREASE, "KERB": KERB_CREASE, "RUNOFF": KERB_CREASE, "GRASS": TERR_CREASE}[key]
            ang, a, b, _ = creases(P, T, fnr)
            if key in ("GRASS", "ROAD"):
                # folds a wheel rolls over: both faces < 40 deg slope, or a fold-over (> 100 deg); cut-bank toes
                # (flat -> steep face) are real terrain and reported separately
                c40 = np.cos(np.radians(40))
                flat = (np.abs(fnr[a, 2]) > c40) & (np.abs(fnr[b, 2]) > c40)
                toe = ang[~flat & (ang <= 100)]
                stats[f"{key.lower()}_banktoe_crease_max_deg"][nm] = round(float(toe.max()), 1) if len(toe) else 0.0
                ang = ang[flat | (ang > 100)]
                if key == "GRASS":
                    dr, _ = road_kd.query(P[T].mean(1)[:, :2])
                    ang = ang[((dr[a] < TERR_R) & (dr[b] < TERR_R))[flat | (np.degrees(np.arccos(np.clip((fnr[a] * fnr[b]).sum(1), -1, 1))) > 100)]]
            mx = float(ang.max()) if len(ang) else 0.0
            stats[f"{key.lower()}_crease_max_deg"][nm] = round(mx, 1)
            if mx > lim:
                (fail if key != "GRASS" else fail)(f"{fn_}:{nm} crease {mx:.0f} deg > {lim}")

# overlaps between collider components (road frame boxes), same side
wb = sorted(wall_boxes, key=lambda w: w[1])
nov = 0
for i, A in enumerate(wb):
    for B in wb[i + 1:]:
        if B[1] > A[2]:
            break
        if A[3] != B[3] or "start" in A[0] + B[0] or "end" in A[0] + B[0]:
            continue
        ov_s = min(A[2], B[2]) - max(A[1], B[1]); ov_d = min(A[5], B[5]) - max(A[4], B[4]); ov_z = min(A[7], B[7]) - max(A[6], B[6])
        if ov_s > 0.05 and ov_d > 0.05 and ov_z > 0.05:
            nov += 1
            if nov <= 20:
                fail(f"overlapping colliders {A[0]} / {B[0]} at s={max(A[1], B[1]):.0f}")
if nov > 20:
    fail(f"... {nov} overlapping collider pairs in total")
# materials / shader permutations / textures per layout (NVIDIA DXCache bloat: dev5 had 189 terrain materials,
# 71 terrain PNGs; a typical community track has tens of materials)
MAX_MATS, MAX_COMBOS, MAX_TEX, MAX_TEX_MB = 80, 80, 64, 400
lay_stats = {}
for ini in sorted(TRACK.glob("models_*.ini")):
    lay = ini.stem[len("models_"):]
    files = [l.split("=", 1)[1].strip() for l in ini.read_text().splitlines() if l.startswith("FILE=")]
    mats, combos, tex, tb, shaders = set(), set(), set(), 0, set()
    for f in files:
        k = K.get(f)
        if k is None:
            fail(f"{lay}: {f} missing"); continue
        for mt in k["materials"]:
            mats.add(mt["name"]); shaders.add(mt["shader"])
            combos.add((mt["shader"], mt["alpha_tested"], mt["blend"], tuple(sorted(mt["textures"].items()))))
        for n, b in k["textures"].items():
            if n not in tex:
                tex.add(n); tb += len(b)
    ext = TRACK / "extension" / "ext_config.ini"
    nrep = len(re.findall(r"^\[SHADER_REPLACEMENT", ext.read_text(), re.M)) if ext.exists() else 0
    lay_stats[lay] = dict(materials=len(mats), shader_texture_combos=len(combos), shaders=sorted(shaders), textures=len(tex),
                          texture_mb=round(tb / 1e6, 1), csp_shader_replacements=nrep)
    if len(mats) > MAX_MATS or len(combos) > MAX_COMBOS or len(tex) > MAX_TEX or tb / 1e6 > MAX_TEX_MB:
        fail(f"{lay}: {len(mats)} materials / {len(combos)} shader-texture combos / {len(tex)} textures / {tb / 1e6:.0f} MB "
             f"(limits {MAX_MATS}/{MAX_COMBOS}/{MAX_TEX}/{MAX_TEX_MB} MB)")
stats["layouts"] = lay_stats
for lay, v in lay_stats.items():
    print(f"  {lay:18s} materials {v['materials']:3d}  combos {v['shader_texture_combos']:3d}  textures {v['textures']:3d} "
          f"({v['texture_mb']} MB)  shaders {','.join(v['shaders'])}  CSP replacements {v['csp_shader_replacements']}")
km = float(F.s[-1]) / 1000
stats["wall_tris_total"] = n_wall_tris; stats["wall_tris_per_km"] = round(n_wall_tris / km)
stats["wall_components"] = len(wall_boxes)
if n_wall_tris / km > MAX_WALL_TRIS_PER_KM:
    fail(f"{n_wall_tris / km:.0f} wall tris per km > {MAX_WALL_TRIS_PER_KM} (physics meshes too high-poly)")

out = dict(ok=not fails, fails=fails, warnings=warns[:200], n_warnings=len(warns),
           stats={k: (dict(sorted(v.items(), key=lambda kv: -kv[1])[:10]) if isinstance(v, dict) and k != "layouts" else v) for k, v in stats.items()})
(ROOT / "reports").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "reports" / "qa_ffb_safety.json", "w"), indent=1)
for s in fails[:60]:
    print("FAIL", s)
print(f"qa_ffb_safety: {len(fails)} fails, {len(warns)} warnings; wall tris {n_wall_tris} ({n_wall_tris / km:.0f}/km), "
      f"{len(wall_boxes)} wall components -> reports/qa_ffb_safety.json")
sys.exit(1 if fails else 0)
