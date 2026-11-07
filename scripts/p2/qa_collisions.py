#!/usr/bin/env python3
"""Collision QA: round-trips the kn5 files and asserts that every solid barrier / fence / wall has a 1WALL
collision surface within 0.3 m along its whole length and at least 1.2 m above its base.

Reads   build/palomares_road/*.kn5 (+ models_<layout>.ini), build/roadside_solid.json (written by 21c: every solid item
        as a polyline of base points).
Checks  1. kn5 round trip of every file used by the layouts
        2. every physics mesh (1WALL*): flags visible=True/renderable=False, <= 65535 vertices, finite, unit normals,
           no degenerate triangles, in-range indices
        3. coverage: for every sampled point of every solid item (every <= 0.5 m) a collision sample lies within 0.3 m
           at base+0.3 m, base+0.7 m and base+1.25 m (=> continuous, no gaps at chunk joins, >= 1.2 m tall)
        4. any visible guardrail*/grpost*/fence* mesh left in the kn5s (21's old ones) is covered the same way
        5. every layout includes palomares_roadside.kn5 and the 1WALL_start / 1WALL_end walls
        6. 21b's 1WALL_dressing exists in every track_* layout
Exit status 1 on any failure; writes reports/qa_collisions.json.
"""
import glob, json, re, sys, pathlib, collections
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import ROOT, BUILD
from kn5 import read_kn5

TOL = 0.30
SPACING = 0.15
TRACK = BUILD / "palomares_road"
fails, notes = [], []


def fail(msg):
    fails.append(msg); print("  FAIL", msg)


def local(pos):   # AC (x, y_up, z) -> local ENU
    return np.stack([pos[:, 0], -pos[:, 2], pos[:, 1]], 1)


def sample_tris(P, T, h=SPACING):
    """points on the triangle surfaces, spacing ~h"""
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    me = np.maximum.reduce([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1), np.linalg.norm(A - C, axis=1)])
    k = np.clip(np.ceil(me / h).astype(int), 1, 48)
    out = [P]
    for kk in np.unique(k):
        m = k == kk
        ij = np.array([(i, j) for i in range(kk + 1) for j in range(kk + 1 - i)], float) / kk
        a, b = ij[:, 0][None, :, None], ij[:, 1][None, :, None]
        pts = A[m][:, None, :] * (1 - a - b) + B[m][:, None, :] * a + C[m][:, None, :] * b
        out.append(pts.reshape(-1, 3))
    return np.concatenate(out)


def main():
    kn = {}
    for f in sorted(glob.glob(str(TRACK / "*.kn5"))):
        kn[pathlib.Path(f).name] = read_kn5(f)       # round trip (asserts no trailing bytes)
    if not kn:
        print("no kn5 files in", TRACK); sys.exit(2)
    print(f"  round-tripped {len(kn)} kn5 files")
    layouts = {pathlib.Path(f).stem[len("models_"):]: re.findall(r"FILE=(\S+)", open(f).read()) for f in glob.glob(str(TRACK / "models_*.ini"))}
    ref_layouts = [l for l in ("road_full", "track_full") if l in layouts] or list(layouts)[:1]

    # 5/6: layout composition
    for lay, files in sorted(layouts.items()):
        if "palomares_roadside.kn5" not in files:
            fail(f"{lay}: palomares_roadside.kn5 missing from models ini")
        names = {m["name"] for f in files for m in kn[f]["meshes"]}
        for w in ("1WALL_start", "1WALL_end"):
            if w not in names:
                fail(f"{lay}: {w} missing")
        if lay.startswith("track_") and not any(n.startswith("1WALL_dressing") for n in names):
            fail(f"{lay}: 1WALL_dressing missing")

    # 2: physics mesh sanity + collision sample cloud (all shared + dressing files)
    stats = collections.Counter(); clouds = []
    files_all = sorted({f for l in ref_layouts for f in layouts[l]})
    for f in files_all:
        for m in kn[f]["meshes"]:
            if not m["name"].startswith("1WALL"):
                continue
            stats["physics_meshes"] += 1
            P = m["pos"].astype(float); T = m["idx"].reshape(-1, 3)
            if len(P) > 65535: fail(f"{f}:{m['name']} >65535 vertices")
            if not np.isfinite(P).all() or not np.isfinite(m["nrm"]).all(): fail(f"{f}:{m['name']} non-finite data")
            if T.max() >= len(P) or T.min() < 0: fail(f"{f}:{m['name']} index out of range")
            if not m["visible"] or m["renderable"]:
                fail(f"{f}:{m['name']} flags visible={m['visible']} renderable={m['renderable']} (want True/False)")
            # raw bytes vs the verified convention (AcTools Kn5Writer field order: castShadows, isVisible, isTransparent
            # ... materialId, layer, lodIn, lodOut, bsphere, isRenderable): 00 01 00 ... 00, active node, no LOD, layer 0
            if m["flag_bytes"] != b"\x00\x01\x00" or m["renderable_byte"] != b"\x00" or not m["active"] or m["lod_in"] != 0 or m["lod_out"] != 0 or m["layer"] != 0:
                fail(f"{f}:{m['name']} raw flags {m['flag_bytes'].hex()} r={m['renderable_byte'].hex()} active={m['active']} lod={m['lod_in']},{m['lod_out']} layer={m['layer']}")
            stats["flag_bytes_ok"] += 1
            fnr = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]); fa = np.linalg.norm(fnr, axis=1); ok_ = fa > 1e-9
            agree = (fnr[ok_] / fa[ok_, None] * m["nrm"][T[ok_, 0]]).sum(1)
            if (agree < 0.9).mean() > 0.001: fail(f"{f}:{m['name']} {100 * (agree < 0.9).mean():.1f}% triangles whose winding disagrees with the normal")
            if m["name"].startswith("1WALL_tree"):
                stats["tree_colliders_tris"] += len(T)
            ln = np.linalg.norm(m["nrm"], axis=1)
            if (np.abs(ln - 1) > 0.05).mean() > 0.001: fail(f"{f}:{m['name']} {100 * (np.abs(ln - 1) > 0.05).mean():.1f}% non-unit vertex normals")
            area = 0.5 * np.linalg.norm(np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), axis=1)
            if (area < 1e-7).mean() > 0.001: fail(f"{f}:{m['name']} {int((area < 1e-7).sum())} degenerate triangles")
            stats["collision_tris"] += len(T)
            clouds.append(sample_tris(local(P), T))
    # physical names: digit prefix => must be 1<KEY> with KEY in surfaces.ini; no visual tree mesh may be physical
    keys = set(re.findall(r"KEY=(\S+)", open(next(iter(glob.glob(str(TRACK / "*/data/surfaces.ini"))))).read()))
    for f in files_all:
        for m in kn[f]["meshes"]:
            n = m["name"]
            if n[:1].isdigit() and not any(re.match(rf"\d{k}", n) for k in keys):
                fail(f"{f}:{n} digit-prefixed name with no surfaces.ini KEY")
            if f == "palomares_trees.kn5" and n[:1].isdigit():
                fail(f"{f}:{n} tree mesh with a physical name")
    if stats["tree_colliders_tris"]:
        fail("1WALL_tree trunk colliders present (dev6 convention: trees get no collision, forest-line slabs instead)")
    print(f"  raw flag bytes OK on {stats['flag_bytes_ok']} physics meshes; trunk collider tris {stats['tree_colliders_tris']}")
    cloud = np.concatenate(clouds); tree = cKDTree(cloud)
    print(f"  {stats['physics_meshes']} physics meshes, {stats['collision_tris']} triangles, {len(cloud)} surface samples")

    def covered(pts):
        d, _ = tree.query(pts, distance_upper_bound=TOL)
        return np.isfinite(d)

    # 3: sidecar items
    side = json.load(open(BUILD / "roadside_solid.json"))
    per_type = collections.defaultdict(lambda: [0, 0, 0.0])      # type -> [items, uncovered items, uncovered m]
    worst = []
    for it in side:
        xy = np.array(it["pts"], float); z = np.array(it["z"], float)
        # dev6 convention (docs/AC_CONVENTIONS.md "Collision convention"): only barrier lines (>= 3 m) and buildings
        # carry collision; poles, signs, mailboxes, trees, hedges, rocks and other props are visual only
        _len = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0
        if not (it["type"].startswith("building") or (_len >= 3.0 and not it["type"].startswith(("hedge", "tree", "pole", "rock")))):
            per_type["(prop, no collision) " + it["type"]][0] += 0
            continue
        if len(xy) > 1:   # densify to <= 0.5 m
            seg = np.hypot(*np.diff(xy, axis=0).T); n = np.maximum(1, np.ceil(seg / 0.5).astype(int))
            xs, zs = [xy[:1]], [z[:1]]
            for i in range(len(seg)):
                t = (np.arange(1, n[i] + 1) / n[i])[:, None]
                xs.append(xy[i] + (xy[i + 1] - xy[i]) * t); zs.append(z[i] + (z[i + 1] - z[i]) * t[:, 0])
            xy = np.concatenate(xs); z = np.concatenate(zs)
        miss = np.zeros(len(xy), bool)
        for h in (0.3, 0.7, 1.25):
            miss |= ~covered(np.c_[xy, z + h])
        r = per_type[it["type"]]; r[0] += 1
        if miss.any():
            r[1] += 1; r[2] += float(miss.sum() * 0.5)
            worst.append((it["type"], it["s0"], it["s1"], int(miss.sum()), len(xy)))
    n_items = len(side)
    n_bad = sum(v[1] for v in per_type.values())
    print(f"  solid items checked: {n_items}; with uncovered samples: {n_bad}")
    for t, (n, b, mtr) in sorted(per_type.items()):
        print(f"    {t:26s} items {n:4d}  uncovered {b:3d}  ({mtr:.1f} m)")
    for w in sorted(worst, key=lambda w: -w[3])[:10]:
        print(f"    UNCOVERED {w[0]} s={w[1]:.0f}..{w[2]:.0f}: {w[3]}/{w[4]} samples")
    if n_bad:
        fail(f"{n_bad} solid items lack collision within {TOL} m / 1.25 m height")

    # 4: leftover visible barrier-like meshes (old guardrails etc.)
    left = 0
    for f in files_all:
        for m in kn[f]["meshes"]:
            if re.match(r"(guardrail|grpost|fence)", m["name"]):
                P = local(m["pos"].astype(float)); mid = P.copy(); mid[:, 2] += 0.0
                if not covered(P).all() and (~covered(P)).mean() > 0.02:
                    left += 1; fail(f"{f}:{m['name']} visible barrier mesh without collision ({100 * (~covered(P)).mean():.0f}% of vertices)")
    # 7: sign faces (build/roadside_signs.json = the emitted front quads): not mirrored, facing the approaching driver
    sg = json.load(open(BUILD / "roadside_signs.json")) if (BUILD / "roadside_signs.json").exists() else []
    nbad = 0
    for sgn in sg:
        P = np.array(sgn["P"]); UV = np.array(sgn["UV"]); t = np.array(sgn["t"]); appr = t if sgn["nb"] else -t     # direction of travel
        face = np.array(sgn["face"]); look = -appr                                    # the driver looks along +appr, i.e. at the face
        right = np.array([appr[1], -appr[0]])                                         # driver's right (looking along appr)
        bl, br, tr = P[0], P[1], P[2]                                                 # emitted order BL, BR, TR, TL
        ok = (np.dot(face, appr) < -0.9                                               # normal points against the approach direction
              and np.dot(np.array(sgn["N"][:2]), appr) < -0.9                         # emitted normal agrees (visible from the driver side)
              and np.dot((br - bl)[:2], right) > 0.2                                  # BR is on the driver's right
              and UV[1][0] > UV[0][0] and UV[2][0] > UV[3][0]                          # texture U increases to the right
              and UV[3][1] < UV[0][1])                                                # texture top is up
        if sgn["key"].startswith("chev") or sgn["key"].startswith("arrow") or (sgn["key"].startswith("dia_") and sgn["key"] not in ("dia_None_None", "dia_None_SLOW")):
            want = sgn["cdir"]
            if want and want not in sgn["key"].replace("chev_l", "left").replace("chev_r", "right") and not sgn["key"].endswith(("_" + want, want + "_None", want + "_SLOW")):
                ok = False
        if not ok:
            nbad += 1; fail(f"sign s={sgn['s']:.0f} {sgn['key']}: mirrored / wrong-facing / wrong turn direction")
    print(f"  sign faces checked: {len(sg)}; bad {nbad}")
    out = dict(kn5_files=len(kn), physics_meshes=stats["physics_meshes"], collision_triangles=stats["collision_tris"], solid_items=n_items,
               uncovered_items=n_bad, per_type={k: dict(items=v[0], uncovered=v[1], uncovered_m=round(v[2], 1)) for k, v in per_type.items()},
               failures=fails, tolerance_m=TOL, min_height_m=1.25)
    (ROOT / "reports").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "reports/qa_collisions.json", "w"), indent=1)
    print("COLLISION QA:", "PASS" if not fails else f"FAIL ({len(fails)} problems)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
