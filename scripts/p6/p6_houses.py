"""dev6: footprint-based houses from data/derived/buildings.geojson (measured LiDAR parts + roof fits).

Registered into scripts/p5/assets.py CATALOG as asset "building.measured" (cls building). Unlike catalog assets the
generator works in ABSOLUTE local track coordinates (x_local, y_local, z NAVD88): MB.meta["world"] = True tells 21e
and scene.py not to transform it. Mesh dict format unchanged (MB -> P, T, UV, N; MB.col = closed boxes).

params (written by scripts/p6/07_ledger_measured.py):
  parts   [{cx, cy, yaw, Lu, Lv, form: flat|shed|gable|hip|tank, ridge: u|v|None, eave, ridge_z, shed_dir}]  local coords
  ground_z  wall base (NAVD88); walls start 0.3 m below it
  roof_rgb  NAIP 2022 median roof colour -> nearest roof palette tile
  wall      wall palette index (NOT measured: GoPro/Mapillary colour sampling not done)
Collision: one closed box per part (MB.cbox, world frame), which the safety pipeline turns into its 8-vertex boxes.
"""
import math
import numpy as np

ROOF_PAL = {"roof": (96, 92, 90), "tank_black": (40, 42, 42), "corr_grey": (150, 152, 150), "galv": (168, 170, 166),
            "paint_white": (225, 225, 220), "corr_red": (132, 44, 36), "corr_green": (70, 96, 72), "stump_top": (150, 120, 84),
            "gravel": (128, 122, 112)}   # existing ledger-atlas tiles (the atlas has no room for new ones)
WALL_PAL = {"p6wall_white": (226, 224, 216), "p6wall_cream": (220, 206, 176), "p6wall_tan": (190, 166, 130), "p6wall_grey": (156, 156, 150),
            "p6wall_brown": (128, 100, 76), "p6wall_red": (128, 48, 40)}
AS = None
WALL_MAP = {"p6wall_white": "siding_white", "p6wall_cream": "siding_tan", "p6wall_tan": "siding_tan", "p6wall_grey": "siding_grey",
            "p6wall_brown": "barn_wall", "p6wall_red": "barn_wall"}


def tiles(A):
    """paint palette tiles (called from assets.build_atlas)"""
    for k, rgb in ROOF_PAL.items():
        pass
    # walls reuse the existing siding tiles (WALL_MAP)


def roof_tile(rgb):
    rgb = np.asarray(rgb or (112, 112, 110), float)
    return min(ROOF_PAL, key=lambda k: float(np.sum((np.asarray(ROOF_PAL[k]) - rgb) ** 2)))


def _part(m, q, zg, wall, roof):
    R = AS.R; th = float(q["yaw"]); Lu, Lv = float(q["Lu"]), float(q["Lv"]); ridge = q.get("ridge"); form = q.get("form", "flat")
    if form in ("gable", "hip") and ridge == "v":           # rotate so the ridge always runs along local u
        th += math.pi / 2; Lu, Lv = Lv, Lu
    c, s = math.cos(th), math.sin(th); cx, cy = float(q["cx"]), float(q["cy"])
    W = lambda u, v, z: [cx + u * c - v * s, cy + u * s + v * c, z]
    eave = float(q["eave"]); rz = float(q["ridge_z"]); base = zg - 0.3
    eave = max(eave, zg + 2.0); rz = max(rz, eave)
    if form == "shed":
        b_, c_ = (q.get("shed_dir") or [0, 0])[:2]; zc = (float(q["eave"]) + float(q["ridge_z"])) / 2
        zf = lambda u, v: max(zc + b_ * u + c_ * v, zg + 2.0)
        if ridge == "v": zf = lambda u, v: max(zc + b_ * v + c_ * (-u), zg + 2.0)   # never: shed has ridge None
    elif form in ("flat", "tank"):
        zf = lambda u, v: eave
    else:
        zf = lambda u, v: eave
    co = [(-Lu / 2, -Lv / 2), (Lu / 2, -Lv / 2), (Lu / 2, Lv / 2), (-Lu / 2, Lv / 2)]
    if form == "tank":
        r = (Lu + Lv) / 4; m.cyl(cx, cy, base, eave, r, 12, R("tank_black"), top=R("tank_black")); return
    for k in range(4):                                       # walls
        (u0, v0), (u1, v1) = co[k], co[(k + 1) % 4]; L = math.hypot(u1 - u0, v1 - v0); nu = max(1, int(round(L / 4.0)))
        out = np.array([(u0 + u1) / 2 * c - (v0 + v1) / 2 * s, (u0 + u1) / 2 * s + (v0 + v1) / 2 * c, 0.0])
        for j in range(nu):
            a0, a1 = j / nu, (j + 1) / nu
            ua, va = u0 + (u1 - u0) * a0, v0 + (v1 - v0) * a0; ub, vb = u0 + (u1 - u0) * a1, v0 + (v1 - v0) * a1
            m.quad([W(ua, va, base), W(ub, vb, base), W(ub, vb, zf(ub, vb)), W(ua, va, zf(ua, va))], wall, out=out)
    ov = 0.4; hu, hv = Lu / 2 + ov, Lv / 2 + ov; k_ = (rz - eave) / max(Lv / 2, 0.5); ez = eave - k_ * ov
    if form in ("flat", "shed"):
        m.quad([W(-hu, -hv, zf(-Lu / 2, -Lv / 2)), W(hu, -hv, zf(Lu / 2, -Lv / 2)), W(hu, hv, zf(Lu / 2, Lv / 2)), W(-hu, hv, zf(-Lu / 2, Lv / 2))], roof, out=np.array([0, 0, 1.0]))
        return
    if form == "gable":
        m.quad([W(-hu, -hv, ez), W(hu, -hv, ez), W(hu, 0, rz), W(-hu, 0, rz)], roof, out=np.array([s, -c, 1.0]))
        m.quad([W(-hu, 0, rz), W(hu, 0, rz), W(hu, hv, ez), W(-hu, hv, ez)], roof, out=np.array([-s, c, 1.0]))
        for su in (-1, 1):                                   # gable-end triangles
            P3 = np.array([W(su * Lu / 2, -Lv / 2, eave), W(su * Lu / 2, Lv / 2, eave), W(su * Lu / 2, 0, rz)])
            if np.dot(np.cross(P3[1] - P3[0], P3[2] - P3[0]), [su * c, su * s, 0]) < 0: P3 = P3[[1, 0, 2]]
            m.tri(P3, wall, [(0, 0), (1, 0), (0.5, 0.4)])
        return
    # hip: ridge along u of length Lu - Lv (>= 0)
    rl = max(Lu - Lv, 0) / 2
    m.quad([W(-hu, -hv, ez), W(hu, -hv, ez), W(rl, 0, rz), W(-rl, 0, rz)], roof, out=np.array([s, -c, 1.0]))
    m.quad([W(-rl, 0, rz), W(rl, 0, rz), W(hu, hv, ez), W(-hu, hv, ez)], roof, out=np.array([-s, c, 1.0]))
    for su in (-1, 1):
        P3 = np.array([W(su * hu, -hv, ez), W(su * hu, hv, ez), W(su * rl, 0, rz)])
        if np.dot(np.cross(P3[1] - P3[0], P3[2] - P3[0]), [su * c, su * s, 1]) < 0: P3 = P3[[1, 0, 2]]
        m.tri(P3, roof, [(0, 0), (1, 0), (0.5, 1.0)])


def g_measured(L, p):
    m = AS.MB(); m.meta["world"] = True
    if not p.get("parts"):          # catalog / contact-sheet preview: a 12 x 9 m gable house at the origin
        p = dict(p, ground_z=0.0, parts=[dict(cx=0, cy=4.5, yaw=0.0, Lu=12, Lv=9, form="gable", ridge="u", eave=3.0, ridge_z=5.2)])
    zg = float(p["ground_z"]); roof = AS.R(roof_tile(p.get("roof_rgb"))); wall = AS.R(WALL_MAP.get(p.get("wall"), "siding_white"))
    for q in p.get("parts", []):
        _part(m, q, zg, wall, roof)
        # collision: one closed box per part, world frame (cbox: cx, cy, L, W, z0, z1, yaw)
        m.cbox(float(q["cx"]), float(q["cy"]), float(q["Lu"]), float(q["Lv"]), zg - 0.5, max(float(q["eave"]), zg + 2.0) + 0.5, float(q["yaw"]))
    return m


def register(assets_mod):
    global AS
    AS = assets_mod
    AS.reg("building.measured", "building", g_measured, (10, 10, 6), dict(shape="none"), True, 18,
           desc="house/barn generated from its measured LiDAR footprint parts and roof fit (dev6, world coordinates)")
