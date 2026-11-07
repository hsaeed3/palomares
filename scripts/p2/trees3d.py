"""Procedural 3D tree prototypes (SpeedTree-like, low poly) for 21_build_geometry.py.

Wood: recursive 3D branching skeleton -> tapered tube meshes (7 sides trunk, 5 limbs, 4 branches, 3 twigs) with UVs
wrapping round (u, inside the species strip of tree_trunk.png) and running along the branch (v, metres / circumference).
Foliage: alpha-tested leaf-cluster quads (0.6-1.6 m) on the outer branches, random orientation biased outward,
double-sided, normals pointing out from the crown centre (spherical) for ksTree-style lighting/translucency.
Atlases (drawn in 22_build_kn5.py): tree_trunk.png = 4 vertical strips (oak, bay, euc, syc);
tree_leaves.png = 4x4 cells, row = species (bottom-up in v), column = cluster variant.
"""
import math
import numpy as np

SPECIES = ("oak", "bay", "euc", "syc", "con", "bare")
STRIP = {"oak": 0, "bay": 1, "euc": 2, "syc": 3, "con": 0, "bare": 3}
# dev6: leaf atlas = photographic Poly Haven leaf clusters (scripts/p2/foliage_photo.py), 4 cols x ROWS rows
ROWS = 5
LEAFROW = {"oak": 0, "bay": 1, "euc": 2, "syc": 3, "con": 4, "bare": 3}
LEAFCOLS = {"syc": (0, 1), "bare": (2, 3)}                  # syc row: cols 0-1 leafy, 2-3 bare winter twigs
H0 = {"oak": 12.0, "bay": 13.0, "euc": 28.0, "syc": 17.0, "con": 18.0, "bare": 17.0}   # prototype height (m)
R0 = {"oak": 8.0, "bay": 4.5, "euc": 5.0, "syc": 6.0, "con": 3.5, "bare": 6.0}       # prototype crown radius (m)
N_PROTO = 4
_UP = np.array([0.0, 0.0, 1.0])


def _frame(d):
    a = _UP if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


class _B:
    def __init__(self):
        self.bP, self.bN, self.bUV, self.bT = [], [], [], []
        self.lP, self.lN, self.lUV, self.lT = [], [], [], []

    def tube(self, pts, radii, sides, strip):
        b0 = len(self.bP); cum = 0.0; rm = max(np.mean(radii), 0.02)
        for i, (p, r) in enumerate(zip(pts, radii)):
            d = _unit((pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]))
            u, v = _frame(d)
            if i: cum += np.linalg.norm(pts[i] - pts[i - 1])
            for j in range(sides + 1):
                a = 2 * math.pi * j / sides; n = u * math.cos(a) + v * math.sin(a)
                self.bP.append(p + n * r); self.bN.append(n)
                self.bUV.append(((strip + 0.02 + 0.96 * j / sides) / 4.0, cum / (2 * math.pi * rm) * 0.5))
        for i in range(len(pts) - 1):
            for j in range(sides):
                a = b0 + i * (sides + 1) + j; b = a + sides + 1
                self.bT += [(a, b, b + 1), (a, b + 1, a + 1)]

    def leaf(self, c, size, centre, row, col, rng, outward=True):
        out = _unit(c - centre + np.array([0, 0, 0.3 * size]))
        n = _unit(out + rng.normal(0, 0.25, 3)) if outward else _unit(rng.normal(0, 1, 3))
        u, v = _frame(n); rot = rng.uniform(0, 2 * math.pi)
        u, v = u * math.cos(rot) + v * math.sin(rot), -u * math.sin(rot) + v * math.cos(rot)
        h = size / 2; b0 = len(self.lP)
        for (su, sv, uu, vv) in ((-1, -1, 0, 0), (1, -1, 1, 0), (1, 1, 1, 1), (-1, 1, 0, 1)):
            p = c + u * su * h + v * sv * h
            self.lP.append(p); self.lN.append(_unit(_unit(p - centre) + np.array([0, 0, 0.25])))
            self.lUV.append(((col + 0.02 + 0.96 * uu) / 4.0, (row + 0.02 + 0.96 * vv) / ROWS))
        self.lT += [(b0, b0 + 1, b0 + 2), (b0, b0 + 2, b0 + 3), (b0, b0 + 2, b0 + 1), (b0, b0 + 3, b0 + 2)]


# species parameters: n main limbs, limb angle from vertical (deg), trunk fraction, recursion depth, kids, spread,
# droop (oak limbs arc out and down), leaf clusters (near, far LOD), cluster size (m, near; far x FAR_LS),
# trunk radius / height, crown volume: base / top as fraction of H, edge droop (fraction of H), lobe irregularity
_SP = {
    "oak": dict(limbs=(4, 6), ang=(50, 74), trunk=0.22, depth=2, kids=(2, 3), spread=0.65, droop=0.10, nl=(170, 40), ls=(1.3, 2.0), r0=0.034,
                cz=(0.26, 1.0), cdroop=0.20, lobe=0.18),
    "bay": dict(limbs=(4, 6), ang=(8, 26), trunk=0.0, depth=2, kids=(2, 3), spread=0.45, droop=0.0, nl=(140, 34), ls=(1.2, 1.9), r0=0.014,
                cz=(0.05, 1.0), cdroop=0.04, lobe=0.12),
    "euc": dict(limbs=(1, 1), ang=(0, 4), trunk=0.45, depth=2, kids=(2, 3), spread=0.55, droop=0.03, nl=(100, 26), ls=(1.2, 1.9), r0=0.020,
                cz=(0.34, 1.0), cdroop=0.06, lobe=0.22),
    "syc": dict(limbs=(2, 4), ang=(15, 35), trunk=0.28, depth=2, kids=(2, 2), spread=0.6, droop=0.0, nl=(120, 30), ls=(1.3, 2.0), r0=0.022,
                cz=(0.22, 1.0), cdroop=0.08, lobe=0.18),
    # Monterey cypress / pine: straight leader, short near-horizontal limbs, conical crown of dense sprays
    "con": dict(limbs=(8, 11), ang=(70, 95), trunk=0.85, depth=1, kids=(2, 2), spread=0.5, droop=0.05, nl=(130, 32), ls=(1.2, 1.8), r0=0.024,
                cz=(0.06, 1.0), cdroop=0.0, lobe=0.10, cone=True),
    # winter (bare) sycamore: the syc skeleton, its outer twigs visible as twig cards, fewer of them
    "bare": dict(limbs=(2, 4), ang=(15, 35), trunk=0.28, depth=2, kids=(2, 2), spread=0.6, droop=0.0, nl=(60, 18), ls=(1.8, 2.6), r0=0.022,
                cz=(0.22, 1.0), cdroop=0.0, lobe=0.18),
}
FAR_LS = 1.9
PE = 2.6  # crown superellipsoid exponent (fuller, rounder-shouldered crowns than an ellipsoid)


def make_proto(kind, seed, lod=0):
    """returns dict(bark=(P,N,UV,T), leaf=(P,N,UV,T)) in metres, base at origin, height ~H0[kind], crown radius ~R0.
    Wood skeleton is scaled to sit inside the crown volume; leaf clusters (overlapping, smaller near / big far) are
    placed at every limb end (no bare stubs) + filling an irregular, low, wide crown ellipsoid incl. its interior."""
    r = np.random.default_rng(1000 * SPECIES.index(kind) + seed)
    S = _SP[kind]; H = H0[kind]; R = R0[kind]; B = _B(); row = LEAFROW[kind]
    z0c, z1c = S["cz"][0] * H, S["cz"][1] * H
    zc, rz = (z0c + z1c) / 2, (z1c - z0c) / 2
    centre = np.array([0, 0, zc])
    tips, ends = [], []
    maxd = S["depth"]
    tubes = []  # deferred: drawn after the skeleton is scaled into the crown

    def grow(p, d, ln, rad, depth):
        nsub = (3, 2, 1, 1)[min(depth, 3)] if not lod else 1
        pts, rr = [p.copy()], [rad]
        for i in range(nsub):
            d = _unit(d + r.normal(0, 0.13 if kind != "euc" else 0.05, 3))
            if S["droop"] and depth >= 1:
                hz = _unit(np.array([d[0], d[1], 0]) + 1e-6)
                d = _unit(d + hz * S["droop"] - _UP * S["droop"] * 0.5 * (depth >= 2))
            if kind in ("syc", "euc", "bay") and depth <= 1:
                d = _unit(d + _UP * 0.08)
            p = p + d * ln / nsub; pts.append(p.copy()); rr.append(rad * (1 - 0.45 * (i + 1) / nsub))
        if depth <= 1 or kind == "bare":  # small branches are hidden inside the leaf clusters: no tubes (bare: shown)
            if not lod or kind == "bay":  # far LOD: trunk only (bay: its stems), limbs hidden in the big clusters
                tubes.append((np.array(pts), rr, (4 if not lod else 3)))
            ends.append(p.copy())
        if depth >= maxd:
            tips.append(p.copy()); return
        k = r.integers(S["kids"][0], S["kids"][1] + 1)
        az0 = r.uniform(0, 2 * math.pi)
        for i in range(k):
            u, v = _frame(d); az = az0 + 2 * math.pi * i / k + r.normal(0, 0.3)
            side = u * math.cos(az) + v * math.sin(az)
            nd = _unit(d * math.cos(S["spread"]) + side * math.sin(S["spread"]))
            grow(p, nd, ln * r.uniform(0.55, 0.75), rr[-1] * (0.85 if i == 0 else 0.7), depth + 1)

    rt = H * S["r0"] * (1.4 if kind == "oak" else 1.0)
    trunk = None
    if kind == "bay":  # multi-stem from the root crown
        for i in range(r.integers(*S["limbs"])):
            az = r.uniform(0, 2 * math.pi); a = math.radians(r.uniform(*S["ang"]))
            d = np.array([math.sin(a) * math.cos(az), math.sin(a) * math.sin(az), math.cos(a)])
            grow(np.array([r.normal(0, 0.15), r.normal(0, 0.15), -0.3]), d, H * r.uniform(0.42, 0.55), rt * r.uniform(0.8, 1.1), 1)
    else:
        lean = math.radians(r.uniform(0, 10)); laz = r.uniform(0, 2 * math.pi)
        d0 = np.array([math.sin(lean) * math.cos(laz), math.sin(lean) * math.sin(laz), math.cos(lean)])
        top = d0 * H * S["trunk"] + np.array([0, 0, -0.3])
        trunk = (np.array([[0, 0, -0.3], top * 0.5, top]), [rt * 1.25, rt * 1.05, rt * 0.95], 5 if not lod else 4)
        nl = r.integers(S["limbs"][0], S["limbs"][1] + 1)
        for i in range(nl):
            az = 2 * math.pi * i / nl + r.normal(0, 0.35); a = math.radians(r.uniform(*S["ang"]))
            d = np.array([math.sin(a) * math.cos(az), math.sin(a) * math.sin(az), math.cos(a)])
            ln = H * (0.45 if kind == "oak" else 0.40 if kind == "euc" else 0.42) * r.uniform(0.8, 1.1)
            if kind == "euc" and i == 0:
                d = _unit(d0 + _UP)
            grow(top, d, ln, rt * (0.85 if nl == 1 else 0.65) * r.uniform(0.85, 1.0), 1)
    # scale the skeleton (not the trunk base) so every branch tip lies inside the crown: 80 % radius, 90 % height
    tp = np.array(tips)
    sxy = 0.80 * R / max(np.percentile(np.hypot(tp[:, 0], tp[:, 1]), 95), 0.5)
    sz = 0.90 * H / max(tp[:, 2].max(), 1.0)
    sc = np.array([sxy, sxy, sz])
    if trunk is not None:
        B.tube(trunk[0] * sc, trunk[1], trunk[2], STRIP[kind])
    for pts, rr, sides in tubes:
        pts = pts * sc
        pts[-1] = pts[-2] + (pts[-1] - pts[-2]) * 0.85  # limb tips pruned back: they end inside a leaf cluster
        B.tube(pts, rr, sides, STRIP[kind])
    ends = [e * sc for e in ends]; tips = [t * sc for t in tips]
    # crown volume: ellipsoid (rx = R, z0c..z1c), lobed outline, edges droop toward the ground (oak skirts)
    ph = r.uniform(0, 2 * math.pi, 3)

    def shell(az, el):
        return 1.0 + S["lobe"] * (0.6 * math.sin(3 * az + ph[0]) * math.cos(el) + 0.4 * math.sin(5 * az + 2 * el + ph[1]))

    def place(q):  # pull a point into the crown volume, apply edge droop
        rel = q - centre; az = math.atan2(rel[1], rel[0])
        rho = math.hypot(rel[0], rel[1]) / R; el = math.atan2(rel[2] / rz, rho + 1e-6)
        e = (rho ** PE + abs(rel[2] / rz) ** PE) ** (1 / PE); lim = shell(az, el) * 0.95
        if e > lim:
            rel = rel * lim / e; rho *= lim / e
        if S.get("cone"):  # conical crown: radius shrinks linearly to the top
            t = np.clip((centre[2] + rel[2] - z0c) / (z1c - z0c), 0, 1)
            rel[:2] *= max(0.08, 1.05 * (1 - t))
        q = centre + rel
        q[2] -= S["cdroop"] * H * min(rho, 1.2) ** 3
        return q

    nlv = S["nl"][1 if lod else 0]
    pts = []
    anchors = ends + tips[::(1 if not lod else 3)]
    for e in anchors[:int(nlv * 0.45)]:  # every limb end (then tips) carries a cluster
        pts.append(place(e + r.normal(0, 0.25, 3)))
    while len(pts) < nlv:  # fill the volume: shell-biased but with interior clusters so the crown isn't hollow
        v = r.normal(0, 1, 3); v /= np.linalg.norm(v); u = r.uniform() ** (1 / 2.2 if not lod else 1 / 1.6)
        az = math.atan2(v[1], v[0]); el = math.asin(np.clip(v[2], -1, 1))
        u *= shell(az, el)
        v = v / (np.abs(v[0] ** 2 + v[1] ** 2) ** (PE / 2) + np.abs(v[2]) ** PE) ** (1 / PE)  # superellipsoid: full dome, no thin top
        pts.append(place(centre + np.array([v[0] * R * u, v[1] * R * u, v[2] * rz * u])))
    for q in pts:
        rel = q - centre
        frac = min(1.0, math.sqrt((rel[0] ** 2 + rel[1] ** 2) / R ** 2 + (rel[2] / rz) ** 2))
        # tint: outer / upper clusters fresh lime (col 3), interior / lower darker old leaves (col 0)
        sc_ = 0.55 * frac + 0.35 * (rel[2] / rz + 1) / 2 + r.normal(0, 0.18)
        col = int(np.clip(sc_ * 4 - 0.6, 0, 3))
        if kind in LEAFCOLS:
            col = LEAFCOLS[kind][int(col >= 2)]
        sz_ = r.uniform(*S["ls"]) * (FAR_LS if lod else 1.0) * (0.8 + 0.3 * frac)
        B.leaf(q, sz_, centre, row, col, r, outward=bool(r.uniform() < 0.68))

    def fin(P, N, UV, T):
        return np.array(P, float).reshape(-1, 3), np.array(N, float).reshape(-1, 3), np.array(UV, float).reshape(-1, 2), np.array(T, np.int64).reshape(-1, 3)
    return dict(bark=fin(B.bP, B.bN, B.bUV, B.bT), leaf=fin(B.lP, B.lN, B.lUV, B.lT))


def instance(proto, x, y, z, h, cr, kind, yaw, lean_dir=None, lean=0.0):
    """scale prototype to LiDAR height + crown radius, random yaw, slight lean; returns transformed copies"""
    s = h / H0[kind]
    sxy = s * float(np.clip(cr / (R0[kind] * s), 0.7, 1.5))
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    M = Rz @ np.diag([sxy, sxy, s])
    if lean and lean_dir is not None:  # shear toward lean_dir (lean = horizontal offset per metre of height)
        Sh = np.eye(3); Sh[0, 2], Sh[1, 2] = lean_dir[0] * lean, lean_dir[1] * lean
        M = Sh @ M
    Minv_T = np.linalg.inv(M).T
    out = {}
    for key, (P, N, UV, T) in proto.items():
        P2 = P @ M.T + np.array([x, y, z]); N2 = N @ Minv_T.T
        N2 /= np.linalg.norm(N2, axis=1, keepdims=True) + 1e-9
        out[key] = (P2, N2, UV, T)
    return out


def shrub_clump(x, y, z, h, w, u0, rng, nq=6):
    """small 3D clump of shrub leaf quads (shrub.png atlas, two styles in u halves); spherical normals"""
    P, N, UV, T = [], [], [], []
    c = np.array([x, y, z + h * 0.45])
    for i in range(nq):
        # varying heights: low fern / poison-oak layer, mid brush, a few tall toyon / coyote-brush tops
        lev = (-0.32, -0.12, 0.0, 0.12, 0.25, -0.25, 0.05, 0.18)[i % 8]
        q = c + np.array([rng.normal(0, w * 0.28), rng.normal(0, w * 0.28), (lev + rng.uniform(-0.06, 0.06)) * h])
        n = _unit(np.array([q[0] - x, q[1] - y, 0]) + rng.normal(0, 0.5, 3) + np.array([0, 0, 0.2]))
        hz = _unit(np.array([-n[1], n[0], 0.0]) + 1e-6); vt = _unit(np.cross(n, hz))
        if vt[2] < 0: vt = -vt
        sx, sz_ = w * rng.uniform(0.45, 0.65), h * rng.uniform(0.55, 0.8)
        b = len(P); base = q - vt * sz_ * 0.5
        for (a, bb, uu, vv) in ((-1, 0, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1), (-1, 1, 0, 1)):
            p = base + hz * a * sx + vt * bb * sz_
            P.append(p); N.append(_unit(_unit(p - c) + np.array([0, 0, 0.3]))); UV.append((u0 + 0.5 * uu, vv))
        T += [(b, b + 1, b + 2), (b, b + 2, b + 3), (b, b + 2, b + 1), (b, b + 3, b + 2)]
    return P, N, UV, T
