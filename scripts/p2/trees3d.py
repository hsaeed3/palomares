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

SPECIES = ("oak", "bay", "euc", "syc")
STRIP = {"oak": 0, "bay": 1, "euc": 2, "syc": 3}
LEAFROW = {"oak": 0, "bay": 1, "euc": 2, "syc": 3}
H0 = {"oak": 12.0, "bay": 13.0, "euc": 28.0, "syc": 17.0}   # prototype height (m)
R0 = {"oak": 8.0, "bay": 4.5, "euc": 5.0, "syc": 6.0}       # prototype crown radius (m)
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

    def leaf(self, c, size, centre, row, col, rng):
        out = _unit(c - centre + np.array([0, 0, 0.3 * size]))
        n = _unit(out * 0.6 + rng.normal(0, 0.6, 3))
        u, v = _frame(n); rot = rng.uniform(0, 2 * math.pi)
        u, v = u * math.cos(rot) + v * math.sin(rot), -u * math.sin(rot) + v * math.cos(rot)
        h = size / 2; b0 = len(self.lP)
        for (su, sv, uu, vv) in ((-1, -1, 0, 0), (1, -1, 1, 0), (1, 1, 1, 1), (-1, 1, 0, 1)):
            p = c + u * su * h + v * sv * h
            self.lP.append(p); self.lN.append(_unit(_unit(p - centre) + np.array([0, 0, 0.25])))
            self.lUV.append(((col + 0.02 + 0.96 * uu) / 4.0, (row + 0.02 + 0.96 * vv) / 4.0))
        self.lT += [(b0, b0 + 1, b0 + 2), (b0, b0 + 2, b0 + 3), (b0, b0 + 2, b0 + 1), (b0, b0 + 3, b0 + 2)]


# species parameters: n main limbs, limb angle from vertical (deg), trunk fraction, recursion depth, kids, spread,
# droop (oak limbs arc out and down), leaf clusters per tip, cluster size (m)
_SP = {
    "oak": dict(limbs=(3, 5), ang=(48, 72), trunk=0.22, depth=3, kids=(2, 3), spread=0.65, droop=0.10, lpt=4, ls=(1.0, 1.8), r0=0.030),
    "bay": dict(limbs=(3, 6), ang=(8, 25), trunk=0.0, depth=3, kids=(2, 3), spread=0.45, droop=0.0, lpt=3, ls=(0.8, 1.4), r0=0.012),
    "euc": dict(limbs=(1, 1), ang=(0, 4), trunk=0.45, depth=3, kids=(2, 3), spread=0.55, droop=0.03, lpt=4, ls=(0.9, 1.6), r0=0.017),
    "syc": dict(limbs=(2, 3), ang=(15, 35), trunk=0.28, depth=3, kids=(2, 2), spread=0.6, droop=0.0, lpt=4, ls=(0.8, 1.5), r0=0.018),
}


def make_proto(kind, seed, lod=0):
    """returns dict(bark=(P,N,UV,T), leaf=(P,N,UV,T)) in metres, base at origin, height ~H0[kind]."""
    r = np.random.default_rng(1000 * SPECIES.index(kind) + seed)
    S = _SP[kind]; H = H0[kind]; B = _B(); row = LEAFROW[kind]
    centre = np.array([0, 0, H * (0.62 if kind != "oak" else 0.55)])
    tips = []
    maxd = S["depth"] - (1 if lod else 0)

    def grow(p, d, ln, rad, depth):
        nsub = (3, 2, 1, 1)[min(depth, 3)]
        pts, rr = [p.copy()], [rad]
        for i in range(nsub):
            d = _unit(d + r.normal(0, 0.13 if kind != "euc" else 0.05, 3))
            if S["droop"] and depth >= 1:
                hz = _unit(np.array([d[0], d[1], 0]) + 1e-6)
                d = _unit(d + hz * S["droop"] - _UP * S["droop"] * 0.5 * (depth >= 2))
            if kind in ("syc", "euc", "bay") and depth <= 1:
                d = _unit(d + _UP * 0.08)
            p = p + d * ln / nsub; pts.append(p.copy()); rr.append(rad * (1 - 0.35 * (i + 1) / nsub))
        sides = (6, 4, 3, 3)[min(depth, 3)]
        if not (lod and depth >= 2):  # far LOD: small branches hidden in the clusters, no tubes
            B.tube(np.array(pts), rr, sides, STRIP[kind])
        end_r = rr[-1]
        if depth >= maxd:
            tips.append((p, d)); return
        if depth >= maxd - 1:  # leaves also along the outer branches
            for q in pts[1:-1]:
                tips.append((q, d))
        k = r.integers(S["kids"][0], S["kids"][1] + 1)
        az0 = r.uniform(0, 2 * math.pi)
        for i in range(k):
            u, v = _frame(d); az = az0 + 2 * math.pi * i / k + r.normal(0, 0.3)
            side = u * math.cos(az) + v * math.sin(az)
            nd = _unit(d * math.cos(S["spread"]) + side * math.sin(S["spread"]))
            grow(p, nd, ln * r.uniform(0.55, 0.75), end_r * (0.85 if i == 0 else 0.7), depth + 1)

    rt = H * S["r0"] * (1.4 if kind == "oak" else 1.0) * 1.0
    if kind == "bay":  # multi-stem from the root crown
        for i in range(r.integers(*S["limbs"])):
            az = r.uniform(0, 2 * math.pi); a = math.radians(r.uniform(*S["ang"]))
            d = np.array([math.sin(a) * math.cos(az), math.sin(a) * math.sin(az), math.cos(a)])
            grow(np.array([r.normal(0, 0.15), r.normal(0, 0.15), -0.3]), d, H * r.uniform(0.42, 0.55), rt * r.uniform(0.8, 1.1), 1)
    else:
        lean = math.radians(r.uniform(0, 10)); laz = r.uniform(0, 2 * math.pi)
        d0 = np.array([math.sin(lean) * math.cos(laz), math.sin(lean) * math.sin(laz), math.cos(lean)])
        top = d0 * H * S["trunk"] + np.array([0, 0, -0.3])
        B.tube(np.array([[0, 0, -0.3], top * 0.5, top]), [rt * 1.25, rt * 1.05, rt * 0.95], 6 if not lod else 5, STRIP[kind])
        nl = r.integers(S["limbs"][0], S["limbs"][1] + 1)
        for i in range(nl):
            az = 2 * math.pi * i / nl + r.normal(0, 0.35); a = math.radians(r.uniform(*S["ang"]))
            d = np.array([math.sin(a) * math.cos(az), math.sin(a) * math.sin(az), math.cos(a)])
            ln = H * (0.45 if kind == "oak" else 0.40 if kind == "euc" else 0.42) * r.uniform(0.8, 1.1)
            if kind == "euc" and i == 0:
                d = _unit(d0 + _UP)
            grow(top, d, ln, rt * (0.85 if nl == 1 else 0.65) * r.uniform(0.85, 1.0), 1)
    # leaf clusters at tips; ~20 % dropped at random -> real sky/sun holes between clumps
    keep = 1.0
    for (p, d) in tips:
        if r.uniform() > keep or r.uniform() < 0.18:
            continue
        for _ in range(S["lpt"] + 1):
            q = p + d * r.uniform(0, 0.6) + r.normal(0, 0.55, 3)
            B.leaf(q, r.uniform(*S["ls"]) * (1.6 if lod else 1.0), centre, row, int(r.integers(0, 4)), r)
    # normalise: crown radius R0 (90th pct of leaf distance), height H
    lp = np.array(B.lP); bp = np.array(B.bP)
    rx = np.percentile(np.hypot(lp[:, 0], lp[:, 1]), 90) if len(lp) else R0[kind]
    zt = max(lp[:, 2].max() if len(lp) else 0, bp[:, 2].max())
    sxy, sz = R0[kind] / max(rx, 0.5), H / max(zt, 1.0)
    sc = np.array([sxy, sxy, sz])

    def fin(P, N, UV, T):
        P = np.array(P) * sc; N = np.array(N) / sc; N /= np.linalg.norm(N, axis=1, keepdims=True)
        return P, N, np.array(UV), np.array(T, np.int64)
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


def shrub_clump(x, y, z, h, w, u0, rng, nq=4):
    """small 3D clump of shrub leaf quads (shrub.png atlas, two styles in u halves); spherical normals"""
    P, N, UV, T = [], [], [], []
    c = np.array([x, y, z + h * 0.45])
    for i in range(nq):
        q = c + np.array([rng.normal(0, w * 0.22), rng.normal(0, w * 0.22), rng.uniform(-0.15, 0.25) * h])
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
