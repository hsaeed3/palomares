"""Robust AC collision helpers (physics meshes named 1WALL_*).

Why closed thin boxes instead of the old double-sided flat strips:
  * a flat strip built from the same vertices in both windings gets vertex normals that cancel to (0,0,0);
    AC (like most engines) is happier with well-formed meshes - the community rule of thumb is "the normal of a
    wall must point outwards", and "walls should be simplest geometry" (assettocorsamods.net, 'BIG PROBLEM WITH
    WALLS').  A closed box has an unambiguous outward winding on every face, valid normals, and a few cm of depth
    so grazing hits at speed cannot slip between the two sheets.
  * the old strips were only 1.0 m above the road edge, so a car with any roll/pitch (or an SUV) rode over them.

Flags: collision meshes are written visible=True, renderable=False (the "RENDERABLE=0" convention reported to work
on assettocorsamods.net); visible=False has been reported to make AC treat the node as absent.
"""
import numpy as np

WALL_FLAGS = dict(visible=True, renderable=False, cast_shadows=False)
MIN_TOP = 1.4          # collision top is at least this far above the local ground / road edge


def box_ribbon(x, y, z0, z1, half=0.07, cap=True):
    """Closed box swept along the polyline (x, y), from z0 to z1 (arrays or scalars), half = half thickness.
    Returns P (n,3), T (m,3), N (n,3).  Every face has its own vertices (flat, outward normals)."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    z0 = np.broadcast_to(np.asarray(z0, float), (n,)); z1 = np.broadcast_to(np.asarray(z1, float), (n,))
    if n < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3))
    t = np.gradient(np.c_[x, y], axis=0)
    ln = np.hypot(t[:, 0], t[:, 1]); ln[ln < 1e-9] = 1.0
    t = t / ln[:, None]; l = np.c_[-t[:, 1], t[:, 0]]            # left normal
    Lx, Ly = x + l[:, 0] * half, y + l[:, 1] * half
    Rx, Ry = x - l[:, 0] * half, y - l[:, 1] * half
    P, T, N = [], [], []

    def face(a, b, c, d, nrm):
        """quad a,b,c,d CCW seen from outside (normal nrm)"""
        k = sum(len(p) for p in P)
        P.append(np.array([a, b, c, d])); N.append(np.tile(nrm, (4, 1))); T.append(np.array([[k, k + 1, k + 2], [k, k + 2, k + 3]]))

    for i in range(n - 1):
        j = i + 1
        pl = lambda X, Y, Z, q: (X[q], Y[q], Z[q])
        # left side (normal = +l), travelling i->j: bottom i, bottom j, top j, top i is CCW seen from the left
        # (looking from +l toward -l with t pointing to the viewer's right): check with cross product below
        for (Xa, Ya, nrm_sign) in ((Lx, Ly, 1.0), (Rx, Ry, -1.0)):
            a = (Xa[i], Ya[i], z0[i]); b = (Xa[j], Ya[j], z0[j]); c = (Xa[j], Ya[j], z1[j]); d = (Xa[i], Ya[i], z1[i])
            nrm = np.cross(np.subtract(b, a), np.subtract(d, a))      # for the left face this points toward -l ...
            want = np.array([l[i, 0] * nrm_sign, l[i, 1] * nrm_sign, 0.0])
            if np.dot(nrm, want) < 0:
                a, b, c, d = b, a, d, c                                   # ... so flip when needed
            nn = np.cross(np.subtract(b, a), np.subtract(d, a)); nn = nn / (np.linalg.norm(nn) + 1e-12)
            face(a, b, c, d, nn)
        # top
        a = (Rx[i], Ry[i], z1[i]); b = (Rx[j], Ry[j], z1[j]); c = (Lx[j], Ly[j], z1[j]); d = (Lx[i], Ly[i], z1[i])
        nn = np.cross(np.subtract(b, a), np.subtract(d, a))
        if nn[2] < 0:
            a, b, c, d = b, a, d, c; nn = -nn
        face(a, b, c, d, nn / (np.linalg.norm(nn) + 1e-12))
    if cap:
        for q, sgn in ((0, -1.0), (n - 1, 1.0)):
            a = (Rx[q], Ry[q], z0[q]); b = (Lx[q], Ly[q], z0[q]); c = (Lx[q], Ly[q], z1[q]); d = (Rx[q], Ry[q], z1[q])
            want = np.array([t[q, 0] * sgn, t[q, 1] * sgn, 0.0])
            nn = np.cross(np.subtract(b, a), np.subtract(d, a))
            if np.dot(nn, want) < 0:
                a, b, c, d = b, a, d, c; nn = -nn
            face(a, b, c, d, nn / (np.linalg.norm(nn) + 1e-12))
    P = np.concatenate(P); T = np.concatenate(T); N = np.concatenate(N)
    return P, T, N


def box_solid(cx, cy, fwd, L, W, z0, z1):
    """closed oriented box (long axis fwd), outward normals"""
    f = np.asarray(fwd, float); f = f / (np.hypot(*f) + 1e-12); l = np.array([-f[1], f[0]])
    a = np.array([cx, cy]) - f * L / 2; b = np.array([cx, cy]) + f * L / 2
    return box_ribbon([a[0], b[0]], [a[1], b[1]], z0, z1, half=W / 2)


class Collider:
    """accumulates closed boxes into ONE 1WALL mesh dict per chunk (splits at 60k vertices)"""
    def __init__(self, prefix):
        self.prefix, self.P, self.T, self.N, self.n = prefix, [], [], [], 0

    def add(self, P, T, N):
        if len(P) == 0:
            return
        self.P.append(P); self.T.append(T + self.n); self.N.append(N); self.n += len(P)

    def ribbon(self, x, y, z0, z1, half=0.07):
        self.add(*box_ribbon(x, y, z0, z1, half))

    def box(self, cx, cy, fwd, L, W, z0, z1):
        self.add(*box_solid(cx, cy, fwd, L, W, z0, z1))

    def meshes(self):
        if not self.P:
            return []
        P = np.concatenate(self.P); T = np.concatenate(self.T); N = np.concatenate(self.N)
        out, k, start = [], 0, 0
        # split by triangles keeping every piece < 60k vertices (faces own their vertices: 4 per quad)
        per = 60000 // 4 * 2
        while start < len(T):
            tt = T[start:start + per]; used, inv = np.unique(tt, return_inverse=True)
            out.append(dict(name=self.prefix if k == 0 else f"{self.prefix}_{k}", material="invisible", P=P[used], T=inv.reshape(-1, 3),
                            UV=np.zeros((len(used), 2)), N=N[used], **WALL_FLAGS))
            start += per; k += 1
        return out


def fix_wall_flags(groups):
    """make every 1WALL_* mesh in every group visible=True/renderable=False"""
    n = 0
    for ms in groups.values():
        for m in ms:
            if m["name"].startswith("1WALL"):
                m.update(WALL_FLAGS); n += 1
                ln = np.linalg.norm(m["N"], axis=1)
                if (np.abs(ln - 1) > 0.05).any():       # e.g. 21b's double-sided strips: shared vertices, opposite windings => zero normals
                    P = m["P"]; T = m["T"].reshape(-1, 3)
                    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]); a = np.linalg.norm(fn, axis=1)
                    keep = a > 1e-9; T, fn, a = T[keep], fn[keep], a[keep]
                    m["P"] = P[T.ravel()]; m["T"] = np.arange(len(T) * 3).reshape(-1, 3)
                    m["N"] = np.repeat(fn / a[:, None], 3, axis=0); m["UV"] = np.zeros((len(m["P"]), 2))
    return n


# ---------------------------------------------------------------------------------------------------------------
# v0.2.0-dev3 physics fixes (applied by scripts/p2/21d_physics.py as a post-process; see docs/AC_CONVENTIONS.md
# "Physics meshes").  AC resolves a car-vs-mesh contact by pushing the body out along the normal of the nearest
# face.  With a 0.14 m closed box, a car that penetrates more than ~7 cm in one physics step (head-on, > ~25 km/h
# at 333 Hz with soft-body overlap) is nearer the BACK face and gets pushed out the far side: it drives through.
# So the face away from the road is moved WALL_DEPTH away: the road-side face is always the nearest exit.
WALL_DEPTH = 1.2


def thicken_away(m, road_xy, depth=WALL_DEPTH, min_depth=0.3):
    """m: 1WALL mesh dict of flat quads (own vertices per face).  Vertices of faces whose normal points away from
    the road (dot > 0.6 with the horizontal away-from-road direction), and every vertex sharing their position (top
    and cap faces), move `depth` along that direction.  Falls back to min_depth where the moved point would get
    closer to the road (hairpins).  Returns number of moved vertices."""
    from scipy.spatial import cKDTree
    P, N, T = m["P"].astype(float), m["N"], m["T"].reshape(-1, 3)
    kd = cKDTree(road_xy)
    d0, j = kd.query(P[:, :2])
    away = P[:, :2] - road_xy[j]; away /= np.linalg.norm(away, axis=1, keepdims=True) + 1e-9
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    dots = (fn[:, None, :2] * away[T][:, :, :]).sum(-1).mean(1)
    back = np.unique(T[dots > 0.6].ravel())
    if not len(back):
        return 0
    key = lambda Q: [tuple(r) for r in np.round(Q, 4)]
    disp = {}
    for kk, i in zip(key(P[back]), back):
        disp.setdefault(kk, away[i])
    moved = np.zeros(len(P), bool); Pn = P.copy()
    for i, kk in enumerate(key(P)):
        a = disp.get(kk)
        if a is not None:
            Pn[i, :2] = P[i, :2] + a * depth; moved[i] = True
    dn, _ = kd.query(Pn[:, :2])
    bad = moved & (dn < d0 + 0.5 * depth)
    for i, kk in enumerate(key(P)):
        if bad[i]:
            Pn[i, :2] = P[i, :2] + disp[kk] * min_depth
    m["P"] = Pn
    fn = np.cross(Pn[T[:, 1]] - Pn[T[:, 0]], Pn[T[:, 2]] - Pn[T[:, 0]]); a = np.linalg.norm(fn, axis=1)
    keep = a > 1e-9; T = T[keep]; fn = fn[keep] / a[keep, None]
    # re-expand to own vertices per triangle so every normal is the flat face normal after the move
    m["P"] = Pn[T.ravel()]; m["T"] = np.arange(len(T) * 3).reshape(-1, 3)
    m["N"] = np.repeat(fn, 3, axis=0); m["UV"] = np.zeros((len(m["P"]), 2))
    return int(moved.sum())


def prism(cx, cy, r, z0, z1, sides=8):
    """closed vertical n-sided prism (tree trunk collider), flat outward normals, own vertices per face"""
    a = np.arange(sides) * 2 * np.pi / sides
    ring = np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]
    P, T, N = [], [], []
    def tri(p, q, s, nrm):
        k = len(P); P.extend([p, q, s]); T.append([k, k + 1, k + 2]); N.extend([nrm] * 3)
    for i in range(sides):
        j = (i + 1) % sides
        b0, b1 = (*ring[i], z0), (*ring[j], z0); t0, t1 = (*ring[i], z1), (*ring[j], z1)
        nm = (a[i] + a[j]) / 2 if j else a[i] + np.pi / sides
        nrm = (np.cos(nm), np.sin(nm), 0.0)
        tri(b0, b1, t1, nrm); tri(b0, t1, t0, nrm)          # CCW seen from outside (ring is CCW from above)
        tri((cx, cy, z1), t0, t1, (0, 0, 1.0))               # top
        tri((cx, cy, z0), b1, b0, (0, 0, -1.0))              # bottom
    return np.array(P, float), np.array(T, np.int64), np.array(N, float)
