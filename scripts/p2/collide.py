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
            out.append(dict(name=f"{self.prefix}_{k}" if (k or len(T) > per) else self.prefix, material="invisible", P=P[used], T=inv.reshape(-1, 3),
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
    return n
