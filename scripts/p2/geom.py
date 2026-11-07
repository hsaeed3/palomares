"""Shared Phase 2 geometry: centerline loading, road cross-section surface, rasters."""
import csv, json, math, pathlib
import numpy as np, yaml, rasterio
from scipy.ndimage import gaussian_filter1d

ROOT = pathlib.Path(__file__).resolve().parents[2]
DER = ROOT / "data/derived"
import os as _os
# PALOMARES_BUILD_DIR: run steps against a copy (e.g. build_quick/, docs/PIPELINE.md) without touching build/
BUILD = pathlib.Path(_os.environ.get("PALOMARES_BUILD_DIR", ROOT / "build"))
CFG = yaml.safe_load(open(ROOT / "config/layouts.yaml"))
SHOULDER_W = 0.8          # nominal default only; per-station widths: Road.sh_w_l / sh_w_r (edge_profile.py)


DEV7 = DER / "dev7"
_TEXT_COLS = ("env_left", "env_right", "struct")


def load_centerline(s_end=None, path=None):
    """Palomares refined centerline (default) or any centerline CSV with the same core columns, e.g. a dev7 road
    (data/derived/dev7/centerline_<road>.csv); text columns (env_*, struct) stay strings, empty numbers -> nan."""
    rows = list(csv.DictReader(open(path or (DER / "centerline.csv"))))
    num = {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0] if k not in _TEXT_COLS}
    for k in _TEXT_COLS:
        if k in rows[0]:
            num[k] = np.array([r[k] for r in rows])
    if s_end is not None:
        m = num["s_m"] <= s_end + 1e-6
        num = {k: v[m] for k, v in num.items()}
    return num


class Road:
    """Road surface as a function of station s and lateral offset d (d>0 = left looking uphill).

    Road(s_end)            : Palomares (data/derived/centerline.csv), as in every Phase 2 step.
    Road(None, name="x")   : a dev7 network road (data/derived/dev7/centerline_x.csv). Same smoothing / surface
                             model; half-widths clipped to edge_clip (default 2.0-12 m: Mission Blvd is up to 19 m
                             wide) and a uniform default edge profile instead of the Palomares GoPro-based one
                             (edge_profile.py reads Palomares-only survey data). See scripts/p7/network.py."""
    def __init__(self, s_end=None, name=None, path=None, edge_clip=None, z_override=None):
        self.name = name or "palomares"
        if name is not None and path is None:
            path = DEV7 / f"centerline_{name}.csv"
        c = load_centerline(s_end, path)
        if z_override is not None:          # network.py: profile fixes (bridge decks, grade limits) before smoothing
            c["z_navd88"] = np.asarray(z_override, float)
        self.raw = c
        self.s = c["s_m"]
        self.x = gaussian_filter1d(c["x_local"], 1.5, mode="nearest")
        self.y = gaussian_filter1d(c["y_local"], 1.5, mode="nearest")
        self.z = gaussian_filter1d(c["z_navd88"], 2.0, mode="nearest")
        # dev4: extra local smoothing where the grade changes > ~0.5 %/m (23_qa 3d spike at s=7866: centreline
        # 0.8 %/m + surface noise > 1 %/m); blends toward a sigma-5 profile only there
        for _ in range(60):
            _d2 = self.z[2:] - 2 * self.z[1:-1] + self.z[:-2]
            _bad = np.nonzero(np.abs(_d2) > 0.006)[0] + 1
            if not len(_bad):
                break
            _m = np.zeros(len(self.z), bool)
            for _o in (-2, -1, 0, 1, 2):
                _m[np.clip(_bad + _o, 1, len(self.z) - 2)] = True
            _zs = self.z.copy(); _zs[1:-1] = 0.25 * self.z[:-2] + 0.5 * self.z[1:-1] + 0.25 * self.z[2:]
            self.z = np.where(_m, _zs, self.z)
        lo, hi = edge_clip or ((2.4, 6.0) if path is None else (2.0, 12.0))
        self.el = gaussian_filter1d(np.clip(c["edge_left_m"], lo, hi), 6, mode="nearest")
        self.er = gaussian_filter1d(np.clip(c["edge_right_m"], lo, hi), 6, mode="nearest")
        self.cl = gaussian_filter1d(np.clip(c["crossfall_left"], -0.10, 0.10), 5, mode="nearest")
        self.cr = gaussian_filter1d(np.clip(c["crossfall_right"], -0.10, 0.10), 5, mode="nearest")
        tx, ty = np.gradient(self.x), np.gradient(self.y)
        n = np.hypot(tx, ty); self.tx, self.ty = tx / n, ty / n
        self.nx, self.ny = -self.ty, self.tx          # left normal
        h = np.unwrap(np.arctan2(self.ty, self.tx))
        self.k = gaussian_filter1d(np.gradient(h), 2, mode="nearest")   # signed curvature, + = left turn
        self.env_l, self.env_r = c["env_left"], c["env_right"]
        if path is None:
            import edge_profile   # per-station shoulder width / material / ditch / edge-line inset (scripts/p2/edge_profile.py)
            for k_, v_ in edge_profile.compute(self).items():
                setattr(self, k_, v_)
        else:                    # network roads: uniform defaults (network.py sets town curbs / canyon shoulders)
            n_ = len(self.s); z0 = np.zeros(n_)
            self.sh_w_l = np.full(n_, 0.6); self.sh_w_r = np.full(n_, 0.6)
            self.sh_mat_l = np.full(n_, "dirt", dtype=object); self.sh_mat_r = self.sh_mat_l.copy()
            self.ditch_l, self.ditch_r = z0.copy(), z0.copy()
            self.line_in_l = np.full(n_, 0.3); self.line_in_r = np.full(n_, 0.3)
            self.fresh = np.zeros(n_, bool)
            self.strip_in_l = np.full(n_, 0.12); self.strip_in_r = np.full(n_, 0.12)

    def sh_w(self, i, side):
        """shoulder width at station index i (array ok); side: +1/'left'/'L' or -1/'right'/'R'"""
        left = side in (1, "left", "L", "l") if not isinstance(side, np.ndarray) else None
        if left is None:
            return np.where(side > 0, self.sh_w_l[i], self.sh_w_r[i])
        return (self.sh_w_l if left else self.sh_w_r)[i]

    def outer(self, i, side):
        """distance from the centreline to the shoulder's outer edge (asphalt half-width + shoulder)"""
        if isinstance(side, np.ndarray):
            return np.where(side > 0, self.el[i] + self.sh_w_l[i], self.er[i] + self.sh_w_r[i])
        left = side in (1, "left", "L", "l")
        return (self.el[i] + self.sh_w_l[i]) if left else (self.er[i] + self.sh_w_r[i])

    def noise(self, s, d):
        # deterministic, low-amplitude (<= ~4 mm), long-wavelength (>= 3 m) surface undulation
        return (0.0022 * np.sin(s * 2 * np.pi / 7.3 + 1.3 * np.sin(d * 0.9))
                + 0.0015 * np.sin(s * 2 * np.pi / 3.1 + d * 1.7 + 0.4)
                + 0.0010 * np.sin(s * 2 * np.pi / 13.7 - d * 0.6))

    def surface_z(self, i, d, noise=True):
        """z at station index i (array ok) and offset d."""
        z = self.z[i] + np.where(d > 0, self.cl[i] * d, self.cr[i] * d)
        return z + (self.noise(self.s[i], d) if noise else 0)

    def xy(self, i, d):
        return self.x[i] + self.nx[i] * d, self.y[i] + self.ny[i] * d


class Raster:
    def __init__(self, path):
        with rasterio.open(path) as r:
            self.a = r.read(1).astype(np.float64); self.t = r.transform; self.nod = r.nodata
        if self.nod is not None:
            self.a[self.a == self.nod] = np.nan
        o = json.load(open(ROOT / "config/origin.json")); self.x0, self.y0 = o["x0"], o["y0"]

    def sample(self, xl, yl):
        """bilinear sample at local coords"""
        X = np.asarray(xl) + self.x0; Y = np.asarray(yl) + self.y0
        col = (X - self.t.c) / self.t.a - 0.5; row = (Y - self.t.f) / self.t.e - 0.5
        c0 = np.floor(col).astype(int); r0 = np.floor(row).astype(int); fc = col - c0; fr = row - r0
        H, W = self.a.shape
        ok = (c0 >= 0) & (r0 >= 0) & (c0 < W - 1) & (r0 < H - 1)
        c0 = np.clip(c0, 0, W - 2); r0 = np.clip(r0, 0, H - 2)
        a = self.a
        v = (a[r0, c0] * (1 - fc) * (1 - fr) + a[r0, c0 + 1] * fc * (1 - fr) + a[r0 + 1, c0] * (1 - fc) * fr + a[r0 + 1, c0 + 1] * fc * fr)
        return np.where(ok, v, np.nan)

    def sample1(self, xl, yl):
        """scalar bilinear sample (same arithmetic as sample(), no numpy overhead)"""
        col = (xl + self.x0 - self.t.c) / self.t.a - 0.5; row = (yl + self.y0 - self.t.f) / self.t.e - 0.5
        c0 = math.floor(col); r0 = math.floor(row); fc = col - c0; fr = row - r0
        H, W = self.a.shape
        if not (0 <= c0 < W - 1 and 0 <= r0 < H - 1):
            return float("nan")
        a = self.a
        return float(a[r0, c0] * (1 - fc) * (1 - fr) + a[r0, c0 + 1] * fc * (1 - fr) + a[r0 + 1, c0] * (1 - fc) * fr + a[r0 + 1, c0 + 1] * fc * fr)


def vertex_normals(P, T):
    N = np.zeros_like(P)
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    n = np.linalg.norm(N, axis=1, keepdims=True); n[n == 0] = 1
    return N / n


def grid_tris(nrow, ncol, offset=0):
    """Quads of an (nrow along s) x (ncol across) grid, CCW seen from +z when columns go right->left
    (d increasing = toward left) and rows go uphill."""
    r, c = np.meshgrid(np.arange(nrow - 1), np.arange(ncol - 1), indexing="ij")
    a = r * ncol + c; b = a + 1; cc = a + ncol; dd = cc + 1
    # rows = forward, cols = left: (a, dd, b) and (a, cc, dd) are CCW seen from +z (QA checks normals)
    T = np.concatenate([np.stack([a, dd, b], -1).reshape(-1, 3), np.stack([a, cc, dd], -1).reshape(-1, 3)])
    return T + offset


def build_lock(name="build"):
    """Serialise pipeline steps that write build/ (21*, 22, 25): concurrent runs clobbered each other's outputs.
    Blocks until the lock is free; held for the life of the process."""
    import fcntl, os, sys, time
    BUILD.mkdir(exist_ok=True)
    fh = open(BUILD / f".{name}.lock", "w")
    t0 = time.time()
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB); break
        except BlockingIOError:
            if time.time() - t0 < 1:
                print(f"[lock] waiting for another pipeline step holding build/.{name}.lock ...", file=sys.stderr)
            time.sleep(5)
    fh.write(f"{os.getpid()} {' '.join(sys.argv)}\n"); fh.flush()
    globals()["_LOCK_FH"] = fh
    return fh
