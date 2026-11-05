"""Refine the OSM centerline onto the real road surface.
Pass A: in OSM-aligned rasters, Viterbi search for the offset whose 5.5 m window is the flattest plane.
Pass B: re-straighten along pass-A line and repeat with a narrow search for precision.
Output: smooth, curvature-continuous centerline resampled at 1 m."""
import numpy as np
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.interpolate import splprep, splev
from common import *
from straight import straighten, frame

HALF = 2.75          # half of minimum road width used for the flatness window

def block_profiles(R, blk=8, step=4):
    z, n = R["z"], R["n"].astype(float)
    ns = z.shape[0]; rows = []
    for i0 in range(0, ns - blk + 1, step):
        zz = z[i0:i0 + blk]; nn = n[i0:i0 + blk]
        w = nn.sum(0); zm = np.where(w > 0, np.nansum(np.where(nn > 0, zz * nn, 0), 0) / np.maximum(w, 1), np.nan)
        rows.append((i0 + blk / 2, zm, w))
    return rows

def cost_matrix(R, rows):
    dd = R["dd"]; nd = R["z"].shape[1]; d = (np.arange(nd) + .5) * dd - R["dm"]
    win = int(2 * HALF / dd) | 1
    C = np.zeros((len(rows), nd)); S = np.zeros((len(rows), nd))
    for r, (_, zm, w) in enumerate(rows):
        wt = (w > 0).astype(float); z0 = np.where(wt > 0, zm, 0)
        zref = np.nanmedian(zm) if np.isfinite(zm).any() else 0
        z0 = np.where(wt > 0, z0 - zref, 0)
        f = lambda a: uniform_filter1d(a, win, mode="constant") * win
        S1, Sd, Sdd, Sz, Sdz, Szz = f(wt), f(wt * d), f(wt * d * d), f(wt * z0), f(wt * d * z0), f(wt * z0 * z0)
        det = S1 * Sdd - Sd ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            b = (S1 * Sdz - Sd * Sz) / det; a = (Sz - b * Sd) / S1
            sse = Szz - a * Sz - b * Sdz
            rms = np.sqrt(np.maximum(sse, 0) / S1)
        cover = S1 / win
        c = np.where(cover > 0.35, rms, 0.08) + 0.6 * np.clip(np.abs(b) - 0.10, 0, None) + 0.03 * (1 - np.clip(cover, 0, 1))
        C[r] = np.nan_to_num(c, nan=0.2); S[r] = np.nan_to_num(b)
    return C, d

def viterbi(C, d, lam, maxjump, prior_w=0.0):
    n, m = C.shape; D = np.abs(d[:, None] - d[None, :])
    trans = lam * D ** 2; trans[D > maxjump] = 1e9
    acc = C[0] + prior_w * np.abs(d); bp = np.zeros((n, m), int)
    for r in range(1, n):
        t = acc[None, :] + trans; bp[r] = t.argmin(1); acc = t.min(1) + C[r] + prior_w * np.abs(d)
    path = np.zeros(n, int); path[-1] = acc.argmin()
    for r in range(n - 1, 0, -1): path[r - 1] = bp[r, path[r]]
    return d[path]

def apply_offset(xy, R, rows, off):
    ss, P, T = frame(xy, 0.5)
    s_b = np.array([r[0] for r in rows]) * R["ds"]
    o = np.interp(ss, s_b, off)
    N = np.c_[-T[:, 1], T[:, 0]]
    return P + o[:, None] * N

def smooth_resample(xy, step=1.0, sigma=6.0):
    seg = np.hypot(*np.diff(xy, axis=0).T); s = np.r_[0, np.cumsum(seg)]
    ss = np.arange(0, s[-1], 0.5)
    P = np.c_[np.interp(ss, s, xy[:, 0]), np.interp(ss, s, xy[:, 1])]
    P[1:-1] = np.c_[gaussian_filter1d(P[:, 0], sigma / 0.5, mode="nearest"), gaussian_filter1d(P[:, 1], sigma / 0.5, mode="nearest")][1:-1]
    tck, u = splprep(P.T, s=len(P) * 0.02 ** 2, k=5)
    uu = np.linspace(0, 1, 20000); Q = np.array(splev(uu, tck)).T
    sq = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]
    so = np.arange(0, sq[-1], step); ui = np.interp(so, sq, uu)
    return so, np.array(splev(ui, tck)).T, tck, ui

if __name__ == "__main__":
    o = origin()
    osm = np.loadtxt(DER / "osm_centerline_utm.csv", delimiter=",", skiprows=1)[:, 1:3] - [o["x0"], o["y0"]]
    RA = straighten(osm, DM=16.0); rows = block_profiles(RA)
    C, d = cost_matrix(RA, rows)
    offA = viterbi(C, d, lam=0.004, maxjump=1.0, prior_w=0.0015)
    offA = gaussian_filter1d(offA, 3)
    xyA = apply_offset(osm, RA, rows, offA)
    _, xyA, _, _ = smooth_resample(xyA, sigma=5.0)
    RB = straighten(xyA, DM=10.0); rowsB = block_profiles(RB)
    CB, dB = cost_matrix(RB, rowsB)
    offB = viterbi(CB, dB, lam=0.01, maxjump=0.5, prior_w=0.004)
    offB = gaussian_filter1d(offB, 4)
    xyB = apply_offset(xyA, RB, rowsB, offB)
    s, xy, tck, ui = smooth_resample(xyB, sigma=4.0)
    np.savez(CACHE / "centerline_refine_debug.npz", offA=offA, offB=offB, sA=np.array([r[0] for r in rows]) * .5,
             sB=np.array([r[0] for r in rowsB]) * .5, CA=C, dA=d)
    np.save(CACHE / "centerline_xy_local.npy", xy)
    print("refined length", s[-1], "pts", len(xy), "offA range", offA.min(), offA.max(), "offB", offB.min(), offB.max())
