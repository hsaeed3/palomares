#!/usr/bin/env python3
"""dev6 step 05b: fence / wall / guardrail lines detected from LiDAR non-ground low points (class 1) within 40 m.

Per 0.5 m cell: hits in the barrier band (0.45-2.0 m above ground) vs the bush bands (0.2-0.4 m and 2.2-3.5 m).
Cells with barrier-band hits and few bush hits are joined (closing), and each component is tested for
LINEARITY (PCA: length >= 8 m, minor-axis RMS <= 0.35 m) and split where it bends (recursive split, 0.4 m tolerance).
NAIP is used to reject green lines (NDVI > 0.35 = clipped hedge -> written as cls=hedge instead).
Class: guardrail = median HAG <= 1.0 and within 4 m of the pavement edge; wall = high density and solid (>= 30% cells
with >= 4 hits per m along); else fence. TYPE (chainlink / wood / wire...) is NOT measurable here; ledger rows snapped
to a line keep their eyeballed type.
-> data/derived/fences_detected.geojson (UTM 10N LineStrings, props: cls, length_m, hag_med, s0, s1, side, off_med, ndvi)
"""
import sys, json, glob, pathlib, math
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import DER, CACHE
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import rasterio

RES = 0.5; BUF = 40.0
cl = np.genfromtxt(DER / "centerline.csv", delimiter=",", names=True)
CX, CY = cl["x_utm"], cl["y_utm"]; KD = cKDTree(np.c_[CX, CY]); TX = np.gradient(CX); TY = np.gradient(CY)
EL, ER = cl["edge_left_m"], cl["edge_right_m"]


def st(x, y):
    d, j = KD.query(np.c_[x, y]); dx, dy = x - CX[j], y - CY[j]
    return cl["s_m"][j] + dx * TX[j] + dy * TY[j], -TY[j] * dx + TX[j] * dy, j


def split(P, tol=0.4):
    """recursive polyline simplification of ordered points -> list of index ranges that are straight"""
    if len(P) < 3: return [(0, len(P) - 1)]
    a, b = P[0], P[-1]; v = b - a; L = np.hypot(*v) + 1e-9
    d = np.abs((P[:, 0] - a[0]) * v[1] - (P[:, 1] - a[1]) * v[0]) / L; k = int(np.argmax(d))
    if d[k] <= tol or k in (0, len(P) - 1): return [(0, len(P) - 1)]
    return split(P[:k + 1], tol) + [(i + k, j + k) for i, j in split(P[k:], tol)]


def rail_profiles(x, y, h, s, off, j, n0):
    """guardrails: per side, a lateral peak of 0.35-0.95 m returns at a constant offset from the pavement edge
    (0.25 m bins, -0.3..+3.5 m beyond the edge) that is >= 2.5x the bins 0.75 m either side, sustained >= 10 m."""
    out = []
    for side, sg in (("L", 1), ("R", -1)):
        edge = np.where(sg > 0, EL[j], ER[j]); b = sg * off - edge
        m = (b > -0.3) & (b < 3.5) & (h >= 0.35) & (h <= 0.95) & (np.sign(off) == sg)
        si = np.floor(s[m] / 2.0).astype(int); bi = np.floor((b[m] + 0.3) / 0.25).astype(int)
        ns = int(s.max() // 2) + 2; nb = 16
        Hh = np.zeros((ns, nb)); np.add.at(Hh, (si, np.clip(bi, 0, nb - 1)), 1)
        Hs = ndi.uniform_filter1d(Hh, 3, axis=0)                     # 6 m along
        pk = Hs.argmax(1); pv = Hs.max(1)
        side_v = np.array([np.mean([Hs[i, max(pk[i] - 3, 0)], Hs[i, min(pk[i] + 3, nb - 1)]]) for i in range(ns)])
        ok = (pv >= 2.0) & (pv >= 2.5 * side_v + 0.3)
        lab, n = ndi.label(ok)
        for sl in ndi.find_objects(lab):
            i0, i1 = sl[0].start, sl[0].stop
            if (i1 - i0) * 2.0 < 10: continue
            bo = float(np.median(pk[i0:i1])) * 0.25 - 0.3 + 0.125
            s0, s1 = i0 * 2.0, i1 * 2.0
            k0, k1 = int(min(s0, len(CX) - 1)), int(min(s1, len(CX) - 1))
            pts = []
            for kk in range(k0, k1 + 1, 2):
                e = EL[kk] if sg > 0 else ER[kk]; nx, ny = -TY[kk], TX[kk]; nn = math.hypot(nx, ny)
                pts.append([round(CX[kk] + sg * nx / nn * (e + bo), 2), round(CY[kk] + sg * ny / nn * (e + bo), 2)])
            out.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=pts),
                            properties=dict(id=f"fd_{n0+len(out)+1:04d}", cls="guardrail", length_m=round(s1 - s0, 1), hag_med=0.65,
                                            s0=s0, s1=s1, side=side, off_beyond_edge=round(bo, 2), detector="lateral_profile",
                                            peak=round(float(np.median(pv[i0:i1])), 1), src="lidar2021")))
    return out


def main():
    xs, ys, hs = [], [], []
    for f in sorted(glob.glob(str(CACHE / "p6/nodes/*.npz"))):
        d = np.load(f); xs.append(d["lx"]); ys.append(d["ly"]); hs.append(d["lh"])
    x, y, h = np.concatenate(xs), np.concatenate(ys), np.concatenate(hs).astype(float)
    s, off, j = st(x, y); m = np.abs(off) <= BUF; x, y, h, s, off, j = x[m], y[m], h[m], s[m], off[m], j[m]
    naip = rasterio.open(DER / "naip2022_r060.tif")
    feats = []
    for c0 in np.arange(0, s.max() + 1, 250.0):
        k = (s >= c0 - 15) & (s < c0 + 265)
        if k.sum() < 50: continue
        xx, yy, hh = x[k], y[k], h[k]
        gx0, gy0 = xx.min() - 2, yy.min() - 2; W = int((xx.max() - gx0) / RES) + 3; H = int((yy.max() - gy0) / RES) + 3
        ci = ((xx - gx0) / RES).astype(int); ri = ((yy - gy0) / RES).astype(int); idx = ri * W + ci
        band = np.bincount(idx[(hh >= 0.45) & (hh <= 2.0)], minlength=W * H).reshape(H, W)
        bush = np.bincount(idx[((hh >= 0.2) & (hh < 0.4)) | ((hh > 2.2) & (hh <= 3.5))], minlength=W * H).reshape(H, W)
        bush_n = ndi.uniform_filter(bush.astype(float), 5) * 25; band_n = ndi.uniform_filter(band.astype(float), 5) * 25
        cand = (band >= 1) & (bush_n < 0.6 * band_n + 1)
        cand = ndi.binary_closing(cand, np.ones((3, 3)))
        lab, n = ndi.label(cand, np.ones((3, 3)))
        for li, sl in enumerate(ndi.find_objects(lab), 1):
            sub = lab[sl] == li
            if sub.sum() < 12: continue
            rr, cc = np.nonzero(sub); px = gx0 + (cc + sl[1].start + 0.5) * RES; py = gy0 + (rr + sl[0].start + 0.5) * RES
            P = np.c_[px, py]; mu = P.mean(0); U, S_, Vt = np.linalg.svd(P - mu, full_matrices=False)
            t = (P - mu) @ Vt[0]; order = np.argsort(t); P = P[order]; t = t[order]
            # order along main axis, smooth -> polyline at 1 m
            nb = max(2, int((t[-1] - t[0]) / 1.0))
            bins = np.linspace(t[0], t[-1] + 1e-6, nb + 1); bi = np.clip(np.digitize(t, bins) - 1, 0, nb - 1)
            cnt = np.bincount(bi, minlength=nb); ok = cnt > 0
            Q = np.c_[np.bincount(bi, P[:, 0], nb)[ok] / cnt[ok], np.bincount(bi, P[:, 1], nb)[ok] / cnt[ok]]
            if len(Q) < 6: continue
            for a, b in split(Q, 0.6):
                seg = Q[a:b + 1]
                if len(seg) < 2: continue
                Lg = float(np.hypot(*(seg[-1] - seg[0])))
                if Lg < 8: continue
                v = (seg[-1] - seg[0]) / Lg; nrm = np.array([-v[1], v[0]])
                # members of this segment: cells within 0.9 m of the chord and inside its span
                tt = (P - seg[0]) @ v; dd = (P - seg[0]) @ nrm; mm = (tt >= -0.5) & (tt <= Lg + 0.5) & (np.abs(dd) < 0.9)
                if mm.sum() < 8: continue
                rms = float(np.sqrt(np.mean(dd[mm] ** 2)))
                fill = len(np.unique(np.floor(tt[mm]).astype(int))) / max(Lg, 1)
                if rms > 0.35 or fill < 0.45: continue
                A, B = seg[0], seg[-1]
                ss, oo, jj = st(np.array([A[0], B[0], (A[0] + B[0]) / 2]), np.array([A[1], B[1], (A[1] + B[1]) / 2]))
                if not (c0 <= ss[2] < c0 + 250): continue
                # points near the segment for height stats
                near = (np.abs((np.c_[xx, yy] - A) @ nrm) < 0.5) & (((np.c_[xx, yy] - A) @ v) >= 0) & (((np.c_[xx, yy] - A) @ v) <= Lg)
                hm = float(np.median(hh[near])) if near.any() else 1.2; dens = near.sum() / Lg
                edge = np.where(oo[2] > 0, EL[jj[2]], ER[jj[2]]); beyond = abs(oo[2]) - edge
                if beyond < -0.5: continue                                    # on the pavement: vehicles
                win = rasterio.windows.from_bounds(min(A[0], B[0]) - 1, min(A[1], B[1]) - 1, max(A[0], B[0]) + 1, max(A[1], B[1]) + 1, naip.transform)
                a_ = naip.read([1, 4], window=win, boundless=True, fill_value=0).astype(float); okp = a_.sum(0) > 0
                ndvi = float(np.median(((a_[1] - a_[0]) / (a_[1] + a_[0] + 1e-6))[okp])) if okp.any() else 0.0
                if ndvi > 0.35: c_ = "hedge"
                elif hm <= 1.0 and beyond <= 4.0: c_ = "guardrail"
                elif dens >= 25 and hm >= 0.9: c_ = "wall"
                else: c_ = "fence"
                feats.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=[[round(A[0], 2), round(A[1], 2)], [round(B[0], 2), round(B[1], 2)]]),
                                  properties=dict(id=f"fd_{len(feats)+1:04d}", cls=c_, length_m=round(Lg, 1), hag_med=round(hm, 2), pts_per_m=round(float(dens), 1),
                                                  rms_m=round(rms, 2), fill=round(fill, 2), s0=round(float(min(ss[:2])), 1), s1=round(float(max(ss[:2])), 1),
                                                  side="L" if oo[2] > 0 else "R", off_beyond_edge=round(float(beyond), 2),
                                                  angle_to_road_deg=round(float(np.degrees(np.arccos(min(1, abs(v @ np.array([TX[jj[2]], TY[jj[2]]]) / math.hypot(TX[jj[2]], TY[jj[2]])))))), 1),
                                                  ndvi=round(ndvi, 2), src="lidar2021+naip2022")))
    feats += rail_profiles(x, y, h, s, off, j, len(feats))
    json.dump(dict(type="FeatureCollection", crs=dict(type="name", properties=dict(name="EPSG:26910")), features=feats), open(DER / "fences_detected.geojson", "w"))
    import collections
    c = collections.Counter(f["properties"]["cls"] for f in feats)
    print(len(feats), "lines", dict(c), "total m", round(sum(f["properties"]["length_m"] for f in feats)))


if __name__ == "__main__":
    main()
