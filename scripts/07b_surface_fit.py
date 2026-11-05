"""High-precision road surface from RAW LiDAR ground returns (no binning).
Per 1 m station, fit z = a + b*u + cl*max(d,0) + cr*min(d,0) over |u|<=1.25 m along and the drivable width
(crowned two-plane model, crown on the centerline; a single superelevated plane is the case cl == cr), with
iterative 2.5-sigma rejection. Replaces z / crossfall / grade in centerline.csv and adds fit diagnostics."""
import csv, json, numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from common import *

o = origin()
rows = list(csv.DictReader(open(DER / "centerline.csv"))); hdr = list(rows[0].keys())
S = np.array([float(r["s_m"]) for r in rows]); XY = np.array([[float(r["x_local"]), float(r["y_local"])] for r in rows])
WL = np.array([float(r["edge_left_m"]) for r in rows]); WR = np.array([float(r["edge_right_m"]) for r in rows])
T = np.gradient(XY, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
tree = cKDTree(XY)
idx = json.load(open(CACHE / "chunks" / "index.json"))
ps, pd, pz = [], [], []
for k in range(idx["n"]):
    d = np.load(CACHE / "chunks" / f"chunk_{k:03d}.npz")
    g = (d["c"] == 2) | (d["c"] == 17)
    x, y, z = d["x"][g].astype(np.float64), d["y"][g].astype(np.float64), d["z"][g].astype(np.float64)
    dist, j = tree.query(np.c_[x, y], distance_upper_bound=8)
    ok = np.isfinite(dist); j = j[ok]; x, y, z = x[ok], y[ok], z[ok]
    rel = np.c_[x - XY[j, 0], y - XY[j, 1]]
    ps.append(S[j] + (rel * T[j]).sum(1)); pd.append(T[j, 0] * rel[:, 1] - T[j, 1] * rel[:, 0]); pz.append(z)
ps, pd, pz = np.concatenate(ps), np.concatenate(pd), np.concatenate(pz)
o_ = np.argsort(ps); ps, pd, pz = ps[o_], pd[o_], pz[o_]
N = len(S); zc = np.full(N, np.nan); cl = np.full(N, np.nan); cr = np.full(N, np.nan); gr = np.full(N, np.nan)
rms = np.full(N, np.nan); npt = np.zeros(N, int)
for i in range(N):
    a, b = np.searchsorted(ps, [S[i] - 1.25, S[i] + 1.25])
    u, dd, zz = ps[a:b] - S[i], pd[a:b], pz[a:b]
    m = (dd < WL[i] - 0.25) & (dd > -WR[i] + 0.25)
    u, dd, zz = u[m], dd[m], zz[m]
    if len(zz) < 12 or (dd > 0.5).sum() < 4 or (dd < -0.5).sum() < 4: continue
    A = np.c_[np.ones_like(u), u, np.maximum(dd, 0), np.minimum(dd, 0)]
    keep = np.ones(len(zz), bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], zz[keep], rcond=None)
        res = zz - A @ coef; sd = max(np.std(res[keep]), 0.01)
        nk = np.abs(res) < 2.5 * sd
        if (nk == keep).all(): break
        keep = nk
    zc[i], gr[i], cl[i], cr[i] = coef; rms[i] = np.std(res[keep]); npt[i] = keep.sum()
bad = ~np.isfinite(zc)
for arr in (zc, cl, cr, gr, rms):
    arr[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), arr[~bad])
# light smoothing only (sigma 1.5 m) so real dips/crowns survive; grade from smoothed z
zc_s = gaussian_filter1d(zc, 1.5); cl_s = gaussian_filter1d(cl, 1.5); cr_s = gaussian_filter1d(cr, 1.5)
grade = np.gradient(zc_s, S)
old = np.array([float(r["z_navd88"]) for r in rows])
for i, r in enumerate(rows):
    r["z_navd88"] = f"{zc_s[i]:.3f}"; r["crossfall_left"] = f"{cl_s[i]:.4f}"; r["crossfall_right"] = f"{cr_s[i]:.4f}"
    r["grade"] = f"{grade[i]:.4f}"; r["fit_rms_m"] = f"{rms[i]:.3f}"; r["fit_npts"] = str(npt[i]); r["fit_interp"] = str(int(bad[i]))
with open(DER / "centerline.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
gj = json.load(open(DER / "centerline.geojson"))
gj["features"][0]["geometry"]["coordinates"] = [[c[0], c[1], round(float(z), 3)] for c, z in zip(gj["features"][0]["geometry"]["coordinates"], zc_s)]
json.dump(gj, open(DER / "centerline.geojson", "w"))
print(f"fitted {N-bad.sum()}/{N} stations; median pts {np.median(npt):.0f}; median rms {np.median(rms)*100:.1f} cm; "
      f"|dz| vs binned median {np.median(np.abs(zc_s-old))*100:.1f} cm max {np.abs(zc_s-old).max()*100:.0f} cm; "
      f"crossfall L/R median {np.median(cl_s)*100:.1f}% / {np.median(cr_s)*100:.1f}%; max grade {np.abs(grade).max()*100:.1f}%")
