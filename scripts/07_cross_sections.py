"""Per-metre cross-sections along the refined centerline:
 width (left/right edge offsets), crossfall left/right half, road surface z, edge-environment class,
 paint-stripe intensity score. Writes data/derived/centerline.csv + centerline.geojson."""
import json, numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
from common import *
from straight import straighten

o = origin()
xy = np.load(CACHE / "centerline_xy_local.npy")
f = CACHE / "straight_refined.npz"
R = dict(np.load(f)) if f.exists() else straighten(xy, DM=14.0)
z, n, I = R["z"].astype(float), R["n"].astype(float), R["i"].astype(float)
DS, DD, DM = float(R["ds"]), float(R["dd"]), float(R["dm"])
nd = z.shape[1]; d = (np.arange(nd) + .5) * DD - DM
seg = np.hypot(*np.diff(xy, axis=0).T); S = np.r_[0, np.cumsum(seg)]
NS = len(xy)

def prof(i0, i1):
    zz, nn = z[i0:i1], n[i0:i1]
    w = nn.sum(0)
    zm = np.where(w > 0, np.nansum(np.where(nn > 0, zz * nn, 0), 0) / np.maximum(w, 1), np.nan)
    im = np.where(w > 0, np.nansum(np.where(nn > 0, I[i0:i1] * nn, 0), 0) / np.maximum(w, 1), np.nan)
    return zm, w, im

def fitline(dd, zz):
    m = np.isfinite(zz)
    if m.sum() < 4: return None
    A = np.c_[np.ones(m.sum()), dd[m]]
    return np.linalg.lstsq(A, zz[m], rcond=None)[0]

TOL, MAXGAP = 0.06, 0.75
out = {k: np.full(NS, np.nan) for k in ("zc", "wl", "wr", "cfl", "cfr", "rise_l2", "rise_l5", "rise_r2", "rise_r5", "cov", "paint_c", "paint_l", "paint_r")}
for k in range(NS):
    i = int(S[k] / DS); i0, i1 = max(0, i - 4), min(z.shape[0], i + 4)
    zm, w, im = prof(i0, i1)
    c = nd // 2
    out["cov"][k] = (w[c - 12:c + 12] > 0).mean()
    left = fitline(d[c:c + 10], zm[c:c + 10]); right = fitline(d[c - 10:c], zm[c - 10:c])
    both = fitline(d[c - 10:c + 10], zm[c - 10:c + 10])
    if both is None: continue
    out["zc"][k] = both[0]
    left = left if left is not None else both; right = right if right is not None else both
    out["cfl"][k], out["cfr"][k] = left[1], right[1]
    # grow edges: last bin within TOL of the extrapolated half-plane, allowing small gaps
    for side, ln, key in ((1, left, "wl"), (-1, right, "wr")):
        last, gap = 1.5, 0.0
        for j in range(8, int(9.5 / DD)):
            col = c + side * j if side > 0 else c - 1 - j
            if col < 0 or col >= nd: break
            dist = abs(d[col])
            if not np.isfinite(zm[col]):
                gap += DD
                if gap > MAXGAP: break
                continue
            if abs(zm[col] - (ln[0] + ln[1] * d[col])) < TOL: last, gap = dist, 0.0
            else:
                gap += DD
                if gap > 0.5: break
        out[key][k] = last
    # edge environment: height relative to road plane extrapolated, 2 m and 5 m beyond the edge
    for side, ln, key, nm in ((1, left, "wl", "l"), (-1, right, "wr", "r")):
        e = out[key][k]
        for off, tag in ((2.0, "2"), (5.0, "5")):
            dpos = side * (e + off)
            sel = (np.abs(d - dpos) < 0.75) & np.isfinite(zm)
            if sel.any():
                zedge = ln[0] + ln[1] * side * e
                out[f"rise_{nm}{tag}"][k] = np.nanmedian(zm[sel]) - zedge
    # paint: intensity ratio of narrow stripe vs surrounding asphalt
    road = (np.abs(d) < min(out["wl"][k], out["wr"][k]) - 0.3) & np.isfinite(im)
    if road.sum() > 6:
        base = np.nanmedian(im[road])
        cs = (np.abs(d) < 0.4) & np.isfinite(im)
        out["paint_c"][k] = np.nanmean(im[cs]) / base if cs.any() else np.nan
        for nm, key, sg in (("l", "wl", 1), ("r", "wr", -1)):
            es = (np.abs(d - sg * (out[key][k] - 0.4)) < 0.3) & np.isfinite(im)
            out[f"paint_{nm}"][k] = np.nanmean(im[es]) / base if es.any() else np.nan

# smoothing along s (robust)
def sm(a, med=9, sig=3):
    a = a.copy(); m = np.isfinite(a)
    a[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), a[m])
    return gaussian_filter1d(median_filter(a, med, mode="nearest"), sig)
wl, wr = sm(out["wl"], 21, 4), sm(out["wr"], 21, 4)
# physically plausible clamp for a 2-lane rural road
wl, wr = np.clip(wl, 2.4, 6.0), np.clip(wr, 2.4, 6.0)
cfl, cfr = sm(out["cfl"], 9, 3), sm(out["cfr"], 9, 3)
zc = sm(out["zc"], 5, 2)

def env(r2, r5):
    if not np.isfinite(r2) and not np.isfinite(r5): return "unknown"
    r2 = 0 if not np.isfinite(r2) else r2; r5 = r2 if not np.isfinite(r5) else r5
    if r2 > 1.5 or r5 > 3.5: return "wall_or_steep_cut" if r2 > 1.5 else "cut_bank"
    if r5 > 1.0: return "cut_bank"
    if r2 < -1.2 or r5 < -3.0: return "steep_drop" if r2 < -1.2 else "drop_off"
    if r5 < -1.0: return "drop_off"
    if r2 < -0.25 and r5 > r2 + 0.2: return "ditch"
    return "flat"
rl = [sm(out[k], 7, 2) for k in ("rise_l2", "rise_l5", "rise_r2", "rise_r5")]
envl = [env(a, b) for a, b in zip(rl[0], rl[1])]; envr = [env(a, b) for a, b in zip(rl[2], rl[3])]

T = np.gradient(xy, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
hd = np.unwrap(np.arctan2(T[:, 1], T[:, 0])); curv = np.gradient(hd) / np.gradient(S)
grade = np.gradient(zc) / np.gradient(S)
X, Y = xy[:, 0] + o["x0"], xy[:, 1] + o["y0"]
lon, lat = T_UTM_LL.transform(X, Y)
hdr = "s_m,x_local,y_local,z_navd88,x_utm,y_utm,lon,lat,width_m,edge_left_m,edge_right_m,crossfall_left,crossfall_right,grade,curvature_1pm,env_left,env_right,ground_cover,paint_center,paint_left,paint_right"
with open(DER / "centerline.csv", "w") as fo:
    fo.write(hdr + "\n")
    for k in range(NS):
        fo.write(f"{S[k]:.2f},{xy[k,0]:.3f},{xy[k,1]:.3f},{zc[k]:.3f},{X[k]:.3f},{Y[k]:.3f},{lon[k]:.7f},{lat[k]:.7f},{wl[k]+wr[k]:.2f},{wl[k]:.2f},{wr[k]:.2f},"
                 f"{cfl[k]:.4f},{cfr[k]:.4f},{grade[k]:.4f},{curv[k]:.5f},{envl[k]},{envr[k]},{out['cov'][k]:.2f},"
                 f"{out['paint_c'][k]:.3f},{out['paint_l'][k]:.3f},{out['paint_r'][k]:.3f}\n")
gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": "Palomares Road refined centerline", "length_m": float(S[-1]),
       "crs_note": "coordinates WGS84 lon/lat/NAVD88 m; see centerline.csv for UTM10N + local"},
       "geometry": {"type": "LineString", "coordinates": [[round(a, 7), round(b, 7), round(c, 3)] for a, b, c in zip(lon, lat, zc)]}}]}
json.dump(gj, open(DER / "centerline.geojson", "w"))
np.savez(CACHE / "xs_raw.npz", **out)
print("width", np.percentile(wl + wr, [0, 5, 50, 95, 100]), "max |grade|", np.abs(grade).max())
