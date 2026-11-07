#!/usr/bin/env python3
"""dev7 D-04: refined centerline + per-metre cross-sections for every dev7 road (Phase-1 method, scripts/06 + 07):
  pass A/B Viterbi search for the flattest 5.5 m strip in road-aligned DTM rasters (dtm_r050_b40), then a curvature-
  continuous quintic spline resampled at 1 m; cross-sections from the DTM (half-plane crossfall, edge growth with
  6 cm tolerance, curb-step detection for town streets) and the ground-intensity raster (paint stripes, asphalt).
  Roads shorter than 40 m are only smoothed (no snap).
Writes data/derived/dev7/centerline_<road>.csv + .geojson, and data/cache/p7/xs_<road>.npz (raw profiles).
Columns: s_m,x_local,y_local,z_navd88,x_utm,y_utm,lon,lat,width_m,edge_left_m,edge_right_m,crossfall_left,
 crossfall_right,grade,curvature_1pm,env_left,env_right,curb_left,curb_right,rise_left_5m,rise_right_5m,
 ground_cover,gdist_m,canopy_frac,bridge17,paint_center,paint_left,paint_right,snap_offset_m
Left/right relative to the line direction (NCR: Mission Blvd -> Sunol; others W->E / S->N); crossfall = dz/dd, d + left.
Run: nice -n5 python3 scripts/p7/04_centerlines.py [road ...]
"""
import sys, json, glob, importlib.util
import numpy as np, rasterio
from scipy.ndimage import map_coordinates, median_filter, gaussian_filter1d
from common7 import *

spec = importlib.util.spec_from_file_location("p1refine", str(ROOT / "scripts" / "06_refine_centerline.py"))
P1 = importlib.util.module_from_spec(spec); sys.path.insert(0, str(ROOT / "scripts")); spec.loader.exec_module(P1)

RS = {k: rasterio.open(DER7 / f"{k}.tif") for k in ("dtm_r050_b40", "int_r050_b40", "gdist_r050_b40", "chm_r100_b150", "cover_r100_b150", "dtm_r100_b150")}
ARR = {k: (r.read(1), ~r.transform) for k, r in RS.items()}
ARR["b17"] = (RS["cover_r100_b150"].read(3).astype(np.float32), ~RS["cover_r100_b150"].transform)
net = json.load(open(DER7 / "network.geojson"))["features"]


def samp(key, x, y, order=1):
    a, inv = ARR[key]; c, r = inv * (np.asarray(x) + X0, np.asarray(y) + Y0)
    return map_coordinates(a, [np.asarray(r) - .5, np.asarray(c) - .5], order=order, mode="constant", cval=np.nan)


def frame(xy, step):
    seg = np.hypot(*np.diff(xy, axis=0).T); s = np.r_[0, np.cumsum(seg)]
    ss = np.arange(0, s[-1], step)
    P = np.c_[np.interp(ss, s, xy[:, 0]), np.interp(ss, s, xy[:, 1])]
    T = np.gradient(P, axis=0); T /= np.maximum(np.linalg.norm(T, axis=1), 1e-9)[:, None]
    return ss, P, T


def straighten(xy, DM, DS=0.5, DD=0.25):
    ss, P, T = frame(xy, DS); N = np.c_[-T[:, 1], T[:, 0]]
    d = (np.arange(int(2 * DM / DD)) + .5) * DD - DM
    X = P[:, None, 0] + d[None] * N[:, None, 0]; Y = P[:, None, 1] + d[None] * N[:, None, 1]
    z = samp("dtm_r050_b40", X, Y); gd = samp("gdist_r050_b40", X, Y, 0); it = samp("int_r050_b40", X, Y, 0)
    n = (np.isfinite(z) & (np.nan_to_num(gd, nan=9) < 1.5)).astype(np.uint16)
    return {"z": np.where(n > 0, z, np.nan), "n": n, "i": it, "ds": DS, "dd": DD, "dm": DM, "zall": z}


def refine(xy):
    if np.hypot(*np.diff(xy, axis=0).T).sum() < 40:
        s, xyo, _, _ = P1.smooth_resample(xy, sigma=2.0); return s, xyo, np.zeros(len(s))
    RA = straighten(xy, 14.0); rows = P1.block_profiles(RA)
    C, d = P1.cost_matrix(RA, rows)
    offA = gaussian_filter1d(P1.viterbi(C, d, lam=0.004, maxjump=1.0, prior_w=0.003), 3)
    xyA = P1.apply_offset(xy, RA, rows, offA); _, xyA, _, _ = P1.smooth_resample(xyA, sigma=5.0)
    RB = straighten(xyA, 8.0); rowsB = P1.block_profiles(RB)
    CB, dB = P1.cost_matrix(RB, rowsB)
    offB = gaussian_filter1d(P1.viterbi(CB, dB, lam=0.01, maxjump=0.5, prior_w=0.006), 4)
    xyB = P1.apply_offset(xyA, RB, rowsB, offB)
    s, xyo, _, _ = P1.smooth_resample(xyB, sigma=4.0)
    # snap offset vs OSM
    from shapely.geometry import LineString, Point
    L = LineString(xy); off = np.array([L.distance(Point(p)) for p in xyo[::1]])
    return s, xyo, off


def fitline(dd, zz):
    m = np.isfinite(zz)
    if m.sum() < 4: return None
    return np.linalg.lstsq(np.c_[np.ones(m.sum()), dd[m]], zz[m], rcond=None)[0]


def sm(a, med=9, sig=3):
    a = a.copy(); m = np.isfinite(a)
    if not m.any(): return a
    a[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), a[m])
    return gaussian_filter1d(median_filter(a, med, mode="nearest"), sig)


def env(r2, r5):
    if not np.isfinite(r2) and not np.isfinite(r5): return "unknown"
    r2 = 0 if not np.isfinite(r2) else r2; r5 = r2 if not np.isfinite(r5) else r5
    if r2 > 1.5 or r5 > 3.5: return "wall_or_steep_cut" if r2 > 1.5 else "cut_bank"
    if r5 > 1.0: return "cut_bank"
    if r2 < -1.2 or r5 < -3.0: return "steep_drop" if r2 < -1.2 else "drop_off"
    if r5 < -1.0: return "drop_off"
    if r2 < -0.25 and r5 > r2 + 0.2: return "ditch"
    return "flat"


def max_half(road):
    """OSM-informed cap on half width (m) for edge growth."""
    lanes = []
    for f in net:
        p = f["properties"]
        nm = slug(p.get("name") or "")
        if nm and (nm == road or road.startswith(nm + "_")) and p["highway"] != "service":
            try: lanes.append(float(p.get("lanes") or 2))
            except ValueError: lanes.append(2)
    ln = max(lanes) if lanes else 2
    return float(np.clip(ln * 1.9 + 1.6, 4.5, 12.0))


def cross_sections(road, xy, s, snap):
    NS = len(xy); DMX = max_half(road) + 7
    R = straighten(xy, DMX, DS=1.0); z, I = R["z"], R["i"]; DD = R["dd"]
    nd = z.shape[1]; d = (np.arange(nd) + .5) * DD - DMX; c = nd // 2
    keys = ("zc", "wl", "wr", "cfl", "cfr", "rl2", "rl5", "rr2", "rr5", "cov", "pc", "pl", "pr", "curbl", "curbr")
    out = {k: np.full(NS, np.nan) for k in keys}
    MAXJ = int(max_half(road) / DD)
    for k in range(NS):
        i0, i1 = max(0, k - 2), min(z.shape[0], k + 3)
        with np.errstate(all="ignore"):
            zm = np.nanmean(z[i0:i1], 0); im = np.nanmean(I[i0:i1], 0)
        out["cov"][k] = np.isfinite(z[k, c - 12:c + 12]).mean()
        both = fitline(d[c - 10:c + 10], zm[c - 10:c + 10])
        if both is None: continue
        left = fitline(d[c:c + 10], zm[c:c + 10]); right = fitline(d[c - 10:c], zm[c - 10:c])
        left = left if left is not None else both; right = right if right is not None else both
        out["zc"][k] = both[0]; out["cfl"][k], out["cfr"][k] = left[1], right[1]
        for side, ln, key, ck in ((1, left, "wl", "curbl"), (-1, right, "wr", "curbr")):
            last, gap, curb = 1.5, 0.0, 0.0
            for j in range(6, MAXJ):
                col = c + j if side > 0 else c - 1 - j
                if col < 0 or col >= nd: break
                if not np.isfinite(zm[col]):
                    gap += DD
                    if gap > 0.75: break
                    continue
                dz = zm[col] - (ln[0] + ln[1] * d[col])
                if abs(dz) < 0.06: last, gap = abs(d[col]), 0.0
                else:
                    # curb: a 8-30 cm rise within 0.5 m that then stays up (sidewalk)
                    col2 = col + side * 3
                    if 0 <= col2 < nd and np.isfinite(zm[col2]):
                        dz2 = zm[col2] - (ln[0] + ln[1] * d[col2])
                        if 0.08 < dz2 < 0.30 and dz > 0.03: curb = dz2; break
                    gap += DD
                    if gap > 0.5: break
            out[key][k] = last; out[ck][k] = curb
        for side, ln, key, nm in ((1, left, "wl", "l"), (-1, right, "wr", "r")):
            e = out[key][k]
            for off, tag in ((2.0, "2"), (5.0, "5")):
                sel = (np.abs(d - side * (e + off)) < 0.75) & np.isfinite(zm)
                if sel.any(): out[f"r{nm}{tag}"][k] = np.nanmedian(zm[sel]) - (ln[0] + ln[1] * side * e)
        road_m = (np.abs(d) < min(out["wl"][k], out["wr"][k]) - 0.3) & np.isfinite(im)
        if road_m.sum() > 6:
            base = np.nanmedian(im[road_m])
            cs = (np.abs(d) < 0.4) & np.isfinite(im)
            out["pc"][k] = np.nanmean(im[cs]) / base if cs.any() else np.nan
            for nm, key, sg in (("l", "wl", 1), ("r", "wr", -1)):
                es = (np.abs(d - sg * (out[key][k] - 0.4)) < 0.3) & np.isfinite(im)
                out[f"p{nm}"][k] = np.nanmean(im[es]) / base if es.any() else np.nan
    np.savez(CACHE7 / f"xs_{road}.npz", **out)
    mh = max_half(road)
    wl, wr = np.clip(sm(out["wl"], 21, 4), 2.0, mh), np.clip(sm(out["wr"], 21, 4), 2.0, mh)
    cfl, cfr = sm(out["cfl"], 9, 3), sm(out["cfr"], 9, 3)
    zc = sm(out["zc"], 5, 2)
    zc = np.where(np.isfinite(zc), zc, samp("dtm_r100_b150", xy[:, 0], xy[:, 1]))
    rl = [sm(out[k], 7, 2) for k in ("rl2", "rl5", "rr2", "rr5")]
    envl = [env(a, b) for a, b in zip(rl[0], rl[1])]; envr = [env(a, b) for a, b in zip(rl[2], rl[3])]
    T = np.gradient(xy, axis=0); T /= np.maximum(np.linalg.norm(T, axis=1), 1e-9)[:, None]
    hd = np.unwrap(np.arctan2(T[:, 1], T[:, 0])); S = s
    curv = np.gradient(hd) / np.maximum(np.gradient(S), 1e-6); grade = np.gradient(zc) / np.maximum(np.gradient(S), 1e-6)
    gd = samp("gdist_r050_b40", xy[:, 0], xy[:, 1], 0)
    N = np.c_[-T[:, 1], T[:, 0]]
    canopy = np.nanmean([samp("chm_r100_b150", xy[:, 0] + f * N[:, 0], xy[:, 1] + f * N[:, 1], 0) > 3 for f in (-3, -1.5, 0, 1.5, 3)], 0)
    b17 = np.nansum([samp("b17", xy[:, 0] + f * N[:, 0], xy[:, 1] + f * N[:, 1], 0) for f in (-2, 0, 2)], 0)
    X, Y = xy[:, 0] + X0, xy[:, 1] + Y0; lon, lat = T_UTM_LL.transform(X, Y)
    hdr = ("s_m,x_local,y_local,z_navd88,x_utm,y_utm,lon,lat,width_m,edge_left_m,edge_right_m,crossfall_left,crossfall_right,grade,"
           "curvature_1pm,env_left,env_right,curb_left,curb_right,rise_left_5m,rise_right_5m,ground_cover,gdist_m,canopy_frac,bridge17,"
           "paint_center,paint_left,paint_right,snap_offset_m")
    with open(DER7 / f"centerline_{road}.csv", "w") as fo:
        fo.write(hdr + "\n")
        for k in range(NS):
            fo.write(f"{S[k]:.2f},{xy[k,0]:.3f},{xy[k,1]:.3f},{zc[k]:.3f},{X[k]:.3f},{Y[k]:.3f},{lon[k]:.7f},{lat[k]:.7f},{wl[k]+wr[k]:.2f},"
                     f"{wl[k]:.2f},{wr[k]:.2f},{cfl[k]:.4f},{cfr[k]:.4f},{grade[k]:.4f},{curv[k]:.5f},{envl[k]},{envr[k]},"
                     f"{out['curbl'][k]:.2f},{out['curbr'][k]:.2f},{rl[1][k]:.2f},{rl[3][k]:.2f},{out['cov'][k]:.2f},{gd[k]:.2f},{canopy[k]:.2f},"
                     f"{b17[k]:.0f},{out['pc'][k]:.3f},{out['pl'][k]:.3f},{out['pr'][k]:.3f},{snap[k]:.2f}\n")
    dump_geojson(DER7 / f"centerline_{road}.geojson", [feat(line_ll(xy, zc), road=road, length_m=round(float(S[-1]), 1),
                 crs_note="WGS84 lon/lat/NAVD88 m; CSV has UTM10N + local (config/origin.json)")])
    return dict(length_m=round(float(S[-1]), 1), width_p5_50_95=np.round(np.percentile(wl + wr, [5, 50, 95]), 2).tolist(),
                z_min=round(float(np.nanmin(zc)), 2), z_max=round(float(np.nanmax(zc)), 2), max_abs_grade=round(float(np.nanmax(np.abs(grade))), 4),
                snap_p50_p95=np.round(np.percentile(snap, [50, 95]), 2).tolist())


if __name__ == "__main__":
    only = sys.argv[1:]
    summ = {}
    sf = DER7 / "centerlines_summary.json"
    if sf.exists(): summ = json.load(open(sf))
    for f in sorted(glob.glob(str(DER7 / "osm_centerline_*.csv"))):
        road = f.split("osm_centerline_")[1][:-4]
        if only and road not in only: continue
        a = np.genfromtxt(f, delimiter=",", names=True); xy = np.c_[a["x_local"], a["y_local"]]
        keep = np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > 0.05]; xy = xy[keep]
        if len(xy) < 2: continue
        s, xyo, snap = refine(xy)
        summ[road] = cross_sections(road, xyo, s, snap)
        print(road, summ[road], flush=True)
    json.dump(summ, open(sf, "w"), indent=1)
