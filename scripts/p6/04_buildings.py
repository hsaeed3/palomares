#!/usr/bin/env python3
"""dev6 step 04: buildings measured from the 2021 USGS LiDAR + NAIP 2022.

The 2021 Alameda delivery has no class 6, so roofs are found as PLANAR patches of single-return class-1 points
(HAG >= 1.8 m) from step 01, then rejected if green in NAIP (NDVI) or too small/low. Per building:
  footprint   rasterised 0.5 m patch mask -> polygon (UTM), dominant orientation from edge directions
  parts       greedy rectangle decomposition of the oriented mask (L / T houses = 2-3 rectangles)
  roof        per part: least-squares fit of flat / shed / gable(ridge along or across) / hip -> eave & ridge z
  colour      NAIP median roof RGB (eroded footprint); NDVI
  ground      DTM 1 m 10th percentile under the footprint
Cross-check: OSM building polygons (osm_buildings_barriers.geojson) by IoU.
-> data/derived/buildings.geojson (UTM 10N, EPSG:26910; props also carry local coords and station/offset)
Run: nice -n5 python3 scripts/p6/04_buildings.py
"""
import sys, json, glob, pathlib, math
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import DER, CACHE, origin
import rasterio, rasterio.features
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from shapely.geometry import shape, Polygon, Point, mapping, box
from shapely import affinity

O = origin(); X0, Y0 = O["x0"], O["y0"]
CELL = 1.0          # planarity cell
RES = 0.5           # footprint raster


def load_pts():
    xs, ys, zs, hs = [], [], [], []
    for f in sorted(glob.glob(str(CACHE / "p6/nodes/*.npz"))):
        d = np.load(f)
        if len(d["bx"]):
            xs.append(d["bx"]); ys.append(d["by"]); zs.append(d["bz"]); hs.append(d["bh"])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs).astype(np.float64), np.concatenate(hs)


def planar_cells(x, y, z):
    """per 1 m cell plane fit (via sums); returns dict cell-> (ok) as boolean per point"""
    gx = np.floor(x / CELL).astype(np.int64); gy = np.floor(y / CELL).astype(np.int64)
    key = gx * 10_000_000 + gy; u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    cx = (gx + 0.5) * CELL; cy = (gy + 0.5) * CELL; dx = x - cx; dy = y - cy
    zm = np.bincount(inv, z) / cnt; dz = z - zm[inv]
    S = lambda a: np.bincount(inv, a, minlength=len(u))
    sxx, syy, sxy, sxz, syz = S(dx * dx), S(dy * dy), S(dx * dy), S(dx * dz), S(dy * dz)
    sx, sy = S(dx), S(dy)
    # centered normal equations for z - zm = b dx + c dy (approx, dx means ~0)
    sxx -= sx * sx / cnt; syy -= sy * sy / cnt; sxy -= sx * sy / cnt
    det = sxx * syy - sxy ** 2 + 1e-9
    b = (sxz * syy - syz * sxy) / det; c = (syz * sxx - sxz * sxy) / det
    r = dz - (b[inv] * (dx - (sx / cnt)[inv]) + c[inv] * (dy - (sy / cnt)[inv]))
    rms = np.sqrt(np.bincount(inv, r * r, minlength=len(u)) / cnt)
    slope = np.hypot(b, c)
    okc = (cnt >= 4) & (rms < 0.15) & (slope < 1.3) & (det > 0.05 * cnt ** 2 * 0.01)
    return okc[inv], cnt[inv]


def orient(poly):
    """dominant edge direction mod 90 deg (length weighted)"""
    c = np.asarray(poly.exterior.coords); e = np.diff(c, axis=0); L = np.hypot(e[:, 0], e[:, 1])
    a = np.mod(np.arctan2(e[:, 1], e[:, 0]), np.pi / 2)
    h, edges = np.histogram(a, bins=36, range=(0, np.pi / 2), weights=L)
    h = h + np.roll(h, 1) + np.roll(h, -1); k = int(np.argmax(h))
    return (edges[k] + edges[k + 1]) / 2


def max_rect(M):
    """largest all-true axis rectangle in bool matrix (histogram stack). returns (area, r0, r1, c0, c1) inclusive-exclusive"""
    best = (0, 0, 0, 0, 0); hgt = np.zeros(M.shape[1], int)
    for r in range(M.shape[0]):
        hgt = np.where(M[r], hgt + 1, 0); st = []
        for c in range(M.shape[1] + 1):
            h = hgt[c] if c < M.shape[1] else 0; start = c
            while st and st[-1][1] >= h:
                s0, hh = st.pop(); a = hh * (c - s0)
                if a > best[0]: best = (a, r - hh + 1, r + 1, s0, c)
                start = s0
            st.append((start, h))
    return best


def fit_roof(u, v, z, Lu, Lv):
    """u, v rect-local (centered), returns dict(form, ridge ('u'|'v'|None), r, k, rms, eave, ridge_z)"""
    out = []
    A = np.c_[np.ones_like(u), u, v]; cf, *_ = np.linalg.lstsq(A, z, rcond=None); res = z - A @ cf
    rms = float(np.sqrt(np.mean(res ** 2))); sl = float(np.hypot(cf[1], cf[2]))
    zc = float(cf[0]); hi = zc + abs(cf[1]) * Lu / 2 + abs(cf[2]) * Lv / 2; lo = zc - abs(cf[1]) * Lu / 2 - abs(cf[2]) * Lv / 2
    out.append(dict(form="flat" if sl < 0.08 else "shed", ridge=None, rms=rms, k=sl, eave=lo if sl >= 0.08 else zc, ridge_z=hi if sl >= 0.08 else zc,
                    shed_dir=[float(cf[1]), float(cf[2])], npar=3 if sl >= 0.08 else 1))
    for ax, d, hw in (("u", np.abs(v), Lv / 2), ("v", np.abs(u), Lu / 2)):
        A = np.c_[np.ones_like(d), -d]; cf, *_ = np.linalg.lstsq(A, z, rcond=None)
        k = float(np.clip(cf[1], 0, 1.3)); r = float(np.mean(z + k * d)); rms = float(np.sqrt(np.mean((z - (r - k * d)) ** 2)))
        out.append(dict(form="gable", ridge=ax, rms=rms, k=k, eave=r - k * hw, ridge_z=r, npar=2))
    if Lu >= Lv: d = np.maximum(np.abs(v), np.abs(u) - (Lu - Lv) / 2); hw = Lv / 2; ax = "u"
    else: d = np.maximum(np.abs(u), np.abs(v) - (Lv - Lu) / 2); hw = Lu / 2; ax = "v"
    A = np.c_[np.ones_like(d), -d]; cf, *_ = np.linalg.lstsq(A, z, rcond=None)
    k = float(np.clip(cf[1], 0, 1.3)); r = float(np.mean(z + k * d)); rms = float(np.sqrt(np.mean((z - (r - k * d)) ** 2)))
    out.append(dict(form="hip", ridge=ax, rms=rms, k=k, eave=r - k * hw, ridge_z=r, npar=2))
    best = min(out, key=lambda o: o["rms"] * (1 + 0.04 * o["npar"]))
    if best["form"] in ("gable", "hip") and best["k"] < 0.08:      # pitched fit with no pitch = flat
        best = out[0] if out[0]["form"] == "flat" else best
    return best


def main():
    x, y, z, h = load_pts(); print(f"{len(x):,} single-return HAG>=1.8 candidate points")
    ok, cnt = planar_cells(x, y, z)
    XA, YA = x.copy(), y.copy()                     # all candidates: planar share per footprint separates roofs from canopy
    x, y, z, h = x[ok], y[ok], z[ok], h[ok]; print(f"{len(x):,} in planar 1 m cells")
    x0, y0 = math.floor(x.min()) - 5, math.ceil(y.max()) + 5
    W, H = int((x.max() + 5 - x0) / RES) + 1, int((y0 - y.min() + 5) / RES) + 1
    c = ((x - x0) / RES).astype(int); r = ((y0 - y) / RES).astype(int)
    M = np.zeros((H, W), bool); M[r, c] = True

    M = ndi.binary_closing(M, np.ones((3, 3)), iterations=2); M = ndi.binary_fill_holes(M); M = ndi.binary_opening(M, np.ones((3, 3)))
    lab, n = ndi.label(M); print(n, "components"); del M
    ca = ((XA - x0) / RES).astype(np.int32); ra = ((y0 - YA) / RES).astype(np.int32); ka = (ca >= 0) & (ca < W) & (ra >= 0) & (ra < H)
    NA = np.bincount(lab[ra[ka], ca[ka]], minlength=n + 1); del XA, YA, ca, ra, ka
    NP = np.bincount(lab[r, c], minlength=n + 1)
    sizes = np.bincount(lab.ravel()); keep = np.nonzero(sizes * RES * RES >= 9.0)[0]; keep = keep[keep > 0]
    tr = from_origin(x0, y0, RES, RES)
    naip = rasterio.open(DER / "naip2022_r060.tif"); dtm = rasterio.open(DER / "dtm_r100_b150.tif"); D1 = dtm.read(1)
    cl = np.genfromtxt(DER / "centerline.csv", delimiter=",", names=True)
    CLX, CLY = cl["x_utm"], cl["y_utm"]; KD = cKDTree(np.c_[CLX, CLY]); TX = np.gradient(CLX); TY = np.gradient(CLY); tn = np.hypot(TX, TY); TX /= tn; TY /= tn
    ptlab = lab[r, c]; order = np.argsort(ptlab); ps = ptlab[order]; starts = np.searchsorted(ps, np.arange(n + 2))
    osm = [shape(f["geometry"]) for f in json.load(open(DER / "osm_buildings_barriers.geojson"))["features"] if f["properties"]["kind"] == "building" and f["geometry"]["type"] == "Polygon"]
    feats, rej = [], dict(green=0, small=0, low=0, rough=0, far=0, thin=0)
    objs = ndi.find_objects(lab)
    for li in keep:
        sl = objs[li - 1]; sub = lab[sl] == li
        t2 = rasterio.transform.from_origin(x0 + sl[1].start * RES, y0 - sl[0].start * RES, RES, RES)
        polys = [shape(g) for g, v in rasterio.features.shapes(sub.astype(np.uint8), mask=sub, transform=t2) if v == 1]
        if not polys: continue
        P = max(polys, key=lambda p: p.area).buffer(0.25, join_style=2).buffer(-0.25, join_style=2).simplify(0.45)
        if P.geom_type != "Polygon" or P.area < 9: rej["small"] += 1; continue
        idx = order[starts[li]:starts[li + 1]]; px, py, pz, ph = x[idx], y[idx], z[idx], h[idx]
        if len(px) < 15: rej["small"] += 1; continue
        pshare = NP[li] / max(NA[li], 1)
        if pshare < 0.55: rej["canopy"] = rej.get("canopy", 0) + 1; continue
        dist, j = KD.query([P.centroid.x, P.centroid.y])
        if dist > 155: rej["far"] += 1; continue
        if np.percentile(ph, 90) < 2.3: rej["low"] += 1; continue
        # NAIP colour + NDVI inside eroded footprint
        Pe = P.buffer(-0.6) if P.buffer(-0.6).area > 2 else P
        win = rasterio.windows.from_bounds(*Pe.bounds, naip.transform).round_offsets().round_lengths()
        a = naip.read(window=win, boundless=True, fill_value=0).astype(float)
        if a.shape[1] == 0 or a.shape[2] == 0: continue
        mk = rasterio.features.geometry_mask([Pe], (a.shape[1], a.shape[2]), naip.window_transform(win), invert=True) & (a[:3].sum(0) > 0)
        if mk.sum() < 3: mk = a[:3].sum(0) > 0
        rgb = np.median(a[:3, mk], axis=1) if mk.any() else np.array([128, 128, 128.])
        nir = np.median(a[3, mk]) if mk.any() else 0
        ndvi_px = (a[3, mk] - a[0, mk]) / (a[3, mk] + a[0, mk] + 1e-6) if mk.any() else np.zeros(1)
        ndvi = float(np.median(ndvi_px)); green_frac = float((ndvi_px > 0.25).mean())
        if ndvi > 0.30 or green_frac > 0.75: rej["green"] += 1; continue   # loose: NAIP roofs lean several m vs LiDAR
        th = orient(P); cth, sth = math.cos(th), math.sin(th)
        cx0, cy0 = P.centroid.x, P.centroid.y
        U = (px - cx0) * cth + (py - cy0) * sth; V = -(px - cx0) * sth + (py - cy0) * cth
        Pr = affinity.rotate(P, -th, origin=(cx0, cy0), use_radians=True)
        bx0, by0, bx1, by1 = Pr.bounds; nc, nr_ = int((bx1 - bx0) / RES) + 1, int((by1 - by0) / RES) + 1
        Mr = rasterio.features.rasterize([Pr], (nr_, nc), transform=from_origin(bx0, by1, RES, RES)).astype(bool)
        tot = Mr.sum(); rects = []; rem = Mr.copy()
        while len(rects) < 4:
            a_, r0, r1, c0, c1 = max_rect(rem)
            if a_ * RES * RES < (9 if not rects else 8) or (rects and rem.sum() < 0.08 * tot): break
            rem[r0:r1, c0:c1] = False
            u0, u1 = bx0 + c0 * RES - (cx0), bx0 + c1 * RES - cx0; v1, v0 = by1 - r0 * RES - cy0, by1 - r1 * RES - cy0
            rects.append([u0, u1, v0, v1])
        if not rects: rej["small"] += 1; continue
        parts = []; rms_all = []
        for u0, u1, v0, v1 in rects:
            m = (U >= u0 - .2) & (U <= u1 + .2) & (V >= v0 - .2) & (V <= v1 + .2)
            Lu, Lv = u1 - u0, v1 - v0; uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
            if m.sum() < 8:
                f = dict(form="flat", ridge=None, rms=0.3, k=0.0, eave=float(np.median(pz)), ridge_z=float(np.median(pz)))
            else:
                f = fit_roof(U[m] - uc, V[m] - vc, pz[m], Lu, Lv)
            rms_all.append(f["rms"])
            wx = cx0 + uc * cth - vc * sth; wy = cy0 + uc * sth + vc * cth
            parts.append(dict(cx=round(wx - X0, 3), cy=round(wy - Y0, 3), yaw=round(th, 5), Lu=round(Lu, 2), Lv=round(Lv, 2), **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()}))
        if np.median(rms_all) > 0.8: rej["rough"] += 1; continue
        # ground under the footprint
        gw = rasterio.windows.from_bounds(*P.buffer(1.0).bounds, dtm.transform).round_offsets().round_lengths()
        G = dtm.read(1, window=gw, boundless=True, fill_value=np.nan)
        gm = rasterio.features.geometry_mask([P.buffer(1.0)], G.shape, dtm.window_transform(gw), invert=True) & np.isfinite(G)
        gz = float(np.percentile(G[gm], 10)) if gm.any() else float(np.median(pz - ph))
        gzmax = float(np.percentile(G[gm], 95)) if gm.any() else gz
        # cylinder test (water tanks)
        circ = 4 * math.pi * P.area / P.length ** 2; mrr = P.minimum_rotated_rectangle; ex = np.asarray(mrr.exterior.coords)
        e1, e2 = np.hypot(*(ex[1] - ex[0])), np.hypot(*(ex[2] - ex[1])); asp = max(e1, e2) / max(min(e1, e2), .1)
        if min(e1, e2) < 2.6 or P.area < 12: rej["small"] += 1; continue
        if asp > 4.5 and min(e1, e2) < 8: rej["thin"] += 1; continue                 # hedge rows / canopy strips
        if P.area < 25 and np.percentile(ph, 90) < 3.2: rej["low"] += 1; continue  # bushes / vehicles in fields      # thin strips = hedges / fence rows / canopy edges
        form = "tank" if (P.area < 80 and circ > 0.82 and asp < 1.25) else max(parts, key=lambda q: q["Lu"] * q["Lv"])["form"]
        dx, dy = cx0 - CLX[j], cy0 - CLY[j]; s = float(cl["s_m"][j] + dx * TX[j] + dy * TY[j]); off = float(-TY[j] * dx + TX[j] * dy)   # + = left
        iou = 0.0
        for q in osm:
            if q.intersects(P): iou = max(iou, q.intersection(P).area / q.union(P).area)
        eave = min(pp["eave"] for pp in parts); ridge = max(pp["ridge_z"] for pp in parts)
        # road-facing distance of the nearest footprint point
        dmin = float(KD.query(np.asarray(P.exterior.coords))[0].min())
        feats.append(dict(type="Feature", geometry=mapping(P), properties=dict(
            id=f"bld_{len(feats)+1:04d}", s=round(s, 1), side="L" if off > 0 else "R", offset_m=round(off, 1), dist_road_m=round(dmin, 1),
            area_m2=round(P.area, 1), n_pts=int(len(px)), ground_z=round(gz, 2), ground_z_hi=round(gzmax, 2),
            eave_z=round(eave, 2), ridge_z=round(ridge, 2), eave_h=round(eave - gz, 2), ridge_h=round(ridge - gz, 2),
            form=form, n_parts=len(parts), parts=parts, roof_rgb=[int(v) for v in rgb], ndvi=round(ndvi, 3),
            planar_share=round(float(pshare), 2), rms_m=round(float(np.median(rms_all)), 3), circ=round(circ, 2), osm_iou=round(iou, 2),
            poly_local=[[round(a - X0, 2), round(b - Y0, 2)] for a, b in P.exterior.coords], src="lidar2021+naip2022")))
    # OSM buildings with no LiDAR match -> listed (not modelled)
    unmatched = [q for q in osm if not any(shape(f["geometry"]).intersects(q) for f in feats)]
    json.dump(dict(type="FeatureCollection", crs=dict(type="name", properties=dict(name="EPSG:26910")), features=feats,
                   meta=dict(rejected=rej, osm_total=len(osm), osm_unmatched=len(unmatched),
                             osm_unmatched_centroids=[[round(q.centroid.x, 1), round(q.centroid.y, 1)] for q in unmatched])),
              open(DER / "buildings.geojson", "w"))
    import collections
    print(len(feats), "buildings", dict(collections.Counter(f["properties"]["form"] for f in feats)), "rejected", rej,
          f"OSM {len(osm)} ({len(unmatched)} without a LiDAR match)")


if __name__ == "__main__":
    main()
