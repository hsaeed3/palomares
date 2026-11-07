#!/usr/bin/env python3
"""dev6 step 05c: near-road understorey layer from understory.tif (LiDAR 0.3-3 m return share, NAIP NDVI) + the
low-point heights of step 01.

Per side, per 1 m of station, two bands beyond the pavement edge: verge 0.3-4 m and bank 4-12 m.
  frac  = mean low-veg return share, ndvi = mean NAIP NDVI (2022-05, spring), h90 = 90th pct HAG of 0.3-3.5 m returns
  verge runs (frac >= 0.12 for >= 6 m, gaps < 3 m bridged) -> hedge rows; type from height + NDVI:
      h90 >= 1.3            -> hedge.coyote_brush   (tall brush / bay / toyon understorey)
      0.8 <= h90 < 1.3      -> hedge.blackberry_thicket
      h90 < 0.8, ndvi>=0.25 -> hedge.grass_verge (tall spring grass)
  bank: shrub points at 3 m local maxima of frac >= 0.3 (>= 7 m apart): shrub.coyote_brush (h >= 1.2) or
        shrub.poison_oak (lower); scale from h90
  ground cover per cell (for reporting/terrain tint): bare (ndvi < 0.08 & frac < 0.05), grass (ndvi >= 0.08 & frac < 0.12), shrub.
-> data/derived/understory.geojson (UTM: LineStrings for rows, Points for shrubs; props s, s1, side, off, asset, frac, ndvi, h90)
"""
import sys, json, glob, pathlib, math
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import DER, CACHE
import rasterio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

cl = np.genfromtxt(DER / "centerline.csv", delimiter=",", names=True)
CX, CY = cl["x_utm"], cl["y_utm"]; TX = np.gradient(CX); TY = np.gradient(CY); tn = np.hypot(TX, TY); TX /= tn; TY /= tn
NX, NY = -TY, TX
U = rasterio.open(DER / "understory.tif"); FR = U.read(1); ND = U.read(2); INV = ~U.transform


def samp(A, x, y):
    c, r = INV * (x, y); r = np.clip(r.astype(int), 0, A.shape[0] - 1); c = np.clip(c.astype(int), 0, A.shape[1] - 1); return A[r, c]


def main():
    xs, ys, hs = [], [], []
    for f in sorted(glob.glob(str(CACHE / "p6/nodes/*.npz"))):
        d = np.load(f); m = (d["lh"] >= 0.3); xs.append(d["lx"][m]); ys.append(d["ly"][m]); hs.append(d["lh"][m])
    x, y, h = np.concatenate(xs), np.concatenate(ys), np.concatenate(hs).astype(float)
    KD = cKDTree(np.c_[CX, CY]); _, j = KD.query(np.c_[x, y]); off = (x - CX[j]) * NX[j] + (y - CY[j]) * NY[j]
    n = len(CX); feats = []; cover = dict(bare=0, grass=0, shrub=0)
    for side, sg in (("L", 1), ("R", -1)):
        edge = cl["edge_left_m"] if sg > 0 else cl["edge_right_m"]
        b = sg * off - edge[j]; ms = np.sign(off) == sg
        for band, (b0, b1) in (("verge", (0.3, 4.0)), ("bank", (4.0, 12.0))):
            oo = np.arange(b0 + 0.5, b1, 1.0)
            px = CX[:, None] + sg * NX[:, None] * (edge[:, None] + oo[None]); py = CY[:, None] + sg * NY[:, None] * (edge[:, None] + oo[None])
            F = samp(FR, px, py); V = samp(ND, px, py)
            if band == "verge":
                for k in ("bare", "grass", "shrub"):
                    pass
                cover["bare"] += int(np.sum((V < 0.08) & (F < 0.05))); cover["grass"] += int(np.sum((V >= 0.08) & (F < 0.12))); cover["shrub"] += int(np.sum(F >= 0.12))
            fr = np.nan_to_num(np.nanmean(F, 1)); nd = np.nan_to_num(np.nanmean(V, 1))
            mm = ms & (b >= b0) & (b < b1)
            h90 = np.zeros(n); jj = j[mm]; hh = h[mm]
            if len(jj):
                o = np.argsort(jj); jj, hh = jj[o], hh[o]; st = np.searchsorted(jj, np.arange(n + 1))
                for i in np.nonzero(np.diff(st) >= 3)[0]: h90[i] = np.percentile(hh[st[i]:st[i + 1]], 90)
            if band == "verge":
                on = ndi.binary_closing(ndi.uniform_filter1d(fr, 5) >= 0.12, np.ones(3))
                lab, nl = ndi.label(on)
                for sl in ndi.find_objects(lab):
                    i0, i1 = sl[0].start, sl[0].stop
                    if i1 - i0 < 6: continue
                    # split long runs at 40 m so type/density follow the data
                    for a in range(i0, i1, 40):
                        e = min(a + 40, i1)
                        if e - a < 4: continue
                        hm = float(np.median(h90[a:e][h90[a:e] > 0])) if (h90[a:e] > 0).any() else 0.6
                        fm = float(fr[a:e].mean()); nm = float(nd[a:e].mean())
                        asset = "hedge.coyote_brush" if hm >= 1.3 else ("hedge.blackberry_thicket" if hm >= 0.8 else ("hedge.grass_verge" if nm >= 0.25 else "hedge.blackberry_thicket"))
                        ofs = float(np.clip(np.median(np.nonzero(F[a:e].mean(0) >= 0.5 * F[a:e].mean(0).max())[0]) + b0, 0.4, 3.0))
                        i_ = np.arange(a, e, 2)
                        pts = np.c_[CX[i_] + sg * NX[i_] * (edge[i_] + ofs), CY[i_] + sg * NY[i_] * (edge[i_] + ofs)]
                        feats.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=np.round(pts, 2).tolist()),
                                          properties=dict(cls="hedge", band="verge", asset=asset, s=float(cl["s_m"][a]), s1=float(cl["s_m"][e - 1]), side=side,
                                                          off=round(ofs, 2), frac=round(fm, 3), ndvi=round(nm, 3), h90=round(hm, 2),
                                                          density=round(float(np.clip(fm / 0.3, 0.3, 1.5)), 2))))
            else:
                G = np.nan_to_num(F); Gm = ndi.maximum_filter(G, size=(5, 3))
                ii, kk = np.nonzero((G == Gm) & (G >= 0.3))
                last = {}
                for i, k in sorted(zip(ii, kk)):
                    if i - last.get(k, -99) < 7: continue
                    last[k] = i; hm = h90[i] if h90[i] > 0 else 1.0
                    feats.append(dict(type="Feature", geometry=dict(type="Point", coordinates=[round(float(px[i, k]), 2), round(float(py[i, k]), 2)]),
                                      properties=dict(cls="shrub", band="bank", asset="shrub.coyote_brush" if hm >= 1.2 else "shrub.poison_oak",
                                                      s=float(cl["s_m"][i]), side=side, off=round(float(oo[k]), 2), frac=round(float(G[i, k]), 3),
                                                      ndvi=round(float(np.nan_to_num(V[i, k])), 3), h90=round(float(hm), 2),
                                                      scale=round(float(np.clip(hm / 1.5, 0.6, 1.6)), 2))))
    tot = sum(cover.values())
    json.dump(dict(type="FeatureCollection", crs=dict(type="name", properties=dict(name="EPSG:26910")), features=feats,
                   meta=dict(verge_cover_share={k: round(v / max(tot, 1), 3) for k, v in cover.items()})), open(DER / "understory.geojson", "w"))
    import collections
    print(collections.Counter((f["properties"]["cls"], f["properties"]["asset"]) for f in feats),
          "hedge m", round(sum(f["properties"]["s1"] - f["properties"]["s"] for f in feats if f["properties"]["cls"] == "hedge")),
          "verge cover", {k: round(v / max(tot, 1), 3) for k, v in cover.items()})


if __name__ == "__main__":
    main()
