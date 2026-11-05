"""Phase-1 features along the refined centerline:
 OSM junctions/driveways/bridges/culverts/streams/stop signs, LiDAR bridge-deck (class 17) spans,
 ground-return gaps, guardrail candidates (height-above-ground signature), steep faces (wall/cliff),
 road markings (intensity), trees from the CHM. Writes data/derived/features.geojson, trees.geojson/csv, gaps.csv, markings.csv."""
import glob, json, csv, numpy as np, rasterio
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter, gaussian_filter, median_filter, label
from shapely.geometry import LineString, Point
from common import *

o = origin()
CL = list(csv.DictReader(open(DER / "centerline.csv")))
S = np.array([float(r["s_m"]) for r in CL]); XY = np.array([[float(r["x_utm"]), float(r["y_utm"])] for r in CL])
WL = np.array([float(r["edge_left_m"]) for r in CL]); WR = np.array([float(r["edge_right_m"]) for r in CL])
ZC = np.array([float(r["z_navd88"]) for r in CL])
T = np.gradient(XY, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
ctree = cKDTree(XY)
line = LineString(XY)
feats = []

def station(x, y):
    dd, j = ctree.query([x, y]); rel = np.array([x, y]) - XY[j]
    return float(S[j] + rel @ T[j]), float(T[j, 0] * rel[1] - T[j, 1] * rel[0]), float(dd)

def addpt(kind, x, y, **p):
    s, d, _ = station(x, y); lon, lat = T_UTM_LL.transform(x, y)
    feats.append({"type": "Feature", "properties": {"kind": kind, "s_m": round(s, 1), "offset_m": round(d, 1), **p},
                  "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]}})

def addspan(kind, s0, s1, **p):
    m = (S >= s0) & (S <= s1)
    if m.sum() < 2: m = np.abs(S - (s0 + s1) / 2) < 1.5
    lon, lat = T_UTM_LL.transform(XY[m, 0], XY[m, 1])
    feats.append({"type": "Feature", "properties": {"kind": kind, "s_start_m": round(s0, 1), "s_end_m": round(s1, 1), "length_m": round(s1 - s0, 1), **p},
                  "geometry": {"type": "LineString", "coordinates": [[round(a, 7), round(b, 7)] for a, b in zip(lon, lat)]}})

def runs(mask, min_len, merge=0.0, step=1.0):
    idx = np.flatnonzero(mask); out = []
    if len(idx) == 0: return out
    st = pr = idx[0]
    for i in idx[1:]:
        if (i - pr) * step > merge + step: out.append((st, pr)); st = i
        pr = i
    out.append((st, pr))
    return [(a, b) for a, b in out if (b - a + 1) * step >= min_len]

# ---------- A. OSM context ----------
main = {1190239384, 116299847, 116299515, 79430605}
nodes, ways = {}, {}
for f in sorted(glob.glob(str(RAW / "osm_ctx" / "*.xml"))):
    r = ET.parse(f).getroot()
    for n in r.iter("node"):
        nodes[n.get("id")] = (float(n.get("lon")), float(n.get("lat")), {t.get("k"): t.get("v") for t in n.iter("tag")})
    for w in r.iter("way"):
        ways[w.get("id")] = ([nd.get("ref") for nd in w.iter("nd")], {t.get("k"): t.get("v") for t in w.iter("tag")})
def wxy(nids):
    ll = np.array([nodes[n][:2] for n in nids if n in nodes])
    x, y = T_LL_UTM.transform(ll[:, 0], ll[:, 1]); return np.c_[x, y]
main_nodes = set(n for wid in main if str(wid) in ways for n in ways[str(wid)][0])
seen = set()
for wid, (nids, t) in ways.items():
    if int(wid) in main: continue
    if "highway" in t:
        shared = [n for n in nids if n in main_nodes]
        for n in shared:
            if (wid, n) in seen: continue
            seen.add((wid, n)); x, y = T_LL_UTM.transform(*nodes[n][:2])
            kind = "driveway" if t.get("service") == "driveway" or t["highway"] in ("service", "track") else "junction"
            addpt(kind, x, y, osm_way=int(wid), highway=t["highway"], service=t.get("service", ""), name=t.get("name", ""))
    if "waterway" in t and len(nids) > 1:
        P = wxy(nids)
        if len(P) < 2: continue
        inter = LineString(P).intersection(line)
        for g in getattr(inter, "geoms", [inter]):
            if g.is_empty or g.geom_type != "Point": continue
            addpt("stream_crossing", g.x, g.y, osm_way=int(wid), waterway=t["waterway"], name=t.get("name", ""),
                  tunnel=t.get("tunnel", ""))
for wid in main:
    w = ways.get(str(wid))
for wid, (nids, t) in ways.items():
    if t.get("bridge") == "yes" and "highway" in t and len(nids) > 1:
        P = wxy(nids)
        ss = [station(*p) for p in P]
        if min(abs(q[1]) for q in ss) < 10:
            addspan("bridge_osm", min(q[0] for q in ss), max(q[0] for q in ss), osm_way=int(wid), name=t.get("name", ""))
for nid, (lon, lat, t) in nodes.items():
    if t.get("highway") in ("stop", "give_way") or "traffic_sign" in t:
        x, y = T_LL_UTM.transform(lon, lat); s, d, dist = station(x, y)
        if dist < 25: addpt("sign", x, y, osm_node=int(nid), sign=t.get("highway", t.get("traffic_sign")))

# ---------- B-E. LiDAR-based ----------
dtm = rasterio.open(DER / "dtm_r050_b40.tif"); dtmA = dtm.read(1)
def sample_dtm(x, y):
    r, c = rasterio.transform.rowcol(dtm.transform, x, y)
    r = np.clip(np.asarray(r), 0, dtmA.shape[0] - 1); c = np.clip(np.asarray(c), 0, dtmA.shape[1] - 1)
    return dtmA[r, c]
idx = json.load(open(CACHE / "chunks" / "index.json"))
NS = len(S)
gcount = np.zeros(NS); bridge17 = np.zeros(NS)
GR_lo = np.zeros((NS, 2)); GR_hi = np.zeros((NS, 2)); GR_n = np.zeros((NS, 2))
for k in range(idx["n"]):
    d = np.load(CACHE / "chunks" / f"chunk_{k:03d}.npz")
    x = d["x"].astype(np.float64) + o["x0"]; y = d["y"].astype(np.float64) + o["y0"]
    dist, j = ctree.query(np.c_[x, y], distance_upper_bound=14)
    ok = np.isfinite(dist); j = np.where(ok, j, 0)
    rel = np.c_[x - XY[j, 0], y - XY[j, 1]]; lat = T[j, 0] * rel[:, 1] - T[j, 1] * rel[:, 0]
    c = d["c"]
    g = ok & ((c == 2) | (c == 17)) & (np.abs(lat) < 2.5); np.add.at(gcount, j[g], 1)
    b = ok & (c == 17) & (np.abs(lat) < 8); np.add.at(bridge17, j[b], 1)
    # guardrail signature: non-ground points 0.35-1.0 m above DTM, 0-1.8 m outside the road edge
    v = ok & (c == 1)
    if v.any():
        hag = d["z"][v] - sample_dtm(x[v], y[v]); jj = j[v]; la = lat[v]
        for side, W, col in ((1, WL, 0), (-1, WR, 1)):
            out_ = side * la - W[jj]
            band = (out_ > -0.3) & (out_ < 1.8)
            np.add.at(GR_n[:, col], jj[band], 1)
            np.add.at(GR_lo[:, col], jj[band & (hag > 0.35) & (hag < 1.05)], 1)
            np.add.at(GR_hi[:, col], jj[band & (hag > 1.3) & (hag < 3.0)], 1)
    print("chunk", k, flush=True)

gaps = []
nog = median_filter((gcount == 0).astype(int), 1) > 0
for a, b in runs(nog, 5.0):
    gaps.append((S[a], S[b] + 1, "no ground returns within 2.5 m of centerline"))
    addspan("lidar_gap", S[a], S[b] + 1, note="no ground returns within 2.5 m of centerline; surface TIN-interpolated")
sparse = (np.convolve(gcount, np.ones(10) / 10, "same") / (5 * 1.0)) < 0.5   # <0.5 pts/m^2 over 10 m
for a, b in runs(sparse & ~nog, 10.0):
    gaps.append((S[a], S[b] + 1, "sparse ground (<0.5 pts/m^2 over 10 m)"))
    addspan("lidar_sparse", S[a], S[b] + 1, note="ground density <0.5 pts/m^2")
with open(DER / "gaps.csv", "w") as f:
    f.write("s_start_m,s_end_m,length_m,type\n")
    for a, b, t in gaps: f.write(f"{a:.0f},{b:.0f},{b-a:.0f},{t}\n")
for a, b in runs(bridge17 > 0, 3.0, merge=3):
    addspan("bridge_lidar_class17", S[a], S[b] + 1, points=int(bridge17[a:b + 1].sum()))
for col, side in ((0, "left"), (1, "right")):
    lo = np.convolve(GR_lo[:, col], np.ones(5), "same"); hi = np.convolve(GR_hi[:, col], np.ones(5), "same")
    cand = (lo >= 6) & (hi < 0.5 * lo)
    for a, b in runs(cand, 10.0, merge=3):
        addspan("guardrail_candidate", S[a], S[b] + 1, side=side, confidence="medium" if b - a > 30 else "low")
envs = {"left": [r["env_left"] for r in CL], "right": [r["env_right"] for r in CL]}
for side, e in envs.items():
    e = np.array(e)
    for kind in ("wall_or_steep_cut", "steep_drop"):
        for a, b in runs(e == kind, 8.0, merge=2):
            addspan("steep_face" if kind == "wall_or_steep_cut" else "steep_drop", S[a], S[b] + 1, side=side,
                    note="rock cut or retaining wall (>1.5 m rise within 2 m of edge)" if kind == "wall_or_steep_cut" else ">1.2 m drop within 2 m of edge; barrier/cliff candidate")

# ---------- F. markings from paint scores ----------
pc = np.array([float(r["paint_center"]) for r in CL]); pl = np.array([float(r["paint_left"]) for r in CL]); pr = np.array([float(r["paint_right"]) for r in CL])
def seg_score(a, L=50):
    out = []
    for s0 in np.arange(0, S[-1], L):
        m = (S >= s0) & (S < s0 + L) & np.isfinite(a)
        out.append(np.nanmedian(a[m]) if m.sum() > 10 else np.nan)
    return np.array(out)
C50, L50, R50 = seg_score(pc), seg_score(pl), seg_score(pr)
def cls(v):
    if not np.isfinite(v): return "unknown"
    return "likely" if v > 1.06 else ("possible" if v > 1.02 else "not_detected")
with open(DER / "markings.csv", "w") as f:
    f.write("s_start_m,s_end_m,center_ratio,center_line,left_ratio,left_edge_line,right_ratio,right_edge_line,assumed_center_type\n")
    for i, s0 in enumerate(np.arange(0, S[-1], 50)):
        f.write(f"{s0:.0f},{min(s0+50,S[-1]):.0f},{C50[i]:.3f},{cls(C50[i])},{L50[i]:.3f},{cls(L50[i])},{R50[i]:.3f},{cls(R50[i])},double_yellow\n")

# ---------- G. trees from CHM ----------
chm = rasterio.open(DER / "chm_r100_b150.tif"); H = chm.read(1); H = np.nan_to_num(H, nan=0)
Hs = gaussian_filter(H, 1.0)
peaks = (Hs == maximum_filter(Hs, size=5)) & (Hs > 4.0)
rr, cc = np.nonzero(peaks)
px, py = rasterio.transform.xy(chm.transform, rr, cc); px, py = np.array(px), np.array(py)
dist, _ = ctree.query(np.c_[px, py]); keep = dist < 60
rr, cc, px, py, dist = rr[keep], cc[keep], px[keep], py[keep], dist[keep]
ph = H[rr, cc]
# reject CHM spikes taller than any local tree (power poles/wires, LiDAR noise): coast live oak/bay/eucalyptus here are < 40 m
m2 = ph <= 40.0
rr, cc, px, py, dist, ph = rr[m2], cc[m2], px[m2], py[m2], dist[m2], ph[m2]
# crown radius: canopy cells (>50% of peak height) assigned to nearest peak
can_r, can_c = np.nonzero(Hs > 3.0)
cx, cy = rasterio.transform.xy(chm.transform, can_r, can_c); cx, cy = np.array(cx), np.array(cy)
ptree = cKDTree(np.c_[px, py]); dd_, jj = ptree.query(np.c_[cx, cy], distance_upper_bound=12)
okc = np.isfinite(dd_) & (Hs[can_r, can_c] > 0.5 * ph[np.where(np.isfinite(dd_), jj, 0)])
area = np.bincount(jj[okc], minlength=len(px)).astype(float)
crown = np.sqrt(np.maximum(area, 1) / np.pi)
gz = rasterio.open(DER / "dtm_r100_b150.tif"); G = gz.read(1)
base = G[rr, cc]
with open(DER / "trees.csv", "w") as f:
    f.write("id,x_utm,y_utm,x_local,y_local,ground_z,height_m,crown_radius_m,dist_to_centerline_m,s_m,offset_m\n")
    tf = []
    moved = 0; dropped = 0
    for i in range(len(px)):
        s, d, _ = station(px[i], py[i])
        # no trunks on the carriageway: a CHM peak over the road is overhanging canopy -> move the trunk
        # to just outside the nearer edge (same station) so the crown still overhangs realistically
        j0 = int(np.clip(round(s), 0, len(S) - 1)); edge_ = WL[j0] if d >= 0 else WR[j0]
        if abs(d) < edge_ + 1.0:
            side_ = 1 if d >= 0 else -1; dn = side_ * (edge_ + 1.5 + 0.3 * crown[i])
            nrm = np.array([-T[j0, 1], T[j0, 0]]); px[i], py[i] = XY[j0] + nrm * dn
            base[i] = float(G[rasterio.transform.rowcol(gz.transform, px[i], py[i])])
            s, d, _ = station(px[i], py[i]); moved += 1
            j1 = int(np.clip(round(s), 0, len(S) - 1))
            if abs(d) < (WL[j1] if d >= 0 else WR[j1]) + 1.0:
                dropped += 1; continue          # hairpin: moved trunk lands on the other leg -> drop
        f.write(f"{i},{px[i]:.2f},{py[i]:.2f},{px[i]-o['x0']:.2f},{py[i]-o['y0']:.2f},{base[i]:.2f},{ph[i]:.1f},{crown[i]:.1f},{dist[i]:.1f},{s:.1f},{d:.1f}\n")
        lon, lat = T_UTM_LL.transform(px[i], py[i])
        tf.append({"type": "Feature", "properties": {"id": i, "height_m": round(float(ph[i]), 1), "crown_radius_m": round(float(crown[i]), 1), "s_m": round(s, 1)},
                   "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7), round(float(base[i]), 2)]}})
json.dump({"type": "FeatureCollection", "features": tf}, open(DER / "trees.geojson", "w"))
summit = int(np.argmax(ZC)); lon, lat = T_UTM_LL.transform(*XY[summit])
feats.append({"type": "Feature", "properties": {"kind": "summit", "s_m": float(S[summit]), "z_navd88": float(ZC[summit])},
              "geometry": {"type": "Point", "coordinates": [lon, lat, float(ZC[summit])]}})
json.dump({"type": "FeatureCollection", "features": feats}, open(DER / "features.geojson", "w"), indent=0)
import collections
print(collections.Counter(f["properties"]["kind"] for f in feats), "trees", len(px), "moved off road", moved, "dropped", dropped, "gaps", len(gaps))
