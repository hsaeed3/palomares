"""dev5 shared scene code: class ids, camera model, chunk cache, class-ID renderer (numpy + PIL painter's z-order).

Camera/lens convention is the one of scripts/p2/27_litmus_fit.py (local coords x,y,z NAVD88):
pose (s, lateral, height, yaw, pitch, roll) relative to the centerline tangent at s; lateral >0 = right of centreline
(as in 27). The GoPro photo is fisheye (f_rel, lam from config/litmus_poses.json); we render a PINHOLE view of
vertical fov FOV and undistort the photo/seg to it (same as 28_litmus_sheet.py).
"""
import csv, json, pathlib, time
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCENE = ROOT / "data/scene"
CACHE = SCENE / "cache"
CHUNK = 200.0
CLASSES = ["unknown", "sky", "road", "marking_yellow", "marking_white", "shoulder_dirt", "terrain_grass", "terrain_bare",
           "rock_cut", "vegetation_tree", "shrub", "trunk", "sign_face", "sign_back", "pole", "guardrail", "fence", "wall",
           "building", "bridge", "vehicle", "person", "water", "mailbox_prop", "other_prop"]
CID = {n: i for i, n in enumerate(CLASSES)}
IGNORE = {0, 20, 21}
_LP = json.load(open(ROOT / "config/litmus_poses.json"))
LENS = (_LP["lens"]["f_rel"], _LP["lens"]["lam"])
FOV = float(_LP["fov"])
AW, AH = 512, 372          # analysis size (pinhole)

# ---------------------------------------------------------------- centerline
_rows = list(csv.DictReader(open(ROOT / "data/derived/centerline.csv")))
CLA = np.array([[float(r[k]) for k in ("s_m", "x_local", "y_local", "z_navd88", "edge_left_m", "edge_right_m")] for r in _rows])
S, X, Y, Z, EL, ER = CLA.T
TX, TY = np.gradient(X), np.gradient(Y); _n = np.hypot(TX, TY); TX /= _n; TY /= _n
S_MAX = float(S[-1])
_KD = cKDTree(np.c_[X, Y])


def station_of(x, y):
    return S[_KD.query(np.c_[np.ravel(x), np.ravel(y)])[1]]


# ---------------------------------------------------------------- camera
def cam_basis(p):
    s, lat, h, yaw, pit, rol = p[:6]
    i = int(np.clip(round(s), 0, len(S) - 2))
    tx, ty = TX[i], TY[i]
    eye = np.array([np.interp(s, S, X) + ty * lat, np.interp(s, S, Y) - tx * lat, np.interp(s, S, Z) + h])
    hd = np.arctan2(ty, tx) + yaw
    fwd = np.array([np.cos(hd) * np.cos(pit), np.sin(hd) * np.cos(pit), np.sin(pit)])
    rt = np.array([np.sin(hd), -np.cos(hd), 0.0]); up = np.cross(rt, fwd)
    c, sn = np.cos(rol), np.sin(rol); rt, up = c * rt + sn * up, -sn * rt + c * up
    return eye, fwd, rt, up


def undistort_map(w_src, h_src, ow=AW, oh=AH, vfov=FOV):
    """(sy, sx) source pixel coords in a fisheye image of size w_src x h_src for each pinhole output pixel"""
    fo = (oh / 2) / np.tan(np.radians(vfov) / 2)
    yy, xx = np.mgrid[0:oh, 0:ow]; u = (xx + .5 - ow / 2) / fo; v = (yy + .5 - oh / 2) / fo
    ru = np.hypot(u, v); th = np.arctan(ru); f = LENS[0] * h_src; lam = LENS[1]
    rd = f * np.tan(lam * th) / lam; sc = np.where(ru > 0, rd / np.maximum(ru, 1e-9), f)
    return h_src / 2 + v * sc - .5, w_src / 2 + u * sc - .5


def undistort(a, order=1, ow=AW, oh=AH):
    a = np.asarray(a); h, w = a.shape[:2]; sy, sx = undistort_map(w, h, ow, oh)
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    if a.ndim == 2:
        out = ndimage.map_coordinates(a, [sy, sx], order=order, cval=0)
    else:
        out = np.stack([ndimage.map_coordinates(a[..., c], [sy, sx], order=order, cval=0) for c in range(a.shape[2])], -1)
    return out, valid


# ---------------------------------------------------------------- chunk cache
ROAD_MAT = {"road_asphalt": "road", "road_asphalt_fresh": "road", "road_edge": "road", "shoulder_dirt": "shoulder_dirt",
            "litter": "shoulder_dirt", "ground": "terrain_grass", "bank": "terrain_bare"}
LINE_MAT = {"paint_white": "marking_white", "paint_white_fresh": "marking_white", "paint_yellow_dbl": "marking_yellow",
            "paint_yellow_dbl_fresh": "marking_yellow", "road_decal": "marking_white"}
TREE_MAT = {"tree_leaves": "vegetation_tree", "tree_trunk": "trunk", "shrub": "shrub"}


def _cls_of(group, mat):
    if group == "road": return ROAD_MAT.get(mat)
    if group == "lines_road": return LINE_MAT.get(mat)
    if group == "trees": return TREE_MAT.get(mat)
    if group == "terrain":
        if mat == "rock": return "rock_cut"
        if mat.startswith("ter_ground"): return "terrain_grass"
        return "terrain_bare"          # ter_bank / ter_litter
    if group == "roadside" and mat == "roadside": return "other_prop"
    if group == "barriers" and mat != "invisible": return "guardrail"
    return None


def build_cache(force=False):
    """build/geometry.pkl -> data/scene/cache/chunk_KKK.npz (tri vertices float32 [n,3,3], cls uint8) + backdrop.npz"""
    CACHE.mkdir(parents=True, exist_ok=True)
    stamp = CACHE / "stamp.json"; src = ROOT / "build/geometry.pkl"
    key = dict(mtime=src.stat().st_mtime, size=src.stat().st_size, v=2)
    if not force and stamp.exists() and json.load(open(stamp)) == key:
        return
    import pickle
    t0 = time.time(); g = pickle.load(open(src, "rb"))["groups"]
    acc = {}
    for gname in ("road", "terrain", "lines_road", "trees", "roadside", "barriers"):
        for m in g.get(gname, []):
            c = _cls_of(gname, m["material"])
            if c is None: continue
            P = np.asarray(m["P"], np.float64); T = np.asarray(m["T"])
            V = P[T].astype(np.float32)                  # [n,3,3]
            cen = V.mean(1)
            ch = (station_of(cen[:, 0], cen[:, 1]) // CHUNK).astype(int)
            for k in np.unique(ch):
                sel = ch == k
                acc.setdefault(k, []).append((V[sel], np.full(sel.sum(), CID[c], np.uint8)))
    for k, L in acc.items():
        np.savez(CACHE / f"chunk_{k:03d}.npz", V=np.concatenate([a for a, _ in L]), C=np.concatenate([b for _, b in L]))
    # backdrop / horizon: coarse 40 m grid of the backdrop DEM (silhouette only)
    import rasterio
    with rasterio.open(ROOT / "data/derived/backdrop_dem_r10.tif") as r:
        a = r.read(1, out_shape=(r.height // 4, r.width // 4)).astype(np.float32); t = r.transform
    o = json.load(open(ROOT / "config/origin.json"))
    hh, ww = a.shape; sx = t.a * 4; sy = t.e * 4
    xs = t.c + (np.arange(ww) + .5) * sx - o["x0"]; ys = t.f + (np.arange(hh) + .5) * sy - o["y0"]
    np.savez(CACHE / "backdrop.npz", Z=a, xs=xs.astype(np.float32), ys=ys.astype(np.float32))
    json.dump(key, open(stamp, "w"))
    print(f"[cache] {len(acc)} chunks in {time.time() - t0:.0f}s")


_CH = {}


def chunk(k):
    if k not in _CH:
        f = CACHE / f"chunk_{k:03d}.npz"
        if f.exists():
            d = np.load(f); _CH[k] = (d["V"], d["C"])
        else:
            _CH[k] = (np.zeros((0, 3, 3), np.float32), np.zeros(0, np.uint8))
    return _CH[k]


_BD = {}


def backdrop_tris(eye, r=4000.0, step=2):
    if not _BD:
        d = np.load(CACHE / "backdrop.npz"); _BD.update(Z=d["Z"], xs=d["xs"], ys=d["ys"])
    Zg, xs, ys = _BD["Z"], _BD["xs"], _BD["ys"]
    ci = np.nonzero(np.abs(xs - eye[0]) < r)[0][::step]; ri = np.nonzero(np.abs(ys - eye[1]) < r)[0][::step]
    if len(ci) < 2 or len(ri) < 2: return np.zeros((0, 3, 3), np.float32)
    gx, gy = np.meshgrid(xs[ci], ys[ri]); gz = Zg[np.ix_(ri, ci)]
    P = np.stack([gx, gy, gz], -1)
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    T = np.concatenate([np.stack([a, b, d], 2).reshape(-1, 3, 3), np.stack([a, d, c], 2).reshape(-1, 3, 3)])
    near = np.hypot(T[:, :, 0].mean(1) - eye[0], T[:, :, 1].mean(1) - eye[1]) > 250   # the near field is the real terrain
    return T[near & np.isfinite(T).all((1, 2)) & (T[:, :, 2] > -100).all(1)]


# ---------------------------------------------------------------- ledger proxies
def load_catalog():
    p = ROOT / "config/asset_catalog.yaml"
    if not p.exists(): return {}
    import yaml
    d = yaml.safe_load(open(p)) or {}
    d = d.get("assets", d)
    if isinstance(d, list): d = {a.get("id"): a for a in d}
    return {k: v for k, v in d.items() if isinstance(v, dict)}


# default proxy per ledger cls: (width along road, depth across, height, z0 above ground, class)
DEFAULT_PROXY = {"sign": (0.8, 0.1, 0.8, 1.6, "sign_face"), "post": (0.15, 0.15, 1.2, 0, "pole"), "pole": (0.3, 0.3, 9, 0, "pole"),
                 "mailbox": (0.5, 0.3, 0.3, 1.0, "mailbox_prop"), "guardrail": (0, 0.2, 0.75, 0, "guardrail"),
                 "fence": (0, 0.1, 1.3, 0, "fence"), "wall": (0, 0.4, 1.2, 0, "wall"), "bridge": (0, 0.4, 1.0, 0, "bridge"),
                 "culvert": (1.0, 1.0, 0.6, 0, "other_prop"), "building": (8, 8, 5, 0, "building"), "driveway": (0, 0, 0, 0, None),
                 "gate": (3, 0.1, 1.3, 0, "fence"), "shrub": (1.5, 1.5, 1.2, 0, "shrub"), "hedge": (0, 1.2, 1.8, 0, "shrub"),
                 "rock": (1.0, 1.0, 0.7, 0, "rock_cut"), "debris": (1.0, 1.0, 0.3, 0, "other_prop"), "prop": (1, 1, 1, 0, "other_prop"),
                 "marking": (0, 0, 0, 0, None), "mound": (0, 0, 0, 0, None), "ditch": (0, 0, 0, 0, None),
                 "tree_density": (0, 0, 0, 0, None), "tree_add": (4, 4, 10, 0, "vegetation_tree"), "tree_remove": (0, 0, 0, 0, None)}
_EDGE = None


def pavement_edges():
    """station-indexed pavement half widths (road.outer of geom.Road), cached in cache/edges.npz"""
    global _EDGE
    if _EDGE is None:
        f = CACHE / "edges.npz"
        if not f.exists():
            import sys; sys.path.insert(0, str(ROOT / "scripts/p2")); import geom
            r = geom.Road(None); i = np.arange(len(r.s))
            np.savez(f, s=r.s, L=r.outer(i, "L"), R=r.outer(i, "R"))
        d = np.load(f); _EDGE = (d["s"], d["L"], d["R"])
    return _EDGE


def ledger_place(o):
    """ledger object -> list of world points along it (linear: s..s1 every 2 m) with (x, y, zground, heading)"""
    es, eL, eR = pavement_edges()
    s0 = float(o["s"]); s1 = o.get("s1"); s1 = float(s1) if s1 is not None else s0
    ss = np.arange(s0, s1 + 1e-6, 2.0) if s1 > s0 + 1 else np.array([s0])
    side = 1 if o.get("side", "R") == "L" else -1
    half = np.interp(ss, es, eL if side > 0 else eR)
    d = side * (half + float(o.get("off", 0) or 0))
    x = np.interp(ss, S, X); y = np.interp(ss, S, Y); z = np.interp(ss, S, Z)
    i = np.clip(np.round(ss).astype(int), 0, len(S) - 1)
    px, py = x - TY[i] * d, y + TX[i] * d
    # same rule as 21e: blend from road-edge height to the DTM ground over the first 1.2 m beyond the edge
    offv = float(o.get("off", 0) or 0)
    if o.get("cls") not in ("building", "bridge", "marking", "driveway") and offv > 0:
        zg = _ground(px, py); w = min(max(offv / 1.2, 0), 1)
        z = np.where(np.isfinite(zg), z * (1 - w) + zg * w, z)
    return px, py, z + float(o.get("dz", 0) or 0), np.arctan2(TY[i], TX[i])


_DTM = None
def _ground(x, y):
    global _DTM
    if _DTM is None:
        import sys; sys.path.insert(0, str(ROOT / "scripts/p2")); import geom
        D = ROOT / "data/derived"; _DTM = (geom.Raster(D / "dtm_r050_b40.tif"), geom.Raster(D / "dtm_r100_b150.tif"))
    z = _DTM[0].sample(np.atleast_1d(x), np.atleast_1d(y)); z2 = _DTM[1].sample(np.atleast_1d(x), np.atleast_1d(y))
    return np.where(np.isfinite(z), z, z2)


def _box(cx, cy, z0, hd, w, dp, h):
    c, s = np.cos(hd), np.sin(hd)
    corners = []
    for a, b in ((-w / 2, -dp / 2), (w / 2, -dp / 2), (w / 2, dp / 2), (-w / 2, dp / 2)):
        corners.append((cx + a * c - b * s, cy + a * s + b * c))
    P = np.array([[x, y, z0] for x, y in corners] + [[x, y, z0 + h] for x, y in corners])
    F = [(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7), (4, 5, 6), (4, 6, 7)]
    return P[np.array(F)]


def load_ledger():
    p = SCENE / "ledger.jsonl"
    if not p.exists(): return []
    out = []
    for ln in open(p):
        ln = ln.strip()
        if ln:
            try: out.append(json.loads(ln))
            except json.JSONDecodeError: pass
    return out


def ledger_tris(ledger, catalog, s_lo, s_hi):
    """proxy triangles for ledger objects in [s_lo, s_hi]; returns V [n,3,3], C, inst (index into ledger, -1 none)"""
    Vs, Cs, Is = [], [], []
    for j, o in enumerate(ledger):
        if o.get("status") == "rejected": continue
        s0 = float(o.get("s", -1e9)); s1 = float(o.get("s1") or s0)
        if s1 < s_lo or s0 > s_hi: continue
        cat = catalog.get(o.get("asset"), {})
        w, dp, h, z0, cl = DEFAULT_PROXY.get(o.get("cls"), (1, 1, 1, 0, "other_prop"))
        sz = cat.get("proxy") or cat.get("size")
        if isinstance(sz, dict):            # assets.py catalog: x fwd (along road), y out, z up
            shp = sz.get("shape", "box")
            if shp == "none":
                continue
            if shp == "plane" and o.get("s1") is not None:   # linear sheet (fence panel) runs along the road
                w, dp, h = sz.get("w", w), sz.get("d", 0.1), sz.get("h", h)
            elif shp == "plane":              # vertical sheet facing the road traffic (spans across, thin along)
                w, dp, h = sz.get("d", 0.1), sz.get("w", w), sz.get("h", h)
            elif shp == "plane_h":          # flat on the ground
                w, dp, h = sz.get("w", w), sz.get("h", sz.get("d", dp)), 0.03
            else:
                w, dp, h = sz.get("w", sz.get("l", w)), sz.get("d", dp), sz.get("h", h)
        elif sz: w, dp, h = list(sz)[:3]
        if o.get("cls") == "marking" and (o.get("params") or {}).get("width"): dp = float(o["params"]["width"])
        z0 = cat.get("z0", z0); cl = cat.get("class", cat.get("seg_class", cl))
        if cl is None or cl not in CID: continue
        sc = float(o.get("scale", 1) or 1); w, dp, h = w * sc, dp * sc, h * sc
        x, y, z, hd = ledger_place(o)
        hd = hd + np.radians(float(o.get("yaw", 0) or 0))
        if len(x) > 1:          # linear object: one box per 2 m piece
            for k in range(len(x) - 1):
                L = np.hypot(x[k + 1] - x[k], y[k + 1] - y[k]); a = np.arctan2(y[k + 1] - y[k], x[k + 1] - x[k])
                V = _box((x[k] + x[k + 1]) / 2, (y[k] + y[k + 1]) / 2, z[k] + z0, a, L, max(dp, .05), h)
                Vs.append(V); Cs.append(np.full(len(V), CID[cl])); Is.append(np.full(len(V), j))
        else:
            V = _box(x[0], y[0], z[0] + z0, hd[0], max(w, .05), max(dp, .05), h)
            Vs.append(V); Cs.append(np.full(len(V), CID[cl])); Is.append(np.full(len(V), j))
            if o.get("cls") == "sign" and z0 > 0.3:        # its post
                V = _box(x[0], y[0], z[0], hd[0], .08, .08, z0)
                Vs.append(V); Cs.append(np.full(len(V), CID["pole"])); Is.append(np.full(len(V), j))
    if not Vs: return np.zeros((0, 3, 3)), np.zeros(0, np.uint8), np.zeros(0, int)
    return np.concatenate(Vs), np.concatenate(Cs).astype(np.uint8), np.concatenate(Is)


# ---------------------------------------------------------------- renderer
def render(p, w=AW, h=AH, ahead=250.0, behind=30.0, ledger=None, catalog=None, backdrop=True, classes=None):
    """class-ID image (uint8 h x w) + instance image (int32, -1 = none) from pose p, pinhole vfov FOV.
    Painter's algorithm (far -> near) with PIL polygons; tris with any vertex behind 0.3 m are dropped."""
    eye, fwd, rt, up = cam_basis(p)
    s = p[0]; k0 = int(max(s - behind, 0) // CHUNK); k1 = int(min(s + ahead, S_MAX) // CHUNK)
    Vs, Cs = [], []
    for k in range(k0, k1 + 1):
        V, C = chunk(k); Vs.append(V); Cs.append(C)
    V = np.concatenate(Vs).astype(np.float64); C = np.concatenate(Cs); I = np.full(len(C), -1)
    if backdrop:
        B = backdrop_tris(eye); V = np.concatenate([B, V]); C = np.concatenate([np.full(len(B), CID["terrain_grass"], np.uint8), C])
        I = np.concatenate([np.full(len(B), -1), I])
    if ledger:
        LV, LC, LI = ledger_tris(ledger, catalog or {}, s - behind, s + ahead)
        V = np.concatenate([V, LV]); C = np.concatenate([C, LC]); I = np.concatenate([I, LI])
    if classes is not None:
        m = np.isin(C, list(classes)); V, C, I = V[m], C[m], I[m]
    D = V - eye
    zc = D @ fwd; u = D @ rt; v = D @ up
    fo = (h / 2) / np.tan(np.radians(FOV) / 2)
    ok = (zc > 0.3).all(1)
    zc, u, v, C, I = zc[ok], u[ok], v[ok], C[ok], I[ok]
    px = w / 2 + fo * u / zc; py = h / 2 - fo * v / zc
    inb = ~((px.max(1) < 0) | (px.min(1) >= w) | (py.max(1) < 0) | (py.min(1) >= h))
    # sort far -> near by nearest vertex depth (small tris -> ok); markings/ledger get a small bias toward camera
    dep = zc.min(1) - np.where((C == 3) | (C == 4), 0.15, 0) - np.where(I >= 0, 0.3, 0)
    order = np.nonzero(inb)[0]; order = order[np.argsort(-dep[order], kind="stable")]
    img = Image.new("L", (w, h), CID["sky"]); dr = ImageDraw.Draw(img)
    inst = Image.new("I", (w, h), -1) if (I >= 0).any() else None; di = ImageDraw.Draw(inst) if inst else None
    pxl, pyl, Cl, Il = px.tolist(), py.tolist(), C.tolist(), I.tolist()
    for t in order.tolist():
        q = ((pxl[t][0], pyl[t][0]), (pxl[t][1], pyl[t][1]), (pxl[t][2], pyl[t][2]))
        dr.polygon(q, fill=Cl[t])
        if di is not None:
            di.polygon(q, fill=Il[t])
    return np.asarray(img), (np.asarray(inst, np.int32) if inst else np.full((h, w), -1, np.int32))
