#!/usr/bin/env python3
"""P: asset / chunk viewer (docs/PIPELINE.md). Renders in seconds, no AC, no Blender, no browser.

  python3 scripts/p2/view.py asset sign.curve_left           # assets/<type>/ (build with assetlib.py first)
  python3 scripts/p2/view.py asset tree.oak --frame 120      # beside GoPro frame 120
  python3 scripts/p2/view.py chunk 9300 [--len 200] [--dir build_quick]   # one 200 m stretch of geometry.pkl

Writes reports/view/<name>.jpg (contact strip: a few angles + the nearest GoPro frame, assets/gopro/NNN.png from
data/scene/poses.json, or a local Mapillary image if one was fetched) and reports/view/<name>.html.
Renderer: numpy pinhole projection + painter's-sorted PIL polygons, Lambert shade, colour = diffuse texture sampled
at each triangle's UVs (alpha-tested texels dropped).
"""
import argparse, io, json, math, pathlib, pickle, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "reports" / "view"
W, H = 640, 400
SUN = np.array([0.35, -0.45, 0.82]); SUN /= np.linalg.norm(SUN)
SKY = (178, 196, 214)


# ------------------------------------------------------------------ rendering
def tri_colors(UV, T, tex, alpha_test=True):
    """UV (n,2), T (m,3), tex PIL RGBA -> (m,3) uint8 colours, (m,) keep mask"""
    a = np.asarray(tex.convert("RGBA"), np.uint8); th, tw = a.shape[:2]
    def at(uv):
        # pipeline convention: mesh v is measured from the image top of the UNflipped tile sheet, the stored PNG/DDS is
        # flipped top-bottom (roadside_tex.Atlas.png) and kn5.py writes -v, so a texture row is (1 - v)
        u = np.mod(uv[..., 0], 1.0); v = np.mod(-uv[..., 1], 1.0)
        return a[np.clip((v * th).astype(int), 0, th - 1), np.clip((u * tw).astype(int), 0, tw - 1)]
    tuv = UV[T]                                   # (m,3,2)
    samples = np.stack([at(tuv.mean(1))] + [at(0.6 * tuv[:, k] + 0.4 * tuv.mean(1)) for k in range(3)], 1)  # (m,4,4)
    best = samples[np.arange(len(T)), samples[:, :, 3].argmax(1)]
    keep = best[:, 3] >= 128 if alpha_test else np.ones(len(T), bool)
    return best[:, :3], keep


def render(P, T, col, eye, target, fov=55.0, w=W, h=H, sky=SKY, far=1e9):
    """P (n,3) world (z up), T (m,3), col (m,3). Returns PIL RGB."""
    eye = np.asarray(eye, float); f = np.asarray(target, float) - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r) + 1e-12; u = np.cross(r, f)
    q = P - eye; xc, yc, zc = q @ r, q @ u, q @ f
    fp = 0.5 * h / math.tan(math.radians(fov) / 2)
    tri_z = zc[T]
    ok = (tri_z > 0.3).all(1) & (tri_z.mean(1) < far)
    T2 = T[ok]; c2 = col[ok]
    if not len(T2):
        return Image.new("RGB", (w, h), sky)
    sx = w / 2 + fp * xc / np.maximum(zc, 1e-3); sy = h / 2 - fp * yc / np.maximum(zc, 1e-3)
    e1 = P[T2[:, 1]] - P[T2[:, 0]]; e2 = P[T2[:, 2]] - P[T2[:, 0]]
    n = np.cross(e1, e2); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.55 + 0.45 * np.abs(n @ SUN)        # double-sided (cards, sheets)
    c3 = np.clip(c2 * shade[:, None], 0, 255).astype(np.uint8)
    order = np.argsort(-zc[T2].mean(1))
    img = Image.new("RGB", (w, h), sky); d = ImageDraw.Draw(img)
    X, Y = sx[T2], sy[T2]
    inside = ~((X.max(1) < 0) | (X.min(1) > w) | (Y.max(1) < 0) | (Y.min(1) > h))
    for k in order:
        if inside[k]:
            d.polygon([(X[k, 0], Y[k, 0]), (X[k, 1], Y[k, 1]), (X[k, 2], Y[k, 2])], fill=tuple(int(v) for v in c3[k]))
    return img


def render_tex(P, T, UV, tex, eye, target, fov=55.0, w=W, h=H, sky=SKY, alpha_test=True):
    """textured painter's renderer (dev7): per-triangle affine texture warp, Lambert shade. Slower than render()
    (about 1 ms per triangle), used for single assets so facades / sign faces can be judged by eye."""
    eye = np.asarray(eye, float); f = np.asarray(target, float) - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r) + 1e-12; u = np.cross(r, f)
    q = P - eye; xc, yc, zc = q @ r, q @ u, q @ f
    fp = 0.5 * h / math.tan(math.radians(fov) / 2)
    ok = (zc[T] > 0.3).all(1); T2 = T[ok]
    img = Image.new("RGBA", (w, h), sky + (255,))
    if not len(T2):
        return img.convert("RGB")
    sx = w / 2 + fp * xc / zc; sy = h / 2 - fp * yc / zc
    e1 = P[T2[:, 1]] - P[T2[:, 0]]; e2 = P[T2[:, 2]] - P[T2[:, 0]]
    n = np.cross(e1, e2); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.55 + 0.45 * np.abs(n @ SUN)
    ta = np.asarray(tex.convert("RGBA")); th, tw = ta.shape[:2]
    for k in np.argsort(-zc[T2].mean(1)):
        t = T2[k]; X = np.c_[sx[t], sy[t]]
        x0, y0 = np.floor(X.min(0)).astype(int); x1, y1 = np.ceil(X.max(0)).astype(int) + 1
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) > w * h:
            continue
        uv = UV[t]; tx = np.mod(uv[:, 0], 1.0) * tw; ty = np.mod(-uv[:, 1], 1.0) * th
        A = np.c_[X - [x0, y0], np.ones(3)]
        try:
            cx = np.linalg.solve(A, tx); cy = np.linalg.solve(A, ty)
        except np.linalg.LinAlgError:
            continue
        gx, gy = np.meshgrid(np.arange(x1 - x0) + 0.5, np.arange(y1 - y0) + 0.5)
        U = np.clip((cx[0] * gx + cx[1] * gy + cx[2]).astype(int), 0, tw - 1); V = np.clip((cy[0] * gx + cy[1] * gy + cy[2]).astype(int), 0, th - 1)
        patch = ta[V, U].astype(np.float32); patch[..., :3] *= shade[k]
        m = Image.new("L", (x1 - x0, y1 - y0), 0); ImageDraw.Draw(m).polygon([tuple(v) for v in (X - [x0, y0])], fill=255)
        ma = np.asarray(m) > 0
        if alpha_test:
            ma &= patch[..., 3] >= 128
        reg = np.asarray(img.crop((x0, y0, x1, y1))).copy(); reg[ma] = patch[ma].astype(np.uint8); reg[..., 3] = 255
        img.paste(Image.fromarray(reg), (x0, y0))
    return img.convert("RGB")


def label(img, text):
    d = ImageDraw.Draw(img); d.rectangle([0, 0, img.width, 18], fill=(0, 0, 0)); d.text((4, 3), text, fill=(255, 255, 255))
    return img


def strip(tiles, path, title):
    OUT.mkdir(parents=True, exist_ok=True)
    cols = 2; rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * W, rows * H + 24), (20, 20, 20))
    ImageDraw.Draw(sheet).text((6, 6), title, fill=(255, 255, 255))
    for i, (t, im) in enumerate(tiles):
        im = im.convert("RGB"); im.thumbnail((W, H)); tile = Image.new("RGB", (W, H), (40, 40, 40))
        tile.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
        sheet.paste(label(tile, t), ((i % cols) * W, 24 + (i // cols) * H))
    sheet.save(path, quality=88)
    html = path.with_suffix(".html")
    html.write_text(f"<!doctype html><meta charset=utf-8><title>{title}</title><body style='background:#111;color:#eee;"
                    f"font-family:sans-serif'><h3>{title}</h3><img src='{path.name}' style='max-width:100%'></body>")
    return path


# ------------------------------------------------------------------ reference photos
def poses():
    return json.load(open(ROOT / "data/scene/poses.json"))["poses"]


def gopro(frame):
    p = ROOT / f"assets/gopro/{int(frame):03d}.png"
    return Image.open(p) if p.exists() else None


def nearest_photo(s):
    ps = [p for p in poses() if p.get("s") is not None]
    p = min(ps, key=lambda p: abs(p["s"] - s))
    im = gopro(p["frame"])
    if im is not None and abs(p["s"] - s) < 300:
        return f"GoPro frame {p['frame']} (s {p['s']:.0f})", im, p
    try:   # local Mapillary sample (scripts/p7 fetch), if any
        fc = json.load(open(ROOT / "data/derived/mapillary_images.geojson"))["features"]
        best = min(fc, key=lambda f: abs(f["properties"]["s"] - s))
        for d in (ROOT / "data/raw").glob("**/mapillary*"):
            for q in d.glob(f"**/{best['properties']['id']}.jpg"):
                return f"Mapillary {best['properties']['id']} (s {best['properties']['s']:.0f}, CC-BY-SA)", Image.open(q), None
    except Exception:
        pass
    return None, None, None


# ------------------------------------------------------------------ asset view
def view_asset(aid, frame=None):
    from dds import to_png
    d = ROOT / "assets" / aid
    man = json.loads((d / "manifest.json").read_text())
    tiles = []
    for part in man["parts"]:
        z = np.load(d / part["mesh"])
        tex = Image.open(io.BytesIO(to_png((ROOT / "assets" / part["texture"]).read_bytes())))
        c, keep = tri_colors(z["UV"], z["T"], tex, part.get("alpha_tested", True))
        part["_m"] = (z["P"], z["T"][keep], c[keep])
    textured = man.get("mly") is not None
    P = np.concatenate([p["_m"][0] for p in man["parts"]])
    off = np.cumsum([0] + [len(p["_m"][0]) for p in man["parts"]])[:-1]
    T = np.concatenate([p["_m"][1] + o for p, o in zip(man["parts"], off)]); C = np.concatenate([p["_m"][2] for p in man["parts"]])
    lo, hi = P.min(0), P.max(0); ctr = (lo + hi) / 2; R = max(np.linalg.norm(hi - lo), 0.5)
    views = (("front", -90, 8), ("3/4", -40, 18), ("side", 0, 8), ("top", -60, 60))
    if man.get("mly") is not None:            # dev7 types: signs face -x, so add the approaching driver's view
        views = (("driver (-x)", 180, 6), ("3/4 road", -135, 15), ("side (-y)", -90, 8), ("top", -60, 60))
    for name, az, el in views:
        a, e = math.radians(az), math.radians(el)
        eye = ctr + 1.5 * R * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        if textured:                                   # dev7: per-pixel textures, one pass per part
            im = None
            for part in man["parts"]:
                z = np.load(d / part["mesh"]); tex = Image.open(io.BytesIO(to_png((ROOT / "assets" / part["texture"]).read_bytes())))
                if im is None:
                    im = render_tex(z["P"], z["T"], z["UV"], tex, eye, ctr, fov=40, alpha_test=part.get("alpha_tested", True))
                else:   # small emissive parts: flat colour on top
                    c, keep = tri_colors(z["UV"], z["T"], tex, False)
                    over = render(z["P"], z["T"], c, eye, ctr, fov=40, sky=(1, 2, 3))
                    oa = np.asarray(over); msk = (oa != (1, 2, 3)).any(2); ia = np.asarray(im).copy(); ia[msk] = oa[msk]; im = Image.fromarray(ia)
            tiles.append((f"{aid} {name}", im))
        else:
            tiles.append((f"{aid} {name}", render(P, T, C, eye, ctr, fov=40)))
    for mid in man.get("mly", [])[:2]:                       # dev7: Mapillary images from the ledger row (CC-BY-SA)
        q = ROOT / f"data/raw/dev7/mapillary/img/{mid}.jpg"
        if q.exists():
            tiles.append((f"Mapillary {mid} (CC-BY-SA)", Image.open(q)))
    frames = [frame] if frame else man.get("frames", [])[:1]
    for f in frames:
        im = gopro(f)
        if im is not None:
            tiles.append((f"GoPro frame {f}", im))
    title = f"{aid}: {man['verts']} verts, {man['tris']} tris, LODs {len(man.get('lods', []))}, src {man['source_hash'][:10]}"
    return strip(tiles, OUT / f"asset_{aid}.jpg", title)


# ------------------------------------------------------------------ chunk view
def view_chunk(s0, length=200.0, bdir=None):
    sys.path.insert(0, str(HERE))
    import geom
    bdir = pathlib.Path(bdir or ROOT / "build_quick")
    t0 = time.time()
    geo = pickle.load(open(bdir / "geometry.pkl", "rb"))
    import kn5_quick
    MATS, TEX = kn5_quick.load_state(geo["groups"], bdir, bdir / "kn5_cache")   # = what 22 quick writes (atlases, terrain remap)
    cl = np.array([[float(r[k]) for k in ("s_m", "x_local", "y_local", "z_navd88")] for r in
                   __import__("csv").DictReader(open(ROOT / "data/derived/centerline.csv"))])
    sel = (cl[:, 0] >= s0 - 40) & (cl[:, 0] <= s0 + length + 260)
    box_lo, box_hi = cl[sel, 1:3].min(0) - 120, cl[sel, 1:3].max(0) + 120
    texc = {}
    Ps, Ts, Cs, n = [], [], [], 0
    for g, ms in geo["groups"].items():
        if g in ("backdrop",) or g.startswith(("kerbs", "dressing")):
            continue
        for m in ms:
            if m["name"].startswith("1WALL") or not m.get("renderable", True) or not m.get("visible", True):
                continue
            P = np.asarray(m["P"], float)
            if not len(P) or (P[:, :2].max(0) < box_lo).any() or (P[:, :2].min(0) > box_hi).any():
                continue
            T = np.asarray(m["T"]).reshape(-1, 3)
            c = P[T].mean(1)[:, :2]
            k = ((c >= box_lo) & (c <= box_hi)).all(1)
            if not k.any():
                continue
            mat = MATS.get(m["material"]); tn = mat.textures.get("txDiffuse") if mat else None
            if tn not in texc:
                texc[tn] = Image.open(io.BytesIO(TEX[tn])) if tn in TEX else Image.new("RGBA", (4, 4), (128, 128, 128, 255))
            col, keep = tri_colors(np.asarray(m["UV"], float), T[k], texc[tn], bool(mat and mat.alpha_tested))
            Ps.append(P); Ts.append(T[k][keep] + n); Cs.append(col[keep]); n += len(P)
    P, T, C = np.concatenate(Ps), np.concatenate(Ts), np.concatenate(Cs)
    print(f"view chunk: {len(T)} tris from {bdir.name} in {time.time() - t0:.1f}s")
    def at(s):
        j = int(np.argmin(np.abs(cl[:, 0] - s))); j2 = min(j + 5, len(cl) - 1)
        t = cl[j2, 1:3] - cl[j, 1:3]; t /= np.linalg.norm(t) + 1e-9
        return cl[j, 1:4], t
    tiles = []
    for ds in (0.0, length / 2):
        p, t = at(s0 + ds)
        eye = p + np.array([0, 0, 1.3]); eye[:2] += np.array([t[1], -t[0]]) * 1.5   # right-hand lane, driver's eye
        tgt = eye + np.array([t[0], t[1], 0.0]) * 40
        tiles.append((f"driver's eye s {s0 + ds:.0f} (uphill)", render(P, T, C, eye, tgt, fov=75, far=700)))
    p, t = at(s0 + length / 2)
    tiles.append((f"overview s {s0:.0f}-{s0 + length:.0f}", render(P, T, C, p + np.array([-t[1] * 90 - t[0] * 60, t[0] * 90 - t[1] * 60, 70]), p, fov=60)))
    lab, im, pose = nearest_photo(s0)
    if im is not None:
        if pose is not None:   # render from the calibrated pose too
            pp, tt = at(pose["s"]); nx = np.array([tt[1], -tt[0]])   # +lateral = right
            yaw = math.radians(pose.get("yaw", 0.0)); fx = np.array([tt[0] * math.cos(yaw) + nx[0] * math.sin(yaw), tt[1] * math.cos(yaw) + nx[1] * math.sin(yaw)])
            eye = pp + np.r_[nx * pose.get("lateral", 0.0), pose.get("height", 1.5)]
            tgt = eye + np.r_[fx * 40, 40 * math.tan(math.radians(-pose.get("pitch", 0.0)))]
            tiles.append((f"model from GoPro pose {pose['frame']}", render(P, T, C, eye, tgt, fov=pose.get("fov_v", 75), far=700)))
        tiles.append((lab, im))
    return strip(tiles, OUT / f"chunk_{int(s0):05d}.jpg", f"chunk s {s0:.0f}-{s0 + length:.0f} from {bdir}  ({len(T)} tris)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["asset", "chunk"])
    ap.add_argument("name")
    ap.add_argument("--frame", type=int)
    ap.add_argument("--len", type=float, default=200.0)
    ap.add_argument("--dir", default=None)
    a = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, str(HERE))
    out = view_asset(a.name, a.frame) if a.what == "asset" else view_chunk(float(a.name), a.len, a.dir)
    print(f"-> {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
