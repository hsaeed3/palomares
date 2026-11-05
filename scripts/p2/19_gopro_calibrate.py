"""Phase 2: calibrate materials from the user's spring GoPro frames (assets/gopro/*.png).

Outputs data/derived/gopro/:
  palette.json        robust per-class colour stats (sRGB + Lab mean/std) over all frames
  palette_by_frame.csv per-frame class means (lets later builds vary colour along the road)
  foliage_XX.png      dense-foliage patches (256 px) used to fill the tree-card crowns
  preview.png         swatches for review

Wet road: the frames were shot on a damp day. Low-saturation road pixels split into two
luminance modes per frame; the darker mode is water. Only the dry mode feeds the asphalt
colour, and the wet share is recorded per frame (column wet_frac).
"""
import csv, glob, json, pathlib
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data/derived/gopro"; OUT.mkdir(parents=True, exist_ok=True)
FR = sorted(glob.glob(str(ROOT / "assets/gopro/*.png")))
W, H = 640, 466  # analysis resolution


def to_lab(rgb):
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def hsv(a):
    r, g, b = (a[..., i] / 255.0 for i in range(3))
    mx = np.max(a, -1) / 255.0; mn = np.min(a, -1) / 255.0; d = mx - mn + 1e-9
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, d / (mx + 1e-9), mx


yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W])[:, None, None]
lane_box = (yy > 0.72) & (yy < 0.96) & (xx > 0.30) & (xx < 0.80)
low = yy > 0.55
CLS = ("asphalt_dry", "asphalt_wet", "litter", "grass", "crown", "crown_shadow", "dirt_cut")
acc = {k: [] for k in CLS}
rows = []
fol = []
rng = np.random.default_rng(0)
for f in FR:
    big = Image.open(f).convert("RGB")
    a = np.asarray(big.resize((W, H)), float)
    a[int(H * 0.93):, int(W * 0.9):] = np.nan  # compass overlay
    a[:int(H * 0.06), int(W * 0.85):] = np.nan  # player controls overlay
    ok = np.isfinite(a).all(-1)
    h, s, v = hsv(np.nan_to_num(a))
    m = {}
    road = lane_box & ok & (s < 0.10) & (v < 0.80) & (v > 0.12)  # drop white/yellow paint, black shadows
    L = v[road]
    if len(L) > 200:
        # two-mode split by Otsu on luminance; dry = upper mode, wet only if the modes are well separated
        hist, edges = np.histogram(L, 64, (0, 1)); p = hist / hist.sum(); c = np.cumsum(p); mu = np.cumsum(p * edges[:-1])
        sb = (mu[-1] * c - mu) ** 2 / (c * (1 - c) + 1e-12); t = edges[np.argmax(sb)]
        lo, hi = L[L < t], L[L >= t]
        separated = len(lo) > 0.08 * len(L) and hi.mean() - lo.mean() > 0.12
        if separated:
            m["asphalt_dry"] = road & (v >= t); m["asphalt_wet"] = road & (v < t); wet = len(lo) / len(L)
        else:
            m["asphalt_dry"] = road; wet = 0.0
    else:
        wet = np.nan
    side = low & ~lane_box & ok
    m["litter"] = side & (h > 15) & (h < 45) & (s > 0.15) & (s < 0.55) & (v > 0.2) & (v < 0.7)
    m["grass"] = low & ok & (h > 60) & (h < 110) & (s > 0.35) & (v > 0.35)
    up = (yy < 0.6) & ok
    m["crown"] = up & (h > 60) & (h < 120) & (s > 0.3) & (v > 0.35)
    m["crown_shadow"] = up & (h > 50) & (h < 140) & (s > 0.25) & (v > 0.08) & (v <= 0.35)
    m["dirt_cut"] = (yy > 0.3) & ok & ~lane_box & (h > 18) & (h < 40) & (s > 0.18) & (s < 0.5) & (v > 0.35) & (v < 0.75)
    row = {"frame": pathlib.Path(f).stem, "wet_frac": round(float(wet), 3) if np.isfinite(wet) else ""}
    for k in CLS:
        px = a[m[k]] if k in m else np.zeros((0, 3))
        if len(px) > 50:
            acc[k].append(px[rng.choice(len(px), min(len(px), 3000), replace=False)])
            row[k] = "#%02x%02x%02x" % tuple(int(x) for x in np.median(px, 0))
        else:
            row[k] = ""
    rows.append(row)
    # foliage patches: 256 px windows (full-res) in upper half that are ~all leaf pixels
    if len(fol) < 40 and int(row["frame"]) % 4 == 0:
        A = np.asarray(big, float); hh, ss, vv = hsv(A)
        leaf = (hh > 50) & (hh < 130) & (ss > 0.2) & (vv > 0.06)
        for _ in range(60):
            y0 = rng.integers(int(A.shape[0] * 0.05), int(A.shape[0] * 0.45)); x0 = rng.integers(0, A.shape[1] - 256)
            if leaf[y0:y0 + 256, x0:x0 + 256].mean() > 0.93 and vv[y0:y0 + 256, x0:x0 + 256].std() > 0.08:
                big.crop((x0, y0, x0 + 256, y0 + 256)).save(OUT / f"foliage_{len(fol):02d}.png"); fol.append(row["frame"]); break

pal = {}
for k in CLS:
    if not acc[k]:
        continue
    px = np.concatenate(acc[k]); lab = to_lab(px)
    pal[k] = {"srgb_median": [int(x) for x in np.median(px, 0)], "lab_mean": lab.mean(0).round(2).tolist(),
              "lab_std": lab.std(0).round(2).tolist(), "n_px": int(len(px))}
wf = np.array([r["wet_frac"] for r in rows if r["wet_frac"] != ""], float)
pal["_meta"] = {"frames": len(FR), "wet_frac_mean": round(float(wf.mean()), 3), "frames_with_wet_patches": int((wf > 0.05).sum()),
                "foliage_patches": fol, "note": "asphalt_dry excludes the darker luminance mode (water) in each frame"}
json.dump(pal, open(OUT / "palette.json", "w"), indent=1)
with open(OUT / "palette_by_frame.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, ["frame", "wet_frac", *CLS]); w.writeheader(); w.writerows(rows)
sw = Image.new("RGB", (80 * len(CLS), 80))
for i, k in enumerate(CLS):
    if k in pal:
        sw.paste(tuple(pal[k]["srgb_median"]), (80 * i, 0, 80 * i + 80, 80))
sw.save(OUT / "preview.png")
print(json.dumps({k: v.get("srgb_median") for k, v in pal.items() if k != "_meta"}), pal["_meta"])
