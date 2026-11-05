"""Terrain macro colour layer (no more tilemap), used by 22_build_kn5.py.

Shader: ksPerPixelMultiMap. AC combines detail as a plain multiply where txDiffuse alpha is 0:
    colour = txDiffuse.rgb * lerp(txDetail(uv * detailUVMultiplier).rgb, 1, txDiffuse.a)
(the Kunos car workflow: AO diffuse with black alpha x body-colour detail; see docs/phase2/CSP_PURE.md).

  txDiffuse = unique per-chunk macro map (MACRO_PX^2 over the chunk's square UV window from 21):
              USGS ortho (1 m, summer-gold) re-graded to the spring GoPro palette (open/bright -> spring grass,
              dark/canopy -> leaf litter), + fbm noise at ~6/20/60 m + unique fbm. Stored as a
              multiplier around MACRO_MEAN (alpha 0 = detail everywhere).
  txDetail  = the calibrated class texture (ground / bank / litter, ~6 m period), raw-brightened by 1/MACRO_MEAN
              so macro x detail keeps the materials.yaml albedo on average.
Scales: ortho (~1.5 m) + fbm 6/20/60 m unique in the macro, ~5-6 m class detail (per-class periods) -> 3+ scales, no visible period.
"""
import io, json
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter, zoom

MACRO_PX = 1024
MACRO_MEAN = 0.8      # raw (sRGB-space) mean of the macro multiplier; detail textures are divided by it
DETAIL_PERIOD = {"ground": 6.1, "bank": 5.3, "litter": 5.7}   # m per detail tile
ROCK_BASE_M, ROCK_DETAIL_MULT = 20.0, 4.3                    # rock: 20 m diffuse tile x 4.65 m detail tile


def _fbm(r, n, cells):
    out = np.zeros((n, n))
    for c, amp in cells:
        k = max(2, int(round(n / c)) + 2)
        g = r.normal(0, 1, (k, k))
        out += amp * zoom(g, n / (k - 1), order=3)[:n, :n]
    return out


class Ortho:
    def __init__(self, path, origin):
        import rasterio
        with rasterio.open(path) as d:
            self.a = d.read().astype(np.float32)   # 3 x H x W
            self.t = d.transform
        self.x0, self.y0 = origin["x0"], origin["y0"]

    def sample(self, X, Y):
        col = (X + self.x0 - self.t.c) / self.t.a - 0.5
        row = (Y + self.y0 - self.t.f) / self.t.e - 0.5
        return np.stack([map_coordinates(b, [row, col], order=1, mode="nearest") for b in self.a], -1)


def macro_map(mc, ortho, pal, lab, rgb, tile_gray, seed):
    """mc: dict(x0, y0, size) local m. Row i <-> v = i/N <-> y = y0 + v*size (kn5 v is image-top-down)."""
    N = MACRO_PX; r = np.random.default_rng(seed)
    t = (np.arange(N) + 0.5) / N * mc["size"]
    X, Y = np.meshgrid(mc["x0"] + t, mc["y0"] + t)
    o = gaussian_filter(ortho.sample(X, Y), (1.5 / (mc["size"] / N),) * 2 + (0,))   # ~1.5 m: ortho grain off
    L = lab(o.clip(0, 255))
    Lo = L[..., 0]
    # open (bright, sunlit dry grass in the summer ortho) -> spring grass; dark (canopy / shade) -> leaf litter
    w = np.clip((Lo - np.median(Lo)) / 18 + 0.55, 0, 1)
    w = gaussian_filter(w, 2.0)
    gp, lp = np.array(pal["grass"]["lab_mean"]), np.array(pal["litter"]["lab_mean"])
    tgt = lp[None, None] * (1 - w[..., None]) + gp[None, None] * w[..., None]
    px = mc["size"] / N
    tgt[..., 0] += 0.25 * (Lo - Lo.mean())
    tgt[..., 0] += _fbm(r, N, ((6 / px, 2.5), (20 / px, 2.5), (60 / px, 3.0)))
    tgt[..., 1] += _fbm(r, N, ((30 / px, 2.5), (90 / px, 2.0)))      # green <-> olive drift
    tgt[..., 2] += _fbm(r, N, ((25 / px, 2.5), (80 / px, 2.5)))      # fresh <-> yellowing
    ref = (gp + lp) / 2
    c = rgb(tgt).astype(np.float32) / 255; c0 = rgb(ref[None, None]).astype(np.float32)[0, 0] / 255
    M = 1 + 0.45 * (c / np.maximum(c0, 1e-3) - 1)
    M[..., 2] = np.minimum(M[..., 2], 0.5 * (M[..., 0] + M[..., 1]))   # spring: never bluer than neutral
    # 17.3 m luminance tile (mid scale)
    # (a baked 17.3 m luminance tile was tried here: it reads as a visible lattice from height -> removed;
    #  mid-scale comes from the unique fbm octaves instead)
    M = np.clip(M, 0.55, 1.25) * MACRO_MEAN
    M = M / M.reshape(-1, 3).mean(0).mean() * MACRO_MEAN
    a = np.dstack([np.clip(M, 0, 1) * 255, np.zeros((N, N))]).astype(np.uint8)
    return Image.fromarray(a, "RGBA")


def tile_gray_from(img, n=512):
    g = np.asarray(img.convert("L").resize((n, n)), np.float32)
    g = gaussian_filter(g, n / 17.3 * 0.6, mode="wrap")   # >= ~1 m features: no aliasing at macro resolution
    return (g - g.mean()) / (g.std() + 1e-6)


def detail_from(png_bytes, mult=1 / MACRO_MEAN):
    a = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"), np.float32) * mult
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def rock_textures(png_bytes):
    """rock: diffuse = calibrated rock tile (alpha 0, raw / MACRO_MEAN), detail = its own luminance at a
    different scale, mean MACRO_MEAN -> 2 non-commensurate scales."""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    a = np.asarray(im, np.float32) / MACRO_MEAN
    d = np.dstack([a.clip(0, 255), np.zeros(a.shape[:2])]).astype(np.uint8)
    g = np.asarray(im.convert("L").rotate(90), np.float32)
    g = MACRO_MEAN * 255 * (1 + 0.35 * (g - g.mean()) / (g.std() + 1e-6))
    det = np.repeat(g.clip(0, 255)[..., None], 3, -1).astype(np.uint8)
    return Image.fromarray(d, "RGBA"), Image.fromarray(det)
