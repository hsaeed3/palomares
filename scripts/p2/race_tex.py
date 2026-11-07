"""Procedural race-dressing textures (no external assets): kerb atlas + normal map, painted run-off,
galvanised W-beam guardrail. Used by 22_build_kn5.py. kn5 v runs bottom-up: images here are drawn with
v=0 at the TOP row and flipped at the end where it matters (kerb atlas is symmetric per block, so no flip)."""
import numpy as np
from PIL import Image, ImageFilter


def _noise(r, H, W, cell, octaves=3):
    out = np.zeros((H, W))
    for o in range(octaves):
        c = max(1, cell // (2 ** o))
        g = r.random((H // c + 2, W // c + 2))
        im = Image.fromarray((g * 255).astype(np.uint8)).resize(((W // c + 2) * c, (H // c + 2) * c), Image.BICUBIC)
        out += (np.asarray(im, float)[:H, :W] / 255 - 0.5) / (2 ** o)
    return out


def _normal_from_height(h, strength):
    gy, gx = np.gradient(h)
    n = np.dstack([-gx * strength, -gy * strength, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8))


RED, WHITE = np.array([196, 28, 32.]), np.array([236, 234, 228.])


def kerb_style(name="palomares"):
    """block colour sequence of a kerb style (config/kerbs.yaml style.<name>.blocks, RGB)"""
    import yaml, pathlib
    st = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[2] / "config/kerbs.yaml")).get("style", {})
    return [np.array(c, float) for c in st.get(name, {}).get("blocks", [RED.tolist(), WHITE.tolist()])]


def kerb_atlas(H=1024, W=1024, seed=11, blocks=None):
    """left half: apex kerb (u 0..0.5, ridged); right half: exit kerb (u 0.5..1, flat, smoother paint).
    v: one texture height = one cycle of the style's blocks (2 = red/white, 3 = Spa-style red/yellow/green);
    kerbs.py maps v = s / (n_blocks * stripe_length_m)."""
    blocks = blocks if blocks is not None else kerb_style("palomares")
    nb = len(blocks); r = np.random.default_rng(seed)
    v = np.arange(H)[:, None] / H; u = np.arange(W)[None, :] / W
    bi = np.minimum((v * nb).astype(int), nb - 1)
    fv = v * nb - bi
    # 6 px soft transition between blocks
    blend = np.clip(np.minimum(fv, 1 - fv) * H / nb / 3.0, 0, 1)
    pal = np.stack(blocks)
    col = pal[np.broadcast_to(bi, (H, W))]
    col = col * (0.85 + 0.15 * blend[..., None]) + (1 - blend[..., None]) * 0.15 * pal.mean(0)
    apex = u < 0.5
    ua = np.where(apex, u * 2, (u - 0.5) * 2)          # 0 = road edge .. 1 = outer edge, per half
    # height field: apex = transverse ridges (10 per block, ~12 cm), exit = fine grain
    ridge = 0.5 + 0.5 * np.cos(2 * np.pi * v * 10 * nb)
    grain = _noise(r, H, W, 8)
    h = np.where(apex, 0.8 * ridge ** 2 + 0.08 * grain, 0.12 * grain)
    # wear: rubber at the inner (road) edge, dirt / dust toward the outer edge, paint chips
    n1 = _noise(r, H, W, 64); n2 = _noise(r, H, W, 16)
    rubber = np.clip((0.35 - ua) / 0.35, 0, 1) * np.clip(0.55 + n1 * 1.2, 0, 1)
    dirt = np.clip((ua - 0.65) / 0.35, 0, 1) * np.clip(0.5 + n2, 0, 1)
    col = col * (1 - 0.55 * rubber[..., None]) + np.array([30, 28, 26.]) * 0.55 * rubber[..., None]
    col = col * (1 - 0.35 * dirt[..., None]) + np.array([118, 98, 72.]) * 0.35 * dirt[..., None]
    chips = (_noise(r, H, W, 4, 2) > 0.45) & (r.random((H, W)) > 0.75)
    col[chips] = col[chips] * 0.6 + np.array([70, 70, 68.]) * 0.4
    shade = np.where(apex, 0.80 + 0.20 * ridge ** 3, 1.0)  # ridge troughs hold grime
    col = col * shade[..., None] * (1 + 0.05 * grain[..., None])
    col[:, W // 2 - 3:W // 2 + 3] = (50, 50, 50)       # gutter between halves (never sampled, mip guard)
    d = Image.fromarray(col.clip(0, 255).astype(np.uint8))
    nm = _normal_from_height(h, 6.0)
    return d, nm


def runoff_tex(S=512, seed=12):
    """green painted run-off (anti-skid paint over asphalt), 2 m repeat, tileable via wrap-around noise"""
    r = np.random.default_rng(seed)
    base = np.array([46, 112, 62.])
    n = _noise(r, S + 64, S + 64, 32)[:S, :S]
    g = r.random((S, S))
    col = base * (0.9 + 0.25 * n[..., None]) * (0.88 + 0.24 * g[..., None])
    worn = (_noise(r, S, S, 24) > 0.28)
    col[worn] = col[worn] * 0.55 + np.array([70, 70, 66.]) * 0.45
    im = Image.fromarray(col.clip(0, 255).astype(np.uint8))
    # make tileable: blend with a half-shifted copy at the seams
    a = np.asarray(im, float); b = np.roll(a, (S // 2, S // 2), (0, 1))
    w = np.minimum(np.minimum(np.arange(S), S - 1 - np.arange(S)) / (S / 4), 1)
    m = np.minimum(w[:, None], w[None, :])[..., None]
    return Image.fromarray((a * m + b * (1 - m)).astype(np.uint8))


def guardrail_wbeam(H=128, W=512, seed=13):
    """galvanised W-beam: v across the 0.35 m beam height (two ridges), u along 4 m (splice bolts at the ends)"""
    r = np.random.default_rng(seed)
    v = (np.arange(H)[:, None] + 0.5) / H
    prof = np.cos(2 * np.pi * v * 2)                     # two crowns, two valleys = the W
    light = 0.78 + 0.22 * prof
    spec = np.clip(prof, 0, 1) ** 6 * 0.25
    base = np.array([168, 172, 176.])
    n = _noise(r, H, W, 32); sp = r.random((H, W))
    col = base * (light + 0.06 * n)[..., None] + 255 * spec[..., None]
    col *= (0.94 + 0.08 * sp)[..., None]
    rust = (_noise(r, H, W, 16) > 0.36) & (v > 0.75)
    col[rust] = col[rust] * 0.7 + np.array([120, 86, 60.]) * 0.3
    col[:2] *= 0.6; col[-2:] *= 0.6
    for x0 in (6, W - 14):                              # splice bolts
        for yb in (0.12, 0.38, 0.62, 0.88):
            y = int(yb * H); col[y - 3:y + 3, x0:x0 + 6] = (60, 62, 64)
    return Image.fromarray(col.clip(0, 255).astype(np.uint8))
