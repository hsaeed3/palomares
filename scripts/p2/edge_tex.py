"""road_edge.png: alpha-tested ragged asphalt-edge atlas (used by the edge_L/R strips built in 21_build_geometry.py).

4 rows (v top-down, as the other atlases): litter, dirt/gravel, grass, ivy. In each row x = along the road (1024 px = 4 m,
tiles in u), y = across: 0 = on the asphalt (road.strip_in inside the edge), EDGE_V = asphalt edge, 1 = shoulder mid.
  * y > ragged line (EDGE_V +- multiscale noise, bites up to ~0.25 m into the asphalt): opaque material = crumbled edge,
    missing chunks of pavement;
  * y < ragged line: sparse alpha-tested debris (gravel pebbles / leaves / grass tufts) thinning toward the lane.
Colours come from the same shoulder textures (after albedo calibration) so the strip matches the shoulder mesh.
"""
import numpy as np
from PIL import Image
from road_surface import pnoise

EDGE_V = 0.55
ROWS = ("litter", "dirt", "grass", "ivy")


def edge_atlas(src, W=1024, H=256, seed=5):
    """src: dict row-name -> PIL RGB image (shoulder material texture)"""
    out = np.zeros((H * 4, W, 4), np.uint8)
    yy = np.mgrid[0:H, 0:W][0] / (H - 1)
    for r, name in enumerate(ROWS):
        im = np.asarray(src[name].convert("RGB").resize((W, W)), float)[:H * 2:2]     # ~2x finer than the shoulder tiling
        n1 = pnoise((H, W), 60, seed + r)[:1].repeat(H, 0)          # along-road wobble of the crumble line
        n2 = pnoise((H, W), 14, seed + 10 + r)                        # bites / chunks
        n3 = pnoise((H, W), 2.5, seed + 20 + r)                       # fine crumbs
        line = EDGE_V - 0.10 * n1 - 0.07 * np.maximum(n2, 0)
        solid = yy + 0.025 * n3 > line
        # debris on the asphalt: density rises toward the edge
        dens = np.clip((yy - 0.05) / (EDGE_V - 0.05), 0, 1) ** 1.6
        thr = {"litter": 1.25, "dirt": 1.15, "grass": 1.9, "ivy": 1.7}[name]
        deb = (pnoise((H, W), 1.6 if name == "dirt" else 3.0, seed + 30 + r) > thr - 0.25 + 1.3 * (1 - dens)) & (yy > 0.04)
        a = solid | deb
        rgb = im.copy()
        rgb[deb & ~solid] *= 0.85                                     # debris slightly darker (shadowed by the asphalt grain)
        edge_band = solid & (yy < line + 0.06)
        rgb[edge_band] *= 0.78                                        # broken-edge shadow lip
        out[r * H:(r + 1) * H, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        out[r * H:(r + 1) * H, :, 3] = a * 255
        out[r * H:r * H + 3, :, 3] = 0; out[(r + 1) * H - 3:(r + 1) * H, :, 3] = 0      # keep rows apart under mip filtering
    return Image.fromarray(out, "RGBA")
