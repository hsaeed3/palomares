"""PNG -> DDS (BC1/DXT1 opaque, BC3/DXT5 with alpha) with a full mip chain, power-of-two sizes.

Kunos and community tracks ship DDS textures with mipmaps inside the kn5.  dev1-dev5 embedded plain PNGs (no mips,
one 4096x5256 non-power-of-two backdrop): the driver then builds or skips mips at load and the GPU samples
full-resolution textures at distance (shimmer, bandwidth), and the NPOT texture is a classic driver-warning source.
Encoder: etcpak (BSD, pip `etcpak`), the same block compressors as the Kunos-era tool chain formats.
"""
import io, struct
import numpy as np
from PIL import Image

MAX_SIZE = 4096
DDSD = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000   # CAPS HEIGHT WIDTH PIXELFORMAT MIPMAPCOUNT LINEARSIZE


def _pow2(n):
    return 1 << max(2, int(round(np.log2(max(n, 4)))))


def header(w, h, mips, fourcc, top_size):
    pf = struct.pack("<II4sIIIII", 32, 0x4, fourcc, 0, 0, 0, 0, 0)
    caps = 0x1000 | 0x400000 | 0x8               # TEXTURE | MIPMAP | COMPLEX
    return b"DDS " + struct.pack("<7I", 124, DDSD, h, w, top_size, 0, mips) + b"\0" * 44 + pf + struct.pack("<5I", caps, 0, 0, 0, 0)


def encode(png_bytes):
    import etcpak
    im = Image.open(io.BytesIO(png_bytes))
    alpha = im.mode in ("RGBA", "LA", "P") and (np.asarray(im.convert("RGBA"))[..., 3] < 250).any()
    im = im.convert("RGBA")
    w, h = _pow2(im.width), _pow2(im.height)
    s = max(w, h) / MAX_SIZE
    if s > 1:
        w, h = max(4, int(w / s)), max(4, int(h / s))
    if (w, h) != im.size:
        im = im.resize((w, h), Image.LANCZOS)
    fourcc = b"DXT5" if alpha else b"DXT1"
    levels, cur = [], im
    while True:
        raw = cur.tobytes()
        levels.append(etcpak.compress_bc3(raw, cur.width, cur.height) if alpha else etcpak.compress_bc1(raw, cur.width, cur.height))
        if cur.width <= 4 or cur.height <= 4:
            break
        cur = cur.resize((cur.width // 2, cur.height // 2), Image.BOX)
    return header(w, h, len(levels), fourcc, len(levels[0])) + b"".join(levels)


def info(b):
    """(fourcc, w, h, mips) of DDS bytes, or None"""
    if b[:4] != b"DDS ":
        return None
    h, w = struct.unpack_from("<II", b, 12); mips = struct.unpack_from("<I", b, 28)[0]
    return b[84:88].decode("ascii", "replace"), w, h, mips


def to_png(b):
    out = io.BytesIO(); Image.open(io.BytesIO(b)).save(out, "PNG"); return out.getvalue()
