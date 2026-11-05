"""Procedural roadside-asset atlas (fences, walls, poles, signs, buildings, cypress). All textures are generated
here (no external assets, so licence-free); MUTCD sign layouts are public-domain designs redrawn procedurally.

ONE RGBA atlas + ONE alpha-tested material ("roadside") => one draw call per 250 m chunk.
Convention (same as 21b dressing atlas): the PNG is stored flipped; UV rects are (u0, v0, u1, v1) with v measured
from the TOP of the unflipped atlas, and a quad's bottom edge uses v1 (see Acc.quad in 21c).
"""
import io, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

S = 2048
PAD = 4
_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT2 = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


def F(sz):
    for p in (_FONT2, _FONT):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


rng = np.random.default_rng(11)


def noise(h, w, sig, amp=1.0):
    n = gaussian_filter(rng.standard_normal((h, w)), sig, mode="wrap")
    return n / (n.std() + 1e-9) * amp


def rgba(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")


def solid_noise(h, w, rgb, var=10, sig=(1.0, 8.0), aniso=None):
    base = np.zeros((h, w, 4)); base[..., :3] = rgb; base[..., 3] = 255
    for s in sig:
        n = noise(h, w, s if aniso is None else (s * aniso[0], s * aniso[1]), var / len(sig))
        base[..., :3] += n[..., None]
    return base


# ----------------------------------------------------------------------------- tiles
def t_wood(w=256, h=128, rgb=(112, 98, 84), planks=0):
    a = solid_noise(h, w, rgb, 14, (1.0, 6.0), aniso=(0.3, 6))      # grain runs along u
    if planks:
        for k in range(1, planks):
            y = k * h // planks; a[max(0, y - 1):y + 1, :, :3] *= 0.55
    return rgba(a)


def t_pole():
    a = solid_noise(256, 128, (66, 52, 42), 14, (1.0, 5.0), aniso=(6, 0.3))   # grain along v (vertical)
    return rgba(a)


def t_metal():
    return rgba(solid_noise(128, 128, (158, 160, 160), 8, (1.0, 5.0)))


def t_concrete():
    a = solid_noise(256, 256, (150, 148, 142), 12, (1.0, 10.0))
    for _ in range(14):                                              # rain streaks
        x = rng.integers(0, 256); a[:, x:x + 2, :3] *= 0.9
    a[:4, :, :3] *= 0.85
    return rgba(a)


def t_stone():
    im = Image.new("RGBA", (256, 256), (70, 66, 60, 255)); d = ImageDraw.Draw(im)
    y = 0
    while y < 256:
        hh = int(rng.integers(26, 44)); x = int(rng.integers(-30, 0))
        while x < 256:
            ww = int(rng.integers(40, 80)); g = int(rng.integers(104, 150))
            col = (g, g - 4 + int(rng.integers(0, 8)), g - 14 + int(rng.integers(0, 8)), 255)
            d.rounded_rectangle((x + 2, y + 2, x + ww - 2, y + hh - 2), 6, fill=col)
            x += ww
        y += hh
    a = np.asarray(im, float); a[..., :3] += noise(256, 256, 1.2, 7)[..., None] + noise(256, 256, 9, 9)[..., None]
    a[..., 1] += np.clip(noise(256, 256, 6, 8), 0, None)                         # moss
    return rgba(a)


def t_chainlink():
    w = h = 128
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    st = 16
    for k in range(-h, w + h, st):
        d.line((k, 0, k + h, h), fill=(150, 154, 156, 255), width=2); d.line((k, h, k + h, 0), fill=(150, 154, 156, 255), width=2)
    return im


def t_rockmesh():
    w = h = 128
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    st = 22
    for k in range(-h, w + h, st):
        d.line((k, 0, k + h, h), fill=(110, 112, 112, 255), width=3); d.line((k, h, k + h, 0), fill=(110, 112, 112, 255), width=3)
    return im


def t_ranchwire():
    w, h = 256, 128
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for y in (6, 30, 56, 86, 120):
        d.line((0, y, w, y), fill=(124, 126, 128, 255), width=2)
    for x in range(8, w, 32):
        d.line((x, 6, x, 120), fill=(124, 126, 128, 255), width=1)
    return im


def t_cable():
    w, h = 256, 64
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for y in (8, 32, 56):
        d.line((0, y, w, y), fill=(150, 152, 152, 255), width=4)
    return im


def t_boards(rgb=(104, 82, 62), n=3, gap=0.22):
    w, h = 256, 128
    a = np.zeros((h, w, 4))
    bh = h / n
    base = solid_noise(h, w, rgb, 16, (1.0, 5.0), aniso=(0.3, 6))
    for k in range(n):
        y0 = int(k * bh + bh * gap / 2); y1 = int((k + 1) * bh - bh * gap / 2)
        a[y0:y1] = base[y0:y1]
    return rgba(a)


def t_splitrail(rgb=(100, 88, 74)):
    w, h = 256, 128
    a = np.zeros((h, w, 4)); base = solid_noise(h, w, rgb, 18, (1.0, 6.0), aniso=(0.3, 6))
    for y0, y1 in ((20, 46), (74, 100)):
        a[y0:y1] = base[y0:y1]
    return rgba(a)


def t_privacy():
    w = h = 256
    a = solid_noise(h, w, (92, 66, 48), 14, (1.0, 5.0), aniso=(6, 0.3))
    for x in range(0, w, 32):
        a[:, x:x + 2, :3] *= 0.45
    a[:10, :, :3] *= 0.8
    return rgba(a)


def t_gate_iron():
    w = h = 256
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = (28, 28, 30, 255)
    for x in range(10, w, 22):
        d.line((x, 12, x, h - 12), fill=c, width=5); d.polygon([(x - 6, 12), (x, 0), (x + 6, 12)], fill=c)
    for y in (20, h // 2, h - 22):
        d.line((0, y, w, y), fill=c, width=7)
    d.rectangle((0, 0, 7, h), fill=c); d.rectangle((w - 8, 0, w, h), fill=c)
    return im


def t_building(kind=0):
    w = h = 256
    cols = ((218, 206, 180), (196, 178, 150))
    a = solid_noise(h, w, cols[kind], 8, (1.0, 12.0))
    im = rgba(a); d = ImageDraw.Draw(im)
    for x0 in (36, 150):
        d.rectangle((x0, 70, x0 + 70, 170), fill=(46, 54, 62, 255)); d.rectangle((x0 - 5, 64, x0 + 75, 176), outline=(235, 232, 226, 255), width=6)
        d.line((x0 + 35, 70, x0 + 35, 170), fill=(235, 232, 226, 255), width=4)
    d.rectangle((0, h - 22, w, h), fill=(120, 112, 100, 255))
    return im


def t_roof():
    w, h = 256, 128
    a = solid_noise(h, w, (92, 62, 54), 12, (1.0, 6.0))
    for y in range(0, h, 16):
        a[y:y + 2, :, :3] *= 0.65
    return rgba(a)


def t_mailbox():
    return rgba(solid_noise(64, 64, (52, 58, 60), 6, (1.0, 4.0)))


def t_delin():
    w, h = 32, 128
    im = Image.new("RGBA", (w, h), (232, 232, 228, 255)); d = ImageDraw.Draw(im)
    d.rectangle((0, 12, w, 36), fill=(236, 170, 30, 255)); d.rectangle((0, 110, w, 128), fill=(190, 190, 186, 255))
    return im


def t_cypress():
    w, h = 256, 512
    a = np.zeros((h, w, 4)); y = np.arange(h)[:, None]; x = np.arange(w)[None, :]
    t = y / h                                                              # 0 top .. 1 bottom
    half = 20 + 78 * np.sin(np.clip(t, 0, 1) * np.pi / 2) ** 0.8 * (t > 0)  # flame silhouette
    half = np.where(t > 0.97, 0, half)
    n = noise(h, w, 3, 1.0)
    m = np.abs(x - w / 2) < half * (1 + 0.12 * n)
    base = solid_noise(h, w, (26, 44, 30), 22, (1.0, 5.0))
    a = np.where(m[..., None], base, 0)
    a[..., 3] = np.where(m, 255, 0)
    return rgba(a)


def t_apron():
    return rgba(solid_noise(128, 128, (100, 98, 94), 12, (1.0, 6.0)))


def t_hump():
    w, h = 256, 128
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for k in range(5):
        x0 = 6 + k * 50
        d.polygon([(x0, h - 4), (x0 + 44, h - 4), (x0 + 22, 6)], fill=(236, 236, 230, 235))
    return im


def t_signback():
    return rgba(solid_noise(64, 64, (140, 142, 142), 6, (1.0, 4.0)))


def t_timber():
    return t_wood(256, 128, (82, 64, 48), planks=3)


# ----------------------------------------------------------------------------- MUTCD signs (public-domain designs)
YEL, BLK, WHT, GRN, RED = (246, 204, 20, 255), (14, 14, 14, 255), (244, 244, 240, 255), (0, 104, 62, 255), (190, 24, 30, 255)


def _txt(d, xy, s, sz, col, anchor="mm"):
    d.text(xy, s, fill=col, font=F(sz), anchor=anchor)


def sign_image(text, shape=None):
    """returns (PIL image 256x256 RGBA, width_m, height_m) for a roadside text/shape description"""
    t = (text or "").strip(); tl = t.lower(); shape = (shape or "").lower()
    im = Image.new("RGBA", (256, 256), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    if "stop" in tl and "stop ahead" not in tl:                               # R1-1
        r = 120; cx = cy = 128
        pts = [(cx + r * np.cos(np.pi / 8 + k * np.pi / 4), cy + r * np.sin(np.pi / 8 + k * np.pi / 4)) for k in range(8)]
        d.polygon(pts, fill=WHT)
        pts = [(cx + (r - 8) * np.cos(np.pi / 8 + k * np.pi / 4), cy + (r - 8) * np.sin(np.pi / 8 + k * np.pi / 4)) for k in range(8)]
        d.polygon(pts, fill=RED); _txt(d, (128, 130), "STOP", 66, WHT)
        return im, 0.75, 0.75, "stop"
    m = re.search(r"(\d{2})\s*(?:mph|m\.p\.h|mi/h)", tl) or re.search(r"speed\s*limit\D{0,6}(\d{2})", tl)
    if m and "advisory" not in tl and "curve" not in tl:                      # R2-1
        d.rectangle((0, 0, 255, 255), fill=WHT); d.rounded_rectangle((6, 6, 249, 249), 16, outline=BLK, width=7)
        _txt(d, (128, 54), "SPEED", 50, BLK); _txt(d, (128, 106), "LIMIT", 50, BLK); _txt(d, (128, 190), m.group(1), 104, BLK)
        return im, 0.6, 0.75, f"mph{m.group(1)}"
    if "chevron" in tl or shape == "chevron":                                 # W1-8
        d.rectangle((0, 0, 255, 255), fill=BLK); d.rectangle((8, 8, 247, 247), fill=YEL)
        right = "right" in tl and "left" not in tl.replace("right-arrow", "right")
        right = ("right" in tl) if "left" not in tl else False
        s = 1 if right else -1
        pts = [(128 - 70 * s, 44), (128 + 40 * s, 128), (128 - 70 * s, 212)]
        d.line(pts, fill=BLK, width=34, joint="curve")
        return im, 0.6, 0.75, "chev_r" if right else "chev_l"
    if re.search(r"\bmile\b|\bmm\b|mile ?marker|milepost", tl):               # D10-1 style
        n = re.search(r"(\d+(?:\.\d+)?)", tl)
        d.rectangle((0, 0, 255, 255), fill=GRN); d.rounded_rectangle((8, 8, 247, 247), 12, outline=WHT, width=6)
        _txt(d, (128, 70), "MILE", 56, WHT); _txt(d, (128, 160), n.group(1) if n else "", 88, WHT)
        return im, 0.3, 0.45, "mile" + (n.group(1) if n else "")
    if "hump" in tl or "bump" in tl or "dip" in tl:                           # W8-1/W17-1
        return _diamond(im, d, "HUMP", None), 0.75, 0.75, "hump"
    if any(k in tl for k in ("curve", "turn", "winding", "bend", "reverse")) or shape == "diamond":
        arrow = "left" if "left" in tl else ("right" if "right" in tl else None)
        lab = "SLOW" if "slow" in tl else None
        return _diamond(im, d, lab, arrow), 0.75, 0.75, f"curve_{arrow}_{lab}"
    if "street" in tl or "blade" in tl or " rd" in tl or "road" in tl:       # street name blade
        d.rectangle((0, 70, 255, 186), fill=GRN); d.rectangle((6, 76, 249, 180), outline=WHT, width=4)
        nm = re.sub(r"\s*street name blade.*", "", t, flags=re.I)[:14].upper() or "PALOMARES RD"
        _txt(d, (128, 128), nm.split(",")[0][:13], 38, WHT)
        return im, 0.9, 0.22, "blade_" + nm[:8]
    if "object marker" in tl or "om3" in tl or "hazard object" in tl:        # OM3
        d.rectangle((0, 0, 255, 255), fill=YEL)
        for k in range(-6, 7):
            d.polygon([(k * 40, 255), (k * 40 + 20, 255), (k * 40 + 20 + 256, 0), (k * 40 + 256, 0)], fill=BLK)
        return im, 0.3, 0.9, "om3"
    if any(k in tl for k in ("fire access", "no parking", "private", "no trespass", "address", "mailbox", "number")):
        d.rectangle((0, 40, 255, 216), fill=WHT); d.rectangle((6, 46, 249, 210), outline=BLK, width=4)
        line = (t.split(",")[0])[:16].upper()
        _txt(d, (128, 128), line, 34, BLK)
        return im, 0.5, 0.35, "txt_" + line[:10]
    # generic warning/info plate
    d.rectangle((0, 0, 255, 255), fill=YEL); d.rounded_rectangle((8, 8, 247, 247), 14, outline=BLK, width=6)
    lab = (t.split(",")[0].split("(")[0])[:18].upper() or "CAUTION"
    parts = lab.split(" ", 1) if len(lab) > 9 else [lab]
    for i, p in enumerate(parts):
        _txt(d, (128, 128 + (i - (len(parts) - 1) / 2) * 62), p, 52 if len(p) < 8 else 38, BLK)
    return im, 0.6, 0.6, "gen_" + lab[:10]


def _diamond(im, d, label, arrow):
    d.polygon([(128, 0), (255, 128), (128, 255), (0, 128)], fill=BLK)
    d.polygon([(128, 14), (241, 128), (128, 241), (14, 128)], fill=YEL)
    if arrow:                                                                  # W1-2 style bent arrow, simplified
        s = -1 if arrow == "left" else 1
        d.line([(128, 214), (128, 140), (128 + 46 * s, 96)], fill=BLK, width=22, joint="curve")
        d.polygon([(128 + 66 * s, 76), (128 + 18 * s, 90), (128 + 52 * s, 126)], fill=BLK)
    if label:
        _txt(d, (128, 128), label, 44, BLK)
    return im


# ----------------------------------------------------------------------------- atlas
class Atlas:
    def __init__(self):
        self.img = Image.new("RGBA", (S, S), (0, 0, 0, 0)); self.rect = {}
        self.x = self.y = self.rowh = 0

    def put(self, name, im):
        if name in self.rect:
            return self.rect[name]
        w, h = im.size
        if self.x + w + 2 * PAD > S:
            self.x, self.y, self.rowh = 0, self.y + self.rowh, 0
        assert self.y + h + 2 * PAD <= S, "roadside atlas full"
        x0, y0 = self.x + PAD, self.y + PAD
        a = np.asarray(im.convert("RGBA"))
        a = np.pad(a, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="edge")             # edge bleed against mip filtering
        self.img.paste(Image.fromarray(a), (self.x, self.y))
        self.rect[name] = (x0 / S, y0 / S, (x0 + w) / S, (y0 + h) / S)
        self.x += w + 2 * PAD; self.rowh = max(self.rowh, h + 2 * PAD)
        return self.rect[name]

    def png(self):
        b = io.BytesIO(); self.img.transpose(Image.FLIP_TOP_BOTTOM).save(b, "PNG", optimize=True); return b.getvalue()


def base_atlas():
    A = Atlas()
    import race_tex as RT
    A.put("wbeam", RT.guardrail_wbeam().convert("RGBA"))
    for name, im in (("wood", t_wood()), ("wood_white", t_wood(256, 128, (214, 212, 204))), ("wood_dark", t_wood(128, 128, (70, 56, 44))),
                     ("timber", t_timber()), ("pole", t_pole()), ("metal", t_metal()), ("concrete", t_concrete()), ("stone", t_stone()),
                     ("chainlink", t_chainlink()), ("rockmesh", t_rockmesh()), ("ranchwire", t_ranchwire()), ("cable", t_cable()),
                     ("boards", t_boards()), ("whiteboards", t_boards((206, 204, 196), 3)), ("splitrail", t_splitrail()),
                     ("whiterail", t_boards((208, 206, 198), 3, 0.5)), ("privacy", t_privacy()), ("gate_iron", t_gate_iron()),
                     ("bldg0", t_building(0)), ("bldg1", t_building(1)), ("roof", t_roof()), ("mailbox", t_mailbox()), ("delin", t_delin()),
                     ("cypress", t_cypress()), ("apron", t_apron()), ("hump", t_hump()), ("signback", t_signback()),
                     ("black", Image.new("RGBA", (16, 16), (24, 24, 26, 255))), ("white", Image.new("RGBA", (16, 16), (214, 214, 210, 255)))):
        A.put(name, im)
    return A
