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


def _canvas(w, h):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); return im, ImageDraw.Draw(im)


def _arrow(d, cx, cy, s, size, w=24):
    """big straight arrow pointing left (s=-1) / right (s=+1)"""
    d.line([(cx - s * size, cy), (cx + s * size * 0.55, cy)], fill=BLK, width=w)
    d.polygon([(cx + s * size, cy), (cx + s * size * 0.35, cy - size * 0.55), (cx + s * size * 0.35, cy + size * 0.55)], fill=BLK)


def sign_image(text, shape=None, dirn_override=None):
    """-> (RGBA image or None, width_m, height_m, key).  Classifies free-text survey descriptions into MUTCD designs
    (public domain): R1-1 STOP, R2-1 SPEED LIMIT, W1-x curve / W1-8 chevron / W1-6 arrow, W8-1 HUMP, W5-2 narrow
    bridge, D10 mile marker, OM3 object marker, street-name blade; everything unreadable gets a grey sign back."""
    tl = ((text or "") + " " + (shape or "")).lower().strip()
    W = lambda p: re.search(p, tl)
    left = W(r"\bleft\b") is not None; right = W(r"\bright\b") is not None
    dirn = "left" if left and not right else ("right" if right and not left else ("right" if right else None))
    if dirn_override:                                   # road-geometry direction (21c) beats the survey wording
        dirn = dirn_override
    if W(r"\bcone\b"):
        return None, 0, 0, "cone"
    if W(r"back of|unreadable|\(reverse\)|\(back"):
        dia = "diamond" in tl
        im, d = _canvas(256, 256)
        if dia:
            d.polygon([(128, 0), (255, 128), (128, 255), (0, 128)], fill=(138, 140, 140, 255))
        else:
            d.rounded_rectangle((0, 20, 255, 236), 10, fill=(138, 140, 140, 255))
        return im, 0.7 if dia else 0.6, 0.7 if dia else 0.5, "back_d" if dia else "back_r"
    if W(r"\bstop\b") and not W(r"stop ahead|bus stop"):
        im, d = _canvas(256, 256); r = 120; cx = cy = 128
        d.polygon([(cx + r * np.cos(np.pi / 8 + k * np.pi / 4), cy + r * np.sin(np.pi / 8 + k * np.pi / 4)) for k in range(8)], fill=WHT)
        d.polygon([(cx + (r - 8) * np.cos(np.pi / 8 + k * np.pi / 4), cy + (r - 8) * np.sin(np.pi / 8 + k * np.pi / 4)) for k in range(8)], fill=RED)
        _txt(d, (128, 130), "STOP", 66, WHT)
        return im, 0.75, 0.75, "stop"
    if W(r"\b(hump|bump)\b"):
        im, d = _canvas(256, 256)
        return _diamond(im, d, "HUMP", None), 0.75, 0.75, "hump"
    m = W(r"(\d{2})\s*(?:mph|m\.p\.h|mi/h)") or W(r"speed\s*limit\D{0,6}(\d{2})")
    if m and not W(r"curve|diamond|advisory"):                                    # R2-1
        im, d = _canvas(256, 320)
        d.rounded_rectangle((0, 0, 255, 319), 18, fill=WHT); d.rounded_rectangle((8, 8, 247, 311), 14, outline=BLK, width=7)
        _txt(d, (128, 68), "SPEED", 54, BLK); _txt(d, (128, 126), "LIMIT", 54, BLK); _txt(d, (128, 232), m.group(1), 112, BLK)
        return im, 0.6, 0.75, f"mph{m.group(1)}"
    if W(r"mile ?marker|milepost|\bmm\b|\bmile\b"):
        n = re.search(r"(\d+(?:\.\d+)?)", tl)
        im, d = _canvas(160, 240)
        d.rounded_rectangle((0, 0, 159, 239), 8, fill=GRN); d.rounded_rectangle((6, 6, 153, 233), 6, outline=WHT, width=4)
        _txt(d, (80, 60), "MILE", 40, WHT); _txt(d, (80, 150), (n.group(1) if n else ""), 52, WHT)
        return im, 0.3, 0.45, "mile" + (n.group(1) if n else "")
    if W(r"object marker|\bom3\b|hazard object"):
        im, d = _canvas(128, 256); d.rectangle((0, 0, 127, 255), fill=YEL)
        for k in range(-6, 7):
            d.polygon([(k * 40, 255), (k * 40 + 20, 255), (k * 40 + 20 + 256, 0), (k * 40 + 256, 0)], fill=BLK)
        return im, 0.3, 0.6, "om3"
    if W(r"street|blade|< ?\w.* rd ?>"):
        nm = re.search(r"<\s*(.*?)\s*>", text or "")
        nm = (nm.group(1) if nm else re.split(r"\s+street|\s+name|,", text or "", flags=re.I)[0]).strip().upper()[:14] or "PALOMARES RD"
        im, d = _canvas(256, 80); d.rectangle((0, 0, 255, 79), fill=GRN); d.rectangle((4, 4, 251, 75), outline=WHT, width=3)
        _txt(d, (128, 42), nm, int(min(36, 236 / (0.7 * max(len(nm), 1)))), WHT)
        return im, 0.9, 0.28, "blade_" + nm
    if W(r"chevron"):                                                            # W1-8
        im, d = _canvas(192, 240); d.rounded_rectangle((0, 0, 191, 239), 8, fill=BLK); d.rounded_rectangle((6, 6, 185, 233), 6, fill=YEL)
        s = 1 if dirn == "right" else -1
        d.line([(96 - 52 * s, 44), (96 + 34 * s, 120), (96 - 52 * s, 196)], fill=BLK, width=30, joint="curve")
        return im, 0.6, 0.75, "chev_" + ("r" if s > 0 else "l")
    if W(r"narrow bridge"):
        im, d = _canvas(256, 256)
        return _diamond(im, d, "NARROW\nBRIDGE", None, 26), 0.75, 0.75, "narrowbridge"
    if W(r"\barrow\b") and not W(r"bent|curve|turn|diamond"):                  # W1-6/W1-7 arrow plate
        im, d = _canvas(256, 192); d.rounded_rectangle((0, 0, 255, 191), 8, fill=BLK); d.rounded_rectangle((6, 6, 249, 185), 6, fill=YEL)
        _arrow(d, 128, 96, 1 if dirn == "right" else -1, 84, 30)
        return im, 0.75, 0.56, "arrow_" + str(dirn)
    if W(r"round|circle|mirror"):
        im, d = _canvas(256, 256)
        if "mirror" in tl:
            d.ellipse((0, 0, 255, 255), fill=(236, 120, 24, 255)); d.ellipse((18, 18, 237, 237), fill=(150, 168, 180, 255))
            return im, 0.6, 0.6, "mirror"
        d.ellipse((0, 0, 255, 255), fill=WHT); d.ellipse((10, 10, 245, 245), outline=RED, width=24)
        d.line((52, 52, 204, 204), fill=RED, width=22)
        return im, 0.6, 0.6, "round_noslash"
    if W(r"diamond|curve|\bturn\b|winding|\bbend\b|warning"):
        im, d = _canvas(256, 256)
        lab = "SLOW" if "slow" in tl else None
        arrow = dirn if W(r"curve|turn|bend|arrow") else None
        return _diamond(im, d, lab, arrow), 0.75, 0.75, f"dia_{arrow}_{lab}"
    if W(r"no dumping|fire access|no parking|private|trespass"):
        im, d = _canvas(256, 192); d.rounded_rectangle((0, 0, 255, 191), 8, fill=WHT); d.rounded_rectangle((6, 6, 249, 185), 6, outline=BLK, width=4)
        lines = ["NO", "DUMPING"] if "dumping" in tl else (["FIRE", "ACCESS"] if "fire" in tl else ["NO", "PARKING"] if "parking" in tl else ["PRIVATE", "ROAD"])
        for i, ln_ in enumerate(lines):
            _txt(d, (128, 66 + i * 60), ln_, 46 if len(ln_) < 8 else 40, BLK)
        return im, 0.6, 0.45, "txt_" + lines[-1]
    im, d = _canvas(256, 192); d.rounded_rectangle((0, 0, 255, 191), 8, fill=WHT); d.rounded_rectangle((6, 6, 249, 185), 6, outline=BLK, width=4)
    return im, 0.5, 0.38, "blank_white"


def _diamond(im, d, label, arrow, fs=44):
    d.polygon([(128, 0), (255, 128), (128, 255), (0, 128)], fill=BLK)
    d.polygon([(128, 14), (241, 128), (128, 241), (14, 128)], fill=YEL)
    if arrow:                                                                  # W1-2 style bent arrow, simplified
        s = -1 if arrow == "left" else 1
        d.line([(128, 214), (128, 140), (128 + 46 * s, 96)], fill=BLK, width=22, joint="curve")
        d.polygon([(128 + 66 * s, 76), (128 + 18 * s, 90), (128 + 52 * s, 126)], fill=BLK)
    if label:
        for i, ln_ in enumerate(label.split("\n")):
            _txt(d, (128, 128 + (i - (label.count("\n")) / 2) * (fs + 6)), ln_, fs, BLK)
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
