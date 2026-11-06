#!/usr/bin/env python3
"""Contact sheet photo | colourised mask for all segmented frames -> reports/dev5/seg_sheet.html"""
import glob, json, os, re
import numpy as np
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEG = os.path.join(ROOT, "data/scene/seg"); OUT = os.path.join(ROOT, "reports/dev5")
TH = os.path.join(OUT, "seg_thumbs"); os.makedirs(TH, exist_ok=True)
PAL = np.array([(0,0,0),(110,170,255),(90,90,90),(255,215,0),(255,255,255),(170,120,60),(80,200,60),
 (200,170,120),(150,80,150),(20,110,20),(140,220,140),(100,50,20),(255,0,0),(120,0,0),(255,0,255),
 (0,255,255),(255,140,0),(160,160,200),(220,40,120),(0,90,200),(60,60,140),(255,120,180),(0,0,255),
 (255,255,120),(200,0,200)], np.uint8)
cls = json.load(open(os.path.join(ROOT, "data/scene/seg_classes.json")))
rows = []
for p in sorted(glob.glob(os.path.join(SEG, "[0-9][0-9][0-9].png"))):
    st = os.path.basename(p)[:3]
    m = np.asarray(Image.open(p)); ph = Image.open(os.path.join(ROOT, "assets/gopro", st + ".png")).convert("RGB")
    tw, thh = 510, 371
    ph.resize((tw, thh)).save(os.path.join(TH, st + "_p.jpg"), quality=80)
    Image.fromarray(PAL[m]).resize((tw, thh), Image.NEAREST).save(os.path.join(TH, st + "_m.png"))
    rows.append(f'<div class=f><b>{st}</b><br><img src="seg_thumbs/{st}_p.jpg"><img src="seg_thumbs/{st}_m.png"></div>')
leg = "".join(f'<span style="background:rgb{tuple(int(x) for x in PAL[int(k)])};padding:2px 6px;margin:2px;display:inline-block;color:#888">{k} {v}</span>' for k, v in cls.items())
html = f"""<!doctype html><meta charset=utf-8><title>Seg Sheet</title><style>body{{background:#222;color:#ddd;font:12px sans-serif}}
.f{{display:inline-block;margin:4px}}img{{width:300px;margin-right:2px}}</style><h3>Segmentation contact sheet ({len(rows)} frames)</h3><p>{leg}</p>{''.join(rows)}"""
open(os.path.join(OUT, "seg_sheet.html"), "w").write(html)
print("wrote", os.path.join(OUT, "seg_sheet.html"), len(rows))
