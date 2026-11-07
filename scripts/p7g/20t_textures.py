#!/usr/bin/env python3
"""dev7 G: CC0 surface textures for the network (ambientCG, 1K JPG) -> data/raw/textures/<id>/ and
$BUILD/dev7/tex/<name>.dds (512 px, mipmapped DXT1 via scripts/p2/dds.py). Idempotent.
  concrete_d : sidewalks + rolled curbs   (ambientCG Concrete034, CC0)
  ballast_d  : rail track beds            (ambientCG Gravel022, CC0, already fetched by 20_fetch_textures)
"""
import io, sys, zipfile, pathlib, requests
from PIL import Image
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[0] / "p2"))
from geom import ROOT, BUILD
import dds
IDS = {"concrete_d": "Concrete034", "ballast_d": "Gravel022"}
out = BUILD / "dev7" / "tex"; out.mkdir(parents=True, exist_ok=True)
for name, aid in IDS.items():
    d = ROOT / "data/raw/textures" / aid
    if not (d.exists() and any(d.glob("*Color.jpg"))):
        d.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"https://ambientcg.com/get?file={aid}_1K-JPG.zip", timeout=120); r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for n in z.namelist():
                if n.endswith(".jpg") and "/" not in n and ".." not in n:
                    (d / n).write_bytes(z.read(n))
    col = next(d.glob("*Color.jpg"))
    im = Image.open(col).convert("RGB").resize((512, 512), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "PNG")
    (out / f"{name}.dds").write_bytes(dds.encode(b.getvalue()))
    print(name, aid, dds.info((out / f"{name}.dds").read_bytes()))
