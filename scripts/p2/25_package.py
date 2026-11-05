"""Phase 2: preview images + docs into the track folder, zip -> dist/palomares_road_vX.Y.Z.zip"""
import shutil, zipfile, sys, pathlib
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import ROOT, BUILD
V = (ROOT / "VERSION").read_text().strip()
TRACK = BUILD / "palomares_road"
for lay, shot in (("road_summit", "s01750.jpg"), ("track_summit", "s01750.jpg")):
    src = ROOT / "reports/visual" / lay / shot
    if not src.exists():
        src = ROOT / "reports/visual/road_summit" / shot
    im = Image.open(src).convert("RGB"); im.thumbnail((1024, 576)); im.save(TRACK / "ui" / lay / "preview.png")
for f in ("CHANGELOG.md", "INSTALL.md"):
    shutil.copy(ROOT / "docs/phase2" / f, TRACK / f)
(ROOT / "dist").mkdir(exist_ok=True)
z = ROOT / "dist" / f"palomares_road_v{V}.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in sorted(TRACK.rglob("*")):
        if p.is_file():
            zf.write(p, "palomares_road/" + str(p.relative_to(TRACK)))
print(z, round(z.stat().st_size / 1e6, 1), "MB")
