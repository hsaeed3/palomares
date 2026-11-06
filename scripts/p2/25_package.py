"""Phase 2: preview images + docs into the track folder, zip -> dist/palomares_road_vX.Y.Z.zip"""
import shutil, zipfile, sys, pathlib
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import ROOT, BUILD
from geom import build_lock; build_lock()  # one build/ writer at a time
V = (ROOT / "VERSION").read_text().strip()
TRACK = BUILD / "palomares_road"
for lay in sorted(d.name for d in (TRACK / "ui").iterdir() if d.is_dir()):
    shot = "s01750.jpg"
    src = next((c for c in (ROOT / "reports/visual" / lay / shot, ROOT / "reports/visual/road_full" / shot, ROOT / "reports/visual/road_summit" / shot) if c.exists()))
    im = Image.open(src).convert("RGB"); im.thumbnail((1024, 576)); im.save(TRACK / "ui" / lay / "preview.png")
for f in ("CHANGELOG.md", "INSTALL.md"):
    shutil.copy(ROOT / "docs/phase2" / f, TRACK / f)
(ROOT / "dist").mkdir(exist_ok=True)
# dev builds install as their own track (no Content Manager cache clashes): palomares_road_dev2, ...
import re
_m = re.search(r"-dev(\d+)$", V)
TRACK_ID = f"palomares_road_dev{_m.group(1)}" if _m else "palomares_road"
z = ROOT / "dist" / f"{TRACK_ID}_v{V}.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in sorted(TRACK.rglob("*")):
        if p.is_file():
            zf.write(p, TRACK_ID + "/" + str(p.relative_to(TRACK)))
print(z, round(z.stat().st_size / 1e6, 1), "MB")
