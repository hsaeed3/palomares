"""Litmus test, step 3: reports/litmus.html + reports/litmus/NNN.jpg — GoPro photo (left) vs track render (right).

The frame is resampled to the same pinhole camera the render uses (intrinsics config/camera.json via
scripts/p5/scene.cam_map, docs/phase2/CAMERA.md), so all panels share one projection; outside the photo = black.
Panels: photo | render | 50/50 blend (alignment check).
Usage: python3 28_litmus_sheet.py [layout=road_full|road_summit fallback]   (after 27_litmus.js)
"""
import html, importlib.util, json, pathlib, sys
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[2]
lay = sys.argv[1] if len(sys.argv) > 1 else ("road_full" if (ROOT / "build/vis/road_full/scene.json").exists() else "road_summit")
spec = importlib.util.spec_from_file_location("fit", pathlib.Path(__file__).with_name("27_litmus_fit.py"))
fit = importlib.util.module_from_spec(spec); spec.loader.exec_module(fit)
cfg = json.load(open(ROOT / "config/litmus_poses.json"))
sys.path.insert(0, str(ROOT / "scripts/p5")); import numpy as np; import scene as SC
out = ROOT / "reports/litmus"; out.mkdir(parents=True, exist_ok=True)
cards = []
for p in cfg["poses"]:
    f = p["frame"]; rp = out / f"render_{lay}_{f:03d}.png"
    if not rp.exists():
        print("missing render", rp, "- run node 27_litmus.js", lay); continue
    ren = Image.open(rp).convert("RGB"); w, h = ren.size
    a, _ = SC.undistort(np.asarray(Image.open(ROOT / f"assets/gopro/{f:03d}.png").convert("RGB")), order=1, ow=w, oh=h, vfov=p["fov"])
    pho = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    sheet = Image.new("RGB", (3 * w + 16, h + 34), (20, 20, 20)); sheet.paste(pho, (0, 34)); sheet.paste(ren, (w + 8, 34))
    sheet.paste(Image.blend(pho, ren, 0.5), (2 * w + 16, 34))
    d = ImageDraw.Draw(sheet)
    d.text((8, 10), f"frame {f:03d}  (resampled to render pinhole)", fill=(255, 255, 255))
    d.text((2 * w + 24, 10), "50/50 blend", fill=(255, 255, 255))
    d.text((w + 16, 10), f"{lay}  s = {p['s']:.0f} m   conf {p['conf']:.2f}{'  pinned' if p['pinned'] else ''}   road IoU {p['road_iou']:.2f}", fill=(255, 255, 255))
    name = f"{f:03d}.jpg"; sheet.save(out / name, quality=88)
    cards.append(f"<figure><img src='litmus/{name}' loading='lazy'><figcaption>Frame {f:03d} · s = {p['s']:.0f} m "
                 f"· lateral {p['lateral_m']:+.2f} m · eye {p['height_m']:.2f} m · "
                 f"yaw {p['yaw_rel_deg']:+.1f}° pitch {p['pitch_deg']:+.1f}° roll {p['roll_deg']:+.1f}° · "
                 f"pose conf {p['conf']:.2f}{' (pinned by eye)' if p['pinned'] else ''}, seg score {p['score']:.2f}, road IoU {p['road_iou']:.2f}</figcaption></figure>")
L = cfg["lens"]
confs = ", ".join(f"{p['frame']:03d}: {p['conf']:.2f}" for p in cfg["poses"])
doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Litmus Photo Match</title><style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--mute:#6b6b66}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#161615;--fg:#ecebe6;--mute:#9a9a94}}}}
:root[data-theme="dark"]{{--bg:#161615;--fg:#ecebe6;--mute:#9a9a94}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:16px}}
img{{width:100%;height:auto;display:block}} figure{{margin:0 0 28px}} figcaption{{color:var(--mute);font-size:13px;margin-top:6px}}
</style></head><body><h1>Litmus: GoPro vs. track ({html.escape(lay)})</h1>
<p>Left: frame resampled to the render's pinhole camera; middle: render from the registered pose (vertical FOV
{cfg['fov']}°); right: 50/50 blend. Frames are rectilinear 360-viewer screenshots; camera calibrated jointly
(docs/phase2/CAMERA.md): f = {L['f_px']:.0f} px, principal-point offset ({L['dcx']:+.0f}, {L['dcy']:+.0f}) px, k1 = {L['k1']:+.3f},
mount height {L['height_m']:.2f} m. </p>
<h2>Litmus (dev5)</h2><p>Poses: <code>data/scene/poses.json</code> (dev5 registration against Mask2Former segmentation of the
GoPro frames, 44 of 219 frames pinned by eye), converted by <code>scripts/p2/27_litmus_poses.py</code>. Frames: one per
1/{cfg['n']} of the road by station, drawn with seed {cfg['seed']} from the {cfg['n_eligible']} frames with conf ≥ {cfg['min_conf']}
not in the known-misaligned list ({', '.join(map(str, cfg['excluded']))}). Frame list is fixed for later builds.
Pose confidence: {confs}. The dev4 sheet (old road-ribbon fit) is replaced.</p>
{''.join(cards)}</body></html>"""
(ROOT / "reports/litmus.html").write_text(doc)
print("wrote reports/litmus.html with", len(cards), "pairs")
