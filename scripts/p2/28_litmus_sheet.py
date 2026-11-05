"""Litmus test, step 3: reports/litmus.html + reports/litmus/NNN.jpg — GoPro photo (left) vs track render (right).

The GoPro frame is undistorted to the same pinhole camera the render uses (lens from config/litmus_poses.json),
so both halves share exactly the same projection; areas outside the photo's field are black (letterbox).
Usage: python3 28_litmus_sheet.py [layout=road_summit]   (after 27_litmus.js)
"""
import html, importlib.util, json, pathlib, sys
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[2]
lay = sys.argv[1] if len(sys.argv) > 1 else "road_summit"
spec = importlib.util.spec_from_file_location("fit", pathlib.Path(__file__).with_name("27_litmus_fit.py"))
fit = importlib.util.module_from_spec(spec); spec.loader.exec_module(fit)
cfg = json.load(open(ROOT / "config/litmus_poses.json"))
cam = (cfg["lens"]["f_rel"], cfg["lens"]["lam"])
out = ROOT / "reports/litmus"; out.mkdir(parents=True, exist_ok=True)
cards = []
for p in cfg["poses"]:
    f = p["frame"]; rp = out / f"render_{lay}_{f:03d}.png"
    if not rp.exists():
        print("missing render", rp, "- run node 27_litmus.js", lay); continue
    ren = Image.open(rp).convert("RGB"); w, h = ren.size
    pho = fit.undistort(Image.open(ROOT / f"assets/gopro/{f:03d}.png").convert("RGB"), cam, p["fov"], (w, h))
    sheet = Image.new("RGB", (2 * w + 8, h + 34), (20, 20, 20)); sheet.paste(pho, (0, 34)); sheet.paste(ren, (w + 8, 34))
    d = ImageDraw.Draw(sheet)
    d.text((8, 10), f"GoPro frame {f:03d}  (undistorted)", fill=(255, 255, 255))
    d.text((w + 16, 10), f"{lay}  s = {p['s']:.0f} m   IoU {p['iou']:.2f}   centre-line res {p['yellow_res_px']:.0f} px", fill=(255, 255, 255))
    name = f"{f:03d}.jpg"; sheet.save(out / name, quality=88)
    cards.append(f"<figure><img src='litmus/{name}' loading='lazy'><figcaption>Frame {f:03d} · s = {p['s']:.0f} m "
                 f"(anchor estimate {p['s_anchor']:.0f} m) · lateral {p['lateral_m']:+.2f} m · eye {p['height_m']:.2f} m · "
                 f"yaw {p['yaw_rel_deg']:+.1f}° pitch {p['pitch_deg']:+.1f}° roll {p['roll_deg']:+.1f}° · "
                 f"fit IoU {p['iou']:.3f}, centre-line residual {p['yellow_res_px']:.0f} px</figcaption></figure>")
L = cfg["lens"]
rej = ", ".join(f"{r['frame']} (IoU {r['iou']:.2f}, {r['yellow_res_px']:.0f} px)" for r in cfg["rejected"]) or "none"
doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Litmus Photo Match</title><style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--mute:#6b6b66}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#161615;--fg:#ecebe6;--mute:#9a9a94}}}}
:root[data-theme="dark"]{{--bg:#161615;--fg:#ecebe6;--mute:#9a9a94}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:16px}}
img{{width:100%;height:auto;display:block}} figure{{margin:0 0 28px}} figcaption{{color:var(--mute);font-size:13px;margin-top:6px}}
</style></head><body><h1>Litmus: GoPro vs. track ({html.escape(lay)})</h1>
<p>Left: GoPro frame undistorted to a pinhole view; right: render from the fitted pose in the same camera
(vertical FOV {cfg['fov']}°). Lens fitted once: f/h = {L['f_rel']}, fisheye factor λ = {L['lam']}
(photo field {L['vfov_photo']}° × {L['hfov_photo']}°). Poses fitted to the road ribbon + yellow centre line (seed {cfg['seed']}).
Rejected draws (poor fit): {html.escape(rej)}.</p>
{''.join(cards)}</body></html>"""
(ROOT / "reports/litmus.html").write_text(doc)
print("wrote reports/litmus.html with", len(cards), "pairs")
