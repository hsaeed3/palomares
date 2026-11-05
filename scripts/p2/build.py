#!/usr/bin/env python3
"""Phase 2 build runner: runs the numbered steps with a header, step lines, timings and a summary.

  python3 scripts/p2/build.py package          # quick installable zip (make package)
  python3 scripts/p2/build.py package --gpu    # same, torch/CUDA for raster sampling if available (make package-gpu)
  python3 scripts/p2/build.py full             # full Phase 2 incl. QA + renders (run_phase2.sh)
"""
import argparse, os, subprocess, sys, time, pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

PACKAGE = [
    ("19_gopro_calibrate.py", "GoPro calibration (palette, foliage crops)"),
    ("20_fetch_textures.py", "Fetching CC0 textures"),
    ("21_build_geometry.py", "Building geometry (road, terrain, trees)"),
    ("21b_track_dressing.py", "Track dressing (gantries, huts, barriers)"),
    ("21c_roadside_assets.py", "Roadside assets (fences, guardrails, poles, signs) + robust collision"),
    ("22_build_kn5.py", "Writing kn5 files + AC track folder"),
    ("25_package.py", "Packaging zip"),
]
FULL = PACKAGE[:6] + [
    ("qa_collisions.py", "Collision QA (every barrier has 1WALL)"),
    ("23_qa.py", "Independent QA"),
    ("24_visual_export.py", "Visual export (kn5 round-trip)"),
    (["node", "24_visual_render.js", "road_summit"], "Rendering road_summit"),
    (["node", "24_visual_render.js", "track_summit"], "Rendering track_summit"),
    ("26_contact_sheet.py", "Contact sheet"),
    PACKAGE[6],
]

TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def fmt(t):
    return f"{t:5.1f}s" if t < 60 else f"{int(t // 60)}m{t % 60:04.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="package", choices=["package", "full"])
    ap.add_argument("--gpu", action="store_true", help="use torch/CUDA where available (falls back to CPU)")
    a = ap.parse_args()
    steps = PACKAGE if a.target == "package" else FULL
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if a.gpu:
        env["PALOMARES_GPU"] = "1"
    if TTY:
        env.setdefault("PALOMARES_TTY", "1")  # child steps keep colours/progress bars when we pipe through
    ver = (ROOT / "VERSION").read_text().strip() if (ROOT / "VERSION").exists() else "?"
    accel = "CPU"
    if a.gpu:
        try:
            import torch
            accel = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU (torch: no CUDA device)"
        except ImportError:
            accel = "CPU (torch not installed)"
    bar = "=" * 64
    print(c("1;36", bar))
    print(c("1;36", f"  Palomares Road -> Assetto Corsa   v{ver}   [{a.target}]"))
    print(c("36", f"  {len(steps)} steps   acceleration: {accel}"))
    print(c("1;36", bar), flush=True)
    times = []
    t_all = time.time()
    for k, (cmd, label) in enumerate(steps, 1):
        argv = [sys.executable, cmd] if isinstance(cmd, str) else cmd
        print(c("1", f"\n[{k}/{len(steps)}] {label}..."), c("2", f"({' '.join(argv[1:]) if isinstance(cmd, str) else ' '.join(cmd)})"), flush=True)
        t0 = time.time()
        rc = subprocess.call(argv, cwd=HERE, env=env)
        dt = time.time() - t0
        times.append((label, dt, rc))
        if rc != 0:
            print(c("1;31", f"\n[{k}/{len(steps)}] FAILED: {label} ({cmd if isinstance(cmd, str) else ' '.join(cmd)}) exit {rc} after {fmt(dt)}"), file=sys.stderr)
            sys.exit(rc)
        print(c("32", f"  done {fmt(dt)}"), flush=True)
    total = time.time() - t_all
    print("\n" + c("1;36", bar))
    print(c("1", "  Summary"))
    for label, dt, _ in times:
        print(f"  {c('32', 'ok')}  {label:44s} {fmt(dt):>8s}")
    print(f"      {'total':44s} {fmt(total):>8s}")
    z = ROOT / "dist" / f"palomares_road_v{ver}.zip"
    if z.exists():
        print(c("1;32", f"\n  -> {z.relative_to(ROOT)}  ({z.stat().st_size / 1e6:.1f} MB)"))
    print(c("1;36", bar))


if __name__ == "__main__":
    main()
