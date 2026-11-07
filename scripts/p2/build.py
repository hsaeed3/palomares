#!/usr/bin/env python3
"""Phase 2 build runner: runs the numbered steps with a header, step lines, timings and a summary.

  python3 scripts/p2/build.py package          # quick installable zip (make package)
  python3 scripts/p2/build.py full             # full Phase 2 incl. QA + renders (run_phase2.sh)
  python3 scripts/p2/build.py quick-seed       # copy build/ -> build_quick/ and run one full 22 there (~9 min, once)
  python3 scripts/p2/build.py quick [--s 9300-9500]   # ledger edit -> 21e + changed kn5 only, in build_quick/ (<1 min)
See docs/PIPELINE.md.
"""
from geom import BUILD
import argparse, os, subprocess, sys, time, pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

PACKAGE = [
    ("19_gopro_calibrate.py", "GoPro calibration (palette, foliage crops)"),
    ("20_fetch_textures.py", "Fetching CC0 textures"),
    ("21_build_geometry.py", "Building geometry (road, terrain, trees)"),
    ("21b_track_dressing.py", "Track dressing (gantries, huts, barriers)"),
    ("21c_roadside_assets.py", "Roadside assets (fences, guardrails, poles, signs) + robust collision"),
    ("21e_ledger_assets.py", "Scene ledger assets (data/scene/ledger.jsonl) + terrain/tree edits"),
    ("21f_far_vegetation.py", "Far-zone vegetation (CHM + NAIP scatter, 60-150 m)"),
    ("21d_physics.py", "Physics post-process (terrain smoothing, tree + wall colliders)"),
    ("21g_road_clear.py", "Road clearance (nothing may stand on the asphalt)"),
    ("22_build_kn5.py", "Writing kn5 files + AC track folder"),
    ("qa_on_road.py", "Gate: no geometry on the asphalt (fails the build)"),
    ("qa_ffb_safety.py", "FFB-safety gate (collision / normals / textures; fails the build)"),
    ("25_package.py", "Packaging zip"),
]
FULL = PACKAGE[:12] + [
    ("racing_line_plot.py", "Racing-line plots (reports/racing_line)"),
    ("qa_drive.py", "Drive simulation + safety QA (reports/safety_qa.md)"),
    ("qa_collisions.py", "Collision QA (every barrier has 1WALL)"),
    ("23_qa.py", "Independent QA"),
    ("24_visual_export.py", "Visual export (kn5 round-trip)"),
    (["node", "24_visual_render.js", "road_summit"], "Rendering road_summit"),
    (["node", "24_visual_render.js", "track_summit"], "Rendering track_summit"),
    ("26_contact_sheet.py", "Contact sheet"),
    PACKAGE[12],
]

TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def fmt(t):
    return f"{t:5.1f}s" if t < 60 else f"{int(t // 60)}m{t % 60:04.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="package", choices=["package", "full", "quick", "quick-seed"])
    ap.add_argument("--s", default=None, help="quick: station range a-b (m) to report / check; default: chunks whose ledger rows changed")
    ap.add_argument("--dir", default=str(ROOT / "build_quick"), help="quick: build dir (never build/ while others use it)")
    ap.add_argument("--no-21e", action="store_true", help="quick: kn5 only (geometry.pkl already updated)")
    ap.add_argument("--from", dest="from_step", default=None,
                    help="package/full: resume at the first step whose script name contains this (e.g. 21b, 22, qa_on_road)")
    a = ap.parse_args()
    if a.target in ("quick", "quick-seed"):
        import quick
        sys.exit(quick.main(a))
    steps = PACKAGE if a.target == "package" else FULL
    if a.from_step:
        k0 = next(i for i, (cmd, _) in enumerate(steps) if a.from_step in (cmd if isinstance(cmd, str) else " ".join(cmd)))
        steps = steps[k0:]
        names = [cmd for cmd, _ in steps if isinstance(cmd, str)]
        ck = BUILD / "geometry_21.pkl"
        if "21_build_geometry.py" not in names and any(n.startswith("21") and n != "21_build_geometry.py" for n in names):
            if not ck.exists():
                sys.exit("--from: no build/geometry_21.pkl checkpoint; run the full package once (21 is not re-entrant)")
            import shutil; shutil.copyfile(ck, BUILD / "geometry.pkl")
            print(f"  restored geometry.pkl from the step-21 checkpoint ({ck.name})", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if TTY:
        env.setdefault("PALOMARES_TTY", "1")  # child steps keep colours/progress bars when we pipe through
    ver = (ROOT / "VERSION").read_text().strip() if (ROOT / "VERSION").exists() else "?"
    bar = "=" * 64
    print(c("1;36", bar))
    print(c("1;36", f"  Palomares Road -> Assetto Corsa   v{ver}   [{a.target}]"))
    print(c("36", f"  {len(steps)} steps   {os.cpu_count()} CPU cores"))
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
        if cmd == "21_build_geometry.py":       # checkpoint: later steps modify geometry.pkl in place and are not re-entrant
            import shutil; shutil.copyfile(BUILD / "geometry.pkl", BUILD / "geometry_21.pkl")
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
