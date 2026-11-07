"""P: quick incremental build (docs/PIPELINE.md). Called by `build.py quick` / `build.py quick-seed`.

quick-seed  copy the 21*-stage outputs of build/ (geometry.pkl + atlases) into build_quick/ under build/'s lock, then
            run ONE full 22_build_kn5.py there: it writes every kn5 and the texture/material cache (~9 min, once).
quick       ledger edit -> 21e on build_quick/geometry.pkl (all rows, ~20 s, idempotent) -> 22 in quick mode: only the
            kn5 files whose input hash changed are rewritten (~15 s) -> round-trip check of the rewritten kn5.
            Reports which 200 m chunks have changed ledger rows since the last quick/seed run.

Not re-run by quick: 19-21c (road, terrain, trees, dressing, roadside), 21d (physics post-process), the AC folder
ini/ui files, QA, packaging.  Edits to those need their own step on the quick dir or a full build (release).
"""
import fcntl, hashlib, json, os, pathlib, shutil, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHUNK = 200.0
SEED_FILES = ["geometry.pkl", "ledger_atlas.png", "roadside_atlas.png", "dressing_atlas.png", "race_assets_atlas.png",
              "roadside_low_assets.json", "roadside_signs.json", "roadside_solid.json", "albedo_calibration.json", "ai_lines.json"]


def _run(argv, env, label):
    t0 = time.time()
    print(f"[quick] {label} ...", flush=True)
    rc = subprocess.call(["nice", "-n", "10", sys.executable, "-u"] + argv, cwd=HERE, env=env)
    dt = time.time() - t0
    print(f"[quick] {label}: exit {rc}, {dt:.1f}s", flush=True)
    if rc:
        sys.exit(rc)
    return dt


def ledger_chunks(path=None):
    """chunk index -> sha1 of its ledger rows (a span row s..s1 counts in every chunk it touches)"""
    path = pathlib.Path(path or os.environ.get("PALOMARES_LEDGER_FILE", ROOT / "data/scene/ledger.jsonl"))
    acc = {}
    if path.exists():
        for ln in path.read_text().splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            s0 = float(r.get("s") or 0.0); s1 = float(r.get("s1") or s0)
            for c in range(int(min(s0, s1) // CHUNK), int(max(s0, s1) // CHUNK) + 1):
                acc.setdefault(c, []).append(json.dumps(r, sort_keys=True))
    return {str(c): hashlib.sha1("\n".join(sorted(v)).encode()).hexdigest() for c, v in acc.items()}


def seed(d):
    src = ROOT / "build"
    d.mkdir(parents=True, exist_ok=True)
    fh = open(src / ".build.lock", "a")
    print(f"[quick] waiting for build/ lock to copy a consistent snapshot ...", flush=True)
    fcntl.flock(fh, fcntl.LOCK_EX)
    try:
        for f in SEED_FILES:
            if (src / f).exists():
                shutil.copy2(src / f, d / f)
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
    env = dict(os.environ, PALOMARES_BUILD_DIR=str(d), PYTHONUNBUFFERED="1")
    env.pop("PALOMARES_KN5_QUICK", None)
    _run(["quick_collision.py", "save"], env, "save 21d collision of the seed")
    dt = _run(["22_build_kn5.py"], env, "full 22_build_kn5.py (seeds kn5 cache)")
    (d / "kn5_cache" / "ledger_chunks.json").write_text(json.dumps(ledger_chunks(), indent=0))
    print(f"[quick] seeded {d} in {dt:.0f}s")
    return 0


def check(track, files):
    sys.path.insert(0, str(HERE))
    from kn5 import read_kn5
    bad = 0
    for f in files:
        k = read_kn5(track / f)
        nmax = max([len(m["pos"]) for m in k["meshes"]] or [0])
        tex_missing = sorted({t for m in k["materials"] for t in m["textures"].values()} - set(k["textures"]))
        ok = nmax <= 65535 and not tex_missing
        bad += not ok
        print(f"[quick] check {f}: {len(k['meshes'])} meshes, max {nmax} verts, {len(k['textures'])} textures"
              f"{'' if ok else '  FAIL missing ' + str(tex_missing)}")
    return bad


def main(a):
    d = pathlib.Path(a.dir).resolve()
    if d == (ROOT / "build").resolve():
        sys.exit("quick: refusing to work in build/ (other agents use it); use build_quick/")
    if a.target == "quick-seed":
        return seed(d)
    if not (d / "kn5_cache" / "state.pkl").exists():
        sys.exit(f"quick: {d} is not seeded; run `python3 scripts/p2/build.py quick-seed` once")
    t_all = time.time()
    lc_file = d / "kn5_cache" / "ledger_chunks.json"
    old = json.loads(lc_file.read_text()) if lc_file.exists() else {}
    new = ledger_chunks()
    changed = sorted({int(c) for c in set(old) | set(new) if old.get(c) != new.get(c)})
    if a.s:
        s0, s1 = (float(v) for v in a.s.split("-"))
        want = set(range(int(s0 // CHUNK), int(s1 // CHUNK) + 1))
        print(f"[quick] --s {a.s}: chunks {sorted(want)}; ledger-changed chunks there: {sorted(want & set(changed))}")
    print(f"[quick] ledger-changed chunks: {[f'{c * CHUNK:.0f}-{(c + 1) * CHUNK:.0f}' for c in changed] or 'none'}")
    env = dict(os.environ, PALOMARES_BUILD_DIR=str(d), PALOMARES_21E_DIR=str(d), PYTHONUNBUFFERED="1")
    if not a.no_21e and (changed or a.s):
        _run(["21e_ledger_assets.py"], env, "21e ledger assets")
        # 21d (terrain passes + shared 1WALL slab rebuild, ~6 min) is NOT idempotent and is not re-run. Instead the
        # collision of the seed build (21d output) is restored: 21e's raw per-prop boxes are dropped. A quick build's
        # collision is therefore that of the last full build; see docs/PIPELINE.md "What quick does not do".
        _run(["quick_collision.py", "restore"], env, "restore 21d collision of the seed")
    env["PALOMARES_KN5_QUICK"] = "1"
    track = d / "palomares_road"
    mt = lambda: {f.name: f.stat().st_mtime_ns for f in track.glob("*.kn5")}
    before = mt()
    _run(["22_build_kn5.py"], env, "22 kn5 (changed files only)")
    after = mt()
    written = sorted(f for f in after if after[f] != before.get(f))
    bad = check(d / "palomares_road", written)
    lc_file.write_text(json.dumps(new, indent=0))
    print(f"[quick] rewrote {written or 'nothing'}; total {time.time() - t_all:.1f}s -> {d / 'palomares_road'}")
    return 1 if bad else 0
