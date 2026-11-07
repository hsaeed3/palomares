#!/usr/bin/env python3
"""dev7 I: one AC track folder palomares_road_dev7 = the 12 Palomares layouts + the dev7 layouts, zipped.

  python3 scripts/p7i/package_dev7.py [--palomares-src build/palomares_road] [--dev7-src build_dev7/palomares_road]
                                      [--no-zip] [--no-qa]
Steps (about 2-3 min, mostly the zip):
  1. stage dist/palomares_road_dev7/ (hard links where possible; sources are only read, build/ is never written)
       - Palomares source: every file (12 layouts, ui/, extension/, kn5) unchanged
       - dev7 source: the dev7 layouts (canyon_*, freeroam*: models_<layout>.ini, <layout>/, ui/<layout>/) and every
         kn5 they reference (G's cut copies palomares_*_dev7.kn5, palomares_niles_*.kn5)
  2. extension/ext_config.ini: Palomares config + the dev7 blocks (between "; >>> dev7" markers), sections renumbered
     never needed: the dev7 block uses CONDITION_7x / MATERIAL_ADJUSTMENT_7x which the Palomares config does not use
     (checked; a clash stops the packager)
  3. every ui/<layout>/ui_track.json: name = "Palomares Road (dev7) — <layout subname>", version
  4. every models_*.ini FILE= must exist in the folder; every layout has data/surfaces.ini, map.png, ai/fast_lane.ai, ui/
  5. qa_on_road (Palomares road on the Palomares kn5, niles_canyon_road on palomares_niles_*) and qa_ffb_safety on
     the staged folder (QA_FFB_* env, PALOMARES_BUILD_DIR pointing at a temp dir that holds the staged folder)
  6. dist/palomares_road_dev7_v<VERSION>.zip (top-level folder palomares_road_dev7/), CHANGELOG / INSTALL / attribution
Report: reports/dev7/package_dev7.json; exit 1 on any failed check.
"""
import argparse, json, os, pathlib, re, shutil, subprocess, sys, time, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
VERSION = "0.3.0-dev7"
NAME = "palomares_road_dev7"
DEV7_LAYOUT = re.compile(r"^(canyon_|freeroam)")


def link_or_copy(a, b):
    b.parent.mkdir(parents=True, exist_ok=True)
    if b.exists():
        b.unlink()
    try:
        os.link(a, b)
    except OSError:
        shutil.copy2(a, b)


def models(folder):
    out = {}
    for f in sorted(folder.glob("models_*.ini")):
        out[f.stem[len("models_"):]] = re.findall(r"^FILE=(.+)$", f.read_text(), re.M)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--palomares-src", default=str(ROOT / "build/palomares_road"))
    ap.add_argument("--dev7-src", default=str(ROOT / "build_dev7/palomares_road"))
    ap.add_argument("--no-zip", action="store_true"); ap.add_argument("--no-qa", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    PS, DS = pathlib.Path(a.palomares_src), pathlib.Path(a.dev7_src)
    STG = ROOT / "dist/_stage_dev7"; OUT = STG / NAME
    assert ROOT / "build" not in OUT.parents
    if STG.exists():
        shutil.rmtree(STG)
    OUT.mkdir(parents=True)
    rep = dict(version=VERSION, palomares_src=str(PS), dev7_src=str(DS), fails=[])
    fail = lambda m: (rep["fails"].append(m), print("FAIL", m))
    # 1. Palomares: everything
    pal = models(PS)
    if len(pal) != 12:
        fail(f"Palomares source has {len(pal)} layouts (expected 12)")
    for f in PS.rglob("*"):
        if f.is_file():
            link_or_copy(f, OUT / f.relative_to(PS))
    # dev7 layouts + referenced kn5
    d7 = {k: v for k, v in models(DS).items() if DEV7_LAYOUT.match(k)}
    clash = sorted(set(d7) & set(pal))
    if clash:
        fail(f"layout name clash {clash}")
    for lay, files in d7.items():
        link_or_copy(DS / f"models_{lay}.ini", OUT / f"models_{lay}.ini")
        for sub in (DS / lay, DS / "ui" / lay):
            for f in sub.rglob("*") if sub.exists() else []:
                if f.is_file():
                    link_or_copy(f, OUT / f.relative_to(DS))
        for k in files:
            if (DS / k).exists():
                if (OUT / k).exists() and k in sum(pal.values(), []):
                    fail(f"{k}: dev7 kn5 would replace a Palomares kn5 of the same name")
                    continue
                link_or_copy(DS / k, OUT / k)
    rep["layouts"] = dict(palomares=sorted(pal), dev7=sorted(d7))
    # 2. ext_config merge
    e = OUT / "extension/ext_config.ini"; base = e.read_text() if e.exists() else ""
    if e.exists():
        e.unlink()                                    # hard link: write a new file, never through to the source
    d7e = (DS / "extension/ext_config.ini").read_text() if (DS / "extension/ext_config.ini").exists() else ""
    blocks = re.findall(r"; >>> dev7.*?; <<< dev7[^\n]*\n?", d7e, re.S)
    secs_base = set(re.findall(r"^\[([^\]]+)\]", base, re.M))
    for b in blocks:
        cl = secs_base & set(re.findall(r"^\[([^\]]+)\]", b, re.M))
        if cl:
            fail(f"ext_config section clash {sorted(cl)}")
    base = re.sub(r"; >>> dev7.*?; <<< dev7[^\n]*\n?", "", base, flags=re.S)
    e.parent.mkdir(parents=True, exist_ok=True); e.write_text(base.rstrip() + "\n\n" + "".join(blocks))
    rep["ext_config_blocks"] = len(blocks)
    # 3. ui names
    for uj in sorted((OUT / "ui").glob("*/ui_track.json")):
        d = json.loads(uj.read_text()); nm = d.get("name", uj.parent.name)
        sub = re.sub(r"\s+v\d[\w.\-]*$", "", nm); sub = re.sub(r"^Palomares Road\s*[—-]\s*", "", sub)
        d["name"] = f"Palomares Road (dev7) — {sub}"; d["version"] = VERSION
        uj.unlink(); uj.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    # 4. structure
    allm = models(OUT)
    for lay, files in allm.items():
        for k in files:
            if not (OUT / k).exists():
                fail(f"{lay}: missing {k}")
        for need in (f"{lay}/data/surfaces.ini", f"{lay}/map.png", f"{lay}/ai/fast_lane.ai", f"ui/{lay}/ui_track.json", f"ui/{lay}/outline.png"):
            if not (OUT / need).exists():
                fail(f"{lay}: missing {need}")
    rep["layout_count"] = len(allm)
    # 5. QA on the staged folder
    if not a.no_qa:
        p2 = ROOT / "scripts/p2"; qa = {}
        def view(tag, lays, pat=None):
            """temp build dir whose palomares_road holds only these layouts' kn5 + data (symlinks): the QA globs
            then see exactly what those layouts load (the shipped Palomares road and G's cut copy never meet)"""
            QB = STG / f"qa_{tag}"; T = QB / "palomares_road"; T.mkdir(parents=True)
            for lay in lays:
                for k in allm[lay]:
                    if not (T / k).exists():
                        os.symlink(OUT / k, T / k)
                (T / lay).mkdir(); os.symlink(OUT / lay / "data", T / lay / "data")
                if pat is None:
                    os.symlink(OUT / f"models_{lay}.ini", T / f"models_{lay}.ini")
                else:                                # split QA run: the layout lists only the kn5 of this run
                    import fnmatch
                    fs = [k for k in allm[lay] if fnmatch.fnmatch(k, pat)]
                    (T / f"models_{lay}.ini").write_text("".join(f"[MODEL_{i}]\nFILE={k}\n\n" for i, k in enumerate(fs)))
                    for k in allm[lay]:
                        if not fnmatch.fnmatch(k, pat) and (T / k).is_symlink():
                            (T / k).unlink()
            return QB
        def run(tag, QB, cmd, extra=None):
            env = dict(os.environ, PALOMARES_BUILD_DIR=str(QB), **(extra or {}))
            r = subprocess.run(["nice", sys.executable] + cmd, cwd=p2, env=env, capture_output=True, text=True, timeout=170)
            last = (r.stdout.strip().splitlines() or [""])[-1]; qa[tag] = dict(rc=r.returncode, last=last); print(f"  {tag}: rc {r.returncode} {last}")
            if r.returncode:
                fail(f"{tag}: {last} {r.stderr[-300:]}")
        QP = view("palomares", sorted(pal))
        run("on_road_palomares", QP, ["qa_on_road.py", "--track", str(QP / "palomares_road")])
        run("ffb_palomares", QP, ["qa_ffb_safety.py"], dict(QA_FFB_OUT="dev7/qa_ffb_package_palomares.json"))
        cl = [k for k in d7 if k.startswith("canyon_")]
        if cl:
            QC = view("canyon", cl)
            run("on_road_niles_canyon", QC, ["qa_on_road.py", "--track", str(QC / "palomares_road"), "--road", "niles_canyon_road", "--kn5", "palomares_niles_*.kn5"])
            run("ffb_canyon", QC, ["qa_ffb_safety.py"], dict(QA_FFB_ROAD_KN5="palomares_niles_ncr.kn5", QA_FFB_ROAD="niles_canyon_road", QA_FFB_OUT="dev7/qa_ffb_package_canyon.json"))
        fl = [k for k in d7 if k.startswith("freeroam")]
        if fl:
            QF = view("freeroam", fl)
            run("on_road_freeroam_palomares", QF, ["qa_on_road.py", "--track", str(QF / "palomares_road"), "--kn5", "palomares_[!n]*.kn5"])
            for rn in ("niles_canyon_road", "niles_blvd", "main_street", "mission_boulevard"):
                run(f"on_road_freeroam_{rn}", QF, ["qa_on_road.py", "--track", str(QF / "palomares_road"), "--road", rn, "--kn5", "palomares_niles_*.kn5"])
            # qa_ffb_safety classifies colliders in ONE road frame: run it per frame on the kn5 that live in it
            # (Palomares frame: shipped Palomares kn5 + G's junction-cut copies; NCR frame: every palomares_niles_*)
            QFp = view("freeroam_pal", fl, "palomares_[!n]*.kn5"); QFn = view("freeroam_nil", fl, "palomares_niles_*.kn5")
            run("ffb_freeroam_palomares", QFp, ["qa_ffb_safety.py"], dict(QA_FFB_KN5="palomares_[!n]*.kn5", QA_FFB_ROAD_KN5="palomares_road_dev7.kn5",
                                                                       QA_FFB_OUT="dev7/qa_ffb_package_freeroam_palomares.json"))
            run("ffb_freeroam_niles", QFn, ["qa_ffb_safety.py"], dict(QA_FFB_KN5="palomares_niles_*.kn5", QA_FFB_ROAD_KN5="palomares_niles_ncr.kn5", QA_FFB_ROAD="niles_canyon_road",
                                                                   QA_FFB_OUT="dev7/qa_ffb_package_freeroam_niles.json"))
        rep["qa"] = qa
        for q in STG.glob("qa_*"):
            shutil.rmtree(q)
    # 6. docs + zip
    (OUT / "INSTALL.md").unlink(missing_ok=True)
    (OUT / "INSTALL.md").write_text(f"# {NAME} v{VERSION}\n\nCopy the folder `{NAME}` into `assettocorsa/content/tracks/`. Layouts: {len(allm)} "
                                   f"(12 Palomares Road, plus {', '.join(sorted(d7))}). Content Manager and CSP recommended (blinking flashers, GrassFX); "
                                   "the track works without CSP.\n")
    ch = ROOT / "docs/phase2/CHANGELOG.md"
    if ch.exists():
        (OUT / "CHANGELOG.md").unlink(missing_ok=True); shutil.copy2(ch, OUT / "CHANGELOG.md")
    att = ROOT / "reports/dev7/mapillary_attribution.csv"
    if att.exists():
        shutil.copy2(att, OUT / "MAPILLARY_ATTRIBUTION.csv")
    if not a.no_zip and not rep["fails"]:
        z = ROOT / f"dist/{NAME}_v{VERSION}.zip"; tmp = z.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for f in sorted(OUT.rglob("*")):
                if f.is_file():
                    zf.write(f, f"{NAME}/{f.relative_to(OUT)}")
        tmp.replace(z); rep["zip"] = str(z.relative_to(ROOT)); rep["zip_mb"] = round(z.stat().st_size / 1e6, 1)
    rep["secs"] = round(time.time() - t0, 1)
    (ROOT / "reports/dev7/package_dev7.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("qa",)}, indent=1))
    sys.exit(1 if rep["fails"] else 0)


if __name__ == "__main__":
    main()
