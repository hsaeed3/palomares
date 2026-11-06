"""dev5 fast scorer: class-ID render of cached geometry + ledger from each frame's pose vs. the frame's segmentation.

  python3 scripts/p5/score.py --frames 20-30            # per-frame scores -> data/scene/scores/NNN.json + summary.csv
  python3 scripts/p5/score.py --frames 22 --overlay     # + reports/dev5/overlay/022.jpg (photo | seg | render | diff)
  python3 scripts/p5/score.py --all                     # all frames, 3 worker processes
  --rebuild-cache   re-read build/geometry.pkl (automatic when its mtime changes)

Pose: data/scene/poses.json (register.py), else the old data/derived/gopro_stations_*.csv guesses.
Seg: data/scene/seg/NNN.png, else a cheap HSV stand-in (marked "seg":"hsv" in the json; use it only for development).
Everything is compared in the undistorted pinhole view (512x372, vfov from config/litmus_poses.json); pixels outside
the fisheye's field and seg classes unknown/vehicle/person are ignored.
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as SC

ROOT = SC.ROOT
OUT = SC.SCENE / "scores"
OVL = ROOT / "reports/dev5/overlay"
PAL = np.zeros((256, 3), np.uint8)
PAL[:25] = [(0, 0, 0), (135, 190, 235), (90, 90, 90), (240, 200, 0), (250, 250, 250), (160, 120, 70), (120, 170, 60),
            (170, 150, 110), (120, 100, 90), (20, 90, 30), (80, 140, 40), (90, 50, 20), (230, 40, 40), (150, 30, 30),
            (200, 200, 60), (190, 190, 210), (200, 120, 200), (130, 80, 160), (230, 120, 30), (40, 200, 200),
            (60, 60, 60), (60, 60, 60), (40, 80, 230), (255, 0, 160), (255, 120, 120)]
# merged groups used for headline IoU (seg models rarely separate grass / bare / rock etc. reliably)
GROUPS = {"sky": [1], "road": [2], "marking_yellow": [3], "marking_white": [4], "shoulder": [5],
          "terrain": [6, 7, 8, 22], "tree": [9], "shrub": [10], "trunk": [11], "vegetation": [9, 10, 11],
          "sign": [12, 13], "pole": [14], "barrier": [15, 16, 17], "building": [18], "bridge": [19], "prop": [23, 24]}


def parse_frames(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


# ---------------------------------------------------------------- poses
def load_poses():
    P = {}
    for f in sorted(glob.glob(str(ROOT / "data/derived/gopro_stations_*.csv"))):
        for r in csv.DictReader(open(f)):
            P[int(r["frame"])] = dict(frame=int(r["frame"]), s=float(r["s_m"]), lateral=float(r["lateral_m"]),
                                      height=float(r["height_m"]), yaw=float(r["yaw_rel_deg"]), pitch=float(r["pitch_deg"]),
                                      roll=float(r["roll_deg"]), src="csv")
    pj = SC.SCENE / "poses.json"
    if pj.exists():
        for r in json.load(open(pj))["poses"]:
            P[r["frame"]] = dict(r, src="poses.json")
    return P


def pvec(r):
    return np.array([r["s"], r["lateral"], r["height"], np.radians(r["yaw"]), np.radians(r["pitch"]), np.radians(r["roll"])])


# ---------------------------------------------------------------- seg
def hsv_standin(f):
    """cheap colour-rule segmentation at 1021x743 (development only)"""
    a = np.asarray(Image.open(ROOT / f"assets/gopro/{f:03d}.png").convert("RGB").resize((1021, 743)), np.float32) / 255
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1); sat = (mx - mn) / (mx + 1e-6); v = mx
    hue = np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360
    H, W = v.shape; yy = np.mgrid[0:H, 0:W][0] / H
    seg = np.full((H, W), 7, np.uint8)                                        # terrain_bare default
    seg[(g > r) & (g > b * 1.05) & (v > 0.15)] = 6                            # greenish -> grass
    seg[(v < 0.28) & (yy < 0.75)] = 9                                         # dark -> tree canopy
    seg[(v < 0.35) & (g >= r) & (g > b) & (yy < 0.75)] = 9
    sky = (b >= r - 0.02) & (v > 0.6) & (sat < 0.45) & (yy < 0.6) | ((v > 0.85) & (sat < 0.12) & (yy < 0.5))
    seg[sky] = 1
    grey = (sat < 0.14) & (v > 0.12) & (v < 0.8) & (yy > 0.35)
    m = ndimage.binary_opening(grey, iterations=2)
    lab, n = ndimage.label(m)
    seeds = lab[H - 12:H - 2, int(W * .3):int(W * .7)].ravel(); seeds = seeds[seeds > 0]
    if len(seeds):
        road = ndimage.binary_closing(lab == np.bincount(seeds).argmax(), iterations=3)
        road = ndimage.binary_fill_holes(np.vstack([road, np.ones((1, W), bool)]))[:-1]
        seg[road] = 2
    else:
        road = np.zeros_like(m)
    rd = ndimage.binary_dilation(road, iterations=4)
    seg[rd & (hue > 30) & (hue < 65) & (sat > 0.3) & (v > 0.45)] = 3
    seg[rd & (sat < 0.15) & (v > 0.8) & (yy > 0.4)] = 4
    seg[int(H * 0.9):, int(W * 0.88):] = 0                                    # GoPro overlay corner
    return seg


def load_seg(f):
    p = SC.SCENE / f"seg/{f:03d}.png"
    if p.exists():
        return np.asarray(Image.open(p)), "seg"
    return hsv_standin(f), "hsv"


# ---------------------------------------------------------------- scoring
def score_frame(f, pose, ledger, catalog, overlay=False, seg_full=None):
    t0 = time.time()
    p = pvec(pose)
    ren, inst = SC.render(p, ledger=ledger, catalog=catalog)
    t_r = time.time() - t0
    if seg_full is None:
        seg_full, src = load_seg(f)
    else:
        src = "given"
    seg, valid = SC.undistort(seg_full, order=0)
    valid &= ~np.isin(seg, list(SC.IGNORE))
    res = dict(frame=f, s=round(float(p[0]), 1), pose_src=pose.get("src"), seg=src, classes={}, groups={}, instances=[])
    for name, ids in [(SC.CLASSES[i], [i]) for i in range(1, 25) if i not in SC.IGNORE] + [("G:" + k, v) for k, v in GROUPS.items()]:
        a = np.isin(ren, ids) & valid; b = np.isin(seg, ids) & valid
        na, nb, ni = int(a.sum()), int(b.sum()), int((a & b).sum())
        if na + nb == 0: continue
        d = dict(iou=round(ni / (na + nb - ni), 4), precision=round(ni / na, 4) if na else None,
                 recall=round(ni / nb, 4) if nb else None, px_render=na, px_seg=nb)
        (res["groups"] if name.startswith("G:") else res["classes"])[name.removeprefix("G:")] = d
    # per-ledger-instance: projected footprint vs seg pixels of a compatible class (group of its render class)
    for j in np.unique(inst[inst >= 0]):
        m = (inst == j) & valid; n = int(m.sum())
        o = ledger[j]; rc = int(np.bincount(ren[inst == j]).argmax()) if (inst == j).any() else 0
        ids = next((v for v in GROUPS.values() if rc in v and len(v) > 1), [rc])
        hit = float(np.isin(seg[m], ids).mean()) if n else 0.0
        res["instances"].append(dict(id=o.get("id"), cls=o.get("cls"), px=n, match=round(hit, 3),
                                     found=bool(n >= 6 and hit >= 0.4), occluded=n < 6))
    g = res["groups"]
    res["score"] = round(float(np.mean([g[k]["iou"] for k in ("sky", "road", "vegetation", "terrain") if k in g])), 4)
    res["t_render_s"] = round(t_r, 3); res["t_total_s"] = round(time.time() - t0, 3)
    if overlay:
        write_overlay(f, ren, seg, valid, res)
    return res


def write_overlay(f, ren, seg, valid, res):
    OVL.mkdir(parents=True, exist_ok=True)
    pho, _ = SC.undistort(np.asarray(Image.open(ROOT / f"assets/gopro/{f:03d}.png").convert("RGB")), order=1)
    gid = np.zeros(256, np.uint8)
    for k, (n, ids) in enumerate(GROUPS.items()):
        if n == "vegetation": continue
        gid[ids] = k + 1
    gid[[9, 10, 11]] = 50
    dis = np.zeros(ren.shape + (3,), np.uint8)
    agree = gid[ren] == gid[seg]
    dis[agree & valid] = (40, 120, 40); dis[~agree & valid] = (220, 40, 40)
    dis[(seg == 2) & (ren != 2) & valid] = (255, 220, 0)       # road in photo, not in render
    dis[(ren == 2) & (seg != 2) & valid] = (255, 0, 255)       # road in render, not in photo
    segc = PAL[seg].copy(); segc[~valid] //= 3
    w, h = SC.AW, SC.AH
    tiles = [pho.astype(np.uint8), segc, PAL[ren], dis]
    # render class edges drawn over the photo so the alignment can be judged by eye
    e = (ndimage.morphological_gradient(ren, size=3) > 0)
    tiles.append(np.where(e[..., None], np.array([255, 0, 255], np.uint8), pho.astype(np.uint8)))
    sheet = Image.new("RGB", (w * 3 + 8, h * 2 + 44), (20, 20, 20))
    pos = [(0, 0), (w + 4, 0), (2 * w + 8, 0), (w + 4, h + 4), (0, h + 4)]
    for t, (x, y) in zip(tiles, pos):
        sheet.paste(Image.fromarray(t), (x, y + 22))
    d = ImageDraw.Draw(sheet); g = res["groups"]
    d.text((6, 4), f"frame {f:03d}  s={res['s']:.0f}  pose={res['pose_src']}  seg={res['seg']}   photo | seg | render", fill=(255, 255, 255))
    d.text((6, h + 30), "photo + render edges        |  disagreement (yellow: photo road only, magenta: render road only)", fill=(255, 255, 255))
    txt = "  ".join(f"{k} {g[k]['iou']:.2f}" for k in ("sky", "road", "marking_yellow", "vegetation", "terrain", "shoulder") if k in g)
    found = sum(i["found"] for i in res["instances"]); vis = sum(not i["occluded"] for i in res["instances"])
    d.text((2 * w + 14, h + 30), "IoU " + txt, fill=(255, 255, 255))
    d.text((2 * w + 14, h + 46), f"ledger found {found}/{vis} visible   score {res['score']:.3f}", fill=(255, 255, 255))
    sheet.save(OVL / f"{f:03d}.jpg", quality=85)


def _work(args):
    f, pose, overlay = args
    r = score_frame(f, pose, _W["ledger"], _W["catalog"], overlay)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(OUT / f"{f:03d}.json", "w"), indent=1)
    return r


_W = {}


def _init():
    try: os.nice(5)
    except OSError: pass
    _W["ledger"] = SC.load_ledger(); _W["catalog"] = SC.load_catalog()


def write_summary():
    rows = []
    for p in sorted(OUT.glob("[0-9][0-9][0-9].json")):
        r = json.load(open(p)); g = r["groups"]
        row = dict(frame=r["frame"], s=r["s"], pose_src=r["pose_src"], seg=r["seg"], score=r["score"])
        for k in GROUPS:
            row["iou_" + k] = g.get(k, {}).get("iou", "")
        vis = [i for i in r["instances"] if not i["occluded"]]
        row["ledger_visible"] = len(vis); row["ledger_found"] = sum(i["found"] for i in vis)
        row["t_s"] = r["t_total_s"]; rows.append(row)
    if rows:
        with open(OUT / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default=None); ap.add_argument("--all", action="store_true")
    ap.add_argument("--overlay", action="store_true"); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rebuild-cache", action="store_true"); a = ap.parse_args()
    try: os.nice(5)
    except OSError: pass
    t0 = time.time(); SC.build_cache(force=a.rebuild_cache)
    poses = load_poses()
    frames = sorted(poses) if a.all or not a.frames else [f for f in parse_frames(a.frames) if f in poses]
    jobs = [(f, poses[f], a.overlay) for f in frames]
    if len(jobs) > 3 and a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(min(a.workers, 3), initializer=_init) as pool:
            res = pool.map(_work, jobs, chunksize=2)
    else:
        _init(); res = [_work(j) for j in jobs]
    write_summary()
    for r in res:
        g = r["groups"]
        print(f"{r['frame']:03d} s={r['s']:7.1f} score {r['score']:.3f}  " + " ".join(
            f"{k}={g[k]['iou']:.2f}" for k in ("sky", "road", "marking_yellow", "vegetation", "terrain") if k in g) +
              f"  ledger {sum(i['found'] for i in r['instances'])}/{sum(not i['occluded'] for i in r['instances'])}  {r['t_total_s']:.2f}s")
    print(f"{len(res)} frames in {time.time() - t0:.1f}s wall")


if __name__ == "__main__":
    main()
