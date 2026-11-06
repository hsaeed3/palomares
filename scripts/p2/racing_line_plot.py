"""Top-down racing-line plots per run direction -> reports/racing_line/
  <dir>_overview.png : whole road, line coloured by speed
  <dir>_tNN.png      : ~1.2 km tiles: asphalt edges, line (speed colours), kerbs (apex red / exit orange),
                       run-off (green ticks), braking onset (x), brake boards (squares), apex (dot)
  summary.json       : per direction: run time, speeds, zones, kerbs, boards
Needs build/racing_line.pkl (racing_line.py) and build/geometry.pkl meta (kerbs, brake_boards) when present."""
import json, pickle, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import ROOT, BUILD, CFG, Road
import racing_line as RL

OUT = ROOT / "reports/racing_line"; OUT.mkdir(parents=True, exist_ok=True)
road = Road(CFG["build_end_m"]); R = RL.compute(road); C = RL.cfg()
meta = {}
try:
    meta = pickle.load(open(BUILD / "geometry.pkl", "rb"))["meta"]
except Exception as e:
    print("no geometry meta:", e)
KERBS = meta.get("kerbs", []); BOARDS = meta.get("brake_boards", {})
xl, yl = road.xy(np.arange(len(road.s)), road.el); xr, yr = road.xy(np.arange(len(road.s)), -road.er)
VMIN, VMAX = 40, 200


def speed_line(ax, D, sl, lw):
    p = np.c_[D["x"][sl], D["y"][sl]]
    seg = np.stack([p[:-1], p[1:]], 1)
    lc = LineCollection(seg, cmap="turbo", norm=plt.Normalize(VMIN, VMAX), linewidths=lw)
    lc.set_array(D["v"][sl][:-1] * 3.6); ax.add_collection(lc); return lc


summary = {}
for dname, sd in (("nb", 1), ("sb", -1)):
    D = R[dname]
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.plot(road.x, road.y, color="0.8", lw=4)
    lc = speed_line(ax, D, slice(None), 1.5)
    for b in BOARDS.get(dname, []):
        i = int(b["s_turn_in"]); ax.plot(road.x[i], road.y[i], "ks", ms=4)
    for s_ in range(0, int(road.s[-1]), 1000):
        ax.annotate(f"{s_ / 1000:.0f} km", (road.x[s_], road.y[s_]), fontsize=7, color="0.3")
    ax.set_aspect("equal"); ax.set_title(f"Palomares racing line {dname.upper()} ({'north' if sd > 0 else 'south'}bound) — speed km/h")
    plt.colorbar(lc, ax=ax, shrink=0.6); fig.savefig(OUT / f"{dname}_overview.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    TILE = 400
    for t, s0 in enumerate(range(0, int(road.s[-1]), TILE)):
        s1 = min(s0 + TILE, int(road.s[-1])); sl = slice(s0, s1 + 1)
        fig, ax = plt.subplots(figsize=(11, 11))
        ax.fill(np.r_[xl[sl], xr[sl][::-1]], np.r_[yl[sl], yr[sl][::-1]], color="0.75", lw=0)
        ax.plot(xl[sl], yl[sl], color="0.3", lw=0.6); ax.plot(xr[sl], yr[sl], color="0.3", lw=0.6)
        ax.plot(road.x[sl], road.y[sl], color="0.55", lw=0.5, ls="--")
        lc = speed_line(ax, D, sl, 2.2)
        for k in KERBS:
            if k["dir"] != dname or k["s1"] < s0 or k["s0"] > s1: continue
            i = np.arange(int(k["s0"]), int(k["s1"]) + 1); e = (road.el[i] + 0.3) if k["side"] == "left" else -(road.er[i] + 0.3)
            x, y = road.xy(i, e); ax.plot(x, y, color="#111111" if k["type"] == "apex" else "#1f5cff", lw=3.5, solid_capstyle="butt")
        for z in D["zones"]:
            for key, mk, col in (("onset", "x", "k"), ("apex", "o", "m")):
                i = z[key]
                if s0 <= i <= s1: ax.plot(D["x"][i], D["y"][i], mk, color=col, ms=6)
        for b in BOARDS.get(dname, []):
            for d_, sb_ in zip(C["board_distances_m"], b["boards"]):
                i = int(round(sb_))
                if s0 <= i <= s1:
                    sg = 1 if b["outside"] == "left" else -1
                    x, y = road.xy(i, sg * (road.el[i] if sg > 0 else road.er[i]) + sg * 1.5)
                    ax.plot(x, y, "s", color="white", mec="k", ms=7); ax.annotate(str(d_), (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        for s_ in range((s0 // 100 + 1) * 100, s1, 100):
            ax.annotate(f"{s_}", (road.x[s_], road.y[s_]), fontsize=6, color="0.25", xytext=(6, 6), textcoords="offset points")
        i_a = s0 if sd > 0 else s1
        ax.annotate("run →" if sd > 0 else "run ←", (road.x[i_a], road.y[i_a]), fontsize=9, weight="bold")
        ax.set_aspect("equal"); ax.set_title(f"{dname.upper()} s {s0}–{s1} m: line by speed, apex kerb black, exit kerb blue, x brake onset, o apex, □ boards")
        plt.colorbar(lc, ax=ax, shrink=0.5, label="km/h")
        fig.savefig(OUT / f"{dname}_t{t:02d}.png", dpi=90, bbox_inches="tight"); plt.close(fig)
    # lateral view: offset of the line across the asphalt vs station (driver's view of where the line goes)
    nr = int(np.ceil(road.s[-1] / 1300)); fig, axs = plt.subplots(nr, 1, figsize=(16, 2.0 * nr))
    for r_, ax in enumerate(axs):
        s0 = r_ * 1300; s1 = min(s0 + 1300, int(road.s[-1])); sl = slice(s0, s1 + 1); ss = road.s[sl]
        ax.fill_between(ss, -road.er[sl], road.el[sl], color="0.85"); ax.axhline(0, color="0.6", lw=0.5, ls="--")
        p = np.c_[ss, D["a"][sl]]; lc2 = LineCollection(np.stack([p[:-1], p[1:]], 1), cmap="turbo", norm=plt.Normalize(VMIN, VMAX), linewidths=2)
        lc2.set_array(D["v"][sl][:-1] * 3.6); ax.add_collection(lc2)
        for k in KERBS:
            if k["dir"] != dname or k["s1"] < s0 or k["s0"] > s1: continue
            i = np.arange(int(k["s0"]), int(k["s1"]) + 1)
            e = road.el[i] if k["side"] == "left" else -road.er[i]
            ax.plot(road.s[i], e, color="#d01010" if k["type"] == "apex" else "#1f5cff", lw=4, solid_capstyle="butt")
        for b in BOARDS.get(dname, []):
            if s0 <= b["s_turn_in"] <= s1: ax.axvline(b["s_turn_in"], color="k", lw=0.8, ls=":")
        ax.set_xlim(s0, s0 + 1300); ax.set_ylim(-6.5, 6.5); ax.set_ylabel("left +  m")
    axs[0].set_title(f"{dname.upper()}: line offset across the asphalt (+ = station-left), colour = speed; kerbs red apex / blue exit; dotted = boarded turn-in")
    axs[-1].set_xlabel("station s (m)"); fig.savefig(OUT / f"{dname}_lateral.png", dpi=80, bbox_inches="tight"); plt.close(fig)
    summary[dname] = dict(flying_time_full_s=round(float(np.sum(1 / np.maximum(D["v_run"], 1))), 1),
                          vmax_kmh=round(float(D["v"].max() * 3.6)), vmin_kmh=round(float(D["v"].min() * 3.6)),
                          braking_zones=len(D["zones"]), kerbs=sum(k["dir"] == dname for k in KERBS),
                          boards=BOARDS.get(dname, []), touch_frac_l=round(float(np.mean(D["gap_l"] < 0.6)), 3),
                          touch_frac_r=round(float(np.mean(D["gap_r"] < 0.6)), 3))
ai = BUILD / "ai_lines.json"
if ai.exists():
    summary["ai_lines"] = json.load(open(ai))
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
print("racing line plots ->", OUT)
