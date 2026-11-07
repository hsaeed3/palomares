#!/usr/bin/env python3
"""dev6 step 07: measured ledger part data/scene/ledger_parts/p6_measured.jsonl + rejection of the eyeballed rows it
replaces (in their own part files, status=rejected, note "dev6: replaced by <id>"). Idempotent: rows rejected by an
earlier run (note tag "[dev6]") are restored first, then matching is redone. Then run scripts/p5/ledger_merge.py.

  buildings   buildings.geojson within 120 m of the road -> building.measured (params: parts, ground_z, roof_rgb, wall)
              eyeballed building rows within 40 m of a measured footprint -> rejected
  fences      fences_detected.geojson segment detector (fence / wall / hedge lines, <= 25 m from the edge):
              ledger fence/wall row on the same side overlapping in station (+-8 m) -> measured row with params.xy
              and the ledger row's TYPE (GoPro), old row rejected; unmatched lines -> new rows, type from the nearest
              ledger fence on that side (<= 300 m) else fence.board_3rail, conf low
  guardrails  lateral-profile detections only SNAP existing ledger guardrails (offset), never add new ones
              (detector recall vs the ledger is 10/42; unconfirmed detections are too noisy)
  understorey understory.geojson verge hedge rows + bank shrubs (<= 12 m); eyeballed shrub / non-clipped hedge rows
              within 10 m of a measured one -> rejected
"""
import sys, json, pathlib, zlib, math
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts/p5"))
from common import DER, origin
import scene

O = origin(); X0, Y0 = O["x0"], O["y0"]
PARTS = ROOT / "data/scene/ledger_parts"; OUT = PARTS / "p6_measured.jsonl"; TAG = "[dev6]"
cl = np.genfromtxt(DER / "centerline.csv", delimiter=",", names=True)
WALLS = ["p6wall_white", "p6wall_cream", "p6wall_tan", "p6wall_grey", "p6wall_brown"]


def row(id_, cls, asset, s, side, off, s1=None, params=None, conf="high", src="lidar", note=""):
    return dict(id=id_, cls=cls, asset=asset, s=round(float(s), 1), side=side, off=round(float(off), 2), dz=0.0, yaw=0.0, scale=1.0,
                s1=None if s1 is None else round(float(s1), 1), params=params or {}, frames=[], conf=conf, src=src, status="placed", note=note)


def main():
    parts = {f: [json.loads(l) for l in open(f) if l.strip()] for f in sorted(PARTS.glob("*.jsonl")) if f != OUT}
    for f, rs in parts.items():                       # undo the previous run
        for r in rs:
            if TAG in (r.get("note") or "") and r.get("status") == "rejected":
                r["status"] = r.pop("_prev_status", "placed") if "_prev_status" in r else "placed"
                r["note"] = r["note"].split(" | " + TAG)[0]
    allr = [(f, r) for f, rs in parts.items() for r in rs]
    pos = {}
    for f, r in allr:
        try: x, y, _, _ = scene.ledger_place(r); pos[r["id"]] = (x, y)
        except Exception: pass
    new, rej = [], {}

    def reject(r, by):
        if r.get("status") != "placed" or r["id"] in rej: return
        rej[r["id"]] = by

    # ---- buildings
    B = json.load(open(DER / "buildings.geojson"))["features"]
    bl = [(f, r) for f, r in allr if r["cls"] == "building" and r.get("status") == "placed"]
    for b in B:
        p = b["properties"]
        if p["dist_road_m"] > 120 or p["ridge_h"] > 15 or p["ridge_h"] < 2.5: continue   # >15 m = canopy patch, <2.5 = not a building
        j = min(int(round(p["s"])), len(cl) - 1); edge = cl["edge_left_m"][j] if p["side"] == "L" else cl["edge_right_m"][j]
        nid = "p6_" + p["id"]
        new.append(row(nid, "building", "building.measured", min(max(p["s"], 0.5), float(cl["s_m"][-1]) - 0.5), p["side"], max(p["dist_road_m"] - edge, 0.0),
                       params=dict(parts=p["parts"], ground_z=p["ground_z"], roof_rgb=p["roof_rgb"], form=p["form"],
                                   wall=WALLS[zlib.crc32(p["id"].encode()) % len(WALLS)], eave_h=p["eave_h"], ridge_h=p["ridge_h"],
                                   area_m2=p["area_m2"], osm_iou=p["osm_iou"]),
                       conf="high" if p["n_pts"] > 150 and p["area_m2"] > 30 else "med", src="lidar",
                       note=f"LiDAR 2021 footprint {p['area_m2']} m2, {p['form']} roof, eave {p['eave_h']} m ridge {p['ridge_h']} m; NAIP 2022 roof colour; wall colour NOT measured"))
        from shapely.geometry import shape, Point
        P = shape(b["geometry"])
        for f, r in bl:
            if r["id"] in pos:
                x, y = pos[r["id"]]; d = P.distance(Point(x[0] + X0, y[0] + Y0))
                if d <= 40: reject(r, nid)

    # ---- fences / walls / hedge lines (segment detector)
    F = json.load(open(DER / "fences_detected.geojson"))["features"]
    fl = [(f, r) for f, r in allr if r["cls"] in ("fence", "wall") and r.get("status") == "placed"]
    for fe in F:
        p = fe["properties"]
        if p.get("detector") == "lateral_profile" or p["off_beyond_edge"] > 25: continue
        xy = np.asarray(fe["geometry"]["coordinates"]) - [X0, Y0]
        # NB order
        a, b = xy[0], xy[-1]
        if np.dot(b - a, [np.interp(p["s0"], cl["s_m"], np.gradient(cl["x_local"])), np.interp(p["s0"], cl["s_m"], np.gradient(cl["y_local"]))]) < 0: xy = xy[::-1]
        match = [r for f, r in fl if r["side"] == p["side"] and r.get("s1") is not None and min(r["s1"], p["s1"]) + 8 > max(r["s"], p["s0"])]
        c_ = p["cls"]; nid = "p6_" + p["id"]
        if c_ == "hedge":
            asset, conf = "hedge.coyote_brush", "med"
        elif match:
            mr = match[0]; asset = mr["asset"]; conf = "high"; c_ = mr["cls"]
            for r in match: reject(r, nid)
        else:
            near = [r for f, r in fl if r["side"] == p["side"] and abs(r["s"] - p["s0"]) < 300 and r["cls"] == "fence"]
            asset = min(near, key=lambda r: abs(r["s"] - p["s0"]))["asset"] if near and c_ == "fence" else ("wall.retaining_concrete" if c_ == "wall" else "fence.board_3rail")
            conf = "low"
        new.append(row(nid, c_, asset, p["s0"], p["side"], max(p["off_beyond_edge"], 0), s1=max(p["s1"], p["s0"] + 1), conf=conf,
                       params=dict(xy=np.round(xy, 2).tolist()), src="lidar",
                       note=f"LiDAR line {p['length_m']} m, median {p['hag_med']} m high; type {'from matched GoPro row' if conf == 'high' else 'NOT measured (guess)'}"))
    # ---- guardrail snapping
    G = [fe["properties"] for fe in F if fe["properties"].get("detector") == "lateral_profile"]
    for f, r in allr:
        if r["cls"] != "guardrail" or r.get("status") != "placed" or r.get("s1") is None: continue
        ov = [(min(r["s1"], g["s1"]) - max(r["s"], g["s0"]), g) for g in G if g["side"] == r["side"]]
        ov = [o for o in ov if o[0] > 0.3 * (r["s1"] - r["s"])]
        if not ov: continue
        g = max(ov, key=lambda o: o[0])[1]; nid = f"p6_{r['id']}_snap"
        new.append(row(nid, "guardrail", r["asset"], r["s"], r["side"], max(g["off_beyond_edge"], 0), s1=r["s1"], params=dict(r.get("params") or {}),
                       src="lidar", conf="high", note=f"GoPro row {r['id']} (type, span) with offset snapped to the LiDAR rail line {g['id']} ({g['off_beyond_edge']} m beyond edge, was {r.get('off')})"))
        reject(r, nid)
    # ---- understorey
    U = json.load(open(DER / "understory.geojson"))["features"]
    meas_s = {"L": [], "R": []}
    for k, u in enumerate(U):
        p = u["properties"]
        if p["cls"] == "hedge":
            new.append(row(f"p6_us_{k:05d}", "hedge", p["asset"], p["s"], p["side"], p["off"], s1=p["s1"], conf="med", src="lidar",
                           params=dict(density=p["density"], seed=k), note=f"LiDAR 0.3-3 m return share {p['frac']}, h90 {p['h90']} m, NAIP NDVI {p['ndvi']}; species NOT measured"))
            meas_s[p["side"]] += list(np.arange(p["s"], p["s1"] + 1, 2.0))
        elif p["off"] <= 12:
            new.append(row(f"p6_us_{k:05d}", "shrub", p["asset"], p["s"], p["side"], p["off"], conf="med", src="lidar",
                           params=dict(scale=p["scale"], seed=k), note=f"LiDAR shrub, h90 {p['h90']} m, share {p['frac']}, NDVI {p['ndvi']}"))
            meas_s[p["side"]].append(p["s"])
    ms = {k: np.sort(np.array(v)) for k, v in meas_s.items()}
    for f, r in allr:
        if r.get("status") != "placed": continue
        if r["cls"] == "shrub" or (r["cls"] == "hedge" and r["asset"] not in ("hedge.clipped", "hedge.cypress_row")):
            a = ms[r["side"]]
            if len(a) and np.min(np.abs(a - r["s"])) <= 10: reject(r, "p6 understorey layer")
    # ---- write
    for f, rs in parts.items():
        ch = False
        for r in rs:
            if r["id"] in rej:
                r["status"] = "rejected"; r["note"] = (r.get("note") or "") + f" | {TAG} replaced by measured {rej[r['id']]}"; ch = True
        if ch or any(TAG in (r.get("note") or "") for r in rs) or True:
            tmp = f.with_suffix(".tmp"); tmp.write_text("".join(json.dumps(r) + "\n" for r in rs)); tmp.replace(f)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in new))
    import collections
    print("p6_measured rows:", dict(collections.Counter(r["cls"] for r in new)))
    print("rejected eyeballed rows:", len(rej), dict(collections.Counter(next(r["cls"] for f, r in allr if r["id"] == i) for i in rej)))


if __name__ == "__main__":
    main()
