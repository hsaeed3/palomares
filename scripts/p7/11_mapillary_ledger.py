"""dev7 M step 3: survey ledger rows (src=mapillary / lidar / osm) for the dev7 network.

Inputs : config/dev7/survey_manual.yaml (by-eye rows from Mapillary frames), data/derived/dev7/mapillary_features.csv
         (Mapillary map features: signs, lights, poles, street furniture), data/derived/dev7/osm_features.geojson (parking,
         benches, hydrants), data/derived/dev7/buildings.geojson (OSM footprints + LiDAR heights), mapillary_index.csv.
Outputs: data/scene/ledger_parts/dev7_mapillary.jsonl (by-eye + map-feature rows), dev7_town_lidar.jsonl (buildings,
         parking from OSM/LiDAR), reports/dev7/mapillary_attribution.csv.  Runtime < 10 s.
Every row has `road` (dev7 centerline id) and `s` on data/derived/dev7/centerline_<road>.csv; side L/R relative to
increasing s; off beyond the pavement edge. ledger_merge.py sends rows with `road` to data/scene/ledger_dev7.jsonl."""
import csv, json, re, sys, pathlib, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, yaml
from shapely.geometry import shape
from common7 import DER7, ROOT, ll2loc
from mly import Roads

PARTS = ROOT / "data/scene/ledger_parts"; REP = ROOT / "reports/dev7"
R = Roads()
CAT = {a["id"]: a for a in yaml.safe_load(open(ROOT / "config/asset_catalog.yaml"))["assets"]}
IDX = {r["id"]: r for r in csv.DictReader(open(DER7 / "mapillary_index.csv"))}
SAMPLE = [r for r in csv.DictReader(open(DER7 / "mapillary_sample.csv")) if r["have"] == "1"]
IMGDIR = ROOT / "data/raw/dev7/mapillary/img"
ON_DISK = {p.stem for p in IMGDIR.glob("*.jpg") if p.stat().st_size > 5000}
by_road = collections.defaultdict(list)
for iid in ON_DISK:
    r = IDX.get(iid)
    if r: by_road[r["road"]].append((float(r["s_m"]), iid))
for v in by_road.values(): v.sort()


def evidence(road, s, s1=None, n=3, win=40.0):
    """image ids on disk nearest the row (the frames it was judged from), at most n"""
    s1 = s if s1 is None else s1
    L = [(min(abs(a - s), abs(a - s1)) if not (s <= a <= s1) else 0.0, iid) for a, iid in by_road.get(road, [])]
    return [iid for d, iid in sorted(L)[:n] if d <= win]


rows, ids = [], collections.Counter()
def add(prefix, road, s, s1, side, off, asset, params, conf, src, note, mly=None, status="placed"):
    a = CAT[asset]; ids[prefix] += 1
    rid = f"d7m_{prefix}_{ids[prefix]:04d}"
    if a["linear"] and s1 is None: s1 = s + 4.0
    rows.append(dict(id=rid, road=road, cls=a["cls"], asset=asset, s=round(float(s), 1), side=side, off=round(float(off), 2),
                     dz=0.0, yaw=0.0, scale=1.0, s1=None if s1 is None else round(float(s1), 1), params=params or {}, frames=[],
                     mly=mly if mly is not None else evidence(road, s, s1), conf=conf, src=src, status=status, note=note))

# ---------------------------------------------------------------- 1. by-eye rows
man = yaml.safe_load(open(ROOT / "config/dev7/survey_manual.yaml"))["rows"]
for road, s, s1, side, off, asset, params, conf, note in man:
    add("eye", road, s, s1, side, off, asset, params, conf, "mapillary" if evidence(road, s, s1) else "osm", note)
flashers = [(r["road"], r["s"]) for r in rows if r["asset"] == "sign.flasher_mast"]
eye_osm_ways = {r["params"].get("osm_way") for r in rows if r["params"].get("osm_way")}

# ---------------------------------------------------------------- 2. Mapillary map features
SIGN = [(r"maximum-speed-limit-(\d+)", "sign.speed_limit"), (r"regulatory--stop--", "sign.stop"), (r"curve-left--", "sign.curve_left"),
        (r"curve-right--", "sign.curve_right"), (r"chevron-left", "sign.chevron_left"), (r"chevron-right", "sign.chevron_right"),
        (r"double-turn-first-left|double-curve-first-left", "sign.reverse_turn"), (r"double-turn-first-right|double-curve-first-right", "sign.reverse_turn"),
        (r"warning--turn-left", "sign.turn_left"), (r"warning--turn-right", "sign.turn_right"), (r"height-restriction", "sign.clearance"),
        (r"bicycles-crossing", "sign.bicycle_warning"), (r"wild-animals", "sign.deer"), (r"falling-rocks", "sign.falling_rocks"),
        (r"slippery-road", "sign.slippery"), (r"side-road-perpendicular-left", "sign.side_road_left"),
        (r"side-road-perpendicular-right", "sign.side_road_right"), (r"warning--traffic-signals", "sign.signal_ahead"),
        (r"stop-ahead", "sign.stop_ahead"), (r"railroad-crossing", "sign.rr_advance"), (r"pedestrians-crossing", "sign.ped_crossing"),
        (r"no-parking-or-no-stopping", "sign.no_parking"), (r"regulatory--yield", "sign.yield"), (r"no-entry", "sign.do_not_enter"),
        (r"one-way", "sign.one_way"), (r"highway-interchange|general-directions", "sign.guide_green"), (r"obstacle-delineator", "sign.object_marker"),
        (r"fine-for-littering", "sign.text_white_rect"), (r"road-narrows", "sign.warning_blank")]
PROP = {"object--support--utility-pole": ("pole.utility_wood", 8.0), "object--street-light": ("pole.streetlight", 8.0),
        "object--bench": ("prop.bench", 3.0), "object--trash-can": ("prop.trash_can", 3.0), "object--fire-hydrant": ("prop.hydrant", 3.0),
        "object--junction-box": ("prop.utility_box", 3.0)}
F = list(csv.DictReader(open(DER7 / "mapillary_features.csv")))
for f in F:
    f["s"] = float(f["s"]); f["off"] = float(f["off"])
groups = collections.defaultdict(list)          # merge repeated detections (2018 + 2023 triangulations) of one object
for f in F:
    v = f["value"]; asset = params = None
    for pat, aid in SIGN:
        m = re.search(pat, v)
        if m:
            asset = aid; params = {"value": int(m.group(1))} if aid == "sign.speed_limit" else {}
            if aid == "sign.reverse_turn": params = {"first": "R" if "first-right" in v else "L"}
            if aid == "sign.text_white_rect": params = {"text": "FINE FOR LITTERING"}
            break
    rad = 8.0
    if asset is None and v in PROP: asset, rad = PROP[v]; params = {}
    if asset is None: continue
    if f["road"] in ("niles_blvd",) and asset == "pole.streetlight": asset = "pole.lamp_ornamental"
    if asset.startswith("sign.") and any(rd == f["road"] and abs(s - f["s"]) < 25 for rd, s in flashers): continue   # panel on a flasher mast
    if abs(f["off"]) > 20: continue                 # far off the road (other street / triangulation error)
    groups[(f["road"], asset, json.dumps(params, sort_keys=True))].append(f)
nmf = 0
for (road, asset, pj), L in groups.items():
    L.sort(key=lambda f: f["s"]); clusters = []
    rad = 8.0 if asset.startswith("sign.") else PROP.get(next((k for k, v in PROP.items() if v[0] == asset), ""), (None, 6.0))[1]
    for f in L:
        if clusters and f["s"] - clusters[-1][-1]["s"] <= rad and (asset.startswith("sign.") or f["side"] == clusters[-1][-1]["side"]):
            clusters[-1].append(f)
        else:
            clusters.append([f])
    for c in clusters:
        new = [f for f in c if f["last_seen"] >= "2023"] or c
        s = float(np.median([f["s"] for f in new])); side = collections.Counter(f["side"] for f in new).most_common(1)[0][0]
        off = float(np.median([f["off"] for f in new if f["side"] == side]))
        off = min(max(off, 0.8 if asset.startswith(("sign.", "pole.")) else 0.4), 15.0)
        p = json.loads(pj); p["dir"] = "nb" if side == "R" else "sb"
        if not asset.startswith("sign."): p.pop("dir")
        p["mly_feature"] = [f["id"] for f in c]
        conf = "high" if len(c) >= 2 and c[0]["last_seen"] >= "2023" else ("med" if max(f["last_seen"] for f in c) >= "2023" else "low")
        note = f"Mapillary map feature x{len(c)} ({c[0]['value']}), last seen {max(f['last_seen'] for f in c)}"
        if max(f["last_seen"] for f in c) < "2023" and road == "niles_canyon_road":
            note += "; 2018 only on a road re-imaged in 2023: may have been removed"
        add("mf", road, s, None, side, off, asset, p, conf, "mapillary", note); nmf += 1

# ---------------------------------------------------------------- 3. OSM street furniture + parking (town)
TOWN = {"niles_blvd", "main_street", "mission_boulevard", "mission_boulevard_2", "mission_boulevard_3", "mission_boulevard_4",
        "sullivan_underpass", "vallejo_way", "vallejo_street", "2nd_street", "3rd_street", "e_street", "f_street", "g_street",
        "h_street", "i_street", "j_street", "l_street", "nichols_avenue", "sycamore_street", "old_canyon_road", "pleasanton_sunol_road",
        "paloma_road", "locomotive_common", "cinematic_common", "dollar_common", "essanay_avenue", "essanay_place", "chase_court",
        "limelight_terrace", "sennett_terrace", "quincy_terrace", "sierra_terrace", "nichols_court", "mayhews_road", "niles_canyon_road"}
mf_props = [(r["road"], r["s"], r["side"], r["asset"]) for r in rows if r["src"] == "mapillary"]
OF = json.load(open(DER7 / "osm_features.geojson"))["features"]
for f in OF:
    p = f["properties"]; g = shape(f["geometry"]); c = g.centroid
    asset = {"bench": "prop.bench", "post_box": "prop.utility_box", "drinking_water": "prop.hydrant"}.get(p.get("amenity"))
    if p.get("kind") == "parking" and p.get("osm_way") not in eye_osm_ways:
        q = R.project_ll(c.x, c.y)
        if q and q["dist"] < 120:
            x, y = ll2loc(*np.asarray(g.exterior.coords.xy))
            poly = np.round(np.c_[x, y], 2).tolist()
            ss = [R.project(float(a), float(b), q["road"])["s"] for a, b in zip(x, y)]
            add("osm", q["road"], min(ss), max(ss), q["side"], max(q["off"] - 0.0, 0.0), "driveway.parking_lot",
                dict(osm_way=p.get("osm_way"), name=p.get("name"), polygon=poly, surface=p.get("surface")), "med", "osm",
                "OSM parking polygon; surface by G", mly=[])
        continue
    if not asset: continue
    q = R.project_ll(c.x, c.y)
    if not q or q["dist"] > 30: continue
    if any(rd == q["road"] and abs(s - q["s"]) < 3 and a == asset for rd, s, sd, a in mf_props): continue
    add("osm", q["road"], q["s"], None, q["side"], max(q["off"], 0.4), asset, dict(osm_node=p.get("osm_node")), "med", "osm",
        f"OSM {p.get('amenity')}", mly=[])

# ---------------------------------------------------------------- 4. town buildings (OSM footprint + LiDAR height)
eye_bld = [(r["road"], r["s"], r["s1"] or r["s"], r["side"]) for r in rows if r["cls"] == "building"]
nb = 0
for f in json.load(open(DER7 / "buildings.geojson"))["features"]:
    p = f["properties"]; g = shape(f["geometry"]); c = g.centroid
    q = R.project_ll(c.x, c.y)
    if not q or q["dist"] > 60 or q["road"] not in TOWN: continue
    covered = any(rd == q["road"] and s0 - 5 <= q["s"] <= s1 + 5 and sd == q["side"] for rd, s0, s1, sd in eye_bld) and q["road"] == "niles_blvd"
    x, y = ll2loc(*np.asarray(g.exterior.coords.xy))
    h = p.get("lidar_height_p90_m")
    add("bld", q["road"], q["s"], None, q["side"], max(q["off"], 0.5), "building.town_generic",
        dict(osm_way=p.get("osm_way"), osm_building=p.get("building"), name=p.get("name"), levels=p.get("building:levels"),
             height_m=h, ground_z=p.get("ground_z"), area_m2=p.get("area_m2"), footprint=np.round(np.c_[x, y], 2).tolist(),
             facade="unverified", by_eye_row=covered),
        "med" if h else "low", "lidar",
        ("footprint OSM, height LiDAR p90; " + ("facade in the by-eye storefront row" if covered else "facade not seen on Mapillary")), mly=[])
    nb += 1

# ---------------------------------------------------------------- 5. LiDAR guardrail candidates not confirmed by eye -> todo
lin = [(r["road"], r["s"], r["s1"], r["side"]) for r in rows if r["cls"] in ("guardrail", "wall", "bridge") and r["s1"]]
gr_todo = []
for f in json.load(open(DER7 / "features.geojson"))["features"]:
    p = f["properties"]
    if p["kind"] != "guardrail_candidate" or p.get("road") != "niles_canyon_road": continue
    s0, s1, sd = float(p["s_start_m"]), float(p["s_end_m"]), "L" if p.get("side") == "left" else "R"
    if any(rd == p["road"] and sd == side and a < s1 + 5 and b > s0 - 5 for rd, a, b, side in lin): continue
    add("lgr", p["road"], s0, max(s1, s0 + 4), sd, 0.6, "guardrail.wbeam_steel", {"lidar_candidate": True}, "low", "lidar",
        "LiDAR guardrail candidate not confirmed on the 75 m Mapillary sample: check a dense frame before building", status="todo")
    gr_todo.append((round(s0), round(s1), sd))
print("LiDAR guardrail candidates left as todo:", gr_todo)

# ---------------------------------------------------------------- write
PARTS.mkdir(parents=True, exist_ok=True)
A = [r for r in rows if r["src"] in ("mapillary",)]; B = [r for r in rows if r["src"] not in ("mapillary",)]
for name, L in (("dev7_mapillary.jsonl", A), ("dev7_town_lidar.jsonl", B)):
    L.sort(key=lambda r: (r["road"], r["s"]))
    (PARTS / name).write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in L))
used = collections.defaultdict(set)
for r in rows:
    for iid in r["mly"]: used[iid].add(r["id"])
for iid in ON_DISK: used.setdefault(iid, set())
with open(REP / "mapillary_attribution.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["image_id", "creator", "captured_at", "sequence", "road", "s_m", "url", "license", "ledger_rows"])
    for iid in sorted(used, key=lambda i: (IDX[i]["road"], float(IDX[i]["s_m"])) if i in IDX else ("", 0)):
        r = IDX.get(iid, {})
        w.writerow([iid, r.get("creator", ""), r.get("captured_at", ""), r.get("sequence", ""), r.get("road", ""), r.get("s_m", ""),
                    f"https://www.mapillary.com/app/?pKey={iid}", "CC-BY-SA 4.0", " ".join(sorted(used[iid]))])
print(f"{len(man)} by-eye, {nmf} map-feature, {nb} buildings, {len(rows)} rows total; {len(A)} -> dev7_mapillary.jsonl, {len(B)} -> dev7_town_lidar.jsonl;",
      len(used), "images attributed")
