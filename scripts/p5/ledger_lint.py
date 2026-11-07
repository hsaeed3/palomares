"""dev5 ledger lint: schema, catalog ids, overlaps, clearance from pavement.

  python3 scripts/p5/ledger_lint.py [--ledger data/scene/ledger.jsonl] [--strict]
Exit code 1 on errors (warnings with --strict).
"""
import argparse, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as SC

CLS = set("sign post pole mailbox guardrail fence wall bridge culvert building driveway gate shrub hedge rock debris prop "
          "marking mound ditch tree_density tree_add tree_remove".split())
LINEAR = {"fence", "guardrail", "wall", "marking", "hedge", "tree_density"}
POINT_MIN_SEP = {"sign": 0.6, "post": 0.3, "pole": 1.0, "mailbox": 0.4, "building": 4.0, "tree_add": 2.0, "shrub": 0.5, "rock": 0.5}
REQ = ["id", "cls", "s", "side", "off", "status"]
DEV7 = SC.ROOT / "data/derived/dev7"


def road_len(road, _c={}):
    """dev7 rows carry `road`: station range is that road's refined centerline (data/derived/dev7/centerline_<road>.csv)"""
    if road not in _c:
        f = DEV7 / f"centerline_{road}.csv"
        _c[road] = float(open(f).read().strip().splitlines()[-1].split(",")[0]) if f.exists() else None
    return _c[road]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ledger", default=str(SC.SCENE / "ledger.jsonl"))   # dev7: --ledger data/scene/ledger_dev7.jsonl
    ap.add_argument("--strict", action="store_true"); a = ap.parse_args()
    cat = SC.load_catalog(); errs, warns = [], []
    E = lambda i, m: errs.append(f"line {i}: {m}"); Wn = lambda i, m: warns.append(f"line {i}: {m}")
    if not os.path.exists(a.ledger):
        print("no ledger at", a.ledger); return 0
    objs, ids = [], collections.Counter()
    for i, ln in enumerate(open(a.ledger), 1):
        if not ln.strip(): continue
        try: o = json.loads(ln)
        except json.JSONDecodeError as e: E(i, f"bad json: {e}"); continue
        miss = [k for k in REQ if k not in o]
        if miss: E(i, f"missing {miss}")
        ids[o.get("id")] += 1; c = o.get("cls")
        if c not in CLS: E(i, f"cls {c!r} not allowed")
        if o.get("status") not in ("placed", "todo", "rejected"): E(i, f"status {o.get('status')!r}")
        if o.get("side") not in ("L", "R"): E(i, f"side {o.get('side')!r}")
        if o.get("conf") not in (None, "high", "med", "low"): Wn(i, f"conf {o.get('conf')!r}")
        if o.get("src") not in (None, "seg", "survey", "lidar", "manual", "mapillary", "naip", "osm"): Wn(i, f"src {o.get('src')!r}")
        try:
            s = float(o.get("s")); off = float(o.get("off", 0))
        except (TypeError, ValueError): E(i, "s/off not numeric"); continue
        smax = SC.S_MAX
        if "road" in o:
            smax = road_len(o["road"])
            if smax is None: E(i, f"road {o['road']!r} has no data/derived/dev7/centerline_*.csv"); continue
        if not 0 <= s <= smax + 0.5: E(i, f"s={s} outside 0..{smax:.0f}")
        s1 = o.get("s1")
        if c in LINEAR and s1 is None: E(i, f"linear {c} needs s1")
        if s1 is not None and float(s1) < s: E(i, "s1 < s")
        if o.get("asset"):
            if cat and o["asset"] not in cat: E(i, f"asset {o['asset']!r} not in config/asset_catalog.yaml")
        elif c not in ("mound", "ditch", "tree_density", "tree_remove", "marking", "driveway"):
            Wn(i, f"{c} without asset (scorer uses a default box)")
        if not cat and o.get("asset"): Wn(i, "config/asset_catalog.yaml missing: asset id unchecked")
        if off < 0 and c != "marking": E(i, f"off={off} < 0: only markings may sit on the pavement")
        if c in ("mound", "ditch"):
            pr = o.get("params") or {}
            r = float(pr.get("radius", 0) or 0)
            if r <= 0: E(i, f"{c} needs params.radius")
            elif off - r < 0.3: E(i, f"{c} within 0.3 m of pavement (off {off} - radius {r} = {off - r:.2f})")
        elif c not in ("marking", "tree_density", "tree_remove", "driveway") and 0 <= off < 0.3 and o.get("status") != "rejected":
            Wn(i, f"{c} only {off} m from pavement edge")
        if o.get("frames") and not all(isinstance(f, int) and 1 <= f <= 219 for f in o["frames"]): E(i, "frames must be ints 1..219")
        o["_line"] = i; objs.append(o)
    for k, n in ids.items():
        if n > 1: errs.append(f"duplicate id {k!r} x{n}")
    # overlaps: same side, same cls, live objects
    live = [o for o in objs if o.get("status") != "rejected"]
    by = collections.defaultdict(list)
    for o in live: by[(o.get("cls"), o.get("side"), o.get("road"))].append(o)
    for (c, side, _road), L in by.items():
        L.sort(key=lambda o: float(o["s"]))
        for p, q in zip(L, L[1:]):
            if c in LINEAR:
                if float(q["s"]) < float(p.get("s1") or p["s"]) - 0.5 and abs(float(q["off"]) - float(p["off"])) < 0.5:
                    Wn(q["_line"], f"{c} {q['id']} overlaps {p['id']} on side {side}")
            elif abs(float(q["s"]) - float(p["s"])) < POINT_MIN_SEP.get(c, 0.3) and abs(float(q["off"]) - float(p["off"])) < POINT_MIN_SEP.get(c, 0.3):
                Wn(q["_line"], f"{c} {q['id']} coincides with {p['id']}")
    for m in errs: print("ERROR", m)
    for m in warns: print("WARN ", m)
    print(f"{len(objs)} objects ({sum(o.get('status') == 'placed' for o in objs)} placed): {len(errs)} errors, {len(warns)} warnings")
    return 1 if errs or (a.strict and warns) else 0


if __name__ == "__main__":
    sys.exit(main())
