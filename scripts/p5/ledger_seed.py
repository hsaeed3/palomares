#!/usr/bin/env python3
"""Seed data/scene/ledger.jsonl from the dev4 survey files data/derived/roadside_assets_*.json.

Every legacy entry 21c used to place becomes ledger rows (src=survey, status=placed, conf from its confidence), so
dev5 starts no worse than dev4. `surface` and `remove_guardrail` entries are not scene objects and stay with 21c.

  python3 scripts/p5/ledger_seed.py            # refuses to overwrite a ledger that has non-survey rows
  python3 scripts/p5/ledger_seed.py --force    # rewrite the survey rows, keep every non-survey row as is

Offsets: legacy offset_m is measured from the ASPHALT edge; the ledger's `off` is beyond the pavement edge
(road.outer = asphalt + shoulder), so off = offset_m - shoulder width at that station (markings stay negative).
"""
import glob, json, math, pathlib, re, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/p2"))
LEDGER = ROOT / "data/scene/ledger.jsonl"

LEGACY_CLS = {   # legacy type -> ledger cls (also used by 21c's replace switch)
    "road_sign": "sign", "utility_pole": "pole", "delineator_post": "post", "guardrail_wbeam": "guardrail",
    "driveway_apron": "driveway", "chainlink_fence": "fence", "wood_split_rail_fence": "fence", "white_rail_fence": "fence",
    "wood_board_fence": "fence", "ranch_wire_fence": "fence", "solid_wood_privacy_fence": "fence", "guardrail_cable": "guardrail",
    "mailbox_cluster": "mailbox", "building": "building", "gate_estate": "gate", "hump_marking": "marking",
    "retaining_wall_concrete": "wall", "retaining_wall_stone": "wall", "retaining_wall_timber": "wall", "rockfall_mesh": "wall",
    "bridge_parapet_concrete": "bridge", "bridge_parapet_stone": "bridge", "culvert_headwall": "culvert", "cypress_row": "hedge",
}
FENCE_ID = {"chainlink_fence": "fence.chainlink", "wood_split_rail_fence": "fence.split_rail", "white_rail_fence": "fence.white_rail",
            "wood_board_fence": "fence.board_3rail", "ranch_wire_fence": "fence.ranch_wire", "solid_wood_privacy_fence": "fence.privacy"}
SIMPLE = {"retaining_wall_concrete": "wall.retaining_concrete", "retaining_wall_stone": "wall.retaining_stone",
          "retaining_wall_timber": "wall.timber_crib", "rockfall_mesh": "wall.rockfall_mesh", "bridge_parapet_concrete": "bridge.parapet_concrete",
          "bridge_parapet_stone": "bridge.parapet_stone", "culvert_headwall": "culvert.headwall", "hump_marking": "marking.hump",
          "guardrail_cable": "guardrail.cable"}
CONF = {"high": "high", "med": "med", "medium": "med", "low": "low"}


def sign_asset(text):
    import roadside_tex as RX
    _, _, _, key = RX.sign_image(text, None)
    if key.startswith("back"):
        return ("sign.back_diamond" if key == "back_d" else "sign.back_rect"), "sb"
    for pre, aid in (("stop", "sign.stop"), ("hump", "sign.hump"), ("mph", "sign.speed_limit"), ("mile", "sign.mile_marker"),
                     ("om3", "sign.object_marker"), ("blade_", "sign.street_name"), ("chev_l", "sign.chevron_left"), ("chev_r", "sign.chevron_right"),
                     ("narrowbridge", "sign.narrow_bridge"), ("arrow_left", "sign.arrow_left"), ("arrow_", "sign.arrow_right"),
                     ("mirror", "sign.mirror"), ("round", "sign.round_prohibition"), ("dia_left", "sign.curve_left"),
                     ("dia_right", "sign.curve_right"), ("dia_None_SLOW", "sign.slow"), ("dia_", "sign.warning_blank"),
                     ("txt_", "sign.text_private"), ("blank", "sign.blank_white")):
        if key.startswith(pre):
            return aid, "nb"
    return "sign.blank_white", "nb"


def building_asset(e):
    a = e.get("attrs") or {}; t = " ".join(str(v) for v in a.values()).lower()
    for k, aid in (("tank", "building.water_tank"), ("barn", "building.barn"), ("stable", "building.stable"), ("shed", "building.shed"),
                   ("garage", "building.garage"), ("trailer", "building.trailer"), ("mobile", "building.trailer"), ("carport", "building.carport"),
                   ("two-stor", "building.house_two_storey"), ("2-stor", "building.house_two_storey"), ("two stor", "building.house_two_storey")):
        if k in t:
            return aid
    return "building.ranch_house"


def legacy_to_rows(e):
    """one legacy survey entry -> list of ledger rows (without ids / off conversion: `off_asphalt` holds offset_m)"""
    t = e["type"]
    if t not in LEGACY_CLS:
        return []
    cls = LEGACY_CLS[t]; a = e.get("attrs") or {}
    s0, s1 = float(e["s0"]), float(e["s1"]); off = e.get("offset_m"); H = e.get("height_m")
    sd = e.get("side", "right"); sides = ["L", "R"] if sd in ("both", "center", None) else ["L" if sd == "left" else "R"]
    base = dict(cls=cls, frames=list(e.get("frames", [])), conf=CONF.get(str(e.get("confidence", "med")).lower(), "med"), src="survey",
                status="placed", note=str(a.get("note", ""))[:160], dz=0.0, yaw=0.0, scale=1.0)
    rows = []

    def add(asset, s, s_1=None, offv=None, **params):
        for sdd in sides:
            r = dict(id=None, **base); r.update(asset=asset, s=round(float(s), 2), s1=None if s_1 is None else round(float(s_1), 2), side=sdd,
                                                 off_asphalt=float(offv if offv is not None else (off if off is not None else 1.0)),
                                                 params={k: v for k, v in params.items() if v is not None})
            rows.append(r)

    if t == "road_sign":
        txt = " ".join(str(a.get(k, "")) for k in ("text", "type_detail", "shape")).strip() or str(a.get("note", ""))
        aid, dr = sign_asset(txt.lower())
        add(aid, 0.5 * (s0 + s1), offv=off if off is not None else 1.5, text=txt, dir=dr, height=H)
    elif t == "utility_pole":
        sp = float(a.get("spacing_m") or 45)
        for s in ([0.5 * (s0 + s1)] if s1 - s0 < 5 else list(np.arange(s0, s1 + 0.1, sp))):
            add("pole.utility_wood", s, offv=off if off is not None else 3, height=H or 10)
    elif t == "delineator_post":
        for s in (np.arange(s0, s1 + 0.01, 12.0) if s1 - s0 > 6 else [0.5 * (s0 + s1)]):
            add("post.delineator_white", s, offv=off if off is not None else 1.2, height=H)
    elif t == "guardrail_wbeam":
        add("guardrail.wbeam_wood" if "wood" in str(a.get("posts", "")).lower() else "guardrail.wbeam_steel", s0, s1, offv=off if off is not None else 0.8,
            height=min(max(float(H or 0.75), 0.6), 0.85))
    elif t == "driveway_apron":
        g = "gravel" in json.dumps(a).lower() or "dirt" in json.dumps(a).lower()
        if s1 - s0 < 3:
            s0, s1 = s0 - 2.5, s1 + 2.5
        add("driveway.gravel" if g else "driveway.asphalt", s0, s1, offv=-0.3)
    elif t in FENCE_ID:
        if s1 - s0 < 3:
            s1 = s0 + 6
        txt = json.dumps(a).lower(); aid = FENCE_ID[t]
        # the dev4 survey had no type for these; its notes name them (seen by eye: frame 206 iron, 191 lattice)
        if re.search(r"wrought|\biron\b|steel picket|picket \(iron", txt):
            aid = "fence.iron_picket"
        elif "lattice" in txt:
            aid = "fence.lattice"
        elif re.search(r"pipe rail|metal pipe|tube rail", txt):
            aid = "fence.pipe_rail"
        add(aid, s0, s1, offv=off if off is not None else 1.0, height=min(max(float(H), 0.6), 3.0) if H else None)
    elif t == "mailbox_cluster":
        n = int(a.get("count") or max(1, min(6, round((s1 - s0) / 0.5) + 1 if s1 > s0 else 2)))
        add("mailbox.single" if n == 1 else "mailbox.cluster", 0.5 * (s0 + s1), offv=off if off is not None else 1.5, count=n)
    elif t == "building":
        txt = " ".join(str(v) for v in a.values()).lower(); m = re.search(r"(\d+)\s*x\s*(\d+)", txt)
        L = float(m.group(1)) if m else max(s1 - s0, 10.0); D = float(m.group(2)) if m else 9.0
        add(building_asset(e), 0.5 * (s0 + s1), offv=off if off is not None else 12, L=min(max(L, 6), 30), D=min(max(D, 5), 20),
            eave=max(3.0, float(H) - 1.5) if H else None)
    elif t == "gate_estate":
        mat = (str(a.get("material", "")) + " " + str(a.get("note", ""))).lower()
        pill = any(k in mat for k in ("pillar", "brick", "stone")) and "not pillar" not in mat
        aid = ("gate.estate_brick_iron" if pill else "gate.chainlink_swing" if "chainlink" in mat else
               "gate.white_rail" if any(k in mat for k in ("white", "horse")) else "gate.pipe_ranch")
        if s1 - s0 < 3:
            s0, s1 = 0.5 * (s0 + s1) - 2.2, 0.5 * (s0 + s1) + 2.2
        add(aid, s0, s1, offv=off if off is not None else 3)
    elif t == "cypress_row":
        cnt = int(a.get("count") or 0)
        sp = float(a.get("spacing_m") or (((s1 - s0) / max(cnt - 1, 1)) if cnt > 1 else 4.5))
        add("hedge.cypress_row", s0, max(s1, s0 + 0.5), offv=off if off is not None else 4, height=H or 14, spacing=min(max(sp, 2.0), 9.0))
    elif t == "hump_marking":
        if s1 - s0 < 2:
            s0, s1 = s0 - 1.5, s1 + 1.5
        sides = ["L"]
        add("marking.hump", s0, s1, offv=None)
        rows[-1]["off_asphalt"] = None
    else:
        aid = SIMPLE[t]; hd = {"culvert_headwall": 0.6}.get(t)
        if s1 - s0 < 3:
            s0, s1 = s0 - 2, s1 + 2
        add(aid, s0, s1, offv=off if off is not None else (1.5 if t == "rockfall_mesh" else 0.3), height=H or hd)
    return rows


def load_legacy():
    """same file loading / fill-replacement / de-dup as 21c.load_assets"""
    files = sorted(glob.glob(str(ROOT / "data/derived/roadside_assets_*.json")))
    base, fills = [], []
    for f in files:
        d = json.load(open(f))
        if isinstance(d, dict):
            d = next((v for v in d.values() if isinstance(v, list)), [])
        for e in d:
            (fills if "fill" in f else base).append(e)
    if fills:
        ffr = {fr for e in fills for fr in e.get("frames", [])}
        lo = min(e["s0"] for e in fills); hi = max(e["s1"] for e in fills)
        base = [e for e in base if not (e.get("confidence") == "low" and ((set(e.get("frames", [])) and set(e.get("frames", [])) <= ffr) or (e["s0"] >= lo and e["s1"] <= hi)))]
    seen, out = set(), []
    for e in base + fills:
        k = (e["type"], e.get("side"), round(e["s0"]), round(e["s1"]))
        if k not in seen:
            seen.add(k); out.append(e)
    return out


def main():
    from geom import Road, CFG
    old = []
    if LEDGER.exists():
        old = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    keep = [r for r in old if r.get("src") != "survey"]
    if keep and "--force" not in sys.argv:
        sys.exit(f"ledger has {len(keep)} non-survey rows; rerun with --force (they are kept, survey rows are rewritten)")
    road = Road(CFG["build_end_m"]); N = len(road.s)
    rows = []
    for e in load_legacy():
        rows += legacy_to_rows(e)
    cnt = {}
    for r in rows:
        oa = r.pop("off_asphalt"); smax = float(road.s[-1]) - 1.0
        r["s"] = min(r["s"], smax)
        if r["s1"] is not None:
            r["s1"] = max(min(r["s1"], smax), r["s"])
        i = int(np.clip(round(r["s"]), 0, N - 1)); sw = float(road.sh_w(i, 1 if r["side"] == "L" else -1))
        if r["cls"] == "marking":
            r["off"] = round(-sw - float(road.el[i]), 2)          # on the centreline; the generator spans both lanes
            r["params"]["width"] = round(float(road.el[i] + road.er[i]) - 1.4, 2)
        elif r["cls"] == "driveway":
            r["off"] = 0.0; r["params"]["inset"] = round(sw + 0.3, 2)   # apron reaches back over the shoulder, 0.3 m onto the asphalt
        else:
            r["off"] = round(max(oa - sw, 0.0), 2)                # 21c also kept furniture >= the shoulder's outer edge
        pre = r["cls"]; cnt[pre] = cnt.get(pre, 0) + 1; r["id"] = f"{pre}_s{cnt[pre]:04d}"
    order = ["id", "cls", "asset", "s", "side", "off", "dz", "yaw", "scale", "s1", "params", "frames", "conf", "src", "status", "note"]
    rows = [{k: r.get(k) for k in order} for r in rows]
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "w") as f:
        for r in keep + rows:
            f.write(json.dumps(r) + "\n")
    print(f"ledger: {len(rows)} survey rows + {len(keep)} kept -> {LEDGER.relative_to(ROOT)}  {cnt}")


if __name__ == "__main__":
    main()
