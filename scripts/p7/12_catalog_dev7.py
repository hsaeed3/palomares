"""dev7 M: add the asset types the Mapillary survey needs to config/asset_catalog.yaml, flagged `dev7: true`.
They have no generator yet (package A builds them); scripts/p5/assets.py --yaml keeps dev7 rows it does not generate.
Idempotent: re-running replaces the dev7 rows. < 1 s."""
import pathlib, yaml
ROOT = pathlib.Path(__file__).resolve().parents[2]; P = ROOT / "config/asset_catalog.yaml"
SEG = {"sign": "sign_face", "pole": "pole", "prop": "other_prop", "wall": "wall", "fence": "fence", "bridge": "bridge",
       "building": "building", "driveway": "road", "marking": "marking_white", "guardrail": "guardrail", "tree_add": "vegetation_tree", "post": "pole"}
# id: (cls, linear, (L, W, H), proxy shape, z0, collision, desc, notes)
NEW = {
 "sign.flasher_mast": ("sign", False, (0.3, 4.0, 6.0), "box", 0.0, True, "galvanised pole with cantilever arm over the lane, yellow warning panel (params.panel) and 2 amber flashing beacons (CSP emissive blink)", "the 'flashing warning signs before the railroad underpasses' on Niles Canyon Rd; params: dir, panel, advisory, beacons, arm_m"),
 "sign.warning_text_rect": ("sign", False, (0.06, 1.2, 2.4), "plane", 1.5, True, "yellow rectangular warning text panel on 2 posts (params.text)", "text is generic font only"),
 "sign.reverse_turn": ("sign", False, (0.06, 0.75, 2.1), "plane", 1.35, True, "MUTCD W1-3/W1-4 reverse turn/curve diamond (params.first L/R)", ""),
 "sign.turn_left": ("sign", False, (0.06, 0.75, 2.1), "plane", 1.35, True, "MUTCD W1-1L sharp turn left diamond", ""),
 "sign.turn_right": ("sign", False, (0.06, 0.75, 2.1), "plane", 1.35, True, "MUTCD W1-1R sharp turn right diamond", ""),
 "sign.side_road_left": ("sign", False, (0.06, 0.75, 2.1), "plane", 1.35, True, "MUTCD W2-2L side road left diamond", ""),
 "sign.side_road_right": ("sign", False, (0.06, 0.75, 2.1), "plane", 1.35, True, "MUTCD W2-2R side road right diamond", ""),
 "sign.text_diamond": ("sign", False, (0.06, 0.9, 2.3), "plane", 1.4, True, "yellow warning diamond with text (params.text, e.g. NARROW SUBWAY)", ""),
 "sign.clearance": ("sign", False, (0.06, 0.9, 2.3), "plane", 1.4, True, "low-clearance / height warning (params.text e.g. 14'-6\")", "mly warning--height-restriction"),
 "sign.bicycle_warning": ("sign", False, (0.06, 0.75, 2.4), "plane", 1.35, True, "W11-1 bicycle diamond (+ SHARE THE ROAD plaque)", ""),
 "sign.deer": ("sign", False, (0.06, 0.75, 2.1), "plane", 1.35, True, "W11-3 deer crossing diamond", ""),
 "sign.falling_rocks": ("sign", False, (0.06, 0.75, 2.1), "plane", 1.35, True, "W8-14 fallen rocks diamond", ""),
 "sign.slippery": ("sign", False, (0.06, 0.75, 2.1), "plane", 1.35, True, "W8-5 slippery when wet diamond", ""),
 "sign.signal_ahead": ("sign", False, (0.06, 0.75, 2.1), "plane", 1.35, True, "W3-3 signal ahead diamond", ""),
 "sign.stop_ahead": ("sign", False, (0.06, 0.75, 2.1), "plane", 1.35, True, "W3-1 stop ahead diamond", ""),
 "sign.rr_advance": ("sign", False, (0.06, 0.9, 2.1), "plane", 1.35, True, "W10-1 round yellow railroad advance warning", ""),
 "sign.ped_crossing": ("sign", False, (0.06, 0.75, 2.1), "plane", 1.35, True, "W11-2 pedestrian crossing (yellow-green)", ""),
 "sign.no_parking": ("sign", False, (0.06, 0.45, 2.1), "plane", 1.5, True, "R7 no parking / no stopping regulatory panel", ""),
 "sign.yield": ("sign", False, (0.06, 0.9, 2.1), "plane", 1.3, True, "R1-2 yield", ""),
 "sign.do_not_enter": ("sign", False, (0.06, 0.75, 2.1), "plane", 1.35, True, "R5-1 do not enter", ""),
 "sign.one_way": ("sign", False, (0.06, 0.9, 2.1), "plane", 1.6, True, "R6-1 one way arrow", ""),
 "sign.guide_green": ("sign", False, (0.1, 2.5, 3.5), "plane", 2.0, True, "green guide / direction sign on 2 posts (params.text)", ""),
 "sign.city_limit": ("sign", False, (0.06, 1.2, 2.4), "plane", 1.6, True, "green community / city-limit name sign (params.text)", ""),
 "sign.route_shield": ("sign", False, (0.06, 0.6, 2.3), "plane", 1.6, True, "California state route shield (params.route), green spade", ""),
 "sign.text_white_rect": ("sign", False, (0.06, 0.6, 2.1), "plane", 1.4, True, "white regulatory text panel (params.text, e.g. FINE FOR LITTERING)", ""),
 "sign.store_text": ("sign", False, (0.1, 3.0, 0.8), "plane", 3.0, False, "plain business-name text board on a facade (params.text, generic font, no logo)", "PLAN.md naming rule: storefronts show only their plain name, e.g. DEVOUT"),
 "pole.traffic_signal": ("pole", False, (0.4, 6.0, 6.5), "cylinder", 0.0, True, "signal pole with mast arm and 3-section heads (params.heads)", "town junctions: Mission/NCR, Mission/Niles"),
 "pole.lamp_ornamental": ("pole", False, (0.3, 1.2, 4.5), "cylinder", 0.0, True, "black ornamental acorn/arm lamp post with banner brackets (Niles Blvd)", ""),
 "prop.bench": ("prop", False, (1.8, 0.6, 0.9), "box", 0.0, True, "street bench (wood slats, black iron frame)", ""),
 "prop.trash_can": ("prop", False, (0.6, 0.6, 1.0), "cylinder", 0.0, True, "street litter bin (black metal)", ""),
 "prop.planter": ("prop", False, (0.9, 0.9, 0.7), "cylinder", 0.0, True, "round concrete planter with shrub", ""),
 "prop.water_barrier": ("prop", False, (1.8, 0.5, 0.8), "box", 0.0, True, "orange plastic water-filled barrier (temporary)", ""),
 "prop.crash_cushion": ("prop", False, (3.0, 0.9, 0.9), "box", 0.0, True, "yellow crash cushion / attenuator at a barrier end", ""),
 "prop.rail_crossing_signal": ("prop", False, (0.4, 2.0, 4.0), "box", 0.0, True, "railroad crossing signal: crossbuck, 2 red flashing lights, optional gate arm (params.gates)", ""),
 "prop.rail_crossbuck": ("prop", False, (0.2, 1.2, 3.0), "plane", 0.0, True, "passive crossbuck on a post", ""),
 "prop.rail_platform": ("prop", False, (60, 4.0, 0.6), "box", 0.0, True, "low station platform (params.length_m) with edge strip", ""),
 "prop.rail_car_generic": ("prop", False, (24, 3.1, 4.3), "box", 0.0, True, "generic period passenger coach / freight car, plain livery, no lettering (params.type, colour)", "rolling stock is generic per PLAN.md"),
 "wall.jersey_barrier": ("wall", True, (1, 0.6, 0.81), "box", 0.0, True, "concrete K-rail / jersey barrier", ""),
 "wall.masonry_parapet": ("wall", True, (1, 0.45, 0.9), "box", 0.0, True, "old low parapet wall (params.material stone|concrete)", ""),
 "wall.rock_cut_face": ("wall", True, (1, 1.0, 10), "box", 0.0, True, "steep exposed rock cut face (terrain + collision, not a built wall)", "use rock_cut material; collision wall where the face is vertical"),
 "wall.retaining_block": ("wall", True, (1, 0.4, 3.5), "box", 0.0, True, "tan split-face concrete block retaining wall (params.fence_on_top)", "Mission Blvd lowered section"),
 "fence.screen_curved": ("fence", True, (1, 0.1, 3.0), "plane", 0.0, True, "curved chain-link protective screen on a bridge over railway", ""),
 "fence.construction_mesh": ("fence", True, (1, 0.05, 1.2), "plane", 0.0, False, "orange plastic construction mesh (temporary)", ""),
 "fence.picket": ("fence", True, (1, 0.05, 1.0), "plane", 0.0, True, "white wooden picket fence", ""),
 "bridge.parapet_balustrade": ("bridge", True, (1, 0.35, 1.1), "box", 0.0, True, "historic panelled concrete bridge parapet with raised sidewalk", ""),
 "bridge.rail_truss_through": ("bridge", False, (46, 5.0, 9.0), "box", 4.6, True, "steel through-truss railway span over the road (params.span_m, clearance_m) on concrete piers", ""),
 "bridge.rail_deck_girder": ("bridge", False, (25, 5.0, 2.5), "box", 4.6, True, "steel deck / plate-girder railway span over the road (params.span_m, count)", ""),
 "bridge.rail_concrete_girder": ("bridge", False, (18, 8.0, 2.0), "box", 4.3, True, "concrete girder railway underpass with concrete abutments, wing walls, pipe railing", ""),
 "bridge.abutment_concrete": ("bridge", False, (6, 3.0, 6.0), "box", 0.0, True, "concrete bridge abutment / pier beside the road", ""),
 "bridge.abutment_stone": ("bridge", False, (6, 3.0, 6.0), "box", 0.0, True, "cut-stone (ashlar) bridge abutment beside the road", ""),
 "building.storefront": ("building", False, (15, 15, 8), "box", 0.0, True, "town storefront (params: storeys, height_m, facade, colour, sign_text)", "facades: brick_firehouse, stucco_arched, flat_parapet, victorian_bay, western_false_front, brick, stucco_mission, cafe_patio, board_batten_gable"),
 "building.storefront_row": ("building", False, (40, 15, 6), "box", 0.0, True, "row of attached storefronts from s to s1 (params.shops: plain names)", ""),
 "building.depot": ("building", False, (25, 8, 7), "box", 0.0, True, "wooden railway depot / freight station (params.facade)", ""),
 "building.town_generic": ("building", False, (12, 10, 6), "box", 0.0, True, "town building from the OSM/LiDAR footprint (params.footprint local xy, height_m LiDAR p90, osm tag); facade unverified", "src lidar"),
 "driveway.parking_lot": ("driveway", True, (10, 20, 0.05), "plane_h", 0.0, False, "asphalt or gravel parking lot (params.polygon local xy when from OSM)", "package G builds the surface"),
 "marking.keep_clear": ("marking", False, (6, 3.0, 0.02), "plane_h", 0.0, False, "KEEP CLEAR stencil", ""),
 "marking.angle_parking": ("marking", True, (1, 5.5, 0.02), "plane_h", 0.0, False, "angled parking stall lines (params.angle_deg)", ""),
 "tree.palm_row": ("tree_add", True, (1, 6, 15), "cylinder", 0.0, True, "row of Canary Island date palms (params.count)", "vegetation module: tall single trunk palm asset"),
 "tree.plane_row": ("tree_add", True, (1, 10, 14), "cylinder", 0.0, True, "row of London plane street trees (params.spacing_m)", ""),
}
d = yaml.safe_load(open(P)); hdr = "".join(l for l in open(P) if l.startswith("#"))
A = [a for a in d["assets"] if not a.get("dev7")]
have = {a["id"] for a in A}
for aid, (cls, lin, (L, W, H), shp, z0, col, desc, notes) in NEW.items():
    if aid in have: continue
    px = dict(shape=shp, w=L, d=W, h=H) if shp != "cylinder" else dict(shape=shp, r=round(W / 2, 2), h=H, w=L, d=W)
    A.append(dict(id=aid, cls=cls, linear=lin, dims_m=dict(L=L, W=W, H=H), proxy=px, z0=z0, collision=col, seg_class=SEG[cls],
                  materials=["ledger"], tris=None, tris_per_10m=None, frames=[], generator=None, desc=desc, dev7=True,
                  notes=("dev7 package M (Mapillary survey); no generator yet, package A builds it. " + notes).strip()))
d["assets"] = A
P.write_text(hdr + yaml.safe_dump(d, sort_keys=False, width=140))
print(sum(a.get("dev7", False) for a in A), "dev7 types;", len(A), "total")
