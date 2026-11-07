"""P quick path: keep the 21d collision of the seed build (docs/PIPELINE.md).

  save     <build>/kn5_cache/collision_seed.pkl <- every 1WALL* mesh per group of <build>/geometry.pkl (after 21d)
  restore  in <build>/geometry.pkl: drop every 1WALL* mesh (21e re-adds raw per-prop boxes), put the saved ones back
           in their original order/position within each group
"""
import pickle, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geom import BUILD, ROOT
if BUILD.resolve() == (ROOT / "build").resolve():
    sys.exit("quick_collision: refuses to run on build/ (set PALOMARES_BUILD_DIR)")
from geom import build_lock; build_lock()

F = BUILD / "kn5_cache" / "collision_seed.pkl"
geo = pickle.load(open(BUILD / "geometry.pkl", "rb"))
G = geo["groups"]
wall = lambda m: m["name"].startswith("1WALL")
if sys.argv[1] == "save":
    F.parent.mkdir(parents=True, exist_ok=True)
    keep = {g: [(i, m) for i, m in enumerate(ms) if wall(m)] for g, ms in G.items()}
    order = {g: [m["name"] for m in ms] for g, ms in G.items()}
    pickle.dump({"walls": keep, "order": order}, open(F, "wb"), protocol=4)
    print(f"saved {sum(map(len, keep.values()))} collision meshes -> {F}")
else:
    st = pickle.load(open(F, "rb"))
    nd = na = 0
    for g in set(G) | set(st["walls"]):
        ms = [m for m in G.get(g, []) if not wall(m)]
        nd += len(G.get(g, [])) - len(ms)
        order = {n: k for k, n in enumerate(st["order"].get(g, []))}
        seed_w = [m for _, m in st["walls"].get(g, [])]
        na += len(seed_w)
        # stable merge: original seed order where known, new (21e) meshes keep their place after their predecessor
        allm = ms + seed_w
        pos = {}
        last = -1.0
        for m in G.get(g, []) + seed_w:
            if m["name"] in order:
                last = order[m["name"]]
            pos.setdefault(id(m), last + 0.5 if m["name"] not in order else order[m["name"]])
        G[g] = sorted(allm, key=lambda m: pos.get(id(m), 1e9))
    pickle.dump(geo, open(BUILD / "geometry.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"collision: dropped {nd} 1WALL meshes, restored {na} from the seed")
