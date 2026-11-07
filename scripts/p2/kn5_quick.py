"""P: incremental kn5 support for 22_build_kn5.py (docs/PIPELINE.md).

A full 22 run (PALOMARES_KN5_QUICK unset) writes every kn5 and caches, in <build>/kn5_cache/:
  state.pkl         MATS (kn5.Material) + T (texture name -> PNG bytes) after albedo calibration / terrain macro atlases
  terrain_uv.pkl    final (UV, material) of every terrain mesh after the macro-atlas remap (22 mutates them)
  kn5_inputs.json   kn5 file -> sha1 of its inputs (mesh arrays + flags, dummies, materials, texture bytes, kn5.py)

A quick run (PALOMARES_KN5_QUICK=1) skips texture generation, loads that cache, re-reads the atlases the 21* steps
write (ledger / roadside / dressing / race), applies the cached terrain remap, then runs only the kn5-writing part of
22 and rewrites a kn5 only when its input hash changed.  Same writer, same inputs => byte-identical kn5.
"""
import hashlib, json, pathlib, pickle, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
HASHES = {}
_HFILE = [None]
ATLASES = {"ledger.png": "ledger_atlas.png", "roadside.png": "roadside_atlas.png",
           "dressing.png": "dressing_atlas.png", "race_assets.png": "race_assets_atlas.png"}
PART_B = ("def merge_by(meshes, key):", "# ------------------------------------------------------------------ AC folder")


def _writer_hash():
    return hashlib.sha1((HERE / "kn5.py").read_bytes()).hexdigest()


def input_hash(nodes, mats, textures):
    h = hashlib.sha1(_writer_hash().encode())
    def node(n):
        h.update(n.name.encode())
        if hasattr(n, "pos"):
            for a in (n.pos, n.nrm, n.uv, n.idx):
                h.update(np.ascontiguousarray(a).tobytes())
            h.update(repr((n.material, n.cast_shadows, n.visible, n.transparent, n.renderable, n.lod_in, n.lod_out, n.layer)).encode())
        else:
            h.update(repr(list(n.matrix) if n.matrix is not None else None).encode())
            for c in getattr(n, "children", None) or []:
                node(c)
    for n in nodes:
        node(n)
    for m in mats:
        h.update(repr((m.name, m.shader, sorted((k, repr(v)) for k, v in m.props.items()), sorted(m.textures.items()),
                       m.blend, m.alpha_tested, m.depth)).encode())
    for t in sorted(textures):
        h.update(t.encode()); h.update(hashlib.sha1(textures[t]).digest())
    return h.hexdigest()


def _load_hashes(kcache):
    _HFILE[0] = pathlib.Path(kcache) / "kn5_inputs.json"
    HASHES.clear()
    if _HFILE[0].exists():
        HASHES.update(json.loads(_HFILE[0].read_text()))


def save_hashes(_track=None):
    if _HFILE[0] is not None:
        _HFILE[0].write_text(json.dumps(HASHES, indent=1, sort_keys=True))


def save_cache(kcache, MATS, T, G):
    kcache = pathlib.Path(kcache); kcache.mkdir(parents=True, exist_ok=True)
    pickle.dump({"MATS": MATS, "T": T}, open(kcache / "state.pkl", "wb"), protocol=4)
    tuv = {m["name"]: (len(m["P"]), np.asarray(m["UV"], np.float64), m["material"]) for m in G.get("terrain", []) if m.get("macro")}
    pickle.dump(tuv, open(kcache / "terrain_uv.pkl", "wb"), protocol=4)
    _load_hashes(kcache); HASHES.clear()           # the full run rewrites every kn5 -> fresh hash table
    print(f"  kn5 cache: {len(T)} textures, {len(MATS)} materials, {len(tuv)} terrain meshes -> {kcache}")


def load_state(G, build, kcache):
    kcache = pathlib.Path(kcache)
    if not (kcache / "state.pkl").exists():
        sys.exit(f"kn5 quick: no cache in {kcache}; run a full 22_build_kn5.py on this build dir once")
    st = pickle.load(open(kcache / "state.pkl", "rb"))
    MATS, T = st["MATS"], st["T"]
    for tex, f in ATLASES.items():
        if (pathlib.Path(build) / f).exists() and tex in T:
            T[tex] = (pathlib.Path(build) / f).read_bytes()
    tuv = pickle.load(open(kcache / "terrain_uv.pkl", "rb"))
    bad = []
    for m in G.get("terrain", []):
        if not m.get("macro"):
            continue
        e = tuv.get(m["name"])
        if e is None or e[0] != len(m["P"]):
            bad.append(m["name"]); continue
        m["UV"], m["material"] = e[1], e[2]
    if bad:
        sys.exit(f"kn5 quick: terrain topology changed ({len(bad)} meshes, e.g. {bad[:3]}); run the full 22_build_kn5.py")
    _load_hashes(kcache)
    return MATS, T


def quick_section(path22, G, build, kcache):
    """source executed inside 22's globals: load the cache, then 22's own kn5-writing block (no texture generation,
    no AC folder ini/ui rewrite)"""
    src = pathlib.Path(path22).read_text()
    a, b = src.index(PART_B[0]), src.index(PART_B[1])
    return ("MATS, T = _KQ.load_state(G, BUILD, KCACHE)\n" + src[a:b] +
            "print('  quick: rewrote', getattr(write, 'written', []) or 'nothing')\n")
