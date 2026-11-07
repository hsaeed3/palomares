#!/usr/bin/env python3
"""P: prepared asset library, assets/<type>/ (docs/PIPELINE.md "Asset library").

  python3 scripts/p2/assetlib.py                 # build every asset whose sources changed (skip the rest)
  python3 scripts/p2/assetlib.py sign.curve_left tree.oak   # only these
  python3 scripts/p2/assetlib.py --force --list

Per asset type, assets/<type>/ holds
  manifest.json   type, cls, source_hash, sources, parts (mesh + texture + material), LODs, collision, verts/tris,
                  GoPro frames where it is seen, build time
  lod0.npz ...    kn5-ready mesh arrays per part (P, N, UV, T; local frame, metres, z up, base at origin)
  collision.json  collision proxy (box/cylinder primitives from the generator, or a bounding box)
  preview.png     4-angle render (scripts/p2/view.py renderer)
Shared textures (DDS, mipmapped, scripts/p2/dds.py) live in assets/_textures/ and are referenced by name.

Types:
  tree3d.<species> scripts/p2/trees3d.py prototypes (variant 0; LOD0 near road, LOD1 = the simplified prototype 21 uses
                  beyond TREE_NEAR), bark + leaf-cluster parts; textures tree_trunk / tree_leaves as made by 22
                  (foliage_photo.py crops), taken from the kn5 cache of a seeded quick build dir.
  every id of scripts/p5/assets.py CATALOG (= config/asset_catalog.yaml): ledger props on the shared 'ledger' atlas.
source_hash = sha1 of the generator source files + the catalog entry + the texture bytes it uses; an asset is
rebuilt only when that hash changes (or --force).
"""
import argparse, hashlib, inspect, io, json, pathlib, pickle, sys, time
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "scripts/p5"))
LIB = ROOT / "assets"
TEXD = LIB / "_textures"
QDIR = ROOT / "build_quick"


def sha(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()


def put_texture(name, png_bytes):
    """PNG -> assets/_textures/<stem>.dds (only if changed); returns the relative path"""
    import dds
    TEXD.mkdir(parents=True, exist_ok=True)
    out = TEXD / (pathlib.Path(name).stem + ".dds"); tag = out.with_suffix(".sha1")
    h = sha(png_bytes)
    if not out.exists() or not tag.exists() or tag.read_text() != h:
        out.write_bytes(dds.encode(png_bytes)); tag.write_text(h)
    return str(out.relative_to(LIB)), h


def bbox_proxy(P):
    lo, hi = P.min(0), P.max(0)
    return [dict(shape="box", min=lo.round(3).tolist(), max=hi.round(3).tolist())]


def preview(d, man):
    import view
    view.OUT = d
    p = view.view_asset(man["type"])
    (d / "preview.png").unlink(missing_ok=True)
    Image.open(p).save(d / "preview.png"); p.unlink(); p.with_suffix(".html").unlink(missing_ok=True)


def write_asset(aid, cls, src_hash, sources, parts, collision, frames, desc, lods_extra=None):
    d = LIB / aid; d.mkdir(parents=True, exist_ok=True)
    mparts, nv, nt = [], 0, 0
    for k, (name, arrays, tex, alpha) in enumerate(parts):
        P, N, UV, T = arrays
        f = f"lod0_{name}.npz"
        np.savez_compressed(d / f, P=np.asarray(P, np.float32), N=np.asarray(N, np.float32), UV=np.asarray(UV, np.float32), T=np.asarray(T, np.int32).reshape(-1, 3))
        mparts.append(dict(name=name, mesh=f, texture=tex, material=name, alpha_tested=alpha))
        nv += len(P); nt += len(T)
    lods = [dict(level=0, parts=[p["mesh"] for p in mparts], lod_out=None)]
    for lv, lparts, lod_out in (lods_extra or []):
        fs = []
        for name, (P, N, UV, T) in lparts:
            f = f"lod{lv}_{name}.npz"
            np.savez_compressed(d / f, P=np.asarray(P, np.float32), N=np.asarray(N, np.float32), UV=np.asarray(UV, np.float32), T=np.asarray(T, np.int32).reshape(-1, 3))
            fs.append(f)
        lods.append(dict(level=lv, parts=fs, lod_out=lod_out))
    (d / "collision.json").write_text(json.dumps(collision, indent=1))
    man = dict(type=aid, cls=cls, desc=desc, source_hash=src_hash, sources=sources, parts=mparts, lods=lods,
               collision="collision.json", verts=nv, tris=nt, frames=list(frames or []), built=time.strftime("%Y-%m-%d %H:%M:%S"))
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    try:
        preview(d, man)
    except Exception as e:   # preview is a convenience; the asset itself is complete
        print(f"  preview failed for {aid}: {e}")
    return man


def current_hash(aid):
    m = LIB / aid / "manifest.json"
    return json.loads(m.read_text()).get("source_hash") if m.exists() else None


# ------------------------------------------------------------------ ledger prop types
def ledger_jobs():
    import assets as AS, yaml
    cat = {a["id"]: a for a in yaml.safe_load(open(ROOT / "config/asset_catalog.yaml"))["assets"]}
    src = (ROOT / "scripts/p5/assets.py").read_bytes() + (ROOT / "scripts/p2/roadside_tex.py").read_bytes()
    for aid, e in AS.CATALOG.items():     # signs paint their faces into the shared atlas lazily: complete it first
        try:
            AS.generate(aid, 10.0 if e["linear"] else None, {"seed": 7})
        except Exception:
            pass
    _LEDGER_TEX.append(put_texture("ledger.png", AS.atlas().png()))
    jobs = {}
    for aid, e in AS.CATALOG.items():
        ce = cat.get(aid, {})
        jobs[aid] = (sha(src, _LEDGER_TEX[0][1], json.dumps(ce, sort_keys=True, default=str), repr(e["dims"]), repr(e["proxy"])),
                     lambda aid=aid, e=e, ce=ce: build_ledger(AS, aid, e, ce))
    return jobs


_LEDGER_TEX = []


def build_ledger(AS, aid, e, ce):
    tex, th = _LEDGER_TEX[0]
    L = float(e["dims"][0]) if e["linear"] else None
    m = AS.generate(aid, max(L or 0.0, 10.0) if e["linear"] else None, {"seed": 7})
    P, T, UV, N = m.arrays()
    if not len(P):
        return None
    col = [dict(c) if isinstance(c, dict) else {"prim": list(map(lambda v: v if isinstance(v, (int, float, str)) else str(v), c)) if isinstance(c, (list, tuple)) else str(c)} for c in m.col] or bbox_proxy(P)
    return dict(cls=e["cls"], parts=[("ledger", (P, N, UV, T), tex, True)], collision=dict(collide=bool(e["collide"]), prims=col, note="from the generator; 21d replaces per-prop boxes by its slab convention"),
                frames=ce.get("frames", []), desc=e.get("desc", ""), sources=["scripts/p5/assets.py", "config/asset_catalog.yaml"], tex_hash=th,
                lods=[(1, [("ledger", (P, N, UV, T))], 450.0)])   # low-poly props: LOD1 = LOD0, cut at 21e LOD_OUT


# ------------------------------------------------------------------ tree species
def tree_jobs():
    import trees3d as T3
    st = QDIR / "kn5_cache/state.pkl"
    src = (HERE / "trees3d.py").read_bytes() + (HERE / "foliage_photo.py").read_bytes()
    tex = {}
    if st.exists():
        T = pickle.load(open(st, "rb"))["T"]
        tex = {k: T[k] for k in ("tree_trunk.png", "tree_leaves.png") if k in T}
    th = sha(*[tex[k] for k in sorted(tex)])
    return {f"tree3d.{sp}": (sha(src, sp, th), lambda sp=sp: build_tree(T3, sp, tex)) for sp in T3.SPECIES}


def build_tree(T3, sp, tex):
    if len(tex) < 2:
        sys.exit("assetlib: tree textures come from build_quick/kn5_cache (run `build.py quick-seed` first)")
    tb, _ = put_texture("tree_trunk.png", tex["tree_trunk.png"]); tl, _ = put_texture("tree_leaves.png", tex["tree_leaves.png"])
    p0, p1 = T3.make_proto(sp, 0, 0), T3.make_proto(sp, 0, 1)
    H, R = T3.H0[sp], T3.R0[sp]
    col = [dict(shape="cylinder", r=0.25 * H / 15.0, z0=0.0, z1=min(4.0, 0.3 * H), note="trunk only; 21d convention: no per-tree collision, forest-line slabs")]
    return dict(cls="tree", parts=[("tree_trunk", p0["bark"], tb, False), ("tree_leaves", p0["leaf"], tl, True)], collision=dict(collide=False, prims=col),
                frames=[], desc=f"trees3d prototype '{sp}' variant 0 (of {T3.N_PROTO}), nominal height {H} m, crown radius {R} m; "
                f"instanced at LiDAR positions scaled to height/crown (21_build_geometry.py)",
                sources=["scripts/p2/trees3d.py", "scripts/p2/foliage_photo.py", "22_build_kn5.py tree textures"],
                lods=[(1, [("tree_trunk", p1["bark"]), ("tree_leaves", p1["leaf"])], None)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    jobs = {**tree_jobs(), **ledger_jobs()}
    if a.only:
        jobs = {k: v for k, v in jobs.items() if k in a.only or any(k.startswith(o.rstrip("*")) for o in a.only if o.endswith("*"))}
    t0 = time.time(); nb = ns = 0
    for aid, (h, fn) in sorted(jobs.items()):
        if not a.force and current_hash(aid) == h:
            ns += 1
            continue
        r = fn()
        if r is None:
            print(f"  {aid}: generator returned no geometry (edit-type entry), skipped"); continue
        man = write_asset(aid, r["cls"], h, r["sources"], r["parts"], r["collision"], r["frames"], r["desc"], r["lods"])
        nb += 1
        print(f"  built {aid:32s} {man['verts']:6d} verts {man['tris']:6d} tris")
    idx = {p.parent.name: {k: json.loads(p.read_text())[k] for k in ("cls", "verts", "tris", "source_hash")} for p in sorted(LIB.glob("*/manifest.json"))}
    (LIB / "_index.json").write_text(json.dumps(idx, indent=1))
    print(f"assetlib: {nb} built, {ns} unchanged, {len(idx)} in library, {time.time() - t0:.1f}s")
    if a.list:
        for k, v in idx.items():
            print(f"  {k:32s} {v['cls']:10s} {v['tris']:6d} tris  {v['source_hash'][:10]}")


if __name__ == "__main__":
    main()
