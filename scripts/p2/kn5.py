"""Minimal KN5 (v5) writer + reader/validator.

Layout follows moppius/blender-assetto-corsa-tools (GPL, format reference only):
header b"sc6969" + uint32 version(5); textures; materials; node tree.
Coordinates handed to Mesh are AC space (Y up). Use ac_from_local() to convert
(east, north, up) -> (x, up, -north), same as the reference exporter's convert_vector3.
Triangle winding: pass CCW-seen-from-the-front in local (ENU) space; the reference
exporter keeps Blender's cyclic order, so we do too.
"""
import struct, io
import numpy as np

MAGIC = b"sc6969"
IDENT = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def ac_from_local(p):
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


class Material:
    def __init__(self, name, shader="ksPerPixel", props=None, textures=None, blend=0, alpha_tested=False, depth=0):
        self.name, self.shader, self.blend, self.alpha_tested, self.depth = name, shader, blend, alpha_tested, depth
        self.props = props or {"ksAmbient": 0.4, "ksDiffuse": 0.4, "ksSpecular": 0.0, "ksSpecularEXP": 1.0, "ksEmissive": (0, 0, 0), "ksAlphaRef": 0.0}
        self.textures = textures or {}  # sampler -> texture name


class Mesh:
    def __init__(self, name, pos, nrm, uv, idx, material, cast_shadows=True, visible=True, transparent=False,
                 renderable=True, lod_in=0.0, lod_out=0.0, layer=0):
        self.name = name
        self.pos = np.asarray(pos, np.float32); self.nrm = np.asarray(nrm, np.float32)
        self.uv = np.asarray(uv, np.float32); self.idx = np.asarray(idx, np.int64).reshape(-1)
        self.material = material
        self.cast_shadows, self.visible, self.transparent, self.renderable = cast_shadows, visible, transparent, renderable
        self.lod_in, self.lod_out, self.layer = lod_in, lod_out, layer


class Dummy:
    """Empty/locator node with a full transform (row-major, translation in last row)."""
    def __init__(self, name, matrix=None, children=None):
        self.name, self.matrix, self.children = name, matrix or IDENT, children or []


def _s(f, x):
    b = x.encode("utf-8"); f.write(struct.pack("<I", len(b))); f.write(b)


def split_mesh(m, limit=65535):
    """Split by triangles so each piece has <= limit unique vertices."""
    tris = m.idx.reshape(-1, 3)
    if len(m.pos) <= limit:
        return [m]
    out, start, k = [], 0, 0
    while start < len(tris):
        used, end = {}, start
        while end < len(tris):
            new = [v for v in tris[end] if v not in used]
            if len(used) + len(new) > limit:
                break
            for v in new:
                used[v] = len(used)
            end += 1
        sel = np.fromiter(used.keys(), np.int64)
        remap = np.full(len(m.pos), -1, np.int64); remap[sel] = np.arange(len(sel))
        out.append(Mesh(f"{m.name}_{k}" if False else m.name, m.pos[sel], m.nrm[sel], m.uv[sel], remap[tris[start:end]].reshape(-1), m.material,
                        m.cast_shadows, m.visible, m.transparent, m.renderable, m.lod_in, m.lod_out, m.layer))
        start, k = end, k + 1
    return out


TEXTURES_AS_DDS = True     # dev6: every embedded texture is DDS (BC1/BC3) with mipmaps, power-of-two (scripts/p2/dds.py)
_DDS_CACHE = {}


def _dds_textures(materials, textures):
    import hashlib, dds
    ren, out = {}, {}
    for n, data in textures.items():
        if data[:4] == b"DDS ":
            out[n] = data; continue
        nn = n.rsplit(".", 1)[0] + ".dds"
        h = hashlib.sha1(data).hexdigest()
        if h not in _DDS_CACHE:
            _DDS_CACHE[h] = dds.encode(data)
        out[nn] = _DDS_CACHE[h]; ren[n] = nn
    import copy
    mats = []
    for m in materials:          # copies: callers reuse their Material objects (with .png names) for other kn5s
        c = copy.copy(m); c.textures = {k: ren.get(v, v) for k, v in m.textures.items()}; mats.append(c)
    return mats, out


def clean_normals_tangents(pos, nrm, uv, idx):
    """unit vertex normals (zero / NaN normals rebuilt from the faces, else up) and UV-derived tangents
    orthogonalised against the normal (dev1-5 wrote tangent = +X everywhere: parallel to the normal on every
    east-facing wall/bank, a degenerate TBN that yields NaN/black or light-leaking normal-mapped shading)."""
    pos = pos.astype(np.float64); n = nrm.astype(np.float64); T = idx.reshape(-1, 3)
    n[~np.isfinite(n)] = 0
    ln = np.linalg.norm(n, axis=1)
    bad = (ln < 0.5) | (ln > 1.5)
    e1 = pos[T[:, 1]] - pos[T[:, 0]]; e2 = pos[T[:, 2]] - pos[T[:, 0]]
    if bad.any():
        fn = np.cross(e1, e2); acc = np.zeros_like(pos)
        for k in range(3):
            np.add.at(acc, T[:, k], fn)
        n[bad] = acc[bad]
        ln = np.linalg.norm(n, axis=1); z = ln < 1e-9
        n[z] = (0, 1, 0); ln[z] = 1
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    du1 = uv[T[:, 1]] - uv[T[:, 0]]; du2 = uv[T[:, 2]] - uv[T[:, 0]]
    r = du1[:, 0] * du2[:, 1] - du2[:, 0] * du1[:, 1]
    ok = np.abs(r) > 1e-12
    tf = np.zeros_like(e1)
    tf[ok] = (e1[ok] * du2[ok, 1:2] - e2[ok] * du1[ok, 1:2]) / r[ok, None]
    tacc = np.zeros_like(pos)
    for k in range(3):
        np.add.at(tacc, T[:, k], tf)
    t = tacc - n * (tacc * n).sum(1, keepdims=True)
    lt = np.linalg.norm(t, axis=1); w = lt < 1e-6
    if w.any():   # no usable UV gradient: any unit vector perpendicular to the normal
        ref = np.where(np.abs(n[w, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
        t[w] = ref - n[w] * (ref * n[w]).sum(1, keepdims=True); lt[w] = np.linalg.norm(t[w], axis=1)
    t /= lt[:, None]
    return n.astype(np.float32), t.astype(np.float32)


def write_kn5(path, materials, textures, nodes):
    """textures: dict name->bytes (PNG/DDS). materials: list[Material]. nodes: list of Mesh/Dummy (top-level)."""
    if TEXTURES_AS_DDS:
        materials, textures = _dds_textures(materials, textures)
    matid = {m.name: i for i, m in enumerate(materials)}
    f = io.BytesIO()
    f.write(MAGIC); f.write(struct.pack("<I", 5))
    f.write(struct.pack("<i", len(textures)))
    for n, data in textures.items():
        f.write(struct.pack("<i", 1)); _s(f, n); f.write(struct.pack("<I", len(data))); f.write(data)
    f.write(struct.pack("<i", len(materials)))
    for m in materials:
        _s(f, m.name); _s(f, m.shader); f.write(struct.pack("<B?i", m.blend, m.alpha_tested, m.depth))
        f.write(struct.pack("<I", len(m.props)))
        for k, v in m.props.items():
            a = [0.0] * 10
            if isinstance(v, (tuple, list)):
                a[3:3 + len(v)] = v if len(v) == 3 else a[3:6]
                if len(v) == 2: a[1:3] = v
                if len(v) == 4: a[6:10] = v
            else:
                a[0] = float(v)
            _s(f, k); f.write(struct.pack("<10f", *a))
        f.write(struct.pack("<I", len(m.textures)))
        for slot, (samp, tex) in enumerate(m.textures.items()):
            _s(f, samp); f.write(struct.pack("<I", slot)); _s(f, tex)
    flat = []
    for n in nodes:
        flat.extend(split_mesh(n) if isinstance(n, Mesh) else [n])

    def wnode(n):
        if isinstance(n, Dummy):
            f.write(struct.pack("<I", 1)); _s(f, n.name); f.write(struct.pack("<I?", len(n.children), True))
            f.write(struct.pack("<16f", *n.matrix))
            for c in n.children: wnode(c)
            return
        assert len(n.pos) <= 65535, n.name
        f.write(struct.pack("<I", 2)); _s(f, n.name); f.write(struct.pack("<I?", 0, True))
        f.write(struct.pack("<???", n.cast_shadows, n.visible, n.transparent))
        nv = len(n.pos)
        assert np.isfinite(n.pos).all() and np.isfinite(n.uv).all(), f"non-finite vertex in {n.name}"
        nrm, tan = clean_normals_tangents(n.pos, n.nrm, n.uv * np.array([1, -1], np.float32), n.idx)
        vb = np.concatenate([n.pos, nrm, n.uv * np.array([1, -1], np.float32), tan], axis=1).astype("<f4")
        f.write(struct.pack("<I", nv)); f.write(vb.tobytes())
        f.write(struct.pack("<I", len(n.idx))); f.write(n.idx.astype("<u2").tobytes())
        c = (n.pos.min(0) + n.pos.max(0)) / 2; r = float(np.linalg.norm(n.pos - c, axis=1).max())
        f.write(struct.pack("<IIff", matid[n.material], n.layer, n.lod_in, n.lod_out))
        f.write(struct.pack("<4f?", *c, r, n.renderable))

    root = Dummy("palomares_root", children=flat)
    wnode(root)
    with open(path, "wb") as out:
        out.write(f.getvalue())


def read_kn5(path):
    b = open(path, "rb").read(); o = [0]

    def rd(fmt):
        v = struct.unpack_from("<" + fmt, b, o[0]); o[0] += struct.calcsize("<" + fmt); return v

    def rs():
        (n,) = rd("I"); s = b[o[0]:o[0] + n].decode("utf-8"); o[0] += n; return s
    assert b[:6] == MAGIC; o[0] = 6
    (ver,) = rd("I")
    if ver > 5: rd("I")
    tex = {}
    (nt,) = rd("i")
    for _ in range(nt):
        rd("i"); name = rs(); (sz,) = rd("I"); tex[name] = b[o[0]:o[0] + sz]; o[0] += sz
    mats = []
    (nm,) = rd("i")
    for _ in range(nm):
        name = rs(); shader = rs(); blend, at = rd("B?")
        rd("i"); (np_,) = rd("I")
        props = {}
        for _ in range(np_):
            k = rs(); props[k] = rd("10f")
        (ns,) = rd("I"); samp = {}
        for _ in range(ns):
            sn = rs(); rd("I"); samp[sn] = rs()
        mats.append(dict(name=name, shader=shader, blend=blend, alpha_tested=at, props=props, textures=samp))
    meshes, dummies = [], []

    def rnode():
        (cls,) = rd("I"); name = rs(); nch, _act = rd("I?")
        if cls == 1:
            m = rd("16f"); dummies.append((name, m))
        else:
            fb = b[o[0]:o[0] + 3]; cs, vis, tr = rd("???"); (nv,) = rd("I")
            vb = np.frombuffer(b, "<f4", nv * 11, o[0]).reshape(nv, 11); o[0] += nv * 44
            (ni,) = rd("I"); ib = np.frombuffer(b, "<u2", ni, o[0]); o[0] += ni * 2
            mid, layer, li, lo = rd("IIff"); rb = b[o[0] + 16:o[0] + 17]; bs = rd("4f?")
            meshes.append(dict(name=name, active=_act, flag_bytes=fb, renderable_byte=rb, layer=layer, lod_in=li, cast_shadows=cs, visible=vis, transparent=tr, renderable=bs[4], pos=vb[:, :3].copy(), nrm=vb[:, 3:6].copy(), uv=vb[:, 6:8].copy(), tan=vb[:, 8:11].copy(), idx=ib.astype(np.int64), material=mid, lod_out=lo))
        for _ in range(nch): rnode()
    rnode()
    assert o[0] == len(b), f"trailing bytes {len(b) - o[0]}"
    return dict(version=ver, textures=tex, materials=mats, meshes=meshes, dummies=dummies)
