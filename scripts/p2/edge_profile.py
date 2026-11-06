"""Per-station, per-side road-edge profile (replaces the constant 0.8 m shoulder).

Sources (all already in data/derived):
  * LiDAR cross-section env_left/env_right (centerline.csv): wall_or_steep_cut / cut_bank / drop_off / steep_drop / ditch / flat
  * GoPro survey A/B (gopro_survey_*.json + gopro_stations_*.csv): left/right_ground material and roadside type per frame
  * roadside_assets_*.json driveway_apron spans: wide dirt / paved aprons

Output arrays (length N, index = 1 m station), attached to geom.Road as attributes:
  sh_w_l, sh_w_r      shoulder width beyond the asphalt edge (m), 0.15 .. 2.6
  sh_mat_l, sh_mat_r  'litter' | 'dirt' | 'grass' | 'ivy' | 'paved'
  ditch_l, ditch_r    depth (m) of a shallow V at the shoulder mid column (0 = none)
  line_in_l, line_in_r  white edge-line centre inset from the asphalt edge (m), 0.15 .. 0.8
  fresh               bool, station inside a fresh_overlay span (config/surface_overrides.yaml)
  strip_in_l, strip_in_r  how far the ragged edge / debris strip reaches onto the asphalt (m), 0.12 .. 0.9
Variation: sums of sines at 5-40 m wavelengths (deterministic per side), abrupt changes at env / survey / apron
boundaries (only a 0.8 m Gaussian, so a 1.5 m change happens over ~3 m).
"""
import csv, glob, json, pathlib
import numpy as np
from scipy.ndimage import gaussian_filter1d

DER = pathlib.Path(__file__).resolve().parents[2] / "data/derived"
MATS = ("litter", "dirt", "grass", "ivy", "paved")
SURVEY_MAT = {"leaf_litter": "litter", "dirt_gravel": "dirt", "green_grass": "grass", "ivy_shrub": "ivy", "paved": "paved"}
# env -> (min width, max width) of the shoulder (GoPro: rock/cut faces leave 0.1-0.4 m; open flats 0.6-1.8 m)
ENV_W = {"wall_or_steep_cut": (0.15, 0.40), "cut_bank": (0.25, 0.95), "drop_off": (0.35, 1.05), "steep_drop": (0.20, 0.60),
         "ditch": (0.90, 1.50), "flat": (0.55, 1.80)}
W_MIN, W_MAX = 0.15, 2.6


def _noise(s, seed, waves=(5.3, 11.7, 23.0, 39.0), amps=(0.18, 0.28, 0.32, 0.22)):
    """smooth pseudo-random field in [0, 1] along s"""
    r = np.random.default_rng(seed)
    v = sum(a * np.sin(2 * np.pi * s / w + r.uniform(0, 2 * np.pi)) for w, a in zip(waves, amps))
    return np.clip(0.5 + v / (2 * sum(amps)) * 1.6, 0, 1)


def _survey():
    st = {}
    for f in sorted(glob.glob(str(DER / "gopro_stations_*.csv"))):
        for r in csv.DictReader(open(f)):
            st[int(r["frame"])] = float(r["s_m"])
    rows = []
    for f in sorted(glob.glob(str(DER / "gopro_survey_*.json"))):
        for r in json.load(open(f)):
            if r["frame"] in st:
                rows.append((st[r["frame"]], r))
    rows.sort(key=lambda t: t[0])
    return np.array([t[0] for t in rows]), [t[1] for t in rows]


def _aprons():
    out = []
    for f in sorted(glob.glob(str(DER / "roadside_assets_*.json"))):
        for e in json.load(open(f)):
            if e["type"] == "driveway_apron":
                a = e.get("attrs") or {}
                txt = (str(a.get("material", "")) + " " + str(a.get("surface", ""))).lower()
                mat = "paved" if ("paved" in txt and "no paving" not in txt and "gravel" not in txt) else "dirt"
                for sd in (("left", "right") if e["side"] == "both" else (e["side"],)):
                    out.append((float(e["s0"]), float(e["s1"]), sd, mat))
    return out


def fresh_spans():
    """[(s0, s1)] freshly repaved spans from config/surface_overrides.yaml"""
    import yaml
    f = DER.parents[1] / "config/surface_overrides.yaml"
    d = (yaml.safe_load(open(f)) or {}) if f.exists() else {}
    return [(float(e["s0"]), float(e["s1"])) for e in d.get("fresh_overlay", [])]


def compute(road):
    s = road.s; N = len(s)
    fresh = np.zeros(N, bool)
    for a0, a1 in fresh_spans():
        fresh |= (s >= a0) & (s <= a1)
    ss, rows = _survey()
    aprons = _aprons()
    out = {}
    for side, sgn, seed in (("left", 1, 101), ("right", -1, 202)):
        env = road.env_l if sgn > 0 else road.env_r
        e = road.el if sgn > 0 else road.er
        # ---- survey material / roadside type at each station: nearest frame, boundary jittered +-8 m
        jit = 8 * (2 * _noise(s, seed + 7, (17.0, 41.0), (1, 1)) - 1)
        j = np.clip(np.searchsorted(ss, s + jit), 1, len(ss) - 1)
        j = np.where(np.abs(ss[j - 1] - (s + jit)) < np.abs(ss[j] - (s + jit)), j - 1, j)
        mat = np.array([SURVEY_MAT.get(rows[k][f"{side}_ground"], "litter") for k in j], dtype=object)
        rtype = np.array([rows[k][side] for k in j], dtype=object)
        # ---- width: env range x smooth noise
        lo = np.array([ENV_W.get(v, (0.4, 1.0))[0] for v in env]); hi = np.array([ENV_W.get(v, (0.4, 1.0))[1] for v in env])
        u = _noise(s, seed)
        w = lo + (hi - lo) * u
        open_ = np.isin(rtype, ["pasture_open", "flat_grass"]) & np.isin(env, ["flat", "drop_off", "ditch"])
        w = np.where(open_, w + 0.35 + 0.5 * _noise(s, seed + 3), w)               # grass verge in pasture
        w = np.where(np.isin(rtype, ["driveway_yard"]) & (env != "wall_or_steep_cut"), w + 0.3 * _noise(s, seed + 5), w)
        # occasional gravel turnout bulges on flat / drop-off ground (abrupt in, abrupt out)
        bump = _noise(s, seed + 11, (60.0, 97.0, 31.0), (1, 0.7, 0.4)) > 0.86
        bump &= np.isin(env, ["flat", "drop_off"])
        w = np.where(bump, np.maximum(w, 1.6 + 0.8 * _noise(s, seed + 13, (9.0, 4.7), (1, 1))), w)
        mat = np.where(bump & (mat != "grass"), "dirt", mat)
        # paved survey ground without an apron = thin paved/gravel edge: treat as gravel shoulder
        mat = np.where(mat == "paved", "dirt", mat)
        # rock / steep cut faces: the shoulder is a little gravel wedge at the toe
        mat = np.where((env == "wall_or_steep_cut") & (mat == "grass"), "dirt", mat)
        # ---- driveway aprons (survey assets): 1.8-2.5 m, tapered 3 m each end
        for a0, a1, sd, am in aprons:
            if sd != side:
                continue
            m = (s >= a0 - 3) & (s <= a1 + 3)
            if not m.any():
                continue
            core = (s >= a0) & (s <= a1)
            tap = np.clip(1 - np.maximum(a0 - s, s - a1) / 3.0, 0, 1)
            wa = 1.8 + 0.7 * _noise(s, seed + 17, (7.0, 13.0), (1, 1))
            w = np.where(m, np.maximum(w, w + (wa - w) * tap), w)
            mat = np.where(core, am, mat)
        # ---- inside of tight corners: keep the offset edge from folding (e + w < 0.7 R)
        kin = np.maximum(sgn * road.k, 0)
        w = np.minimum(w, np.maximum(0.7 / np.maximum(kin, 1e-6) - e, W_MIN))
        w = np.clip(gaussian_filter1d(w, 0.8, mode="nearest"), W_MIN, W_MAX)
        # ---- material runs shorter than 4 m merge into their predecessor
        mat = mat.astype(str)
        k0 = 0
        for k in range(1, N + 1):
            if k == N or mat[k] != mat[k0]:
                if k - k0 < 4 and k0 > 0:
                    mat[k0:k] = mat[k0 - 1]
                k0 = k
        # ---- ditch: shallow V at the mid column where the LiDAR says ditch; a slight gutter at cut banks
        dz = np.where(env == "ditch", 0.14 + 0.10 * _noise(s, seed + 19), 0.0)
        dz = np.where((env == "cut_bank") & (w > 0.6) & (mat != "paved"), 0.05 + 0.05 * _noise(s, seed + 23), dz)
        dz = np.where(mat == "paved", 0.0, gaussian_filter1d(dz, 1.5, mode="nearest"))
        dz = np.minimum(dz, 0.25 * w)        # keep the V slopes < ~1:2
        # ---- white edge line inset 0.15-0.8 m (near the edge at walls, often well inside on open sections)
        li = 0.15 + 0.6 * _noise(s, seed + 29, (21.0, 47.0, 83.0), (1, 0.8, 0.6))
        li = np.where(env == "wall_or_steep_cut", np.minimum(li, 0.3), li)
        # ---- ragged edge / debris reach onto the asphalt
        si = 0.12 + 0.35 * _noise(s, seed + 31, (4.1, 9.7, 18.0), (1, 0.8, 0.6))
        wash = _noise(s, seed + 37, (13.0, 29.0, 7.0), (1, 0.8, 0.5)) > 0.8                # wash-out patches
        si = np.where(wash & np.isin(mat, ["dirt", "litter"]), si + 0.4, si)
        si = np.where(mat == "grass", si * 0.6, si)
        si = np.minimum(gaussian_filter1d(si, 1.0, mode="nearest"), np.maximum(li - 0.02, 0.12) + 0.5)
        # freshly repaved spans: crisp pavement edge, clean gravel / dirt backfill (grass verges kept), no wash-out
        mat = np.where(fresh & (mat != "grass") & (mat != "paved"), "dirt", mat)
        si = np.where(fresh, 0.12, si)
        k = side[0]
        out[f"sh_w_{k}"] = w; out[f"sh_mat_{k}"] = mat; out[f"ditch_{k}"] = dz
        out[f"line_in_{k}"] = li; out[f"strip_in_{k}"] = si
    out["fresh"] = fresh
    return out
