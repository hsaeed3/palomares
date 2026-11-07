#!/usr/bin/env python3
"""dev7 D-08: reports/dev7/data_report.md + coverage map PNG (LiDAR work unit per 25 m cell, ground density, gaps,
Mapillary images). Reads only dev7 derived products (and the node cache for the work-unit map)."""
import glob, json, csv, collections
import numpy as np, rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common7 import *

T = json.load(open(DER7 / "termini.json")); SUM = json.load(open(DER7 / "centerlines_summary.json"))
SC = json.load(open(DER7 / "scope.json")); CC = json.load(open(CACHE7 / "class_counts.json"))
F = json.load(open(DER7 / "features.geojson"))["features"]
gaps = list(csv.DictReader(open(DER7 / "gaps.csv")))
mi = list(csv.DictReader(open(DER7 / "mapillary_index.csv")))
mc = list(csv.DictReader(open(DER7 / "mapillary_coverage.csv")))
nc = np.genfromtxt(DER7 / "centerline_niles_canyon_road.csv", delimiter=",", names=True, dtype=None, encoding=None)
# ---- work-unit / density map ----
g = json.load(open(CACHE7 / "grid.json")); CS = 25.0
nx, ny = int(g["w"] / CS) + 1, int(g["h"] / CS) + 1
wu = np.zeros((3, ny, nx), np.int64)
for f in glob.glob(str(CACHE7 / "nodes" / "*.npz")):
    k = [i for i, d in enumerate(EPT_SETS) if f.split("/")[-1].startswith(d)][0]
    d = np.load(f)
    if len(d["gx"]) == 0: continue
    c = ((d["gx"] - g["x0_local"]) / CS).astype(int); r = ((g["y1_local"] - d["gy"]) / CS).astype(int)
    np.add.at(wu[k], (r.clip(0, ny - 1), c.clip(0, nx - 1)), 1)
dens = wu.sum(0) / CS ** 2
dom = np.where(wu.sum(0) > 0, wu.argmax(0), -1)
share = {EPT_SETS[k]: round(float(wu[k].sum() / wu.sum() * 100), 1) for k in range(3)}
fig, ax = plt.subplots(2, 1, figsize=(14, 8.5))
ext = [g["x0_local"], g["x0_local"] + nx * CS, g["y1_local"] - ny * CS, g["y1_local"]]
cm = matplotlib.colors.ListedColormap(["#4c78a8", "#f58518", "#54a24b"])
ax[0].imshow(np.ma.masked_less(dom, 0), cmap=cm, vmin=-.5, vmax=2.5, extent=ext, interpolation="nearest")
ax[0].set_title("Dominant 2021 work unit per 25 m cell (blue=CA_AlamedaCo_1, orange=_2, green=_3); all QL1")
im = ax[1].imshow(np.ma.masked_equal(dens, 0), cmap="viridis", vmin=0, vmax=12, extent=ext, interpolation="nearest")
plt.colorbar(im, ax=ax[1], fraction=.02, label="ground+bridge returns / m^2")
ax[1].set_title("Ground-return density; red x = gaps > 5 m on a centerline; white dots = Mapillary images")
for a in ax:
    for f in glob.glob(str(DER7 / "centerline_*.csv")):
        c = np.genfromtxt(f, delimiter=",", names=True, dtype=None, encoding=None); a.plot(c["x_local"], c["y_local"], "k-", lw=.5)
    a.set_aspect("equal"); a.set_xlabel("local x (m)"); a.set_ylabel("local y (m)")
mx, my = ll2loc(np.array([float(r["lon"]) for r in mi]), np.array([float(r["lat"]) for r in mi]))
ax[1].plot(mx, my, ".", color="w", ms=.6, alpha=.5)
if gaps:
    gx, gy = ll2loc(np.array([float(r["lon"]) for r in gaps]), np.array([float(r["lat"]) for r in gaps])); ax[1].plot(gx, gy, "rx", ms=5)
plt.tight_layout(); plt.savefig(REP7 / "lidar_coverage.png", dpi=100); plt.close()
# ---- canyon stats ----
s, z, gr = nc["s_m"], nc["z_navd88"], nc["grade"]
g25 = np.convolve(gr, np.ones(25) / 25, mode="same")
W = nc["width_m"]
# ---- report ----
L = []
w = L.append
w("# dev7 data report (work package D)\n")
w("Scope: Niles town streets, Niles Canyon Rd (SR-84) from Mission Blvd to the Sunol stop, and the connector to the existing Palomares Rd south end. "
  "All products are in `data/derived/dev7/`. Coordinates use UTM 10N (EPSG:26910) and NAVD88 m, with the same fixed local origin as Phase 1 (`config/origin.json`, x0=%.1f, y0=%.1f). "
  "Pipeline: `scripts/p7/01..08` (numbered, idempotent).\n" % (X0, Y0))
w("## Termini (confirmed in OSM)\n")
t = T["niles_canyon_road"]
w(f"| Terminus | lon, lat | OSM node | Notes |\n|---|---|---|---|")
w(f"| NCR west: Mission Blvd | {t['west_mission_blvd']['lonlat'][0]:.6f}, {t['west_mission_blvd']['lonlat'][1]:.6f} | {t['west_mission_blvd']['osm_node']} | {t['west_mission_blvd']['desc']} |")
w(f"| NCR east: Sunol | {t['east_sunol']['lonlat'][0]:.6f}, {t['east_sunol']['lonlat'][1]:.6f} | {t['east_sunol']['osm_node']} | {t['east_sunol']['desc']} |")
nb = T["niles_blvd_start"]
w(f"| Niles Blvd start | {nb['lonlat'][0]:.6f}, {nb['lonlat'][1]:.6f} | {nb['osm_node']} | Niles Blvd / {', '.join(nb['cross_streets'])}: {nb['desc']} |")
w(f"| Devout Coffee | {nb['devout_coffee']['lonlat'][0]:.6f}, {nb['devout_coffee']['lonlat'][1]:.6f} | {nb['devout_coffee']['osm_node']} | OSM amenity=cafe, {nb['devout_coffee']['s_along_niles_blvd_m']:.0f} m along the Niles Blvd centerline from the start |")
p = T["palomares_south_end"]
w(f"| Palomares Rd south end | {p['lonlat'][0]:.6f}, {p['lonlat'][1]:.6f} | - | on NCR at OSM station {p['ncr_station_m']:.0f} m (distance {p['dist_to_nearest_ncr_m']} m); refined NCR station in features (junction) |\n")
w("The Sunol \"traffic light / stop sign\" is the all-way STOP at Niles Canyon Rd / Main St / Pleasanton-Sunol Rd. OSM has stop nodes there and no signal. "
  "The nearest traffic signals are on Calaveras Rd / I-680, about 1.3 km east, outside the scope. SR-84 continues east as Paloma Rd, and the first 345 m are included as a stub.\n")
w("## Lengths (refined centerlines, 1 m)\n")
w("| Road | Role | Length (m) | Width p5/p50/p95 (m) | z min-max (m) | max abs grade | snap vs OSM p50/p95 (m) |\n|---|---|---|---|---|---|---|")
roles = {k: v.get("role", "") for k, v in T["roads"].items()}
tot = collections.defaultdict(float)
for k in sorted(SUM, key=lambda k: -SUM[k]["length_m"]):
    v = SUM[k]; rl = roles.get(k) or ("canyon" if k == "niles_canyon_road" else "town")
    tot[rl] += v["length_m"]
    w(f"| {k} | {rl} | {v['length_m']:.0f} | {'/'.join(f'{x:.1f}' for x in v['width_p5_50_95'])} | {v['z_min']:.1f}-{v['z_max']:.1f} | {v['max_abs_grade']*100:.1f}% | {v['snap_p50_p95'][0]:.1f}/{v['snap_p50_p95'][1]:.1f} |")
w("\nTotals by role: " + ", ".join(f"{k} {v/1000:.2f} km" for k, v in tot.items()) + f"; all roads {sum(tot.values())/1000:.2f} km.\n")
w("Mission Blvd is split into its OSM carriageway pieces (`mission_boulevard`, `_2`..`_4`). Roads under 40 m are smoothed only and not snapped. "
  "Widths come from DTM flat-strip growth (6 cm tolerance), with curb steps detected on town streets. They are capped by OSM lanes (lanes x 1.9 m + 1.6 m per side), so treat town widths with curbs and parked cars as approximate.\n")
w("## Niles Canyon Rd profile (Mission Blvd to Sunol)\n")
w("![profile](profile_niles_canyon_road.png)\n")
jhi, jlo = int(np.nanargmax(z)), int(np.nanargmin(z))
w(f"- Length {s[-1]:.0f} m. Start z {z[0]:.1f} m, end z {z[-1]:.1f} m, net {z[-1]-z[0]:+.1f} m.")
w(f"- Highest point {z[jhi]:.1f} m at s={s[jhi]:.0f} m; lowest {z[jlo]:.1f} m at s={s[jlo]:.0f} m. The canyon climbs gently eastward along Alameda Creek, with no summit in the hill-climb sense. The \"climb\" is the creek gradient.")
w(f"- Grade (25 m mean): p50 {np.percentile(g25,50)*100:+.2f}%, p5 {np.percentile(g25,5)*100:+.2f}%, p95 {np.percentile(g25,95)*100:+.2f}%, max abs {np.abs(g25).max()*100:.1f}% (s={s[np.abs(g25).argmax()]:.0f} m). Share of length > 4%: {(np.abs(g25)>0.04).mean()*100:.1f}%.")
w(f"- Total ascent {np.clip(np.diff(z),0,None).sum():.0f} m, descent {np.clip(-np.diff(z),0,None).sum():.0f} m (W to E, on the smoothed profile).")
w(f"- Width p5/p50/p95 {np.percentile(W,5):.1f}/{np.percentile(W,50):.1f}/{np.percentile(W,95):.1f} m. Curvature: tightest radius {1/np.abs(np.convolve(nc['curvature_1pm'],np.ones(9)/9,'same')).max():.0f} m.")
cr = [e["properties"] for e in F if e["properties"].get("road") == "niles_canyon_road"]
for kind in ("rail_underpass", "road_over_rail", "rail_road_crossing", "bridge_osm", "bridge_lidar_class17", "culvert_candidate", "stream_crossing_bridge", "traffic_signals", "stop"):
    xs = [q for q in cr if q["kind"] == kind]
    if xs:
        def fmt(q):
            st = q.get("s_m", q.get("s_start_m")); en = q.get("s_end_m")
            ex = f" (DSM {q['dsm_over_road_m']} m over road)" if "dsm_over_road_m" in q else ""
            nm = f" {q.get('rail') or q.get('name') or q.get('stream') or ''}".rstrip()
            return f"{st:.0f}" + (f"-{en:.0f}" if en is not None else "") + nm + ex
        w(f"- **{kind}** ({len(xs)}): " + "; ".join(fmt(q) for q in sorted(xs, key=lambda q: q.get("s_m", q.get("s_start_m", 0)))) )
w("")
w("## Features (all roads)\n")
cnt = collections.Counter((e["properties"]["kind"]) for e in F)
w("| kind | count |\n|---|---|")
for k, v in sorted(cnt.items()): w(f"| {k} | {v} |")
w("\nThe railway underpass(es) on SR-84 are where the Niles Canyon Railway / UP lines cross over the road. `dsm_over_road_m` is the first-surface height above the road surface, which is the top of the structure, not the clearance. "
  "The flashing warning signs before the underpass are not in OSM and LiDAR can't resolve them. Package M must place them from Mapillary.\n")
w("## LiDAR coverage and quality\n")
w("![coverage](lidar_coverage.png)\n")
w(f"- Source: USGS 3DEP EPT `CA_AlamedaCo_1/2/3_2021` (QL1, the same sets as Phase 1), streamed and reprojected EPSG:3857 to UTM 10N. Share of ground returns by work unit: {share}.")
w(f"- Classes present: {CC['class_counts']}. There is no 3-6 vegetation or building classification, so trees and buildings come from the CHM (DSM - DTM). Class 17 (bridge deck) is present ({CC['class_counts'].get('17', 0):,} points) and is used as drivable ground.")
w(f"- Ground returns (deduplicated): {CC['ground_pts_dedup']:,}. Areas: town polygon {SC['areas_km2']['town']} km2, 40 m detail {SC['areas_km2']['detail']} km2, 150 m terrain {SC['areas_km2']['terrain']} km2.")
w("- Rasters: `dtm_r050_b40` (0.5 m, detail area), `dtm_r100_b150` (1 m), `dsm_r100_b150`, `chm_r100_b150`, `gdist_r050_b40` (gap diagnostic), `int_r050_b40` (intensity), `cover_r100_b150` (first/ground/bridge counts).\n")
w("## Gaps and uncertain sections\n")
if gaps:
    w("Ground-return gaps longer than 5 m on a centerline (gap length = run with no return within 1 m, plus 2 x max distance):\n")
    w("| road | s start-end (m) | gap (m) | canopy | bridge pts | likely cause |\n|---|---|---|---|---|---|")
    for r in gaps:
        cause = "bridge/structure" if int(r["bridge17"]) > 0 else ("dense canopy" if float(r["canopy_frac"]) > .5 else "structure/vehicle/water or overpass")
        w(f"| {r['road']} | {float(r['s_start_m']):.0f}-{float(r['s_end_m']):.0f} | {float(r['gap_len_m']):.1f} | {float(r['canopy_frac']):.2f} | {r['bridge17']} | {cause} |")
else:
    w("No ground-return gaps over 5 m on any centerline.")
w("")
w("## Mapillary coverage (index only, CC-BY-SA 4.0)\n")
yrs = collections.Counter(r["captured_at"][:4] for r in mi)
w(f"{len(mi):,} images within 40 m of a dev7 road (`mapillary_index.csv`). Years: {dict(sorted(yrs.items()))}. Panoramas: {sum(int(r['is_pano']) for r in mi)}.\n")
w("| road | images | 100 m bins covered | newest |\n|---|---|---|---|")
by = collections.defaultdict(list)
for r in mc: by[r["road"]].append(r)
for k, rows in sorted(by.items(), key=lambda kv: -sum(int(r["n_images"]) for r in kv[1])):
    n = sum(int(r["n_images"]) for r in rows); cov = sum(int(r["n_images"]) > 0 for r in rows); yr = max(int(r["newest_year"]) for r in rows)
    w(f"| {k} | {n} | {cov}/{len(rows)} | {yr or '-'} |")
w("\nNiles Canyon Rd has full coverage in both directions. In the town, Niles Blvd and Mission Blvd are covered, but most side streets (2nd St, 3rd St, G/H/I/J St) have no Mapillary images. Package M will have to use NAIP or LiDAR for those, or mark them unverified.\n")
w("## Float precision\n")
w("See `docs/dev7/DECISIONS.md` (D7-1). The farthest dev7 point is about 10.1 km from the fixed origin (the Sunol stub), and Niles town is about 8.6 km away.\n")
w("## Attribution\n\nOpenStreetMap contributors (ODbL 1.0); USGS 3DEP (public domain); Mapillary contributors (CC-BY-SA 4.0, index only).")
(REP7 / "data_report.md").write_text("\n".join(L) + "\n")
print("report written", len(L), "lines; work-unit share", share)
