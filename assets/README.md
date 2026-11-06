# assets/ — viewer captures at corners

Rendered from `reports/viewer.html` by `scripts/12_screens.js` (headless Chromium). The data overlays are hidden in these captures. `_drive` is a driver's-eye view about 45 m before the apex, in the uphill direction and right-hand lane. `_orbit` is an oblique overview centred on the apex. The corners are the 12 tightest well-separated bends by smoothed curvature, plus the start and the summit. Corner numbers are not official turn numbers.

| # | station | radius | turns | elevation | driver's eye | overview |
|---|---|---|---|---|---|---|
| corner_01 | 78 m | 19 m | left | 51 m | ![](corner_01_s00078_drive.jpg) | ![](corner_01_s00078_orbit.jpg) |
| corner_02 | 1186 m | 30 m | left | 108 m | ![](corner_02_s01186_drive.jpg) | ![](corner_02_s01186_orbit.jpg) |
| corner_03 | 1906 m | 23 m | left | 167 m | ![](corner_03_s01906_drive.jpg) | ![](corner_03_s01906_orbit.jpg) |
| corner_04 | 3295 m | 28 m | left | 244 m | ![](corner_04_s03294_drive.jpg) | ![](corner_04_s03294_orbit.jpg) |
| corner_05 | 4282 m | 37 m | left | 269 m | ![](corner_05_s04281_drive.jpg) | ![](corner_05_s04281_orbit.jpg) |
| corner_06 | 4974 m | 27 m | left | 298 m | ![](corner_06_s04973_drive.jpg) | ![](corner_06_s04973_orbit.jpg) |
| corner_07 | 7065 m | 31 m | right | 354 m | ![](corner_07_s07064_drive.jpg) | ![](corner_07_s07064_orbit.jpg) |
| corner_08 | 8943 m | 44 m | right | 233 m | ![](corner_08_s08942_drive.jpg) | ![](corner_08_s08942_orbit.jpg) |
| corner_09 | 11205 m | 45 m | left | 170 m | ![](corner_09_s11204_drive.jpg) | ![](corner_09_s11204_orbit.jpg) |
| corner_10 | 12526 m | 26 m | left | 150 m | ![](corner_10_s12525_drive.jpg) | ![](corner_10_s12525_orbit.jpg) |
| corner_11 | 13847 m | 41 m | right | 127 m | ![](corner_11_s13846_drive.jpg) | ![](corner_11_s13846_orbit.jpg) |
| corner_12 | 14582 m | 29 m | right | 122 m | ![](corner_12_s14581_drive.jpg) | ![](corner_12_s14581_orbit.jpg) |
| start | 30 m | — |  | 46 m | ![](start_sr84_s00000_drive.jpg) | ![](start_sr84_s00000_orbit.jpg) |
| summit | 7391 m | — |  | 372 m | ![](summit_s07390_drive.jpg) | ![](summit_s07390_orbit.jpg) |

What the visuals represent: terrain is the LiDAR DTM draped with 1 m USGS imagery, plus CC0 detail textures (ambientCG Grass004, Rock051, Ground037). Trees use real LiDAR positions, heights and crown sizes. They are procedural trunks with alpha leaf cards (LeafSet024), tinted from the imagery at each tree. Roadside grass is ambientCG Foliage001/002 cards on open ground. These are review visuals, not the final AC art.
