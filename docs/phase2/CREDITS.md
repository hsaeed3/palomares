# Credits and asset sources: race kerbs and track dressing (track_* layouts)

The kerb, run-off, guardrail and dressing art is **procedural**: numpy and PIL code in this repo. It uses no downloaded or ripped assets.

| Asset | Source | Script |
|---|---|---|
| Kerb atlas (apex ridged and exit flat red/white, wear, rubber, dirt) and kerb normal map | procedural | `scripts/p2/race_tex.py` `kerb_atlas()` |
| Green painted run-off | procedural | `race_tex.py` `runoff_tex()` |
| Galvanised W-beam guardrail texture | procedural | `race_tex.py` `guardrail_wbeam()` |
| Dressing atlas: banner, sponsor boards, conveyor belt, tyres, TecPro-style blocks, concrete, catch fence, distance boards, marshal and flag tiles, start lights | procedural (PIL drawing) | `scripts/p2/21b_track_dressing.py` |
| Font | DejaVu Sans Bold (Bitstream Vera licence, free to embed) | system font |

**Branding is fictional.** "Palomares Hill Climb" (PHC) and its roundel, plus the sponsors "Sunol Ridge Motor Oil", "Canyon Tyre", "Eden Valley Roasters" and "Redtail Timing", are invented for this track. Any likeness to a real business is unintended. The art uses no real trademarks or logos: no FIA, IMSA, tyre-brand or other marks. "TecPro-style" describes only the shape and colour of a generic modular block barrier.

Other textures in the track come from CC0 sources (ambientCG, Poly Haven), as listed in `docs/ENVIRONMENT.md` and `scripts/p2/20_fetch_textures.py`.

## Race event assets (`scripts/p2/race_assets.py`, track_* layouts)

Downloaded models are all **CC0** from Poly Haven (https://polyhaven.com/license). We use only the 1k glTF mesh and diffuse map, packed into `build/race_assets_atlas.png`. The files are cached in `data/raw/race_assets/polyhaven/`.

| Asset | URL | Author | Licence |
|---|---|---|---|
| Fire extinguisher | https://polyhaven.com/a/korean_fire_extinguisher_01 | Um Joorin | CC0 |
| Portable generator | https://polyhaven.com/a/portable_generator | James Ray Cock | CC0 |
| Oil drum | https://polyhaven.com/a/Barrel_01 | Jorge Camacho | CC0 |
| Jerrycan | https://polyhaven.com/a/metal_jerrycan | Sean Buckley | CC0 |
| Picnic table | https://polyhaven.com/a/wooden_picnic_table | Ulan Cabanilla | CC0 |
| Gas bottle | https://polyhaven.com/a/propane_tank | Poly Haven | CC0 |
| Side-road geometry (used to place barricades) | https://api.openstreetmap.org (`data/raw/race_assets/osm/side_ways.json`) | © OpenStreetMap contributors | ODbL |

**Modelled in this repo (procedural numpy + PIL, real dimensions):**
- Modular aluminium grandstand: stepped treads, seats, scaffold sub-frame, stairs, handrails, cantilever roof, fascia and cladding
- Catch fence and steel crowd-control barriers
- Pop-up gazebos: scissor frame, hip roof, printed valance
- Portable toilets and enclosed race-car trailers
- Straw bales, traffic cones, Type III ROAD CLOSED barricades and a scaffold TV tower

All branding is fictional (PHC).
