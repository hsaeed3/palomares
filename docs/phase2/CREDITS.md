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
