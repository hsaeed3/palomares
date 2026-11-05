# AC conventions used, and how each was verified

| Convention | Used as | Verified against |
|---|---|---|
| kn5 binary layout (v5: header, textures, materials, node tree, 44-byte vertices, u16 indices) | `scripts/p2/kn5.py` | moppius/blender-assetto-corsa-tools exporter source (read, not vendored); our reader round-trips every file to the last byte |
| Axis mapping (east, north, up) → AC (x, y=up, z=−north); triangle order kept; UV v flipped | kn5.py `ac_from_local` | same exporter (`convert_vector3`, `uv = (u, −v)`) |
| Embedded PNG textures | 22_build_kn5.py | exporter embeds PNG when the source isn't DDS |
| Physics by name prefix `1ROAD`, `1GRASS`, `1KERB`, `1WALL` + `surfaces.ini` KEYs | build + data/surfaces.ini | common community practice; **not** checked in-game |
| `models_<layout>.ini` `[MODEL_n] FILE/POSITION/ROTATION` multi-layout folder, `ui/<layout>/ui_track.json` | 22_build_kn5.py | community docs; **not** checked against a downloaded track |
| Spawn locators: Y up, **−Z forward** | `locator()` in 22_build_kn5.py | AC-Tools Blender add-on description (extensions.blender.org/add-ons/ac-tools). Sources disagree (+Z vs −Z) → **needs in-game check** |
| A-B timing `AC_AB_START_L/R`, `AC_AB_FINISH_L/R` | timing kn5 | CLAUDE.md spec; vanilla support unverified |
| Shaders `ksPerPixel`, `ksPerPixelMultiMap` (road), `ksPerPixelAT` (foliage) | materials | standard Kunos shader names; property sets not checked in-game |
