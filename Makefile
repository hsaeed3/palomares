all: phase1 phase2

phase1:
	bash scripts/run_all.sh

phase2:
	bash scripts/p2/run_phase2.sh

# Quick installable zip (no QA / renders): uses committed data/derived + reports/visual previews
package:
	cd scripts/p2 && python3 19_gopro_calibrate.py && python3 20_fetch_textures.py && python3 21_build_geometry.py && python3 21b_track_dressing.py && python3 22_build_kn5.py && python3 25_package.py
