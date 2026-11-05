all: phase1 phase2

phase1:
	bash scripts/run_all.sh

phase2:
	bash scripts/p2/run_phase2.sh

# Quick installable zip (no QA / renders): uses committed data/derived + reports/visual previews
package:
	python3 scripts/p2/build.py package

# Roadside assets (21c, after 21/21b) and the collision QA (needs the kn5s from 22)
roadside:
	python3 scripts/p2/21c_roadside_assets.py

qa-collisions:
	python3 scripts/p2/qa_collisions.py
