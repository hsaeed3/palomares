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

# dev5 scene ledger: build hook (after 21c, before 21d), catalog yaml + contact sheet, seed from the survey files
ledger-assets:
	python3 scripts/p2/21e_ledger_assets.py

catalog:
	python3 scripts/p5/assets.py --yaml --sheet

ledger-seed:
	python3 scripts/p5/ledger_seed.py

physics:
	python3 scripts/p2/21d_physics.py

qa-collisions:
	python3 scripts/p2/qa_collisions.py

# Download the prebuilt zip (Git LFS, with a direct-download fallback) into dist/
download:
	bash scripts/download_build.sh
