#!/bin/bash
# dev7 G gates for one road: qa_on_road, qa_ffb_safety, qa_net.  usage: scripts/p7g/gates.sh <road> <tag> [qa_net args]
set -u
R=$1; TAG=$2; shift 2
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export PALOMARES_BUILD_DIR=${PALOMARES_BUILD_DIR:-$ROOT/build_dev7}
cd "$ROOT/scripts/p2"
rc=0
nice python3 qa_on_road.py --track "$PALOMARES_BUILD_DIR/palomares_road" --road "$R" --kn5 "palomares_niles_${TAG}*.kn5" | tail -1 || rc=1
QA_FFB_KN5="palomares_niles_${TAG}*.kn5" QA_FFB_ROAD_KN5="palomares_niles_${TAG}.kn5" QA_FFB_ROAD="$R" QA_FFB_OUT="dev7/qa_ffb_${TAG}.json" \
  nice python3 qa_ffb_safety.py | tail -6 || rc=1
nice python3 "$ROOT/scripts/p7g/qa_net.py" "$R" --kn5 "palomares_niles_${TAG}*.kn5" "$@" | tail -1 || rc=1
exit $rc
