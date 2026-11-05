#!/usr/bin/env bash
# Download the prebuilt track zip (stored with Git LFS) into dist/.
# Usage: make download [VERSION=0.2.0-dev1] [BRANCH=claude/brave-archimedes-gmx0t1]
set -euo pipefail
cd "$(dirname "$0")/.."
V="${VERSION:-$(cat VERSION)}"
F="dist/palomares_road_v${V}.zip"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
is_zip() { [ -f "$1" ] && head -c 2 "$1" | grep -q PK; }

echo "==> Palomares Road prebuilt track v${V}"
if is_zip "$F"; then echo "    already downloaded: $F ($(du -h "$F" | cut -f1))"; exit 0; fi
mkdir -p dist
if command -v git-lfs >/dev/null 2>&1 || git lfs version >/dev/null 2>&1; then
  echo "==> git lfs pull ($F)"
  git lfs install --local >/dev/null 2>&1 || true
  git lfs pull --include="$F" && is_zip "$F" && { echo "    done: $F ($(du -h "$F" | cut -f1))"; exit 0; }
  echo "    git lfs pull did not produce the zip; falling back to direct download"
fi
URL="https://media.githubusercontent.com/media/hsaeed3/palomares/${BRANCH}/${F}"
echo "==> downloading $URL"
curl -fL --progress-bar -o "$F.part" "$URL" && mv "$F.part" "$F"
is_zip "$F" || { echo "!! download is not a zip (private repo? use git lfs pull with your GitHub login)"; exit 1; }
echo "    done: $F ($(du -h "$F" | cut -f1))"
echo "    install: drag it onto Content Manager, or unzip palomares_road/ into assettocorsa/content/tracks/"
