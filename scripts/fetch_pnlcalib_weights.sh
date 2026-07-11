#!/usr/bin/env bash
# Download the PnLCalib HRNet weights (~506 MB) used by --pnl calibration.
# The weights are NOT committed (git-ignored); run this once after cloning.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party/PnLCalib/weights"
BASE="https://github.com/mguti97/PnLCalib/releases/download/v1.0.0"
mkdir -p "$DIR"

for w in SV_kp SV_lines; do
  if [ -s "$DIR/$w" ]; then
    echo "✓ $w already present — skipping"
    continue
  fi
  echo "↓ downloading $w (~253 MB) ..."
  curl -SL --fail -o "$DIR/$w" "$BASE/$w"
done

echo "done → $DIR"
