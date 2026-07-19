#!/usr/bin/env bash
# Fetch the dedicated soccer-ball detection weights (git-ignored, ~130 MB).
#
# Model: football-ball-detection.pt from roboflow/sports (examples/soccer) --
# a YOLOv8x fine-tuned on broadcast soccer footage (Roboflow Universe dataset
# football-ball-detection-rejhg, trained at imgsz=1280). Public Google Drive
# file, no API key needed. Licence note: a YOLOv8 fine-tune is AGPL-3.0 via
# ultralytics (already in requirements.txt); see third_party/README.md.
#
# Usage:  bash scripts/fetch_ball_weights.sh
set -euo pipefail

DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/weights"
DEST="$DEST_DIR/football-ball-detection.pt"
GDRIVE_ID="1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"

mkdir -p "$DEST_DIR"
if [ -f "$DEST" ]; then
    echo "already present: $DEST"
    exit 0
fi

if ! python -c "import gdown" >/dev/null 2>&1; then
    echo "installing gdown (Google Drive downloader) ..."
    pip install --quiet gdown
fi

python -m gdown "https://drive.google.com/uc?id=$GDRIVE_ID" -O "$DEST"
echo "wrote $DEST"
