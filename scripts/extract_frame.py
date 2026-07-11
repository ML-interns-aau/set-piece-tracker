"""Extract a single frame from a clip and save it as a PNG for manual calibration.

Usage:
    python scripts/extract_frame.py --clip data/raw/clips/5416922_1105.mp4 --frame 0
    python scripts/extract_frame.py --clip data/raw/clips/5416922_1105.mp4 --frame 25

Open the saved PNG in any image viewer, hover over pitch markings to read pixel
coordinates, then fill in a calibration JSON (copy data/samples/calibration_points.example.json
as a template) and pass it via --calib to demo_geometry.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract a frame as PNG for manual calibration.")
    ap.add_argument("--clip", required=True, help="path to a corner-kick clip")
    ap.add_argument("--frame", type=int, default=0, help="frame index to extract (default: 0)")
    ap.add_argument("--out", default=None, help="output PNG path (default: outputs/frame_<N>.png)")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    if not clip_path.exists():
        raise SystemExit(f"clip not found: {clip_path}")

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open clip: {clip_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {args.frame}")

    out_path = Path(args.out) if args.out else Path("outputs") / f"frame_{args.frame}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    h, w = frame.shape[:2]
    print(f"saved frame {args.frame} ({w}x{h}) to {out_path}")
    print()
    print("Next steps:")
    print(f"  1. Open {out_path} in an image viewer")
    print("  2. Hover over pitch markings and note pixel (u, v) coordinates")
    print("  3. Copy data/samples/calibration_points.example.json and fill in your points")
    print("  4. Run: python scripts/demo_geometry.py --clip <clip> --calib your_points.json --overlay")


if __name__ == "__main__":
    main()
