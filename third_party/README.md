# third_party

Vendored external code. Kept here (not in `src/`) so its licence and provenance
stay clearly separated from our own code.

## PnLCalib

- **Source:** https://github.com/mguti97/PnLCalib (paper: *"No Bells, Just
  Whistles: Sports Field Registration by Leveraging Geometric Properties"* /
  PnLCalib — sports field registration via points-and-lines optimisation).
- **What it is:** an HRNet keypoint + line detector plus a camera-calibration
  solver. Given a broadcast soccer frame it localises pitch keypoints and solves
  a full camera model.
- **Why it's here:** corner-kick footage packs defenders directly onto the
  penalty-area lines, so classical marking detection cannot recover the box.
  PnLCalib is robust to that occlusion. We use it as the sole calibrator via the
  adapter `src/geometry/pnl_calibration.py`,
  which converts its output into our `Calibration` (interface I7) in the
  `pitch.py` metric convention.

### ⚠️ Licence: GPL-2.0

PnLCalib is licensed **GPL-2.0** (see `PnLCalib/LICENSE`). By vendoring it into
this repository, distribution of the combined work is subject to GPL-2.0. This
was a deliberate, accepted choice for this project; keep our own first-party code
in `src/` so the boundary stays clear, and be aware of the obligation if this
repo is ever distributed externally.

### Weights (not committed — ~506 MB)

The HRNet weights are git-ignored (see `.gitignore`). Download the single-view
weights from the PnLCalib v1.0.0 release into `third_party/PnLCalib/weights/`:

```bash
cd third_party/PnLCalib/weights
curl -SL -o SV_kp    https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp
curl -SL -o SV_lines https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines
```

### Run it

```bash
# via our pipeline demo (recommended):
python scripts/demo_geometry.py --clip data/raw/clips/XXXX.mp4 --pnl --no-detect --overlay

# or PnLCalib's own inference (upstream interface):
cd third_party/PnLCalib
python inference.py --weights_kp weights/SV_kp --weights_line weights/SV_lines \
    --pnl_refine --input_path examples/messi_sample.png --input_type image \
    --save_path out.png --device cpu
```

Local edits to the vendored tree: none to the model/solver code. The adapter
injects `transform2`/`device` (globals PnLCalib defines only under `__main__`)
at import time rather than patching the files.
