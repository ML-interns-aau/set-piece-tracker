# Set Piece Tracker

Offline video-analytics pipeline that turns a corner-kick clip into verified player
positions, velocities, and zone/delivery features at two key moments — the kick
(`t_kick`) and first contact (`t_contact`). See **`CLAUDE.md`** for the full design,
architecture, data model, and project status.

This README is the quick "how do I run it" guide.

## Setup

```bash
# Python 3.14
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On a CPU-only machine, install CPU torch first so pip doesn't pull the multi-GB CUDA
stack (see the note at the top of `requirements.txt`).

### Model weights (not committed)

Two sets of weights are needed and are **git-ignored** (too large to commit):

- **YOLO** (`yolo11m.pt`) — player/ball detection. Ultralytics downloads it on first use,
  or drop it in the repo root.
- **PnLCalib** (`SV_kp`, `SV_lines`, ~506 MB) — the calibration model. Fetch them with:
  ```bash
  scripts/fetch_pnlcalib_weights.sh
  ```
  They land in `third_party/PnLCalib/weights/` (git-ignored). See `third_party/README.md`.

## Run the geometry & moments demo on one clip

```bash
python scripts/demo_geometry.py --clip data/raw/clips/<CLIP>.mp4 --overlay
```

Calibration is always the vendored **PnLCalib** model (per-frame, robust to the
corner-kick player wall; needs the weights above) — it is the sole calibration path, so
there is no flag to select it and no classical/manual fallback. Useful flags:

- `--overlay` — write `overlay.mp4` with the pitch markings + ball trail + `t_kick`/
  `t_contact` markers burned in.
- `--device 0` — run YOLO ball detection on the GPU (default `cpu`). PnLCalib
  **auto-selects the GPU** when CUDA is available, independent of this flag.
- `--no-detect` — calibrate + draw the overlay only, skip YOLO ball detection.
- `--max-frames N` — cap frames processed. Per-frame calibration + YOLO run on every
  frame, so on CPU use e.g. `--max-frames 40` for a quick look.
- `--frame N` — frame used for the reported corner side + calibration summary (default 0).
- `--conf F` — YOLO ball-detection confidence (default 0.25); `--out DIR` — output
  directory (default `outputs/demo`).

Outputs land in `outputs/demo/` (git-ignored):
- `overlay.mp4` — **watch this**: the cyan pitch markings should stay glued to the real
  painted lines across the whole clip (that's the per-frame calibration tracking the
  camera). The ball trail should sit on the ball; `t_kick`/`t_contact` should land on the
  real kick and first touch.
- `summary.json` — corner side, calibration reprojection error, key-moment frames, trajectory.
- `ball_track.csv` — per-frame ball position in pixels **and** pitch metres.

## Run the tests

```bash
python -m pytest -q
```

Pure logic — no clip or weights needed: geometry/moments (homography application,
orientation, trajectory fit, key-moment detection) **and** the features & reliability
plane (zone containment, the 13-feature computation, reliability scoring). The PnLCalib
calibration model itself needs torch + weights and is not exercised by this suite.

## Status & where to pick up

**Done — Geometry & Moments plane (FR-007–011):** corner-side/orientation, penalty-area
calibration (the vendored **PnLCalib** learned model is the sole path → a per-frame
`CalibrationTrack` that tracks camera pan/zoom), ball smoothing + projectile trajectory
fit, and `t_kick`/`t_contact` detection. Unit-tested (except the PnLCalib model itself).

**Done — Features & Reliability plane (FR-012–016), pure logic** (`src/features/`): the
versioned zone model (Appendix C polygons, provisional zones flagged), the 13-feature
computation (Appendix A schema), and per-position reliability scoring (which grades
extrapolated positions gracefully rather than zeroing them). Fully unit-tested; it runs on
`PlayerPosition` records, so it waits on the **I10** producer below to feed it real clips.

**Next (see `CLAUDE.md` for the plan):** the **I5 foot-point pipeline** (per-frame player
foot points) — it unblocks the taker-foot cross-check for `t_kick`, player-gating for
`t_contact`, and **I10** (player positions/velocities at the moments) that feeds the
features plane above. Then overlay verification and batch export. Monocular ball-height
estimation is still open (delivery-height metrics return `None` until then).
