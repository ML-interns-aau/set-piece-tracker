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

## Run the manual verification UI

Two pieces: render an overlay video for a clip, then start the review server.

```bash
# 1. Render team-colored/GK-highlighted boxes + zone overlays + event markers.
#    --tracks/--calib are optional (see scripts/render_overlay.py --help for
#    their JSON formats); omit them to render zones/markers with no player boxes.
python scripts/render_overlay.py --clip data/raw/clips/<CLIP>.mp4 \
    --events outputs/verification/<CLIP>_events.json \
    --out outputs/verification/<CLIP>_overlay.mp4

# 2. Start the local review server (stdlib only — no cv2 GUI needed since
#    requirements.txt pins opencv-python-headless).
python -m src.verification.ui --clip <CLIP> \
    --original data/raw/clips/<CLIP>.mp4 \
    --overlay outputs/verification/<CLIP>_overlay.mp4 \
    --events outputs/verification/<CLIP>_events.json \
    --log outputs/verification/<CLIP>_reviews.jsonl
```

Open `http://127.0.0.1:8765/` — original clip left, overlay right, kept in lockstep.

| Key | Action |
|---|---|
| Space | Play / pause both videos |
| ← / → | Step one frame back / forward |
| V | Verdict: Verified |
| R | Verdict: Rejected |
| E | Verdict: Needs Correction |

Each verdict appends one line to the `--log` JSONL file (`clip_id`, `frame`, `event`,
`verdict`, `reviewer`, `timestamp`); reloading the page re-derives which events are
already reviewed from that file, so review sessions resume where they left off.

**Corrections** (FR-019, `src/verification/correction_schema.py`/`correction.py`) are
layered on top of an events file rather than edited in place:

```bash
python scripts/apply_corrections.py \
    --pipeline-output outputs/verification/<CLIP>_events.json \
    --corrections outputs/verification/<CLIP>_corrections.json
# writes outputs/verification/<CLIP>_events_corrected.json — refuses to overwrite
# either input file, so the original stays auditable underneath.
```

A correction file is `{"clip_id": ..., "corrections": [{"frame": ..., "action":
"change_event" | "change_player" | "delete_event" | "add_event", ...}]}` — see the
module docstring in `src/verification/correction_schema.py` for the exact field set
each action requires.

## Run the tests

```bash
python -m pytest -q
```

Pure logic — no clip or weights needed: geometry/moments (homography application,
orientation, trajectory fit, key-moment detection), the features & reliability plane
(zone containment, the 13-feature computation, reliability scoring), and the
verification plane (overlay drawing primitives against synthetic frames, the
correction-file schema/apply logic, and the review server's HTTP/JSON routes — see
`tests/test_ui.py`'s module docstring for what that last one does *not* cover, namely
browser/keyboard automation). The PnLCalib calibration model itself needs torch +
weights and is not exercised by this suite.

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

**Done — Verification plane (FR-017–019), `src/verification/`:** overlay rendering
(team-colored + GK-highlighted boxes, zone overlays, burned-in event markers), a
stdlib-only local web app for manual review (side-by-side video, keyboard shortcuts,
verdict logging with resume), and a correction-file system (frame-indexed
change/delete/add edits applied as a separate, auditable pass — never in place). This
plane operates on its own event vocabulary (`src/verification/events.py`) bridged from
the real `t_kick`/`t_contact` key moments today; it does not yet have a real
shot/pass/header/etc. event-detection stage to consume (see that module's docstring).

**Next (see `CLAUDE.md` for the plan):** the **I5 foot-point pipeline** (per-frame player
foot points) — it unblocks the taker-foot cross-check for `t_kick`, player-gating for
`t_contact`, and **I10** (player positions/velocities at the moments) that feeds the
features plane above and, downstream, the events this verification plane reviews. Then
batch export. Monocular ball-height estimation is still open (delivery-height metrics
return `None` until then).
