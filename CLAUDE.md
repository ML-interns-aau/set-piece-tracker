# CLAUDE.md — Set Piece Tracker

Guidance for Claude Code (and humans) working in this repository.

## What this is

**Set Piece Tracker** is an offline video-analytics pipeline that turns a single
**corner-kick clip** into verified player positions, velocities, and a fixed set of
zone/delivery features at two **key moments**: the kick (`t_kick`) and first contact
(`t_contact`). First delivery is ~100 corner-kick clips from women's football, feeding an
English FA set-piece research programme; a second batch of 100 follows on acceptance.

End-to-end per clip: ingest & catalog → detect & track players → classify team & identify
defending goalkeeper → normalize corner orientation → calibrate the zoomed penalty-area
view to metric pitch coordinates → reconstruct ball trajectory & locate `t_kick`/
`t_contact` → emit per-player positions/velocities/reliability at both moments → compute
the 13 zone/delivery features → render an overlay video → mandatory manual verification/
correction → write per-clip and per-batch outputs.

- **Offline, batch-oriented, file-based.** No real-time, no streaming, no database.
  Outputs are CSV/JSON data files plus one overlay MP4 per clip.
- **No ground truth is supplied.** Mandatory human overlay-verification is the system's
  only accuracy reference — a first-class pipeline stage, not QA polish.

### Source of truth

The **technical design doc** and **PRD** are authoritative — they define the planes, the
FR/NFR list, the feature dictionary (Appendix A), the zone geometry (Appendix C), the
schemas, and the component interfaces (I1–I13). Keep them in this repo under `docs/` and
read them before non-trivial work. When this file and the design doc disagree, the design
doc wins. This file summarizes them so an agent can act without re-reading everything.

## Current state (as of this writing)

The repo holds an early **perception skeleton** and little else. Everything else in the
design is unbuilt — do not assume a pipeline runner, catalog, calibration, key-moment
detection, feature computation, reliability scoring, overlay renderer, verification
workflow, or exporters exist yet.

```
src/
├── __init__.py
└── engine/
    ├── detector.py          # FootballDetector — YOLO person+ball detection
    ├── tracker.py           # FootballTracker  — ByteTrack multi-object tracking
    └── team_classifier.py   # TeamClassifier   — team split + goalkeeper detection (FR-005)
```

> The current `src/engine/` grouping is a starting point, **not** the target layout.
> New work should follow the clean-architecture structure below, and this skeleton
> should be folded into it (`src/engine/*` → `src/perception/`).

## Goals & success metrics (targets pending ratification)

| Goal | Metric | Target |
|---|---|---|
| Maximal coverage | camera-visible players carrying a track at `t_kick`/`t_contact` | ≥ 90% per clip; missing players listed |
| Trustworthy positions | manual overlay verdict per player | ≥ 95% confirmed; every position carries a source flag |
| Velocities at moments | tracked players with a velocity estimate | ≥ 80% |
| Complete first batch | clips processed or flagged unusable w/ reason | 100/100 accounted for |
| Features per corner | feature rows complete for processed clips | 100% |
| Visible iteration | daily progress update w/ example overlays | every working day |

Per-clip processing runs in minutes on a GPU workstation; the 100-clip batch runs
unattended overnight; a routine manual check is ~1–2 min/clip.

## Architecture: six logical planes

Build in roughly this order. The **per-clip catalog row is the load-bearing interface**
across all planes, so a failure or pending manual check on one clip never blocks the batch.

| Plane | FRs | Nature | Status |
|---|---|---|---|
| **1. Ingestion & catalog** | FR-001–003 | net-new | not started |
| **2. Perception** (detect, track, team + **GK**) | FR-004/005 | adapted CV primitives | skeleton in `src/engine/` |
| **3. Geometry & moments** (orientation, penalty-area calibration, ball trajectory, `t_kick`/`t_contact`) | FR-006–011 | mostly net-new | not started |
| **4. Features & reliability** (positions, velocities, 13 features, reliability score) | FR-012–016 | net-new | not started |
| **5. Verification** (overlay render + mandatory manual check/correction) | FR-017–019 | net-new | not started |
| **6. Reporting/export** (per-clip files, batch feature table, coverage/reliability report) | FR-020–023 | net-new | not started |

### Detailed design notes (per plane)

- **5.1 Ingestion & catalog (FR-001–003).** Unpack the clip archive; register each clip
  (id, filename, resolution, fps, duration, status). Batch and single-clip entry points
  share one per-clip function → resumption = "skip anything already processed." A
  discontinuity check (histogram/scene-cut delta) flags cuts/replays; if a flagged
  segment overlaps the kick-to-contact window, mark the clip `unusable:discontinuity`.
- **5.2 Detection & tracking (FR-004).** Ultralytics YOLO (person class) + ByteTrack (via
  `supervision`). Re-tune conf/NMS/track-buffer for crowded, occluded box conditions in
  the pilot phase; interfaces (frame in → detections/tracked IDs out) don't change.
  Coverage is measured, not assumed.
- **5.3 Team & GK classification (FR-005).** HSV + KMeans jersey-color clustering for the
  attacking/defending split. GK is net-new: fit a 3-cluster model when kit colors support
  it (silhouette/inertia check vs. the 2-cluster fit), label the smallest most-isolated
  cluster as GK candidate (`gk_confidence=high`); else fall back to position/role
  heuristics (nearest goal line, inside GK area), flag `gk_confidence=low` for the manual
  pass.
- **5.4 Occlusion handling & extrapolation (FR-006).** Bridge detection gaps with linear
  interpolation (short gaps) / constant-velocity extrapolation (boundary gaps). Every such
  value is written `position_source=extrapolated`, never merged with detected values.
- **5.5 Corner-side detection & orientation (FR-007).** Detect the corner arc near a frame
  edge at clip start → fixes left/right side. Normalize all positions to one canonical
  orientation so near-/far-post mean the same across clips. Manual override in the catalog.
- **5.6 Penalty-area calibration (FR-008), net-new.** Corner footage never shows all four
  pitch corners, so solve the homography from a subset of standard markings (penalty-area
  corners, goal-area corners, goalposts 7.32 m apart, penalty spot 11 m, corner arc) —
  any ≥ 4 give a well-conditioned correspondence to FIFA-standard metric coords. Manual
  clickable-point fallback per clip. Reprojection error feeds the reliability score.
- **5.7 Ball detection & trajectory (FR-009).** YOLO ball class + Kalman smoother
  (predicted-position flag, divergence guard via optical flow). Fit a projectile model
  (constant horizontal velocity, gravity-only vertical) anchored to the calibrated pitch
  plane → max height, height at target, max speed. Monocular height is uncertain — report
  a confidence interval, validate against human-judged high/low samples before trusting.
- **5.8 Key-moment detection (FR-010/011).** `t_kick` = first frame ball speed crosses a
  min threshold from a dead-ball baseline, cross-checked vs. taker's foot proximity.
  `t_contact` = first post-kick trajectory discontinuity coincident with a player bbox.
  Burn both decision frames into the overlay; manual moment-tagging override is the
  fallback.
- **5.9 Positions & velocity at moments (FR-012).** Per moment, per player: pitch (x,y) m,
  velocity vector (at `t_kick`, from the up-to-2 s pre-kick window, adaptively shortened),
  team, GK flag, `position_source`, `reliability_score`. This record is the atomic unit
  downstream consumes.
- **5.10 Zone model (FR-013).** Pure coordinate-space polygon/rectangle tests against a
  calibrated position. **Versioned** (`zone_geometry_version`). NEAR/EDGE/short-pass zones
  are provisional — mark them so in code and output.
- **5.11 Feature computation (FR-014/015).** Zone-occupancy counts evaluated at `t_kick`
  by default (evaluation moment is a parameter, extendable to `t_contact`); delivery
  features from the trajectory fit. Every feature is deterministically recomputable from
  stored positions + trajectory fit alone — a geometry/definition change never re-runs
  detection or tracking.
- **5.12 Reliability scoring (FR-016).** Per-position score from detection confidence,
  tracking continuity, extrapolation share, and calibration reprojection error. A
  first-class field that travels with every position into features and exports.
- **5.13 Overlay video (FR-017).** Team-colored boxes, ID labels, ball trail
  (detected/predicted styling), **plus** GK highlight, zone-boundary overlays, and
  burned-in `t_kick`/`t_contact` frame markers. Goal: a human compares output vs. footage
  in 1–2 min without reading raw coordinates.
- **5.14 Manual verification & correction (FR-018/019).** Checker logs players visible vs.
  tracked at each moment, confirms/corrects team & GK, and records positions. Corrections
  live in a separate correction-file (clip, moment, player, field, value) applied as a
  final pass — originals stay auditable underneath; corrected values are re-flagged
  `manually_corrected` and excluded from automated accuracy self-assessment.
- **5.15 Progress artifacts (FR-020).** A lightweight script over the catalog + overlays
  emitting counters and sample frames/clips for the daily report.
- **5.16 Batch orchestration (FR-021–023).** Iterate the catalog with per-clip failure
  containment. Per clip: positions file, feature row, overlay (optionally full per-frame
  tracks). Per batch: combined feature table + coverage/reliability report. Document all
  schemas, units, coordinate convention, and active `zone_geometry_version` with the data.

## Target structure (clean architecture)

Organize by bounded context (the planes), not by framework. **The dependency rule:
everything points inward toward `domain/`; `domain/` depends on nothing.** Keep the core
logic pure and testable; isolate OpenCV/YOLO/file-IO at the edges.

```
src/
├── domain/          # pure data models + contracts (no cv2/ultralytics/pandas imports)
│                    #   Clip, TrackedPlayer, Position, KeyMoments, FeatureRow, schemas + versions
├── ingestion/       # Plane 1: clip registration, catalog, discontinuity check
├── perception/      # Plane 2: detection, tracking, team + GK classification  (← current src/engine)
├── geometry/        # Plane 3: orientation normalization, calibration, ball trajectory, key moments
├── features/        # Plane 4: zone model, feature computation, reliability scoring
├── verification/    # Plane 5: overlay rendering, manual-correction workflow
├── reporting/       # Plane 6: per-clip + batch exporters, progress artifacts
├── pipeline/        # orchestration: per-clip runner + batch runner (per-clip failure containment)
└── config/          # configuration (defaults live in code; overridable per environment)
```

Guidelines:
- **Each plane exposes a narrow interface** matching the I1–I13 boundaries below; planes
  communicate via `domain` types, not by reaching into each other's internals.
- **Pure core, impure edges.** Zone containment, projectile fit, reliability combination,
  and correction application are pure functions — unit-testable without any clip. Video
  decode, model inference, and file writes stay at the boundary.
- **File formats are contracts** (see Data Model below). Make outputs self-describing
  (units, coordinate convention, active `zone_geometry_version`).

## Data model & schema

All data is file-based (CSV/JSON + one overlay MP4 per clip); no database.

### Clip catalog (one row per clip)
`clip_id`, `filename`, `resolution`, `fps`, `duration_s`, `status`
(`pending`/`processed`/`verified`/`unusable`), `unusable_reason`, `corner_side`,
`calibration_quality`, and verification fields (`checker`, `date`, `verdict`,
`corrections_count`).

### Positions file (one row per player per key moment)

| Field | Type | Description | Units / Range |
|---|---|---|---|
| `clip_id` | string | clip identifier | — |
| `moment` | string | key moment this row describes | `t_kick` / `t_contact` |
| `player_id` | int | anonymous track ID | — |
| `team` | string | attacking or defending | `attacking` / `defending` |
| `is_goalkeeper` | bool | defending goalkeeper flag | `true` / `false` |
| `pitch_x`, `pitch_y` | float | calibrated pitch position | m |
| `velocity_x`, `velocity_y` | float | velocity components (`t_kick` only, from pre-kick window) | m/s |
| `position_source` | string | provenance | `detected` / `extrapolated` / `manually_corrected` |
| `reliability_score` | float | combined confidence (detection, continuity, extrapolation share, calibration) | 0–1 |
| `velocity_window_s` | float | actual pre-kick seconds used (`t_kick` only) | s |

### Feature row (one row per corner)
`clip_id`, `corner_side`, `t_kick_frame`, `t_contact_frame`, `zone_geometry_version`,
plus the 13 feature columns of Appendix A (`num_short_pass_options` … `pass_hight_in_m_at_target`).

### Batch coverage/reliability report (one row per clip)
players visible (manual count) vs. players tracked at each moment, mean reliability
score, verification status.

## Feature dictionary (13 features — Appendix A is authoritative)

Exact column names/order are in PRD Appendix A; the ordering runs
`num_short_pass_options` … `pass_hight_in_m_at_target` (note the delivered spelling
`hight`). The goalkeeper is **excluded** from counts where specified.

Zone-occupancy counts (1–10): number of short-pass options; defenders in NEAR; attackers
& defenders in GK area; attackers & defenders in PENALTY; attackers & defenders in EDGE;
defenders on near post; defenders on far post.

Delivery metrics (11–13): maximum trajectory height (loftedness, m); maximum pass speed
(m/s); ball height at target point (`pass_hight_in_m_at_target`, m).

## Zone geometry (Appendix C — provisional per §11)

Pure coordinate-space regions tested against calibrated positions, tagged with
`zone_geometry_version`:
- **GK area** = goal area, **PENALTY area**, **NEAR area**, **EDGE area**,
  **near-post** / **far-post** bands, **short-pass target zones**.
- **NEAR, EDGE, and short-pass zones are provisional** pending FA confirmation — mark them
  provisional in code and output until confirmed. A confirmation triggers recomputation
  from stored positions, never re-tracking.

## Component interfaces (I1–I13)

Contracts between planes; keep these stable. `player`/`ball` classes, pixel bboxes
`(x1,y1,x2,y2)`, pitch coords in meters.

- **I1 Ingest→Detection:** `clip_id`; `frames` iterator of `(frame_idx, image[BGR])`; `fps`.
- **I2 Detection→Tracking** (per frame): `frame_idx`; `detections[].bbox` (px), `.class`
  (player/ball), `.confidence` (0–1).
- **I3 Tracking→Classification** (per frame): `frame_idx`; `tracks[].track_id` (persistent
  within clip), `.bbox`, `.class`, `.confidence`.
- **I4 Classification→downstream** (per track, once/clip): `track_id`; `team`
  (attacking/defending); `is_goalkeeper`; `gk_confidence` (`high`=3-cluster / `low`=role
  heuristic, flagged for manual check).
- **I5 Gap bridging→geometry** (per frame, per track): `frame_idx`; `track_id`;
  `foot_point` (u,v px = bottom-center of bbox, the point the homography maps); `source`
  (detected/extrapolated); `gap_frames` (0 if detected).
- **I6 Corner-side→normalization:** `corner_side` (left/right, broadcast frame);
  `side_source` (auto / manual override).
- **I7 Calibration→metric consumers:** `H` (3×3, pixel→pitch meters, post-normalization);
  `reprojection_error` (m, feeds reliability); `points_used` (≥ 4).
- **I8 Ball trajectory→moments & features:** `ball_track[]` `(frame_idx, x, y)` in m;
  `fit.max_height_m`; `fit.max_speed_ms`; `fit.height_at_target_m` (null if no cross into
  PEN/GK); `fit.confidence_interval` (m, on height).
- **I9 Key-moment→snapshot:** `t_kick_frame`; `t_contact_frame` (null if ball reaches no
  player).
- **I10 Snapshot (I4+I5+I7+I9)→Positions at moments:** per moment/track, map `foot_point`
  through `H`, compute velocity over the pre-moment window → rows of the Positions file.
- **I11 Positions→Features:** Positions rows (+ I8 fit) → one Feature row.
- **I12 everything→Overlay:** I3 tracks + I4 labels + I8 ball_track + I9 moments + zone
  polygons → `overlay.mp4`.
- **I13 Overlay+Positions→Verification:** checker reads overlay + Positions rows; writes
  correction records + catalog verdict fields; apply-corrections re-emits Positions rows
  with `manually_corrected` flags.

## Domain facts & conventions (must match the schema)

- **Two key moments:** `t_kick` (ball motion onset from dead-ball baseline) and
  `t_contact` (first trajectory discontinuity coincident with a player bbox). `t_contact`
  may be `null`; delivery features still compute up to that point.
- **Provenance is mandatory.** `position_source ∈ {detected, extrapolated,
  manually_corrected}`. Never merge extrapolated with detected; never fabricate a position
  to fill a gap — unsupported positions are output as **missing**, counted against
  coverage.
- **Reliability score (0–1)** is a first-class field on every position — not a diagnostic
  side-channel.
- **Coordinates:** meters, **canonical orientation** (corner side normalized). Velocities
  in m/s. Foot point = bottom-center of bbox.
- **Pitch model (FIFA standard):** 105 × 68 m; penalty area 40.32 × 16.5 m; goal area
  18.32 × 5.5 m; goal width 7.32 m; penalty spot 11 m.
- **Zone geometry is versioned**; NEAR/EDGE/short-pass provisional; features recomputable
  from stored positions + trajectory fit alone.
- **Manual corrections are layered**, not overwritten in place; corrected values excluded
  from automated accuracy self-assessment.
- **Per-clip failure containment:** one bad clip → `unusable:<reason>`, batch continues.

### Sentinel IDs & class IDs currently in code

- `detector.py` detects **COCO class 0 (person)** and **class 32 (sports ball)**,
  `imgsz=960`, agnostic NMS.
- `tracker.py`: ball is assigned `BALL_TRACKER_ID = -99`; players get ByteTrack IDs.
- `team_classifier.py`: `UNKNOWN_ID = -1`, `REFEREE_ID = -2`, `GK_ID = -3`. Teams `0`/`1`.
  `detect_goalkeeper=False` reproduces plain 2-team behavior.

## Failure modes to handle (design §9)

- Corrupt/unreadable clip → `unusable:corrupt_file`; batch continues.
- Cut/replay over the kick-to-contact window → `unusable:discontinuity`.
- Player occluded through both moments → reported missing, counted against coverage.
- GK kit ≈ outfield kit → position/role fallback, `gk_confidence=low`, manual resolve.
- >/< 22 players (subs, staff) → all tracked; only clearly-in-play counted for coverage.
- < 4 calibration markings → manual reference-point entry; flag for review if neither works.
- No pre-kick footage → shorter/omitted `t_kick` velocity; record `velocity_window_s`.
- Moment misdetected → visible via burned-in overlay marker; manual moment-tag fallback.
- Ball reaches no player → `t_contact=null`; delivery features from kick-onward trajectory.
- Short corner outside every zone → zero-occupancy for all zones (not forced into nearest);
  flag for zone-geometry review.
- Zone geometry revised → recompute features under new `zone_geometry_version`, no re-track.
- Verification backlog → clips stay `processed` (not `verified`), excluded from accepted
  count; a visible catalog state, not a silent gap.

## Development

- **Python 3.14** (`.venv/` present; `python 3.14.5`).
- **Dependencies** (`requirements.txt`): `opencv-python-headless`, `numpy`,
  `supervision`, `ultralytics`, `scikit-learn`.
- **YOLO weights:** code defaults to `yolo11m.pt`. Weights are not committed; the pilot
  phase (PRD Phase 2) decides which weights to standardize on before batch runs.
- **Key parameters** (proposed, pending pilot tuning): calibration ≥ 4 penalty-area/
  goal-area/goalpost/penalty-spot points; velocity window adaptive up to 2 s pre-kick;
  zone geometry per Appendix C (provisional). Perception thresholds (detection conf/NMS,
  track buffer, team-cluster count/refit interval) start from sensible defaults, not final.
- **Setup:**
  ```bash
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- There are **no tests, no CLI entry point, and no CI** yet. When adding them, unit-test
  the pure logic (zone containment, projectile fit, reliability combination, correction
  application) independent of any specific clip; validate end-to-end on 3–5 pilot clips
  before batch scale.

## Working norms for this repo

- **Match the schema exactly.** New outputs conform to the Data Model above and are
  self-describing (units, coordinate convention, active `zone_geometry_version`).
- **Verification is load-bearing.** Anything affecting a checker's ability to spot an
  error in ~1–2 min/clip (overlay clarity, burned-in moment markers, GK highlight) is
  product-critical, not cosmetic.
- **Follow the phase gates** (pilot 3–5 clips → stabilize → full 100-clip batch → confirm
  zone geometry → stakeholder review). Don't jump to batch scale before pilot validation
  of calibration/key-moment/GK logic.
- **Privacy:** no identity recognition (face/jersey OCR/roster) — anonymous track IDs with
  team + GK flags only. Processing stays local; overlay videos carry the same distribution
  restrictions as source footage.
- Commit/push only when asked. Team members commit under their own names.
