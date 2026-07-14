"""Per-position reliability score (FR-016, design 5.12).

A first-class 0..1 confidence that travels with every position into the
features and the export (not a diagnostic side-channel). It combines four
inputs already produced upstream:

  * **detection confidence** at the source frame(s)              -> higher is better
  * **tracking continuity** (gap length bridged by FR-006)       -> shorter gap is better
  * **extrapolation share** (fraction of the window estimated)   -> less is better
  * **calibration quality** (FR-008 reprojection error, metres)  -> smaller is better

Each input is mapped to a 0..1 sub-score, then the sub-scores are combined by a
weighted geometric mean so that a weak factor drags the whole score down rather
than being averaged away. The result is monotonic in every input and clamped to
[0, 1].

**Graceful, not binary (design 5.6).** The continuous quality signals — gap
length, extrapolation share, reprojection error — *degrade smoothly*; none of
them alone forces the score to zero. In particular an extrapolated position
(``position_source=extrapolated``, FR-006/5.4) is a normal output the score is
built to *grade*, so a short interpolation between strong detections still
scores meaningfully and is distinguished from a long, wild extrapolation. The
extrapolation sub-score therefore floors at :data:`EXTRAPOLATION_FLOOR` rather
than reaching 0.

**The one hard gate** is a *missing* calibration (``reprojection_error_m=None``):
with no homography there is no metric position at all, so the design returns
"no calibration" and routes the frame to manual review rather than poisoning
downstream (design 5.6 "Safety"). That case — and only that case, plus a
degenerate detection confidence of exactly 0 (see caller contract on
``ReliabilityInputs``) — yields a score of 0.

Everything here is a pure function of scalar inputs, unit-tested against
constructed cases — no clip, model, or file needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Half-widths / scales controlling how fast each sub-score falls off. Proposed
# defaults pending pilot tuning (kept explicit so tests pin the behaviour).
GAP_HALF_FRAMES = 12.0          # gap at which continuity sub-score = 0.5
REPROJ_HALF_ERROR_M = 1.0       # reprojection error at which calibration = 0.5

# Floor for the extrapolation sub-score at full extrapolation (share = 1.0). A
# fully-estimated position (the FR-006/5.4 normal case) keeps a small non-zero
# sub-score so gap length and detection confidence still differentiate a short
# interpolation from a long extrapolation, per the "degrade gracefully rather
# than binarily" principle (design 5.6). Not 0: the score grades extrapolation,
# it does not veto it (only a missing calibration vetoes — see module docstring).
EXTRAPOLATION_FLOOR = 0.1

# Relative weights of the four sub-scores in the geometric mean. They need not
# sum to 1 (they are normalised internally); the ratios are what matter.
WEIGHT_DETECTION = 1.0
WEIGHT_CONTINUITY = 1.0
WEIGHT_EXTRAPOLATION = 1.0
WEIGHT_CALIBRATION = 1.0


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else float(v)


def detection_subscore(detection_confidence: float) -> float:
    """Detection confidence is already 0..1; just clamp it."""
    return _clamp01(detection_confidence)


def continuity_subscore(gap_frames: float, half_frames: float = GAP_HALF_FRAMES) -> float:
    """Map a bridged-gap length (frames) to 0..1 — 0 gap -> 1, decaying smoothly.

    Uses ``half_frames`` as the gap at which the sub-score is 0.5.
    """
    g = max(0.0, float(gap_frames))
    return half_frames / (half_frames + g)


def extrapolation_subscore(
    extrapolation_share: float, floor: float = EXTRAPOLATION_FLOOR
) -> float:
    """Fraction of the window that was estimated -> [floor, 1] (observed -> 1).

    Linearly interpolates from 1.0 at share = 0 down to ``floor`` at share = 1,
    so a fully-extrapolated position keeps a small non-zero sub-score (design
    5.6: grade extrapolation gracefully, don't veto it).
    """
    share = _clamp01(extrapolation_share)
    return 1.0 - (1.0 - floor) * share


def calibration_subscore(
    reprojection_error_m: float, half_error_m: float = REPROJ_HALF_ERROR_M
) -> float:
    """Map calibration reprojection error (m) to 0..1 — 0 error -> 1, decaying.

    Uses ``half_error_m`` as the error at which the sub-score is 0.5. A missing
    calibration (``None``) scores 0.
    """
    if reprojection_error_m is None:
        return 0.0
    e = max(0.0, float(reprojection_error_m))
    return half_error_m / (half_error_m + e)


@dataclass(frozen=True)
class ReliabilityInputs:
    """The four upstream signals a reliability score is built from (FR-016)."""

    detection_confidence: float      # 0..1
    gap_frames: float                # frames bridged by extrapolation (0 if detected)
    extrapolation_share: float       # 0..1 fraction of the window estimated
    reprojection_error_m: float | None  # calibration error, m (None if uncalibrated)


def reliability_score(
    inputs: ReliabilityInputs,
    *,
    w_detection: float = WEIGHT_DETECTION,
    w_continuity: float = WEIGHT_CONTINUITY,
    w_extrapolation: float = WEIGHT_EXTRAPOLATION,
    w_calibration: float = WEIGHT_CALIBRATION,
    gap_half_frames: float = GAP_HALF_FRAMES,
    reproj_half_error_m: float = REPROJ_HALF_ERROR_M,
    extrapolation_floor: float = EXTRAPOLATION_FLOOR,
) -> float:
    """Combine the four sub-scores into a 0..1 reliability score (FR-016).

    Weighted **geometric** mean: a weak sub-score pulls the result down (rather
    than being averaged away), while the score stays monotonic in every input.
    Only a *missing* calibration or a zero detection confidence forces the score
    to 0; the continuous quality signals degrade gracefully (design 5.6).
    """
    subs = (
        (detection_subscore(inputs.detection_confidence), w_detection),
        (continuity_subscore(inputs.gap_frames, gap_half_frames), w_continuity),
        (extrapolation_subscore(inputs.extrapolation_share, extrapolation_floor),
         w_extrapolation),
        (calibration_subscore(inputs.reprojection_error_m, reproj_half_error_m),
         w_calibration),
    )
    total_w = sum(w for _, w in subs)
    if total_w <= 0.0:
        raise ValueError("weights must sum to a positive value")

    # weighted geometric mean = exp( sum(w_i * ln s_i) / sum(w_i) );
    # any zero sub-score forces the product to 0 without a log-domain error.
    if any(s <= 0.0 for s, w in subs if w > 0.0):
        return 0.0

    log_sum = sum(w * math.log(s) for s, w in subs)
    return _clamp01(math.exp(log_sum / total_w))
