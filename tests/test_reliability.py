"""FR-016 - reliability-score formula against constructed input cases."""

from __future__ import annotations

import pytest

from src.features.reliability import (
    EXTRAPOLATION_FLOOR,
    ReliabilityInputs,
    calibration_subscore,
    continuity_subscore,
    detection_subscore,
    extrapolation_subscore,
    reliability_score,
)


def _perfect() -> ReliabilityInputs:
    return ReliabilityInputs(
        detection_confidence=1.0,
        gap_frames=0.0,
        extrapolation_share=0.0,
        reprojection_error_m=0.0,
    )


def test_perfect_data_scores_near_one():
    assert reliability_score(_perfect()) == 1.0


def test_low_detection_confidence_lowers_score():
    base = reliability_score(_perfect())
    lowered = reliability_score(
        ReliabilityInputs(0.3, 0.0, 0.0, 0.0)
    )
    assert lowered < base
    assert 0.0 < lowered < 1.0


def test_large_extrapolation_share_lowers_score():
    base = reliability_score(_perfect())
    lowered = reliability_score(ReliabilityInputs(1.0, 0.0, 0.8, 0.0))
    assert lowered < base


def test_full_extrapolation_lowers_but_does_not_zero_score():
    # design 5.6: extrapolation is a normal output the score GRADES, not vetoes.
    # A fully-estimated position is heavily penalised but stays non-zero so the
    # other factors can still differentiate it.
    full = reliability_score(ReliabilityInputs(1.0, 0.0, 1.0, 0.0))
    assert 0.0 < full < reliability_score(_perfect())


def test_short_interpolation_beats_long_extrapolation():
    # the case that motivated the graceful floor (FR-006/5.4): both positions
    # are fully estimated (share=1.0) with equally strong anchors, differing
    # only in bridged-gap length. The score must still distinguish them.
    short_interp = reliability_score(ReliabilityInputs(0.9, 2.0, 1.0, 0.2))
    long_extrap = reliability_score(ReliabilityInputs(0.9, 30.0, 1.0, 0.2))
    assert short_interp > long_extrap > 0.0


def test_poor_calibration_lowers_score():
    base = reliability_score(_perfect())
    lowered = reliability_score(ReliabilityInputs(1.0, 0.0, 0.0, 3.0))
    assert lowered < base


def test_missing_calibration_zeroes_score():
    assert reliability_score(ReliabilityInputs(1.0, 0.0, 0.0, None)) == 0.0


def test_gap_lowers_score_monotonically():
    scores = [
        reliability_score(ReliabilityInputs(1.0, g, 0.0, 0.0))
        for g in (0.0, 5.0, 15.0, 40.0)
    ]
    # strictly decreasing as the bridged gap grows
    assert all(a > b for a, b in zip(scores, scores[1:]))


def test_combined_factors_are_monotonic():
    good = reliability_score(ReliabilityInputs(0.95, 2.0, 0.05, 0.2))
    mid = reliability_score(ReliabilityInputs(0.8, 6.0, 0.2, 0.6))
    poor = reliability_score(ReliabilityInputs(0.5, 20.0, 0.5, 1.5))
    assert good > mid > poor
    assert all(0.0 < s < 1.0 for s in (good, mid, poor))


def test_score_is_clamped_to_unit_interval():
    # out-of-range detection confidence is clamped, not amplified
    s = reliability_score(ReliabilityInputs(5.0, 0.0, -0.5, 0.0))
    assert 0.0 <= s <= 1.0
    assert s == 1.0


# --- sub-score unit behaviour ----------------------------------------------
def test_subscores_at_half_points():
    # continuity 0.5 at the half-frames constant; calibration 0.5 at half-error
    assert continuity_subscore(12.0, half_frames=12.0) == 0.5
    assert calibration_subscore(1.0, half_error_m=1.0) == 0.5


def test_subscore_bounds():
    assert detection_subscore(2.0) == 1.0
    assert detection_subscore(-1.0) == 0.0
    assert extrapolation_subscore(0.0) == 1.0
    # fully-extrapolated floors at EXTRAPOLATION_FLOOR, not 0 (graceful degrade)
    assert extrapolation_subscore(1.0) == pytest.approx(EXTRAPOLATION_FLOOR)
    assert continuity_subscore(0.0) == 1.0
    assert calibration_subscore(0.0) == 1.0
    assert calibration_subscore(None) == 0.0


def test_extrapolation_subscore_is_linear_between_floor_and_one():
    # half-estimated sits halfway between the floor and 1
    mid = extrapolation_subscore(0.5)
    assert mid == pytest.approx(1.0 - (1.0 - EXTRAPOLATION_FLOOR) * 0.5)
