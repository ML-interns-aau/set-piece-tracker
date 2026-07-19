"""BallFinder pure logic (tiers, temporal gate, tiling) with stubbed models.

No torch/ultralytics needed: a stub model reproduces the ultralytics result
contract for scripted detections, so the tier fallback and gating logic are
tested without weights or a GPU.
"""

from __future__ import annotations

import numpy as np

from src.engine.ball_finder import BALL_CLASS_ID, BallFinder


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _T(np.asarray(xyxy, dtype=np.float64).reshape(-1, 4))
        self.conf = _T(np.asarray(conf, dtype=np.float64))
        self.cls = _T(np.asarray(cls, dtype=np.float64))

    def __len__(self):
        return len(self.xyxy._a)


class _T:
    """Minimal tensor stand-in: .cpu().numpy() chain."""

    def __init__(self, a):
        self._a = a

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def astype(self, t):
        return self._a.astype(t)

    def __len__(self):
        return len(self._a)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class StubModel:
    """Scripted detector: maps (image height, width) -> detections.

    ``script`` is a callable(image, classes) -> (xyxy, conf, cls) lists.
    Records every call for assertions.
    """

    def __init__(self, script):
        self.script = script
        self.calls = []

    def __call__(self, image, classes=None, **kw):
        self.calls.append((image.shape, tuple(classes or ())))
        xyxy, conf, cls = self.script(image, classes)
        return [_Result(_Boxes(xyxy, conf, cls) if len(xyxy) else None)]


def _frame(h=720, w=1280):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _finder(script, shared=True, **kw):
    m = StubModel(script)
    models = (m, m) if shared else (m, StubModel(script))
    f = BallFinder(_models=models, **kw)
    return f, models


def test_full_frame_hit_uses_full_tier():
    def script(image, classes):
        # ball at (100, 100) + one player, only on full frames
        if image.shape[:2] == (720, 1280):
            return ([[95, 95, 105, 105], [400, 300, 430, 380]], [0.4, 0.9],
                    [BALL_CLASS_ID, 0])
        return ([], [], [])

    f, _ = _finder(script)
    ball, players, tier = f.process_frame(_frame())
    assert tier == "full"
    np.testing.assert_allclose(ball, [95, 95, 105, 105])
    assert len(players) == 1
    assert f.tier_counts["full"] == 1


def test_roi_tier_recovers_after_miss():
    state = {"n": 0}

    def script(image, classes):
        state["n"] += 1
        h, w = image.shape[:2]
        if (h, w) == (720, 1280) and state["n"] == 1:
            return ([[95, 95, 105, 105]], [0.4], [BALL_CLASS_ID])
        if (h, w) == (320, 320):  # ROI crop around the last position
            # centre (140, 140): a 56 px move, inside the 60 px one-frame gate
            return ([[135, 135, 145, 145]], [0.3], [BALL_CLASS_ID])
        return ([], [], [])

    f, _ = _finder(script)
    ball1, _, tier1 = f.process_frame(_frame())
    assert tier1 == "full"
    ball2, _, tier2 = f.process_frame(_frame())
    assert tier2 == "roi"
    # ROI top-left was clipped to (0, 0) around (100, 100): offset adds nothing
    assert ball2 is not None


def test_tiled_tier_covers_whole_frame():
    def script(image, classes):
        h, w = image.shape[:2]
        if (h, w) == (720, 1280):
            return ([], [], [])
        # detect the ball only in the tile containing frame point (1200, 650)
        return ([], [], [])

    f, _ = _finder(script)
    # verify the tile grid covers the far corner of the frame
    tiles = []
    orig = f._ball_candidates

    def spy(image, imgsz, offset=(0.0, 0.0)):
        tiles.append((offset, image.shape[:2]))
        return orig(image, imgsz, offset)

    f._ball_candidates = spy
    f._tiled_candidates(_frame())
    covered_x = max(off[0] + shape[1] for off, shape in tiles)
    covered_y = max(off[1] + shape[0] for off, shape in tiles)
    assert covered_x >= 1280 and covered_y >= 720
    # adjacent tiles overlap by at least the configured amount
    xs = sorted({off[0] for off, _ in tiles})
    assert all(b - a <= f.tile_wh[0] - f.tile_overlap for a, b in zip(xs, xs[1:]))


def test_temporal_gate_rejects_far_jump():
    state = {"n": 0}

    def script(image, classes):
        state["n"] += 1
        if image.shape[:2] != (720, 1280):
            return ([], [], [])
        if state["n"] == 1:
            return ([[95, 95, 105, 105]], [0.4], [BALL_CLASS_ID])
        # next full frame: only a false positive across the pitch (white sock)
        return ([[1195, 595, 1205, 605]], [0.9], [BALL_CLASS_ID])

    f, _ = _finder(script)
    _, _, tier1 = f.process_frame(_frame())
    assert tier1 == "full"
    ball2, _, tier2 = f.process_frame(_frame())
    # 1100+ px jump >> max_jump_px -> rejected everywhere -> miss
    assert tier2 == "miss" and ball2 is None
    assert f.tier_counts["miss"] == 1


def test_gate_accepts_speed_scaled_jump_after_misses():
    f, _ = _finder(lambda image, classes: ([], [], []))
    f._last_ball_xy = (100.0, 100.0)
    f._missed = 4  # allowed = 60 * 5 = 300 px
    got = f._gate([(np.array([340.0, 100.0, 350.0, 110.0]), 0.5)])  # ~245 px away
    assert got is not None


def test_dedicated_ball_model_uses_class_zero_and_own_pass():
    person_calls, ball_calls = [], []

    class PersonModel(StubModel):
        pass

    def person_script(image, classes):
        person_calls.append(classes)
        return ([[400, 300, 430, 380]], [0.9], [0])

    def ball_script(image, classes):
        ball_calls.append(classes)
        return ([[95, 95, 105, 105]], [0.4], [0])  # class 0 = ball in fine-tune

    pm, bm = StubModel(person_script), StubModel(ball_script)
    f = BallFinder(_models=(pm, bm))
    assert f.ball_class_id == 0
    ball, players, tier = f.process_frame(_frame())
    assert tier == "full"
    np.testing.assert_allclose(ball, [95, 95, 105, 105])
    assert len(players) == 1
    assert pm.calls and bm.calls  # both models ran their own pass
