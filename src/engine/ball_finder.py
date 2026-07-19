"""Tiered ball detection: full-frame -> ROI re-check -> tiled sweep (FR-004/009).

Stock COCO YOLO at full-frame resolution misses a broadcast-footage ball most
of the time: at 1280x720 the ball is ~5-15 px, below the detectability floor
once the frame is downscaled to the model's input size, and a dead ball at the
corner arc sits half-occluded by the taker's feet (where agnostic NMS lets the
person box suppress it). This module fixes recall with three stacked, widely
used small-object techniques (see docs/ links in third_party/README.md's
ball-detection note):

1. **Full-frame pass** at native-or-better ``imgsz`` with a *low* ball
   confidence floor and non-agnostic NMS (players keep their own pass).
2. **ROI re-check**: if the full frame missed and we have a prior (last
   accepted ball, Kalman prediction, or a fixed seed such as the corner arc),
   re-run the model on a small crop around it -- the ball becomes ~4x larger
   in effective resolution for the cost of one tiny inference.
3. **Tiled sweep**: if still nothing, slice the frame into overlapping tiles
   and detect per tile (each tile is upsampled to the model input, which is
   the whole trick). Slowest tier, only runs on miss frames.

A **single-ball temporal gate** filters every candidate: the accepted ball
must be within a speed-scaled jump radius of the last accepted position,
which absorbs the white-sock/head/line false positives that a low confidence
floor lets through. Candidates that fail the gate are dropped, the frame
counts as a miss, and the Kalman smoother downstream extrapolates
(``position_source`` provenance is unchanged).

The class is model-agnostic: anything with the ultralytics ``model(frame,
**kwargs)`` call contract works. A ball-specific fine-tune (recommended:
roboflow/sports ``football-ball-detection.pt``, a broadcast-soccer YOLOv8x --
see ``scripts/fetch_ball_weights.sh``) is passed via ``ball_model_path`` and
handles the ball tiers, while the stock COCO model keeps detecting players;
without it, the stock model's COCO "sports ball" class is used for both.
"""

from __future__ import annotations

import numpy as np

BALL_CLASS_ID = 32       # COCO "sports ball" (stock model)
PERSON_CLASS_ID = 0      # COCO "person"
BALL_MODEL_CLASS_ID = 0  # dedicated single-class ball fine-tune


class BallFinder:
    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        ball_model_path: str | None = None,
        device: str = "cpu",
        imgsz: int = 1280,
        ball_conf: float = 0.05,
        person_conf: float = 0.25,
        roi_size: int = 320,
        tile_wh: tuple[int, int] = (640, 460),
        tile_overlap: int = 100,
        max_jump_px: float = 60.0,
        person_class_id: int = PERSON_CLASS_ID,
        _models: tuple | None = None,  # test seam: (person_model, ball_model)
    ) -> None:
        if str(device).isdigit():
            device = f"cuda:{device}"
        if _models is not None:
            self.model, self.ball_model = _models
        else:
            from ultralytics import YOLO  # lazy heavy import, matches detector.py

            self.model = YOLO(model_path)
            self.model.to(device)
            if ball_model_path is not None:
                self.ball_model = YOLO(ball_model_path)
                self.ball_model.to(device)
            else:
                self.ball_model = self.model
        # a dedicated fine-tune is single-class (0 = ball); the stock COCO
        # model shares weights with the person pass and uses class 32
        self.ball_class_id = BALL_MODEL_CLASS_ID if ball_model_path or (
            _models is not None and _models[1] is not _models[0]
        ) else BALL_CLASS_ID
        self.device = device
        self.imgsz = imgsz
        self.ball_conf = ball_conf
        self.person_conf = person_conf
        self.roi_size = roi_size
        self.tile_wh = tile_wh
        self.tile_overlap = tile_overlap
        self.max_jump_px = max_jump_px
        self.person_class_id = person_class_id
        self._last_ball_xy: tuple[float, float] | None = None
        self._missed = 0
        self.tier_counts = {"full": 0, "roi": 0, "tiled": 0, "miss": 0}

    # --- raw inference ---------------------------------------------------

    def _infer(self, model, image: np.ndarray, imgsz: int, conf: float, classes: list[int]):
        """One model call -> (boxes_xyxy (N,4), confs (N,), class_ids (N,))."""
        res = model(
            image,
            classes=classes,
            conf=conf,
            imgsz=imgsz,
            agnostic_nms=False,  # a person box must not suppress the ball
            verbose=False,
            device=self.device,
        )[0]
        b = res.boxes
        if b is None or len(b) == 0:
            return (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
        return (
            b.xyxy.cpu().numpy(),
            b.conf.cpu().numpy(),
            b.cls.cpu().numpy().astype(int),
        )

    def _ball_candidates(self, image: np.ndarray, imgsz: int, offset=(0.0, 0.0)):
        """Ball boxes from one inference, offset back into frame coordinates."""
        xyxy, conf, cls = self._infer(
            self.ball_model, image, imgsz, self.ball_conf, [self.ball_class_id]
        )
        mask = cls == self.ball_class_id
        xyxy = xyxy[mask] + np.array([*offset, *offset])
        return list(zip(xyxy, conf[mask]))

    # --- detection tiers --------------------------------------------------

    def _roi_candidates(self, frame: np.ndarray, center: tuple[float, float]):
        h, w = frame.shape[:2]
        half = self.roi_size // 2
        cx = int(np.clip(center[0], half, max(half, w - half)))
        cy = int(np.clip(center[1], half, max(half, h - half)))
        x0, y0 = cx - half, cy - half
        crop = frame[y0:y0 + self.roi_size, x0:x0 + self.roi_size]
        if crop.size == 0:
            return []
        return self._ball_candidates(crop, imgsz=640, offset=(x0, y0))

    def _tiled_candidates(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        tw, th = self.tile_wh
        step_x = max(1, tw - self.tile_overlap)
        step_y = max(1, th - self.tile_overlap)
        cands = []
        for y0 in range(0, max(1, h - self.tile_overlap), step_y):
            for x0 in range(0, max(1, w - self.tile_overlap), step_x):
                x1, y1 = min(x0 + tw, w), min(y0 + th, h)
                tile = frame[y0:y1, x0:x1]
                if tile.shape[0] < 32 or tile.shape[1] < 32:
                    continue
                cands.extend(self._ball_candidates(tile, imgsz=640, offset=(x0, y0)))
        return cands

    # --- temporal gate ----------------------------------------------------

    def _gate(self, candidates) -> tuple[np.ndarray, float] | None:
        """Best candidate consistent with the recent ball position, or None.

        With no prior, the highest-confidence candidate is accepted (track
        start). With a prior, the allowed jump grows with consecutive misses
        (the ball keeps moving while undetected).
        """
        if not candidates:
            return None
        if self._last_ball_xy is None:
            return max(candidates, key=lambda c: c[1])
        lx, ly = self._last_ball_xy
        allowed = self.max_jump_px * (1 + self._missed)
        gated = []
        for box, conf in candidates:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            dist = float(np.hypot(cx - lx, cy - ly))
            if dist <= allowed:
                gated.append((box, conf, dist))
        if not gated:
            return None
        # prefer confident candidates, tie-broken toward the prior
        box, conf, _ = max(gated, key=lambda g: (g[1], -g[2]))
        return box, conf

    # --- public API ---------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray, prior_xy: tuple[float, float] | None = None
    ) -> tuple[np.ndarray | None, np.ndarray, str]:
        """Detect ball + players in one frame.

        ``prior_xy`` is an optional external ball-position hint for the ROI
        tier (e.g. the Kalman prediction from ``BallSmoother``, or the corner
        arc's pixel position before the kick). Returns ``(ball_bbox_xyxy |
        None, players_xyxy (N, 4), tier)`` with ``tier`` in
        ``full/roi/tiled/miss`` for diagnostics.
        """
        if self.ball_model is self.model:
            # shared COCO model: players + ball in one full-frame pass
            xyxy, conf, cls = self._infer(
                self.model, frame, self.imgsz, min(self.ball_conf, self.person_conf),
                [self.person_class_id, self.ball_class_id],
            )
            players = xyxy[(cls == self.person_class_id) & (conf >= self.person_conf)]
            ball_mask = cls == self.ball_class_id
            candidates = list(zip(xyxy[ball_mask], conf[ball_mask]))
        else:
            # dedicated ball model: players from the stock model, ball from its own pass
            xyxy, conf, cls = self._infer(
                self.model, frame, self.imgsz, self.person_conf, [self.person_class_id]
            )
            players = xyxy[cls == self.person_class_id]
            candidates = self._ball_candidates(frame, imgsz=self.imgsz)

        best = self._gate(candidates)
        tier = "full"
        if best is None:
            roi_center = prior_xy or self._last_ball_xy
            if roi_center is not None:
                best = self._gate(self._roi_candidates(frame, roi_center))
                tier = "roi"
        if best is None:
            best = self._gate(self._tiled_candidates(frame))
            tier = "tiled"

        if best is None:
            self._missed += 1
            self.tier_counts["miss"] += 1
            return None, players, "miss"

        box, _ = best
        self._last_ball_xy = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        self._missed = 0
        self.tier_counts[tier] += 1
        return np.asarray(box, dtype=np.float64), players, tier
