"""Ball position smoother: Kalman filter + optical-flow divergence guard.

Adapted from the general-purpose football-tracker ball tracker (the fast,
frequently-occluded small-object case it was built for). Cleaned up and
decoupled from ``supervision``: ``update`` takes a plain bbox (or ``None`` when
the ball wasn't detected this frame) and returns a smoothed pixel position plus
a ``predicted`` flag.

This produces the *pixel* ball path; mapping it through the calibration
homography (``geometry.calibration.apply_homography``) yields the metric
pitch-plane ball track that ``geometry.trajectory`` consumes (interface I8).
The ``predicted`` flag mirrors the detected/extrapolated provenance rule.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class BallSmoother:
    def __init__(
        self,
        max_trail: int = 25,
        max_missed: int = 30,
        diverge_px: float = 150.0,
    ) -> None:
        self.max_trail = max_trail
        self.max_missed = max_missed
        self.diverge_px = diverge_px
        self._kf = self._build_kalman()
        self._initialised = False
        self._missed_count = 0
        self.trail: deque[tuple[float, float]] = deque(maxlen=max_trail)
        self._prev_gray: np.ndarray | None = None

    @staticmethod
    def _build_kalman() -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32)
        dt = 1.0
        kf.transitionMatrix = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        return kf

    def _optical_flow_estimate(self, gray: np.ndarray) -> tuple[float, float] | None:
        if self._prev_gray is None or len(self.trail) == 0:
            return None
        last = np.array([[list(self.trail[-1])]], dtype=np.float32)
        try:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, last, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
        except cv2.error:
            return None
        if p1 is not None and st is not None and st[0][0] == 1:
            nx, ny = p1[0][0]
            return float(nx), float(ny)
        return None

    def update(
        self,
        frame: np.ndarray,
        ball_bbox_xyxy: np.ndarray | tuple[float, float, float, float] | None,
    ) -> tuple[float, float, bool]:
        """Advance one frame. Returns (cx, cy, predicted).

        ``predicted`` is True when the position was estimated (Kalman/optical
        flow) rather than measured from a detection this frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if ball_bbox_xyxy is not None:
            x1, y1, x2, y2 = (float(v) for v in ball_bbox_xyxy)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            if not self._initialised:
                self._kf.statePre = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
                self._kf.statePost = self._kf.statePre.copy()
                self._initialised = True
            else:
                self._kf.predict()
                self._kf.correct(measurement)
            self._missed_count = 0
            self.trail.append((cx, cy))
            self._prev_gray = gray
            return cx, cy, False

        self._missed_count += 1
        if self._missed_count > self.max_missed:
            self._initialised = False
            self._prev_gray = gray
            if self.trail:
                return (*self.trail[-1], True)
            return 0.0, 0.0, True

        if self._initialised:
            pred = self._kf.predict()
            px, py = float(pred[0][0]), float(pred[1][0])
            if self.trail:
                lx, ly = self.trail[-1]
                if np.hypot(px - lx, py - ly) > self.diverge_px:
                    of = self._optical_flow_estimate(gray)
                    if of is not None:
                        px, py = of
            self.trail.append((px, py))
            self._prev_gray = gray
            return px, py, True

        of = self._optical_flow_estimate(gray)
        self._prev_gray = gray
        if of is not None:
            self.trail.append(of)
            return (*of, True)
        if self.trail:
            return (*self.trail[-1], True)
        return 0.0, 0.0, True

    def get_trail(self) -> list[tuple[float, float]]:
        return list(self.trail)
