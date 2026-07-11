"""Manual per-clip calibration fallback (FR-008): click known markings.

This is the impure edge -- it opens an OpenCV window -- so it is kept out of
``calibration`` (which stays pure and unit-testable). Clicked pixel points are
saved to JSON so a clip's manual calibration is reproducible and auditable.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.domain.models import Calibration, Source
from src.domain.pitch import REFERENCE_POINTS
from src.geometry.calibration import calibrate_from_markings

# Ordered from most-to-least reliably visible in a corner broadcast view.
DEFAULT_CLICK_ORDER = [
    "near_post",
    "far_post",
    "goal_area_gl_left",
    "goal_area_gl_right",
    "goal_area_front_left",
    "goal_area_front_right",
    "pen_area_gl_left",
    "pen_area_gl_right",
    "pen_area_front_left",
    "pen_area_front_right",
    "penalty_spot",
]


def manual_calibrate(
    frame: np.ndarray,
    point_names: list[str] | None = None,
    save_path: str | Path | None = None,
    window: str = "Manual calibration",
) -> Calibration:
    """Guided click-to-calibrate on a single frame.

    Cycles through ``point_names`` (default :data:`DEFAULT_CLICK_ORDER`). For each
    prompted marking either left-click its location or press ``n`` to skip it.
    Keys: ``u`` undo last, ``s`` solve+save (needs >= 4 clicked), ``q``/Esc abort.
    """
    names = point_names or DEFAULT_CLICK_ORDER
    for n in names:
        if n not in REFERENCE_POINTS:
            raise KeyError(f"unknown reference point: {n}")

    clicked: dict[str, tuple[float, float]] = {}
    order: list[str] = []          # names in click order, for undo
    idx = {"i": 0}                 # index into `names` (mutable for callback)
    base = frame.copy()

    def _redraw() -> np.ndarray:
        img = base.copy()
        for name, (u, v) in clicked.items():
            cv2.circle(img, (int(u), int(v)), 5, (0, 0, 255), -1)
            cv2.putText(img, name, (int(u) + 6, int(v) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        target = names[idx["i"]] if idx["i"] < len(names) else "(all done)"
        cv2.putText(img, f"Click: {target}   [{len(clicked)} set]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, "n skip  u undo  s save  q quit",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return img

    def _on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and idx["i"] < len(names):
            name = names[idx["i"]]
            clicked[name] = (float(x), float(y))
            order.append(name)
            idx["i"] += 1
            cv2.imshow(window, _redraw())

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)
    cv2.imshow(window, _redraw())

    try:
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("n") and idx["i"] < len(names):
                idx["i"] += 1
                cv2.imshow(window, _redraw())
            elif key == ord("u") and order:
                last = order.pop()
                clicked.pop(last, None)
                idx["i"] = names.index(last)
                cv2.imshow(window, _redraw())
            elif key == ord("s") and len(clicked) >= 4:
                break
            elif key in (ord("q"), 27):
                raise KeyboardInterrupt("manual calibration aborted")
    finally:
        cv2.destroyWindow(window)

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"pixel_points": clicked}, indent=2), encoding="utf-8")

    return calibrate_from_markings(clicked, source=Source.MANUAL)


def calibration_from_saved(path: str | Path) -> Calibration:
    """Rebuild a manual calibration from a previously-saved click file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    points = {k: tuple(v) for k, v in data["pixel_points"].items()}
    return calibrate_from_markings(points, source=Source.MANUAL)
