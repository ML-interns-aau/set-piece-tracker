"""PnLCalib adapter: learned pitch calibration -> our Calibration (FR-008, I7).

Corner-kick footage packs defenders onto the penalty-area lines, so classical
line/marking detection (see ``auto_calibration``) cannot recover the box. The
vendored PnLCalib model (``third_party/PnLCalib``, HRNet keypoint + line
detection) is robust to that clutter: it localises pitch keypoints across the
whole field and solves a camera model, from which we take the ground-plane
homography.

This module is the *impure edge*: it pulls in torch and the vendored (GPL-2.0)
PnLCalib code lazily, so the pure geometry/domain packages stay importable
without those heavy dependencies. Its output is converted into our own
``Calibration`` in the ``pitch.py`` metric convention (analysed goal at x = 0),
identical to what ``calibration``/``auto_calibration`` produce, so nothing
downstream needs to know which calibrator was used.

Setup (not committed): the ~506 MB weights live in ``third_party/PnLCalib/weights``
(``SV_kp``, ``SV_lines``); see ``third_party/PnLCalib/README.md``. Requires torch,
torchvision, scipy, shapely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.domain.models import Calibration, CalibrationTrack, Source
from src.domain.pitch import (
    PENALTY_AREA_DEPTH_M,
    PENALTY_AREA_WIDTH_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    _CENTRE_Y,
)

_PNL_DIR = Path(__file__).resolve().parents[2] / "third_party" / "PnLCalib"
_WEIGHTS_DIR = _PNL_DIR / "weights"

# PnLCalib's published single-view detection thresholds.
KP_THRESHOLD = 0.3434
LINE_THRESHOLD = 0.7867
# Reject a calibration whose own reprojection error (pixels, on the model's
# detected correspondences) exceeds this -- a coarse trust gate on top of the
# model already returning None when it cannot calibrate at all.
MAX_REPROJ_PX = 30.0

_models: tuple | None = None  # lazy singleton: (model_kp, model_line, pnl_inference_module, device)


def _load_models() -> tuple:
    """Lazily build the HRNet models and load weights (once per process)."""
    global _models
    if _models is not None:
        return _models

    import sys
    import yaml
    import torch
    import torchvision.transforms as T

    if not (_WEIGHTS_DIR / "SV_kp").exists() or not (_WEIGHTS_DIR / "SV_lines").exists():
        raise FileNotFoundError(
            f"PnLCalib weights not found in {_WEIGHTS_DIR}. Download SV_kp and SV_lines "
            "from the PnLCalib v1.0.0 GitHub release (see third_party/PnLCalib/README.md)."
        )
    if str(_PNL_DIR) not in sys.path:
        sys.path.insert(0, str(_PNL_DIR))

    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l
    import inference as pnl_inf

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # These are defined only under PnLCalib's __main__; inject them for library use.
    pnl_inf.transform2 = T.Resize((540, 960))
    pnl_inf.device = device

    cfg = yaml.safe_load(open(_PNL_DIR / "config" / "hrnetv2_w48.yaml"))
    cfg_l = yaml.safe_load(open(_PNL_DIR / "config" / "hrnetv2_w48_l.yaml"))

    model_kp = get_cls_net(cfg)
    model_kp.load_state_dict(torch.load(str(_WEIGHTS_DIR / "SV_kp"), map_location=device))
    model_kp.to(device).eval()

    model_line = get_cls_net_l(cfg_l)
    model_line.load_state_dict(torch.load(str(_WEIGHTS_DIR / "SV_lines"), map_location=device))
    model_line.to(device).eval()

    _models = (model_kp, model_line, pnl_inf, device)
    return _models


def _ground_homography(P: np.ndarray) -> np.ndarray:
    """Ground-plane (Z=0) homography [Xc, Yc, 1] (centred world) -> image pixels."""
    return np.column_stack([P[:, 0], P[:, 1], P[:, 3]]).astype(np.float64)


def _decide_use_near(Hwi: np.ndarray, frame_shape: tuple[int, int]) -> bool:
    """Which goal is in view? True => world X=0 goal is the analysed goal (our x=0).

    PnLCalib calibrates the whole 105 x 68 m pitch (a goal at X=0 and X=105); we
    anchor our metric frame on whichever goal the corner clip actually shows.
    """
    half_l = PITCH_LENGTH_M / 2.0

    def _project(xc: float, yc: float) -> np.ndarray:
        p = Hwi @ np.array([xc, yc, 1.0])
        return p[:2] / p[2] if abs(p[2]) > 1e-9 else np.array([1e9, 1e9])

    h, w = frame_shape
    near_goal, far_goal = _project(-half_l, 0.0), _project(+half_l, 0.0)

    def _inside(pt: np.ndarray) -> bool:
        return -0.5 * w < pt[0] < 1.5 * w and -0.5 * h < pt[1] < 1.5 * h

    near_in, far_in = _inside(near_goal), _inside(far_goal)
    if near_in != far_in:
        return near_in
    centre = np.array([w / 2.0, h / 2.0])   # both/neither: goal closest to frame centre
    return np.hypot(*(near_goal - centre)) <= np.hypot(*(far_goal - centre))


def _calibration_from_projection(
    P: np.ndarray,
    frame_shape: tuple[int, int],
    rep_err_px: float,
    points_used: int,
    source: Source,
    use_near: bool,
) -> Calibration:
    """Build our pixel->metre ``Calibration`` from a projection, given the goal choice.

    ``use_near`` is passed in (not re-decided) so a whole clip shares one analysed-goal
    convention even as the camera moves. Inverts the ground-plane homography and
    remaps so the analysed goal sits at our x = 0 (``pitch.py`` convention).
    """
    half_l, half_w = PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M / 2.0
    h, w = frame_shape
    Hwi = _ground_homography(P)

    # M maps our metric (x, y, 1) -> centred world (Xc, Yc, 1) with analysed goal at x=0.
    if use_near:                          # analysed goal is world X=0: X=x, Y=y
        M = np.array([[1.0, 0.0, -half_l], [0.0, 1.0, -half_w], [0.0, 0.0, 1.0]])
    else:                                 # analysed goal is world X=105: X=105-x, Y=68-y
        M = np.array([[-1.0, 0.0, half_l], [0.0, -1.0, half_w], [0.0, 0.0, 1.0]])

    H = np.linalg.inv(Hwi @ M)
    H = H / H[2, 2]

    # Pixel reprojection error -> metres via the local scale of H at the image centre.
    from src.geometry.calibration import apply_homography
    c = np.array([[w / 2.0, h / 2.0], [w / 2.0 + 1.0, h / 2.0]])
    cm = apply_homography(H, c)
    m_per_px = float(np.hypot(*(cm[1] - cm[0]))) or 0.0
    return Calibration(H=H, reprojection_error_m=rep_err_px * m_per_px,
                       points_used=points_used, source=source)


def _projection_to_calibration(
    P: np.ndarray,
    frame_shape: tuple[int, int],
    rep_err_px: float,
    points_used: int,
    source: Source,
) -> Calibration:
    """Single-frame convenience: decide the analysed goal from this frame, then build."""
    use_near = _decide_use_near(_ground_homography(P), frame_shape)
    return _calibration_from_projection(P, frame_shape, rep_err_px, points_used, source, use_near)


def pnl_calibrate(frame: np.ndarray, source: Source = Source.AUTO) -> Calibration | None:
    """Calibrate one frame with PnLCalib. Returns ``None`` if it cannot be trusted."""
    calib, _ = pnl_calibrate_detailed(frame, source)
    return calib


def _calibrate_frame_raw(frame: np.ndarray) -> dict | None:
    """Run PnLCalib on one frame; return raw model output or ``None`` if untrusted.

    Returns ``{cam_params, rep_err_px, n_keypoints}`` where ``cam_params`` is
    PnLCalib's camera model (rotation matrix, focal, principal point, position) --
    the low-dimensional form we interpolate across a clip.
    """
    model_kp, model_line, pnl_inf, _device = _load_models()
    from utils.utils_calib import FramebyFrameCalib

    h, w = frame.shape[:2]
    cam = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)
    params = pnl_inf.inference(cam, frame, model_kp, model_line,
                               KP_THRESHOLD, LINE_THRESHOLD, True)
    n_keypoints = len(getattr(cam, "keypoints_dict", {}) or {})
    if params is None:
        return None
    rep_err_px = float(params.get("rep_err", float("nan")))
    if not np.isfinite(rep_err_px) or rep_err_px > MAX_REPROJ_PX:
        return None
    return {"cam_params": params["cam_params"], "rep_err_px": rep_err_px,
            "n_keypoints": n_keypoints, "mode": params.get("mode")}


def pnl_calibrate_detailed(
    frame: np.ndarray,
    source: Source = Source.AUTO,
) -> tuple[Calibration | None, dict]:
    """Like :func:`pnl_calibrate` but also returns model metadata for debugging."""
    _model_kp, _model_line, pnl_inf, _device = _load_models()
    raw = _calibrate_frame_raw(frame)
    info: dict = {
        "n_keypoints": None if raw is None else raw["n_keypoints"],
        "rep_err_px": None if raw is None else raw["rep_err_px"],
        "mode": None if raw is None else raw["mode"],
    }
    if raw is None:
        return None, info

    P = pnl_inf.projection_from_cam_params({"cam_params": raw["cam_params"]})
    calib = _projection_to_calibration(P, frame.shape[:2], raw["rep_err_px"],
                                       raw["n_keypoints"], source)
    return calib, info


# --- clip-level (moving-camera) calibration -------------------------------------

# Defaults (pilot; not final). Static/cut are judged by how far the *projected pitch
# markings* move between adjacent keyframes (pixels) -- captures pan, zoom and
# translation together, which a rotation-only test misses.
KEYFRAME_STRIDE = 1          # per-frame calibration by default (clips are short); set
                             # stride > 1 to calibrate every Nth frame and interpolate
                             # the gaps (cheaper, for long clips / batch runs).
STATIC_TOL_PX = 4.0          # marking shift below this across all frames => treat as static
                             # (one shared H, which also removes per-frame jitter)
DISCONTINUITY_TOL_PX = 250.0 # marking jump above this between adjacent keyframes => cut


def _params_to_calibration(cam_params, pnl_inf, frame_shape, rep_err_px, n_kp, source, use_near):
    P = pnl_inf.projection_from_cam_params({"cam_params": cam_params})
    return _calibration_from_projection(P, frame_shape, rep_err_px, n_kp, source, use_near)


def build_calibration_track(
    clip_path,
    frame_indices=None,
    stride: int = KEYFRAME_STRIDE,
    static_tol_px: float = STATIC_TOL_PX,
    discontinuity_tol_px: float = DISCONTINUITY_TOL_PX,
    source: Source = Source.AUTO,
) -> CalibrationTrack | None:
    """Calibrate a whole clip per frame, handling camera pan/zoom (and cuts).

    With the default ``stride=1`` every frame is calibrated directly by PnLCalib
    (best accuracy; fine for the short corner clips). With ``stride > 1`` only every
    Nth frame is calibrated and the gaps are filled by interpolating the camera
    model between the bracketing keyframes -- **SLERP** on the rotation, linear on
    focal length / principal point / position (the correct way to interpolate a
    camera pose) -- which is cheaper for long clips or batch runs.

    A genuinely static clip (projected markings move < ``static_tol_px`` across all
    frames) collapses to a single shared homography, which also removes per-frame
    jitter. A marking jump beyond ``discontinuity_tol_px`` between adjacent
    keyframes is treated as a camera cut: calibrations are not interpolated across
    it and the boundary frame is flagged (feeds the unusable:discontinuity check).
    Returns ``None`` if no frame could be calibrated.
    """
    import cv2
    from scipy.spatial.transform import Rotation, Slerp

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_shape = (h, w)

    wanted = (sorted({int(f) for f in frame_indices}) if frame_indices is not None
              else list(range(n_frames)))
    if not wanted:
        cap.release()
        return None

    lo, hi = wanted[0], wanted[-1]
    kf_indices = list(range(lo, hi + 1, max(1, stride)))
    if kf_indices[-1] != hi:
        kf_indices.append(hi)

    # --- calibrate keyframes ---
    kfs: list[dict] = []
    _mk, _ml, pnl_inf, _dev = _load_models()
    for idx in kf_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        raw = _calibrate_frame_raw(frame)
        if raw is None:
            continue
        cp = raw["cam_params"]
        P = pnl_inf.projection_from_cam_params({"cam_params": cp})
        kfs.append({
            "idx": idx,
            "cam_params": cp,
            "R": Rotation.from_matrix(np.array(cp["rotation_matrix"], dtype=np.float64)),
            "focal": np.array([cp["x_focal_length"], cp["y_focal_length"]], dtype=np.float64),
            "pp": np.array(cp["principal_point"], dtype=np.float64),
            "pos": np.array(cp["position_meters"], dtype=np.float64),
            "rep_px": raw["rep_err_px"],
            "n_kp": raw["n_keypoints"],
            "use_near": _decide_use_near(_ground_homography(P), frame_shape),
        })
    cap.release()
    if not kfs:
        return None

    # analysed-goal convention fixed once for the whole clip (majority vote)
    use_near = sum(k["use_near"] for k in kfs) >= (len(kfs) / 2.0)

    # Per-keyframe homography (fixed convention), used both for stability analysis
    # and directly for non-interpolated frames.
    for k in kfs:
        k["calib"] = _params_to_calibration(k["cam_params"], pnl_inf, frame_shape,
                                            k["rep_px"], k["n_kp"], source, use_near)

    # Stability = how far the projected pitch markings move between keyframes (px).
    _half_pa = PENALTY_AREA_WIDTH_M / 2.0
    _probes = np.array([
        [0.0, _CENTRE_Y - _half_pa], [0.0, _CENTRE_Y + _half_pa],
        [PENALTY_AREA_DEPTH_M, _CENTRE_Y - _half_pa], [PENALTY_AREA_DEPTH_M, _CENTRE_Y + _half_pa],
        [0.0, _CENTRE_Y], [PENALTY_AREA_DEPTH_M, _CENTRE_Y],
    ], dtype=np.float64)

    def _project_markings(calib: Calibration) -> np.ndarray:
        Hinv = np.linalg.inv(calib.H)
        hom = np.c_[_probes, np.ones(len(_probes))] @ Hinv.T
        return hom[:, :2] / hom[:, 2:3]

    proj = [_project_markings(k["calib"]) for k in kfs]

    def _shift(i: int, j: int) -> float:
        return float(np.max(np.linalg.norm(proj[i] - proj[j], axis=1)))

    shifts = [_shift(i, i + 1) for i in range(len(kfs) - 1)]
    max_shift = max(shifts, default=0.0)

    # --- static fast-path ---
    if len(kfs) == 1 or max_shift < static_tol_px:
        best = min(range(len(kfs)), key=lambda i: kfs[i]["rep_px"])
        return CalibrationTrack(per_frame={f: kfs[best]["calib"] for f in wanted}, static=True)

    # --- discontinuities (camera cuts): segment id per keyframe ---
    seg_id = [0] * len(kfs)
    discontinuities: list[int] = []
    for i in range(1, len(kfs)):
        if shifts[i - 1] > discontinuity_tol_px:
            seg_id[i] = seg_id[i - 1] + 1
            discontinuities.append(kfs[i]["idx"])
        else:
            seg_id[i] = seg_id[i - 1]

    idxs = [k["idx"] for k in kfs]

    def _bracket(f: int):
        """Return (a, b) keyframe dicts bracketing frame f, and whether to interpolate."""
        import bisect
        pos = bisect.bisect_right(idxs, f)
        a = kfs[pos - 1] if pos > 0 else None
        b = kfs[pos] if pos < len(kfs) else None
        if a is None:
            return b, b, False
        if b is None:
            return a, a, False
        if seg_id[idxs.index(a["idx"])] != seg_id[idxs.index(b["idx"])]:
            # across a cut: snap to the nearer keyframe, do not interpolate
            return (a, a, False) if (f - a["idx"]) <= (b["idx"] - f) else (b, b, False)
        return a, b, a["idx"] != b["idx"]

    per_frame: dict[int, Calibration] = {}
    for f in wanted:
        a, b, interp = _bracket(f)
        if not interp:
            per_frame[f] = a["calib"]   # exact keyframe (or snapped across a cut)
            continue
        t = (f - a["idx"]) / (b["idx"] - a["idx"])
        R = Slerp([a["idx"], b["idx"]], Rotation.concatenate([a["R"], b["R"]]))([f])[0]
        focal = (1 - t) * a["focal"] + t * b["focal"]
        pp = (1 - t) * a["pp"] + t * b["pp"]
        pos = (1 - t) * a["pos"] + t * b["pos"]
        cp = {
            "rotation_matrix": R.as_matrix().tolist(),
            "x_focal_length": float(focal[0]), "y_focal_length": float(focal[1]),
            "principal_point": pp.tolist(), "position_meters": pos.tolist(),
        }
        rep = (1 - t) * a["rep_px"] + t * b["rep_px"]
        per_frame[f] = _params_to_calibration(cp, pnl_inf, frame_shape, rep,
                                              a["n_kp"], source, use_near)

    return CalibrationTrack(per_frame=per_frame, static=False,
                            discontinuity_frames=tuple(discontinuities))
