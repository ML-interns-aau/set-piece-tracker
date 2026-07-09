import supervision as sv
from ultralytics import YOLO
import numpy as np
import pathlib
import platform
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
class FootballDetector:
    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        conf: float = 0.25,
        iou: float = 0.60,
        device: str = "cpu",
    ):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.CLASS_NAMES_DICT = self.model.model.names
        self.conf   = conf
        self.iou    = iou
        self.device = device
    def detect(self, frame: np.ndarray) -> sv.Detections:
        results = self.model(
            frame,
            classes=[0, 32],
            conf=self.conf,
            iou=self.iou,
            imgsz=960,
            agnostic_nms=True,
            verbose=False,
            device=self.device,
        )[0]
        return sv.Detections.from_ultralytics(results)
    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        detections = self.detect(frame)
        return detections[detections.class_id == 0]
    def detect_ball(self, frame: np.ndarray) -> sv.Detections:
        detections = self.detect(frame)
        return detections[detections.class_id == 32]
