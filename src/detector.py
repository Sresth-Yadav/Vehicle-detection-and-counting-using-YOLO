"""
detector.py
-----------
Thin wrapper around Ultralytics YOLO that runs detection + multi-object
tracking (ByteTrack) in one call and returns clean, easy-to-use results
filtered down to vehicle classes only.
"""

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from ultralytics import YOLO


@dataclass
class Track:
    """A single tracked vehicle detection in the current frame."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def centroid(self):
        cx = int((self.x1 + self.x2) / 2)
        cy = int((self.y1 + self.y2) / 2)
        return cx, cy


class VehicleDetector:
    """
    Loads a YOLO model once and exposes `track(frame)` which returns a
    list of `Track` objects for every vehicle detected in that frame,
    with a persistent track_id assigned by ByteTrack across frames.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        vehicle_classes: Dict[int, str] = None,
        confidence: float = 0.35,
        iou: float = 0.5,
        tracker_cfg: str = "bytetrack.yaml",
    ):
        self.model = YOLO(model_path)
        self.vehicle_classes = vehicle_classes or {
            2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
        }
        self.confidence = confidence
        self.iou = iou
        self.tracker_cfg = tracker_cfg
        self.class_ids = list(self.vehicle_classes.keys())

    def track(self, frame: np.ndarray) -> List[Track]:
        """
        Run detection + tracking on a single BGR frame.
        Returns a list of Track objects (empty list if nothing found).
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=self.class_ids,
            conf=self.confidence,
            iou=self.iou,
            tracker=self.tracker_cfg,
            verbose=False,
        )

        tracks: List[Track] = []
        if not results:
            return tracks

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return tracks

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), tid, c, cf in zip(xyxy, ids, cls, conf):
            class_name = self.vehicle_classes.get(int(c), str(int(c)))
            tracks.append(
                Track(
                    track_id=int(tid),
                    class_id=int(c),
                    class_name=class_name,
                    confidence=float(cf),
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                )
            )
        return tracks
