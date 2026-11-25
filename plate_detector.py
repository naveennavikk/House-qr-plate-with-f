"""Detects whether any QR plate is visible in a property image.

The matcher can fail to locate small, distant plates, so on its own it cannot
tell "plate not found" from "plate not shown". This YOLO detector, trained on
property photos with plates located by the matcher (training/train_plate_detector.py),
answers the narrower question "is there a QR plate anywhere in this photo?".
"""

import threading
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "plate_detector.pt"


class PlateDetector:
    def __init__(self, model_path=DEFAULT_MODEL, device="cpu", imgsz=1024):
        self.model = YOLO(str(model_path))
        self.device = device
        self.imgsz = imgsz
        # Ultralytics predictors are not thread-safe; the API calls this from a pool.
        self._lock = threading.Lock()

    def detect(self, image: Image.Image, min_conf=0.05):
        """Return [(confidence, [x1, y1, x2, y2]), ...] sorted by confidence."""
        rgb = np.asarray(image.convert("RGB"))[:, :, ::-1]  # ultralytics expects BGR arrays
        with self._lock:
            result = self.model.predict(rgb, imgsz=self.imgsz, conf=min_conf, device=self.device, verbose=False)[0]
        boxes = [
            (float(c), [float(v) for v in xyxy])
            for c, xyxy in zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist())
        ]
        return sorted(boxes, key=lambda b: -b[0])

    def best_confidence(self, image: Image.Image) -> float:
        boxes = self.detect(image)
        return boxes[0][0] if boxes else 0.0
