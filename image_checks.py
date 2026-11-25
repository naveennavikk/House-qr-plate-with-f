"""Extra inspector check: sideways (horizontally shot) photos.

A small YOLO classifier trained by training/train_image_checks.py:

* orientation_cls.pt (upright / sideways): photos are saved as portrait files,
  but when the phone was held sideways the scene inside is rotated 90 degrees.
  Only the top square of the photo is classified, so the ID/GPS watermark that
  the app stamps upright at the bottom does not affect the decision.
"""

import threading
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

MODELS = Path(__file__).resolve().parent / "models"


class _Classifier:
    def __init__(self, path, positive, imgsz, device):
        self.model = YOLO(str(path))
        self.index = list(self.model.names.values()).index(positive)
        self.imgsz = imgsz
        self.device = device
        self._lock = threading.Lock()  # ultralytics predictors are not thread-safe

    def probability(self, image: Image.Image) -> float:
        with self._lock:
            probs = self.model.predict(image, imgsz=self.imgsz, device=self.device, verbose=False)[0].probs
        return float(probs.data[self.index])


class OrientationChecker(_Classifier):
    def __init__(self, device="cpu"):
        super().__init__(MODELS / "orientation_cls.pt", "sideways", 224, device)

    def sideways_probability(self, image: Image.Image) -> float:
        image = image.convert("RGB")
        side = min(image.size)
        top_square = image.crop((0, 0, side, side)).resize((320, 320), Image.Resampling.LANCZOS)
        return self.probability(top_square)
