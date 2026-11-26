"""Extra checks: sideways (horizontally shot) photos and plate close-ups sent
as the property image.

A small YOLO classifier trained by training/train_image_checks.py:

* orientation_cls.pt (upright / sideways): photos are saved as portrait files,
  but when the phone was held sideways the scene inside is rotated 90 degrees.
  Only the top square of the photo is classified, so the ID/GPS watermark that
  the app stamps upright at the bottom does not affect the decision.
"""

import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
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


def plate_closeup_fraction(image: Image.Image) -> float:
    """Width of a readable QR code as a fraction of the photo width (0 if none).

    A property photo shows the house with a small plate. When the plate
    close-up is sent as the property image (e.g. the two photos swapped), its
    QR code is large and readable. Only decodable, square QR boxes count:
    OpenCV also returns junk boxes on house photos, but those never decode.
    On held-out data, >= 0.15 caught 238/250 close-ups and 0/450 property photos.
    """
    gray = np.asarray(ImageOps.exif_transpose(image).convert("L"))
    best = 0.0
    for scale in (1.0, 0.5):
        scaled = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        try:
            text, points, _ = cv2.QRCodeDetector().detectAndDecode(scaled)
        except cv2.error:
            continue
        if not text or points is None:
            continue
        p = np.asarray(points).reshape(-1, 2) / scale
        w, h = p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min()
        inside = p.min() >= -2 and p[:, 0].max() <= gray.shape[1] + 2 and p[:, 1].max() <= gray.shape[0] + 2
        if inside and h > 0 and 0.75 <= w / h <= 1.33 and w / gray.shape[1] <= 0.8:
            best = max(best, float(w / gray.shape[1]))
    return best
