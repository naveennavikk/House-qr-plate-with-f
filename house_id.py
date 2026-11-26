"""Reads the house ID stamped at the bottom-left of each photo (e.g. "ID: HPSBA-C118430").

The capture app writes four white lines into every photo: ID, LAT, LONG and the
date. RapidOCR (PP-OCR models on onnxruntime) reads the bottom-left area and the
"HPSBA-<letter><digits>" line is taken as the house ID. It read 762/762
labelled photos correctly (Accept + every rejection folder).

OCR reads the series letter "I" as "1" (and "O" as "0"), and some older IDs
have no series letter at all ("HPSBA14240"), so IDs are compared with
comparison_key(), which treats I/1 and O/0 as the same character.
"""

import os
import re
import threading

import numpy as np
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR

# CPU threads per OCR engine. The OCR runs alongside the matcher, so a small
# number avoids the two fighting over cores (QR_OCR_THREADS overrides; 0 = all).
OCR_THREADS = int(os.getenv("QR_OCR_THREADS", "2"))

ID_RE = re.compile(r"HPSBA-?([A-Z0-9|])-?(\d{3,8})")


def normalize_id(value) -> str | None:
    """'hpsba - c 118430' / 'HPSBA-C118430' -> 'HPSBA-C118430'; None if not an HPSBA ID."""
    if value is None:
        return None
    text = re.sub(r"[\s.,'`:_~]", "", str(value).upper())
    m = ID_RE.search(text)
    return f"HPSBA-{m.group(1).replace('|', 'I')}{m.group(2)}" if m else None


def comparison_key(house_id: str) -> str:
    """Key for comparing IDs: drops the prefix/dash and folds I->1, O->0."""
    return house_id.removeprefix("HPSBA-").translate(str.maketrans("IO", "10"))


class HouseIdReader:
    def __init__(self):
        # One OCR engine per thread, so both photos can be read in parallel.
        self._local = threading.local()

    def _engine(self) -> RapidOCR:
        engine = getattr(self._local, "engine", None)
        if engine is None:
            kwargs = {"intra_op_num_threads": OCR_THREADS} if OCR_THREADS > 0 else {}
            engine = self._local.engine = RapidOCR(**kwargs)
        return engine

    def read(self, image: Image.Image) -> str | None:
        """Return the normalized house ID printed at the bottom-left, or None if unreadable."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        w, h = image.size
        crop = np.asarray(image.crop((0, int(h * 0.76), int(w * 0.7), int(h * 0.96))))[:, :, ::-1]
        # The stamped text is always upright, so the text-angle classifier is skipped.
        lines, _ = self._engine()(crop, use_cls=False)
        for _box, text, _conf in lines or []:
            house_id = normalize_id(text)
            if house_id:
                return house_id
        return None
