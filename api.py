"""HTTP API for connecting the QR plate workflow to an existing frontend."""

import base64
import binascii
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import BoundedSemaphore

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, RootModel
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

from house_checker import HouseDoorChecker
from matcher import HybridImageMatcher
from image_checks import OrientationChecker
from plate_detector import PlateDetector

APP_ROOT = Path(__file__).resolve().parent
MODEL_PATHS = [
    ("YOLO26n", APP_ROOT / "models" / "yolo26n_cls_best.pt"),
    ("YOLO11n", APP_ROOT / "models" / "yolo11n_cls_best.pt"),
]
MAX_IMAGE_BYTES = 25 * 1024 * 1024
# A property image wider than tall by this factor counts as horizontal (landscape)
# and is rejected; the small margin keeps near-square photos from being rejected.
HORIZONTAL_ASPECT_RATIO = 1.05
HORIZONTAL_IMAGE_REMARK = "Image is in horizontal position; send it properly in vertical position."
# Orientation classifier (image_checks.py): a portrait file whose scene is
# rotated 90 degrees (phone held sideways) is also a horizontal picture. 0.95 was
# first chosen on held-out data: 20/23 horizontal pairs caught (the 3 misses are
# not 90-degree rotations), 1/800 good photos flagged. Raised to 0.99 because an
# upright photo looking up a tree trunk (HPSBA-A232416, a "remove plate" case)
# scored 0.982; on 62 labelled horizontal photos 0.99 catches 26 vs 28 at 0.95.
SIDEWAYS_THRESHOLD = float(os.getenv("QR_SIDEWAYS_THRESHOLD", "0.99"))
# House checker (house_checker.py): after a plate match, the property image must
# show a house with its door/gate visible, otherwise the pair is rejected.
# QR_HOUSE_THRESHOLD overrides the threshold stored in models/house_checker/probe.json.
HOUSE_CHECK_THRESHOLD = float(os.environ["QR_HOUSE_THRESHOLD"]) if os.getenv("QR_HOUSE_THRESHOLD") else None
HOUSE_IMAGE_REMARK = "Property image is not proper."
# Plate detector (plate_detector.py), used when the matcher does not accept:
#  * best confidence < PLATE_PRESENT_THRESHOLD -> reject "plate not shown";
#  * matcher says "not visible" but confidence >= PLATE_VISIBLE_THRESHOLD ->
#    accept the match (plate found, background not checkable) and continue.
# Held-out data (64 "plate not shown" pairs, 400 accept pairs): the second rule
# recovers 20 of the matcher's 30 false "not visible" rejections without
# letting any "plate not shown" pair through (their max was 0.51).
PLATE_PRESENT_THRESHOLD = float(os.getenv("QR_PLATE_PRESENT_THRESHOLD", "0.1"))
PLATE_VISIBLE_THRESHOLD = float(os.getenv("QR_PLATE_VISIBLE_THRESHOLD", "0.55"))
#  * matcher accepted without a background check, but the detector finds no
#    plate at all (< PLATE_VETO_THRESHOLD) -> reject "plate not shown". Held-out:
#    6/8 such "plate not shown" accepts caught, 4/400 good pairs rejected.
PLATE_VETO_THRESHOLD = float(os.getenv("QR_PLATE_VETO_THRESHOLD", "0.05"))
PLATE_NOT_SHOWN_REMARK = "Plate not shown in property image."
# Average "Accept+" probability required for automatic approval. 0.8 was chosen
# on the validation split (see training/reports): it approved ~65% of good
# plates while letting through far fewer rejects than the previous models.
# Lowered to 0.75 to auto-approve more plates (fewer manual reviews).
ACCEPT_THRESHOLD = float(os.getenv("QR_ACCEPT_THRESHOLD", "0.75"))
# Run the inspector models and the matcher's detector on the GPU when one is
# available; QR_DEVICE overrides it (e.g. "cpu", "cuda:0", "cuda:1").
DEVICE = os.getenv("QR_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
INFERENCE_WORKERS = max(
    1,
    int(os.getenv("QR_INFERENCE_WORKERS", str(min(4, os.cpu_count() or 1)))),
)
MAX_PENDING_PAIRS = max(
    INFERENCE_WORKERS,
    int(os.getenv("QR_MAX_PENDING_PAIRS", str(max(100, INFERENCE_WORKERS * 4)))),
)
INFERENCE_POOL = ThreadPoolExecutor(max_workers=INFERENCE_WORKERS, thread_name_prefix="qr-inference")
PENDING_SLOTS = BoundedSemaphore(MAX_PENDING_PAIRS)

CLASS_TO_REMARK = {
    "gap between plate and wall": "Gap between plate and wall; make sure the plate sits flush against the wall.",
    "nails not proper": "nails are not properly fixed ; secure it with at least two properly fixed nails.",
    "rescan scanify image": "Rescan the Scanify image clearly.",
    "stick plate properly": "Stick plate properly.",
    "write o for owner and r1, r2 for rent": "Mark owner plates as O and rental plates as R1 or R2, as applicable.",
    "write r1,r2 or o1, o2 on plate": "Mark owner plates as O and rental plates as R1 or R2, as applicable.",
    "rescan properly scanify image": "Rescan the Scanify image clearly.",
    "horizontal pictures": "Take the pictures in portrait (vertical) orientation.",
    "house id no. missmatch": "House ID number on the plate does not match the property record.",
    "plate is damaged after installation": "Plate is damaged after installation; replace it with a new plate.",
    "plate missing after installation": "Plate is missing after installation; reinstall the plate.",
    "plate not visible in property image": "The plate is not visible in the property image; retake it with the plate clearly visible.",
    "property image not proper": "Property image is not proper; retake a clear photo of the property.",
    "show house entrance": "Show the house entrance in the property image.",
    "show property image long distance": "Take the property image from a longer distance so the whole house and plate are visible.",
    "remove plates": "Remove the QR plate from its current spot and place it in a more convenient area.",
    "remove dustbin sticker": "Remove the dustbin sticker.",
    "remove plate when wall is crack": "Wall is cracked; remove the plate.",
    "write house id no. on plate with marker": "Write the house ID number on the plate with a marker.",
}

allowed_origins = [
    origin.strip()
    for origin in os.getenv("QR_API_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]


class ImagePair(BaseModel):
    QRCodeImage: str
    HouseImage: str
    userId: int | str | None = None
    houseId: int | str | None = None
    QRStatus: int | None = None
    Remark: str | None = None


class BatchRequest(BaseModel):
    items: list[ImagePair] = Field(min_length=1, max_length=100)


class BatchRecordList(RootModel[list[ImagePair]]):
    root: list[ImagePair] = Field(min_length=1, max_length=100)


app = FastAPI(title="QR Plate Match and Inspect API", version="1.0.0")
WEB_ROOT = APP_ROOT / "web"
app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)
@lru_cache(maxsize=1)
def get_matcher():
    return HybridImageMatcher(device=DEVICE)


@lru_cache(maxsize=1)
def get_house_checker():
    return HouseDoorChecker(threshold=HOUSE_CHECK_THRESHOLD, device=DEVICE)


@lru_cache(maxsize=1)
def get_orientation_checker():
    return OrientationChecker(device=DEVICE)


@lru_cache(maxsize=1)
def get_plate_detector():
    return PlateDetector(device=DEVICE)


@lru_cache(maxsize=2)
def get_model(path: str):
    return YOLO(path)


def decode_image(value: str, field_name: str) -> Image.Image:
    """Decode raw base64 or a data:image/...;base64,... value."""
    encoded = "".join(value.strip().split())
    if encoded.lower().startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower():
            raise HTTPException(400, f"{field_name} must be a base64 encoded image.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, f"{field_name} contains invalid base64 data.") from None
    if not raw:
        raise HTTPException(400, f"{field_name} is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"{field_name} exceeds the 25 MB image limit.")
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            # Apply the camera's EXIF rotation so the image is analysed the way
            # it is displayed (phones often store portrait photos sideways).
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, f"{field_name} is not a supported image.") from None


def result(images: ImagePair, qr_status=None, remark=None, manual_review=False, reason=None):
    """Return only the fields the existing frontend needs to update its record.

    For manual review, ``QRStatus`` is null and ``Remark`` (also returned as
    ``manual_review_reason``) explains why a person is needed.
    """
    if manual_review:
        remark = reason or "The result could not be decided automatically."
    response = {
        "QRStatus": qr_status,
        "Remark": remark,
        "manual_review": manual_review,
    }
    if manual_review:
        response["manual_review_reason"] = remark
    if images.userId is not None:
        response["userId"] = images.userId
    if images.houseId is not None:
        response["houseId"] = images.houseId
    return response


def inspect_reference(reference: Image.Image, images: ImagePair):
    predictions = []
    for name, path in MODEL_PATHS:
        if not path.is_file():
            continue
        output = get_model(str(path)).predict(source=reference, device=DEVICE, verbose=False)[0]
        probabilities = output.probs.data.detach().cpu().numpy().astype(float)
        predictions.append(
            (name, {str(output.names[i]): float(probabilities[i]) for i in range(len(probabilities))})
        )
    if not predictions:
        raise HTTPException(503, "Inspector model weights are missing from the server.")

    classes = set().union(*(scores.keys() for _, scores in predictions))
    averages = {
        label: float(np.mean([scores.get(label, 0.0) for _, scores in predictions]))
        for label in classes
    }
    highest = max(averages, key=averages.get)
    winners = [max(scores, key=scores.get).strip().casefold() for _, scores in predictions]
    nail_manual = highest.strip().casefold() == "accept+" and any(w == "nails not proper" for w in winners)
    normalized = " ".join(highest.strip().split()).casefold()
    reason = None
    if nail_manual:
        reason = (
            "Inspector leans towards approval, but one model flagged the nails as not "
            "properly fixed; check the nails manually."
        )
    elif any(word in normalized for word in ("shop", "store", "merchant", "metal")):
        reason = (
            f"Inspector classified the plate as '{highest.strip()}' "
            f"({averages[highest]:.0%} confidence); this category needs manual review."
        )
    if reason:
        return result(images, qr_status=None, remark=images.Remark, manual_review=True, reason=reason)
    if normalized == "accept+":
        # Auto-approve only confident predictions; uncertain ones go to a person.
        if averages[highest] < ACCEPT_THRESHOLD:
            ranked = sorted(averages.items(), key=lambda kv: kv[1], reverse=True)
            runner_up = next((label for label, _ in ranked if label != highest), None)
            reason = (
                f"Inspector approval confidence is {averages[highest]:.0%}, below the "
                f"{ACCEPT_THRESHOLD:.0%} needed for automatic approval."
            )
            if runner_up:
                reason += f" Next most likely issue: '{runner_up.strip()}' ({averages[runner_up]:.0%})."
            return result(images, qr_status=None, remark=images.Remark, manual_review=True, reason=reason)
        return result(images, qr_status=1, remark="")
    remark = CLASS_TO_REMARK.get(normalized, f"Inspector rejection: {highest.strip()}.")
    return result(images, qr_status=0, remark=remark)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/")
def frontend():
    return FileResponse(WEB_ROOT / "index.html")


@app.post("/api/inspect")
def inspect_pair(images: ImagePair):
    future = submit_pair(images)
    if future is None:
        raise HTTPException(503, "Image processing is at capacity; retry this request shortly.")
    return future.result()


def submit_pair(images: ImagePair):
    """Submit one pair to the bounded shared inference pool, or decline if full."""
    if not PENDING_SLOTS.acquire(blocking=False):
        return None
    try:
        future = INFERENCE_POOL.submit(process_pair, images)
    except Exception:
        PENDING_SLOTS.release()
        raise
    future.add_done_callback(lambda _: PENDING_SLOTS.release())
    return future


def process_pair(images: ImagePair):
    if not images.QRCodeImage.strip():
        raise HTTPException(422, "QRCodeImage must contain a base64 encoded image.")
    if not images.HouseImage.strip():
        raise HTTPException(422, "HouseImage must contain a base64 encoded image.")
    reference = decode_image(images.QRCodeImage, "QRCodeImage")
    property_image = decode_image(images.HouseImage, "HouseImage")
    # Property photos must be taken in portrait (vertical) orientation.
    if property_image.width > property_image.height * HORIZONTAL_ASPECT_RATIO:
        return result(images, qr_status=0, remark=HORIZONTAL_IMAGE_REMARK)
    # Portrait files can still hold a sideways scene (photo taken horizontally).
    # Checked on both photos before matching, which a rotated photo would fail.
    try:
        orientation = get_orientation_checker()
        if any(orientation.sideways_probability(img) >= SIDEWAYS_THRESHOLD for img in (reference, property_image)):
            return result(images, qr_status=0, remark=HORIZONTAL_IMAGE_REMARK)
    except Exception as exc:
        raise HTTPException(500, f"Orientation check failed: {exc}") from exc
    try:
        match = get_matcher().verify_presence(reference, property_image)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Image matching failed: {exc}") from exc

    # The matcher could not confirm the plate: ask the plate detector whether any
    # QR plate is visible in the property image at all.
    if not match.get("contains_reference"):
        try:
            plate_confidence = get_plate_detector().best_confidence(property_image)
        except Exception as exc:
            raise HTTPException(500, f"Plate detection failed: {exc}") from exc
        if plate_confidence < PLATE_PRESENT_THRESHOLD:
            return result(images, qr_status=0, remark=PLATE_NOT_SHOWN_REMARK)
        if match.get("reject_remark") == HybridImageMatcher.REMARK_NOT_VISIBLE:
            if plate_confidence >= PLATE_VISIBLE_THRESHOLD:
                # The matcher missed a plate the detector clearly sees (usually a
                # small, distant plate whose background cannot be compared).
                # Plate found + background not checkable -> accept the match and
                # continue to the inspector and house checks (no manual review).
                match = {
                    "contains_reference": True,
                    "background_checked": False,
                    "detection_method": "plate detector: plate found; background not checkable",
                    "plate_confidence": plate_confidence,
                }
            else:
                return result(images, qr_status=0, remark=PLATE_NOT_SHOWN_REMARK)

    # The matcher accepted without checking the background (it only saw a
    # QR-like box). If the plate detector finds no plate at all, the matcher
    # boxed something else on the wall: the plate is not shown.
    elif match.get("background_checked") is False:
        try:
            plate_confidence = get_plate_detector().best_confidence(property_image)
        except Exception as exc:
            raise HTTPException(500, f"Plate detection failed: {exc}") from exc
        if plate_confidence < PLATE_VETO_THRESHOLD:
            return result(images, qr_status=0, remark=PLATE_NOT_SHOWN_REMARK)

    if match.get("needs_manual_review"):
        return result(
            images,
            qr_status=None,
            remark=images.Remark,
            manual_review=True,
            reason=match.get("manual_review_reason"),
        )
    if not match.get("contains_reference"):
        return result(
            images,
            qr_status=0,
            remark=match.get("reject_remark")
            or "Reference plate or its surrounding background does not match the property image.",
        )
    # As in the UI workflow, inspect only the reference plate photo after a match.
    inspection = inspect_reference(reference, images)
    # A plate on a tree, pole or banner must be moved; such photos naturally
    # fail the house check, so the remove-plate remark wins.
    if inspection.get("Remark") == CLASS_TO_REMARK["remove plates"]:
        return inspection
    # Otherwise the property image must be a proper house photo (house with its
    # door/gate visible) before any plate verdict is reported.
    try:
        house = get_house_checker().check(property_image)
    except Exception as exc:
        raise HTTPException(500, f"House image check failed: {exc}") from exc
    if not house.accepted:
        return result(images, qr_status=0, remark=HOUSE_IMAGE_REMARK)
    return inspection


@app.post("/api/inspect/batch")
def inspect_batch(batch: BatchRequest | BatchRecordList):
    """Process up to 100 pairs concurrently within the shared inference limit."""
    items = batch.items if isinstance(batch, BatchRequest) else batch.root
    results = []
    submitted = []
    for index, images in enumerate(items):
        submitted.append((index, images, submit_pair(images)))

    for index, images, future in submitted:
        try:
            if future is None:
                raise HTTPException(503, "Image processing is at capacity; retry this item shortly.")
            item_result = future.result()
            item_result["success"] = True
        except HTTPException as exc:
            item_result = result(images, qr_status=None, remark=images.Remark)
            item_result.update({"success": False, "error": str(exc.detail)})
        except Exception as exc:
            item_result = result(images, qr_status=None, remark=images.Remark)
            item_result.update({"success": False, "error": f"Image analysis failed: {exc}"})
        item_result["index"] = index
        results.append(item_result)
    return {"results": results, "count": len(results)}


@app.get("/api/inspect/batch")
def batch_endpoint_help():
    """Explain the POST contract when this API URL is opened in a browser."""
    return {
        "message": "This endpoint processes images with POST. In Postman, choose Body > raw > JSON.",
        "method": "POST",
        "url": "/api/inspect/batch",
        "maximum_pairs_per_request": 100,
        "body_example": [
            {
                "houseId": 7044,
                "QRCodeImage": "<base64 reference plate image>",
                "HouseImage": "<base64 property image>",
            }
        ],
        "also_accepts_items_wrapper": True,
    }
