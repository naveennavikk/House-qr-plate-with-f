"""HTTP API for connecting the QR plate workflow to an existing frontend."""

import base64
import binascii
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import BoundedSemaphore

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, RootModel
from ultralytics import YOLO

from house_checker import HouseDoorChecker
from house_id import HouseIdReader, comparison_key, normalize_id
from house_source import SourceError, fetch_houses, fetch_ulbs
from image_checks import OrientationChecker, plate_closeup_fraction
from matcher import HybridImageMatcher
from plate_detector import PlateDetector

APP_ROOT = Path(__file__).resolve().parent
MODEL_PATHS = [
    ("YOLO26n", APP_ROOT / "models" / "yolo26n_cls_best.pt"),
    ("YOLO11n", APP_ROOT / "models" / "yolo11n_cls_best.pt"),
]
# The inspector above was retrained without a Shops class (no shop data was
# available). The previous weights still recognise shops, so they are kept
# only for that: if they classify the plate as Shops, the pair goes to manual review.
LEGACY_SHOP_MODEL_PATHS = [
    APP_ROOT / "models" / "legacy_shops" / "yolo26n_cls_best.pt",
    APP_ROOT / "models" / "legacy_shops" / "yolo11n_cls_best.pt",
]
LEGACY_REMOVE_PLATES_THRESHOLD = float(os.getenv("QR_LEGACY_REMOVE_PLATES_THRESHOLD", "0.6"))
MAX_IMAGE_BYTES = 25 * 1024 * 1024
# A property image wider than tall by this factor counts as horizontal (landscape)
# and is rejected; the small margin keeps near-square photos from being rejected.
HORIZONTAL_ASPECT_RATIO = 1.05
HORIZONTAL_IMAGE_REMARK = "Image is in horizontal position; send it properly in vertical position."
# House ID check (house_id.py), run before every other task: the IDs stamped at
# the bottom-left of both photos and the request's houseNo must all agree.
# An ID that cannot be read is skipped (policy to be decided after testing).
HOUSE_ID_MISMATCH_REMARK = "House ID mismatch"
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
# A plate close-up sent as the property image (e.g. the two photos swapped) has a
# large, readable QR code; the property image is then not proper. See
# image_checks.plate_closeup_fraction (238/250 close-ups vs 0/450 property photos).
PLATE_CLOSEUP_THRESHOLD = float(os.getenv("QR_PLATE_CLOSEUP_THRESHOLD", "0.15"))
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
# Independent stages of one pair (house ID OCR of both photos, orientation and
# the matcher) run concurrently on this pool; results are still applied in the
# original order, so decisions are unchanged. Separate from INFERENCE_POOL so a
# pair waiting on its stages can never block the stages themselves.
STAGE_POOL = ThreadPoolExecutor(max_workers=INFERENCE_WORKERS * 4, thread_name_prefix="qr-stage")

CLASS_TO_REMARK = {
    "gap between plate and wall": "Gap between plate and wall; make sure the plate sits flush against the wall.",
    "nails not proper": "Nails are not proper,fix properly",
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
    # House ID code, e.g. "HPSBA-C118430"; compared with the IDs stamped on both photos.
    houseNo: str | None = None
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
def get_house_id_reader():
    return HouseIdReader()


def house_ids_mismatch(reference_id: str | None, property_id: str | None, house_no) -> bool:
    """True when any two of the readable IDs (both photo stamps + houseNo) differ."""
    ids = [reference_id, property_id, normalize_id(house_no)]
    keys = {comparison_key(i) for i in ids if i}
    return len(keys) > 1


@lru_cache(maxsize=1)
def get_orientation_checker():
    return OrientationChecker(device=DEVICE)


@lru_cache(maxsize=1)
def get_plate_detector():
    return PlateDetector(device=DEVICE)


@lru_cache(maxsize=4)
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
    if images.houseNo is not None:
        response["houseNo"] = images.houseNo
    return response


def legacy_top_class(reference: Image.Image) -> tuple[str, float] | None:
    """Top class and its average probability from the legacy (pre-retraining) models."""
    predictions = []
    for path in LEGACY_SHOP_MODEL_PATHS:
        if not path.is_file():
            continue
        output = get_model(str(path)).predict(source=reference, device=DEVICE, verbose=False)[0]
        probabilities = output.probs.data.detach().cpu().numpy().astype(float)
        predictions.append({str(output.names[i]): float(probabilities[i]) for i in range(len(probabilities))})
    if not predictions:
        return None
    averages = {
        label: float(np.mean([scores.get(label, 0.0) for scores in predictions]))
        for label in set().union(*predictions)
    }
    highest = max(averages, key=averages.get)
    return highest, averages[highest]


def inspect_reference(reference: Image.Image, images: ImagePair):
    legacy = legacy_top_class(reference)
    if legacy is not None:
        legacy_class, legacy_confidence = legacy
        if "shop" in legacy_class.casefold():
            return result(
                images, qr_status=None, remark=images.Remark, manual_review=True,
                reason=(
                    f"Inspector classified the plate as 'Shops' ({legacy_confidence:.0%} confidence); "
                    "this category needs manual review."
                ),
            )
        # The legacy models were better at plates on trees/poles: trust a
        # confident REMOVE PLATES from them (good held-out plates scored <= 0.52).
        if legacy_class.strip().casefold() == "remove plates" and legacy_confidence >= LEGACY_REMOVE_PLATES_THRESHOLD:
            return result(images, qr_status=0, remark=CLASS_TO_REMARK["remove plates"])
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


@app.on_event("startup")
def warm_up_models():
    """Load every model (and run each once) at startup, so the first request is
    as fast as later ones instead of paying several seconds of lazy loading."""
    if os.getenv("QR_SKIP_WARMUP"):
        return
    blank = Image.new("RGB", (762, 1016), (200, 200, 200))
    get_house_id_reader().read(blank)
    get_orientation_checker().sideways_probability(blank)
    get_matcher()._region_detections(blank, "qr code . plate .")
    get_plate_detector().detect(blank)
    for _name, path in MODEL_PATHS:
        if path.is_file():
            get_model(str(path)).predict(source=blank, device=DEVICE, verbose=False)
    legacy_top_class(blank)
    get_house_checker().check(blank)


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
    is_landscape = property_image.width > property_image.height * HORIZONTAL_ASPECT_RATIO
    reader, orientation, matcher = get_house_id_reader(), get_orientation_checker(), get_matcher()
    # Start the independent stages together. The matcher is started speculatively;
    # its result is only used if the earlier checks pass.
    reference_id = STAGE_POOL.submit(reader.read, reference)
    property_id = STAGE_POOL.submit(reader.read, property_image)
    closeup = STAGE_POOL.submit(plate_closeup_fraction, property_image)
    sideways = STAGE_POOL.submit(
        lambda: any(orientation.sideways_probability(img) >= SIDEWAYS_THRESHOLD for img in (reference, property_image))
    )
    matching = None if is_landscape else STAGE_POOL.submit(matcher.verify_presence, reference, property_image)

    # First task: the house IDs must agree; on a mismatch no other task runs.
    try:
        if house_ids_mismatch(reference_id.result(), property_id.result(), images.houseNo):
            if matching is not None:
                matching.cancel()  # skips the matcher if it has not started yet
            return result(images, qr_status=0, remark=HOUSE_ID_MISMATCH_REMARK)
    except Exception as exc:
        raise HTTPException(500, f"House ID check failed: {exc}") from exc
    # Property photos must be taken in portrait (vertical) orientation.
    if is_landscape:
        return result(images, qr_status=0, remark=HORIZONTAL_IMAGE_REMARK)
    # Portrait files can still hold a sideways scene (photo taken horizontally).
    # Checked on both photos before matching, which a rotated photo would fail.
    try:
        if sideways.result():
            if matching is not None:
                matching.cancel()
            return result(images, qr_status=0, remark=HORIZONTAL_IMAGE_REMARK)
    except Exception as exc:
        raise HTTPException(500, f"Orientation check failed: {exc}") from exc
    # The property image must show the house, not a close-up of the plate.
    try:
        if closeup.result() >= PLATE_CLOSEUP_THRESHOLD:
            if matching is not None:
                matching.cancel()
            return result(images, qr_status=0, remark=HOUSE_IMAGE_REMARK)
    except Exception as exc:
        raise HTTPException(500, f"Property image check failed: {exc}") from exc
    try:
        match = matching.result()
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
        except Exception as exc:  # noqa: BLE001 - one bad pair must not fail the whole batch
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


# ---------------------------------------------------------------------------
# Houses from the ICTSBM house details API (house_source.py): every house of a
# day is fetched server side and run through process_pair, exactly like a
# POST /api/inspect request. Results are streamed to the page one line (JSON)
# per house as soon as each house is checked; nothing is written back.

THUMB_SIZE = 220


def thumbnail(value) -> str | None:
    """Small JPEG data URL of a base64 photo for the results page (None if unreadable)."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        image = decode_image(value, "image")
    except HTTPException:
        return None
    image.thumbnail((THUMB_SIZE, THUMB_SIZE))
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def check_house(index: int, record: dict) -> dict:
    row = {
        "index": index,
        "houseId": record.get("houseId"),
        "userId": record.get("userId"),
        "houseNo": record.get("ReferanceId"),
        "employee": record.get("qrEmpName"),
        "modified": record.get("modified"),
        "propertyType": record.get("PropertyType"),
        "masterId": record.get("MasterId"),
        "qrThumb": thumbnail(record.get("QRCodeImage")),
        "houseThumb": thumbnail(record.get("HouseImage")),
    }
    started = time.perf_counter()
    missing = [name for name in ("QRCodeImage", "HouseImage") if not str(record.get(name) or "").strip()]
    if missing:
        row.update(status="Error", QRStatus=None, Remark=f"Missing {' and '.join(missing)} in the house API data.")
        return row
    try:
        pair = ImagePair(
            QRCodeImage=record["QRCodeImage"],
            HouseImage=record["HouseImage"],
            userId=record.get("userId"),
            houseId=record.get("houseId"),
            houseNo=record.get("ReferanceId"),
        )
        outcome = process_pair(pair)
    except HTTPException as exc:
        row.update(status="Error", QRStatus=None, Remark=str(exc.detail))
    except Exception as exc:  # noqa: BLE001 - one bad house must not stop the others
        row.update(status="Error", QRStatus=None, Remark=f"Image analysis failed: {exc}")
    else:
        status = "Manual review" if outcome.get("manual_review") else {1: "Accepted", 0: "Rejected"}.get(outcome.get("QRStatus"), "Manual review")
        row.update(status=status, QRStatus=outcome.get("QRStatus"), Remark=outcome.get("Remark"))
    row["seconds"] = round(time.perf_counter() - started, 2)
    return row


HOUSE_BATCH_SIZE = int(os.getenv("HOUSE_BATCH_SIZE", "100"))


def stream_house_checks(records: list[dict]):
    """Check the houses in batches of HOUSE_BATCH_SIZE; one JSON line per house and per finished batch."""
    total_started = time.perf_counter()
    batches = [records[i:i + HOUSE_BATCH_SIZE] for i in range(0, len(records), HOUSE_BATCH_SIZE)]
    yield json.dumps({"type": "start", "count": len(records), "batchSize": HOUSE_BATCH_SIZE, "batches": len(batches)}) + "\n"
    pending = set()
    try:
        for number, batch in enumerate(batches, start=1):
            batch_started = time.perf_counter()
            offset = (number - 1) * HOUSE_BATCH_SIZE
            queue = list(enumerate(batch, start=offset))
            queue.reverse()
            while queue or pending:
                # Keep the shared inference pool busy without exceeding its pending limit.
                while queue and (PENDING_SLOTS.acquire(blocking=False) if pending else PENDING_SLOTS.acquire(timeout=60)):
                    index, record = queue.pop()
                    future = INFERENCE_POOL.submit(check_house, index, record)
                    future.add_done_callback(lambda _: PENDING_SLOTS.release())
                    pending.add(future)
                if not pending:
                    continue
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield json.dumps({"type": "house", "batch": number, **future.result()}) + "\n"
            seconds = round(time.perf_counter() - batch_started, 1)
            yield json.dumps({"type": "batch", "batch": number, "count": len(batch), "seconds": seconds}) + "\n"
    finally:
        for future in pending:  # page closed: drop houses that have not started
            future.cancel()
    yield json.dumps({"type": "done", "seconds": round(time.perf_counter() - total_started, 1)}) + "\n"


@app.get("/api/ulbs")
def list_ulbs():
    """ULBs for the page's dropdown: [{"name": ..., "appId": ...}]."""
    try:
        return fetch_ulbs()
    except SourceError as exc:
        raise HTTPException(502, str(exc)) from None


@app.get("/api/houses/check")
def check_houses(
    app_id: int | None = Query(None, alias="appId"),
    from_day: date = Query(alias="fromDate"),
    to_day: date | None = Query(None, alias="toDate"),
    search: str = Query("", alias="search"),
):
    """Fetch the houses between fromDate and toDate from the house API and check every one
    (streamed, one JSON line per house)."""
    to_day = to_day or from_day
    if to_day < from_day:
        raise HTTPException(422, "toDate must not be before fromDate.")
    try:
        records = fetch_houses(from_day, to_day, search, app_id)
    except SourceError as exc:
        raise HTTPException(502, str(exc)) from None
    return StreamingResponse(stream_house_checks(records), media_type="application/x-ndjson")
