# QR Plate Match and Inspect API

This repository contains the FastAPI service, a small HTML/CSS/JavaScript test page, the image matcher, and trained inspector weights. Your team's backend can call the same API endpoints. The service compares the reference QR plate and its surrounding background with the property image. If they match, the house checker (`house_checker.py`) confirms the property image shows a house with its door or gate visible; if it does not, the pair is rejected with the remark "Property image is not proper." Only after both checks pass does the inspector analyze the reference plate image. The house checker downloads its CLIP encoder (about 290 MB) to `models/house_checker/` on first use.

## Run the API

Install the dependencies in your Python environment:

```powershell
python -m pip install -r requirements.txt
```

From this folder, start the service:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` to test one or multiple image pairs in the browser. The interactive API schema is at `http://localhost:8000/docs`; health check: `http://localhost:8000/health`.

## Backend request

Send `POST http://localhost:8000/api/inspect` with JSON containing the two required images. Each image can be raw base64 or a `data:image/...;base64,...` string. Optional `houseId` and `userId` values are echoed back for record correlation. Other fields in the existing property record may also be included and are ignored by the image analysis.

```json
{
  "houseId": 7044,
  "QRCodeImage": "<base64 reference plate image>",
  "HouseImage": "<base64 property image>"
}
```

For multiple pairs in one request, send `POST http://localhost:8000/api/inspect/batch` with the frontend's JSON array directly. Each item uses the same record shape. The endpoint also accepts the wrapped `{"items": [...]}` format. A batch can contain up to 100 pairs; results are returned in the same order and include the zero-based `index` plus any supplied IDs. A bad image in one item is reported for that item and does not stop the other pairs from being processed.

```json
[
  {
    "houseId": 7044,
    "QRCodeImage": "<base64 reference plate image 1>",
    "HouseImage": "<base64 property image 1>"
  },
  {
    "houseId": 7045,
    "QRCodeImage": "<base64 reference plate image 2>",
    "HouseImage": "<base64 property image 2>"
  }
]
```

## Response

`QRStatus` follows the existing application's convention:

- `1`: approve; `Remark` is empty.
- `0`: reject; `Remark` explains the rejection.
- `null`: manual review; `Remark` (also returned as `manual_review_reason`) explains why the pair needs a person (for example, a blurry reference image, a partial background match, or low inspector confidence).

Example manual review:

```json
{
  "QRStatus": null,
  "Remark": "The plate could not be located in the property image.",
  "manual_review": true,
  "manual_review_reason": "The plate could not be located in the property image.",
  "houseId": 7044
}
```

Example automatic rejection:

```json
{
  "QRStatus": 0,
  "Remark": "Reference plate or its surrounding background does not match the property image.",
  "manual_review": false,
  "houseId": 7044
}
```

For manual review the calling backend can store the `Remark` so reviewers see why the pair was flagged, while leaving `QRStatus` unset. It can set `QRStatusDate` and `modified` according to its own record-update rules.

See [LOAD_TEST_REPORT.md](LOAD_TEST_REPORT.md) for the local synthetic throughput and latency measurements, test limitations, and recommended scale tests.

## Plate-visibility and orientation checks

- `models/plate_detector.pt` (`plate_detector.py`) is a YOLO detector that answers "is any QR plate visible in the property image?". When the matcher cannot confirm the plate and the detector finds none, or when the matcher accepted without a background check but the detector finds no plate at all, the pair is rejected with "Plate not shown in property image." Thresholds: `QR_PLATE_PRESENT_THRESHOLD` (0.1), `QR_PLATE_VISIBLE_THRESHOLD` (0.55), `QR_PLATE_VETO_THRESHOLD` (0.05). Retrain with `training/train_plate_detector.py`.
- `models/orientation_cls.pt` (`image_checks.py`) catches photos taken sideways: the file is portrait but the scene is rotated 90 degrees. Both images are checked before matching, and a sideways photo is rejected with the horizontal-image remark. Threshold: `QR_SIDEWAYS_THRESHOLD` (0.99). Retrain with `training/train_image_checks.py`.
