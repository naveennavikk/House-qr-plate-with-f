"""Train the "is a QR plate visible?" detector (models/plate_detector.pt).

Steps:
  1. label  - run the matcher on Accept pairs (reference plate + "-House" photo).
              Where it confidently locates the plate, keep that box as a label.
  2. build  - turn those boxes into a YOLO dataset of whole-plate boxes. Tiny
              (<18 px) template hits are dropped: they are often blank wall.
  3. train  - fine-tune yolo11n at 1024 px (property photos are 762x1016, so
              this keeps small, distant plates at native resolution).

Usage (from the repository root):
  python training/train_plate_detector.py label --accept-dir <Accept folder> --limit 1900 --skip holdout_ids.txt
  python training/train_plate_detector.py build
  python training/train_plate_detector.py train --weights yolo11n.pt
Then copy training/runs/plate/weights/best.pt to models/plate_detector.pt.

Keep the pairs used to evaluate the API out of the labels (--skip), together
with the "Plate not show in property image" rejection folder.
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = ROOT / "training" / "data" / "plate_detector"
LABELS = WORK / "labels.jsonl"
DATASET = WORK / "dataset"


def parse_box(value):
    if value is None:
        return None
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(value).replace("np.int64", "").replace("np.float64", ""))]
    return nums[:4] if len(nums) >= 4 else None


def label(args):
    from PIL import Image
    from matcher import HybridImageMatcher

    files = set(os.listdir(args.accept_dir))
    pairs = []
    for f in sorted(files):
        m = re.match(r"(HPSBA-[A-Z]\d+)-House\.jpe?g$", f)
        if not m:
            continue
        for ext in (".jpeg", ".jpg"):
            if m.group(1) + ext in files:
                pairs.append((m.group(1), os.path.join(args.accept_dir, m.group(1) + ext), os.path.join(args.accept_dir, f)))
                break
    random.Random(0).shuffle(pairs)
    skip = set(Path(args.skip).read_text().split()) if args.skip else set()
    pairs = [p for p in pairs[: args.limit] if p[0] not in skip]
    WORK.mkdir(parents=True, exist_ok=True)
    matcher = HybridImageMatcher()
    with open(LABELS, "w") as fh:
        for key, ref, prop in pairs:
            prop_img = Image.open(prop)
            r = matcher.verify_presence(Image.open(ref), prop_img)
            if r.get("contains_reference") and r.get("candidate_qr_box") is not None:
                fh.write(json.dumps({
                    "id": key, "prop": prop, "prop_size": list(prop_img.size),
                    "detection_method": r["detection_method"], "candidate_qr_box": str(r["candidate_qr_box"]),
                }) + "\n")
                fh.flush()


def build(_args):
    from PIL import Image

    shutil.rmtree(DATASET, ignore_errors=True)
    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True)
        (DATASET / "labels" / split).mkdir(parents=True)
    rows = [json.loads(line) for line in open(LABELS)]
    random.Random(1).shuffle(rows)
    kept = 0
    for i, r in enumerate(rows):
        W, H = r["prop_size"]
        scale = min(1, 1400 / max(W, H))  # the matcher's working resolution
        x1, y1, x2, y2 = [v / scale for v in parse_box(r["candidate_qr_box"])]
        w, h = x2 - x1, y2 - y1
        if w < 18 or h < 18:
            continue
        if not r["detection_method"].startswith("Grounding"):
            # QR symbol box -> whole plate (QR on the left, text panel to the right).
            x1, y1, x2, y2 = x1 - .1 * w, y1 - .15 * h, x2 + 1.35 * w, y2 + .15 * h
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        split = "val" if i % 10 == 0 else "train"
        Image.open(r["prop"]).convert("RGB").save(DATASET / "images" / split / f"{r['id']}.jpg", quality=95)
        (DATASET / "labels" / split / f"{r['id']}.txt").write_text(
            f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"
        )
        kept += 1
    (DATASET / "data.yaml").write_text(
        f"path: {DATASET}\ntrain: images/train\nval: images/val\nnames:\n  0: plate\n"
    )
    print(f"labelled {len(rows)}, kept {kept}")


def train(args):
    from ultralytics import YOLO

    YOLO(args.weights).train(
        data=str(DATASET / "data.yaml"), imgsz=1024, epochs=20, patience=8, batch=16,
        device=args.device, project=str(ROOT / "training" / "runs"), name="plate", exist_ok=True,
        mosaic=1.0, close_mosaic=3, plots=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("label")
    s.add_argument("--accept-dir", required=True)
    s.add_argument("--limit", type=int, default=1900)
    s.add_argument("--skip")
    sub.add_parser("build")
    s = sub.add_parser("train")
    s.add_argument("--weights", default="yolo11n.pt")
    s.add_argument("--device", default="cpu")
    args = p.parse_args()
    {"label": label, "build": build, "train": train}[args.cmd](args)
