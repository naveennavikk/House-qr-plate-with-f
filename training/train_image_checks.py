"""Train the orientation classifier (models/orientation_cls.pt) used by image_checks.py.

The training data is generated automatically: upright Accept photos make the
"upright" class, and the same photos rotated by 90 or 270 degrees make the
"sideways" class. Only the top square of each photo is used, which excludes
the ID/GPS watermark that the app stamps upright at the bottom (the same crop
is used at inference).

Usage (from the repository root):
  python training/train_image_checks.py --accept-dir <Accept folder> [--skip holdout_ids.txt]
Then copy training/runs/orient/weights/best.pt to models/orientation_cls.pt.

Evaluate on the "SCANIFY IMAGE AND PROPERTY IMAGE HORIZONTAL" rejection folder
and on held-out Accept photos before changing SIDEWAYS_THRESHOLD in api.py.
"""

import argparse
import random
import re
import shutil
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "training" / "data" / "orientation"


def build(accept_dir: Path, skip: set[str], limit: int):
    shutil.rmtree(DATASET, ignore_errors=True)
    rng = random.Random(0)
    files = []
    for f in sorted(accept_dir.iterdir()):
        m = re.match(r"(HPSBA-[A-Z]\d+)", f.name)
        if m and m.group(1) not in skip and f.suffix.lower() in {".jpg", ".jpeg"} and "copy" not in f.name.lower():
            files.append(f)
    rng.shuffle(files)
    for i, f in enumerate(files[:limit]):
        im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        if im.width > im.height:
            continue
        side = min(im.size)
        square = im.crop((0, 0, side, side)).resize((320, 320), Image.Resampling.LANCZOS)
        split = "val" if i % 10 == 0 else "train"
        for cls, img in (("upright", square), ("sideways", square.rotate(rng.choice((90, 270)), expand=True))):
            out = DATASET / split / cls
            out.mkdir(parents=True, exist_ok=True)
            img.save(out / f"{i}.jpg", quality=92)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--accept-dir", required=True, type=Path)
    p.add_argument("--skip", help="file with house IDs to keep out of training")
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--weights", default=str(ROOT / "models" / "yolo11n_cls_best.pt"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    skip = set(Path(args.skip).read_text().split()) if args.skip else set()
    build(args.accept_dir, skip, args.limit)

    from ultralytics import YOLO

    YOLO(args.weights).train(
        data=str(DATASET), imgsz=224, epochs=10, batch=64, device=args.device,
        project=str(ROOT / "training" / "runs"), name="orient", exist_ok=True, plots=False, fliplr=0.5,
    )


if __name__ == "__main__":
    main()
