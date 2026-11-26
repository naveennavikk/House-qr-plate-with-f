"""Build the accept/reject rule classification dataset.

Sources (all read-only), under QR_DATA_ROOT
(default ~/Downloads/Images for qr plate and qr plate house):
  * rejection rules data train/<rule>/, Rejection plate photos 1/<rule>/,
    BHIWANDI, SOLAPUR, Ahilyanagar and New folder sub-folders - reject rules
    (folder names are mapped onto the inspector classes by CLASS_ALIASES)
  * Acc, Accept, Accept-, accept--                  - accepted plate + "-House" pairs
  * the older prepared dataset the shipped models were trained on (optional)

Output: training/data/prepared/{train,val,test}/<class>/  plus
        training/data/holdout_accept/  (accept pairs never used for training,
        used by evaluate_rules.py to measure false rejections).

Images are split by house ID, so a plate photo and its property photo
never end up on both sides of the split.
"""
import hashlib
import os
import json
import random
import re
import shutil
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
DOCS = Path(os.getenv("QR_DATA_ROOT", Path.home() / "Downloads" / "Images for qr plate and qr plate house"))
REJECT_ROOT = DOCS / "rejection rules data train"
EXTRA_REJECT_ROOTS = [
    DOCS / "Rejection plate photos 1",
    DOCS / "BHIWANDI" / "BHIWANDI",
    DOCS / "SOLAPUR" / "SOLAPUR",
    DOCS / "Ahilyanagar" / "Ahilyanagar",
    DOCS / "New folder",
]
ACCEPT_DIRS = [DOCS / n for n in ("Accept", "Acc", "Accept-", "accept--")]
OLD_PREPARED = Path.home() / "Downloads" / "data train classification and yolo" / \
    "QR_House_Plate_AI_Inspector_Hybrid_v2" / "data" / "prepared"
OUT = HERE / "data" / "prepared"
HOLDOUT = HERE / "data" / "holdout_accept"

SEED = 42
ACCEPT_CLASS = "Accept+"
ACCEPT_TRAIN_GROUPS = 600      # per house-ID series (D and I): ~2400 accept images
HOLDOUT_GROUPS = 300           # accept pairs reserved for false-reject testing
MAX_SIDE = 640
EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp"}

# Folders that describe the same rule are trained as one class.
CLASS_ALIASES = {
    "write r1,r2 or o1, o2 on plate": "Write O for owner and R1, R2 for Rent",
    "write r1 and r2 and o1 and o2 on plate": "Write O for owner and R1, R2 for Rent",
    "write o1,o2 or r1,r2 on plate": "Write O for owner and R1, R2 for Rent",
    "rescan properly scanify image": "Rescan Scanify Image",
    "rescan scanify image": "Rescan Scanify Image",
    "rescan properly": "Rescan Scanify Image",
    "rescan proper scanify image": "Rescan Scanify Image",
    "nails": "Nails Not Proper",
    "nail not proper": "Nails Not Proper",
    "nails not proper": "Nails Not Proper",
    "damage plates": "plate is damaged After Installation",
    "damage plate": "plate is damaged After Installation",
    "damages plate": "plate is damaged After Installation",
    "plate missing after installation": "PLATE MISSING AFTER INSTALLATION",
    "plate not show in property image": "PLATE NOT VISIBLE IN PROPERTY IMAGE",
    "plate not visible in property image": "PLATE NOT VISIBLE IN PROPERTY IMAGE",
    "property image not proper": "Property Image not proper",
    "remove dustbin stickers from doors and walls": "REMOVE DUSTBIN STICKER",
    "remove dust bin sticker": "REMOVE DUSTBIN STICKER",
    "remove dustbin sticker": "REMOVE DUSTBIN STICKER",
    "remove plate": "REMOVE PLATES",
    "remove plates": "REMOVE PLATES",
    "remove plate when wall is crack": "REMOVE PLATE WHEN WALL IS CRACK",
    "show house entrance": "SHOW HOUSE ENTRANCE",
    "show property image long distance": "Show Property Image Long Distance",
    "stick plate properly": "Stick plate properly",
    "write house id no. on plate with marker": "WRITE HOUSE ID NO. ON PLATE WITH MARKER",
    "write house id number on plate with marker": "WRITE HOUSE ID NO. ON PLATE WITH MARKER",
    "scanify image and plate id mismatch": "HOUSE ID NO. MISSMATCH",
    "mismatch house id and plate id": "HOUSE ID NO. MISSMATCH",
    "mismatched id no": "HOUSE ID NO. MISSMATCH",
    "house id no. missmatch": "HOUSE ID NO. MISSMATCH",
    "scanify image and property image horizontal": "Horizontal Pictures",
    "scan scanify image and property image horizontal": "Horizontal Pictures",
    "horizontal pictures": "Horizontal Pictures",
    "gap between plate and wall": "GAP BETWEEN PLATE AND WALL",
    "accept+": ACCEPT_CLASS,
}
# Rules handled outside the inspector (or without a class in the shipped
# models) are not trained.
SKIP_FOLDERS = {
    "washroom", "remove plate from banner", "put plate plain surface wall",
    "put plate properly", "show only one plate in property image", "scratch",
}

ID_RE = re.compile(r"(hpsba-[a-z]?\d+)", re.I)


def canonical(name):
    key = " ".join(name.split()).casefold()
    return CLASS_ALIASES.get(key, " ".join(name.split()))


def group_key(path, class_name):
    m = ID_RE.search(path.stem)
    return m.group(1).casefold() if m else f"{class_name}/{path.stem}".casefold()


def images_in(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_jpg(src, dest):
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, "JPEG", quality=92)


def split_groups(groups):
    """Split a list of image groups ~70/15/15, keeping each group intact."""
    random.shuffle(groups)
    n = len(groups)
    if n >= 5:
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
    elif n >= 3:
        n_val, n_test = 1, 1
    elif n == 2:
        n_val, n_test = 1, 0
    else:
        n_val, n_test = 0, 0
    return {
        "val": groups[:n_val],
        "test": groups[n_val:n_val + n_test],
        "train": groups[n_val + n_test:],
    }


def main():
    random.seed(SEED)
    if not REJECT_ROOT.is_dir():
        raise FileNotFoundError(REJECT_ROOT)

    # 1. Collect reject images: new rule folders first, then old classes.
    class_files = {}   # class -> {hash: path}
    sources = [d for d in sorted(REJECT_ROOT.iterdir()) if d.is_dir()]
    for root in EXTRA_REJECT_ROOTS:
        if root.is_dir():
            sources += [
                d for d in sorted(root.iterdir())
                if d.is_dir() and " ".join(d.name.split()).casefold() not in SKIP_FOLDERS
            ]
    if OLD_PREPARED.is_dir():
        for split in ("train", "val", "test"):
            sources += [d for d in sorted((OLD_PREPARED / split).iterdir()) if d.is_dir()]
    seen = {}          # hash -> class (first wins; drops cross-class duplicates)
    conflicts = 0
    for folder in sources:
        cls = canonical(folder.name)
        for img in images_in(folder):
            h = digest(img)
            if h in seen:
                conflicts += seen[h] != cls
                continue
            seen[h] = cls
            class_files.setdefault(cls, {})[h] = img

    reject_ids = {
        group_key(p, c)
        for c, files in class_files.items() if c != ACCEPT_CLASS
        for p in files.values()
        if ID_RE.search(p.stem)
    }

    # 2. Accept pairs from the Accept folders, excluding any house ID that was
    #    ever rejected and any exact duplicate of a reject image.
    accept_groups = {}
    for folder in ACCEPT_DIRS:
        if not folder.is_dir():
            continue
        for img in images_in(folder):
            key = group_key(img, ACCEPT_CLASS)
            if key in reject_ids:
                continue
            accept_groups.setdefault(key, {})[img.name.casefold()] = img
    # Only keep complete plate + house pairs.
    pairs = {k: list(v.values()) for k, v in accept_groups.items() if len(v) == 2}
    keys = sorted(pairs)
    random.shuffle(keys)
    holdout_keys = keys[:HOLDOUT_GROUPS]
    # Balance the house-ID series so I-series plates are as well represented
    # as the (far more numerous) D-series ones.
    by_series = {}
    for k in keys[HOLDOUT_GROUPS:]:
        m = re.match(r"hpsba-([a-z])", k)
        by_series.setdefault(m.group(1) if m else "?", []).append(k)
    train_keys = [k for ks in by_series.values() for k in ks[:ACCEPT_TRAIN_GROUPS]]
    accept_files = class_files.setdefault(ACCEPT_CLASS, {})
    for k in train_keys:
        for img in pairs[k]:
            h = digest(img)
            if h not in seen:
                seen[h] = ACCEPT_CLASS
                accept_files[h] = img

    # 3. Write the split dataset.
    for target in (OUT, HOLDOUT):
        if target.exists():
            shutil.rmtree(target)
    summary = {}
    for cls, files in sorted(class_files.items()):
        grouped = {}
        for img in files.values():
            grouped.setdefault(group_key(img, cls), []).append(img)
        splits = split_groups(list(grouped.values()))
        summary[cls] = {}
        for split, groups in splits.items():
            count = 0
            for group in groups:
                for img in group:
                    count += 1
                    name = re.sub(r"[^\w.-]+", "_", img.stem)[:60]
                    save_jpg(img, OUT / split / cls / f"{name}_{digest(img)[:8]}.jpg")
            summary[cls][split] = count
    for k in holdout_keys:
        for img in pairs[k]:
            kind = "house" if img.stem.casefold().endswith("-house") else "plate"
            save_jpg(img, HOLDOUT / kind / f"{img.stem}.jpg")

    report = {
        "seed": SEED,
        "classes": summary,
        "cross_class_duplicates_dropped": conflicts,
        "accept_pairs_available": len(pairs),
        "accept_pairs_trained": len(train_keys),
        "accept_pairs_holdout": len(holdout_keys),
    }
    (HERE / "reports").mkdir(exist_ok=True)
    (HERE / "reports" / "split_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
