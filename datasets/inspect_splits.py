#audit all three dataset splits
from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np

from camvid_classes import load_class_mapping, rgb_mask_to_class_mask

ROOT = Path("data/CamVid")

class_to_id, color_to_id = load_class_mapping()
id_to_class = {v: k for k, v in class_to_id.items()}

splits = ["train", "val", "test"]


def normalized_stem(path):
    stem = path.stem
    return stem[:-2] if stem.endswith("_L") else stem


for split in splits:
    image_dir = ROOT / split
    mask_dir = ROOT / f"{split}_labels"

    images = {
        normalized_stem(p): p
        for p in image_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    }

    masks = {
        normalized_stem(p): p
        for p in mask_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    }

    missing_masks = sorted(set(images) - set(masks))
    orphan_masks = sorted(set(masks) - set(images))

    print(f"\n{'=' * 45}")
    print(f"{split.upper()} SPLIT")
    print(f"Images: {len(images)}")
    print(f"Masks: {len(masks)}")
    print(f"Images without masks: {len(missing_masks)}")
    print(f"Masks without images: {len(orphan_masks)}")

    class_pixel_counts = Counter()
    classes_present = set()
    dimension_errors = 0
    unknown_colors = set()

    for name in sorted(set(images) & set(masks)):
        image = Image.open(images[name]).convert("RGB")
        mask = Image.open(masks[name]).convert("RGB")

        if image.size != mask.size:
            dimension_errors += 1
            continue

        rgb = np.asarray(mask)

        known_colors = set(color_to_id.keys())
        mask_colors = set(map(tuple, np.unique(
            rgb.reshape(-1, 3), axis=0
        )))

        unknown_colors.update(mask_colors - known_colors)

        class_mask = rgb_mask_to_class_mask(mask, color_to_id)

        for class_id in range(11):
            pixels = int(np.sum(class_mask == class_id))

            if pixels > 0:
                classes_present.add(class_id)
                class_pixel_counts[class_id] += pixels

    print("Dimension mismatches:", dimension_errors)
    print("Unknown RGB colors:", len(unknown_colors))

    print("\nClass coverage:")
    for class_id in range(11):
        name = id_to_class[class_id]
        pixels = class_pixel_counts[class_id]
        status = "PRESENT" if class_id in classes_present else "MISSING"
        print(f"{class_id}: {name:15} | {status:7} | pixels: {pixels:,}")

    if missing_masks:
        print("\nExamples of images without masks:", missing_masks[:5])

    if orphan_masks:
        print("Examples of masks without images:", orphan_masks[:5])

    if unknown_colors:
        print("Unknown colors:", list(unknown_colors)[:10])