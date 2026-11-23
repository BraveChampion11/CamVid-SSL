from pathlib import Path
import sys
import os
import csv

import numpy as np
from PIL import Image

# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

# Ensure camvid_classes.py resolves data paths correctly
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "datasets"))

from camvid_classes import (
    load_class_mapping,
    rgb_mask_to_class_mask,
    IGNORE_INDEX
)

DATA_DIR = ROOT / "data" / "CamVid"
MASK_DIR = DATA_DIR / "train_labels"
SPLIT_DIR = ROOT / "data" / "splits"

OUTPUT_DIR = ROOT / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]
BUDGETS = ["10", "20", "30"]

# --------------------------------------------------
# Load class mapping
# --------------------------------------------------

class_to_id, color_to_id = load_class_mapping()

id_to_class = {
    class_id: name
    for name, class_id in class_to_id.items()
}

NUM_CLASSES = len(class_to_id)

print("Classes:")
for class_id, name in id_to_class.items():
    print(f"{class_id}: {name}")

# --------------------------------------------------
# Find corresponding mask
# --------------------------------------------------

def get_mask_path(image_name):

    image_stem = Path(image_name).stem

    # Standard CamVid mask naming convention
    mask_path = MASK_DIR / f"{image_stem}_L.png"

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found for image: {image_name}\n"
            f"Expected: {mask_path}"
        )

    return mask_path


# --------------------------------------------------
# Audit one labeled subset
# --------------------------------------------------

def audit_subset(seed, budget):

    manifest = (
        SPLIT_DIR
        / f"seed_{seed}"
        / f"labeled_{budget}.txt"
    )

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    image_names = [
        name.strip()
        for name in manifest.read_text().splitlines()
        if name.strip()
    ]

    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    images_present = np.zeros(NUM_CLASSES, dtype=np.int64)

    total_pixels = 0
    ignored_pixels = 0

    for image_name in image_names:

        mask_path = get_mask_path(image_name)

        with Image.open(mask_path) as rgb_mask:

            class_mask = rgb_mask_to_class_mask(
                rgb_mask,
                color_to_id
            )

        # Count target-class pixels
        counts = np.bincount(
            class_mask[class_mask != IGNORE_INDEX].astype(np.int64),
            minlength=NUM_CLASSES
        )

        pixel_counts += counts
        images_present += (counts > 0).astype(np.int64)

        total_pixels += class_mask.size
        ignored_pixels += np.count_nonzero(
            class_mask == IGNORE_INDEX
        )

    selected_pixels = int(pixel_counts.sum())

    results = []

    print("\n" + "=" * 75)
    print(f"SEED {seed} | LABELED DATA: {budget}%")
    print(f"Labeled images: {len(image_names)}")
    print(f"Selected-class pixels: {selected_pixels:,}")
    print(f"Ignored pixels: {ignored_pixels:,}")
    print("-" * 75)

    print(
        f"{'Class':<18}"
        f"{'Pixels':>14}"
        f"{'Pixel %':>12}"
        f"{'Images':>12}"
        f"{'Status':>12}"
    )

    for class_id in range(NUM_CLASSES):

        name = id_to_class[class_id]
        pixels = int(pixel_counts[class_id])
        percentage = (
            100 * pixels / selected_pixels
            if selected_pixels > 0 else 0
        )
        image_count = int(images_present[class_id])

        status = "PRESENT" if image_count > 0 else "ABSENT"

        print(
            f"{name:<18}"
            f"{pixels:>14,}"
            f"{percentage:>11.3f}%"
            f"{image_count:>12}"
            f"{status:>12}"
        )

        results.append({
            "seed": seed,
            "budget": f"{budget}%",
            "class_id": class_id,
            "class_name": name,
            "pixel_count": pixels,
            "pixel_percentage": round(percentage, 5),
            "images_present": image_count,
            "total_labeled_images": len(image_names),
            "absent": image_count == 0,
            "ignored_pixels": ignored_pixels,
            "total_mask_pixels": total_pixels
        })

    absent_classes = [
        id_to_class[class_id]
        for class_id in range(NUM_CLASSES)
        if images_present[class_id] == 0
    ]

    if absent_classes:
        print("\nWARNING — Classes absent from this subset:")
        print(", ".join(absent_classes))
    else:
        print("\nAll 11 classes are represented.")

    return results


# --------------------------------------------------
# Audit all 9 subsets
# --------------------------------------------------

def main():

    all_results = []

    for seed in SEEDS:
        for budget in BUDGETS:

            results = audit_subset(seed, budget)
            all_results.extend(results)

    # Save results to CSV
    output_file = OUTPUT_DIR / "subset_class_distribution.csv"

    fieldnames = list(all_results[0].keys())

    with open(output_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 75)
    print("AUDIT COMPLETED")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()