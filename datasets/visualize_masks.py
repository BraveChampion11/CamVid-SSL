# It will generate 10 visual QA panels using only the seed_0/labeled_10.txt
# subset, so we won't inspect masks from the unlabeled pool.
#
# The script creates:
# - 6 reproducibly selected random examples.
# - 4 examples highlighting Bicyclist, SignSymbol, Column_Pole and Pedestrian.
# - Each panel shows the original image, class mask and overlay.
# - Ignored pixels are displayed in neutral grey.

from pathlib import Path
import csv
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from camvid_classes import (
    load_class_mapping,
    rgb_mask_to_class_mask,
    IGNORE_INDEX
)


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data/CamVid/train"
MASK_DIR = ROOT / "data/CamVid/train_labels"

LABELED_FILE = ROOT / "data/splits/seed_0/labeled_10.txt"

OUTPUT_DIR = ROOT / "evaluation/visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
IGNORE_COLOR = (255, 0, 255)
ALPHA = 0.45

# --------------------------------------------------
# 2. Load class mapping and RGB palette
# --------------------------------------------------

class_to_id, color_to_id = load_class_mapping()

id_to_color = {}

with open(ROOT / "data/CamVid/class_dict.csv", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if int(row["class_11"]) == 1:
            class_id = class_to_id[row["name"].strip()]
            id_to_color[class_id] = (
                int(row["r"]),
                int(row["g"]),
                int(row["b"])
            )

id_to_name = {v: k for k, v in class_to_id.items()}

print("Loaded classes:")
for class_id, name in id_to_name.items():
    print(f"{class_id}: {name}")

# --------------------------------------------------
# 3. Read labeled subset
# --------------------------------------------------

with open(LABELED_FILE, "r") as f:
    filenames = [line.strip() for line in f if line.strip()]

print(f"\nImages in labeled 10% subset: {len(filenames)}")

# --------------------------------------------------
# 4. Helper: load image and convert mask
# --------------------------------------------------

def load_sample(filename):

    image_path = IMAGE_DIR / filename

    mask_name = Path(filename).stem + "_L.png"
    mask_path = MASK_DIR / mask_name

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    image = Image.open(image_path).convert("RGB")
    rgb_mask = Image.open(mask_path).convert("RGB")

    if image.size != rgb_mask.size:
        raise ValueError(f"Image-mask size mismatch: {filename}")

    class_mask = rgb_mask_to_class_mask(
        rgb_mask,
        color_to_id
    )

    return np.asarray(image), class_mask


# --------------------------------------------------
# 5. Helper: generate colored mask
# --------------------------------------------------

def create_color_mask(class_mask):

    height, width = class_mask.shape

    color_mask = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    color_mask[:] = IGNORE_COLOR

    for class_id, color in id_to_color.items():
        color_mask[class_mask == class_id] = color

    return color_mask


# --------------------------------------------------
# 6. Helper: visualize one sample
# --------------------------------------------------

def visualize_sample(filename, label):

    image, class_mask = load_sample(filename)

    color_mask = create_color_mask(class_mask)

    valid_pixels = class_mask != IGNORE_INDEX

    overlay = image.copy().astype(np.float32)

    overlay[valid_pixels] = (
        (1 - ALPHA) * image[valid_pixels]
        + ALPHA * color_mask[valid_pixels]
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    axes[1].imshow(color_mask)
    axes[1].set_title("Ground Truth Mask")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{label}: {filename}",
        fontsize=13
    )

    plt.tight_layout()

    output_name = f"{label}_{Path(filename).stem}.png"

    plt.savefig(
        OUTPUT_DIR / output_name,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_name}")


# --------------------------------------------------
# 7. Select six reproducible random samples
# --------------------------------------------------

rng = random.Random(RANDOM_SEED)

random_samples = rng.sample(
    filenames,
    min(6, len(filenames))
)

print("\nGenerating random examples...")

for index, filename in enumerate(random_samples, 1):
    visualize_sample(
        filename,
        f"random_{index}"
    )


# --------------------------------------------------
# 8. Select examples containing rare classes
# --------------------------------------------------

rare_classes = [
    "Bicyclist",
    "SignSymbol",
    "Column_Pole",
    "Pedestrian"
]

# Count each rare class in each labeled image.
class_candidates = {name: [] for name in rare_classes}

print("\nSearching for rare-class examples...")

for filename in filenames:

    _, class_mask = load_sample(filename)

    for class_name in rare_classes:

        class_id = class_to_id[class_name]

        pixel_count = int(
            np.sum(class_mask == class_id)
        )

        if pixel_count > 0:
            class_candidates[class_name].append(
                (filename, pixel_count)
            )


# Choose the image containing the largest number of pixels
# for each rare class. Prefer distinct images when possible.
used_images = set()

for class_name in rare_classes:

    candidates = sorted(
        class_candidates[class_name],
        key=lambda item: item[1],
        reverse=True
    )

    if not candidates:
        print(f"WARNING: No example found for {class_name}")
        continue

    selected = next(
        (item for item in candidates if item[0] not in used_images),
        candidates[0]
    )

    filename, pixel_count = selected
    used_images.add(filename)

    print(
        f"{class_name}: {filename} "
        f"({pixel_count:,} pixels)"
    )

    visualize_sample(
        filename,
        f"rare_{class_name}"
    )


# --------------------------------------------------
# 9. Completion
# --------------------------------------------------

print("\nVisual QA generation completed.")
print(f"Output directory: {OUTPUT_DIR}")
