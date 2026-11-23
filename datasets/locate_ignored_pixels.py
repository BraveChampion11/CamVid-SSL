#Find every unmatched pixel
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import numpy as np
import csv

ROOT = Path("data/CamVid")
CSV_PATH = ROOT / "class_dict.csv"

# Read all class definitions.
selected_colors = {}
excluded_colors = {}

with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        name = row["name"].strip()
        color = (int(row["r"]), int(row["g"]), int(row["b"]))

        if int(row["class_11"]) == 1:
            selected_colors[color] = name
        else:
            excluded_colors[color] = name

known_colors = set(selected_colors) | set(excluded_colors)

for split in ["train", "val", "test"]:

    mask_dir = ROOT / f"{split}_labels"

    selected_counts = Counter()
    excluded_counts = Counter()
    unknown_counts = Counter()

    # Group unknown pixel coordinates by file and RGB colour.
    locations = defaultdict(list)

    total_pixels = 0

    for path in sorted(mask_dir.glob("*.png")):

        # Read original mask directly. No resizing or augmentation.
        mask = np.asarray(Image.open(path).convert("RGB"))

        height, width, _ = mask.shape
        total_pixels += height * width

        colors, counts = np.unique(
            mask.reshape(-1, 3),
            axis=0,
            return_counts=True
        )

        for color_array, count in zip(colors, counts):

            color = tuple(int(x) for x in color_array)
            count = int(count)

            if color in selected_colors:
                selected_counts[selected_colors[color]] += count

            elif color in excluded_colors:
                excluded_counts[excluded_colors[color]] += count

            else:
                unknown_counts[color] += count

                # Find exact coordinates of this unknown colour.
                matches = np.all(mask == color_array, axis=-1)
                rows, cols = np.where(matches)

                for y, x in zip(rows, cols):
                    locations[(path.name, color)].append((int(x), int(y)))

    print("\n" + "=" * 60)
    print(f"{split.upper()} RAW MASK AUDIT")
    print("=" * 60)

    print("Total pixels:", f"{total_pixels:,}")
    print("Selected-class pixels:",
          f"{sum(selected_counts.values()):,}")
    print("Known excluded-class pixels:",
          f"{sum(excluded_counts.values()):,}")
    print("Unknown-colour pixels:",
          f"{sum(unknown_counts.values()):,}")

    print("\nKnown excluded categories:")
    for name, count in sorted(excluded_counts.items()):
        print(f"{name:22} {count:,}")

    print("\nUnknown colours:")
    for color, count in unknown_counts.most_common():
        print(f"{color}: {count:,} pixels")

    print("\nExact locations of unknown pixels:")

    for (filename, color), coordinates in sorted(locations.items()):
        print(f"\nFile: {filename}")
        print(f"RGB: {color}")
        print(f"Count: {len(coordinates)}")
        print(f"Coordinates (x, y): {coordinates}")

    assert (
        sum(selected_counts.values())
        + sum(excluded_counts.values())
        + sum(unknown_counts.values())
        == total_pixels
    ), "Pixel accounting mismatch!"