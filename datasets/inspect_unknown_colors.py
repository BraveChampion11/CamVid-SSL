#diagnose the unknown colours first
from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np

from camvid_classes import load_class_mapping

ROOT = Path("data/CamVid/test_labels")

_, color_to_id = load_class_mapping()

known_colors = np.array(list(color_to_id.keys()), dtype=np.int16)
unknown_counts = Counter()

total_pixels = 0
unknown_pixels = 0

for path in ROOT.glob("*.png"):
    mask = np.asarray(Image.open(path).convert("RGB"))

    colors, counts = np.unique(
        mask.reshape(-1, 3),
        axis=0,
        return_counts=True
    )

    total_pixels += mask.shape[0] * mask.shape[1]

    for color, count in zip(colors, counts):
        color_tuple = tuple(int(x) for x in color)

        if color_tuple not in color_to_id:
            unknown_counts[color_tuple] += int(count)
            unknown_pixels += int(count)

print("Total pixels:", f"{total_pixels:,}")
print("Unknown pixels:", f"{unknown_pixels:,}")
print("Unknown pixel percentage:", round(
    100 * unknown_pixels / total_pixels, 4
), "%")

print("\nTop 20 unknown colors:")

for color, count in unknown_counts.most_common(20):
    distances = np.linalg.norm(
        known_colors - np.array(color),
        axis=1
    )

    nearest_index = np.argmin(distances)
    nearest_color = tuple(known_colors[nearest_index].tolist())

    print(
        f"{color} | pixels: {count:,} | "
        f"nearest known color: {nearest_color} | "
        f"distance: {distances[nearest_index]:.2f}"
    )