from pathlib import Path
from collections import Counter
import csv
import numpy as np
from PIL import Image

ROOT = Path("data/CamVid")

# Read class dictionary
csv_path = ROOT / "class_dict.csv"

class_mapping = {}

with open(csv_path, newline="") as file:
    reader = csv.reader(file)
    header = next(reader)

    print("CSV Header:", header)

    for row in reader:
        if len(row) >= 4:
            name = row[0].strip()
            rgb = tuple(int(x.strip()) for x in row[1:4])
            class_mapping[rgb] = name

print("\nClasses defined:", len(class_mapping))

# Inspect all training masks
color_counts = Counter()

for mask_path in (ROOT / "train_labels").glob("*"):
    if not mask_path.is_file():
        continue

    mask = np.array(Image.open(mask_path).convert("RGB"))

    colors, counts = np.unique(
        mask.reshape(-1, 3),
        axis=0,
        return_counts=True
    )

    for color, count in zip(colors, counts):
        color_counts[tuple(color)] += int(count)

print("\nUnique colors across all training masks:",
      len(color_counts))

print("\nObserved classes:")

for color, count in sorted(color_counts.items()):
    name = class_mapping.get(color, "UNKNOWN COLOR")
    print(f"{name:25} RGB={color} Pixels={count}")

# Check unmatched colors
unmatched = set(color_counts) - set(class_mapping)

print("\nUnmatched colors:", unmatched)