#create the class-mapping module
import csv
import numpy as np
from pathlib import Path

CLASS_CSV = Path("data/CamVid/class_dict.csv")
IGNORE_INDEX = 255


def load_class_mapping():
    class_to_id = {}
    color_to_id = {}

    with open(CLASS_CSV, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            name = row["name"].strip()
            color = (
                int(row["r"]),
                int(row["g"]),
                int(row["b"])
            )
            flag = int(row["class_11"])

            if flag == 1:
                class_id = len(class_to_id)
                class_to_id[name] = class_id
                color_to_id[color] = class_id
            else:
                color_to_id[color] = IGNORE_INDEX

    if len(class_to_id) != 11:
        raise ValueError(
            f"Expected 11 selected classes, found {len(class_to_id)}"
        )

    return class_to_id, color_to_id


def rgb_mask_to_class_mask(rgb_mask, color_to_id):
    """
    Convert an RGB CamVid mask into a single-channel class-index mask.
    Excluded categories are assigned IGNORE_INDEX (255).
    """
    rgb_mask = np.asarray(rgb_mask.convert("RGB"))
    height, width, _ = rgb_mask.shape

    class_mask = np.full(
        (height, width),
        IGNORE_INDEX,
        dtype=np.uint8
    )

    for color, class_id in color_to_id.items():
        matches = np.all(rgb_mask == color, axis=-1)
        class_mask[matches] = class_id

    return class_mask


if __name__ == "__main__":
    class_to_id, color_to_id = load_class_mapping()

    print("Class mapping:")
    for name, class_id in class_to_id.items():
        print(f"{class_id}: {name}")

    print("\nTotal target classes:", len(class_to_id))
    print("Ignore index:", IGNORE_INDEX)