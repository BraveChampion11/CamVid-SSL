#eed to decide how to handle excluded classes.
import csv
from pathlib import Path

csv_path = Path("data/CamVid/class_dict.csv")

selected_classes = []
excluded_classes = []

with open(csv_path, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        name = row["name"].strip()
        flag = int(row["class_11"])

        if flag == 1:
            selected_classes.append(name)
        else:
            excluded_classes.append(name)

print("Selected 11-class categories:")
for index, name in enumerate(selected_classes):
    print(f"{index}: {name}")

print("\nNumber of selected classes:", len(selected_classes))

print("\nExcluded categories:")
for name in excluded_classes:
    print(name)