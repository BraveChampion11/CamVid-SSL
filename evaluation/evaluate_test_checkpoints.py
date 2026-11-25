
import csv
import math
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.easegnet import EASegNet
from datasets.camvid_dataset import CamVidDataset
from evaluation.segmentation_metrics import SegmentationMetrics


# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_CLASSES = 11
IGNORE_INDEX = 255
BOUNDARY_TOLERANCE = 2

BATCH_SIZE = 4
NUM_WORKERS = 4
IMAGE_SIZE = (480, 360)

CLASS_NAMES = [
    "Bicyclist",
    "Building",
    "Car",
    "Column_Pole",
    "Fence",
    "Pedestrian",
    "Road",
    "Sidewalk",
    "SignSymbol",
    "Sky",
    "Tree",
]

SUPERVISED_DIR = Path("checkpoints/supervised")
MEAN_TEACHER_DIR = Path("checkpoints/mean_teacher")

OUTPUT_DIR = Path("evaluation/test_results")

OVERALL_CSV = OUTPUT_DIR / "test_results_all_runs.csv"
PER_CLASS_CSV = OUTPUT_DIR / "test_results_per_class.csv"
SUMMARY_CSV = OUTPUT_DIR / "test_results_summary.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Expected checkpoint inventory
# --------------------------------------------------

EXPECTED_CHECKPOINTS = []

for seed in range(3):
    for budget in (10, 20, 30):
        EXPECTED_CHECKPOINTS.append(
            (
                "Supervised",
                seed,
                budget,
                SUPERVISED_DIR /
                f"easegnet_seed{seed}_{budget}pct_best.pth",
            )
        )

        EXPECTED_CHECKPOINTS.append(
            (
                "Mean Teacher",
                seed,
                budget,
                MEAN_TEACHER_DIR /
                f"easegnet_mean_teacher_seed{seed}_{budget}pct_best.pth",
            )
        )


def check_files():
    missing = [
        str(path)
        for _, _, _, path in EXPECTED_CHECKPOINTS
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing expected checkpoints:\n" + "\n".join(missing)
        )

    if len(EXPECTED_CHECKPOINTS) != 18:
        raise RuntimeError("Expected exactly 18 checkpoints.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for output_file in (OVERALL_CSV, PER_CLASS_CSV, SUMMARY_CSV):
        if output_file.exists():
            raise FileExistsError(
                f"{output_file} already exists. "
                "Move or rename previous results before running again."
            )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_checkpoint(method, seed, budget, checkpoint_path, test_loader):
    print("\n" + "=" * 65)
    print(f"Method:     {method}")
    print(f"Seed:       {seed}")
    print(f"Budget:     {budget}%")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 65)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if method == "Supervised":
        state_dict_key = "model_state_dict"
    else:
        # Primary Mean Teacher comparison uses the STUDENT.
        # The teacher is not used in the primary results.
        state_dict_key = "student_state_dict"

    if state_dict_key not in checkpoint:
        raise KeyError(
            f"{checkpoint_path} does not contain {state_dict_key}"
        )

    model = EASegNet(
        num_classes=NUM_CLASSES,
        pretrained=False,
    ).to(DEVICE)

    model.load_state_dict(checkpoint[state_dict_key], strict=True)
    model.eval()

    evaluator = SegmentationMetrics(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        boundary_tolerance=BOUNDARY_TOLERANCE,
    )

    images_processed = 0

    with torch.inference_mode():
        for images, targets, filenames in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            outputs = model(images)

            if outputs.shape[1] != NUM_CLASSES:
                raise ValueError(
                    f"Expected {NUM_CLASSES} output classes; "
                    f"received {outputs.shape[1]}."
                )

            if outputs.shape[2:] != targets.shape[1:]:
                raise ValueError(
                    "Prediction and target dimensions do not match."
                )

            predictions = outputs.argmax(dim=1)
            evaluator.update(predictions, targets)

            images_processed += len(filenames)

            if images_processed % 40 == 0:
                print(
                    f"Processed {images_processed}/"
                    f"{len(test_loader.dataset)} images"
                )

    if images_processed != len(test_loader.dataset):
        raise RuntimeError(
            f"Processed {images_processed} images, "
            f"expected {len(test_loader.dataset)}."
        )

    results = evaluator.compute()

    checkpoint_update = checkpoint["update"]
    best_validation_miou = checkpoint["best_mIoU"]

    print(f"Checkpoint update: {checkpoint_update}")
    print(f"Best validation mIoU: {best_validation_miou:.4f}")
    print(f"Test mIoU:       {results['mIoU']:.4f}")
    print(f"Test Dice:       {results['Dice']:.4f}")
    print(f"Test Pixel Acc:  {results['pixel_accuracy']:.4f}")
    print(f"Test Boundary F1:{results['Boundary_F1']:.4f}")

    del model, checkpoint
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "method": method,
        "seed": seed,
        "budget": budget,
        "checkpoint": str(checkpoint_path),
        "checkpoint_update": checkpoint_update,
        "best_validation_mIoU": best_validation_miou,
        "images_evaluated": images_processed,
        "mIoU": results["mIoU"],
        "Dice": results["Dice"],
        "pixel_accuracy": results["pixel_accuracy"],
        "Boundary_F1": results["Boundary_F1"],
        "evaluated_classes": results["evaluated_classes"],
        "evaluated_boundary_classes": results["evaluated_boundary_classes"],
        "per_class_IoU": results["per_class_IoU"],
        "per_class_Boundary_F1": results["per_class_Boundary_F1"],
    }


# --------------------------------------------------
# CSV output
# --------------------------------------------------

def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calculate_summary(overall_rows):
    metrics = [
        "mIoU",
        "Dice",
        "pixel_accuracy",
        "Boundary_F1",
    ]

    summary_rows = []

    for method in ("Supervised", "Mean Teacher"):
        for budget in (10, 20, 30):
            matching = [
                row for row in overall_rows
                if row["method"] == method
                and row["budget"] == budget
            ]

            if len(matching) != 3:
                raise RuntimeError(
                    f"Expected 3 seeds for {method}, {budget}%."
                )

            summary = {
                "method": method,
                "budget": budget,
                "number_of_seeds": len(matching),
            }

            for metric in metrics:
                values = [float(row[metric]) for row in matching]
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values)

            summary_rows.append(summary)

    return summary_rows


def main():
    check_files()

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading CamVid TEST dataset...")

    test_dataset = CamVidDataset(
        split="test",
        manifest=None,
        image_size=IMAGE_SIZE,
        augment=False,
    )

    if len(test_dataset) != 233:
        raise RuntimeError(
            f"Expected 233 test images, found {len(test_dataset)}."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print("Test images:", len(test_dataset))
    print("Expected checkpoints:", len(EXPECTED_CHECKPOINTS))

    overall_rows = []
    per_class_rows = []

    for method, seed, budget, checkpoint_path in EXPECTED_CHECKPOINTS:
        result = evaluate_checkpoint(
            method,
            seed,
            budget,
            checkpoint_path,
            test_loader,
        )

        per_class_iou = result.pop("per_class_IoU")
        per_class_boundary = result.pop("per_class_Boundary_F1")

        overall_rows.append(result)

        for class_id, class_name in enumerate(CLASS_NAMES):
            per_class_rows.append({
                "method": method,
                "seed": seed,
                "budget": budget,
                "class_id": class_id,
                "class_name": class_name,
                "IoU": per_class_iou[class_id],
                "Boundary_F1": per_class_boundary[class_id],
            })

    if len(overall_rows) != 18:
        raise RuntimeError("Not all 18 checkpoints were evaluated.")

    if len(per_class_rows) != 18 * NUM_CLASSES:
        raise RuntimeError("Per-class results are incomplete.")

    summary_rows = calculate_summary(overall_rows)

    write_csv(
        OVERALL_CSV,
        overall_rows,
        [
            "method",
            "seed",
            "budget",
            "checkpoint",
            "checkpoint_update",
            "best_validation_mIoU",
            "images_evaluated",
            "mIoU",
            "Dice",
            "pixel_accuracy",
            "Boundary_F1",
            "evaluated_classes",
            "evaluated_boundary_classes",
        ],
    )

    write_csv(
        PER_CLASS_CSV,
        per_class_rows,
        [
            "method",
            "seed",
            "budget",
            "class_id",
            "class_name",
            "IoU",
            "Boundary_F1",
        ],
    )

    write_csv(
        SUMMARY_CSV,
        summary_rows,
        [
            "method",
            "budget",
            "number_of_seeds",
            "mIoU_mean",
            "mIoU_std",
            "Dice_mean",
            "Dice_std",
            "pixel_accuracy_mean",
            "pixel_accuracy_std",
            "Boundary_F1_mean",
            "Boundary_F1_std",
        ],
    )

    print("\n" + "=" * 65)
    print("FINAL TEST EVALUATION COMPLETED")
    print("=" * 65)
    print("Checkpoints evaluated:", len(overall_rows))
    print("Test images per checkpoint:", len(test_dataset))
    print("Total checkpoint-image evaluations:", len(overall_rows) * len(test_dataset))
    print("\nSaved files:")
    print(OVERALL_CSV)
    print(PER_CLASS_CSV)
    print(SUMMARY_CSV)

    print("\nSUMMARY: TEST mIoU (mean ± sample SD)")
    for row in summary_rows:
        print(
            f"{row['method']:14s} | "
            f"{row['budget']:2d}% | "
            f"{row['mIoU_mean']:.4f} ± {row['mIoU_std']:.4f}"
        )


if __name__ == "__main__":
    main()
