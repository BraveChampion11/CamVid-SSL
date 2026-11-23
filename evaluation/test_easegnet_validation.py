#Load pretrained EASegNet.
#Load the 101 validation images.
#Run inference on your RTX 4000 Ada GPU.
#Pass predictions and targets to SegmentationMetrics.
#Print mIoU, Dice, Pixel Accuracy and Boundary F1.
#Check tensor dimensions and metric outputs.


import torch
from torch.utils.data import DataLoader

from models.easegnet import EASegNet
from datasets.camvid_dataset import CamVidDataset
from evaluation.segmentation_metrics import SegmentationMetrics


def main():
    """Runs a validation smoke test on the CamVid validation dataset."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    dataset = CamVidDataset(
        split="val",
        manifest=None,
        image_size=(480, 360),
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    print(f"Validation images: {len(dataset)}")

    model = EASegNet(
        num_classes=11,
        pretrained=True
    ).to(device)

    model.eval()

    evaluator = SegmentationMetrics(
        num_classes=11,
        ignore_index=255,
        boundary_tolerance=2
    )

    total_images = 0

    with torch.inference_mode():
        for batch_index, (images, targets, filenames) in enumerate(loader):

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            if outputs.shape[1] != 11:
                raise ValueError(
                    f"Expected 11 output classes, got {outputs.shape[1]}"
                )

            if outputs.shape[2:] != targets.shape[1:]:
                raise ValueError(
                    "Model output dimensions do not match target dimensions."
                )

            predictions = outputs.argmax(dim=1)

            evaluator.update(predictions, targets)

            total_images += images.size(0)

            if batch_index == 0:
                print("Input shape:", tuple(images.shape))
                print("Output shape:", tuple(outputs.shape))
                print("Target shape:", tuple(targets.shape))
                print("Prediction shape:", tuple(predictions.shape))

            if (batch_index + 1) % 10 == 0:
                print(
                    f"Processed {total_images}/{len(dataset)} images"
                )

    results = evaluator.compute()

    print("\n========== VALIDATION RESULTS ==========")
    print(f"Images evaluated: {total_images}")
    print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    print(f"mIoU: {results['mIoU']:.4f}")
    print(f"Dice: {results['Dice']:.4f}")
    print(f"Boundary F1: {results['Boundary_F1']:.4f}")

    print("\nPer-class IoU:")
    for class_id, score in enumerate(results["per_class_IoU"]):
        print(f"Class {class_id}: {score:.4f}")

    print("\nPer-class Boundary F1:")
    for class_id, score in enumerate(
        results["per_class_Boundary_F1"]
    ):
        print(f"Class {class_id}: {score:.4f}")

    print("\nEvaluated semantic classes:", results["evaluated_classes"])
    print(
        "Evaluated boundary classes:",
        results["evaluated_boundary_classes"]
    )

    if total_images != len(dataset):
        raise RuntimeError("Not all validation images were evaluated.")

    print("\nVALIDATION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
