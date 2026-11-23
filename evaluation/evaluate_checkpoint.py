import torch
from evaluation.segmentation_metrics import SegmentationMetrics


def evaluate_model(model, val_loader, device):
    """Evaluates a segmentation model on a validation dataset."""

    evaluator = SegmentationMetrics(
        num_classes=11,
        ignore_index=255,
        boundary_tolerance=2
    )

    model.eval()

    with torch.no_grad():
        for images, targets, filenames in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            evaluator.update(predictions, targets)

    results = evaluator.compute()

    return results