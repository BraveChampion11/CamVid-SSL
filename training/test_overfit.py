import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW

from datasets.camvid_dataset import CamVidDataset
from models.easegnet import EASegNet


def calculate_scores(logits, targets, num_classes=11):
    """Calculate pixel accuracy and mean IoU for a batch."""

    predictions = logits.argmax(dim=1)

    valid = (
        (targets != 255)
        & (targets >= 0)
        & (targets < num_classes)
    )

    targets = targets[valid]
    predictions = predictions[valid]

    if targets.numel() == 0:
        return 0.0, 0.0

    encoded = targets * num_classes + predictions

    confusion = torch.bincount(
        encoded,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes).float()

    true_positives = torch.diag(confusion)
    union = (
        confusion.sum(dim=0)
        + confusion.sum(dim=1)
        - true_positives
    )

    valid_classes = union > 0
    iou = true_positives[valid_classes] / union[valid_classes]

    pixel_accuracy = true_positives.sum() / confusion.sum()
    mean_iou = iou.mean()

    return pixel_accuracy.item(), mean_iou.item()


def main():
    """Run a single-image overfitting diagnostic for EASegNet."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    dataset = CamVidDataset(
        split="train",
        manifest=None,
        image_size=(480, 360),
        augment=False
    )

    image, mask, filename = dataset[0]

    images = image.unsqueeze(0).to(device)
    targets = mask.unsqueeze(0).to(device)

    valid_pixels = (targets != 255).sum().item()

    if valid_pixels == 0:
        raise ValueError("Selected image contains no valid target pixels.")

    print(f"Selected image: {filename}")
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {targets.shape}")
    print(f"Valid pixels: {valid_pixels}")

    model = EASegNet(
        num_classes=11,
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    model.train()

    with torch.no_grad():
        initial_logits = model(images)
        initial_loss = criterion(initial_logits, targets).item()

    print(f"\nInitial loss: {initial_loss:.6f}")
    print("\nStarting overfit test...\n")

    for iteration in range(1, args.iters + 1):

        optimizer.zero_grad()

        logits = model(images)

        if logits.shape[-2:] != targets.shape[-2:]:
            raise ValueError("Model output and target dimensions do not match.")

        loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss encountered.")

        loss.backward()

        if iteration == 1:
            gradient_norm = sum(
                parameter.grad.norm().item()
                for parameter in model.parameters()
                if parameter.grad is not None
            )

            print(f"Initial gradient norm: {gradient_norm:.6f}")

            if gradient_norm == 0:
                raise ValueError("No gradients detected.")

        optimizer.step()

        if iteration % 10 == 0 or iteration == 1:

            with torch.no_grad():
                pixel_accuracy, mean_iou = calculate_scores(
                    logits,
                    targets
                )

            print(
                f"Iteration {iteration:3d}/{args.iters} | "
                f"Loss: {loss.item():.6f} | "
                f"Pixel Accuracy: {pixel_accuracy:.4f} | "
                f"mIoU: {mean_iou:.4f}"
            )

    model.eval()

    with torch.no_grad():
        final_logits = model(images)
        final_loss = criterion(final_logits, targets).item()

        final_accuracy, final_miou = calculate_scores(
            final_logits,
            targets
        )

    print("\n========== OVERFIT TEST RESULTS ==========")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Pixel Accuracy: {final_accuracy:.4f}")
    print(f"mIoU:           {final_miou:.4f}")

    if final_loss < initial_loss:
        print("\nOVERFIT TEST PASSED: Loss decreased.")
    else:
        print("\nOVERFIT TEST FAILED: Loss did not decrease sufficiently.")


if __name__ == "__main__":
    main()