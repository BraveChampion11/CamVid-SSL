import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets.camvid_dataset import CamVidDataset
from models.easegnet import EASegNet
from evaluation.segmentation_metrics import SegmentationMetrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed):
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Seed NumPy and Python random generators inside a DataLoader worker."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_loader(dataset, batch_size, shuffle, num_workers, seed):
    """Create a reproducibly seeded DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0
    )


def validate(model, loader, criterion, device):
    """Compute validation loss and dataset-level segmentation metrics."""
    evaluator = SegmentationMetrics(
        num_classes=11,
        ignore_index=255,
        boundary_tolerance=2
    )
    model.eval()
    weighted_loss = 0.0
    valid_pixels_total = 0

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks)
            valid_pixels = int((masks != 255).sum().item())
            weighted_loss += loss.item() * valid_pixels
            valid_pixels_total += valid_pixels
            evaluator.update(logits.argmax(dim=1), masks)

    if valid_pixels_total == 0:
        raise ValueError("Validation set contains no valid labeled pixels.")

    metrics = evaluator.compute()
    metrics["loss"] = weighted_loss / valid_pixels_total
    return metrics


def append_metrics(csv_path, row):
    """Append a metrics row to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, model, optimizer, scheduler, update, best_miou, metrics, config):
    """Save model, optimizer, scheduler, metrics, and experiment configuration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "update": update,
            "best_mIoU": best_miou,
            "validation_metrics": metrics,
            "config": config
        },
        path
    )


def main():
    """Train EASegNet on a selected CamVid labeled-data budget."""
    parser = argparse.ArgumentParser(
        description="Supervised EASegNet training on CamVid."
    )
    parser.add_argument("--budget", type=int, choices=[10, 20, 30, 100], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if min(args.updates, args.batch_size, args.eval_every) < 1:
        raise ValueError("Updates, batch size, and evaluation interval must be positive.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.budget == 100:
        manifest = None
    else:
        manifest = (
            PROJECT_ROOT / "data" / "splits" /
            f"seed_{args.seed}" / f"labeled_{args.budget}.txt"
        )
        if not manifest.is_file():
            raise FileNotFoundError(f"Labeled subset manifest not found: {manifest}")

    train_dataset = CamVidDataset(
        split="train",
        manifest=manifest,
        image_size=(480, 360),
        augment=True
    )
    val_dataset = CamVidDataset(
        split="val",
        manifest=None,
        image_size=(480, 360),
        augment=False
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty.")

    train_loader = create_loader(
        train_dataset, args.batch_size, True, args.num_workers, args.seed
    )
    val_loader = create_loader(
        val_dataset, args.batch_size, False, args.num_workers, args.seed
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Annotation budget: {args.budget}%")
    print(f"Seed: {args.seed}")
    print(f"Optimizer updates: {args.updates}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    model = EASegNet(num_classes=11, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates
    )

    run_name = f"easegnet_seed{args.seed}_{args.budget}pct"
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "supervised"
    log_dir = PROJECT_ROOT / "logs" / "supervised"
    best_checkpoint = checkpoint_dir / f"{run_name}_best.pth"
    latest_checkpoint = checkpoint_dir / f"{run_name}_latest.pth"
    csv_path = log_dir / f"{run_name}.csv"

    config = {
        "model": "EASegNet",
        "dataset": "CamVid",
        "budget_percent": args.budget,
        "seed": args.seed,
        "manifest": str(manifest) if manifest else None,
        "optimizer_updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "loss": "CrossEntropyLoss",
        "ignore_index": 255,
        "image_size": [480, 360],
        "encoder_pretraining": "MobileNetV2 ImageNet pretrained",
        "augmentation_enabled": True
    }

    model.train()
    data_iterator = iter(train_loader)
    best_miou = float("-inf")
    interval_loss = 0.0
    interval_updates = 0

    print("\nStarting supervised training...\n")

    for update in range(1, args.updates + 1):
        try:
            images, masks, _ = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            images, masks, _ = next(data_iterator)

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)

        if logits.shape[-2:] != masks.shape[-2:]:
            raise ValueError(
                f"Model output size {logits.shape[-2:]} does not match "
                f"mask size {masks.shape[-2:]}."
            )

        loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at optimizer update {update}.")

        loss.backward()
        optimizer.step()
        scheduler.step()

        interval_loss += loss.item()
        interval_updates += 1

        if update % 50 == 0 or update == 1:
            print(
                f"Update {update}/{args.updates} | "
                f"Training loss: {loss.item():.5f}"
            )

        if update % args.eval_every == 0 or update == args.updates:
            average_train_loss = interval_loss / interval_updates
            metrics = validate(model, val_loader, criterion, device)
            current_miou = float(metrics["mIoU"])
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"\nUpdate {update}/{args.updates} | "
                f"Train loss: {average_train_loss:.5f} | "
                f"Val loss: {metrics['loss']:.5f} | "
                f"Pixel accuracy: {metrics['pixel_accuracy']:.4f} | "
                f"mIoU: {current_miou:.4f} | "
                f"Dice: {metrics['Dice']:.4f} | "
                f"Boundary F1: {metrics['Boundary_F1']:.4f} | "
                f"LR: {current_lr:.7g}"
            )

            row = {
                "update": update,
                "average_training_loss": average_train_loss,
                "validation_loss": metrics["loss"],
                "validation_pixel_accuracy": metrics["pixel_accuracy"],
                "validation_mIoU": current_miou,
                "validation_Dice": metrics["Dice"],
                "validation_Boundary_F1": metrics["Boundary_F1"],
                "learning_rate": current_lr,
                "evaluated_classes": metrics["evaluated_classes"],
                "evaluated_boundary_classes": metrics["evaluated_boundary_classes"],
                "per_class_IoU": json.dumps(metrics["per_class_IoU"])
            }
            append_metrics(csv_path, row)

            save_checkpoint(
                latest_checkpoint, model, optimizer, scheduler,
                update, best_miou, metrics, config
            )

            if current_miou > best_miou:
                best_miou = current_miou
                save_checkpoint(
                    best_checkpoint, model, optimizer, scheduler,
                    update, best_miou, metrics, config
                )
                print(f"New best checkpoint saved: {best_checkpoint}")

            print(f"Metrics log: {csv_path}\n")
            interval_loss = 0.0
            interval_updates = 0
            model.train()

    print("========== TRAINING COMPLETE ==========")
    print(f"Total optimizer updates: {args.updates}")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Latest checkpoint: {latest_checkpoint}")
    print(f"Metrics log: {csv_path}")


if __name__ == "__main__":
    main()
