import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets.camvid_dataset import CamVidDataset
from datasets.camvid_unlabeled_dataset import CamVidUnlabeledDataset
from models.easegnet import EASegNet
from evaluation.segmentation_metrics import SegmentationMetrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NUM_CLASSES = 11
IGNORE_INDEX = 255


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_loader(dataset, batch_size, shuffle, num_workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0
    )


def validate(model, loader, criterion, device):
    evaluator = SegmentationMetrics(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
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

            valid_pixels = int(
                (masks != IGNORE_INDEX).sum().item()
            )

            weighted_loss += loss.item() * valid_pixels
            valid_pixels_total += valid_pixels

            evaluator.update(logits.argmax(dim=1), masks)

    if valid_pixels_total == 0:
        raise ValueError("Validation set contains no valid pixels.")

    metrics = evaluator.compute()
    metrics["loss"] = weighted_loss / valid_pixels_total

    return metrics


def append_metrics(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys())
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


@torch.no_grad()
def update_teacher(student, teacher, ema_decay):
    """
    EMA-update teacher parameters.

    Copy buffers directly rather than EMA-averaging BatchNorm running
    statistics and counters.
    """

    student_parameters = dict(student.named_parameters())
    teacher_parameters = dict(teacher.named_parameters())

    for name, teacher_parameter in teacher_parameters.items():
        student_parameter = student_parameters[name]

        teacher_parameter.mul_(ema_decay).add_(
            student_parameter,
            alpha=1.0 - ema_decay
        )

    student_buffers = dict(student.named_buffers())
    teacher_buffers = dict(teacher.named_buffers())

    for name, teacher_buffer in teacher_buffers.items():
        teacher_buffer.copy_(student_buffers[name])


def save_checkpoint(
    path,
    student,
    teacher,
    optimizer,
    scheduler,
    update,
    best_miou,
    metrics,
    config
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "update": update,
            "best_mIoU": best_miou,
            "validation_metrics": metrics,
            "config": config
        },
        path
    )


def next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)

    return batch, iterator


def main():
    parser = argparse.ArgumentParser(
        description="Mean Teacher semi-supervised CamVid training."
    )

    parser.add_argument(
        "--budget",
        type=int,
        choices=[10, 20, 30],
        required=True
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=[0, 1, 2],
        default=0
    )
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--unsup-weight", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.1667)

    args = parser.parse_args()

    if min(
        args.updates,
        args.batch_size,
        args.eval_every
    ) < 1:
        raise ValueError(
            "Updates, batch size and evaluation interval must be positive."
        )

    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1).")

    if args.unsup_weight < 0:
        raise ValueError("Unsupervised weight cannot be negative.")

    if not 0.0 < args.warmup_fraction <= 1.0:
        raise ValueError("Warmup fraction must be in (0, 1].")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 1. Resolve labeled manifest
    # ---------------------------------------------------------

    manifest = (
        PROJECT_ROOT
        / "data"
        / "splits"
        / f"seed_{args.seed}"
        / f"labeled_{args.budget}.txt"
    )

    if not manifest.is_file():
        raise FileNotFoundError(
            f"Labeled manifest not found: {manifest}"
        )

    # ---------------------------------------------------------
    # 2. Create datasets
    # ---------------------------------------------------------

    labeled_dataset = CamVidDataset(
        split="train",
        manifest=manifest,
        image_size=(480, 360),
        augment=True
    )

    unlabeled_dataset = CamVidUnlabeledDataset(
        labeled_manifest=manifest,
        image_size=(480, 360)
    )

    val_dataset = CamVidDataset(
        split="val",
        manifest=None,
        image_size=(480, 360),
        augment=False
    )

    if (
        len(labeled_dataset) == 0
        or len(unlabeled_dataset) == 0
        or len(val_dataset) == 0
    ):
        raise ValueError("A training or validation dataset is empty.")

    print(f"Labeled images: {len(labeled_dataset)}")
    print(f"Unlabeled images: {len(unlabeled_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Annotation budget: {args.budget}%")
    print(f"Seed: {args.seed}")

    # ---------------------------------------------------------
    # 3. Create DataLoaders
    # ---------------------------------------------------------

    labeled_loader = create_loader(
        labeled_dataset,
        args.batch_size,
        True,
        args.num_workers,
        args.seed
    )

    unlabeled_loader = create_loader(
        unlabeled_dataset,
        args.batch_size,
        True,
        args.num_workers,
        args.seed + 100
    )

    val_loader = create_loader(
        val_dataset,
        args.batch_size,
        False,
        args.num_workers,
        args.seed + 200
    )

    if len(labeled_loader) == 0 or len(unlabeled_loader) == 0:
        raise ValueError(
            "A training loader has zero batches. "
            "Reduce the batch size."
        )

    # ---------------------------------------------------------
    # 4. Initialize student and teacher
    # ---------------------------------------------------------

    student = EASegNet(
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(device)

    # Teacher starts as an exact copy of the initialized student.
    teacher = copy.deepcopy(student).to(device)

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    teacher.eval()

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX
    )

    optimizer = AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.updates
    )

    # ---------------------------------------------------------
    # 5. Output paths and configuration
    # ---------------------------------------------------------

    run_name = (
        f"easegnet_mean_teacher_seed{args.seed}_{args.budget}pct"
    )

    checkpoint_dir = (
        PROJECT_ROOT / "checkpoints" / "mean_teacher"
    )

    log_dir = PROJECT_ROOT / "logs" / "mean_teacher"

    best_checkpoint = (
        checkpoint_dir / f"{run_name}_best.pth"
    )

    latest_checkpoint = (
        checkpoint_dir / f"{run_name}_latest.pth"
    )

    csv_path = log_dir / f"{run_name}.csv"

    # Do not silently append results from an earlier experiment.
    if csv_path.exists():
        raise FileExistsError(
            f"Log already exists: {csv_path}\n"
            "Move or rename the old log before starting a fresh run."
        )

    config = {
        "model": "EASegNet",
        "method": "Mean Teacher",
        "dataset": "CamVid",
        "budget_percent": args.budget,
        "seed": args.seed,
        "manifest": str(manifest),
        "optimizer_updates": args.updates,
        "labeled_batch_size": args.batch_size,
        "unlabeled_batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "supervised_loss": "CrossEntropyLoss",
        "unsupervised_loss": "Confidence-masked pseudo-label CrossEntropy",
        "confidence_threshold": args.confidence_threshold,
        "ema_decay_max": args.ema_decay,
        "unsupervised_weight": args.unsup_weight,
        "warmup_fraction": args.warmup_fraction,
        "warmup_strategy": "Supervised-only warm-up, teacher synchronized, then confidence-masked Mean Teacher",
        "warmup_updates": max(
            1,
            int(args.updates * args.warmup_fraction)
        ),
        "ignore_index": IGNORE_INDEX,
        "num_classes": NUM_CLASSES,
        "image_size": [480, 360],
        "encoder_pretraining": "MobileNetV2 ImageNet pretrained",
        "teacher_initialization": "Exact copy of initialized student",
        "teacher_buffer_update": "Copy student buffers",
        "teacher_input": "Weakly augmented",
        "student_unlabeled_input": "Strong color augmentation",
        "test_set_used_during_training": False
    }

    log_dir.mkdir(parents=True, exist_ok=True)

    with (log_dir / f"{run_name}_config.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=4)

    # ---------------------------------------------------------
    # 6. Training
    # ---------------------------------------------------------

    labeled_iterator = iter(labeled_loader)
    unlabeled_iterator = iter(unlabeled_loader)

    best_miou = float("-inf")

    warmup_updates = config["warmup_updates"]

    interval_total_loss = 0.0
    interval_supervised_loss = 0.0
    interval_unsupervised_loss = 0.0
    interval_coverage = 0.0
    interval_updates = 0

    print("\nStarting Mean Teacher training...\n")

    for update in range(1, args.updates + 1):

        student.train()
        teacher.eval()

        # -----------------------------------------------------
        # Get labeled batch
        # -----------------------------------------------------

        labeled_batch, labeled_iterator = next_batch(
            labeled_iterator,
            labeled_loader
        )

        labeled_images, labeled_masks, _ = labeled_batch

        labeled_images = labeled_images.to(
            device,
            non_blocking=True
        )

        labeled_masks = labeled_masks.to(
            device,
            non_blocking=True
        )

        # -----------------------------------------------------
        # Get unlabeled batch
        # -----------------------------------------------------

        unlabeled_batch, unlabeled_iterator = next_batch(
            unlabeled_iterator,
            unlabeled_loader
        )

        weak_images, strong_images, _ = unlabeled_batch

        weak_images = weak_images.to(
            device,
            non_blocking=True
        )

        strong_images = strong_images.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        # -----------------------------------------------------
        # Supervised student loss
        # -----------------------------------------------------

        labeled_logits = student(labeled_images)

        if labeled_logits.shape[-2:] != labeled_masks.shape[-2:]:
            raise ValueError(
                "Student output dimensions do not match labeled masks."
            )

        supervised_loss = criterion(
            labeled_logits,
            labeled_masks
        )

        # -----------------------------------------------------
        # Supervised-only warm-up, then Mean Teacher loss
        # -----------------------------------------------------

        if update <= warmup_updates:
            # Train only on labeled data during warm-up.
            # No teacher pseudo-labels are used in this phase.
            unsupervised_loss = labeled_logits.sum() * 0.0
            coverage = 0.0

        else:
            # Teacher predicts pseudo-labels from the weak view.
            with torch.no_grad():
                teacher_logits = teacher(weak_images)

                teacher_probabilities = torch.softmax(
                    teacher_logits,
                    dim=1
                )

                confidence, pseudo_labels = (
                    teacher_probabilities.max(dim=1)
                )

                confidence_mask = (
                    confidence >= args.confidence_threshold
                )

            # Student learns from the strong color-augmented view.
            unlabeled_logits = student(strong_images)

            if unlabeled_logits.shape[-2:] != pseudo_labels.shape[-2:]:
                raise ValueError(
                    "Student output dimensions do not match pseudo-labels."
                )

            per_pixel_loss = F.cross_entropy(
                unlabeled_logits,
                pseudo_labels,
                reduction="none"
            )

            if confidence_mask.any():
                unsupervised_loss = per_pixel_loss[
                    confidence_mask
                ].mean()
            else:
                # Differentiable zero if no pixels pass the threshold.
                unsupervised_loss = unlabeled_logits.sum() * 0.0

            coverage = confidence_mask.float().mean().item()

        # Activate the unsupervised objective only after warm-up.
        current_unsup_weight = (
            args.unsup_weight if update > warmup_updates else 0.0
        )

        total_loss = (
            supervised_loss
            + current_unsup_weight * unsupervised_loss
        )

        if not torch.isfinite(total_loss):
            raise ValueError(
                f"Non-finite total loss at update {update}."
            )

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # -----------------------------------------------------
        # Teacher synchronization and EMA update
        # -----------------------------------------------------

        if update == warmup_updates:
            # Start SSL from the supervised student, not from a
            # lagging teacher that has averaged the initial weights.
            teacher.load_state_dict(student.state_dict())
            current_ema_decay = 0.0
            print(
                f"\\nTeacher synchronized with student "
                f"after {warmup_updates} supervised updates."
            )

        elif update > warmup_updates:
            # Restart the EMA schedule after synchronization.
            ssl_update = update - warmup_updates
            current_ema_decay = min(
                args.ema_decay,
                1.0 - (1.0 / (ssl_update + 1))
            )

            update_teacher(
                student,
                teacher,
                current_ema_decay
            )

        else:
            # Teacher is unused during supervised-only warm-up.
            current_ema_decay = 0.0

        interval_total_loss += total_loss.item()
        interval_supervised_loss += supervised_loss.item()
        interval_unsupervised_loss += unsupervised_loss.item()
        interval_coverage += coverage
        interval_updates += 1

        if update % 50 == 0 or update == 1:
            print(
                f"Update {update}/{args.updates} | "
                f"Total loss: {total_loss.item():.5f} | "
                f"Sup loss: {supervised_loss.item():.5f} | "
                f"Unsup loss: {unsupervised_loss.item():.5f} | "
                f"Coverage: {coverage:.3f} | "
                f"EMA: {current_ema_decay:.4f}"
            )

        # -----------------------------------------------------
        # Validation and checkpointing
        # -----------------------------------------------------

        if (
            update % args.eval_every == 0
            or update == args.updates
        ):

            average_total_loss = (
                interval_total_loss / interval_updates
            )

            average_supervised_loss = (
                interval_supervised_loss / interval_updates
            )

            average_unsupervised_loss = (
                interval_unsupervised_loss / interval_updates
            )

            average_coverage = (
                interval_coverage / interval_updates
            )

            metrics = validate(
                student,
                val_loader,
                criterion,
                device
            )

            current_miou = float(metrics["mIoU"])
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"\nValidation at update {update}/{args.updates}\n"
                f"Train total loss: {average_total_loss:.5f}\n"
                f"Train supervised loss: {average_supervised_loss:.5f}\n"
                f"Train unsupervised loss: {average_unsupervised_loss:.5f}\n"
                f"Pseudo-label coverage: {average_coverage:.4f}\n"
                f"Validation loss: {metrics['loss']:.5f}\n"
                f"Pixel accuracy: {metrics['pixel_accuracy']:.4f}\n"
                f"mIoU: {current_miou:.4f}\n"
                f"Dice: {metrics['Dice']:.4f}\n"
                f"Boundary F1: {metrics['Boundary_F1']:.4f}\n"
                f"Learning rate: {current_lr:.7g}\n"
            )

            row = {
                "update": update,
                "average_training_total_loss": average_total_loss,
                "average_supervised_loss": average_supervised_loss,
                "average_unsupervised_loss": average_unsupervised_loss,
                "average_pseudo_label_coverage": average_coverage,
                "validation_loss": metrics["loss"],
                "validation_pixel_accuracy": metrics["pixel_accuracy"],
                "validation_mIoU": current_miou,
                "validation_Dice": metrics["Dice"],
                "validation_Boundary_F1": metrics["Boundary_F1"],
                "learning_rate": current_lr,
                "evaluated_classes": metrics["evaluated_classes"],
                "evaluated_boundary_classes": metrics[
                    "evaluated_boundary_classes"
                ],
                "per_class_IoU": json.dumps(
                    metrics["per_class_IoU"]
                )
            }

            append_metrics(csv_path, row)

            # Save latest student and teacher states.
            save_checkpoint(
                latest_checkpoint,
                student,
                teacher,
                optimizer,
                scheduler,
                update,
                best_miou,
                metrics,
                config
            )

            if current_miou > best_miou:
                best_miou = current_miou

                save_checkpoint(
                    best_checkpoint,
                    student,
                    teacher,
                    optimizer,
                    scheduler,
                    update,
                    best_miou,
                    metrics,
                    config
                )

                print(
                    f"New best checkpoint saved: {best_checkpoint}"
                )

            print(f"Metrics log: {csv_path}\n")

            interval_total_loss = 0.0
            interval_supervised_loss = 0.0
            interval_unsupervised_loss = 0.0
            interval_coverage = 0.0
            interval_updates = 0

    print("========== MEAN TEACHER TRAINING COMPLETE ==========")
    print(f"Total optimizer updates: {args.updates}")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Latest checkpoint: {latest_checkpoint}")
    print(f"Metrics log: {csv_path}")


if __name__ == "__main__":
    main()