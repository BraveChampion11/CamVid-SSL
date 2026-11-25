import torch
from torch.utils.data import DataLoader

from models.easegnet import EASegNet
from datasets.camvid_unlabeled_dataset import CamVidUnlabeledDataset
from datasets.camvid_dataset import CamVidDataset
from evaluation.segmentation_metrics import SegmentationMetrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = (
    "checkpoints/mean_teacher/"
    "easegnet_mean_teacher_seed0_10pct_latest.pth"
)

MANIFEST = "data/splits/seed_0/labeled_10.txt"

# Load datasets
unlabeled_dataset = CamVidUnlabeledDataset(
    labeled_manifest=MANIFEST,
    image_size=(480, 360)
)

val_dataset = CamVidDataset(
    split="val",
    manifest=None,
    image_size=(480, 360),
    augment=False
)

unlabeled_loader = DataLoader(
    unlabeled_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

# Load checkpoint
checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

student = EASegNet(num_classes=11, pretrained=False).to(DEVICE)
teacher = EASegNet(num_classes=11, pretrained=False).to(DEVICE)

student.load_state_dict(checkpoint["student_state_dict"])
teacher.load_state_dict(checkpoint["teacher_state_dict"])

student.eval()
teacher.eval()


def confidence_report(model, name):
    confidences = []

    with torch.no_grad():
        for weak_images, _, _ in unlabeled_loader:
            weak_images = weak_images.to(DEVICE)

            probabilities = torch.softmax(model(weak_images), dim=1)
            confidence, _ = probabilities.max(dim=1)

            confidences.append(confidence.cpu().flatten())

    confidence = torch.cat(confidences)

    print(f"\n{name} CONFIDENCE")
    print("-" * 35)
    print(f"Mean:   {confidence.mean().item():.4f}")
    print(f"Median: {confidence.median().item():.4f}")
    print(f"Maximum: {confidence.max().item():.4f}")

    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
        coverage = (confidence >= threshold).float().mean().item()
        print(f"Coverage >= {threshold:.2f}: {coverage:.4%}")


def validation_report(model, name):
    evaluator = SegmentationMetrics(
        num_classes=11,
        ignore_index=255,
        boundary_tolerance=2
    )

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            evaluator.update(logits.argmax(dim=1), masks)

    metrics = evaluator.compute()

    print(f"\n{name} VALIDATION")
    print("-" * 35)
    print(f"mIoU:       {metrics['mIoU']:.4f}")
    print(f"Dice:       {metrics['Dice']:.4f}")
    print(f"Pixel Acc:  {metrics['pixel_accuracy']:.4f}")
    print(f"Boundary F1:{metrics['Boundary_F1']:.4f}")


print("Device:", DEVICE)
print("Checkpoint update:", checkpoint["update"])

confidence_report(student, "STUDENT")
confidence_report(teacher, "TEACHER")

validation_report(student, "STUDENT")
validation_report(teacher, "TEACHER")

print("\nDIAGNOSTIC COMPLETE")