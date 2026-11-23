#Test the dataset loader
import torch
from torch.utils.data import DataLoader

from datasets.camvid_dataset import CamVidDataset

# Load 10% labeled training data
dataset = CamVidDataset(
    split="train",
    manifest="data/splits/seed_0/labeled_10.txt",
    augment=True
)

print("Dataset size:", len(dataset))
print("Classes:", dataset.class_to_id)

# Load one batch
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

images, masks, filenames = next(iter(loader))

print("\nBatch information:")
print("Images:", images.shape)
print("Masks:", masks.shape)
print("Filenames:", filenames)

print("\nValue checks:")
print("Image range:", images.min().item(), images.max().item())
print("Mask classes:", torch.unique(masks).tolist())

assert images.shape == (2, 3, 360, 480)
assert masks.shape == (2, 360, 480)

valid_classes = set(range(11)) | {255}
assert set(torch.unique(masks).tolist()).issubset(valid_classes)

print("\nDATA PIPELINE TEST PASSED")