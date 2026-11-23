from pathlib import Path
import random

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "CamVid"
SPLIT_DIR = ROOT / "data" / "splits"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# Required labeled subset sizes
SUBSET_SIZES = {
    "10": 37,
    "20": 73,
    "30": 110
}

SEEDS = [0, 1, 2]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def get_image_names(directory):
    """Return sorted image filenames from a directory."""
    return sorted(
        file.name
        for file in directory.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_names(filepath, names):
    """Save one image filename per line."""
    filepath.write_text("\n".join(names) + "\n")


def validate_split(train_names, val_names, test_names, labeled, unlabeled):
    """Check split integrity."""

    train_set = set(train_names)
    val_set = set(val_names)
    test_set = set(test_names)

    labeled_set = set(labeled)
    unlabeled_set = set(unlabeled)

    # No duplicates
    assert len(labeled) == len(labeled_set)
    assert len(unlabeled) == len(unlabeled_set)

    # Labeled and unlabeled must be disjoint
    assert labeled_set.isdisjoint(unlabeled_set)

    # Together they must contain every training image
    assert labeled_set | unlabeled_set == train_set

    # No validation/test leakage
    assert labeled_set.isdisjoint(val_set)
    assert labeled_set.isdisjoint(test_set)
    assert unlabeled_set.isdisjoint(val_set)
    assert unlabeled_set.isdisjoint(test_set)


def main():

    # Read image filenames
    train_names = get_image_names(TRAIN_DIR)
    val_names = get_image_names(VAL_DIR)
    test_names = get_image_names(TEST_DIR)

    print("Dataset sizes:")
    print(f"Train: {len(train_names)}")
    print(f"Val:   {len(val_names)}")
    print(f"Test:  {len(test_names)}")

    # Confirm expected dataset size
    assert len(train_names) == 367
    assert len(val_names) == 101
    assert len(test_names) == 233

    # Check that split filenames do not overlap
    assert set(train_names).isdisjoint(set(val_names))
    assert set(train_names).isdisjoint(set(test_names))
    assert set(val_names).isdisjoint(set(test_names))

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:

        print(f"\nGenerating splits for seed {seed}...")

        # Reproducible random ordering
        shuffled = train_names.copy()
        random.Random(seed).shuffle(shuffled)

        # Nested labeled subsets
        labeled_10 = shuffled[:SUBSET_SIZES["10"]]
        labeled_20 = shuffled[:SUBSET_SIZES["20"]]
        labeled_30 = shuffled[:SUBSET_SIZES["30"]]

        labeled_subsets = {
            "10": labeled_10,
            "20": labeled_20,
            "30": labeled_30
        }

        seed_dir = SPLIT_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for percentage, labeled in labeled_subsets.items():

            labeled_set = set(labeled)

            # Unlabeled pool is the complement of labeled pool
            unlabeled = [
                name for name in train_names
                if name not in labeled_set
            ]

            # Validate each pair
            validate_split(
                train_names,
                val_names,
                test_names,
                labeled,
                unlabeled
            )

            # Save manifests
            save_names(
                seed_dir / f"labeled_{percentage}.txt",
                labeled
            )

            save_names(
                seed_dir / f"unlabeled_{percentage}.txt",
                unlabeled
            )

            print(
                f"{percentage}%: "
                f"Labeled = {len(labeled)}, "
                f"Unlabeled = {len(unlabeled)}"
            )

        # Verify nested relationship
        assert set(labeled_10).issubset(set(labeled_20))
        assert set(labeled_20).issubset(set(labeled_30))

        print("Nested subset checks: PASSED")

    print("\nAll splits generated and validated successfully.")
    print(f"Saved in: {SPLIT_DIR}")


if __name__ == "__main__":
    main()