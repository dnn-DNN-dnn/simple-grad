#!/usr/bin/env python3
"""Download MNIST once and save it in train_mnist.py's local format."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from datasets import load_dataset, load_from_disk


REQUIRED_SPLITS = ("train", "test")
REQUIRED_COLUMNS = {"image", "label"}


def validate_mnist(dataset):
    """Validate the structure consumed by examples/train_mnist.py."""
    missing_splits = [name for name in REQUIRED_SPLITS if name not in dataset]
    if missing_splits:
        raise ValueError(f"MNIST is missing splits: {missing_splits}")

    for name in REQUIRED_SPLITS:
        split = dataset[name]
        missing_columns = REQUIRED_COLUMNS.difference(split.column_names)
        if missing_columns:
            raise ValueError(
                f"MNIST {name!r} is missing columns: {sorted(missing_columns)}"
            )
        if len(split) == 0:
            raise ValueError(f"MNIST {name!r} split must not be empty")


def ensure_mnist(destination, dataset_name="ylecun/mnist"):
    """Download MNIST unless a valid saved DatasetDict already exists."""
    destination = Path(destination)
    if destination.exists():
        try:
            dataset = load_from_disk(str(destination))
            validate_mnist(dataset)
        except Exception as exc:
            raise RuntimeError(
                f"{destination} exists but is not a valid saved MNIST dataset; "
                "remove it or choose another --output directory"
            ) from exc
        print(
            f"MNIST already present at {destination} "
            f"(train={len(dataset['train']):,}, test={len(dataset['test']):,})"
        )
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-download-",
            dir=destination.parent,
        )
    )
    staging_dataset = staging_root / "dataset"
    try:
        print(f"Downloading {dataset_name}...")
        dataset = load_dataset(dataset_name)
        validate_mnist(dataset)
        dataset.save_to_disk(str(staging_dataset))
        os.replace(staging_dataset, destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    print(
        f"Saved MNIST to {destination} "
        f"(train={len(dataset['train']):,}, test={len(dataset['test']):,})"
    )
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/data/mnist"),
        help="local DatasetDict directory used by train_mnist.py",
    )
    parser.add_argument(
        "--dataset",
        default="ylecun/mnist",
        help="Hugging Face dataset identifier",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_mnist(args.output, args.dataset)


if __name__ == "__main__":
    main()
