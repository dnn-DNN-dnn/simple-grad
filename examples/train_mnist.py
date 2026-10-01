#!/usr/bin/env python3
"""Train a small MNIST classifier entirely with simple-grad."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from datasets import load_from_disk

from simplegrad import (
    Adam,
    Conv2d,
    Flatten,
    Linear,
    ReLU,
    Sequential,
    Tensor,
    cross_entropy,
    estimate_peak_memory,
    format_memory_estimate,
    release_graph,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("tmp/data/mnist"))
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/runs/mnist/task1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def load_mnist(path, train_size, test_size, seed):
    """Load deterministic subsets from a DatasetDict saved on disk."""
    dataset = load_from_disk(str(path))
    if "train" not in dataset or "test" not in dataset:
        raise ValueError("MNIST DatasetDict must contain train and test splits")

    def load_split(name, size):
        split = dataset[name]
        if not {"image", "label"}.issubset(split.column_names):
            raise ValueError(f"{name} split must contain image and label columns")
        if size <= 0 or size > len(split):
            raise ValueError(f"{name}_size must be between 1 and {len(split)}")

        split = split.shuffle(seed=seed).select(range(size))
        images = np.stack(
            [np.asarray(image, dtype=np.float32) for image in split["image"]]
        )
        if images.shape[1:] != (28, 28):
            raise ValueError(f"expected 28x28 MNIST images, got {images.shape[1:]}")
        images = images[:, None, :, :] / np.float32(255.0)
        labels = np.asarray(split["label"], dtype=np.int64)
        return images, labels

    train_images, train_labels = load_split("train", train_size)
    test_images, test_labels = load_split("test", test_size)
    return train_images, train_labels, test_images, test_labels


def build_model(depth, width):
    """Build repeated Conv/ReLU blocks followed by a ten-class Linear."""
    final_side = 28 - 2 * depth
    if depth <= 0 or width <= 0 or final_side <= 0:
        raise ValueError("depth and width must be positive and preserve spatial size")

    layers = []
    in_channels = 1
    for _ in range(depth):
        layers += [Conv2d(in_channels, width, 3), ReLU()]
        in_channels = width
    layers += [Flatten(), Linear(width * final_side**2, 10)]
    return Sequential(*layers)


def train_epoch(model, optimizer, images, labels, batch_size, rng):
    """Train on every example once and return weighted loss and accuracy."""
    order = rng.permutation(len(images))
    loss_sum = 0.0
    correct = 0

    for start in range(0, len(images), batch_size):
        indices = order[start : start + batch_size]
        inputs = Tensor(images[indices])
        targets = labels[indices]

        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.data) * len(indices)
        correct += int(np.sum(np.argmax(logits.data, axis=1) == targets))
        release_graph(loss)

    return loss_sum / len(images), correct / len(images)


def evaluate(model, images, labels, batch_size):
    """Return held-out correct count and accuracy."""
    correct = 0
    for start in range(0, len(images), batch_size):
        inputs = Tensor(images[start : start + batch_size])
        targets = labels[start : start + batch_size]
        logits = model(inputs)
        correct += int(np.sum(np.argmax(logits.data, axis=1) == targets))
        release_graph(logits)
    return correct, correct / len(images)


def write_metrics(path, metrics):
    """Write metrics atomically to avoid leaving a partial result file."""
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / ".metrics.json.tmp"
    destination = path / "metrics.json"
    temporary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    train_images, train_labels, test_images, test_labels = load_mnist(
        args.data_dir,
        args.train_size,
        args.test_size,
        args.seed,
    )

    np.random.seed(args.seed)
    model = build_model(args.depth, args.width)
    memory_estimate = estimate_peak_memory(
        model,
        args.batch_size,
        tuple(train_images.shape[1:]),
    )
    print(format_memory_estimate(memory_estimate), flush=True)
    print(
        "memory breakdown: "
        f"parameters_bytes={memory_estimate.parameters} "
        f"gradients_bytes={memory_estimate.gradients} "
        f"optimizer_state_bytes={memory_estimate.optimizer_state} "
        "saved_activations_bytes="
        f"{memory_estimate.activations_saved_for_backward} "
        f"transient_other_bytes={memory_estimate.transient_other}",
        flush=True,
    )
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed + 1)

    metrics = {
        "config": vars(args).copy(),
        "memory_estimate": memory_estimate.to_dict(),
        "epochs": [],
    }
    metrics["config"]["data_dir"] = str(args.data_dir)
    metrics["config"]["output_dir"] = str(args.output_dir)

    process_started = perf_counter()
    total_training_seconds = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_started = perf_counter()
        print(f"epoch={epoch}/{args.epochs} training...", flush=True)
        training_started = perf_counter()
        loss, train_accuracy = train_epoch(
            model,
            optimizer,
            train_images,
            train_labels,
            args.batch_size,
            rng,
        )
        training_seconds = perf_counter() - training_started
        total_training_seconds += training_seconds

        evaluation_started = perf_counter()
        correct, test_accuracy = evaluate(
            model,
            test_images,
            test_labels,
            args.batch_size,
        )
        evaluation_seconds = perf_counter() - evaluation_started
        epoch_seconds = perf_counter() - epoch_started
        metrics["epochs"].append(
            {
                "epoch": epoch,
                "loss": loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "test_correct": correct,
                "test_total": len(test_images),
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "epoch_seconds": epoch_seconds,
            }
        )
        print(
            f"epoch={epoch}/{args.epochs} loss={loss:.6f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"test_accuracy={test_accuracy:.4f} ({correct}/{len(test_images)}) "
            f"train_time={training_seconds:.3f}s epoch_time={epoch_seconds:.3f}s",
            flush=True,
        )

    metrics["final_accuracy"] = metrics["epochs"][-1]["test_accuracy"]
    metrics["total_training_seconds"] = total_training_seconds
    metrics["total_process_seconds"] = perf_counter() - process_started
    destination = write_metrics(args.output_dir, metrics)
    print(
        f"total_train_time={metrics['total_training_seconds']:.3f}s "
        f"total_process_time={metrics['total_process_seconds']:.3f}s"
    )
    print(f"metrics={destination.resolve()}")


if __name__ == "__main__":
    main()
