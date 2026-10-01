#!/usr/bin/env python3
"""Generate the immutable, analytical Task 2 prediction grid."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from simplegrad import (
    Conv2d,
    Flatten,
    Linear,
    ReLU,
    Sequential,
    estimate_peak_memory,
    format_bytes,
    format_parameter_count,
)


GRID = (
    ("depth-1", 1, 8, 16),
    ("baseline", 2, 8, 16),
    ("depth-3", 3, 8, 16),
    ("width-4", 2, 4, 16),
    ("width-16", 2, 16, 16),
    ("batch-8", 2, 8, 8),
    ("batch-32", 2, 8, 32),
)


def build_model(depth, width):
    final_side = 28 - 2 * depth
    layers = []
    in_channels = 1
    for _ in range(depth):
        layers.extend((Conv2d(in_channels, width, 3), ReLU()))
        in_channels = width
    layers.extend((Flatten(), Linear(width * final_side**2, 10)))
    return Sequential(*layers)


def current_commit(repo_root):
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def generate_predictions(repo_root):
    np.random.seed(2026)
    configurations = []
    assumptions = None
    for label, depth, width, batch_size in GRID:
        estimate = estimate_peak_memory(
            build_model(depth, width),
            batch_size,
            (1, 28, 28),
        )
        serialized = estimate.to_dict()
        if assumptions is None:
            assumptions = serialized.pop("assumptions")
        else:
            assert serialized.pop("assumptions") == assumptions
        configurations.append(
            {
                "label": label,
                "depth": depth,
                "width": width,
                "batch_size": batch_size,
                **serialized,
                "display": {
                    "num_params": format_parameter_count(
                        estimate.parameter_count
                    ),
                    "memory": format_bytes(estimate.peak_bytes),
                    "breakdown": {
                        name: format_bytes(value)
                        for name, value in estimate.breakdown.to_dict().items()
                    },
                },
            }
        )

    return {
        "schema_version": 1,
        "artifact_kind": "analytical_prediction_only",
        "allocation_policy_base_commit": current_commit(repo_root),
        "estimator_version": 1,
        "input_shape": [1, 28, 28],
        "policy": {
            "dtype": "float32",
            "gradients": "one buffer per trainable parameter",
            "loss": "cross_entropy",
            "optimizer": "two Adam moments per trainable parameter",
            "peak": "parameters + gradients + optimizer state + saved activations",
        },
        "assumptions": assumptions,
        "configurations": configurations,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/task2/predictions-original.json"),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print JSON instead of writing the output file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow replacing an existing prediction artifact",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    predictions = generate_predictions(repo_root)
    rendered = json.dumps(predictions, indent=2) + "\n"
    if args.stdout:
        print(rendered, end="")
        return

    destination = args.output
    if not destination.is_absolute():
        destination = repo_root / destination
    if destination.exists() and not args.force:
        raise FileExistsError(
            f"refusing to overwrite original predictions: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, destination)
    print(destination)


if __name__ == "__main__":
    main()
