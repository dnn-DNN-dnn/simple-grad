#!/usr/bin/env python3
"""Compare tracemalloc with simple-grad accounting for one training step."""

import argparse
import gc
import json
import os
import platform
import tracemalloc
from pathlib import Path

import numpy as np

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
    get_memory_stats,
    release_graph,
    reset_peak_memory,
)


def build_model(depth, width):
    """Build the same convolutional architecture as train_mnist.py."""
    final_side = 28 - 2 * depth
    if depth <= 0 or width <= 0 or final_side <= 0:
        raise ValueError("depth and width must preserve a positive spatial size")

    layers = []
    in_channels = 1
    for _ in range(depth):
        layers.extend((Conv2d(in_channels, width, 3), ReLU()))
        in_channels = width
    layers.extend((Flatten(), Linear(width * final_side**2, 10)))
    return Sequential(*layers)


def training_step(model, optimizer, input_data, targets):
    """Run and release one complete forward, backward, and Adam step."""
    inputs = Tensor(input_data)
    optimizer.zero_grad()
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()
    release_graph(loss)
    optimizer.step()
    loss_value = float(loss.data)
    del loss, logits, inputs
    return loss_value


def measure(args):
    rng = np.random.default_rng(args.seed)
    # Benchmark fixtures are intentionally created before tracing. Both
    # trackers therefore measure the simple-grad model and step, not the source
    # batch owned by a dataset or data loader.
    input_data = rng.normal(
        size=(args.batch_size, 1, 28, 28)
    ).astype(np.float32)
    targets = rng.integers(0, 10, size=args.batch_size, dtype=np.int64)

    tracemalloc.start()
    np.random.seed(args.seed)
    model = build_model(args.depth, args.width)
    optimizer = Adam(model.parameters(), lr=1e-3)
    estimate = estimate_peak_memory(model, args.batch_size, (1, 28, 28))

    # Initialize lazy Adam state before measuring the steady-state step.
    training_step(model, optimizer, input_data, targets)
    gc.collect()

    reset_peak_memory()
    tracemalloc.reset_peak()
    logical_baseline = get_memory_stats().current_bytes
    traced_baseline, _ = tracemalloc.get_traced_memory()

    loss_value = training_step(model, optimizer, input_data, targets)
    logical = get_memory_stats()
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    gc.collect()
    tracemalloc.stop()

    logical_increment = logical.peak_bytes - logical_baseline
    traced_increment = traced_peak - traced_baseline
    increment_gap = traced_increment - logical_increment

    return {
        "definition": (
            "one steady-state training step: forward, backward, and Adam; "
            "Adam is warmed up and graph storage is released before reset"
        ),
        "configuration": {
            "batch_size": args.batch_size,
            "depth": args.depth,
            "width": args.width,
            "input_shape": [1, 28, 28],
            "dtype": "float32",
            "seed": args.seed,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "loss": loss_value,
        "analytical_estimate_bytes": estimate.peak_bytes,
        "simplegrad": {
            "baseline_bytes": logical_baseline,
            "peak_bytes": logical.peak_bytes,
            "peak_increment_bytes": logical_increment,
            "at_peak": logical.at_peak.to_dict(),
            "peak_source": logical.peak_source,
        },
        "tracemalloc": {
            "baseline_bytes": traced_baseline,
            "current_after_step_bytes": traced_current,
            "peak_bytes": traced_peak,
            "peak_increment_bytes": traced_increment,
        },
        "comparison": {
            "peak_increment_gap_bytes": increment_gap,
            "tracemalloc_to_simplegrad_increment_ratio": (
                traced_increment / logical_increment
            ),
            "gap_as_percent_of_tracemalloc_increment": (
                100.0 * increment_gap / traced_increment
            ),
        },
        "limitations": [
            "the gap includes unaccounted NumPy temporaries and Python bookkeeping",
            "tracemalloc may miss memory allocated directly by native BLAS/einsum code",
            "the two trackers can reach their peaks at different instants",
            "results are specific to this shape, environment, and NumPy build",
        ],
    }


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/runs/memory-tracker-comparison.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    payload = measure(args)
    write_json_atomic(args.output, payload)

    logical = payload["simplegrad"]
    traced = payload["tracemalloc"]
    comparison = payload["comparison"]
    print(
        f"simplegrad: baseline={logical['baseline_bytes']:,} bytes, "
        f"peak={logical['peak_bytes']:,} bytes, "
        f"increment={logical['peak_increment_bytes']:,} bytes"
    )
    print(
        f"tracemalloc: baseline={traced['baseline_bytes']:,} bytes, "
        f"peak={traced['peak_bytes']:,} bytes, "
        f"increment={traced['peak_increment_bytes']:,} bytes"
    )
    print(
        f"increment gap={comparison['peak_increment_gap_bytes']:,} bytes, "
        "tracemalloc/simplegrad="
        f"{comparison['tracemalloc_to_simplegrad_increment_ratio']:.2f}x"
    )
    print(f"results={args.output.resolve()}")


if __name__ == "__main__":
    main()
