#!/usr/bin/env python3
"""Benchmark training-step overhead from Conv2D memory accounting."""

import argparse
import gc
import json
import os
import statistics
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

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
    release_graph,
)


CONV_PIXEL_TEMPORARIES = {
    "conv2d.forward_einsum",
    "conv2d.backward_input_einsum",
    "conv2d.backward_weight_einsum",
}
CONV_REUSED_WORKSPACES = {
    "conv2d.forward_einsum_workspace",
    "conv2d.backward_einsum_workspace",
}


def build_model(depth, width):
    final_side = 28 - 2 * depth
    layers = []
    in_channels = 1
    for _ in range(depth):
        layers.extend((Conv2d(in_channels, width, 3), ReLU()))
        in_channels = width
    layers.extend((Flatten(), Linear(width * final_side**2, 10)))
    return Sequential(*layers)


def bypass_conv_pixel_accounting():
    """Disable Conv2D contraction-workspace accounting for diagnostic A/B."""
    import simplegrad.ops as ops

    original = getattr(ops, "temporary_result", None)
    if original is None:
        raise RuntimeError("this source tree does not contain Task 2 accounting")

    @contextmanager
    def selective_temporary_result(shape, dtype, source, factory):
        if source in CONV_PIXEL_TEMPORARIES:
            yield factory()
            return
        with original(shape, dtype, source, factory) as array:
            yield array

    original_allocate_empty = ops.allocate_empty

    class NoopHandle:
        def release(self):
            pass

    def selective_allocate_empty(shape, dtype, category, source):
        if source in CONV_REUSED_WORKSPACES:
            return np.empty(shape, dtype=dtype), NoopHandle()
        return original_allocate_empty(shape, dtype, category, source)

    ops.temporary_result = selective_temporary_result
    ops.allocate_empty = selective_allocate_empty


def run_step(model, optimizer, input_data, targets, lifecycle):
    phase = {}
    started = perf_counter()
    inputs = Tensor(input_data)
    optimizer.zero_grad()
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    phase["forward_seconds"] = perf_counter() - started

    started = perf_counter()
    loss.backward()
    phase["backward_seconds"] = perf_counter() - started

    if lifecycle == "release-before-adam":
        started = perf_counter()
        release_graph(loss)
        phase["release_seconds"] = perf_counter() - started

        started = perf_counter()
        optimizer.step()
        phase["optimizer_seconds"] = perf_counter() - started
    else:
        started = perf_counter()
        optimizer.step()
        phase["optimizer_seconds"] = perf_counter() - started

        started = perf_counter()
        release_graph(loss)
        phase["release_seconds"] = perf_counter() - started

    checksum = float(loss.data)
    checksum += float(logits.data.sum(dtype=np.float64))
    del loss, logits, inputs
    return phase, checksum


def median(values):
    return statistics.median(values)


def benchmark(args):
    if args.mode == "bypass-conv-accounting":
        bypass_conv_pixel_accounting()

    rng = np.random.default_rng(args.seed)
    input_data = rng.normal(
        size=(args.batch_size, 1, 28, 28)
    ).astype(np.float32)
    targets = rng.integers(0, 10, size=args.batch_size, dtype=np.int64)

    np.random.seed(args.seed)
    model = build_model(args.depth, args.width)
    optimizer = Adam(model.parameters(), lr=1e-3)

    checksum = 0.0
    for _ in range(args.warmup):
        gc.collect()
        _, checksum = run_step(
            model, optimizer, input_data, targets, args.lifecycle
        )

    samples = []
    for _ in range(args.iterations):
        # Collection before timing prevents completed Task 1 traversal cycles
        # from accumulating and is excluded from both implementations.
        gc.collect()
        started = perf_counter()
        phase, checksum = run_step(
            model, optimizer, input_data, targets, args.lifecycle
        )
        phase["step_seconds"] = perf_counter() - started
        samples.append(phase)

    keys = sorted(samples[0])
    return {
        "configuration": {
            "depth": args.depth,
            "width": args.width,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "mode": args.mode,
            "lifecycle": args.lifecycle,
        },
        "median": {key: median([sample[key] for sample in samples]) for key in keys},
        "checksum": checksum,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--mode",
        choices=("standard", "bypass-conv-accounting"),
        default="standard",
    )
    parser.add_argument(
        "--lifecycle",
        choices=("task1", "release-before-adam"),
        default="release-before-adam",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = benchmark(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
