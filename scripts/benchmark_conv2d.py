#!/usr/bin/env python3
"""Compare three Conv2d implementations over forward/backward steps."""

import argparse
import gc
import json
import os
import platform
import statistics
import tracemalloc
from pathlib import Path
from time import perf_counter

import numpy as np

from simplegrad import (
    Tensor,
    conv2d,
    conv2d_im2col,
    conv2d_strided,
    release_graph,
)


OPERATIONS = {
    "patch": conv2d,
    "strided": conv2d_strided,
    "im2col": conv2d_im2col,
}


def _step(operation, x, weight, bias, upstream):
    x.grad.fill(0)
    weight.grad.fill(0)
    bias.grad.fill(0)
    output = operation(x, weight, bias)
    output.backward(upstream)
    checksum = float(output.data.sum(dtype=np.float64))
    checksum += float(x.grad.sum(dtype=np.float64))
    checksum += float(weight.grad.sum(dtype=np.float64))
    checksum += float(bias.grad.sum(dtype=np.float64))
    release_graph(output)
    return checksum


def benchmark_operation(
    name,
    operation,
    *,
    batch_size,
    in_channels,
    out_channels,
    side,
    kernel_size,
    warmup,
    iterations,
    seed,
):
    rng = np.random.default_rng(seed)
    x = Tensor(
        rng.normal(size=(batch_size, in_channels, side, side)).astype(np.float32),
        requires_grad=True,
    )
    weight = Tensor(
        rng.normal(
            size=(out_channels, in_channels, kernel_size, kernel_size)
        ).astype(np.float32),
        requires_grad=True,
    )
    bias = Tensor(
        rng.normal(size=(out_channels,)).astype(np.float32),
        requires_grad=True,
    )
    output_side = side - kernel_size + 1
    upstream = rng.normal(
        size=(batch_size, out_channels, output_side, output_side)
    ).astype(np.float32)

    checksum = 0.0
    for _ in range(warmup):
        gc.collect()
        checksum = _step(operation, x, weight, bias, upstream)

    durations = []
    for _ in range(iterations):
        gc.collect()
        started = perf_counter()
        checksum = _step(operation, x, weight, bias, upstream)
        durations.append(perf_counter() - started)

    gc.collect()
    tracemalloc.start()
    baseline_current, _ = tracemalloc.get_traced_memory()
    checksum = _step(operation, x, weight, bias, upstream)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    return {
        "implementation": name,
        "median_step_seconds": statistics.median(durations),
        "min_step_seconds": min(durations),
        "max_step_seconds": max(durations),
        "traced_peak_increment_bytes": peak - baseline_current,
        "checksum": checksum,
    }


def run_benchmark(args):
    if min(
        args.batch_size,
        args.in_channels,
        args.out_channels,
        args.side,
        args.kernel_size,
        args.iterations,
    ) <= 0:
        raise ValueError("shape arguments and iterations must be positive")
    if args.kernel_size > args.side:
        raise ValueError("kernel_size must not exceed side")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")

    results = [
        benchmark_operation(
            name,
            operation,
            batch_size=args.batch_size,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            side=args.side,
            kernel_size=args.kernel_size,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
        )
        for name, operation in OPERATIONS.items()
    ]
    patch = results[0]
    for result in results[1:]:
        if not np.isclose(
            patch["checksum"], result["checksum"], rtol=1e-4, atol=1e-5
        ):
            raise RuntimeError(
                f"patch and {result['implementation']} produced different "
                "benchmark checksums"
            )

    return {
        "definition": (
            "one step is Conv2d forward plus backward for input, weight, and bias; "
            "setup and cyclic-garbage collection are excluded from timing"
        ),
        "memory_metric": (
            "peak Python/NumPy allocation bytes traced during one isolated step, "
            "above the pre-step traced baseline"
        ),
        "configuration": {
            "batch_size": args.batch_size,
            "in_channels": args.in_channels,
            "out_channels": args.out_channels,
            "side": args.side,
            "kernel_size": args.kernel_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "dtype": "float32",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "results": results,
        "comparison_to_patch": {
            result["implementation"]: {
                "speedup": (
                    patch["median_step_seconds"]
                    / result["median_step_seconds"]
                ),
                "peak_memory_ratio": (
                    result["traced_peak_increment_bytes"]
                    / patch["traced_peak_increment_bytes"]
                ),
            }
            for result in results[1:]
        },
    }


def _write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=8)
    parser.add_argument("--out-channels", type=int, default=8)
    parser.add_argument("--side", type=int, default=26)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/runs/conv2d-benchmark.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload = run_benchmark(args)
    _write_json_atomic(args.output, payload)

    for result in payload["results"]:
        print(
            f"{result['implementation']}: "
            f"median={result['median_step_seconds'] * 1000:.3f} ms, "
            f"peak={result['traced_peak_increment_bytes']:,} bytes"
        )
    for name, comparison in payload["comparison_to_patch"].items():
        print(
            f"{name} vs patch: speedup={comparison['speedup']:.2f}x, "
            f"peak memory ratio={comparison['peak_memory_ratio']:.2f}x"
        )
    print(f"results={args.output.resolve()}")


if __name__ == "__main__":
    main()
