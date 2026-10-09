#!/usr/bin/env python3
"""Reconcile Task 2 predictions with deterministic measured accounting."""

import argparse
import gc
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from predict_memory import GRID, build_model
from simplegrad import (
    Adam,
    MemoryLimitExceeded,
    Tensor,
    cross_entropy,
    estimate_largest_batch_size,
    estimate_peak_memory,
    format_bytes,
    get_memory_stats,
    release_graph,
    reset_peak_memory,
    set_memory_limit,
)


INPUT_SHAPE = (1, 28, 28)
DEFAULT_CAP_BYTES = 6_000_000


def _current_commit(repo_root):
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _run_step(model, optimizer, batch_size):
    inputs = Tensor(np.zeros((batch_size,) + INPUT_SHAPE, dtype=np.float32))
    targets = np.arange(batch_size, dtype=np.int64) % 10
    loss = None
    try:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        release_graph(loss)
        optimizer.step()
    finally:
        if loss is not None:
            release_graph(loss)


def _collect_cycles():
    # Collection makes every configuration begin from the same baseline and
    # verifies that explicit graph release left no tracked graph storage live.
    gc.collect()


def measure_configuration(depth, width, batch_size):
    """Measure one steady-state step after lazy Adam initialization."""
    np.random.seed(2026)
    model = build_model(depth, width)
    optimizer = Adam(model.parameters())
    _run_step(model, optimizer, batch_size)
    _collect_cycles()

    reset_peak_memory()
    _run_step(model, optimizer, batch_size)
    _collect_cycles()
    stats = get_memory_stats()
    result = stats.to_dict()

    estimate = estimate_peak_memory(model, batch_size, INPUT_SHAPE)
    expected_persistent = (
        estimate.parameters + estimate.parameters + estimate.optimizer_state
    )
    if stats.current_bytes != expected_persistent:
        raise RuntimeError(
            "measurement did not return to its persistent baseline: "
            f"expected {expected_persistent}, got {stats.current_bytes}"
        )

    del optimizer, model
    gc.collect()
    if get_memory_stats().current_bytes != 0:
        raise RuntimeError("configuration cleanup left tracked allocations live")
    return result


def _comparison(predicted, measured):
    difference = measured - predicted
    percent = 0.0 if measured == 0 else abs(difference) / measured * 100.0
    return {
        "predicted_bytes": predicted,
        "measured_bytes": measured,
        "difference_bytes": difference,
        "absolute_error_percent": round(percent, 6),
    }


def _compare_breakdowns(prediction, measurement):
    measured = measurement["at_peak"]
    return {
        category: _comparison(predicted, measured[category])
        for category, predicted in prediction["breakdown"].items()
    }


def _confirm_under_cap(depth, width, batch_size, cap_bytes):
    """Run a steady-state step under an absolute cap and report the outcome."""
    np.random.seed(2026)
    model = build_model(depth, width)
    optimizer = Adam(model.parameters())
    _run_step(model, optimizer, 1)
    _collect_cycles()
    reset_peak_memory()

    error = None
    fits = True
    try:
        set_memory_limit(cap_bytes)
        _run_step(model, optimizer, batch_size)
    except MemoryLimitExceeded as exc:
        fits = False
        error = {
            "source": exc.source,
            "category": exc.category,
            "current_bytes": exc.current_bytes,
            "requested_bytes": exc.requested_bytes,
            "would_be_bytes": exc.would_be_bytes,
            "limit_bytes": exc.limit_bytes,
        }
    finally:
        set_memory_limit(None)
        _collect_cycles()

    stats = get_memory_stats().to_dict()
    del optimizer, model
    gc.collect()
    if get_memory_stats().current_bytes != 0:
        raise RuntimeError("cap confirmation left tracked allocations live")
    return {"fits": fits, "measurement": stats, "oom": error}


def generate_reconciliation(repo_root, original_path, cap_bytes):
    original = json.loads(original_path.read_text(encoding="utf-8"))
    originals_by_label = {
        item["label"]: item for item in original["configurations"]
    }

    configurations = []
    for label, depth, width, batch_size in GRID:
        model = build_model(depth, width)
        estimate = estimate_peak_memory(model, batch_size, INPUT_SHAPE)
        reconciled = estimate.to_dict()
        del model
        gc.collect()
        measured = measure_configuration(depth, width, batch_size)
        original_prediction = originals_by_label[label]
        configurations.append(
            {
                "label": label,
                "depth": depth,
                "width": width,
                "batch_size": batch_size,
                "original_prediction": {
                    "peak_bytes": original_prediction["peak_bytes"],
                    "breakdown": original_prediction["breakdown"],
                },
                "reconciled_prediction": reconciled,
                "measurement": measured,
                "original_error": _compare_breakdowns(
                    original_prediction, measured
                ),
                "reconciled_error": _compare_breakdowns(reconciled, measured),
                "peak_error": {
                    "original": _comparison(
                        original_prediction["peak_bytes"],
                        measured["peak_bytes"],
                    ),
                    "reconciled": _comparison(
                        reconciled["peak_bytes"], measured["peak_bytes"]
                    ),
                },
            }
        )

    cap_depth = 2
    cap_width = 8
    cap_model = build_model(cap_depth, cap_width)
    largest_batch = estimate_largest_batch_size(
        cap_model, INPUT_SHAPE, cap_bytes
    )
    fitting_prediction = estimate_peak_memory(
        cap_model, largest_batch, INPUT_SHAPE
    )
    rejected_prediction = estimate_peak_memory(
        cap_model, largest_batch + 1, INPUT_SHAPE
    )
    del cap_model
    gc.collect()

    cap_result = {
        "cap_bytes": cap_bytes,
        "cap_display": format_bytes(cap_bytes),
        "depth": cap_depth,
        "width": cap_width,
        "predicted_largest_batch_size": largest_batch,
        "largest_batch_prediction_bytes": fitting_prediction.peak_bytes,
        "next_batch_prediction_bytes": rejected_prediction.peak_bytes,
        "largest_batch_confirmation": _confirm_under_cap(
            cap_depth, cap_width, largest_batch, cap_bytes
        ),
        "next_batch_confirmation": _confirm_under_cap(
            cap_depth, cap_width, largest_batch + 1, cap_bytes
        ),
    }

    return {
        "schema_version": 1,
        "artifact_kind": "task2_prediction_measurement_reconciliation",
        "original_prediction_artifact": str(
            original_path.relative_to(repo_root)
        ),
        "original_prediction_base_commit": original[
            "allocation_policy_base_commit"
        ],
        "measurement_worktree_base_commit": _current_commit(repo_root),
        "measurement_policy": (
            "one steady-state training step after Adam warm-up; deterministic "
            "logical NumPy payload bytes; graph released before Adam; "
            "iterative backward traversal"
        ),
        "margin_percent": 10,
        "configurations": configurations,
        "cap_experiment": cap_result,
    }


def _write_json_atomic(destination, payload):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/task2/predictions-original.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/task2/reconciliation.json"),
    )
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP_BYTES)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cap <= 0:
        raise ValueError("cap must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    original_path = args.predictions
    if not original_path.is_absolute():
        original_path = repo_root / original_path
    output_path = args.output
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    payload = generate_reconciliation(repo_root, original_path, args.cap)
    _write_json_atomic(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
