import gc
import json
import os
import subprocess
import sys

import numpy as np

from simplegrad import (
    Adam,
    Linear,
    Tensor,
    cross_entropy,
    get_memory_event_trace,
    get_memory_stats,
    release_graph,
    reset_peak_memory,
)
from simplegrad.memory import MemoryTracker


def test_memory_tracker_records_current_peak_and_peak_breakdown():
    tracker = MemoryTracker()

    parameter = tracker.reserve(12, "parameters", "test.parameter")
    workspace = tracker.reserve(20, "transient_other", "test.workspace")
    workspace.release()

    stats = tracker.stats()
    assert stats.current_bytes == 12
    assert stats.current.parameters == 12
    assert stats.current.transient_other == 0
    assert stats.peak_bytes == 32
    assert stats.at_peak.parameters == 12
    assert stats.at_peak.transient_other == 20
    assert stats.peak_source == "test.workspace"

    parameter.release()
    assert tracker.stats().current_bytes == 0


def test_tensor_and_module_allocations_use_expected_categories():
    gc.collect()
    baseline = get_memory_stats().current
    reset_peak_memory()

    tensor = Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
    layer = Linear(3, 2)
    current = get_memory_stats().current

    assert current.parameters - baseline.parameters == (3 * 2 + 2) * 4
    assert current.gradients - baseline.gradients == (3 + 3 * 2 + 2) * 4
    assert (
        current.activations_saved_for_backward
        - baseline.activations_saved_for_backward
        == 3 * 4
    )

    del tensor, layer
    gc.collect()
    assert get_memory_stats().current == baseline


def _training_step(model, optimizer):
    inputs = Tensor(np.arange(12, dtype=np.float32).reshape(4, 3))
    logits = model(inputs)
    loss = cross_entropy(logits, np.array([0, 1, 0, 1], dtype=np.int64))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    release_graph(loss)


def test_complete_training_step_cleans_graph_memory_and_has_repeatable_peak():
    gc.collect()
    model = Linear(3, 2)
    optimizer = Adam(model.parameters())

    # Warm up once so both compared intervals start with initialized Adam state.
    _training_step(model, optimizer)
    gc.collect()
    persistent = get_memory_stats().current
    assert persistent.activations_saved_for_backward == 0
    assert persistent.transient_other == 0

    reset_peak_memory()
    _training_step(model, optimizer)
    gc.collect()
    first = get_memory_stats()
    assert first.current == persistent

    reset_peak_memory()
    _training_step(model, optimizer)
    gc.collect()
    second = get_memory_stats()

    assert second.current == persistent
    assert second.peak_bytes == first.peak_bytes
    assert second.at_peak == first.at_peak
    assert second.peak_source == first.peak_source
    assert sum(second.at_peak.to_dict().values()) == second.peak_bytes
    assert get_memory_event_trace()


def test_peak_is_identical_across_hash_seeds():
    program = """
import json
import numpy as np
from simplegrad import Adam, Linear, Tensor, cross_entropy, get_memory_stats, reset_peak_memory

np.random.seed(1)
model = Linear(3, 2)
optimizer = Adam(model.parameters())
reset_peak_memory()
inputs = Tensor(np.ones((4, 3), dtype=np.float32))
loss = cross_entropy(model(inputs), np.array([0, 1, 0, 1]))
loss.backward()
optimizer.step()
stats = get_memory_stats()
print(json.dumps({
    "peak_bytes": stats.peak_bytes,
    "at_peak": stats.at_peak.to_dict(),
    "peak_source": stats.peak_source,
}, sort_keys=True))
"""

    results = []
    for seed in ("1", "999"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        results.append(json.loads(completed.stdout))

    assert results[0] == results[1]
