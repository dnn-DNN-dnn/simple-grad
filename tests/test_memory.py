import gc
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from simplegrad import (
    Adam,
    Conv2d,
    Flatten,
    Linear,
    MemoryLimitExceeded,
    ReLU,
    Sequential,
    Tensor,
    cross_entropy,
    get_memory_event_trace,
    get_memory_limit,
    get_memory_stats,
    release_graph,
    reset_peak_memory,
    set_memory_limit,
)
from simplegrad.memory import MemoryTracker, create_array


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


def test_memory_limit_allows_exact_boundary_and_rejects_atomically():
    tracker = MemoryTracker()
    tracker.set_limit(20)

    allocation = tracker.reserve(20, "parameters", "test.exact_limit")
    before = tracker.stats()
    with pytest.raises(MemoryLimitExceeded) as raised:
        tracker.reserve(1, "transient_other", "test.too_large")

    error = raised.value
    assert error.limit_bytes == 20
    assert error.current_bytes == 20
    assert error.requested_bytes == 1
    assert error.would_be_bytes == 21
    assert error.category == "transient_other"
    assert error.source == "test.too_large"
    assert tracker.stats() == before

    allocation.release()
    tracker.set_limit(0)
    assert tracker.limit_bytes == 0
    tracker.set_limit(None)
    assert tracker.limit_bytes is None


def test_setting_limit_below_current_memory_fails_without_enabling_it():
    tracker = MemoryTracker()
    allocation = tracker.reserve(8, "parameters", "test.parameter")

    with pytest.raises(MemoryLimitExceeded) as raised:
        tracker.set_limit(7)

    assert raised.value.current_bytes == 8
    assert raised.value.requested_bytes == 0
    assert tracker.limit_bytes is None
    allocation.release()


@pytest.mark.parametrize(
    ("value", "error"),
    [(-1, ValueError), (True, TypeError), (1.5, TypeError), ("10", TypeError)],
)
def test_memory_limit_rejects_invalid_values(value, error):
    with pytest.raises(error):
        set_memory_limit(value)
    assert get_memory_limit() is None


def test_array_creation_rolls_back_reservation_when_factory_fails():
    def fail():
        raise RuntimeError("native allocation failed")

    # Exercise the same reserve/create/rollback ordering on the global helper
    # while comparing against its live baseline.
    gc.collect()
    baseline = get_memory_stats().current_bytes
    with pytest.raises(RuntimeError, match="native allocation failed"):
        create_array(
            (4,),
            np.float32,
            "transient_other",
            "test.failed_factory",
            fail,
        )
    assert get_memory_stats().current_bytes == baseline


def test_cap_rejection_happens_before_array_factory_runs():
    gc.collect()
    baseline = get_memory_stats().current_bytes
    factory_called = False

    def factory():
        nonlocal factory_called
        factory_called = True
        return np.zeros(1, dtype=np.float32)

    set_memory_limit(baseline)
    try:
        with pytest.raises(MemoryLimitExceeded):
            create_array(
                (1,),
                np.float32,
                "transient_other",
                "test.preallocation_rejection",
                factory,
            )
    finally:
        set_memory_limit(None)

    assert not factory_called
    assert get_memory_stats().current_bytes == baseline


def test_tensor_and_module_allocations_use_expected_categories():
    gc.collect()
    baseline = get_memory_stats().current
    reset_peak_memory()

    tensor = Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
    layer = Linear(3, 2)
    current = get_memory_stats().current

    assert current.parameters - baseline.parameters == (3 * 2 + 2) * 4
    assert current.gradients - baseline.gradients == (3 * 2 + 2) * 4
    assert current.transient_other - baseline.transient_other == 3 * 4
    assert (
        current.activations_saved_for_backward
        - baseline.activations_saved_for_backward
        == 3 * 4
    )

    del tensor, layer
    gc.collect()
    assert get_memory_stats().current == baseline


def test_conv2d_reuses_one_tracked_einsum_workspace_per_phase():
    """Workspace event count is constant rather than proportional to pixels."""
    gc.collect()
    layer = Conv2d(1, 2, 3)
    inputs = Tensor(np.ones((2, 1, 5, 5), dtype=np.float32), requires_grad=True)
    reset_peak_memory()

    output = layer(inputs)
    output.backward(np.ones_like(output.data))
    events = get_memory_event_trace()

    for source in (
        "conv2d.forward_einsum_workspace",
        "conv2d.backward_einsum_workspace",
    ):
        matching = [event.action for event in events if event.source == source]
        assert matching == ["allocate", "release"]
    assert not any(
        event.source
        in {
            "conv2d.forward_einsum",
            "conv2d.backward_input_einsum",
            "conv2d.backward_weight_einsum",
        }
        for event in events
    )

    release_graph(output)
    del output, inputs, layer
    gc.collect()


def _training_step(model, optimizer):
    inputs = Tensor(np.arange(12, dtype=np.float32).reshape(4, 3))
    logits = model(inputs)
    loss = cross_entropy(logits, np.array([0, 1, 0, 1], dtype=np.int64))
    optimizer.zero_grad()
    loss.backward()
    release_graph(loss)
    optimizer.step()


def test_complete_training_step_cleans_graph_memory_and_has_repeatable_peak():
    gc.collect()
    model = Linear(3, 2)
    optimizer = Adam(model.parameters())

    # Warm up once so both compared intervals start with initialized Adam state.
    _training_step(model, optimizer)
    persistent = get_memory_stats().current
    assert persistent.activations_saved_for_backward == 0
    assert persistent.transient_other == 0

    reset_peak_memory()
    _training_step(model, optimizer)
    first = get_memory_stats()
    assert first.current == persistent

    reset_peak_memory()
    _training_step(model, optimizer)
    second = get_memory_stats()

    assert second.current == persistent
    assert second.peak_bytes == first.peak_bytes
    assert second.at_peak == first.at_peak
    assert second.peak_source == first.peak_source
    assert sum(second.at_peak.to_dict().values()) == second.peak_bytes
    assert get_memory_event_trace()


def test_iterative_backward_releases_graph_without_cyclic_collection():
    """Graph cleanup returns to baseline even while cyclic GC is disabled."""
    gc.collect()
    model = Linear(3, 2)
    optimizer = Adam(model.parameters())
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        _training_step(model, optimizer)
        current = get_memory_stats().current
        assert current.activations_saved_for_backward == 0
        assert current.transient_other == 0
    finally:
        if was_enabled:
            gc.enable()

    del optimizer, model
    gc.collect()


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
from simplegrad import release_graph
release_graph(loss)
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


def test_tensor_construction_oom_rolls_back_partially_created_data():
    gc.collect()
    baseline = get_memory_stats().current_bytes
    set_memory_limit(baseline + 16)
    try:
        with pytest.raises(MemoryLimitExceeded) as raised:
            Tensor(np.ones(4, dtype=np.float32), requires_grad=True)

        assert raised.value.source == "tensor.grad"
        assert raised.value.requested_bytes == 16
        assert get_memory_stats().current_bytes == baseline
    finally:
        set_memory_limit(None)

    assert get_memory_limit() is None


def _safe_training_step(model, optimizer):
    loss = None
    try:
        inputs = Tensor(np.arange(12, dtype=np.float32).reshape(4, 3))
        loss = cross_entropy(
            model(inputs),
            np.array([0, 1, 0, 1], dtype=np.int64),
        )
        optimizer.zero_grad()
        loss.backward()
        release_graph(loss)
        optimizer.step()
    finally:
        if loss is not None:
            release_graph(loss)


def test_complete_training_step_obeys_exact_cap_and_next_lower_cap_ooms():
    gc.collect()
    model = Linear(3, 2)
    optimizer = Adam(model.parameters())

    # Initialize lazy Adam state, then measure a steady-state step.
    _safe_training_step(model, optimizer)
    gc.collect()
    reset_peak_memory()
    _safe_training_step(model, optimizer)
    gc.collect()
    required_peak = get_memory_stats().peak_bytes

    set_memory_limit(required_peak)
    try:
        _safe_training_step(model, optimizer)
        gc.collect()
        assert get_memory_stats().peak_bytes <= required_peak
    finally:
        set_memory_limit(None)

    reset_peak_memory()
    set_memory_limit(required_peak - 1)
    try:
        with pytest.raises(MemoryLimitExceeded) as raised:
            _safe_training_step(model, optimizer)
        assert raised.value.would_be_bytes > required_peak - 1
        assert get_memory_stats().current_bytes <= required_peak - 1
    finally:
        set_memory_limit(None)


def test_task1_convnet_configuration_raises_inside_allocation_path():
    gc.collect()
    model = Sequential(
        Conv2d(1, 2, 3),
        ReLU(),
        Flatten(),
        Linear(18, 3),
    )
    optimizer = Adam(model.parameters())
    baseline = get_memory_stats().current_bytes
    input_bytes = 4 * 1 * 5 * 5 * np.dtype(np.float32).itemsize
    inputs = None
    loss = None

    # The input fits exactly in the remaining budget; the first convolution
    # workspace is the allocation that must be rejected inside simple-grad.
    set_memory_limit(baseline + input_bytes)
    try:
        with pytest.raises(MemoryLimitExceeded) as raised:
            inputs = Tensor(np.ones((4, 1, 5, 5), dtype=np.float32))
            loss = cross_entropy(
                model(inputs),
                np.array([0, 1, 2, 0], dtype=np.int64),
            )
            optimizer.zero_grad()
            loss.backward()
            release_graph(loss)
            optimizer.step()

        assert raised.value.source == "conv2d.forward_output_workspace"
        assert raised.value.current_bytes == baseline + input_bytes
        assert raised.value.would_be_bytes > raised.value.limit_bytes
    finally:
        if loss is not None:
            release_graph(loss)
        set_memory_limit(None)
        del inputs, loss
        gc.collect()
