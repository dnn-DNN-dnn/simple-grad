import pytest

from simplegrad import (
    Conv2d,
    Flatten,
    Linear,
    MemoryBreakdown,
    MemoryEstimate,
    ReLU,
    Sequential,
    estimate_peak_memory,
    format_bytes,
    format_memory_estimate,
    format_parameter_count,
)
from simplegrad.module import Module
from simplegrad.tensor import Tensor


def build_tiny_model():
    return Sequential(
        Conv2d(1, 2, 3),
        ReLU(),
        Flatten(),
        Linear(18, 3),
    )


def build_mnist_model(depth=2, width=8):
    side = 28 - 2 * depth
    layers = []
    in_channels = 1
    for _ in range(depth):
        layers.extend((Conv2d(in_channels, width, 3), ReLU()))
        in_channels = width
    layers.extend((Flatten(), Linear(width * side**2, 10)))
    return Sequential(*layers)


def test_result_invariants_and_serialization():
    breakdown = MemoryBreakdown(
        parameters=4,
        gradients=4,
        optimizer_state=8,
        activations_saved_for_backward=12,
    )
    estimate = MemoryEstimate(
        parameter_count=1,
        peak_bytes=28,
        breakdown=breakdown,
        assumptions=("test assumption",),
    )

    assert breakdown.total == breakdown.total_bytes == 28
    assert estimate.at_peak is breakdown
    assert estimate.parameters == 4
    assert estimate.transient_other == 0
    assert estimate.to_dict() == {
        "parameter_count": 1,
        "peak_bytes": 28,
        "breakdown": breakdown.to_dict(),
        "assumptions": ["test assumption"],
    }

    with pytest.raises(ValueError, match="breakdown total"):
        MemoryEstimate(1, 29, breakdown)
    with pytest.raises(TypeError, match="integer"):
        MemoryBreakdown(parameters=True)


def test_human_readable_formatters_preserve_exact_values():
    assert format_parameter_count(2_345_678) == "2.3M (2,345,678)"
    assert format_parameter_count(46_754) == "0.05M (46,754)"
    assert format_bytes(3_000_123) == "3 MB (3,000,123 bytes)"
    assert format_bytes(2_376_480) == "2.4 MB (2,376,480 bytes)"
    assert format_bytes(999) == "999 B (999 bytes)"

    estimate = estimate_peak_memory(build_tiny_model(), 4, (1, 5, 5))
    assert format_memory_estimate(estimate) == (
        "num params: 0.0001M (77), memory: 3.7 KB (3,736 bytes)"
    )

    with pytest.raises(ValueError, match="non-negative"):
        format_bytes(-1)
    with pytest.raises(TypeError, match="MemoryEstimate"):
        format_memory_estimate(object())


def test_tiny_model_matches_hand_calculation():
    estimate = estimate_peak_memory(build_tiny_model(), 4, (1, 5, 5))

    # Trainable parameters:
    # Conv: 2*1*3*3 + 2 = 20
    # Linear: 3*18 + 3 = 57
    assert estimate.parameter_count == 77
    assert estimate.parameters == 77 * 4
    assert estimate.gradients == 77 * 4
    assert estimate.optimizer_state == 77 * 2 * 4

    # Saved activations:
    # input=100, Conv=72, ReLU=72, Flatten=72, Linear=12,
    # cross-entropy probabilities=12.
    assert estimate.activations_saved_for_backward == 340 * 4
    # Adam dominates the simple transient model: five arrays for the largest
    # parameter (54 float32 elements), plus two retained int64 batch vectors.
    assert estimate.transient_other == 5 * 54 * 4 + 4 * 2 * 8
    assert estimate.peak_bytes == 3736


def test_estimator_does_not_execute_supported_modules(monkeypatch):
    model = build_tiny_model()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("model execution is forbidden")

    for module_type in (Sequential, Conv2d, ReLU, Flatten, Linear):
        monkeypatch.setattr(module_type, "__call__", fail_if_called)

    assert estimate_peak_memory(model, 4, (1, 5, 5)).peak_bytes > 0


def test_estimate_is_repeatable():
    model = build_mnist_model()

    assert estimate_peak_memory(model, 16, (1, 28, 28)) == (
        estimate_peak_memory(model, 16, (1, 28, 28))
    )


def test_batch_size_changes_activations_and_may_change_transient_workspace():
    model = build_mnist_model(depth=2, width=8)
    small = estimate_peak_memory(model, 8, (1, 28, 28))
    large = estimate_peak_memory(model, 16, (1, 28, 28))

    assert small.parameter_count == large.parameter_count
    assert small.parameters == large.parameters
    assert small.gradients == large.gradients
    assert small.optimizer_state == large.optimizer_state
    assert large.activations_saved_for_backward == (
        2 * small.activations_saved_for_backward
    )
    assert large.peak_bytes - small.peak_bytes == (
        small.activations_saved_for_backward
        + large.transient_other
        - small.transient_other
    )


def test_transient_workspace_is_nonzero_and_scales_for_large_batches():
    model = build_mnist_model(depth=2, width=8)
    small = estimate_peak_memory(model, 1, (1, 28, 28))
    large = estimate_peak_memory(model, 128, (1, 28, 28))

    assert small.transient_other > 0
    assert large.transient_other > small.transient_other


def test_depth_and_width_change_prediction():
    shallow = estimate_peak_memory(build_mnist_model(1, 8), 4, (1, 28, 28))
    deep = estimate_peak_memory(build_mnist_model(3, 8), 4, (1, 28, 28))
    narrow = estimate_peak_memory(build_mnist_model(2, 4), 4, (1, 28, 28))
    wide = estimate_peak_memory(build_mnist_model(2, 16), 4, (1, 28, 28))

    assert shallow.peak_bytes != deep.peak_bytes
    assert narrow.parameter_count < wide.parameter_count
    assert narrow.peak_bytes < wide.peak_bytes


def test_shared_parameters_count_once_and_activations_count_per_call():
    shared = Linear(3, 3)
    shared_model = Sequential(Flatten(), shared, ReLU(), shared)
    single_model = Sequential(Flatten(), Linear(3, 3))

    shared_estimate = estimate_peak_memory(shared_model, 4, (1, 1, 3))
    single_estimate = estimate_peak_memory(single_model, 4, (1, 1, 3))

    assert shared_estimate.parameter_count == 12
    assert shared_estimate.parameters == single_estimate.parameters == 48
    assert shared_estimate.activations_saved_for_backward == 72 * 4
    assert single_estimate.activations_saved_for_backward == 48 * 4


def test_biasless_conv_reduces_parameter_count():
    with_bias = Sequential(
        Conv2d(1, 2, 3, bias=True),
        Flatten(),
        Linear(18, 3),
    )
    without_bias = Sequential(
        Conv2d(1, 2, 3, bias=False),
        Flatten(),
        Linear(18, 3),
    )

    biased = estimate_peak_memory(with_bias, 2, (1, 5, 5))
    unbiased = estimate_peak_memory(without_bias, 2, (1, 5, 5))

    assert biased.parameter_count - unbiased.parameter_count == 2


def test_flatten_negative_dimension_is_shape_analyzed():
    model = Sequential(Flatten(start_dim=-3), Linear(18, 3))

    assert estimate_peak_memory(model, 4, (1, 2, 9)).peak_bytes > 0


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "4"])
def test_estimator_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        estimate_peak_memory(build_tiny_model(), batch_size, (1, 5, 5))


@pytest.mark.parametrize(
    "input_shape",
    [(1, 5), (1, 5, 5, 1), (1, 0, 5), (True, 5, 5)],
)
def test_estimator_rejects_invalid_input_shape(input_shape):
    with pytest.raises(ValueError, match="input_shape"):
        estimate_peak_memory(build_tiny_model(), 4, input_shape)


def test_estimator_reports_shape_mismatch_with_module_path():
    model = Sequential(
        Conv2d(1, 2, 3),
        ReLU(),
        Flatten(),
        Linear(17, 3),
    )

    with pytest.raises(ValueError, match=r"model\.3 Linear expected"):
        estimate_peak_memory(model, 4, (1, 5, 5))


def test_estimator_rejects_unsupported_module():
    class Unsupported(Module):
        def __init__(self):
            self.weight = Tensor([1.0], requires_grad=True)

        def parameters(self):
            return [self.weight]

    with pytest.raises(TypeError, match="Unsupported"):
        estimate_peak_memory(Unsupported(), 4, (1, 5, 5))
