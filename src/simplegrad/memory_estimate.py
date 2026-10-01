"""Simple, pure shape-based memory estimation.

The estimator intentionally uses a compact analytical model:

    parameters + parameter gradients + Adam state + saved activations
    + largest modeled transient workspace

It does not execute the model or inspect runtime allocations.
"""

from dataclasses import dataclass
from math import prod
from numbers import Integral

import numpy as np

from .module import Conv2d, Flatten, Linear, Module, ReLU, Sequential
from .memory import MemoryBreakdown
from .tensor import Tensor


_FLOAT32_BYTES = np.dtype(np.float32).itemsize
_BYTE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB", "EB")
_ASSUMPTIONS = (
    "float32 parameter, gradient, optimizer-state, and activation arrays",
    "one gradient buffer per trainable parameter",
    "two Adam moment buffers per trainable parameter",
    "saved activations include the input, each leaf module output, and loss probabilities",
    "intermediate Tensor gradient buffers are excluded",
    "transient/other includes retained target indices plus the largest modeled workspace",
    "opaque NumPy, BLAS, and einsum internal workspace is excluded",
)


def _non_negative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _compact_decimal(value, decimals=1):
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def format_bytes(byte_count):
    """Format bytes with a decimal unit and retain the exact byte count.

    Decimal units are used: one KB is 1,000 bytes and one MB is 1,000,000
    bytes. The exact integer remains visible for reproducible memory reports.
    """
    byte_count = _non_negative_integer(byte_count, "byte_count")
    scaled = float(byte_count)
    unit_index = 0
    while scaled >= 1000 and unit_index < len(_BYTE_UNITS) - 1:
        scaled /= 1000
        unit_index += 1
    readable = _compact_decimal(scaled)
    return f"{readable} {_BYTE_UNITS[unit_index]} ({byte_count:,} bytes)"


def format_parameter_count(parameter_count):
    """Format a parameter count in millions and retain the exact count."""
    parameter_count = _non_negative_integer(parameter_count, "parameter_count")
    millions = parameter_count / 1_000_000
    if millions >= 1:
        decimals = 1
    elif millions >= 0.01:
        decimals = 2
    else:
        decimals = 4
    readable = _compact_decimal(millions, decimals=decimals)
    return f"{readable}M ({parameter_count:,})"


def format_memory_estimate(estimate):
    """Return the concise training-log representation of an estimate."""
    if not isinstance(estimate, MemoryEstimate):
        raise TypeError("estimate must be a MemoryEstimate")
    return (
        f"num params: {format_parameter_count(estimate.parameter_count)}, "
        f"memory: {format_bytes(estimate.peak_bytes)}"
    )


@dataclass(frozen=True)
class MemoryEstimate:
    """Result of the simplified analytical memory model."""

    parameter_count: int
    peak_bytes: int
    breakdown: MemoryBreakdown
    assumptions: tuple[str, ...] = _ASSUMPTIONS

    def __post_init__(self):
        object.__setattr__(
            self,
            "parameter_count",
            _non_negative_integer(self.parameter_count, "parameter_count"),
        )
        object.__setattr__(
            self,
            "peak_bytes",
            _non_negative_integer(self.peak_bytes, "peak_bytes"),
        )
        if not isinstance(self.breakdown, MemoryBreakdown):
            raise TypeError("breakdown must be a MemoryBreakdown")
        if self.peak_bytes != self.breakdown.total_bytes:
            raise ValueError("peak_bytes must equal the breakdown total")
        try:
            assumptions = tuple(self.assumptions)
        except TypeError as exc:
            raise TypeError("assumptions must be an iterable of strings") from exc
        if not all(isinstance(item, str) and item for item in assumptions):
            raise ValueError("assumptions must contain non-empty strings")
        object.__setattr__(self, "assumptions", assumptions)

    @property
    def at_peak(self):
        """Compatibility alias for the category breakdown."""
        return self.breakdown

    @property
    def parameters(self):
        return self.breakdown.parameters

    @property
    def gradients(self):
        return self.breakdown.gradients

    @property
    def optimizer_state(self):
        return self.breakdown.optimizer_state

    @property
    def activations_saved_for_backward(self):
        return self.breakdown.activations_saved_for_backward

    @property
    def transient_other(self):
        return self.breakdown.transient_other

    def to_dict(self):
        return {
            "parameter_count": self.parameter_count,
            "peak_bytes": self.peak_bytes,
            "breakdown": self.breakdown.to_dict(),
            "assumptions": list(self.assumptions),
        }


def _shape_tuple(shape, name, expected_rank=None):
    try:
        result = tuple(shape)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of positive integers") from exc
    if expected_rank is not None and len(result) != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}, got {result}")
    if any(
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, Integral)
        or size <= 0
        for size in result
    ):
        raise ValueError(f"{name} must contain only positive integers")
    return tuple(int(size) for size in result)


def _numel(shape):
    return int(prod(shape))


def _trainable_parameter_stats(model):
    count = 0
    largest = 0
    seen = set()
    for parameter in model.parameters():
        if not isinstance(parameter, Tensor):
            raise TypeError("model parameters must be Tensors")
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        if not parameter.requires_grad:
            continue
        if parameter.data.dtype != np.float32:
            raise ValueError("the estimator supports only float32 parameters")
        parameter_count = _numel(parameter.data.shape)
        count += parameter_count
        largest = max(largest, parameter_count)
    if count == 0:
        raise ValueError("estimate_peak_memory requires trainable parameters")
    return count, largest


def _flatten_shape(shape, start_dim, path):
    if isinstance(start_dim, bool) or not isinstance(start_dim, Integral):
        raise TypeError(f"{path} Flatten start_dim must be an integer")
    rank = len(shape)
    normalized = int(start_dim) + rank if start_dim < 0 else int(start_dim)
    if normalized < 0 or normalized >= rank:
        raise ValueError(
            f"{path} Flatten start_dim {start_dim} is invalid for rank {rank}"
        )
    return shape[:normalized] + (_numel(shape[normalized:]),)


def _analyze_module(module, input_shape, input_requires_grad, path):
    """Return output shape, saved elements, workspace bytes, and grad flag."""
    if isinstance(module, Sequential):
        if not module.modules:
            raise ValueError(f"{path} Sequential must contain at least one module")
        shape = input_shape
        saved_elements = 0
        largest_workspace = 0
        requires_grad = input_requires_grad
        for index, child in enumerate(module.modules):
            shape, child_saved, child_workspace, requires_grad = _analyze_module(
                child,
                shape,
                requires_grad,
                f"{path}.{index}",
            )
            saved_elements += child_saved
            largest_workspace = max(largest_workspace, child_workspace)
        return shape, saved_elements, largest_workspace, requires_grad

    if isinstance(module, Conv2d):
        if len(input_shape) != 4:
            raise ValueError(
                f"{path} Conv2d expected rank-4 BCHW input, got {input_shape}"
            )
        batch, channels, height, width = input_shape
        if channels != module.in_channels:
            raise ValueError(
                f"{path} Conv2d expected {module.in_channels} input channels, "
                f"got {channels}"
            )
        if height != width:
            raise ValueError(f"{path} Conv2d requires square input, got {input_shape}")
        if module.kernel_size > height:
            raise ValueError(
                f"{path} Conv2d kernel {module.kernel_size} exceeds input side {height}"
            )
        output_side = height - module.kernel_size + 1
        output_shape = (
            batch,
            module.out_channels,
            output_side,
            output_side,
        )
        weight_shape = (
            module.out_channels,
            module.in_channels,
            module.kernel_size,
            module.kernel_size,
        )
        weight_bytes = _numel(weight_shape) * _FLOAT32_BYTES
        input_gradient_bytes = (
            _numel(input_shape) * _FLOAT32_BYTES
            if input_requires_grad
            else 0
        )
        input_einsum_bytes = (
            input_shape[0]
            * module.in_channels
            * module.kernel_size
            * module.kernel_size
            * _FLOAT32_BYTES
            if input_requires_grad
            else 0
        )
        bias_reduction_bytes = (
            module.out_channels * _FLOAT32_BYTES
            if module.bias is not None
            else 0
        )
        backward_workspace = (
            input_gradient_bytes
            + weight_bytes
            + max(
                input_einsum_bytes,
                weight_bytes,
                bias_reduction_bytes,
            )
        )
        forward_workspace = _numel(output_shape) * _FLOAT32_BYTES
        return (
            output_shape,
            _numel(output_shape),
            max(forward_workspace, backward_workspace),
            True,
        )

    if isinstance(module, ReLU):
        # Backward simultaneously creates a bool mask and float32 product.
        workspace = _numel(input_shape) * (
            np.dtype(np.bool_).itemsize + _FLOAT32_BYTES
        )
        return input_shape, _numel(input_shape), workspace, input_requires_grad

    if isinstance(module, Flatten):
        output_shape = _flatten_shape(input_shape, module.start_dim, path)
        return output_shape, _numel(output_shape), 0, input_requires_grad

    if isinstance(module, Linear):
        if len(input_shape) != 2 or input_shape[1] != module.in_features:
            raise ValueError(
                f"{path} Linear expected shape (batch, {module.in_features}), "
                f"shape propagation produced {input_shape}"
            )
        output_shape = (input_shape[0], module.out_features)
        forward_workspace = _numel(output_shape) * _FLOAT32_BYTES
        input_contribution = _numel(input_shape) * _FLOAT32_BYTES
        weight_contribution = (
            module.in_features * module.out_features * _FLOAT32_BYTES
        )
        backward_workspace = max(input_contribution, weight_contribution)
        return (
            output_shape,
            _numel(output_shape),
            max(forward_workspace, backward_workspace),
            True,
        )

    raise TypeError(
        f"estimate_peak_memory does not support module {type(module).__name__}"
    )


def _validate_arguments(model, batch_size, input_shape):
    if not isinstance(model, Module):
        raise TypeError("model must be a simplegrad Module")
    if (
        isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, Integral)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    return int(batch_size), _shape_tuple(input_shape, "input_shape", expected_rank=3)


def estimate_peak_memory(model, batch_size, input_shape):
    """Estimate peak bytes by summing persistent state and saved activations.

    The function is analytical: it reads module and parameter metadata, performs
    shape propagation, and never calls the model or allocates dummy Tensors.
    """
    batch_size, input_shape = _validate_arguments(
        model,
        batch_size,
        input_shape,
    )
    parameter_count, largest_parameter_count = _trainable_parameter_stats(model)
    parameter_bytes = parameter_count * _FLOAT32_BYTES

    full_input_shape = (batch_size,) + input_shape
    (
        output_shape,
        module_activation_count,
        module_workspace_bytes,
        _,
    ) = _analyze_module(
        model,
        full_input_shape,
        False,
        "model",
    )
    if len(output_shape) != 2:
        raise ValueError(
            "cross_entropy requires rank-2 logits, "
            f"shape propagation produced {output_shape}"
        )

    # The input is retained by the first operation. Each leaf module output is
    # retained by the graph, and fused cross-entropy retains probabilities with
    # the same shape as the logits.
    saved_activation_count = (
        _numel(full_input_shape)
        + module_activation_count
        + _numel(output_shape)
    )
    batch_size, _ = output_shape
    retained_target_and_index_bytes = batch_size * 2 * np.dtype(np.int64).itemsize
    logits_bytes = _numel(output_shape) * _FLOAT32_BYTES
    # Shifted logits and exponentials coexist. Four batch-sized float32 arrays
    # cover normalizers, log-normalizers, indexed logits, and per-example loss.
    loss_workspace_bytes = (
        2 * logits_bytes
        + 4 * batch_size * _FLOAT32_BYTES
        + _FLOAT32_BYTES
    )
    # Adam's corrected moments, denominator, numerator, and update can overlap.
    adam_workspace_bytes = 5 * largest_parameter_count * _FLOAT32_BYTES
    transient_other_bytes = retained_target_and_index_bytes + max(
        module_workspace_bytes,
        loss_workspace_bytes,
        adam_workspace_bytes,
        _FLOAT32_BYTES,
    )
    breakdown = MemoryBreakdown(
        parameters=parameter_bytes,
        gradients=parameter_bytes,
        optimizer_state=2 * parameter_bytes,
        activations_saved_for_backward=(
            saved_activation_count * _FLOAT32_BYTES
        ),
        transient_other=transient_other_bytes,
    )
    return MemoryEstimate(
        parameter_count=parameter_count,
        peak_bytes=breakdown.total_bytes,
        breakdown=breakdown,
        assumptions=_ASSUMPTIONS,
    )
