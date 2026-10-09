import numpy as np

from .memory import (
    allocate_empty,
    allocate_zeros,
    create_array,
    temporary_result,
)
from .tensor import Tensor, _accumulate_grad


def relu(tensor):
    """Apply ReLU and use derivative zero at an input of exactly zero."""
    if not isinstance(tensor, Tensor):
        raise TypeError("relu input must be a Tensor")

    with temporary_result(
        tensor.data.shape,
        np.float32,
        "relu.forward_result",
        lambda: np.maximum(0, tensor.data),
    ) as result_data:
        out = Tensor(
            result_data,
            requires_grad=tensor.requires_grad,
            _source="relu.output",
        )
    
    def _backward():
        # The closure retains the input Tensor and recreates the boolean mask
        # during backward; forward does not save a separate mask allocation.
        with temporary_result(
            tensor.data.shape,
            np.float32,
            "relu.backward_contribution",
            lambda: (tensor.data > 0) * out.grad,
        ) as contribution:
            _accumulate_grad(tensor, contribution)
    
    out._backward = _backward
    out._prev = (tensor,)
    return out


def sigmoid(tensor):
    """Apply the elementwise logistic sigmoid."""
    if not isinstance(tensor, Tensor):
        raise TypeError("sigmoid input must be a Tensor")

    with temporary_result(
        tensor.data.shape,
        np.float32,
        "sigmoid.forward_result",
        lambda: 1 / (1 + np.exp(-tensor.data)),
    ) as result_data:
        out = Tensor(
            result_data,
            requires_grad=tensor.requires_grad,
            _source="sigmoid.output",
        )
    
    def _backward():
        with temporary_result(
            tensor.data.shape,
            np.float32,
            "sigmoid.backward_contribution",
            lambda: out.data * (1 - out.data) * out.grad,
        ) as contribution:
            _accumulate_grad(tensor, contribution)
    
    out._backward = _backward
    out._prev = (tensor,)
    return out


def matmul(tensor1, tensor2):
    """Multiply two rank-2 Tensors."""
    if not isinstance(tensor1, Tensor) or not isinstance(tensor2, Tensor):
        raise TypeError("matmul operands must be Tensors")
    if tensor1.data.ndim != 2 or tensor2.data.ndim != 2:
        raise ValueError("matmul currently supports only rank-2 Tensors")
    if tensor1.data.shape[1] != tensor2.data.shape[0]:
        raise ValueError(
            "matmul inner dimensions must match: "
            f"got {tensor1.data.shape} and {tensor2.data.shape}"
        )

    output_shape = (tensor1.data.shape[0], tensor2.data.shape[1])
    with temporary_result(
        output_shape,
        np.float32,
        "matmul.forward_result",
        lambda: tensor1.data @ tensor2.data,
    ) as result_data:
        out = Tensor(
            result_data,
            requires_grad=tensor1.requires_grad or tensor2.requires_grad,
            _source="matmul.output",
        )
    
    def _backward():
        with temporary_result(
            tensor1.data.shape,
            np.float32,
            "matmul.backward_left",
            lambda: out.grad @ tensor2.data.T,
        ) as first_contribution:
            _accumulate_grad(tensor1, first_contribution)
        with temporary_result(
            tensor2.data.shape,
            np.float32,
            "matmul.backward_right",
            lambda: tensor1.data.T @ out.grad,
        ) as second_contribution:
            _accumulate_grad(tensor2, second_contribution)
    
    out._backward = _backward
    out._prev = (tensor1, tensor2)
    return out


def reshape(tensor, shape):
    """Reshape a Tensor while preserving its autograd connection.

    Tensor construction currently owns/copies operation output storage, so the
    returned Tensor owns its reshaped data even when NumPy could create a view.
    """
    if not isinstance(tensor, Tensor):
        raise TypeError("reshape input must be a Tensor")

    try:
        reshaped_data = np.reshape(tensor.data, shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cannot reshape tensor with shape {tensor.data.shape} to {shape}"
        ) from exc

    original_shape = tensor.data.shape
    out = Tensor(
        reshaped_data,
        requires_grad=tensor.requires_grad,
        _source="reshape.output",
    )

    def _backward():
        _accumulate_grad(tensor, out.grad.reshape(original_shape))

    out._backward = _backward
    out._prev = (tensor,)
    return out


def flatten(tensor, start_dim=1):
    """Flatten dimensions from ``start_dim`` through the final dimension."""
    if not isinstance(tensor, Tensor):
        raise TypeError("flatten input must be a Tensor")
    if not isinstance(start_dim, int) or isinstance(start_dim, bool):
        raise TypeError("start_dim must be an integer")

    ndim = tensor.data.ndim
    normalized_dim = start_dim + ndim if start_dim < 0 else start_dim
    if ndim == 0 or normalized_dim < 0 or normalized_dim >= ndim:
        raise ValueError(
            f"start_dim {start_dim} is invalid for a rank-{ndim} tensor"
        )

    flattened_size = int(np.prod(tensor.data.shape[normalized_dim:], dtype=np.int64))
    output_shape = tensor.data.shape[:normalized_dim] + (flattened_size,)
    return reshape(tensor, output_shape)


def conv2d(x, weight, bias=None):
    """Apply a stride-1, no-padding 2D convolution to BCHW input.

    Args:
        x: Input Tensor with shape ``(batch, in_channels, height, width)``.
        weight: Filter Tensor with shape
            ``(out_channels, in_channels, kernel, kernel)``.
        bias: Optional Tensor with shape ``(out_channels,)``.
    """
    if not isinstance(x, Tensor) or not isinstance(weight, Tensor):
        raise TypeError("conv2d input and weight must be Tensors")
    if bias is not None and not isinstance(bias, Tensor):
        raise TypeError("conv2d bias must be a Tensor or None")
    if x.data.ndim != 4:
        raise ValueError(f"conv2d input must be BCHW rank 4, got {x.data.shape}")
    if weight.data.ndim != 4:
        raise ValueError(
            f"conv2d weight must be OIHW rank 4, got {weight.data.shape}"
        )

    batch_size, in_channels, height, width = x.data.shape
    out_channels, weight_channels, kernel_height, kernel_width = weight.data.shape
    if height != width:
        raise ValueError("conv2d currently requires square input")
    if kernel_height != kernel_width:
        raise ValueError("conv2d currently requires a square kernel")
    if in_channels != weight_channels:
        raise ValueError(
            "conv2d input and weight channel counts must match: "
            f"got {in_channels} and {weight_channels}"
        )
    if kernel_height <= 0 or kernel_height > height:
        raise ValueError(
            f"kernel size {kernel_height} is invalid for input side {height}"
        )
    if bias is not None and bias.data.shape != (out_channels,):
        raise ValueError(
            f"conv2d bias must have shape {(out_channels,)}, got {bias.data.shape}"
        )

    output_side = height - kernel_height + 1
    output_shape = (batch_size, out_channels, output_side, output_side)
    output_data, output_handle = allocate_empty(
        output_shape,
        np.float32,
        "transient_other",
        "conv2d.forward_output_workspace",
    )

    try:
        # Allocate and account for one small contraction result, then reuse it
        # at every output position. This has the same peak liveness as one
        # temporary per pixel without thousands of tracker events.
        patch_result, patch_result_handle = allocate_empty(
            (batch_size, out_channels),
            np.float32,
            "transient_other",
            "conv2d.forward_einsum_workspace",
        )
        try:
            # Each patch is a view into the input. This avoids materializing a
            # full im2col array whose size scales with every output position.
            for output_y in range(output_side):
                for output_x in range(output_side):
                    patch = x.data[
                        :,
                        :,
                        output_y : output_y + kernel_height,
                        output_x : output_x + kernel_width,
                    ]
                    np.einsum(
                        "bcij,ocij->bo",
                        patch,
                        weight.data,
                        out=patch_result,
                        optimize=False,
                    )
                    output_data[:, :, output_y, output_x] = patch_result
        finally:
            patch_result_handle.release()

        if bias is not None:
            output_data += bias.data[None, :, None, None]

        requires_grad = x.requires_grad or weight.requires_grad or (
            bias is not None and bias.requires_grad
        )
        out = Tensor(
            output_data,
            requires_grad=requires_grad,
            _source="conv2d.output",
        )
    finally:
        output_handle.release()

    def _backward():
        workspace_handles = []
        input_gradient = None
        weight_gradient = None
        try:
            if x.requires_grad:
                input_gradient, input_handle = allocate_zeros(
                    x.data.shape,
                    np.float32,
                    "transient_other",
                    "conv2d.backward_input",
                )
                workspace_handles.append(input_handle)
            if weight.requires_grad:
                weight_gradient, weight_handle = allocate_zeros(
                    weight.data.shape,
                    np.float32,
                    "transient_other",
                    "conv2d.backward_weight",
                )
                workspace_handles.append(weight_handle)

            # A single flat scratch allocation is large enough for any one
            # contraction. Input, weight, and bias views reuse this storage;
            # their contractions execute sequentially and never coexist.
            input_contribution_elements = (
                batch_size * in_channels * kernel_height * kernel_width
                if input_gradient is not None
                else 0
            )
            weight_contribution_elements = (
                weight.data.size if weight_gradient is not None else 0
            )
            bias_contribution_elements = (
                out_channels
                if bias is not None and bias.requires_grad
                else 0
            )
            scratch_elements = max(
                input_contribution_elements,
                weight_contribution_elements,
                bias_contribution_elements,
            )
            if scratch_elements:
                scratch, scratch_handle = allocate_empty(
                    (scratch_elements,),
                    np.float32,
                    "transient_other",
                    "conv2d.backward_einsum_workspace",
                )
                workspace_handles.append(scratch_handle)
            else:
                scratch = None

            input_contribution = (
                scratch[:input_contribution_elements].reshape(
                    batch_size,
                    in_channels,
                    kernel_height,
                    kernel_width,
                )
                if input_contribution_elements
                else None
            )
            weight_contribution = (
                scratch[:weight_contribution_elements].reshape(weight.data.shape)
                if weight_contribution_elements
                else None
            )
            bias_contribution = (
                scratch[:bias_contribution_elements]
                if bias_contribution_elements
                else None
            )

            for output_y in range(output_side):
                for output_x in range(output_side):
                    output_gradient = out.grad[:, :, output_y, output_x]
                    patch = x.data[
                        :,
                        :,
                        output_y : output_y + kernel_height,
                        output_x : output_x + kernel_width,
                    ]

                    if input_gradient is not None:
                        np.einsum(
                            "bo,ocij->bcij",
                            output_gradient,
                            weight.data,
                            out=input_contribution,
                            optimize=False,
                        )
                        input_gradient[
                            :,
                            :,
                            output_y : output_y + kernel_height,
                            output_x : output_x + kernel_width,
                        ] += input_contribution

                    if weight_gradient is not None:
                        np.einsum(
                            "bo,bcij->ocij",
                            output_gradient,
                            patch,
                            out=weight_contribution,
                            optimize=False,
                        )
                        weight_gradient += weight_contribution

            if input_gradient is not None:
                _accumulate_grad(x, input_gradient)
            if weight_gradient is not None:
                _accumulate_grad(weight, weight_gradient)
            if bias is not None and bias.requires_grad:
                out.grad.sum(
                    axis=(0, 2, 3),
                    dtype=np.float32,
                    out=bias_contribution,
                )
                _accumulate_grad(bias, bias_contribution)
        finally:
            for handle in reversed(workspace_handles):
                handle.release()

    out._backward = _backward
    parents = (x, weight) if bias is None else (x, weight, bias)
    out._prev = parents
    return out


def cross_entropy(logits, targets):
    """Return stable mean softmax cross-entropy for class-index targets."""
    if not isinstance(logits, Tensor):
        raise TypeError("cross_entropy logits must be a Tensor")
    if logits.data.ndim != 2:
        raise ValueError(
            f"cross_entropy logits must have shape (batch, classes), got {logits.data.shape}"
        )

    batch_size, class_count = logits.data.shape
    if batch_size == 0:
        raise ValueError("cross_entropy does not support an empty batch")
    if class_count == 0:
        raise ValueError("cross_entropy requires at least one class")

    target_view = np.asarray(targets)
    if target_view.shape != (batch_size,):
        raise ValueError(
            f"targets must have shape {(batch_size,)}, got {target_view.shape}"
        )
    if not np.issubdtype(target_view.dtype, np.integer):
        raise TypeError("cross_entropy targets must contain integer class indices")

    temporary_handles = []
    saved_handles = []
    try:
        target_data, target_handle = create_array(
            (batch_size,),
            np.int64,
            "transient_other",
            "cross_entropy.targets",
            lambda: target_view.astype(np.int64, copy=True),
        )
        saved_handles.append(target_handle)
        if np.any(target_data < 0) or np.any(target_data >= class_count):
            raise ValueError(
                f"target class indices must be in [0, {class_count})"
            )
        row_maxima, handle = create_array(
            (batch_size, 1),
            np.float32,
            "transient_other",
            "cross_entropy.row_maxima",
            lambda: logits.data.max(axis=1, keepdims=True),
        )
        temporary_handles.append(handle)
        shifted, handle = create_array(
            logits.data.shape,
            np.float32,
            "transient_other",
            "cross_entropy.shifted",
            lambda: logits.data - row_maxima,
        )
        temporary_handles.append(handle)
        exponentials, handle = create_array(
            logits.data.shape,
            np.float32,
            "transient_other",
            "cross_entropy.exponentials",
            lambda: np.exp(shifted),
        )
        temporary_handles.append(handle)
        normalizers, handle = create_array(
            (batch_size, 1),
            np.float32,
            "transient_other",
            "cross_entropy.normalizers",
            lambda: exponentials.sum(axis=1, keepdims=True, dtype=np.float32),
        )
        temporary_handles.append(handle)
        probabilities, handle = create_array(
            logits.data.shape,
            np.float32,
            "activations_saved_for_backward",
            "cross_entropy.probabilities",
            lambda: exponentials / normalizers,
        )
        saved_handles.append(handle)
        log_normalizers, handle = create_array(
            (batch_size,),
            np.float32,
            "transient_other",
            "cross_entropy.log_normalizers",
            lambda: np.log(normalizers[:, 0]),
        )
        temporary_handles.append(handle)
        batch_indices, handle = create_array(
            (batch_size,),
            np.int64,
            "transient_other",
            "cross_entropy.batch_indices",
            lambda: np.arange(batch_size, dtype=np.int64),
        )
        saved_handles.append(handle)
        per_example, handle = create_array(
            (batch_size,),
            np.float32,
            "transient_other",
            "cross_entropy.per_example",
            lambda: log_normalizers - shifted[batch_indices, target_data],
        )
        temporary_handles.append(handle)
        with temporary_result(
            (),
            np.float32,
            "cross_entropy.loss_scalar",
            lambda: np.asarray(per_example.mean(dtype=np.float32)),
        ) as loss_data:
            out = Tensor(
                loss_data,
                requires_grad=logits.requires_grad,
                _source="cross_entropy.output",
            )
        for handle in saved_handles:
            out._save_handle(handle)
        saved_handles.clear()
    finally:
        for handle in reversed(temporary_handles):
            handle.release()
        for handle in reversed(saved_handles):
            handle.release()

    def _backward():
        if not logits.requires_grad:
            return
        with temporary_result(
            logits.data.shape,
            np.float32,
            "cross_entropy.backward_logits",
            lambda: probabilities.copy(),
        ) as logits_gradient:
            logits_gradient[batch_indices, target_data] -= np.float32(1.0)
            logits_gradient *= out.grad / np.float32(batch_size)
            _accumulate_grad(logits, logits_gradient)

    out._backward = _backward
    out._prev = (logits,)
    return out
