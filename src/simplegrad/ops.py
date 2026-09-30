import numpy as np

from .tensor import Tensor, _accumulate_grad


def relu(tensor):
    """Apply ReLU and use derivative zero at an input of exactly zero."""
    if not isinstance(tensor, Tensor):
        raise TypeError("relu input must be a Tensor")

    out = Tensor(np.maximum(0, tensor.data), requires_grad=tensor.requires_grad)
    
    def _backward():
        # The closure retains the input Tensor and recreates the boolean mask
        # during backward; forward does not save a separate mask allocation.
        _accumulate_grad(tensor, (tensor.data > 0) * out.grad)
    
    out._backward = _backward
    out._prev = {tensor, }
    return out


def sigmoid(tensor):
    """Apply the elementwise logistic sigmoid."""
    if not isinstance(tensor, Tensor):
        raise TypeError("sigmoid input must be a Tensor")

    out = Tensor(1 / (1 + np.exp(-tensor.data)), requires_grad=tensor.requires_grad)
    
    def _backward():
        _accumulate_grad(tensor, out.data * (1 - out.data) * out.grad)
    
    out._backward = _backward
    out._prev = {tensor, }
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

    out = Tensor(tensor1.data @ tensor2.data, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    def _backward():
        _accumulate_grad(tensor1, out.grad @ tensor2.data.T)
        _accumulate_grad(tensor2, tensor1.data.T @ out.grad)
    
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
    out = Tensor(reshaped_data, requires_grad=tensor.requires_grad)

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
    output_data = np.empty(
        (batch_size, out_channels, output_side, output_side),
        dtype=np.float32,
    )

    # Each patch is a view into the input. This avoids materializing a full
    # im2col array whose size scales with every output position.
    for output_y in range(output_side):
        for output_x in range(output_side):
            patch = x.data[
                :,
                :,
                output_y : output_y + kernel_height,
                output_x : output_x + kernel_width,
            ]
            output_data[:, :, output_y, output_x] = np.einsum(
                "bcij,ocij->bo",
                patch,
                weight.data,
                optimize=False,
            )

    if bias is not None:
        output_data += bias.data[None, :, None, None]

    requires_grad = x.requires_grad or weight.requires_grad or (
        bias is not None and bias.requires_grad
    )
    out = Tensor(output_data, requires_grad=requires_grad)

    def _backward():
        input_gradient = np.zeros_like(x.data) if x.requires_grad else None
        weight_gradient = (
            np.zeros_like(weight.data) if weight.requires_grad else None
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
                    input_gradient[
                        :,
                        :,
                        output_y : output_y + kernel_height,
                        output_x : output_x + kernel_width,
                    ] += np.einsum(
                        "bo,ocij->bcij",
                        output_gradient,
                        weight.data,
                        optimize=False,
                    )

                if weight_gradient is not None:
                    weight_gradient += np.einsum(
                        "bo,bcij->ocij",
                        output_gradient,
                        patch,
                        optimize=False,
                    )

        if input_gradient is not None:
            _accumulate_grad(x, input_gradient)
        if weight_gradient is not None:
            _accumulate_grad(weight, weight_gradient)
        if bias is not None and bias.requires_grad:
            bias_gradient = out.grad.sum(axis=(0, 2, 3), dtype=np.float32)
            _accumulate_grad(bias, bias_gradient)

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

    target_data = np.asarray(targets)
    if target_data.shape != (batch_size,):
        raise ValueError(
            f"targets must have shape {(batch_size,)}, got {target_data.shape}"
        )
    if not np.issubdtype(target_data.dtype, np.integer):
        raise TypeError("cross_entropy targets must contain integer class indices")
    target_data = target_data.astype(np.int64, copy=True)
    if np.any(target_data < 0) or np.any(target_data >= class_count):
        raise ValueError(
            f"target class indices must be in [0, {class_count})"
        )

    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    normalizers = exponentials.sum(axis=1, keepdims=True, dtype=np.float32)
    probabilities = exponentials / normalizers
    log_normalizers = np.log(normalizers[:, 0])
    batch_indices = np.arange(batch_size)
    per_example = log_normalizers - shifted[batch_indices, target_data]
    loss_data = per_example.mean(dtype=np.float32)
    out = Tensor(loss_data, requires_grad=logits.requires_grad)

    def _backward():
        if not logits.requires_grad:
            return
        logits_gradient = probabilities.copy()
        logits_gradient[batch_indices, target_data] -= np.float32(1.0)
        logits_gradient *= out.grad / np.float32(batch_size)
        _accumulate_grad(logits, logits_gradient)

    out._backward = _backward
    out._prev = (logits,)
    return out
