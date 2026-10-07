import numpy as np
import pytest
import torch
from torch.nn import functional as F 

from simplegrad.tensor import Tensor
from simplegrad.ops import (
    conv2d,
    conv2d_im2col,
    conv2d_strided,
    cross_entropy,
    flatten,
    matmul,
    relu,
    reshape,
    sigmoid,
)


## ReLU only selects the input or exact zero, and backward multiplies by an exact 0/1 mask; NumPy's default tolerance should pass.
def test_relu():
    pt_x = torch.rand((2, 2), requires_grad=True)
    x = Tensor.from_torch(pt_x)

    expected = F.relu(pt_x)
    result = relu(x)
    np.testing.assert_allclose(result.data, expected.detach().numpy())

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(x.grad.data, pt_x.grad.detach().numpy())


## Sigmoid uses exp and division, whose float32 kernels and rounding can differ between NumPy and PyTorch; use the PyTorch default float32 test tolerance: rtol=1.3e-6, atol=1e-5.
def test_sigmoid():
    pt_x = torch.rand((2, 2), requires_grad=True)
    x = Tensor.from_torch(pt_x)

    expected = F.sigmoid(pt_x)
    result = sigmoid(x)
    np.testing.assert_allclose(result.data, expected.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(x.grad.data, pt_x.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)


## Matmul accumulates products in implementation-dependent orders; use the PyTorch default float32 test tolerance: rtol=1.3e-6, atol=1e-5.
def test_matmul():
    pt_a = torch.rand((3, 4), requires_grad=True)
    pt_b = torch.rand((4, 5), requires_grad=True)
    a = Tensor.from_torch(pt_a)
    b = Tensor.from_torch(pt_b)

    expected = pt_a @ pt_b
    result = matmul(a, b)
    np.testing.assert_allclose(result.data, expected.detach().numpy(), 
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(a.grad.data, pt_a.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(b.grad.data, pt_b.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)


## These basic differentiable ops use PyTorch's default float32 test tolerance, rtol=1.3e-6 and atol=1e-5, for forward and backward comparisons.
def test_elemwise_add():
    pt_a = torch.rand((4, 4), requires_grad=True)
    pt_b = torch.rand((4, 4), requires_grad=True)
    a = Tensor.from_torch(pt_a)
    b = Tensor.from_torch(pt_b)

    expected = pt_a + pt_b
    result = a + b
    np.testing.assert_allclose(result.data, expected.detach().numpy(), 
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(a.grad.data, pt_a.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(b.grad.data, pt_b.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)


def test_elemwise_mul():
    pt_a = torch.rand((4, 4), requires_grad=True)
    pt_b = torch.rand((4, 4), requires_grad=True)
    a = Tensor.from_torch(pt_a)
    b = Tensor.from_torch(pt_b)

    expected = pt_a * pt_b
    result = a * b
    np.testing.assert_allclose(result.data, expected.detach().numpy(), 
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(a.grad.data, pt_a.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(b.grad.data, pt_b.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)


## Scalar multiplication performs the same elementwise float32 operation in
## both libraries; NumPy's default tolerance should pass.
@pytest.mark.parametrize("scalar", [-2.5, np.float32(0.5), np.int64(3)])
def test_tensor_scalar_multiplication_matches_torch(scalar):
    """Tensor-scalar multiplication propagates only into the Tensor."""
    data = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    upstream = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    x = Tensor(data, requires_grad=True)
    pt_x = torch.tensor(data, requires_grad=True)

    result = x * scalar
    expected = pt_x * scalar
    result.backward(upstream)
    expected.backward(torch.tensor(upstream))

    np.testing.assert_allclose(result.data, expected.detach().numpy())
    np.testing.assert_allclose(x.grad, pt_x.grad.numpy())


def test_reverse_scalar_multiplication_and_negation():
    """Reverse multiplication and unary negation use scalar multiplication."""
    x = Tensor([1.0, -2.0, 3.0], requires_grad=True)
    upstream = np.array([1.0, 2.0, -0.5], dtype=np.float32)

    result = -x
    result.backward(upstream)

    np.testing.assert_allclose((2 * x).data, 2 * x.data)
    np.testing.assert_allclose(result.data, -x.data)
    np.testing.assert_allclose(x.grad, -upstream)


@pytest.mark.parametrize("value", [True, "2", object()])
def test_scalar_multiplication_rejects_non_real_values(value):
    """Only real numeric scalars and Tensor operands are supported."""
    with pytest.raises(TypeError, match="Tensor or real scalar"):
        Tensor([1.0]) * value


## Forward add and its unreduced gradient should pass NumPy's defaults. The broadcast-bias gradient is a reduction, so it uses rtol=1.3e-6, atol=1e-5.
def test_elemwise_add_backward_reduces_broadcast_bias():
    """A broadcast bias receives one summed gradient per original element."""
    pt_x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    pt_x.requires_grad_()
    pt_bias = torch.tensor([0.5, -1.0, 2.0, 3.0], requires_grad=True)
    upstream = torch.linspace(-1.0, 1.0, 24).reshape(2, 3, 4)
    x = Tensor.from_torch(pt_x)
    bias = Tensor.from_torch(pt_bias)

    expected = pt_x + pt_bias
    result = x + bias
    expected.backward(upstream)
    result.backward(upstream.numpy())

    np.testing.assert_allclose(result.data, expected.detach().numpy())
    np.testing.assert_allclose(x.grad, pt_x.grad.numpy())
    # NumPy and PyTorch reduce float32 values in slightly different orders.
    np.testing.assert_allclose(
        bias.grad,
        pt_bias.grad.numpy(),
        rtol=1.3e-6,
        atol=1e-5,
    )


## Broadcast multiplication is exact before reduction; reduced operand
## gradients allow float32 summation-order differences at rtol=1.3e-6,
## atol=1e-5.
def test_elemwise_mul_backward_reduces_each_broadcast_operand():
    """Each operand reduces its local derivative over its expanded axes."""
    pt_a = torch.tensor(
        [[[1.0, -2.0, 3.0]], [[-4.0, 5.0, 6.0]]],
        requires_grad=True,
    )
    pt_b = torch.tensor(
        [[[0.5], [1.5], [-2.0], [3.0]]],
        requires_grad=True,
    )
    upstream = torch.linspace(-2.0, 2.0, 24).reshape(2, 4, 3)
    a = Tensor.from_torch(pt_a)
    b = Tensor.from_torch(pt_b)

    expected = pt_a * pt_b
    result = a * b
    expected.backward(upstream)
    result.backward(upstream.numpy())

    np.testing.assert_allclose(result.data, expected.detach().numpy())
    np.testing.assert_allclose(a.grad, pt_a.grad.numpy(), rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(b.grad, pt_b.grad.numpy(), rtol=1.3e-6, atol=1e-5)


## These tests inject exact values and exercise selection or layout changes,
## not approximate numerical kernels; NumPy's default tolerance should pass.
def test_relu_uses_zero_derivative_at_zero():
    """ReLU matches PyTorch for negative, zero, and positive inputs."""
    pt_x = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
    upstream = torch.tensor([1.5, -2.0, 0.5])
    x = Tensor.from_torch(pt_x)

    expected = F.relu(pt_x)
    result = relu(x)
    expected.backward(upstream)
    result.backward(upstream.numpy())

    np.testing.assert_allclose(result.data, expected.detach().numpy())
    np.testing.assert_allclose(x.grad, pt_x.grad.numpy())


def test_reshape_forward_and_backward_match_torch():
    """Reshape preserves values and reverses the shape change in backward."""
    pt_x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    pt_x.requires_grad_()
    upstream = torch.linspace(-2.0, 2.0, 24).reshape(2, 12)
    x = Tensor.from_torch(pt_x)

    expected = pt_x.reshape(2, 12)
    result = reshape(x, (2, 12))
    expected.backward(upstream)
    result.backward(upstream.numpy())

    np.testing.assert_allclose(result.data, expected.detach().numpy())
    np.testing.assert_allclose(x.grad, pt_x.grad.numpy())


def test_flatten_forward_and_backward_match_torch():
    """Flatten preserves the batch prefix and its gradient layout."""
    pt_x = torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4)
    pt_x.requires_grad_()
    upstream = torch.linspace(-1.0, 1.0, 48).reshape(2, 24)
    x = Tensor.from_torch(pt_x)

    expected = torch.flatten(pt_x, start_dim=1)
    result = flatten(x, start_dim=1)
    expected.backward(upstream)
    result.backward(upstream.numpy())

    np.testing.assert_allclose(result.data, expected.detach().numpy())
    np.testing.assert_allclose(x.grad, pt_x.grad.numpy())


## Convolution uses different multiply-accumulate kernels in this library and
## PyTorch; rtol=1e-4 and atol=1e-5 allow float32 accumulation-order error.
@pytest.mark.parametrize(
    "operation",
    [conv2d, conv2d_strided, conv2d_im2col],
    ids=["patch", "strided", "im2col"],
)
@pytest.mark.parametrize(
    ("batch_size", "in_channels", "out_channels", "side", "kernel", "use_bias"),
    [
        (1, 2, 3, 5, 1, False),
        (2, 2, 3, 5, 3, True),
    ],
)
def test_conv2d_forward_and_backward_match_torch(
    operation,
    batch_size,
    in_channels,
    out_channels,
    side,
    kernel,
    use_bias,
):
    """Every convolution backend matches each PyTorch trainable input."""
    rng = np.random.default_rng(2026 + kernel)
    x_data = rng.normal(size=(batch_size, in_channels, side, side)).astype(np.float32)
    weight_data = rng.normal(
        size=(out_channels, in_channels, kernel, kernel)
    ).astype(np.float32)
    bias_data = rng.normal(size=(out_channels,)).astype(np.float32)

    pt_x = torch.tensor(x_data, requires_grad=True)
    pt_weight = torch.tensor(weight_data, requires_grad=True)
    pt_bias = torch.tensor(bias_data, requires_grad=True) if use_bias else None
    x = Tensor(x_data, requires_grad=True)
    weight = Tensor(weight_data, requires_grad=True)
    bias = Tensor(bias_data, requires_grad=True) if use_bias else None

    expected = F.conv2d(pt_x, pt_weight, pt_bias, stride=1, padding=0)
    result = operation(x, weight, bias)
    upstream = torch.linspace(
        -1.5,
        2.0,
        expected.numel(),
        dtype=torch.float32,
    ).reshape(expected.shape)
    expected.backward(upstream)
    result.backward(upstream.numpy())

    np.testing.assert_allclose(
        result.data,
        expected.detach().numpy(),
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(x.grad, pt_x.grad.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        weight.grad,
        pt_weight.grad.numpy(),
        rtol=1e-4,
        atol=1e-5,
    )
    if use_bias:
        np.testing.assert_allclose(
            bias.grad,
            pt_bias.grad.numpy(),
            rtol=1e-4,
            atol=1e-5,
        )


def test_conv2d_validates_geometry():
    """Unsupported input geometry fails before numerical work begins."""
    weight = Tensor(np.ones((2, 1, 3, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="square input"):
        conv2d(Tensor(np.ones((1, 1, 5, 4))), weight)
    with pytest.raises(ValueError, match="channel counts"):
        conv2d(Tensor(np.ones((1, 2, 5, 5))), weight)
    with pytest.raises(ValueError, match="bias must have shape"):
        conv2d(Tensor(np.ones((1, 1, 5, 5))), weight, Tensor(np.ones(3)))


## Cross-entropy combines reductions, exp, log, normalization, and mean. NumPy
## and PyTorch may round each stage differently, so use rtol=1e-4, atol=1e-5.
def test_cross_entropy_forward_and_backward_match_torch():
    """Fused mean cross-entropy matches PyTorch with a scalar upstream."""
    logits_data = np.array(
        [
            [1.5, -0.2, 0.3, 2.1],
            [-1.0, 0.4, 2.3, -0.7],
            [0.1, -0.5, 0.8, 1.2],
        ],
        dtype=np.float32,
    )
    targets = np.array([3, 2, 0], dtype=np.int64)
    pt_logits = torch.tensor(logits_data, requires_grad=True)
    logits = Tensor(logits_data, requires_grad=True)

    expected = F.cross_entropy(pt_logits, torch.tensor(targets), reduction="mean")
    result = cross_entropy(logits, targets)
    upstream = np.array(0.75, dtype=np.float32)
    expected.backward(torch.tensor(0.75))
    result.backward(upstream)

    np.testing.assert_allclose(result.data, expected.detach().numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(logits.grad, pt_logits.grad.numpy(), rtol=1e-4, atol=1e-5)


## Extreme logits amplify differences in exp/log paths; retain the cross-entropy
## tolerance rtol=1e-4, atol=1e-5 while also requiring finite results.
def test_cross_entropy_is_stable_for_extreme_logits():
    """Max shifting keeps extreme-logit loss and gradients finite."""
    logits_data = np.array(
        [[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 0.0]],
        dtype=np.float32,
    )
    targets = np.array([0, 2], dtype=np.int64)
    pt_logits = torch.tensor(logits_data, requires_grad=True)
    logits = Tensor(logits_data, requires_grad=True)

    expected = F.cross_entropy(pt_logits, torch.tensor(targets))
    result = cross_entropy(logits, targets)
    expected.backward()
    result.backward()

    assert np.isfinite(result.data)
    assert np.all(np.isfinite(logits.grad))
    np.testing.assert_allclose(result.data, expected.detach().numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(logits.grad, pt_logits.grad.numpy(), rtol=1e-4, atol=1e-5)


## Repeated and distinct targets use the same reduction-heavy cross-entropy
## path, so their forward and backward checks use rtol=1e-4, atol=1e-5.
@pytest.mark.parametrize(
    "targets",
    [
        np.array([0, 1, 2, 3], dtype=np.int64),
        np.array([1, 1, 1, 2], dtype=np.int64),
    ],
)
def test_cross_entropy_handles_all_classes_and_repeated_targets(targets):
    """Indexed loss gradients work for distinct and repeated class labels."""
    logits_data = np.linspace(-1.5, 2.0, 16, dtype=np.float32).reshape(4, 4)
    pt_logits = torch.tensor(logits_data, requires_grad=True)
    logits = Tensor(logits_data, requires_grad=True)

    expected = F.cross_entropy(pt_logits, torch.tensor(targets))
    result = cross_entropy(logits, targets)
    expected.backward()
    result.backward()

    np.testing.assert_allclose(result.data, expected.detach().numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(logits.grad, pt_logits.grad.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("targets", "error", "message"),
    [
        (np.array([[0, 1]]), ValueError, "targets must have shape"),
        (np.array([0.0, 1.0]), TypeError, "integer class indices"),
        (np.array([0, 3]), ValueError, "must be in"),
        (np.array([0, -1]), ValueError, "must be in"),
    ],
)
def test_cross_entropy_rejects_invalid_targets(targets, error, message):
    """Class targets must have valid integer indices and batch shape."""
    logits = Tensor(np.ones((2, 3), dtype=np.float32), requires_grad=True)

    with pytest.raises(error, match=message):
        cross_entropy(logits, targets)
