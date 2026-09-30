import numpy as np
import torch
from torch.nn import functional as F 

from simplegrad.tensor import Tensor
from simplegrad.ops import relu, sigmoid, matmul


def test_relu():
    pt_x = torch.rand((2, 2), requires_grad=True)
    x = Tensor.from_torch(pt_x)

    expected = F.relu(pt_x)
    result = relu(x)
    np.testing.assert_allclose(result.data, expected.detach().numpy())

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(x.grad.data, pt_x.grad.detach().numpy())


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
