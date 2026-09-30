import numpy as np
import torch
from torch.nn import functional as F 

from simplegrad.tensor import Tensor
from simplegrad.ops import relu, sigmoid, matmul


def test_autograd():
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    b = Tensor(np.random.randn(3, 3), requires_grad=True)
    c = Tensor(np.random.randn(3, 3), requires_grad=True)
    d = Tensor(np.random.randn(3, 3), requires_grad=True)

    b_T = b.transpose()
    z1 = matmul(a, b_T)
    z2 = z1 + c
    z3 = z2 * d
    z4 = relu(z3)
    res = sigmoid(z4)

    pt_a, pt_b, pt_c, pt_d = a.to_torch(), b.to_torch(), c.to_torch(), d.to_torch()
    pt_b_T = pt_b.T
    pt_z1 = pt_a @ pt_b_T
    pt_z2 = pt_z1 + pt_c
    pt_z3 = pt_z2 * pt_d
    pt_z4 = F.relu(pt_z3)
    expected = F.sigmoid(pt_z4)

    np.testing.assert_allclose(res.data, expected.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)

    pt_z1.retain_grad()
    pt_z2.retain_grad()
    pt_z3.retain_grad()
    pt_z4.retain_grad()

    res.backward()
    expected.backward(torch.ones_like(expected))

    np.testing.assert_allclose(z4.grad, pt_z4.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(z3.grad, pt_z3.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(z2.grad, pt_z2.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(z1.grad, pt_z1.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)


def test_backward_accumulates_leaf_gradients_without_repropagating_old_gradients():
    """Repeated traversals add leaf grads but do not reuse stale intermediates."""
    x = Tensor([1.0, -2.0, 3.0], requires_grad=True)
    result = x * x
    upstream = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result.backward(upstream)
    first_gradient = x.grad.copy()
    result.backward(upstream)

    np.testing.assert_allclose(first_gradient, 2 * x.data * upstream)
    np.testing.assert_allclose(x.grad, 2 * first_gradient)


def test_backward_accumulates_contributions_from_branches():
    """Every path from a branched graph contributes to the shared leaf."""
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    squared = x * x
    result = squared + x
    upstream = np.array([3.0, 2.0, 1.0], dtype=np.float32)

    result.backward(upstream)

    np.testing.assert_allclose(x.grad, (2 * x.data + 1) * upstream)
