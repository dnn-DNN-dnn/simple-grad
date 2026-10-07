import numpy as np
import pytest
import torch

from simplegrad.tensor import Tensor, _accumulate_grad, release_graph, sum_to_shape


## These functions copy or cast the same source values; they do not perform numerically different algorithms.
def test_from_torch():
    pt_tensor = torch.rand((2, 2))
    tensor = Tensor.from_torch(pt_tensor)

    assert pt_tensor.requires_grad == tensor.requires_grad
    np.testing.assert_allclose(tensor.data, pt_tensor.numpy())


def test_to_torch():
    tensor = Tensor(np.random.rand(2, 2))
    pt_tensor = tensor.to_torch()

    assert pt_tensor.requires_grad == tensor.requires_grad
    np.testing.assert_allclose(tensor.data, pt_tensor.numpy())


def test_transpose():
    tensor = Tensor(np.random.rand(2, 3, 4))
    transposed = tensor.transpose()

    d11, d12 = tensor.data.shape[-2:]
    d21, d22 = transposed.data.shape[-2:]
    assert d11 == d22 and d21 == d12, f"{d11=}, {d12=}, {d21=}, {d22=}"


## These functions perform the same operations; they must not show different values.
def test_backward_accepts_array_upstream():
    """An ndarray seed computes the requested vector-Jacobian product."""
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    result = x * x
    upstream = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    result.backward(upstream)

    np.testing.assert_allclose(x.grad, 2 * x.data * upstream)


def test_backward_accepts_tensor_upstream():
    """A Tensor may provide upstream values without joining the graph."""
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    result = x * x

    result.backward(Tensor([3.0, 2.0, 1.0]))

    np.testing.assert_allclose(x.grad, [6.0, 8.0, 6.0])


def test_backward_rejects_broadcastable_upstream_shape():
    """Backward rejects implicit broadcasting even when NumPy permits it."""
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    result = x * x

    with pytest.raises(ValueError, match="exactly match"):
        result.backward(np.ones((3,), dtype=np.float32))


def test_backward_rejects_non_numeric_upstream():
    """Backward reports a clear error for a non-numeric gradient seed."""
    x = Tensor([1.0], requires_grad=True)

    with pytest.raises(TypeError, match="numeric array-like"):
        (x * x).backward(["not-a-number"])


def test_backward_rejects_tensor_without_gradients():
    """A tensor outside autograd cannot start a backward traversal."""
    x = Tensor([1.0, 2.0])

    with pytest.raises(RuntimeError, match="does not require gradients"):
        x.backward()


## These tests intentinally inject values to test the algorithm's exactness; assert_allclose should pass with default tolerances.
def test_accumulate_grad_preserves_gradient_buffer():
    """Central accumulation updates values without replacing the buffer."""
    x = Tensor([1.0, 2.0], requires_grad=True)
    original_buffer = x.grad

    _accumulate_grad(x, [3.0, 4.0])
    _accumulate_grad(x, np.array([0.5, 1.5], dtype=np.float64))

    assert x.grad is original_buffer
    assert x.grad.dtype == np.float32
    np.testing.assert_allclose(x.grad, [3.5, 5.5])


def test_accumulate_grad_requires_exact_shape():
    """Central accumulation rejects broadcasting before it hides an op bug."""
    x = Tensor(np.zeros((2, 3)), requires_grad=True)

    with pytest.raises(ValueError, match="exactly match"):
        _accumulate_grad(x, np.ones((3,), dtype=np.float32))


def test_accumulate_grad_ignores_tensor_without_gradients():
    """Backward rules may safely call the helper for constant operands."""
    constant = Tensor([1.0, 2.0])

    _accumulate_grad(constant, [3.0, 4.0])

    assert constant.grad is None

## These tests intentinally inject values to test the algorithm's exactness; assert_allclose should pass with default tolerances.
def test_sum_to_shape_reduces_leading_and_singleton_dimensions():
    """Broadcast dimensions are summed while original dimensions remain."""
    gradient = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    result = sum_to_shape(gradient, (1, 4))

    assert result.shape == (1, 4)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, gradient.sum(axis=(0, 1))[None, :])


def test_sum_to_shape_reduces_to_scalar():
    """A scalar input receives the sum of every broadcast output gradient."""
    gradient = np.arange(6, dtype=np.float32).reshape(2, 3)

    result = sum_to_shape(gradient, ())

    assert result.shape == ()
    np.testing.assert_allclose(result, gradient.sum())


def test_sum_to_shape_rejects_incompatible_shape():
    """Reduction cannot repair dimensions that were not broadcast-compatible."""
    with pytest.raises(ValueError, match="cannot broadcast"):
        sum_to_shape(np.ones((2, 3), dtype=np.float32), (2, 4))


def test_release_graph_removes_links_but_preserves_gradients():
    """Explicit cleanup releases graph state after gradients are consumed."""
    x = Tensor([1.0, 2.0], requires_grad=True)
    intermediate = x * x
    result = intermediate + x
    result.backward()
    expected_gradient = x.grad.copy()

    release_graph(result)

    assert result._prev == ()
    assert intermediate._prev == ()
    assert x._prev == ()
    np.testing.assert_array_equal(x.grad, expected_gradient)
