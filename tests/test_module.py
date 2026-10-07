import numpy as np
import pytest
import torch
from torch import nn

from simplegrad.tensor import Tensor
from simplegrad.module import Conv2d, Flatten, Linear, ReLU, Sequential
from simplegrad.ops import cross_entropy


## State loading copies the same float32 values without numerical computation;
## NumPy's default tolerance should pass.
def test_load_state_dict_from_torch():
    D_IN, D_OUT = 32, 64
    pt_module = nn.Linear(D_IN, D_OUT, bias=True)
    module = Linear(D_IN, D_OUT)
    module.load_state_dict(pt_module.state_dict())

    np.testing.assert_allclose(module.weight.data, pt_module.weight.detach().numpy())
    np.testing.assert_allclose(module.bias.data, pt_module.bias.detach().numpy())


## Linear contains a matmul reduction, so use PyTorch's float32 test defaults:
## rtol=1.3e-6 and atol=1e-5.
def test_linear_module():
    B, D_IN, D_OUT = 8, 32, 64
    pt_module = nn.Linear(D_IN, D_OUT, bias=True)
    module = Linear(D_IN, D_OUT)
    module.load_state_dict(pt_module.state_dict())

    x = Tensor(np.random.rand(B, D_IN))
    pt_x = x.to_torch()

    expected = pt_module(pt_x)
    result = module(x) 

    np.testing.assert_allclose(result.data, expected.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)


## Both expected and actual weights come from the same seeded values and are
## cast to float32; NumPy's default tolerance should pass.
def test_linear_and_conv2d_use_small_normal_weights_and_zero_biases():
    """Module initialization retains the original 0.01-normal policy."""
    random_state = np.random.get_state()
    try:
        np.random.seed(17)
        expected_linear_weight = np.random.randn(3, 4) * 0.01
        np.random.seed(17)
        linear = Linear(4, 3)

        np.random.seed(29)
        expected_conv_weight = np.random.randn(2, 1, 3, 3) * 0.01
        np.random.seed(29)
        convolution = Conv2d(1, 2, 3)
    finally:
        np.random.set_state(random_state)

    np.testing.assert_allclose(
        linear.weight.data,
        expected_linear_weight.astype(np.float32),
    )
    np.testing.assert_array_equal(linear.bias.data, np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(
        convolution.weight.data,
        expected_conv_weight.astype(np.float32),
    )
    np.testing.assert_array_equal(
        convolution.bias.data,
        np.zeros(2, dtype=np.float32),
    )


## Linear forward and all backward paths inherit matmul and reduction rounding;
## use PyTorch's float32 defaults, rtol=1.3e-6 and atol=1e-5.
def test_linear_backward_and_bias_match_torch():
    """Linear propagates into input, weight, and broadcast bias."""
    pt_module = nn.Linear(4, 3)
    module = Linear(4, 3)
    module.load_state_dict(pt_module.state_dict())
    x_data = np.arange(8, dtype=np.float32).reshape(2, 4) / 4
    pt_x = torch.tensor(x_data, requires_grad=True)
    x = Tensor(x_data, requires_grad=True)
    upstream = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 1.5, 2.0]])

    expected = pt_module(pt_x)
    result = module(x)
    expected.backward(upstream)
    result.backward(upstream.numpy())

    np.testing.assert_allclose(result.data, expected.detach().numpy(), rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(x.grad, pt_x.grad.numpy(), rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(
        module.weight.grad,
        pt_module.weight.grad.numpy(),
        rtol=1.3e-6,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        module.bias.grad,
        pt_module.bias.grad.numpy(),
        rtol=1.3e-6,
        atol=1e-5,
    )


def test_linear_state_dict_is_an_owned_snapshot():
    """Exported state cannot mutate live module parameters."""
    module = Linear(3, 2)
    state = module.state_dict()
    original_weight = module.weight.data.copy()

    state["weight"].fill(123)

    np.testing.assert_array_equal(module.weight.data, original_weight)


def test_linear_load_state_dict_preserves_parameter_storage():
    """Loading copies values in place so Tensor and grad identities remain valid."""
    module = Linear(3, 2)
    weight_storage = module.weight.data
    gradient_storage = module.weight.grad
    state = {
        "weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "bias": np.array([0.5, -0.5], dtype=np.float32),
    }

    module.load_state_dict(state)

    assert module.weight.data is weight_storage
    assert module.weight.grad is gradient_storage
    np.testing.assert_array_equal(module.weight.data, state["weight"])


## Conv2d uses different multiply-accumulate kernels in NumPy and PyTorch;
## rtol=1e-4 and atol=1e-5 allow float32 accumulation-order differences.
@pytest.mark.parametrize("implementation", ["patch", "strided", "im2col"])
def test_conv2d_module_matches_torch(implementation):
    """Conv2d stores PyTorch-compatible parameters and delegates correctly."""
    pt_module = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=0, bias=True)
    module = Conv2d(2, 3, kernel_size=3, implementation=implementation)
    module.load_state_dict(pt_module.state_dict())
    x_data = np.linspace(-1.0, 1.0, 2 * 2 * 5 * 5, dtype=np.float32).reshape(2, 2, 5, 5)
    pt_x = torch.tensor(x_data, requires_grad=True)
    x = Tensor(x_data, requires_grad=True)

    expected = pt_module(pt_x)
    result = module(x)

    np.testing.assert_allclose(result.data, expected.detach().numpy(), rtol=1e-4, atol=1e-5)
    assert [parameter.data.shape for parameter in module.parameters()] == [
        (3, 2, 3, 3),
        (3,),
    ]


def test_conv2d_without_bias_and_unsupported_geometry():
    """Bias is optional and unsupported convolution options fail explicitly."""
    module = Conv2d(1, 2, kernel_size=3, bias=False)

    assert module.bias is None
    assert module.parameters() == [module.weight]
    with pytest.raises(ValueError, match="supports only"):
        Conv2d(1, 2, kernel_size=3, stride=2)
    with pytest.raises(ValueError, match="implementation"):
        Conv2d(1, 2, kernel_size=3, implementation="unknown")


def test_sequential_convnet_shape_and_parameter_order():
    """Composition produces logits and parameters in stable module order."""
    model = Sequential(
        Conv2d(1, 2, kernel_size=3),
        ReLU(),
        Flatten(),
        Linear(2 * 4 * 4, 5),
    )

    result = model(Tensor(np.ones((3, 1, 6, 6), dtype=np.float32)))

    assert result.data.shape == (3, 5)
    assert [parameter.data.shape for parameter in model.parameters()] == [
        (2, 1, 3, 3),
        (2,),
        (5, 32),
        (5,),
    ]


def test_sequential_convnet_backward_reaches_every_parameter():
    """The complete classifier graph carries loss gradients to every parameter."""
    rng = np.random.default_rng(2026)
    model = Sequential(
        Conv2d(1, 2, kernel_size=3),
        ReLU(),
        Flatten(),
        Linear(2 * 4 * 4, 3),
    )
    inputs = Tensor(rng.normal(size=(2, 1, 6, 6)).astype(np.float32))
    targets = np.array([0, 2], dtype=np.int64)

    loss = cross_entropy(model(inputs), targets)
    loss.backward()

    assert np.isfinite(loss.data)
    for parameter in model.parameters():
        assert np.all(np.isfinite(parameter.grad))
        assert np.any(parameter.grad != 0)


def test_sequential_deduplicates_shared_parameters():
    """A reused module exposes each parameter once to an optimizer."""
    shared = Linear(3, 3)
    model = Sequential(shared, ReLU(), shared)

    assert model.parameters() == [shared.weight, shared.bias]


def test_module_zero_grad_reuses_buffers():
    """Gradient reset keeps persistent buffer identity for memory accounting."""
    module = Linear(2, 3)
    module.weight.grad.fill(7)
    module.bias.grad.fill(-2)
    weight_grad = module.weight.grad
    bias_grad = module.bias.grad

    module.zero_grad()

    assert module.weight.grad is weight_grad
    assert module.bias.grad is bias_grad
    np.testing.assert_array_equal(module.weight.grad, 0)
    np.testing.assert_array_equal(module.bias.grad, 0)
