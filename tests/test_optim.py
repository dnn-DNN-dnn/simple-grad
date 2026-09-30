import numpy as np
import pytest
import torch

from simplegrad.optim import Adam
from simplegrad.tensor import Tensor


def _torch_state(torch_optimizer, torch_parameter):
    """Return PyTorch's Adam state using names comparable to simple-grad."""
    state = torch_optimizer.state[torch_parameter]
    return {
        "step": int(state["step"].item()),
        "exp_avg": state["exp_avg"].detach().numpy(),
        "exp_avg_sq": state["exp_avg_sq"].detach().numpy(),
    }


def test_adam_defaults():
    """Adam exposes the assignment's PyTorch-default hyperparameters."""
    optimizer = Adam([Tensor([1.0], requires_grad=True)])

    assert optimizer.lr == 1e-3
    assert optimizer.beta1 == 0.9
    assert optimizer.beta2 == 0.999
    assert optimizer.eps == 1e-8


def test_adam_matches_pytorch_across_multiple_steps():
    """Parameters and both moments match PyTorch over stateful updates."""
    first_data = np.array([[1.0, -2.0, 3.0], [0.5, 1.5, -0.25]], dtype=np.float32)
    second_data = np.array([0.25, -0.75, 2.0, -1.0], dtype=np.float32)
    first = Tensor(first_data, requires_grad=True)
    second = Tensor(second_data, requires_grad=True)
    torch_first = torch.tensor(first_data, requires_grad=True)
    torch_second = torch.tensor(second_data, requires_grad=True)
    optimizer = Adam([first, second], lr=3e-3, betas=(0.8, 0.95), eps=1e-7)
    torch_optimizer = torch.optim.Adam(
        [torch_first, torch_second],
        lr=3e-3,
        betas=(0.8, 0.95),
        eps=1e-7,
        foreach=False,
        fused=False,
    )

    gradient_steps = [
        (
            np.array([[0.1, -0.2, 0.3], [0.0, 0.5, -0.4]], dtype=np.float32),
            np.array([0.2, 0.0, -0.3, 0.7], dtype=np.float32),
        ),
        (
            np.array([[-0.4, 0.1, 0.0], [0.6, -0.2, 0.3]], dtype=np.float32),
            np.array([-0.5, 0.25, 0.1, -0.2], dtype=np.float32),
        ),
        (
            np.zeros((2, 3), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        ),
        (
            np.array([[1.0, -0.5, 0.2], [-0.3, 0.8, -0.9]], dtype=np.float32),
            np.array([0.4, -0.6, 0.9, 0.3], dtype=np.float32),
        ),
        (
            np.array([[0.05, 0.15, -0.25], [0.35, -0.45, 0.55]], dtype=np.float32),
            np.array([-0.1, 0.2, -0.3, 0.4], dtype=np.float32),
        ),
    ]

    for expected_step, (first_gradient, second_gradient) in enumerate(
        gradient_steps,
        start=1,
    ):
        first.grad[...] = first_gradient
        second.grad[...] = second_gradient
        torch_first.grad = torch.tensor(first_gradient)
        torch_second.grad = torch.tensor(second_gradient)

        optimizer.step()
        torch_optimizer.step()

        np.testing.assert_allclose(
            first.data,
            torch_first.detach().numpy(),
            rtol=1e-4,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            second.data,
            torch_second.detach().numpy(),
            rtol=1e-4,
            atol=1e-6,
        )
        for parameter, torch_parameter in (
            (first, torch_first),
            (second, torch_second),
        ):
            state = optimizer.state[parameter]
            torch_state = _torch_state(torch_optimizer, torch_parameter)
            assert state["step"] == expected_step == torch_state["step"]
            np.testing.assert_allclose(
                state["exp_avg"],
                torch_state["exp_avg"],
                rtol=1e-5,
                atol=1e-7,
            )
            np.testing.assert_allclose(
                state["exp_avg_sq"],
                torch_state["exp_avg_sq"],
                rtol=1e-5,
                atol=1e-7,
            )


def test_adam_skips_missing_gradient_with_per_parameter_step():
    """A skipped parameter creates no state and starts its own counter later."""
    first = Tensor([1.0, -1.0], requires_grad=True)
    second = Tensor([0.5, 2.0], requires_grad=True)
    torch_first = torch.tensor([1.0, -1.0], requires_grad=True)
    torch_second = torch.tensor([0.5, 2.0], requires_grad=True)
    optimizer = Adam([first, second])
    torch_optimizer = torch.optim.Adam(
        [torch_first, torch_second],
        foreach=False,
        fused=False,
    )

    first.grad[...] = [0.25, -0.5]
    second.grad = None
    torch_first.grad = torch.tensor([0.25, -0.5])
    torch_second.grad = None
    optimizer.step()
    torch_optimizer.step()

    assert first in optimizer.state
    assert second not in optimizer.state
    np.testing.assert_array_equal(second.data, torch_second.detach().numpy())

    first.grad[...] = [-0.1, 0.2]
    second.grad = np.array([0.3, -0.4], dtype=np.float32)
    torch_first.grad = torch.tensor([-0.1, 0.2])
    torch_second.grad = torch.tensor([0.3, -0.4])
    optimizer.step()
    torch_optimizer.step()

    assert optimizer.state[first]["step"] == 2
    assert optimizer.state[second]["step"] == 1
    np.testing.assert_allclose(first.data, torch_first.detach().numpy(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(second.data, torch_second.detach().numpy(), rtol=1e-4, atol=1e-6)


def test_adam_deduplicates_parameters_by_identity():
    """A repeated Tensor receives one state entry and one update per step."""
    parameter = Tensor([1.0, -2.0], requires_grad=True)
    reference = Tensor([1.0, -2.0], requires_grad=True)
    parameter_data = parameter.data
    optimizer = Adam([parameter, parameter])
    reference_optimizer = Adam([reference])
    parameter.grad[...] = [0.5, -0.25]
    reference.grad[...] = parameter.grad

    optimizer.step()
    reference_optimizer.step()

    assert optimizer.parameters == [parameter]
    assert parameter.data is parameter_data
    assert len(optimizer.state) == 1
    np.testing.assert_array_equal(parameter.data, reference.data)


def test_adam_reuses_moment_buffers_across_steps():
    """Persistent optimizer-state arrays are updated rather than replaced."""
    parameter = Tensor([1.0, -2.0], requires_grad=True)
    optimizer = Adam([parameter])
    parameter.grad[...] = [0.5, -0.25]
    optimizer.step()
    first_moment = optimizer.state[parameter]["exp_avg"]
    second_moment = optimizer.state[parameter]["exp_avg_sq"]

    parameter.grad[...] = [-0.1, 0.3]
    optimizer.step()

    assert optimizer.state[parameter]["exp_avg"] is first_moment
    assert optimizer.state[parameter]["exp_avg_sq"] is second_moment
    assert optimizer.state[parameter]["step"] == 2


def test_adam_zero_grad_preserves_buffer_identity():
    """Gradient reset clears values without creating replacement arrays."""
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad[...] = [3.0, -4.0]
    gradient_buffer = parameter.grad
    optimizer = Adam([parameter])

    optimizer.zero_grad()

    assert parameter.grad is gradient_buffer
    np.testing.assert_array_equal(parameter.grad, np.zeros(2, dtype=np.float32))


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"lr": 0}, ValueError, "lr must be positive"),
        ({"lr": float("inf")}, ValueError, "lr must be finite"),
        ({"betas": (-0.1, 0.999)}, ValueError, "beta1"),
        ({"betas": (0.9, 1.0)}, ValueError, "beta2"),
        ({"betas": (0.9,)}, TypeError, "pair"),
        ({"eps": 0}, ValueError, "eps must be positive"),
    ],
)
def test_adam_rejects_invalid_hyperparameters(kwargs, error, message):
    """Invalid optimizer settings fail during construction."""
    parameter = Tensor([1.0], requires_grad=True)

    with pytest.raises(error, match=message):
        Adam([parameter], **kwargs)


def test_adam_rejects_invalid_parameters_and_gradient_shapes():
    """Only trainable Tensors with shape-compatible gradients may update."""
    with pytest.raises(ValueError, match="at least one"):
        Adam([])
    with pytest.raises(TypeError, match="must be Tensors"):
        Adam([np.array([1.0])])
    with pytest.raises(ValueError, match="must require gradients"):
        Adam([Tensor([1.0])])

    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad = np.ones((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="gradient shape"):
        Adam([parameter]).step()
