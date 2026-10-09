import numpy as np

from .ops import conv2d, flatten, matmul, relu
from .tensor import Tensor


def _positive_integer(value, name):
    """Validate a positive integer constructor argument."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _state_array(value, name):
    """Convert a NumPy- or Torch-like state value to owned float32 data."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.array(value, dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"state value {name!r} must be numeric") from exc


class Module:
    """Base class for the explicit module types supported by simple-grad."""

    def parameters(self):
        return []

    def zero_grad(self):
        """Reset parameter gradients without replacing their buffers."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.fill(0)


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.in_features = _positive_integer(in_features, "in_features")
        self.out_features = _positive_integer(out_features, "out_features")
        # Retain the original library's simple initialization while storing
        # weights in PyTorch-compatible (out_features, in_features) order.
        self.weight = Tensor(
            np.random.randn(self.out_features, self.in_features) * 0.01,
            requires_grad=True,
            _data_category="parameters",
            _grad_category="gradients",
            _source="linear.weight",
        )
        self.bias = Tensor(
            np.zeros(self.out_features),
            requires_grad=True,
            _data_category="parameters",
            _grad_category="gradients",
            _source="linear.bias",
        )

    def __call__(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("Linear input must be a Tensor")
        if x.data.ndim != 2 or x.data.shape[1] != self.in_features:
            raise ValueError(
                f"Linear expected input shape (batch, {self.in_features}), "
                f"got {x.data.shape}"
            )
        return matmul(x, self.weight.transpose()) + self.bias

    def parameters(self):
        return [self.weight, self.bias]

    def load_state_dict(self, state_dict):
        expected_keys = {"weight", "bias"}
        if set(state_dict) != expected_keys:
            raise ValueError(
                f"Linear state keys must be {sorted(expected_keys)}, got {sorted(state_dict)}"
            )
        weight = _state_array(state_dict["weight"], "weight")
        bias = _state_array(state_dict["bias"], "bias")
        if weight.shape != self.weight.data.shape:
            raise ValueError(
                f"weight must have shape {self.weight.data.shape}, got {weight.shape}"
            )
        if bias.shape != self.bias.data.shape:
            raise ValueError(
                f"bias must have shape {self.bias.data.shape}, got {bias.shape}"
            )
        self.weight.data[...] = weight
        self.bias.data[...] = bias

    def state_dict(self):
        return {
            "weight": self.weight.data.copy(),
            "bias": self.bias.data.copy(),
        }


class Conv2d(Module):
    """Square stride-1 convolution without padding or dilation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        self.in_channels = _positive_integer(in_channels, "in_channels")
        self.out_channels = _positive_integer(out_channels, "out_channels")
        self.kernel_size = _positive_integer(kernel_size, "kernel_size")
        if not isinstance(bias, (bool, np.bool_)):
            raise TypeError("bias must be a boolean")
        if stride != 1 or padding != 0 or dilation != 1 or groups != 1:
            raise ValueError(
                "Conv2d supports only stride=1, padding=0, dilation=1, groups=1"
            )

        # Match Linear's small-normal policy for now. Initialization can be
        # revisited using training evidence without changing the layer API.
        self.weight = Tensor(
            np.random.randn(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
            * 0.01,
            requires_grad=True,
            _data_category="parameters",
            _grad_category="gradients",
            _source="conv2d.weight",
        )
        self.bias = (
            Tensor(
                np.zeros(self.out_channels),
                requires_grad=True,
                _data_category="parameters",
                _grad_category="gradients",
                _source="conv2d.bias",
            )
            if bias
            else None
        )

    def __call__(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("Conv2d input must be a Tensor")
        if x.data.ndim != 4 or x.data.shape[1] != self.in_channels:
            raise ValueError(
                f"Conv2d expected BCHW input with {self.in_channels} channels, "
                f"got {x.data.shape}"
            )
        return conv2d(x, self.weight, self.bias)

    def parameters(self):
        return [self.weight] if self.bias is None else [self.weight, self.bias]

    def load_state_dict(self, state_dict):
        expected_keys = {"weight"} if self.bias is None else {"weight", "bias"}
        if set(state_dict) != expected_keys:
            raise ValueError(
                f"Conv2d state keys must be {sorted(expected_keys)}, got {sorted(state_dict)}"
            )
        weight = _state_array(state_dict["weight"], "weight")
        if weight.shape != self.weight.data.shape:
            raise ValueError(
                f"weight must have shape {self.weight.data.shape}, got {weight.shape}"
            )
        self.weight.data[...] = weight
        if self.bias is not None:
            bias = _state_array(state_dict["bias"], "bias")
            if bias.shape != self.bias.data.shape:
                raise ValueError(
                    f"bias must have shape {self.bias.data.shape}, got {bias.shape}"
                )
            self.bias.data[...] = bias

    def state_dict(self):
        state = {"weight": self.weight.data.copy()}
        if self.bias is not None:
            state["bias"] = self.bias.data.copy()
        return state


class ReLU(Module):
    """Module wrapper for the functional ReLU operation."""

    def __call__(self, x):
        return relu(x)


class Flatten(Module):
    """Flatten a Tensor from ``start_dim`` through its final dimension."""

    def __init__(self, start_dim=1):
        self.start_dim = start_dim

    def __call__(self, x):
        return flatten(x, self.start_dim)


class Sequential(Module):
    """Apply a fixed, ordered collection of modules."""

    def __init__(self, *modules):
        if not all(isinstance(module, Module) for module in modules):
            raise TypeError("Sequential children must be Module instances")
        self.modules = tuple(modules)

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def parameters(self):
        parameters = []
        seen = set()
        for module in self.modules:
            for parameter in module.parameters():
                parameter_id = id(parameter)
                if parameter_id not in seen:
                    seen.add(parameter_id)
                    parameters.append(parameter)
        return parameters
