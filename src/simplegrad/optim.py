from numbers import Real

import numpy as np

from .memory import temporary_array, track_array
from .tensor import Tensor


def _finite_real(value, name):
    """Return a finite Python float while rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class Adam:
    """Adam optimizer with PyTorch defaults and per-parameter state.

    Moment arrays are created lazily the first time a parameter has a gradient.
    Updates operate directly on Tensor data, so optimizer steps never extend the
    autograd graph.
    """

    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        try:
            supplied_parameters = list(parameters)
        except TypeError as exc:
            raise TypeError("parameters must be an iterable of Tensors") from exc

        if not supplied_parameters:
            raise ValueError("Adam requires at least one parameter")

        self.parameters = []
        seen = set()
        for parameter in supplied_parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("Adam parameters must be Tensors")
            if not parameter.requires_grad:
                raise ValueError("Adam parameters must require gradients")
            parameter_id = id(parameter)
            if parameter_id not in seen:
                seen.add(parameter_id)
                self.parameters.append(parameter)

        self.lr = _finite_real(lr, "lr")
        if self.lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise TypeError("betas must be a pair of real numbers")
        self.beta1 = _finite_real(betas[0], "beta1")
        self.beta2 = _finite_real(betas[1], "beta2")
        if not 0 <= self.beta1 < 1:
            raise ValueError("beta1 must satisfy 0 <= beta1 < 1")
        if not 0 <= self.beta2 < 1:
            raise ValueError("beta2 must satisfy 0 <= beta2 < 1")

        self.eps = _finite_real(eps, "eps")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

        # Tensor uses identity hashing, so state is naturally per parameter and
        # retaining the key prevents object-id reuse from aliasing old state.
        self.state = {}

    def zero_grad(self):
        """Zero existing parameter gradients without replacing their arrays."""
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.fill(0)

    def _initialize_state(self, parameter):
        """Create float32 moment buffers for one parameter on first use."""
        first_moment = np.zeros_like(parameter.data, dtype=np.float32)
        second_moment = np.zeros_like(parameter.data, dtype=np.float32)
        state = {
            "step": 0,
            "exp_avg": first_moment,
            "exp_avg_sq": second_moment,
            "_handles": (
                track_array(
                    first_moment,
                    "optimizer_state",
                    "adam.exp_avg",
                ),
                track_array(
                    second_moment,
                    "optimizer_state",
                    "adam.exp_avg_sq",
                ),
            ),
        }
        self.state[parameter] = state
        return state

    def step(self):
        """Update every parameter that currently has a gradient."""
        beta1 = np.float32(self.beta1)
        beta2 = np.float32(self.beta2)
        one_minus_beta1 = np.float32(1.0 - self.beta1)
        one_minus_beta2 = np.float32(1.0 - self.beta2)
        learning_rate = np.float32(self.lr)
        epsilon = np.float32(self.eps)

        for parameter in self.parameters:
            if parameter.grad is None:
                continue

            try:
                gradient = np.asarray(parameter.grad, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise TypeError("parameter gradients must be numeric arrays") from exc
            if gradient.shape != parameter.data.shape:
                raise ValueError(
                    "parameter gradient shape must match parameter data: "
                    f"expected {parameter.data.shape}, got {gradient.shape}"
                )

            state = self.state.get(parameter)
            if state is None:
                state = self._initialize_state(parameter)

            state["step"] += 1
            step = state["step"]
            first_moment = state["exp_avg"]
            second_moment = state["exp_avg_sq"]

            # Update persistent moments in place to keep their allocation and
            # identity stable across training steps.
            first_moment *= beta1
            first_update = one_minus_beta1 * gradient
            with temporary_array(first_update, "adam.first_moment_update"):
                first_moment += first_update
            del first_update
            second_moment *= beta2
            squared_gradient = gradient * gradient
            with temporary_array(squared_gradient, "adam.squared_gradient"):
                second_update = one_minus_beta2 * squared_gradient
                with temporary_array(second_update, "adam.second_moment_update"):
                    second_moment += second_update
                del second_update
            del squared_gradient

            bias_correction1 = np.float32(1.0 - self.beta1**step)
            bias_correction2 = np.float32(1.0 - self.beta2**step)
            corrected_first_moment = first_moment / bias_correction1
            with temporary_array(corrected_first_moment, "adam.corrected_first"):
                corrected_second_moment = second_moment / bias_correction2
                with temporary_array(corrected_second_moment, "adam.corrected_second"):
                    square_root = np.sqrt(corrected_second_moment)
                    square_root_handle = track_array(
                        square_root,
                        "transient_other",
                        "adam.square_root",
                    )
                    try:
                        denominator = square_root + epsilon
                    finally:
                        square_root_handle.release()
                        del square_root
                    with temporary_array(denominator, "adam.denominator"):
                        numerator = learning_rate * corrected_first_moment
                        with temporary_array(numerator, "adam.numerator"):
                            update = numerator / denominator
                            with temporary_array(update, "adam.update"):
                                # Raw ndarray mutation is deliberately outside autograd.
                                parameter.data[...] -= update
                            del update
                        del numerator
                    del denominator
                del corrected_second_moment
            del corrected_first_moment
