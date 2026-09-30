import numpy as np
import torch


def _noop_backward():
    """Shared no-op used by leaf and released Tensor nodes."""


def sum_to_shape(gradient, shape):
    """Reduce a broadcasted gradient back to an input's original shape."""
    try:
        gradient_data = np.asarray(gradient, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError("gradient must be numeric array-like") from exc

    try:
        target_shape = tuple(shape)
    except TypeError as exc:
        raise TypeError("shape must be an iterable of non-negative integers") from exc

    if any(
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or size < 0
        for size in target_shape
    ):
        raise TypeError("shape must contain only non-negative integers")

    if len(target_shape) > gradient_data.ndim:
        raise ValueError(
            f"cannot reduce gradient shape {gradient_data.shape} to {target_shape}"
        )

    leading_dimensions = gradient_data.ndim - len(target_shape)
    aligned_target = (1,) * leading_dimensions + target_shape
    incompatible_axes = [
        axis
        for axis, (gradient_size, target_size) in enumerate(
            zip(gradient_data.shape, aligned_target)
        )
        if target_size != 1 and target_size != gradient_size
    ]
    if incompatible_axes:
        raise ValueError(
            f"shape {target_shape} cannot broadcast to gradient shape "
            f"{gradient_data.shape}"
        )

    reduction_axes = tuple(range(leading_dimensions)) + tuple(
        axis
        for axis, (gradient_size, target_size) in enumerate(
            zip(gradient_data.shape, aligned_target)
        )
        if axis >= leading_dimensions
        and target_size == 1
        and gradient_size != 1
    )
    if reduction_axes:
        gradient_data = gradient_data.sum(
            axis=reduction_axes,
            keepdims=True,
            dtype=np.float32,
        )

    return gradient_data.reshape(target_shape)


def _accumulate_grad(tensor, contribution):
    """Add one gradient contribution to a Tensor's existing buffer."""
    if not tensor.requires_grad:
        return
    if tensor.grad is None:
        raise RuntimeError("a tensor requiring gradients must have a gradient buffer")

    try:
        contribution_data = np.asarray(contribution, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError("gradient contribution must be numeric array-like") from exc

    if contribution_data.shape != tensor.data.shape:
        raise ValueError(
            "gradient contribution shape must exactly match the tensor shape: "
            f"expected {tensor.data.shape}, got {contribution_data.shape}"
        )

    # Slice assignment preserves the persistent gradient-buffer object.
    tensor.grad[...] += contribution_data


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = _noop_backward
        self._prev = set()

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    @staticmethod
    def from_torch(tensor):
        return Tensor(tensor.detach().cpu().numpy(), requires_grad=tensor.requires_grad)

    def to_torch(self):
        return torch.tensor(self.data, requires_grad=self.requires_grad,
                            dtype=torch.float32)

    def transpose(self):
        out = Tensor(np.swapaxes(self.data, -1, -2), requires_grad=self.requires_grad) 

        def _backward():
            _accumulate_grad(self, np.swapaxes(out.grad, -1, -2))

        out._backward = _backward 
        out._prev = {self, }
        return out 
    
    def __add__(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward():
            _accumulate_grad(self, sum_to_shape(out.grad, self.data.shape))
            _accumulate_grad(other, sum_to_shape(out.grad, other.data.shape))
        
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "exponent must be an int or float"

        out = Tensor(self.data**other, requires_grad=self.requires_grad)

        def _backward():
            _accumulate_grad(self, (other * self.data**(other-1)) * out.grad)
        out._backward = _backward
        out._prev = {self}

        return out

    def __mul__(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            self_contribution = sum_to_shape(
                other.data * out.grad,
                self.data.shape,
            )
            other_contribution = sum_to_shape(
                self.data * out.grad,
                other.data.shape,
            )
            _accumulate_grad(self, self_contribution)
            _accumulate_grad(other, other_contribution)

        out._backward = _backward
        out._prev = {self, other}
        return out
    
    def backward(self, upstream=None):
        """Backpropagate an optional gradient through this tensor's graph."""
        if not self.requires_grad:
            raise RuntimeError("cannot call backward on a tensor that does not require gradients")

        if upstream is None:
            upstream_data = np.ones_like(self.data)
        else:
            if isinstance(upstream, Tensor):
                upstream = upstream.data
            try:
                upstream_data = np.asarray(upstream, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise TypeError("upstream must be a Tensor or numeric array-like") from exc

            if upstream_data.shape != self.data.shape:
                raise ValueError(
                    "upstream gradient shape must exactly match the output shape: "
                    f"expected {self.data.shape}, got {upstream_data.shape}"
                )

        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)

                topo.append(node)

        build_topo(self)

        for tensor in topo:
            if tensor._prev and tensor.requires_grad:
                tensor.grad.fill(0)

        if self._prev:
            self.grad[...] = upstream_data
        else:
            _accumulate_grad(self, upstream_data)

        for tensor in reversed(topo):
            tensor._backward()

    def __neg__(self): # -self
            return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1


def release_graph(root):
    """Remove backward closures and parent links reachable from ``root``."""
    if not isinstance(root, Tensor):
        raise TypeError("release_graph root must be a Tensor")

    nodes = []
    visited = set()
    stack = [root]
    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)
        nodes.append(node)
        stack.extend(node._prev)

    for node in nodes:
        node._prev = ()
        node._backward = _noop_backward
