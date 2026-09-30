import numpy as np
import torch

from .tensor import Tensor, _accumulate_grad


def relu(tensor):
    out = Tensor(np.maximum(0, tensor.data), requires_grad=tensor.requires_grad)
    
    def _backward():
        _accumulate_grad(tensor, (tensor.data > 0) * out.grad)
    
    out._backward = _backward
    out._prev = {tensor, }
    return out


def sigmoid(tensor):
    out = Tensor(1 / (1 + np.exp(-tensor.data)), requires_grad=tensor.requires_grad)
    
    def _backward():
        _accumulate_grad(tensor, out.data * (1 - out.data) * out.grad)
    
    out._backward = _backward
    out._prev = {tensor, }
    return out


def matmul(tensor1, tensor2):
    assert isinstance(tensor2, Tensor), "Operand must be a Tensor"
    out = Tensor(tensor1.data @ tensor2.data, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    def _backward():
        _accumulate_grad(tensor1, out.grad @ tensor2.data.T)
        _accumulate_grad(tensor2, tensor1.data.T @ out.grad)
    
    out._backward = _backward
    out._prev = {tensor1, tensor2}
    return out

