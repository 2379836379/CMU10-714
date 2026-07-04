from typing import Optional
import numpy as np

from ..autograd import NDArray
from ..autograd import Tensor, TensorOp
from .ops_mathematic import *


class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        arr = Z if isinstance(Z, np.ndarray) else Z.numpy()
        shifted = arr - np.max(arr, axis=-1, keepdims=True)
        out = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        return out if isinstance(Z, np.ndarray) else type(Z)(out.astype(np.float32), device=Z.device)

    def gradient(self, out_grad: Tensor, node: Tensor):
        softmax = exp(node)
        summed = summation(out_grad, axes=(-1,))
        summed = reshape(summed, (*out_grad.shape[:-1], 1))
        summed = broadcast_to(summed, out_grad.shape)
        return out_grad - softmax * summed


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        arr = Z if isinstance(Z, np.ndarray) else Z.numpy()
        axes = self.axes
        if isinstance(axes, int):
            axes = (axes,)
        max_z = np.max(arr, axis=axes, keepdims=True)
        res = np.log(np.sum(np.exp(arr - max_z), axis=axes, keepdims=True)) + max_z
        out = np.squeeze(res) if axes is None else np.squeeze(res, axis=axes)
        return out if isinstance(Z, np.ndarray) else type(Z)(out.astype(np.float32), device=Z.device)

    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0]
        axes = self.axes
        if isinstance(axes, int):
            axes = (axes,)
        if axes is None:
            reshape_shape = (1,) * len(Z.shape)
        else:
            axes = tuple(axis if axis >= 0 else axis + len(Z.shape) for axis in axes)
            reshape_shape = list(Z.shape)
            for axis in axes:
                reshape_shape[axis] = 1
            reshape_shape = tuple(reshape_shape)
        node_b = broadcast_to(reshape(node, reshape_shape), Z.shape)
        out_b = broadcast_to(reshape(out_grad, reshape_shape), Z.shape)
        return exp(Z - node_b) * out_b


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
