from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=-1, keepdims=True)
        shifted = Z - max_z
        logsumexp = array_api.log(
            array_api.sum(array_api.exp(shifted), axis=-1, keepdims=True)
        )
        return shifted - logsumexp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        softmax = exp(node)
        summed = summation(out_grad, axes=(-1,))
        summed = reshape(summed, (*out_grad.shape[:-1], 1))
        summed = broadcast_to(summed, out_grad.shape)
        return out_grad - softmax * summed
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        axes = self.axes
        if isinstance(axes, int):
            axes = (axes,)

        max_z = array_api.max(Z, axis=axes, keepdims=True)
        shifted = Z - max_z
        res = array_api.log(
            array_api.sum(array_api.exp(shifted), axis=axes, keepdims=True)
        ) + max_z

        if axes is None:
            return array_api.squeeze(res)
        return array_api.squeeze(res, axis=axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
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

        node_reshaped = reshape(node, reshape_shape)
        out_grad_reshaped = reshape(out_grad, reshape_shape)
        node_broadcast = broadcast_to(node_reshaped, Z.shape)
        out_grad_broadcast = broadcast_to(out_grad_reshaped, Z.shape)
        return exp(Z - node_broadcast) * out_grad_broadcast
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)