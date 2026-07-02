"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a ** b) * log(a)
        return grad_a, grad_b
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a,self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return out_grad * self.scalar * (a ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs  # 获取两个输入张量
        # 对 a 的梯度：∂(a/b)/∂a = 1/b
        grad_a = out_grad / b
        # 对 b 的梯度：∂(a/b)/∂b = -a/b^2
        grad_b = out_grad * (-a / (b ** 2))  # 或 -a / (b ** 2)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # 张量除以标量：a / scalar
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # y = a / scalar，所以 dy/da = 1/scalar
        # 最终梯度 = out_grad / scalar
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            axes = list(range(len(a.shape)))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            return array_api.transpose(a, axes=axes)

        if len(self.axes) == 2:
            axes = list(range(len(a.shape)))
            i, j = self.axes
            axes[i], axes[j] = axes[j], axes[i]
            return array_api.transpose(a, axes=axes)

        return array_api.transpose(a, axes=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return transpose(out_grad)

        if len(self.axes) == 2:
            return transpose(out_grad, axes=self.axes)

        inverse_axes = [0] * len(self.axes)
        for i, axis in enumerate(self.axes):
            inverse_axes[axis] = i
        return transpose(out_grad, axes=tuple(inverse_axes))
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 获取原始输入张量 a
        a, = node.inputs
        # 获取原始形状
        original_shape = a.shape
        # 梯度需要重塑回原始形状
        return reshape(out_grad, original_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        a, = node.inputs
        original_shape = a.shape
        aligned_shape = (1,) * (len(self.shape) - len(original_shape)) + original_shape

        reduce_axes = tuple(
            i for i, (in_dim, out_dim) in enumerate(zip(aligned_shape, self.shape))
            if in_dim == 1 and out_dim != 1
        )

        grad = out_grad
        if reduce_axes:
            grad = summation(grad, axes=reduce_axes)
        return reshape(grad, original_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return array_api.sum(a)
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        a, = node.inputs
        original_shape = a.shape
        axes = self.axes

        if axes is None:
            return broadcast_to(reshape(out_grad, (1,) * len(original_shape)), original_shape)

        if isinstance(axes, int):
            axes = (axes,)

        axes = tuple(axis if axis >= 0 else axis + len(original_shape) for axis in axes)
        reshape_shape = list(original_shape)
        for axis in axes:
            reshape_shape[axis] = 1

        grad = reshape(out_grad, tuple(reshape_shape))
        return broadcast_to(grad, original_shape)



def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        a, b = node.inputs

        grad_a = matmul(out_grad, transpose(b))
        grad_b = matmul(transpose(a), out_grad)

        if len(grad_a.shape) > len(a.shape):
            grad_a = summation(grad_a, axes=tuple(range(len(grad_a.shape) - len(a.shape))))
        if len(grad_b.shape) > len(b.shape):
            grad_b = summation(grad_b, axes=tuple(range(len(grad_b.shape) - len(b.shape))))

        reduce_axes_a = tuple(
            i for i, (gdim, adim) in enumerate(zip(grad_a.shape, a.shape))
            if adim == 1 and gdim != 1
        )
        if reduce_axes_a:
            grad_a = summation(grad_a, axes=reduce_axes_a)
            grad_a = reshape(grad_a, a.shape)

        reduce_axes_b = tuple(
            i for i, (gdim, bdim) in enumerate(zip(grad_b.shape, b.shape))
            if bdim == 1 and gdim != 1
        )
        if reduce_axes_b:
            grad_b = summation(grad_b, axes=reduce_axes_b)
            grad_b = reshape(grad_b, b.shape)

        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)

class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # y = -x，所以 dy/dx = -1
        # 最终梯度 = out_grad * (-1)
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        # y = log(a)，所以 dy/da = 1/a
        # 最终梯度 = out_grad / a
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # y = exp(a)，所以 dy/da = exp(a) = y
        # 需要获取前向传播的结果
        # 对于一元操作，node 的 outputs 包含前向传播结果
        a, = node.inputs
        # 方法1：重新计算 exp(a)
        # return out_grad * exp(a)
        
        # 方法2：从 node 获取前向传播结果（更高效）
        # 注意：在计算图中，node 的 outputs 可能包含前向传播的值
        # 但有些框架要求使用 inputs 重新计算
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        relu_grad = Tensor((a.realize_cached_data() > 0).astype(a.dtype))
        return out_grad * relu_grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

