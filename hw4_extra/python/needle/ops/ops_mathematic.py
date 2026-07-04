"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

import numpy as np

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp


def _ndarray_from_numpy(arr, device):
    if isinstance(device, str) or device is None:
        return arr
    return type(device.rand(1))(arr.astype(np.float32), device=device)


def _to_numpy(a):
    return a if isinstance(a, np.ndarray) else a.numpy()


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
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad * b * (a ** (b - 1)), out_grad * (a ** b) * log(a)


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        a, = node.inputs
        return out_grad * self.scalar * (a ** (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad / b, out_grad * (-a / (b ** 2))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            axes = list(range(len(a.shape)))
            axes[-1], axes[-2] = axes[-2], axes[-1]
        else:
            axes = list(range(len(a.shape)))
            i, j = self.axes
            axes[i], axes[j] = axes[j], axes[i]
        if isinstance(a, np.ndarray):
            return np.transpose(a, axes=axes)
        return a.permute(tuple(axes))

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        if isinstance(a, np.ndarray):
            return a.reshape(self.shape)
        return a.compact().reshape(self.shape)

    def gradient(self, out_grad, node):
        a, = node.inputs
        return reshape(out_grad, a.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        if isinstance(a, np.ndarray):
            return np.broadcast_to(a, self.shape)
        if len(self.shape) > len(a.shape):
            a = a.reshape((1,) * (len(self.shape) - len(a.shape)) + a.shape)
        return a.broadcast_to(self.shape)

    def gradient(self, out_grad, node):
        a, = node.inputs
        original_shape = a.shape
        aligned_shape = (1,) * (len(self.shape) - len(original_shape)) + original_shape
        reduce_axes = tuple(i for i, (in_dim, out_dim) in enumerate(zip(aligned_shape, self.shape)) if in_dim == 1 and out_dim != 1)
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
        axes = self.axes
        if isinstance(axes, list):
            axes = tuple(axes)
        if isinstance(a, np.ndarray):
            return np.sum(a, axis=axes)
        if axes is None:
            return a.sum()
        if isinstance(axes, tuple) and len(axes) > 1:
            out = a
            for ax in sorted([x if x >= 0 else x + len(a.shape) for x in axes], reverse=True):
                out = out.sum(axis=ax)
            return out
        return a.sum(axis=axes)

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
        return a @ b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = matmul(out_grad, transpose(b))
        grad_b = matmul(transpose(a), out_grad)
        if len(grad_a.shape) > len(a.shape):
            grad_a = summation(grad_a, axes=tuple(range(len(grad_a.shape) - len(a.shape))))
        if len(grad_b.shape) > len(b.shape):
            grad_b = summation(grad_b, axes=tuple(range(len(grad_b.shape) - len(b.shape))))
        reduce_axes_a = tuple(i for i, (gdim, adim) in enumerate(zip(grad_a.shape, a.shape)) if adim == 1 and gdim != 1)
        if reduce_axes_a:
            grad_a = reshape(summation(grad_a, axes=reduce_axes_a), a.shape)
        reduce_axes_b = tuple(i for i, (gdim, bdim) in enumerate(zip(grad_b.shape, b.shape)) if bdim == 1 and gdim != 1)
        if reduce_axes_b:
            grad_b = reshape(summation(grad_b, axes=reduce_axes_b), b.shape)
        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return np.log(a) if isinstance(a, np.ndarray) else a.log()

    def gradient(self, out_grad, node):
        a, = node.inputs
        return out_grad / a


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return np.exp(a) if isinstance(a, np.ndarray) else a.exp()

    def gradient(self, out_grad, node):
        a, = node.inputs
        return out_grad * exp(a)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return np.maximum(a, 0) if isinstance(a, np.ndarray) else a.maximum(0)

    def gradient(self, out_grad, node):
        a, = node.inputs
        mask = Tensor((a.realize_cached_data().numpy() > 0).astype(np.float32) if not isinstance(a.realize_cached_data(), np.ndarray) else (a.realize_cached_data() > 0).astype(np.float32), device=a.device, requires_grad=False)
        return out_grad * mask


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return np.tanh(a) if isinstance(a, np.ndarray) else a.tanh()

    def gradient(self, out_grad, node):
        return out_grad * ((node * node) * (-1) + 1)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, args):
        if len(args) == 0:
            raise ValueError('stack expects at least one tensor')
        first = args[0]
        arr = np.stack([_to_numpy(x) for x in args], axis=self.axis)
        return arr if isinstance(first, np.ndarray) else type(first)(arr.astype(np.float32), device=first.device)

    def gradient(self, out_grad, node):
        return split(out_grad, axis=self.axis)


def stack(args, axis=0):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, A):
        pieces = [_to_numpy(A).take(i, axis=self.axis) for i in range(A.shape[self.axis])]
        if isinstance(A, np.ndarray):
            return tuple(pieces)
        return tuple(type(A)(x.astype(np.float32), device=A.device) for x in pieces)

    def gradient(self, out_grad, node):
        return stack(list(out_grad), axis=self.axis)


def split(a, axis=0):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: tuple[int, ...]):
        self.axes = axes

    def compute(self, a):
        arr = np.flip(_to_numpy(a), self.axes)
        return arr if isinstance(a, np.ndarray) else type(a)(arr.astype(np.float32), device=a.device)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, dilation: int, axes: tuple[int, ...]):
        self.dilation = dilation
        self.axes = axes

    def compute(self, a):
        arr = _to_numpy(a)
        new_shape = list(arr.shape)
        for ax in self.axes:
            new_shape[ax] = arr.shape[ax] * (self.dilation + 1)
        out = np.zeros(new_shape, dtype=arr.dtype)
        idx = [slice(None)] * arr.ndim
        for ax in self.axes:
            idx[ax] = slice(0, new_shape[ax], self.dilation + 1)
        out[tuple(idx)] = arr
        return out if isinstance(a, np.ndarray) else type(a)(out.astype(np.float32), device=a.device)

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.dilation, self.axes)


def dilate(a, dilation, axes):
    return Dilate(dilation, axes)(a)


class UnDilate(TensorOp):
    def __init__(self, dilation: int, axes: tuple[int, ...]):
        self.dilation = dilation
        self.axes = axes

    def compute(self, a):
        idx = [slice(None)] * len(a.shape)
        for ax in self.axes:
            idx[ax] = slice(0, a.shape[ax], self.dilation + 1)
        return a[tuple(idx)]

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.dilation, self.axes)


def undilate(a, dilation, axes):
    return UnDilate(dilation, axes)(a)


def _conv_numpy(X, W, stride=1, padding=0):
    if padding > 0:
        X = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    N, H, W_in, C_in = X.shape
    K = W.shape[0]
    C_out = W.shape[3]
    out_h = (H - K) // stride + 1
    out_w = (W_in - K) // stride + 1
    out = np.zeros((N, out_h, out_w, C_out), dtype=X.dtype)
    for i in range(out_h):
        hs = i * stride
        for j in range(out_w):
            ws = j * stride
            patch = X[:, hs:hs + K, ws:ws + K, :]
            out[:, i, j, :] = np.tensordot(patch, W, axes=([1, 2, 3], [0, 1, 2]))
    return out


def _conv_backward_numpy(X, W, out_grad, stride=1, padding=0):
    X_pad = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0))) if padding > 0 else X
    N, H_pad, W_pad, C_in = X_pad.shape
    K = W.shape[0]
    out_h, out_w, C_out = out_grad.shape[1], out_grad.shape[2], out_grad.shape[3]
    dX_pad = np.zeros_like(X_pad)
    dW = np.zeros_like(W)
    for kh in range(K):
        for kw in range(K):
            x_slice = X_pad[:, kh:kh + out_h * stride:stride, kw:kw + out_w * stride:stride, :]
            dW[kh, kw, :, :] = np.tensordot(x_slice, out_grad, axes=([0, 1, 2], [0, 1, 2]))
            dx_piece = np.tensordot(out_grad, W[kh, kw, :, :], axes=([3], [1]))
            dX_pad[:, kh:kh + out_h * stride:stride, kw:kw + out_w * stride:stride, :] += dx_piece
    if padding > 0:
        dX = dX_pad[:, padding:-padding, padding:-padding, :]
    else:
        dX = dX_pad
    return dX, dW


class Conv(TensorOp):
    def __init__(self, stride: int = 1, padding: int = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        out = _conv_numpy(_to_numpy(A), _to_numpy(B), self.stride, self.padding)
        return out if isinstance(A, np.ndarray) else type(A)(out.astype(np.float32), device=A.device)

    def gradient(self, out_grad, node):
        X, W = node.inputs
        dX, dW = _conv_backward_numpy(X.numpy(), W.numpy(), out_grad.numpy(), self.stride, self.padding)
        return Tensor(dX, device=X.device, requires_grad=False), Tensor(dW, device=W.device, requires_grad=False)


def conv(a, b, stride=1, padding=0):
    return Conv(stride=stride, padding=padding)(a, b)


from .ops_tuple import make_tuple
