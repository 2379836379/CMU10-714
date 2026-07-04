"""Sequence modules."""
from needle.autograd import Tensor
from needle import ops
import needle.init as init
from .nn_basic import Parameter, Module, Tanh, ReLU
import math


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return (1 + ops.exp(-x)) ** (-1)


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        super().__init__()
        bound = math.sqrt(1 / hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        self.act = Tanh() if nonlinearity == 'tanh' else ReLU()

    def forward(self, X, h=None):
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
        out = X @ self.W_ih + h @ self.W_hh
        if self.bias:
            out = out + ops.broadcast_to(self.bias_ih.reshape((1, self.hidden_size)), out.shape)
            out = out + ops.broadcast_to(self.bias_hh.reshape((1, self.hidden_size)), out.shape)
        return self.act(out)


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype) for i in range(num_layers)]

    def forward(self, X, h0=None):
        seq_len, bs, _ = X.shape
        inputs = list(ops.split(X, axis=0))
        if h0 is None:
            hs = [init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype) for _ in range(self.num_layers)]
        else:
            hs = list(ops.split(h0, axis=0))
        outputs = []
        for t in range(seq_len):
            x = inputs[t]
            for l, cell in enumerate(self.rnn_cells):
                hs[l] = cell(x, hs[l])
                x = hs[l]
            outputs.append(x)
        return ops.stack(outputs, axis=0), ops.stack(hs, axis=0)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        super().__init__()
        bound = math.sqrt(1 / hidden_size)
        self.hidden_size = hidden_size
        self.bias = bias
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def forward(self, X, h=None):
        bs = X.shape[0]
        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        gates = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
            gates = gates + ops.broadcast_to(self.bias_ih.reshape((1, 4 * self.hidden_size)), gates.shape)
            gates = gates + ops.broadcast_to(self.bias_hh.reshape((1, 4 * self.hidden_size)), gates.shape)
        gates = gates.reshape((bs, 4, self.hidden_size))
        i, f, g, o = list(ops.split(gates, axis=1))
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = self.tanh(g)
        o = self.sigmoid(o)
        c = f * c0 + i * g
        h = o * self.tanh(c)
        return h, c


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = [LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias, device=device, dtype=dtype) for i in range(num_layers)]

    def forward(self, X, h=None):
        seq_len, bs, _ = X.shape
        inputs = list(ops.split(X, axis=0))
        if h is None:
            hs = [init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype) for _ in range(self.num_layers)]
            cs = [init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype) for _ in range(self.num_layers)]
        else:
            hs = list(ops.split(h[0], axis=0))
            cs = list(ops.split(h[1], axis=0))
        outputs = []
        for t in range(seq_len):
            x = inputs[t]
            for l, cell in enumerate(self.lstm_cells):
                hs[l], cs[l] = cell(x, (hs[l], cs[l]))
                x = hs[l]
            outputs.append(x)
        return ops.stack(outputs, axis=0), (ops.stack(hs, axis=0), ops.stack(cs, axis=0))


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        seq_len, bs = x.shape
        x_flat = x.reshape((seq_len * bs,))
        one_hot = init.one_hot(self.num_embeddings, x_flat, device=x.device, dtype=self.weight.dtype)
        out = one_hot @ self.weight
        return out.reshape((seq_len, bs, self.embedding_dim))
