"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys
sys.path.append("python/")
import needle as ndl
import needle.nn as nn
from apps.models import *

device = ndl.cpu()


def parse_mnist(image_filename, label_filename):
    with gzip.open(image_filename, 'rb') as f:
        _, num, rows, cols = struct.unpack('>4I', f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32).reshape(num, rows * cols) / 255.0
    with gzip.open(label_filename, 'rb') as f:
        _, num = struct.unpack('>2I', f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8)
    return X, y


def softmax_loss(Z, y_one_hot):
    return (ndl.logsumexp(Z, axes=(1,)) - (Z * y_one_hot).sum((1,))).sum() / Z.shape[0]


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    n = X.shape[0]
    for i in range(0, n, batch):
        Xb = ndl.Tensor(X[i:i + batch], requires_grad=False)
        yb = y[i:i + batch]
        y_one_hot = np.zeros((len(yb), W2.shape[1]), dtype=np.float32)
        y_one_hot[np.arange(len(yb)), yb] = 1
        yb_t = ndl.Tensor(y_one_hot, requires_grad=False)
        logits = ndl.relu(Xb @ W1) @ W2
        loss = softmax_loss(logits, yb_t)
        loss.backward()
        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
    return W1, W2


def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    np.random.seed(4)
    if opt is None:
        model.eval()
    else:
        model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for X, y in dataloader:
        logits = model(X)
        loss = loss_fn(logits, y)
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
        total_loss += loss.numpy() * X.shape[0]
        total_correct += (logits.numpy().argmax(axis=1) == y.numpy()).sum()
        total += X.shape[0]
    return total_correct / total, total_loss / total


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam, lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    np.random.seed(4)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_epochs):
        acc, loss = epoch_general_cifar10(dataloader, model, loss_fn(), opt)
    return acc, loss


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    np.random.seed(4)
    return epoch_general_cifar10(dataloader, model, loss_fn(), opt=None)


def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None, clip=None, device=None, dtype="float32"):
    np.random.seed(4)
    if opt is None:
        model.eval()
    else:
        model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    h = None
    for i in range(0, data.shape[0] - 1, seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        out, h = model(X, h)
        loss = loss_fn(out, y)
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            if clip is not None and hasattr(opt, 'clip_grad_norm'):
                opt.clip_grad_norm(clip)
            opt.step()
        if isinstance(h, tuple):
            h = tuple(v.detach() for v in h)
        elif h is not None:
            h = h.detach()
        total_loss += loss.numpy() * y.shape[0]
        total_correct += (out.numpy().argmax(axis=1) == y.numpy()).sum()
        total_tokens += y.shape[0]
    return total_correct / total_tokens, total_loss / total_tokens


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD, lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None, device=None, dtype="float32"):
    np.random.seed(4)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_epochs):
        acc, loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn(), opt=opt, clip=clip, device=device, dtype=dtype)
    return acc, loss


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss, device=None, dtype="float32"):
    np.random.seed(4)
    return epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn(), opt=None, device=device, dtype=dtype)


def loss_err(h, y):
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
