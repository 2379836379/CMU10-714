"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE

    # 读取图像文件
    with gzip.open(image_filename, 'rb') as f:
        # 读取魔数、图像数量、行数、列数
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        # 读取图像数据，并转换为float32
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols).astype(np.float32)
        # 归一化到[0.0, 1.0]
        X = X / 255.0

    # 读取标签文件
    with gzip.open(label_filename, 'rb') as f:
        # 读取魔数和标签数量
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        # 读取标签数据，保持uint8类型
        y = np.frombuffer(f.read(), dtype=np.uint8)

    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR CODE
    # 计算 log-sum-exp: log(sum(exp(Z_i))) 对每个样本
    # axis=1 表示对每个样本的类别维度求和
    log_sum_exp = ndl.log(ndl.summation(ndl.exp(Z), axes=1))
    log_sum_exp = ndl.reshape(log_sum_exp, (Z.shape[0], 1))
    log_sum_exp = ndl.broadcast_to(log_sum_exp, Z.shape)
    log_softmax = Z - log_sum_exp
    
    # 交叉熵损失：-sum(y_one_hot * log_softmax)
    loss_per_sample = -ndl.summation(y_one_hot * log_softmax, axes=1)
    
    # 返回平均损失
    return ndl.summation(loss_per_sample) / Z.shape[0]
    ### END YOUR SOLUTION



def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    num_classes = W2.shape[1]

    for i in range(0, num_examples, batch):
        X_batch = ndl.Tensor(X[i : i + batch])
        y_batch = y[i : i + batch]

        y_one_hot = np.zeros((y_batch.shape[0], num_classes), dtype=np.float32)
        y_one_hot[np.arange(y_batch.shape[0]), y_batch] = 1
        y_batch_tensor = ndl.Tensor(y_one_hot)

        logits = ndl.relu(X_batch @ W1) @ W2
        loss = softmax_loss(logits, y_batch_tensor)
        loss.backward()

        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())

    return W1, W2
    ### END YOUR CODE


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
