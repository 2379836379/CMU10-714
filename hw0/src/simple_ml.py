import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x+y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
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
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
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


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    z_y = Z[np.arange(len(y)), y]
    
    # 计算 log-sum-exp: log(sum(exp(Z_i))) 对每个样本
    # axis=1 表示对每个样本的类别维度求和
    log_sum_exp = np.log(np.sum(np.exp(Z), axis=1))
    # 计算每个样本的损失: log_sum_exp - z_y
    # 然后取平均值
    loss = np.mean(log_sum_exp - z_y)
    return loss
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """

    ### BEGIN YOUR CODE
    m = X.shape[0]  # 样本数量
    # 按顺序遍历批次（不随机打乱）
    for i in range(0, m, batch):
        # 获取当前批次的样本
        X_batch = X[i:i+batch]  # (batch_size, input_dim)
        y_batch = y[i:i+batch]  # (batch_size,)
    
        # 前向传播：计算logits
        logits = X_batch @ theta  # (batch_size, num_classes)
        # 计算softmax概率（数值稳定版本）
        # 减去每个样本的最大值防止溢出
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        Z = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (batch_size, num_classes)
        # 创建one-hot标签矩阵
        batch_size = len(y_batch)
        I_y = np.zeros((batch_size, Z.shape[1]), dtype=np.float32)
        I_y[np.arange(batch_size), y_batch] = 1.0
        # 计算梯度：grad = (1/batch) * X_batch^T @ (Z - I_y)
        grad = (X_batch.T @ (Z - I_y)) / batch_size  # (input_dim, num_classes)
        
        # 更新参数（梯度下降）
        theta -= lr * grad
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    m = X.shape[0]  # 样本数量
    for i in range(0, m, batch):
        # 获取当前批次
        X_batch = X[i:i+batch]  # (batch_size, input_dim)
        y_batch = y[i:i+batch]  # (batch_size,)
        batch_size = len(y_batch)
        
        # ========== 前向传播 ==========
        # 第一层：Z1 = ReLU(X W1)
        Z1 = np.maximum(0, X_batch @ W1)  # (batch_size, hidden_dim)
        # 第二层：logits = Z1 W2
        logits = Z1 @ W2  # (batch_size, num_classes)
        
        # ========== Softmax概率 ==========
        # 数值稳定：减去最大值
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (batch_size, num_classes)
        
        # ========== 反向传播 ==========
        # One-hot标签矩阵
        I_y = np.zeros((batch_size, probs.shape[1]), dtype=np.float32)
        I_y[np.arange(batch_size), y_batch] = 1.0
        
        # G2 = probs - I_y (输出层误差)
        G2 = probs - I_y  # (batch_size, num_classes)
        
        # W2的梯度: dW2 = (1/batch) * Z1^T @ G2
        dW2 = (Z1.T @ G2) / batch_size  # (hidden_dim, num_classes)
        
        # G1 = ReLU导数 * (G2 @ W2^T) (隐藏层误差)
        relu_grad = (Z1 > 0).astype(np.float32)  # (batch_size, hidden_dim)
        G1 = relu_grad * (G2 @ W2.T)  # (batch_size, hidden_dim)
        
        # W1的梯度: dW1 = (1/batch) * X^T @ G1
        dW1 = (X_batch.T @ G1) / batch_size  # (input_dim, hidden_dim)
        # ========== 参数更新 ==========
        W1 -= lr * dW1
        W2 -= lr * dW2
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
