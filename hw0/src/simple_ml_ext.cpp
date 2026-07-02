#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // 遍历每个批次
    for (size_t i = 0; i < m; i += batch) {
        // 当前批次的实际大小（最后一个批次可能小于batch）
        size_t batch_size = std::min(batch, m - i);
        
        // ========== 分配临时数组 ==========
        // logits: (batch_size, k)
        float *logits = new float[batch_size * k]();
        // probs: (batch_size, k) - softmax概率
        float *probs = new float[batch_size * k]();
        // I_y: (batch_size, k) - one-hot标签
        float *I_y = new float[batch_size * k]();
        // grad: (n, k) - 梯度
        float *grad = new float[n * k]();
        
        // ========== 前向传播：计算logits = X_batch @ theta ==========
        // 对于批次中的每个样本
        for (size_t j = 0; j < batch_size; j++) {
            // 对于每个类别
            for (size_t c = 0; c < k; c++) {
                float sum = 0.0f;
                // 对于每个特征维度
                for (size_t d = 0; d < n; d++) {
                    // X[i+j, d] * theta[d, c]
                    sum += X[(i + j) * n + d] * theta[d * k + c];
                }
                logits[j * k + c] = sum;
            }
        }
        
        // ========== 计算Softmax概率 ==========
        for (size_t j = 0; j < batch_size; j++) {
            // 找到当前样本的最大logit值（数值稳定）
            float max_logit = logits[j * k];
            for (size_t c = 1; c < k; c++) {
                if (logits[j * k + c] > max_logit) {
                    max_logit = logits[j * k + c];
                }
            }
            
            // 计算exp并求和
            float sum_exp = 0.0f;
            for (size_t c = 0; c < k; c++) {
                probs[j * k + c] = expf(logits[j * k + c] - max_logit);
                sum_exp += probs[j * k + c];
            }
            
            // 归一化得到概率
            for (size_t c = 0; c < k; c++) {
                probs[j * k + c] /= sum_exp;
            }
        }
        
        // ========== 创建One-Hot标签矩阵 ==========
        for (size_t j = 0; j < batch_size; j++) {
            // 获取当前样本的真实标签
            unsigned char label = y[i + j];
            // 设置对应位置为1
            I_y[j * k + label] = 1.0f;
        }
        
        // ========== 计算梯度 ==========
        // grad = (1/batch_size) * X_batch^T @ (probs - I_y)
        for (size_t d = 0; d < n; d++) {
            for (size_t c = 0; c < k; c++) {
                float sum = 0.0f;
                for (size_t j = 0; j < batch_size; j++) {
                    // X[i+j, d] * (probs[j, c] - I_y[j, c])
                    sum += X[(i + j) * n + d] * (probs[j * k + c] - I_y[j * k + c]);
                }
                grad[d * k + c] = sum / batch_size;
            }
        }
        
        // ========== 更新参数 ==========
        // theta = theta - lr * grad
        for (size_t j = 0; j < n * k; j++) {
            theta[j] -= lr * grad[j];
        }
        
        // ========== 释放临时数组 ==========
        delete[] logits;
        delete[] probs;
        delete[] I_y;
        delete[] grad;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
