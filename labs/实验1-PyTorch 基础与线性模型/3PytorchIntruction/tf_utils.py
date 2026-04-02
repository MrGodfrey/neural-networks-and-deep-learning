"""
pytorch_compat_utils.py（tf_utils.py）
---------------------------------------
原文件使用已废弃的 TensorFlow 1.x API（tf.placeholder / tf.Session）。
本模块已重写为纯 PyTorch 实现，对外接口保持不变，方便旧代码直接复用。
"""

import h5py
import math

import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------
# 数据加载（与 improv_utils.load_dataset 保持一致）
# -----------------------------------------------------------------------
def load_dataset():
    """
    从 HDF5 文件中读取手势数字数据集。

    返回:
    train_set_x_orig -- 训练集图像，形状 (1080, 64, 64, 3)
    train_set_y_orig -- 训练集标签，形状 (1, 1080)
    test_set_x_orig  -- 测试集图像，形状 (120, 64, 64, 3)
    test_set_y_orig  -- 测试集标签，形状 (1, 120)
    classes          -- 类别名称数组
    """
    train_file = h5py.File('datasets/train_signs.h5', 'r')
    train_x = np.array(train_file['train_set_x'][:])
    train_y = np.array(train_file['train_set_y'][:])

    test_file = h5py.File('datasets/test_signs.h5', 'r')
    test_x  = np.array(test_file['test_set_x'][:])
    test_y  = np.array(test_file['test_set_y'][:])
    classes = np.array(test_file['list_classes'][:])

    train_y = train_y.reshape(1, train_y.shape[0])
    test_y  = test_y.reshape(1,  test_y.shape[0])

    return train_x, train_y, test_x, test_y, classes


# -----------------------------------------------------------------------
# 随机小批量划分
# -----------------------------------------------------------------------
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    将数据 (X, Y) 随机打乱后划分为若干小批量。

    参数:
    X              -- 输入数据，形状 (特征数, 样本数)
    Y              -- 标签矩阵，形状 (1, 样本数)
    mini_batch_size -- 每个批量的样本数
    seed           -- 随机种子

    返回:
    mini_batches -- 列表，每个元素为 (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]
    np.random.seed(seed)

    perm = list(np.random.permutation(m))
    X_s = X[:, perm]
    Y_s = Y[:, perm].reshape(Y.shape[0], m)

    mini_batches = []
    n_complete = math.floor(m / mini_batch_size)

    for k in range(n_complete):
        s, e = k * mini_batch_size, (k + 1) * mini_batch_size
        mini_batches.append((X_s[:, s:e], Y_s[:, s:e]))

    if m % mini_batch_size != 0:
        s = n_complete * mini_batch_size
        mini_batches.append((X_s[:, s:], Y_s[:, s:]))

    return mini_batches


# -----------------------------------------------------------------------
# One-Hot 编码（NumPy 版）
# -----------------------------------------------------------------------
def convert_to_one_hot(Y, C):
    """
    将整数标签数组转换为 One-Hot 矩阵。

    参数:
    Y -- 标签数组（任意形状，会被展平）
    C -- 类别总数

    返回:
    形状 (C, N) 的 One-Hot 矩阵（numpy.ndarray）
    """
    return np.eye(C)[Y.reshape(-1)].T


# -----------------------------------------------------------------------
# 前向传播（PyTorch 实现，取代原 TF1.x 版本）
# -----------------------------------------------------------------------
def forward_propagation_for_predict(X, parameters):
    """
    实现三层全连接网络的前向传播：
    LINEAR → ReLU → LINEAR → ReLU → LINEAR

    参数:
    X          -- 输入张量，形状 (12288, N)
    parameters -- 字典，包含 W1, b1, W2, b2, W3, b3

    返回:
    Z3 -- 输出 logits，形状 (6, N)
    """
    def _t(v):
        if isinstance(v, torch.Tensor):
            return v.float()
        return torch.tensor(np.array(v), dtype=torch.float32)

    W1, b1 = _t(parameters['W1']), _t(parameters['b1'])
    W2, b2 = _t(parameters['W2']), _t(parameters['b2'])
    W3, b3 = _t(parameters['W3']), _t(parameters['b3'])

    X_t = _t(X)
    A1 = F.relu(W1 @ X_t + b1)
    A2 = F.relu(W2 @ A1  + b2)
    Z3 = W3 @ A2 + b3
    return Z3


# -----------------------------------------------------------------------
# 预测（取代原 TF1.x tf.Session 实现）
# -----------------------------------------------------------------------
def predict(X, parameters):
    """
    使用已训练的参数字典对输入 X 进行类别预测。

    参数:
    X          -- numpy 数组，形状 (12288, N) 或 (N, 12288)
    parameters -- 字典，包含 W1, b1, W2, b2, W3, b3

    返回:
    predictions -- 形状 (N,) 的 numpy 数组，每个元素为预测类别索引
    """
    X_arr = np.array(X)
    if X_arr.shape[0] != 12288:
        X_arr = X_arr.T

    X_t = torch.tensor(X_arr, dtype=torch.float32)

    with torch.no_grad():
        Z3 = forward_propagation_for_predict(X_t, parameters)
        preds = torch.argmax(Z3, dim=0)

    return preds.numpy()
