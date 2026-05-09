import math
import numpy as np
import h5py
import matplotlib.pyplot as plt


def load_dataset():
    """从 HDF5 文件加载手语数字数据集（0-5）。"""
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """将数据集划分为随机小批量。

    参数：
    X -- 输入数据，形状为 (m, H, W, C)
    Y -- 标签向量，形状为 (m, n_y)
    mini_batch_size -- 每个小批量的大小
    seed -- 随机种子，用于保证结果可复现

    返回：
    mini_batches -- 包含 (mini_batch_X, mini_batch_Y) 元组的列表
    """
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    # 步骤 1：打乱 (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # 步骤 2：按 mini_batch_size 划分（不含末尾不完整批量）
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    # 处理末尾不完整的批量
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches


def convert_to_one_hot(Y, C):
    """将整数标签转换为 one-hot 编码。"""
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
