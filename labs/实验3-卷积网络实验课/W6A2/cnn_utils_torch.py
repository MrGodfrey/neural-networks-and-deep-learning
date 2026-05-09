import numpy as np
import h5py


def load_happy_dataset():
    """
    从 HDF5 文件中加载情绪分类（Happy House）数据集。

    返回值：
        X_train -- 训练集图像，形状 (m_train, H, W, C)，像素值未归一化
        Y_train -- 训练集标签，形状 (1, m_train)，取值 0 或 1
        X_test  -- 测试集图像，形状 (m_test, H, W, C)
        Y_test  -- 测试集标签，形状 (1, m_test)
        classes -- 类别名称数组
    """
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_signs_dataset():
    """
    从 HDF5 文件中加载手语数字（SIGNS）数据集。

    返回值：
        X_train -- 训练集图像，形状 (m_train, H, W, C)，像素值未归一化
        Y_train -- 训练集标签，形状 (1, m_train)，取值 0~5
        X_test  -- 测试集图像，形状 (m_test, H, W, C)
        Y_test  -- 测试集标签，形状 (1, m_test)
        classes -- 类别名称数组
    """
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


def convert_to_one_hot(Y, C):
    """
    将整数类别标签转换为 one-hot 编码。

    参数：
        Y -- 标签数组，形状任意（将被展平）
        C -- 类别总数

    返回值：
        形状为 (C, m) 的 one-hot 矩阵
    """
    return np.eye(C)[Y.reshape(-1)].T
