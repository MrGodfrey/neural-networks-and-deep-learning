import numpy as np


def sigmoid(Z):
    """
    Sigmoid 激活函数。

    参数:
        Z -- 任意形状的 numpy 数组（线性输出）

    返回:
        A -- sigmoid(Z)，与 Z 形状相同
        cache -- 缓存 Z，供反向传播使用
    """
    A = 1.0 / (1.0 + np.exp(-Z))
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    """
    Sigmoid 激活函数的反向传播。

    参数:
        dA -- 激活值的上游梯度，任意形状
        cache -- 前向传播缓存的 Z

    返回:
        dZ -- 关于 Z 的梯度
    """
    Z = cache
    s = 1.0 / (1.0 + np.exp(-Z))
    dZ = dA * s * (1.0 - s)
    assert dZ.shape == Z.shape
    return dZ


def relu(Z):
    """
    ReLU 激活函数：f(z) = max(0, z)。

    参数:
        Z -- 任意形状的 numpy 数组（线性输出）

    返回:
        A -- relu(Z)，与 Z 形状相同
        cache -- 缓存 Z，供反向传播使用
    """
    A = np.maximum(0, Z)
    assert A.shape == Z.shape
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    ReLU 激活函数的反向传播。

    参数:
        dA -- 激活值的上游梯度，任意形状
        cache -- 前向传播缓存的 Z

    返回:
        dZ -- 关于 Z 的梯度（Z <= 0 处梯度置零）
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert dZ.shape == Z.shape
    return dZ


def tanh_activation(Z):
    """
    Tanh 激活函数（附加函数）。

    参数:
        Z -- 任意形状的 numpy 数组（线性输出）

    返回:
        A -- tanh(Z)，与 Z 形状相同
        cache -- 缓存 Z，供反向传播使用
    """
    A = np.tanh(Z)
    cache = Z
    return A, cache


def tanh_backward(dA, cache):
    """
    Tanh 激活函数的反向传播（附加函数）。
    导数公式：d/dZ tanh(Z) = 1 - tanh²(Z)

    参数:
        dA -- 激活值的上游梯度，任意形状
        cache -- 前向传播缓存的 Z

    返回:
        dZ -- 关于 Z 的梯度
    """
    Z = cache
    t = np.tanh(Z)
    dZ = dA * (1.0 - t ** 2)
    assert dZ.shape == Z.shape
    return dZ

