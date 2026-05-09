import numpy as np


def linear_forward_test_case():
    """生成线性前向传播的测试数据（随机种子=1）。"""
    np.random.seed(1)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    return A, W, b


def linear_activation_forward_test_case():
    """生成线性-激活前向传播的测试数据（随机种子=2）。"""
    np.random.seed(2)
    A_prev = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    return A_prev, W, b


def L_model_forward_test_case():
    """生成两层网络前向传播的测试数据（随机种子=1）。"""
    np.random.seed(1)
    X = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return X, parameters


def compute_cost_test_case():
    """生成损失函数计算的测试数据。"""
    Y = np.asarray([[1, 1, 0]])
    aL = np.array([[.8, .9, 0.4]])
    return Y, aL


def linear_backward_test_case():
    """生成线性反向传播的测试数据（随机种子=1）。"""
    np.random.seed(1)
    dZ = np.random.randn(3, 4)
    A = np.random.randn(5, 4)
    W = np.random.randn(3, 5)
    b = np.random.randn(3, 1)
    linear_cache = (A, W, b)
    return dZ, linear_cache


def linear_activation_backward_test_case():
    """生成线性-激活反向传播的测试数据（随机种子=2）。"""
    np.random.seed(2)
    dA = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    Z = np.random.randn(1, 2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)
    return dA, linear_activation_cache


def L_model_backward_test_case():
    """生成 L 层网络反向传播的测试数据（随机种子=3）。"""
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    Z1 = np.random.randn(3, 2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3, 2)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    Z2 = np.random.randn(1, 2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)
    return AL, Y, caches


def update_parameters_test_case():
    """生成参数更新的测试数据（随机种子=2/3）。"""
    np.random.seed(2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    np.random.seed(3)
    dW1 = np.random.randn(3, 4)
    db1 = np.random.randn(3, 1)
    dW2 = np.random.randn(1, 3)
    db2 = np.random.randn(1, 1)
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return parameters, grads


def L_model_forward_test_case_2hidden():
    """生成含两个隐藏层的 L 层网络前向传播测试数据（随机种子=6）。"""
    np.random.seed(6)
    X = np.random.randn(5, 4)
    W1 = np.random.randn(4, 5)
    b1 = np.random.randn(4, 1)
    W2 = np.random.randn(3, 4)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    return X, parameters


def print_grads(grads):
    """打印梯度字典中的 dW1、db1、dA1。"""
    print("dW1 = " + str(grads["dW1"]))
    print("db1 = " + str(grads["db1"]))
    print("dA1 = " + str(grads["dA1"]))
