import numpy as np

def sigmoid(x):
    """
    计算 sigmoid 函数值。

    参数:
    x -- 标量或任意大小的 numpy 数组

    返回:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s

def relu(x):
    """
    计算 ReLU 函数值。

    参数:
    x -- 标量或任意大小的 numpy 数组

    返回:
    s -- relu(x)
    """
    s = np.maximum(0, x)

    return s

def dictionary_to_vector(parameters):
    """
    将参数字典展平并拼接为一个列向量（满足特定形状要求）。
    展平顺序为: W1, b1, W2, b2, W3, b3。
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:

        # 将参数展平为列向量
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(theta):
    """
    将列向量还原为参数字典（满足特定形状要求）。
    还原顺序为: W1, b1, W2, b2, W3, b3。
    """
    parameters = {}
    parameters["W1"] = theta[: 20].reshape((5, 4))
    parameters["b1"] = theta[20: 25].reshape((5, 1))
    parameters["W2"] = theta[25: 40].reshape((3, 5))
    parameters["b2"] = theta[40: 43].reshape((3, 1))
    parameters["W3"] = theta[43: 46].reshape((1, 3))
    parameters["b3"] = theta[46: 47].reshape((1, 1))

    return parameters

def gradients_to_vector(gradients):
    """
    将梯度字典展平并拼接为一个列向量（满足特定形状要求）。
    展平顺序为: dW1, db1, dW2, db2, dW3, db3。
    """

    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # 将梯度展平为列向量
        new_vector = np.reshape(gradients[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta