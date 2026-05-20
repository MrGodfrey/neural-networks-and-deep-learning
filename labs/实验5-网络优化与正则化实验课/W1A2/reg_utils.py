import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
import scipy.io


def sigmoid(x):
    """
    计算 sigmoid 函数。

    参数：
    x -- 标量或任意大小的 numpy 数组

    返回：
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def relu(x):
    """
    计算 ReLU 激活函数。

    参数：
    x -- 标量或任意大小的 numpy 数组

    返回：
    s -- relu(x)，即 max(0, x)
    """
    s = np.maximum(0, x)
    return s


def load_planar_dataset(seed):
    """
    生成平面螺旋形数据集（单参数版本，用于可视化）。

    参数：
    seed -- 随机种子，保证可复现性

    返回：
    X -- 形状为 (2, m) 的特征矩阵
    Y -- 形状为 (1, m) 的标签向量（0 或 1）
    """
    np.random.seed(seed)

    m = 400          # 样本总数
    N = int(m / 2)   # 每类样本数
    D = 2            # 特征维度
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    max_radius = 4   # 螺旋最大半径

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = max_radius * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def initialize_parameters(layer_dims):
    """
    初始化神经网络参数。

    参数：
    layer_dims -- 包含每层维度的列表，例如 [2, 20, 3, 1]

    返回：
    parameters -- 包含 "W1","b1",...,"WL","bL" 的字典：
                    Wl 的形状为 (layer_dims[l], layer_dims[l-1])
                    bl 的形状为 (layer_dims[l], 1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1])
        assert parameters['b' + str(l)].shape == (layer_dims[l], 1)

    return parameters


def forward_propagation(X, parameters):
    """
    实现三层神经网络的前向传播：LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID。

    参数：
    X          -- 输入数据，形状为 (输入维度, 样本数)
    parameters -- 包含 W1,b1,W2,b2,W3,b3 的字典

    返回：
    A3    -- 最后一层的激活输出（预测概率）
    cache -- 包含中间变量的元组，供反向传播使用
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


def backward_propagation(X, Y, cache):
    """
    实现三层神经网络的反向传播（无正则化）。

    参数：
    X     -- 输入数据，形状为 (输入维度, 样本数)
    Y     -- 真实标签向量，形状为 (1, 样本数)
    cache -- forward_propagation() 返回的缓存元组

    返回：
    gradients -- 包含各参数梯度的字典
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def update_parameters(parameters, grads, learning_rate):
    """
    使用梯度下降法更新参数。

    参数：
    parameters    -- 包含当前参数的字典
    grads         -- 包含各参数梯度的字典
    learning_rate -- 学习率（标量）

    返回：
    parameters -- 更新后的参数字典
    """
    n = len(parameters) // 2

    for k in range(n):
        parameters["W" + str(k + 1)] -= learning_rate * grads["dW" + str(k + 1)]
        parameters["b" + str(k + 1)] -= learning_rate * grads["db" + str(k + 1)]

    return parameters


def predict(X, y, parameters):
    """
    使用训练好的模型对数据集进行预测。

    参数：
    X          -- 待预测的数据集，形状为 (特征数, 样本数)
    y          -- 真实标签，形状为 (1, 样本数)
    parameters -- 训练好的模型参数

    返回：
    p -- 预测结果（0 或 1）
    """
    m = X.shape[1]
    p = np.zeros((1, m), dtype=int)

    a3, caches = forward_propagation(X, parameters)

    for i in range(a3.shape[1]):
        p[0, i] = 1 if a3[0, i] > 0.5 else 0

    print("准确率: " + str(np.mean(p[0, :] == y[0, :])))

    return p


def compute_cost(a3, Y):
    """
    计算交叉熵损失函数（不含正则化）。

    参数：
    a3 -- 前向传播输出的预测概率，形状与 Y 相同
    Y  -- 真实标签向量

    返回：
    cost -- 标量损失值
    """
    m = Y.shape[1]

    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost = 1. / m * np.nansum(logprobs)

    return cost


def predict_dec(parameters, X):
    """
    用于绘制决策边界的辅助预测函数。

    参数：
    parameters -- 模型参数字典
    X          -- 输入数据，形状为 (特征数, 样本数)

    返回：
    predictions -- 模型预测结果（布尔数组，True 表示正类）
    """
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3 > 0.5)
    return predictions


def load_planar_dataset(randomness, seed):
    """
    生成带有随机噪声的平面螺旋形数据集（双参数版本）。

    参数：
    randomness -- 控制噪声强度的标量
    seed       -- 随机种子

    返回：
    X -- 形状为 (2, m) 的特征矩阵
    Y -- 形状为 (1, m) 的标签向量（0 或 1）
    """
    np.random.seed(seed)

    m = 50
    N = int(m / 2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        if j == 0:
            t = np.linspace(j, 4 * 3.1415 * (j + 1), N)
            r = 0.3 * np.square(t) + np.random.randn(N) * randomness
        if j == 1:
            t = np.linspace(j, 2 * 3.1415 * (j + 1), N)
            r = 0.2 * np.square(t) + np.random.randn(N) * randomness

        X[ix] = np.c_[r * np.cos(t), r * np.sin(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def plot_decision_boundary(model, X, y):
    """绘制分类模型的决策边界。"""
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


def load_2D_dataset():
    """
    从 datasets/data.mat 加载二维数据集并可视化训练集散点图。

    返回：
    train_X -- 训练集特征，形状为 (2, 训练样本数)
    train_Y -- 训练集标签，形状为 (1, 训练样本数)
    test_X  -- 测试集特征，形状为 (2, 测试样本数)
    test_Y  -- 测试集标签，形状为 (1, 测试样本数)
    """
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral)

    return train_X, train_Y, test_X, test_Y
