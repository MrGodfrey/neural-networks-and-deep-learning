import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io
import sklearn
import sklearn.datasets


def sigmoid(x):
    """
    计算 sigmoid 激活函数。

    参数：
    x -- 标量或任意形状的 numpy 数组

    返回：
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def relu(x):
    """
    计算 ReLU 激活函数。

    参数：
    x -- 标量或任意形状的 numpy 数组

    返回：
    s -- relu(x) = max(0, x)
    """
    s = np.maximum(0, x)
    return s


def load_params_and_grads(seed=1):
    """
    生成用于测试的随机参数和梯度。

    参数：
    seed -- 随机种子，保证可复现性

    返回：
    W1, b1, W2, b2, dW1, db1, dW2, db2 -- 参数与对应梯度
    """
    np.random.seed(seed)
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 3)
    b2 = np.random.randn(3, 1)

    dW1 = np.random.randn(2, 3)
    db1 = np.random.randn(2, 1)
    dW2 = np.random.randn(3, 3)
    db2 = np.random.randn(3, 1)

    return W1, b1, W2, b2, dW1, db1, dW2, db2


def initialize_parameters(layer_dims):
    """
    初始化神经网络各层的参数（He 初始化）。

    参数：
    layer_dims -- 列表，包含每层神经元数量，如 [2, 4, 1]

    返回：
    parameters -- 字典，包含 "W1", "b1", ..., "WL", "bL"：
                    Wl -- 权重矩阵，形状 (layer_dims[l], layer_dims[l-1])
                    bl -- 偏置向量，形状 (layer_dims[l], 1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # 网络层数

    for l in range(1, L):
        parameters['W' + str(l)] = (np.random.randn(layer_dims[l], layer_dims[l - 1])
                                    * np.sqrt(2 / layer_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert parameters['W' + str(l)].shape[0] == layer_dims[l], layer_dims[l - 1]
        assert parameters['W' + str(l)].shape[0] == layer_dims[l], 1

    return parameters


def compute_cost(a3, Y):
    """
    计算二分类交叉熵损失（累积值，未除以样本数）。

    参数：
    a3 -- 前向传播的输出（预测概率），形状与 Y 相同
    Y  -- 真实标签向量

    返回：
    cost_total -- 损失值之和（用于小批量场景下的逐批累积）

    注意：
    在小批量训练中，先对整个 epoch 的损失求和，
    最后再除以训练样本总数 m。
    """
    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost_total = np.sum(logprobs)
    return cost_total


def forward_propagation(X, parameters):
    """
    实现三层神经网络的前向传播：
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

    参数：
    X          -- 输入数据，形状 (输入维度, 样本数)
    parameters -- 字典，包含 "W1","b1","W2","b2","W3","b3"

    返回：
    a3    -- 输出层激活值（预测概率）
    cache -- 各层中间变量，用于反向传播
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    return a3, cache


def backward_propagation(X, Y, cache):
    """
    实现三层神经网络的反向传播。

    参数：
    X     -- 输入数据，形状 (输入维度, 样本数)
    Y     -- 真实标签，形状 (1, 样本数)
    cache -- forward_propagation() 返回的中间变量

    返回：
    gradients -- 字典，包含各参数和激活值的梯度
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = 1.0 / m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    gradients = {
        "dz3": dz3, "dW3": dW3, "db3": db3,
        "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
        "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1,
    }
    return gradients


def predict(X, y, parameters):
    """
    使用训练好的参数对数据集进行预测，并打印准确率。

    参数：
    X          -- 输入数据，形状 (特征数, 样本数)
    y          -- 真实标签，形状 (1, 样本数)
    parameters -- 训练好的模型参数

    返回：
    p -- 预测结果（0 或 1），形状 (1, 样本数)
    """
    m = X.shape[1]
    p = np.zeros((1, m), dtype=int)

    a3, caches = forward_propagation(X, parameters)

    for i in range(a3.shape[1]):
        if a3[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("准确率：" + str(np.mean((p[0, :] == y[0, :]))))
    return p


def load_2D_dataset():
    """
    从 .mat 文件加载二维数据集，并绘制散点图。

    返回：
    train_X, train_Y -- 训练集特征与标签
    test_X,  test_Y  -- 测试集特征与标签
    """
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral)
    return train_X, train_Y, test_X, test_Y


def plot_decision_boundary(model, X, y):
    """
    绘制模型的决策边界。

    参数：
    model -- 接受输入矩阵并返回预测值的函数
    X     -- 输入数据，形状 (2, 样本数)
    y     -- 真实标签
    """
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


def predict_dec(parameters, X):
    """
    用于绘制决策边界时的预测函数。

    参数：
    parameters -- 模型参数字典
    X          -- 输入数据，形状 (特征数, 样本数)

    返回：
    predictions -- 预测结果（True/False）
    """
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3 > 0.5)
    return predictions


def load_dataset():
    """
    生成二分类模拟数据集（月牙形，模拟叶片特征分布），并绘制散点图。

    返回：
    train_X -- 特征矩阵，形状 (2, 样本数)
    train_Y -- 标签矩阵，形状 (1, 样本数)，0=健康，1=患病
    """
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=0.2)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y
