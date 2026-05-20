import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

def sigmoid(x):
    """
    计算 sigmoid 激活函数。

    参数：
    x -- 任意形状的标量或 numpy 数组

    返回：
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s

def relu(x):
    """
    计算 ReLU 激活函数。

    参数：
    x -- 任意形状的标量或 numpy 数组

    返回：
    s -- relu(x)
    """
    s = np.maximum(0, x)
    return s

def forward_propagation(X, parameters):
    """
    实现三层神经网络的前向传播：LINEAR->ReLU->LINEAR->ReLU->LINEAR->Sigmoid。

    参数：
    X -- 输入数据，形状 (输入维度, 样本数)
    parameters -- 参数字典，包含 "W1","b1","W2","b2","W3","b3"

    返回：
    a3 -- 最终输出激活值
    cache -- 中间结果缓存，用于反向传播
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
    X -- 输入数据，形状 (输入维度, 样本数)
    Y -- 标签向量，形状 (1, 样本数)
    cache -- forward_propagation 返回的缓存

    返回：
    gradients -- 包含各参数梯度的字典
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = 1. / m * (a3 - Y)
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

    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    return gradients

def update_parameters(parameters, grads, learning_rate):
    """
    使用梯度下降更新参数。

    参数：
    parameters -- 包含参数的字典
    grads -- 包含梯度的字典
    learning_rate -- 学习率（标量）

    返回：
    parameters -- 更新后的参数字典
    """
    L = len(parameters) // 2
    for k in range(L):
        parameters["W" + str(k + 1)] -= learning_rate * grads["dW" + str(k + 1)]
        parameters["b" + str(k + 1)] -= learning_rate * grads["db" + str(k + 1)]
    return parameters

def compute_loss(a3, Y):
    """
    计算二分类交叉熵损失。

    参数：
    a3 -- 前向传播输出，与 Y 形状相同
    Y -- 标签向量，与 a3 形状相同

    返回：
    loss -- 损失值（标量）
    """
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    loss = 1. / m * np.nansum(logprobs)
    return loss

def predict(X, y, parameters):
    """
    使用训练好的参数对输入数据进行预测。

    参数：
    X -- 输入数据
    y -- 真实标签
    parameters -- 训练好的参数字典

    返回：
    p -- 预测结果（0 或 1）
    """
    m = X.shape[1]
    p = np.zeros((1, m), dtype=int)

    a3, _ = forward_propagation(X, parameters)
    for i in range(m):
        p[0, i] = 1 if a3[0, i] > 0.5 else 0

    print("准确率：" + str(np.mean(p[0, :] == y[0, :])))
    return p

def plot_decision_boundary(model, X, y):
    """绘制决策边界。"""
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
    用于绘制决策边界的辅助函数。

    参数：
    parameters -- 参数字典
    X -- 输入数据，形状 (m, K)

    返回：
    predictions -- 预测结果（0/1 布尔数组）
    """
    a3, _ = forward_propagation(X, parameters)
    predictions = (a3 > 0.5)
    return predictions

def load_dataset():
    """
    生成二维月牙形分类数据集（训练集和测试集）。

    返回：
    train_X, train_Y, test_X, test_Y -- 训练集与测试集的特征和标签
    """
    # noise=0.15 使数据集具有适度重叠，便于验证不同初始化方法的分类效果
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=0.15)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_moons(n_samples=100, noise=0.15)

    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    plt.title("二维月牙形数据集")
    plt.show()

    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y
