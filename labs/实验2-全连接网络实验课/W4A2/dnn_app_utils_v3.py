import numpy as np
import matplotlib.pyplot as plt
import h5py


def sigmoid(Z):
    """
    Sigmoid 激活函数。

    参数:
    Z -- 任意形状的 numpy 数组

    返回:
    A -- sigmoid(Z)，与 Z 形状相同
    cache -- Z 本身，反向传播时使用
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    """
    ReLU 激活函数：max(0, Z)。

    参数:
    Z -- 线性层的输出，任意形状

    返回:
    A -- 激活后的输出，与 Z 形状相同
    cache -- Z 本身，反向传播时使用
    """
    A = np.maximum(0, Z)
    assert A.shape == Z.shape
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    ReLU 的反向传播。

    参数:
    dA -- 激活后的梯度，任意形状
    cache -- 前向传播存储的 Z

    返回:
    dZ -- 关于 Z 的梯度
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert dZ.shape == Z.shape
    return dZ


def sigmoid_backward(dA, cache):
    """
    Sigmoid 的反向传播。

    参数:
    dA -- 激活后的梯度，任意形状
    cache -- 前向传播存储的 Z

    返回:
    dZ -- 关于 Z 的梯度
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert dZ.shape == Z.shape
    return dZ


def load_data():
    """从 H5 文件加载猫咪数据集，返回训练集、测试集及类别标签。"""
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters(n_x, n_h, n_y):
    """
    初始化两层网络的参数（W 随机小值，b 为零）。

    参数:
    n_x -- 输入层维度
    n_h -- 隐藏层维度
    n_y -- 输出层维度

    返回:
    parameters -- 包含 W1, b1, W2, b2 的字典
    """
    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert W1.shape == (n_h, n_x)
    assert b1.shape == (n_h, 1)
    assert W2.shape == (n_y, n_h)
    assert b2.shape == (n_y, 1)

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


def initialize_parameters_deep(layer_dims):
    """
    初始化 L 层网络的参数，使用 He 初始化缩放。

    参数:
    layer_dims -- 各层维度列表，如 [n_x, n_1, ..., n_L]

    返回:
    parameters -- 包含 W1,b1,...,WL,bL 的字典
    """
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = (
            np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        )
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1])
        assert parameters['b' + str(l)].shape == (layer_dims[l], 1)

    return parameters


def linear_forward(A, W, b):
    """
    线性前向传播：Z = W·A + b。

    参数:
    A -- 上一层激活值，形状 (上一层大小, 样本数)
    W -- 权重矩阵，形状 (当前层大小, 上一层大小)
    b -- 偏置向量，形状 (当前层大小, 1)

    返回:
    Z -- 预激活值
    cache -- (A, W, b)，反向传播时使用
    """
    Z = W.dot(A) + b
    assert Z.shape == (W.shape[0], A.shape[1])
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    线性 + 激活的前向传播。

    参数:
    A_prev -- 上一层激活值
    W -- 权重矩阵
    b -- 偏置向量
    activation -- 激活函数类型："sigmoid" 或 "relu"

    返回:
    A -- 激活后的输出
    cache -- (linear_cache, activation_cache)
    """
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    else:
        raise ValueError(f'未知激活函数: {activation}，请传入 "sigmoid" 或 "relu"')

    assert A.shape == (W.shape[0], A_prev.shape[1])
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    """
    L 层网络前向传播：[线性->ReLU]*(L-1) -> 线性->Sigmoid。

    参数:
    X -- 输入数据，形状 (输入维度, 样本数)
    parameters -- initialize_parameters_deep() 的输出

    返回:
    AL -- 最终激活输出（预测概率）
    caches -- 所有层的 cache 列表
    """
    caches = []
    A = X
    L = len(parameters) // 2

    # 前 L-1 层：线性 -> ReLU
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu"
        )
        caches.append(cache)

    # 最后一层：线性 -> Sigmoid
    AL, cache = linear_activation_forward(
        A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid"
    )
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])
    return AL, caches


def compute_cost(AL, Y):
    """
    计算交叉熵损失。

    参数:
    AL -- 预测概率向量，形状 (1, 样本数)
    Y -- 真实标签，形状 (1, 样本数)

    返回:
    cost -- 标量损失值
    """
    m = Y.shape[1]
    cost = -(1.0 / m) * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)
    assert cost.shape == ()
    return cost


def linear_backward(dZ, cache):
    """
    单层线性部分的反向传播。

    参数:
    dZ -- 关于线性输出的梯度
    cache -- 前向传播存储的 (A_prev, W, b)

    返回:
    dA_prev -- 关于上一层激活的梯度
    dW -- 关于 W 的梯度
    db -- 关于 b 的梯度
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1.0 / m) * np.dot(dZ, A_prev.T)
    db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    线性 + 激活层的反向传播。

    参数:
    dA -- 当前层激活后的梯度
    cache -- (linear_cache, activation_cache)
    activation -- "sigmoid" 或 "relu"

    返回:
    dA_prev, dW, db
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        raise ValueError(f'未知激活函数: {activation}')

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    L 层网络反向传播：[线性->ReLU]*(L-1) -> 线性->Sigmoid。

    参数:
    AL -- 前向传播的最终输出
    Y -- 真实标签
    caches -- L_model_forward() 返回的 caches 列表

    返回:
    grads -- 梯度字典，包含 dA, dW, db
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    # 输出层梯度初始化
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # 最后一层（Sigmoid）
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = (
        linear_activation_backward(dAL, current_cache, activation="sigmoid")
    )

    # 其余层（ReLU），从后往前
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, activation="relu"
        )
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    梯度下降更新参数。

    参数:
    parameters -- 当前参数字典
    grads -- L_model_backward() 返回的梯度字典
    learning_rate -- 学习率

    返回:
    parameters -- 更新后的参数字典
    """
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

    return parameters


def predict(X, y, parameters):
    """
    用训练好的 L 层网络对输入数据进行预测。

    参数:
    X -- 输入数据，形状 (特征数, 样本数)
    y -- 真实标签，形状 (1, 样本数)
    parameters -- 训练好的参数字典

    返回:
    p -- 预测结果（0 或 1），形状 (1, 样本数)
    """
    m = X.shape[1]

    # 前向传播得到概率
    probas, _ = L_model_forward(X, parameters)

    # 向量化：概率 > 0.5 预测为 1，否则为 0
    p = (probas > 0.5).astype(int)

    print("准确率: " + str(np.mean(p == y)))
    return p


def plot_costs(costs, learning_rate=0.0075):
    """
    绘制训练损失曲线。

    参数:
    costs -- 每 100 次迭代记录的损失列表
    learning_rate -- 学习率（显示在图标题中）
    """
    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per 100)')
    plt.title(f'Learning rate = {learning_rate}')
    plt.show()


def print_mislabeled_images(classes, X, y, p):
    """
    绘制被错误分类的图像。

    参数:
    classes -- 类别标签数组
    X -- 图像数据，形状 (特征数, 样本数)
    y -- 真实标签
    p -- 预测标签
    """
    # p + y == 1 时表示预测与真实不一致（一个为0一个为1）
    mislabeled_indices = np.asarray(np.where((p + y) == 1))
    num_images = len(mislabeled_indices[0])

    plt.rcParams['figure.figsize'] = (40.0, 40.0)
    fig = plt.figure()
    fig.suptitle("Mislabeled Images", fontsize=20, fontweight='bold')

    for i in range(num_images):
        index = mislabeled_indices[1][i]
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        pred_label = classes[int(p[0, index])].decode("utf-8")
        true_label = classes[y[0, index]].decode("utf-8")
        plt.title(f"Predicted: {pred_label}\nActual: {true_label}")
