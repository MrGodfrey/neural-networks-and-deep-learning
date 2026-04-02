"""
pytorch_utils.py（improv_utils.py）
------------------------------------
手势数字分类实验的 PyTorch 辅助工具模块。
提供数据加载、小批量生成、预测、参数初始化及模型训练等功能。
"""

import h5py
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# 数据加载
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
    将训练数据 (X, Y) 随机打乱后划分为若干小批量。

    参数:
    X              -- 输入数据，形状 (特征数, 样本数)
    Y              -- 标签向量，形状 (1, 样本数)
    mini_batch_size -- 每个小批量的样本数
    seed           -- 随机种子（用于可重复性）

    返回:
    mini_batches -- 列表，每个元素为 (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]
    np.random.seed(seed)

    # 随机打乱列索引
    perm = list(np.random.permutation(m))
    X_shuffled = X[:, perm]
    Y_shuffled = Y[:, perm].reshape(Y.shape[0], m)

    mini_batches = []
    n_complete = math.floor(m / mini_batch_size)

    for k in range(n_complete):
        start, end = k * mini_batch_size, (k + 1) * mini_batch_size
        mini_batches.append((X_shuffled[:, start:end],
                             Y_shuffled[:, start:end]))

    # 处理最后不足一个批量的余量
    if m % mini_batch_size != 0:
        start = n_complete * mini_batch_size
        mini_batches.append((X_shuffled[:, start:],
                             Y_shuffled[:, start:]))

    return mini_batches


# -----------------------------------------------------------------------
# One-Hot 编码转换（NumPy 版）
# -----------------------------------------------------------------------
def convert_to_one_hot(Y, C):
    """
    将整数标签向量转换为 One-Hot 矩阵（NumPy 实现）。

    参数:
    Y -- 标签数组，形状任意（会被展平）
    C -- 类别数

    返回:
    one_hot -- 形状 (C, N) 的 One-Hot 矩阵
    """
    return np.eye(C)[Y.reshape(-1)].T


# -----------------------------------------------------------------------
# 预测
# -----------------------------------------------------------------------
def predict(X, parameters):
    """
    使用已训练的参数字典对输入 X 进行类别预测。

    参数:
    X          -- numpy 数组，形状 (12288, N) 或 (N, 12288)；
                  若第一维不为 12288，则自动转置。
    parameters -- 字典，键为 W1, b1, W2, b2, W3, b3（nn.Parameter 或普通张量）

    返回:
    predictions -- 形状 (N,) 的 numpy 数组，每个元素为预测类别索引
    """
    def _to_t(v):
        if isinstance(v, torch.Tensor):
            return v.detach().float()
        return torch.tensor(np.array(v), dtype=torch.float32)

    p = {k: _to_t(v) for k, v in parameters.items()}
    X_t = torch.tensor(np.array(X), dtype=torch.float32)
    if X_t.shape[0] != 12288:
        X_t = X_t.T

    with torch.no_grad():
        A1 = F.relu(p['W1'] @ X_t + p['b1'])
        A2 = F.relu(p['W2'] @ A1  + p['b2'])
        Z3 = p['W3'] @ A2  + p['b3']        # (6, N)
        preds = torch.argmax(Z3, dim=0)      # (N,)

    return preds.numpy()


# -----------------------------------------------------------------------
# 参数初始化
# -----------------------------------------------------------------------
def initialize_parameters():
    """
    使用 Xavier 正态分布初始化三层全连接网络的权重参数。

    网络结构:
        输入层 12288  →  隐藏层 25  →  隐藏层 12  →  输出层 6

    返回:
    parameters -- 字典，包含键 W1, b1, W2, b2, W3, b3（均为 nn.Parameter）
    """
    torch.manual_seed(1)

    W1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(25, 12288)))
    b1 = nn.Parameter(torch.zeros(25, 1))
    W2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(12, 25)))
    b2 = nn.Parameter(torch.zeros(12, 1))
    W3 = nn.Parameter(nn.init.xavier_normal_(torch.empty(6, 12)))
    b3 = nn.Parameter(torch.zeros(6, 1))

    return {"W1": W1, "b1": b1,
            "W2": W2, "b2": b2,
            "W3": W3, "b3": b3}


# -----------------------------------------------------------------------
# 损失计算（平均交叉熵）
# -----------------------------------------------------------------------
def compute_cost(Z3, Y):
    """
    计算多分类交叉熵损失（平均值）。

    参数:
    Z3 -- 形状 (6, N) 的 logits 张量（前向传播的最后输出）
    Y  -- 形状 (N,) 的 long 类型标签张量

    返回:
    cost -- 标量 torch.Tensor
    """
    return F.cross_entropy(Z3.T, Y, reduction='mean')


# -----------------------------------------------------------------------
# 完整模型训练
# -----------------------------------------------------------------------
def model(X_train, Y_train, X_test, Y_test,
          learning_rate=0.0001, num_epochs=1500,
          minibatch_size=32, print_cost=True):
    """
    训练三层全连接神经网络：LINEAR → ReLU → LINEAR → ReLU → LINEAR → Softmax。

    参数:
    X_train        -- (1080, 12288) 的 float32 张量或 numpy 数组
    Y_train        -- (1080,) 的 long 类型类别索引
    X_test         -- (120, 12288) 的 float32 张量或 numpy 数组
    Y_test         -- (120,) 的 long 类型类别索引
    learning_rate  -- Adam 优化器的学习率
    num_epochs     -- 训练轮数
    minibatch_size -- 小批量大小
    print_cost     -- 若为 True，则每 100 轮打印一次损失

    返回:
    parameters -- 训练后的参数字典
    """
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, TensorDataset

    def _to_t(arr, dtype):
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype)
        return torch.tensor(np.array(arr), dtype=dtype)

    X_train = _to_t(X_train, torch.float32)
    Y_train = _to_t(Y_train, torch.long)
    X_test  = _to_t(X_test,  torch.float32)
    Y_test  = _to_t(Y_test,  torch.long)

    costs = []
    parameters = initialize_parameters()
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    W3, b3 = parameters['W3'], parameters['b3']

    optimizer = torch.optim.Adam([W1, b1, W2, b2, W3, b3], lr=learning_rate)

    train_loader = DataLoader(TensorDataset(X_train, Y_train),
                              batch_size=minibatch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test,  Y_test),
                              batch_size=minibatch_size)

    for epoch in range(num_epochs):
        epoch_cost = 0.0
        n_batches  = 0

        for mb_X, mb_Y in train_loader:
            optimizer.zero_grad()
            # 前向传播（权重矩阵要求输入形状为 (特征数, 样本数)）
            A1 = F.relu(W1 @ mb_X.T + b1)
            A2 = F.relu(W2 @ A1     + b2)
            Z3 = W3 @ A2 + b3                          # (6, N)
            loss = F.cross_entropy(Z3.T, mb_Y, reduction='mean')
            loss.backward()
            optimizer.step()
            epoch_cost += loss.item()
            n_batches  += 1

        epoch_cost /= n_batches

        if print_cost and epoch % 100 == 0:
            print(f"第 {epoch} 轮损失：{epoch_cost:.6f}")
        if print_cost and epoch % 5 == 0:
            costs.append(epoch_cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('损失')
    plt.xlabel('迭代次数（每 5 轮）')
    plt.title(f"学习率 = {learning_rate}")
    plt.show()

    # 计算最终准确率
    with torch.no_grad():
        def _acc(loader):
            correct, total = 0, 0
            for mb_X, mb_Y in loader:
                A1 = F.relu(W1 @ mb_X.T + b1)
                A2 = F.relu(W2 @ A1     + b2)
                Z3 = W3 @ A2 + b3
                correct += (torch.argmax(Z3, dim=0) == mb_Y).sum().item()
                total   += mb_Y.shape[0]
            return correct / total

        tr_acc = _acc(DataLoader(TensorDataset(X_train, Y_train),
                                 batch_size=minibatch_size))
        te_acc = _acc(test_loader)

    print("参数训练完成！")
    print(f"训练集准确率：{tr_acc:.4f}")
    print(f"测试集准确率：{te_acc:.4f}")

    return parameters
