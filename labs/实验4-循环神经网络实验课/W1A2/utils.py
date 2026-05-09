import numpy as np


def softmax(x):
    """
    计算 softmax 函数。

    参数:
        x -- 输入向量或矩阵

    返回:
        softmax 归一化后的概率分布
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def smooth(loss, cur_loss):
    """
    对损失值进行指数平滑，以减少训练曲线的波动。

    参数:
        loss    -- 历史平滑损失
        cur_loss -- 当前步骤的原始损失

    返回:
        平滑后的新损失值
    """
    return loss * 0.999 + cur_loss * 0.001


def print_sample(sample_ix, ix_to_char):
    """
    将采样得到的字符索引列表转换为字符串并打印，首字母大写。

    参数:
        sample_ix  -- 字符索引列表
        ix_to_char -- 索引到字符的映射字典
    """
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]
    print('%s' % (txt,), end='')


def get_sample(sample_ix, ix_to_char):
    """
    将采样得到的字符索引列表转换为字符串，首字母大写后返回。

    参数:
        sample_ix  -- 字符索引列表
        ix_to_char -- 索引到字符的映射字典

    返回:
        首字母大写的字符串
    """
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]
    return txt


def get_initial_loss(vocab_size, seq_length):
    """
    计算均匀分布假设下的初始交叉熵损失，用于损失平滑的起始值。

    参数:
        vocab_size  -- 词表大小（字符种类数）
        seq_length  -- 序列长度（名字的平均字符数）

    返回:
        初始损失值（浮点数）
    """
    return -np.log(1.0 / vocab_size) * seq_length


def initialize_parameters(n_a, n_x, n_y):
    """
    用小随机数初始化 RNN 的权重矩阵和零偏置向量。

    参数:
        n_a -- 隐藏状态的维度
        n_x -- 输入（词表）的维度
        n_y -- 输出（词表）的维度

    返回:
        parameters -- 包含以下键的字典:
            Wax -- 输入到隐藏层的权重矩阵，形状 (n_a, n_x)
            Waa -- 隐藏层到隐藏层的权重矩阵，形状 (n_a, n_a)
            Wya -- 隐藏层到输出层的权重矩阵，形状 (n_y, n_a)
            b   -- 隐藏层偏置，形状 (n_a, 1)
            by  -- 输出层偏置，形状 (n_y, 1)
    """
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x) * 0.01
    Waa = np.random.randn(n_a, n_a) * 0.01
    Wya = np.random.randn(n_y, n_a) * 0.01
    b = np.zeros((n_a, 1))
    by = np.zeros((n_y, 1))
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    return parameters


def rnn_step_forward(parameters, a_prev, x):
    """
    执行 RNN 单步前向传播。

    参数:
        parameters -- 权重字典，含 Waa, Wax, Wya, by, b
        a_prev     -- 上一时刻隐藏状态，形状 (n_a, 1)
        x          -- 当前时刻输入，形状 (n_x, 1)

    返回:
        a_next -- 当前时刻隐藏状态，形状 (n_a, 1)
        p_t    -- 当前时刻输出概率分布，形状 (n_y, 1)
    """
    Waa = parameters['Waa']
    Wax = parameters['Wax']
    Wya = parameters['Wya']
    by  = parameters['by']
    b   = parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
    p_t = softmax(np.dot(Wya, a_next) + by)
    return a_next, p_t


def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    """
    执行 RNN 单步反向传播，累加梯度。

    参数:
        dy         -- 输出层的误差，形状 (n_y, 1)
        gradients  -- 累积梯度字典（原地修改）
        parameters -- 权重字典
        x          -- 当前时刻输入，形状 (n_x, 1)
        a          -- 当前时刻隐藏状态，形状 (n_a, 1)
        a_prev     -- 上一时刻隐藏状态，形状 (n_a, 1)

    返回:
        gradients -- 更新后的梯度字典
    """
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next']
    daraw = (1 - a * a) * da
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients


def update_parameters(parameters, gradients, lr):
    """
    使用梯度下降法更新 RNN 的所有参数。

    参数:
        parameters -- 当前参数字典
        gradients  -- 对应梯度字典
        lr         -- 学习率

    返回:
        parameters -- 更新后的参数字典
    """
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']   += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    return parameters


def rnn_forward(X, Y, a0, parameters, vocab_size=27):
    """
    在整个输入序列上执行 RNN 前向传播，并计算交叉熵损失。

    参数:
        X          -- 输入字符索引列表（第一个元素为 None，表示起始标记）
        Y          -- 目标字符索引列表
        a0         -- 初始隐藏状态，形状 (n_a, 1)
        parameters -- 权重字典
        vocab_size -- 词表大小，默认 27

    返回:
        loss  -- 序列的交叉熵损失
        cache -- 包含 (y_hat, a, x) 的元组，供反向传播使用
    """
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a0)
    loss = 0

    for t in range(len(X)):
        x[t] = np.zeros((vocab_size, 1))
        if X[t] is not None:
            x[t][X[t]] = 1
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t - 1], x[t])
        loss -= np.log(y_hat[t][Y[t], 0])

    cache = (y_hat, a, x)
    return loss, cache


def rnn_backward(X, Y, parameters, cache):
    """
    在整个序列上执行 RNN 反向传播（BPTT）。

    参数:
        X          -- 输入字符索引列表
        Y          -- 目标字符索引列表
        parameters -- 权重字典
        cache      -- rnn_forward 返回的缓存 (y_hat, a, x)

    返回:
        gradients -- 包含所有参数梯度的字典
        a         -- 所有时刻隐藏状态的字典
    """
    gradients = {}
    (y_hat, a, x) = cache
    Waa = parameters['Waa']
    Wax = parameters['Wax']
    Wya = parameters['Wya']
    by  = parameters['by']
    b   = parameters['b']

    gradients['dWax'] = np.zeros_like(Wax)
    gradients['dWaa'] = np.zeros_like(Waa)
    gradients['dWya'] = np.zeros_like(Wya)
    gradients['db']   = np.zeros_like(b)
    gradients['dby']  = np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])

    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t - 1])

    return gradients, a
