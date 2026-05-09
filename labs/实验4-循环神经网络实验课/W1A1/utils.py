import numpy as np


def softmax(x):
    """对向量 x 计算 softmax 归一化概率。"""
    shifted = np.exp(x - np.max(x))
    return shifted / shifted.sum(axis=0)


def smooth(loss, cur_loss):
    """对损失值进行指数平滑，用于可视化训练曲线。"""
    return loss * 0.999 + cur_loss * 0.001


def print_sample(sample_ix, ix_to_char):
    """将字符索引序列转换为字符串并打印。"""
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt,))


def get_initial_loss(vocab_size, seq_length):
    """计算均匀分布假设下的初始交叉熵损失（基准值）。"""
    return -np.log(1.0 / vocab_size) * seq_length


def initialize_parameters(n_a, n_x, n_y):
    """
    以小随机值初始化字符级 RNN 的参数。

    参数:
        n_a -- 隐藏状态维度
        n_x -- 输入（词汇表）维度
        n_y -- 输出（词汇表）维度

    返回:
        parameters -- 字典，包含 Wax, Waa, Wya, b, by
    """
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x) * 0.01
    Waa = np.random.randn(n_a, n_a) * 0.01
    Wya = np.random.randn(n_y, n_a) * 0.01
    b   = np.zeros((n_a, 1))
    by  = np.zeros((n_y, 1))
    return {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}


def rnn_step_forward(parameters, a_prev, x):
    """
    执行单步 RNN 前向传播。

    返回:
        a_next -- 新隐藏状态
        p_t    -- 下一字符的概率分布
    """
    Waa = parameters['Waa']
    Wax = parameters['Wax']
    Wya = parameters['Wya']
    by  = parameters['by']
    b   = parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
    p_t    = softmax(np.dot(Wya, a_next) + by)
    return a_next, p_t


def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    """执行单步 RNN 反向传播，累加梯度到 gradients 字典。"""
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby']  += dy
    da     = np.dot(parameters['Wya'].T, dy) + gradients['da_next']
    daraw  = (1 - a * a) * da  # tanh 反向
    gradients['db']   += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients


def update_parameters(parameters, gradients, lr):
    """按梯度下降规则更新参数。"""
    parameters['Wax'] -= lr * gradients['dWax']
    parameters['Waa'] -= lr * gradients['dWaa']
    parameters['Wya'] -= lr * gradients['dWya']
    parameters['b']   -= lr * gradients['db']
    parameters['by']  -= lr * gradients['dby']
    return parameters


def rnn_forward(X, Y, a0, parameters, vocab_size=71):
    """
    对整个序列执行 RNN 前向传播并计算交叉熵损失。

    参数:
        X          -- 输入字符索引列表
        Y          -- 目标字符索引列表
        a0         -- 初始隐藏状态
        parameters -- 模型参数字典
        vocab_size -- 词汇表大小

    返回:
        loss  -- 序列交叉熵损失
        cache -- (y_hat, a, x) 供反向传播使用
    """
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a0)
    loss  = 0

    for t in range(len(X)):
        x[t] = np.zeros((vocab_size, 1))
        x[t][X[t]] = 1
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t - 1], x[t])
        loss -= np.log(y_hat[t][Y[t], 0])

    return loss, (y_hat, a, x)


def rnn_backward(X, Y, parameters, cache):
    """
    对整个序列执行 RNN 反向传播（时间反向传播）。

    返回:
        gradients -- 各参数梯度字典
        a         -- 各时间步隐藏状态字典
    """
    gradients = {}
    y_hat, a, x = cache
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
