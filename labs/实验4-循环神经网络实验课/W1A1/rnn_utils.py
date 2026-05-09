import numpy as np


def softmax(x):
    """对输入向量或矩阵按列计算 softmax 概率分布。"""
    shifted = np.exp(x - np.max(x))
    return shifted / shifted.sum(axis=0)


def sigmoid(x):
    """逐元素 sigmoid 激活函数，将输入映射到 (0, 1) 区间。"""
    return 1.0 / (1.0 + np.exp(-x))


def initialize_adam(parameters):
    """
    为 Adam 优化器初始化一阶矩估计 v 和二阶矩估计 s。

    参数:
        parameters -- 字典，包含各层权重 W1..WL 和偏置 b1..bL

    返回:
        v -- 梯度的指数加权移动平均，初始化为与参数同形的零矩阵
        s -- 梯度平方的指数加权移动平均，初始化为与参数同形的零矩阵
    """
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t,
                                 learning_rate=0.01,
                                 beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    使用 Adam 算法更新网络参数。

    参数:
        parameters    -- 当前参数字典 {W1, b1, ..., WL, bL}
        grads         -- 梯度字典 {dW1, db1, ..., dWL, dbL}
        v             -- 一阶矩估计（由 initialize_adam 初始化）
        s             -- 二阶矩估计（由 initialize_adam 初始化）
        t             -- 当前迭代步数（用于偏差修正）
        learning_rate -- 学习率 α
        beta1         -- 一阶矩衰减系数
        beta2         -- 二阶矩衰减系数
        epsilon       -- 数值稳定项，防止除零

    返回:
        parameters -- 更新后的参数字典
        v          -- 更新后的一阶矩估计
        s          -- 更新后的二阶矩估计
    """
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L + 1):
        # 更新一阶矩
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]

        # 偏差修正（一阶矩）
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)

        # 更新二阶矩
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)

        # 偏差修正（二阶矩）
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)

        # 参数更新
        parameters["W" + str(l)] -= (
            learning_rate * v_corrected["dW" + str(l)]
            / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        )
        parameters["b" + str(l)] -= (
            learning_rate * v_corrected["db" + str(l)]
            / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)
        )

    return parameters, v, s
