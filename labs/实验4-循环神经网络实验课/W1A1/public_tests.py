import numpy as np
from rnn_utils import softmax, sigmoid


def rnn_cell_forward_tests(target):
    """验证单步 RNN 单元前向传播的正确性。"""

    # 测试 1：仅偏置有效（权重矩阵设为零）
    a_prev = np.zeros((5, 10))
    xt = np.zeros((3, 10))
    params = {
        'Waa': np.random.randn(5, 5),
        'Wax': np.random.randn(5, 3),
        'Wya': np.zeros((2, 5)),   # 设为零以检查学生是否误用 a_prev
        'ba':  np.random.randn(5, 1),
        'by':  np.random.randn(2, 1),
    }

    a_next, yt_pred, cache = target(xt, a_prev, params)

    assert a_next.shape  == (5, 10), f"a_next 形状错误，期望 (5,10)，实际 {a_next.shape}"
    assert yt_pred.shape == (2, 10), f"yt_pred 形状错误，期望 (2,10)，实际 {yt_pred.shape}"
    assert cache[0].shape == (5, 10), "cache 中 a_next 形状错误"
    assert cache[1].shape == (5, 10), "cache 中 a_prev 形状错误"
    assert cache[2].shape == (3, 10), "cache 中 xt 形状错误"
    assert len(cache[3].keys()) == 5, "cache 中参数数量应为 5"

    assert np.allclose(np.tanh(params['ba']), a_next), "测试1：a_next 中 ba 相关计算有误"
    assert np.allclose(softmax(params['by']), yt_pred), "测试1：yt_pred 中 by 相关计算有误"

    # 测试 2：仅输入 xt 有效
    a_prev = np.zeros((5, 10))
    xt = np.random.randn(3, 10)
    params['Wax'] = np.random.randn(5, 3)
    params['Wya'] = np.random.randn(2, 5)
    params['ba']  = np.zeros((5, 1))
    params['by']  = np.zeros((2, 1))

    a_next, yt_pred, cache = target(xt, a_prev, params)

    assert np.allclose(np.tanh(np.dot(params['Wax'], xt)), a_next), \
        "测试2：a_next 中 xt 相关计算有误"
    assert np.allclose(softmax(np.dot(params['Wya'], a_next)), yt_pred), \
        "测试2：yt_pred 中 a_next 相关计算有误"

    # 测试 3：仅历史隐状态 a_prev 有效
    a_prev = np.random.randn(5, 10)
    xt = np.zeros((3, 10))
    params['Waa'] = np.random.randn(5, 5)
    params['ba']  = np.zeros((5, 1))
    params['by']  = np.zeros((2, 1))

    a_next, yt_pred, cache = target(xt, a_prev, params)

    assert np.allclose(np.tanh(np.dot(params['Waa'], a_prev)), a_next), \
        "测试3：a_next 中 a_prev 相关计算有误"
    assert np.allclose(softmax(np.dot(params['Wya'], a_next)), yt_pred), \
        "测试3：yt_pred 中 a_next 相关计算有误"

    print("\033[92mAll tests passed")


def rnn_forward_test(target):
    """验证完整 RNN 前向传播在多个时间步上的正确性。"""
    np.random.seed(17)
    T_x, m, n_x, n_a, n_y = 13, 8, 4, 7, 3

    x = np.random.randn(n_x, m, T_x)
    a0 = np.random.randn(n_a, m)
    params = {
        'Waa': np.random.randn(n_a, n_a),
        'Wax': np.random.randn(n_a, n_x),
        'Wya': np.random.randn(n_y, n_a),
        'ba':  np.random.randn(n_a, 1),
        'by':  np.random.randn(n_y, 1),
    }

    a, y_pred, caches = target(x, a0, params)

    assert a.shape     == (n_a, m, T_x), f"a 形状错误，期望 {(n_a, m, T_x)}，实际 {a.shape}"
    assert y_pred.shape == (n_y, m, T_x), f"y_pred 形状错误，期望 {(n_y, m, T_x)}，实际 {y_pred.shape}"
    assert len(caches[0]) == T_x, f"caches[0] 长度应等于 T_x={T_x}"

    assert np.allclose(a[5, 2, 2:6], [0.99999291, 0.99332189, 0.9921928, 0.99503445]), \
        "a 的数值结果不正确"
    assert np.allclose(y_pred[2, 1, 1:5], [0.19428, 0.14292, 0.24993, 0.00119], atol=1e-4), \
        "y_pred 的数值结果不正确"
    assert np.allclose(caches[1], x), "caches[1] 应等于输入 x"

    print("\033[92mAll tests passed")


def lstm_cell_forward_test(target):
    """验证单步 LSTM 单元前向传播的正确性（门控值、状态及输出形状与数值）。"""
    np.random.seed(212)
    m, n_x, n_a, n_y = 8, 4, 7, 3

    x  = np.random.randn(n_x, m)
    a0 = np.random.randn(n_a, m)
    c0 = np.random.randn(n_a, m)
    params = {
        'Wf': np.random.randn(n_a, n_a + n_x),
        'bf': np.random.randn(n_a, 1),
        'Wi': np.random.randn(n_a, n_a + n_x),
        'bi': np.random.randn(n_a, 1),
        'Wo': np.random.randn(n_a, n_a + n_x),
        'bo': np.random.randn(n_a, 1),
        'Wc': np.random.randn(n_a, n_a + n_x),
        'bc': np.random.randn(n_a, 1),
        'Wy': np.random.randn(n_y, n_a),
        'by': np.random.randn(n_y, 1),
    }

    a_next, c_next, y_pred, cache = target(x, a0, c0, params)

    assert len(cache) == 10, "cache 长度不应改变（应为 10）"

    # 形状检查
    shape_checks = [
        (cache[4], (n_a, m), "cache[4](ft)"),
        (cache[5], (n_a, m), "cache[5](it)"),
        (cache[6], (n_a, m), "cache[6](cct)"),
        (cache[1], (n_a, m), "cache[1](c_next)"),
        (cache[7], (n_a, m), "cache[7](ot)"),
        (cache[0], (n_a, m), "cache[0](a_next)"),
        (cache[8], (n_x, m), "cache[8](xt)"),
        (cache[2], (n_a, m), "cache[2](a_prev)"),
        (cache[3], (n_a, m), "cache[3](c_prev)"),
    ]
    for arr, expected, name in shape_checks:
        assert arr.shape == expected, f"{name} 形状错误：{arr.shape} != {expected}"

    assert a_next.shape  == (n_a, m), f"a_next 形状错误：{a_next.shape}"
    assert c_next.shape  == (n_a, m), f"c_next 形状错误：{c_next.shape}"
    assert y_pred.shape  == (n_y, m), f"y_pred 形状错误：{y_pred.shape}"

    # 数值检查
    assert np.allclose(cache[4][0, 0:2], [0.32969833, 0.0574555]),  "ft 数值有误"
    assert np.allclose(cache[5][0, 0:2], [0.0036446,  0.9806943]),  "it 数值有误"
    assert np.allclose(cache[6][0, 0:2], [0.99903873, 0.57509956]), "cct 数值有误"
    assert np.allclose(cache[1][0, 0:2], [0.1352798,  0.39884899]), "c_next 数值有误"
    assert np.allclose(cache[7][0, 0:2], [0.7477249,  0.71588751]), "ot 数值有误"
    assert np.allclose(cache[0][0, 0:2], [0.10053951, 0.27129536]), "a_next 数值有误"
    assert np.allclose(y_pred[1], [0.417098, 0.449528, 0.223159, 0.278376,
                                    0.68453,  0.419221, 0.564025, 0.538475]), "y_pred 数值有误"

    print("\033[92mAll tests passed")


def lstm_forward_test(target):
    """验证完整 LSTM 前向传播在多时间步上的形状与数值正确性。"""
    np.random.seed(45)
    n_x, m, T_x, n_a, n_y = 4, 13, 16, 3, 2

    x  = np.random.randn(n_x, m, T_x)
    a0 = np.random.randn(n_a, m)
    params = {
        'Wf': np.random.randn(n_a, n_a + n_x),
        'bf': np.random.randn(n_a, 1),
        'Wi': np.random.randn(n_a, n_a + n_x),
        'bi': np.random.randn(n_a, 1),
        'Wo': np.random.randn(n_a, n_a + n_x),
        'bo': np.random.randn(n_a, 1),
        'Wc': np.random.randn(n_a, n_a + n_x),
        'bc': np.random.randn(n_a, 1),
        'Wy': np.random.randn(n_y, n_a),
        'by': np.random.randn(n_y, 1),
    }

    a, y, c, caches = target(x, a0, params)

    assert a.shape == (n_a, m, T_x), f"a 形状错误：{a.shape} != {(n_a, m, T_x)}"
    assert c.shape == (n_a, m, T_x), f"c 形状错误：{c.shape} != {(n_a, m, T_x)}"
    assert y.shape == (n_y, m, T_x), f"y 形状错误：{y.shape} != {(n_y, m, T_x)}"
    assert len(caches[0])    == T_x, f"caches[0] 长度应为 {T_x}"
    assert len(caches[0][0]) == 10,  "caches[0][0] 长度应为 10"

    assert np.allclose(a[2, 1, 4:6], [-0.01606022,  0.0243569]),  "a 数值有误"
    assert np.allclose(c[2, 1, 4:6], [-0.02753855,  0.05668358]), "c 数值有误"
    assert np.allclose(y[1, 1, 4:6], [0.70444592,   0.70648935]), "y 数值有误"

    print("\033[92mAll tests passed")
