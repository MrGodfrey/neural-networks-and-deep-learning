import numpy as np
from outputs import *


def zero_pad_test(target):
    """测试 zero_pad 函数的正确性。"""
    # 测试 1
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = target(x, 3)
    print("x.shape =\n", x.shape)
    print("x_pad.shape =\n", x_pad.shape)
    print("x[1,1] =\n", x[1, 1])
    print("x_pad[1,1] =\n", x_pad[1, 1])

    assert type(x_pad) == np.ndarray, "输出必须是 numpy 数组"
    assert x_pad.shape == (4, 9, 9, 2), f"形状错误：{x_pad.shape} != (4, 9, 9, 2)"
    print(x_pad[0, 0:2, :, 0])
    assert np.allclose(x_pad[0, 0:2, :, 0], [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], 1e-15), "行未用零填充"
    assert np.allclose(x_pad[0, :, 7:9, 1].transpose(), [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], 1e-15), "列未用零填充"
    assert np.allclose(x_pad[:, 3:6, 3:6, :], x, 1e-15), "内部数值不一致"

    # 测试 2
    np.random.seed(1)
    x = np.random.randn(5, 4, 4, 3)
    pad = 2
    x_pad = target(x, pad)

    assert type(x_pad) == np.ndarray, "输出必须是 numpy 数组"
    assert x_pad.shape == (5, 4 + 2 * pad, 4 + 2 * pad, 3), f"形状错误：{x_pad.shape} != {(5, 4 + 2 * pad, 4 + 2 * pad, 3)}"
    assert np.allclose(x_pad[0, 0:2, :, 0], [[0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0]], 1e-15), "行未用零填充"
    assert np.allclose(x_pad[0, :, 6:8, 1].transpose(), [[0, 0, 0, 0, 0, 0, 0, 0],
                                                           [0, 0, 0, 0, 0, 0, 0, 0]], 1e-15), "列未用零填充"
    assert np.allclose(x_pad[:, 2:6, 2:6, :], x, 1e-15), "内部数值不一致"

    print("\033[92m所有测试通过！")


def conv_single_step_test(target):
    """测试 conv_single_step 函数的正确性。"""
    np.random.seed(3)
    a_slice_prev = np.random.randn(5, 5, 3)
    W = np.random.randn(5, 5, 3)
    b = np.random.randn(1, 1, 1)

    Z = target(a_slice_prev, W, b)
    expected_output = np.float64(-3.5443670581382474)

    assert (type(Z) == np.float64 or type(Z) == np.float32), "输出必须转换为浮点型"
    assert np.isclose(Z, expected_output), f"数值错误。期望值：{expected_output}，实际值：{Z}"

    print("\033[92m所有测试通过！")


def conv_forward_test_1(z_mean, z_0_2_1, cache_0_1_2_3):
    """测试 conv_forward 第一组输出。"""
    test_count = 0
    z_mean_expected = 0.5511276474566768
    z_0_2_1_expected = [-2.17796037, 8.07171329, -0.5772704, 3.36286738, 4.48113645, -2.89198428, 10.99288867, 3.03171932]
    cache_0_1_2_3_expected = [-1.1191154, 1.9560789, -0.3264995, -1.34267579]

    if np.isclose(z_mean, z_mean_expected):
        test_count += 1
    else:
        print("\033[91m测试1：Z 的均值不正确。期望：", z_mean_expected, "\n实际输出：", z_mean, "，请确认步幅计算是否正确\033[90m\n")

    if np.allclose(z_0_2_1, z_0_2_1_expected):
        test_count += 1
    else:
        print("\033[91m测试1：Z[0,2,1] 不正确。期望：", z_0_2_1_expected, "\n实际输出：", z_0_2_1, "，请确认步幅计算是否正确\033[90m\n")

    if np.allclose(cache_0_1_2_3, cache_0_1_2_3_expected):
        test_count += 1
    else:
        print("\033[91m测试1：cache_conv[0][1][2][3] 不正确。期望：", cache_0_1_2_3_expected, "\n实际输出：", cache_0_1_2_3, "\033[90m")

    if test_count == 3:
        print("\033[92m测试1：所有测试通过！")


def conv_forward_test_2(target):
    """测试 conv_forward 第二组输出。"""
    # 测试 1
    np.random.seed(3)
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)

    Z, cache_conv = target(A_prev, W, b, {"pad": 3, "stride": 1})
    Z_shape = Z.shape
    assert Z_shape[0] == A_prev.shape[0], f"m 错误。当前：{Z_shape[0]}，期望：{A_prev.shape[0]}"
    assert Z_shape[1] == 9, f"n_H 错误。当前：{Z_shape[1]}，期望：9"
    assert Z_shape[2] == 11, f"n_W 错误。当前：{Z_shape[2]}，期望：11"
    assert Z_shape[3] == W.shape[3], f"n_C 错误。当前：{Z_shape[3]}，期望：{W.shape[3]}"

    # 测试 2
    Z, cache_conv = target(A_prev, W, b, {"pad": 0, "stride": 2})
    assert (Z.shape == (2, 2, 3, 8)), "形状错误，不要在函数中硬编码 pad 和 stride"

    # 测试 3
    W = np.random.randn(5, 5, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    Z, cache_conv = target(A_prev, W, b, {"pad": 6, "stride": 1})
    Z_shape = Z.shape
    assert Z_shape[0] == A_prev.shape[0], f"m 错误。当前：{Z_shape[0]}，期望：{A_prev.shape[0]}"
    assert Z_shape[1] == 13, f"n_H 错误。当前：{Z_shape[1]}，期望：13"
    assert Z_shape[2] == 15, f"n_W 错误。当前：{Z_shape[2]}，期望：15"
    assert Z_shape[3] == W.shape[3], f"n_C 错误。当前：{Z_shape[3]}，期望：{W.shape[3]}"

    Z_means = np.mean(Z)
    expected_Z = -0.5384027772160062

    expected_conv = np.array([[1.98848968, 1.19505834, -0.0952376, -0.52718778],
                               [-0.32158469, 0.15113037, -0.01862772, 0.48352879],
                               [0.76896516, 1.36624284, 1.14726479, -0.11022916],
                               [0.38825041, -0.38712718, -0.58722031, 1.91082685],
                               [-0.45984615, 1.99073781, -0.34903539, 0.25282509],
                               [1.08940955, 0.02392202, 0.39312528, -0.2413848],
                               [-0.47552486, -0.16577702, -0.64971742, 1.63138295]])

    assert np.isclose(Z_means, expected_Z), f"Z 均值错误。期望：{expected_Z}，实际：{Z_means}"
    assert np.allclose(cache_conv[0][1, 2], expected_conv), "Z 中的数值错误"

    print("\033[92m测试2：所有测试通过！")


def pool_forward_test_1(target):
    """测试 pool_forward 第一组输出。"""
    # 测试 1
    A_prev = np.random.randn(2, 5, 7, 3)
    A, cache = target(A_prev, {"stride": 2, "f": 2}, mode="average")
    A_shape = A.shape
    assert A_shape[0] == A_prev.shape[0], f"测试1 - m 错误。当前：{A_shape[0]}，期望：{A_prev.shape[0]}"
    assert A_shape[1] == 2, f"测试1 - n_H 错误。当前：{A_shape[1]}，期望：2"
    assert A_shape[2] == 3, f"测试1 - n_W 错误。当前：{A_shape[2]}，期望：3"
    assert A_shape[3] == A_prev.shape[3], f"测试1 - n_C 错误。当前：{A_shape[3]}，期望：{A_prev.shape[3]}"

    # 测试 2
    A_prev = np.random.randn(4, 5, 7, 4)
    A, cache = target(A_prev, {"stride": 1, "f": 5}, mode="max")
    A_shape = A.shape
    assert A_shape[0] == A_prev.shape[0], f"测试2 - m 错误。当前：{A_shape[0]}，期望：{A_prev.shape[0]}"
    assert A_shape[1] == 1, f"测试2 - n_H 错误。当前：{A_shape[1]}，期望：1"
    assert A_shape[2] == 3, f"测试2 - n_W 错误。当前：{A_shape[2]}，期望：3"
    assert A_shape[3] == A_prev.shape[3], f"测试2 - n_C 错误。当前：{A_shape[3]}，期望：{A_prev.shape[3]}"

    # 测试 3
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 5, 3)

    A, cache = target(A_prev, {"stride": 1, "f": 2}, mode="max")

    assert np.allclose(A[1, 1], np.array([[1.19891788, 0.74055645, 0.07734007],
                                           [0.31515939, 0.84616065, 0.07734007],
                                           [0.69803203, 0.84616065, 1.2245077],
                                           [0.69803203, 1.12141771, 1.2245077]])), "A[1, 1] 的值错误"

    assert np.allclose(cache[0][1, 2], np.array([[0.16938243, 0.74055645, -0.9537006],
                                                   [-0.26621851, 0.03261455, -1.37311732],
                                                   [0.31515939, 0.84616065, -0.85951594],
                                                   [0.35054598, -1.31228341, -0.03869551],
                                                   [-1.61577235, 1.12141771, 0.40890054]])), "cache 的值错误"

    A, cache = target(A_prev, {"stride": 1, "f": 2}, mode="average")

    assert np.allclose(A[1, 1], np.array([[0.11583785, 0.34545544, -0.6561907],
                                           [-0.2334108, 0.3364666, -0.69382351],
                                           [0.25497093, -0.21741362, -0.07342615],
                                           [-0.04092568, -0.01110394, 0.12495022]])), "A[1, 1] 的值错误"

    print("\033[92m所有测试通过！")


def pool_forward_test_2(target):
    """测试 pool_forward 第二组输出（含步幅验证）。"""
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 5, 3)

    A, cache = target(A_prev, {"stride": 2, "f": 3}, mode="max")

    assert np.allclose(A[0], np.array([[[1.74481176, 0.90159072, 1.65980218],
                                         [1.74481176, 1.6924546, 1.65980218]],
                                        [[1.13162939, 1.51981682, 2.18557541],
                                         [1.13162939, 1.6924546, 2.18557541]]])), "最大池化 A[0] 值错误，请确认步幅计算"

    A, cache = target(A_prev, {"stride": 2, "f": 3}, mode="average")

    assert np.allclose(A[1], np.array([[[-0.17313416, 0.32377198, -0.34317572],
                                         [0.02030094, 0.14141479, -0.01231585]],
                                        [[0.42944926, 0.08446996, -0.27290905],
                                         [0.15077452, 0.28911175, 0.00123239]]])), "平均池化 A[1] 值错误，请确认步幅计算"

    print("\033[92m所有测试通过！")
