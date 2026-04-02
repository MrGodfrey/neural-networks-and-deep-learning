import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_utils import single_test, multiple_test


# -----------------------------------------------------------------------
# 练习 1 测试：linear_function
# -----------------------------------------------------------------------
def linear_function_test(target):
    """验证 linear_function 的返回类型与形状。"""
    result = target()
    test_cases = [
        {
            "name": "datatype_check",
            "input": [],
            "expected": torch.tensor([0.0]),
            "error": "返回值应为 torch.Tensor"
        },
        {
            "name": "shape_check",
            "input": [],
            "expected": torch.zeros(4, 1),
            "error": "返回张量的形状应为 (4, 1)"
        },
    ]
    # 手动执行（返回值已经计算过了，用 single_test 的 target 包装一下）
    success = 0
    for tc in test_cases:
        try:
            if tc["name"] == "datatype_check":
                assert isinstance(result, torch.Tensor)
                success += 1
            elif tc["name"] == "shape_check":
                assert result.shape == tc["expected"].shape
                success += 1
        except AssertionError:
            print("错误：" + tc["error"])

    if success == len(test_cases):
        print("\033[92m linear_function 全部测试通过。")
    else:
        raise AssertionError("linear_function 未通过全部测试，请检查实现。")


# -----------------------------------------------------------------------
# 练习 2 测试：sigmoid
# -----------------------------------------------------------------------
def sigmoid_test(target):
    """验证 sigmoid 的返回类型、dtype 以及数值正确性。"""
    test_inputs = [0.0, -1.0, 2.0]
    expected = [0.5, 0.2689414, 0.8807970]

    success = 0
    for z, exp in zip(test_inputs, expected):
        result = target(z)
        try:
            assert isinstance(result, torch.Tensor), "返回值应为 torch.Tensor"
            assert result.dtype == torch.float32, "dtype 应为 torch.float32"
            assert abs(result.item() - exp) < 1e-5, f"sigmoid({z}) 数值错误"
            success += 1
        except AssertionError as e:
            print("错误：", e)

    if success == len(test_inputs):
        print("\033[92m sigmoid 全部测试通过。")
    else:
        raise AssertionError("sigmoid 未通过全部测试，请检查实现。")


# -----------------------------------------------------------------------
# 练习 3 测试：one_hot_matrix
# -----------------------------------------------------------------------
def one_hot_matrix_test(target):
    """验证 one_hot_matrix 的输出形状与数值。"""
    # 情形1：标量整数标签
    result1 = target(2, 5)
    expected1 = torch.tensor([0., 0., 1., 0., 0.])

    # 情形2：列表形式标签
    result2 = target([0], 4)
    expected2 = torch.tensor([1., 0., 0., 0.])

    success = 0
    for result, expected, desc in [
        (result1, expected1, "标量标签 2，C=5"),
        (result2, expected2, "列表标签 [0]，C=4"),
    ]:
        try:
            assert isinstance(result, torch.Tensor), f"{desc}：返回值应为 torch.Tensor"
            assert result.shape == expected.shape, f"{desc}：形状不匹配"
            assert torch.allclose(result, expected), f"{desc}：数值不匹配"
            success += 1
        except AssertionError as e:
            print("错误：", e)

    if success == 2:
        print("\033[92m one_hot_matrix 全部测试通过。")
    else:
        raise AssertionError("one_hot_matrix 未通过全部测试，请检查实现。")


# -----------------------------------------------------------------------
# 练习 4 测试：initialize_parameters
# -----------------------------------------------------------------------
def initialize_parameters_test(target):
    """验证参数字典的键、形状与类型。"""
    params = target()

    expected_shapes = {
        "W1": (25, 12288), "b1": (25, 1),
        "W2": (12, 25),    "b2": (12, 1),
        "W3": (6, 12),     "b3": (6, 1),
    }

    success = 0
    for key, shape in expected_shapes.items():
        try:
            assert key in params, f"参数字典缺少键 '{key}'"
            assert isinstance(params[key], nn.Parameter), \
                f"'{key}' 应为 nn.Parameter"
            assert tuple(params[key].shape) == shape, \
                f"'{key}' 形状应为 {shape}，实际为 {tuple(params[key].shape)}"
            success += 1
        except AssertionError as e:
            print("错误：", e)

    if success == len(expected_shapes):
        print("\033[92m initialize_parameters 全部测试通过。")
    else:
        raise AssertionError("initialize_parameters 未通过全部测试，请检查实现。")


# -----------------------------------------------------------------------
# 练习 5 测试：forward_propagation
# -----------------------------------------------------------------------
def forward_propagation_test(target, params, X_sample):
    """验证前向传播的输出形状以及梯度可传播性。"""
    X = torch.tensor(X_sample[:3].T, dtype=torch.float32)  # (12288, 3)
    Z3 = target(X, params)

    success = 0
    try:
        assert isinstance(Z3, torch.Tensor), "Z3 应为 torch.Tensor"
        success += 1
    except AssertionError as e:
        print("错误：", e)

    try:
        assert Z3.shape == (6, 3), f"Z3 形状应为 (6, 3)，实际为 {Z3.shape}"
        success += 1
    except AssertionError as e:
        print("错误：", e)

    # 验证梯度可以反向传播
    try:
        loss = Z3.sum()
        loss.backward()
        assert params['W1'].grad is not None, "W1 梯度为 None，检查是否使用了 nn.Parameter"
        success += 1
    except Exception as e:
        print("梯度错误：", e)

    if success == 3:
        print("\033[92m forward_propagation 全部测试通过。")
    else:
        raise AssertionError("forward_propagation 未通过全部测试，请检查实现。")


# -----------------------------------------------------------------------
# 练习 6 测试：compute_total_loss
# -----------------------------------------------------------------------
def compute_total_loss_test(target):
    """验证交叉熵损失函数的数值与类型。"""
    torch.manual_seed(42)
    # logits: (6, 2)，labels: (2,)
    logits = torch.tensor(
        [[2.4048, 5.0334],
         [-0.4588, -0.5373],
         [ 0.4031, -0.5984],
         [-0.1519, -0.5378],
         [-0.7914, -1.0213],
         [-0.3568,  0.2692]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)

    result = target(logits, labels)

    success = 0
    try:
        assert isinstance(result, torch.Tensor), "返回值应为 torch.Tensor"
        success += 1
    except AssertionError as e:
        print("错误：", e)

    try:
        assert result.shape == torch.Size([]), "结果应为标量张量"
        success += 1
    except AssertionError as e:
        print("错误：", e)

    # 验证数值：两个样本的交叉熵之和
    ref = F.cross_entropy(logits.T, labels, reduction='sum')
    try:
        assert torch.isclose(result, ref, atol=1e-4), \
            f"损失值不匹配：期望 {ref.item():.4f}，实际 {result.item():.4f}"
        success += 1
    except AssertionError as e:
        print("错误：", e)

    if success == 3:
        print("\033[92m compute_total_loss 全部测试通过。")
    else:
        raise AssertionError("compute_total_loss 未通过全部测试，请检查实现。")
