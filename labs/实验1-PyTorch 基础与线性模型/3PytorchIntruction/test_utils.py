import numpy as np


# -----------------------------------------------------------------------
# 辅助函数：递归检查数据类型
# -----------------------------------------------------------------------
def datatype_check(expected_output, target_output, error):
    """递归比较 expected_output 与 target_output 的类型是否一致。"""
    success = 0
    if isinstance(target_output, dict):
        for key in target_output.keys():
            try:
                success += datatype_check(expected_output[key],
                                          target_output[key], error)
            except Exception:
                print("类型错误：{} — 变量 {}，期望类型 {} 但得到 {}".format(
                    error, key,
                    type(expected_output[key]), type(target_output[key])))
        return 1 if success == len(target_output) else 0

    elif isinstance(target_output, (tuple, list)):
        for i in range(len(target_output)):
            try:
                success += datatype_check(expected_output[i],
                                          target_output[i], error)
            except Exception:
                print("类型错误：{} — 位置 {}，期望类型 {} 但得到 {}".format(
                    error, i,
                    type(expected_output[i]), type(target_output[i])))
        return 1 if success == len(target_output) else 0

    else:
        assert isinstance(target_output, type(expected_output))
        return 1


# -----------------------------------------------------------------------
# 辅助函数：递归检查数值是否相等
# -----------------------------------------------------------------------
def equation_output_check(expected_output, target_output, error):
    """递归比较 expected_output 与 target_output 的数值是否几乎相等。"""
    success = 0
    if isinstance(target_output, dict):
        for key in target_output.keys():
            try:
                success += equation_output_check(expected_output[key],
                                                 target_output[key], error)
            except Exception:
                print("数值错误：{} — 变量 {}".format(error, key))
        return 1 if success == len(target_output) else 0

    elif isinstance(target_output, (tuple, list)):
        for i in range(len(target_output)):
            try:
                success += equation_output_check(expected_output[i],
                                                 target_output[i], error)
            except Exception:
                print("数值错误：{} — 位置 {}".format(error, i))
        return 1 if success == len(target_output) else 0

    else:
        if hasattr(target_output, 'shape'):
            np.testing.assert_array_almost_equal(target_output, expected_output)
        else:
            assert target_output == expected_output
        return 1


# -----------------------------------------------------------------------
# 辅助函数：递归检查形状是否一致
# -----------------------------------------------------------------------
def shape_check(expected_output, target_output, error):
    """递归比较 expected_output 与 target_output 的形状是否一致。"""
    success = 0
    if isinstance(target_output, dict):
        for key in target_output.keys():
            try:
                success += shape_check(expected_output[key],
                                       target_output[key], error)
            except Exception:
                print("形状错误：{} — 变量 {}".format(error, key))
        return 1 if success == len(target_output) else 0

    elif isinstance(target_output, (tuple, list)):
        for i in range(len(target_output)):
            try:
                success += shape_check(expected_output[i],
                                       target_output[i], error)
            except Exception:
                print("形状错误：{} — 位置 {}".format(error, i))
        return 1 if success == len(target_output) else 0

    else:
        if hasattr(target_output, 'shape'):
            assert target_output.shape == expected_output.shape
        return 1


# -----------------------------------------------------------------------
# 单测试用例驱动（函数直接接受位置参数列表）
# -----------------------------------------------------------------------
def single_test(test_cases, target):
    """
    对 target 函数运行一组测试用例，每个用例包含：
      name     -- 'datatype_check' / 'equation_output_check' / 'shape_check'
      input    -- 传入 target 的参数列表
      expected -- 期望输出
      error    -- 失败时的提示信息
    """
    success = 0
    for tc in test_cases:
        try:
            if tc['name'] == 'datatype_check':
                assert isinstance(target(*tc['input']), type(tc['expected']))
                success += 1
            elif tc['name'] == 'equation_output_check':
                assert np.allclose(tc['expected'], target(*tc['input']))
                success += 1
            elif tc['name'] == 'shape_check':
                assert tc['expected'].shape == target(*tc['input']).shape
                success += 1
        except Exception:
            print("错误：" + tc['error'])

    if success == len(test_cases):
        print("\033[92m 全部测试通过。")
    else:
        print('\033[92m', success, " 个测试通过")
        print('\033[91m', len(test_cases) - success, " 个测试未通过")
        raise AssertionError(
            "函数 {} 未通过全部测试。请检查公式，避免在函数内部使用全局变量。".format(
                target.__name__))


# -----------------------------------------------------------------------
# 多测试用例驱动（先调用函数，再对返回值做多项检查）
# -----------------------------------------------------------------------
def multiple_test(test_cases, target):
    """
    对 target 函数运行一组测试用例，先获取输出，再分别做类型、数值、形状检查。
    """
    success = 0
    for tc in test_cases:
        try:
            result = target(*tc['input'])
            if tc['name'] == 'datatype_check':
                success += datatype_check(tc['expected'], result, tc['error'])
            elif tc['name'] == 'equation_output_check':
                success += equation_output_check(tc['expected'], result, tc['error'])
            elif tc['name'] == 'shape_check':
                success += shape_check(tc['expected'], result, tc['error'])
        except Exception:
            print("错误：" + tc['error'])

    if success == len(test_cases):
        print("\033[92m 全部测试通过。")
    else:
        print('\033[92m', success, " 个测试通过")
        print('\033[91m', len(test_cases) - success, " 个测试未通过")
        raise AssertionError(
            "函数 {} 未通过全部测试。请检查公式，避免在函数内部使用全局变量。".format(
                target.__name__))
