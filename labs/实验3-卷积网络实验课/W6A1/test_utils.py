import numpy as np


def datatype_check(expected_output, target_output, error):
    """递归检查输出的数据类型是否与期望一致。"""
    success = 0
    if isinstance(target_output, dict):
        for key in target_output.keys():
            try:
                success += datatype_check(expected_output[key], target_output[key], error)
            except:
                print("错误：{} 变量 {}，得到 {} 但期望类型 {}".format(
                    error, key, type(target_output[key]), type(expected_output[key])))
        return 1 if success == len(target_output.keys()) else 0
    elif isinstance(target_output, (tuple, list)):
        for i in range(len(target_output)):
            try:
                success += datatype_check(expected_output[i], target_output[i], error)
            except:
                print("错误：{} 变量 {}，得到 {} 但期望类型 {}".format(
                    error, i, type(target_output[i]), type(expected_output[i])))
        return 1 if success == len(target_output) else 0
    else:
        assert isinstance(target_output, type(expected_output))
        return 1


def equation_output_check(expected_output, target_output, error):
    """递归检查输出的数值是否与期望一致。"""
    success = 0
    if isinstance(target_output, dict):
        for key in target_output.keys():
            try:
                success += equation_output_check(expected_output[key], target_output[key], error)
            except:
                print("错误：{} 变量 {}。".format(error, key))
        return 1 if success == len(target_output.keys()) else 0
    elif isinstance(target_output, (tuple, list)):
        for i in range(len(target_output)):
            try:
                success += equation_output_check(expected_output[i], target_output[i], error)
            except:
                print("错误：{} 位置 {} 的变量。".format(error, i))
        return 1 if success == len(target_output) else 0
    else:
        if hasattr(target_output, 'shape'):
            np.testing.assert_array_almost_equal(target_output, expected_output)
        else:
            assert target_output == expected_output
        return 1


def shape_check(expected_output, target_output, error):
    """递归检查输出的形状是否与期望一致。"""
    success = 0
    if isinstance(target_output, dict):
        for key in target_output.keys():
            try:
                success += shape_check(expected_output[key], target_output[key], error)
            except:
                print("错误：{} 变量 {}。".format(error, key))
        return 1 if success == len(target_output.keys()) else 0
    elif isinstance(target_output, (tuple, list)):
        for i in range(len(target_output)):
            try:
                success += shape_check(expected_output[i], target_output[i], error)
            except:
                print("错误：{} 变量 {}。".format(error, i))
        return 1 if success == len(target_output) else 0
    else:
        if hasattr(target_output, 'shape'):
            assert target_output.shape == expected_output.shape
        return 1


def single_test(test_cases, target):
    """对单个函数运行一组测试用例。"""
    success = 0
    for test_case in test_cases:
        try:
            if test_case['name'] == "datatype_check":
                assert isinstance(target(*test_case['input']), type(test_case["expected"]))
                success += 1
            if test_case['name'] == "equation_output_check":
                assert np.allclose(test_case["expected"], target(*test_case['input']))
                success += 1
            if test_case['name'] == "shape_check":
                assert test_case['expected'].shape == target(*test_case['input']).shape
                success += 1
        except:
            print("错误：" + test_case['error'])

    if success == len(test_cases):
        print("\033[92m 所有测试通过。")
    else:
        print('\033[92m', success, " 个测试通过")
        print('\033[91m', len(test_cases) - success, " 个测试失败")
        raise AssertionError(
            "函数 {} 未通过所有测试。请检查公式，避免在函数内部使用全局变量。".format(target.__name__))


def multiple_test(test_cases, target):
    """对返回复合结构的函数运行一组测试用例。"""
    success = 0
    for test_case in test_cases:
        try:
            target_answer = target(*test_case['input'])
            if test_case['name'] == "datatype_check":
                success += datatype_check(test_case['expected'], target_answer, test_case['error'])
            if test_case['name'] == "equation_output_check":
                success += equation_output_check(test_case['expected'], target_answer, test_case['error'])
            if test_case['name'] == "shape_check":
                success += shape_check(test_case['expected'], target_answer, test_case['error'])
        except:
            print("错误：" + test_case['error'])

    if success == len(test_cases):
        print("\033[92m 所有测试通过。")
    else:
        print('\033[92m', success, " 个测试通过")
        print('\033[91m', len(test_cases) - success, " 个测试失败")
        raise AssertionError(
            "函数 {} 未通过所有测试。请检查公式，避免在函数内部使用全局变量。".format(target.__name__))
