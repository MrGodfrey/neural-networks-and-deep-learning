import numpy as np
from copy import deepcopy


def datatype_check(expected_output, target_output, error, level=0):
    success = 0
    if level == 0:
        try:
            assert isinstance(target_output, type(expected_output))
            return 1
        except:
            return 0
    else:
        if isinstance(expected_output, (tuple, list, np.ndarray, dict)):
            range_values = expected_output.keys() if isinstance(expected_output, dict) else range(len(expected_output))
            if len(expected_output) != len(target_output) or not isinstance(target_output, type(expected_output)):
                return 0
            for i in range_values:
                try:
                    success += datatype_check(expected_output[i], target_output[i], error, level - 1)
                except:
                    pass
            return 1 if success == len(expected_output) else 0
        else:
            try:
                assert isinstance(target_output, type(expected_output))
                return 1
            except:
                return 0


def equation_output_check(expected_output, target_output, error):
    success = 0
    if isinstance(expected_output, (tuple, list, dict)):
        range_values = expected_output.keys() if isinstance(expected_output, dict) else range(len(expected_output))
        if len(expected_output) != len(target_output):
            return 0
        for i in range_values:
            try:
                success += equation_output_check(expected_output[i], target_output[i], error)
            except:
                print("Error: {} for variable in position {}.".format(error, i))
        return 1 if success == len(expected_output) else 0
    else:
        try:
            if hasattr(expected_output, 'shape'):
                np.testing.assert_array_almost_equal(target_output, expected_output)
            else:
                assert target_output == expected_output
        except:
            return 0
        return 1


def shape_check(expected_output, target_output, error):
    success = 0
    if isinstance(expected_output, (tuple, list, dict, np.ndarray)):
        range_values = expected_output.keys() if isinstance(expected_output, dict) else range(len(expected_output))
        if len(expected_output) != len(target_output):
            return 0
        for i in range_values:
            try:
                success += shape_check(expected_output[i], target_output[i], error)
            except:
                print("Error: {} for variable {}.".format(error, i))
        return 1 if success == len(expected_output) else 0
    else:
        return 1


def single_test(test_cases, target):
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
            print("Error: " + test_case['error'])
    if success == len(test_cases):
        print("\033[92m All tests passed.")
    else:
        print('\033[92m', success, " Tests passed")
        print('\033[91m', len(test_cases) - success, " Tests failed")
        raise AssertionError(
            "Not all tests were passed for {}. Check your equations and avoid using global variables inside the function.".format(target.__name__))


def multiple_test(test_cases, target):
    success = 0
    for test_case in test_cases:
        try:
            test_input = deepcopy(test_case['input'])
            target_answer = target(*test_input)
        except:
            raise AssertionError("Unable to successfully run test case for {}.".format(target.__name__))
        try:
            if test_case['name'] == "datatype_check":
                success += datatype_check(test_case['expected'], target_answer, test_case['error'])
            if test_case['name'] == "equation_output_check":
                success += equation_output_check(test_case['expected'], target_answer, test_case['error'])
            if test_case['name'] == "shape_check":
                success += shape_check(test_case['expected'], target_answer, test_case['error'])
        except:
            print("Error: " + test_case['error'])
    if success == len(test_cases):
        print("\033[92m All tests passed.")
    else:
        print('\033[92m', success, " Tests passed")
        print('\033[91m', len(test_cases) - success, " Tests failed")
        raise AssertionError(
            "Not all tests were passed for {}. Check your equations and avoid using global variables inside the function.".format(target.__name__))

def forward_propagation_test(target):
    x, theta = 2, 4
    expected_output = 8
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [x, theta],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    single_test(test_cases, target)

def backward_propagation_test(target):
    x, theta = 3, 4
    expected_output = 3
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [x, theta],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    single_test(test_cases, target)

def gradient_check_test(target):
    x, theta = 3, 4
    expected_output = 7.814075313343006e-11
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [x, theta],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    single_test(test_cases, target)


def predict_test(target):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
        [-0.02311792,  0.03137121],
        [-0.0169217 , -0.01752545],
        [ 0.00935436, -0.05018221]]),
     'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
     'b1': np.array([[ -8.97523455e-07],
        [  8.15562092e-06],
        [  6.04810633e-07],
        [ -2.54560700e-06]]),
     'b2': np.array([[  9.14954378e-05]])}
    expected_output = np.array([[True, False, True]])

    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    single_test(test_cases, target)
