import numpy as np
from termcolor import colored
import torch
import torch.nn as nn

def test_happy_model(happyModel):
    model = happyModel()
    layers = list(model.children())

    # 1. 层数检查
    assert len(layers) == 8, \
        f"模型应有 8 层，当前为 {len(layers)} 层"

    # 2. 各层类型检查
    expected_types = [
        nn.ZeroPad2d, nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
        nn.MaxPool2d, nn.Flatten, nn.Linear, nn.Sigmoid
    ]
    for i, (layer, expected) in enumerate(zip(layers, expected_types)):
        assert isinstance(layer, expected), \
            f"第 {i+1} 层应为 {expected.__name__}，当前为 {type(layer).__name__}"

    # 3. 各层关键参数检查
    pad, conv, bn, _, pool, _, fc, _ = layers

    assert pad.padding == (3, 3, 3, 3), \
        f"ZeroPad2d padding 应为 3，当前为 {pad.padding}"
    assert conv.in_channels == 3 and conv.out_channels == 32, \
        f"Conv2d 通道数应为 (3->32)，当前为 ({conv.in_channels}->{conv.out_channels})"
    assert conv.kernel_size == (7, 7) and conv.stride == (1, 1), \
        f"Conv2d kernel_size 应为 7，stride 应为 1"
    assert bn.num_features == 32, \
        f"BatchNorm2d num_features 应为 32，当前为 {bn.num_features}"
    assert pool.kernel_size == 2 and pool.stride == 2, \
        f"MaxPool2d kernel_size 和 stride 均应为 2"
    assert fc.in_features == 32768 and fc.out_features == 1, \
        f"Linear 应为 (32768->1)，当前为 ({fc.in_features}->{fc.out_features})"

    # 4. 前向传播输出形状检查
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(4, 3, 64, 64)
        out = model(dummy)
    assert out.shape == (4, 1), \
        f"输出形状应为 (4, 1)，当前为 {tuple(out.shape)}"
    assert ((out >= 0) & (out <= 1)).all(), \
        "Sigmoid 输出应在 [0, 1] 范围内"

    print("All tests passed")


def test_convolutional_model(convolutional_model):
    model = convolutional_model((64, 64, 3))

    # 1. 必须是 nn.Module 的子类
    assert isinstance(model, nn.Module), \
        "convolutional_model 返回值应为 nn.Module 的实例"

    # 2. 提取所有子模块（仅直接子层）
    named = dict(model.named_children())

    # 3. 检查各层是否存在且类型正确
    for attr, expected_type in [
        ('conv1', nn.Conv2d), ('relu1', nn.ReLU), ('pool1', nn.MaxPool2d),
        ('conv2', nn.Conv2d), ('relu2', nn.ReLU), ('pool2', nn.MaxPool2d),
        ('flatten', nn.Flatten), ('fc', nn.Linear),
    ]:
        assert attr in named, \
            f"模型中缺少层：{attr}"
        assert isinstance(named[attr], expected_type), \
            f"层 {attr} 类型应为 {expected_type.__name__}，当前为 {type(named[attr]).__name__}"

    conv1, pool1, conv2, pool2, fc = (
        named['conv1'], named['pool1'],
        named['conv2'], named['pool2'],
        named['fc'],
    )

    # 4. conv1 参数
    assert conv1.in_channels == 3 and conv1.out_channels == 8, \
        f"conv1 通道数应为 (3->8)，当前为 ({conv1.in_channels}->{conv1.out_channels})"
    assert conv1.kernel_size == (4, 4) and conv1.stride == (1, 1), \
        f"conv1 kernel_size 应为 4，stride 应为 1"

    # 5. pool1 参数
    assert pool1.kernel_size == 8 and pool1.stride == 8, \
        f"pool1 kernel_size 和 stride 均应为 8，当前 kernel={pool1.kernel_size} stride={pool1.stride}"

    # 6. conv2 参数
    assert conv2.in_channels == 8 and conv2.out_channels == 16, \
        f"conv2 通道数应为 (8->16)，当前为 ({conv2.in_channels}->{conv2.out_channels})"
    assert conv2.kernel_size == (2, 2) and conv2.stride == (1, 1), \
        f"conv2 kernel_size 应为 2，stride 应为 1"

    # 7. pool2 参数
    assert pool2.kernel_size == 4 and pool2.stride == 4, \
        f"pool2 kernel_size 和 stride 均应为 4，当前 kernel={pool2.kernel_size} stride={pool2.stride}"

    # 8. 全连接层参数
    assert fc.in_features == 64 and fc.out_features == 6, \
        f"fc 应为 (64->6)，当前为 ({fc.in_features}->{fc.out_features})"

    # 9. 前向传播输出形状检查
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(4, 3, 64, 64)
        out = model(dummy)
    assert out.shape == (4, 6), \
        f"输出形状应为 (4, 6)，当前为 {tuple(out.shape)}"

    print("All tests passed")

