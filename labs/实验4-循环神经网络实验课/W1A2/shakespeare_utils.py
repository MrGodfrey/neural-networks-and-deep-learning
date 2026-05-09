# 莎士比亚风格文本生成辅助工具
from __future__ import print_function
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import sys
import io


def build_data(text, Tx=40, stride=3):
    """
    通过滑动窗口从文本语料库中构建训练样本。

    参数:
        text   -- 字符串，原始文本语料
        Tx     -- 每个训练样本的序列长度（字符数），默认 40
        stride -- 滑动窗口的步长，默认 3

    返回:
        X -- 训练输入列表，每个元素为长度 Tx 的字符串
        Y -- 训练标签列表，每个元素为对应窗口后一个字符
    """
    X = []
    Y = []
    for i in range(0, len(text) - Tx, stride):
        X.append(text[i: i + Tx])
        Y.append(text[i + Tx])
    print('训练样本数量:', len(X))
    return X, Y


def vectorization(X, Y, n_x, char_indices, Tx=40):
    """
    将字符串列表转换为神经网络所需的数值张量。

    参数:
        X           -- 训练输入列表（字符串）
        Y           -- 训练标签列表（单个字符）
        n_x         -- 字符表大小（one-hot 向量维度）
        char_indices -- 字符到索引的映射字典
        Tx          -- 序列长度，默认 40

    返回:
        x -- 输入张量，形状 (m, Tx, n_x)
        y -- 标签张量，形状 (m, n_x)
    """
    m = len(X)
    x = np.zeros((m, Tx, n_x), dtype=np.bool_)
    y = np.zeros((m, n_x), dtype=np.bool_)
    for i, sentence in enumerate(X):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[Y[i]]] = 1
    return x, y


def sample(preds, temperature=1.0):
    """
    根据概率分布采样下一个字符的索引，支持温度参数控制多样性。

    参数:
        preds       -- 模型输出的概率向量
        temperature -- 采样温度；值越小输出越确定，值越大输出越随机

    返回:
        采样得到的字符索引（整数）
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    out = np.random.choice(range(len(chars)), p=probas.ravel())
    return out


def on_epoch_end(epoch, logs):
    """每个训练轮次结束时的回调函数（占位，可按需扩展）。"""
    None


print("正在加载文本数据...")
text = io.open('shakespeare.txt', encoding='utf-8').read().lower()

Tx = 40
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print("正在构建训练集...")
X, Y = build_data(text, Tx, stride=3)
print("正在向量化训练集...")
x, y = vectorization(X, Y, n_x=len(chars), char_indices=char_indices)
print("正在加载模型...")
model = load_model('models/model_shakespeare_kiank_350_epoch.h5')


def generate_output():
    """
    交互式生成莎士比亚风格的文本续写。

    提示用户输入一段开头文字，模型将续写约 400 个字符。
    """
    generated = ''
    usr_input = input("请输入诗歌的开头（模型将为你续写）: ")
    sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
    generated += usr_input

    sys.stdout.write("\n\n续写结果：\n\n")
    sys.stdout.write(usr_input)

    for i in range(400):
        x_pred = np.zeros((1, Tx, len(chars)))
        for t, char in enumerate(sentence):
            if char != '0':
                x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature=1.0)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

        if next_char == '\n':
            continue
