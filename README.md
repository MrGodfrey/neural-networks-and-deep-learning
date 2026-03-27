# 神经网络与深度学习 (Neural Networks and Deep Learning)

欢迎来到**数据科学与大数据技术专业**大三核心课程《神经网络与深度学习》的开源代码与讲义仓库。本仓库旨在为选修该课程的同学们提供一站式的学习资源，同时也欢迎任何对深度学习感兴趣的朋友学习和贡献。

**仓库地址：**
- **GitHub (国外)**: [https://github.com/MrGodfrey/neural-networks-and-deep-learning](https://github.com/MrGodfrey/neural-networks-and-deep-learning)
- **Gitee (国内)**: [https://gitee.com/lvqiscu/neural-networks-and-deep-learning](https://gitee.com/lvqiscu/neural-networks-and-deep-learning)

## 📖 课程简介

本课程以邱锡鹏教授的《神经网络与深度学习》为主要教材，分为四个主要部分：
1. **机器学习基础**：概率论、线性模型等。
2. **基础模型**：前馈神经网络、卷积神经网络 (CNN)、循环神经网络 (RNN)。
3. **进阶模型**：网络优化、注意力机制、生成模型等。
4. **大模型与前沿技术**：Transformer、大语言模型 (LLM)、强化学习及相关前沿应用。

## 📂 仓库目录结构

本仓库包含理论讲义和实验课代码两部分，将随着课程的推进逐步更新：

```text
.
├── README.md
├── notes/               # 每周的理论课讲义 (为 LaTeX 源码的 .tex 文件)
│   ├── Outline.tex      # 课程大纲
│   ├── w1.tex           # 第一周理论讲义
│   ├── w2.tex           # 第二周理论讲义
│   └── ...
├── slides/              # 每周的课程 slides
│   ├── w1.pdf           # 第一周课程
│   ├── w2.pdf           # 第二周课程
│   └── ...
└── labs/                # 六次实验课的参考代码与说明 (将在实验课后陆续更新)
    ├── lab1_linear_models/       # 实验一：线性模型的实现与应用
    ├── lab2_feedforward_nn/      # 实验二：前馈神经网络的实现与应用
    ├── lab3_cnn/                 # 实验三：卷积神经网络的实现与应用
    ├── lab4_rnn/                 # 实验四：循环神经网络的实现与应用
    ├── lab5_optimization/        # 实验五：网络优化与正则化方法的实现与应用
    └── lab6_attention/           # 实验六：注意力机制的实现与应用
```

## 💻 实验课说明

本学期共安排 **6 次实验课**。实验课的代码示例与指导文档均放置在 `labs/`  下。我们将主要使用 **Python** 和 **PyTorch** 框架来搭建和训练模型。建议同学们在实验课前配置好相应的本地环境或使用云端环境。


##  项目分享

如果你做了有趣的项目，或者完成了任何与本课程相关、希望分享给大家的内容，我专门建立了一个项目仓库：

[neural-networks-and-deep-learning-student-project](https://github.com/MrGodfrey/neural-networks-and-deep-learning-student-project)

请移步该仓库，并通过 Pull Request 的方式提交你认为值得分享的内容。

## 💬 Issue 使用指南 (Q&A)

如果你在课程学习或实验过程中遇到任何问题，欢迎在 GitHub 的 [Issues](https://github.com/MrGodfrey/neural-networks-and-deep-learning/issues) 区提问与互动！这里是我们交流和答疑的主要平台。

**你可以在 Issues 中提问或发起的情况：**
- **理论解惑**：对讲义中的公式推导、概念理解有疑问。
- **实验求助**：实验代码报错、环境配置遇到困难、或是某个模型无法收敛。
- **拓展探讨**：分享你对深度学习相关知识的见解，或者提出实验代码的优化建议。
- **问题反馈**：指出讲义、作业或实验指导书中的错漏（如果你暂时不方便提交 PR，也可以在这里反馈）。

**提问小贴士：**
1. 提问前请先在 Issues 列表中搜索，看看是否已有同学提出了类似问题并得到了解答。
2. 尽量提供充足的信息。如果是代码问题，请务必附上完整的报错截图或日志、关键代码片段以及你的环境版本（如操作系统、Python 和 PyTorch 版本）。
3. 互相帮助。如果你知道其他同学提问的答案，欢迎直接回复解答！



## 🤝 参与贡献 (Pull Request 指南)

这份课程资料不仅供大家学习，也是一个协作构建的项目。如果同学们在阅读讲义时发现了错别字、推导错误，或者在实验代码中发现了 Bug、想补充更好的代码注释及实战实现，非常欢迎你们提交 Pull Request (PR) 做出贡献！

> **⚠️ 重要说明：**
> **本项目的 Gitee 仓库仅作为国内镜像供大家快速访问。我们目前只接受在 GitHub 上提交的 Pull Request 进行代码和文档的合并协作。**

**参与贡献的步骤如下：**

1. **Fork 本仓库**：点击本页面右上角的 `Fork` 按钮，将本仓库复制到你的个人 GitHub 账号下。
2. **克隆到本地**：
   ```bash
   git clone https://github.com/MrGodfrey/neural-networks-and-deep-learning
   cd neural-networks-and-deep-learning
   ```
3. **创建分支**：建议基于你要修改的内容创建一个带有描述性名称的新分支。
   ```bash
   git checkout -b fix-typo-w1
   ```
4. **修改代码或文档**：使用编辑器修改 `.tex` 源码或 `labs/` 里的 Python 代码。
5. **提交修改**：
   ```bash
   git add .
   git commit -m "docs: 修复第一周讲义中的错别字"
   ```
6. **推送到远程仓库**：
   ```bash
   git push origin fix-typo-w1
   ```
7. **提交 Pull Request**：回到本仓库的 GitHub 页面，点击 `Compare & pull request` 按钮，简要描述你的修改内容，然后提交。我们会尽快审核并合并你的代码！

### 常见贡献方向：
- 修正 `notes/` 目录下讲义中的错别字或 LaTeX 公式拼写。
- 补充或完善 `labs/` 目录下的代码注释。
- 提供实验课题目的其他可行解法或进阶思路。


## 📚 参考资源

- **主教材**：邱锡鹏《神经网络与深度学习》 [(官方资源)](https://nndl.github.io/)
- **推荐扩展与框架**：
  - 李沐《动手学深度学习》: [zh-v2.d2l.ai](https://zh-v2.d2l.ai/)
  - PyTorch 官方网站: [pytorch.org](http://pytorch.org)

## 📄 许可证 (License)

本仓库采用双重许可：

- **笔记讲义 & 讲义 (Notes/slides)**：`notes/` 和 `slides/` 目录下的所有内容受 [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](notes/LICENSE) 保护。你可以自由分享和演绎，但必须署名且不可用于商业目的。
- **实验代码 (Labs)**：`labs/` 目录下的所有代码采用 [MIT License](labs/LICENSE) 授权。你可以自由地使用、修改和分发这些代码。
