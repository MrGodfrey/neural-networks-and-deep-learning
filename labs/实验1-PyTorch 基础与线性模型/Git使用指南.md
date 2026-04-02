# Git 使用指南

在深度学习和协作开发中，代码版本控制至关重要。Git 是一种分布式版本控制系统，能帮助你记录文件的每一次修改，追踪代码的变化，并方便地与他人合作。

## 一、Git 基本概念

理解以下三个核心区域，是掌握 Git 的关键：
1. **工作区（Working Directory）**：你真正在电脑上看到和编辑的文件目录。
2. **暂存区（Staging Area）**：一个隐藏的区域（文件列表），用于临时存储你准备进行版本提交的修改。
3. **本地仓库（Local Repository）**：Git 存储历史记录和各个版本的地方。

工作流大致为： `工作区修改 -> (add) -> 暂存区 -> (commit) -> 仓库`

## 二、基础操作

### 1. 初始化仓库 (git init)
在你的项目文件夹中（例如 `nndlClass`），打开终端并输入以下命令以初始化一个新的 Git 仓库：
```bash
git init
```

### 2. 查看状态 (git status)
在做任何操作之前或之后，随时查看当前状态：
```bash
git status
```
这会显示哪些文件被修改了、哪些还没被追踪。

### 3. 添加到暂存区 (git add)
将修改的文件添加到暂存区：
```bash
git add <filename>  # 添加指定文件
git add .           # 添加当前目录下的所有修改
```

### 4. 提交版本 (git commit)
将暂存区的更改永久保存到本地仓库并附上一条说明：
```bash
git commit -m "添加了线性回归模型代码"
```

### 5. 查看历史记录 (git log)
查看所有过去的提交记录：
```bash
git log
# 或者使用简化版
git log --oneline
```

### 6. 查看修改差异 (git diff)
看看你到底改了什么内容（在 `git add` 之前使用）：
```bash
git diff
```

## 三、远程协作

你的代码常常需要备份到 GitHub、Gitee 等托管平台，或是团队合作开发。

### 1. 克隆远程仓库 (git clone)
如果你想把云端的代码完整拉取到本地（自动完成 `init` 并关联远程）：
```bash
git clone https://github.com/username/repository.git
```

### 2. 关联远程仓库 (git remote)
如果是你自己在本地创建的仓库，想将其推送到云端，需要先关联远程地址：
```bash
git remote add origin https://github.com/username/repository.git
```

### 3. 拉取更新 (git pull)
把远程仓库最新的修改拉取并合并到本地：
```bash
git pull origin main
```

### 4. 推送到远程 (git push)
将你的本地提交同步到远程仓库：
```bash
git push -u origin main
```

## 四、分支管理

分支（Branch）允许你在不同于主线（通常是 `main` 或 `master`）的环境中安全地开发新功能，互不影响。

### 1. 创建分支并切换 (git branch / git checkout)
```bash
git branch feature-logistic   # 创建新分支
git checkout feature-logistic # 切换到新分支

# 上面两步可以合并为一个命令：
git checkout -b feature-logistic
```

### 2. 查看所有分支
```bash
git branch
```

### 3. 合并分支 (git merge)
当你在分支上完成了工作（例如写完了二分类模型代码，并已 commit），切回 `main` 并把修改合并进来：
```bash
# 1. 切换回主分支
git checkout main
# 2. 将 feature-logistic 的代码合并到当前的 main 分支中
git merge feature-logistic
```

---
**配置提示**：第一次使用 Git 前，建议配置你的名字和邮箱（这些信息会出现在每次 commit 记录中）：
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```