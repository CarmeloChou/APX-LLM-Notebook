# Git

## git 工作区、暂存区、仓库

```
工作区 (Working Directory)  →  暂存区 (Staging Area)  →  仓库 (Repository)
      ↓                            ↓                          ↓
[拍原始照片]                [选要精修的照片]            [保存最终作品到相册]
```

![](./Image/git.png)

## git diff

- git diff = 工作区 vs 暂存区
- git diff --cached = 暂存区 vs 最后一次提交
- git diff HEAD = 工作区 vs 最后一次提交
- git checkout -- file = 用暂存区覆盖工作区
- git restore file = 新版的 git checkout -- file

## git pull

代理模式：```git -c https.proxy=127.0.0.1:7897 pull```

## git stash

git stash是 Git 中一个极其有用的功能，用于临时保存工作目录的修改，让你可以切换分支或执行其他操作，稍后再恢复这些修改。

```bash
# 场景：你正在 feature 分支上开发
git status
# 修改了多个文件，但还没完成

# 这时需要紧急切换到 main 分支修复 bug
git checkout main
# ❌ 错误：Git 会阻止，因为你有未提交的修改

# ✅ 正确做法：使用 git stash
git stash
# 保存所有修改，工作目录变干净
git checkout main
# 现在可以切换到其他分支
```

一般的处理流程如下，遇到紧急开发时，手头的工作还未保存，使用git stash临时性存储，解决紧急问题后，回到开发分支。

```bash
# 1. 开始新功能开发
git checkout -b feature/new-feature

# 2. 开发过程中...
echo "代码修改" >> file1.py
git add file1.py
echo "更多修改" >> file2.py  # 还没有 add

# 3. 需要处理紧急任务
git stash save "新功能开发中-已完成文件1"
Saved working directory and index state On feature/new-feature: 新功能开发中-已完成文件1

# 4. 切换分支处理紧急任务
git checkout main
git checkout -b hotfix/urgent
# 修复问题，提交
git add .
git commit -m "修复紧急问题"
git checkout main
git merge hotfix/urgent
git branch -d hotfix/urgent

# 5. 回到功能开发
git checkout feature/new-feature
git stash pop
# 修改全部恢复，继续开发
```

## git checkout

git checkout是 Git 中最多功能、最重要的命令之一，用于切换分支、恢复文件、创建分支等多种操作。

```bash
# 1. 切换分支
git checkout <branch-name>

# 2. 创建并切换到新分支
git checkout -b <new-branch>

# 3. 恢复文件到指定版本
git checkout <commit> -- <file>

# 4. 查看历史版本
git checkout <commit>
```
