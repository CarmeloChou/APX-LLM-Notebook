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