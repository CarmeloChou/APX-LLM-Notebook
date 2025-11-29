# GQA&MQA

论文：[[GQA Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints]](./Paper/GQA Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.pdf)

模式：

MHA各个头单独使用一份kv投影，MQA使用1组kv投影，GQA将多个头共用一组kv。

MHA由于每个头单独计算kv，计算开销太大。后来改进为MQA，计算速率提高，但整体效果和精度下降，GQA采用折衷方式，对注意力头进行分组计算kv，减少开销同时对精度消耗也有控制。

![](./Image/GQA.jpg)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    """
    分组注意力机制，kv头数小于多头注意力机制头数大于MQA头数
    参数：
    嵌入维度 n_embd
    kv注意力头数 n_head
    """
    # 当gqa_head = 1 时，退化为MQA，gqa_head 小于 n_head，则利用广播机制实现分组注意力机制
    def __init(self, n_embd, n_head, gqa_head):
        super().__init__()
        assert n_embd % n_head == 0, "嵌入维度需要被注意力头数整除"
        assert n_head % gqa_head == 0, "头数需要被分组头数整除"
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.gqa_head = gqa_head
        self.head_dim = n_embd // n_head
        self.group_size = n_head // gqa_head
        
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, gqa_head * self.head_dim)
        self.v_proj = nn.Linear(n_embd, gqa_head * self.head_dim)
        self.mlp = nn.Linear(n_embd, n_embd)
        
    def forward(self, X):
        B, T, C = X.shape
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)
        
        q = q.view(B, T, n_head, self.head_dim).transpose(1, 2) # B, n_head, T, head_dim
        k = k.view(B, T, self.gqa_head, self.head_dim).transpose(1, 2) # B, gqa_head, T, head_dim
        v = v.view(B, T, self.gqa_head, self.head_dim).transpose(1, 2)
        
        # 关键：将K、V广播到与Q相同的头数
        # 通过repeat_interleave实现分组共享
        k = k.repeat_interleave(self.group_size, dim=1)  # [B, gqa_head, T, head_dim] -> [B, n_head, T, head_dim]
        v = v.repeat_interleave(self.group_size, dim=1)
        
        score = F.softmax(q @ k.tanspose(-1, -2) / torch.sqrt(n_embd//n_head)) @ V
        score = score.transpose(1, 2).contiguous().view(B, T, C)
        score = self.mlp(score)
        
        return score
```

---------------

**补充：repeat_interleave作用**

这里自动广播无法实现。自动广播实现：**两个张量只能从尾部开始逐维度比较，并且每个维度必须满足以下条件之一：**

1. 维度大小相等
2. 其中一个维度大小为1
3. 其中一个张量缺少该维度

```python
original = torch.tensor([
    [[k1]],
    [[k2]],
    [[k3]],
    [[k4]],
]) # 4, 1, 1
expand = original.repeat_interleave(2, dim=0)
[
    [[k1]],
    [[k1]],
    [[k2]],
    [[k2]],
    [[k3]],
    [[k3]],
    [[k4]],
    [[k4]],
] # 8, 1, 1
```

```python
# 替代repeat_interleave的更简洁写法
k_expanded = k_small.repeat(1, self.group_size, 1, 1)  # 在维度1上重复group_size次
v_expanded = v_small.repeat(1, self.group_size, 1, 1)

# 也可以使用stack
# stack 表示在指定维度上堆叠，如果原来维度为3， 4.在0维度上堆叠表示 2，3，4；在1维度上表示3，2，4；在2维度上表示3，4，2
k = torch.stack([k, k], dim=1)
# flatten 用法，从开始维度到终止维度展平，也可以直接使用-2，表示在倒数两个维度上展平
k.flatten(1,2) 

# 还可以切片
k_expand = torch.zeros_like(q)
k_expand(:, 0::2, ...) = k
k_expand(:, 1::2, ...) = k

```

