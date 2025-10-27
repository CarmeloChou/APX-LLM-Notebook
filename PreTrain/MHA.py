import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    参数：
    输入批次：batch_size
    输入词表量：vocab_size
    注意力头数量：head_num
    嵌入维度：nemd

    作用：
    对输入的批次数据进行多头注意力计算
"""

    def __init__(self,  nemd, head_num):
        super().__init__()
        self.nemd = nemd
        self.head_num = head_num
        # 判断能否被整除
        assert nemd % head_num == 0
        self.head_dim = nemd // head_num
        self.mlp1 = nn.Linear(nemd, 3*nemd)
        self.proj = nn.Linear(nemd, nemd)

    
    def forward(self, X):
        B, T, C = X.shape
        # 初始化中的函数和参数使用必须带上self
        q, k, v = self.mlp1(X).split(self.nemd, dim=-1)
        q = q.view(B, -1, self.head_num, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.head_num, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.head_num, self.head_dim).transpose(1, 2)
        #view使用-1自动计算而不是 :
        score = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1) # B, head_num, T, T
        score = score @ v
        # 必须使用contiguous保持内存一致
        score = score.transpose(1, 2).contiguous().view(B, -1, self.nemd)
        score = self.proj(score)

        return score