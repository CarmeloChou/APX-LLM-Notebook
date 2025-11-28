# MoE

论文：[[DeepSeekMoE Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models]](./Paper/DeepSeekMoE Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.pdf)

DeepSeekMoE架构：设立n个专家，激活其中的k个；设立共享专家，使得部分通用知识在模型中流通。整体架构如下图所示

![DeepSeekMoE架构](./Image/DeepSeekMoE.png)

![](./Image/稀疏模型与稠密模型.jpg)

- 稀疏MoE层：代替了传统Transformer中的FFN层。MoE层包含若干专家，每个专家本身是独立的神经网络，这些专家通常是FFN，也可以是更复杂的网络结构。甚至可以是MoE堆叠，形成层级MoE架构
- 门控网络/路由：决定Token被发送到哪个专家。可分为Token Choice和Expert Choice，分别控制不同token分给不同的专家，或者全部token分给某个专家。Token choice可能存在语义中断的问题，某些token被集中分配到某些expert上，未分配的expert训练不充分，失去了MoE的意义。Expert Choice性能弱于Token Choice，expert choice会有单词丢失问题，可能会影响单词推理时无法看到后续的token

## Token Choice

```python
# Token Choice
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicMoELayer(nn.Module):
    def __init__(self, n_embd, expert_size, num_experts, top_k=2):
        """
        n_embd: 嵌入维度
        expert_size：每个专家的中间层维度
        num_experts: 专家数量
        top_k: 每个token选择的专家数量
        """
        super().__init__()
        self.n_embd = n_embd
        self.expert_size = expert_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 门控网络（router网络）
        self.gate = nn.Linear(n_embd, num_experts, bias=False)
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
        		nn.Linear(n_embd, expert_size),
                nn.GELU()
                nn.Linear(expert_size, n_embd)
            ) for _ in range(num_experts)
        ])
        
        def forward(self, X):
            """
            X:[B, T, C]
            return : [B, T, C]
            """
            B, T, n_embd = X.shape
            original_shape = X.shape
            
            # 展平B T维度
            X_flat = X.view(-1, n_embd) # B*T C
            
            # 计算门控分数
            gate_logits = self.gate(X_flat)
            
            # 选择top_k个专家
            top_k_gate, top_k_indices = torch.topk(gate_logits, k=self.top_k, dim=-1) 
            
            gate_wei = F.softmax(top_k_gate, dim=-1) # B*T 2
            
            output = torch.zeros_like(X_flat)
            
            for expert_idx in range(self.num_experts):
                expert_mask = (top_k_indices == expert_idx).any(dim=-1) #B*T, 1
                
                if expert_mask.any():
                    expert_input = X_flat[expert_mask] #top_k=True, nembd
                    expert_output = self.experts[expert_idx](expert_input) # top_k=True, n_embd
                    
                    token_expert_mask = top_k_indices[expert_mask] == expert_idx # top_k=True, top_k
                    token_wei = gate_wei[expert_mask] # B*T, top_k
                    
                    expert_wei = torch.sum(token_wei * token_expert_mask.float(), dim=-1, keepdim=True)
                    
                    output[expert_mask] += expert_output * expert_wei
                    
                    output = output.reshape(original_shape)
                    return output
```

---

补充：torch索引机制

```python
a[[1,0,0]], a[[True, False, False]]
# 以上二者结果不同，整数数组索引表示取第1行、第0行、第0行
# 布尔掩码表示，只保留行数=True的那一行或多行
```

## Expert Choice

```python
class Expert_Choice_Efficient(nn.Module):
    def __init__(self, n_embd, experts_num, expert_dim, top_k=2):
        super().__init__()
        self.n_embd = n_embd
        self.experts_num = experts_num
        self.expert_dim = expert_dim
        self.top_k = top_k

        # 共享的门控网络，输出为专家数量
        self.gate_network = nn.Linear(n_embd, experts_num)

        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, n_embd)
            ) for _ in range(experts_num)
        ])

    def forward(self, X):
        B, T, C = X.shape
        x_flat = X.view(-1, C)  # [B*T, C]
        total_tokens = x_flat.shape[0]

        # 计算所有token对所有专家的分数
        gate_scores = self.gate_network(x_flat)  # [B*T, experts_num]

        # 每个专家选择top_k个token
        top_k = min(self.top_k, total_tokens)
        expert_topk_scores, expert_topk_indices = torch.topk(
            gate_scores.t(), k=top_k, dim=1
        )  # [experts_num, top_k]

        # 初始化输出和usage统计
        output = torch.zeros_like(x_flat)
        usage_mask = torch.zeros(total_tokens, dtype=torch.bool, device=x_flat.device)

        # 批量处理所有专家
        for expert_id in range(self.experts_num):
            token_indices = expert_topk_indices[expert_id]  # [top_k]
            token_scores = expert_topk_scores[expert_id]  # [top_k]
            
            if len(token_indices) > 0:
                # 标记被使用的token
                usage_mask[token_indices] = True
                
                # 专家处理
                selected_tokens = x_flat[token_indices]
                expert_output = self.experts[expert_id](selected_tokens)
                
                # 计算权重
                weights = F.softmax(token_scores, dim=0).unsqueeze(-1)
                weighted_output = expert_output * weights
                
                # 使用index_add_高效累加
                output.index_add_(0, token_indices, weighted_output)

        # 处理未被选中的token（残差连接）
        if (~usage_mask).any():
            output[~usage_mask] = x_flat[~usage_mask]

        return output.view(B, T, C)
```

token choice 不会造成语义中断，因为相邻的token更倾向于选择同一个expert。而expert choice则可能造成语义中断，相邻的token可能选择不同的expert

## Global Choice

## Shared Experts

# 优化策略

## Load Balance 负载均衡

指部分专家之间的分配计算不平衡，可能某些专家处于高负荷计算，但某些专家长期处于饥饿状态。可能影响模型的计算能力。

## 辅助损失函数

## 路由优化

## 处理丢弃令牌

