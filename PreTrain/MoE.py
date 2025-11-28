import torch
import torch.nn as nn
import torch.nn.functional as F

class Token_Choice(nn.Module):
    def __init__(self, n_embd, experts_num, expert_dim, top_k=2):
        super.__init__()
        self.n_embd = n_embd
        self.experts_num = experts_num
        self.expert_dim = expert_dim
        self.top_k = top_k

        # 门控网络
        self.gate = nn.Parameter(n_embd, experts_num)

        # 专家网络
        self.experts = nn.ModuleList({
            nn.Sequential(
                nn.Linear(n_embd, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, experts_num)
            ) for _ in range(experts_num)
        }) 

    def forward(self, X):
        B, T, C = X.shape

        # 展平B,T 维度
        x_flat = X.view(-1, C)

        # 计算门控分数，对所有token B*T批量计算
        x_score = self.gate(x_flat)

        # 选择topk个专家
        top_k_gate, top_k_indice = torch.topk(x_score, self.top_k, dim=-1)

        # 计算topk的数值权重，加权后传递给专家网络   
        gate_wei = F.softmax(top_k_gate, dim=-1)

        output = torch.zeros_like(x_flat)

        for expert_id in range(self.experts_num):
            expert_mask = (expert_id == top_k_indice).any(dim=-1) # B*T, 1

            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_out = self.experts[expert_id](expert_input)

                token_expert_mask = top_k_indice[expert_mask] == expert_id # B*T, 2
                token_wei = top_k_gate[expert_mask] # B*T, 2

                expert_wei = torch.sum(token_wei * token_expert_mask.float(), dim=-1, keepdim=True) # B*T, 2
                 # 不同的Token对于同一专家需要累计计算
                output[expert_mask] += expert_wei * gate_wei[expert_mask]

        return output.view(B, T, C)

