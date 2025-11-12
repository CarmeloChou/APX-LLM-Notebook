# Lora & PEFT method

## PEFT方法分类

附加方法：保持原有模型参数不变，引入少量新的可训练参数。这些新参数位于特定的模型架构之中。

- 适配器微调：添加适配器层，对原有模型参数冻结。相当于额外训练一个补丁附加到原来的模型参数上，实现微调效果。

![](./Image/适配器微调.jpg)

```python
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, original_layer, down_dim):
        """
        参数：
        w_original 原始模型层（如nn.Linear, nn.Conv2d等）
        down_dim 降维的维度数
        """
        super().__init__()
        self.original_layer = original_layer
        self.down_dim = down_dim
        
        # 冻结原始层参数
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # 获取输入维度
        if hasattr(original_layer, "in_features"):
            in_dim = original_layer.in_features
            out_dim = original_layer.out_features
        elif hasattr(original_layer, "in_channels"):
            in_dim = original_layer.in_channels
            out_dim = original_layer.out_features
        else:
            raise ValueError("无法确定原始层的输入维度")
            
        self.down_proj = nn.Linear(in_dim, down_dim, bias=False)
        self.up_proj = nn.Linear(down_dim, out_dim, bias=False)
        self.activation = nn.ReLU()
        
        # 初始化适配器权重为接近0的小值，确保训练初期适配器影响很小
        nn.init.zeros_(self.up_proj.weight)
                
    def forward(self, X):
        # 原始层前向传播
        original_output = self.original_layer(X)
        
        # 适配器前向传播
        adapter_output = self.down_proj(X)
        adapter_output = self.activation(adapter_output)
        adapter_output = self.up_proj(adapter_output)
        
        # 残差连接：原始输出 + 适配器输出
        output = original_output + adapter_output
        return output
```



- [[*前缀微调* **Prefix-Tunning**]](./Paper/Prefix-Tuning: Optimizing Continuous Prompts for Generation.pdf)：将一系列可训练的连续向量（即“前缀”）添加到每个注意力层的输入之前。这些前缀在微调过程中作为任务特定的指令被学习。

![](./Image/前缀微调.png)



- *提示微调* 是一种简化方法，其中可训练向量仅添加到初始输入嵌入序列中。像 P-Tuning 这样的变体进一步完善了这一思路。