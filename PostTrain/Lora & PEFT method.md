# Lora & PEFT method

## PEFT方法分类

###  附加方法

保持原有模型参数不变，引入少量新的可训练参数。这些新参数位于特定的模型架构之中。

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

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrefixTunning(nn.Module):
    def __init__(self, model, prefix_len, hidden_size, num_heads, num_layers):
    """
    	参数：
    	model : 初始模型
    	prefix_len : 前缀长度，相当于在T的维度增加
    	hidden_size : 隐藏层维度
    	num_layers : Transformer层层数
    """
    super().__init__()
    self.model = model
    self.prefix_len = prefix_len
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_layers = num_layers
    
    # 冻结初始模型参数
    for p in self.model.parameters():
        p.requires_grad = False
    
    # 为模型每一层创建前缀
    self.prefix_embd = nn.Parameters(
    	torch.rand(prefix_len, hidden_size)
    )
    
    # 为每一层的key和value创建前缀
    self.prefix_keys = nn.ParameterList([
        nn.Parameter(torch.randn(prefix_len, hidden_size))
        for _ in range(num_layers)
    ])
    self.prefix_values = nn.ParameterList([
        nn.Parameter(torch.randn(prefix_len, hidden_size))
        for _ in range(num_layers)
    ])
    
    def forward(self, input_ids, attention_mask=None):
        B, T, C = input_ids.shape
        
        # 获取原始输入嵌入
        input_embd = self.model.get_input_embeddings()(input_ids)
    	
        # 为每个样本添加前缀
        prefix_embd = self.prefix_embeddings.unsqueeze(0).expand(B, -1, -1)
        combined_embd = torch.cat([prefix_embd, input_embd])
        
    	# 扩展注意力掩码以包含前缀
        if attention_mask is not None:
            prefix_mask = torch.ones(B, self.prefix_length, device=input_ids.device)
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            combined_mask = None
        
        # 修改模型的注意力机制以包含前缀
        outputs = self.custom_forward_with_prefix(input_embeds, combined_mask)
        return outputs
    
    def custom_forward_with_prefix(self, input_embeds, attention_mask):
        """自定义前向传播，在每一层的注意力机制中注入前缀"""
        
        # 这里需要重写Transformer的forward方法
        # 在实际实现中，这通常需要修改模型的源代码
        # 或者使用hook机制来注入前缀
        
        # 简化示例：实际实现会更复杂
        hidden_states = input_embeds
        
        for layer_idx, layer in enumerate(self.model.transformer.h):
            # 获取当前层的前缀key和value
            prefix_key = self.prefix_keys[layer_idx]
            prefix_value = self.prefix_values[layer_idx]
            
            # 扩展前缀到batch大小
            batch_size = input_embeds.shape[0]
            prefix_key = prefix_key.unsqueeze(0).expand(batch_size, -1, -1)
            prefix_value = prefix_value.unsqueeze(0).expand(batch_size, -1, -1)
            
            # 在这里修改注意力机制来包含前缀
            # 实际实现需要重写attention计算
            layer_outputs = self.prefix_attention_layer(
                layer, hidden_states, prefix_key, prefix_value, attention_mask
            )
            hidden_states = layer_outputs[0]
        
        return hidden_states
```



- *提示微调* 是一种简化方法，其中可训练向量仅添加到初始输入嵌入序列中。像 P-Tuning 这样的变体进一步完善了这一思路。

附加方法的主要优势在于预训练知识（冻结权重）与任务特定适应（新参数）之间的明确分离。这种模块化简化了多任务学习和部署，因为可以按需加载不同的适配器或前缀，而无需修改大型基础模型。

### 选择性方法

选择性方法采取更直接的方法，即仅解冻并微调原始预训练模型参数中经过仔细选择的*一小部分*。其余参数保持冻结。

示例包括：

- 仅微调网络中的偏置项。
- 选择性地更新最后一层或对适应任务很重要的特定层。
- 像 FishMask 或 Diff Pruning 这样更复杂的技术尝试识别并训练对目标任务很重要的子网络。

尽管直观，选择性方法的有效性在很大程度上取决于识别出要微调的*正确*参数子集。这种识别可能不简单。虽然在训练过程中可能比完全微调的内存消耗少，但它们可能需要比附加方法或重参数化方法微调更多的参数才能达到相似的性能。此外，管理不同的任务适应需要存储修改后参数的单独副本或应用复杂的修补机制。

### 重参数化方法

重参数化方法修改了权重更新的表示或应用方式，而不是直接添加参数或选择子集。此类中最突出的技术使用低秩近似。

- **低秩适应 (LoRA)：** LoRA 基于这样的假设：适应所需的权重变化（ΔWΔ*W*）具有较低的“内在秩”。LoRA 不学习完整的 ΔWΔ*W* 矩阵，而是通过训练两个小得多的低秩矩阵 B*B* 和 A*A* 来近似它，使得 ΔW≈BAΔ*W*≈*B**A*。原始权重矩阵 W*W* 保持冻结，前向传播计算 h=(W+BA)x*h*=(*W*+*B**A*)*x*。训练期间，仅更新 B*B* 和 A*A*。这大幅度减少了可训练参数的数量，通常降至总数的不到0.1%。从数学上看，对于预训练权重矩阵 W0∈Rd×k*W*0∈R*d*×*k*，更新表示为： W=W0+ΔW=W0+BA*W*=*W*0+Δ*W*=*W*0+*B**A* 其中 B∈Rd×r*B*∈R*d*×*r*，A∈Rr×k*A*∈R*r*×*k*，且秩 r≪min⁡(d,k)*r*≪min(*d*,*k*)。

LoRA 在参数效率和性能之间实现了很好的平衡。低秩更新矩阵 B*B* 和 A*A* 非常紧凑。重要的是，一旦训练完成，更新 BA*B**A* 可以合并回原始权重中（W=W0+BA*W*=*W*0+*B**A*），与原始模型相比，消除了任何推理延迟开销。这一特性对于部署场景尤其有吸引力。