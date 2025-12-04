# Lora & PEFT method

## PEFT方法分类

###  附加方法：注入新参数

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
- 前缀微调在注意力机制中只作用于KV矩阵，不对Q进行变化。

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

### 选择性方法：微调部分参数

选择性方法采取更直接的方法，即仅解冻并微调原始预训练模型参数中经过仔细选择的*一小部分*。其余参数保持冻结。

示例包括：

- 仅微调网络中的偏置项。
- 选择性地更新最后一层或对适应任务很重要的特定层。
- 像 FishMask 或 Diff Pruning 这样更复杂的技术尝试识别并训练对目标任务很重要的子网络。

尽管直观，选择性方法的有效性在很大程度上取决于识别出要微调的*正确*参数子集。这种识别可能不简单。虽然在训练过程中可能比完全微调的内存消耗少，但它们可能需要比附加方法或重参数化方法微调更多的参数才能达到相似的性能。此外，管理不同的任务适应需要存储修改后参数的单独副本或应用复杂的修补机制。

### 重参数化方法：修改权重更新

重参数化方法修改了权重更新的表示或应用方式，而不是直接添加参数或选择子集。此类中最突出的技术使用低秩近似（LORA）。

## LORA简介

论文：[低秩适应LoRA](./Papers/LoRA Low-Rank Adaptation of Large Language Models.pdf)

![](E:\DATA\LLM\APX-LLM-Notebook\PostTrain\Image\LoRA.png)

- A、B矩阵有一个初始化为0，有一个为高斯初始化，确保在第一次前向传播时，LoRA是无效的，有助于避免A、B均是随机初始化导致开始训练与原始模型有较大偏差。
- 1/r : 当数据经过低秩矩阵B，送入激活函数之前，会出现与1/r的线性相关的波动，可以使用1/r缩放因子来抵消数据波动；r还可以看作平衡参数，控制A、B的类别，r较大时，A、B看作全面但存在冗余信息，r较小时看作精炼但不全面信息。
- α：在训练LoRA时，α可看作模型对新知识的侧重程度
- 论文中，LoRA只应用于W~q~和W~v~。

预训练模型已经包含了大量的通用知识，当针对特定任务进行微调时，如情感分析或者代码生成，我们没有从根本上改变原有通用模型对相关任务的理解。而是通过微调来使其现有的能力适应特定的任务和模式。这种引导和适配增量的过程，可以通过在**高维权重空间**沿着相对较少的方向或维度修改原始权重来表示。

也就是说，通用模型是一个全才，但在特有的指示方向上并没有那么强势，这也就是其高维空间向量中数值表达较少的方向，这种理念又与SVD分解相似。通过分解得到低秩矩阵，减少学习参数。ΔW∈Rd×kΔ*W*∈R*d*×*k*
$$
W = W_0 + \Delta W \\
$$

$$
\Delta W ≈ BA
$$

$$
h = W_0x + \Delta W_0x = W_0x +BAx
$$

$$
缩放因子： \alpha \space {对BA施加更新幅度，使用 \frac{\alpha}{r}进行缩放，有助于稳定训练} \\
h = W_0x + \frac{\alpha}{r}BAx
$$

一般而言，使用随机高斯值初始化A，零初始化B。这是为了保证训练开始时BA为0，使得初始LoRA模型与原有预训练模型完全一致

- 参数量：r * （d + k)   r << min(d, k)，数量远小于微调w_0的 d * k参数量。增加r也会增加内存需求量
- 近似能力：r决定了BA矩阵秩的上限。更高的r允许BA近似更复杂的Δ*W* 。如果真实固有秩较高，r较小可能导致欠拟合。如果过高，可能通过捕捉噪声或虚假相关性导致数过拟合，并且增加计算成本。
- 选取r的方法：
  1. 测试，r=4，8，16等等值
  2. 计算开销，资源有限需要权衡
  3. 性能饱和，随着r的增加，会有临界值，达到之后性能会趋于平稳
  4. 任务复杂度
- 选取α。α过高更侧重LoRA的调整。加速适应目标任务，但可能产生对与训练阶段的遗忘；α较低，减小LoRA影响，更好的保留基础模型的能力，提高泛化能力
  1. 设定α=r
  2. 设定α为固定值，如16，32，64
  3. 将α视为独立的超参数。使用网格搜索、随机搜索或者贝叶斯优化进行系统调整。

LoRA层的实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    """
    将标准 Linear 层替换为LoRA层
    """
    def __init__(self, original_layer, alpha, rank, lora_drop_rate):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.original_layer = original_layer
        self.alpha = alpha
        self.rank = rank
        self.lora_drop_rate = lora_drop_rate
        
        # 将原始权重和偏置注册为不可训练的参数
        self.weight = nn.Parameter(original_layer.weight.detach().clone())
        self.weight.requires_grad = False

        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.detach().clone())
            self.bias.requires_grad = False
        else:
            # 使用 register_parameter 确保 'bias' 属性存在，即使它为 None
            self.register_parameter('bias', None)
            
        # 创建并初始化AB
        self.lora_A = nn.Parameter(torch.Tensor(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.Tensor(self.out_features, rank))
        
        # LoRA 路径的可选 dropout 层
        if lora_drop_rate > 0.0:
            self.lora_drop = nn.Dropout(p=lora_drop_rate)
        else:
            self.lora_drop = nn.Identity() # 作为一个直通层
        
        # 缩放因子
        if rank > 0:
            self.scaling = self.alpha / self.rank
        else:
            self.scaling = 1.0 # 如果秩为 0，避免除以零

        # 初始化 LoRA 参数
        self.reset_lora_parameters()
        
	 def reset_lora_parameters(self):
        """ 初始化 LoRA 矩阵 A 和 B。 """
        if self.rank > 0:
            # 使用 Kaiming uniform 初始化 A，以获得更好的梯度流动
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # 将 B 初始化为零，以便初始适应项为零
            nn.init.zeros_(self.lora_B)
            
	def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 执行修改后的前向传播。 """
        # 计算原始（不变的）线性变换
        # 使用 F.linear 可以避免在混合张量时出现设备放置问题
        result = F.Linear(x, self.weight, self.bias)

        # 如果 rank > 0，计算 LoRA 调整
        if self.rank > 0:
            # 在 LoRA 矩阵之前对输入 x 应用 dropout
            x_lora = self.lora_dropout(x)

            # 计算 x @ A^T
            # 输入 x_lora (N, d_in), 权重 lora_A (r, d_in) -> 输出 (N, r)
            after_A = F.Linear(x_lora, self.lora_A.T)

            # 计算 (x @ A^T) @ B^T
            # 输入 after_A (N, r), 权重 lora_B (d_out, r) -> 输出 (N, d_out)
            lora_adjustment = F.Linear(after_A, self.lora_B.T)

            # 将缩放后的 LoRA 调整添加到原始结果中
            result += lora_adjustment * self.scaling

        return result
    
    def train(self, mode: bool = True):
        """ 确保原始权重在训练期间保持不变。 """
        # 防御性重写父类中的train，确保self.weight不需要计算梯度
        super().train(mode)
        # 在模式更改后显式设置 requires_grad 为 False
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        # 确保 LoRA 参数可训练（它们默认是可训练的）
        # self.lora_A.requires_grad = True
        # self.lora_B.requires_grad = True

    def extra_repr(self) -> str:
        """ 向模块表示添加 LoRA 特定信息。 """
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'rank={self.rank}, alpha={self.alpha}')
```

LoRA也可应用于卷积层或嵌入层，实际使用中主要应用在Linear层中，尤其是**注意力机制及前馈网络**

### LoRA在Transformer中的应用

主要考虑LoRA在何处有效融合进Transformer架构。Trams former通常包含MHA以及FFN，这些依赖线性变换，是LoRA适应的主要目标。

常见调整MHA以及FNN

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank, alpha):
        """
        original_layer : 原始线性层
        rank ： 秩
        alpha ： 缩放规模
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # 冻结原始参数
        for p in original_layer.parameters():
            p.requires_grad = False

         # 创建LoRA矩阵A和B（作为Parameter，不是Linear层）
        self.lora_A = nn.Parameter(torch.Tensor(rank, original_layer.in_features))   # (r, d_in)
        self.lora_B = nn.Parameter(torch.Tensor(original_layer.out_features, rank))  # (d_out, r)

        # 初始化lora，A采用kaiming均匀分布，B使用0初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
	def forward(self, X):
        # 冻结参数传播结果
        result = self.original_layer(X)
        
        if self.rank > 0:
        # 更高效的计算：x -> A -> B，避免大矩阵
        lora_result = (x @ self.lora_A.t()) @ self.lora_B.t()  # x @ A^T @ B^T
        result += lora_result * (self.alpha / self.rank)
        
        return result

original_layer = nn.Linear(in_features=512, out_features=512)
lora_layer = LoRALinear(original_layer, rank=8, alpha=16)
```

---

补充：

```python
# nn.Linear 和 nn.Parameters的区别
# nn.Parameter是一个可训练的张量
self.lora_A = nn.parameter(torch.Tensor(rank, in_features))

# nn.Linear是完整的神经网络层
nn.Linear(rank, in_features)包含了weight和bias
```

## 提示词微调及P-Tunning

### 提示词微调

不改变模型内部权重，通过在提示词前加入可训练的前缀达到微调效果。可极大减少参数训练量。

```python
import torch
import torch.nn as nn

class BasicPromptTuning:
    def basic_process(self, original_input):
        """
        基本流程示例：
        原始输入: [我 爱 机 器 学 习]
        添加提示词: [P1 P2 P3 P4] + [我 爱 机 器 学 习]
        """
        
        # 1. 原始输入embedding
        input_embeddings = self.get_embeddings(original_input)  # [seq_len, hidden_size]
        
        # 2. 可学习的提示词（随机初始化）
        prompt_embeddings = self.learnable_prompts  # [prompt_len, hidden_size]
        
        # 3. 拼接
        combined_embeddings = torch.cat([prompt_embeddings, input_embeddings], dim=0)
        
        return combined_embeddings
```

#### 提示词初始化的影响

提示词嵌入P~j~的初始化方式可以明显影响训练稳定性和最终性能。常用策略有：

1. **随机初始化：** 使用小的随机值初始化向量，类似于其他网络权重的初始化方式。这种方法简单，但可能需要仔细调整学习率，并可能导致更长的收敛时间。
2. **词汇初始化：** 使用模型词汇表中与目标任务相关的特定词汇的平均嵌入来初始化提示词嵌入。例如，对于摘要任务，你可以使用“Summarize”、“TLDR”、“Abstract”、“Condense”等词汇的嵌入进行初始化。这可以为优化过程提供一个更好的起点。

### P-Tunning

#### P-Tunning-v1：引入提示词编码器

P-Tuning (v1) 观察到手动选择最优提示词的离散性很难实现，并且独立的提示词嵌入（如基础提示词微调中所示）可能缺乏表现力。它提出了两个主要方面：

1. **提示词编码器：** P-Tuning没有直接学习静态提示词嵌入**P~j~**，而是使用一个小型神经网络（通常是BiLSTM后接MLP），称为“提示词编码器”。该编码器将虚拟提示词标记序列作为输入，并动态生成实际的连续提示词嵌入。这使得提示词嵌入之间可以存在依赖关系，从而可能增加它们的表现力。
2. **锚定标记：** 有时会在输入中散布任务特定的锚定标记，以进一步引导模型。

然而，P-Tuning v1仍然主要集中在输入嵌入层周围进行修改，有时会出现优化不稳定的情况。

#### P-Tunning-v2：层级特定提示词

P-Tuning v2（常被称为深度提示词微调）通过将可训练的提示词嵌入应用于Transformer的*每个层*，而不仅仅是输入层，从而大幅增强了该方法。

在这种设置中，对于每个层l*l*，都会维护一组可训练的提示词嵌入**P~j~^(l)^**。这些嵌入会添加到进入该层的隐藏状态序列之前。这类似于Prefix Tuning，后者也在每个层注入可调参数。然而，P-Tuning v2通常只添加前缀*向量*，而Prefix Tuning通常涉及专门为注意力机制学习前缀键值对。

通过允许对模型内部计算在每个层进行直接影响，P-Tuning v2克服了浅层提示词微调的局限性，即初始提示词的影响可能在更深层消散。它在具有挑战性的NLU基准测试上表现出接近完全微调的性能，同时保持了高参数效率**（尽管略低于基础提示词微调的效率，但仍远高于LoRA或完全微调）**。它有效地为冻结的LLM提供了一种层级“引导机制”。



## 微调方法对比

| 微调方法              | 英文全称                   | 是否改变**原始模型**参数？ | 是否引入**额外**可训练参数？ | 核心思想简介                                                 |
| :-------------------- | :------------------------- | :------------------------- | :--------------------------- | :----------------------------------------------------------- |
| **全量微调**          | **Full Fine-Tuning**       | **是**                     | 否                           | 最经典的方法。用新数据在所有任务上更新模型**全部**参数。效果好，但成本极高，易发生灾难性遗忘。 |
| **SFT**               | **Supervised Fine-Tuning** | **是**                     | 否                           | SFT 指的是**任务/目标**，而不是具体技术。它通常使用“全量微调”作为实现手段，用有监督的数据（指令-回答对）来微调模型，使其遵循指令。 |
| **提示词微调/软提示** | **Prompt Tuning**          | **否**                     | **是**                       | 在输入层，将硬提示（人工设计的文本）替换为一段**可训练的软提示向量**（一组可学习的Token嵌入）。只训练这些向量，模型主体冻结。 |
| **P-Tuning**          | -                          | **否**                     | **是**                       | 提示词微调的增强版。为了解决软提示难以训练的问题，它通常引入一个轻量级的**提示编码器**（如LSTM/MLP）来生成这些软提示向量，训练的是这个编码器。 |
| **Prefix-Tuning**     | **Prefix-Tuning**          | **否**                     | **是**                       | 与P-Tuning类似，但不是只在输入加前缀，而是在模型的**每一层**（而不仅仅是输入层）都添加一组可训练的“前缀向量”。这些向量作为上下文，引导模型生成期望的输出。 |
| **Adapter Tuning**    | **Adapter Tuning**         | **否**                     | **是**                       | 在Transformer的每个模块（如FFN层）内部插入一个小的、拥有瓶颈结构的**前馈神经网络**（Adapter）。只训练这些Adapter，冻结模型主体。 |
| **LoRA**              | **Low-Rank Adaptation**    | **否**                     | **是**                       | 认为模型在微调过程中的参数更新量（ΔW）是**低秩**的。通过用两个小矩阵（A和B）的乘积来近似这个更新量 ΔW = A * B。训练时只训练A和B，然后将 ΔW 加到冻结的原始参数W上。 |

# 进阶LoRA实现方法

在之前建立的低秩适配（LoRA）基础理解之上，本章着重介绍高级实现方法和变体，以提高其性能、效率和适用性。我们将考察超出基础LoRA设置的技术，以应对实际LLM微调场景中遇到的具体问题。

## B、A初始化策略

### 默认初始化：B为零，A为高斯分布

最广泛采用且通常为默认的LoRA初始化策略是：

1. **初始化矩阵A\*A\*** 使用标准随机初始化方法，通常是Kaiming均匀或高斯分布（N(0,σ2)N(0,*σ*2)），且方差较小。这能确保投影到低秩空间的初始表示具有一定的变动性。
2. **初始化矩阵B\*B\*** 为全零。

**优点：**

- **稳定性：** 从ΔW=0Δ*W*=0开始能避免对预训练模型精心学习到的表示造成任何初始干扰。这通常会带来更稳定的训练动态，尤其是在初始阶段。
- **保留预训练知识：** 确保微调过程完全从基础模型的状态开始，没有来自LoRA层的任何初始随机扰动。

**缺点：**

- **初始学习可能较慢：** 由于B*B*从零开始，梯度信息需要回流以将B*B*更新为非零值，然后才能发生有意义的自适应。这可能会使收敛的初始阶段略微慢于ΔWΔ*W*从一开始就非零的方法。

这种策略在流行的库中默认实施，例如Hugging Face的PEFT (`peft`)。例如，`LoraLayer`通常将`lora_B`权重初始化为零，并使用Kaiming均匀初始化来初始化`lora_A`权重。

### 默认初始化：B为零，A为高斯分布

另一种方法是使用随机分布初始化矩阵A*A*和B*B*，通常是精心选择（通常较小）方差σ2*σ*2的高斯分布（N(0,σ2)N(0,*σ*2)）。

在这种情况下，当t=0*t*=0时，自适应项ΔW=BAΔ*W*=*B**A*将是一个非零矩阵，尽管如果σ*σ*较小，其项的值也可能较小。

**原理：**

这里的想法是，从小的非零随机自适应开始，可能让模型更快地学习到所需的调整，从而可能加速收敛。初始的随机ΔWΔ*W*提供了一个即时但可能带噪声的自适应方向。

**注意事项：**

- **方差选择（σ^2^）：** 这是一个重要的超参数。如果σ*σ*过大，初始的随机ΔW可能会显著干扰预训练权重W0，导致训练不稳定或立即性能下降。如果σ*σ*过小，其作用可能微不足道，并类似于将B*B*初始化为零。最佳方差通常需要调优。
- **与学习率的相互作用：** 与将B*B*零初始化策略相比，非零的初始ΔW可能需要更小的初始学习率以保持稳定性。

**优点：**

- **潜在的更快初始收敛：** 模型不需要“打破B=0的对称性”，并且如果随机初始化提供有用信号，则可能在初始步骤中更快地自适应。

**缺点：**

- **不稳定性风险：** 如果未适当缩放，初始随机ΔW可能会干扰预训练模型的功能。
- **对超参数的敏感性增加：** 需要更仔细地调优初始化方差（σ^2^）以及可能的学习率。

## 训练后合并LoRA权重

合并LoRA权重涉及计算有效权重更新 ΔW=αrBAΔ*W*=*r**α**B**A*，并将其直接添加到原始权重矩阵 W0*W*0。结果是一个新的权重矩阵 W~merged~，它包含了学到的适配：
$$
W_{merged}=W_0+ΔW=W_0+\frac{α}{r} BA
$$
计算完成后，W~merged~ 在模型层中替代原始权重矩阵 W0。使用这个特定合并模型进行推理时，不再需要独立的LoRA矩阵 A 和 B。该层随后像使用 W~merged~ 的标准层一样运行：
$$
h=W_{merged}x
$$
此计算在训练完成后离线进行。你为模型中每个经LoRA适配的层执行此计算。

### 合并优点

1. 部署简化，模型结构与原模型相同
2. 潜在推理加速，算法路径减少，直接进行推理

### 合并缺点

1. 灵活性丧失，合并操作不可逆，一旦合并，失去了在同一模型基础上切换不同适配器的能力。
2. 存储影响，lora适配器比基础模型小，存储大量小型适配器在存储效率上更显著。

# 量化LoRA（QLoRA）原理

本质是通过压缩原始大模型参数的存储空间，腾出空间来给lora训练。。。

虽然标准LoRA大幅减少了*可训练*参数的数量，但微调大型语言模型仍构成主要的内存难题。主要瓶颈通常不是适配器权重本身，而是加载庞大*基础模型*并进行计算所需的内存。即使模型被冻结，基础模型的权重（通常为FP16或BF16等16比特格式）也会占用大量GPU内存。此外，在前向和反向传播过程中计算的激活值会显著增加内存占用，这使得在没有高端多GPU配置的情况下，微调亿级参数模型通常不可行。

QLoRA（量化低秩适配）直接解决了这个内存瓶颈。它引入了一种技术，通过大幅减少基础模型的内存占用，而不显著牺牲性能来微调大型语言模型。核心思想是将预训练的基础模型加载为极低精度（通常为4比特）的量化权重，同时以更高精度格式（如BFloat16）训练LoRA适配器。

## 量化方法

### 4比特量化（Normal Float4）NF4

QLoRA的核心是NF4数据类型。与标准整数或浮点数量化方案不同，NF4专门为通常围绕零点正态分布的权重设计，这是预训练神经网络权重的一个常见特点。

量化映射，将F16压缩到4比特范围，在需要高精度计算时，再反量化映射回F16，会有一定的精度损耗，但大大缩小的存储空间。

#### **步骤1：分块**

- **输入**：一个一维的FP16张量 `W`（例如，有10000个元素）。
- **操作**：将其重塑为 `[num_blocks, block_size]`，例如 `[157, 64]`（因为 157 * 64 = 10048，可能需要填充）。

#### **步骤2：为每个块计算归一化范围**

1. **计算块的统计量**：对于每个块，计算其值的均值（μ）和标准差（σ）。在理想情况下，我们假设这些块来自一个均值为0，标准差为1的正态分布 *N*(0,1)。

2. **确定理论分位数**：NF4的“尺子”是预先定义好的，它基于 *N*(0,1)分布的理论分位数。这把尺子有16个等级（2^4=16），但它不是均匀的。等级的设置目标是让每个量化区间内的概率质量相等。

   - 具体来说，这16个等级 *q**i*对应于正态分布的以下分位数：

     [−1.0,−0.6962,−0.5251,−0.3949,−0.2844,−0.1848,−0.0911,0.0,0.0911,0.1848,0.2844,0.3949,0.5251,0.6962,1.0]

   - 注意：两端的边界被设置为-1和1，而不是无穷大，因为极端值概率极低，这样可以更好地覆盖99.99%以上的常见数值。

#### **步骤3：将块内的值映射到NF4等级**

1. **归一化当前块的值**：使用当前块的均值μ和标准差σ对值 `v`进行归一化（或至少是缩放），使其分布更接近 *N*(0,1)。
   - `v_normalized = (v - μ) / σ`(实际操作中可能简化，例如只除以一个绝对最大值)
2. **查找最近邻**：将 `v_normalized`与预定义的NF4等级数组进行比较，找到数值上最接近的那个等级索引 `index`（0到15）。
   - 例如，`v_normalized = 0.25`，它与等级 0.2844（索引11）和 0.1848（索引10）比较，更接近0.2844，因此 `index = 11`。

#### **步骤4：存储量化常数**

- **量化常数**：通常是一个**缩放因子（scale）** 和一个**零点（zero_point）**。

- **计算**：`scale = (max_value_in_block - min_value_in_block) / (max_quant_level - min_quant_level)`

  `zero_point = min_quant_level - (min_value_in_block / scale)`

  - 但在NF4的上下文中，由于等级是固定的非均匀值，实际操作会更复杂一些，可能需要一个优化步骤来找到最优的 `scale`和 `zero_point`，使得反量化后的值与原始值的误差最小（例如，使用最小二乘法）。

最终，这个块的权重在内存中被存储为：

- **4-bit 索引数组**：一个 `block_size`长度的数组，每个元素是4位，表示对应权重值的NF4等级索引。
- **量化常数**：一对FP16的数值（`scale`和 `zero_point`）。

**反量化的实际操作过程（前向传播时）**

当需要计算时，例如需要计算 `Y = W * X`：

1. **定位块**：根据要计算的元素，定位到对应的权重块。
2. **索引到值**：读取该块的4-bit索引数组。对于每个索引 `i`，通过查表操作，找到对应的NF4等级值 `q_value = NF4_LEVELS[i]`。这个 `NF4_LEVELS`就是步骤2中提到的那个固定的16个值的数组。
3. **反量化到目标范围**：使用该块的 `scale`和 `zero_point`，将 `q_value`转换回一个近似于原始FP16权重的值。
   - `dequantized_value = q_value * scale + zero_point`
4. **计算**：使用这个反量化后的 `dequantized_value`进行矩阵乘法等计算。

### 双重量化（DQ）

双重量化的目标非常明确：**进一步压缩存储“量化常数”所需的空间**。

在普通的4位量化中，对于每个数据块，我们都需要存储一个FP32（或FP16）的**量化常数**（通常是`scale`，有时还包括`zero_point`）。当模型很大、分块很多时，这些常数本身也会占据可观的内存。

双重量化的思路是：**将这些量化常数也进行量化**。

#### **第一级量化：对主权重进行4位量化**

这一步是标准流程：

1. 将FP16权重张量 `W`分成多个块（例如，每个块包含64个参数）。
2. 对每个块 `i`进行NF4量化，得到：
   - **4位权重索引** `W_int4(i)`
   - **FP32的量化常数**（例如缩放因子 `scale1(i)`）。

此时，`scale1(i)`仍然是全精度的。

#### **第二级量化：对量化常数进行8位量化**

1. 收集**所有**第一级量化产生的量化常数（即所有的 `scale1(i)`），形成一个“常数张量”。
2. 对这个“常数张量”**整体**进行**一次**8位量化（通常是均匀量化）。
3. 这次量化会产生：
   - **8位的常数索引** `scale_int8`
   - **一个第二级的、全局的FP32量化常数** `scale2`和 `zero_point2`（对于整个“常数张量”而言，只需要这一套常数）。

### 微调期间的运行流程

1. **加载与量化**

- 预训练的基础模型 $W$ 被加载，其权重立即被量化为 **NF4格式**，即 $W^{NF4}$。
- 使用**双重量化技术**压缩相关量化常数。
- 原始高精度权重被丢弃，释放大量 GPU 内存。
- $W^{NF4}$ 权重被**冻结**，训练期间不会更新。

2. **初始化适配器**

- **LoRA适配器**（$A$ 和 $B$ 矩阵）被添加到目标层（例如注意力层）。
- 适配器以更高精度格式（通常为 **BFloat16 (BF16)**）初始化。
- 这些适配器权重是**唯一将进行训练的参数**。

3. **前向传播**

处理输入 $x$ 时：

- 输入通过冻结的 $W^{NF4}$ 层传播。
- 对于包含 LoRA 适配器的层，4 比特基础模型权重 $W^{NF4}$ 的必要块会**即时反量化**回计算精度（BF16）。
- 输出计算为基础模型输出和 LoRA 适配器输出的总和：

  $$
  y = \text{反量化}(W^{NF4})x + \alpha \cdot xBA
  $$

  其中：
  - $\text{反量化}(W^{NF4})$ 表示反量化的基础模型权重块。
  - $B$ 和 $A$ 是 BF16 格式的 LoRA 矩阵。
  - $\alpha$ 是 LoRA 缩放因子。
- 矩阵乘法和加法运算在 **BF16 精度**下进行。

4. **反向传播**

- 计算梯度，但梯度**仅流经适配器权重 $A$ 和 $B$**。
- 梯度计算通常在 **BF16** 中进行。
- 由于$(W^{NF4})$ 被冻结，**不会为基础模型计算或存储梯度**，从而节省大量内存和计算资源。

5. **优化器更新步**

- 优化器（通常是 **AdamW**，可能是与分页优化器一起使用的 **8 比特 AdamW** 等内存高效变体）**仅更新 LoRA 适配器权重 $A$ 和 $B$**。
- 更新基于计算出的梯度进行。

### 分页内存优化

分页优化器的核心思想借鉴了计算机操作系统的**虚拟内存**或**分页机制**。当物理内存（RAM）不足时，操作系统会将暂时不用的数据从内存“换出”到硬盘上的“虚拟内存”空间，等需要时再“换入”内存。

分页优化器做的事情与此完全类似，只不过是发生在**GPU显存**和**CPU内存**之间。

# 优化、部署与实际考量

优化部分主要在于PEFT的基础设施要求、优化器、学习率、适配器等软硬件渠道，这部分想通过后续实践直接实操，理论太过枯燥，且不具有标准性。

## PEFT训练于推理的性能分析

### 性能指标

性能指标包括训练指标及推理指标。

**训练指标：**

- **GPU利用率：**衡量 GPU 计算单元的活跃程度。持续低利用率（例如，低于 70-80%）通常表明瓶颈在其他地方，例如数据加载或 CPU 处理。`nvidia-smi` 等工具提供实时视图。
- **内存使用：** 跟踪 GPU RAM 的消耗量。重要方面包括峰值内存使用（决定作业是否适合硬件）和平均使用量。PEFT 显著减少了*参数*内存，但激活和优化器状态（特别是使用 AdamW 等优化器时）仍占用大量内存。分析工具可以显示内存分配峰值。
- **训练吞吐量：**量化训练速度，通常以每秒样本数、每秒 token 数或每秒步数衡量。更高的吞吐量通常意味着更快的训练收敛和更低的计算成本。这通常由训练脚本报告，或可从日志中获得。 "**实际运行时间：** 训练周期或整个作业所花费的总时间。这受吞吐量、数据加载时间以及任何系统等待的影响。"
- **I/O等待时间：**等待从存储读取数据所花费的时间。大量的 I/O 等待可能导致 GPU 饥饿，从而降低利用率。系统级工具或框架分析器有时可以突出显示这一点。

**推理指标**：

- **延迟：** 处理单个推理请求（或一批）所需的时间。通常衡量从请求到达至响应生成的端到端时间。低延迟对于实时应用非常重要。
- **吞吐量：** 单位时间内处理的推理请求数量（例如，每秒请求数）。更高的吞吐量对于同时服务许多用户很重要。延迟和吞吐量之间通常存在权衡，尤其是在使用批处理时。
- **GPU 利用率（推理）：** 类似于训练，表明 GPU 在推理期间的使用效率。除非处理持续大量请求或大型批次，否则可能低于训练时的利用率。
- **GPU 内存使用（推理）：** 主要反映存储基础模型和活跃 PEFT 适配器所需的内存。对于 LoRA，合并权重消除了在推理期间存储单独 A 和 B 矩阵的需要，与动态适配器加载相比，略微减少了内存使用。

### 性能分析工具

- `nvidia-smi` (NVIDIA 系统管理界面)：

  一个命令行工具，提供 GPU 利用率、内存使用、温度和功耗的实时监控。非常适合在运行时进行快速检查和基本监控。

  ```bash
  # 每秒更新一次 GPU 状态
  watch -n 1 nvidia-smi
  ```

- **PyTorch 分析器 (`torch.profiler`)：** 直接集成到 PyTorch 中，此工具可分析训练或推理脚本中的 CPU 和 GPU 操作。它可以跟踪操作员执行时间、GPU 内核启动和内存分配事件。结果可以在 TensorBoard 或 Chrome 的 `chrome://tracing` 工具中轻松查看。

  ```python
  import torch
  from torch.profiler import profile, record_function, ProfilerActivity
  
  # 假设模型、数据加载器、标准、优化器已定义
  
  # 用于分析特定代码块的上下文管理器
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
               record_shapes=True, 
               profile_memory=True) as prof:
      # 使用 record_function 标记代码的特定部分
      with record_function("data_loading"):
          # 模拟或获取数据批次
          inputs = get_next_batch() 
  
      with record_function("model_forward_backward"):
          outputs = model(inputs)
          loss = criterion(outputs, labels) # 假设标签可用
          loss.backward()
  
      with record_function("optimizer_step"):
          optimizer.step()
          optimizer.zero_grad()
  
  # 打印按 CUDA 时间排序的聚合统计数据
  print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
  
  # 打印按 CPU 时间排序的聚合统计数据
  print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
  
  # 可选地导出跟踪数据以进行详细的时间线可视化
  # prof.export_chrome_trace("peft_train_trace.json") 
  ```

- **Python 分析器 (`cProfile`、`line_profiler`)：** 标准 Python 工具，可用于识别 CPU 密集型代码中的性能瓶颈，例如复杂的数据预处理或主模型执行之外的自定义逻辑。`line_profiler` 提供逐行计时信息，并要求修饰要分析的函数。

# 评估PEFT性能及局限性

## 常用指标

### 自然语言理解（NLU）任务的衡量指标

NLU任务通常包括分类、序列标注或问答。PEFT方法通常在GLUE（通用语言理解评估）或SuperGLUE等既有基准上进行评估，这些基准涵盖了多种此类任务。

#### 分类任务

对于情感分析、主题分类或自然语言推理等任务，目标是将标签分配给给定输入文本，可应用标准分类衡量指标：

- **准确率：** 最简单的衡量指标，表示正确预测的比例。尽管直观，但在数据不平衡的数据集上可能会产生误导。 $准确率=\frac{正确预测的数量}{预测总数}$
- **精确率、召回率和F1分数：**这些衡量指标提供了更详细的视角，尤其适用于不平衡的类别。
  - **精确率：** 衡量积极预测的准确性。$精确率=\frac{TP}{TP+FP}(其中TP = 真阳性，FP = 假阳性）$。
  - **召回率（敏感度）：** 衡量实际阳性中被正确识别的比例。$召回率=\frac{TP}{TP+FN}（其中FN = 假阴性）$。
  - **F1分数：** 精确率和召回率的调和平均值，提供了一个平衡两者的单一分数。$F1=2×\frac{精确率×召回率}{精确率+召回率}$。这通常是基准中分类任务的主要衡量指标。
- **马修斯相关系数（MCC）：** 即使对于不平衡的类别，也被认为是一种平衡的衡量方式，取值范围从-1（完全不一致）到+1（完美预测）。 $MCC=\frac{TP×TN−FP×FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}（其中TN = 真阴性）$。

```
真阳性：模型预测为垃圾邮件，而它确实是垃圾邮件。]
假阳性：模型预测为垃圾邮件，但它实际上不是垃圾邮件（即误伤了正常邮件）。
真阴性：模型预测为非垃圾邮件，而它确实不是垃圾邮件。
假阴性：模型预测为非垃圾邮件，但它实际上是垃圾邮件（即漏掉了垃圾邮件）。
```



评估PEFT时，我们使用这些衡量指标来比较例如LoRA适配器所达到的性能与完全微调的基础模型的性能。目标通常是在使用少得多的可训练参数的同时，达到接近完全微调的性能（例如，F1分数相差1-2个百分点内）。

#### 问答（QA）任务

对于抽取式问答任务（如SQuAD - 斯坦福问答数据集），答案是给定上下文中的一段文本，常用衡量指标包括：

- **精确匹配（EM）：** 衡量与真实答案完全匹配的预测百分比。这是一个严格的衡量指标。
- **F1分数：** 在词元级别计算，将预测和真实值视为词元集合。它衡量预测答案范围与真实答案范围之间的重叠度，为部分正确的答案提供部分得分。这通常被认为比EM更准确的衡量指标。

### 自然语言生成（NLG）任务的衡量指标

评估生成文本（例如，摘要、翻译、对话）本质上比评估NLU任务更为复杂，因为可能存在多个有效输出。衡量指标通常依赖于将生成文本与一个或多个参考文本进行比较。

- **BLEU（双语评估替补）：** 主要用于机器翻译，BLEU衡量生成文本与参考译文之间的n-gram精确率重叠度。它会惩罚过短的句子。分数越高表示与参考文本的相似度越好。
- **ROUGE（召回率导向的摘要评估替补）：**通常用于摘要任务，ROUGE衡量n-gram召回率重叠度。其变体包括：
  - **ROUGE-N：** 衡量n-gram的重叠度（例如，ROUGE-1用于unigram，ROUGE-2用于bigram）。
  - **ROUGE-L：** 衡量生成摘要与参考摘要之间最长公共子序列（LCS），捕捉句子级别的结构相似性。
- **METEOR（明确排序翻译评估指标）：** 也用于翻译和生成任务，METEOR考虑精确匹配、词干匹配、同义词匹配和释义，根据这些标准对预测和参考进行对齐。它包含对错误词序的惩罚。
- **困惑度（PPL）：** 一种内在评估衡量指标，衡量概率模型预测样本的优劣。困惑度越低，表示模型对测试数据越不感到“惊讶”，暗示着更好的语言建模能力。虽然在训练期间有用，但它并不总是与人类对下游任务质量的判断完全相关。

## 灾难性遗忘

灾难性遗忘 (CF) 是神经网络中一个常见现象，指模型在连续训练多个任务时，学习新任务后，突然丧失在之前任务上的表现。这发生的原因是，新任务所需的参数更新会覆盖用于记忆旧任务的参数。大型语言模型 (LLM) 在大量通用数据上预训练，在为特定下游任务进行微调时，遗忘这些基本知识是一个重要顾虑。完全微调会更新所有模型参数，尤其容易受到此问题的影响。了解不同的 PEFT 方法在缓解灾难性遗忘方面与完全微调相比表现如何，是评估这些方法时的一个主要考量因素。

参数高效微调方法在一定程度上是为了应对完全微调的计算负担而设计的，但其架构本身就可能对灾难性遗忘提供保护。主要原因有：

1. **冻结基础模型参数：** 大多数 PEFT 技术 (LoRA、适配器、前缀/提示微调) 保持绝大部分原始预训练模型权重冻结。微调仅影响少量新增或修改的参数。由于核心知识存在于数百万或数十亿个冻结参数中，因此不太可能被专注于少量可调参数的更新所清除。
2. **任务特定参数隔离：** 新引入的参数 (如 LoRA 矩阵 A 和 B，或适配器层) 是专门针对下游任务优化的。这将任务特定的适应与嵌入在基础模型权重中的通用知识隔离开来。在 LoRA 等方法中，更新 ΔW=BAΔ*W*=*B**A* 被添加到原始权重 W0*W*0，从而将原始功能与适应区分开来。
3. **有限的更新能力：** LoRA 等方法中更新的低维度 (由秩 r*r* 控制) 或适配器中的瓶颈维度，本身就限制了微调过程在整个模型行为上引发剧烈变化的能力。虽然足以适应新任务，但这种有限的能力可能不足以完全覆盖预训练期间学到的复杂、分布式表征。

## PEFT当前局限性

### 性能与完全微调的比较

尽管PEFT方法通常能以大幅减少的可训练参数实现非常接近完全微调的性能，但性能差距仍然可能存在，尤其是对于：

- **高度复杂的任务：** 需要复杂推理、多步骤逻辑或跨长上下文信息整合的任务，可能仍然能从完全微调的全局参数更新中获得更多益处。
- **大量知识更新：** 当目标是根本改变或向基础模型注入大量新事实知识时，PEFT方法（它们只修改一小部分参数）可能不如重新训练网络更大一部分有效。完全微调允许对模型的内部知识表示进行更广泛的调整。
- **极低参数预算：** 使用极少参数的方法（例如，使用极短提示的提示微调，或使用极低秩 r*r* 的LoRA）可能没有足够的容量来完全捕捉目标任务的细节，导致相比于具有更多可训练参数的方法或完全微调，性能上限较低。

研究继续寻求混合方法和对PEFT技术的改进（例如，在不同层之间改变秩，或结合不同的PEFT方法），以弥补这些剩余的性能差距，同时保留效率优势。

### 超参数敏感性和调整难度

PEFT方法引入了新的超参数，需要仔细调整才能获得最佳结果。这些参数包括：

- **LoRA：** 秩 (r*r*)、缩放因子 (α*α*)、目标模块（要适应哪些层）。
- **适配器微调：** 瓶颈维度、插入位置。
- **前缀/提示微调：** 前缀长度、初始化方法。

找到最佳组合可能不简单，并且通常需要大量实验，这可能会抵消训练期间获得的部分计算节省。此外，最佳超参数可能无法很好地泛化到不同的基础模型、数据集或任务上，对于新应用需要重新调整。更自动化超参数优化的策略（例如，使用贝叶斯优化等方法）或开发不那么敏感的PEFT变体，都是活跃的研究方面。

### 多个适配器的组合与干扰

在尝试组合多个PEFT模块时，例如同时使用多个LoRA适配器用于多任务学习或动态任务切换，会面临一个重要的实际问题。尽管适配器轻量级，但简单地加载多组权重可能导致：

- **参数干扰：** 像LoRA这样的加性方法会修改相同的基础权重。叠加多个LoRA更新 (W0+ΔW1+ΔW2) 可能导致不可预测的相互影响，或与单独使用每个适配器相比性能下降。
- **内存占用增加：** 尽管每个适配器都很小，但同时加载多个会增加推理时的内存使用。

研究正在寻求方法以实现更好的适配器组合，包括：

- 训练后有效合并适配器的方法。
- 用于任务特定适配器路由或门控的方法。
- 明确鼓励适配器正交性或最小化干扰的训练策略。

### 理解其潜在机制

尽管我们有功能性的实现和假设（例如LoRA的低秩假设），但对某些PEFT方法为何如此有效以及如何运作的深刻理论理解仍在发展中。重要的待解决问题包括：

- LoRA中的低秩更新具体捕获了哪些语言或功能方面？
- 学习到的前缀或提示究竟如何修改模型的内部表示和注意力模式？
- 为什么某些层（如注意力层）通常比其他层更适合作为PEFT的目标？
- 我们能否预先判断哪种PEFT方法最适合特定任务和模型架构？

开发针对PEFT的更好可解释性工具和理论框架，对于设计更有效和可靠的适应技术很重要。

### 适应范围：知识与风格

正在进行的研究关注PEFT引起的改变的*本质*。当前的证据显示，许多PEFT方法擅长适应模型的*风格*、*格式*或*任务特定行为*，但在根本性更新或注入*新事实知识*方面，可能不如完全微调有效。这种区别对于要求模型学习大量新信息与主要需要行为适应的应用很重要。研究旨在增强PEFT方法的知识注入能力。

### 量化相互影响

QLoRA显示了将PEFT与量化结合的潜力。然而，激进量化（例如4比特）与低秩更新之间的相互影响很复杂。潜在的问题包括：

- **误差累积：** 量化和低秩近似都引入误差。它们的综合效应可能比预期更大地降低性能。
- **最佳量化策略：** 标准量化技术应用于基础模型权重和PEFT更新的组合时是否最佳？为PEFT量身定制的量化方案可能会产生更好的结果。

需要进一步研究来理解这些相互影响，并制定将PEFT与各种量化方法稳定结合的最佳实践。

### 安全影响

PEFT的安全方面相对研究不足。待解决的问题包括：

- 与完全微调的模型相比，PEFT模型更容易或更不容易受到对抗性攻击或数据投毒？
- 适配器机制本身是否可能被用作新的攻击途径，例如通过注入恶意适配器？
- PEFT如何影响模型隐私以及提取训练数据的可能性？

随着PEFT的广泛应用，了解其安全特性将变得越来越重要。

### 缩放法则与可预测性

PEFT的有效性如何随模型大小、数据集大小和可训练PEFT参数数量的增加而缩放？为不同的PEFT方法建立可靠的缩放法则，将使实践者能够更好地预测新应用和更大模型的性能和资源需求，类似于预训练大型语言模型时观察到的缩放法则。

这些局限性与待解决的问题表明PEFT是一个活跃的方面。持续的研究不断改进现有方法，开发新途径，并更深刻地理解如何高效、有效地适应大型语言模型，用于各种下游应用。
