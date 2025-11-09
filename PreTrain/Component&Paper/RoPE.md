# RoPE

论文：[[RoFormer: Enhanced Transformer with Rotary Position Embedding]](./Paper/RoFormer Enhanced Transformer with Rotary Position Embedding.pdf)

绝对位置编码问题：训练阶段如果是2048个窗口位置，但当2049输入时，模型无法确定其位置关系。如果循环使用当作1来看，则可能造成信息损失或错乱。

绝对位置编码：学习的是相对位置之间的处理模式。如1和5，模型考虑的是4个单位的相对关系，到了2049和2053，模型考虑的依然是4个单位的相对关系。理论上可以无限延申，模型学习的是相对位置之间的模式关系。

![[RoPE]](./Image/RoPE.png)

这里将嵌入维度除以2，两两分组看作一个向量，对向量进行实部和虚部的换算。如：
$$
 X =  \begin {bmatrix} x_1, x_2, x_3, x_4 \end {bmatrix} \\
 这里将x_1, x_3看作向量实部，x_2, x_4 看作向量虚部
$$


![](D:\001-Coding\DATA\APX-LLM-Notebook\PreTrain\Component&Paper\Image\RoP计算原理简版.JPG)

```python
import torch
import torch.nn as nn


class RoPE(nn.Module):
    
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim%2 == 0 , "维度需要被2整除"
  
        i = torch.arange(0, dim, 2)
        theta = base ** (-i / dim)
        
        self.register_buffer('theta', theta)

        self.theta = base ** (-i / n_embd)
        self.register_buffer('theta', self.theta)

    def forward(self, X):
        positions = torch.arange(0, X.size(1))
        angles = positions.unsqueeze(-1) * self.theta.unsqueeze(0)
        print(angles.shape)

        cos_val = torch.cos(angles).unsqueeze(0)
        print(cos_val.shape)
        sin_val = torch.sin(angles).unsqueeze(0)
        print(sin_val.shape)

        X_real = X[..., 0::2]
        X_imag = X[..., 1::2]

        X_rotate_real = X_real * cos_val - X_imag * sin_val
        X_rotate_imag = X_real * sin_val + X_imag * cos_val

        X_rotate = torch.stack([X_rotate_real, X_rotate_imag], dim=-1)
        X_rotate = X_rotate.flatten(-2)
        
        return X_rotate
```



| 特性         | 第一种写法（正确）                                           | 第二种写法（错误）                                           |
| :----------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **逻辑**     | 先计算值，再注册。                                           | 先设为普通属性，再尝试注册为同名缓冲区。                     |
| **关键区别** | 使用局部变量 `theta`作为值传递给 `register_buffer`。         | 使用了已经存在的实例属性 `self.theta`作为值传递给 `register_buffer`。 |
| **结果**     | 成功创建一个名为 `'theta'`的缓冲区，可通过 `self.theta`访问。 | 因属性名冲突而导致程序报错。                                 |
| **推荐度**   | **✅ 推荐**                                                   | **❌ 错误**                                                   |

| 特性             | `self.theta = ...`（直接赋值）   | `self.register_buffer('theta', ...)`（缓冲区注册） |
| :--------------- | :------------------------------- | :------------------------------------------------- |
| **是否创建属性** | ✅ 是，创建普通Python属性         | ✅ 是，创建PyTorch缓冲区属性                        |
| **默认设备**     | 取决于赋值时的计算设备           | 初始时取决于传入张量的设备                         |
| **模型移动时**   | ❌ **不会自动移动**，需要手动处理 | ✅ **自动跟随模型移动**（CPU/GPU）                  |
| **模型保存时**   | ❌ 不会被保存到模型checkpoint     | ✅ 自动保存到模型checkpoint                         |
| **梯度计算**     | 不参与（与缓冲区相同）           | 不参与梯度计算（默认）                             |
| **推荐场景**     | 临时变量、配置参数               | 需要持久化、需要设备同步的张量                     |