# RMSNorm

论文：[[《Root Mean Square Layer Normalization》]](./Paper/《Root Mean Square Layer Normalization》.pdf)

LayerNorm的计算开销对于层数较少的模型来说可能微不足道。但当模型更深和更大时，其计算开销变得更为明显。

RMSNorm性能表现可与LayerNorm抗衡，但其速度提升了大概7%-64%(取决于模型架构、硬件等等要素)。
$$
RMSNorm公式：\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}} \odot g
$$
论文指出，任何后续的**线性变换**（如全连接层 *Wx*+*b*）本身已经包含了**偏置参数 b**。这个偏置项足以学习并补偿掉输入数据的均值偏移。因此，在归一化阶段进行显式的、固定的中心化可能是冗余的。

求导过程：
$$
a = Wx;\\
s = RMS(a) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}{a_i}} = \sqrt{\frac{1}{n}a^Ta};\\ 
v = \frac{a}{RMS(a)};\\
y = g \odot v;
$$

$$
\begin{align*}
    \frac{\partial v}{\partial a} &= \frac{\partial}{\partial a}\left(\frac{a}{s}\right) \\
    &= I \cdot \frac{1}{s} + a \cdot \frac{\partial}{\partial a}\left(\frac{1}{s}\right) \\
    &= I \cdot \frac{1}{s} - \frac{a}{s^2} \cdot \frac{\partial s}{\partial a} \\
    &= I \cdot \frac{1}{s} - \frac{a}{s^2} \cdot \frac{1}{2} \cdot ({\frac{1}{n}a^Ta})^{-\frac{1}{2}} \cdot \frac{1}{n} \cdot  2a^T \\
    &= I \cdot \frac{1}{s} -  \frac{a}{s^2} \cdot s^{-1} \cdot \frac{1}{n} \cdot a^T \\
    &= \frac{1}{s} \cdot (I - \frac{aa^T}{ns^2})
\end{align*}
$$

$$
\frac{\partial y}{\partial v} = 
\begin{bmatrix} 
\frac{\partial y_1}{\partial v_1} & \cdots & \frac{\partial y_1}{\partial v_n} \\
\frac{\partial y_2}{\partial v_1} & \cdots & \frac{\partial y_2}{\partial v_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial v_1} & \cdots & \frac{\partial y_m}{\partial v_n}
\end{bmatrix} 
= 
\begin{bmatrix}
g_1 & 0 & \cdots & 0 \\
0 & g_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & g_n
\end{bmatrix} 
= \operatorname{diag}(g)
$$



----

## 补充：分子布局&分母布局

**1、分子布局的本质**：分子是**标量**、**列向量**、矩阵向量化后的**列向量**；分母是**标量**、列向量**转置**后的**行向量**、矩阵的**转置矩阵**、矩阵向量化后的列向量**转置**后的**行向量**。

**2、分母布局的本质**：分子是**标量**、列向量**转置**后的**行向量**、矩阵向量化后的列向量**转置**后的**行向量**；分母是**标量**、**列向量**、**矩阵自己**、矩阵向量化后的**列向量**。

思考一下，其实我们可以再简洁一些：**谁转置了，就是另一方的布局**。分子转置了，就是分母布局；分母转置了，就是分子布局。

![矩阵求导](.\Image\矩阵求导.jpg)不严谨地说，分子布局是分子为列向量，分母布局是分母为列向量。