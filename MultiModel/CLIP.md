# CLIP

Contrastive Language-Image Pre-Training[Learning Transferable Visual Models From Natural Language Supervision.pdf](./Paper/Learning Transferable Visual Models From Natural Language Supervision.pdf)

大概构思一下设计理念来源。在文本可以通过注意力进行上下文学习的时候，如何打通图片和文本的桥梁？是否能够通过现有的ins图片+描述来进行训练？这种数据是天然的标注数据，如果可以，那么就能够打通二者之间的壁垒，进行文生图或者图生文操作。

![](E:\DATA\LLM\APX-LLM-Notebook\MultiModel\Image\CLIP.png)

CLIP 是一个通过**对比学习**在海量互联网图文对上训练的**多模态模型**，它能将任意图像和文本映射到**同一语义空间**，实现强大的**零样本**跨模态理解能力。

- 数据：互联网数据对

- 让匹配的（图像，文本）对在向量空间中靠近，不匹配的推远

- 损失函数：对称交叉熵损失（InfoNCE）

- 架构设计

  ```python
  # 双编码器架构
  图像编码器（ViT或CNN） → 图像特征向量
  文本编码器（Transformer）→ 文本特征向量
  
  # 对比学习
  相似度 = 矩阵乘法(图像特征, 文本特征.T)
  # 目标：对角线相似度高，非对角线相似度低
  ```

**CLIP的本质是建立了一个视觉和语言的“翻译系统”**，让计算机能像人一样，看到图片就想到对应的描述，看到文字就能想象对应的画面。

InfoNCE损失函数：其中sim为余弦相似度

$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(\mathbf{q}, \mathbf{k^+}) / \tau)}{\sum_{i=1}^{N} \exp(\text{sim}(\mathbf{q}, \mathbf{k_i}) / \tau)}$

$\mathcal{L} = -\log \left[ \frac{\exp(\operatorname{sim}(\mathbf{q}, \mathbf{k}^+) / \tau)}{\exp(\operatorname{sim}(\mathbf{q}, \mathbf{k}^+) / \tau) + \sum_{i=1}^{N-1} \exp(\operatorname{sim}(\mathbf{q}, \mathbf{k}_i^-) / \tau)} \right]$