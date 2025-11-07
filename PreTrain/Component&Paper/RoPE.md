# RoPE

论文：[[RoFormer: Enhanced Transformer with Rotary Position Embedding]](./Paper/RoFormer Enhanced Transformer with Rotary Position Embedding.pdf)

绝对位置编码问题：训练阶段如果是2048个窗口位置，但当2049输入时，模型无法确定其位置关系。如果循环使用当作1来看，则可能造成信息损失或错乱。

绝对位置编码：学习的是相对位置之间的处理模式。如1和5，模型考虑的是4个单位的相对关系，到了2049和2053，模型考虑的依然是4个单位的相对关系。理论上可以无限延申，模型学习的是相对位置之间的模式关系。

![[RoPE]](./Image/RoPE.png)