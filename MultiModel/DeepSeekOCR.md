# DeepSeekOCR

[DeepSeekOCR](./Paper/DeepSeek-OCR:ContextsOpticalCompression.pdf)

## DeepEncoder

- **局部感知**(SAM-base)：采用窗口注意力机制，精准识别细节
- **16倍卷积压缩**：将4096个patch token压缩至256个，大幅减少计算量
- **全局语义理解**(CLIP-large)：提取整页文档的语义上下文信息

## DeepSeek-3B-MoE解码器

基于混合专家架构，拥有30亿参数但每次推理仅激活约5.7亿参数，实现“大模型能力，小模型效率”

## 多分辨率处理模式

DeepSeek-OCR提供5种智能分辨率模式：

- **Tiny模式**(64token)：适用于简单收据、票据
- **Base模式**(256token)：标准文档处理
- **Gundam模式**(可变token)：复杂技术文档，支持动态超高分辨率

## Segment Anything

[segment anything](./Paper/SegmentAnything.pdf)

![](./Image/segmentanything.jpg)

为了实现图像分割的基础模型（基础模型指那些在数据集外的数据仍有着良好表现的模型，这种能力通常是通过提示词工程实现，当使用互联网数据进行扩展和训练时，这些模型展现出惊人的zero-shot或者few-shot表现），本文从三个角度出发：

- 任务。什么样的任务可以让模型实现zero-shot
- 模型。相关的模型架构是怎么样的
- 数据。怎样的数据可以驱动这样的任务和模型