# DeepSeekOCR

[DeepSeekOCR](./Paper/DeepSeek-OCR:ContextsOpticalCompression.pdf)

## DeepEncoder

- **局部感知**(SAM-base)：采用窗口注意力机制，精准识别细节
- **16倍卷积压缩**：将4096个patch token压缩至256个，大幅减少计算量
- **全局语义理解**(CLIP-large)：提取整页文档的语义上下文信息

为解决上下文压缩的灵活性，需要满足以下特性的编码器：

1. 能够处理高分辨率；
2. 在高分辨率下激活率低；
3. 少量视觉令牌；
4. 支持多分辨率输入；
5. 参数量适中。

现有开源编码器无法满足上述条件，本文设计了一个编码器，命名为DeepEncoder。

编码器架构：SAM+CLIP

## Data Engine

DeepSeek-OCR 构建了复杂且多样的训练数据，包括 OCR 1.0 数据，主要包含传统的 OCR 任务，如场景图像 OCR 和文档 OCR；OCR 2.0 数据，主要包括复杂人工图像的解析任务，如常见图表、化学式和平面几何解析数据；通用视觉数据，主要用于为 DeepSeek-OCR 注入一定的通用图像理解能力，并保留通用视觉接口。

数据集构建：数据类型+位置+内容

- OCR 2.0 Data：

![](./Image/DeepSeekOCRData.jpg)

- OCR 2.0 Data

![](./Image/DeepSeekOCRData1.jpg)

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