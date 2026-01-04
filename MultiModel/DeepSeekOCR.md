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