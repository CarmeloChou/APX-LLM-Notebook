# vLLM(Very Large Language Model serving)

> 由加州大学伯克利分校 Sky Computing Lab 发起、现由开源社区维护的**高性能大语言模型推理与服务引擎**，主要用于在生产环境中高效部署和运行 LLM（如 Llama、Qwen、Mistral 等）。

## 特性

| **特性**                              | **说明**                                                     |
| :------------------------------------ | :----------------------------------------------------------- |
| **PagedAttention**                    | 借鉴 OS 虚拟内存分页机制管理 KV Cache，按需分配显存、支持非连续存储和共享前缀，显存利用率可达 90%+，大幅减少碎片 |
| **Continuous Batching（连续批处理）** | 每个解码步动态将新请求加入 batch、已完成请求立即退出，避免等待整批结束，GPU 利用率显著提升 |
| **高性能内核优化**                    | 集成 FlashAttention、FlashInfer、CUDA Graph 等，支持INT4/INT8/FP8/AWQ/GPTQ 量化，降低显存提升速度 |
| **OpenAI 兼容 API**                   | 启动后提供与 OpenAI 一致的 `/v1/chat/completions`接口，可无缝替换原 OpenAI SDK 调用 |
| **广泛硬件与模型支持**                | 支持 NVIDIA/AMD GPU、部分 CPU 及华为昇腾等加速卡；兼容 HuggingFace 格式的主流 Transformer/MoE/多模态模型 |
| **分布式推理**                        | 支持张量并行、流水线并行等，可在多卡/多机上部署超大参数模型  |

## PagedAttention

### Block Table

传统大模型推理在自回归解码时，每生成一个新 token 都要对**历史所有 token 的 Key/Value 做 Attention**，因此必须把之前所有 token 的 K/V 存下来，这就是 **KV Cache**。传统实现会为每个请求**提前申请一整块连续显存**（比如按最大长度 2048），哪怕实际只用了 200 个 token，剩余显存既浪费又产生碎片，而且相同 system prompt 不能共享，并发一高显存就爆——这是第一个痛点。vLLM 提出的 **PagedAttention** 就是受操作系统虚拟内存启发，把 KV Cache 切成一个个固定大小的 **Block（页，比如每页存 16 个 token 的 K/V）**，这些 Block 在显存中**可以不连续存放**，每个请求维护一张 **Block Table** 做逻辑→物理映射：

```
逻辑 Token 位置：  0  1  2 ... 15 | 16 17 ... 31
                       ↓                ↓
Block Table:        Block#3  →  物理 Block@GPU_mem_7
                   (放 token 0~15)   (放 token 16~31)
```

示意：

```
┌─────────────── 请求 A 的 KV ───────────────┐
│ 逻辑块0 → 物理块 [███]  (显存地址 0x1000) │
│ 逻辑块1 → 物理块 [███]  (显存地址 0x4080) │ ← 不必连续
│ 逻辑块2 → 物理块 [███]  (显存地址 0x2200) │
└───────────────────────────────────────────┘
         ↑ Block Table 做映射
```

- **按需分配 Block**，不预开连续大块 → 显存碎片极小
- 不同请求可 **共享同一个物理 Block**（如相同 system prompt 存在 Block#3），引用计数 +1 即可
- 超长上下文靠追加 Block，而不是一次性预留

Attention 计算时，vLLM 按 Block Table 读出对应 K/V 送入 FlashAttention / FlashInfer 内核，**逻辑上等价于全量 KV，物理上却是分页存储**。

### Continuous Batching

第二个痛点是 **Static Batching**：传统推理要等一个 batch 所有请求都生成完才换下一批，短请求白白等最长请求。vLLM 用 **Continuous Batching（持续批处理）**，在 **每一个 decoding step（每生成一个 token）** 都重新组织 batch：

- 已完成请求立刻移出
- 新到达请求立刻加入
- 剩余请求继续留在 batch 里算

示意：

```
Step 1 decode: [Req A, Req B, Req C]
Step 2 decode: [Req A, Req B, Req D]   ← C 完了，D 进来
Step 3 decode: [Req A, Req E, Req F]   ← B 完了
```

这保证 GPU 矩阵乘始终接近满负荷，极大提升吞吐。

------

综合起来，一次生成流程是：

```
Prompt → Tokenize
  ↓
查共享 Prefix Block（命中则复用物理 Block）
  ↓
对已有 Block 做 PagedAttention
  ↓
采样下一个 token → 追加新 KV Block（按需分配）
  ↓
若未完成 → 下一 Step 重新组 Continuous Batch
  ↓
返回完整回答
```

### 处理不同用户请求的高并发

> **vLLM 中：每个序列一张 Block Table → 指向物理 KV Block；相同 prompt 前缀 → 不同请求指向同一物理 Block（引用计数）；Continuous Batching 只是每步选哪些序列参与计算，不改变 Block 映射关系。**

示意：

```
Req A Block Table:  [Phy#3, Phy#7, Phy#9]
Req B Block Table:  [Phy#3, Phy#8, Phy#11]
Req C Block Table:  [Phy#5, Phy#6]
```

- 一个请求可以用 **1～N 个 Block**（取决于已生成 token 数）
- 不同请求 **绝不共用 Block Table**
- 但 **可以共用表里指向的同一个物理 Block**

------

> **不同用户进来 → 各自新建 Block Table → 若输入 token 相同（通常是 prompt 前缀）→ 新请求 Block Table 直接指向已有物理 Block → 不重算、不额外占显存**

这就是 **Prefix KV Cache Hit / Copy-on-Write 共享**。

示意（System Prompt 相同）：

```
物理 Block #3  [You are a helpful assistant...]
        ↑                ↑
   Req A BT       Req B BT   ← 直接引用，ref_cnt = 2
```

只有当某个请求 **要继续生成新 token（超出共享前缀）** 时：

- 才分配新物理 Block
- 不影响其他请求的共享 Block

vLLM 判断能否共享是按：

```
hash( token_ids_of_prefix )
```

不是字符串相似、不是语义相同。

所以：

- `"你是一个助手"`vs `"你是一个 AI 助手"`→ ❌ 不共享
- 同一 tokenizer、同一 token list → ✅ 共享

Curio Court Block 1, 20 Ping Hong Ln, Ping Shan, Yuen Long, N.T., HongKong