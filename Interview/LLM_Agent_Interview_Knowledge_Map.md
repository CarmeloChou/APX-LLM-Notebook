# LLM 与 Agent 面试复习：知识框架、问题与答案

> 面向 LLM 应用算法 / Agent 工程师岗位。  
> 阅读顺序：先看每章的知识框架建立地图，再用“问题—答案”训练口述。答案默认是 60～90 秒面试版本；追问用于继续展开。

---

# 一、总框架：从模型到可靠产品

```text
文本与数据
  → Tokenization 与向量表示
  → Transformer 与语言建模
  → 预训练
  → 后训练与对齐
  → 推理、部署与性能
  → RAG、数据库、知识图谱
  → Tool、Skill、Agent、协议
  → Harness、评估、安全与产品工程
```

面试主线可以概括为：

> LLM 把文本转换为 token，并用 Transformer 预测下一个 token；预训练赋予通用语言和知识能力，后训练使它更会遵循指令。实际业务中，动态且需核验的知识通过 RAG、数据库和知识图谱提供，确定性操作由工具执行。Agent 负责在目标和反馈下规划、选择能力与综合结果，Harness 则用状态、Schema、权限、验证、重试和观测把它约束成可交付的产品。

---

# 二、Tokenization 与表示

## 知识框架

```text
文本
  → tokenizer（BPE / WordPiece / Unigram）
  → token IDs
  → embedding lookup
  → 向量序列
```

需要掌握：子词切分、特殊 token、词表与模型匹配、chat template、token 成本、生成模型 embedding 与检索 embedding 的区别。

## 问题 1：为什么 LLM 需要 Tokenizer？

**答案：**

模型只能做数值计算，不能直接处理字符串。Tokenizer 把文本拆成 token 并映射为整数 ID，随后模型通过 embedding 表把 ID 查成向量。子词级 Tokenizer 在词表规模、序列长度和未登录词之间做了折中：高频片段是完整 token，低频词可以由更小片段拼出来。因此它既比字符级序列短，又不像词级切分那样容易出现 OOV。

**追问：中文一个字是否一定是一个 token？**

不一定。取决于词表和算法。常见汉字可能是单 token，罕见字符、组合词、标点或 UTF-8 字节序列可能对应多个 token。工程上要以实际 tokenizer 计数，因为 token 数决定上下文容量和调用成本。

## 问题 2：为什么 Tokenizer 必须与模型权重匹配？

**答案：**

token ID 本质是 embedding 矩阵的行索引。模型训练时已经学会“某个 ID 对应某种向量和使用模式”。更换 tokenizer 会改变文本到 ID 的映射，模型看到的输入语义就错位了。新增 token 时还必须扩展输入 embedding 和输出语言模型头，并通过训练让新增行获得可用表示。

## 问题 3：生成模型的 embedding 和 RAG embedding 有什么区别？

**答案：**

生成模型内部 embedding 服务于逐 token 预测，不一定适合直接做文本语义检索。RAG embedding 模型通常用对比学习训练，让 query 与相关文档更接近、与无关文档更远，因此更适合向量召回。两者都叫向量，但训练目标和使用位置不同。

---

# 三、Transformer 与语言建模

## 知识框架

```text
Decoder-only Transformer
  token embedding + position information
    → Transformer Block × N
       ├─ causal self-attention
       ├─ residual / normalization
       └─ FFN
    → LM head
    → vocabulary logits
    → sampling next token
```

关键词：Q/K/V、缩放点积注意力、causal mask、多头注意力、GQA、RoPE、FFN、SwiGLU、RMSNorm、残差、MoE、长上下文。

## 问题 1：请解释 Self-Attention 的 Q、K、V。

**答案：**

Self-Attention 让每个 token 根据当前上下文，从其他 token 聚合相关信息。输入向量 X 分别经过三组线性投影得到 Q、K、V：Q 可以理解为当前位置想查找什么，K 是每个位置提供的匹配索引，V 是该位置真正携带的内容。用 Q 和所有 K 的点积算相关性，归一化成权重后，对 V 加权求和。这样一个 token 就能动态吸收与自己相关的上下文信息。

公式的纯文本写法：

```text
Q = X × Wq
K = X × Wk
V = X × Wv
Attention = softmax((Q × K^T) / sqrt(dk)) × V
```

**追问：为什么除以 sqrt(dk)？**

维度变大时，点积的数值范围和方差会变大，Softmax 容易过度尖锐，导致梯度不稳定。除以 `sqrt(dk)` 是尺度归一化。

## 问题 2：为什么 Decoder-only 模型需要 Causal Mask？

**答案：**

语言模型的训练目标是根据前文预测下一个 token。如果当前位置能看到未来 token，就发生信息泄漏：训练损失很低，但推理时未来内容不存在。Causal Mask 会把未来位置的 attention score 屏蔽掉，因此第 t 个位置只能看自己和之前的 token，训练与自回归生成保持一致。

## 问题 3：Attention、FFN 和残差连接各自做什么？

**答案：**

Attention 负责 token 之间的信息交互，例如长距离依赖和指代关系；FFN 对每个 token 独立做非线性特征变换，可以理解为在 token 内部加工特征。残差连接让层学习对原表示的增量修改，保留已有信息并改善深层网络的梯度传播。现代 LLM 的大量参数通常在 FFN 中，而注意力决定上下文如何通信。

## 问题 4：RoPE、MHA、MQA、GQA 分别解决什么问题？

**答案：**

Attention 本身不感知顺序，RoPE 通过对 Q/K 做与位置相关的旋转，把相对位置信息自然地反映到注意力分数中。MHA 每个头都有独立的 Q/K/V，表达力强但 KV cache 大；MQA 让多个 query 头共享一组 K/V，大幅省显存；GQA 按组共享 K/V，在质量和推理成本之间折中，因此常用于现代模型。

## 问题 5：MoE 为什么能“参数很大但计算不那么大”？

**答案：**

MoE 通常把 FFN 替换为多个专家网络，router 为每个 token 只选择 top-k 个专家执行。总参数量可以很大，但单个 token 激活的参数有限，因此 FLOPs 不随总专家数线性增长。难点是专家负载不均、跨设备通信和路由不稳定，所以要有负载均衡损失和专家并行设计。MoE 不必然意味着端到端延迟更低。

---

# 四、预训练、微调与对齐

## 知识框架

```text
Pretraining：学习通用文本分布
  → SFT：学习指令和对话格式
  → Preference optimization / RL：学习偏好与安全边界
  → 可选领域适配：Full FT / LoRA / QLoRA
```

## 问题 1：预训练到底在训练什么？

**答案：**

以自回归 LLM 为例，训练目标是预测下一个 token，即最大化 `P(x_t | x_1 ... x_(t-1))`。训练时使用 teacher forcing：预测当前位置时前文使用真实 token，因此可以并行计算所有位置的损失。海量且多样的数据使模型学到语言规律、世界知识和部分任务模式；但它本质仍是概率预测器，并不保证事实始终正确。

## 问题 2：预训练数据为什么如此重要？

**答案：**

模型能力不仅取决于参数规模，也强烈受数据质量、覆盖和配比影响。数据管线一般包括解析、质量过滤、去重、隐私和安全处理、语言/领域识别、配比采样和 tokenization。比如提高代码或数学数据比例会改变相应能力；重复和低质量数据会浪费计算并增加记忆、污染和过拟合风险。Scaling law 的工程含义是，在固定算力下模型大小和训练 token 必须合理配比，而不是只堆参数。

## 问题 3：SFT、RLHF、DPO 的区别？

**答案：**

SFT 用高质量“指令—回答”样本继续做交叉熵训练，让基础模型学会助手的格式和任务响应。传统 RLHF 先训练 reward model 表达人类偏好，再用 PPO 优化策略，同时通过 KL 约束避免偏离参考模型过多。DPO 直接用 chosen/rejected 偏好对优化相对概率，不显式训练 reward model，也不需要完整 PPO，工程更简单、通常更稳定。无论哪种方法，偏好数据质量都是关键上限。

## 问题 4：LoRA 为什么有效？何时不用 LoRA？

**答案：**

LoRA 假设下游适配所需的权重更新可以由低秩矩阵近似：冻结原权重 W，只训练低秩增量 `ΔW = B × A`。这样训练参数、优化器状态和显存都显著减少。它适合风格、输出格式、稳定的领域任务和轻量适配；但对于频繁变化、需要来源可追溯的事实知识，更推荐写入数据库或 RAG。否则知识难更新、难引用，也可能产生“记住但说错”的幻觉。

---

# 五、推理与模型服务

## 知识框架

```text
请求 → Prefill（处理输入）→ KV Cache
     → Decode（逐 token 生成）→ 采样 → 流式输出
```

关键词：TTFT、TPOT、KV cache、continuous batching、PagedAttention、prefix cache、量化、吞吐与延迟。

## 问题 1：Prefill 和 Decode 有什么区别？

**答案：**

Prefill 一次性处理用户输入和已有上下文，为每层计算并缓存 K/V；它并行度高，通常偏计算密集，并决定首 token 时间 TTFT。Decode 阶段每次只生成一个 token，复用历史 KV cache，但因为逐 token 串行、需要频繁读写 cache，常更受显存带宽限制，并决定每 token 时间 TPOT。优化服务必须分别看这两阶段，不能只看平均 tokens/s。

## 问题 2：KV Cache 为什么重要，又为什么麻烦？

**答案：**

生成新 token 时，历史 token 的 K/V 不会变化，缓存它们就避免重复计算历史 attention。代价是 cache 随层数、上下文长度、KV heads、head dimension、精度和并发增长，占用大量显存。因此 GQA、PagedAttention、prefix cache、上下文控制和请求调度都与 KV cache 管理密切相关。

## 问题 3：temperature、top-k、top-p 怎么理解？

**答案：**

模型先输出词表 logits。temperature 改变概率分布的尖锐程度：低温更确定，高温更多样。top-k 只在概率最高的 k 个 token 中采样；top-p 选择累计概率达到 p 的最小候选集，候选规模会随分布变化。事实问答和结构化输出一般用较低随机性；创意写作可以更高。它们无法弥补检索不足或事实错误。

---

# 六、RAG、SQL 与知识图谱

## 知识框架

```text
文件解析/OCR → 保留结构与元数据 → chunk → index
query → 改写/分解 → hybrid retrieval → rerank
→ context construction → 带引用回答 → 分层评估
```

## 问题 1：RAG 的核心难点是什么？

**答案：**

RAG 的难点不是接入向量库，而是确保模型拿到的证据正确、完整、可追溯。上游要可靠解析 PDF、表格和标题层级，并保留文件、页码、版本等元数据；中间要选择合适 chunk 粒度、混合召回和重排；下游要按 token 预算构建上下文、处理重复和冲突，并把结论绑定到引用。若只拼接 top-k 文本，往往会出现召回错、上下文噪声大、模型忽略证据或引用不支持结论的问题。

## 问题 2：BM25、向量检索、Rerank 如何分工？

**答案：**

BM25 擅长专有名词、标题、编号和精确关键词；向量检索擅长同义表达和语义相似。通常先用 hybrid retrieval 做高召回，再把几十个候选交给 cross-encoder reranker 精排到少量证据。Cross-encoder 精度高但成本高，不适合直接扫全库。参数如 top-k、融合权重和 chunk 规模应通过标注评估集调优。

## 问题 3：SQL、Vector DB、Graph DB 的能力边界？

**答案：**

SQL 适合精确过滤、聚合统计、事务和稳定结构化属性；向量库适合非结构化文本的语义近似检索；图数据库适合多跳关系、邻居扩展和路径问题。在产业研究里，企业、政策、地区等实体属性和统计放 SQL，政策原文段落放向量索引，产业链—企业—地区—项目之间的关系放图中。它们互补，不是替代关系。

## 问题 4：什么才算 GraphRAG？

**答案：**

不是把数据放进图数据库就叫 GraphRAG。图必须实际参与检索或推理，例如先识别 query 中的实体，再扩展相关邻居和多跳路径；或基于图社区生成摘要；或用关系约束过滤候选证据。图谱设计首先要服务任务：定义实体、关系、别名、唯一 ID、来源和置信度，并处理实体消歧。

---

# 七、Tool、Skill、Agent、Workflow 与 Harness

## 知识框架

```text
Agent：理解目标、选择、规划、综合
Skill：面向任务的方法包和约束
Tool：确定性、可测试的原子操作
Workflow：固定且可预测的流程
Harness：状态、Schema、权限、预算、验证、重试、观测
```

## 问题 1：Agent、Skill、Tool、Harness 的区别？

**答案：**

Tool 是原子执行能力，例如数据库查询、网页搜索、计算、渲染 HTML，应该有明确输入输出并可独立测试。Skill 是完成一类任务的方法包，包含适用条件、领域指令、可用工具、步骤和输出契约，例如“政策分析 Skill”。Agent 是以 LLM 为决策器，在目标和环境反馈下选择 Skill、规划、调用工具并决定是否结束。Harness 是运行控制层，负责状态管理、Schema 校验、权限、超时、重试、预算、trace 和评估。工业系统的核心不是让模型做更多，而是让模型只做不确定性判断，让系统接管确定性控制。

## 问题 2：Function Calling 的真实执行流程是什么？

**答案：**

LLM 不会真的执行函数。它根据工具定义输出结构化调用意图，例如工具名和 JSON 参数；runtime 先校验 schema 和权限，再调用真实 API 或函数，然后将类型化结果或错误返回给模型作为 observation。完整链路是：模型意图 → 参数验证 → 权限/限流 → 工具执行 → 结果标准化 → 重试或降级 → trace。工具描述和参数 schema 会直接影响模型选择正确工具的概率。

## 问题 3：为什么企业 Agent 常用 Workflow 外壳，而不是完全自主？

**答案：**

完全自主的 ReAct 灵活，但容易循环、偏离目标、成本不可控且难复现。报告生成、审批、数据处理等任务往往有稳定步骤，更适合代码化 workflow；把需要语义判断的局部，例如检索路由、研究计划调整、证据综合，交给 Agent。这样既保留适应性，又能提供状态机、重试、审计和可测性。多 Agent 只有在角色真的独立、可并行且接口清晰时才值得引入。

## 问题 4：Reflection 怎样才有效？

**答案：**

“请检查一下”太开放，效果不稳定。Reflection 应转为验证器：关键字段是否完整、每个主张是否有引用、引用是否支持主张、证据是否冲突、是否超过 token/时间预算。并设置明确触发条件、最大迭代、重复 observation 检测和停止条件。Schema 能保证格式，不保证事实；事实性仍需证据校验和评估集。

## 问题 5：MCP 和 A2A 的边界？

**答案：**

Function calling 是模型表达工具调用意图的格式。MCP 标准化 Host/Client/Server 之间暴露和使用 tools、resources、prompts，重点是模型应用与外部能力的连接和复用，不负责高级规划。A2A 更关注 Agent 之间如何声明能力、分配任务、同步状态、传递消息和 artifact，适用于协作和长任务。简言之：MCP 偏“接工具和上下文”，A2A 偏“Agent 协作”。

---

# 八、Schema、可靠性、安全与评估

## 知识框架

```text
Intent Schema → Plan Schema → Tool I/O Schema
→ Evidence Schema → Report Schema → Rendering Schema

每一层：validate → observe → retry/fallback → trace → evaluate
```

## 问题 1：多层 Schema 能解决什么，不能解决什么？

**答案：**

多层 Schema 把开放生成逐层收敛：意图 schema 约束任务输入，plan schema 约束步骤，tool schema 约束执行，evidence schema 绑定主张与来源，report schema 约束交付物，rendering schema 让展示与内容分离。它能提高格式稳定性、可解析性、测试性和故障恢复能力；但不能自动保证事实正确、证据充分或结论合理。因此还需要 retrieval verification、规则校验、引用检查、人审和端到端评估。

## 问题 2：Agent 系统如何防止幻觉和越权？

**答案：**

对事实问题，要求答案基于 evidence package，并在缺证据时明确不确定性；对动作问题，模型只提出意图，工具层实行最小权限、allowlist、参数化查询、读写隔离、超时和人工确认。网页、文档和工具输出都视为不可信输入，防范间接 prompt injection。还要限制最大步数、token、时间和成本，并完整记录工具调用审计日志。

## 问题 3：怎么评估一个 Agent，而不是凭感觉？

**答案：**

先建立覆盖真实任务、歧义、缺数据、冲突信息、长任务和恶意输入的 golden set，再分层测量。检索看 Recall@K、NDCG 和引用覆盖；工具看选择和参数准确率、成功率；Agent 看任务成功率、平均步数、循环率与恢复率；报告看事实、引用、完整性和一致性；系统看 P95 延迟、单位成本和失败率。每次失败应能通过 trace 定位到解析、检索、规划、工具或生成环节。

---

# 九、结合产业研究 Agent 的项目答法

## 知识框架

```text
源文件 / 网页 / 数据表
  → Object storage + PostgreSQL + Vector index + Graph
  → 查询与采集 Tools
  → 产业研究、政策、企业、案例 Skills
  → Research Agent（产出 evidence package）
  → Report workflow（生成、审查、HTML 渲染）
  → Web 产品、任务状态与历史报告
```

## 问题：请介绍你的 Agent 项目架构和关键取舍。

**答案模板：**

我做的是面向产业研究和募投报告的 Agent 系统。底层保留源文件和页码等追溯信息；结构化实体和统计放 PostgreSQL，非结构化段落用向量检索，产业链和企业关系用图谱表示。系统不是让一个模型直接写完整报告，而是先由 Research Agent 将需求解析为受 Schema 约束的研究计划，调用政策、企业、案例和图谱等工具，形成每条结论都可追溯的 evidence package。之后 Report workflow 按章节填充报告 Schema，先做引用和完整性校验，再由确定性 HTML 模板渲染。

我的关键取舍是 Workflow 外壳加局部 Agent 路由：数据库查询、渲染和校验保持确定性；模型主要负责需求理解、检索选择、信息综合和必要的重新规划。这样可以控制幻觉、成本和可观测性。对于未集成或无法验证的信息，系统会明确返回知识缺口，或在授权下调用外部检索，而不是编造结论。

---

# 十、八周复习与输出清单

| 周 | 主题 | 面试输出 |
|---|---|---|
| 1 | Tokenizer、embedding、attention、mask | 不看资料讲清 token 到 logits |
| 2 | RoPE、GQA、FFN、Norm、MoE | 讲清现代 LLM block 与取舍 |
| 3 | 预训练、SFT、LoRA、DPO/RLHF | 训练与对齐对比表 |
| 4 | Prefill、Decode、KV cache、量化、服务 | 能解释慢、显存不足、并发低 |
| 5 | RAG、hybrid、rerank、引用与评估 | 一份自己的检索链路与失败案例 |
| 6 | KG、GraphRAG、SQL/Vector/Graph | 产业 ontology 与真实查询 |
| 7 | Tool、Skill、Agent、MCP、Harness | Tool/Plan/Evidence 三份 Schema |
| 8 | 产品化、安全、评估、项目演练 | 三分钟项目介绍和录屏 demo |

## 最终自检

- 能否解释一个 token 从输入到输出概率的完整路径？
- 能否区分预训练、微调、RAG 和工具调用分别解决什么问题？
- 能否说明 KV cache 为什么影响长上下文与并发？
- 能否指出 RAG 的召回错误、重排错误、上下文错误和生成错误？
- 能否清楚划分 Tool、Skill、Agent、Workflow、Harness、MCP、A2A？
- 能否给出带权限、验证、重试、预算和 trace 的工具调用链？
- 能否用三分钟讲清自己的项目问题、架构、取舍、结果和局限？

