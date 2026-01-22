# 学习路线

源自知乎@CodeCrafter的回答。https://www.zhihu.com/question/1893668410774254363/answer/1953423901766944177

**现在的Ai agent课程很糟**，因为这玩意太新了，根本没有形成成熟的，可以被称之为课程的知识体系。

```
现在的技术栈，每周都在变。今天LangChain发个新版本，明天LlamaIndex就重构了核心模块，后天OpenAI又给你整个Function Calling v2。那些录播课，讲师录完的时候技术都可能是过时的。他们教的，顶多算是对某个时间点上，某个框架的API文档的“视频翻译”。这玩意儿的核心，压根就不是调API。调API一个实习生学一天就会了。真正的难点，那些课程根本没讲，或者说讲师自己也没搞明白。

作者：CodeCrafter
链接：https://www.zhihu.com/question/1893668410774254363/answer/1953423901766944177
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

## 阶段一：打好地基

对LLM本身要有体感。要不然就像没学过C语言就要直接写操作系统内核。

- Prompt Engineering：本身不是玄学，是一门手艺。Few-shot，Chain of Thought、self-consistency基本功必会。**《ChatGPT Prompt Engineering for Developers》**非常好的入门课程，一下午就可以过完。

- RAG：这是目前90%以上Agent应用的核心。你必须搞懂RAG的全流程：文档加载(Loading)、切分(Splitting)、向量化(Embedding)、存储(Storing)、检索(Retrieving)。每个环节用什么技术，有什么优缺点，你心里得有数。比如，文本切分按固定长度切、按句子切、还是按markdown标题切，效果天差地别。检索也不是只有向量相似度，多路检索、混合检索、重排(rerank)这些都得懂。如果你想搞清楚大模型怎么跟外部知识库结合、怎么在实际业务里用起来，字节RAG实践手册就是很好的参考。下载地址：[字节跳动6万字RAG 实践手册流出...](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/ns0HSA4Mp_TtsvAajoehVA)（纯free，自由获取）**它不是只讲概念，还从工程实践的角度拆了整个流程：数据怎么准备、知识库怎么构建、检索和生成怎么配合、性能和延迟怎么优化**。

- 微调：知道什么时候该用微调，什么时候不该用。别动不动就说“我拿我的业务数据fine-tune一个模型”，成本高，效果还不一定好。多数情况下，一个好的RAG系统比一个草率的微调模型效果好得多。想要搞懂微调，可以看看[字节内部大模型微调实践手册.pdf](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/ZLq0kH7Qdt9sWgTBXkr96w)（纯free，自由获取），它能让你看到字节在一线的探索，这个手册里结合了**抖音、飞书、电商、智能客服等 50 多个真实业务场景，**总结了**SFT、ReFT、Adapter、LoRA**等方法在千亿甚至万亿参数模型上的应用实践，还特别关注低资源和零资源场景下的微调策略，把微调方法、模型评估、部署监控全流程都搞清楚了。

这个阶段，你的目标是能不依赖任何框架，手撸一个简单的RAG问答机器人。就用OpenAI的API，自己用FAISS或者Chroma建个向量库，自己管理prompt，跑通整个流程。当你能做到这一点，你对LLM应用的理解就上了一个台阶。

  