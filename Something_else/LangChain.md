# LangChain

>  中文文档：[理念 | LangChain 中文文档](https://langchain-doc.cn/v1/python/langchain/philosophy.html#_2023-06)

# LangGraph

最初的 LangChain 有两个重点：LLM 抽象层，以及帮助快速构建常用应用的高层接口；然而缺少一个低层编排层，让开发者能精准控制 Agent 的执行流程。于是出现了：LangGraph。

LangGraph 加入了实践中发现必需的功能：流式输出、持久化执行、短期记忆、人类介入（human-in-the-loop）等。

## 流式输出

- **流式（Streaming）**：就像在线看 YouTube/Bilibili。视频缓冲几秒后就开始播放，边下载边播放。对于大模型，这意味着模型**每生成一个字（或几个字），就立刻通过网络发送给前端**，用户能看到文字像打字机一样“逐字蹦出来”。
- **非流式（普通模式）**：就像下载电影。你必须等整个文件（比如 2GB）完全下载好后，播放器才能开始播放。对于大模型，这意味着用户要盯着屏幕转圈几十秒，然后文字瞬间全部出现。

# 核心组件

## Agent

### 静态模型

创建时配置一次，执行过程保持不变

```python
from langchain.agents import create_agent

agent = create_agetn(
	"openai:gpt-5",
    tools = tool
)

# 可以直接使用提供商包初始化模型实例
from langchain_openai import ChatOpenai
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import ChatTongyi # 或者使用Openai兼容

model = ChatOpenAI(
	model="gpt-5",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
)
agent = create_agent(model, tools=tools)
```

### 动态模型

```python
from langchian_openai import ChatOpenAI
from langchain_agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ChatOpenAI(model='gpt-4o-mini')
advanced_model = ChatOpenAI(model='gpt-4o')

@wrap_model_call
def dynamic_model_selection(request: ModelReuqst, handler) -> ModelResponse:
    message_count = len(request.state["messages"])
    
    if ...:
        model = advanced_model
    else:
        model = basic_model
        
    request.model = model
    return handler(request)

agent = create_agent(
    model=basic_model,  # 默认模型
    tools=tools,
    middleware=[dynamic_model_selection]
)
```