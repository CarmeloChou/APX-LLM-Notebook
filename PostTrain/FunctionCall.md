# FC（function call ）vs prompt

一般而言，工具嵌入有两种模式，一种模型支持FC，另一种不支持，需要通过prompt植入提示词。

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个助手，可以使用以下工具：
{tools}

当需要使用工具时，请严格回复如下JSON格式，不要包含任何其他文字：
{{"tool": "工具名", "args": {{"参数名": "值"}}}}"""),
    ("human", "{input}")
])
```

对于提示词而言，在模型不支持FC的情况下可以使用，但在配合ReAct模式时，需要手写if-else判断下一步动作。能用FC就用FC，直接将tools绑定到llm的api端口，能够直接进行调用，并且系统提示词可以瘦身。

> FC是SFT阶段的一种格式输出，能够让模型输出需要调用的工具格式。

```python
# 格式一般如下
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "北京今天天气怎么样？"}
  ],
  "tools": [{"name": "search_weather", "description": "查询天气", "parameters": {...}}],
  "completion": {
    "tool_calls": [{
      "name": "search_weather",
      "arguments": "{\"city\": \"北京\"}"
    }]
  }
}
```

二者对比如下：

```bash
原生 Function Calling          纯文本 Prompt 模拟
┌─────────────────────┐       ┌─────────────────────────┐
│ API tools 参数      │       │ System Prompt 里的文字   │
│ (结构化 schema)     │       │ (自然语言描述)           │
└────────┬────────────┘       └──────────┬──────────────┘
         ▼                               ▼
  模型输出 tool_calls              模型输出一段文本
  (结构化、可靠)                   (可能格式错误)
         ▼                               ▼
  SDK 自动解析+执行               手写正则/JSON解析
  + 自动多轮循环                  + 手写容错 + 手写循环
         ▼                               ▼
      ✅ 省心                      ⚠️ 费力且脆弱
```

