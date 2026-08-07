# Agent 项目工作计划与续接文档

> 目标：在招聘窗口内完成一个可演示、可解释、可持续迭代的“大模型应用 / Agent”项目。
>
> 当前原则：先完成稳定主链路；遇到真实问题再定向学习；不为了学习而扩展架构。

## 1. 项目定位

**项目暂定**：产业研究与报告生成 Agent。

用户输入产业研究问题，系统支持多轮追问，结合本地知识库（后续加入检索）生成带来源的结构化分析或报告。

目标岗位：Python 大模型应用开发、RAG / Agent 工程师、AI 后端工程师。

## 2. 当前已经完成

- 跑通 `FastAPI -> ChatService -> Agent Runtime -> LangGraph / LLM -> Response` 最小链路。
- 使用 Swagger 验证结构化请求与响应。
- 区分三类数据：
  - `ChatRequest`：HTTP 请求数据；
  - Agent State：当前工作流的 `messages`、检索结果、回答等；
  - `RunnableConfig`：`user_id`、`session_id/thread_id` 等运行上下文。
- 使用 InMemory 实现会话消息保存的原型。
- 发现并解决多用户会话串话：LangGraph Checkpointer 的 `thread_id` 不能只用局部 `session_id`。

当前隔离策略：

```python
thread_id = f"{user_id}_{session_id}"
```

因此 `001 / 1` 与 `002 / 1` 会被视为两个不同的 LangGraph 工作流。

## 3. 已确认的关键设计结论

### 3.1 Request、State、Config 的边界

```text
ChatRequest
  用户本次提交：message、user_id、session_id

Agent State
  工作流运行数据：messages、documents、plan、answer

RunnableConfig
  运行身份与配置：thread_id、user_id、模型配置等
```

不要把 `user_id`、`session_id` 当作需要模型推理的 Agent State；它们是运行上下文和访问边界。

### 3.2 会话隔离

会话关系：

```text
User
  └── Chat Session
        └── Chat Messages
```

必须保证：一个 session 只能属于一个 user。读取历史时，必须按 `user_id + session_id` 验证归属。

### 3.3 短期对话记忆与长期记忆

第一版：由 LangGraph Checkpointer 管理短期对话/工作流状态。

后续：自己实现的 Store 管理用户偏好、历史报告、长期任务信息。

不要同时：

1. 手动把完整历史拼成一个新的 user prompt；
2. 又让 LangGraph Checkpointer 恢复 `messages`。

否则会发生重复注入，模型可能复述旧问题。

### 3.4 当前 InMemory 的边界

`dict` 版本适合学习和单进程开发，但服务重启、多个进程或多个容器后数据会丢失或不一致。

下一阶段要实现 PostgreSQL 持久化。

## 4. 当前目标架构（冻结第一版）

```text
Swagger / Web UI
  -> FastAPI API
  -> ChatService
  -> LangGraph Agent Runtime
  -> Tool Layer
  -> PostgreSQL / Vector Retrieval
  -> Evidence-based Answer / Report
```

暂不实现：复杂前端、Redis、Graph DB、多 Agent、长期记忆、复杂权限、Kubernetes。

## 5. 本周下一步：会话持久化

### 5.1 第一目标

把会话和消息从 InMemory 迁移到 PostgreSQL，实现：

```text
服务重启后，历史仍能恢复；
不同用户不能读取彼此的会话；
未来可部署多个后端实例。
```

### 5.2 数据表（第一版）

```text
chat_sessions
- id              # 全局唯一 UUID；也是产品层 session_id
- user_id         # 该会话的拥有者
- created_at
- updated_at

chat_messages
- id
- session_id      # 外键，指向 chat_sessions.id
- role            # user / assistant / tool
- content
- created_at
```

`user_id` 表示“谁”；`chat_sessions.id` 表示“这个人的哪一段对话”。一个用户可拥有多段会话。

### 5.3 存储抽象

不要让新接口继承现有 `Memory`。改成：

```text
SessionStore（抽象能力契约）
  ├── InMemorySessionStore（当前字典实现）
  └── PostgresSessionStore（后续数据库实现）
```

接口第一版：

```python
class SessionStore:
    async def create_session(self, user_id: str) -> str: ...
    async def get_messages(self, user_id: str, session_id: str) -> list[dict]: ...
    async def append_message(
        self, user_id: str, session_id: str, message: dict
    ) -> None: ...
```

`ChatService` 只依赖 `SessionStore`，不关心底层是字典还是 PostgreSQL。

### 5.4 SQLAlchemy 最小心智模型

```text
Engine       -> 数据库连接工厂 / 连接池
AsyncSession -> 一次数据库操作的临时工作区，不是聊天会话
ORM Model    -> Python 类对应数据库表
commit       -> 确认写入
rollback     -> 失败时撤销本次未提交操作
```

本阶段只需要掌握：建表、插入、按 `user_id + session_id` 查询、`commit`、`rollback`。

## 6. 两周推进计划

### Day 1：稳定当前会话行为

- 确保 `thread_id` 全局唯一。
- 每次只把本轮 user message 传给 LangGraph。
- 不再把 InMemory history 拼成 `query_merge`。
- 验证同用户同会话连续、同用户不同会话隔离、不同用户同 session 值隔离。

### Day 2：重构目录与接口

目标目录：

```text
src/
  api/
  schemas/
  services/
  agent/
  infrastructure/
  stores/
```

- 将现有 `Memory` 改名为 `InMemorySessionStore`。
- 定义 `SessionStore` 抽象接口。
- 保持 Swagger 功能不变。

### Day 3：PostgreSQL 连通与最小读写

- 连接 Docker 中已有 PostgreSQL。
- 创建 `chat_sessions` 与 `chat_messages` 表。
- 插入一个会话与两条消息。
- 按 session 查询消息并按时间排序。

### Day 4：实现 PostgresSessionStore

- 实现创建会话、追加消息、读取消息。
- 读取时验证 `session_id` 属于当前 `user_id`。
- 用依赖注入将 Store 提供给 ChatService。

### Day 5：切换持久化与验证

- 将 ChatService 从 InMemory 切换到 PostgresSessionStore。
- 重启服务后验证会话历史仍存在。
- 增加基础日志：request_id、user_id、thread_id、耗时、异常。

### Week 2：Agent 从聊天升级为可用能力

1. 增加一个标准化 Tool（先用本地假资料也可以）。
2. 在 Agent State 中加入 `documents` 与 `citations`。
3. 实现 `query -> tool -> evidence -> answer`。
4. 接入 PostgreSQL 文档查询。
5. 再加入向量检索，形成最小 RAG。

## 7. 每日执行方式

每天只设定一个可验证交付，不设“学习某个框架”的开放目标。

示例：

```text
今天目标：重启服务后，001 的 session 能恢复，002 无法读取该 session。
```

开始前写五分钟设计卡：

```text
功能：
输入：
输出：
状态/数据保存在哪里：
失败时怎么办：
如何验证：
```

完成后复盘：

```text
今天实现了什么？
最困惑的概念是什么？
AI 的实现比我多考虑了什么？
下一版最值得改进的一点是什么？
```

## 8. 使用 AI 的规则

你负责：需求边界、输入输出、状态归属、验收条件、关键取舍。

AI 可加速：样板代码、ORM 模型、CRUD、测试扩展、重构建议、错误定位。

核心模块在交给 AI 前，至少先写：接口签名、伪代码、一个成功场景和一个失败场景。

## 9. 跨电脑续接

将代码和本文档同步到私有 Git 仓库。

每次开始：

```text
git pull
阅读本文档的“当前已经完成”和“下一步”
```

每次结束：

```text
更新本文档
git add .
git commit -m "描述本次功能"
git push
```

不要提交 `.env`、API Key、数据库密码。提交 `.env.example` 即可。
