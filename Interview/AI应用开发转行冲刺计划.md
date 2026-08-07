# 🎯 AI 大模型应用开发 — 转行冲刺计划（8~10 周）

> **核心策略：以 Agent 项目为主线，反向补齐 Web 后端、数据库、工程化能力。**
>
> 不做 Demo，做**工业级 AI 应用**，面试中能讲清每一个设计决策。

---

## 一、策略定位

| 维度 | ❌ 错误路线 | ✅ 正确路线 |
|------|------------|------------|
| 学习方式 | 先学 FastAPI → 数据库 → 前端 → Agent | 以 Agent 项目为主线，缺什么补什么 |
| 目标 | "学会 FastAPI" | "做出完整 AI 应用，理解为什么这样设计" |
| 面试定位 | "学习过 RAG 和 Agent 的人" | **"能独立设计和实现 AI 应用系统的人"** |

**已有储备：**
- ✅ RAG 思想
- ✅ Agent Loop / LangGraph 设计
- ✅ 数据库内容（募投报告知识库）
- ✅ OCR pipeline
- ✅ 领域数据

**待补齐：**
- ⚠️ Web 服务化（FastAPI）
- ⚠️ 工程组织（项目结构、分层架构）
- ⚠️ 数据库生产化（PostgreSQL + SQLAlchemy）
- ⚠️ 部署（Docker）
- ⚠️ 监控 / 日志 / 测试
- ⚠️ 前端基础（React 够用即可）

---

## 二、最终系统架构

```
                         用户
                          |
                      React/Web UI
                          |
                      FastAPI
                          |
                --------------------
                |                  |
         LangGraph Agent       User/Auth
                |
          ----------------
          |              |
        Tools           State
          |
    -------------------------
    |           |           |
 PostgreSQL   VectorDB   Web Search
    |
 Knowledge Base
    |
 Report Generator
    |
 HTML/PDF 输出

外围设施：Docker · Logging · Tracing · Evaluation
```

---

## 三、阶段 0：全局架构认知（第 1 周前 3~5 天）🖊️ **只画图，不写代码**

### 输出物 1：用户流程图

```
用户输入："分析 XX 地区新能源产业投资机会"
    ↓
[Query 理解]  识别意图：产业分析 + 地区筛选
    ↓
[任务规划]   分解子任务：检索 → 判断 → 搜索 → 整理 → 生成
    ↓
[检索产业数据库]  查询 PostgreSQL 知识库
    ↓
[判断信息充分性]
    ├── 充分 → 整理证据
    └── 不足 → 网络搜索补全 → 整理证据
    ↓
[生成报告]  Agent 组装结构化报告
    ↓
[HTML 展示]  前端渲染，用户查看/下载
```

### 输出物 2：系统模块图

| 层 | 模块 | 职责 |
|----|------|------|
| **前端** | React UI | 输入框、文件上传、流式展示、Markdown 渲染、报告查看 |
| **API 层** | FastAPI | 路由、参数校验、用户管理、调用 Agent |
| **Agent 层** | LangGraph | 状态流转、决策、Tool 调度 |
| **Tool 层** | Tools | 数据库查询、网络搜索、文件处理、报告生成 |
| **数据层** | PostgreSQL / VectorDB | 结构化数据 + 向量检索 |

### 输出物 3：State 设计（**本周重点**）

```python
class AgentState(TypedDict):
    """贯穿整个 Agent 生命周期的状态"""
    
    # ── 用户输入 ──
    query: str                    # 原始问题
    
    # ── 路由与意图 ──
    intent: str                   # 意图分类：政策查询 | 案例分析 | 产业分析
    sub_tasks: list[str]          # 分解后的子任务
    
    # ── 检索上下文 ──
    retrieved_docs: list[Document]  # 检索到的文档
    web_search_results: list[dict]  # 网络搜索结果
    evidence: list[Evidence]       # 整理后的证据条目
    
    # ── Agent 推理 ──
    tool_history: list[ToolCall]   # 工具调用历史
    reflection: str                # 自检结果："信息充分" | "需要补充XX"
    retry_count: int               # 重试次数
    
    # ── 最终输出 ──
    answer: str                    # 文本回答
    report: Report                 # 结构化报告（含章节、数据、图表引用）
```

**设计原则：**
- **贯穿流程的** → 放 State（query, intent, docs, report）
- **局部变量** → 放节点内部（临时计算结果、中间变量）

### 输出物 4：模块职责表

| 模块 | 输入 | 输出 | 关键决策 |
|------|------|------|----------|
| **Router** | query | intent + sub_tasks | 政策查询 or 产业分析？ |
| **Retriever** | intent + sub_tasks | retrieved_docs | 检索策略选择 |
| **WebSearch** | sub_tasks | web_search_results | 触发条件判断 |
| **Reflection** | evidence | 充分 / 需补充 | 质量阈值 |
| **Generator** | evidence | report | 模板选择 |

### 输出物 5：能力缺口自查表

| 类别 | 具体技能 | 掌握程度 | 优先级 |
|------|----------|----------|--------|
| **技术** | FastAPI 路由 + Pydantic + async | 0 → 会用 | 🔴 最高 |
| **技术** | PostgreSQL + SQLAlchemy ORM | 0 → 建模+CRUD | 🔴 最高 |
| **技术** | React 基础（输入/流式/Markdown） | 0 → 够用 | 🟡 中 |
| **工程** | Docker Compose 多容器 | 0 → 能配 | 🟡 中 |
| **工程** | Logging (request_id 链路追踪) | 0 → 实现 | 🟡 中 |
| **工程** | Pytest 测试 | 有基础 → 规范化 | 🟢 后 |
| **工程** | Evaluation 评估体系 | 0 → 建立 | 🟢 后 |
| **业务** | 报告生成模板 | 有基础 → 产品化 | 🟡 中 |

---

## 四、阶段 1：FastAPI + LangGraph 打通（第 1~2 周）

> **目标：浏览器 → FastAPI → LangGraph → LLM → 返回结果。不做数据库。**

### 学习重点（只掌握需要的部分）

| FastAPI 知识点 | 用到什么程度 | 为什么要学 |
|---------------|-------------|-----------|
| `@app.post("/chat")` 路由 | 定义 API 端点 | 最基本的入口 |
| Pydantic Schema (`ChatRequest`, `ChatResponse`) | 请求/响应模型 | 类型安全 + 自动文档 |
| `async def` | 异步处理函数 | LLM 调用/网络请求/数据库查询都适合异步 |
| Middleware | 注入 `request_id` + 日志 | 链路追踪的基础 |
| StreamingResponse | 流式返回 | 打字机效果 |

### 项目结构

```
app/
├── main.py                 # FastAPI 入口
├── api/
│   └── chat.py             # /chat 路由
├── agent/
│   ├── graph.py            # LangGraph 图定义
│   ├── state.py            # AgentState
│   └── nodes.py            # 各节点实现
├── schemas/
│   └── request.py          # Pydantic 模型
├── core/
│   ├── config.py           # 配置（API key、模型参数）
│   └── logger.py           # 统一日志
└── tests/
    └── test_chat.py
```

### 验收标准
- [ ] `curl -X POST /chat -d '{"message":"分析新能源"}'` 能拿到 Agent 返回结果
- [ ] 每个请求有 `request_id`，日志中可追踪
- [ ] 有 2 个 Pytest 测试用例

---

## 五、阶段 2：数据库生产化（第 3 周）

> **目标：SQLite → PostgreSQL，建立分层数据访问。**

### 数据建模

```sql
-- 产业知识库
industry (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(255),
    category    VARCHAR(100),     -- 新能源/基建/民生...
    region      VARCHAR(100),
    metadata    JSONB
)

-- 文档
document (
    id          SERIAL PRIMARY KEY,
    industry_id INTEGER REFERENCES industry(id),
    title       VARCHAR(500),
    content     TEXT,
    source      VARCHAR(255),
    doc_type    VARCHAR(50),      -- 募投报告/政策文件/案例
    created_at  TIMESTAMP
)
```

### 分层架构（**关键设计决策**）

```
Agent → Tool → Repository → Database
         ↑
    Agent 不能直接写 SQL
```

```python
# Repository 层（数据访问抽象）
class DocumentRepository:
    async def search(self, query: str, limit: int = 10) -> list[Document]:
        """全文检索 + 向量检索"""
        
    async def get_by_industry(self, industry_id: int) -> list[Document]:
        """按产业查询"""
        
    async def create(self, doc: DocumentCreate) -> Document:
        """插入文档"""
```

### 学习重点

| SQLAlchemy 知识点 | 用到什么程度 |
|-------------------|-------------|
| Declarative Base + Model 定义 | 建表 |
| AsyncSession | 异步查询 |
| `select()`, `where()`, `join()` | 基础 CRUD |
| Alembic | 数据库迁移 |

### 验收标准
- [ ] PostgreSQL 跑在 Docker 里，SQLAlchemy 能连上
- [ ] DocumentRepository 有 3 个方法的测试
- [ ] 知识库查询 Tool 通过 Repository 访问，不直接写 SQL

---

## 六、阶段 3：完善 Agent 工程能力（第 4~5 周）

### 3.1 Router — 意图路由

```
用户问题
    │
    ▼
┌─────────────────┐
│  意图分类       │
│  政策查询？     │ → 政策检索流程
│  案例分析？     │ → 案例检索流程
│  产业分析？     │ → 综合分析流程
└─────────────────┘
```

### 3.2 Tool 系统 — 统一接口

```python
class Tool:
    name: str
    description: str           # 给 LLM 看的描述
    
    async def execute(self, **kwargs) -> ToolResult:
        """统一执行入口"""
```

**工具清单：**

| Tool | 功能 | 依赖 |
|------|------|------|
| `database_search` | 知识库检索 | PostgreSQL + VectorDB |
| `web_search` | 网络搜索补全 | Search API |
| `report_generate` | 生成结构化报告 | LLM + 模板 |
| `file_upload` | 上传新文档 | 文件存储 |

### 3.3 Reflection — 质量自检

```python
def reflection_node(state: AgentState) -> AgentState:
    """
    评估当前证据是否充分
    ├── 充分 → 进入生成
    └── 不足 → 返回检索（最多重试 2 次）
    """
```

**重点：不要过度设计。** 先做简单的规则判断（文档数量/相关性分数），后面再优化。

### 验收标准
- [ ] 3 种意图各有一条完整流程
- [ ] Tool 统一通过 `Tool.execute()` 调用
- [ ] Reflection 能触发重新检索

---

## 七、阶段 4：前端展示（第 5~6 周）

> **目标：不做漂亮网站，做 AI 应用界面。**

### 需要掌握的最小前端集

| React 知识点 | 对应功能 |
|-------------|---------|
| 受控组件 (`useState`) | 输入框 |
| `useEffect` + `fetch` | API 调用 |
| SSE / EventSource | 流式展示（打字机效果） |
| `react-markdown` | 报告渲染 |
| 简单 CSS | 布局够用就行 |

### 界面要素
- 输入框（支持多行）
- 文件上传按钮
- 流式回复区域（逐字出现）
- Markdown 报告展示区
- 历史对话列表（可选）

### 验收标准
- [ ] 输入问题 → 流式看到 Agent 回复
- [ ] Markdown 表格/列表正常渲染
- [ ] 文件上传能触发后端处理

---

## 八、阶段 5：工程化（第 7~8 周）

> **这是区分 Demo 和项目的地方。**

### 8.1 Docker — 容器化

```
docker-compose.yml
├── frontend    (React, port 3000)
├── backend     (FastAPI, port 8000)
├── postgres    (port 5432)
└── redis       (port 6379, 缓存/任务队列)
```

### 8.2 Logging — 链路追踪

每条日志必须包含：

```
request_id=abc123 user=test query="新能源" tool=database_search latency=0.23s status=ok
```

### 8.3 Evaluation — 测试集

建立 100 个问题的测试集，每个问题标注：
- 期望意图
- 期望检索结果（至少 3 篇相关文档）
- 期望答案要点

评价维度：
- retrieval accuracy（检索准确率）
- answer correctness（答案正确性）
- latency（延迟）

### 8.4 测试

- 每个 Repository 方法有测试
- 每个 Tool 有测试
- API 端点有集成测试

### 验收标准
- [ ] `docker-compose up` 一键启动全栈
- [ ] 日志中可追踪完整请求链路
- [ ] 100 题测试集 + 评估脚本

---

## 九、冲刺后：面试叙事线

用**一句话**介绍你的项目：

> "我独立设计和实现了一个面向产业投资分析的 AI Agent 系统，支持从用户问题理解 → 知识库检索 → 报告生成的完整闭环，后端用 FastAPI + LangGraph，前端用 React，全栈 Docker 部署。"

用**三个设计决策**展示深度：

1. **为什么 Agent 不直接写 SQL，而是要经过 Repository 层？**
   → 解耦、可测试、安全（防注入）、方便切换数据源

2. **为什么用 async？**
   → LLM 调用 2~30 秒，数据库查询 IO 密集，异步避免阻塞

3. **为什么要做 Reflection 节点？**
   → 知识库可能不完整，需要自检 + 补全机制

---

## 十、每日时间安排（适配上班族）

### 工作日晚上（2 小时）

| 时间段 | 做什么 | 原则 |
|--------|--------|------|
| 30 min | **复盘架构**，画图：今天改了什么？影响哪个模块？ | 不学新东西 |
| 90 min | **写代码**，专注一个模块 | 不切换上下文 |

### 白天碎片时间

| 适合 | 不适合 |
|------|--------|
| 读 FastAPI 文档某一节 | 刷视频课程 |
| 看 PostgreSQL 概念 | 零散看代码 |
| 读优秀项目源码（如 LangChain 源码） | 漫无目的刷技术文章 |

### 周末

| 时间 | 重点 |
|------|------|
| 周六上午 | 数据库建模 / 迁移 |
| 周六下午 | Agent 流程完善 |
| 周日上午 | Docker 部署 / 测试 |
| 周日下午 | 整理文档 / 写 README |

---

## 十一、Hermes 使用策略

| ✅ 让 Hermes 做 | ❌ 不要让 Hermes 做 |
|-----------------|---------------------|
| 代码生成（给清楚接口） | 替你思考架构 |
| 重构（提取函数、加类型） | 替你定义 State |
| 补测试、补文档 | 替你设计模块 |
| Debug（贴错误日志） | 代你做技术选型 |

**高效 Prompt 模板：**

> "根据这个 State 设计，实现 `database_search` Tool。要求：不直接暴露 SQL，返回 Document 模型，有 logging，支持 async，添加 pytest 测试。"

---

## 十二、今晚任务清单 ✅

- [ ] **画图 1**：《我的 AI 应用整体架构》
- [ ] **画图 2**：《一次用户请求生命周期》（从输入到 HTML 报告）
- [ ] **画图 3**：《模块职责表》（每个模块：输入 / 输出）
- [ ] **画图 4**：《我还缺什么能力》（技术 / 工程 / 业务三栏）
- [ ] 确认本周第一阶段的具体起点（FastAPI 环境搭建）

---

> **一句话总结：不要把过去一年积累的知识散装存放，压缩成一个完整作品。两个月后，你的面试牌就从「学过」变成「做过」。**
