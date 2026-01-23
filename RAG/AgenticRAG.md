# AgenticRAG

[Building an Advanced Agentic RAG Pipeline that Mimics a Human Thought Process | by Fareed Khan | Level Up Coding](https://levelup.gitconnected.com/building-an-advanced-agentic-rag-pipeline-that-mimics-a-human-thought-process-687e1fd79f61)

标准RAG系统只查找、总结既有文件，并不会进行思考。Agentic RAG可以阅读、校正、链接、推理，使得其更像一个专家而不是检索工具。

本文试图构建一个Agentic RAG管线，模仿人类阅读以及理解问题的能力。

![](./Image/AgenticRAG.jpg)

流程如下：

- 简历丰富的知识库。agent不止阅读文件，还会仔细分析文档，并且使用LLM添加总结和关键词，创建不同层次的理解。
- 整合专家队伍。每个专家不会精通所有事件，并且也不是agent本身。该队伍依赖专家工具：文档工具、数据库分析工具等。
- 门控网络。在执行前，门控网络会检查问题是否清晰和具体。如果不够清晰具体，会被要求重新分类。
- 规划师指定计划。一旦问题评估技术，规划者将请求分解为一步步的工具调用，保证进程架构清晰，避免回答冲突。
- 作者评估结果。每个工具的输入会被检查质量以及一致性。如果结果很弱或者存在矛盾，重新评估问题以及校正自身。
- 策略者将所有的细节串联起来，最终的回答不仅是事实的列表。策略者还会寻找相关性、模式、理论，将数据转化为更高深的视角分析。
- 对抗性测试。系统会被红队机器人挑战，提出棘手、误导或有偏见的问题，确保系统在压力下依然稳健可靠。
- 最终，它发展成超越简单问答的功能，新增了认知**记忆（从**过去互动中学习）、**守望台**（主动监控重要事件）和**神谕**（解读图表等视觉数据）等功能。

[FareedKhan-dev/agentic-rag: Agentic RAG to achieve human like reasoning](https://github.com/FareedKhan-dev/agentic-rag/?source=post_page-----687e1fd79f61---------------------------------------)

本质上是套了多层壳的agent

# Scalable，Production-Grade Agentic Pipeline

本文是同一作者的推文，我打算以此构建一个agentic rag项目。

![](./Image/RAGpipeline.jpg)

Agentic RAG 系统依然包括三大板块：数据库、LLM、Tools（API、MCP、CLI）。本文将其细分为六个方面：

- 数据摄取层：通过文档加载、分块、索引将原始数据转换为结构化知识；可扩展，支持 S3、RDBMS 和 Ray。
- **AI 计算层：**高效运行 LLM 和嵌入，将模型映射到 GPU/CPU 以实现低延迟、大规模推理。
- **智能AI管道：**通过API、缓存和分布式执行，支持智能体推理、查询增强和工作流编排。
- **工具和沙箱：**为计算、搜索和 API 测试提供安全的环境，不会影响生产工作负载。
- **基础设施即代码 (IaC)：**实现可复制、可扩展的基础设施的自动化部署、联网、集群设置和自动扩展。
- **部署与评估：**处理密钥、数据库、集群、监控和日志记录，以确保大规模可靠运行。0

## 数据源准备

本文使用了50个真实文档以及950个噪音文档。（打开作者仓库，里面只有5个真是文档）

```bash
# clone noisy data repository
git clone https://github.com/tpn/pdfs.git
# create noisy_data directory
mkdir noisy_data

# randomly sample 950 documents from tpn/pdfs repository
find ./pdfs -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" -o -name "*.html" \) | shuf -n 950 | xargs -I {} cp {} ./noisy_data/
# create data directory
mkdir data

# copy true data to data directory
cp -r ./true_data/* ./data/

# copy noisy data to data directory
cp -r ./noisy_data/* ./data/
# count total number of documents in data directory
find ./data -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" -o -name "*.html" \) | wc -l

### OUTPUT
1000
```

## 创建企业代码库

### 工作管线及架构

通常一个agentic RAG管线代码库包括一个向量数据库、一系列AI模型以及一个获取管线。然而，当管线越来越复杂的时候，我们需要将整体架构细分为更小的、可管理的组成部分。以下为RAG管线：

```bash
scalable-rag-core/                     # Minimal production RAG system
├── infra/                             # Core cloud infrastructure
│   ├── terraform/                     # Cluster, networking, storage
│   └── karpenter/                     # Node autoscaling (CPU/GPU)
│
├── deploy/                            # Kubernetes deployment layer
│   ├── helm/                          # Databases & stateful services
│   │   ├── qdrant/                    # Vector database
│   │   └── neo4j/                     # Knowledge graph
│   ├── ray/                           # Ray + Ray Serve (LLM & embeddings)
│   └── ingress/                       # API ingress
│
├── models/                            # Model configuration (infra-agnostic)
│   ├── embeddings/                    # Embedding models
│   ├── llm/                           # LLM inference configs
│   └── rerankers/                     # Cross-encoder rerankers
│
├── pipelines/                         # Offline & async RAG pipelines
│   └── ingestion/                     # Document ingestion flow
│       ├── loaders/                   # PDF / HTML / DOC loaders
│       ├── chunking/                  # Chunking & metadata
│       ├── embedding/                 # Embedding computation
│       ├── indexing/                  # Vector + graph indexing
│       └── graph/                     # Knowledge graph extraction
│
├── libs/                              # Shared core libraries
│   ├── schemas/                       # Request/response schemas
│   ├── retry/                         # Resilience & retries
│   └── observability/                 # Metrics & tracing
│
├── services/                          # Online serving layer
│   ├── api/                           # RAG API
│   │   └── app/
│   │       ├── agents/                # Agentic orchestration
│   │       │   └── nodes/             # Planner / Retriever / Responder
│   │       ├── clients/               # Vector DB, Graph DB, Ray clients
│   │       ├── cache/                 # Semantic & response caching
│   │       ├── memory/                # Conversation memory
│   │       ├── enhancers/             # Query rewriting, HyDE
│   │       ├── routes/                # Chat & retrieval APIs
│   │       └── tools/                 # Vector search, graph search
│   │
│   └── gateway/                       # Rate limiting / API protection
```

以上结构看起来比较复杂，最重要的包括以下四个部分：

- deploy：部署的参数配置，包括Ray、入口控制、密钥管理
- infra：Infrastructure as code（IaC）代码
- pipelines：数据处理及工作管理管线，包括文档加载、文档分块、嵌入计算、图处理、图编码
- services：应用服务，包括API、防火墙参数、沙盒环境

每个部分都由单独的文件夹组成，**具有较好的可编辑性和拓展性**。

### 开发工作流

**最重要的第一步是配置本地开发环境**。可扩展项目通常自动处理开发环境问题，避免每次有新开发者加入就要重新配置环境。开发环境通常包含以下三方面：

- `.env.example`:分享环境变量的标准方法。开发者可以复制相关内容到`.env`并且改为他们自己想要的数值（取决于开发节奏、开发阶段、生产语境）
- `Makefile`：包括多种自动化任务指令，包括：building，testing和部署应用
- `docker-composed.yml`：**最重要的布局文件**，包括在Dockers容器中本地运行整个RAG管线的所有服务

```bash
# .env.example
# Copy this to .env and fill the values

# --- APP SETTINGS ---
ENV=dev
LOG_LEVEL=INFO
SECRET_KEY=change_this_to_a_secure_random_string_for_jwt

```

第一步定义基础的应用设置，比如环境、log等级、[JWT认证密钥](./JWT.md)

```bash
# --- DATABASE (Aurora Postgres) ---
DATABASE_URL=postgresql+asyncpg://ragadmin:changeme@localhost:5432/rag/db

# ---CACHE(Redis) ---
RESIS_URL=redis://localhost:6379/0

# ---VECTOR DB(Qdrant) ---
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=rag_collection

# --- GRAPH DB (Neo4j)
NEO4J_URL=bolt://localhost:7678
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

RAG管线多数基于数据仓库，但我们正在创建可扩张的解决路径，因此这不仅仅是向量存储了。我们需要存储不同路径文件之间的联系，这意味着我们需要多种不同的存储方式。

在RAG管线中，我们使用最常见以及最受欢迎的数据库：

1. Aurora Postgres for chat history and metadata storage.
2. Redis for caching frequently accessed data.（经常访问的数据）
3. Qdrant as our vector database for storing embeddings.（嵌入向量）
4. Neo4j as our graph database for storing relationships between entities.（存储实例之间的关系）

```bash
# --- AWS(Infrastructure) ---
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
S3_BUCKET_NAME=rag-platform-docs-dev

# ---RAY CLUSTER (AI Engines) ---
# In k8s, these point to internal Service DNS
# Locally, you might port-forward
RAY_LLM_ENDPOINT=http://localhost:8000/llm
RAY_EMBED_ENDPOINT=http://localhost:8000/embed

# --- OBSERVABILITY ---
OTEL_EXPORTER_OTLP_ENDPOINT=http://locahost:4317
```

至此，我们已经处理了大量的数据，我们需要一个有效的方式存储和检索文档。为此，我们使用AWS S3作为我们的基础文档存储，并结合集群式**Ray Serving 来托管我们的 AI 模型（LLM、嵌入、重排序器）**。

接下来创建`Makefile`文件自动化处理一般任务，如building、testing以及应用部署。

[Makefile 基本操作](./Makefile.md)

```makefile
# Makefile

.PHONY: help install dev up down deploy test
help:
 @echo "RAG Platform Commands:"
 @echo "  make install    - Install Python dependencies"
 @echo "  make dev        - Run FastAPI server locally"
 @echo "  make up         - Start local DBs (Docker)"
 @echo "  make down       - Stop local DBs"
 @echo "  make deploy     - Deploy to AWS EKS via Helm"
 @echo "  make infra      - Apply Terraform"
install:
 pip install -r services/api/requirements.txt

# Run Local Development Environment
up:
 docker-compose up -d
down:
 docker-compose down

# Run the API locally (Hot Reload)
dev:
 uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env

# Infrastructure
infra:
 cd infra/terraform && terraform init && terraform apply

# Kubernetes Deployment
deploy:

 # Update dependencies
 helm dependency update deploy/helm/api

 # Install/Upgrade
 helm upgrade --install api deploy/helm/api --namespace default
 helm upgrade --install ray-cluster kuberay/ray-cluster -f deploy/ray/ray-cluster.yaml
test:
 pytest tests/
```

Makefile创建后，创建各类命令用来管理开发工作流程。

最后生成一个`docker-compose.yml`文件来定义Docker容器在本地运行整个RAG管道所需的所有服务。

```yaml
version: '3.8'

services:
  # 1. Postgres(聊天记录)
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: ragadmin
      POSTGRES_PASSWORD: changeme
      POSTGRES_DB: rag_db
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data

  # 2. Redis(缓存)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  # 3. Qdrant(矢量数据库)
  qdrant:
    image: qdrant/qdrant:v1.7.3
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  # 4. Neo4j(图数据库)
  neo4j:
    image: neo4j:5.16.0-community
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_dbms_memory_pagecache_size: 1G
    ports:
      - "7474:7474" # HTTP
      - "7687:7687" # Bolt
    volumes:
      - neo4j_data:/data

  # 5. MinIO(S3 Mock)
  minio:
    image: minio/minio
    command: server /data
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin

volumes:
  pg_data:
  qdrant_data:
  neo4j_data:
```

在小型项目中，我们通常使用 .NET Framework`pip`或`virtualenv`.NET Framework 来管理依赖项。然而，在大型项目中，我们使用 Docker 容器来隔离 RAG 流水线的不同组件。

在我们的`yaml`文件中，我们为每个服务指定了不同的端口，以避免冲突，并在我们的管道运行时便于监控，这也是大规模项目中的最佳实践

### 核心共享实用程序

现在我们已经搭建好了项目结构和开发流程，首先需要的就是一个唯一的ID生成策略。**当用户向我们的RAG机器人发送聊天信息时，很多事情会同时发生**，将它们全部映射起来有助于我们追踪与该特定聊天会话相关的问题，并涵盖RAG流程的各个环节。

**这是生产系统中非常常见的做法，其中所有内容都与用户操作** **（例如点击或请求）**相关联，以便以后更容易进行监控、调试和跟踪。

为此，我们将创建一个`libs/utils/ids.py`文件来处理聊天会话、文件上传和 OpenTelemetry 跟踪的唯一 ID 生成。

```python
# libs/utils/ids.py
import uuid
import hashlib

def generate_session_id() -> str:
    """为聊天会话生成标准 UUID"""
    return str(uuid.uuid4()) # 生成随机 UUID（版本4）

def generate_file_id(content: bytes) -> str:
    """
    基于文件内容生成确定性ID
    防止重复上传同一个文件
    """
    return hashlib.md5(content).hexdigest() #根据二进制内容生成32位十六进制字符串（如 d41d8cd98f00b204e9800998ecf8427e）

def generate_trace_id() -> str:
    """为OpenTelemetry跟踪生成ID"""
    return uuid.uuid4().hex #生成不带连字符的 32 位十六进制字符串
```

同样，为了进行性能监控和优化，我们需要测量 RAG 流水线中各个函数的执行时间。让我们创建一个`libs/utils/timing.py`文件来处理同步和异步函数的执行时间测量。

```python
import functools
import time
import logging

# 获取logger
logger = logging.getLogger(__name__)

def measure_time(func):
    """
    用于记录同步函数执行时间的装饰器
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # 毫秒
        logger.info(f"函数 '{func.__name__}' 耗时 {execution_time:.2f} 毫秒")
        return result
    return wrapper

def measure_time_async(func):
    """
    用于记录异步函数执行时间的装饰器。
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # 毫秒
        logger.info(f"异步函数 '{func.__name__}' 耗时 {execution_time:.2f} 毫秒")
        return result
    return wrapper
```

[代码装饰器（DRY）](./DRY.md)

这里我们定义了两个装饰器`measure_time`，`measure_time_async`分别用于记录同步函数和异步函数的执行时间。

最后，我们需要一个重试机制。在生产级 RAG 系统中，我们通常**使用指数退避算法来处理**RAG 流水线中的错误。让我们创建一个`libs/retry/backoff.py`文件来实现这个功能。

```python
def exponential_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0):
    """
    Decorator for Exponential Backoff with Jitter.
    Retries async functions upon exception.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while True: # 这里一直保持程序运行，即使失败只是保持重试，而不会终止服务运行
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if retries >= max_retries:
                        logger.error(f"Max retries reached for {func.__name__}: {e}")
                        raise e
                    
                    # Algorithm: base * (2 ^ retries) + random_jitter
                    # Jitter prevents "Thundering Herd" problem on the server
                    delay = min(base_delay * (2 ** retries), max_delay)
                    jitter = random.uniform(0, 0.5)
                    sleep_time = delay + jitter
                    
                    logger.warning(f"Error in {func.__name__}: {e}. Retrying in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                    retries += 1
        return wrapper
    return decorator
```



我们使用公式`base * (2 ^ retries) + random_jitter`来计算每次重试前的延迟。这有助于我们避免**“群殴”** **问题，**即多个客户端同时重试的情况。

## 数据摄取层

RAG流程的第一部分（无论规模大小）都是将文档导入系统。在小型项目中，一个简单的脚本按顺序读取文件即可满足需求。

然而，在**企业级 RAG 管道**中，摄取是一项高吞吐量的异步任务，必须同时处理数千个文件，而不会导致 API 服务器崩溃。

![](./Image/数据提取.jpg)

**Ray Data 允许我们创建**任务的有向无环图 (DAG)，这些任务可以在集群中的多个节点上并行执行。

这样我们就可以独立地扩展解析（CPU 密集型）和嵌入（GPU 密集型）任务。

### 文档加载和配置

首先，我们需要一个集中式的配置来管理数据摄取参数。像数据块大小或数据库集合这样的值如果硬编码到生产环境中，将会造成灾难性的后果。

![](./Image/文档处理.jpg)

让我们创建`pipelines/ingestion/config.yaml`一个包含所有数据摄取管道配置的文件。

```yaml
# pipelines/ingestion/config.yaml

chunking:
	# 512个token是RAG的黄金比例（上下文足够，噪声不多）
	chunk_size: 512
	
	# 重叠部分确保在分割点不会丢失上下文
	chunk_overlap: 50
	
	# 用于递归分割的分隔符（段落 -> 句子 -> 单词）
	separators: [ "\n\n", "\n", " ", ""]
	
embedding:
	# 要使用的Ray Serve 端点 endpoint
	: "http://ray-serve-embed:8000/embed"
	batch_size: 100
	
graph:
	# 控制 LLM 提取速度与成本
	concurrency: 10
	
	# 如果为 true，则严格遵循shcema.py 本体
	enforce_schema: true
	
vector_db:
	collection_name: "rag_collection"
	distance_metric: "Cosine"
```

现在我们需要加载器。在企业系统中，PDF 文件非常占用资源。将一个 100MB 的 PDF 文件加载到内存中可能会导致 Kubernetes 工作进程因内存不足 (OOM) 而被终止。

我们需要一个单独的处理文件，例如`pipelines/ingestion/loaders/pdf.py`，在其中我们使用`unstructured`临时文件来有效地管理内存。

```python
# pipelines/ingestion/loaders/pdf.py
import tempfile
from unstructured.partition.pdf import partition_pdf

def parse_pdf_bytes(file_bytes: bytes, filename: str):
    """
    使用临时文件解析PDF文件流以提高内存效率
    """
    text_content = ""
    # 使用磁盘存储而不是RAM以防止处理大文件时工作进程崩溃
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file.flush()
        
        # "hi_res" 策略使用OCR和布局分析
        elements = partition_pdf(filename=tmp_file.name, strategy="hi_res")
        for el in elements:
            text_content += str(el) + "\n"
    return text_content, {"filename" : filename, "type" : "pdf"}
```

对于其他格式，我们需要更轻量级的解析器。让我们`pipelines/ingestion/loaders/docx.py`为 Word 文档创建解析器：

```python
# pipelines/ingestion/loaders/docx.py
import docx
import io

def parse_docx_bytes(file_bytes: bytes, filename: str):
    """解析 .docx文件，提取文本和简单表格"""
    doc = docx.Document(io.BytesIO(file_bytes))
    full_text = []
    
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
            
    return "\n\n".join(full_text), {"filename" :filename, "type" : "docx"}
```

对于`pipelines/ingestion/loaders/html.py`网页内容，我们必须去除脚本和样式，以避免用 CSS 或 JavaScript 代码污染我们的矢量图。

```python
# pipelines/ingestion/loaders/html.py
from bs4 import BeautifulSoup

def parse_html_bytes(file_bytes: bytes, filename: str):
    """从HTML中清除脚本/样式，提取纯文本"""
    soup = BeautifulSoup(file_bytes, "html.parser")
    
    # 移除LLM会混淆的垃圾元素
    for script in soup(["script", "style", "meta"]):
        script.decompose()
    
    return soup.get_text(separator= "\n"), {"filename" : filename, "type" : "html"}
```

### 组块和知识图谱

提取原始文本后，在 RAG 中，接下来我们需要对其进行转换。我们定义了一个分割器，`pipelines/ingestion/chunking/splitter.py`将文本分割成 512 个词元的块，这是许多嵌入模型的标准限制。

![](./Image/文本分割器.jpg)

```python
# pipelines/ingestion/chunking/splitter.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_txt(text: str, chun_size: int = 512, overlap: int = 50):
    """将文本分割成重叠的块，以保留边界处得上下文。"""
    splitter = RecursiveCharacterTextSplitter(
    	chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.create_documents([text])
    
    # 映射到Ray管道的字典格式
    return [{"text" : c.page_content, "metadata" : {"chunk_index" : i}} for i, c in enumerate(chunks)]
```

```python
def split_chunk(text: str, chunk_size: int = 512, overlap : int = 50) -> list:
    len_text = len(text)
    all_chunk = []
    assert chunk_size-overlap >= 1; "分块字符数与重叠数至少为1"
    for i in range(0, len_text, chunk_size-overlap):
        chunk = text[i:chunk_size]
        all_chunk.append(chunk)
    return all_chunk
```

我们还对这些数据块进行丰富`pipelines/ingestion/chunking/metadata.py`。在分布式系统中，去重非常重要，因此我们生成内容哈希值。

```python
# pipelines/ingestion/chunking/metadata.py
import hashlib
import datetime

def enrich_metadata(base_metadata: dict, content: str) -> dict:
    """添加哈希值和时间戳以进行去重和新鲜度跟踪"""
    return {
        **base_metadata,
        "chunk_hash": hashlib.md5(content.encode("utf-8")).hexdigest(),
        "ingested_at": datetime.datetime.utcnow().isoformat()
    }
```

[字典解包运算符](./Knowledge_Added/字典解包运算符.md)

现在我们将创建用于生成嵌入的 GPU 工作负载。我们不会在数据导入脚本中加载模型（启动速度较慢），而是调用**Ray Serve**端点。

这样一来，我们的数据摄取任务只需向持续运行的模型服务发出 HTTP 请求即可。我们需要创建`pipelines/ingestion/embedding/compute.py`单独的组件来管理它：

```python
# pipelines/ingestion/embedding/compute.py
import httpx

class BatchEmbedder:
    """Ray Actor，用于批量处理文本块并调用嵌入服务"""
    def __init__(self):
        # 我们指向内部k8s服务DNS
        self.endpoint = "http://ray-serve-embed:8000/embed"
        self.client =  httpx.Client(timeout=30.0)
    def __call__(self, batch):
        """向GPU服务发送一批文本"""
        response = self.client.post(
        	self.endpoint,
            json = {"text" : batch["text"], "task_type" : "document"}
        )
        batch["vector"] = response.json()["embeddings"]
        return batch
```

同时，我们提取知识图谱。为了保持图谱的整洁，我们必须定义一个严格的模式`pipelines/ingestion/graph/schema.py`。否则，LLM 会生成随机的关系类型。

```python
# pipelines/ingestion/graph/schema.py
from typing import Literal

# 将LLM限制为仅包含以下实体/关系
VALID_NODE_LABELS = Literal["Person", "Organization", "Location", "Concept", "Product"]

VALID_RELATION_TYPES = Literal["WORKS_FOR", "LOCATED_IN", "RELATES_TO", "PART_OF"]

class GraphSchema:
    @staticmethod
    def get_system_prompt() -> str:
        return f"提取节点/边。允许得标签:{VALID_NODE_LABELS.__args__}..."
```

我们在 中应用此模式`pipelines/ingestion/graph/extractor.py`。它使用 LLM 来理解文本的*结构，而不仅仅是语义相似性。*

```python
# pipelines/ingestion/graph/extractor.py
import httpx
from pipelines.ingestion.graph.schema import GraphSchema

class GraphExtractor:
    """
    用于图提取的Ray Actor类
    调用内部LLM服务提取实体
    """
    def __init__(self):
        # 指向内部Ray Serve LLM端点
        # 我们使用内部k8s DNS 名称
        self.llm_endpoint = "http://ray-serve-llm:8000/llm/chat"
        self.client = httpx.Client(timeout = 60.0) # 较长的超时时间用于推理
    def __call__(self, batch: Dict[str, Any]) -> Dict[str,Any]:
        """
        处理一批文本块
        """
        nodes_list = []
        edges_list = []
        
        # 遍历批次中的文本块
        for text in batch["text"]:
            try:
                # 1. 构建提示
                prompt = f"""
                {GraphSchema.get_system_prompt()}
                
                输入文本：
                {text}
                """
                # 2.调用LLM(Llama-3-70B)
                response = self.client.post(
                	self.llm_endpoint,
                    json={
                        "messages" : [{"role" : "user", "content" : prompt}],
                        "temperature" : 0.0, 
                        "max_tokens" : 1024
                    }
                )
                reponse.raise_for_status()
                
                # 3.解析JSON输出
                # 我们假设模型返回有效的JSON（通过约束解码或后处理保证）
                content = response.json()["choices"][0]["message"]["content"]
                graph_data = json.loads(content)
                
                # 4.追加到结果
                nodes_list.append(graph_data.get("nodes", []))
                edges_list.append(graph_data.get("edges", []))
			except Exception as e:
                # 记录错误但不导致管道崩溃，此块返回空图
                print(f"图提取失败，块：{e}")
                nodes_list.append([])
                edges_list.append([])
	
	# 将图数据添加到批次
    batch["graph_nodes"] = nodes_list
    batch["graph_edges"] = edges_list
    return batch
```

### 高通量索引

在大规模随机存取存储器（RAG）中，我们不逐条插入记录，而是执行批量写入。批量处理可以降低GPU和CPU的负载，从而减少内存使用，使分区可以用于后续的其他任务。

![](./Image/高吞吐量索引.jpg)

对于向量，我们使用`pipelines/ingestion/indexing/qdrant.py`。这会处理与我们的 Qdrant 集群的连接并执行原子 upsert 操作。

[Qdrant Python 客户端文档](https://python-client.qdrant.org.cn/)

```python
# pipelines/ingestion/indexing/qdrant.py

from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

class QdrantIndexer:
    """使用批量upsert将向量写入Qdrant"""
    def __init__(self):
        self.client = QdrantClient(host="qdrant-service", post=6333)
    def write(self, batch):
        points = [
            models.PointStruct(id = str(uuid.uuid4()), vector=row["vector"], payload=row["metadata"])
            for row in batch if "vector" in row
        ]
       self.client.upsert(collection_name="rag_collection", points=points)
```

对于该图，我们创建了它`pipelines/ingestion/indexing/neo4j.py`。我们使用 Cypher`MERGE`语句来确保幂等性，如果我们运行两次数据摄取，我们不希望出现重复节点，这样可以避免 CPU 或 GPU 节点上可能出现的内存问题。

```python
# pipelines/ingestion/indexing/neo4j.py
from neo4j import GraphDatabase

class Neo4jIndexer:
    """使用幂等MERGE查询写入图数据。"""
	def __init__(self):
        self.driver = GraphDatabase.driver("bolt://neo4j-cluster:7687", auth=("neo4j", "pass"))
    def write(self, batch):
        with self.driver.session() as session:
            # 将批处理数据扁平化并执行单个事务以提高性能
            session.execute_write(self._merge_graph_data, batch)
```

### 使用 Ray 实现事件驱动型工作流

最后，我们需要将所有这些组件整合在一起。我们不需要单个脚本，而需要一个流水线，其中读取、分块和嵌入操作可以在我们 CPU 集群的不同节点上并行执行。为此，我们需要创建`pipelines/ingestion/main.py`一个调度器。

[k8s集群](./Knowledge_Added/k8s集群.md)

我们将使用**Ray Data**创建一个延迟执行的 DAG（有向无环图）。

```python
# pipelines/ingestion/main.py
import ray
from pipelines.ingestion.embedding.compute import BatchEmbedder
from pipelines.ingestion.indexing.qdrant import QdrantIndexer

def main(bucket_name:str, prefix:str):
    """
    主要编排流程
    """
    # 1.使用Ray Data从S3读取数据（延迟加载）
    # 这会自动将读取任务分配到各个工作进程
    ds = ray.data.read_binary_files(
    	paths=f"s3://{bucket_name}/{prefix}",
        include_paths=True
    )
    
    # 2.解析和分块（映射阶段）
    # num_cpus=1 告诉Ray为每个解析任务预留1个cpu核心
    chunked_ds = ds.map_batches(
    	process_batch,
        batch_size=10, # 每个工作进程一次处理10个文件
        num_cpus=1
    )
    
    # 3.FORK: 分支A - 向量嵌入（GPU密集型）
    # 我们使用一个类Actor（BatchEmbedder）来维护与Ray Serve的连接
    vector_ds = chunked_ds.map_batches(
    	BatchEmbedder,
        concurrency = 5,
        num_gpus = 0.2,
        bacth_size = 100
    )
    
    # 4.FORK：分支B-图提取（LLM密集型）
    # 速度较慢，因此我们可以设置更高的并发性或专用节点
    graph_ds = chunked_ds.map_batches(
    	GraphExtractor,
        concurrency = 10,
        num_gpus = 0.5,
        batch_size = 5
    )
    
    # 5.索引（写入数据库）
    # 触发执行
    vector_ds.write_datasource(QdrantIndexer())
    graph_ds.write_datasource(Neo4jIndexer())
    
    print("数据导入作业已成功完成。")
```

为了在我们的 Kubernetes 集群上运行此程序，我们需要定义运行时环境`pipelines/jobs/ray_job.yaml`。这确保我们的工作进程安装了所有必要的 Python 依赖项。

```yaml
# pipelines/jobs/ray_job.yaml
entrypoint: "python pipelines/ingestion/main.py"
runtime_env:
	working_dir: "./"
	pip: ["boto3", "qdrant-client", "neo4j", "langchain", "unstructured"]
```

在企业架构中，我们不会手动触发此操作。**我们使用事件驱动模式**。**当文件上传到 S3 时**，会触发一个事件，该事件会触发在`pipelines/jobs/s3_event_handler.py`.

```python
# pipelines/jobs/s3_event_handler.py
from ray.job_submission import JobSubmissionClient

def handle_s3_event(event, context):
    """由S3 上传触发 -> 提交Ray作业"""
    client = JobSubmissionClient("http://rag-ray-cluster-head-svc:8265")
    client.submit_job(
    	entrypoint=f"python pipelines/ingestion/main.py {bucket} {key}",
        runtime_env={"working_dir" : "./"}
    )
```

最后，为了测试整个过程，我们`scripts/bulk_upload_s3.py`使用多线程上传将之前准备好的噪声数据集上传到我们的 S3 存储桶中。

```python
# scripts/bulk_upload_s3.py
from concurrent.futures import ThreadPoolExecutor

def upload_directory(dir_path, bucket_name):
    """高性能多线程S3上传器"""
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 将本地文件映射到S3上传任务
        executor.map(upload_file, files_to_upload)
```

我们现在已经构建了数据摄取层。现在，只需向 Kubernetes 集群添加更多节点，我们的系统就可以扩展到处理数百万份文档。

在下一部分中，我们将使用分布式模式构建**模型服务器层**，该层将包含我们的模型。

## AI计算层

现在我们有了数据摄取管道，接下来需要处理这些数据的流程。在单体应用中，您可以将 LLM 直接加载到 API 服务器中。

然而，在**企业级红黄绿灯平台**中，这是一个严重的错误……

> 在 Web 服务器中加载 700 亿参数模型会严重影响请求吞吐量，并使扩展成为不可能。

![](./Image/计算层.jpg)我们需要将 FastAPI 与 AI 模型解耦。我们将使用**RayServe**将模型托管为独立的微服务，这些微服务可以根据 GPU 可用性和传入流量自动扩展。

### 模型配置及硬件映射

在生产环境中，我们绝不会将模型参数硬编码到代码中。我们需要灵活的配置，以便能够在不重写代码的情况下切换模型、调整量化级别和优化批次大小。

![](./Image/模型配置.jpg)

让我们来定义一下我们的核心`models/llm/llama-70b.yaml`。我们使用**Llama-3-70B-Instruct**，但由于 70B的参数在 FP16 模式下需要约 140GB 的显存，我们使用**AWQ 量化技术**将其适配到更经济实惠的 GPU 上，这实际上是……

> 许多公司将数据和代理上作为 RAG 系统的重要组成部分，而不是用于生成最终答案的 AI 模型。

```bash
# models/llm/llama-70b.yaml
model_config:
	# HuggingFace Model ID
	model_id: "meta-llama/Meta-Llama-3-70B-Instruct"
	
	# Quantization: AWQ is SOTA for high-throuput inference on Nvidia GPUs
	quantization: "awq"
	
	# 上下文窗口 : Llama-3 支持8k，我们设置在8192
	max_model_len: 8192
	
	# 批处理：对于“数千用户并发”至关重要
	max_num_seqs: 128
	
	# 硬件要求（映射到AWS实例）
	gpu_memory_utilization: 0.90
	tensor_parallel_size: 4
	
	#停止标记
	stop_token_ids: [128001, 128009] # <|eot_id|> 特定于Llama-3分词器
```

注意这里`tensor_parallel_size: 4`。这是企业级配置。它告诉服务引擎，这个单个模型对于单个 GPU 来说太大了，因此必须将权重矩阵无缝地分配到 4 个 GPU 上。

我们还保留了一个较小模型的配置，`models/llm/llama-7b.yaml`我们可以将其用于查询重写或摘要等较轻的任务，以节省成本。

```bash
# models/llm/llama-7b.yaml
model_config:
	model_id: "meta-llama/Meta-Llama-3-8B-Instruct"
	
	# 8B可以轻松放入单个T4或A10GPU上
	quantization: "awq"
	
	max_model_len: 8192
	
	#由于模型尺寸小，吞吐量可能更高
	max_num_seqs: 256
	gpu_memory_utilization: 0.85
	tensor_parallel_size: 1
	
	stop_token_ids: [128001, 128009]
```

在检索方面，我们在 . 中定义了我们的嵌入模型配置`models/embeddings/bge-m3.yaml`。BGE-M3 非常出色，因为它能够处理密集、稀疏和多语言检索，这对于用户可以使用非英语语言提问的全球企业平台来说非常重要。

```bash
# models/embeddings/bge-m3.yaml
model_config:
	model_id: "BAAI/bge-m3"
	
	# 嵌入生成的批次大小（越大，数据导入速度越快）
	batch_size: 32
	
	# 余弦相似度向量归一化
	normalize_embeddings: true
	
	# 精度：FP16在T4/A10 GPU上速度更快
	dtype: "float16"
	
	# 最大序列长度（BGE-M3支持8192，但为了提高RAG精度，以512分块
	max_seq_length: 8192
```

最后，为了提高准确率，我们将使用 中定义的重排序器`models/rerankers/bge-reranker.yaml`。该模型对检索到的前几条文档进行重新评分，以在它们进入 LLM 之前过滤掉假阳性，从而显著减少幻觉。

```bash
# models/rerankers/bge-reranker.yaml
model_config:
	model_id: "BAAI/bge-reranker-v2-m3"
	
	# 精度设置
	dtype: "float16"
	
	# 输入对（查询+文档）的最大长度
	max_length: 512
	
	# 重排序的批次大小
	batch_size: 16
```

### 使用 vLLM 和 Ray 为 AI 模型提供服务

现在我们需要实际运行这些模型。**标准的 HuggingFace 流水线速度太慢，无法满足高并发生产环境的需求**。

> 我们将使用**vLLM**，这是一个高吞吐量的服务引擎，它使用 PagedAttention 来高效地管理内存。

![](./Image/服务逻辑.jpg)我们将 vLLM 封装在 Ray Serve 部署中`services/api/app/models/vllm_engine.py`。该脚本处理引擎的初始化，并公开一个用于生成的异步端点。

```python
# services/api/app/models/vllm_engine.py
from ray import serve
from vllm import AsyncLLMEngine, EngineArgs, SamplingParams
from transformers import AutoTokenizer # 用于带有聊天模板的分词器
import os

@serve.deployment(autoscaling_config={"min_replicas":1, "max_replicas":10}, ray_actor_ootions={"num_gpus": 1})
class VLLMMDeployment:
    def __init__(self):
        model_id = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-70B-Instruct")
        # 1.加载分词器以正确格式聊天
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        args = EngineArgs(
        	model=model_id,
            quantization="awq",
            gpu_memory_utilization=0.90,
            max_model_len=8192
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(args)
    async def __call__(self, request):
        body = await request.json()
        messages = body.get("messages", [])
        
        # 2.使用标准模板应用程序
        # 这可以正确处理特定模型的系统提示、特殊令牌和角色
        prompt = self.tokenizer.apply_chat_template(
        	messages,
            tokenize = False,
            add_generation_prompt = True
        )
        
        sampling_params = SamplingParams(
        	temperature = body.get("temperature", 0.7),
            max_token_ids = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        )
        
        request_id = str(os.urandom(8).hex())
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        text_output = final_output.outputs[0].text
        
        return {"choices": [{"messages":{"content":text_output, "role": "assistant"}}]}
    
app = BLLMDeployment.bi
```

