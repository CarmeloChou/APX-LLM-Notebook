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

第一步定义基础的应用设置，比如环境、log等级、[JWT认证密钥](./Knowledge_Added/JWT.md)

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



我们使用公式`base * (2 ^ retries) + random_jitter`来计算每次重试的延迟。这有助于我们避免**“群殴”** **问题，**即多个客户端同时重试的情况。

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
	# 要使用的Ray Serve 端点
    endpoint
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
    
app = BLLMDeployment.bind()
```

通过使用` @serve.deployment`，我们告诉 Ray 管理此类的生命周期。这`autoscaling_config`是关键的企业级特性：如果流量激增，Ray 会检测到负载并自动启动更多副本，并在空闲时缩减规模以节省云成本。

### 服务嵌入和重新排序

嵌入模型不需要像 vLLM 那样庞大的后端，但它们确实需要高效的批处理**。**

> **如果 50 个用户同时进行搜索，我们希望在一次 GPU 处理中对所有 50 个查询进行编码，而不是 50 次顺序处理**。

我们在 中实现了这一点`services/api/app/models/embedding_engine.py`。请注意，我们的分配方式`num_gpus: 0.5 `允许我们将两个嵌入模型打包到单个 GPU 上，从而大大节省了成本。

```python
# services/api/app/models/embedding_engine.py
from ray import serve
from sentence_transformers import SentenceTransformer
import os
import torch

@serve.deployment(
	num_replicas=1,
    ray_actor_options={"num_gpus":0.5} # 共享GPU
)

class EmbedDeployment:
    def __init__(self):
        # 将模型加载到GPU上
        model_name = "BAAI/bge-m3"
        self.model = SentenceTransformer(model_name, device="cuda")
        
        # 编译以提高速度（可选，需要PyTorch 2.0+）
        self.model = torch.compile(self.model)
    async def __call__(self, request):
        body = await request.json()
        texts = body.get("text")
        task_type = body.get("task_type", "document")
        
        # BGE-M3对指令的处理方式不同，此处简化：
        if isinstance(texts, str):
            texts = [texts]
            
        # 编码
        embeddings = self.model.encode(
        	texts,
            batch_size = 32,
            normalize_embeddings = True
        )
        
      	return {"embeddings": embeddings.tolist()}

app = EmbedDeployment.bind()
```

此部署方案使用同一个共享资源池来处理文档嵌入（数据摄取期间）和查询嵌入（聊天期间）。我们利用`torch.compile`该资源池优化模型执行图，最大限度地发挥 GPU 的性能。

### 异步内部客户端

最后，我们的 API 层需要一种与这些 Ray 服务通信的方式。由于这些模型作为独立的微服务运行，我们通过 HTTP 进行通信。

![](./Image/异步调用.jpg)

我们需要非阻塞客户端来确保 FastAPI 服务器在等待 GPU 时保持响应。让我们创建一个`services/api/app/clients/ray_llm.py`包含连接池和重试机制的 LLM 通信处理程序。

```python
# services/api/app/clients/ray_llm.py
import httpx
import logging
import backoff
from typing import List, Dict, Optional
from services.api.app.config import settings

logger = logging.getLogger(__name__)

clas RayLLMClient:
    """
    具有正确连接池的异步客户端
    """
    def __init__(self):
        self.endpoint = settings.RAY_LLM_ENDPOINT
        # 客户端在startup_event中初始化
        self.client: Optional [httpx.AsyncClient] = None
    async def start(self):
        """在应用程序启动期间调用"""
        # 限制：防止打开过多的Ray连接
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=5)
        self.client = httpx.AsyncClient(
        	timeout = 120.0,
            limits = limits
        )
        logger.info("Ray LLM 客户端已初始化")
    async def close(self):
        """在应用关闭期间调用"""
        if self.client:
            await self.client.aclose()
    @backoff.on_exception(backoff.expo, httpx.HTTPError, max_tries=3)
    async def chat_completion(self, message:List[Dict], temperature:float=0.7, json_model:bool=False)->str:
        if not self.client:
            raise RuntimeError("客户端未初始化。请先调用start()")
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024            
        }
        
        response = await self.client.post(self.endpoint, json=payload)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]

# 全局实例（由main.py中的Lifespan管理)
llm_client = RayLLMClient()
```

这里所用的机制`backoff`对分布式系统至关重要。如果网络出现故障或 Ray 繁忙，系统不会崩溃；而是会进行指数级等待并重试，从而确保高可用性。

同样，我创建了`services/api/app/clients/ray_embed.py`的嵌入服务。该客户端处理**“查询”**嵌入（用于搜索）和**“文档”**嵌入（用于摄取）之间的区别。

> 在检索增强生成（RAG）系统中，“查询嵌入”通常指对用户搜索问题生成的向量表示，用于实时检索；而“文档嵌入”指对知识库中文档块生成的向量，用于预先构建可搜索的索引。这种区分确保了搜索的准确性和效率。

随着这些客户端的编码完成，我们的 API 代码（我们接下来将构建）可以将 700 亿参数模型视为简单的异步函数调用，完全消除 GPU 管理和分布式推理的复杂性。

在下一部分中，我们将构建**Agentic 管道（将 RAG 转换为 Agentic RAG）**，该逻辑决定何时使用这些模型来回答用户问题。

## 智能体人工智能管道

我们有数据摄取管道和运行在分布式集群上的 AI 模型。在一个简单的 RAG 应用中，你可能只需要将检索调用链接到 LLM 生成调用即可。

![](./Image/智能体管道.jpg)

然而，对于**企业代理平台而言**，线性链是脆弱的。

> 当用户改变话题、提出数学问题或说话含糊不清时，它们就会失效。

我们正在使用**FastAPI**和**LangGraph构建一个**事件驱动代理。这使得我们的系统能够**“推理”**用户的意图循环，自我纠正，并动态地选择工具，同时异步处理数千个并发的 WebSocket 连接。

### API 基础与可观测性

首先，我们需要定义环境和安全标准。企业级 API 不能是一个**黑盒**，我们需要结构化的日志和追踪信息来调试特定查询为何耗时 5 秒而不是 500 毫秒。

![](./Image/FastAPI.jpg)

让我们验证一下我们的依赖项`services/api/requirements.txt`。我们引入这些依赖项`fastapi`是为了提高速度、`langgraph`便于编排和`opentelemetry`便于观察。

```bash
# services/api/requirements.txt

# 核心框架
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.6.0
pydantic-settings==2.1.0
simpleeval==0.9.13 # 安全的数学计算

# 异步数据库和缓存
sqlalchemy==2.0.25
asyncpg==0.29.0
redis==5.0.1

# AI和LLM客户端
openai==1.10.0 # 标准客户端，通常用于兼容的端点
anthropic==0.8.0 # 如果使用Claude作为备用
tiktoken==0.2.2
sentence-transformers==2.3.1
transformers==4.37.0 # 用于分词器模板

# 图数据库和向量数据库
neo4j==5.16.0 
qdrant-client==1.7.3 

# 代理框架
langchain==0.1.5 
langgraph==0.0.21 

# 可观测性与运维
opentelemetry-api==1.22.0 
opentelemetry-sdk==1.22.0 
opentelemetry-exporter-otlp==1.22.0 
prometheus-client==0.19.0 
python-json-logger==2.0.7 
backoff==2.2.1 

# 安全
python-jose[cryptography]==3.3.0 
passlib[bcrypt]==1.7.4 
python-multipart==0.0.6 

# 实用工具
boto3==1.34.34 
httpx==0.26.0 
tenacity==8.2.3
```

此需求文件确保我们拥有高性能系统所需的所有异步驱动程序（`asyncpg`，`redis`）和可观测性工具（ ）。其中包含和重点突出是为了提高可靠性，即内部调用失败时自动重试的逻辑。`opentelemetry``backoff``tenacity`

接下来，我们`services/api/app/config.py`使用 Pydantic 设置进行创建。这会在启动时验证所有数据库 URL 和 API 密钥是否存在，从而防止后续运行时崩溃。

```python
# services/api/app/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    应用程序配置。
    自动读取环境变量（不区分大小写）
    """
    # 通用
    ENV: str = "prod"
    LOG_LEVEL: str = "INFO"
    
    # 数据库（Aurora Postgres）
    DATABASE_URL: str # 例如，postgresql+asyncpg://user:pass@host:5432/db
    
    # Redis(缓存)
    REDIS_URL: str
    
    # 向量数据库(Qdrant)
    QDRANT_HOST: str = "qdrant-service"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "rag_collection"
        
    # 图数据库（Neo4j)
    NEO4J_URL: str = "bolt://neo4j-cluster:7687"
	NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str #敏感信息
    
    # AWS S3(文档)
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str
    
    # Ray Serve(内部LLM/嵌入)
    RAY_LLM_ENDPOINT: str = "http://llm-service:8000/llm"
    RAY_EMBED_ENDPOINT: str = "http://embed-service:8000/embed"
        
    # 安全
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    
    class Config:
        env_file = ".env"

# 实例化单例
settings = Settings()
```

它作为整个 API 的中心数据源。通过使用 Pydantic，我们确保如果缺少关键密钥`JWT_SECRET_KEY`，Pod 将立即启动失败**（快速失败原则）**，并立即向运维团队发出警报。

为了使日志能够被 Datadog 或 Splunk 等工具读取，我们将实现结构化的 JSON 日志记录`services/api/app/logging.py`。

```python
# services/api/app/logging.py
import logging
import json
import sys
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    将日志记录格式化为JSON对象。
    包含时间戳、级别和消息。
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        
        # 添加额外信息（如有）
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        # 添加额外字段（例如：user_id, trace_id)
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        
        return json.dumps(log_record)
    
    def setup_logging():
        """
        配置根日志记录输出JSON格式到标准输出
        """
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # 移除默认处理程序以避免重复日志
        if root_logger.handlers:
            root_logger.handlers = []
        
        root_logger.addHandler(handler)
        
        # 静默部分杂乱的库
        logging.getLogger("uvicorn.access").disabled = True
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
# 导入时初始化
setup_logging()
```

当运行 50 个 Pod 时，标准的文本日志就显得力不从心了。JSON 日志允许我们像查询数据库一样查询日志，按错误级别或特定请求 ID 进行筛选，从而跨分布式节点追踪错误。

我们还启用了分布式追踪功能`services/api/app/observability.py`。该功能可以追踪从 API 到 Redis，再到 Vector DB，最后到 Ray 集群的请求流。

```python
# services/api/app/observability.py
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from libs.observability.tracing import configure_tracing

def setup_observability(app: FastAPI):
    """
    配置OpenTelemetry并将其附加到FastAPI应用
    """
    # 1.配置追踪器（将数据发送到Jaeger/Datadog)
    configure_tracing(service_name="rag-api-service")
    
    # 2.自动检测FastAPI
    # 这将自动为每个请求创建span
    FastAPIInstrumentor.instrument_app(app)
```

我们都知道安全不容妥协。我们在系统中实现了 JWT（JSON Web Token）验证`services/api/app/auth/jwt.py`。该中间件确保只有授权用户才能查询我们昂贵的 GPU 资源。

```python
# services/api/app/auth/jwt.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from services.api.app.config import settings
import time

# OAuth2 scheme 告诉Swagger UI将token发送到哪里
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    验证Authorization标头中的JWT Token
    解码用户信息（ID、角色、权限）
    """
    credentials_exception = HTTPException(
    	status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        #1. 解码令牌
        # 使用配置中定义的密钥验证签名
        payload = jwt.decode(
        	token,
            settins.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        user_id: str = payload.get("sub")
        role: str = payload.get("role", "user")
            
        if user_id is None:
            raise credentials_exception
        
        # 2.检查过期时间（如果jwt.decode已经执行此操作，则此步骤是多余的，但为了安全起见，这样做比较好）
        exp = payload.get("exp")
        if exp and time.time() > exp:
            raise HTTPException(status_code=401, detail="令牌已过期")
        return {
            "id": user_id,
            "role": role,
            "permissions": payload.get("permissions", [])
        }
    except JWTError:
        raise credentials_exception
```

最后，我们在`libs/schemas/chat.py`中定义数据合约。这确保前端发送的内容与我们预期的完全一致，并接收一致的响应结构。

```python
# lib/schemas/chat.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
        
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = True
    
    # 高级用户可选过滤器（例如，“仅搜索人力资源档案”）
    filters: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """
    非流式调用的标准响应结构
    """
    answer: str
    session_id: str
    # 引用对于“工业”RAG建立信任至关重要
    citations: List[Dict[str, str]] = []
    latency_ms: float

class RetrievalResult(BaseModel):
    """
    由检索节点使用
    """
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]
```

这种严格的类型定义使我们能够自动生成 Swagger 文档，并防止“垃圾数据”导致我们的代理在管道下游崩溃。

### 异步数据网关

可扩展的 API 绝不能阻塞主线程。我们需要为所有数据库提供异步客户端，这样单个 API 工作进程就可以处理数百个等待的连接，同时数据库也在处理数据。

![](./Image/异步数据网关.jpg)

首先，我们需要构建`services/api/app/clients/redis.py`一个负责高速缓存和速率限制的程序。

```python
# services/api/app/clients/redis.py
import redis.asyncio as redis
from services.api.app.config import settings

class RedisClient:
    """
    单例Redis连接池
    用于速率限制和语义缓存存储
    """
    def __init__(self):
        self.redis = None
    
    async def connect(self):
    	if not self.redis:
            # decode_responses=True 表示我们返回的是字符串而不是字节
            self.redis = redis.from_url(
            	settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
	
    async def close(self):
        if self.redis:
            await self.redis.close()

	def get_client(self):
        """返回原始Redis客户端实例"""
        return self.redis

# 全局实例
redis_client = RedisClient()
```

这种单例模式可以高效地重用连接，从而防止经常困扰架构不良应用程序的**“连接过多”错误。**

接下来，我们将创建`services/api/app/clients/qdrant.py`向量搜索。我们使用 Qdrant 客户端的异步版本来执行非阻塞相似性搜索。

```python
# services/api/app/clients/qdrant/py
from qdrant_client import QdrantClient, AsyncQdrantClient
from services.api.app.config import settins

class VectorDBClient:
    """
    Qdrant的异步客户端
    """
    def __init__(self):
        self.client = AsyncQdrantClient(
        	host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            # 在生产环境中，我们启用gRPC以获得略微更快的性能
            prefer_grpc=True
        )
        
	async def search(self, vector:list[float], limit:int=5):
        """
        执行语义搜索
        """
        return await self.client.search(
        	collection_name=settins.QDRANT_COLLECTION,
            query_vector=vector,
            limit=limit,
            with_payload=True
        )
    
# 全局实例
qdrant_client=VectorDBClient()
```

对于`services/api/app/clients/neo4j.py`图搜索，此驱动程序管理连接池以高效运行 Cypher 查询。

```python
# services/api/app/clients/neo4j.py
from neo4j import GraphDatabase, AsyncGraphDatabase
from services.api.app.config import settings
import logging

logger = logging.getLogger(__name__)

class Neo4jClient:
    """
    Neo4j驱动程序的单例包装器
    支持异步执行，以处理高并发API
    """
    def __init__(self):
        self._driver = None
    def connect(self):
        """初始化连接池"""
        if not self.driver:
            try:
                # 创建带身份验证的驱动程序
                self._driver = AsyncGraphDatabase.driver(
                settings.NEO4J_USER, settings.NEO4J_PASSWORD
                )
                logger.info("成功连接到Neo4j")
            except Exception as e:
                logger.error(f"连接到Neor4j失败{e}")
                raise
                
	async def close(self):
        """关闭连接池"""
        if self._driver:
            await self._driver.close()
	
    async def query(self, cypher_query:str, parameters: dict = None):
        """
        执行Cypher查询并返回结果
        """
        if not self._driver:
            await self.connect()
		
        async with self._driver.session() as session:
            result = await session.run(cypher_query, parameters or {})
            return [record.data() async for record in result]

# 全局实例
neo4j_client = Neo4jClient()
```

### 上下文记忆和语义缓存

智能体的智能程度取决于其记忆力。我们需要使用Postgres来存储完整的对话历史记录，以便LLM能够回忆起5回合前说过的话。

![](./Image/上下文记忆.jpg)

为此，我们需要在`services/api/app/memory/models.py`中定义模式。

```python
# services/api/app/memory/models.py
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text, DataTime, JSON
from datetime import datetime

Base = declarative_base()

class ChatHistory(Base):
    """
    用于 chat_history 表的SQLAlchemy模型。存储上下文原始对话日志
    """
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 会话ID将消息连接到特定对话线程
    session_id = Column(String(255), index=True, nullable=False)
	
    # 用于多租户的用户ID
    user_id = Column(String(255), index=True, nullable=False)
    
    # 角色：'user','assistant' or 'system'
    role = Column(String(50), nullable=False)
    
    # 实际消息内容
    content = Column(Text, nullable=False)
    
    # 元数据：令牌使用情况、延迟、使用的模型版本
    metadata_ = Column(JSON, default={}, nullable=True)
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
```

然后我们将实现 CRUD 逻辑`services/api/app/memory/postgres.py`。我们按时间倒序获取历史记录，以便将最新上下文提供给 LLM。

```python
# services/api/app/memory/postgres.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, JSON, DateTime, Integer, Text
from datetime import datetime
from services.api.app.config import settings

# 1.数据库设置
Base = declarative_base()

# 2.定义聊天记录表
class ChatHistory(Base):
    """
    存储每次对话回合
    """
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True) # 用户对话ID
    user_id = Column(String, index=True)
    role = Column(String) # 'user'or 'assistant'
    content = Column(Text) # 文本消息
    metadata_ = Column(JSON, default={}) #额外信息（延迟、令牌）
    created_at = Column(DataTime, default=datetime.utcnow)
    
# 3.异步引擎和会话
engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(
	bind = engine, class_ = AsyncSession, expire_on_commit=False
)

class PostgresMemory:
    """
    用于持久化会话状态管理器
    """
    async def add_message(self, session_id: str, role: str, content: str, user_id: str):
        async with AsyncSessionLocal() as session:
            async with session.begin():
                msg = ChatHistory(
                	session_id = session_id,
                    role = role,
                    content = content,
                    user_id = user_id
                )
                session.add(msg)
                # 通过 async with session.begin() 自动提交
                
	async def get_history(self, session_id: str, limit: int=10):
        """
        获取上下文窗口中的最后N条消息
        """
        from sqlalchemy import select
        async with AsyncSessionLoacl() as session:
            result = await session.execute(
            	select(ChatHistory)
                .where(ChatHistory.session_id == session_id)
                .order_by(ChatHistory.create_at.desc())
                .limit(limit)
            )
            # 反向排序以获取时间顺序（最早 -> 最新)
            return result.scalars().all()[::-1]

postgres_memory = PostgresMemory()
```

为了节省成本，我们在系统中实现了**语义缓存**`services/api/app/cache/semantic.py`。如果一个用户问**“什么是 Kubernetes？”**，另一个用户问**“解释一下 K8s”**，则嵌入相似度会很高。

> 在 70B 型号上，我们可以立即提供缓存的答案，而无需消耗 GPU 积分。

```python
# services/api/app/cache/semantic.py
import json
import logging
from typing import Optional
from services.api.app.clients.ray_embed import embed_client
from services.api.app.client.qdrant import qdrant_client
from services.api.app.config import settings

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    使用向量搜索实现语义缓存
    我们不进行精确的字符串匹配，而是根据语义进行匹配
    """
    async def get_cached_response(self, query:str, threshold: float=0.95) -> Optional[str]:
        """
        检查缓存中是否存在类似的查询
        """
        try:
            # 1.嵌入传入的查询（快速CPU/GPU调用）
            vector = await embed_client.embed_query(query)
            
            # 2.在Qdrant中的特定“缓存”集合中搜索
            # 如果配置了Redis Vector， 则在Redis Vector中搜索
            results = await qdrant_client.client.search(
            	collection_name = "semantic_cache",
                query_vector = vector,
                limit = 1,
                with_payload = True,
                score_threshold = threshold # 仅搜索极其相似的查询
            )
			if results:
                logger.info(f"语义缓存命中！得分{results[0].score}")
                return results[0].payload["answer"]
		
        except Exception as e:
            logger.warning(f"于一缓存查找失败{e}")
            
		return None

    async def set_cached_response(self, query: str, answer: str):
        """将回答保存到缓存中"""
        try:
            # 1.嵌入查询
            vector = await embed_client.embed_query(query)
            
            # 2.保存到Vector DB
            import uuid
            from qdrant_client.http import models
            
            await qdrant_client.client.upsert(
            	collection_name = "semantic_cache",
                points = [
                    moedels.PointStruct(
                    	id = str(uuid.uuid4()),
                        vector = vector,
                        payload = {"query": query, "answer": answer}
                    )
                  
                ]
            )
            
		except Exception as e:
            logger.warning(f"写入语义缓存失败{e}")

semantic_cache = SemanticCache()
```

这肯定会降低企业环境中常见问题解答 (FAQ) 的延迟。

### 使用 LangGraph 的工作流

这是智能体的核心。我们将对话视为一个状态机。智能体可以在**“思考”**、**“检索”**、**“使用工具”**和**“回答”**之间转换。

![](./Image/LangGraph工作流.jpg)首先，我们定义`AgentState`in `services/api/app/agents/state.py`。该字典承载着对话在图节点间流动时的上下文。

```python
# services/api/app/agents/state.py
from typing import TypedDict, Annotated, List, Union
import operator

class AgentState(TypedDict):
    """
    LangGraph中节点间传递的状态对象。
    跟踪对话历史和当前步骤数据
    """
    # 使用 operator.add 表示新消息追加，而不是覆盖
    messages: Annotated[List[dict], operator.add]
    
    # 从RAG（向量+图）中检索的上下文
    documents: List[str]
        
    # 当前处理中的问题
    current_query: str
        
	# 规划器内部暂存的
    plan: List[str]
```

规划**节点**（`services/api/app/agents/nodes/planner.py`）会查看用户查询并决定如何处理。它充当路由器的角色，将流量导向检索、工具或直接回答。

```python
# services/api/app/agents/nodes/planner.py
import json
import logging
from services.api.app.agents.state import AgentState
from services.api.app.clients.ray_llm import llm_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
您是 RAG 规划代理。
分析用户查询和对话历史记录。
决定下一步：
1. 如果用户问候（Hello/Hi），则输出“direct_answer”。
2. 如果用户询问需要数据的具体问题，则输出“retrieve”。
3. 如果用户询问数学/代码，则输出“tool_use”。
仅输出 JSON 格式：
{ 
    "action": "retrieve" | "direct_answer" | "tool_use", 
    "refined_query": "独立搜索查询", 
    "reasoning": "您选择此操作的原因" 
} 
"""
```

然后我们就可以简单地定义异步规划器节点……

```python
async def planner_node(state: AgentState) -> dict:
    """
    决定在LangGraph中的路径
    """
    logger.info("规划节点：正在分析查询...")
    # 提取最新用户消息
    # state['messages']是一个字典或对象列表
    last_message = state["messages"][-1]
    user_query = last_message.content if hasattr(last_message, 'content') else last_message['content']
    
    # 调用LLM进行规划
    try:
        response_text = await llm_client.chat_completion(
        	messages = [
                { "role" : "system" , "content" : SYSTEM_PROMPT}, 
                { "role" : "user" , "content" : user_query} 
            ], 
            temperature= 0.0  # 确定性规划
        )
        
        # 解析JSON
        plan = json.loads(response_text)
        
        logger.ingfo(f"Plan derived: {plan['action']}")
        
        # 更新状态
        return {
             "current_query" : plan.get( "refined_query" , user_query), 
            "plan" : [plan[ "reasoning" ]] 
        }
    
     except Exception as e: 
        logger.error( f"Planning failed: {e} " ) 
        # 回退：假设我们需要搜索
        return { 
            "current_query" : user_query, 
            "plan" : [ "Error in planning, defaulting to retrieval." ] 
        }
```

这个节点至关重要，因为它能防止系统对简单的问候语或无关的问题进行代价高昂的搜索。

我们还需要一个**检索节点**（`services/api/app/agents/nodes/retriever.py`），它将执行我们的**混合搜索**。

它并行调用 Qdrant（向量）和 Neo4j（图）`asyncio.gather`，从而兼具语义检索和结构检索的优势。

```python
# services/api/app/agents/nodes/retriever.py
import asyncio
from typing import Dict, List
from services.api.app.agents.state import AgentState
from services.api.app.clients.qdrant import qdrant_client
from services.api.app.clients.neo4j import neo4j_client
from services.api.app.clients.ray_embed import embed_client
import logging

logger = logging.getLogger(__name__)
async def retrieve_node(state: AgentState) -> Dict:
    """
    执行混合检索
    1.嵌入用户查询
    2.同时运行向量搜索和图搜索
    3.合并并去重结果
    """
    query = state["current_query"]
    logger.info(f"正在检索{query}的上下文")
	
    # 步骤1：获取查询的嵌入（调用Ray Serve）
    # 等待此操作，需要Qdrant向量
    query_vector = await embed_client.embed_query(query)
    
    # 步骤2：定义并执行任务
    # 任务A：向量搜索
    async def run_vector_search():
        results = await qdrant_client.search(vector=query_vector, limit=5)
        # 格式：“内容[来源：第一页]“
        return [f"{r.paload['text']} [来源：{r.payload['metadata']['filename']}]" for r in results]
    
    # 任务B：图搜索（结构关系）
    # 我们使用关键字来匹配或者预先定义编码模板
    async def run_graph_search():
        cypher = """
        CALL db.index.fulltext.queryNodes("entity_index", $query) YIELD node, score
        MATCH (node)-[r]->(neighbor)
        RETURN node.name + ' ' + type(r) + ' ' + neighbor.name as text
        LIMIT 5
        """
        # 注意：Lucene 语法的全文搜索可能需要模糊匹配 (~)。
        try:
            result = await neo4j_client.query(cypher, {"query": query})
            return [r['text'] for r in results]
        except Exception as e:
            logger.error(f"图搜索失败: {e}")
            return []
        
	# 步骤3 ：并行运行
    vector_docs, graph_docs = await asyncio.gather(run_vector_search(), run_graphe_search())
    
    # 步骤4：合并和去重
    # 我们优先考虑图结果中的特定事实，向量结果中的一般上下文。
    combined_docs = list(set(vector_docs + graph_docs))
    
    logger.info(f"检索到{len(combined_docs)}个文档。")
    
    # 更新状态
    return {"documents": combined_docs}
```

响应**节点**（`services/api/app/agents/nodes/responder.py`）接收检索到的文档，并通过 Ray Serve 使用 Llama-70B 模型合成最终答案。

```python
# services/api/app/agents/nodes/responder.py
from services.api.app.agents.state import AgentState
from services.api.app.clients.ray_llm import llm_client

async def generate_node(state: AgentState) -> dict:
    """
    使用检索到的文档合成最终答案
    """
    query = state["current_query"]
    documents = state.get("documents", [])
    
    # 构建上下文字符串
    context_str = "\n\n".join(documents)
    
    prompt = f"""
    您是一位乐于助人的企业助手，请使用以下上下文回答用户的问题
    
    上下文：
    {context_str}
    
    问题：
    {query}
    
    说明：
    1.使用[Source:Filename]引用来源。
    2.如果答案不在上下文中，请说”我的文档中没有该信息“
    3.请保持简洁和专业
    """
    
    #调用LLM
    answer = await llm_client.chat_completion(
    	messages = [{"role": "user", "content": prompt}],
        temperature = 0.3 # 低创意，高保真度
    )
    
    # 返回用于更新状态的字典（添加AI信息）
    return {
        "messages":[{"role": "assistant", "content":answer}]
    }
```

最后，我们将所有内容组合在一起`services/api/app/agents/graph.py`。这定义了工作流程：开始 -> 规划器 -> （检索器或工具） -> 响应器 -> 结束。

```python
# services/api/app/agents/graph.py
from langgraph.graph import StateGraph, END
from services.api.app.agents.state import AgentState
from services.api.app.nodes.retriever import retrieve_node
from services.api.app.agents.nodes.responder import generate_node
from services.api.app.agents.nodes.planner import planner_node

# 初始化图
workflow = StateGraph(AgentState)

# 1.定义节点（逻辑步骤）
# 这些函数（上面导入的）将在下一个nodes/文件夹中实现
workflow.add_node("planner", planner_node) # 重写查询/决定步骤
workflow.add_node("retriever", retrieve_node) # 调用Qdrant和Neo4j
workflow.add_node("responder", generate_node) # 调用Ray Serve LLM

# 2.定义边（流程）
# 开始->计划->检索->生成->结束
workflow.set_entry_point("planner")
workflow.add_edge("planner", "responder")
workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END) # 在更复杂的代理中，如果答案错误，我们可以循环返回

# 3.编译图
# 这将创建可运行的应用程序
agent_app = workflow.compile()
```

### 查询增强和应用程序入口点

为了提高检索准确率，我们使用了先进的 RAG 技术，其中之一是**HyDE**（假设文档嵌入），我们将在 中实现它`services/api/app/enhancers/hyde.py`。

![](./Image/查询增强.jpg)

> 它要求语言学习模型 (LLM) 产生一个虚假的答案，将其嵌入文本中，并用它来查找具有相似语义模式的真实文档。
>
> 这里应该可以自定义，还有adaptive rag等模式，可以看RAG all tech/ RAG anything

```python
# services/api/app/enhancers/hyde.py
from services.api.app.clients.ray_llm import llm_client

SYSTEM_PROMPT = """
您是一位乐于助人的助手。
请编写一个假设的段落来回答用户的问题。
它不需要完全正确，但必须使用
相关文档应有的正确词汇和结构。
问题：{question} 
"""

async def generate_hypothetical_document(question: str) -> str:
    """生成一个虚拟文档以改进向量相似度检索"""
    try:
        hypothetical_doc = await llm_client.chat_completion(
        messages=[
            {"role":"system", "content":SYSTEM_PROMPT.format(question=question)},
                  ],
              temperature = 0.7
        )
        return hypothetical_doc
    except Exception:
        return question # 回退
        
```

我们还使用解析器解决共引用（例如， “**它多少钱？”**`services/api/app/enhancers/query_rewriter.py` ） 。这确保搜索引擎能够获得完整的查询，例如“ ***Kubernetes\*****多少钱？”**。

```python
# services/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from services.api.app.clients.neo4j import neo4j_client
from services.api.app.clients.ray_llm import embed_client
from services.api.app.cache.redis import redis_client
from services.api.app.routes import chat, upload, health

@asynccontextmanager
async def lifespan(app: FastAPI):
    """集中式资源管理，在此处初始化所有连接池
    """
    # 1.启动
    print("正在初始化客户端...")
    neo4j_client.connect()
    await redis_client.connect()
    await llm_client.start()
    await embed_client.start()
    
    yield
    
    # 2.关闭
    print("正在关闭客户端...")
    await neo4j_client.close()
    await redis_client.close()
    await llm_client.close()
    await embed_client.close()
    
# FastAPI应用程序
app = FastAPI(title='企业RAG平台', version="1.0.0", lifespan=lifespan)

# 包含路由
app.include_router(chat.router, prefix="/api/v1/chat", tags=["聊天"])
app.include_router(upload.router, prefix="/api/v1/upload", tags=["上传"])
app.include_router(health.router, prefix="/health", tags=["健康"])

if __name__ == "__main__":
    import uvicorn
    # 在生产环境中，此程序通过Docker中的Gunicorn/Uvicorn运行
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

我们现在有了一个能够进行规划、检索和推理的智能体。但是，智能体系统需要与外部知识进行交互。在下一节中，我们将构建**工具和沙箱**层，使我们的智能体能够安全地执行代码并搜索网络。

## 工具和沙箱

在**企业级 RAG 平台**中，赋予 AI 模型直接执行代码或查询内部数据库的权限**会带来巨大的安全风险**。这样一来，黑客可以伪造 LLM 提示符，并访问关键的数据库信息。

![](./Image/工具箱和沙盒.jpg)

**我们需要分层策略**。我们将构建一套安全、确定性的工具，对于高风险操作（例如 Python 执行），我们将创建一个隔离的、加固的沙箱环境。

### 安全代码沙箱

允许逻辑逻辑模型（LLM）生成和执行Python代码功能强大，它可以解决复杂的数学问题、生成图表或分析CSV文件。但这也很危险。逻辑逻辑模型可能会产生幻觉`os.system("rm -rf /")`或试图窃取数据。

![](./Image/代码沙盒.jpg)

为了解决这个问题，我们构建了一个**沙箱微服务**。这是一个独立的 Docker 容器，用于运行不受信任的代码。**它没有网络访问权限，CPU/内存资源有限，并且超时时间很短**。

首先，我们将编写代码`services/sandbox/Dockerfile`，创建一个非root用户来限制容器内的权限。

```dockerfile
# services/sandbox/Dockerfile
# 1.使用最小基础镜像以减少攻击面
FROM python:3.10-slim

# 2.设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1\
	PYTHONUBUFFERED=1

# 3.创建非root用户（安全最佳实践）
RUN useradd -m -u 1000 sandbox_user

# 4.安顿依赖项
WORKDIR /app
COPY runner.py
RUN pip install flask

# 5.安全限制（应用于容器内部，由k8s limits.yaml 强制执行）
# 我们限制用户进程权限
USER sandbox_user

# 7.入口
EXPOSE 8080
CMD ["python", "runner.py"]
```

接下来是`services/sandbox/runner.py`.这是我们简单的 Flask 服务器，它接收代码，在单独的进程中执行代码（这样如果服务器挂起，我们就可以终止它），并捕获标准输出。

```python
# services/sandbox/runner.py 
from flask import Flask, request, jsonify 
import sys 
import io 
import contextlib 
import multiprocessing 

app = Flask(__name__)

def execute_code_safe(code:str, queue):
    """
    在单独的进程中运行代码，以允许硬超时。
    捕获标准输出。
    """ 
    # 重定向标准输出以捕获 print() 语句
    buffer = io.StringIO() 
     try : 
        with contextlib.redirect_stdout(buffer): 
            # 受限的全局变量可以增加轻微的安全层。
            exec (code, { "__builtins__" : __builtins__}, {}) 
        queue.put({ "status" : "success" , "output" : buffer.getvalue()}) 
    except Exception as e: 
        queue.put({ "status" : "error" , "output" : str (e)}) 
        
       
@app.route( "/execute" , methods=[ "POST" ] ) 
def  run_code (): 
    data = request.json 
    code = data.get( "code" , "" ) 
    timeout = data.get( "timeout" , 5 ) # 最大超时时间为 5 秒
    queue = multiprocessing.Queue() 
    p = multiprocessing.Process(target=execute_code_safe, args=(code, queue)) 
    p.start() 
    
    # 阻塞直到超时
    p.join(timeout) 
    
    if p.is_alive(): 
        p.terminate() 
        return jsonify({ "output" : "错误：执行超时。" }), 408 
        
    if  not queue.empty(): 
        result = queue.get() 
        return jsonify(result) 
        
    return jsonify({ "output" : "No output produced." }) 

if __name__ == "__main__" : 
    # Run on port 8080
     app.run(host= "0.0.0.0" , port= 8080 )
```

我们在 . 中定义了资源限制`services/sandbox/limits.yaml`。这将确保即使用户运行`while True`循环或内存炸弹，也只会导致沙箱 pod 崩溃，而不会导致节点崩溃。

```yaml
# services/sandbox/limits.yaml

# python代码解释器的硬性限制
runtime:
	# 终止信号前的最大执行实践
	timeout_seconds: 10
	
	# 脚本可分配的最大内存容量
	memory_limit_mb: 512
	
	# 最大CPU核心（防止加密货币挖矿循环）
	cpu_limit: 0.5
	
	# 网络许可：STRICTLY DENY
	allow_network: false
	files:
	# 最大输入数据
	max_input_size_mb: 5
	
	# 允许的模型（白名单）-可选，代码分析处理
	allowed_imports: ["math", "datetime", "json", "pandas", "numpy"]
```

我们同样需要在`services/sandbox/network-policy.yaml`中书写K8s网络规则。这是一个防火墙规则，禁止所有的外来访问。沙盒不能连接到互联网、数据库以及其他的服务。

```yaml
# services/sandbox/network-policy.yaml
apiVersion: networking.k8s.io/v1

kind: NetworkPolicy

metadata:
	name: sandbox-deny-egress
	namespace: default
	
spec:
	podSelector:
		matchLabels:
			app: sanbox-service
	policyTypes:
	
	- Egress
	
	# 默认禁止所有外来访问
	# 沙盒不能向外调用互联网以及任何服务
	egress: []
```

最终，这个API接口需要一个代理来和沙盒通信。`services/api/app/tools/sanbox.py`处理这个给内部沙盒服务发送HTTP请求。

```python
# services/api/app/tools/sandbox.py
import httpx
from services.api.app.config import settings

# 帮助在k8s中查找沙盒服务
SANDBOX_URL = "http://sandbox-service:8080/execute"

async def run_python_code(code:str) -> str:
    """
    工具：python解释器
    在安全、隔离的沙盒环境中运行python代码
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
            	SANDBOX_URL,
                json={"code":code, "timeout":5},
                timeout=6.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return f"Output:\n{data['output']}"
                else:
                    return f"Execution Error:\n{data['output']}"
			else:
                return f"Sandbox Error: Status {response.status_code}"
	
    except Exception as e:
        return f"Sandbox Connection Failed: {str(e)}"
```

### 固定工具及搜索工具

代码执行存在风险，因此我们尽可能选择更安全、确定性的工具。对于数学运算，我们在 `services/api/app/tools/calculator.py`使用`simpleeval`。它会解析表达式并进行计算，而无需使用 ` `eval()`.`

![](./Image/搜索工具.png)

```python
# services/api/app/tools/calculator.py
from simpleeval import simple_eval

def calculate(expression: str) -> str:
    """
    使用simpleeval安全地计算数学表达式。
    防止远程代码执行（RCE）。
    """
    # 1.长度限制，防止ReDoS攻击或内存耗尽
    if len(expression) > 100:
        return "错误：表达式过长。"
    try:
        # simple_eval解析抽象语法树（AST），并且只允许使用数学运算符。
        # 它无法访问全局变量、内置函数或操作系统/系统模块
        result = simple_eval(expression)
        return str(result)
    
    except Exception as e:
        return f"错误：{str(e)}"
```

为了检索知识，我们将图数据库和向量数据库作为工具公开，`services/api/app/tools/graph_search.py`使代理能够查询 Neo4j。

> 请注意，我们首先使用 LLM 提取*实体*，然后运行固定的 Cypher 查询。

我们**绝不**让LLM直接编写原始Cypher查询，以防止注入攻击。

```python
# services/api/app/tools/graph_search.py
from services.api.app.clients.neo4j import neo4j_client
from services.api.app.clients.ray_llm import llm_client
import json

SYSTEM_PROMPT = """
你是知识图谱助手。请从用户的问题中提取核心主题以执行搜索。
问题：{question}
仅输出JSON:{"entities":["list", "of", "names"]}
"""
```

然后我们可以定义我们的`graph_tool`异步方法……

```python
async def search_graph_tool(question: str) -> str:
    """通过提取实体安全地搜索图"""
    try:
        response_text = await llm_client.chat_completion(
        	messages=[{"role":"system", "content": SYSTEM_PROMPT.format(question=question)}],
            temperature= 0.0, 
            json_mode= True
        )
        data = json.loads(response_text)
        entities = data.get("entities", [])
        
        # 执行参数化查询（安全）
        cypher_query = """
        UNWIND $names AS target_name
        CALL db.index.fulltext.queryNodes("entity_index", target_name) YIELD node, score
        MATCH(node)-[r]-(neighbor)
        RETURN node.name AS source, type(r) AS rel, neighbor.name AS target
        LIMIT 10
        """
        results = await neo4j_client.query(cypher_query, {"names": entities})
        return str(results) if results else "未找到连接"
    except Exception as e:
        return f"图搜索错误：{str(e)}"
    
```

同样，我们需要编写代码`services/api/app/tools/vector_search.py`来暴露 Qdrant 端口。当规划器判断用户想要**“查找文档”**而不是**“回答问题”**时，这将非常有用。

```python
# services/api/app/tools/vector_search.py
from services.api.app.clients.qdrant import qdrant_client
from services.api.app.clients.ray_embed import embed_client

async def search_vector_tool(query: str) -> str:
    """工具：在向量数据库中搜索文档"""
    try:
        vector = await embed_client.embed_query(query)
        results = await qdrant_client.search(vector, limit=3)
    	fomatted = ""
        for r in results:
            meta = r.payload.get("metadata", {})
            formatted += f"-{r.payload.get("text", "")[:200]} ... [Source: {meta.get("filename")}]\n"
		return formatted if formatted else "未找到相关文档。"
    except Exception  as e:
        return f"搜索错误：{str(e)}"
```

为了获取外部信息，我们添加了此功能`services/api/app/tools/web_search.py`。它使用类似 Tavily（针对代理商优化）的 API 来获取实时信息。您可以使用任何其他您选择的提供商，但 Tavily 的可靠性高且价格实惠。

```python
# services/api/app/tools/web_search.py
import httpx
import os

async def web_search_tool(query: str) -> str:
    """工具：搜索互联网"""
    api_key = os.getenv("TAVILY_API_KEY")
    
    if not api_key: return "网络搜索已禁用"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
            	"https://api.tavily.com/search",
                json = {"api_key", "query", "max_results", 3}
            )
            
            data = response.json()
            results = data.get("results", [])
            return "\n".join([f"- {r["title"]} : {r["content"]}" for r in results] )

	except Exception as e:
        return f"Web search Error: {str(e)}"
```

### API路由和网关逻辑

工具准备就绪后，我们需要通过 REST 端点公开系统。`services/api/app/routes/chat.py`这是主要入口点。

![](./Image/路由逻辑.png)

```python
# services/api/app/routes/chat.py
import uuid
import json
import logging
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from services.api.app.auth.jwt import get_current_user
from services.api.app.agents.graph import agent_app
from services.api.app.agents.state import AgentState
from services.api.app.memory.postgres import postgres_memory
from services.api.app.cache.semantic import semantic_cache

router = APIRouter()
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

@router.post("/stream")
async def chat_stream(req: ChatRequest, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    """主聊天端点（流式传输）。协调缓存->历史记录->代理"""
    session_id = req.session_id or str(uuid.uuid4())
    user_id = user["id"]
    
    # 1. 检查缓存
    cached_ans = await semantic_cache.get_cached_response(req.message) 
    if cached_ans: 
        async  def  stream_cache (): 
            yield json.dumps({ "type" : "answer" , "content" : cached_ans}) + "\n" 
        return StreamingResponse(stream_cache(), media_type= "application/x-ndjson" ) 

    # 2. 加载历史
    history_objs = await postgres_memory.get_history(session_id, limit= 6 ) 
    history_dicts = [{ "role" : msg.role, "content" : msg.content} for msg in history_objs] 
    history_dicts.append({ "role" : "user" , "content" : req.message}) 

    # 3.初始化代理状态
    initial_state = AgentState(messages=history_dicts, current_query=req.message, documents=[], plan=[]) 
    async  def  event_generator (): 
        final_answer = "" 
        async  for event in agent_app.astream(initial_state): 
            node_name = list (event.keys())[ 0 ] 
            node_data = event[node_name] 
            
            # 发送状态更新
            yield json.dumps({ "type" : "status" , "node" : node_name}) + "\n" 
            if node_name == "responder"  and  "messages"  in node_data: 
                final_answer = node_data[ "messages" ][- 1 ][ "content" ] 
                yield json.dumps({ "type" : "answer" , "content" : final_answer}) + "\n" 
        
        # 后台：保存到数据库和缓存
        if final_answer: 
            await postgres_memory.add_message(session_id, "user" , req.message, user_id) 
            await postgres_memory.add_message(session_id, "assistant" , final_answer, user_id) 
            await semantic_cache.set_cached_response(req.message, final_answer) 
```

类似地，我们也可以创建健康监测功能……

```python
# services/api/app/routes/health.py
@router.get("/readiness")
async def readiness(response: Response):
    """
    K8s就绪探测
    检查与关键依赖项（Redis、数据库）的连接
    如果失败，K8s将停止向此Pod发送流量
    """
    status_report = {"redis": "down", "neo4j": "down"}
    is_healthy = True
    
    # 1.检查Redis
    try:
        r = redis_client.get_client()
        if await r.ping():
            status_report["redis"] = "up"
	except Exception:
        is_healthy = False
        
	# 2.检查Neo4j（仅连接性）
    try:
        # 驱动程序是单例，检查是否已初始化
        if neo4j_client._driver:
            status_report["neo4j"] = "up"
		else:
            is_healthy = False
	except Exception:
        is_healthy = False

	if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

	return status_report
```

最后，**保护我们的 API 免受滥用至关重要**。我们在网关层使用速率限制来实现这一功能`services/gateway/rate_limit.lua`。该机制运行在 Nginx 或 Kong 中，通过检查 Redis 来确保用户没有发送垃圾请求。

```lua
# services/gateway/rate_limit.lua
local redis = require "resty.redis"
local red = redis:new()

red:set_timeout(100)

local ok, err = red:connect("rag-redis-prod", 6379)

if not ok then return ngx.exit(500) end
local key = "rate_limit:" ..ngx.var.remote_addr

local limit = 100

local current = red:incr(key)

if current == 1 then red:expire(key, 60) end

if current > limit then
    ngx.status = 429
    ngx.say("速率限制已超出")
    return ngx.exit(429)
end
```

> 通过将此逻辑移至 Lua/Nginx，我们可以在不良流量到达我们的 Python API*之前*将其拦截，从而节省宝贵的 CPU 周期。

我们现在已经构建了整个应用栈：数据摄取、模型、代理和工具。下一节中，我们将配置基础设施，以便在 AWS 上运行这个庞大的系统。

## 基础设施即代码（IaC）

软件栈搭建完毕后，我们需要一个运行它的地方。在**企业级 RAG 平台**中，你不能仅仅在 AWS 控制台中点击按钮。那样会导致**“点击操作”的**偏离、安全漏洞以及环境的不可复现性。

![](./Image/基础设施.jpg)

1. 我们使用**Terraform**将整个云环境以代码形式定义。这使我们能够在几分钟内快速启动完全相同的本地`dev`、本地`staging`和本地`prod`环境。
2. 我们还使用**Karpenter**进行智能的、即时的节点扩展，这比标准的 AWS 自动扩展组更快、更便宜。

### 基础设施与网络

所有云架构都始于网络。我们需要一个虚拟私有云 (VPC)，它将我们的敏感数据库与公共互联网隔离，同时允许我们的 API 处理流量。

![](./Image/网络.jpg)

首先，我们开始构建`infra/terraform/main.tf`。我们将配置远程状态后端。这对于团队协作至关重要：如果没有它，两位工程师`terraform apply`同时操作可能会导致基础架构状态损坏。我们将状态文件存储在带有 DynamoDB 锁定的版本化 S3 存储桶中。

```bash
# infra/terraform/main.tf
terraform{
	required_version = ">=1.5.0"
	
	backend "s3" {
		bucket = "rag-platform-terraform-state-prod-001"
		key = "platform/terraform.tfstate"
		region = "us-east-1"
		encrypto = true
		dynamodb_table = "terraform-state-lock"
	}
	
    required_providers {
        aws = { source = "hashicorp/aws", version = "~> 5.0"}
        kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.23"}
    }
}

provider "aws" {
	region = var.aws_region
	default_tags {
		tags = { Project = "Enterprise-RAG", ManagedBy = "Terraform"}
	}
}
```

这里我们主要初始化 AWS 和 Kubernetes 提供商。该`backend "s3"`代码块确保我们基础设施的**敏感数据**存储在云端，经过加密和锁定，而不是存储在开发人员的笔记本电脑上。

我们在 . 中定义了可自定义的参数`infra/terraform/variables.tf`。这提高了可重用性，我们可以简单地通过覆盖这些变量来部署`us-west-2`或更改 IP 范围，而无需触及复杂的逻辑文件。

```bash
# infra/terraform/variables.tf
variable "aws_region" {
	description = "AWS region to deploy resources"
	default = "us-east-1"
}

variable "cluster_name" {
	description = "Name of the EKS Cluster"
	default = "rag-platform-cluter"
}

variable "vpc_cidr" {
	description = "CIDR block for the VPC"
	default = "10.0.0.0/16"
}
```

此文件负责定义我们各个模块所需的输入。它充当我们基础设施模块的 API，定义了适用于大多数情况的默认值，同时也允许进行自定义。

现在，`infra/terraform/vpc.tf`我们需要构建一个三层网络架构。这可以从物理上将负载均衡器（公有网络）与应用服务器（私有网络）和数据存储（数据库）隔离开来，从而大幅减少攻击面。

```bash
# infra/terraform/vpc.tf
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.1.0"

name = "${var.cluster_name}-vpc"

  cidr = var.vpc_cidr

  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  private_subnets = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  database_subnets= ["10.0.201.0/24", "10.0.202.0/24", "10.0.203.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = false # High Availability requires one NAT per AZ
  enable_dns_hostnames = true

  # Tags are required for Kubernetes Load Balancers to auto-discover subnets
  public_subnet_tags = { "kubernetes.io/role/elb" = "1" }
  private_subnet_tags = { "kubernetes.io/role/internal-elb" = "1" }
}
```

在这里，我设置了`enable_nat_gateway = true`允许我们的私有应用程序 Pod 从互联网安全地下载 Docker 镜像和 Python 包，而无需将它们暴露给外部连接。这些标签至关重要：如果没有它们，Kubernetes 中的 AWS 负载均衡控制器将不知道将 ALB 部署在哪里。

### 计算集群（EKS 和 IAM）

我们平台的云通信部分采用的是**Amazon EKS**（弹性 Kubernetes 服务）集群。我们正在部署控制平面来协调所有容器。此设置需要具备安全性和细粒度的权限管理能力。

![](./Image/计算集群.jpg)在此版本中`infra/terraform/eks.tf`，我们启用了 OIDC（OpenID Connect）。该连接允许 Kubernetes 服务账户承担 AWS IAM 角色（IRSA)。这消除了对长期有效的 AWS 访问密钥的需求，从而避免了企业环境中的一项重大安全风险。

```bash
# infra/terraform/eks.tf
module "eks" {
	source = "terraform-aws-modules/eks/aws"
	version = "~> 19.0"
	
}

cluster_name = var.cluster_name
cluter_version = "1.29"
vpc_id = module.vpc.vpc_id
subnet_ids = module.vpc.private_subnets
enable_irsa = true

# 我们只定义最小系统节点组
# 应用扩张在后续Karpenter中处理
eks_managed_node_groups = {
	system = {
		name   			= "system-nodes"
		instance_types 	= ["m6i.large"]
		min_size		= 2
		max_size 		= 5
		desired_size	= 2
	}
}
```

此模块配置 EKS 控制平面和一个小型**“系统”**节点组。我们保持此节点组规模较小，因为它仅负责运行系统关键型 Pod，例如 CoreDNS 和 Karpenter。繁重的计算任务将由稍后动态创建的节点完成。

接下来，我们将编写代码`infra/terraform/iam.tf`来实现摄取管道的**最小权限原则。**

我们创建了一个特定的 IAM 策略，该策略*仅*授予对文档存储桶的访问权限，并将其绑定到摄取作业使用的特定 Kubernetes 服务帐户。

```bash
# infra/terraform/iam.tf
resource "aws_iam_policy" "ingestion_policy" {
  name        = "RAG_Ingestion_S3_Policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Effect   = "Allow"
        Resource = [aws_s3_bucket.documents.arn, "${aws_s3_bucket.documents.arn}/*"]
      }
    ]
  })
}

module "ingestion_irsa_role" {
  source    = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-account-eks"
  role_name = "rag-ingestion-role"
  
  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["default:ray-worker"]
    }
  }
  role_policy_arns = { policy = aws_iam_policy.ingestion_policy.arn }
}
```

> 这样可以确保，即使我们的 Ray Worker 遭到入侵，攻击者也无法访问我们 AWS 账户的其他部分，例如数据库备份或账单信息。

### 托管数据存储

在 Kubernetes 内部运行有状态数据库在运维上既复杂又有风险。

> 对于生产级系统，我们使用 AWS 托管服务来自动处理备份、修补和故障转移。

![](./Image/数据存储管理.jpg)

我们通过部署**Aurora Serverless v2**（Postgres）来实现`infra/terraform/rds.tf`这一点。这对于像与 RAG 聊天这样的可变工作负载至关重要。它可以根据活跃连接数自动扩展或缩减计算容量（ACU），从而确保高峰时段的高性能和低谷时段的低成本。

```bash
# infra/terraform/rds.tf
module "aurora" {
  source  = "terraform-aws-modules/rds-aurora/aws"
  name           = "${var.cluster_name}-postgres"
  engine         = "aurora-postgresql"
  instance_class = "db.serverless" 
  
  instances = {
    one = {}
    two = {} # HA Multi-AZ
  }

serverlessv2_scaling_configuration = {
    min_capacity = 2
    max_capacity = 64
  }
  vpc_id               = module.vpc.vpc_id
  db_subnet_group_name = module.vpc.database_subnet_group_name
  
  # Only allow traffic from within the VPC
  security_group_rules = {
    vpc_ingress = { cidr_blocks = [module.vpc.vpc_cidr_block] }
  }
}
```

这`serverlessv2_scaling_configuration`使得数据库能够从 2 个 ACU（便宜）无缝扩展到 64 个 ACU（强大），而不会断开连接。

为了实现高速缓存，我们需要编写代码并在其中使用**ElastiCache（Redis）**`infra/terraform/redis.tf`。

我们特意选择了`cache.t4g.medium`这些实例。这些实例运行在 AWS Graviton (ARM) 处理器上，与标准 x86 实例相比，性价比最高可提升 40%。

```bash
# infra/terraform/redis.tf
 resource "aws_elasticache_replication_group"  "redis" { 
  replication_group_id = "rag-redis-prod"
   description = "用于 RAG 语义缓存的 Redis"
   node_type = "cache.t4g.medium"
   num_cache_clusters = 2 # 主节点 + 副本节点
  port = 6379 
  
  subnet_group_name = aws_elasticache_subnet_group.redis_subnet.name 
  security_group_ids = [aws_security_group.redis_sg.id] 
  
  at_rest_encryption_enabled = true
   transit_encryption_enabled = true
 }
```

我们需要配置一个包含两个节点的复制组，以确保高可用性。**如果主节点发生故障，AWS 会自动将副本提升为主节点，从而保持缓存在线**。

我们将创建一个空间来存放原始文件。它会`infra/terraform/s3.tf`设置一个**启用传输加速功能**的 S3 存储桶。全球企业的用户遍布各地，传输加速功能利用 Amazon CloudFront 全球分布式边缘节点，通过优化的 AWS 主干网络将上传文件路由到存储桶，从而显著加快大文件传输速度。

```bash
# infra/terraform/s3.tf
resource "aws_s3_bucket" "documents" {
  bucket = "rag-platform-documents-prod"
}

resource "aws_s3_bucket_accelerate_configuration" "docs_accel" {
  bucket = aws_s3_bucket.documents.id
  status = "Enabled"
}

resource "aws_s3_bucket_lifecycle_configuration" "docs_lifecycle" {
  bucket = aws_s3_bucket.documents.id
  rule {
    id     = "archive-old-files"
    status = "Enabled"
    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING" # Auto-optimizes cost
    }
  }
}
```

30 天后，系统`lifecycle_configuration`会自动将旧文件移至更便宜的存储类别（如 Glacier），这是控制长期存储成本的标准 FinOps 做法。

由于我们选择在 Kubernetes 上自托管 Neo4j（为了节省与 AuraDB Enterprise 相比的许可成本），因此我们在 . 中定义了其安全组`infra/terraform/neo4j.tf`。我们需要严格控制对这些 pod 的网络访问。

```bash
# infra/terraform/neo4j.tf
resource "aws_security_group" "neo4j_sg" {
  name        = "neo4j-access-sg"
  vpc_id      = module.vpc.vpc_id

ingress {
    description = "Internal Bolt Protocol"
    from_port   = 7687
    to_port     = 7687
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
}
```

此安全组充当防火墙，仅允许来自我们 VPC 内部的流量访问 Neo4j Bolt 端口。外部互联网流量将被完全阻止。

最后，`infra/terraform/outputs.tf`导出我们刚刚创建的端点。Terraform 模块类似于函数，这些输出是返回值，我们可以轻松查询这些值，以便在下一阶段配置 Kubernetes Secrets。

```bash
# infra/terraform/neo4j.tf
resource "aws_security_group" "neo4j_sg" {
  name        = "neo4j-access-sg"
  vpc_id      = module.vpc.vpc_id

ingress {
    description = "Internal Bolt Protocol"
    from_port   = 7687
    to_port     = 7687
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
}
```

此安全组充当防火墙，仅允许来自我们 VPC 内部的流量访问 Neo4j Bolt 端口。外部互联网流量将被完全阻止。

最后，`infra/terraform/outputs.tf`导出我们刚刚创建的端点。Terraform 模块类似于函数，这些输出是返回值，我们可以轻松查询这些值，以便在下一阶段配置 Kubernetes Secrets。

```bash
# infra/terraform/outputs.tf
output "aurora_db_endpoint" {
  value = module.aurora.cluster_endpoint
}

output "redis_primary_endpoint" {
  value = aws_elasticache_replication_group.redis.primary_endpoint_address
}
output "s3_bucket_name" {
  value = aws_s3_bucket.documents.id
}
```

### 使用 Karpenter 实现自动扩缩容

标准集群自动扩缩容机制反应迟缓，需要等待 Pod 调度失败后才会添加节点。而**Karpenter**则更加主动智能，它会分析待调度 Pod 的具体资源需求（例如 GPU 类型、内存），并在几秒钟内启动最合适的 EC2 实例。

![](./Karpenter自动缩放.jpg)

我们在配置文件中定义了 CPU 资源分配器`infra/karpenter/provisioner-cpu.yaml`。对于我们的无状态 API 和 Web 服务器，我们可以容忍中断，因此我们使用**竞价型实例**。与按需实例相比，这可以为我们节省大约 70% 的计算成本。

```yaml
# infra/karpenter/provisioner-cpu.yaml
apiVersion: karpenter.sh/v1beta1
kind: Provisioner
metadata:
  name: cpu-provisioner
spec:
  requirements:
    - key: "karpenter.k8s.aws/instance-family"
      operator: In
      values: ["m6i", "c6i"]
    - key: "karpenter.sh/capacity-type"
      operator: In
      values: ["spot"] # Cost optimization
  limits:
    resources:
      cpu: 1000
  consolidation:
    enabled: true # Repacks pods to kill empty nodes
```

该`consolidation: true`设置指示 Karpenter 主动移动 pod，以更紧密地排列节点并终止未充分利用的实例，从而确保我们不会为空置空间付费。

对于我们的人工智能工作负载，我们进行了定义`infra/karpenter/provisioner-gpu.yaml`。成本控制的关键就在这里。我们设置了生存时间 (TTL) 参数。

```yaml
# infra/karpenter/provisioner-gpu.yaml
apiVersion: karpenter.sh/v1beta1
kind: Provisioner
metadata:
  name: gpu-provisioner
spec:
  requirements:
    - key: "karpenter.k8s.aws/instance-category"
      operator: In
      values: ["g"] # g5 instances (Nvidia A10G)
    - key: "karpenter.sh/capacity-type"
      operator: In
      values: ["on-demand", "spot"]
  
  # Kill the node if it's empty for 30 seconds
  ttlSecondsAfterEmpty: 30
```

## 部署

**在生产级 RAG 平台**中，部署不仅仅是`kubectl apply -f file.yaml`。

> 我们需要管理配置偏差，安全地处理密钥，并确保我们的数据库能够抵御故障。

![](./Image/部署层.jpg)

我们使用**Helm**来打包应用程序。Helm 允许我们将部署定义为模板，这样就可以轻松地**针对不同的环境更改值（例如副本数或存储大小），而无需重写数千行 YAML 代码**。

### 集群引导和密钥

在部署自定义代码之前，我们需要安装集群。这包括入口控制器（流量控制）、外部密钥操作员（安全卫士）和 KubeRay 操作员（AI 管理器）。

![](./Image/集群引导.jpg)

我们使用 Helm 脚本来实现自动化`scripts/bootstrap_cluster.sh`。该脚本会为这些关键系统组件安装 Helm Chart。

```sh
# scripts/bootstrap_cluster.sh 
#!/bin/bash 

# 1. 安装 KubeRay Operator（管理 Ray 集群）
helm repo add kuberay https://ray-project.github.io/kuberay-helm/ 
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0 

# 2. 安装外部密钥（同步 AWS Secrets Manager 到 Kubernetes）
helm repo add external-secrets https://charts.external-secrets.io 
helm install external-secrets external-secrets/external-secrets 

# 3. 安装 Nginx Ingress（负载均衡控制器）
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx 
helm install ingress-nginx ingress-nginx/ingress-nginx
```

在尝试部署应用程序之前，我们要确保集群中所有必要的控制器都在运行。这相当于我们平台的**“预检”**。

接下来，为了处理密钥。在生产环境中，我们绝不会提交`.env`文件。我们使用 AWS Secrets Manager 来存储数据库密码和 API 密钥。我们使用**External Secrets Operator**来获取这些密钥，并将它们安全地注入到 Kubernetes Pod 中。

我们在以下地方定义了这一点`deploy/secrets/external-secrets.yaml`。

```yaml
# deploy/secrets/external-secrets.yaml 
apiVersion:  external-secrets.io/v1beta1 
kind:  ExternalSecret 
metadata: 
  name:  app-secrets 
spec: 
  refreshInterval:  1h              # 每小时检查一次轮换
  secretStoreRef: 
    name:  aws-secrets-manager      # 连接到 AWS 
    kind:  ClusterSecretStore 
  target: 
    name:  app-env-secret           # 创建一个名为“app-env-secret”的 Kubernetes Secret 
  data: 
  -  secretKey:  NEO4J_PASSWORD 
    remoteRef: 
      key:  prod/rag/db_creds       # AWS Secret 名称
      property:  neo4j_password
```

此配置告诉 Kubernetes：“前往 AWS Secrets Manager，查找`prod/rag/db_creds`并获取`neo4j_password`，并为我的 Pod 创建一个本地 Kubernetes Secret”。**这样可以将我们的敏感数据完全排除在 Git 历史记录之外。**

### 数据库和入口部署

现在我们部署有状态工作负载。虽然我们使用 RDS 来托管 Postgres，但出于性能和成本方面的考虑，我们选择自行托管 Qdrant 和 Neo4j。

![](./Image/数据库和入口.jpg)

我们使用标准的 Helm Chart，但会覆盖默认值以使其适用于生产环境。对于 Qdrant，我们将进行修改，`deploy/helm/qdrant/values.yaml`以确保数据持久化和高可用性。

```yaml
# deploy/helm/qdrant/values.yaml 

# 集群模式：运行 3 个副本以实现容错
replicaCount:  3 

config: 
  storage: 
    # 使用 Memmap 进行存储（将向量存储在磁盘上，但映射到 RAM）
    # 允许处理大于 RAM 的数据集。
    on_disk_payload:  true 
  
  service: 
    enable_tls:  false  # TLS 由 Ingress/Mesh 处理

resources: 
  requests: 
    cpu:  "2" 
    memory:  "4Gi" 
  limits: 
    cpu:  "4" 
    memory:  "8Gi" 

# 持久化：使用快速 SSD（io1 或 gp3）
persistence: 
  size:  50Gi 
  storageClassName:  gp3
```

使用`replicaCount: 3`此功能是为了在某个节点发生故障时，我们的 Vector DB 仍能保持在线。`on_disk_payload`这是 RAG 的一项关键优化，它可以防止 Qdrant 在索引数百万个文档（超出 RAM 限制）时崩溃。

对于 Neo4j，我们需要配置类似的结构`deploy/helm/neo4j/values.yaml`。

```yaml
# deploy/helm/neo4j/values.yaml 

neo4j: 
  name:  "neo4j-cluster" 
  edition:  "community" 
  
  core: 
    # 社区版不支持集群。
    # 必须为 1。如果需要高可用性，请升级到企业版。
    numberOfServers:  1  
    
    resources: 
      requests: 
        cpu:  "2" 
        memory:  "8Gi" 
    
  volumes: 
    data: 
      mode:  "default"  # 动态绑定
      storageClassName:  "gp3" 
      size:  "100Gi"
```

我们分配了内存（`8Gi`），因为图遍历会消耗大量内存。我们还附加了一个**100GB 的持久卷，以确保我们的知识图谱**在 Pod 重启后仍然有效。

为了将我们的 API 暴露给互联网，我们在 . 中配置了一个 Ingress 资源`deploy/ingress/nginx.yaml`。它充当我们的 API 网关，将来自 AWS 负载均衡器的 HTTP 流量路由到我们的内部服务。

```yaml
# deploy/ingress/nginx.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  annotations:
    # Use AWS Load Balancer Controller
    kubernetes.io/ingress.class: nginx

    # Increase body size limit just in case (though we use S3 direct upload)
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"

    # Timeout settings for streaming responses
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
spec:
  rules:
  - host: api.your-rag-platform.com
    http:
      paths:

      # Route /chat to the API
      - path: /chat
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80

      # Route /upload to the API (for presigned generation)
      - path: /upload
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80
```

这里的注解很重要。`proxy-read-timeout`**设置为 1 小时是因为生成包含 70 字节模型的长答案有时需要** **时间**，我们不希望 Nginx 过早断开连接。

### Ray AI 集群部署

最后，我们部署了运营的核心——Ray集群。这很复杂，因为它包含一个**头节点（管理节点）**和**多个工作节点（执行节点）**，这些节点需要独立扩展。

![](./Image/Ray集群部署.jpg)

首先，我们在 . 中定义扩展逻辑`deploy/ray/autoscaling.yaml`。此配置映射告诉 Ray Autoscaler 当负载增加时要添加多少个工作进程。

```yaml
# deploy/ray/autoscaling.yaml 
autoscaling: 
  enabled:  true 
  upscaling_speed:  1.0  # 激进的扩容
  idle_timeout_minutes:  5  # 如果工作节点空闲 5 分钟则终止
  
  worker_nodes: 
    gpu_worker_group: 
      min_workers:  0 
      max_workers:  20 
      resources: { "CPU":  4 , "memory":  32Gi , "GPU":  1 }
```

此配置适用于**归零模式**……

> 如果 5 分钟内无人使用 AI 模型，Ray 会终止工作 Pod。Karpenter 随后会检测空节点并终止 EC2 实例，从而节省大量资金。

我们在 . 中定义集群结构`deploy/ray/ray-cluster.yaml`。这是一个由 KubeRay 操作员管理的自定义资源定义 (CRD)。

```yaml
# deploy/ray/ray-cluster.yaml 
apiVersion:  ray.io/v1 

kind:  RayCluster 

metadata: 
  name:  rag-ray-cluster 

spec: 
  rayVersion:  '2.9.0' 

  headGroupSpec: 
    serviceType:  ClusterIP 

    template: 
      spec: 
        containers: 
        -  name:  ray-head 
          image:  rayproject/ray:2.9.0-py310-gpu 
          resources: { requests: { cpu:  "2" , memory:  "8Gi" } } 

workerGroupSpecs: 

  # 用于推理的 GPU 工作进程
  -  groupName:  gpu-workers 
    replicas:  0 
    minReplicas:  0 
    maxReplicas:  20 

    template: 
      spec: 
        containers: 
        -  name:  ray-worker 
          image:  rayproject/ray:2.9.0-py310-gpu 

          resources: 
            limits: { nvidia.com/gpu:  1 } 
            requests: { nvidia.com/gpu:  1 } 

        # 污点容忍度确保这些 pod仅允许在 GPU 节点上进行访问
        ：
        -  key:  "nvidia.com/gpu" 
          operator:  "exist"
```

该`tolerations`部分确保我们昂贵的 AI 工作负载仅在具有 GPU 的节点上调度，与我们之前在 Karpenter 配置中设置的 Taints 相匹配。

现在我们使用 . 将实际模型部署到此集群上`RayService`。这种抽象方式可以处理我们模型的高可用性和零停机时间升级。

对于嵌入模型，我们需要创建`deploy/ray/ray-serve-embed.yaml`以便在集群上提供嵌入模型服务。

```yaml
# deploy/ray/ray-serve-embed.yaml 
apiVersion:  ray.io/v1 

kind:  RayService 

metadata: 
  name:  embed-service 

spec: 
  serveConfigV2:  | 
    applications: 
      - name: bge-m3 
        import_path: services.api.app.models.embedding_engine:app 
        deployments: 
          - name: EmbedDeployment 
            autoscaling_config: 
              min_replicas: 1 
              max_replicas: 5 
            ray_actor_options: 
              num_gpus: 0.5 # 共享 GPU（每张卡 2 个模型）
```

而我们需要创建的 LLM`deploy/ray/ray-serve-llm.yaml`将使用 VLLM 为 Meta Llama 3 模型提供服务，以实现高效推理。

```yaml
# deploy/ray/ray-serve-llm.yaml 
apiVersion:  ray.io/v1 

kind:  RayService 

metadata: 
  name:  llm-service 

spec: 
  serveConfigV2:  | 
    applications: 
      - name: llama3 
        import_path: services.api.app.models.vllm_engine:app 
        runtime_env: 
          pip: ["vllm==0.3.0"] 
          env_vars: 
            MODEL_ID: "meta-llama/Meta-Llama-3-70B-Instruct" 
        deployments: 
          - name: VLLMDeployment 
            autoscaling_config: 
              min_replicas: 1 
              max_replicas: 10 
            ray_actor_options: 
              num_gpus: 1
```

通过将这些定义为`RayService`对象，我们便能掌控运维能力。如果我们更新了这些对象`MODEL_ID`，运维人员将使用新模型启动新的 Pod，等待它们恢复正常运行，然后无缝切换流量，确保用户零停机时间。

最后，当我们完成测试或需要在开发环境中节省成本时，我们也必须`scripts/cleanup.sh`通过拆除所有内容来节省成本。

```sh
# scripts/cleanup.sh 
# !/bin/bash 

echo  "⚠️警告：此操作将销毁所有云资源⚠️" 
echo  "包括：EKS 集群、数据库（RDS/Neo4j/Redis）、S3 存储桶、负载均衡器。" 
echo  "开发/测试环境的成本节约措施。" 
echo  "" 
read -p "确定吗？输入“DESTROY”：" confirm 

if [ " $confirm " != "DESTROY" ]; then 
    echo  "已中止。" 
    exit 1 
fi 

echo  "🔹 1. 删除 Kubernetes 资源 (Helm)...."
 helm uninstall api || true
 helm uninstall qdrant || true
 helm uninstall ray-cluster || true
 kubectl delete -f deploy/ray/ || true 

echo  "🔹 2. 等待负载均衡器清理..." 
sleep 20 

echo  "🔹 3. 运行 Terraform Destroy..." 
cd infra/terraform 
terraform destroy -auto-approve 

echo  "✅ 所有资源已销毁。"
```

这个脚本是一个安全阀。它确保我们不会意外地让一个包含 20 个 GPU 节点的集群在周末运行，从而产生巨额账单。

**我们结合使用了 Terraform、Helm 和 Ray，实现了对基础设施的全面控制，**同时自动化了分布式系统的复杂性。现在，我们将探讨如何在生产环境中监控和评估这一架构。

## 评估与运营

在**企业级红黄绿灯平台**中，评估至关重要，而且大多数评估工作都在开发阶段进行。我们需要制定一套策略，涵盖**可观测性**（指标/追踪）、**评估**（准确性检查）和**运维**（负载测试/维护）。

> 在生产系统中，我们看到的评价通常是用户对答案的点赞/踩。

虽然这种反馈很有用，但往往过于零散和滞后，无法在问题影响众多用户之前发现问题。

### 可观测性和可追踪性

**无法衡量的东西就无法改进**。在涉及 Ray、Kubernetes 和多个数据库的分布式系统中，如果没有分布式追踪，就无法找到瓶颈。

![](./Image/追踪逻辑.jpg)

可观测性也是一种评估方式。因为它能帮助我们了解系统性能和用户行为，所以当令牌使用量超出预算或延迟峰值出现时，可能会对我们的预算和用户满意度造成影响。

我们在 . 中定义了自定义指标`libs/observability/metrics.py`。我们将使用 Prometheus 计数器来跟踪代币使用情况（成本），并使用直方图来跟踪延迟（性能）。

```python
# lib/observability/metrices/py
from prometheus_client import Counter, Histogram

# 1.计数器：只递增（例如，请求总数）
REQUEST_COUNT = Counter(
	"rag_api_requests_total",
    "请求总数",
    ["method", "endpoint", "status"]
)

# 2.直方图跟踪分布（例如，延迟、Token 计数）
 REQUEST_LATENCY = Histogram( 
    "rag_api_latency_seconds" , 
    "请求延迟" , 
    [ "endpoint" ] 
) 

TOKEN_USAGE = Counter( 
    "rag_llm_tokens_total" , 
    "LLM Token 消耗总数" , 
    [ "model" , "type" ] # type=prompt vs completion
 ) 

def track_request (method: str , endpoint: str , status: int): 
    """用于递增请求计数器的辅助函数"""
     REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
```

这使我们能够构建 Grafana 仪表板，准确地显示 Llama-3 消耗的令牌数量，并确定特定端点是否变慢。

为了进行追踪，我们将创建`libs/observability/tracing.py`。我们使用**OpenTelemetry**，它允许我们可视化从 API 到向量搜索再到 LLM 生成的完整请求生命周期。

```python
# libs/observability/tracing.py 
from opentelemetry import trace 
from opentelemetry.sdk.trace import TracerProvider 
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter 

def configure_tracing (service_name: str): 
    """
    设置 OpenTelemetry 追踪。
    """ 
    # 1. 创建追踪器提供程序
    provider = TracerProvider() 
    
    # 2. 配置导出器
    # 开发环境：打印到控制台。
    # 生产环境：发送到 OTLP 收集器（例如 Jaeger/Grafana）
     processor = BatchSpanProcessor(ConsoleSpanExporter()) 
    provider.add_span_processor(processor) 
    
    # 3. 设置全局追踪器
    trace.set_tracer_provider(provider) 
    
    return trace.get_tracer(service_name)
```

当用户报告**“响应缓慢”**时，我们可以查看跟踪 ID，例如，**可以看到 90% 的时间都花在了等待 Neo4j 图查询上**，从而准确地找出需要优化的地方。

### 持续评估流程

我们如何确定 RAG 流程的准确性？我们不可能手动检查每个答案。我们使用**“LLM-as-a-Judge”**来实现自动化质量控制。

按回车键或点击查看完整尺寸的图片

![](./Image/持续评估.jpg)

首先，我们需要一个**“黄金数据集”，**即一组问题及其对应的真实答案。我们根据项目初期参考的`eval/datasets/golden.json`Kubernetes 文档（例如`pods_autoscale.html`）创建该数据集。

```json
// eval/datasets/golden.json 
[ 
  { 
    "question" :  "解释水平 Pod 自动扩缩器 (HPA) 和集群自动扩缩器之间的区别。" , 
    "ground_truth" :  "HPA 根据 CPU/指标扩展 Pod 数量。集群自动扩缩器在 Pod 无法调度时扩展节点数量。" , 
    "contexts" :  [ 
      "水平 Pod 自动扩缩器是 Kubernetes 的一个组件，用于调整副本数量……" , 
      "集群自动扩缩器是一个自动调整 Kubernetes 集群大小的工具……" 
    ] 
  }
   ... 
]
```

此文件用作我们的基准测试。每次更新提示或检索逻辑时，我们都会使用此数据集运行系统。

1. 通常，黄金数据集是通过更大型的线性模型或人工标注合成生成的。为了简化操作，我们使用 Gemini（谷歌最新的模型）构建了该数据集。
2. 在处理私有数据时，我们会创建一个子流水线来生成黄金数据集。具体做法是，调用强大的逻辑生命周期管理（LLM）工具读取私有文档并生成问答对。这样，我们就能避免将敏感数据暴露给第三方服务。

为了对结果进行评分，我们将编写判断逻辑`eval/judges/llm_judge.py`。该脚本将使用强大的模型（例如 GPT-4 或 Llama-3-70B）将我们系统的实际答案与真实答案进行比较。

```python
# eval/judges/llm_judge.py 
from pydantic import BaseModel 
from services.api.app.clients.ray_llm import llm_client 

class Grade(BaseModel): 
    score: int
    reasoning: str

JUDGE_PROMPT = """
您是一位公正的评委，正在评估一个 RAG 系统。
您将收到一个问题、一个标准答案和系统的答案。
请根据 1 到 5 的等级对系统的答案进行评分：
1：完全错误或完全错误。3 
：部分正确，但缺少关键细节。5 
：完美、全面，并且与标准答案的逻辑完全一致。
仅输出 JSON：{{"score": int, "reasoning": "string"}}
问题：{question}
标准答案：{ground_truth}
系统答案：{system_answer} 
""" 

async def grade_answer(question: str, ground_truth: str , system_answer: str ) -> Grade: 
    """
    调用 LLM 对单个 QA 进行评分
    """
    import json
    
    try:
        response_text = await llm_client.chat_completion(
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question,
                ground_truth=ground_truth,
                system_answer=system_answer
            )}],
            temperature=0.0
        )
        
        # Parse JSON output
        data = json.loads(response_text)
        return Grade(**data)
        
    except Exception as e:
        return Grade(score=0, reasoning=f"Judge Error: {e}")
```

**我们使用Ragas**来协调整个过程，Ragas 是一个用于评估 RAG 系统的常用框架。主要的评估脚本将位于`eval/ragas/run.py`.

**Ragas 是一个行业标准框架，它计算忠实度（人工智能是否产生了幻觉？）和上下文召回率（我们是否找到了正确的文档？）等指标**。

让我们来实施这个……

```python
# eval/ragas/run.py 
from ragas import evaluate 
from ragas.metrics import faithfulness, answer_relevancy 
from datasets import Dataset 

def  run_evaluation(questions, answers, contexts, ground_truths ): 
    """
    运行 Ragas 评估套件。
    """

     data = { 
        "question" : questions, 
        "answer" : answers, 
        "contexts" : contexts, 
        "ground_truth" : ground_truths 
    } 

    dataset = Dataset.from_dict(data) 

    results = evaluate( 
        dataset=dataset, 
        metrics=[faithfulness, answer_relevancy], 
    ) 
    
    df = results.to_pandas() 

    df.to_csv( "eval/reports/evaluation_results.csv" , index= False ) 

    print (results)
```

该脚本将生成一份报告，`eval/reports/evaluation_results.csv`其中提供了我们管道准确性的可视化摘要。

> 我们在 CI/CD 流水线中使用此功能：如果“忠实度”得分低于 0.8，我们将阻止部署。

### 卓越运营与维护

最后，我们需要一些工具来维护生产环境中的系统。代码变更、负载增加和数据漂移都是不可避免的。

![](./Image/维护部署.jpg)

为了确保我们的 Karpenter 自动扩缩容设置正常工作，我们使用 Locust 模拟高**流量**`scripts/load_test.py`。

```python
# scripts/load_test.py
from locust import HttpUser, task, between
import os

class RAGUser(HttpUser):
    # Wait 1 to 5 seconds between tasks
    wait_time = between(1, 5)
    
    @task
    def chat_stream_task(self):
        """
        Simulates a user sending a chat message.
        """
        headers = {
            "Authorization": f"Bearer {os.getenv('AUTH_TOKEN')}" # Load token from env
        }
        
        # Example query
        payload = {
            "message": "What is the warranty policy for the new X1 processor?",
            "session_id": "loadtest-user-123"
        }
        
        # Use streaming=True to handle the SSE response
        with self.client.post(
            "/api/v1/chat/stream", 
            json=payload, 
            headers=headers, 
            stream=True,
            name="/chat/stream" # Group results under this name
        ) as response:

            if response.status_code != 200:
                response.failure("Failed request")
            else:
                # Iterate through the stream to simulate a real client
                for line in response.iter_lines():
                    if line:
                        pass # In a real test, you might validate the JSON
                response.success()
```

运行此脚本有助于我们调整`ray/autoscaling.yaml`配置，确保**当 100 个用户同时加入时，新的 GPU 节点能够足够快地启动**。

随着应用程序的演进，数据库架构也会随之改变。我们需要创建`scripts/migrate_db.py`（封装）机制来安全地处理架构迁移，避免停机。

```python
# scripts/migrate_db.py 
import subprocess 
import os 
from services.api.app.config import settings 

def  run_migrations(): 
    """
    将 Alembic 迁移应用到数据库。
    """ 
    print("正在运行数据库迁移..." ) 
    
    # Alembic 需要数据库 URL。我们通过环境变量传递它。
    env = os.environ.copy() 
    env[ "DATABASE_URL" ] = settings.DATABASE_URL 
    
    try : 
        # 'alembic upgrade head' 命令应用所有待处理的迁移。
         subprocess.run( 
            [ "alembic" , "upgrade" , "head" ], 
            check= True , 
            env=env, 
            cwd=os.path.dirname(os.path.abspath(__file__)) # 从此脚本目录运行
        ) 
        print ( "✅ 迁移已成功应用。" ) 
    except subprocess.CalledProcessError as e: 
        print ( f"❌ 迁移失败：{e} " ) 
        exit( 1 ) 

if __name__ == "__main__" : 
    run_migrations()
```

最后，当我们部署新版本时，缓存是空的，这会导致前几个用户响应速度较慢。我们通常会`scripts/warmup_cache.py`在将流量切换到新 Pod 之前，预先在 Redis 和 Qdrant 中填充常用查询。

```python
# scripts/warmup_cache.py 
import asyncio 
from services.api.app.cache.semantic import semantic_cache 

# 常见问题解答列表
FAQ = [ 
    ( "你们的营业时间是什么？" , "我们的营业时间是上午 9 点到下午 5 点。" ), 
    ("退货政策是什么？" , "30 天内可退货。" )
] 

async  def  warmup (): 
    print ( "🔥 正在预热语义缓存..." ) 
    for question, answer in FAQ: 
        await semantic_cache.set_cached_response(question, answer) 
    print ( "✅ 缓存预热完成。" ) 

if __name__ == "__main__" : 
    asyncio.run(warmup())
```

这些常见问题解答源自我们的黄金数据集和真实用户查询。通过将它们预加载到缓存中，我们确保最常见问题能够立即得到解答，从而在部署后立即提升用户体验。

## 端到端执行

我们已经编写了代码，定义了基础设施，并搭建了评估流程。现在，我们需要实际**部署**所有内容，以便运行我们的 RAG 平台。

> 虽然 Terraform 可以自动完成大部分繁重的工作，但我们仍需要在 AWS 控制台中执行一些初始的手动步骤来引导我们的“远程状态”后端。这是一次性设置。

我们需要创建一个 S3 存储桶来存储 Terraform 状态文件，并创建一个 DynamoDB 表来处理状态锁定（以防止并发修改）。

1. 您可以根据需要为存储桶分配多个规则，例如版本控制、加密和生命周期策略，以优化成本。请务必选择区域`us-east-1`（弗吉尼亚北部），因为 EKS 在该区域拥有最丰富的 GPU 实例类型。
2. 类似地，我们需要创建一个`DynamoDB`用于状态锁定的表，因为这对于防止多人同时应用更改至关重要。

我们需要将分区键设置为`LockID`类型为 `<type>` 的键`String`。其余设置可以保留默认值。您可以给表起一个类似 `<name>` 的名称`terraform-state-lock`，并确保它与 S3 存储桶位于同一区域，以获得最佳性能。

### 创建 EKS 集群

现在我们的 Terraform 后端已经准备就绪。我们需要切换到终端。我们将创建 VPC、EKS 集群控制平面和数据库。

> **注意：**此步骤会创建 EKS*控制平面*和一个用于运行系统工具的小型“系统节点组”（包含 2 个小型 CPU）。**它不会立即创建 GPU 节点。**为了节省成本，我们仅在软件实际需要时才创建 GPU 节点。

运行基础设施构建命令，它将负责初始化和应用 Terraform 代码。

```bash
# From the root of the project
make infra
```

该命令`terraform init`随后运行`terraform apply`

```bash
#### OUTPUT ######
Initializing the backend...
Successfully configured the backend "s3"!

Terraform will perform the following actions:
  # module.vpc.aws_vpc.this will be created
  + resource "aws_vpc" "this" {
      + cidr_block = "10.0.0.0/16"
      + tags       = { "Name" = "rag-platform-cluster-vpc" }
      ...
    }
  # module.eks.aws_eks_cluster.this will be created
  + resource "aws_eks_cluster" "this" {
      + name     = "rag-platform-cluster"
      ...
    }
Plan: 48 to add, 0 to change, 0 to destroy.

Do you want to perform these actions?
  Enter a value: yes

aws_s3_bucket.documents: Creating...
module.vpc.aws_nat_gateway.this[0]: Creating...
module.eks.aws_eks_cluster.this: Creating...
module.aurora.aws_rds_cluster.this: Creating...

Apply complete! Resources: 48 added, 0 changed, 0 destroyed.
```

接下来为我们分配资源

```bash
Outputs:
aurora_db_endpoint = "rag-platform-cluster-postgres.cluster-c8s7d6f5.us-east-1.rds.amazonaws.com"
redis_primary_endpoint = "rag-redis-prod.ng.0001.use1.cache.amazonaws.com"
s3_bucket_name = "rag-platform-documents-prod"
```

你可以发现输出中少量资源被创建了，包括VPC、EKS集群、RDS Postgres以及ElastiCache Redis。

这个过程通常需要15-20分钟。一旦完成，我们获得了一个准备就绪的K8s集群

现在我们配置 `kubectl` 与我们的新集群交互并且安装系统控制器。

```bash
# Update kubeconfig to point to the new cluster
aws eks update-kubeconfig --region us-east-1 --name rag-platform-cluster

# Run bootstrap script
./scripts/bootstrap_cluster.sh

# Run bootstrap script
./scripts/bootstrap_cluster.sh
```

我们将得到

```bash
#### OUTPUT ######
"kuberay" has been added to your repositories
Release "kuberay-operator" does not exist. Installing it now.
NAME: kuberay-operator
STATUS: deployed

Release "external-secrets" does not exist. Installing it now.
NAME: external-secrets
STATUS: deployed

Release "ingress-nginx" does not exist. Installing it now.
NAME: ingress-nginx
STATUS: deployed
```

您可以看到 KubeRay Operator、External Secrets Operator 和 Nginx Ingress Controller 已成功安装。

目前，我们的集群已经安装完毕，但**GPU实例数为零**。

接下来，我们需要**Karpenter**来管理 GPU 节点的动态配置。我们应用之前定义的 Karpenter 配置。

Ray 配置明确要求`nvidia.com/gpu: 1`。

1. Kubernetes 会尝试调度该 Pod。
2. 它将失败（待定），因为我们没有 GPU 节点。
3. **Karpenter**将检测到此待处理的 pod。
4. Karpenter 将自动调用 AWS EC2 API 来购买`g5.xlarge`实例。

```bash
# 应用 Ray 集群和 Ray 服务定义
kubectl apply -f deploy/ray/
```

仔细观察

```bash
kubectl get pods -w 

# 初始输出：
名称 就绪 状态 重启次数 运行时间
rag-ray-cluster-head-8k2j1 0/1 待处理 0 5s <-- 等待节点ray 
-worker-gpu-group-0 0/1 待处理 0 5s <-- 等待GPU节点
```

> *此时此刻，如果您查看 AWS EC2 控制台，您会看到一个名为“*`*karpenter-rag-platform-cluster-...*`*正在初始化”的新实例。我们没有点击“创建实例”，是软件自动创建的。*

我等了大约**45 秒***……*然后 pod 的状态就变成了 Running：

```bash
rag - ray - cluster - head - 8 k2j1           1/1运行中0 45秒ray - worker - gpu - group - 0 0/1     容器创建中0 50秒ray - worker - gpu - group - 0 1/1运行中0 95秒                  
```

Karpenter 已成功及时创建/更新了所需的基础设施。`vLLM`工作进程内的引擎现在正在将 Llama-3-70B 权重（约 40GB 量化后）加载到新创建实例的显存中。

### 推理和延迟测试

如果你还记得第一部分的内容，我们准备了 1000 份文档（很多无关信息）。我们将把它们上传到 S3 并触发 Ray 作业。如果当前集群过于繁忙，该作业会启动*额外的*CPU 节点，这一切都得益于我们的 Karpenter 架构。

```bash
# 上传数据（使用 Terraform 输出中的存储桶名称）
 python scripts/bulk_upload_s3.py ./data rag-platform-documents-prod 

# 触发数据摄取作业
python -m pipelines.jobs.s3_event_handler
```

这将触发文档上传、嵌入和分块创建知识图谱的摄取任务。

```bash
### 输出 ###
作业提交 ID：ray_job_ingest_1000_docs 


（Ray 数据）2025-12-28 10:00:01 -- 执行 DAG InputDataBuffer[Input] -> TaskPoolMapOperator[ParsePDF] -> TaskPoolMapOperator[Embed] 

（Ray 数据）2025-12-28 10:00:05 -- 阶段 1 (ReadS3)：找到 1000/1000 个文件。

（Ray 数据）2025-12-28 10:00:45 -- 阶段 2 (Parse)：100%|██████████| 1000/1000 [00:40<00:00, 25.00 个文档/秒] 

(Ray 数据) 2025-12-28 10:01:10 -- 阶段 3 (嵌入): 100%|██████████| 5000/5000 个数据块 [00:25<00:00, 200.00 个数据块/秒] 

(Ray 数据) 2025-12-28 10:01:15 -- 阶段 4 (WriteQdrant): 已插入 5000 个向量。

(Ray 数据) 2025-12-28 10:01:20 -- 阶段 5 (WriteNeo4j): 已合并 1200 个节点和 3500 个关系。... 

[已截断]

作​​业“ray_job_ingest_1000_docs”成功。
```

请注意吞吐量。由于 Ray 将 PDF 解析并行化到多个 CPU 核心，并将嵌入批量处理到 GPU，我们在不到 2 分钟的时间内处理了 1000 份企业文档。

现在知识图谱和向量数据库已经填充完毕，让我们提出一个需要智能体“规划”的复杂问题。

```bash
#获取负载均衡器 URL kubectl 
get ingress 
# 输出：api.your-rag-platform.com 

# Curl 请求
curl -X POST  https://api.your-rag-platform.com/api/v1/chat/stream \   -H "Content-Type: application/json " \   -d '{     " message " : "根据文档比较 Kubernetes中HPA与VPA的成本。",     " session_id ": " demo-session- 1 "   }'
```

通过查看控制台输出，我们可以看到整个 rag 处理流程的工作原理……

```bash
#### RAG 流水线输出 #####

 { "type" : "status" , "node" : "planner" , "info" : "Intent: Retrieve" } 
{ "type" : "status" , "node" : "retriever" , "info" : "Searching Qdrant & Neo4j..." } 
... 
{ "type" : "status" , "node" : "retriever" , "info" : "Found 5 related chunks." } 
{ "type" : "answer" , "content" : "Based on the provided documentation:" } 
... 
{ "type" : "answer" , "content" : "The **Horizo​​ntal Pod Autoscaler (HPA)** scales the number of Pod in a replication controller, deployment, replica set, or stateful set based on observations CPU utilization or other selected metrics." } 
{ "type" : "answer" , "content" : " HPA 本身不产生直接成本，但它通过扩展 Pod 来增加云成本，这可能会触发集群自动扩缩器添加新节点。" } 
{ "type" : "answer" , "content" : " 相比之下，**垂直 Pod 自动扩缩器 (VPA)** 会根据使用情况自动调整 Pod 的 CPU 和内存请求/限制。" } 
... 
{ "type" : "answer" , "content" : " VPA 优化现有资源，并通过合理调整容器大小来降低成本，确保您不会过度配置资源。" } 
{ "type" : "answer" , "content" : " **主要成本差异**：HPA 通过增加容量（节点/Pod）来增加成本，而 VPA 通过调整资源效率来优化成本。" } 
{ "type" : "answer" , "content" : " [来源：pods_autoscale.html]" } 
... 
{ "type" : "status" , "node" : "responder" , "info" : "Streaming Complete" }
```

您可以看到，代理首先识别出意图为“检索”。然后，它在 Qdrant 和 Neo4j 中执行混合搜索，找到相关文档，并综合生成带有引用的全面答案。

**虽然答案质量很重要，但在企业环境中，延迟直接关系到用户体验。如果用户需要盯着一个旋转的加载图标看10秒钟，他们就会离开。**

我们来看一下刚刚运行的 HPA 与 VPA 查询的内部延迟细分情况。由于我们启用了结构化日志记录，因此可以准确地看到时间都消耗在了哪里。

```bash
### 输出 ###
 { 
  "request_id":  "req-a1b2-c3d4" , 
  "total_latency_ms":  2850 , 
  "breakdown": { 
    "planner_node_ms":  420 , 
    "retrieval_node_ms":  780 , 
    "generation_ttft_ms":  1650 
   } 
}
```

我们通常会在 2-4 秒内收到响应。以下是这种架构能够实现如此高速度的原因：

1. **并行检索：**耗时`retrieval_node_ms`780 毫秒。这包括*向量*搜索（Qdrant）和图查询（Neo4j）。由于我们`asyncio.gather`在 API 代码中使用了并行处理，这些查询在不同的 CPU 线程上同时运行。如果按顺序运行，仅此步骤就需要 1.5 秒。
2. **vLLM 加速：**首次词元到达时间`generation_ttft_ms`(Time To First Token) 为 1.6 秒。这是 70B 模型读取上下文并开始说话所需的时间。标准的 HuggingFace 流水线在此处需要 4-5 秒。vLLM 的**PagedAttention**优化了内存访问，从而将这一时间缩短。

但是，随着我们添加更多代理，例如“代码审查员”或“网络搜索员”，这种延迟自然会增加。

> 如果我们简单地将它们串联起来，响应时间可能会超过 10 秒。

为了解决这个问题，我们将进一步**分散网络**。不再让一个代理循环在单个 CPU 上运行，而是生成多个 Ray Actor 并行运行不同的代理，即使复杂度增加，也能保持约 3 秒的运行时间。

### Redis 和 Grafana 仪表盘分析

最后，我们可以验证系统在高负载下的表现。我们运行 Locust 负载测试，模拟 500 个并发用户带来的流量高峰。

```bash
locust -f scripts/load_test.py --headless -u 500 -r 10 --host https://api.your-rag-platform.com
```

![](./Image/Ray表盘.png)

从 Ray Dashboard 的峰值可以看出，我们导入的 1000 个文档已经使 GPU 达到满负荷运行。自动扩缩器检测到负载增加，并请求更多 GPU 节点（最多 3 个）。

本次测试中，我们将最大 GPU 节点数设置为 3 个（总共 16 个 GPU），CPU 核心数设置为 192 个。这足以处理 500 个并发用户，延迟较低，但显然可以根据您的预算和预期负载进行调整。

![](./Image/Grafana仪表盘.jpg)

当我们访问 Grafana 控制面板时，可以看到请求延迟最初飙升至 5 秒，但随着新节点上线，延迟稳定在 1.2 秒左右。

自动扩缩容日志显示了以下内容……

```bash
### 自动扩缩容日志 ###
 [自动扩缩容日志]

已触发扩容：待处理任务数 > 100。

当前节点：1 个 (g5.xlarge)。

目标节点：3 个 (g5.xlarge)。

正在通过 Karpenter 启动 3 个新节点...
```

我们看到 GPU 利用率飙升。Ray 自动扩缩器立即检测到队列深度增加，并请求增加 2 个 GPU 节点。Karpenter 立即响应了这一请求。90 秒内，集群容量翻倍，请求延迟也趋于稳定。

它可以处理嘈杂的数据，高效地进行处理，并能自动扩展以满足用户需求，并在用户离开时缩减至零。
