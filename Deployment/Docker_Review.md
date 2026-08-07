# Docker_Review

## Dockerfile vs Docker compose

二者不是二选一的关系，Dockerfile注重单一应用构建，Docker compose注重整体编排。

| 特性         | Dockerfile                     | Docker Compose                                |
| :----------- | :----------------------------- | :-------------------------------------------- |
| **核心作用** | **构建** (Build) 自定义镜像    | **编排** (Orchestrate) 多容器应用             |
| **操作对象** | 单个镜像 (Image)               | 多个容器/服务 (Services)                      |
| **文件命名** | `Dockerfile` (默认无后缀)      | `docker-compose.yml` / `compose.yaml`         |
| **执行命令** | `docker build`                 | `docker compose up/down`                      |
| **语法格式** | DSL (领域特定语言)，指令式     | YAML 格式，声明式                             |
| **关注点**   | 环境依赖、代码打包、运行时配置 | 服务间网络、数据卷挂载、环境变量、启动顺序    |
| **类比**     | 🏗️ 建筑蓝图 (如何造一栋房子)    | 🏙️ 城市规划 (如何把房子、水电、道路组合成社区) |

Dockerfile：文本文件，告诉Docker如何一步步组装一个镜像

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

Docker Compose : YAML文件，定义了完整应用的所有服务及相互关系。`docker compose up`会解析整个文件，按依赖关系决定启动先后，每个容器各自走自己的镜像+挂载流程

典型配置：service, volumes, networks, depends_on, environment

```yaml
services:
	web:
		build: .
		ports: ["8000:8000"]
		depends_on: [db]
     db:
     	image: postgres:16
     	volumes: [pgdata:/var/lib/postgresql/data]
volumes:
	pgdata:
```

