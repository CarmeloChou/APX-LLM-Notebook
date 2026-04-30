# 镜像

三种命名方式

```yaml
# 1.直接命名
nginx
# 等价于
nginx:latest

# 2.指定版本
ubuntu:20.04

# 3.完整版
registry.cn-hangzhou.aliyuncs.com/mycompany/backend:1.2.0
```

# 容器

## 存储层和数据持久化

> 由于容器的特殊性，容器存储层与容器生命周期一致。容器存储层可对底层镜像进行修改，但只是复制镜像数据修改，只在当前容器可见，底层镜像不会改变。删除容器时，容器存储层消失，相关的数据也会消失。Docker实践中，容器存储层应该保持**无状态**，持久化数据使用**数据卷**或者**绑定挂载**。

```
## 使用数据卷（推荐）
$ docker run -v mydata:/var/lib/mysql mysql
## 使用绑定挂载
$ docker run -v /host/path:/container/path nginx
```

数据跳过存储层，直接写入宿主机

## 容器生命周期

![](./Image/容器生命周期.jpg)

```bash
## 创建并启动容器（最常用）
$ docker run nginx
## 分步操作
$ docker create nginx # 创建容器（不启动）
$ docker start abc123 # 启动容器
## 停止容器
$ docker stop abc123 # 优雅停止（发送 SIGTERM，等待后发送 SIGKILL）
$ docker kill abc123 # 强制停止（直接发送 SIGKILL）
## 暂停/恢复（不常用，但有时有用）
$ docker pause abc123 # 暂停容器内所有进程
$ docker unpause abc123 # 恢复
## 删除容器
$ docker rm abc123 # 删除已停止的容器
$ docker rm -f abc123 # 强制删除运行中的容器
```

# 仓库

## Registery&Repository&Tag

Docker Registry 是镜像分发和管理的核心组件。

> Docker Registry 是存储和分发 Docker 镜像的服务，类似于代码的 GitHub 或包管理的 npm。

![](./Image/仓库.jpg)

| 概念              | 说明               | 示例                           |
| ----------------- | ------------------ | ------------------------------ |
| Registry          | 存储镜像的服务     | Docker Hub、ghcr.io            |
| Repository (仓库) | 同一软件的镜像集合 | nginx、mysql、 mycompany/myapp |
| Tag (标签)        | 仓库内的版本标识   | latest、1.25、alpine           |

## 镜像加速器

由于网络原因，在国内直接访问 Docker Hub 可能会很慢。可以配置 镜像加速器 (Registry Mirror) 来加速下载。配置示例如下：

```json
// /etc/docker/daemon.json
{
	"registry-mirrors": [
	"https://your-accelerator-url"
	]
}
```

