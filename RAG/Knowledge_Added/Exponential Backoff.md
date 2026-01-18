# 指数避让函数（Exponential Backoff）

指数退避算法是一种**智能重试策略**，主要用于处理失败的操作请求（如网络请求、API调用、数据库连接等）。它的核心思想是：**失败后不要立即重试，而是等待一段时间，且等待时间随着失败次数指数增长**。

## 一、为什么需要指数退避？

### 问题场景：

1. **服务过载**：服务器繁忙，立即重试会加重负担
2. **瞬时故障**：网络闪断、临时拥塞
3. **速率限制**：API 有调用频率限制

### 没有指数退避的问题：

```python
# 简单的重试 - 很糟糕！
while attempts < 5:
    try:
        response = call_api()
        break
    except:
        time.sleep(1)  # 固定等待1秒
        attempts += 1
```

**问题**：所有客户端同时重试 → **重试风暴** → 服务器压力更大

## 二、指数退避的工作原理

### 基本公式：

text

```
等待时间 = base_delay × (2 ^ 重试次数) + 随机抖动
```



### 示例（base_delay = 1秒）：

| 重试次数   | 计算          | 等待时间范围 |
| :--------- | :------------ | :----------- |
| 第一次重试 | 1 × 2¹ = 2秒  | 1.5-2.5秒    |
| 第二次重试 | 1 × 2² = 4秒  | 3-5秒        |
| 第三次重试 | 1 × 2³ = 8秒  | 6-10秒       |
| 第四次重试 | 1 × 2⁴ = 16秒 | 12-20秒      |
| 第五次重试 | 1 × 2⁵ = 32秒 | 24-40秒      |

------

## 三、关键组件

### 1. **指数增长**

python

```
# 核心算法
delay = base_delay * (2 ** retry_count)  # 指数增长
```



### 2. **随机抖动（Jitter）**

python

```
# 添加随机性，避免所有客户端同时重试
jitter = random.uniform(0.75, 1.25)  # ±25% 随机
actual_delay = delay * jitter
```



### 3. **最大退避和最大重试**

python

```
MAX_RETRIES = 5
MAX_DELAY = 60  # 最大等待60秒
```

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
            while True:
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

