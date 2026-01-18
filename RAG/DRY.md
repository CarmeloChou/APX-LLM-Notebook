# 代码装饰器DRY（don’t repeat yourself）

被发明用来避免重复代码。

之前，如果想要对不同函数运行时间进行统计，需要重复构建代码。

## **作用1：分离关注点**

```python
# 每个函数都要写重复的代码
def function1():
    start_time = time.time()  # ⚠️ 重复代码
    # 实际功能
    end_time = time.time()    # ⚠️ 重复代码
    print(f"耗时: {end_time - start_time}")

def function2():
    start_time = time.time()  # ⚠️ 重复代码
    # 实际功能
    end_time = time.time()    # ⚠️ 重复代码
    print(f"耗时: {end_time - start_time}")
```

代码装饰器之后，解决了重复问题

```python
# 定义一个通用功能
def timer_decorator(func):
    def wrapper():
        start_time = time.time()
        result = func()  # 执行原函数
        end_time = time.time()
        print(f"耗时: {end_time - start_time}")
        return result
    return wrapper

# 优雅地应用到多个函数
@timer_decorator
def function1():
    # 只写核心逻辑
    pass

@timer_decorator
def function2():
    # 只写核心逻辑
    pass
```

```python
# 业务函数只关心业务逻辑
def process_data(data):
    """只关注数据处理逻辑"""
    return [x * 2 for x in data]

# 性能监控由装饰器负责
@timer_decorator
def process_data(data):
    """自动获得计时功能"""
    return [x * 2 for x in data]
```

## **作用2：非侵入式修改**

- **不修改源代码**：原函数不知道被计时
- **可插拔**：随时添加或移除装饰器
- **不影响测试**：可以测试不带计时的函数

## **作用3：保持函数签名**