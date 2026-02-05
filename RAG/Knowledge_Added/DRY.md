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

## **作用2：非侵入式修改**

- **不修改源代码**：原函数不知道被计时
- **可插拔**：随时添加或移除装饰器
- **不影响测试**：可以测试不带计时的函数

## **作用3：保持函数签名**

---

## 补充

存在异步函数、同步函数的分类情况，都需要写下来

```python
from functools import wraps
def time_count(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.perf_counter()
        result = func(*args, **kargs)
        end = time.perf_counter()
        print(f"普通操作耗时：{end-start}")
        return result
    
    @wraps(func)
    async def asyncwrapper(*args, **kargs):
        start = time.perf_counter()
        result = await func(*args, **kargs)
        end = time.perf_counter()
        print(f"普通操作耗时：{end-start}")
        return result
    return asyncwrapper if asyncio.iscoroutinefunction(func) else wrapper
```

## 类装饰器

```python
def log_class(cls):
    """类装饰器，在调用方法前后打印日志"""
    class Wrapper:
        def __init__(self, *args, **kwargs):
            self.wrapped = cls(*args, **kwargs)  # 实例化原始类
        
        def __getattr__(self, name):
            """拦截未定义的属性访问，转发给原始类"""
            return getattr(self.wrapped, name)
        
        def display(self):
            print(f"调用 {cls.__name__}.display() 前")
            self.wrapped.display()
            print(f"调用 {cls.__name__}.display() 后")
    
    return Wrapper  # 返回包装后的类

@log_class
class MyClass:
    def display1(self):
        print("这是 MyClass 的 display 方法")

obj = MyClass()
obj.display1()
```

这里getattr的作用是属性转发，这个类的初始化相当于是Wrapper的初始化，一开始是寻找Wrapper中是否存在display1，不存在的时候使用getattr，调用Myclass中的display1，查找成功，运行成功
