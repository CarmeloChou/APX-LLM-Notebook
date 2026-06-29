# python 模块

## aysncio

协程模块，异步非阻塞，程序运行时间取决于最长的等待时间。

协程是 `asyncio` 的核心概念之一。它是一个特殊的函数，可以在执行过程中暂停，并在稍后恢复执行。协程通过 `async def` 关键字定义，并通过 `await` 关键字暂停执行，等待异步操作完成。

### 使用方法

async 定义异步函数、上下文、迭代器

```python

async def func():
    pass

async with lock:
    pass

async for i in range(100):
    pass

```

await等待异步操作完成

```python
async def func():
    data = await func1()
    return data
```



### 核心函数

由于Jupyter Notebook本身就是在`asyncio.run()`中，因而，在循环事件中调用回报错。只需要直接调用`await func()`即可。

| 方法/函数                       | 说明                          | 示例                                                 |
| :------------------------------ | :---------------------------- | :--------------------------------------------------- |
| **`asyncio.run(coro)`**         | 运行异步主函数（Python 3.7+） | `asyncio.run(main())`                                |
| **`asyncio.create_task(coro)`** | 创建任务并加入事件循环        | `task = asyncio.create_task(fetch_data())`           |
| **`asyncio.gather(\*coros)`**   | 并发运行多个协程              | `await asyncio.gather(task1, task2)`                 |
| **`asyncio.sleep(delay)`**      | 异步等待（非阻塞）            | `await asyncio.sleep(1)`                             |
| **`asyncio.wait(coros)`**       | 控制任务完成方式              | `done, pending = await asyncio.wait([task1, task2])` |

```python
import aysncio
import time
import random

# 打印单词的简单示例
async def print_test(words, delay):
    print(words)
    await asyncio.sleep(delay)
    return words

async def print_all():
    words_list = ["a", "b", "c", "zhou"]
    start = time.perf_counter()

    tasks = [print_test(word, random.randrange(0, 5, 1)) for word in words_list]
    # 并发进行多个任务
    results = await asyncio.gather(*tasks)

    for result in results:
        print(result)
	# 打印时间取决于随机的最长时间
    print(f"耗时：{time.perf_counter() - start}")
```

### 异步锁

```python
class BankAccount:
    def __init__(self, balance=100):
        self.balance = balance
    
    async def unsafe_transfer(self, amount):
        """不安全的转账（有竞态条件）"""
        # 模拟一些处理
        await asyncio.sleep(0.01)
        
        # 读取余额
        current = self.balance
        await asyncio.sleep(0.01)  # ❌ 这里可能被切换！
        
        # 计算新余额
        new_balance = current + amount
        await asyncio.sleep(0.01)  # ❌ 这里可能又被切换！
        
        # 写入余额
        self.balance = new_balance
        print(f"转账 {amount}，新余额: {self.balance}")

async def unsafe_demo():
    """演示竞态条件"""
    account = BankAccount(100)
    
    # 并发执行100次转账
    tasks = []
    for i in range(100):
        task = asyncio.create_task(account.unsafe_transfer(1))
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    print(f"最终余额（期望200）: {account.balance}")
    # 可能输出：最终余额: 101（因为竞态条件！）

# asyncio.run(unsafe_demo())
```

以上输出结果为101，因为数据一直在覆写，而没有叠加。需要加锁

```python
class SafeBankAccount:
    def __init__(self, balance=100):
        self.balance = balance
        self.lock = asyncio.Lock()
    
    async def unsafe_transfer(self, amount):
        async with self.lock:
            """不安全的转账（有竞态条件）"""
            # 模拟一些处理
            await asyncio.sleep(0.01)
            
            # 读取余额
            current = self.balance
            await asyncio.sleep(0.01) 
            
            # 计算新余额
            new_balance = current + amount
            await asyncio.sleep(0.01)  
            
            # 写入余额
            self.balance = new_balance
            print(f"转账 {amount}，新余额: {self.balance}")

async def safe_demo():
    """演示竞态条件"""
    account = SafeBankAccount(100)
    
    # 并发执行100次转账
    tasks = []
    for i in range(100):
        task = asyncio.create_task(account.unsafe_transfer(1))
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    print(f"最终余额（期望200）: {account.balance}")


# asyncio.run(safe_demo())
```

加上互斥锁，数据为叠加递增的。

## threading

### 什么是线程？

你可以把线程想象成办公室里的员工：

- 一个单线程程序就像只有一个员工，他必须顺序完成打印文档、回复邮件、泡咖啡等所有工作。
- 多线程程序则像拥有多个员工，他们可以**同时**进行不同的任务，大大提高了工作效率。

在计算机科学中：

- **进程**：一个运行中的程序，拥有独立的内存空间（例如，你同时打开的浏览器和音乐播放器就是两个进程）。
- **线程**：进程内的一个独立执行流，是 CPU 调度的基本单位。同一个进程内的所有线程**共享该进程的内存空间**（如全局变量）。

### Python 的线程与全局解释器锁 (GIL)

Python有一个叫做全局解释器锁 (Global Interpreter Lock， GIL) 的机制，GIL 确保了在任意时刻，只有一个线程可以执行 Python 字节码。

**这意味着什么？** 对于 CPU 密集型任务（如科学计算、图像处理），由于 GIL 的存在，多线程通常无法利用多核优势来提升计算速度，甚至可能因为线程切换的开销而变慢。

**那么，Python 多线程的用武之地在哪里？** 对于 I/O 密集型任务（如网络请求、读写文件、等待用户输入），线程在等待 I/O 操作完成时会释放 GIL，从而让其他线程运行。这可以显著提升程序的整体响应速度和效率，因为你在等待一个网页响应时，程序可以去处理另一个任务。

在 Python 中，可以通过继承 `threading.Thread` 类或直接使用 `threading.Thread` 构造函数来创建线程。

[进程、线程和协程之间的区别和联系_进程和线程和协程-CSDN博客](https://blog.csdn.net/daaikuaichuan/article/details/82951084)

```python
import threading
import time
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

class test_thread(threading.Thread):
    def __init__(self, thread_id=None, data=100):
        super().__init__()
        self.result = [i for i in range(data)]
        if thread_id is not None:
            self.name = f"Thread-{thread_id}"

    def print_all(self):
        for i in self.result:
            print(f"This is {self.name}")
            time.sleep(0.1)
            print(i)

    @time_count
    def run(self):
        self.print_all()

num_thread = 4
threads = []
results = []
for i in range(num_thread):
    thread = test_thread(i, 100//num_thread) # 这里使用/计算结果为浮点数，需要使用整数
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

```

## logging

logging日志模块，为python自带的日志系统，可追踪程序的运行状态、调试错误以及记录重要信息。

>- **DEBUG**：详细的调试信息，通常用于开发阶段。
>- **INFO**：程序正常运行时的信息。
>- **WARNING**：表示潜在的问题，但程序仍能正常运行。
>- **ERROR**：表示程序中的错误，导致某些功能无法正常工作。
>- **CRITICAL**：表示严重的错误，可能导致程序崩溃。

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# 可以修改日志输出的格式
logging.basicConfig(
	level=logging.DEBUG,
    # 输出的格式，时间-名称-r
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.debug("这是一条调试信息")
logging.info("这是一条普通信息")
logging.warning("这是一条警告信息")
logging.error("这是一条错误信息")
logging.critical("这是一条严重错误信息")

```

```bash
2026-01-30 09:55:19 - root - DEBUG - 这是一条调试信息
2026-01-30 09:55:19 - root - INFO - 这是一条调试信息
2026-01-30 09:55:19 - root - WARNING - 这是一条调试信息
2026-01-30 09:55:19 - root - ERROR - 这是一条错误信息
2026-01-30 09:55:19 - root - CRITICAL - 这是一条严重错误信息
```

```python
# 手动清除logging
root_logger = logging.getLogger()
# 2. 清除所有处理器
for handler in root_logger.handlers[:]:  # 使用[:]创建副本
    root_logger.removeHandler(handler)
    handler.close()  # 重要：释放资源
```

```python
# 自定义输出格式，其实就是重写方法
import logging
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_dict = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage()
        }
        return json.dumps(log_dict, ensure_ascii=False)
    
logger = logging.getLogger("MyAgent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

logger.info("测试信息")
# 清除所有handler
logger.handlers.clear()
```



## `__name__`模块

```python
if __name__ == "__main__":
    do something....
# 是常见的接口判断，这意味着，程序运行是由当前模块产生的。
```

name的常见用法。当`__name__`被作为模块导入时，可以识别出导入的模块，这意味着，可以区分和记录不同模块代码的调用，从而进行debug或者日志log辅助开发。

example：

```python
# ./__name__test.py
if __name__ == "__main__":
    print("在当前函数调用")
else:
    print(f"在{__name__}模块中调用")    
  
```

```python
# 当作为模块导入时，会打印出__name__test
# 当直接调用时，会打印 __main__
import __name__test # 相当于cpp中的#include
```

## 上下文管理器

上下文管理器是 Python 中用于**管理资源**的对象，确保资源被正确获取和释放。

```python
# 最经典的例子：文件操作
with open('file.txt', 'r') as f:  # 进入上下文
    content = f.read()             # 使用资源
                                  # 离开上下文，自动关闭文件

# 等价于：
f = open('file.txt', 'r')
try:
    content = f.read()
finally:
    f.close()  # 确保文件被关闭
```

同步上下文管理器

```python
class DatabaseConnection:
    def __init__(self, host, port, database):
        """初始化方法：设置连接参数"""
        self.host = host
        self.port = port
        self.database = database
        self.connection = None  # 实际的连接对象，在__enter__中创建
        print(f"初始化连接参数：{host}:{port}/{database}")
    
    def __enter__(self):
        """进入上下文时调用：建立实际连接"""
        print(f"正在连接到 {self.host}:{self.port}/{self.database}")
        # 模拟建立连接
        self.connection = f"Connection to {self.database}"
        print("连接成功！")
        return self  # 返回给 as 后面的变量
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时调用：关闭连接"""
        print(f"关闭连接：{self.connection}")
        self.connection = None
        if exc_type:
            print(f"发生异常：{exc_type.__name__}: {exc_val}")
        print("连接已关闭")
        return False  # 不抑制异常

# 使用
print("=== 开始使用数据库 ===")
with DatabaseConnection("localhost", 5432, "mydb") as db:
    print(f"使用连接：{db.connection}")
    # 模拟数据库操作
    print("执行查询...")
    # 如果没有异常，正常退出
print("=== 使用结束 ===")

# 输出：
# === 开始使用数据库 ===
# 初始化连接参数：localhost:5432/mydb
# 正在连接到 localhost:5432/mydb
# 连接成功！
# 使用连接：Connection to mydb
# 执行查询...
# 关闭连接：Connection to mydb
# 连接已关闭
# === 使用结束 ===
```

异步上下文管理器

```python
class AsyncDatabaseConnections:
    async def 
```

### 特殊方法

py特殊方法以双下划线开头和结尾的方法，可以实现特定的行为。

```python
__enter__(self)          # 进入with块时调用
# 这里退出必须包含四个参数
- self: 实例本身。
- exc_type: 异常类型（如果没有异常，则为 None）。
- exc_val: 异常值（如果没有异常，则为 None）。
- exc_tb: 异常的 traceback 对象（如果没有异常，则为 None）。
__exit__(self, exc_type, exc_val, exc_tb)  # 离开with块时调用
```



## Pydantic

**Pydantic** 是一个 Python 数据验证和设置管理库，**用类型注解来验证数据**。

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str | None = None

# 自动验证
user = User(name="张三", age=25)  # ✅
user = User(name="李四", age="二十五")  # ❌ 自动报错：age 不是整数

user = User(name="王五", age="30")  # ✅ 字符串"30"自动转成整数30

print(User.schema_json(indent=2))
# 输出标准的 JSON Schema，可供API文档使用
```

## typing

typing是 Python 的类型提示标准库，用于为代码添加类型注解，提高代码的可读性和可维护性，并支持静态类型检查。

```python
from typing import List, Dict, Tuple, Optional

# List-列表
names: List[str] = ["Alice", "Bob"]
scores: List[int] = [1, 2, 3]

# Dict
student: Dict[str, any] = {"name": "Alice", "age": 20, "scores": [95, 87]}
config: Dict[str, int] = {"timeout": 30, "max_retires": 3}
    
# Tuple
point: Tuple[float, float] = (1.5, 2.5)
coords: Tuple[int, ... = (1, 2, 3, 4)
mixed: Tuple[str, int, bool] = ("Alice", 20, True)
```

```python
from typing import Union
# 让一个变量/参数有多种类型选择，但必须是明确指定的类型之一
def get_item(id: Union[int, str]) -> Item:
    # id要么是int，要么是str
    if isinstance(id, int):
        # 处理数字id
    else:
        # 处理字符串id
        
Optinal[X] 就是 Union[X, None]
```

## subprocess

Python 的一个**标准库**，无需安装，直接导入即可使用。可管理子进程，**并控制它们的输入、输出和错误流**。

```python
import subprocess

result = subprocess.run(
['dir'],
shell=True,
capture_output=True,
encoding='gbk'
)

result.check_returncode()
print(result)
```

使用Popen创建子进程可以替代run方法

```python
proc = subprocess.Popen(['timeout', '1'])
while proc.poll() is None:
    print('Working...')
    print('Exit status', proc.poll())
```

## Yield

它的作用是把一个普通的函数变成**生成器（Generator）**，让函数具备了“暂停”和“记忆”的能力。

- **`return`（普通函数）**：就像一个只会把结果一次性打包扔给你的快递员。函数一旦遇到 `return`，就会立刻结束并销毁所有内部状态。
- **`yield`（生成器函数）**：就像一个可以多次互动的自动售货机。每次遇到 `yield`，它会吐出一个值并**暂停**，同时**记住**当前的所有状态（比如变量是多少、执行到哪一行了）。等你下次再要数据时，它会从上次暂停的地方继续往下执行。

```python
def number_generator(n):
    for i in range(n):
        print(f"准备产出数字: {i}")
        yield i  # 产出 i，然后暂停在这里，等待下一次调用

# 调用函数时，代码其实并没有立刻执行，而是返回了一个生成器对象
gen = number_generator(3) 

# 每次调用 next() 或使用 for 循环，才会真正触发一次执行
print(next(gen))  # 输出: 准备产出数字: 0  换行 0
print(next(gen))  # 输出: 准备产出数字: 1  换行 1
print(next(gen))  # 输出: 准备产出数字: 2  换行 2
```

```python
def echo():
    while True:
        # 暂停并等待外部传入值，传入的值会被赋值给 received
        received = yield 
        print(f"我收到了: {received}")

g = echo()
next(g)           # 必须先启动生成器，让它运行到第一个 yield 处暂停
g.send("你好")    # 输出: 我收到了: 你好 （同时继续循环到下一个 yield 暂停）
g.send("Python")  # 输出: 我收到了: Python
```

## Path

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
```

等价于告诉 Python：

> “如果你找不到模块，也去这个目录找。”

将当前文件所在目录的src文件夹加入Python模块搜索路径。可以直接实现：

```python
import something # 来自 src/somthing,.py
```



## 设计模式

### 单例模式

这里cls指类本身，这时self还没创建出来，必须使用new创建，new类似于cpp中的构造函数。init单纯负责初始化，创建之后，才能进行初始化赋值。

```python
class ThreadSafeSingleton:
    """线程安全的单例模式"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    print("创建线程安全的单例实例")
        return cls._instance
    
    def __init__(self):
        # 确保只初始化一次
        if not hasattr(self, '_initialized'):
            self.data = {}
            self._initialized = True
```

简单的单例：

```python
class ThreadSingle:
    def __init__(self):
        self.connection_string = "database://localhost:5432/mydb"
        print("数据库连接已创建")
               
    def query(self, sql):
        return f"执行查询: {sql}"
            
```

使用new：当类未被创建时，创建类并初始化，第二次依然会被初始化，所以会显示第二个实例。这里可以加一个判断，来保证名称不会被改写。

```python
class TreadSingle:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            print("创建新的单例实例")
        else:
            print("返回已存在的单例实例")
        return cls._instance
    
    def __init__(self, name):
        # 注意：__init__ 每次都会被调用
        self.name = name
        print(f"初始化实例，名称: {name}")
        
# 测试代码
print("=== 测试单例模式 ===")
s1 = Singleton("第一个实例")
s2 = Singleton("第二个实例")

print(f"s1 的 ID: {id(s1)}")
print(f"s2 的 ID: {id(s2)}")
print(f"s1 和 s2 是同一个对象吗？ {s1 is s2}")
print(f"s1 名称: {s1.name}")
print(f"s2 名称: {s2.name}")  # 注意：这里会显示"第二个实例"
```

