# Cpp

## lambda表达式

cpp11引入的匿名函数对象

```cpp
[捕获列表](参数列表) -> 返回类型 {
    函数体
}

// 定义一个lambda并立即调用
[]() {
    std::cout << "Hello Lambda!" << std::endl;
}();  // 注意最后的()表示立即调用

// 或者赋值给变量
auto sayHello = []() {
    std::cout << "Hello Lambda!" << std::endl;
};
sayHello();  // 调用

```



```c++
// 传统写法
auto lambda = [](int x) {
    // 这里不能调用 lambda(x-1)，因为 lambda 在自身定义中不可见
    return x * ???;  // 无法递归
};
```

```cpp
#include <functional>
std::function<int(int)> factorial;
factorial = [&factorial](int n) -> int {
    return n <= 1 ? 1 : n * factorial(n - 1);
};
// 问题：std::function 有类型擦除开销
```

```c++
// C++23
auto fibonacci = [](this auto&& self, int n) -> int {
    if (n <= 1) return n;
    return self(n - 1) + self(n - 2);
};

cout << fibonacci(10) << endl;  // 55
```

