# Cpp

## lambda表达式

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

