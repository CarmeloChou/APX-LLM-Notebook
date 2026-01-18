# Makefile

[makefile教程](./Paper/Makefile教程-廖雪峰-2025-06-16.pdf)

Linux的`make`程序用来自动化编译大型源码，很多时候，在Linux下编译安装软件，只需要敲一个make就可以全自动化完成。

`make`能自动化完成这些工作，是因为项目提供了一个`Makefile`文件，它负责告诉`make`，应该如何编译和链接程序。

`Makefile`相当于Java项目的`pom.xml`，Node工程的`package.json`，Rust项目的`Cargo.toml`，不同之处在于，`make`虽然最初是针对C语言开发，但它实际上并不限定C语言，而是可以应用到任意项目，甚至不是编程语言。此外，`make`主要用于Unix/Linux环境的自动化开发，掌握`Makefile`的写法，可以更好地在Linux环境下做开发，也可以为后续开发Linux内核做好准备。

## Makefile规则

编写Makefile文件，指定操作规则和逻辑，程序会自动化运行。

我们举个例子：在当前目录下，有3个文本文件：`a.txt`，`b.txt`和`c.txt`。

现在，我们要合并`a.txt`与`b.txt`，生成中间文件`m.txt`，再用中间文件`m.txt`与`c.txt`合并，生成最终的目标文件`x.txt`，整个逻辑如下图所示：

```
┌─────┐ ┌─────┐ ┌─────┐
│a.txt│ │b.txt│ │c.txt│
└─────┘ └─────┘ └─────┘
   │       │       │
   └───┬───┘       │
       │           │
       ▼           │
    ┌─────┐        │
    │m.txt│        │
    └─────┘        │
       │           │
       └─────┬─────┘
             │
             ▼
          ┌─────┐
          │x.txt│
          └─────┘
```

根据上述逻辑，我们来编写`Makefile`。

```makefile
# 目标文件：依赖文件1 依赖文件2
m.txt : a.txt b.txt
	cat a.txt b.txt > m.txt
```

编写一个简单的测试脚本，囊括主要基础知识。输入`make help`时，我们希望看到当下Makefile文件中的指令内容。本测试Makefile主要功能为删除测试文件，内容如下：

```makefile
.PHONY: help del del2        # 第1行

help:                        # 第2行
	@echo "test makefile induction"    # 第3行
	@echo "  make del  - delete test.txt"  # 第4行
	@echo "  make del2 - delete test2.txt" # 第5行

del:                         # 第6行
	@rm -f test.txt           # 第7行
	@echo "has deleted test.txt"  # 第8行

del2:                        # 第9行
	rm -f test2.txt          # 第10行
	echo "has deleted test2.txt" # 第11行
```

其中`.PHONY`指伪目标内容，它包含本次makefile中的所有指令名称，执行`del2`时，并没有一个del2的文件，所以每次执行make del2，都会执行上述命令。**PHONY的作用就是将del2视为规则而不是文件。**

一般来说，并不需要用`.PHONY`标识`clean`等约定俗成的伪目标名称，除非有人故意搞破坏，手动创建名字叫`clean`的文件。

命令加上@让程序不要打印该指令，比如`del`指令执行时，只会输出`has deleted test.txt`而`del2`指令执行时，会输出`rm -f test2.txt` 、 `echo "has deleted test2.txt"` 、`has deleted test2.txt`。

如果删除文件时文件已经不再。如果想忽略错误，继续执行后续命令，可以在需要忽略错误的命令前加上`-`，或者在删除后加上`-f`：

```makefile
ignore_error:
	-rm zzz.txt
	echo 'ok'
```