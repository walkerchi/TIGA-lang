---
title: Tiga — 支持 PyTorch 的 Graph Message Passing JIT 编译器
homepage: true
description: 使用 PyTorch Tensor 编写可微图程序。Tiga 编译 graph message passing，支持稀疏聚合、图神经网络算子与 GPU 图计算，提供示例、梯度接口和性能对比。
hide:
  - navigation
  - toc
---

<div class="tiga-home-hero" markdown>

# ![Tiga](assets/tiga-logo.svg){ width="300" .tiga-brand-logo } { #tiga }

面向 [graph message passing](https://en.wikipedia.org/wiki/Graph_neural_network#Message_passing)
程序的可微 [JIT 编译器](https://en.wikipedia.org/wiki/Just-in-time_compilation)。

</div>

定义图，编写每条边传递的消息，再指定消息如何合并。调用程序即可得到节点结果，
并自动推导梯度，无需手写反向函数。日常使用不要求理解编译器 IR。

直接支持 [PyTorch](https://en.wikipedia.org/wiki/PyTorch) 的 `torch.Tensor`，
无需额外包装；通过 `torch.autograd.grad` 或 `.backward()` 求导，
可集成到现有 PyTorch 模型与训练流程中。

## 从这里开始 { #choose-a-path }

<div class="gf-learning-paths" markdown>

<div class="gf-learning-path" markdown>

### 使用 Tiga

1. [安装并运行第一个程序](getting-started.md)。
2. [理解图、字段和消息](programming-model.md)。
3. [确认执行方式与排查问题](execution.md)。
4. [选择一个完整示例](examples.md)。

从 Python 开始，不要求预先了解 MLIR。

</div>

<div class="gf-learning-path" markdown>

### 开发编译器

1. [理解编译器内部的分层](compiler-pipeline.md)。
2. [沿着一个真实程序逐层阅读 IR](ir-walkthrough.md)。
3. [构建、测试并定位实现代码](development.md)。

先了解 Python 编程模型，再进入内部实现。IR 教程先解释符号，
再展示 pass 与真实编译输出。

</div>

</div>

## 程序长什么样 { #a-small-semantic-surface }

下面的示例在 NVIDIA GPU 上计算邻居值的加权和，并通过 PyTorch 求导：

```python
import torch
import tiga as tg

device = torch.device("cuda")

class WeightedAggregation(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x

graph = tg.Graph.from_csr(
    torch.tensor([0, 2, 3, 5], dtype=torch.int64, device=device),
    torch.tensor([0, 2, 1, 0, 1], dtype=torch.int64, device=device),
    num_src=3,
)
x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
weight = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], device=device, requires_grad=True)
program = WeightedAggregation()
out = program(
    graph=graph,
    src={"x": x},
    dst={},
    edge={"weight": weight},
)
print(out.tolist())  # [11.0, 8.0, 17.0]
dx, dw = torch.autograd.grad(out.sum(), (x, weight))
print(dx.tolist())   # [7.0, 10.0, 3.0]
print(dw.tolist())   # [1.0, 3.0, 2.0, 1.0, 2.0]
```

图提供边，`edge()` 提供每条边的消息，
[reducer](https://en.wikipedia.org/wiki/Fold_(higher-order_function)) `tg.sum()`
负责按目标节点合并消息。`tg` 只是 `tiga` 的 import 别名，不是另一个包。

图索引、输入、输出和梯度均位于 GPU，无需包装 Tensor 或切换求导接口。
运行环境配置见[安装指南](getting-started.md)；
[执行诊断](execution.md)说明如何检查所选执行路径。

## 目前支持什么 { #what-is-executable-today }

支持 CPU 与 NVIDIA GPU 执行，以及面向大图的分层存储和分布式计算。
常规计算直接使用 `torch.Tensor`；分层存储与分布式等高级运行时功能
使用原生 `tg.Tensor`，见[内存与分布式](memory-and-distributed.md)。
具体操作、数据类型与执行路径的覆盖范围见[支持矩阵](roadmap.md)。

## 稀疏图与图神经网络算子

邻居加权聚合是一种[稀疏矩阵](https://baike.baidu.com/item/稀疏矩阵)运算，
自定义边函数则把它扩展到更一般的图程序。
[PyTorch sparse 与 message passing 指南](pytorch-sparse-message-passing.md)
用一个可核对结果的例子串起 `torch.sparse.mm`、CSR 图与
[图神经网络](https://baike.baidu.com/item/图神经网络)（graph neural network，GNN）聚合。
Tiga 编译其中的算子，不替代完整的模型库与训练框架。

## 深入理解实现 { #from-semantics-to-provider-code }

编译器把图上的计算逐步变成遍历方式和执行计划。这些是内部决策，
不是需要额外编写的 API。[编译器入门](compiler-pipeline.md)解释分层目的；
[IR 实例教程](ir-walkthrough.md)展示真实的 `gf.apply`、
`gf_iter.traverse`、`gf_kernel.launch` 和 task 操作。
[技术报告](https://github.com/walkerchi/tiga-lang-paper)进一步讲述编程模型与编译器设计，
并提供性能评估及复现数据。

## 评估性能 { #retained-structure-changes-the-algorithm }

[性能与扩展性](experiments.zh.md)对比指定工作负载的执行时间、显存开销、
大图容量和分布式执行。[测量协议](performance.md)定义计时与正确性核对方式，
[基准归档](benchmark-results.md)提供更多测量案例。

查函数签名时使用 [Python API 参考](api.md)；了解项目定位时阅读[同类对比](comparison.md)。
