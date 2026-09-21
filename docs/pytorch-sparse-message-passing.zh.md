---
title: PyTorch 稀疏矩阵乘法与 Graph Message Passing
description: 在 GPU 上对照 Tiga message passing 与 torch.sparse.mm，核对节点和边的梯度，理解 CSR 稀疏聚合与图神经网络 GNN 的关系。
---

# PyTorch 稀疏矩阵乘法与 graph message passing

[PyTorch](https://en.wikipedia.org/wiki/PyTorch) 通过 `torch.sparse` 提供稀疏矩阵运算，
Tiga 则通过沿图边的 [message passing](https://en.wikipedia.org/wiki/Graph_neural_network#Message_passing)
描述计算。对于邻居加权求和，两者表达的是同一个运算；一般的自定义边函数则不一定能写成矩阵乘法。

## 稀疏矩阵如何表示图

在 [compressed sparse row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
（CSR）格式中，每一行表示一个目标节点，存储的列索引表示源节点，对应的数值表示边权。
源节点 j 到目标节点 i 的边贡献如下：

$$
y_i = \sum_{e=(j\to i)} w_e x_j.
$$

当节点携带特征向量时，这就是稀疏矩阵与稠密矩阵乘法（SpMM）。
`torch.sparse.mm(adjacency, features)` 与 Tiga 中的 `edge.weight * src.x`
消息加 sum reducer 都表达该加权和。
[图编程模型](programming-model.md#graph-is-a-relation-not-a-csr-tensor)通过示意图解释连接关系；下例明确采用“行是目标、列是源”的约定。

## GPU 示例：同时核对前向结果与梯度

先完成[安装](getting-started.md)，包括与 CUDA 环境匹配的 Torch。
下例为两个实现创建独立的输入，核对输出、节点梯度和边权梯度。这是正确性检查，不是性能测试。

运行 [`python examples/pytorch_sparse_message_passing.py --device cuda`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/pytorch_sparse_message_passing.py)。
同一份源码也支持 `--device cpu`。

```python
--8<-- "examples/pytorch_sparse_message_passing.py"
```

两种实现都返回普通 `torch.Tensor`。五条存储边各自拥有独立的边权。
在这个 sum 运算中，没有入边的行输出零；相同源、目标之间的重复边按每条存储边分别累加。
其他 reducer 的[空邻域规则](reducers.md)需要单独检查。
Torch 的稀疏格式、设备及梯度支持以
[`torch.sparse.mm` 文档](https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html)为准；
Tiga 的覆盖范围见[支持矩阵](roadmap.md)。

## 与图神经网络有什么关系

[图神经网络](https://baike.baidu.com/item/图神经网络)（graph neural network，GNN）
把图聚合与特征变换、非线性运算组合起来。上面的加权和只是聚合模块，
不等于完整、已归一化的 graph convolutional network（GCN）层。
自环、度归一化、可学习变换与训练目标仍由具体模型定义。
参见[多特征 GCN 形状的聚合示例](examples/message-passing.md#gcn-aggregation)
与 [Torch 模型集成](examples/programs-and-interop.md)。

PyTorch Geometric（PyG）提供图学习层与训练工具；Tiga 聚焦于编译图程序，
包括自定义边消息及其已支持的梯度，不是所有 PyG 层的直接替代品。
[同类对比](comparison.md)进一步解释这些抽象边界。

## 不止固定的稀疏邻接矩阵

Radius 与 k-nearest-neighbor 图根据坐标或特征确定邻居，stencil 根据网格规则确定邻居。
[动态关系示例](examples/dynamic-relations.md)介绍这些构造器、距离度量与边界条件。
执行时是物化边，还是在计算中生成候选，取决于关系与所选 backend。

性能比较需要固定图、特征、dtype、设备与梯度需求，并区分建图、编译、预热后执行和内存开销。
[性能报告](experiments.md)提供稀疏聚合、radius/kNN、大图与分布式案例；
这些结果不代表 Tiga 对任意图神经网络或 `torch.sparse` 运算都更快。
