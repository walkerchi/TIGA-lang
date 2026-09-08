# Tiga 与同类对比 { #how-tiga-compares }

Tiga 是面向 message passing 与图计算的**编译器**——不是 framework，也不
是通用深度学习编译器。有意义的对比取决于今天手头用的是什么。

![旗舰稀疏矩阵-向量案例上的实测对比：Tiga 的融合 kernel 领先，PyG 的 gather-scatter MessagePassing 与 eager index_add 路径落后一个数量级](assets/results/spmm-provider-compare.svg)

上面的案例是 GNN 最典型的内层循环——在不规则 CSR 关系上加权求和
（N = 131 072，度数 ≈ 16，特征宽 16，RTX 5070 Ti、L2 热缓存实测）。
同一张关系、同一份数学，每个实现都与 `torch.sparse.mm` 对齐过正确性。
复现命令：`python -m benchmarks.sparse_compute.weighted_aggregation --topology irregular --locality random --device cuda --index-dtype i32 --nodes 131072 --degree 16 --features 16,64`。

## 一览

| 同类 | 定位 | Tiga 的区别 |
|---|---|---|
| [PyG](https://pytorch-geometric.readthedocs.io/) | GNN framework；其 gather-scatter MessagePassing 不做融合 | 实测的是这条路径（PyG 2.8.0）：Tiga 的融合 kernel 在上面案例约为它的 29 倍。PyG 另有 `message_and_aggregate()` 稀疏路径，不在本次测量范围内 |
| [DGL](https://www.dgl.ai/) | 带融合 `gspmm`/`gsddmm` 算子的 GNN framework | 规则 SpMM 上接近 `torch.sparse.mm`；Tiga 额外保留关系来源、生成式拓扑与编译器生成的 VJP |
| [torch.compile](https://pytorch.org/docs/stable/torch.compiler.html) | 通用 PyTorch 编译器 | 无法图捕获 `torch.sparse.mm`；对 gather-scatter 写法只能追平，谈不上融合 |
| [Taichi](https://www.taichi-lang.org/) / [Warp](https://nvidia.github.io/warp/) | Python kernel JIT | 逐线程 kernel 靠手写；Tiga 从关系与 edge/node UDF 出发 |
| [Triton](https://triton-lang.org/) / [TileLang](https://github.com/tile-ai/tilelang) | tile / kernel DSL | 是 Tiga lowering 的 backend，不是竞品 |
| [Graphiler](https://github.com/xiezhq-hermann/graphiler) | 建在 DGL + TorchScript 上的研究型 GNN 编译器，2022 年后停更 | 同一生态位，但 Tiga 在维护且多 backend |
| [GALA](https://github.com/ADAPT-uiuc/GALA-GNN-Acceleration-LAnguage) | 研究型 GNN DSL + 编译器（OOPSLA 2025） | 最接近的在世学术同行；自带前端与 benchmark 套件，不是即插即用的对比项 |

## 什么时候用 Tiga，什么时候不用

该用 Tiga：message passing、稀疏聚合、图上的注意力——并且前向 kernel
和它的梯度都由同一份定义生成，跑 CPU 或 GPU。

该用别的：大规模稠密 [Transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) 训练（PyTorch）、手调 stencil/PDE（Taichi、Devito）、纯图分析如固定拓扑上的 [PageRank](https://baike.baidu.com/item/PageRank)（GraphIt、Gunrock）。

这些取舍背后的完整设计调研——Ebb、Simit、TACO、GraphIt、Taichi、
Legion 等——保存在仓库的
[`docs/RELATED_WORK.md`](https://github.com/walkerchi/TIGA-lang/blob/main/docs/RELATED_WORK.md)。
