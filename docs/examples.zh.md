# 示例 { #examples }

每个示例都是一个普通的用户程序——没有一个调用 Tiga 的算法
算子。下面的每个页面都会先解释所解决的问题（配有示意图和公式），
然后展示完整可运行的程序。所有示例都可以在源码目录下直接运行：

```bash
python examples/message_passing_autograd.py   # any example runs the same way
```

### [消息传递与图算法](examples/message-passing.md) { #message-passing-and-graph-algorithms }

| 示例 | 计算内容 |
|---|---|
| `message_passing_autograd.py` | 自定义邻居聚合，反向传播由编译器自动生成 |
| `gcn.py` | [GCN](https://en.wikipedia.org/wiki/Graph_neural_network) 层：邻居特征的加权和（SpMM），自动求梯度 |
| `diffusion.py` | 在图上执行一步显式欧拉扩散 |
| `custom_reducer.py` | 以自定义 (sum, count) reducer 代数编写的邻居均值 |
| `compiler_probes/pagerank.py` | 20 次 [PageRank](https://en.wikipedia.org/wiki/PageRank) 迭代，捕获为单个卷起的 repeat 算子——一个编译器探针 |

### [动态与生成的关系](examples/dynamic-relations.md) { #dynamic-and-generated-relations }

| 示例 | 计算内容 |
|---|---|
| `radius_autograd.py` | 可微的半径图：截断距离内的点成为边 |
| `knn_message_passing.py` | 精确的 k 近邻图，作为消息传递的输入 |

### [分布式与内存层级](examples/distributed-memory.md) { #distributed-and-memory-hierarchy }

| 示例 | 计算内容 |
|---|---|
| `distributed_halo.py` | 一个图切分到两个进程，自动进行 [halo](https://en.wikipedia.org/wiki/Halo_(computer_science)) 交换 |
| `hierarchical_memory.py` | 可链式调用的 `.disk()` / `.cpu()` 张量换出——gf 管理文件，具名换出可跨进程共享 |
| `paged_giant_graph.py` | 100 万节点的图落盘后按页流式处理（带 prefetch），结果写回磁盘 |
| `auto_offload.py` | RAM 预算让超限的图自动分页落盘，含 prefetch 流水线 |

### [Tensor kernel、程序与 Torch 互操作](examples/programs-and-interop.md) { #tensor-kernels-programs-and-torch-interop }

| 示例 | 计算内容 |
|---|---|
| `tensor_matmul.py` | [矩阵乘法](https://baike.baidu.com/item/矩阵乘法) `C = A·B`，两个操作数的梯度均由编译器推导 |
| `complex_autograd.py` | 对[复数](https://baike.baidu.com/item/复数)张量求导（共轭 Wirtinger 梯度） |
| `linear_recurrence.py` | 用通用的 map/cumsum/contract 把因果线性注意力递归融合成单个 CUDA kernel |
| `graph_program.py` | `@gf.jit` 自动跨 kernel 捕获：水平 kernel 融合 |
| `joint_autograd.py` | 前向与反向编译为单个可执行程序 |
| `torch_interop.py` | torch 张量通过零拷贝共享存储调用 Tiga UDF |
| `torch_library.py` | UDF 注册为 `torch.library` 算子，兼容 Inductor |
| `edge_nn_message_passing.py` | 每条边上一个 `torch.nn` MLP，编译为单个融合的 tile kernel |

### [可视化](examples/visualization.md) { #visualization }

| 示例 | 计算内容 |
|---|---|
| `gpu_heatmap.py` | 用张量广播与算术运算将标量场转换为 RGB [热力图](https://baike.baidu.com/item/热力图) |
| `visualize_fields.py` | 在三角剖分的圆盘上松弛稳态热传导（mean reducer 实现 Jacobi 迭代），渲染为粒子、Delaunay 网格与 GIF 视频 |
| `visualize_mesh.py` | 加载 OBJ icosphere，沿网格边扩散高斯包，渲染填充/wireframe/粒子图与环绕 GIF，并导出 PLY/OBJ 供 Blender 使用 |
| `visualize_gaussians.py` | 把彩色各向异性 3-D 高斯圆环渲染为 splats 与 ray-marched 密度体，并导出环绕 GIF |

### [线性求解器与控制流](examples/solvers.md) { #linear-solvers-and-control-flow }

| 示例 | 计算内容 |
|---|---|
| `fem_poisson.py` | 在图 stencil 上用无矩阵[共轭梯度法](https://baike.baidu.com/item/共轭梯度法)求解[泊松方程](https://baike.baidu.com/item/泊松方程) |
| `meshfree_linear_solve.py` | 同样的求解器风格，作用于由点位置生成的半径图 |
| `solvers.py` | 单个 `linear_solve` 入口——`method="cg" / "bicgstab" / "richardson"`——以 `@gf.jit` 下的普通 Python 循环实现 |
| `nonlinear_solve.py` | 非线性扩散 `−∇·((1+u²)∇u) = f`，以 Picard 定点迭代求解，VJP 穿透迭代循环 |

### [注意力与稠密关系](examples/attention.md) { #attention-and-dense-relations }

| 示例 | 计算内容 |
|---|---|
| `full_attention.py` | 无掩码的精确[注意力](https://baike.baidu.com/item/注意力机制)——每个 query 都对每个 key 做注意力 |
| `causal_dense_relation.py` | 用三角图表达的因果注意力——无需任何掩码代码 |
| `varlen_causal_attention.py` | 打包的变长序列，带逐序列因果掩码（`cu_seqlens`） |
| `tile_pruned_attention.py` | 注意力计算会跳过不可能影响 [softmax](https://en.wikipedia.org/wiki/Softmax_function) 结果的 key tile |

## 原生示例目前的覆盖边界 { #current-native-example-boundary }

上述示例都是可以真实运行的程序，而不是 API 示意。覆盖边界统一维护在一个地方：
[现在能跑什么](index.md#what-is-executable-today)。
