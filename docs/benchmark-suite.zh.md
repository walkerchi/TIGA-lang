# 基准测试套件 { #benchmark-suite }

`benchmarks/` 目录树是 [实测结果](benchmark-results.md) 中每一行数据背后的
可执行证据库，按**负载语义**分组，绝不按 provider 分组：Triton、Torch、
厂商库与 Tiga 生成的代码在同一用例内是平级参与者，而非顶层分类。

## 目录布局 { #layout }

| 目录 | 范围 | 入口点 |
|---|---|---|
| `sparse_compute/` | SpMV/SpMM、稀疏归约、扩散、融合 | `weighted_aggregation`, `cpu_relation`, `diffusion_roofline`, `fusion` |
| `graph_operations/` | 拓扑构建/重建与生成的关系 | `radius_build`, `radius_pipeline`, `radius_roofline`, `knn_build` |
| `graph_algorithms/` | 代表性的编译器探针，而非 [NetworkX](https://en.wikipedia.org/wiki/NetworkX) 式的全覆盖 | [PageRank](https://en.wikipedia.org/wiki/PageRank)（循环/收敛）、[BFS](https://en.wikipedia.org/wiki/Breadth-first_search)（前沿）、三角形计数（交集） |
| `neural_networks/` | 具有通用 Tiga 语义的 NN 负载 | `dense_attention`, `linear_attention`, `sparse_attention`, `online_softmax`, `dense_matmul` |
| `compiler/` | 编译/[JIT](https://en.wikipedia.org/wiki/Just-in-time_compilation)/缓存/provider 转译——不是算法结果 | `provider_gate`, `jit_latency`, `tensor_fusion`, `cpu_pointwise` |
| `memory_hierarchy/` | 寄存器/shared/HBM/RAM/NVMe 的放置与流水化 | pinned↔HBM DMA 与 RAM↔NVMe 换出 |
| `distributed/` | 分区/[halo](https://en.wikipedia.org/wiki/Halo_(computer_science))/集合通信/重叠 | 基于 stdlib 与真实 [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 的双进程精确 halo 交换 |
| `large_graphs/` | 十亿边数据集、验证契约、内存分层规划 | `plan`，声明式 `suite.json` |
| `autograd/` | 生成的反向 [kernel](https://en.wikipedia.org/wiki/Compute_kernel) 与框架、手写 backward 的对比 | `message_passing_backward`, `reducer_product_backward`, `dynamic_radius_backward` |
| `visualization/` | 渲染相关的预处理 kernel | `heatmap` |
| `providers/` | provider 一致性，而非性能 | `conformance` |
| `common/` | [roofline](https://en.wikipedia.org/wiki/Roofline_model) 校准、公平性门禁、输出 schema、绘图 | 支撑模块 |
| `kernels/` | 手写性能 oracle | 从不被 `tiga` 导入 |

## 运行用例 { #running-cases }

入口点以模块方式从仓库根目录运行，因此导入与当前工作目录无关。
大多数入口支持 `--quick`（减少采样数量）和 `--fail-on-gate`
（验收门禁不通过时以非零码退出）：

```bash
python -m benchmarks.sparse_compute.weighted_aggregation --quick
python -m benchmarks.sparse_compute.weighted_aggregation --topology lognormal --locality random --features 1
python -m benchmarks.graph_operations.radius_pipeline --quick --fail-on-gate
python -m benchmarks.graph_operations.knn_build --quick --fail-on-gate
python -m benchmarks.neural_networks.dense_attention --quick --fail-on-gate
python -m benchmarks.neural_networks.dense_attention --quick --causal --fail-on-gate
python -m benchmarks.neural_networks.linear_attention --quick --fail-on-gate
python -m benchmarks.neural_networks.sparse_attention --quick --fail-on-gate
python -m benchmarks.neural_networks.radius_edge_mlp --quick
python -m benchmarks.compiler.provider_gate --quick
python -m benchmarks.autograd.message_passing_backward --quick
python -m benchmarks.autograd.dynamic_radius_backward --quick
python -m benchmarks.memory_hierarchy.transfer --quick
python -m benchmarks.distributed.halo_exchange --quick
python -m benchmarks.distributed.automatic_overlap --quick
python -m benchmarks.providers.conformance
python -m benchmarks.large_graphs.plan
python -m benchmarks.common.check_outputs
```

`benchmarks.compiler.provider_gate` 还需要构建好的 `gf-opt` 与
`gf-translate`；它们会从 `build/*/bin/` 自动发现，也可通过
`TIGA_OPT` / `TIGA_TRANSLATE` 显式指定。

`benchmarks.neural_networks.radius_edge_mlp` 还需要
`warp-lang` 包（`pip install warp-lang`）；NVIDIA Warp 是手写
的对标 kernel，绝不是 Tiga 的依赖。

## 产物与公平性规则 { #artifact-and-fairness-rules }

- 每一项性能声明都基于完全相同的语义进行对比，产物保存在
  `output/roofline/<operation>/<case>/` 下——原始样本、
  中位数、[bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) [置信区间](https://baike.baidu.com/item/置信区间)、设备/软件元数据，以及
  算术强度模型。
- 机器可读 JSON 是权威数据，但不是给人看的界面：每个
  实测目录都会生成一份 SVG 可视化（带 PNG 回退），
  正式 roofline 用例还会生成操作级的 `summary.svg`。
- `kernels/` 下的手写 oracle 代码可以设定性能目标，但
  绝不会作为 Tiga 编译器的输出计入报告。
- `evidence_manifest.json` 声明了汇总编译器报告的面板过滤器、
  provider 顺序与基线；若缺少精确匹配的分组或 provider，渲染就会失败，
  因此无法悄悄拿有利的测量结果顶替。
- 静态稀疏生成器覆盖固定、有界均匀、离散幂律、
  连续对数正态与指数等度分布族。连续用例均设置种子、
  按请求的均值重新缩放、显式设置上限，并在每个结果中记录
  min/mean/p50/p95/p99/max、零度占比与变异系数。

[方法论页面](performance.md) 定义了语义匹配、缓存状态、
冷 JIT 开销核算、验收门禁、通用产物布局与复现命令。
