# 基准结果 { #benchmark-results }

运行时间与内存对比、大图容量和分布式开销见[性能与扩展性](experiments.zh.md)。本页为历史测量归档。

本页是**历史测量**，不是当前 checkout 的性能结果。存档包含 18 个 operation 的 44 个已注册 case，以及六个辅助数据文件。GPU 数据来自记录中的 RTX 5070 Ti 环境，CPU 数据属于配套主机；不计算跨设备综合分数。

[复现图表](benchmark-reproducibility.zh.md) · [下载原始证据](assets/results/evidence.zip) · [逐图输入索引](assets/results/chart-inputs.json) · [完整矩阵](#the-full-matrix)

!!! warning "范围与来源"

    部分历史文件缺少原始代码 revision、驱动/包版本或逐次采样。存档可以复现图表，但不保证新测量得到相同性能。`--quick` 是 smoke 测量，不是所有存档 case 的精确复现命令。

## 怎样读图 { #reading-the-figures }

新的大图容量测量单列于[十亿边实验](memory.zh.md#billion-edge-capacity)。
它的完整前向计时不能与下方历史 warm operator 计时混用。

加速比大于 1 表示指定候选快于该行对标实现，图中同时保留毫秒数。不同工作负载、不同计时边界的比值不取平均。`F` 是 feature 宽度，不是浮点精度；内存图区分逐边中间量估算和实测峰值分配。小屏幕可横向滚动图表；点击图可打开完整 SVG。

## GPU { #gpu }

=== "Sparse"

    Weighted CSR aggregation 对每个目标行计算加权和。选中的拓扑切片为 131,072 行、平均度 16、FP32 数值；索引宽度以各存档 case 为准。`F=1/16/64` 表示 feature 宽度。

    [![选定稀疏工作负载：prepared Tiga 对比 torch.sparse.mm](assets/results/sparse-relations.svg)](assets/results/sparse-relations.svg)

    候选为 `tiga.prepared_auto`，对标为 `torch.sparse.mm`，cache 为 hot；准备阶段不计入这个 warm 时间。这是固定切片，不代表所有已注册拓扑都获胜。[全部稀疏 case](#matrix-weighted_aggregation)。

=== "Edge-NN"

    边 MLP 读取相对位置和源节点 feature，再按目标聚合。融合与 eager 路径计算同一个选定工作负载；手写 Triton tile 是 oracle，不是编译器生成代码。

    [![边 MLP 前向时延与加速比](assets/results/edge-nn-message-passing.svg)](assets/results/edge-nn-message-passing.svg)

    [![逐边中间量的内存估算](assets/results/edge-nn-memory.svg)](assets/results/edge-nn-memory.svg)

    第二张图的零表示不物化逐边 activation 数组，**不是总设备内存为零**。GiB 来自中间 Tensor 尺寸估算，不是 allocator 峰值测量。

    [![边 MLP 前向加反向与实测峰值内存](assets/results/edge-nn-backward.svg)](assets/results/edge-nn-backward.svg)

    训练图比较完整前向+反向和实测峰值分配，计时口径不同于仅前向。[前向 case](#matrix-radius_edge_mlp) · [反向 case](#matrix-edge_nn_backward) · [程序例子](examples/attention.zh.md)。

=== "GAT"

    这个图 attention case 包含逐边评分、按目标归一化和聚合，不是 dense SDPA。

    [![GAT 训练时延与实测峰值内存](assets/results/gat-attention.svg)](assets/results/gat-attention.svg)

    对标为同一个已记录图上的 eager autograd。这是辅助存档，不在 44 个已注册 case 中；准确输入路径见[逐图索引](assets/results/chart-inputs.json)。

=== "Attention"

    Dense exact、linear、tile-pruned sparse attention 是不同的数学工作负载。每行只与自己的匹配实现比较，跨行不能用于排名。

    [![三种 attention 工作负载及各自对标](assets/results/attention.svg)](assets/results/attention.svg)

    [Dense case](#matrix-dense_attention) · [Linear case](#matrix-linear_attention) · [Sparse case](#matrix-sparse_attention)。完整矩阵给出 shape、精度和 provider 名称。

=== "Dynamic relations"

    Radius 拓扑构建、已有拓扑上的计算、重建再计算，具有不同计时边界。Reuse 排除构建，end-to-end 包含构建。

    [![明确阶段边界的 radius 管线时延](assets/results/dynamic-boundaries.svg)](assets/results/dynamic-boundaries.svg)

    单阶段比值不能证明端到端加速比。[构建 case](#matrix-radius_graph_build) · [消费 case](#matrix-radius_distance_aggregation)。

## 编译器整体 { #across-the-compiler }

以下每行都选择明确的存档工作负载及匹配基线。选择由生成器固定，不会自动扫描最大加速比。

[![选定编译器变换](assets/results/transformations.svg)](assets/results/transformations.svg)

[![专用原语对比](assets/results/primitive-parity.svg)](assets/results/primitive-parity.svg)

基线名称和精确筛选条件见 `benchmarks/evidence_manifest.json` 与 `benchmarks/common/plot_docs_results.py`；条形旁保留候选/对标的绝对时延。[PageRank](#matrix-pagerank) · [kNN](#matrix-knn_graph) · [Matmul 校准](#matrix-dense_matmul_calibration) · [Heatmap](#matrix-visualization_heatmap)。

## CPU { #cpu }

=== "Relations"

    同一个 FP32/i32、度 16 的 relation loop，在两种节点数下对比 SciPy CSR matvec 与 Torch sparse，线程数为 16；同时呈现胜出与落后的切片。

    [![CPU relation 的结果随规模与对标变化](assets/results/cpu-relations.svg)](assets/results/cpu-relations.svg)

    131k case 已注册，16k 对比属于辅助存档。此结果不能证明跨 CPU、线程数或 NUMA placement 的扩展性。[已注册 CPU relation](#matrix-cpu_relation)。

=== "Distributed"

    历史内部 overlap 探测，不代表公开执行策略（固定先通信、再计算）。两个 CPU 进程使用人为增加延迟的传输链路；这不是 NVLink、RDMA 或多 GPU 基准。

    [![一条已记录的 CPU 重叠 trace](assets/results/distributed-overlap.svg)](assets/results/distributed-overlap.svg)

    时间线取某个 rank 的首个 sample，中位数标注来自完整采样集合；不表示每次采样的重叠均相同。[可运行分布式例子](examples/distributed-memory.zh.md#two-process-halo-exchange)。

## 完整矩阵 { #the-full-matrix }

表格从同一批已注册存档输入生成，包含较慢路径。这是详细证据视图，不是另一批独立测量；每个 operation 列出 case 条件、计时范围和对标实现。

--8<-- "docs/includes/full-matrix.zh.md"
