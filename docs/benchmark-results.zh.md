# 基准测试结果 { #benchmark-results }

本页上的每个结果都回答三个问题：计算的是什么问题、对标实现（peer）是
哪一个、结论的适用范围有多窄。页面按设备划分，因为**每种设备的对标
实现集合不同**：GPU 结果对标 cuSPARSE、手写 Triton/Warp [kernel](https://en.wikipedia.org/wiki/Compute_kernel) 以及
官方的 Flash/FLA/FSA [attention](https://en.wikipedia.org/wiki/Attention_(machine_learning)) 实现，而 CPU 结果对标 SciPy、Torch
CPU sparse 以及运行时自身的强制串行调度——两组结果从不混进同一张图。
完整的证据矩阵位于各 case 的产物以及下方的[完整矩阵](#the-full-matrix)中。

!!! note "快照"

    GPU 结果来自仓库的 NVIDIA RTX 5070 Ti 环境；CPU 结果来自与其配对
    的主机 CPU（16 线程）。均为已注册的 case 结果，而非对其他设备
    或形状的外推。原始 JSON/SVG/HTML 产物生成于
    `output/roofline/<operation>/<case>/`，并有意不纳入 Git。

## GPU { #gpu }

在 RTX 5070 Ti 上测得。对标实现：`torch.sparse.mm`（cuSPARSE）、手写
Triton 和 NVIDIA Warp kernel，以及官方 Flash SDPA / FLA / FSA
attention 实现。

=== "稀疏"

    **问题。** 每个 [GCN](https://en.wikipedia.org/wiki/Graph_neural_network)、扩散步和图 kernel 的内层循环：对每个节点的
    入边做一次加权求和。

    $$
    \text{out}[i, f] \;=\; \sum_{e\,=\,(j \to i)} w_e \cdot x[j, f]
    $$

    这里的图是一个 131,072 行、平均度为 16 的 CSR 关系；让这个问题
    变难的是度倾斜——下面的切片覆盖从规则度数到不规则（0–32）、
    幂律、对数正态、指数分布的各种邻域。对标实现是相同 FP32 语义下
    的 `torch.sparse.mm`（cuSPARSE）。

    <object class="gf-chart" type="image/svg+xml" data="/assets/results/sparse-relations.svg" aria-label="各关系族上编译器 tile 相对 torch.sparse.mm 的加速比">
      <img src="/assets/results/sparse-relations.png" alt="各关系族上编译器 tile 相对 torch.sparse.mm 的加速比">
    </object>

    每一行都是同一个公式作用在不同的图切片上——悬停任意一行可查看
    其确切定义。编译器生成的行×邻居×特征 tile 在每一个已注册拓扑上
    都优于稀疏库——在固定度 F16 向量上最高达 **4.32×**，即使在负载
    均衡（而非 tile 划分）成为瓶颈的严重倾斜社交图切片上，也有
    1.2–1.5×。

    已注册 case：[↓ 完整矩阵](#matrix-weighted_aggregation)

    ??? info "基准代码来源 — `sparse_compute/weighted_aggregation.py`"

        ```bash
        python -m benchmarks.sparse_compute.weighted_aggregation --quick
        ```

        ```python
        --8<-- "benchmarks/sparse_compute/weighted_aggregation.py"
        ```

=== "Edge-NN"

    **问题。** 一种 PointNet++ 风格的消息：一个小型 [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) 直接读取边
    本身——逐边计算无法提升（hoist）为节点级预计算，而物化
    `[E, ·]` 消息会让内存随边数爆炸：

    $$
    \text{message}_e \;=\; \mathrm{MLP}\big(\,\text{pos}_j - \text{pos}_i \,\|\, x_j\,\big),
    \qquad
    \text{out}[i] \;=\; \sum_{e\,=\,(j \to i),\; \|\text{pos}_j - \text{pos}_i\| \le r} \text{message}_e
    $$

    唯一的出路是融合 tile kernel，让消息只存在于 tile 内部。
    Tiga 用 `gf.nn.trace` 捕获 MLP 并生成该 kernel；对标实现
    是一个采用相同 tile 划分的手写 Triton oracle、一个手写 NVIDIA
    Warp 标量 kernel，以及 eager 的 gather → MLP → `index_add` 回退
    （MLP 11→16→8，FP32，度 ≈ 31；全部与 eager oracle 交叉核对，
    最大绝对误差 ≤ 5e-5）。

    ![Edge-MLP 在各 provider 下的时延与逐边内存](assets/results/edge-nn-message-passing.svg)

    同一份工作负载，四种实现——provider 名称的含义如下：

    - `tiga-compiled-tile` — 边 UDF 的 MLP 由 `gf.nn.trace`
      捕获，并由 Tiga 编译为融合 tile kernel；
    - `triton-fused-tile` — 采用相同 tile 划分的手写 `@triton.jit`
      oracle，即性能参照；
    - `warp-fused` — 手写 NVIDIA Warp 标量 kernel（哈希网格 +
      kernel 内 MLP）；
    - `tiga-eager` — 同一 UDF 跑在 eager 回退上，也是目前任何
      无法捕获的 UDF 会走的路径。

    编译生成的 kernel 与手写 oracle 的差距在 1.00–1.04× 以内，比
    NVIDIA Warp 基线快 **8.5×**，比 PyTorch eager 基线快 **11.8×**，
    且逐边激活内存为零。

    已注册 case：[↓ 完整矩阵](#matrix-radius_edge_mlp)

    **为什么 Warp 只有 1.4×？** 仅靠融合还不够。Warp kernel 消除了
    所有 O(E) 访存流量，但用标量 FMA 来求 MLP——每条边约 800 条指令，
    约为 GPU fp32 峰值的 1.3%——瓶颈在指令 issue 而非内存带宽，
    优势只随规模增长（16k 粒子时 0.89× → 131k 时 1.30× → 262k
    时 1.39×）。PyTorch eager 浪费内存，但其 cuBLAS GEMM 的指令效率
    高。tile kernel 是把两个问题同时修掉的点——`tl.dot` 恢复了指令
    效率，*而且*没有任何逐边数据离开 tile——11.8× 正是由此而来。

    ![逐边激活内存：eager 对比融合 tile](assets/results/edge-nn-memory.svg)

    **训练也是融合的。** grad 模式下的调用通过 [autograd](https://en.wikipedia.org/wiki/Automatic_differentiation) 桥运行同一
    个前向 tile；反向是一个符号 VJP 重算 tile kernel——逐边激活在
    tile 内部重放，权重梯度以 tile 外积的形式累加，因此两个方向上
    都不存在 [E, ·] 张量：

    $$
    \mathrm{d}W_l \mathrel{+}= a_{l-1}^{\top}\, \mathrm{d}h_l,
    \qquad
    \mathrm{d}a_{l-1} = \mathrm{d}h_l\, W_l,
    \qquad
    \mathrm{d}\,\text{message}_e = \mathrm{d}\,\text{out}[\text{dst}(e)]
    $$

    ![Edge-NN 训练步与峰值内存对比 eager autograd](assets/results/edge-nn-backward.svg)

    在 131k 粒子 / 4.0M 条边上，一次前向+反向步比 eager Torch
    autograd 快 **13.45×**，峰值内存低 **18.4×**（65 MiB 对比
    1.2 GiB）——eager 要为其反向保存 [E, ·] 的 gather、隐藏层激活和
    消息；重算 VJP 只保存 O(N) 的输入。

    已注册 case：[↓ 完整矩阵](#matrix-edge_nn_backward)

    ??? info "基准代码来源 — `autograd/edge_nn_backward.py`"

        ```bash
        python -m benchmarks.autograd.edge_nn_backward --quick
        ```

        ```python
        --8<-- "benchmarks/autograd/edge_nn_backward.py"
        ```

    ??? info "基准代码来源 — `neural_networks/radius_edge_mlp.py`"

        用 `python -m benchmarks.neural_networks.radius_edge_mlp
        --quick` 运行。各实现的核心部分并排对比如下：

        === "Tiga"

            UDF 是普通 Python；`gf.nn.trace` 让 MLP 对编译器可见，
            编译器随即生成融合 tile kernel。

            ```python
            --8<-- "benchmarks/neural_networks/radius_edge_mlp.py:tiga"
            ```

        === "Torch eager（基线）"

            同样的计算改用手写：逐边 gather → 在 `[E, ·]` 上跑 MLP →
            `index_add`。每个中间量都被物化。

            ```python
            --8<-- "benchmarks/neural_networks/radius_edge_mlp.py:torch"
            ```

        === "NVIDIA Warp（基线）"

            手写融合 kernel：哈希网格邻居查询、kernel 内 MLP、逐线程
            累加器——没有任何逐边数据触及内存。

            ```python
            --8<-- "benchmarks/neural_networks/radius_edge_mlp.py:warp"
            ```

        === "Triton tile（oracle）"

            编译器所瞄准的形态：每个程序一个 128 边的块，两层 MLP
            均为 `tl.dot`，带掩码的 atomic 段归约加法。

            ```python
            --8<-- "benchmarks/neural_networks/radius_edge_mlp.py:triton"
            ```

    **GAT 风格 attention：用 `gf.online_softmax()` 做 nn 打分。** 同一
    条边 nn 链也能融合进 attention——只需更换 reducer：

    $$
    s_e \;=\; \mathrm{MLP}\big(\,\text{pos}_j - \text{pos}_i \,\|\, x_j \,\|\, x_i\,\big),
    \qquad
    \text{out}[i] \;=\; \sum_{e\,=\,(j \to i)} \mathrm{softmax}_e(s)\, x_j
    $$

    [Softmax](https://en.wikipedia.org/wiki/Softmax_function) 把同一目标行的边耦合在一起，加法型 atomic tile 表达不了
    这种耦合。Tiga 切换为**行中心** kernel：一个程序负责一条
    CSR 行，按块流式处理该行的边，在寄存器中计算打分，并把
    online softmax 状态 $(m, l, \text{acc})$ 携带在 `scf.for` 的
    iter_args 中——这就是 [FlashAttention](https://en.wikipedia.org/wiki/FlashAttention) 的重缩放，只是搬到了图边
    上。训练同样是融合的：前向只持久化逐行的 $(m, l)$，反向按块
    重放打分链，并在 tile 内应用 softmax [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) 伴随，
    $\mathrm{d}s_e = w_e\,(\langle \mathrm{d}\,\text{out}_i, v_e \rangle -
    \langle \mathrm{d}\,\text{out}_i, \text{out}_i \rangle)$，把梯度
    流式写回字段、位置和权重。

    ![GAT 边 attention 训练步与峰值内存对比 eager autograd](assets/results/gat-attention.svg)

    在 131k 粒子 / 4.0M 条边上，一次前向+反向步比 eager Torch
    autograd 快 **3.6×**，峰值内存低 **35×**（39 MiB 对比
    1.4 GiB——eager 要物化 [E] 打分、[E] 权重和 [E, F] 加权消息；
    融合 kernel 只保留 O(N) 状态）。

    ??? info "基准代码来源 — `neural_networks/gat_attention.py`"

        ```bash
        python -m benchmarks.neural_networks.gat_attention --quick
        ```

        ```python
        --8<-- "benchmarks/neural_networks/gat_attention.py"
        ```

=== "Attention"

    **问题。** 三个工作负载共享编译器机制，但并不共享同一个数学
    问题——因此从不把它们平均成一个"attention 加速比"：

    - 精确稠密 attention，$\;\text{softmax}(QK^{\top}/\sqrt{d} + M)\,V$；
    - 写成未归一化因果递推的线性 attention；
    - 带语义准入阈值的 tile 剪枝稀疏 attention。

    ![精确稠密、线性与 tile 剪枝稀疏 attention 对比各自匹配的对标实现](assets/results/attention.svg)

    每一项都在 B1/H16/N4096/D64 FP16（线性：L64/T512 FP32）下与
    各自匹配的对标实现做了核验：**1.074×** 对比精确 Flash SDPA，
    **1.037×** 对比官方 FLA 递推，**1.182×** 对比使用相同剪枝阈值
    的官方 FSA kernel（与精确 attention 的精度对比单独报告）。

    已注册 case：[↓ dense_attention](#matrix-dense_attention) · [↓ linear_attention](#matrix-linear_attention) · [↓ sparse_attention](#matrix-sparse_attention)

    ??? info "基准代码来源 — `neural_networks/dense_attention.py`"

        ```bash
        python -m benchmarks.neural_networks.dense_attention --quick
        ```

        ```python
        --8<-- "benchmarks/neural_networks/dense_attention.py"
        ```

    ??? info "基准代码来源 — `neural_networks/linear_attention.py`"

        ```bash
        python -m benchmarks.neural_networks.linear_attention --quick
        ```

        ```python
        --8<-- "benchmarks/neural_networks/linear_attention.py"
        ```

    ??? info "基准代码来源 — `neural_networks/sparse_attention.py`"

        ```bash
        python -m benchmarks.neural_networks.sparse_attention --quick
        ```

        ```python
        --8<-- "benchmarks/neural_networks/sparse_attention.py"
        ```

=== "Reducers 与 VJP"

    **问题。** 两个由生成 reducer 完成的工作负载，跑在同一个 CSR
    关系上：一个 online softmax，每行只保留运行中的最大值/分母/
    分子状态，

    $$
    \text{out}[i] \;=\; \frac{\sum_{e=(j \to i)} e^{s_e - m_i}\, v_e}
    {\sum_{e=(j \to i)} e^{s_e - m_i}},
    \qquad m_i = \max_{e=(j \to i)} s_e
    $$

    以及一个乘积 reducer 的反向模式 VJP——两者都以 reducer 代数写成，
    都由编译生成，从不手工推导。

    编译器的稳定元组状态 online softmax 对比匹配的手写 Triton
    kernel 达到 **1.007×**；零安全的生成乘积 VJP 为 **1.021×**；
    半径几何反向（梯度穿过 `edge.distance` 本身）对比匹配的
    Triton 为 **1.079×**，对比 Torch autograd 为 **4.36×**。

    已注册 case：[↓ online_softmax](#matrix-online_softmax) · [↓ message_passing_backward](#matrix-message_passing_backward)

    ??? info "基准代码来源 — `neural_networks/online_softmax.py`"

        ```bash
        python -m benchmarks.neural_networks.online_softmax --quick
        ```

        ```python
        --8<-- "benchmarks/neural_networks/online_softmax.py"
        ```

    ??? info "基准代码来源 — `autograd/message_passing_backward.py`"

        ```bash
        python -m benchmarks.autograd.message_passing_backward --quick
        ```

        ```python
        --8<-- "benchmarks/autograd/message_passing_backward.py"
        ```

=== "半径图"

    **问题。** 关系本身由几何计算得到，并随几何变化——邻居是截断
    半径内的点对，消息按距离加权：

    $$
    E \;=\; \{\, (j \to i) : \|\text{pos}_j - \text{pos}_i\| \le r \,\},
    \qquad
    \text{out}[i] \;=\; \sum_{(j \to i) \in E} \|\text{pos}_j - \text{pos}_i\| \cdot x_j
    $$

    报告把**构建**关系与**使用**关系分开：生成的 cell 目录
    （0.335 ms）替代了物化式 CSR 构建（3.31 ms），一趟全新的从位置
    到输出的 pass（0.740 ms）让几何、距离和归约端到端保持融合——
    在 N=32,768、度 ≈ 32 时，比先物化再 `torch.sparse.mm`
    （3.36 ms）快 4.5×。

    ![半径管线各阶段：构建、使用、复用、全新构建+使用](assets/results/dynamic-boundaries.svg)

    已注册的 24-case 2D/3D 矩阵还覆盖周期盒与斜胞重建，并将最小
    镜像过滤融合在 kernel 内。此结果不外推到自定义无界度量、非
    均匀占据或不同的粒子分布；[动态图策略](dynamic-graphs.md)一页
    解释了编译器何时选择每种实现方式。

    已注册 case：[↓ radius_distance_aggregation](#matrix-radius_distance_aggregation) · [↓ radius_graph_build](#matrix-radius_graph_build) · [↓ knn_graph](#matrix-knn_graph)

    ??? info "基准代码来源"
        === "`graph_operations/radius_pipeline.py`"

            ```bash
            python -m benchmarks.graph_operations.radius_pipeline --quick
            ```

            ```python
            --8<-- "benchmarks/graph_operations/radius_pipeline.py"
            ```

## 编译器层面的汇总 { #across-the-compiler }

同一份已注册 GPU 证据的两个汇总视图。第一，保留关系结构在哪些地方
让编译器得以改变执行计划（而不仅仅是 dispatch 同一个稠密原语）：

![每种编译器变换的已注册加速比](assets/results/transformations.svg)

第二，相对成熟的专用实现，这层抽象的代价有多大——这是覆盖度与
追平程度的证据，而不是使用编译器的理由：

![各原语族与最快匹配对标实现的时延比](assets/results/primitive-parity.svg)

已注册 case：[↓ pagerank](#matrix-pagerank) · [↓ dense_matmul_calibration](#matrix-dense_matmul_calibration) · [↓ horizontal_fusion](#matrix-horizontal_fusion) · [↓ visualization_heatmap](#matrix-visualization_heatmap)

??? info "基准代码来源 — `graph_algorithms/pagerank.py`"

    ```bash
    python -m benchmarks.graph_algorithms.pagerank --quick
    ```

    ```python
    --8<-- "benchmarks/graph_algorithms/pagerank.py"
    ```

??? info "基准代码来源 — `graph_operations/knn_build.py`"

    ```bash
    python -m benchmarks.graph_operations.knn_build --quick
    ```

    ```python
    --8<-- "benchmarks/graph_operations/knn_build.py"
    ```

??? info "基准代码来源 — `neural_networks/dense_matmul.py`"

    ```bash
    python -m benchmarks.neural_networks.dense_matmul --quick
    ```

    ```python
    --8<-- "benchmarks/neural_networks/dense_matmul.py"
    ```

??? info "基准代码来源 — `visualization/heatmap.py`"

    ```bash
    python -m benchmarks.visualization.heatmap --quick
    ```

    ```python
    --8<-- "benchmarks/visualization/heatmap.py"
    ```

## CPU { #cpu }

在配对的主机 CPU（16 线程）上测得。这里的对标实现是另一组：
SciPy 的 `csr_matvec`、CPU 上的 `torch.sparse.mm`，以及运行时自身
的强制串行调度——它们都不出现在上面的 GPU 图表中。

=== "稀疏关系"

    **问题。** 与 GPU 稀疏标签页相同的加权 CSR SpMV——

    $$
    \text{out}[i] \;=\; \sum_{e\,=\,(j \to i)} w_e \cdot x[j]
    $$

    ——但主机侧的基线是另一组：SciPy 的 `csr_matvec` 和 CPU 上的
    `torch.sparse.mm`。编译器将遍历 lowering 为一个融合的 16 线程
    LLVM 循环：关系游走、取权重和归约在单趟内完成，没有物化的消息
    缓冲区。

    ![CPU 融合关系循环对比 SciPy 与 Torch CPU sparse](assets/results/cpu-relations.svg)

    在 N=131,072（度 16，FP32，热缓存）下，生成的循环运行
    0.124 ms——对比 `scipy.csr_matvec` 快 **6.32×**（95% CI 下限
    5.54），对比 Torch CPU sparse 快 **20.66×**。较弱的一端也如实
    给出：在 N=16,384 时，运行时的固定开销主导了 23 µs 的 Torch
    调用（**0.42×**），而 SciPy 仍落后 1.62×。此结论针对大 N 流式
    场景，而非玩具规模下的单次调用时延。

    已注册 case：[↓ cpu_relation](#matrix-cpu_relation) · [↓ cpu_pointwise_fusion](#matrix-cpu_pointwise_fusion)

    ??? info "基准代码来源 — `sparse_compute/cpu_relation.py`"

        ```bash
        python -m benchmarks.sparse_compute.cpu_relation --quick
        ```

        ```python
        --8<-- "benchmarks/sparse_compute/cpu_relation.py"
        ```

=== "分布式"

    **问题。** 跨两个进程的一步消息传递：每个 rank 拥有一半节点，
    需要邻居的 ghost 层，并且应当把 [halo](https://en.wikipedia.org/wiki/Halo_(computer_science)) 交换隐藏在内部计算之后，
    而不是将其串行化。

    ![自动 interior/halo 重叠时间线与端到端对比](assets/results/distributed-overlap.svg)

    两个真实 CPU 进程，走公开的 `Graph.halo()` 路径，65,536 个
    实体，度 16，特征宽度 64，25% 边界，以及显式的 5 ms 接收延迟
    模型：自动的 `interior ∥ halo → boundary` 调度把中位数
    14.10 ms 的通信隐藏了起来，以 38.87 ms 完成，对比强制串行的
    41.47 ms（**1.067×**）。这是受控链路模型下调度器排序的证据——而非
    跨节点或 NCCL 测量。

    ??? info "基准代码来源"
        === "`distributed/automatic_overlap.py`"

            ```bash
            python -m benchmarks.distributed.automatic_overlap --quick
            ```

            ```python
            --8<-- "benchmarks/distributed/automatic_overlap.py"
            ```

## 复现与检查 { #reproduce-and-inspect }

每个正式 case 都会写出原始样本、中位数、bootstrap 置信区间、
设备/软件元数据、算术强度模型、[roofline](https://en.wikipedia.org/wiki/Roofline_model) 上限，以及 SVG 优先、PNG 回退
的图表。[性能方法论](performance.md)定义了通用的产物布局和复现
命令；[基准测试套件](benchmark-suite.md)一页列出了每个入口。

## 完整矩阵 { #the-full-matrix }

本节由 `python -m benchmarks.common.full_matrix` 依据
`benchmarks/evidence_manifest.json` 与 `output/roofline/` 下的测量
产物生成——每一行是一个已注册 gate；没有注册 gate 的 case，每一行
是一个实测 provider，加速比相对最优的非 Tiga 对标实现取得。

--8<-- "docs/includes/full-matrix.zh.md"

## 尚未声明的结论 { #what-is-not-yet-claimed }

- 在真实的 provider 插件与硬件产物出现之前，不声明向 ROCm/Hygon、
  Metal 或 PPU 的性能可移植性；
- 不声明非均匀分布或自定义无界度量下的半径构建 SOTA；
- 不声明 N8192/D3/k32 之外的精确 kNN 性能；
- 不从单 GPU 绑定测试外推多 GPU NCCL 重叠或吞吐；
- 不从单一度分布或特征宽度外推普适的稀疏性能。
