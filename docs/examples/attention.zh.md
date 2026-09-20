# 注意力与稠密关系 { #attention-and-dense-relations }

前四个示例需要 NVIDIA GPU 和 `cuda` extra；GAT 根据设备可用性选择 CPU/CUDA。
核心片段不包含全部初始化，运行命令对应完整脚本。Torch 需在 Tiga 之前单独安装。

用通用的关系（relation）与 reducer 语义来表达[注意力](https://baike.baidu.com/item/注意力机制)类工作负载。
掩码（mask）从来不是代码——它就是图本身：稠密关系是全量注意力，三角关系是
因果注意力，而块对角 [CSR](https://en.wikipedia.org/wiki/Sparse_matrix) 则是带 `cu_seqlens` 的 varlen 因果注意力。tile 剪枝
是叠加在同一个 online [softmax](https://en.wikipedia.org/wiki/Softmax_function) reducer 之上的另一层独立的近似语义。

- [`python examples/full_attention.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/full_attention.py)
- [`python examples/causal_dense_relation.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/causal_dense_relation.py)
- [`python examples/varlen_causal_attention.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/varlen_causal_attention.py)
- [`python examples/tile_pruned_attention.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/tile_pruned_attention.py)
- [`python examples/gat_edge_attention.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/gat_edge_attention.py)

<span id="dense-and-causal"></span>

## 全量注意力，无掩码 { #full-attention-no-mask }

**它是什么。** 注意力：N 个位置中每个位置都携带一个 query 向量、一个 key 向量
和一个宽度为 D 的 value 向量。每个输出位置是所有 value 向量的加权平均，权重
是 query·key 相似度分数上的 softmax——因此代价是完整的 N×N 分数矩阵。没有
掩码时，每个 query 都会关注每个 key。用图的术语来说，每条边 (j→i) 把目的位置
i（query）与源位置 j（key 和 value）配对；稠密关系包含全部 N×N 条边。

![8×8 attention 矩阵，所有格子填满：行是 query 位置、列是 key 位置，填充格代表边 j→i——全量注意力拥有全部 N² 条边，没有任何掩码。](../assets/examples/full-attention.svg)

$$
s_{hij} = \langle k_{hj},\, q_{hi}\rangle \cdot D^{-1/2}, \qquad
\mathrm{out}_{hi} = \sum_{j} \frac{e^{s_{hij}}}{\sum_{l} e^{s_{hil}}}\, v_{hj}
$$

这是下面所有内容的精确特例——与 FlashAttention 相同的分块 online softmax
机制，只是每个 tile 都被接纳。

```python
--8<-- "examples/full_attention.py:core"
```

??? example "完整源码：examples/full_attention.py（可直接运行）"

    ```python
    --8<-- "examples/full_attention.py"
    ```

??? info "本机编译产物（Ryzen 7 255 · RTX 5070 Ti 实测）"

    === "CUDA"

        ```text
        ### FullAttentionTorchExecutor
        backend: cuda
        provider: triton-nvidia@3.6.0
        lowering: gf-kernel-to-ttir-dense-streaming-reduction
        passes: gf-verify-domain, gf-lower-domain-to-iter, gf-lower-iter-to-kernel, capture-structured-message, expand-reducer-regions, select-cartesian-coordinate-hierarchy, select-dense-query-neighbor-tile, lower-vector-contraction, lower-streaming-reducer-state, gf-kernel-to-ttir, provider-compile-serialized-ttir
        provider cache key: triton-nvidia | cuda:nvidia | 3.6.0 | 2.11.0+cu128 | af81e84448f193cc | cuda:120:warp32
        accepted: [machine-schedule] admitted dense-query-key-tile
        unknown: [pipeline] no compiler-controlled asynchronous pipeline admitted
        remark: [planning] Cartesian relation remained implicit
        remark: [planning] multi-value message and reducer state were fused into tiles
        remark: [planning] no pairwise score or normalized-weight tensor was materialized
        variant cache: hits=0, misses=1
        ```

## 因果稠密关系 { #causal-dense-relation }

**它是什么。** 因果注意力是带“禁止向前看”约束的注意力：位置 i 只能对
j ≤ i 的位置求平均。这是从左到右生成文本的[语言模型](https://baike.baidu.com/item/语言模型)的规则，其中每个位置
只能读取过去。该示例把掩码表达为图本身：三角关系恰好包含所有满足 j ≤ i
的边 (j→i)，因此边函数完全不需要任何掩码代码。同上，源 j 提供 key 和
value，目的 i 提供 query。

![同一个 attention 矩阵，只有下三角（j ≤ i）被填充：上三角的格子不是被掩码挡掉的分数，而是根本不存在的边。](../assets/examples/causal-dense-relation.svg)

$$
s_{hij} = \langle k_{hj},\, q_{hi}\rangle \cdot D^{-1/2} \quad (j \le i), \qquad
\mathrm{out}_{hi} = \sum_{j \le i} \frac{e^{s_{hij}}}{\sum_{l \le i} e^{s_{hil}}}\, v_{hj}
$$

`Graph.triangular()` 把下三角（含对角）拓扑一路携带穿过
Domain → Iter → Kernel，普通的 online reducer 经 lowering 得到因果流式 TTIR。
Torch 只作为外部 [CUDA](https://en.wikipedia.org/wiki/CUDA) 存储出现。

```python
--8<-- "examples/causal_dense_relation.py:core"
```

??? example "完整源码：examples/causal_dense_relation.py（可直接运行）"

    ```python
    --8<-- "examples/causal_dense_relation.py"
    ```

??? info "本机编译产物（Ryzen 7 255 · RTX 5070 Ti 实测）"

    === "CUDA"

        ```text
        ### CausalWeightedValueTorchExecutor
        backend: cuda
        provider: triton-nvidia@3.6.0
        lowering: gf-kernel-to-ttir-dense-streaming-reduction
        passes: gf-verify-domain, gf-lower-domain-to-iter, gf-lower-iter-to-kernel, capture-structured-message, expand-reducer-regions, select-cartesian-coordinate-hierarchy, select-dense-query-neighbor-tile, lower-vector-contraction, lower-streaming-reducer-state, gf-kernel-to-ttir, provider-compile-serialized-ttir
        provider cache key: triton-nvidia | cuda:nvidia | 3.6.0 | 2.11.0+cu128 | af81e84448f193cc | cuda:120:warp32
        accepted: [machine-schedule] admitted dense-query-key-tile
        unknown: [pipeline] no compiler-controlled asynchronous pipeline admitted
        remark: [planning] Cartesian relation remained implicit
        remark: [planning] multi-value message and reducer state were fused into tiles
        remark: [planning] no pairwise score or normalized-weight tensor was materialized
        variant cache: hits=0, misses=1
        ```

## 用图组合表达 varlen 因果注意力 { #varlen-causal-attention-with-cu_seqlens }

**它是什么。** 真实的训练批次会把许多长度各异的序列打包进一个扁平[张量](https://baike.baidu.com/item/张量)，
并用累积长度 `cu_seqlens = [0, L₁, L₁+L₂, …]` 标记边界（即 flash-attn 的
varlen API 约定）。每个序列**仅在自身内部**做因果注意力——位置 i 只有当
j 与 i 位于同一序列且 j ≤ i 时才能读取位置 j。作为掩码，它是一个由小三角
组成的块对角矩阵；作为图，它是一个恰好包含这些边的显式 CSR 关系，共
Σ L(L+1)/2 条。边界处理无需任何代码：跨序列注意力就是一条不存在的边。

![cu_seqlens = [0, 3, 5] 对应的 5×5 attention 矩阵：块对角上的两个因果小三角——位置 0–2 在序列 0 内部互相注意，位置 3–4 在序列 1 内部；所有跨序列的格子都没有边。](../assets/examples/varlen-causal-attention.svg)

$$
\mathrm{out}_{hi} = \sum_{\substack{j \le i \\ \mathrm{seq}(j)=\mathrm{seq}(i)}}
\frac{e^{s_{hij}}}{\sum_{\substack{l \le i \\ \mathrm{seq}(l)=\mathrm{seq}(i)}} e^{s_{hil}}}\, v_{hj},
\qquad
\mathrm{seq}(i) = b \;\Leftrightarrow\; \mathrm{cu}_b \le i < \mathrm{cu}_{b+1}
$$

该示例用 `tg.Graph.cat` 组合每个序列的三角图，运行同一个无掩码 edge UDF，
包括长度为 1 的序列。块 `k` 只在内部互相注意。当前 `cat` 会物化 CSR，
尚非隐式组合关系。兼容构造器 `tg.Graph.cu_seqlens(boundaries, causal=True)`
在累计边界以零开头时产生相同关系。（测试会将全部三个注意力示例与
`F.scaled_dot_product_attention` 的结果逐一比对。）目前执行会回退到精确的
eager oracle：编译后的流式路径覆盖上面的稠密/三角关系以及基于字段分数的 CSR
段 softmax，而在通用 CSR 上由边计算分数属于编译器边界，而非语义边界。

```python
--8<-- "examples/varlen_causal_attention.py:core"
```

??? example "完整源码：examples/varlen_causal_attention.py（可直接运行）"

    ```python
    --8<-- "examples/varlen_causal_attention.py"
    ```

??? info "本机编译产物（Ryzen 7 255 · RTX 5070 Ti 实测）"

    === "CUDA"

        ```text
        ### VarlenCausalAttentionTorchExecutor
        backend: reference
        provider: torch
        lowering: (unspecified)
        passes: validate-domain, select-reference-csr
        remark: [planning] reference evaluator selected; no performance codegen artifact exists
        remark: [planning] cross-apply fusion is handled by the @tg.jit/@tg.program capture boundary rather than the single-apply reference evaluator
        variant cache: hits=0, misses=1
        ```

<span id="sparse-and-pruned"></span>

## 基于 tile 剪枝的稀疏注意力 { #tile-pruned-sparse-attention }

### 为什么要跳过一块 { #why-prune-a-tile }

这是一个**用精度换取潜在性能的实验性选项**，不是使用 attention 的必需步骤。
普通 attention 对每个允许的 key 计算 score，再用 softmax 权重累加 value。
当某一整块 key 的 score 都很低时，继续加载该块 value 并累加，可能贡献很小。
tile 是一小块 query 与一小块 key 的配对；批量跳过是为了适合 GPU 的分块执行。

例如，已见最高 score 为 10，下一块最高 score 只有 1，两者指数权重的比例约为：

$$
\exp(1 - 10) \approx 0.00012
$$

若 `block_prune_threshold=0.01`，这块会被跳过。这个比较针对整个 query block，
不是每个 query 各自的误差保证；很多低分项相加，或 value 很大时，仍可能造成明显误差。

| 仍然执行 | 被剪 tile 跳过 |
|---|---|
| 读取 key、计算 query/key score、判断阈值 | 读取 value、后续 softmax 与 value 累加 |

因此它不减少全部 score 计算，也不保证加速。阈值越大越激进；
精确计算应省略 `block_prune_threshold`，不是把它设为 0。
当前仅支持指定 generated-CUDA dense 前向路径，不支持 backward。
示例中的 `width / nodes` 只是演示参数，不是通用调参公式。

### 具体规则与边界 { #tile-admission-rule }

**它是什么。** 与第一个示例相同的注意力，但是近似的：tile 剪枝把 key 按块
（tile）处理，跳过最高分数远低于当前 query block 最大值的 tile。阈值本身
不保证被省略的总概率或输出误差上界；这些还取决于被剪 tile 的数量与 value
的幅度，需要按工作负载验证精度。FlashAttention 正是
每个 tile 都被接纳的精确特例；本示例添加了接纳规则。

![8×8 attention 矩阵按 2×2 tile 分组：靠近对角线的 tile 被接纳（实心靛蓝），其余被剪枝（打叉）——先计算 score，再剪枝；被剪 tile 的 value 不会被加载。](../assets/examples/tile-pruned-attention.svg)

图中对角附近被保留仅是示意，不是固定局部窗口；实际保留哪些块由数据的 score 决定。

$$
s_{hij} = \langle k_{hj},\, q_{hi}\rangle \cdot D^{-1/2}, \qquad
\mathrm{out}_{hi} = \sum_{j} \frac{e^{s_{hij}}}{\sum_{l} e^{s_{hil}}}\, v_{hj}
$$

emitter 将整个 query/key tile 的最大 score 与当前 query block 的滚动
最大值比较。以自然对数 score 为单位，准入规则使用比例
`rho = block_prune_threshold`：

$$
m_{\mathrm{tile}} \ge m_{\mathrm{run,block}} + \ln(\rho),
\qquad 0 < \rho \le 1
$$

TTIR 等价地使用以 2 为底的 score 和 `log2(rho)`。key 加载和 score 计算发生
在准入判断之前；剪枝跳过的是 value 加载及后续 softmax/value 累加。


阈值的工作机制：这是内置 `tg.online_softmax()` reducer 的一个**构造参数，
由编译器按结构识别**——不是 subclass reducer 的扩展机制。阈值随 reducer 的
特化键进入 `gf.reducer` IR，成为一个可选属性；只有 generated-CUDA dense
streaming provider 会消费该属性，在 tile 主循环里生成一个动态的 `scf.if`，把
低分 tile 的载荷整个跳过。其余所有路径——eager oracle、原生 CSR、edge-nn
attention、以及全部反向 pass——都 fail closed，而不是静默退回精确 softmax，
因此这个近似永远是显式选择。

subclass `tg.Reducer` 无法复刻这个机制：structured-dense lowering 只接受内置
online-softmax 结构，tile 接纳逻辑写在 TTIR emitter 里，而不是从用户代数
推导出来的。subclass `OnlineSoftmaxReducer` 再设置 `block_prune_threshold`
是可行的——但那用的还是同一个内置参数，不是新的扩展点。

```python
--8<-- "examples/tile_pruned_attention.py:core"
```

??? example "完整源码：examples/tile_pruned_attention.py（可直接运行）"

    ```python
    --8<-- "examples/tile_pruned_attention.py"
    ```

??? info "本机编译产物（Ryzen 7 255 · RTX 5070 Ti 实测）"

    === "CUDA"

        ```text
        ### TilePrunedAttentionTorchExecutor
        backend: cuda
        provider: triton-nvidia@3.6.0
        lowering: gf-kernel-to-ttir-dense-streaming-reduction
        passes: gf-verify-domain, gf-lower-domain-to-iter, gf-lower-iter-to-kernel, capture-structured-message, expand-reducer-regions, select-cartesian-coordinate-hierarchy, select-dense-query-neighbor-tile, lower-vector-contraction, lower-streaming-reducer-state, gf-kernel-to-ttir, provider-compile-serialized-ttir
        provider cache key: triton-nvidia | cuda:nvidia | 3.6.0 | 2.11.0+cu128 | af81e84448f193cc | cuda:120:warp32
        accepted: [machine-schedule] admitted dense-query-key-tile
        unknown: [pipeline] no compiler-controlled asynchronous pipeline admitted
        remark: [planning] Cartesian relation remained implicit
        remark: [planning] multi-value message and reducer state were fused into tiles
        remark: [planning] no pairwise score or normalized-weight tensor was materialized
        variant cache: hits=0, misses=1
        ```

<span id="graph-attention"></span>

## 用 nn 打分的 GAT 边注意力 { #gat-edge-attention-with-an-nn-score }

**它是什么。** [图注意力（GAT）](https://baike.baidu.com/item/图注意力网络)：不用点积，每条边的注意力*分数*由一个小型
[神经网络](https://baike.baidu.com/item/人工神经网络)产生，该网络读取边几何信息和两个端点的特征；随后对入边的 softmax
像之前一样对源特征加权。可学习的参数就集中在这个分数网络中。

$$
s_e = \mathrm{MLP}\big(\,\text{pos}_j - \text{pos}_i \,\|\, x_j \,\|\, x_i\,\big),
\qquad
\mathrm{out}_i = \sum_{e\,=\,(j \to i)}
\frac{e^{s_e}}{\sum_{e'=(j' \to i)} e^{s_{e'}}}\, x_j
$$

UDF 沿用第一个示例的结构，只有一处变化——`tg.nn.trace(attn)` 产生分数。
在 CUDA 上编译为一个以行为中心的融合 [kernel](https://en.wikipedia.org/wiki/Compute_kernel)：每个目的行一个程序，分数 MLP
在寄存器中按边块求值，online softmax 状态保存在循环寄存器中——不存在任何
[E, ·] 形状的分数或注意力权重张量。梯度模式的调用走融合的反向路径（分数链
重放 + tile 内 softmax Jacobian 伴随），因此训练时同样不会按边物化任何数据。
launch geometry 采用注意力示例的默认值（`block_e=16, num_warps=1`）；
`tg.nn.trace` 仍接受显式覆盖值。

```python
--8<-- "examples/gat_edge_attention.py:core"
```

??? example "完整源码：examples/gat_edge_attention.py（可直接运行）"

    ```python
    --8<-- "examples/gat_edge_attention.py"
    ```

??? info "本机编译产物（Ryzen 7 255 · RTX 5070 Ti 实测）"

    === "CUDA"

        ```text
        ### GATTorchExecutor
        backend: cuda
        provider: triton-nvidia@3.6.0
        lowering: gf-python-emit-edge-nn-attention-vjp
        passes: capture-edge-nn-subgraph, translate-fx-to-message-dag, prove-edge-nn-attention-structure, python-emit-edge-nn-attention-ttir, symbolic-edge-nn-attention-vjp, python-emit-edge-nn-attention-vjp-ttir, provider-compile-serialized-ttir
        provider cache key: triton-nvidia | cuda:nvidia | 3.6.0 | 2.11.0+cu128 | af81e84448f193cc | cuda:120:warp32
        remark: [planning] training path: forward persists only the O(N) softmax state (m, l); backward replays the score chain per row chunk — no [E, ·] tensor in either direction
        variant cache: hits=0, misses=2
        ```
