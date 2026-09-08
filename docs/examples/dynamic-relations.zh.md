# 动态与生成式关系 { #dynamic-and-generated-relations }

拓扑由点的位置计算生成，而不是以 [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR_or_CRS)) 形式导入。关系如何物化、如何复用由
provider 决定，用户 UDF 无需改动。

- [`python examples/radius_autograd.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/radius_autograd.py)
- [`python examples/knn_message_passing.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/knn_message_passing.py)

## 可微的半径关系 { #differentiable-radius-relation }

**这是什么。** 半径图由点的位置构建：只要两点的[欧氏距离](https://baike.baidu.com/item/欧几里得距离)不超过截断半径 r，
就在它们之间连一条边，因此拓扑是从数据算出来的，而不是预先存储的。示例在
这张图上做一次消息传递求和：每个节点 i 收集截断半径内每个邻居 j 的值 `x[j]`，
并按两点间的距离加权。然后求输出对位置和取值的梯度，也就是
[向量–雅可比积（VJP）](https://en.wikipedia.org/wiki/Automatic_differentiation)——同一计算的反向 pass。

![半径关系：p0 和 p2 位于 p1 的截断半径 r 内，因此按距离加权的边流入 p1；p0 和 p2 之间的距离大于 r，不连边](../assets/examples/radius-autograd.svg)

$$
\text{out}_i \;=\; \sum_{e\,=\,(j \to i),\; \lVert p_j - p_i \rVert \,\le\, r} \lVert p_j - p_i \rVert \cdot x_j
$$

示例不依赖 Torch：关系基于当前位置快照重建；`edge.distance` 可微，最小镜像
几何支持[周期边界](https://baike.baidu.com/item/周期性边界条件)；`gf.autograd.grad` 直接对半径关系本身求导，在同一张固定的
拓扑快照上生成关于位置和源值的 VJP。

```python
--8<-- "examples/radius_autograd.py:core"
```

??? example "完整源码：examples/radius_autograd.py（可直接运行）"

    ```python
    --8<-- "examples/radius_autograd.py"
    ```

??? info "本机编译产物（Ryzen 7 255 · RTX 5070 Ti 实测）"

    === "CPU"

        ```text
        ### DistanceWeightedSum
        backend: cpu:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: materialize-fixed-radius-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        executable cache: hits=0, misses=1
        ```

    === "CUDA"

        ```text
        ### DistanceWeightedSum
        backend: cuda:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: materialize-fixed-radius-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        executable cache: hits=0, misses=1
        ```

## 精确 kNN 接入 MessagePassing { #exact-knn-feeding-messagepassing }

**这是什么。** [k 近邻](https://baike.baidu.com/item/K近邻算法)（kNN）：给定 N 个点，为每个点找出距离最近的另外
k 个点（这里按平方欧氏距离计算，排除自身）。示例把每个点的 k 个邻居当作
图的边，在其上做消息传递求和：每个点把邻居的值 `x[j]` 乘上边权后累加
（这里的边权全为 1，所以结果就是普通的邻居求和）。

![kNN 关系：高亮一个查询点，来自其三个最近邻的箭头携带加权消息流入它；其余点均被忽略](../assets/examples/knn-message-passing.svg)

$$
\text{out}_i \;=\; \sum_{j\,\in\,\mathrm{knn}(i,\,k)} w_{(j \to i)} \cdot x_j
$$

精确 kNN 仍然是过程式的：当前位置快照按需惰性重建，固定 k 的 CSR ABI
会重新绑定到一个生成的 MessagePassing TTIR 使用方。选取与使用是两个独立的
契约，但会融合成单次启动——不物化任何 CSR 行指针或选中列的张量。

```python
--8<-- "examples/knn_message_passing.py:core"
```

??? example "完整源码：examples/knn_message_passing.py（可直接运行）"

    ```python
    --8<-- "examples/knn_message_passing.py"
    ```

??? info "本机编译产物（Ryzen 7 255 · RTX 5070 Ti 实测）"

    === "CUDA"

        ```text
        ### NeighborSumTorchExecutor
        backend: cuda
        provider: triton-nvidia@3.6.0
        lowering: gf-kernel-to-ttir-ranked-select-consume
        passes: gf-verify-domain, gf-lower-domain-to-iter, gf-lower-iter-to-kernel, capture-ranked-relation, select-ranked-pairs-coordinate-hierarchy, select-candidate-tile, emit-local-stable-topk, emit-hierarchical-topk-merge, fuse-selected-edge-consumer, gf-kernel-to-ttir, provider-compile-serialized-ttir
        provider cache key: triton-nvidia | cuda:nvidia | 3.6.0 | 2.11.0+cu128 | af81e84448f193cc | cuda:120:warp32
        remark: [planning] exact candidate ranking and edge consumption share one launch
        remark: [planning] candidate tiles retain a power-of-two padded stable key state
        remark: [planning] no CSR row pointer or selected column tensor is materialized
        executable cache: hits=0, misses=1
        ```
