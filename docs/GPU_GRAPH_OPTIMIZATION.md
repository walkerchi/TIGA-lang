# GraphForge GPU Graph 硬件优化调研与实验路线

> 非规范调研与实验候选。实现和发布门槛以仓库根目录 `PROJECT.md` 为准。

状态：M1/M2 设计输入  
日期：2026-08-06

## 1. 结论

GraphForge 不应在第一版设计细粒度用户语言。第一阶段只提供粗粒度
`MessagePassing`，覆盖 `StaticGraph` 与 `DynamicGraph`，建立语义正确、复杂度合理的
naive CPU/GPU lowering。然后依据 profile 实现一个收益明显的硬件优化，最后才从真实
优化中归纳细粒度语言需要暴露的稳定概念。

建议路线：

```text
coarse MessagePassing semantics
  → naive StaticGraph / DynamicGraph
  → profile traversal, feature, atomic, build and memory traffic
  → subgraph/work-tile hardware optimization
  → prove end-to-end speedup and identify reusable compiler primitives
  → design fine-grained graph language from evidence
```

## 2. “一个 subgraph 放到一个 SM”应如何准确表达

普通 CUDA/HIP kernel 不能把某个 thread block 永久指定给某个物理 SM。Block 由硬件
scheduler 动态放置，而且不同 block 之间不能依赖固定执行顺序。可依赖的局部协作边界
是 CTA/thread block/workgroup：同一 block 的线程共享 shared memory/LDS 和 barrier。

因此 portable compiler IR 应表达：

```text
Graph/Relation in HBM
    ↓ partition/bucket/tile
WorkTile = subgraph fragment / row range / edge range / cell neighborhood
    ↓ map
CTA/workgroup
    ↓ stage selected working set
shared/LDS + registers
```

硬件 scheduler 再把 CTA 放到 SM/CU。只有需要 device-side dynamic work queue、跨轮次
locality或极端负载不均衡时，才考虑 persistent CTA workers；这不是 naive baseline。

“把 subgraph 放进 shared memory”也通常不是复制完整子图。Shared/LDS 容量有限，真正
适合 staging 的对象可能是：

- compact local vertex IDs、row offsets 或 edge tile；
- 被多个 edge 重用的 source/destination feature tile；
- particle cell 中的位置、类型和短 feature；
- per-row/per-node partial reducer state；
- layout conversion 或 cooperative-copy buffer。

全图 topology 和 field home instance 通常仍在 HBM。

## 3. 相关工作的直接启发

### 3.1 CUDA execution model

[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
明确指出 block 调度顺序没有保证；block 内线程共享 shared memory/L1/register-file
资源。因此 GraphForge 高层使用 target-neutral `workgroup/subgroup`，不提供
`subgraph.bind(sm_id)` 语义。

### 3.2 Gunrock：分层 load balancing

[Gunrock](https://arxiv.org/abs/1701.01170) 的重要启发是图 workload 需要根据规模和
degree 在 thread、warp、block 和全局层级之间切换，而不是固定“一行一个 warp”或
“一边一个 thread”。首批 schedule family 应至少有 edge、row、split-row 和 bucket。

### 3.3 GNNAdvisor：neighbor × feature 二维 workload

[GNNAdvisor](https://www.usenix.org/system/files/osdi21-wang-yuke.pdf) 根据 graph 与 model
特征驱动二维 workload management，并使用 coarse-grained neighbor partitioning。
这说明 MessagePassing 的性能空间不只有 graph axis；feature width 决定第二个并行轴、
reduction 组织和 shared/register footprint。

### 3.4 FeatGraph：粗粒度 sparse template + 细粒度 payload

[FeatGraph](https://arxiv.org/abs/2008.11359) 联合优化 graph traversal 与 feature
dimension computation。对 GraphForge 的启发是：第一版 coarse MessagePassing 足以
捕获 sparse traversal template，同时 message region 提供 tensor payload；不必先公开
任意细粒度 graph language。

### 3.5 GraphCage：cache-aware subgraph blocking

[GraphCage](https://arxiv.org/abs/1904.02241) 表明普通 dense cache blocking 不能直接
搬到图上；必须把 blocking overhead、subgraph sparsity 和 load balancing 一起考虑。
论文报告 cache-centric blocking 相比已有优化实现可获得明显收益，但适用性依赖图和
迭代阶段。GraphForge 必须允许 cost model 拒绝 subgraph tiling。

### 3.6 TC-GNN：稀疏块到规则 Tensor Core tile

[TC-GNN](https://arxiv.org/abs/2112.02052) 使用 graph translation 把部分稀疏 GNN
计算转成 Tensor Core 可处理的规则块。它验证了“sparse topology + dense tile payload”
路线，但 conversion、padding 和 sparsity pattern 都是成本；不应作为第一版通用
lowering。

### 3.7 Dynamic radius graph：cell list 与 reuse

[LAMMPS neighbor-list internals](https://docs.lammps.org/Developer_par_neigh.html) 使用
spatial bin、neighbor stencil 和 Verlet skin；neighbor list 构建后复用多个 timestep。
这说明 DynamicGraph 的首个正确优化不是任意动态图增量算法，而是让 builder、reuse
condition 和 MessagePassing consumer 在同一 cost model 中。

## 4. 第一版 Naive 实现

“Naive”必须指优化简单，而不是复杂度错误。

### 4.1 StaticGraph

输入：COO/CSR、node/edge Field、粗粒度 Message/Reducer/Update。

CPU baseline：

```text
for dst:
    state = init
    for edge in CSR row(dst):
        state = combine(state, message(edge))
    out[dst] = finalize(state)
```

GPU baseline：

1. edge-centric：一批 edge/thread，atomic reduce；
2. row-centric：一个 program/warp 处理一个或若干 CSR row，局部 reduce；
3. 只做简单安全 dispatch，不做 graph reorder、subgraph packing 或 persistent kernel。

### 4.2 DynamicGraph

首个 DynamicGraph 定义为 geometric radius graph，而不是支持任意 edge mutation：

```python
graph = gf.RadiusGraph(
    positions=position,
    cutoff=cutoff,
    periodic_box=box,
)
```

Reference 可以对小输入使用 all-pairs 以定义语义；performance baseline 必须使用复杂度
合理的 uniform cell list：

```text
bin particles
  → build cell offsets/members
  → enumerate neighbor cells
  → materialize COO/CSR neighbor list
  → reuse StaticGraph MessagePassing kernels
```

第一版不做 generate-consume fusion、Verlet reuse、cell tile shared staging 或动态排序。
这样能独立测量 build 与 consume 成本。

## 5. 第一轮硬件优化候选

### 5.1 Degree-aware hierarchical mapping

```text
small row   → thread/lane
medium row  → subgroup/warp
large row   → CTA or split-row + second reduction
```

优点：实现风险低，几乎所有不均匀图都需要。缺点：主要解决 load balance，不一定改善
随机 feature traffic。

2026-08-14 implementation note：compiler 已能生成 packed degree worklist、typed additive
bucket TTIR 与 `direct-filter + compact high-degree tail`。正式 social-like power-law 矩阵覆盖
local/random、i32/i64、hot/cold；autotune 对 random gather 选择 register-resident chunked
worklist，对 local graph 选择 reusable-output native CSR，八个已登记 bucket 均通过严格门槛。
edge-balanced/merge-path/persistent queue 仍是未登记分布的候选 schedule，不属于当前支持声明。

### 5.2 Static subgraph/work-tile blocking

将一段 destination rows 与其 source frontier 形成 tile：

```text
row range
  → collect/compact source IDs
  → stage reused source features into shared/LDS or rely on L2 tile
  → process local edge stream
  → reduce/store destination tile
```

收益来自 source feature reuse 和更规则的 feature-axis computation；成本包括 partition、
compact metadata、boundary duplication、shared capacity、barrier 和 occupancy。StaticGraph
可跨多次 MessagePassing amortize preprocessing，因此是 subgraph optimization 的首选。

### 5.3 Neighbor × feature 2D tiling

对 feature width 16–256 的 workload，同时 tile neighbor 和 feature：

```text
CTA.x → destination/neighbor partition
CTA.y or subgroup/lane → feature tile
```

它通常比“完整 subgraph 放进 shared”更容易先实现，也能直接控制 coalescing、reducer
register pressure 和 occupancy。

### 5.4 Dynamic cell-tile staging/fusion

一个 CTA 处理一个或多个 spatial cells，把本 cell/neighbor cell 的 position/short
feature tile 分批搬入 shared，然后计算 pair message。后续可选择：

- materialize neighbor list；
- generate-consume fused，不写 edge list；
- 使用 skin/rebuild condition 跨 timestep 复用。

该方案适合 feature 很短的粒子仿真；与 StaticGraph subgraph packing 的成本结构不同。

### 5.5 Persistent CTA queue

固定数量 CTA 从 device queue 获取 degree bucket、row chunk 或 cell tile，可以改善极端
skew 和动态图负载均衡，也可复用部分 worker state。代价是 queue atomic、退出协议、
occupancy 限制和更复杂的跨 backend lowering。只在普通 overdecomposition 仍有明显
tail 时实现。

### 5.6 Sparse-to-MMA translation

只针对高 feature width、适合 semiring/linear message、局部密度足够的 tile。必须把
translation、padding 和 preprocessing amortization 纳入 end-to-end 时间。它属于第二
轮优化，不作为 MessagePassing 通用保证。

## 6. 建议选择的“明显效果”优化实验

先保留两个候选，由 naive profile 决定只实现其中一个完整 vertical slice：

### 候选 A：StaticGraph feature aggregation

Workload：`sum(weight * x[src])`，feature width `16/32/64/128`，同一 topology 重用多次。

比较：

```text
edge atomic
CSR row
degree bucket + split-row
neighbor × feature 2D tile
subgraph tile + compact source-feature cache
```

这个候选最直接验证 subgraph/CTA、dense payload tile 和 preprocessing amortization。

### 候选 B：Dynamic radius interaction

Workload：3D positions、短 feature、cutoff interaction，多 timestep。

比较：

```text
cell-list build + materialized CSR + consume
cell tile shared staging
generate-consume fusion
Verlet skin + rebuild/reuse
```

这个候选最直接验证 DynamicGraph，以及 builder 与 consumer 的联合优化。

选择标准不是单 kernel 峰值，而是：

- naive profile 中该瓶颈至少占 end-to-end 时间的 30%；
- 优化在至少两类输入分布上达到 `>=1.5x` end-to-end speedup；
- preprocessing/build/copy/padding 全部计时；
- 不适用输入能自动 fallback，回退不超过 10%；
- CPU、GPU reference correctness 与空/极端 degree case 全部通过。

`1.5x` 是初始工程门槛，可在获得本机数据后调整。

## 7. 从优化反推细粒度语言

只有实现并验证上述优化后，才决定细粒度语言是否需要暴露：

```text
neighbor/feature dependent axes
work tile / degree bucket
reducer partial state
materialize vs generate-consume
reuse/rebuild condition
cooperative staging intent
```

即使需要，也优先暴露语义和合法性信息，不暴露 `SM id`、具体 shared-memory 字节布局
或 CUDA-only primitive。没有被两个以上真实优化共同需要的概念，先保留为内部 IR，
不进入 public API。
