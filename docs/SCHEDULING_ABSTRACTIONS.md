# GraphForge 调度抽象调研：从 Tensor Axis 到多空间映射

> 非规范调研记录。已经采纳的调度契约只在仓库根目录 `PROJECT.md` 生效。

状态：设计建议  
日期：2026-08-06

## 1. 结论

可以直接以 tensor/iteration axis 为调度入口，而且规则 tensor、stencil、FFT stage、
contraction 的大部分局部优化都应这样表达。但“把每个 axis 放到某个硬件或存储层级”
不够成为 GraphForge 的完整调度抽象。

原因是高性能实现同时包含六类相互关联、但语义不同的决策：

```text
Schedule
  = IterationTransform   迭代域如何 split/fuse/reorder/segment
  + AccessLayout         逻辑坐标如何访问并布局每个 tensor/field
  + ExecutionMapping     迭代点由 device/block/warp/lane/vector 谁执行
  + StoragePlan          哪个 region/tile 的哪个版本驻留在哪个 memory space
  + PipelinePlan         producer/copy/compute/store 何时执行，如何同步和复用 buffer
  + TaskPlacement        partition、halo、collective 和跨设备 task 如何放置
```

这不是用户必须手写的六组 annotation，而是 compiler 的内部决策空间和
`explain()` 输出结构。M0–M2 用户输入是 coarse MessagePassing；compiler 将其 lower
成仍保留 iteration/relation/reduction 结构的内部 IR，再构造这些映射。

如果把这六件事压成一个 `axis -> level` 表，规则 GEMM 的简单例子看起来很漂亮，遇到
CSR、online softmax、异步 copy、同一迭代域上的多种 tensor layout 或 distributed halo
时就会产生歧义。更合适的统一方式是：保留多个有类型的空间，并显式描述它们之间的
映射。

## 2. 为什么单一 axis hierarchy 不够

### 2.1 迭代轴不等于 tensor 轴

矩阵乘的逻辑迭代域是 `(m, n, k)`，但 A、B、C 的访问分别是 `(m, k)`、`(k, n)`、
`(m, n)`。一个迭代轴可能不存在于某个 tensor 中，也可能通过 affine、gather 或
relation map 访问多个 tensor 轴。MLIR Linalg 因而把 `iterator_types` 与每个 operand
的 `indexing_maps` 分开，而不是给 tensor dimension 直接绑定 thread。

GraphForge 也应区分：

- `IterAxis`：计算实例的逻辑坐标；
- `AccessMap`：迭代坐标到 Field/Relation 坐标；
- `Layout`：Field 坐标到物理地址、lane 或 fragment 坐标。

### 2.2 sparse/ragged 不是矩形 axis

CSR 的邻居轴是依赖父坐标的：

```text
dst in [0, N)
neighbor in [rowptr[dst], rowptr[dst + 1])
```

它的 extent 随 `dst` 改变。radius relation 甚至需要运行时生成这个集合。普通
`tensor<N, max_degree>` axis 只能通过 padding/mask 表达，会隐藏有效工作量和
degree skew。

因此需要 `segmented/dependent/generated` axis，以及从逻辑 dimension 到物理
coordinate level 的映射。TACO 与 MLIR SparseTensor 的 dense/compressed/singleton
level、Finch 的 looplet，以及 CoRa 的 ragged dimension 都说明：稀疏结构是
coordinate hierarchy，不只是 stride 不同的 dense tensor。

### 2.3 execution mapping 不等于 memory placement

`dst_outer -> block` 描述由哪个 program instance 执行；`Q_tile -> shared` 描述数据
驻留；`feature_inner -> lane` 可能同时决定协作加载和寄存器 fragment layout。这些
约束有关联，但不是同一种关系。

特别是一个 tile 可以先在 HBM，异步复制到 shared，再由各 lane 按不同 fragment
layout 读入 register。它不是被唯一地“放在 shared axis”上。

### 2.4 pipeline 不是空间轴映射

double buffering 至少还要描述：

- 哪个 producer/copy 与哪个 consumer/compute 构成 stage；
- `k+1` 的 copy 与 `k` 的 compute 是否可 overlap；
- buffer 数量、phase、barrier/token 和复用条件；
- fill/drain、尾块 predicate 和容量约束。

TileLang 的 `Pipelined`、MLIR NVGPU 的 async-copy token/group/wait 都把这些作为时序和
依赖，而不是 layout。FlashAttention 的性能也来自 IO-aware tiling、online reducer
和数据搬运时序的联合，而非只选择 thread tile。

### 2.5 存储和机器拓扑不总是一棵树

register/shared/HBM 在一个 kernel 内近似层次结构；HBM、CPU RAM、NVMe、peer HBM、
remote RAM 之间则是带多条 transfer path、不同 engine、并发能力和一致性范围的图。
一个只读 Region 还可能同时有多个 replica。因此长期 storage placement 应映射到
`MemoryTopology`，不能只用整数 level。

### 2.6 distributed tensor sharding 很强，但仍不是完整 distributed schedule

GSPMD 和 MLIR Shard 证明 `tensor axis -> device mesh axis` 是非常好的分片标注：它能
统一 data/model/spatial parallel，并由编译器传播和插入 collective。GraphForge 应
直接学习这一层。

但不规则 relation 还需要 graph/mesh partition、owned/ghost region、halo packing、
负载平衡和迁移；communication/computation overlap 又需要 task/event DAG。因此
sharding 是 `TaskPlacement` 的重要子集，不是完整 runtime schedule。

## 3. 相关系统带来的具体启发

| 系统 | 核心抽象 | 对 GraphForge 的启发 | 不能直接覆盖的部分 |
|---|---|---|---|
| Halide | algorithm/schedule 分离；split/reorder/compute_at/store_at | axis transform 之外还要有 producer-consumer placement 和 storage lifetime | irregular coordinate、distributed |
| TVM TensorIR | block、loop、buffer region、cache、tensorize | schedule 作用于 iteration/block，访问 region 单独分析 | relation provenance、dynamic graph runtime |
| MLIR Linalg/Transform | iterator types + operand indexing maps；独立 transform IR | 分开 iteration/access；专家与自动调度走同一种 transform | sparse/dynamic relation 需扩展 |
| CuTe | hierarchical shape/stride 与 layout composition/divide | 用可组合 layout 映射数据和 thread fragment，避免枚举 layout 名称 | 主要面向 dense、单 kernel、NVIDIA |
| TileLang | tile instruction、fragment layout、显式 memory scope、software pipeline | `gf.kernel` 需要 tile、copy、pipeline 和 reducer 同时可见 | Domain/Relation/Task 不是其目标 |
| TACO / MLIR SparseTensor | per-level format、coordinate hierarchy、coiteration | logical dimension 与 storage level 分开；复用成熟 sparse theory | geometric builder、task runtime |
| Finch | looplet 和逐步 lowering 的 structured iteration | generated/run/sequence/control-flow 结构不能过早拍平 | GPU mapping/pipeline 仍需下层 IR |
| CoRa | ragged dimension 与 dimension graph、最少 padding | dependent extent 是一等信息；schedule 需支持 ragged fusion | 通用 sparse format 与 distributed |
| GraphIt | graph iteration space；direction、segment、parallel、layout schedule | traversal direction、degree segmentation 是 axis transform 之外的一等决策 | 连续 tensor tile、memory pipeline |
| DaCe | dataflow、memlet、storage location 与 stateful graph | 跨 kernel 数据移动和 lifetime 需要 data/task flow | 不替代 relation/domain IR |
| GSPMD / MLIR Shard | tensor dimension 到 device mesh axis 的 sharding | 规则 distributed tensor 采用轴分片和传播 | irregular partition、halo overlap |

主要资料：

- [MLIR Linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [MLIR Transform tutorial](https://mlir.llvm.org/docs/Tutorials/transform/)
- [TVM TensorIR](https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html)
- [CuTe Layout Algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)
- [TileLang instructions](https://tilelang.com/programming_guides/instructions.html) 与
  [software pipeline](https://www.tilelang.com/programming_guides/software_pipeline.html)
- [MLIR SparseTensor dialect](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/)
- [Finch paper](https://arxiv.org/abs/2404.16730)
- [CoRa paper](https://proceedings.mlsys.org/paper_files/paper/2022/file/afe8a4577080504b8bec07bbe4b2b9cc-Paper.pdf)
- [GraphIt paper](https://arxiv.org/abs/1805.00923)
- [DaCe SDFG paper](https://arxiv.org/abs/1902.10345)
- [GSPMD paper](https://arxiv.org/abs/2105.04663) 与
  [MLIR Shard dialect](https://mlir.llvm.org/docs/Dialects/Shard/)
- [MLIR GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/) 与
  [NVGPU dialect](https://mlir.llvm.org/docs/Dialects/NVGPU/)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)

## 4. 建议的统一模型

### 4.0 先统一 coarse MessagePassing 的内部 IR，不先设计细粒度语言

M0–M2 用户只写 coarse `MessagePassing`，其中 edge/optional-node region 允许 tensor 和
标量表达式，但 node/edge/neighbor traversal 不通过 public loop syntax 暴露：

```python
class Diffusion(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * (src.u - dst.u)
```

Compiler 将 Graph/Relation、Message、Reducer 和 Effect lower 成内部细粒度 SSA/
Iteration IR，同一份内部 IR 再按 target lower：

```text
                         ┌→ CPU: scf/affine loops + vector + parallel runtime
gf.domain → gf.iter ─────┤
                         └→ GPU: tile/map + working-set promotion + pipeline
                                  → gf.kernel → CUDA/HIP/PPU provider
```

GPU 路径通过 reuse/liveness、协作访问、tile footprint、layout conversion、memory
capacity 和 occupancy 推导 register/shared/LDS；随后插入 cooperative copy、barrier、
double buffer 和 async pipeline。CPU 路径不模拟 shared memory，同一逻辑 temporary
lower 成 SSA/vector、stack allocation 或 cache-blocked loop。

“GPU block-tensor/dense lowering”指把有效 relation item 上的 feature/message/reducer
计算变成规则 tile、vector、dot/MMA 和 reduction；它不要求把 sparse graph 变成
`N×N` dense adjacency。CSR coordinate stream 可以保持稀疏，degree bucket 内部则可按
cost model 选择小范围 padding/masking，以规则计算换取更高硬件利用率。

这里“细粒度”首先是 compiler IR 的性质，不是 public language commitment。只有 naive
Static/Dynamic Graph 和至少一项硬件优化完成后，才从被验证的 dependent axis、work
tile、partial reducer、materialize/reuse 等需求反推用户语法。

### 4.1 IterationDomain：允许矩形、依赖和生成轴

轴必须有稳定名字和类型，不能只按 loop number 引用：

```text
axis kind:
  parallel       独立空间轴
  reduction      具有 reducer 代数的归约轴
  sequential     有 loop-carried dependency
  scan           有顺序/结合律约束的前缀轴
  segmented      extent 依赖父坐标，如 CSR row
  generated      由 relation builder/iterator 动态产生
  stage          FFT/time integration 等算法 stage；不是普通数据维
```

`IterationDomain` 是轴及其依赖构成的 DAG/coordinate hierarchy，而不是永远矩形的
shape tuple。规则子域可以直接 lower 到 `linalg/affine/scf`；structured sparse 子域
尽量 lower 到 `sparse_tensor`；GraphForge 只保留 upstream 难以表达的 relation
provenance、generate 和 traversal policy。

### 4.2 AccessMap 与 Layout：使用可组合映射

建议把 layout 看作函数而不是枚举：

```text
iteration coordinate --AccessMap--> logical field coordinate
logical field coordinate --DataLayout--> physical address
logical tile coordinate --ThreadLayout--> (lane, lane-local coordinate)
```

最小组合子包括 `compose`、`product`、`divide/tile`、`permute`、`broadcast`、
`pad`、`swizzle` 和 `vectorize`。M0/M1 不必实现完整 CuTe 类型代数，但 IR 不能锁死在
`row_major/col_major` 两个字符串上。

Sparse layout 不能硬套 stride algebra；它由 dimension-to-level map、level properties
和 position/coordinate buffers 表达。dense layout 与 sparse coordinate layout 在
`AccessMap` 处汇合，而不是强行成为同一种底层表示。

### 4.3 ExecutionMapping：映射到逻辑 machine axes

使用 target-neutral machine vocabulary：

```text
cluster / device / program / workgroup / subgroup / lane / vector / sequential
```

Target capability 再把 `subgroup` 解释为 NVIDIA warp、AMD wave 或 PPU 对应实体。映射
允许多轴 co-map、一个轴分层 split 后映射多级 machine axis，并携带 predication、
replication 和 ownership 约束。

不要在高层 schedule 写死 `warp_size=32`；schedule 参数可以是 target-dependent symbol，
specialization 后才成为常量。

### 4.4 StoragePlan：Region/Instance 而不是 tensor axis 标签

StoragePlan 选择：

- materialize、cache、recompute 还是 stream；
- Region tile 在什么 memory space 建立 PhysicalInstance；
- instance 的 layout、容量、alignment、version、lifetime 和 ownership；
- producer 在何处 `compute_at`，值在何处 `store_at`；
- spill/evict/writeback/replicate 的约束。

用户可以写 `.cache(q, "workgroup", at=k_outer)`，但其语义应展开成 instance、copy 和
lifetime，而不是给 q 的某个轴永久加一个 scope 属性。

### 4.5 PipelinePlan：显式 stage、buffer 和 event

PipelinePlan 作用于 executable statements/dataflow edges：

```text
copy(K_tile): HBM -> shared       stage 0, async
copy(V_tile): HBM -> shared       stage 0, async
score + online_reduce             stage 1
output_accumulate                 stage 2
store                             epilogue
```

它记录 `num_stages`、buffer rotation、prefetch distance、async token、barrier scope 和
资源预算。通用层只表达 dependence；backend 选择 `cp.async/TMA`、AMD/PPU 对应机制或
同步 copy fallback。

### 4.6 TaskPlacement：mesh sharding 与 relation partition 并存

规则 tensor 使用：

```text
tensor axis -> device mesh axis -> inferred collective
```

Relation 使用：

```text
entity/relation partition -> owned/ghost regions -> halo/update/migration tasks
```

两者最终都生成 `gf.task` 的 region privilege、copy/collective 和 Event DAG，pipeline
planner 才能把 interior compute 与 halo transfer overlap。

## 5. MLIR 中的落点

不建议创建一个永久承载所有信息的 `gf.schedule` 大 dialect。建议两层表示：

1. **Transform program**：主要由 compiler/autotuner 生成，采用 MLIR Transform dialect 风格，
   通过稳定 handle/name 匹配 payload IR；GraphForge 只添加 relation/segment/storage/
   pipeline 所需 extension op。专家可以外置 override，但普通用户不需要接触。
2. **Scheduled payload IR**：变换应用后，结果显式存在 `gf.iter`、`linalg`、`gf.kernel`、
   `gf.storage` 和 `gf.task` 中；codegen 不依赖隐藏 Python schedule 对象。

概念 IR：

```mlir
gf.iter.domain @attn {
  %dst  = gf.iter.axis parallel [0, %num_nodes)
  %nbr  = gf.iter.axis segmented parent(%dst) offsets(%rowptr)
  %feat = gf.iter.axis parallel [0, %feature_width)
  gf.iter.reduce %nbr
      reducer = #gf.reducer<online_softmax>
}

// Separate transform program, schematic syntax.
transform.sequence failures(propagate) {
  %op = transform.gf.match_relation @attn
  %dst_o, %dst_i = transform.gf.tile_axis %op["dst"] by [128]
  transform.gf.bind %dst_o to "program"
  transform.gf.bucket_axis %op["nbr"] by #gf.degree_classes
  transform.gf.cache %op["K", "V"] in "workgroup" at %dst_i
  transform.gf.pipeline %op along "nbr" stages 3
}
```

建议增加的 pass：

```text
gf-normalize-iteration-domain
gf-lower-relations-to-coordinate-hierarchy
gf-apply-schedule-transforms
gf-infer-access-and-layout
gf-map-execution
gf-plan-physical-instances
gf-form-software-pipeline
gf-lower-sharding-and-halo
gf-verify-schedule
gf-report-schedule
```

`gf-verify-schedule` 至少检查 reduction/scan 代数、race、ragged bounds、layout
injectivity、barrier convergence、instance capacity、async buffer lifetime 和 distributed
region version。

## 6. 用户 API 建议：M0–M2 只有 coarse API

普通用户只写 coarse MessagePassing 语义。Kernel 首次调用自动 JIT，编译器从内部
细粒度 IR 选择 schedule：

```python
class Diffusion(gf.MessagePassing):
    reducer = gf.sum(dtype=gf.float32)

    def edge(self, src, dst, edge):
        return src.x - dst.x

kernel = Diffusion()
output = kernel(graph=graph, src={"x": x}, dst={"x": x})
```

编译器可用的信息包括 named/dependent IterAxis、AccessMap、Effect、Relation
provenance/statistics、reducer algebra、依赖、reuse distance、value lifetime，以及 target
capability 和 runtime profile。layout、placement、copy 和 pipeline 是编译结果，不是
Field 定义的一部分。

内存预算、可用设备和允许使用的 storage tier 无法从计算代码推导，作为独立部署输入：

```python
deployment = gf.DeploymentPolicy(
    memory_budget={"hbm": "12GiB"},
    allowed_tiers=("hbm", "pinned", "ram", "nvme"),
)
# 通过独立 runtime/deployment context 提供，不写入 Field 或 physics source。
```

M0–M2 不公开 `gf.Schedule`。Compiler 可以在内部生成 Transform artifact 以便 dump、
测试和复现 autotune 结果，但不承诺其 Python syntax 或稳定 ABI。完成 naive
Static/Dynamic Graph 与一项显著优化后，再决定哪些 override 值得成为专家接口。

API 必须区分三类信息：

- **Semantic declaration**：Effect、Reducer、persistence、external ownership、
  determinism；影响正确性或可观察行为；
- **Deployment constraint**：设备、容量、允许的 storage/transport；来自运行环境；
- **Optimization decision**：tile、layout、placement、prefetch、pipeline、sharding；默认由
  compiler/runtime 产生，可被外置 Transform artifact 覆盖。

API 不应要求所有 op 都具有同名轴。例如 FFT 可以暴露 `batch/stage/butterfly/element`，
stencil 暴露 `time/x/y/z/offset`，CSR 暴露 `dst/neighbor/feature`。公共的是 axis kind、
transform 和映射协议，而不是固定 axis 列表。

## 7. Online softmax 例子：为什么需要六类决策

对每个 destination 的邻居做 attention：

```text
iteration:
  dst: parallel
  neighbor: segmented reduction
  feature: parallel/reduction

reducer state in registers:
  (m, l, o)

execution:
  dst tile -> program/workgroup
  neighbor chunk -> subgroup/lane + sequential stream
  feature tile -> lane-local vector/fragment

storage:
  Q accumulator -> register
  K/V tile -> workgroup/shared
  input/output -> HBM

pipeline:
  prefetch K/V chunk k+1 while computing chunk k
  rotate 2/3 shared buffers with async token/barrier

dynamic variants:
  small degree row kernel
  medium degree subgroup kernel
  large degree split-row + tuple-state combine
```

这里只用 `dst/neighbor/feature -> block/warp/lane` 无法表达 `(m,l,o)` 的 lifetime、K/V
的 shared instance、async copy 的 phase，以及 split-row 后 tuple reducer 的 combine。
这正是将多个映射分开的最小反例。

## 8. 分阶段实现建议

### M0/M1：只实现 naive lowering 所需内部结构

- named `IterAxis`，支持 `parallel/reduction/segmented/generated`；
- iteration/access mapping 分离；
- StaticGraph edge-atomic/CSR-row 与 CPU nested loops；
- Dynamic RadiusGraph cell-list materialize + consume；
- 不实现 public Schedule、通用 working-set promotion 或 pipeline language。

### M2：只实现 profile 选中的一项优化

- Static neighbor×feature/subgraph tile，或 Dynamic cell-tile/fusion/reuse；
- 只加入该优化需要的 tile、working-set、layout、copy/pipeline pass；
- profitability guard、naive fallback 和 end-to-end benchmark。

### M3 之后扩展

- 从已验证的至少两个 workload 中归纳可复用内部 transform；
- 再决定是否需要 public fine-grained traversal 与专家 Schedule；
- affine stencil axis、halo tile、time skew；
- layout composition、swizzle、fragment mapping；
- backend async-copy lowering。

### M6/M7 扩展

- HBM/RAM/NVMe instance 与 pipeline；
- device mesh sharding propagation；
- relation partition、owned/ghost/halo；
- task/event overlap 与 runtime feedback。

## 9. 需要实验回答的问题

1. named axis + dependent extent 能否同时自然 lower CSR、stencil 和 block-sparse
   attention，而不引入三套 schedule API？
2. layout 使用简单 affine map 到什么程度后必须引入 CuTe 风格层次代数？
3. degree bucket 是 iteration transform、variant dispatch 还是二者组合，哪种 IR 最稳定？
4. pipeline 自动推导能覆盖哪些 producer/consumer pattern，何时必须要求专家标注？
5. target-neutral `subgroup` 在 warp32、wave64 和 PPU 上需要哪些 capability/legality
   约束？
6. Transform program 的 canonical form 和 payload hash 能否将 GraphForge optimization
   控制在毫秒级，并把慢速 vendor compilation 隔离到 persistent cache？
7. online softmax 的 row/split-row 与 1/2/3-stage pipeline 在 degree、feature width 和
   shared/register budget 上的 crossover 在哪里？

首个原型不应追求完整 schedule language。应先证明同一套 named/dependent axis、
layout、mapping 和 pipeline IR 能生成 static CSR sum、online softmax 与 affine stencil
三类结构不同的 kernel；这比只把单个 GEMM tile 做得漂亮更能验证抽象。
