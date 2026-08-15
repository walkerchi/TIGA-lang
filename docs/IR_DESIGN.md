# GraphForge IR implementation notes

> 非规范实现笔记。唯一规范性设计是仓库根目录 `PROJECT.md`；冲突时以其为准。

状态：architecture draft  
日期：2026-08-14

本文定义 GraphForge 从 lazy Python/Torch program 到 CPU/GPU/distributed executable 的
IR 边界。目标不是立刻实现所有 op，而是让 M0 的最小 dialect 不锁死 generated relation、
auto-fusion、hierarchical storage 和 distributed MessagePassing。

---

## 1. 设计原则

1. **Graph schema 与 graph data 分开。** IR 保存 Entity/Relation schema、provenance、
   lifecycle 和性质；实际 CSR indices、positions、Field values 是 runtime operands。
2. **GraphProgram 使用 SSA value semantics。** `edge()` 读取 invocation-input snapshot，
   `node()` 产生新 Field version；物理 in-place 是后期 bufferization 结果。
3. **Assembly 是 generalized MessagePassing。** HyperRelation 的局部 contribution 通过 typed
   destination map 归约；不存在绕开 relation semantics 的 public `gf.assemble()`。
4. **Distribution 不改变数学语义。** 单设备 reduction 可拆成 rank-local partial reducers，
   最终由 destination owner combine，`node()` 只执行一次。
5. **Materialized/Paged/Generated 是 plan，不是 Graph 类型。** 同一个 logical relation 可有
   多种 ResolvedRelation。
6. **不为每条 edge 创建 IR op。** IR 描述 relation schema 和 iteration domain；runtime
   graph size 不线性膨胀 compiler IR。
7. **快速路径优先。** Canonical semantic IR、template traversal、分层 cache 和 bounded
   specialization 防止 JIT 退化为全程序 autotune。

---

## 2. 总体层级

```text
Python lazy FieldValue DAG / torch.compile region
  ↓ capture
gf.domain
  GraphProgram SSA, Entity/Field/Relation, edge/node, Reducer, Effect
  ↓ whole-program legality/canonicalization/fusion candidates
gf.task
  partition, ownership, halo, page/copy/collective, Event DAG
  ↓ per compute task
gf.iter
  entity/port/neighbor/feature axes, generated/paged/materialized iteration, reduce/route
  ↓ target planning
gf.kernel
  work tile, workgroup/subgroup/lane mapping, memory promotion, local pipeline
  ↓ dialect conversion
upstream MLIR
  arith/math/scf/affine/vector/tensor/memref/gpu/linalg/async/LLVM
  ↓ provider/backend
PTX/cubin, HIP/hsaco, PPU binary, CPU object
```

不是每个 target 都必须物理经过所有 dialect。M0 reference evaluator 可直接解释
`gf.domain`；简单 CPU stencil 可从 `gf.domain` 直接进入 `affine/scf/vector`；但各层的语义
边界必须一致，inspection 也使用这些稳定 stage 名。

---

## 3. `gf.domain`：语义 IR

### 3.1 核心类型

以下语法是 schematic MLIR，不是最终 parser spelling：

```mlir
!gf.entity_set<"node", id = i64, coord = [index, index]>
!gf.entity_ref<@nodes>

!gf.field<@nodes, f32>
!gf.field<@nodes, tensor<3xf32>>
!gf.field_value<@nodes, tensor<3xf32>, version = ?>

!gf.relation<
  sources = [@nodes],
  destinations = [@nodes],
  arity = 2,
  origin = #gf.origin<external>,
  lifecycle = #gf.lifecycle<frozen>
>

!gf.relation_item<@edges>
!gf.snapshot<@edges>
!gf.reducer_state<@sum_f32>
!gf.event
```

`Field` 是 schema；`FieldValue` 是某个 invocation/version 的 SSA value。高层 IR 不允许通过
同一个 FieldValue handle 原地改变可见值。

### 3.2 Entity 与 Field schema

```mlir
gf.entity_set @nodes {
  id_type = i64,
  coordinate_rank = 2,
  global_cardinality = #gf.dynamic
}

gf.field @u on @nodes : f32 {
  mutability = "versioned",
  semantic_role = "state"
}
```

Runtime binding 将 `%u : !gf.field_value<@nodes, f32>` 绑定到 Torch Tensor、GraphForge
Region 或 distributed local shard。Global cardinality 和 runtime Tensor address 不进入
symbol identity。

### 3.3 Relation origin、lifecycle 与 realization

三个维度必须独立：

```text
origin:      External | Procedural
lifecycle:   Frozen | Rebuildable
realization: Materialized | Paged | Generated
```

External CSR：

```mlir
gf.relation @adj (%row_ptr, %col_idx) : ... {
  origin = #gf.external<format = "csr", index = i32>,
  lifecycle = #gf.frozen,
  properties = {sorted = true, symmetric = false}
}
```

Affine stencil：

```mlir
gf.relation @dx : ... {
  origin = #gf.procedural<"affine_stencil">,
  lifecycle = #gf.frozen,
  ports = [
    #gf.port<id = 0, offset = [-1, 0], side = "minus">,
    #gf.port<id = 1, offset = [ 1, 0], side = "plus">
  ],
  lattice_basis = #gf.identity_basis<2>,
  boundary = #gf.boundary<periodic, periodic>
}
```

Port 保存 topology/orientation facts；特定 derivative 的 numerical coefficient 属于 edge/node
operator。非 Cartesian lattice 通过 `physical_displacement = basis * integer_offset` 生成
derived edge value，因此 axial hex ports 仍是固定 affine domain。

规则平铺不能表示成 Cartesian/hex 等 case enum。`gf.domain` 的通用 schema 是：

```mlir
gf.index_domain @cells {
  rank = 2,
  set = #gf.presburger_set<(i, j) : ...>,
  boundary_action = #gf.periodic_quotient<...>
}

gf.indexed_complex @complex on @cells {
  generators = dense<...> : tensor<2x2xf64>,
  sites = [
    #gf.site<"A", offset = [...]>,
    #gf.site<"B", offset = [...]>
  ]
}

gf.relation @nearest on @complex {
  links = [
    #gf.endpoint_map<dst_site = "A", src_site = "B",
                     map = (i, j) -> (i, j)>,
    #gf.endpoint_map<dst_site = "A", src_site = "B",
                     map = (i, j) -> (i - 1, j + 1)>
  ]
}
```

Formal core：

```text
IndexDomain D ⊂ Z^d
Entity (i, site), i ∈ D
EndpointMap: (i, dst_site) → (A i + b, src_site)
Embedding: x(i, site) = B i + c_site, or runtime coordinate Field
BoundaryAction: quotient/map/ghost/mask
```

Cartesian、triangular、honeycomb、Kagome、FCC/BCC、staggered/MAC grids 都只是不同的
domain/sites/maps/embedding。有限 polygonal patch 可用 Presburger pieces；一般 aperiodic
generator 使用 `GeneratedTileProducer`；unstructured mesh 使用 explicit cell-complex incidence。
Pass 通过 interfaces 查询 endpoint map 是否 affine、port domain 是否 finite、embedding 是否
constant，而不是 `switch (tiling_kind)`。

Unstructured `Mesh` 不是另一种 kernel input，而是若干 EntitySets/Relations/Fields 的 Python
handle。Domain IR 保存 typed oriented incidence：

```mlir
gf.entity_set @vertices {dimension = 0}
gf.entity_set @faces    {dimension = 2}
gf.entity_set @cells    {dimension = 3}

gf.relation @cell_faces (%offsets, %indices, %orientation)
  : !gf.relation<sources = [@cells], destinations = [@faces],
                   origin = external, lifecycle = frozen>

gf.field @position on @vertices : tensor<3xf64>
gf.field @normal   on @faces    : tensor<3xf64>
gf.field @volume   on @cells    : f64
```

Incidence 可 fixed-arity、type-bucketed 或 ragged。`cell adjacency through faces` 是 relation
composition：

```text
cell_dst ← incidence → face ← incidence → cell_src
```

Planner 可 materialize CSR，也可遍历 incidence composition 而不写出 cell-cell adjacency。
Orientation 是 relation item property，确保同一个 face 对左右 cells 产生相反 normal/sign。
Curvilinear mesh 只把 embedding/Jacobian/metric 改为 runtime Fields；AMR 增加 level、parent/
child 与 coarse-fine interface Relations。它们都 lower 到同一个 `gf.apply`。

Radius relation：

```mlir
gf.relation @radius (%positions, %cutoff, %box) : ... {
  origin = #gf.procedural<"radius">,
  lifecycle = #gf.rebuildable<dependencies = [0, 1, 2],
                              predicate = "field_version_or_skin">,
  properties = {directed = true}
}
```

编译后的函数接收 graph/relation handle 或 underlying buffers；relation actual contents 和
logical version 不进入 code cache key。Version 使 materialization/statistics/plan 失效。

### 3.4 Snapshot resolve

```mlir
%snapshot, %ready = gf.resolve %relation
  budget(%memory_budget)
  : !gf.relation<...> -> (!gf.snapshot<...>, !gf.event)
```

`gf.resolve` 的结果在 whole-program planning 后具体化为：

```text
MaterializedInstance
PagedTileStream
GeneratedTileProducer
```

Domain IR 中 resolve 表示 snapshot/version correctness barrier，不承诺独立 builder kernel 或
完整 adjacency allocation。

### 3.5 Reducer definition

Reducer 是 symbol op，regions 可内联、专门化或映射到 backend primitive：

```mlir
gf.reducer @sum_f32 : f32 -> f32 -> f32 {
  properties = [associative, commutative, atomic_add],
  determinism = #gf.determinism<order_independent_not_bitwise>
  identity { gf.return 0.0 : f32 }
  lift(%x: f32) { gf.return %x : f32 }
  combine(%a: f32, %b: f32) {
    %c = arith.addf %a, %b : f32
    gf.return %c : f32
  }
  finalize(%x: f32) { gf.return %x : f32 }
}
```

Online softmax 的 state 是 `(m, l, o)`；`combine` 使用稳定 rescale 公式。PortGather 的 state
是 compile-time port-indexed tuple/vector + validity mask，并声明 `unique_by_port` 或重复项
行为。Verifier 不尝试证明任意 user reducer 的结合律；builtin reducer 由系统保证，custom
reducer 的 algebra property 是显式契约并进入 determinism diagnostics。

Public `NeighborhoodKernel.compute(center, nbr, ...)` 是 fixed finite port domain 的 ergonomic
frontend；capture 时规范化为 `PortGather + local region`，与 MessagePassing 共用 `gf.apply`、
iteration/storage/distributed lowering。它不是可用于任意 ragged/dynamic neighbor stream 的
随机访问容器。

### 3.6 Unified MessagePassing op

Public `edge()/node()` capture 为一个 `gf.apply`：

```mlir
%u_next = gf.apply %snapshot
    fields(%u_src, %u_dst, %weight)
    params(%dt)
    reducer(@sum_f32)
    effects [read(%u_src), read(%u_dst), read(%weight)]
    : ... -> !gf.field_value<@nodes, f32> {

  ^edge(%src: !gf.entity_ref<@nodes>,
        %dst: !gf.entity_ref<@nodes>,
        %item: !gf.relation_item<@adj>):
    %su = gf.read %u_src[%src] : f32
    %du = gf.read %u_dst[%dst] : f32
    %w  = gf.read %weight[%item] : f32
    %d  = arith.subf %su, %du : f32
    %m  = arith.mulf %w, %d : f32
    gf.edge_yield %m : f32

  ^node(%dst: !gf.entity_ref<@nodes>, %total: f32):
    %old = gf.read %u_dst[%dst] : f32
    %du  = arith.mulf %dt, %total : f32
    %new = arith.addf %old, %du : f32
    gf.node_yield %new : f32
}
```

语义：

```text
contribution[e] = edge(input snapshot)
aggregate[d]    = reducer(contribution[e] where destination(e)=d)
output[d]       = node(destination input snapshot, aggregate[d])
```

`node` region 可省略，此时 output 是 reducer finalized result。不存在单独 `old` 参数；node
通过 destination snapshot 读旧值。Bufferization 只有证明无 read-after-overwrite 后才能令
output 与 input storage alias。

### 3.7 HyperRelation 与 assembly

Binary MessagePassing 是 `gf.apply` 的特例。HyperRelation item 可有固定或 ragged endpoints：

```text
item = element
endpoints(item) = [node_0, ..., node_{P-1}]
```

Element region 产生 port-indexed local contributions；IR 使用 typed route terminator：

```mlir
^edge(%element, %endpoints, %item):
  %re = gf.call_local @element_residual(%endpoints, %geometry)
        : ... -> tensor<Pxf32>
  gf.route_yield %re to %endpoints by #gf.endpoint_identity
                 reducer @sum_f32
```

`route_yield` 是 `gf.apply` 内部 terminator，不是 public `gf.assemble()`。它声明 local result
axis 如何映射到 destination EntitySet。Lowering 可选择：

- matrix-free element tile：compute once → scatter residual；
- coloring/atomics/segmented reduction；
- assemble sparse matrix：destination 改为 typed `(row_dof, col_dof)` EntitySet；
- distributed partial route：按 destination owner combine。

### 3.8 GraphProgram SSA 与 auto-fusion

普通 Python function 中连续 leaf calls 形成一个 `func.func`/GraphProgram：

```mlir
func.func @euler_rhs(%mesh, %q, %geometry) -> !gf.field_value<...>
    attributes {gf.program} {
  %q_face = gf.apply @reconstruct ...
  %flux = gf.apply @riemann ...
  %residual = gf.apply @divergence ...
  %out = gf.local_map @add_source(%q, %residual)
  return %out
}
```

GraphForge-native `FieldValue` 默认 lazy：leaf call 返回 deferred SSA handle；首次 external
observation、unsupported escape、mutation/barrier、memory-pressure flush 或显式 materialize
触发 planning/JIT。`@gf.program` 只作为可选 capture/AOT/export/debug boundary。

Auto-fusion pass 从 SSA def-use 和 Effect 推导候选，不以 decorator 作为 fusion hint：

```text
legal if:
  producer result has no observable escape
  snapshots/versions are compatible
  iteration domains can be composed
  effects and ownership preserve ordering
  reducer route/finalize can be transformed

profitable if:
  saved bytes/launch/communication > duplicate work
  register/shared footprint respects capability
  occupancy and parallelism stay acceptable
```

典型 rewrite：

- local map → edge/node inline；
- same-relation multi-consumer traversal fusion；
- generated relation builder → consumer fusion；
- reconstruction → face flux tile chaining；
- face flux → cell divergence partial routing；
- pointwise time update → node region fusion。

通信、global reduction、unsupported op、external Tensor observation、alias barrier 或过高 resource
cost 会形成 fusion boundary。`explain()` 必须给出每条未融合 SSA edge 的原因。

---

## 4. `gf.task`：storage/distributed execution graph

### 4.1 为什么在 `gf.iter` 之前

Distributed/out-of-core planning 先把一个 logical apply 拆成 local compute、transfer、halo 和
combine tasks；每个 compute task 再 lower 自己的 relation iteration。单设备 in-memory
程序可使用 trivial task graph，不必显式打印这一层。

### 4.2 Region、instance 与 event

```mlir
%owned = gf.region.subset %u [#gf.partition_owned]
%ghost = gf.region.subset %u [#gf.partition_ghost]
%tile = gf.storage.acquire %stream[%tile_id] budget(%bytes)
%copy_done = gf.task.copy %tile_src to %tile_hbm
%event = gf.task.launch @kernel(...) depends_on(%copy_done)
gf.storage.release %tile after(%event)
```

`Region` 是 logical Field subset + version；`PhysicalInstance` 是某 memory space 的 layout/
compression copy。Register/shared placement 不使用这些 long-lived ops，而在 `gf.kernel` 中表示。

### 4.3 Distributed MessagePassing 的不变语义

全局定义不变：

```text
aggregate[d] = reduce(contribution[e] for e where destination(e)=d)
```

Partition 后要求每个 destination 有唯一 logical owner。Planner 可选择：

#### Owner-compute / pull

```text
destination owner stores/iterates incoming relation items
remote source Field → ghost/halo fetch
owner computes all contributions
owner runs node() once
```

适合 destination-partitioned CSR、stencil 和 spatial radius。

#### Edge-compute / push

```text
edge/element owner computes contributions
local combine by destination ID
send (destination, partial reducer state) to destination owner
owner combines received states
owner runs node() once
```

适合 edge/element geometry 很大、已按 element 分区或 source feature 搬移代价较高的情况。

#### Hybrid/2D

Source feature、edge 和 destination axes 分到 device mesh，不同维度分别 broadcast/reduce。
Dense all-pairs、超长程 interaction 和极高 degree graph 可选择此计划。

Reducer 必须提供跨 rank 合法的 `combine`。若 reducer 不可 combine，只有将全部 messages 按
规定顺序送到 owner 才能保语义；planner 应拒绝不可承受的 distributed plan，而不是静默
改变结果。

### 4.4 Overlap task graph

Owner-compute stencil/radius 的典型 task IR：

```text
halo_pack(local boundary fields)
  → halo_exchange ───────────────→ boundary_apply
interior_apply ──────────────────→ merge/finalize → node/output version
```

Interior 和 exchange 无依赖，可并行；boundary 等待 halo event。对于 push plan：

```text
local edge compute
  → partial combine
  → pack by destination owner
  → exchange
  → owner combine
  → node
```

Snapshot version、halo version 和 output version 必须匹配；verifier 禁止混合不同 timestep 的
ghost data。

### 4.5 Public distributed binding

用户不写通信 loop，但必须提供不可推导的语义 metadata：global entity ID、local shard 与
global EntitySet 的映射、或者 partition manifest。示意：

```python
mesh = gf.DeviceMesh("cuda", (2, 4), names=("rack", "gpu"))
graph = gf.load("mesh.gfg").halo(
    mesh,
    partition=gf.ByDestination(mesh_axis="gpu", balance="edges"),
    depth="auto",
)
u = gf.Field.from_local(local_u, entities=graph.nodes)

u_next = Diffusion()(graph=graph, src={"u": u}, dst={"u": u})
```

`Graph.halo()` 是 declarative logical transformation，仍返回普通 `gf.Graph`，不会立即通信。
Process group/device mesh 可绑定 deployment config、Torch DeviceMesh、`torchrun`/MPI/vendor
launcher。Partition 算法、halo packing、transport、overlap 和 kernels 是 planner/runtime 决策；
ownership/global ID 不能凭空自动推断。

---

## 5. `gf.iter`：target-independent iteration IR

当前最小实现的独立 dialect spelling 是 `gf_iter.traverse/yield`。它已能表达
`compressed-row` 与 `generated-neighborhood`、`destination-major` ordering，并完整保留
edge/optional-node regions 和 reducer/effect metadata。下面的 Axis ops 是后续渐进展开，不应
在尚无优化需求时一次性预建。

### 5.1 Axis model

```text
EntityAxis:   destination/source/entity set
SegmentAxis:  CSR row, bucket, page segment
NeighborAxis: extent depends on destination/segment
PortAxis:     small compile-time stencil/element endpoint domain
FeatureAxis:  dense payload axis
CellAxis:     geometric bin/neighborhood
```

Axis 可以 dependent/ragged；不要为了使用普通 tensor loop 把 neighbor extent padding 成全局
最大 degree。

### 5.2 CSR lowering

```text
forall dst in D:
  state = reducer.identity
  for p in row_ptr[dst] .. row_ptr[dst+1]:
    src = col_idx[p]
    state = reducer.combine(state, edge(src, dst, p))
  out[dst] = node(dst, reducer.finalize(state))
```

`gf.iter` 保留 `dst → segment → neighbor` 依赖关系，后续才能选择 row-per-thread、warp-per-row、
split-row、edge-atomic 或 degree bucket。

### 5.3 Affine stencil lowering

```text
forall (i, j) in owned grid:
  for port in static Ports:
    (si, sj) = affine_map(i, j, port.offset)
    state = combine(state, edge((si, sj), (i, j), port))
```

Affine map 和 boundary region 尽量 lower 到 upstream `affine`/`vector`，不生成 CSR。

### 5.4 Generated radius lowering

```text
forall owned destination_cell:
  for neighboring_cell_port:
    for destination_particle_tile:
      for source_particle_tile:
        if distance(src, dst) <= cutoff:
          reduce edge(src, dst, derived_geometry)
```

Materialized plan 将中间层替换为 COO/CSR segment；generated-fused plan 直接消费 predicate
结果；paged plan 在 SegmentAxis 外增加 tile acquire/release。

### 5.5 HyperRelation route lowering

```text
forall element:
  endpoints = relation.endpoints(element)
  local = element_kernel(endpoints)
  for port in endpoint_ports:
    route local[port] to endpoints[port] using reducer
```

Route 可 lower 成 atomics、coloring、sort/segment、owner partial state 或 destination-tile
accumulator。

---

## 6. `gf.kernel`：target-aware physical IR

当前最小实现的独立 dialect spelling 是 `gf_kernel.launch/yield`，包含
`csr-row/generated-tile` skeleton、degree worklist/split-row planning 和类型化 storage/task
边界。Domain→Iter→Kernel→TTIR 已可运行并在 CUDA Driver runtime 中执行；下面更一般的
WorkTile/Promotion/Pipeline 仍是设计，不应把 schematic 当作已实现 op。

这一层才引入硬件 mapping 和 kernel-local memory：

```text
WorkTile
  logical axes/subset
  → program/workgroup/subgroup/lane/vector/sequential mapping

Promotion
  Field/indices/partial state slice
  → register/shared/LDS/local memory

Pipeline
  async copy → barrier → compute → writeback
  stages/buffers/events
```

Schematic：

```mlir
gf.kernel.launch @radius_tile mapping(#gf.workgroup) {
  %src_smem = gf.kernel.promote %src_tile to #gf.memory<shared>
  %dst_smem = gf.kernel.promote %dst_tile to #gf.memory<shared>
  gf.kernel.pipeline stages(3) {
    gf.kernel.async_copy ...
    gf.kernel.compute ...
  }
}
```

这些 op 由 compiler/autotuner 生成，不进入 coarse public API。CPU lowering 可删除 GPU
mapping，将 WorkTile 变为 cache-blocked/vectorized loops。

### 6.1 `gf_tensor` 与 `gf_control`：可微数据流和有界迭代

`gf_tensor` 是 runtime Tensor DAG、自动 VJP 与 provider codegen 共用的 typed SSA 层；它不
是用 Python 字符串拼接出来的第二套 IR。静态 CSR 的 relation lowering 保留
`gather → edge algebra → csr_segment_sum → node algebra`，其中
`csr_segment_sum` 携带由 Graph analysis 证明的 `degree_min/degree_max`。这些属性只决定后期
schedule；数值语义仍然是 CSR 行归约。未知度数使用 `(0, 0)` 并进入任意长行的 tiled-loop
fallback，绝不根据 `E/N` 猜测 uniform relation。

`gf_control.repeat` 表达固定次数的 loop-carried Tensor state：

```mlir
%rank1 = "gf_control.repeat"(%rank0, %row_ptr, %col_idx, %weight) <{
  iterations = 20 : i64, num_carried = 1 : i64
}> ({
^bb0(%rank: tensor<Nxf32>, %row: tensor<?xi64>,
     %col: tensor<Exi64>, %w: tensor<Exf32>):
  %next = ... : tensor<Nxf32>
  "gf_control.yield"(%next) : (tensor<Nxf32>) -> ()
}) : (...) -> tensor<Nxf32>
```

`num_carried` 允许前缀中的多个不同 shape/type Tensor 同时成为 SSA result；其余 operands 是
显式 immutable captures。循环体只 capture 一次，iteration count 不增加 compiler IR 节点。
CPU lowering 产生 `scf.for`，并为每个 carried value 复用两个 ping-pong MemRef；单状态 CUDA
runtime 复用两个 device buffer 和 prepared executable，多状态 CUDA loop plan 尚待实现。
CPU pass 还会将 body 内 rank-0 reduction 及其 scalar algebra 提升为每迭代一次的临时值，
避免 CG 的 dot product 因被多个 vector update 引用而退化成 `O(N^2)`；多次 use
的 vector SSA（例如 `A(p)` 和 next residual）也按 block SSA 顺序每迭代只物化一次，
单 use vector 仍融合到 consumer。当前
fixed-degree weighted CSR + 逐节点 epilogue 会结构匹配到二维 row×neighbor TTIR tile，每次
迭代一次 launch；未知/长尾 CSR 使用一行一 program、任意长度分块循环。这里没有
PageRank-named op 或 codegen case。`gf_control.while` 使用 rank-0
`gf_tensor.compare` condition 和强制 `max_iterations`；CPU 降到 `scf.while`，
收敛后不再执行 body，也不把 scalar 转回 Python。GPU command-graph/cooperative
persistent plan 及 distributed collective condition 仍未实现，因此 PageRank 仍只能作为
`G0` 的 partial evidence。

Reverse mode 对 fixed repeat 已有自动 correctness fallback：autograd transform 将 body 按
iteration 特化，并直接复用已有 pointwise、CSR relation 和 reducer VJP，用户不写 backward。
这条路径的反向 IR/compile work 为 O(iterations)，还不是反向 `gf_control.repeat`；因此只作为
语义覆盖，不登记性能。后续 control-autodiff pass 需要依据 save/recompute budget 生成反向循环
和 state tape，再与 checkpoint/hierarchy planner 合并。

---

## 7. Pass pipeline

### 7.1 Semantic normalization

```text
gf-verify-domain
gf-infer-field-access-and-effects
gf-canonicalize-origin-lifecycle
gf-normalize-edge-node-apply
gf-canonicalize-reducers
gf-build-program-ssa
gf-identify-fusion-candidates
```

### 7.2 Whole-program planning

```text
gf-resolve-shape-and-statistics-buckets
gf-select-materialized-paged-generated
gf-select-ownership-and-partition-plan
gf-form-profitable-fusion-groups
gf-plan-storage-and-events
gf-domain-to-task
```

### 7.3 Iteration lowering

```text
gf-lower-relation-to-axes
gf-lower-route-and-reducer
gf-select-traversal-skeleton
gf-bucket-or-split-ragged-work
gf-task-compute-to-iter
```

### 7.4 Target planning/codegen

```text
gf-map-work-tiles
gf-plan-promotion
gf-plan-local-pipeline
gf-bufferize-field-versions
gf-iter-to-kernel
gf-kernel-to-triton / gpu / vector / LLVM / source provider
```

每项复杂 pass 都必须有 naive fallback。M0/M1 只实现其中最小子集，但未实现 op 必须产生
明确 diagnostic，不能 silent miscompile。

---

## 8. Verification

### 8.1 Domain verifier

- Field 的 EntitySet 与 relation endpoint type 匹配；
- edge/node 只访问声明/推导出的 Field 和参数；
- output Field version 不在同一 apply 中被当成 input snapshot 读取；
- relation port、boundary、sortedness、uniqueness property 使用合法；
- reducer message/state/result type 匹配；
- HyperRelation route 的 destination map 和 local tensor axis 匹配；
- persistent edge state 不能绑定到不稳定 generated edge ID；
- runtime parameter 与 specialization constant 明确区分。

### 8.2 Task/distributed verifier

- 每个 destination 有唯一 logical owner；
- node region 在 final combine 后且只执行一次；
- halo/partial state version 与 snapshot version 相同；
- transfer/launch/release Event 无 use-before-ready；
- cross-device reducer 具有合法 combine；
- deterministic mode 的 ordering/algorithm 满足声明。

### 8.3 Kernel verifier

- register/shared/LDS allocation 不超过 target capability；
- barrier 对所有参与线程可达且 memory scope 正确；
- promoted tile 生命周期覆盖全部 use；
- masked/padded work 不越界；
- atomic dtype/op 被 backend 支持。

---

## 9. Cache 与快速编译

### 9.1 不进入 code cache key

```text
actual CSR/COO contents
Graph logical version
positions/cutoff runtime values
runtime pointer/address
exact rank-local partition contents
```

### 9.2 进入 semantic/planning key

```text
canonical GraphProgram/gf.domain hash
Entity/Field/Relation schema
origin/lifecycle/enumerator kind
edge/node/reducer IR
dtype/index width/rank
shape/degree/statistics bucket
target capabilities and compiler/provider versions
determinism/effect/compile options
```

### 9.3 分层 cache

```text
semantic cache: canonical gf.domain and per-region hashes
fusion cache:   GraphProgram fusion groups and guards
planning cache: traversal/materialization/partition/storage plan
provider cache: Triton/LLVM/source artifact
runtime cache:  loaded module, launch metadata, communication plan
```

### 9.4 JIT latency策略

- small builtin reducer/edge/node regions canonicalize and hash independently；
- traversal 使用少量 pre-verified skeleton，不为每个公式复制 planner；
- first-call 使用 heuristic quick plan，不阻塞完整 autotune；
- autotune winner 后台/离线持久化，guarded replacement 不改变 semantic key；
- graph size 使用 bounded buckets，避免每个 N/degree 重新编译；
- 不展开 runtime entities/edges；只展开很小的 static PortAxis；
- distributed rank 共用 binary，rank ID/partition metadata 是 runtime operands；
- inspection artifact 从 cache 读取，不触发无关 target compilation。

---

## 10. Inspection contract

```python
kernel = Diffusion()
u_next = kernel(graph=graph, src={"u": u}, dst={"u": u}, dt=dt)

print(kernel.ir("domain"))
print(kernel.ir("task"))
print(kernel.ir("iteration"))
print(kernel.ir("kernel"))

variant = kernel.inspect()
print(variant.explain())
print(variant.fusion_groups())
print(variant.relation_plan())
print(variant.distribution_plan())
print(variant.memory_plan())
print(variant.artifacts())
```

`explain()` 至少报告：

```text
selected snapshot/realization
fusion groups and barriers
traversal and reducer strategy
ownership/halo/partial-combine plan
working-set and memory-space estimates
compile/cache/build/transfer/consume timings
fallbacks, guards and rejected candidates
```

---

## 11. M0 最小实现切片

M0 不实现完整 task/iter/kernel dialect，但 Domain IR 必须能 round-trip 以下内容：

1. EntitySet、Field schema/value；
2. External+Frozen CSR Relation；
3. Procedural+Rebuildable Radius schema；
4. origin/lifecycle/realization interface；
5. binary `gf.apply` 的 edge/optional node regions；
6. builtin sum Reducer；
7. read/write/reduce Effects 与 snapshot version；
8. ResolvedRelation interface；
9. straight-line GraphProgram SSA；
10. target-independent canonical hash 和 source location。

M1 增加 CSR/radius baseline lowering；M4 增加 IndexDomain/IndexedComplex/EndpointMap、
AffineRelation、PortGather、HyperRelation route 和 auto-fusion；M6/M7 实现
PagedTileStream/storage task 与 distributed task lowering。

---

## 12. 历史原型问题及当前决议

1. `gf.apply` 使用多 region op，还是 reducer/node 作为 symbol call 更利于 region hashing；
2. HyperRelation fixed/ragged endpoints 是否统一为一个 type，还是分开以简化 verifier；
3. GraphForge lazy Field 与 Torch Tensor-subclass bridge 的可维护性和 graph-break 行为；
4. Domain-level fusion 与 target-level tile fusion 的 cost model 如何分工；
5. distributed ownership/partition attr 哪些进入 IR，哪些只属于 deployment plan；
6. custom reducer algebra property 的 trust/verification/debug contract；
7. upstream `sparse_tensor` coordinate hierarchy 能复用到什么程度，何时保留专用 relation op；
8. codegen skeleton + generated region 在 Triton/LLVM provider 中的最小链接边界。

这些问题已由 M0 round-trip、vertical slice 和 composition prototype 收口：apply/reducer
保留 region 以支持结构分析与 hashing；fixed/ragged endpoint 共享 relation contract、物理
instance 分开；Torch 仅是可选 adapter；Domain fusion 与 tile fusion 分属语义/物理层；
ownership 进入 IR、transport placement 留在 deployment plan；reducer 属性必须由结构证明或
fail-closed；通用 sparse hierarchy 可复用但不抹平 generated relation；TTIR 是 vendor provider
边界。Python API 不暴露 target-specific schedule。
