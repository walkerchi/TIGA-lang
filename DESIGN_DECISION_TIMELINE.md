# Tiga 设计决策时间线

状态：历史记录，不是现行规范  
整理日期：2026-08-12  
现行规范：[PROJECT.md](PROJECT.md)

本文按讨论和实现发生的先后顺序，记录 Tiga 从“科学仿真是否都能抽象为
MessagePassing”发展为 relation-oriented compiler 的过程。每个节点尽量保留当时的问题、
候选方案、验证、结论和后续影响。

时间线中的状态含义：

- **固定**：已进入当前项目规范，后续实现不得无理由违背；
- **已落地**：已有代码和测试，但覆盖范围可能仍有限；
- **实验**：做过原型或测量，尚未成为稳定实现；
- **否决**：明确不再沿用的设计；
- **待完成**：接口或 IR 已预留，但执行能力尚未交付。

精确日期没有单独记录的早期讨论以阶段编号排序，不伪造日历日期。2026-08-05、
2026-08-10 和 2026-08-12 是仓库中存在明确证据的检查点。

---

## T00：提出“科学仿真统一为 MessagePassing”的初始假设

**问题**

能否把科学仿真统一描述为：沿 relation/edge 产生 message，在 destination 上聚合，再更新
node？如果可以，就能构造一个类似 Triton、TileLang、CuTe DSL 的稀疏计算基础设施，并把
Static Graph、Radius Graph、CPU/GPU、国产加速器、distributed 和多级存储统一起来。

**最初候选**

```text
for edge(src, dst) in graph:
    message = edge_fn(src, dst, edge)
for dst in nodes:
    aggregate = reducer(messages_to(dst))
    output[dst] = node_fn(dst, aggregate)
```

用户继承 `MessagePassing`，实现 message/edge 与 aggregate/update；JIT 根据 graph 和硬件生成
kernel。

**分析**

这个模型非常适合：

- GNN、粒子相互作用、邻域搜索后的计算；
- CSR/COO 稀疏聚合、SpMV/SpMM；
- stencil、有限体积 face flux、有限元 incidence；
- diffusion、Laplacian、局部守恒残差；
- attention 中的 query-key relation 与在线归约。

但不能把它当成整个系统唯一的最低层 primitive。FFT、谱方法、dense GEMM/BEM、全局线性
求解、排序、扫描等即使能“画成图”，也可能丢失 butterfly、矩阵乘、分层低秩、scan tree 等
更强的结构，甚至把原本的 `O(n log n)` 算法退化为显式全连接边。

**结论：固定**

Tiga 采用 **relation/message/reducer 作为主要领域语义**，但 IR 和 runtime 必须允许
dense contraction、scan、FFT、solver/library call 等其他结构化 primitive。所谓“图是通用
表示”只说明语义可表达，不代表物理执行必须枚举 edge。

---

## T01：FFT、stencil 是特殊图，但不能只保留“图”这一层信息

**追问**

FFT、stencil 难道不也是特殊图结构？

**澄清**

是。关键区别不是“能不能表示为图”，而是 compiler 是否保留足够强的结构：

- stencil 是有规则 offset、方向、边界作用和重复 pattern 的 implicit relation；
- FFT 是多级 butterfly relation，还带 stage、radix、twiddle 和跨 stage 数据布局；
- dense matmul 是完整 Cartesian relation，但其 tensor contraction 结构应映射 tensor core，
  不应变成逐 edge scalar loop；
- scan/sort 有顺序和分层合并约束，不能被普通 commutative reducer 取代。

**结论：固定**

Graph 是逻辑 relation；physical lowering 可以是 CSR traversal、规则 loop、block-sparse tile、
tensor contraction、butterfly 或 library dispatch。Compiler 的任务是从 relation schema 和
reducer algebra 中恢复/保留这些结构，而不是把所有结构抹平成 adjacency list。

---

## T02：效率不能依靠一个万能 kernel，必须分离语义与调度

**问题**

统一抽象之后如何保证效率？

**候选方案**

1. 每种算法手写一个 backend kernel；
2. 用户手工指定 block、shared memory、pipeline；
3. 前端只描述语义，compiler 根据 relation/reducer/shape/硬件选择 skeleton 和 schedule；
4. 提供粗粒度自动路径，并在确有需要后增加细粒度语言。

**分析**

方案 1 会把 Tiga 变成 kernel library；方案 2 会把硬件细节泄漏到仿真代码并破坏跨
CPU/GPU；方案 3 是目标，但早期 optimizer 不可能自动解决全部图不规则性；因此需要方案 3
和方案 4 的分阶段组合。

效率来源被拆为：

- 不物化本可 implicit/generated 的 relation；
- destination-major、edge-major、split-row 等 traversal 选择；
- degree/shape/locality proof；
- fusion，减少 relation/index/field 重复流量；
- reducer algebra recognition，例如 sum、max、online softmax；
- row/neighbor/feature tile；
- register/shared/LDS/HBM promotion 和 async pipeline；
- library dispatch 与 generated code 之间的性能 gate；
- distributed interior/boundary overlap。

**结论：固定**

Public semantics 与 physical schedule 分层。默认自动调度；只有 profiling 证明 compiler 缺少
某个必要自由度时，才把它提升为可复用的细粒度语言原语。

---

## T03：先写 PROJECT.md，再进入工程阶段

**动作**

理论分析基本稳定后，开始把范围、接口、IR、backend、性能契约和里程碑集中写入
`PROJECT.md`。

**早期工程范围**

- coarse `MessagePassing`；
- Static CSR Graph 与 Dynamic Radius Graph；
- 一个 CPU backend 和一个 GPU baseline；
- accuracy benchmark、performance benchmark、SOTA/CUDA 对照；
- 可检查的 JIT artifact。

**结论：固定**

先形成端到端 vertical slice，再从 profiling 反推硬件优化和细粒度 DSL。不能一开始设计一个
看似完备但没有任何测量证据的“终极调度语言”。

---

## T04：补充 related work 和 background

**问题**

是否需要调研文献与现有项目，而不是闭门造车？

**调研对象**

- PyG/DGL/GraphBLAS：message passing、segment reduction、semiring；
- Triton/TileLang/CuTe：tile、layout、GPU codegen；
- Taichi：领域前端、SNode/IR、多 backend JIT；
- MLIR：多层 dialect、progressive lowering、pass infrastructure；
- Legion/Regent/Charm++/Kokkos/RAJA：region、task、placement、distributed；
- FlashAttention/online softmax：不物化 dense relation 与 tuple-state reduction；
- Gunrock/GraphIt/GraphBLAST：图 traversal、direction optimization、load balance；
- out-of-core graph systems：paged relation、streaming、partition。

**得到的启发**

- Graph schema、reducer algebra、storage/task dependency 必须在高层 IR 保留；
- tensor axis mapping 不能替代 sparse/ragged coordinate hierarchy；
- layout/pipeline 是 target planning 的结果，不应成为每个用户程序的必写注解；
- backend portability 依赖稳定的 Tiga Kernel IR，而不是稳定的 Triton 内部 IR；
- benchmark 必须语义匹配、包含构图成本和冷/热 JIT，而不是只挑有利数字。

**结论：已落地**

调研写入 `docs/RELATED_WORK.md`，但规范性结论合并回 `PROJECT.md`。`docs/` 不拥有独立于
`PROJECT.md` 的产品规则。

---

## T05：项目既要接入 Torch，也不能依赖 Torch

**初始诉求**

Tiga 应能直接接入 Torch，甚至在其覆盖范围内替代 Torch。

**设计演化**

早期实现大量使用 Torch tensor、`index_add_` 和 sparse ops，方便 correctness 与性能对照。
随后发现 reducer、graph/core、compiler/runtime 都依赖 Torch，会使项目无法成为独立 compiler，
也无法在没有 Torch 的国产卡或嵌入式 runtime 上运行。

**最终分层**

```text
tiga core:
  Graph / Reducer / MessagePassing / Tensor / runtime / compiler

optional adapter:
  tiga.interop.torch
  Torch Tensor binding / semantic oracle / library provider
```

**结论：固定，核心已落地**

Torch 是可选 adapter 和 oracle，不是 Tiga IR、Reducer、Graph 或 runtime 的基类。Public
API 接受 Torch tensor 时可以 lazy 进入 interop；无 Torch 环境必须仍能加载 compiler、native
Tensor/runtime 和 bundled tools。

---

## T06：从第一天采用 MLIR，但 progressive 实现

**争议**

是否应该一开始就加入 MLIR、自定义 IR 和 passes？

**结论形成过程**

只写 `triton.py` backend 能快速出结果，但会造成：

- frontend semantics 与某个 CUDA kernel 模板绑定；
- 无法自然表示 Graph lifecycle、Reducer regions、Effects、Storage、Task；
- CPU、ROCm/DCU、Metal、PPU 只能继续堆平行实现；
- fusion 和 distributed legality 没有正式 IR 承载。

另一方面，一开始实现所有 dialect/pass/backend 也会使项目停留在架构图。

**结论：固定**

从第一天建立 out-of-tree MLIR project 和 dialect skeleton，但按 vertical slice progressive
lowering：

```text
Python typed capture
  → gf.domain
  → gf.iter
  → gf.kernel
  → provider input / LLVM

storage/distributed side:
  gf.storage + gf.task
```

M0 允许简单 pass 和 naive skeleton，但 source-of-truth 必须是 typed IR，不能是 Python/字符串
模板。

---

## T07：Reducer 从 `sum` 枚举升级为可编译 UDF kernel

**问题**

Reduce 能否表示 softmax，尤其 online softmax？更复杂 reducer 是否只能加 builtin？

**早期错误方向**

`SumReducer.reduce()` 直接调用 `torch.index_add_`。这只是 reference evaluator，不是 compiler
abstraction，也无法表达 online softmax 的多状态合并。

**新语义**

Reducer 被定义为四个可继承、可捕获、可编译的 region：

```text
identity()               -> state
lift(message)            -> state
combine(left, right)     -> state
finalize(state)          -> result
```

它必须声明 associativity/commutativity/determinism 等性质。State 可以是 tuple，例如稳定 online
softmax 的 `(max, denominator, numerator)`。

**验证**

- scalar sum 由 region 结构识别，不依赖类名；
- online softmax 使用稳定 max-correction 合并；
- 用户 tuple-state mean 可通过 native `gf.reducer` 四个 regions round-trip；
- reducer region ABI verifier 检查 message/state/result 类型。

**结论：固定，scalar 路径已落地**

Reducer 是一等 kernel algebra。Builtin 只是便捷构造器；用户 UDF 与 builtin 进入同一 IR。
结构可证明时进入优化 schedule，不能证明时进入 correctness-first generic lowering 或明确
unsupported。

---

## T08：hierarchical memory 与 pipeline 不能靠用户在业务代码里硬写

**被质疑的 API**

```python
gf.Placement(home="ram", cache=("hbm",), spill="nvme")
gf.Schedule().tile("dst", 4096).cache(..., space="shared").pipeline(...)
```

**问题**

这种写法把一个 target 的物理决策混入算法语义。相同代码换到 CPU、不同 GPU 或 distributed
机器时，`shared`、tile size、stage count 可能完全不适用。

**Taichi/Triton 带来的修正**

用户应通过细粒度访问模式让 compiler 看见 reuse/lifetime；compiler 自动 lower 到 register、
shared/LDS、HBM 或 CPU loop。粗粒度 frontend 看不出足够信息时，可以使用 policy/hint，但它
不能成为 correctness 语义。

**最终模型**

- 语义层：Field/Relation/Effect/version；
- storage IR：Region、PhysicalInstance、Residency、Transfer、Event；
- kernel planning：tile、promotion、buffer、pipeline；
- target legalization：register/shared/LDS/HBM/RAM/NVMe/P2P；
- 用户只在必须时声明 capacity/persistence/external ownership 等约束。

**结论：固定，IR 骨架已落地，完整执行待完成**

层级存储和 pipeline 必须进入 compiler contract，但默认由 compiler 决策。不能把所有 placement
细节都暴露成业务代码里的 chained schedule。

---

## T09：冻结第一版 public abstraction——coarse MessagePassing + Static/Dynamic Graph

**讨论结果**

项目首先只提供 graph/relation 接口，允许粗粒度 message/reducer/node 描述；compiler 自动
lower 成 GPU tile、dense contraction、CPU loop 或 library call。细粒度语言暂不定型。

**第一版范围**

- Static external CSR/COO；
- Dynamic/procedural radius relation；
- implicit dense Cartesian relation；
- sum reducer 和可扩展 UDF reducer；
- optional node；
- CPU/reference 与 CUDA vertical slice。

**结论：固定**

先实现正确、可测、可检查的 coarse compiler，然后只为已经观察到的性能瓶颈增加细粒度表达。

---

## T10：Lazy JIT 必须由调用触发，并提供完整 artifact inspection

**问题**

为什么需要用户显式 `gf.compile(Diffusion(), graph=...)`？既然叫 JIT，继承 Kernel 后调用时
就应自动编译。

**结论**

Public usage 改为：

```python
program = Diffusion()
out = program(graph=graph, src=..., dst=..., edge=...)
```

第一次调用根据 UDF、Graph schema、dtype/shape、target/provider 和 runtime guards 产生 variant；
后续命中 cache。显式 compile 只保留为 ahead-of-time/prewarm/tooling 能力，不是普通用法。

**调试契约**

```python
program.explain()
program.ir("domain")
program.ir("iter")
program.ir("kernel")
program.ir("task")
program.ir("gf.kernel.ttir")
program.code("ttgir" | "llir" | "ptx")
program.cache_info
```

**状态：已落地**

`Kernel/CompiledVariant` 已保存 pass、provider identity、remarks 与 artifacts。尚未生成的 artifact
必须抛出清晰错误，不能伪造。

---

## T11：Graph 的 Static/Dynamic 区分从“类名”改为 topology lifecycle

**问题**

Graph 如何定义？怎样区分 static 与 dynamic？`Graph.radius` 什么时候 build？

**初始候选**

- `StaticGraph` 与 `DynamicGraph` 两套互不相干的类；
- `Graph.radius()` 立即生成 CSR；
- runtime 中始终保存完整 adjacency。

**问题分析**

同一个 radius graph 可以每步 rebuild，也可以 build 一次后冻结；一个 external CSR 也可能被
替换版本。真正影响编译的是 relation 的 origin、lifecycle、realization 和 version，而不是
Python 类名。

**最终 schema**

```text
origin: external | procedural
lifecycle: frozen | rebuildable
realization: materialized | generated | paged | distributed
version/snapshot
num_src / num_dst
ordering / uniqueness / degree bounds
```

**结论：固定**

Public 保持统一 `Graph`。Static/Dynamic 是 topology lifecycle；materialized/generated 是
physical realization，二者正交。

---

## T12：大 Radius/Adjacency 不应物化，Graph 是 relation handle 而非 CSR 容器

**问题**

Radius Graph 很大时不可能存进内存；能否像 FlashAttention 一样在 tile 内生成并消费邻居？
图若超过 HBM/RAM，甚至分布式，用户接口是否改变？

**结论形成**

`Graph.radius(...)` 表示可生成 relation，而不是承诺立即返回完整 `row_ptr/col_idx`。Compiler
可选择：

- all-pairs reference，仅用于小规模正确性；
- cell list/hash grid/BVH 构建 materialized CSR；
- generated tile，builder-consumer fusion，不生成全图；
- paged tile stream，从 RAM/NVMe 读取；
- partition-local relation + halo/remote partial combine。

**结论：固定，default Euclidean generated vertical slice 已落地**

用户始终操作同一个 Graph handle；residency 是 physical instance 的属性。SSD-backed graph
不应使用一套完全不同的计算 API。

---

## T13：Stencil 必须表示方向、边界和任意 tiling，而不是只列几个 case

**问题演化**

最初用 `Port(offset=(-1,0), coefficient=-1)` 表示 x 导数，随后遇到：

- y/z 方向；
- 六方密铺；
- 非正交、非规则 tiling；
- nonlinear coefficient、material tensor、geometry-dependent flux；
- unstructured mesh。

**被否决的方向**

- 为 Cartesian/hex 分别堆特殊 Graph class；
- 把 derivative coefficient 固定在 Graph 中；
- 让 edge 和 node 重复相同公式或缩放；
- 用晦涩的 `IndexedComplex(generators, sites, relations)` 直接作为主 public API。

**最终认识**

- relation 描述 topology/incidence/geometry/port orientation；
- field 和 UDF 描述随状态变化的 nonlinear coefficient；
- structured grid 可用 offset relation；
- arbitrary tiling/unstructured mesh 使用 entities + incidence + geometry；
- local frame、normal、displacement、measure 是 derived edge/port field；
- 图示和 inspect 工具是复杂 topology API 的必要组成。

**结论：固定，完整 mesh frontend 待完成**

不能用有限枚举穷举平铺类型，也不能把物理公式塞进 Graph topology。Graph 提供关系和几何，
UDF 提供计算。

---

## T14：消除 message/update、edge/node 的语义重复

**问题**

早期示例同时出现 `message()`、`aggregate()`、`update()`，又出现 `edge()`、`node()`；部分
示例把简单缩放分别写在 edge 和 node，造成概念重复。

**统一语义**

```text
message[e]  = edge(src_snapshot, dst_snapshot, edge_fields, params)
aggregate[d] = reducer(messages_to(d))
output[d]   = node(dst_snapshot[d], aggregate[d], params)  # optional
            = aggregate[d]                                 # omitted
```

Reducer 自己拥有 identity/lift/combine/finalize。`node()` 只在确实需要 old-state update、source
term 或 post-reduction nonlinear transform 时出现。

**结论：固定**

Public semantic regions 只有必需的 `edge()` 和可选 `node()`；不再同时提供同义的
message/update 命名。`node()` 每个 destination 只执行一次，并且必须在全部 partial state
combine/finalize 后执行。

---

## T15：复杂 PDE program 是多个 relation kernel 的 SSA program，并应自动融合

**问题**

Euler/FVM 可能需要 reconstruction、Riemann flux、divergence、source。`assemble` 是否是另一个
特殊 API？多个 `MessagePassing` 调用是否自动 fuse？

**分析**

`assemble` 不是新的基础语义。有限元/有限体积 assembly 可表示为 element/face/cell incidence
上的 message + reducer。复杂算法由多个 relation kernel 组成 GraphProgram SSA：

```text
reconstruct → flux → divergence → source/update
```

Fusion 不能简单“全部合成一个 kernel”。必须检查 relation equivalence、snapshot version、
effect、alias、reducer reassociation、node-once、working set 和 distributed barrier。

**结论：固定，基础 fusion analysis 已落地**

GraphProgram 默认 lazy composition，compiler 自动形成 fusion candidate。Fusion 被分解为 effect/
version canonicalization、legality analysis、group formation、region fusion、physical planning 和
profitability，而不是一个 `autofuse=True` 开关或单个 mega-pass。

---

## T16：Graph version 使用不可变 snapshot，而不是 `replace_csr` 原地替换

**问题**

为什么要 `graph.replace_csr(new_row_ptr, new_col_idx)`，而不是直接 new 一个 graph，再释放旧
对象？

**分析**

真正需要保留的不是 Python object mutation，而是 compiled program 对 relation snapshot 的
版本依赖。原地替换容易使旧 executable、异步 kernel、distributed task 观察到错误 topology。

**结论：固定**

Topology update 产生新 logical snapshot/version；旧 snapshot 在仍有引用/Event 时继续有效。
Runtime 可以在证明安全后复用 physical buffer，但 public semantics 不依赖原地 mutation。

---

## T17：超大图的 native 使用方式与内存图一致

**被质疑的 API**

```python
graph = gf.Graph.open("dataset.gfg")
```

单独的 `open()` 容易让人误解 SSD graph 是另一种算法对象。

**结论**

Graph 是 logical handle；文件、RAM、HBM、NVMe、remote shard 都是其 PhysicalInstance。用户仍然
写同一个：

```python
out = Program()(graph=graph, src=..., dst=...)
```

Loader/constructor 可以从文件产生 handle，但不改变 MessagePassing 接口。Compiler/runtime
通过 Region、Residency、Transfer、Event 和 bounded tile stream 决定数据移动。

**状态更新（2026-08-13）：IR contract 与 NVMe execution 均已落地**

`HierarchyRuntime` 已执行 capacity/version accounting、RAM↔NVMe spill、pinned↔HBM
async DMA，并由 `gf_storage.transfer/release` 的 bundle plan 驱动；正式 artifact 位于
`output/memory_hierarchy/hbm_pinned_nvme/`。

---

## T18：建立 correctness、performance、roofline 与 SOTA 四层验证

**要求**

实现第一版之后必须与开源 SOTA 比；若没有合适 SOTA，则与手写 CUDA/Triton 对比。所有 operation
都应有独立 roofline，而不是只测 weighted aggregation。

**验证分层**

1. semantic/reference correctness；
2. generated/native executable correctness；
3. warm kernel、cold JIT、ready-to-result、peak memory；
4. matched semantics 下与最快可运行 peer 的统计 gate。

**Benchmark families**

- sparse compute：weighted aggregation、SpMV/SpMM、diffusion、fusion；
- graph operations：radius build、build+consume、kNN；
- neural networks：dense/linear/sparse attention；
- compiler：JIT/cache/provider translation；
- memory hierarchy 与 distributed：能力落地后加入。

**结论：固定**

每个 operation 使用独立目录：

```text
output/roofline/<operation>/<case>/
```

主 roofline 中同一 logical operation 的 provider 必须使用相同 x-axis FLOP/byte 模型。不能把
build、consume、materialized、generated 的不同语义点混在一张图上，也不能用 reference 点冒充
`auto`。

---

## T19：第一次 roofline 暴露“性能完全不行”与图表语义问题

**观察**

最初 weighted aggregation roofline 中 Tiga 明显慢于 `torch.sparse.mm`；radius build 与
aggregation 图上还出现 roofline 下方大片空白、不同 provider x-axis 不一致、标签拥挤等问题。

**根因分类**

- reference/scatter path 被误当成 compiler result；
- materialized build+consume 与 precomputed consume 混在一起；
- arithmetic intensity 对不同 provider 使用了不同 byte 定义；
- warm execution 与 compile/build 时间混淆；
- 小 workload 受 launch overhead 支配，却用峰值 roof 解释；
- output 中残留重复、无用历史文件。

**修正**

- 明确 `auto` 是 planner 选中的 executable，`reference` 只定义语义；
- 每个 operation/case 单独存 JSON、roofline、latency、report；
- full semantic roofline 与 provider operational roofline 分开；
- 所有 peer 使用同一 logical FLOP/ideal-byte x-axis；
- 结果记录 hot/cold cache、confidence interval 和 skipped baseline 原因；
- 清理 output，只保留规范 artifact。

**结论：固定**

Roofline 是诊断工具，不是装饰图。图上空白本身不一定是 bug，但 x-axis 不同、operation 不匹配
或把 build cost 隐藏掉是方法错误。

---

## T20：Backend 从 `triton.py` 改为 Tiga Kernel IR + provider boundary

**争议**

为什么 backend 文件叫 `triton.py`？为什么存在 CUDA backend，而不是统一 lower TTIR，再使用各
厂商 Triton fork 从 TTIR 到 assembly？

**被否决的架构**

```text
MessagePassing → workload-specific @triton.jit Python function
```

它无法证明自己是 compiler lowering，也无法支持没有兼容 Triton 的硬件。

**最终架构**

```text
gf.domain → gf.iter → gf.kernel
                         ├→ versioned serialized TTIR provider
                         │    ├→ NVIDIA TTGIR/LLVM/PTX/cubin
                         │    └→ AMD/厂商 fork 的目标 IR/binary
                         ├→ LLVM CPU
                         ├→ Metal/MSL provider
                         └→ vendor-specific PPU provider
```

**关键限制**

TTIR/TTGIR 是 Triton 内部 dialect，不保证跨版本/厂商稳定。Tiga 的稳定边界是
`gf.kernel`；`gf.kernel → TTIR` 位于 provider plugin，并把 provider version/target 纳入 cache
key。Tiga 固定 MLIR 与 vendor Triton MLIR 可能 ABI 不兼容，因此二者通过 serialized
TTIR/worker 隔离，而不是链接进同一 `MLIRContext`。

**结论：固定，NVIDIA vertical slice 已落地**

“多 backend”不能靠把 CUDA 改名为 TTIR 来宣称。ROCm/DCU、Metal、PPU 只有在真机 correctness、
artifact、JIT、roofline 和 conformance 通过后才算支持。

---

## T21：禁止 workload 特调进入 compiler source

**触发问题**

源码中出现 `generated_radius_distance_sum_domain_mlir`、attention/diffusion 特定函数和大量
Triton kernel，违背“Tiga 只是 compiler”的定位。

**边界重申**

- `python/tiga`：typed frontend、compiler/runtime/provider adapter；
- `lib/`：dialect、analysis、passes、IR translation；
- `examples/`：用户算法；
- `benchmarks/kernels/`：手写 CUDA/Triton 性能 oracle；
- 不允许 Tiga runtime import benchmark oracle。

Compiler 可以有结构化 pattern，例如“scalar multiply + additive reducer”“stable streaming tuple
algebra”“dense contraction”，但不能匹配 `Diffusion`、`Attention`、字段名或 workload 名。

**结论：固定，持续通过 source scan 审核**

优化以 op/type/region/algebra/shape proof 为键，不以 Python 类名和算法名称为键。

---

## T22：DenseGraph 作为 DynamicGraph 的 implicit extreme，并验证 FlashAttention

**动机**

如果 DynamicGraph 可以边生成边消费，那么 complete Cartesian relation 应是最简单的 implicit
Graph。Dense attention 可用它检验：compiler 是否真的能保留 contraction 与 online reducer，
而不是物化 `N²` edge/score。

**设计**

```python
class DenseAttention(gf.MessagePassing):
    reducer = gf.online_softmax()

    def edge(self, src, dst, edge, scale):
        score = (src.key * dst.query).sum(dim=-1) * scale
        return self.reducer(score, src.value)
```

**验证**

- `gf.cartesian → gf_iter.traverse → gf_kernel.dense_launch`；
- reducer tuple state 在 tile 内维护；
- provider TTIR 使用 contraction/dot 与 stable streaming update；
- 不生成 pairwise score tensor；
- 与 Torch SDPA 做 correctness/performance 对照。

**结论：已落地，覆盖仍有限**

Tiga core 中没有 `attention()` 算子。Dense attention 只是用户程序和 benchmark。FLA/FSA
应作为 separate matched workloads 放在 examples/benchmarks，不能加入 core 源码。

---

## T23：项目定位收缩为 compiler-centered execution system

**问题**

若要替代 Torch，是否还要实现 `nn`、optimizer、dataset？

**结论形成**

Tiga 需要最小 runtime 承载 compiler artifact，并需要 Tensor/autograd 表达输入、views、
broadcast 和梯度；但复制完整训练框架会稀释 compiler 主线。

**最终范围**

- 实现 `gf.Tensor`、Buffer/Device/Stream/Event、basic autograd；
- 支持 reshape、permute、broadcast、axis reduction、complex dtype；
- Torch/DLPack/external buffer interop；
- 暂不实现 optimizer、dataset、通用 `nn` package。

**结论：固定，Tensor/部分 autograd 已落地**

Tiga 是 compiler，加上承载 compiler 的最小 execution substrate，不是另一个 PyTorch。

---

## T24：CPU 不能生成字符串 C/C++，必须走 MLIR→LLVM

**触发问题**

`cpu_tensor.py`、`mlir_bridge.py`、`tensor_mlir.py` 曾出现 source string emission，看不出真正
compiler pipeline。

**修正架构**

```text
Tensor DAG
  → native C++ OpBuilder 构造 gf_tensor
  → canonicalization/VJP
  → SCF/MemRef
  → LLVM dialect
  → MLIR ExecutionEngine
  → CPU machine code
```

Python 文件只负责 typed descriptor、runtime binding 和 inspection，不拼 MLIR/C/C++ source。

**验证：已落地**

- native extension 构造并验证 Tensor IR；
- CPU executable 在进程内编译执行；
- complex、view、broadcast、reshape、reduction 和 VJP 有回归；
- core 无 Torch import；
- source scan 检查 Python compiler 中无 MLIR module 模板。

---

## T25：源码结构按 compiler concepts 拆分，而不是堆在少数文件

**要求**

Graph、Reducer、MessagePassing、Tensor、Autograd、Distributed、Kernel 应各自成为清晰目录；
benchmark 按 workload 分类，不按 Triton/Torch provider 分类。

**结论：已落地**

当前 Python package 分为 `graph/`、`reducer/`、`message_passing/`、`tensor/`、`autograd/`、
`distributed/`、`kernel/`、`compiler/`、`runtime/`、`interop/torch/`、`codegen/`。Benchmarks
分为 sparse compute、graph operations、neural networks、compiler、memory hierarchy、distributed
和 handwritten oracle。

拆目录不是目的；每个目录必须对应明确 IR/runtime 责任，不能把 Torch oracle 换个位置后继续
冒充 compiler。

---

## T26：不均衡 Graph 的 load balance 必须是 compiler partition/schedule 问题

**问题**

Graph degree 极不均衡时如何 load balance？是否需要 partition？

**策略空间**

- one-row-per-program：适合中等规则 degree；
- multi-row tile：适合大量小 degree row；
- split-row/segmented reduction：适合超大 hub；
- edge-balanced partition：均匀分 edge，再 combine destination partial；
- degree bucketing：small/medium/large row 使用不同 variant；
- reorder/locality partition：改善 cache/coalescing；
- distributed vertex/edge cut，owner 完成 reducer/node。

**正确性约束**

split-row 只有在 reducer `combine` 合法时成立；deterministic/strict float 模式不能擅自改变顺序；
node 必须由 destination owner 在所有 partial 到齐后只执行一次。

**结论：固定，基础 degree proof 已落地，完整 load-balance 待完成**

Partition 不应成为用户手写 kernel loop。Compiler 根据 degree histogram、state size、target 和
distributed ownership 选择，并保留 runtime guard/fallback。

---

## T27：Distributed 使用 DTensor-like placement，但通信 primitive 在 Kernel/Task 底层

**诉求**

用户不应手工调用 send/recv；最好直接对 Graph 声明 halo，类似 Torch DTensor。

**Public 模型**

Graph placement 声明 mesh、partition、ownership 和 halo depth；MessagePassing 代码保持不变。

**Compiler lowering**

```text
partition/owner map
  → halo_pack
  → halo_exchange
  → halo_unpack
  → interior apply  ─┐
  → boundary apply  ─┴→ partial combine → owner node
```

通信、Event、stream dependency 和 overlap 是 `gf.task/gf.storage` primitive，不是用户 UDF。

**验证：planning 已落地**

`Graph.halo()` 可生成可检查的 typed Task IR，包括 pack/exchange/unpack、interior/boundary overlap
和 join。Snapshot-version verifier 检查跨 storage/task 的一致性。

**结论更新（2026-08-13）：core neighbor execution 已落地，vendor transport 待外部 provider**

真实 owner/ghost map、byte sizing、pack/exchange/unpack bundle lowering、bundle executable
resolver 和 Torch-free 两进程 transport 已执行。普通 distributed MessagePassing 仍明确拒绝
单 shard 语义，直到 sharded
Tensor 自动绑定和 NCCL/MPI/RCCL provider 完成。

---

## T28：2026-08-05——建立 query × cache benchmark 覆盖矩阵

**动作**

把 performance query 与 hot/cold cache、degree、feature、index dtype、topology/locality、build/
consume 生命周期组成矩阵，避免只测单一友好 case。

**关键决策**

- query 是 operation semantics，不是 provider 名；
- hot/cold working set 分别参考 L2/DRAM roof；
- compile time、graph build、consume、ready-to-result 分开；
- radius build、aggregation、build+consume 是不同 query；
- dense matmul calibration 只校准 roof，不是 sparse operation 的 SOTA baseline。

**结论：固定**

Benchmark registry 与 output contract 成为测试的一部分；缺失项显示 `PENDING/SKIP reason`，不补造
数据点。

---

## T29：2026-08-10——性能调优从 reference 转向 compiler-generated TTIR

**动作**

对 weighted CSR aggregation、radius build/consume、dense attention 等进行本机 RTX 5070 Ti
测量；输出迁移到 per-operation layout。

**主要观察**

- 原 reference/index_add 路径慢于 Torch sparse 是预期，不应作为 `auto` 性能；
- fixed/bounded-ragged CSR 需要 row×neighbor tile，而不是 scalar row loop；
- generated radius 只有在 builder 与 consumer 融合、避免 CSR/edge distance materialization 时
  才有明显价值；
- TTIR→provider binary 的冷编译远大于 warm execution，必须单独报告和缓存；
- 某些 bucket 超过本机 peer 不代表所有 topology/degree/feature 已达 SOTA。

**结论：实验结果进入性能基线**

任何“performance-ready”声明必须绑定具体 operation/case/hardware/provider/CI，不外推到未测
ROCm/DCU、periodic radius、skewed occupancy 或 distributed。

---

## T30：2026-08-12——全面 compiler audit，清除伪 compiler 路径

**触发**

对整个项目检查后发现：core 仍有 Torch dependency、Reducer 是 eager 实现、Graph/core 使用
Torch、CPU compiler 生成字符串、MLIR bridge 是文本模板、distributed IR 看不出执行作用。

**修正**

- Tensor frontend 改为 native C++ OpBuilder；
- CPU 走 `gf_tensor → SCF/MemRef → LLVM ExecutionEngine`；
- scalar CSR/dense/generated-radius/structured dense capture 改为 native OpBuilder；
- Reducer 四 regions 成为 typed symbol；
- Python Domain MLIR source template 被禁止；
- target-independent Domain→Iter→Kernel→Task passes 在同一 MLIRContext 内执行；
- 仅在 vendor TTIR boundary 序列化文本；
- core Torch import 收缩到 `interop/torch`；
- wheel 捆绑 native extension、tools、runtime、libMLIR/libLLVM 与相对 RPATH。

**验证**

- Python 与 MLIR lit 回归；
- strict docs build；
- source-boundary scans；
- clean venv、无 Torch wheel smoke；
- bundled dynamic library resolution；
- 本地 compatibility SDK 与正式 pinned release SDK 明确区分。

**结论：已落地**

Tiga 的 compiler claim 必须由 typed OpBuilder、verified IR、passes、provider artifacts 和
runtime execution共同支撑。文件名叫 `compiler` 或打印一段 MLIR 不构成 compiler。

---

## T31：Optional node 获得正式 ABI，不再靠后端猜参数位置

**问题**

Edge/reducer 已能进入 IR，但 optional node block 读取了哪些 destination field/param，后端没有
显式映射，无法安全生成 code。

**修正**

增加：

```text
node_input_indices
node_input_segment_sizes
region_kinds = [edge, optional-node]
```

属性从 Domain 经过 Iter 保留到 Kernel；verifier 检查 index 范围、segment 覆盖、node argument
顺序和 reducer finalized result 类型。Fusion 合并 inputs 时必须重映射第二个 apply 的 node
indices。

**验证：已落地**

Dense diffusion 的 `edge → sum → node` 在 GPU provider 实际执行，与 tensor 公式最大误差约
FP32 rounding 量级。Negative lit 覆盖越界 index 和 reducer/node ABI mismatch。

**结论**

Node input mapping 是 semantic ABI，不是 codegen 临时猜测。它同样是未来 fusion、distributed
owner-node 和 bufferization 的必要信息。

---

## T32：Generic scalar Reducer 从“只能 inspect”进入真实 TTIR execution

**问题**

用户 tuple-state reducer 虽能进入 Domain/Iter/Kernel，却只有 online-softmax/weighted-sum 特定
translator 能生成 TTIR。

**实现策略**

建立结构化 scalar algebra emitter，解释 edge、identity、lift、combine、finalize 和 optional
node 中的 FP32 `arith`/`math.exp` operations。它不读取 Python reducer 名或 workload 名。

**覆盖**

- implicit dense scalar reducer；
- materialized CSR scalar reducer；
- tuple state，例如 mean `(sum,count)`；
- runtime scalar params；
- optional destination node update；
- unsupported op/type 明确报错。

**验证：已落地**

- vendor Triton 实际生成 TTGIR/LLVM IR/PTX/cubin；
- dense tuple mean 与 `torch.mean` 一致；
- dense/CSR edge→reduce→node 正确；
- public lazy JIT 第一次编译，第二次 executable cache hit；
- scalar param 值变化复用同一 variant 且结果正确；
- 新增 `examples/custom_reducer.py`。

**结论**

Compiler 形成两级策略：generic correctness lowering 保证 UDF 可执行；algebra/shape proof 再选择
高性能 schedule。不能为了优化覆盖率而拒绝所有非 builtin reducer，也不能把 serial fallback
宣传为 SOTA。

---

## T33：2026-08-12——本地 wheel 与测试闭环

**验证结果（当时检查点）**

- Python suite：发现 87 项并通过，其中 2 项按环境跳过；
- MLIR lit：23/23；
- MkDocs strict：通过；
- binary wheel：约 65 MiB；
- 全新 venv、无 Torch：native Tensor、Reducer MLIR、`gf-opt`、`gf-translate` smoke 通过；
- core source scan：Torch import 仅位于 `interop/torch`；
- Python compiler source scan：无 Domain/Kernel MLIR 字符串模板。

**限制**

本地 wheel 使用 MLIR 20 compatibility SDK 并显式关闭 strict version check，只证明 packaging
链正确。正式 release contract 仍是 pinned LLVM/MLIR 22.1.8 的 manylinux/macOS CI，且需完成
auditwheel、trusted publishing 和目标平台 conformance。

**结论：本地验证完成，正式发布待完成**

---

## T34：最近一次 additive-tile 性能实验——有收益，但尚未合入稳定 source

**动机**

Generic CSR reducer 采用 one-row serial loop，只保证正确性。目标是从 reducer regions 证明
zero-identity、逐分量 additive tuple state，然后在 bounded degree 上对 neighbor tile 使用
`tt.reduce`，同时保留任意 edge/lift/finalize/node algebra。

**实验观察**

在一版临时 additive-tile binary 上，RTX 5070 Ti、regular/random CSR、FP32 diffusion 得到：

- quick `N=8192, degree=16`：Tiga 约 0.026 ms，快于当次手写 Triton CSR；
- `N=131072` degree scan：degree 32/64 达到或超过当次最快 peer；
- degree 2–16 明显落后，说明 one-row-per-program 在小 degree 下 occupancy 不足，需要 multi-row
  tile；
- roof utilization 仍不高，不能外推为通用 SOTA。

**为什么没有记为已落地**

实验期间共享工作区被另一个长期运行进程同时改写，出现 binary 含优化而 source 中对应 emitter
缺失的 source-of-truth 不一致。稳定源码随后只保留 generic serial path。因此该测量只能作为
调度方向证据，不能作为当前 Tiga capability 或正式 benchmark artifact。

**结论：实验，未合入**

下一次实现必须：

1. 先把 reducer algebra proof 做成独立、可测试 analysis；
2. additive tiled emitter 以 source + lit + provider compile + public runtime 四者同时提交；
3. degree 2–16 使用 multi-row tile，degree 32–64 使用 row×neighbor tile，大 hub 使用 split-row；
4. 每个 degree bucket 重新跑 matched hot/cold roofline 和 confidence gate；
5. 在 source 与 build artifact hash 一致后才更新 `PROJECT.md` 性能状态。

---

## 当前固定架构（时间线汇总）

```text
User program
  ├─ gf.Tensor expressions
  └─ MessagePassing(edge, Reducer, optional node) + Graph relation
                         │
                         ▼
Native typed capture / C++ OpBuilder
  ├─ gf_tensor
  └─ gf.domain: relation + apply + reducer regions + effects/version
                         │
                         ▼
gf.iter: coordinate hierarchy / ordering / relation lifecycle
                         │
                         ▼
gf.kernel: traversal skeleton + schedule proof + local regions
  ├─ LLVM CPU path
  ├─ versioned serialized TTIR provider → TTGIR/LLVM/PTX/cubin
  ├─ future Metal/MSL provider
  └─ future ROCm/DCU/PPU providers

Orthogonal planning:
  gf.storage: Region / PhysicalInstance / Transfer / Event
  gf.task: Partition / Halo / Launch / Collective / overlap
```

核心边界：

- Tiga 是 compiler-centered execution system，不是算法/kernel library；
- Graph/relation 是主领域语义，但不是唯一底层计算 primitive；
- Reducer 是可继承、可编译 algebra kernel；
- Torch 是可选 adapter；
- TTIR 是版本化 provider input，不是稳定公共 IR；
- hierarchical memory、pipeline、partition、halo 由 compiler/runtime 规划；
- 用户默认只写 coarse semantics；细粒度 DSL 必须由已测性能需求驱动；
- 每项性能结论绑定 matched operation、case、硬件、provider 和统计证据。

---

## 仍未完成的关键决策/工程闭环

按当前优先级：

1. 把 reducer algebra recognition 从 translator helper 提升为正式 analysis op/pass；
2. 完成 generic scalar tuple reducer 的 multi-row/bounded/split-row GPU schedules；
3. 扩展 vector message/state、多输出 product reducer 和 fused apply；
4. 为 relation kernel 建立真正的 CPU vector/parallel LLVM lowering；
5. 完成 CUDA Driver runtime plugin，减少对 Torch-owned launch/allocation 的依赖；
6. 实现真实 hierarchical storage capacity/liveness、RAM/NVMe tile streaming；
7. 实现 owner/ghost map、halo transport、collective 和多进程 task runtime；
8. 在真机上建立 ROCm/DCU、Metal、PPU provider 与 conformance；
9. 完成 per-operation accuracy/performance/roofline 矩阵，补 kNN、vector SpMM、FLA/FSA；
10. 用 pinned LLVM/MLIR 22.1.8 建立 manylinux/macOS wheel CI 和 PyPI trusted publishing；
11. 只有在上述 profiling 反复暴露同一调度缺口后，再决定 fine-grained DSL 的第一组 public
    primitives。

这份时间线保留“为什么变成现在这样”；任何新决策应先更新 `PROJECT.md` 的现行规范，再在本文
末尾追加带验证证据的新时间节点。

---

## T35：2026-08-15——删除 LinearOperator：solver 是 sugar，不是 core 接口

**动机**

`gf.linalg.LinearOperator` 把 MessagePassing 包一层 `(shape, matvec=lambda ...)` 才能交给
solver。但 bound 到 Graph 的 MessagePassing kernel 本身就是 matrix-free linear operator：
shape 可从 graph schema/rhs 推导，`symmetric` 只是自我声明而非验证，`parameters` 元数据
与 capture 已有的 Tensor 依赖重复。包装类既不给 compiler 新信息，又强迫用户多写一层
lambda。同时 `PROJECT.md` §1.1 早已把 “solver 等算法类” 划在 core 之外，`gf.linalg`
住在 core 里与此矛盾。

**决策**

明确 primitive / grammar-sugar 分层：

- primitive（有 IR 语义）：`gf_tensor`、`gf_control.repeat/while`、relation apply、reducer
  regions、VJP transform；
- sugar（纯 Python 组合 primitive，放 `examples/`）：solver。`gf.linalg` 整个从
  core 删除，`dot/vector_norm/richardson/cg` 移入 `examples/solvers.py`。
  `PROJECT.md` §1.1 明确 sugar 层规则：不引入新 IR 语义、不被 core import、不进
  core wheel、不占 core gate，性能声明挂在算法自身；sugar 同时是 primitive
  surface 的 dogfooding——算法层若必须绕过公开 surface，说明 primitive 有缺口，应补
  primitive 而不是让 sugar 依赖内部 API。（曾短暂落在 `extensions/gfext/`，评估后
  认为单独一层不必要，`examples/` 即正确位置。）

Solver 直接消费 MessagePassing 实例：`cg(kernel, rhs, graph=..., field="u", edge=...,
params=...)`，`field` 命名未知量字段，每次迭代绑定 `src/dst`；常量 edge fields 与 UDF
params 一次绑定；纯 Tensor 代数（对角 scaling、Jacobi）用 plain `Tensor -> Tensor`
callable。`LinearOperator`、`rmatvec/adjoint_apply`、`parameters`、`symmetric` 标志全部
删除，不留 deprecated alias。未来若 distributed collective、pipelined CG 或 implicit VJP
需要算法结构作为调度/微分证据，才引入 retained `gf_linalg.solve` op 作为 primitive，
Python 表面仍是 thin staging。

**验证**

- `tests/python/test_solvers.py`（原 test_linalg.py）11 项通过：MessagePassing 直接作
  operator 的正/负绑定测试、callable 形式、fixed/tolerance CG 的 IR 断言不变；
- 全量 Python suite：235 passed、0 failed、10 subtests；
- `examples/fem_poisson.py` 与 `examples/meshfree_linear_solve.py` 直接运行数值正确
  （max error 9.2e-10 / residual norm 3.2e-7），`gf_control.repeat/while` capture 不变；
- `mkdocs build --strict` 通过；
- 全仓 `LinearOperator`/`gf.linalg` 引用清零（仅 L0 ledger 中以“无包装”措辞提及）。

**结论：已落地**

---

## T36：2026-08-15——class 形式的 bounded control

**动机**

`gf.repeat/gf.while_loop` 的 lambda 形式与项目自身惯例不一致：§2.3 早已规定“class 便于
命名、复用、检查 IR 和持有 specialization，lambda 只是匿名 shorthand”。讨论中否定了
Taichi/AutoGraph 式 AST 捕获路线——它需要一个 Python 子集前端子系统（源码可得性、
闭包规则、诊断质量），却只换拼写；现有 proxy tracing 已足够表达循环契约，且 loop-carried
state 与 `max_iterations` 本来就是必须显式的资源/调度契约，不应由前端猜测。

**决策**

新增 `gf.control.Repeat` / `gf.control.While`：subclass override `body`（While 另有
`condition`），captured constants 是普通实例属性。class 是主拼写，functional
`repeat/while_loop` 保留为匿名 shorthand；两者经同一条 tracing 路径 lower 到同一个
`gf_control.repeat/while` op，零 IR 改动。`@gf.program` 边界的 AST 捕获仍保留为未来可选
语法糖，不是当前工作项。

**验证**

- `tests/python/test_solvers.py` 新增 class 形式测试：实例属性作为 region capture、
  与 functional 形式相同的 `gf_control.repeat/while` IR、CPU `scf.while` lowering 与
  `max_iterations` artifact 断言，未 override 时 fail-closed；12 项全部通过。

**结论：已落地**

---

## T37：2026-08-15——编排层 `@gf.jit` AST 子集

**动机**

`gf.repeat/gf.while_loop` 的三元组写法不够 Pythonic。讨论确认 AST 路线可以做，但边界
必须划清：AST 只作用于编排层（调用 MessagePassing/Tensor 代数的外层函数），永不进入
`edge()`/`node()` UDF region（那里维持 proxy tracing）。关键约束是 `max_iterations`
资源契约不能丢——Python 循环语法本身没有承载它的位置。

**决策**

- 新增 `@gf.jit` / `@gf.jit(max_iterations=k)`（`python/tiga/jit.py`）：capture 期
  AST 重写，`for i in range(k)` → `gf_control.repeat`，裸 `while cond:` →
  `gf_control.while`（上界取装饰器参数），`for` body 首句 `if cond: break` → 提前退出
  的 bounded while（上界取 range 长度）。循环变量 `i` 被读取时 desugar 为额外 carried
  rank-0 i64 state。
- loop-carried 变量由静态规则推导：body 内赋值 ∩ 循环前已定义，且必须是 Tensor；
  循环内新建变量是 body 局部值。`continue`、body 中部 `break`、`while True`、非 range
  迭代器、loop `else`、data-dependent `if` 全部 fail-closed，诊断带源码行号。
- 简单比较条件的 break-if 做结构取反（`<=` → `>`），保持 lowered IR 与手写 functional
  形式逐字一致；复杂条件回退 `!= True`。
- 闭包变量在装饰时按值快照进 staged globals；lambda/REPL/exec 无源码时报错。
- 顺带修复独立安全问题：`Tensor.__bool__` 现在 raise TypeError——此前
  `if tensor_scalar:` 会静默走真分支。

**验证**

- `tests/python/test_jit.py` 8 项：for/while/for+break 与 functional 形式 IR 等价
  （`num_carried`、op 计数）、carried 自动推导、循环变量 desugar、MessagePassing 在
  循环体内、8 类 fail-closed；
- `examples/solvers.py` 的 `cg`/`richardson` 改用 `@gf.jit` 自然循环写法后，
  `test_solvers.py` 12 项断言（含 `num_carried=4`、`scf.while`、`arith.cmpf ogt`、
  VJP 数值）一字未改全部通过——transform 不改变 IR；
- 全量 Python suite 通过；ruff 与 mkdocs strict 通过。

**结论：已落地**

## T38：2026-08-17——公开文档单一属主化、全英文化与 distributed 测试闭环

**动因**

文档站要对外公开。审计发现三类系统性问题：同一事实多页重复（benchmark 数字在
results/BENCHMARKS/dynamic-graphs 三处近乎逐字重复；provider 状态矩阵在四处重复；
policy 在五处重复）；六个页面全中文（BENCHMARKS/IR_DESIGN/RELATED_WORK/
SCHEDULING_ABSTRACTIONS/GPU_GRAPH_OPTIMIZATION/COMPILER_BOOTSTRAP）与英文公开站不一致；
免责声明密度超过正文。同时 distributed 只有手工 benchmark，没有可持续的测试闭环。

**决策**

- 单一属主：benchmark 数字只属于 `benchmark-results.md`；方法/协议合并为
  `performance.md`（"Methodology and protocol"）；provider/status 矩阵只属于
  `roadmap.md`；贡献政策集中于 `development.md`。`BENCHMARKS.md`（中文工作日志）
  删除，协议内容并入 `performance.md`；`COMPILER_BOOTSTRAP.md` 删除，独有内容
  （LLVM 22.1.8 pin 政策、`gf-translate` 进程边界理由、wheel 内容清单）并入
  `development.md`。
- 全英文化：`IR_DESIGN.md` 逐节翻译（内容不动）；`RELATED_WORK.md` /
  `SCHEDULING_ABSTRACTIONS.md` / `GPU_GRAPH_OPTIMIZATION.md` 翻译并删内部 onboarding
  材料；`memory-and-distributed.md` 残留中文句子译出。
- Examples/Benchmark 提升为顶部 tab；Examples 扩为总览 + 6 个主题页，22 个示例全部
  进文档；新增 `benchmark-suite.md` 导览。
- distributed 测试：新增 `test_distributed_partitions.py`（非均匀分区对称性、拓扑校验、
  pack/unpack round-trip，纯 stdlib）、`test_distributed_benchmark_smoke.py`（halo/MPI/NCCL
  loopback benchmark 冒烟 + 单 GPU 拒绝双 GPU gate 的负向测试 + env 门禁的 overlap 冒烟）；
  `test_distributed_runtime.py` 增加不规则图两进程 vs 单进程 reference 精确对照；两个
  torch-free 文件接入 CI torch-free job。
- `PROJECT.md` §0.1 重写为"实现状态总览"：已实现（带证据）/ 部分实现（fail-closed
  边界）/ 未实现 TODO / 距工业级的 14 项工程差距。

**验证**

- `mkdocs build --strict` 零警告；部署到公开站点后本机 Caddy 抽查各页 tab/链接一致。
- Python suite 252 passed、1 skipped（env 门禁）、10 subtests；新增 distributed 测试在本机
  零 skip 全过（mpiexec 两进程与 NCCL loopback 真实执行）。

**结论：已落地**
