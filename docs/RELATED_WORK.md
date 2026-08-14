# GraphForge Related Work

状态：首轮定向调研  
日期：2026-08-06

## 1. 调研目的

本调研不试图罗列所有 HPC、图计算和编译器项目，而是回答 GraphForge 当前最关键
的工程问题：

1. Relation 是否足以作为科学计算的核心数据模型？
2. 如何分离物理语义、数据结构、schedule 和 runtime？
3. 如何描述 sparse、structured、dynamic 和 hierarchical relation？
4. 如何在不同硬件上选择 traversal、layout 和 reduction？
5. distributed、memory hierarchy 和 overlap 应该属于哪一层？
6. GraphForge 与已有工作相比真正的新贡献是什么？

首轮结论：关系模型作为仿真 DSL 核心已有强先例，尤其是 Ebb 和 Simit。GraphForge
不能把“simulation as relations/message passing”本身当作唯一创新点，而应在这些工作
基础上解决它们没有联合解决的现代工程问题。

## 2. 最接近 GraphForge 的项目

### 2.1 Ebb

资料：

- [Ebb paper](https://arxiv.org/abs/1506.07577)
- [Stanford project page](https://graphics.stanford.edu/~mdfisher/ebb.html)

Ebb 是目前与 GraphForge 核心思想最接近的工作。它将系统分为三层：

```text
simulation code
domain/data-structure libraries
CPU/GPU runtime
```

不同几何域通过统一 relational data model 实现。计算代码与几何数据结构分离，runtime
负责 CPU/GPU 映射。论文在多个仿真上报告了接近手写 GPU code 的性能。

需要学习：

- relation、key、field、group 和 projection 的具体语义；
- domain library 如何在 relation model 上定义 grid、triangle mesh、tet mesh；
- 编译器如何从 field access 推导 race、locality 和 reduction；
- relation storage 如何与 simulation kernel 解耦；
- 为什么三层架构能够添加新 domain 而不扩大 runtime API。

对 GraphForge 的直接影响：

- `EntitySet + Relation + Field` 应作为 M0 核心，而不是只做 `CSRRelation` wrapper；
- Relation schema 必须表达 key uniqueness、arity、cardinality、inverse、grouping 和
  mutability；
- domain library 应建立在公共 Relation API 上，stencil/mesh/radius 不是编译器中的
  硬编码前端；
- field access/effect 是 schedule 和并行正确性的核心输入。

不能简单照搬：Ebb 的工作重点是当时的 CPU/GPU 仿真 DSL。GraphForge 还需要验证
动态图生成、现代 GPU tile/schedule、快速 Python JIT、多 backend capability、原生
distributed task IR 和 tiered storage。

### 2.2 Simit

资料：

- [Simit paper and project](https://simit-lang.org/tog16)
- [Simit language](https://simit-lang.org/language)
- [Getting started](https://simit-lang.org/getting-started)

Simit 使用 hypergraph 表示物理系统，同时允许用户使用全局 vector/matrix/tensor
语言。它的 assembly construct 建立局部图元素与全局线性代数之间的映射，编译器可
将某些全局操作转回 graph 上的 in-place computation，避免真正构造中间稀疏矩阵。

需要学习：

- vertex/edge set 和任意 arity hyperedge；
- graph field 与 system vector/matrix 的 type relation；
- assembly construct 如何避免 indexing bug，并为编译器保留 provenance；
- index expression fusion 与 in-place lowering；
- local graph computation 与 global algebra 的组合方式。

对 GraphForge 的直接影响：

- 普通 binary Message Passing 不够，IR 需要为 `HyperRelation` 预留多输入 message；
- 局部 relation program 与 global solver/library call 之间需要显式 assembly/view；
- matrix/tensor 不能只是失去来源信息的普通 tensor，应该知道它由哪个 relation
  assemble 而来；
- 调用 PETSc、FFT、BLAS 等外部库时仍应保留结果与 relation/entity 的映射。

### 2.3 Liszt 与 OP2

资料：

- [Liszt paper](https://graphics.stanford.edu/hackliszt/liszt_sc2011.pdf)
- [OP2 papers](https://op-dsl.github.io/papers.html)
- [OP2 documentation](https://op2-dsl.readthedocs.io/en/latest/)

Liszt 针对 mesh PDE，通过受限 mesh statements 暴露并行性、局部性和同步，并生成
cluster、SMP 与 GPU code。OP2 用 sets、maps、data 和 access descriptors 表达非
结构网格计算，再通过 source transformation/codegen 映射到多种 backend。

需要学习：

- set/map/dat 的最小 API；
- `READ/WRITE/RW/INC/MIN/MAX` 等 access descriptor；
- indirect increment 导致的 coloring、atomics 或 partition 策略；
- halo 和 distributed mesh partition 的生成方式；
- restricted DSL 如何换取可分析性。

对 GraphForge 的直接影响：M0 effect system 至少需要 `read`、`write`、`reduce(op)`；
不能只靠扫描表达式猜测所有 alias 和跨 field effect。

## 3. 图计算与 Message Passing 系统

### 3.1 GraphIt

资料：[GraphIt project and paper](https://graphit-lang.org/)。

GraphIt 最值得学习的是 algorithm language 与 schedule language 的分离。输入图的
大小与结构会改变 locality、work efficiency 和 parallelism 的权衡，因此 traversal
和 layout optimization 必须能够组合，而不是隐藏在一个固定 executor 中。

GraphForge 应学习：

- schedule 作为独立、可序列化对象；
- push/pull、edge/vertex traversal、frontier 和 direction switching；
- graph structure statistics 如何驱动 schedule；
- schedule transformation 的合法性检查。

GraphIt 面向 graph analytics；GraphForge 还需要 typed field、几何关系、连续数值
kernel、hyperrelation、时间演化和 global solver。

### 3.2 Gunrock

资料：[Gunrock paper](https://arxiv.org/abs/1701.01170)。

Gunrock 使用 vertex/edge frontier 的 data-centric GPU abstraction，说明 irregular
graph workload 不能只看静态 SpMV；active set、filter、advance、compute 等 frontier
primitive 对非均匀迭代十分重要。

GraphForge 后续应为 active relation subset/frontier 预留表示，但 M0 不需要实现完整
graph analytics runtime。

### 3.3 GraphBLAS

资料：[GraphBLAS specification](https://graphblas.org/graphblas-api-cpp/)。

GraphBLAS 用 semiring 定义广义 sparse matrix/vector operation。它为 Reducer 的
代数契约提供了成熟参照：identity、monoid、semiring、mask、accumulator 和 descriptor
需要清晰区分。

GraphForge 不应复制整个 GraphBLAS API，但 reducer/type verifier 应借鉴其严格性。

### 3.4 PyG 与 DGL

资料：

- [PyG MessagePassing API](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.MessagePassing.html)
- [DGL gspmm](https://www.dgl.ai/dgl_docs/en/1.1.x/generated/dgl.ops.gspmm.html)

PyG 的 `message/aggregate/update` 是良好的用户心智模型；DGL 将 message/reduce 归约
到 fused GSpMM/GSDDMM kernel，证明常见代数组合适合 template fusion。

局限是 adjacency 通常已经是显式 sparse assignment matrix，relation provenance、
几何 builder、hierarchy 和 solver/task semantics 不是其核心。GraphForge 可提供
PyTorch/PyG 互操作，但不能把 PyG execution model 当作 Domain IR。

## 4. Sparse 与 Structured Compiler

### 4.1 TACO

资料：[TACO publications](https://tensor-compiler.org/publications.html)。

TACO 的关键贡献是将 sparse format 描述成 per-dimension level 的组合，并使用
iteration graph/merge lattice 生成多格式 sparse tensor algebra。GraphForge 的
`ExplicitRelation` 物理格式不应使用 CSR/COO 枚举硬编码，而应借鉴 levelized format
capability：ordered、unique、compressed、singleton、dense、locate 等。

### 4.2 MLIR SparseTensor dialect

资料：[MLIR SparseTensor dialect](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/)。

该 dialect 已实现 level encoding、iteration graph、iteration lattice、iteration
space 与 sparse runtime bridge。GraphForge 不应重复实现通用 sparse tensor
iteration theory。

建议边界：

- `gf.domain` 保存 Entity/Relation/geometry/effect/provenance；
- 能表示为 tensor algebra 的部分 lower 到 `linalg + sparse_tensor`；
- 动态 geometric builder、frontier 和 distributed relation 保留在 GraphForge
  dialect/runtime；
- 实验验证 upstream SparseTensor 对 GPU irregular reduction 的性能与可扩展性后，
  再决定复用范围。

### 4.3 Finch

资料：

- [Finch repository](https://github.com/finch-tensor/Finch.jl)
- [Finch paper](https://arxiv.org/abs/2404.16730)

Finch 同时处理 sparse 和 structured arrays，包括 run-length、banded、triangular、
blocks、不同 background value 和 control flow。它使用高层 FinchLogic 做 fusion/
scheduling，再 lower 到更明确控制流的 FinchNotation。

需要学习：

- looplet/structured coiteration；
- format language；
- high-level logic IR 与 low-level control IR 的边界；
- algebraic zero/background value 如何消除工作；
- compiler code dump 和可调试性。

Finch 对 GraphForge `gf.iter` 的设计价值很高，尤其能避免把 structured relation
重新退化为 sparse coordinate list。

## 5. 科学计算 DSL

### 5.1 Taichi

资料：

- [Taichi paper](https://yuanming.taichi.graphics/publication/2019-taichi/)
- [Sparse data structures](https://docs.taichi-lang.org/docs/sparse)

Taichi 将 computation 与 hierarchical data structure 分离，通过可组合 SNode 表达
dense、pointer、bitmasked 等结构。这证明高层结构信息可用于自动生成 sparse traversal
和 memory maintenance。

需要学习：

- staged Python frontend 和 kernel specialization；
- hierarchical IR 与 offload/task splitting；
- SNode tree、activation 和 sparse iteration；
- JIT cache、diagnostics 和 framework interoperability。

同时应吸取 backend feature erosion 的教训：高层 sparse data structure 若对每个
backend 都要求大量特殊 runtime 支持，维护成本很高。GraphForge 的核心 Relation
能力必须分层，并允许 backend 明确拒绝或 fallback，不能宣称所有 target 等价。

### 5.2 Devito

资料：[Devito paper](https://arxiv.org/abs/1707.03776) 与
[project](https://www.devitoproject.org/index.html)。

Devito 从 symbolic PDE/finite-difference expression 生成优化 stencil code，说明
保留方程、时间维、空间 offset 和边界信息比提前转成 edge list 更有价值。

GraphForge 不一定承担 PDE 离散化，但 `AffineRelation` 应能作为 Devito/FEniCS 类
前端的低层目标，同时保留 stencil/time-step metadata。

### 5.3 NVIDIA Warp

资料：[Warp documentation](https://nvidia.github.io/warp/latest/index.html)。

Warp 展示了现代 Python simulation JIT 应具备的体验：typed kernel、CPU/GPU JIT、
geometry/physics primitives、PyTorch/JAX 互操作和 differentiability。

GraphForge 应借鉴它的错误信息、类型体验、cache 和 interop；区别在于 GraphForge
试图让编译器理解 Relation 与 schedule，而不是主要让用户手写 per-thread kernel。

## 6. Kernel DSL 与硬件可移植性

### 6.1 Triton

资料：[Triton programming model](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)。

Triton 的 blocked program 对规则 tile computation 很高效，适合作为首个 NVIDIA
backend 和部分 `gf.kernel` lowering。但它不是 GraphForge Domain IR，也不天然解决
动态图构建、分布式 task、storage hierarchy 和所有 irregular sparse schedule。

### 6.2 TileLang

资料：[TileLang paper](https://arxiv.org/abs/2504.17577)。

TileLang 强调 tile primitive、layout 和 pipeline 的可组合控制。GraphForge 后期的
expert `Schedule` 和 `gf.kernel` 应参考这种显式但可组合的设计，而不是不断增加
backend-specific decorator。

### 6.3 CAKE

资料：[CAKE: Compiler–Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/abs/2608.12629)。

CAKE 的核心不是再做一层高阶算子，而是让 agent 编辑 typed、hardware-explicit 的物理
schedule：named resources、warp roles、barrier、pipeline 和 instruction form 都可见；地址、
phase bit、descriptor encoding 等机械信息由 lowering 推导。其 verifier/cost model 返回定位到
具体 resource/role/stage 的结构化 finding，并把重复失败沉淀成 IR primitive、verifier rule、
cost calibration 和 corpus regression。它还明确把单 shape 搜索与 dispatcher portfolio
generalization 分成两个阶段，后者必须验证 guard overlap/gap、tail、held-out shape 和 fallback。

这直接加强 GraphForge 已有的 bottom-up fine-language gate：普通用户仍只写 coarse
`MessagePassing`；compiler 先自动产生 machine schedule。只有多个已实现优化反复需要相同的
resource/role/barrier/pipeline 决策时，才把它们从 `gf.kernel` 中提炼成 expert/agent 可变换的
schedule vocabulary。调试接口不应只返回 PTX 或一个 latency，而应返回带 source location、
legality disposition、resource estimate、bottleneck attribution 和建议修复位置的 finding。

不能直接照搬的部分同样重要。CAKE 当前是 NVIDIA Ampere–Blackwell 的单设备 schedule IR，
论文明确没有验证非 NVIDIA transfer cost；它不覆盖 GraphForge 的 Relation provenance、
generated topology、hierarchical remote storage 或 distributed task/event semantics。GraphForge
因此只借鉴其“显式但可验证的 machine schedule”和 harness 方法，不把 `mma/tmem/TMA` 等
厂商指令提升为跨厂商 semantic IR，也不要求模拟用户手写 memory placement。

### 6.3 Kokkos

资料：[Kokkos programming model](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ProgrammingModel.html)。

Kokkos 明确拆分 Execution Space、Execution Pattern、Execution Policy、Memory Space、
Memory Layout 和 Memory Trait。它是 GraphForge capability/machine model 的重要参考。

需要注意：性能可移植并不意味着同一 launch 参数或 layout 在所有 target 上最优。
GraphForge 应统一 semantic schedule vocabulary，由 target 独立选择具体参数。

## 7. 动态邻域与粒子系统

### 7.1 LAMMPS

资料：

- [Neighbor-list internals](https://docs.lammps.org/Developer_par_neigh.html)
- [Accelerator package options](https://docs.lammps.org/latest/package.html)

LAMMPS 的工程经验直接支持将 topology builder 和 reuse policy 设为一等 IR：

- Verlet list 使用 force cutoff 加 skin，复用若干 timestep 后重建；
- neighbor list 通常是最占内存的数据结构之一；
- CPU 常偏好 half list/Newton on；GPU 可能用 full list/Newton off，以重复计算换取
  thread safety、减少 atomics 或通信；
- GPU 可选择 device build、CPU build 或 hybrid build；
- transposed layout 有时更快，但转换会增加额外内存。

这意味着 `symmetric=True` 不能直接决定只存一半边。GraphForge schedule 必须能在
half+atomic/coloring 与 full+duplicated-compute 之间选择。

### 7.2 HOOMD-blue

资料：[HOOMD neighbor lists](https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/module-nlist.html)。

HOOMD 提供 Cell、Tree、Stencil 等不同 neighbor builder，说明 radius relation 的
物理实现必须可替换，并依据 density、cutoff ratio、box 和粒子分布调度。

## 8. Distributed 与异构 Runtime

### 8.1 Legion / Realm

资料：

- [Legion publications](https://legion.stanford.edu/publications/index.html)
- [Realm overview](https://legion.stanford.edu/realm/)

Legion 用 logical region 和 privilege 表达 locality、independence、partition 与 data
usage；Realm 使用分布式 event-based runtime，使 task、copy 和 synchronization 都
可异步组合。

GraphForge 应学习其逻辑/物理 instance 分离、region privilege、mapper 和 event，
但第一阶段不应自己实现完整 Legion。可行路线是让 `gf.task` 的语义足够清晰，后续
评估复用 Realm/Legion 或实现更小的 relation-specific runtime。

### 8.2 StarPU

资料：[StarPU project](https://starpu.gitlabpages.inria.fr/)。

StarPU 让应用提供 task 的 CPU/GPU implementation 和 constraint，runtime 管理依赖、
异构调度、数据复制、cluster communication 和异步执行。它说明 backend executable
variant 与 task/data runtime 可以正交。

对 GraphForge 的影响：`Executable` 不应只是一段同步 callable；它应具有输入输出
region、target requirement、estimated cost 和异步 Event。

## 9. 综合比较

| 项目 | 最强设计 | GraphForge 应复用的思想 | 尚未覆盖的 GraphForge 目标 |
|---|---|---|---|
| Ebb | relational simulation model | Entity/Relation/Field、domain library 分层 | 现代多级 IR、动态图、distributed/storage 全栈 |
| Simit | hypergraph + global algebra | HyperRelation、assembly provenance | 动态 topology、通用 kernel schedule/runtime |
| GraphIt | algorithm/schedule 分离 | 可组合 schedule、structure-aware choice | 连续数值、geometry、solver |
| TACO/MLIR Sparse | format/iteration theory | level format、iteration space/lattice | geometric builder、task/runtime |
| Finch | structured coiteration | structured format、logic/control IR 分层 | simulation entity/effect/distributed |
| Taichi | data structure/computation 分离 | staged frontend、hierarchical storage | relation/global solver/distributed semantics |
| Devito | symbolic stencil compiler | affine/time structure preservation | 非结构与动态图 |
| Warp | Python simulation kernel JIT | typed UX、interop、autodiff 路径 | relation-aware automatic scheduling |
| Triton/TileLang | GPU tile kernel DSL | 最终 kernel lowering、expert schedule | Domain/Task/Relation runtime |
| CAKE | agent-facing typed machine schedule、localized diagnostics、portfolio stage | 从生产 kernel bottom-up 提炼 resource/role/barrier/pipeline；compiler harness 可共同演化 | relation/dynamic graph、distributed/storage、非 NVIDIA portability |
| LAMMPS/HOOMD | dynamic neighbor engineering | builder/reuse/half-full policy | 通用 compiler IR |
| Legion/Realm | distributed region/event runtime | privilege、instance、mapper、overlap | relation-specific kernel generation |

## 10. 对 PROJECT.md 的具体修正

首轮调研后，以下设计应提升为明确工程要求：

1. M0 Relation 不能只有 CSR 参数，必须先定义逻辑 schema；
2. schema 增加 arity、key/cardinality、functional/multi、inverse、mutability、lifetime；
3. Field access 增加显式 `read/write/reduce` effect；
4. IR 为 hyperedge 和 relation-derived tensor/assembly 保留扩展点；
5. Schedule IR 参考 GraphIt，与 Domain IR 分开序列化；
6. ExplicitRelation format 后续对接 MLIR SparseTensor level，而不是自建完整 sparse
   iteration theory；
7. radius builder 必须同时支持 materialize、generate-consume 和 reuse/rebuild；
8. symmetric relation 不隐含 half storage，由 schedule 决定 full/half；
9. `Executable.launch()` 从一开始返回 Event，避免同步 API 锁死 distributed overlap；
10. GraphForge 的差异化表述应以 Ebb/Simit 为基线，而不是声称首次提出 relation-based
    simulation。

## 11. 必读顺序

### P0：开始 M0 前

1. Ebb：完整阅读 relation/data model 与 compiler/runtime sections；
2. Simit：完整阅读 hypergraph、assembly 和 in-place lowering；
3. GraphIt：schedule language 与 graph iteration space；
4. Finch/Looplets：structured format 和 coiteration；
5. MLIR SparseTensor：encoding、iteration space/lattice 和 lowering pipeline；
6. LAMMPS neighbor internals：skin、rebuild、half/full 与 device build；
7. Legion logical region/Realm event：只读核心模型，不先实现。

### P1：M1/M2 期间

- TACO format abstraction；
- OP2/Liszt access descriptor 和 partition；
- Triton MLIR dialect/lowering；
- TileLang schedule/layout；
- Kokkos execution/memory spaces；
- Gunrock frontier/load balance；
- Warp frontend/JIT/cache。

### P2：进入对应功能前

- Devito：`AffineRelation`；
- HOOMD/LAMMPS source：`GeometricRelation`；
- StarPU/Realm：`gf.task` runtime；
- GraphBLAS：完整 reducer/semiring 与 mask；
- PETSc/FEniCS：solver/assembly interoperability。

## 12. 需要用原型回答的问题

文献不能替代实验。M0/M1 应明确回答：

1. Ebb 风格 Relation schema 能否自然映射 PyTorch Tensor 且保持零拷贝？
2. restricted symbolic proxy 能否给出比 AST/bytecode tracing 更好的错误信息与缓存？
3. 三种不同 message algebra 能否复用同一个 edge/node skeleton，而不是重新手写
   Triton kernel？
4. `AffineRelation` 不物化 CSR 相比显式 CSR stencil 的内存和性能收益是多少？
5. static CSR 在 degree、feature width、skew 变化时，简单 cost model 是否能选择
   正确 schedule？
6. symmetric pair interaction 在 half+atomic 与 full+duplicate 之间的 crossover 在
   NVIDIA、CPU、DCU 上分别在哪里？
7. radius relation 何时应 materialize，何时应 generate-consume，Verlet list 重用多少
   step 才能 amortize build？
8. frontend、optimization、backend compile、cache lookup 各占多少 JIT latency？

只有这些问题获得数据后，才进入完整 MLIR、distributed 和更多 backend。

## 13. 调度抽象专项调研

关于 Halide/TVM/MLIR Linalg/CuTe 的 dense axis 与 layout、TACO/Finch/CoRa 的
sparse/ragged coordinate hierarchy、TileLang/NVGPU 的 pipeline，以及 GSPMD/MLIR
Shard 的 distributed sharding 比较，见
[`SCHEDULING_ABSTRACTIONS.md`](SCHEDULING_ABSTRACTIONS.md)。专项结论是：axis 是统一
调度入口，但 iteration、access/layout、execution、storage、pipeline 和 task placement
必须是可组合的独立映射，不能压成单个 `axis -> level` 属性。

GPU graph hardware mapping、subgraph/work-tile、neighbor×feature 二维划分、cache
blocking、persistent CTA 和 Dynamic RadiusGraph 的实施顺序见
[`GPU_GRAPH_OPTIMIZATION.md`](GPU_GRAPH_OPTIMIZATION.md)。该路线明确先实现 coarse
MessagePassing 的 naive Static/Dynamic backend，再由 profile 选择一项显著优化，最后
才决定细粒度 public language。
