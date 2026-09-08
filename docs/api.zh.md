# Python API { #python-api }

本页是刻意保持精简的公开 API 表面的平铺参考。概念性背景见各小节链接的
指南页；本页只列出签名、约束与返回值。

## Tensor 与 autograd { #tensor-and-autograd }

`float16/float32/float64`、`complex64/complex128`、`int32/int64` 与 `bool`
均为 Tiga 原生 dtype。只有浮点与[复数](https://baike.baidu.com/item/复数)值可以要求梯度。
广播遵循右对齐的 NumPy 语义。

### gf.tensor(data, *, dtype=None, device="cpu", requires_grad=False) { #gftensor }

从嵌套 Python 序列或标量创建一个由 Tiga 原生运行时支撑的 Tensor。
省略 `dtype` 时按数据推断。

### gf.empty(shape, *, dtype=gf.float32, device="cpu", requires_grad=False) { #gfempty }

分配一个未初始化的连续 Tensor。

### gf.zeros_like(value, *, requires_grad=False) { #gfzeros_like }

分配一个与 `value` 的形状、dtype 和设备相同的全零 Tensor。

### gf.ones_like(value, *, requires_grad=False) { #gfones_like }

分配一个与 `value` 的形状、dtype 和设备相同的全一 Tensor。

### gf.from_torch(value, *, requires_grad=None) { #gffrom_torch }

零拷贝包装一个稠密 Torch tensor。示例：
[可选的 Torch 互操作](examples/programs-and-interop.zh.md#optional-torch-interoperability)。

### Tensor.reshape(*shape) { #tensorreshape }

返回一个新形状的惰性视图；一个 `-1` 维度会被推断。

### Tensor.permute(*axes) { #tensorpermute }

返回轴重排后的惰性视图。

### Tensor.transpose(dim0, dim1) { #tensortranspose }

返回交换两个轴后的惰性视图。`Tensor.T` 反转全部轴。

### Tensor.squeeze(dim=None) { #tensorsqueeze }

移除长度为 1 的维度；给定 `dim` 时只作用于该维。

### Tensor.unsqueeze(dim) { #tensorunsqueeze }

在 `dim` 处插入一个长度为 1 的维度。

### Tensor.broadcast_to(shape) { #tensorbroadcast_to }

返回惰性广播视图。`Tensor.expand(*shape)` 是同一视图的变参写法。

### 算术、比较与 matmul 运算符 { #tensor-operators }

`+`、`-`、`*`、`/`、一元 `-`、比较运算与 `@` 构建惰性表达式节点。
将 `Tensor` 用作 Python 布尔值（`if tensor:`）会抛出 `TypeError`。

### Tensor.matmul(other) { #tensormatmul }

rank-2 矩阵乘法；复数输入使用共轭 Wirtinger VJP。示例：
[矩阵乘法与自动推导的梯度](examples/programs-and-interop.zh.md#matrix-multiplication-with-derived-gradients)。

### Tensor.exp() { #tensorexp }

逐元素指数。

### Tensor.sqrt() { #tensorsqrt }

逐元素平方根。

### Tensor.conj() { #tensorconj }

逐元素复共轭；实数 dtype 下为恒等。示例：
[复数 VJP](examples/programs-and-interop.zh.md#complex-vjp)。

### Tensor.cumsum(dim, *, reverse=False) { #tensorcumsum }

沿一个轴的包含式扫描。示例：
[用 map/cumsum/contract 拼出线性递推](examples/programs-and-interop.zh.md#linear-recurrence-from-mapcumsumcontract)。

### Tensor.sum(axis=None, *, keepdims=False) { #tensorsum }

沿一个轴、多个轴或全部轴归约。

### Tensor.gather(index) { #tensorgather }

按索引 Tensor 选取行；VJP 是 segment sum。

### Tensor.segment_sum(index, num_segments) { #tensorsegment_sum }

按每行的段索引把行求和进 `num_segments` 个桶。

### Tensor.checkpoint() { #tensorcheckpoint }

把一个非叶子值标记为显式的反向保存点（IR 中为
`gf_tensor.checkpoint`），对该值覆盖 checkpoint 规划器的选择。

### Tensor.realize() { #tensorrealize }

强制对延迟表达式做 JIT 编译与执行，返回物化后的 Tensor。

### Tensor.prepare() { #tensorprepare }

在首次 `realize()` 之后返回一个零参数的热 callable，以稳定的
输入/输出缓冲区重复提交已编译的 DAG。冷编译开销仍可通过
`Tensor.execution` 观察到。

### Tensor.execution { #tensorexecution }

属性，在物化后返回 codegen/启动诊断：语义哈希、编译/启动/物化耗时、
MLIR checkpoint 计划、已保存字节数以及所选 backend。物化前为 `None`。

### Tensor.generated_code(kind=None) { #tensorgenerated_code }

返回真实的编译器阶段产物，如 `"gf_tensor"`、`"cpu_loop"`、`"llvm"` 或
`"ptx"`；缺省时选取已生成的最低层阶段。指南：
[编译器管线](compiler-pipeline.zh.md#inspecting-a-compiled-program)。

### Tensor.mlir(*, verify=False) { #tensormlir }

返回规范化的、与目标无关的 `gf_tensor` IR；`verify=True` 会用原生
解析器与 C++ verifier 做一次往返校验。这是编译器契约；
`Tensor.expression()` 只是非正式的调试捕获。

### Tensor.tolist() / Tensor.to_numpy() / Tensor.to_torch(*, copy=False) { #tensor-host-interop }

显式的主机或 Torch 拷贝。`to_numpy()` 从原生存储（或可选的 Torch 侧
存储）执行主机拷贝；`to_torch()` 默认共享存储，`copy=True` 时拷贝。

### gf.autograd.grad(output, inputs, *, grad_output=None, allow_unused=False, checkpoint="auto") { #gfautogradgrad }

构建函数式的反向模式 [VJP](https://en.wikipedia.org/wiki/Automatic_differentiation) 表达式，不改动输入。
`checkpoint` 选择反向所需前向值的存储方式：`save` 把非叶子值标记为
保存，`recompute` 保留完整前向表达式，`auto` 生成
`gf_tensor.checkpoint_candidate` 交给与目标无关的 checkpoint 规划器。
非标量或复数输出必须显式给出 `grad_output`。示例：
[矩阵乘法与自动推导的梯度](examples/programs-and-interop.zh.md#matrix-multiplication-with-derived-gradients)。

### gf.autograd.value_and_grad(function, *, argnums=0) { #gfautogradvalue_and_grad }

返回 `function` 的一个变换：同时产出前向输出和 `argnums` 选定的
位置参数的梯度，不涉及 `.grad` 变更。

### gf.autograd.joint_plan(output, inputs, *, checkpoint="auto") { #gfautogradjoint_plan }

构建一个可执行、可检查的联合前向/反向任务 DAG，使用同一份快照版本。
`plan.run()` 返回 `(output, gradients)`；`plan.explain()` 渲染该
bundle。示例：
[联合前向/反向 DAG](examples/programs-and-interop.zh.md#joint-forwardbackward-dag)。

### gf.autograd.grad_mlir(output, input, *, grad_output=None, lower=True) { #gfautogradgrad_mlir }

以 [MLIR](https://en.wikipedia.org/wiki/MLIR_(software)) 文本返回显式 VJP 请求，或（`lower=True`）其
`gf-tensor-vjp` 结果，供检查。

## Graph { #graph }

Tiga 张量之上的不可变逻辑关系。物化 [CSR](https://en.wikipedia.org/wiki/Sparse_matrix) 拥有两个索引张量；
dense、triangular、radius 与 kNN 关系保持隐式或过程式，绝不会被静默
物化成 O(N²) 邻接。

### Graph.from_csr(row_ptr, col_idx, *, num_src=None, sorted_by_dst=True, validate="basic") { #graphfrom_csr }

从 CSR 索引数组创建外部冻结关系（Tiga Tensor，或经 interop
适配器路由的 Torch tensor）。`num_src` 缺省为 `max(col_idx) + 1`；
`validate="full"` 额外检查端点、单调性与索引范围。示例：
[GCN 聚合](examples/message-passing.zh.md#gcn-aggregation)。

### Graph.from_coo(src, dst, *, num_src=None, num_dst=None) { #graphfrom_coo }

从坐标列表创建同样的冻结关系，内部做一次按目标点的稳定排序，写成
规范 CSR。

### Graph.regular(num_nodes, degree, *, device="cpu") { #graphregular }

创建确定性的定度数关系——目标 `i` 从源 `(i * degree + k) % num_nodes`
收集——基准测试与冒烟测试不再手工组装 `arange`/`%` CSR 数组。

### Graph.dense(num_src, num_dst=None, *, device=None, index_dtype=gf.int64) { #graphdense }

创建隐式笛卡尔关系；不分配 `N×N` 索引张量。`num_dst` 缺省等于
`num_src`。示例：
[全量注意力，无掩码](examples/attention.zh.md#full-attention-no-mask)。

### Graph.triangular(num_entities, *, device=None, index_dtype=gf.int64) { #graphtriangular }

创建隐式下三角含对角关系（`src <= dst`）：在 IR 中仍是图拓扑，
任何 MessagePassing reducer 都可使用；dense lowering 使用有界
源 tile 加配对掩码，而不物化 CSR。示例：
[因果稠密关系](examples/attention.zh.md#causal-dense-relation)。

### Graph.cu_seqlens(cu_seqlens, *, causal=True) { #graphcu_seqlens }

为 flash-attn 风格的打包变长序列创建物化的块对角关系：序列 `k` 中的
位置 `i` 从源 `[s_k, i]`（`causal=True`）或从所属序列整体
（`causal=False`）收集。`cu_seqlens` 必须以 0 开头且单调。示例：
[带 cu_seqlens 的 varlen 因果注意力](examples/attention.zh.md#varlen-causal-attention-with-cu_seqlens)。

### Graph.cat(graphs) { #graphcat }

把多个关系组合成一个块对角关系：块 `k` 的目标点只与块 `k` 的源点
相连，索引按源点累计数量偏移。`cat([Graph.triangular(n) for n in lengths])`
等价于 `Graph.cu_seqlens(cumsum(lengths), causal=True)`；用 `Graph.dense`
块则等价于 `causal=False` 变体。组合结果是一份物化的 CSR 快照。示例：
[带 cu_seqlens 的 varlen 因果注意力](examples/attention.zh.md#varlen-causal-attention-with-cu_seqlens)。

### Graph.stencil(dims, offsets, *, periodic=False, device="cpu") { #graphstencil }

在按行主序线性化的节点上创建物化的规则网格 stencil 关系；每个目标点
按 offsets 顺序从 `coord(dst) + offset` 收集。非周期边界截断网格外
源点，周期边界按取模回绕。示例：
[无矩阵 FEM 算子与求解循环](examples/solvers.zh.md#matrix-free-fem-operator-and-solver-loop)、
[磁盘上的超大图](examples/distributed-memory.zh.md#a-disk-resident-giant-graph)。

### Graph.radius(positions, cutoff, *, exclude_self=True, fields=None, metric=None, select=None, periodic=None) { #graphradius }

从 rank-2 浮点位置创建可重建的过程式关系；邻接按当前快照重新计算。
`metric`/`select` 是自定义 builder UDF，`periodic` 提供盒长 `[D]` 或
晶格向量 `[D,D]`。示例：
[可微的半径关系](examples/dynamic-relations.zh.md#differentiable-radius-relation)。

### Graph.knn(positions, k, *, candidates=None, exclude_self=None) { #graphknn }

创建精确的过程式 [kNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) 关系；除非显式物化或编译器选定的使用方需要，
否则不分配邻接结构。省略 `candidates` 时为自 kNN 并默认
`exclude_self=True`；传入 candidates 时创建二分的 query→candidate
关系。示例：
[精确 kNN 接入 MessagePassing](examples/dynamic-relations.zh.md#exact-knn-feeding-messagepassing)。

### Graph.open(path, *, device="cpu") { #graphopen }

打开一个持久化图而不急于加载 CSR 数组；结果是由分页存储支撑的
普通 `Graph`，在规划器请求有界行范围之前只读取清单（manifest）。
`gf.load(path, *, device="cpu")` 是模块级别名。示例：
[磁盘上的超大图](examples/distributed-memory.zh.md#a-disk-resident-giant-graph)。

### gf.save(graph, path, *, fields=None) { #gfsave }

以带版本号的 `.gfg` 格式持久化静态 CSR 图（manifest 加
`row_ptr.bin`/`col_idx.bin`）；`gf.load` 负责重新打开。
`fields={"src": {...}, "dst": {...}, "edge": {...}}` 会把节点/边字段
额外写成定长行，使其随拓扑一起从磁盘分页读取。

### Graph.fields(role, *, requires_grad=False) { #graphfields }

在用 `fields=` 保存的 `paged_csr` 图上，返回该角色下惰性读取的字段壳
（完整 shape、payload 在磁盘）映射，可直接传入 kernel 调用。
`requires_grad=True` 让壳可微；此时分页执行同时支持前向与
`gf.autograd.grad`。

### graph.halo(mesh, *, partition=None, depth="auto") { #graphhalo }

返回带有 owned/ghost 需求的同一种逻辑 `Graph` 类型；`partition` 缺省
为 `gf.ByDestination()`。这是声明式的：通信以类型化的 pack、
exchange、unpack、interior 和 boundary 任务插入到用户 kernel 之下。
示例：
[双进程 halo 交换](examples/distributed-memory.zh.md#two-process-halo-exchange)。

### graph.paged_rows(begin, end) { #graphpaged_rows }

编译器/运行时钩子，返回一个有界的 CSR 目标行分页；只对用 `gf.load`
打开的 `paged_csr` 图有效。

### graph.resolve_csr() { #graphresolve_csr }

显式物化并返回 `(row_ptr, col_idx)`——这是检查或调试请求，不属于
正常的编译器消费路径。

### graph.transpose() { #graphtranspose }

返回一个交换了端点角色的物化快照。

### graph.explain() { #graphexplain }

渲染一行摘要：origin、lifecycle、realization、实体与边数量、设备，
以及存在的 halo/分页后端。示例：
[磁盘上的超大图](examples/distributed-memory.zh.md#a-disk-resident-giant-graph)。

### Graph 内省属性 { #graph-properties }

`graph.schema`（`GraphSchema` dataclass）、`graph.device`、
`graph.num_edges`（过程式关系在实现前为 `None`）、`graph.placement`
与 `graph.is_distributed`。

### 规划器统计 { #graph-planner-statistics }

`graph.degree_bounds()`、`graph.degree_statistics()`、
`graph.degree_histogram()`、`graph.fixed_degree()` 与
`graph.source_index_span_ratio(samples=4096)` 暴露规划器所消费的
物理统计量；结果按快照缓存。

## MessagePassing { #messagepassing }

subclass `MessagePassing`，设置类属性 `reducer`，并实现
`edge(src, dst, edge, **params)`；`node(dst, aggregate, **params)`
可选，缺省为恒等。节点字段通过 `src={...}`/`dst={...}` 端点角色映射
绑定，或通过同构图上的单个 `ndata={...}` 映射。指南：
[消息传递](message-passing.zh.md)。

### program(graph=..., src=None, dst=None, edge=None, ndata=None, **params) { #messagepassing-call }

调用 kernel 实例即触发惰性 [JIT](https://en.wikipedia.org/wiki/Just-in-time_compilation) 选择与执行，返回聚合后的节点 Tensor。
在 `paged_csr` 图上，调用按有界目标行分页流式执行，并额外接受三个
调用时参数：`page_rows`（页高，缺省 100 000 或
`TIGA_PAGED_PAGE_ROWS`）、`prefetch=True`（让页读取与计算重叠）
与 `prefetch_depth`（并发预取的页数，缺省 2 或
`TIGA_PAGED_PREFETCH_DEPTH`）。示例：
[GCN 聚合](examples/message-passing.zh.md#gcn-aggregation)、
[磁盘上的超大图](examples/distributed-memory.zh.md#a-disk-resident-giant-graph)。

### gf.runtime.auto_offload(ram) { #gfruntimeautoffload }

上下文管理器：生效期间，每个显式 CSR 构造器（`Graph.from_csr`、
`Graph.stencil`、`Graph.cat` 等）在 CSR 超过 `ram` 字节时把拓扑持久化
为 `.gfg` 并返回 `paged_csr` 图，kernel 调用随之自动分页并带 prefetch。
环境变量 `TIGA_GRAPH_RAM_BUDGET` 设定进程级默认预算；上下文管理器
优先。offload 产物落在进程级临时目录、退出时清理；设置
`TIGA_SPILL_DIR` 后可跨进程存活。仅 CPU 图；字段也可随拓扑落盘
（`gf.save(..., fields=...)`），前向与反向都能分页执行。
示例：
[RAM 预算下的自动 offload](examples/distributed-memory.zh.md#automatic-graph-offload)。

### program.reference(**kwargs) { #messagepassingreference }

通过可选的 Torch interop oracle 显式执行语义实现；这是正确性参照，
不是原生 backend。

### program.prepare(*, graph, src=None, dst=None, edge=None, ndata=None, **params) { #messagepassingprepare }

可选地将一个经过验证的标量 CSR CUDA 绑定冻结为零参数热提交。普通
执行仍是惰性 JIT，从不需要此调用。

### program.explain() { #messagepassingexplain }

以文本渲染最近一个变体的规划器/lowering/缓存信息。

### program.diagnostics { #messagepassingdiagnostics }

属性，返回类型化的 `AnalysisFinding` 记录——阶段、处置结论
（`accepted`/`rejected`/`warning`/`unknown`/`remark`）以及可选的
资源、目标契约、估计与建议动作。`explain()` 是这些记录的人类可读
形式。

### program.schedules { #messagepassingschedules }

属性，返回由原生编译器从已验证的 `gf.kernel` IR 中提取的类型化
`MachineSchedule` 记录：调度族、行/邻居 tile、子组数量、流水线深度、
具名资源、执行角色、生产者/消费者交接以及获准的指令类别。
`pipeline_stages == 1` 是明确的否定结果——编译器未接纳由它控制的
异步流水线——因此不能把 provider 的指令重排说成已证明的重叠。

### program.ir(stage="domain") { #messagepassingir }

返回 `domain`、`iter`、`kernel`、`task`、provider 输入或 provider IR
文本（如可用）。指南：
[检查编译产物](message-passing.zh.md#inspecting-the-compilation)。

### program.code(kind="ptx") { #messagepassingcode }

返回最近变体的代码产物，如 `ttgir`、`llir` 或 `ptx`。

### program.cache_info { #messagepassingcache_info }

属性，返回可执行缓存的命中/未命中/变体计数。

### program.last_variant 与 program.variants { #messagepassinglast_variant }

最近一次调用（或每个已缓存特化）的 `CompiledVariant` 记录：backend、
provider、lowering、pass、产物、诊断与调度。首次调用前访问
`last_variant` 会抛出异常。

## Reducer { #reducers }

reducer 是可执行的编译器输入，不是 eager 张量运算。
[reducer 指南](reducers.zh.md)以文字形式解释代数契约、属性与
lowering 选择。

### gf.sum(*, identity=0, deterministic=False) { #gfsum }

满足结合律与交换律的加法 reducer。示例：
[GCN 聚合](examples/message-passing.zh.md#gcn-aggregation)。

### gf.mean(*, deterministic=False) { #gfmean }

基于内置元组状态 `(sum, count)` 的邻居均值 reducer；度为 0 的行产生
NaN。指南：[gf.mean()](reducers.zh.md#gfmean)。

### gf.prod(*, deterministic=False) { #gfprod }

乘法 reducer，被提升为零安全乘积 IR；度为 0 的行产生幺元 1。示例：
[边 gate 的乘积](reducers.zh.md#product-of-edge-gates)。

### gf.online_softmax(*, accumulation_dtype=None, deterministic=False, block_prune_threshold=None) { #gfonline_softmax }

稳定的多值流式 reducer；edge 区域返回 `reducer(score, value)`
（一个 `OnlineSoftmaxItem`）。位于 `(0, 1]` 的
`block_prune_threshold` 是面向 dense 流式 tile 的语义近似策略，不是
调度提示。示例：
[全量注意力，无掩码](examples/attention.zh.md#full-attention-no-mask)、
[基于 tile 剪枝的稀疏注意力](examples/attention.zh.md#tile-pruned-sparse-attention)。

### class gf.Reducer { #gfreducer }

用户自定义 reducer 基类：subclass 并实现 `identity()`、
`lift(*messages)`、`combine(left, right)` 与 `finalize(state)`；
状态可以是标量或元组。类属性 `associative`、`commutative` 与
`deterministic` 声明代数性质——并行 lowering 要求显式的结合律声明，
Tiga 不会从 Python 源码证明结合律。示例：
[用户自定义 reducer](examples/message-passing.zh.md#user-defined-reducer)；
指南：[自定义 reducer](reducers.zh.md#inventing-your-own)。

### reducer(*messages) { #reducer-call }

在 `edge()` 内调用 reducer，把它与一个或多个边局部消息绑定成一个
staged `ReducerCall`。

### reducer.mlir(*, message_dtypes=None, symbol=None) { #reducermlir }

把四个区域捕获为经过验证的原生 `gf.reducer` op 并返回 MLIR 文本；
`message_dtypes` 缺省为单个 `float32` 消息。指南：
[检查 reducer](reducers.zh.md#inspecting-a-reducer)。

??? info "自动 reducer VJP 覆盖范围"

    加法/稳定代数以及捕获的、满足结合律的标量 reducer，会通过 CPU LLVM
    与 CUDA TTIR lowering 为平衡树或确定性树。乘积被提升为零安全 CSR
    乘积/VJP IR。空行/参差行和有序非交换元组状态已被覆盖；未注册的
    reducer 形态仍只有正确性保证。

## nn { #nn }

### gf.nn.trace(module, *, block_e=None, num_warps=None) { #gfnntrace }

包装一个 `torch.nn` 模块，使其可在 edge UDF 内调用：对 Torch tensor
走 eager（输入沿特征维拼接），或被捕获为可编译子图，进入融合的
edge-NN tile lowering。`block_e`（edge-tile 大小，[16, 1024] 内的
2 的幂）与 `num_warps` 只是 launch geometry，从不改变数值；缺省值
按 lowering 区分——edge-centric sum tile 为 128/4，row-centric 注意力
tile 为 16/1。示例：
[用 nn 打分的 GAT 边注意力](examples/attention.zh.md#gat-edge-attention-with-an-nn-score)、
[Edge nn 模块与融合 tile kernel](examples/programs-and-interop.zh.md#edge-nn-modules-with-a-fused-tile-kernel)；
指南：[Edge nn 模块（CUDA，torch interop）](message-passing.zh.md#edge-nn-modules-cuda-torch-interop)。

## Control 与 jit { #control }

Tiga 语义值之上的结构化编译器控制流：循环体只捕获一次，lowering
为 `gf_control.repeat` / `gf_control.while` op，而不是作为 Python
循环执行。

### @gf.jit 或 @gf.jit(max_iterations=k) { #gfjit }

AST 路由：把 `for i in range(k)` 捕获为 `gf_control.repeat`，把裸
`while cond:` 捕获为由装饰器参数限定上界的 `gf_control.while`。
`for` 循环体开头的 `if cond: break` 是提前退出。循环携带变量是在
循环前赋值的 Tensor；`continue`、循环体中部的 `break`、`while True`、
非 range 迭代以及循环 `else` 均 fail closed。MessagePassing UDF 区域
永远不会被 AST 变换。`@gf.jit` 同时自动激活组合上下文：直线部分里
平级的 MessagePassing 调用被捕获为 `GraphProgram` SSA 叶子并在合法处
融合，叶子可直接参与张量算术；staged 循环体内的 kernel 调用保持逐
迭代语义（内联进循环体，绝不注册成顶层叶子）。捕获字段带
`requires_grad` 时叶子可微：反向逐叶子内联展开求 VJP（不融合），
共享输入的梯度贡献累加；不支持的叶子形态 fail closed。示例：
[固定迭代的 PageRank 探针](examples/message-passing.zh.md#fixed-iteration-pagerank-probe)、
[无矩阵 FEM 算子与求解循环](examples/solvers.zh.md#matrix-free-fem-operator-and-solver-loop)。

### gf.repeat(initial, body, *, iterations) { #gfrepeat }

匿名简写，捕获一个携带一个或多个 Tensor 状态的定次循环；`body` 只
trace 一次，且必须为每个携带状态返回一个形状、dtype、设备均不变的
Tensor。CPU lowering 为每个携带值分配两个可复用缓冲区。

### gf.while_loop(initial, condition, body, *, max_iterations) { #gfwhile_loop }

匿名简写，捕获无主机标量检查的有界数据依赖控制流；`condition` 必须
返回 rank-0 布尔 Tensor，`max_iterations` 作为有限资源守卫保留在
IR 中。CPU lowering 为 `scf.while`；在 provider 循环计划就绪之前 CUDA
fail closed。

### class gf.control.Repeat { #gfcontrolrepeat }

同一个 `gf_control.repeat` op 的构建器形式：subclass 并重写 `body`；
捕获的常量是普通实例属性。以 `kernel(initial, iterations=k)` 调用
实例时，`body` 恰好 trace 一次。

### class gf.control.While { #gfcontrolwhile }

`gf_control.while` 的构建器形式：subclass 并重写 `condition` 与
`body`；以 `kernel(initial, max_iterations=k)` 调用实例时，适用与
`gf.while_loop` 相同的上界。

定常求解器是这些原语之上的语法糖，不属于核心 API：
[examples/solvers.py](https://github.com/walkerchi/TIGA-lang/blob/main/examples/solvers.py)
提供 `dot`、`vector_norm`、`richardson` 和 `cg`；参见
[线性求解器与控制流](examples/solvers.zh.md)。

## 分布式与 halo { #distributed-and-halo }

### gf.DeviceMesh(device_type, shape, *, names=()) { #gfdevicemesh }

描述逻辑设备（`shape` 为 int 或元组），不初始化进程组。部署时
绑定到 Torch `DeviceMesh`、MPI、NCCL/RCCL 或厂商通信器。示例：
[双进程 halo 交换](examples/distributed-memory.zh.md#two-process-halo-exchange)。

### gf.ByDestination(mesh_axis=0, balance="auto") { #gfbydestination }

把目标实体与最终 reducer 状态分配到 mesh 分片的分区策略；
`balance` 取 `"edges"`、`"entities"` 或 `"auto"`。

### gf.GraphPlacement(mesh, partition, halo_depth) { #gfgraphplacement }

由 `graph.halo(...)` 附加到 Graph 快照上的逻辑所有权与 ghost 需求；
`halo_depth` 为非负整数或 `"auto"`。

### gf.HaloMap { #gfhalomap }

具体的单 rank 拥有者/ghost 映射：`owned_begin`/`owned_end`、
`ghost_ids`、`receive_from` 与 `send_to`，以及 `owned_entities`、
`ghost_entities`、`bytes_for(itemsize, trailing_elements=1)` 辅助
方法。

### gf.derive_halo_map(row_ptr, col_idx, *, num_entities, world_size, rank, peer_requests=None) { #gfderive_halo_map }

从本 rank 拥有的 CSR 行推导精确的 ghost，不依赖任何框架；
`peer_requests[p]` 列出 peer `p` 请求的本地实体 ID，用于填充
`send_to`。指南：
[精确说明所有权与 ghost](memory-and-distributed.zh.md#ownership-and-ghosts-exactly)。

### gf.collective_halo_maps(row_ptr, col_idx, *, num_entities, world_size) { #gfcollective_halo_maps }

为所有 rank 创建互相一致的接收/发送映射。

### gf.exchange_halo(halo, owned_data, *, element_bytes, transport) { #gfexchange_halo }

在一条 transport 上对编译器推导的邻居 [halo](https://en.wikipedia.org/wiki/Halo_(computer_science)) 完成 pack、
exchange 与 unpack，返回 `HaloBuffer`。`owned_data` 是从
`halo.owned_begin` 起按目标拥有者排列的本地字节存储；不涉及任何
框架张量类型。

### DistributedRuntime(transport, *, progress_threads=1) { #distributedruntime }

在图 kernel 之下管理异步通信推进。作为上下文管理器使用时成为活跃
runtime；没有活跃 runtime 时调用分布式图会 fail closed。
`runtime.exchange_halo(halo, owned_data, *, element_bytes)` 启动 halo
推进并返回一个与 provider 无关的 completion；
`runtime.last_execution_trace` 报告最近一次自动分片执行的耗时。

### DistributedRuntime.from_provider(name, *, progress_threads=1, **options) { #distributedruntimefrom_provider }

在不变的图 API 之下绑定部署用 transport 插件：默认注册 `"mpi"`
（mpi4py 包装）、`"tcp"`（零依赖的跨机器套接字）与 `"nccl"`
（基于 `libnccl` 的分组 send/recv）；第三方通过
`tiga.transport` entry-point 组注册。示例：
[双进程 halo 交换](examples/distributed-memory.zh.md#two-process-halo-exchange)、
[跨进程与跨机器运行](examples/distributed-memory.zh.md#running-across-processes-and-machines)；
指南：[runtime 与 transport](memory-and-distributed.zh.md#runtimes-and-transports)。

??? info "当前分布式执行边界"

    CPU rank 本地执行、反向 halo VJP、分页 CSR 分片以及真实的双进程
    [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 传输均已可执行。[CUDA](https://en.wikipedia.org/wiki/CUDA) 原生缓冲区使用独立的
    通信/编译器流和经过验证的事件排序。单 rank 的 NCCL 测试夹具只证明
    provider 绑定；真正的多 GPU NCCL/RCCL 正确性与重叠性能仍是
    未关闭的门禁。参见
    [内存层次与分布式执行](memory-and-distributed.zh.md)。

## Spill 与磁盘 { #spill-and-disk }

### Tensor.disk(*, name=None) { #tensordisk }

把负载换出到磁盘并释放内存缓冲区；此后的任何读取都会惰性重新加载。
匿名换出在 tensor 被回收或进程退出时删除；具名换出持久保存在
`TIGA_SPILL_DIR`（默认 `~/.cache/tiga/spill`）中。示例：
[把张量换出到磁盘](examples/distributed-memory.zh.md#spilling-tensors-to-disk)。

### Tensor.cpu() { #tensorcpu }

确保负载驻留在 CPU 内存中，并显式重新加载换出的数据；读取时也会自动加载。

### gf.from_disk(name) { #gffrom_disk }

在任意进程中把具名的磁盘换出挂载为惰性加载的 Tensor；名称不存在时
抛出 `FileNotFoundError`。示例：
[磁盘上的超大图](examples/distributed-memory.zh.md#a-disk-resident-giant-graph)。

## Program 与 visualize { #program-and-visualize }

### @gf.program { #gfprogram }

为无法提供源码的直线代码保留的兼容捕获边界：函数内的 MessagePassing
调用成为同一个 `GraphProgram`（类型化无环 SSA 组合）的 apply。
`@gf.jit` 会自动激活同一上下文（并额外捕获循环），新代码应直接使用
`@gf.jit`；两者叠加合法且幂等。装饰器之外的普通叶子调用仍是自动
JIT。示例：
[跨 kernel SSA 捕获](examples/programs-and-interop.zh.md#gfprogram-ssa-capture)。

### GraphProgram.apply(kernel, *, graph, src, dst, edge=None, **params) { #graphprogramapply }

向组合中加入一个 MessagePassing 叶子，返回其类型化 `ProgramValue`。

### GraphProgram.outputs(*values) { #graphprogramoutputs }

选定在组合边界上被观察的值。裸 `ProgramValue` 叶子直接选定其
producer；Tensor 表达式则贡献其传递引用的全部 program 叶子——
因此在叶子之上叠加算术是合法的输出写法，而不会谎称并入了叶子
kernel 的融合。

### GraphProgram.run() { #graphprogramrun }

对选定输出做 JIT 与执行；相互独立的叶子在合法处水平融合，有依赖的
apply 通过类型化的运行时依赖 DAG 执行。

### GraphProgram.ir(stage="domain") { #graphprogramir }

返回 `domain`、`fused`、`iteration`/`iter` 或 `kernel` IR 文本。调用
`ir` 即触发组合、验证与 lowering 的观察边界。

### GraphProgram.code(kind="ptx") { #graphprogramcode }

返回 provider 产物，如 `ttir` 或 `ptx`；对有依赖的 apply 返回按
apply 组织的产物映射。

### GraphProgram.explain() 与 GraphProgram.semantic_hash { #graphprogramexplain }

渲染 apply/融合计数以及组合后 domain IR 的内容哈希。

### ProgramValue.materialize() { #programvaluematerialize }

在此观察边界对所属程序做 JIT 与执行，返回具体值。`value.ir(stage)`
检查所属程序。

### gf.visualize.heatmap(values, *, vmin=0.0, vmax=1.0, low=None, high=None, cmap=None) { #gfvisualizeheatmap }

把一个连续 rank-2 浮点 Tensor 的热力图编译为一个融合 kernel，返回惰性
RGB `Raster`。默认 colormap 为 `viridis`；`cmap` 选择其他多停靠点
[colormap](https://zh.wikipedia.org/wiki/%E9%A2%9C%E8%89%B2%E6%98%A0%E5%B0%84)
——取自 `gf.visualize.colormaps()` 的名称、等间距 RGB 颜色列表，或位置
在 [0, 1] 内严格递增的 `(position, RGB)` 停靠点列表——`low`/`high` 则
选择朴素的双色 ramp（缺省的一端回落到对应的 viridis 端点）。双色 ramp
下超出 `[vmin,vmax]` 的值在惰性 Tensor 中不截断，保持表达式可微；多
停靠点 colormap 与 Matplotlib 一样钳制到端点颜色。颜色变换就是普通的
Tensor 广播/算术（多停靠点 ramp 用 `sqrt(x²)` 构造 tent 函数），编译器
核心中不存在可视化操作。示例：
[GPU 热力图可视化准备](examples/visualization.zh.md#gpu-heatmap-visualization-prep)。

### gf.visualize.colormaps() { #gfvisualizecolormaps }

返回内置热力图 colormap 的名称：`viridis`、`magma`、`plasma`、
`inferno`、`jet`、`coolwarm`、`gray`——同名 Matplotlib colormap 的
八停靠点采样，在融合 kernel 内做分段线性插值。

### Raster { #raster }

惰性 RGB 光栅，其像素仍是 Tiga Tensor：`realize()`、
`prepare()`、`execution`、`mlir(*, verify=False)` 与
`generated_code(kind)` 暴露与 `Tensor` 相同的编译器/运行时路径；
`to_numpy(*, clip=True)`、`save(path)`（Pillow）与
`show(**imshow_options)`（Matplotlib）是可选的主机互操作/编码辅助
方法。

### gf.visualize.Camera { #gfvisualizecamera }

主机端透视 camera，用于点云渲染：
`Camera(position, target, fov=45.0, up=(0.0, 0.0, 1.0))`；
`Camera.auto(positions, *, fov=45.0, margin=1.2)` 用包围球拟合 target
与距离——(N, 2) 平面输入按俯视处理，3-D 输入沿最小方差主轴取景，
各向同性点云回退到等轴测方向；
`Camera.from_angles(elevation=30.0, azimuth=-60.0, *, positions=None, distance=None, fov=45.0)`
把 camera 放在围绕 target 的球面上。
`camera.world_to_ndc(points_xyz)` 把 (N, 3) 点投影为
`(ndc, depth, visible)`。示例：
[粒子/视频](examples/visualization.zh.md#particles-and-video)。

### gf.visualize.particles(positions, values=None, *, camera="auto", width=512, height=512, point_radius=2.0, vmin=None, vmax=None, low=None, high=None, cmap=None) { #gfvisualizeparticles }

把每个点以圆盘溅射到主机端标量场——取点的值，`values=None` 时累加
单位密度——然后复用 `heatmap` 的 Tensor 表达式为该标量场上色，返回
惰性 `Raster`。`camera` 接受 `"auto"`、`Camera` 实例或
`(elevation, azimuth)` 元组；`vmin`/`vmax` 默认取标量场的数据范围；
`cmap` 选用多停靠点 colormap（见 [`heatmap`](#gfvisualizeheatmap)），
覆盖 `low`/`high`。示例：
[从散点生成 Delaunay mesh](examples/visualization.zh.md#simulation-meshes-load-shade-orbit)。

### gf.visualize.delaunay(positions, values=None, *, camera="auto", width=512, height=512, vmin=None, vmax=None, low=None, high=None, cmap=None, wireframe=False) { #gfvisualizedelaunay }

对 positions 做三角剖分——(N, 2) 平面输入直接剖分，3-D 输入先经
camera 投影——每个三角形填充顶点值均值（`values=None` 时用均匀值；
`wireframe=True` 改为绘制三角形边线），再经 `heatmap` 为标量场上色
（`cmap` 覆盖 `low`/`high`），返回惰性 `Raster`。示例：
[mesh 渲染](examples/visualization.zh.md#simulation-meshes-load-shade-orbit)。

### gf.visualize.mesh(positions, faces, values=None, *, camera="auto", width=512, height=512, wireframe=False, vmin=None, vmax=None, low=None, high=None, cmap=None) { #gfvisualizemesh }

渲染给定三角网格——`delaunay` 是从点现算三角剖分，`mesh` 接受显式的
`faces`（(M, 3) 索引数组，例如仿真网格或经 `load_obj` 加载的模型）。
三角形按顶点深度均值从远到近绘制（painter's algorithm），近处三角形
遮挡远处者；每个三角形填充其有限顶点值的均值（`values=None` 时用均匀
值；顶点全非有限的三角形跳过），`wireframe=True` 改为画边线。camera
处理、平面输入与相机后方裁剪策略与 `delaunay` 一致。示例：
[仿真网格：加载、着色、环绕](examples/visualization.zh.md#simulation-meshes-load-shade-orbit)。

### gf.visualize.splats(positions, colors, scales, *, rotations=None, opacities=None, camera="auto", width=512, height=512, cutoff=3.0, background=(0,0,0)) { #gfvisualizesplats }

把各向异性 3-D 高斯渲染为惰性 `Raster`（EWA splatting，与 3-D Gaussian
Splatting 相同）：协方差 `R·diag(scales²)·Rᵀ` 经透视投影的 Jacobian 投成
2-D 椭圆，在 `cutoff` 个标准差内求值，颜色按 front-to-back transmittance
从近到远做 alpha 合成。`colors` (N, 3) 为 [0, 1] 内的 RGB，`scales`
(N, 3) 或 (N,) 为正的标准差，`rotations` (N, 4) 为 `(w, x, y, z)`
quaternion，`opacities` (N,) 在 [0, 1] 内。相机后方、亚像素或衰减到
1/255 以下的高斯被跳过。示例：
[Gaussian splats 与体渲染](examples/visualization.zh.md#gaussian-splats-and-volumes)。

### gf.visualize.volume(density, *, camera="auto", width=512, height=512, steps=128, cmap="viridis", vmin=None, vmax=None, scale=8.0, background=(0,0,0)) { #gfvisualizevolume }

对密度网格 (X, Y, Z)（映射到单位立方体）做 ray marching，返回惰性
`Raster`：每条像素光线上的三线性采样发出归一化密度的 `cmap` 颜色，
并按 `alpha = 1 - exp(-density·scale·Δt)` 吸收（emission-absorption
模型）。`vmin`/`vmax` 默认取数据范围；`camera` 为 `"auto"` 时取景
整个立方体。示例：
[Gaussian splats 与体渲染](examples/visualization.zh.md#gaussian-splats-and-volumes)。

### gf.visualize.load_ply(path) { #gfvisualizeloadply }

解析 PLY 文件（ASCII 或 binary_little_endian），返回
`(positions, faces, extras)`：由 vertex 的 `x y z` 属性构成的 (N, 3)
float64 数组、face 的列表属性构成的 (M, K) int64 数组（无 face 时为
`None`），以及其余每个 vertex 属性组成的 dict——3-D Gaussian Splatting
载荷（`f_dc_*`、`opacity`、`scale_*`、`rot_*`）的入口。示例：
[Gaussian splats 与体渲染](examples/visualization.zh.md#gaussian-splats-and-volumes)。

### gf.visualize.load_obj(path) { #gfvisualizeloadobj }

把极简 Wavefront OBJ 文件解析为 `(positions, faces)`：`v x y z` 顶点
行变成 (N, 3) float64 数组，`f` 面行（含 `f a/b/c` 形式，只取顶点
索引）变成 (M, 3) 整数数组，1-based 索引就地解析、多边形扇形三角化。
空文件、缺少顶点/面、索引越界都会抛出 `ValueError`。示例：
[仿真网格：加载、着色、环绕](examples/visualization.zh.md#simulation-meshes-load-shade-orbit)。

### gf.visualize.export_ply(path, positions, *, faces=None, values=None, low=None, high=None, cmap=None) { #gfvisualizeexportply }

把几何体导出为 binary_little_endian [PLY](https://en.wikipedia.org/wiki/PLY_(file_format))
供 [Blender](https://en.wikipedia.org/wiki/Blender_(software)) 导入：顶点
携带 `x y z` float32（(N, 2) 平面输入补 `z = 0`），可选 `faces` (M, 3)
写成 `vertex_indices` 列表元素，`values` 额外携带 `red green blue`
uint8（与 `heatmap` 同一 colormap——默认 `viridis`，可用 `cmap` 或
`low`/`high` 覆盖——按数据范围归一化）与保存原始场值的
`scalar_value` float32。positions 与 values 经一次主机拷贝（GPU tensor
同样）进入单个结构化缓冲区一次落盘，百万点导出远低于一秒。示例：
[仿真网格：加载、着色、环绕](examples/visualization.zh.md#simulation-meshes-load-shade-orbit)。

### gf.visualize.export_obj(path, positions, faces) { #gfvisualizeexportobj }

写出文本 Wavefront OBJ——`load_obj` 的精确逆操作：`v` 行按 float64
round-trip 精度写出、`f` 行为 1-based，因此 `load_obj(export_obj(...))`
恒等恢复坐标与面。示例：
[仿真网格：加载、着色、环绕](examples/visualization.zh.md#simulation-meshes-load-shade-orbit)。

### gf.visualize.export_vdb(path, positions, values, *, voxel_size=0.05) { #gfvisualizeexportvdb }

把点云与标量场栅格化为 [OpenVDB](https://en.wikipedia.org/wiki/OpenVDB)
dense grid。需要可选依赖 `pyopenvdb`——无依赖的 PLY 导出已覆盖 Blender
互导路径，仅在明确需要体素网格时才引入 OpenVDB；缺失时抛出
`ModuleNotFoundError`。

### gf.visualize.save_video(frames, path, *, fps=30) { #gfvisualizesavevideo }

把任意帧迭代对象——`Raster` 或 [0, 1] 范围内的浮点 (H, W, 3) 数组——
编码为 `.gif`（Pillow）或 `.mp4`（原始 RGB 逐帧管道送入 ffmpeg 子
进程）；生成器逐帧消费，不囤积。示例：
[粒子/视频](examples/visualization.zh.md#particles-and-video)。

### gf.compiler.Dim / TensorSpec / ShapeSpecializer { #gfcompiler-symbolic-shapes }

有界符号签名：`Dim(name, minimum=1, maximum=None, multiple_of=1)`、
`TensorSpec(shape, *, dtype=gf.float32, device=None)` 与
`ShapeSpecializer(specs, compiler)`。同一个符号在多个参数中重复出现
时必须绑定到同一长度；最小值/最大值/可整除性守卫在编译前检查，每个
合法绑定形成一个具体的 MLIR 特化/缓存键。指南：
[Tensor 运行时与 autograd](runtime-and-autograd.zh.md#current-alpha-slice)。
