# 消息传递 { #message-passing }

前置内容：[编程模型](programming-model.md)。本页是公共接口契约：
先给完整程序，再说明字段、调用、允许的表达式与支持范围。
常见故障见[执行与排错](execution.md)；可选 nn 和编译检查内容放在后面。

## 三段式完整程序 { #a-complete-program-in-three-pieces }

```python
import torch
import tiga as tg

class WeightedSum(tg.MessagePassing):
    reducer = tg.sum()                       # 1. how messages combine

    def edge(self, src, dst, edge):          # 2. what each edge sends
        return edge.weight * src.x

    def node(self, dst, aggregate):          # 3. optional node update
        return aggregate + dst.bias

graph = tg.Graph.from_csr(
    torch.tensor([0, 2, 3, 5], dtype=torch.int64),
    torch.tensor([0, 2, 1, 0, 1], dtype=torch.int64),
    num_src=3,
)
num_edges = 5
bias = torch.tensor([0.1, 0.2, 0.3])
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
weight = torch.tensor([0.5] * num_edges, requires_grad=True)
out = WeightedSum()(graph=graph, src={"x": x}, dst={"bias": bias},
                    edge={"weight": weight})
dx, dw = torch.autograd.grad(out.sum(), (x, weight))
print(out.tolist())  # approximately [2.1, 1.2, 1.8]
print(dx.tolist())   # [1.0, 1.0, 0.5]
```

无需手写反向函数，返回值可直接参与 `torch.autograd.grad` 和 `.backward()`。

## Subclass 契约 { #the-subclass-contract }

- `reducer` —— 类属性（默认 `tg.sum()`），也可以在 `__init__` 里按实例
  设置。代数的支持范围取决于执行路径；见 [Reducers](#reducers)。
- `edge(self, src, dst, edge, **params)` —— **必需**。返回一条边的消息。
- `node(self, dst, aggregate, **params)` —— **可选**；默认原样返回聚合结果。

调用时额外传入的关键字参数会作为 `params` 转发。UDF 声明了
`**params` 时会收到全部参数；否则只传入签名中出现的名字，因此
`def node(self, dst, aggregate, dt)` 会从调用处取到 `dt=...`。

位置命名空间 `src, dst, edge` 总会被传入，所以签名必须接受它们 ——
但用不到的那个不引用即可。只有真正读取的属性才会进入捕获的 IR；
没必要 `del` 掉未使用的命名空间（旧版示例曾把这当作惯例，其实对
编译没有任何影响）。

## 字段命名空间 { #field-namespaces }

边是有向的 —— 每条边是一个 `(j→i)` 对 —— `src`/`dst` 命名的是两个端点
*角色*，而不是两个节点集合。在 `edge` 内部，三个 staged 命名空间暴露
调用时传入的字段：

| 命名空间 | 逐边取值 | 来源 |
|---|---|---|
| `src.<name>` | 按边收集的源端字段 | `src={...}` |
| `dst.<name>` | 按边展开的目标端字段 | `dst={...}` |
| `edge.<name>` | 边字段 | `edge={...}` 加上隐式图字段 |

在同构关系上，同一个节点集合扮演两个角色，因此一个 `ndata={...}` 映射
会把每个字段同时绑定到 `src` 和 `dst` —— 无需重复声明：

```python
kernel(graph=ring, ndata={"u": u}, edge={"conductivity": c}, dt=0.1)
```

`ndata=` 不能与 `src=`/`dst=` 组合使用；二分关系
（`num_src != num_dst`，例如注意力中的 query 节点与 key/value 节点）
必须使用显式的 `src=`/`dst=` 对 —— 这正是 `ndata` 无法表达的情形。

生成的关系自带隐式边字段 —— `Graph.radius` 和 `Graph.knn` 提供可微的
`edge.displacement` 和 `edge.distance`。在 `edge={...}` 里用同名遮蔽
图提供的字段会直接报错，而不是静默覆盖。

`node` 收到的是*未做 gather* 的目标端命名空间，加上聚合后的 `aggregate`。

## 调用与执行 { #calling-and-execution }

调用仅限关键字参数。Torch 字段返回形状为 `(num_dst, *feature_shape)` 的
`torch.Tensor`：

```python
out = kernel(graph=graph, src={...}, dst={...}, edge={...}, dt=0.1)
```

- 原生路径：字段值必须是 `tg.Tensor`，首维分别为 `num_src` /
  `num_dst` / `num_edges`，且位于图所在设备上。
- 默认直接传入 `torch.Tensor` 字段，内部处理适配，无需公开调用
  `from_torch` / `to_torch`。
- 全部使用原生字段时返回原生 `tg.Tensor`，属于高级接口。
- 原生调用捕获延迟表达式，读取结果时才执行。`auto` 可能对小表达式使用
  `python-oracle`；`TIGA_TENSOR_BACKEND=native` 要求原生编译。物化后检查
  `out.execution`；`kernel.cache_info` 统计 capture 变体，不是原生编译次数。

## reducer { #reducers }

reducer 是被捕获的代数结构，而不是 `"sum"` 这样的字符串 —— 完整契约、
内置签名和 lowering 阶梯见 [reducers 指南](reducers.md)。本页只需要
区分两件事：`reducer = tg.sum()` 定义聚合规则，`edge()` 返回每条边的消息。
普通 sum/mean/prod 不需要在 `edge()` 里再次调用 reducer。

下面 online-softmax 的现有接口要接收 `score` 和 `value` 两个输入，
所以用 `self.reducer(score, value)` 把它们打包成 staged 条目。
**这不是在单条边里执行聚合**；真正的聚合仍在所有入边之间进行。
当前多输入接口要求这个显式绑定，不能直接换成 `return score, value`。

```python
class Attention(tg.MessagePassing):
    reducer = tg.online_softmax()

    def edge(self, src, dst, edge, scale):
        score = (src.key * dst.query).sum(dim=-1) * scale
        return self.reducer(score, src.value)   # online-softmax item
```

用户自定义 reducer 继承 `tg.Reducer`，可以携带元组状态 —— 均值就是
`(sum, count)`，见可运行的
[自定义 reducer 示例](examples/message-passing.md#user-defined-reducer)。

## UDF 里能写什么 { #what-udf-code-may-contain }

UDF 构建的是惰性 Tensor 表达式 DAG，而不是任意 Python。支持的包括：
张量与标量的广播算术、一元 `-`、`exp`、`sqrt`、`conj`、`matmul`/`@`、
比较运算、`reshape`/`permute`/`transpose`、`unsqueeze`/`squeeze`、
`broadcast_to`、`.sum(dim=...)`、`gather`、`cumsum`。
以下情况会被拒绝（fail closed，直接报错）：

- 对张量做 Python 真值判断（`if tensor:`）抛出 `TypeError`；
- 使用 `tg.sum()` 时，`edge` 必须恰好返回一个 staged Tensor 表达式；
- 多消息返回需要自定义 reducer，且各消息共享同一个尾部特征形状。

MessagePassing UDF 永远不会经过 AST 变换 —— [`@tg.jit` 控制
流](api.md#control)作用于 kernel 外围，而不是其内部。

## Autograd 与 checkpoint { #autograd-and-checkpoints }

默认的 Torch 输入使用 `torch.autograd.grad` 或 `.backward()`，见本页完整例子。
下面的 `tg.autograd` 和 checkpoint 策略仅适用于原生 Tensor 高级接口，
不能直接传入 Torch Tensor。

```text
tg.autograd.grad(output, inputs, *, grad_output=None,
                 allow_unused=False, checkpoint="auto")
```

任何带 `requires_grad=True` 的 `src`/`dst`/`edge` 字段都是可微的
（float/complex dtype），半径关系的隐式 `distance`/`displacement`
字段同样可微 —— 位置梯度会流经生成的几何量。`checkpoint="save"` 在
VJP 中插入显式的 `Tensor.checkpoint()` 保存，`"recompute"` 把前向计算
融合进反向，`"auto"` 则把预算决策交给编译器。

## 哪些图可以传给 kernel { #which-graphs-a-kernel-can-consume }

所有 `Graph` 构造器都能用在同一个 `MessagePassing` 调用里，区别只在于
底层实际走哪条执行路径：

| 构造器 | Torch 张量调用 | 编译后的 CUDA 特化 | 原生 `tg.Tensor`（自动生成 VJP） |
|---|---|---|---|
| `Graph.from_csr` / `from_coo` | ✓（eager oracle） | ✓ 加权求和、edge-nn tile、nn 注意力 | ✓ |
| `Graph.radius` | ✓（eager oracle） | ✓ 生成的半径 kernel、edge-nn tile | ✓（含位置 VJP） |
| `Graph.knn` | ✓（eager oracle） | ✓ 固定度数的加权求和 kernel | — |
| `Graph.dense` / `triangular` | ✓（eager oracle） | ✓ 稠密流式注意力 | — |
| `graph.halo(...)` | — | — | ✓ 分布式 rank 本地执行 |

Torch 适配器为其支持的语义提供 eager 参考执行，但这不是原生路径的
通用回退机制。不支持的原生关系或 lowering 会报错，梯度的覆盖范围也可能
小于前向求值范围。详见[按路径划分的支持矩阵](roadmap.md)。

## 用法模式 { #usage-patterns }

以下列出 `MessagePassing` 支持的所有用法，每种都给出语义公式和
可运行示例。记号约定：`e = (j → i)` 表示一条边，`m_e` 是它的消息，
结果按目标节点 `i` 给出。

**标量消息求和。** [message_passing_autograd.py](examples/message-passing.md#differentiable-messagepassing-udf)。

$$
\text{out}_i = \sum_{e=(j\to i)} w_e\, x_j
$$

**多特征聚合（[GCN](https://en.wikipedia.org/wiki/Graph_neural_network)/SpMM）。** [gcn.py](examples/message-passing.md#gcn-aggregation)。

$$
\text{out}_i = \sum_{j\in\mathcal{N}(i)} w_{ji}\, \mathbf{x}_j, \qquad \mathbf{x}_j \in \mathbb{R}^F
$$

**双侧消息。** 一个表达式里同时使用两个端点角色；[diffusion.py](examples/message-passing.md#directional-diffusion)。

$$
m_{ji} = u_j - u_i
$$

**逐边数据。** `c` 保存在关系本身上；[message_passing_autograd.py](examples/message-passing.md#differentiable-messagepassing-udf)。

$$
m_e = c_e\, T_j
$$

**可选的节点更新。** [message_passing_autograd.py](examples/message-passing.md#differentiable-messagepassing-udf)。

$$
\text{out}_i = \text{aggregate}_i + b_i
$$

**编译器生成的梯度。** 编译器为任何带 `requires_grad` 的字段推导梯度；[message_passing_autograd.py](examples/message-passing.md#differentiable-messagepassing-udf)。

$$
\partial L / \partial x, \qquad \partial L / \partial w
$$

**自定义 [reducer](https://en.wikipedia.org/wiki/Fold_(higher-order_function)) 代数。** 用元组状态表示均值；[custom_reducer.py](examples/message-passing.md#user-defined-reducer)。

$$
\begin{aligned}
\text{lift}(v) &= (v, 1) \\
(s, n) \oplus (s', n') &= (s + s', n + n') \\
\text{out}_i &= s / n
\end{aligned}
$$

**带隐式几何量的半径关系。** 位置可微；[radius_autograd.py](examples/dynamic-relations.md#differentiable-radius-relation)。

$$
m_e = f(\mathbf{p}_j - \mathbf{p}_i,\; \lVert \mathbf{p}_j - \mathbf{p}_i \rVert)
$$

**精确 [kNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) 关系。** 关系本身作为 UDF 的输入；[knn_message_passing.py](examples/dynamic-relations.md#exact-knn-feeding-messagepassing)。

$$
\mathcal{N}(i) = \operatorname{kNN}(\mathbf{p}_i, k)
$$

**作为[注意力](https://baike.baidu.com/item/注意力机制)掩码的稠密与三角关系。** [full_attention.py](examples/attention.md#full-attention-no-mask)、[causal_dense_relation.py](examples/attention.md#causal-dense-relation)、[varlen_causal_attention.py](examples/attention.md#varlen-causal-attention-with-cu_seqlens)。

$$
\text{out}_i = \sum_{j \le i} \operatorname{softmax}_j(\mathbf{q}_i \mathbf{k}_j^\top)\, \mathbf{v}_j
$$

**online [softmax](https://en.wikipedia.org/wiki/Softmax_function) reducer。** 以幺半群方式流式计算，不物化任何分数矩阵；[tile_pruned_attention.py](examples/attention.md#tile-pruned-sparse-attention)。

$$
\text{out}_i = \dfrac{\sum_e e^{s_e}\, \mathbf{v}_e}{\sum_e e^{s_e}}
$$

**nn 打分的注意力（GAT）。** 以行为中心的融合 [kernel](https://en.wikipedia.org/wiki/Compute_kernel)，前向和反向都有；[gat_edge_attention.py](examples/attention.md#gat-edge-attention-with-an-nn-score)。

$$
s_e = \operatorname{MLP}_\theta([\mathbf{h}_i \,\|\, \mathbf{h}_j]), \qquad \text{out}_i = \sum_j \alpha_{ij}\, \mathbf{v}_j
$$

**在 `edge` 内使用 torch.nn 模块。** 融合为单个 tile kernel，可端到端训练；[edge_nn_message_passing.py](examples/programs-and-interop.md#edge-nn-modules-with-a-fused-tile-kernel)。

$$
m_e = \operatorname{MLP}_\theta([\mathbf{p}_j - \mathbf{p}_i \,\|\, \mathbf{x}_j]), \qquad \text{out}_i = \sum_e m_e
$$

**零拷贝 torch 互操作。** torch 张量直接作为字段，无需转换层；[torch_interop.py](examples/programs-and-interop.md#optional-torch-interoperability)。

**kernel 外的固定次数循环。** 以卷起的 `gf_control.repeat` 形式只捕获一次（[PageRank](https://en.wikipedia.org/wiki/PageRank)）；[pagerank.py](examples/message-passing.md#fixed-iteration-pagerank-probe)。

$$
\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} + \omega\,(\mathbf{b} - A \mathbf{x}^{(t)})
$$

**分布式 [halo](https://en.wikipedia.org/wiki/Halo_(computer_science)) 交换。** `out_rank[i]` 在分区关系上按 rank 本地计算；[distributed_halo.py](examples/distributed-memory.md#two-process-halo-exchange)。

## Edge nn 模块（CUDA，torch 互操作） { #edge-nn-modules-cuda-torch-interop }

`tg.nn.trace` 包装一个 `torch.nn` 模块，使其可以在 `edge()` 内被调用。
同一份源码驱动两条路径：eager 执行会拼接参数并调用模块；编译器证明
链式结构后生成一个融合的 tile kernel，消息全程不离开 tile ——
任何地方都不会物化 O(E) 的消息张量。

```python
class EdgeMLP(tg.MessagePassing):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = tg.nn.trace(mlp)          # nn.Sequential(Linear, ReLU, Linear)

    def edge(self, src, dst, edge):
        # message_e = MLP([pos_src − pos_dst ‖ x_src]); out = Σ_e message_e
        return self.mlp(edge.displacement, src.x)

with torch.no_grad():                        # inference: compiled tile kernel
    out = EdgeMLP(mlp)(graph=graph, src={"x": x}, dst={})
```

契约与当前限制 —— 任何超出范围的情形都会回退到精确的 eager oracle，
结果不变：

| 方面 | 契约 |
|---|---|
| 模块结构 | 一条**线性链** —— 单流、无分支 —— 由 `nn.Linear`、边局部逐元素算子和按特征的 `nn.LayerNorm` 组成 |
| 逐元素算子 | 模块或函数形式：`ReLU`、`GELU`（精确版）、`Sigmoid`、`Tanh`、`SiLU`、`ELU`、`LeakyReLU`、`Hardtanh`/`torch.clamp`（含 `ReLU6`）、`Hardsigmoid`、`Hardswish`、`Mish`、`SELU`、`Softplus`、`exp`、`log`、`sqrt`、`rsqrt`、`abs`、`sin`、`cos`、`square` |
| 标量算术 | 常量可出现在链中任意位置 —— `x * c`、`x + c`、`c - x`、`x / c`、`c / x`、`x ** p`、`-x` —— 用于温度缩放和仿射平移 |
| `nn.LayerNorm(width)` | 在每条边自己的特征轴上归一化该边的消息向量（结构上天然边局部），并为 `weight`/`bias` 提供符号梯度；`elementwise_affine=False` 也可用；`nn.Identity` 会被跳过 |
| 输入 | 逐边字段 gather（`src`/`dst`/`edge` 字段，加上隐式 `displacement`）；字段和权重均为 float32 |
| 可融合的 reducer | `tg.sum()` → 边中心 tile（本节）；`tg.online_softmax()` → GAT 风格注意力，其中 nn 为每条边产出一个标量分数、`value` 是字段 gather，编译为以行为中心的 online-softmax tile kernel，含前向和融合反向（[示例](examples/attention.md#gat-edge-attention-with-an-nn-score)） |
| Launch geometry | 是调优旋钮而非语义：`tg.nn.trace(mlp, block_e=256, num_warps=8)` 设置 tile 大小（2 的幂，16–1024）。默认值按 lowering 区分：sum tile 为 128/4 —— 在 400 万边半径工作负载上扫描得到的最优点（2.9 ms 前向+反向，对比 64 时的 3.3 ms 和 256 时的 3.5 ms）—— 注意力 tile 为 16/1，在典型注意力度数下小块浪费的 lane 更少 |
| 不支持的结构 | 在 `tg.nn.trace` 时以 `NotImplementedError` 拒绝：`BatchNorm`（统计量跨整个边批次计算 —— 天然跨边）、dropout（重算式 VJP 无法重放的随机状态）、分支/残差（超出已证明的线性链结构）；超出运行时支持范围（reducer、dtype、设备）的调用回退到精确的 eager oracle |
| 训练 | 同样融合：grad 模式下的调用经由 autograd 桥运行同一个 tile 前向，反向是符号 VJP 重算 tile kernel —— 逐边激活在 tile 内重放，因此两个方向上都不存在 `[E, ·]` 张量。梯度流向字段、位置和所有模块权重 |
| emitter 后端 | TTIR 由 Python 侧的 emitter 产生（阶段 1 前向，阶段 2 VJP）；后续阶段会把 emitter 移入 C++ `gf-kernel-to-ttir` 翻译 |

这些限制为什么存在：融合 kernel 独立处理每个边 tile —— 消息在寄存器里
算完、就地归约，全程不存在 `[E, ·]` 激活。`BatchNorm` 在构造上就违反这一点
（每条边都耦合到整个批次），支持它意味着物化全部消息再跑第二遍。
dropout 虽然是边局部的，但它是随机的；反向会在 tile 里重放前向、必须逐位
复现同样的值，随机 mask 做不到 —— 何况 dropout 在推理时本来就是恒等。
分支和残差原则上是边局部的，拒绝它们是工程边界而非本质限制：结构证明和
符号 VJP 重放目前只覆盖线性链。eager 回退则是一份契约而非托词 —— 语义由
oracle 定义，编译 kernel 必须精确复现 oracle，超出范围的调用结果保持完全一致。

在半径 edge-MLP 工作负载（`benchmarks/neural_networks/radius_edge_mlp.py`，
810 万边）上，编译后的前向与手写融合 Triton oracle 的差距在 1% 以内，
比 eager 路径快约 12 倍，且逐边激活内存为零。训练步
（`benchmarks/autograd/edge_nn_backward.py`，400 万边）比 eager Torch
autograd 快 **13.5 倍**，峰值内存仅为 **1/18**（65 MiB 对比 1.2 GiB）。

## 查看编译产物 { #inspecting-the-compilation }

每个 kernel 实例都可以直接查看自己的编译信息：

```python
kernel = WeightedSum()
out = kernel(graph=graph, src={"x": x}, dst={}, edge={"weight": weight})

print(kernel.explain())
# backend: reference
# provider: torch
# lowering: (unspecified)
# passes: validate-domain, select-reference-csr
# remark: [planning] reference evaluator selected; no performance codegen artifact exists
# remark: [planning] cross-apply fusion is handled by the @tg.jit/@tg.program capture boundary rather than ...
# variant cache: hits=0, misses=1
```

如果调用走的是编译后的 CUDA kernel，`explain()` 会报告真实的 lowering
—— 例如 `provider: triton`、`lowering: edge-nn-tile`，并附上 pass
列表（`capture-message-passing-udf`、`prove-edge-nn-tile-structure`、
`gf-kernel-to-ttir`、……）。除 `explain()` 之外还有：

- `kernel.ir("domain" | "iter" | "kernel" | "task" | "kernel_ttir")` 和
  `kernel.code("ptx" | ...)` —— 查看编译器各阶段 IR 和最终生成的代码；
- `kernel.diagnostics` / `kernel.schedules` —— 类型化的 `AnalysisFinding`
  与 `MachineSchedule` 记录；
- `kernel.cache_info`、`kernel.last_variant` —— 缓存命中统计和本次编译
  变体的完整记录。

精确签名见 [API 参考](api.md#messagepassing)。
