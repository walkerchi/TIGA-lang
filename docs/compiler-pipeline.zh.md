# 编译器入门 { #compiler-pipeline }

本页面向编译器开发者，前置内容是 [Python 编程模型](programming-model.md)。
运行程序或定位执行问题时，先阅读[执行与排错](execution.md)。

## 为什么需要内部表示 { #the-pipeline-at-a-glance }

Python 程序描述结果；编译器还需要决定如何访问边、划分工作和搬运数据。
[中间表示（IR）](https://baike.baidu.com/item/中间表示)把这些决策记录为可供后续变换
检查的结构。IR 是编译器的内部数据，不是额外的用户 API。

以邻居加权求和为例：乘法与求和定义了计算含义；按 CSR 行遍历，或将工作
分配给 GPU block，则是在决定执行方式。不同 IR 层把这些问题分开表达。

| 层次 | 真实操作名举例 | 回答的问题 |
|---|---|---|
| Domain IR | `gf.relation`、`gf.apply`、`gf.reducer` | 程序描述哪张图、什么计算？ |
| Iter IR | `gf_iter.traverse` | 按什么顺序访问关系？ |
| Kernel IR | `gf_kernel.launch` | 如何安排计算与局部资源？ |
| Task / Storage IR | `gf_task.launch`、`gf_storage.transfer` | 任务依赖哪些数据或事件？ |
| Tensor IR | `gf_tensor.mul`、`gf_tensor.reduce_sum` | 需要求值哪些 Tensor 表达式与梯度？ |

旧架构材料中的 `gf.domain`、`gf.iter`、`gf.kernel` 是层次标签，
不是实际操作名。[dialect](https://en.wikipedia.org/wiki/MLIR_(software))
把一组相关操作组织在一起；`gf_iter.traverse` 是其中一条具体指令。
某些检查接口还接受 `"gf.iter"` 这样的阶段别名；它是 API 参数，不是 MLIR 语法。

**下一步阅读[用实例读懂 IR](ir-walkthrough.md)**，对照真实输入、输出、符号含义
和逐步复现命令。

## 不是每个程序都经过同一条路线 { #execution-paths }

- **原生 Tensor 程序**（包括原生 CSR message passing）捕获 Tensor 表达式。
  受支持的 CPU 编译路径将 `gf_tensor` 操作逐步转换到 LLVM，梯度也有对应表达。
- **关系编译路径**保留 `gf.apply`，再
  [lowering](https://en.wikipedia.org/wiki/Compiler#Back_end) 到 `gf_iter`
  和 `gf_kernel`。实例教程从编译器测试输入直接验证这条路线。
- **Task 与 storage 规划**在选定路径需要时增加依赖。
  Task IR 不是每次调用都必须经过的“第四步”。
- **参考求值**可能在 `auto` 下被用于小规模原生表达式；它不证明任何编译
  路线实际执行过。

[支持矩阵](roadmap.md)记录具体入口的覆盖范围。架构能力图不能当作一次调用的执行轨迹。

## provider 是什么？ { #what-is-a-provider }

[backend](https://en.wikipedia.org/wiki/Compiler#Back_end)面向具体执行环境。
在 Tiga 中，provider 是目标工具链与运行时的适配层，不是云服务商。

当前 NVIDIA 路径把序列化的 Triton IR（TTIR）交给 Triton，继续生成设备代码；
CPU 路径使用 [LLVM](https://en.wikipedia.org/wiki/LLVM)。
序列化用于隔离 Tiga 和 provider 可能不兼容的 MLIR 构建。
此前的调度也会使用目标硬件能力，并非所有目标决策都从 provider 边界才开始。

## 查看编译产物 { #inspecting-a-compiled-program }

先检查结果实际如何执行，不要预设它经过了全部阶段：

| 问题 | 检查方式 |
|---|---|
| 这个原生结果是否通过 JIT 执行？ | 先物化，再检查 `output.execution` |
| 原生 Tensor 生成了什么代码？ | `output.generated_code()`；`output.mlir(verify=True)` 查看并验证 Tensor IR |
| MessagePassing 选了什么计划？ | `kernel.explain()` |
| 当前变体是否包含 Iter / Kernel IR？ | 仅在产物存在时调用 `kernel.ir("iter")` / `kernel.ir("kernel")` |

`kernel.ir("domain")` 返回变体的语义计划，不一定是可以解析的 MLIR。
`"gf.kernel.ttir"` 是可用的检查接口参数，不是实际操作名。
不存在的产物会报错；循环索取全部阶段并不是通用的检查方法。

原生梯度通过反向模式自动微分推导
[VJP](https://en.wikipedia.org/wiki/Automatic_differentiation)，
其表示方法见[运行时与 autograd](runtime-and-autograd.md)。

## 继续深入 { #details }

| 开发问题 | 下一章 |
|---|---|
| 真实操作经过 pass 后怎样改变？ | [用实例读懂 IR](ir-walkthrough.md) |
| 源码在哪里，修改后如何测试？ | [构建与贡献](development.md) |
| 动态邻居和不规则度数如何处理？ | [动态关系](dynamic-graphs.md) |
| 梯度与延迟 Tensor 如何实现？ | [Tensor 运行时与 autograd](runtime-and-autograd.md) |
| 传输和通信如何表示？ | [内存与分布式执行](memory-and-distributed.md) |
| 求解器循环与隐式梯度如何工作？ | [线性求解器](linear-solvers.md) |
