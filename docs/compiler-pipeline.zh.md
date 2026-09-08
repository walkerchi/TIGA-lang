# 编译器流水线 { #compiler-pipeline }

Tiga 把同一个 Python 程序编译成不同的物理算法，选择依据是关系本身的结构，
而不是工作负载的名字。为了让这些选择既正确又可检查，lowering 依次经过四级 IR ——
Domain、Iter、Kernel、Task IR —— 每一级只固定一类决策，最后程序经由一条稳定的
provider 边界离开编译器。

<figure class="gf-figure gf-figure--architecture">
  <object type="image/svg+xml" data="/assets/compiler-pipeline-overview.svg" aria-label="Tiga 编译器流水线：Domain、Iter、Kernel、Task IR，随后是 provider 交接">
    <img src="/assets/compiler-pipeline-overview.svg" alt="Tiga 编译器流水线：Domain、Iter、Kernel、Task IR，随后是 provider 交接">
  </object>
  <figcaption><a href="/assets/compiler-pipeline-overview.svg">打开完整尺寸的 SVG</a>。每一级只固定一类决策；provider 交接是唯一与厂商相关的步骤。</figcaption>
</figure>

## 流水线总览 { #the-pipeline-at-a-glance }

| 阶段 | 固定的决策 | 消费方 |
|---|---|---|
| Domain IR — `gf.domain` | 程序的含义：关系、field 角色、UDF region、reducer | 语义优化：canonicalization、fusion 合法性、自动 [VJP](https://en.wikipedia.org/wiki/Automatic_differentiation) |
| Iter IR — `gf.iter` | 如何遍历：坐标层级；dense、sparse、ragged 或 generated 顺序 | 迭代 lowering：builder–consumer fusion、行界与 tile 边界 |
| Kernel IR — `gf.kernel` | 如何启动：launch geometry、tile、mask、局部工作集 | provider 翻译：GPU 上的序列化 TTIR，CPU 上的 [MLIR](https://en.wikipedia.org/wiki/MLIR_%28software%29) → [LLVM](https://en.wikipedia.org/wiki/LLVM) |
| Task IR — `gf.task` + `gf.storage` | 数据放在哪里、何时移动：物理实例、owned/ghost/halo 集合、事件 DAG | 运行时规划：异步传输、halo 交换、通信重叠 |

GPU 一侧交接的是序列化 TTIR；CPU 路径则是与之平行的上游 MLIR 到 LLVM lowering。
两条边界都不会把厂商特定的布局编码泄回稳定的 IR。

storage 与 task 信息并不是 Python 侧的调度提示：cost 与依赖事实会在本地 task
生成之前回馈到迭代与 kernel 决策中。

## 查看编译产物 { #inspecting-a-compiled-program }

每个编译后的 kernel 或 program 同时就是一个检查句柄：

```python
program(...)
program.explain()        # backend、provider、lowering、pass 序列、缓存状态

for stage in ("domain", "iter", "kernel", "task", "gf.kernel.ttir"):
    program.ir(stage)    # 各流水线阶段的 IR 文本

program.code("ttgir")    # 生成的代码；还支持 "llir" 与 "ptx"
```

`explain()` 报告 backend、provider 身份、所选 lowering、pass 序列、缓存状态以及优化
备注。某个阶段若没有产生 artifact，会抛出明确的错误 —— 语义路径或外部库路径绝不会
假装自己生成过 PTX。

带真实 `explain()` 输出的完整示例见
[检查编译结果](message-passing.md#inspecting-the-compilation)。

## 细节展开 { #details }

??? info "各 dialect 携带的信息"

    - `gf.domain` 保留数学含义：Entity/Field/Relation schema、`edge()`/`node()`
      UDF region、带类型的 reducer region 以及 effect。`gf.tensor` 则让 shape、
      dtype 与广播语义保持显式，并携带 map/reduce/scan/contract 与 VJP 请求。
    - `gf.iter` 把 sparse、ragged、dense 与 generated 迭代显式化 —— 坐标层级与
      遍历顺序 —— 而不绑定任何一家 GPU 厂商。
    - `gf.kernel` 固定 launch geometry、tile、mask 与局部工作集，同时把厂商特定的
      布局编码挡在稳定的编译器边界之外。
    - `gf.storage` 保留逻辑 Region、PhysicalInstance、内存空间、快照版本、异步传输
      与生命周期事件，覆盖 register、shared、HBM、RAM、NVMe 与分布式层级。
    - `gf.task` 保留分区、owned/ghost/halo 语义以及事件 DAG。每个本地计算 task 都
      经由同一条 `gf.iter → gf.kernel` 路径 lowering，因此通信留在 Python kernel
      之下，而不会被错误地塞进单个设备 kernel 内部。

??? info "各阶段的 pass 清单"

    | 阶段 | 引入的信息 | 代表性变换 | 检查方式 |
    |---|---|---|---|
    | 捕获 | field 角色、关系来源、UDF region、shape/dtype 守卫 | region 验证、effect 发现、语义哈希 | `ir("domain")`、`Tensor.mlir()` |
    | 语义优化 | reducer 代数、Tensor DAG 与 VJP 请求 | canonicalization、fusion 合法性、自动 VJP、checkpoint 候选 | `ir("domain")`、VJP IR |
    | 迭代 lowering | 坐标层级、遍历顺序、generated/materialized 选择 | builder–consumer fusion、行界、dense/三角 tile 边界 | `ir("iter")` |
    | Kernel 调度 | launch geometry、tile、mask、局部工作集与归约 | degree 分桶、行切分、特征 tiling、流水线合法性 | `ir("kernel")`、`schedules` |
    | 存储/任务规划 | 物理实例、容量、版本、owned/ghost 集合与事件 | spill/重计算、异步传输、halo task、通信重叠 | `ir("task")`、`explain()` |
    | Provider 翻译 | provider ABI 与合法目标操作 | Tiga kernel IR → 序列化 TTIR，或 CPU MLIR → LLVM | `ir("gf.kernel.ttir")`、`code(...)` |

    原生 pass 位于 `lib/Transforms/` 下 —— 例如 `LowerDomainToIter`、
    `LowerIterToKernel`、`PlanDegreeBuckets`、`PlanSplitRows`、
    `SelectKernelSchedule`、`FusionPasses`、`TensorVJP`、
    `PlanTensorCheckpoints` 与 `PlanDistributedTasks`。

??? info "为什么 GPU 边界是序列化 TTIR"

    Tiga 与厂商的 Triton fork 各自携带自己的 MLIR 版本。在边界处序列化 TTIR
    可以避免两个版本被链接进同一个进程，同时让交接保持可检查、可缓存。provider 可以
    是 NVIDIA、ROCm 或某个厂商 Triton fork；Tiga 不会把厂商 MLIR 链接进
    核心。边界下游，厂商 Triton 继续经由 TTGIR 与 LLVM IR 生成 PTX、cubin 或厂商
    ISA，而运行时负责启动编译好的 task 并遵守事件 DAG 的依赖关系。
