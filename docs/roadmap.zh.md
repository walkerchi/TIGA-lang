# 当前状态与路线图 { #current-status-and-roadmap }

本页为公开摘要。权威的完成台账是
`PROJECT.md §15.3`；本页陈述若与该台账冲突，以台账
为准。一项功能只有在正确性、可检查的编译产物和所声明的性能/一致性门禁
全部具备之后，才会摘掉 **partial** 标记。

## 支持矩阵 { #support-matrix }

| 目标 | 状态 | 可执行路径 | 剩余发布门禁 |
|---|---|---|---|
| NVIDIA [CUDA](https://en.wikipedia.org/wiki/CUDA) | 本地 alpha | `gf.domain/tensor → gf.iter → gf.kernel → serialized TTIR → vendor Triton → PTX/cubin`；原生 CUDA Driver 运行时 | 更全的 shape/layout/dtype 与多设备覆盖矩阵 |
| [CPU](https://en.wikipedia.org/wiki/Central_processing_unit) | 本地 alpha | `gf_tensor → Vector/SCF/MemRef → LLVM → ExecutionEngine`；融合的关系区间并行 | ragged/幂律负载的 NUMA 支持及更全的 dtype/layout 覆盖矩阵 |
| AMD ROCm / Hygon DCU | provider 待定 | provider ABI 与一致性输入已存在 | 厂商 TTIR→hsaco 插件及真实硬件产物 |
| Apple Metal | provider 待定 | provider ABI 与一致性输入已存在 | 合法 IR→MSL/metallib 插件及 Apple 硬件 CI |
| PPU | provider 待定 | provider ABI 与一致性输入已存在 | 厂商编译器/运行时插件及硬件 CI |

仅被 [Torch](https://en.wikipedia.org/wiki/PyTorch) 或 Triton 检测到并不等于支持。provider 必须在目标硬件上通过正确性、
产物检查、冷/热编译、运行时以及与参考相匹配的性能验证。

## 已打通的端到端功能 { #implemented-vertical-slices }

<div class="gf-feature-grid">
  <div class="gf-card"><div class="gf-card__label">编译器</div><h3>可检查的原生 IR</h3><p>Tensor、关系与 reducer 捕获；Domain→Iter→Kernel；自动 VJP；CPU LLVM 与 GPU TTIR 翻译。</p></div>
  <div class="gf-card"><div class="gf-card__label">运行时</div><h3>独立于 Torch 的执行</h3><p>CPU 缓冲区/ExecutionEngine 与 CUDA Driver 分配、stream、事件、模块、启动及锁页 DMA。</p></div>
  <div class="gf-card"><div class="gf-card__label">结构</div><h3>静态 + 生成式关系</h3><p>CSR、稠密/三角、生成的欧氏半径、度感知稀疏调度以及 builder–consumer 融合。</p></div>
  <div class="gf-card"><div class="gf-card__label">微分</div><h3>编译器派生的 VJP</h3><p>Tensor 视图/广播/归约/扫描/矩阵乘法、复数值以及关系/reducer 家族，无需用户编写反向 kernel。</p></div>
  <div class="gf-card"><div class="gf-card__label">内存</div><h3>物理规划</h3><p>容量、版本、checkpoint/spill、锁页内存↔HBM DMA 以及 RAM↔NVMe 执行均在 IR 中表示。</p></div>
  <div class="gf-card"><div class="gf-card__label">分布式</div><h3>Owned / ghost / halo 任务</h3><p>双进程 MPI 正确性、CPU 重叠以及用户 kernel 之下的 CUDA stream 依赖排序。</p></div>
</div>

## 进行中的收尾项 { #active-closure-items }

| 台账 | 状态 | 已有内容 | 收尾所需 |
|---|---|---|---|
| K0 · 精确过程式 kNN | **partial — ranked M0 已测量** | `gf.ranked_relation` → ranked-pairs → ranked 启动；稳定的候选 tile top-k、带掩码的任意 k≤64、层级合并、selected-edge 融合、活跃坐标重绑定，三项 N/D/k 门禁已通过 | 大 k、通用度量 UDF、受内存预算约束的 spill/任务规划、反向 lowering |
| X0 · 分布式执行 | **partial** | 类型化 [halo](https://en.wikipedia.org/wiki/Halo_(computer_science)) Event DAG、[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 双进程前向/[VJP](https://en.wikipedia.org/wiki/Automatic_differentiation)、CPU 重叠、NCCL 单 rank 绑定与 CUDA 提交顺序测试夹具 | 真实 2+ GPU 上的 NCCL 正确性、profiler 重叠、性能及 RCCL 证据 |
| P0 · 发布工程 | **partial** | 锁定版本的 LLVM 构建、本地 wheel 审计、无 Torch 冒烟测试、sdist 重建与托管编译器 CI | 完整 CPython/Linux/macOS 发布矩阵、可信发布与首个 [PyPI](https://en.wikipedia.org/wiki/Python_Package_Index) 版本 |
| G0 · 图算法探针 | **partial** | 固定迭代次数的 PageRank 编译器/控制路径及已注册性能用例 | 反向结构化循环/tape 的性能、设备端收敛、代表性 frontier 与有序交集 IR 探针 |
| L0 · 无矩阵求解器 | **partial** | `examples/solvers.py` 中的求解器语法糖（MessagePassing kernel 即算子，无包装器）、多结果 `repeat/while` 加 CPU 类型化双缓冲与 `scf.while`、用 MessagePassing 施加 FEM 刚度、带可选预条件子的固定/残差驱动 [CG](https://en.wikipedia.org/wiki/Conjugate_gradient_method) | 多状态 CUDA 循环规划、分布式归约、结构化反向循环、隐式伴随 VJP，以及相匹配的前向/反向产物 |
| 厂商 provider | **pending** | provider ABI、插件入口点与 fail-closed 的一致性命令 | 各厂商可独立分发的 provider 及目标硬件产物 |

精确 kNN 门禁计时的是单次启动内实时的全对重建与结果取用；
不以缓存的空间目录复用替代重建计时。

## 工程推进顺序 { #engineering-order }

1. 在已测量的 FP32/k≤64 M0 契约之上继续扩展 ranked-relation
   lowering，包括有界的 scratch/spill 任务和自动生成的反向。
2. 使用已注册的真实度分布，扩展通用稀疏/向量/高度数及非线性融合覆盖。
3. 验证真实多设备上的通信/计算重叠，并加入拓扑感知的划分成本模型。
4. 运行完整发布矩阵，并发布首批签名的 PyPI 产物。
5. 只有当厂商 provider 的工具链和硬件一致性测试能够持续运行时，
   才将其加入。
6. 借助 FEM/求解器探针加入有界的多状态控制流和由残差守卫的隐式微分，
   且不引入以工作负载命名的 [kernel](https://en.wikipedia.org/wiki/Compute_kernel)。
