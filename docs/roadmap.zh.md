# 当前状态与路线图 { #current-status-and-roadmap }

## 支持环境 { #latest-validation }

Tiga 处于 alpha 阶段。支持的安装环境为 Linux x86-64、CPython 3.11/3.12
和 glibc ≥ 2.38。Torch adapter 使用 Torch 2.11.x，CUDA 编译另需 Triton 3.6.x；
Torch 单独安装。安装方法见[快速开始](getting-started.zh.md)，
发行可用性见[发布状态](support.zh.md#publication)。

## 支持矩阵 { #support-matrix }

| 目标 | 执行路径 | 可用性 |
|---|---|---|
| NVIDIA CUDA | Domain/Tensor → Iter → Kernel → TTIR → Triton → PTX/cubin | 已实现受支持操作、shape 和 dtype 的执行 |
| CPU | Tensor → Vector/SCF/MemRef → LLVM → ExecutionEngine | 原生 JIT，支持无 Torch 执行 |
| AMD ROCm / Hygon DCU | provider 扩展接口 | 暂无受支持的设备实现 |
| Apple Metal | provider 扩展接口 | 暂无受支持的设备实现 |
| PPU | provider 扩展接口 | 暂无受支持的设备实现 |

Torch 或 Triton 能检测到设备，不代表 Tiga 已提供对应 provider。
通过[执行诊断](execution.zh.md)查看所选 provider。

## 功能支持矩阵 { #feature-support }

图构造与图执行的覆盖范围不同。下表描述实际消费图的执行路径及其梯度行为。

| 路径 | 已支持行为 | 约束 |
|---|---|---|
| 原生 Tensor MessagePassing | 物化 CSR、默认欧氏 radius，前向与 VJP | 此入口拒绝其他关系实现 |
| 原生 paged CSR | CPU 前向/VJP、CUDA FP32 前向；[十亿边测量](memory.zh.md#billion-edge-capacity) | CUDA backward 暂不支持；输出驻留设备，staging 使用主机内存 |
| Torch 接口 | 普通 Torch 输入/输出与 autograd；dense、triangular 和生成式关系 | 融合取决于 relation、dtype 和 shape；部分 CSR backward 重放 Torch 运算 |
| 生成式精确 kNN | FP32 邻居选择与融合计算，k ≤ 64 | 受目标/shape 守卫限制；大 k、通用度量 UDF 和生成式 backward 需其他路径或暂不支持 |
| 图组合 | `Graph.cat([Graph.triangular(n), ...])`、stencil 和 CSR 构造 | `cat` 物化 CSR；离散邻居选择不参与求导 |
| Reducer | sum、mean、product、online softmax 与受支持的自定义代数 | lowering 依赖声明的代数性质 |
| Tensor dtype | f16/f32/f64、i32/i64、complex64/128 和 bool 存储 | 操作/backend 覆盖范围小于存储覆盖范围 |
| EdgeNN | 捕获 Torch 模块；有守卫的 CUDA tile 前向及输入、位置、参数 VJP | 仅支持可 trace 的操作；求导时固定已选邻居 |
| 分布式 | CPU/MPI、CUDA TCP/NCCL halo 交换与反向 VJP | 固定 ownership，先通信再计算；不按 GPU 处理速度自动重分区 |
| 分层存储 | 原生 Tensor 预算、可选整 Tensor LRU 换出/恢复、可微 copy 与快照 | 不管理 Torch-owned 存储，不为任意超显存 kernel 自动分块 |
| Tile 剪枝 attention | 显式选择的近似 CUDA dense 前向 | backward 暂不支持；近似效果取决于阈值与输入 |
| 可视化 | `tg.visualize.gaussians`、volume、mesh 与 field | Gaussian 几何/渲染使用主机计算，不参与求导 |

[API 例子](api-examples.zh.md)、[内存与存储](memory.zh.md)和
[分布式执行](memory-and-distributed.zh.md)提供可运行入口。
对应工作负载的测量见[性能与扩展性](experiments.zh.md)。

## 编译器能力 { #implemented-vertical-slices }

编译器在 Domain、Iter 和 Kernel IR 中保留关系结构与 reducer 代数，
推导受支持的 VJP，并 lowering 到 CPU LLVM 或 GPU TTIR。
Task/Storage IR 描述依赖、放置和数据搬运。
[IR 实例教程](ir-walkthrough.zh.md)展示一个可执行程序经过各层的结果。

## 计划扩展 { #active-closure-items }

| 方向 | 计划内容 |
|---|---|
| 生成式关系 | 更大 k、通用度量 lowering、受预算约束的 scratch 与生成式 backward |
| 分布式图 | 算力感知分区、更大规模的分区局部拓扑与故障恢复 |
| 分页 | 减少主机 staging 复制、直接在设备组装输出、重叠页搬运与计算 |
| 图算法 | 设备侧收敛、结构化反向循环、frontier 与交集遍历 |
| 求解器 | 多状态 CUDA 循环、分布式归约与隐式伴随微分 |
| Provider | 具备硬件正确性和性能覆盖的其他厂商 backend |
| 安装与发行 | 扩展平台/Python 覆盖，提供按版本发布的公开文档 |

## 开发重点 { #engineering-order }

扩展操作与梯度覆盖，改善大图 staging 和分区，并为每个受支持 backend
维护可复现测试。新增 provider 需要独立的工具链与硬件验证。

构建与测试方法见[开发指南](development.zh.md)，问题反馈见[支持指南](support.zh.md#bug-report)。
