# Roadmap

状态：以仓库根目录 `PROJECT.md` §15.3 的单一收口台账为准  
日期：2026-08-14

GraphForge 通过扩展通用 compiler coverage 成长，不向 core 添加按 workload
命名的手写 kernel。本页只把规范台账映射成可执行工程顺序；状态与
`PROJECT.md` 冲突时以后者为准。

## Backend status

| Target | 可执行状态 | 下一道真实门槛 |
|---|---|---|
| NVIDIA CUDA | `gf.kernel → TTIR → Triton CUDA → PTX/cubin`，GraphForge CUDA Driver runtime 自主管理 context/buffer/stream/event/module/launch | 扩展 shape/layout/dtype 与性能矩阵 |
| CPU | `gf_tensor → Vector/SCF/MemRef → LLVM → ExecutionEngine`，pointwise vector+tail 与 relation range-parallel 均有正式 gate | 扩展 ragged/power-law、NUMA 与更多 dtype/layout 矩阵 |
| AMD ROCm/Hygon DCU | provider plugin ABI 和 conformance 输入已固定；没有本机 plugin/hardware | vendor TTIR→hsaco plugin、真机 correctness/artifact/roofline |
| Apple Metal/MPS | provider plugin ABI 和 conformance 输入已固定；没有本机 plugin/hardware | TTIR/legal IR→MSL/metallib plugin、Apple 真机 CI |
| PPU | provider plugin ABI 和 conformance 输入已固定；没有 vendor toolchain/hardware | vendor IR/binary/runtime plugin、真机 CI |

“被 Torch/Triton 检测到”不算支持。provider 只有在 correctness、artifact
inspection、cold/warm compile、roofline 和 matched-peer gate 都在目标硬件上运行后才能
进入支持矩阵。`benchmarks/providers/conformance.py --require` 会让缺失外部 plugin 的 CI
显式失败，而不是生成伪结果。

## 已完成的本机主链

- native OpBuilder 捕获 Tensor、CSR/dense/generated-radius、edge/node/reducer regions；
- Domain→Iter→Kernel→TTIR 和 CPU LLVM lowering，不生成 C/C++ 或 MLIR 字符串源码；
- CUDA Driver allocator、stream/event、module/function/launch 和 pinned-host 异步 DMA；
  native CUDA Tensor 已在禁止 import Torch 的子进程中完成 TTIR 编译、Driver 发射和回读；
- Static CSR、default radius builder/consumer fusion、power-law load balance 性能门槛；
- CPU contiguous Vector IR/LLVM、懒创建 persistent worker pool，以及 fused CSR relation
  destination-range parallelism；
- Tensor view/broadcast/reduction/complex、semantic VJP、online softmax、checkpoint
  liveness/cost/spill 和 joint forward/backward Task DAG；
- hierarchy capacity/version/physical replicas、HBM↔pinned RAM、RAM↔NVMe；
- typed halo overlap DAG、owner/ghost map 和真实两进程 Torch-free exact exchange；
- persistent isolated vendor compile worker、content-addressed cache；
- optional functional `torch.library` adapter、FakeTensor/meta/autograd/Inductor tests；
- LLVM 22.1.8 pinned-SDK clean build；当前 61/61 lit、220 Python tests + 10 subtests；
- `gf.kernel` 的 provider-neutral machine-schedule ABI 已由选择 pass 生成、dialect
  verifier 校验，并通过 native binding 暴露为 `kernel.schedules` 与 structured findings；
- manylinux/macOS release workflow、本地 manylinux_2_38 wheel audit、两次独立 no-Torch
  clean-venv smoke、sdist→wheel rebuild 与 strict twine。

## 剩余 compiler/runtime 收口项

### C4 — general reducer reverse mode（DONE）

当前 additive scalar/tuple reducer、stable online-softmax，以及结构等价但名字任意的 user
stable-weighted reducer 已自动求导。无法关联为这些 algebra 的 associative scalar reducer
也会生成 balanced/deterministic reduction tree；product 与非交换 affine-composition 已覆盖
empty/ragged/order，scalar/vector product forward/VJP 已通过 CPU LLVM 与 CUDA TTIR differential。
未被已知代数证明命中的 reducer 仍保留 balanced/deterministic tree 作为 correctness fallback；
captured product monoid 已成为一等 CSR product/VJP IR，反向不用除法，因此零 message 仍正确。
uniform-degree analysis 将其映射为每个 row/feature 一个 CTA，并一次写回整行 edge 梯度。
N=131072、degree=16 的正式 artifact 对 tuned handwritten Triton 为 1.021x
（95% CI low=1.016），同时超过 `torch.autograd`，因此 C4 的代表性 general-reducer
backward gate 已关闭。Stable tuple reducer 的 N=131072、degree=32 compiler schedule 采用
32-row tile/1 warp，正式 artifact 对 matched handwritten Triton 为 1.007x（95% CI low=1.005），
online-softmax gate 也已关闭。

### X0 — automatic sharded execution

已有 typed `halo_pack → halo_exchange → halo_unpack`、表达 interior/boundary dependency 的 Task IR、真实 transport、
provider-neutral completion、bundle→transport executable resolver，以及 CPU contiguous
rank-local Tensor 的自动 owned+ghost CSR binding和两进程 reverse-halo VJP。CUDA rank-local
Buffer 已完成 staged halo，device-buffer transport 也已走通无 host staging 的 native TTIR
forward/reverse VJP；当前机器只有一张 GPU，所以这些测试是 provider/binding correctness，
不是多卡性能声明。
版本化 `.gfg` PhysicalInstance binding 与 paged destination partition 已完成：rank 只读取
本地 CSR row/edge range，普通 `gf.load(...).halo(...)` MessagePassing 两进程 forward/VJP
通过。`graphforge.transport` plugin ABI 和 mpi4py/MPICH provider 也已由真实
`mpiexec -n 2` 执行；16 MiB halo artifact 的端到端吞吐为 1.892 GB/s。Torch-free NCCL
device-buffer provider、单 rank 真实 communicator + D2D local-path byte gate，以及完整
device-transport MessagePassing forward/reverse VJP binding 已完成。forward executor 已将
device halo enqueue 到 communication stream，在独立 compiler stream realize interior，等待后
运行 boundary，并以 CUDA `gf_tensor.scatter_rows` 合并；fixture 记录的 host 时间只证明提交与
依赖顺序，不算 NCCL P2P 或真实 GPU overlap 证据。
CPU host-transport 自动执行器已把 owned rows 拆成 interior/boundary：先启动 halo worker，
实际 realize interior，再等待 ghost、运行 boundary，并通过一等 `gf_tensor.scatter_rows`
线性拼回 owned row domain。两进程 artifact 在显式 5 ms receive-delay 模型下测得
13.7224 ms overlap，38.2262 ms 对 40.8632 ms serialized（1.069x）；该结果证明执行顺序和
受控模型下的收益，不冒充真实多节点网络。剩余 RCCL，以及两个以上真实设备的端到端 NCCL
correctness、device-direct overlap timeline 和通信吞吐 artifact。用户 API 仍只操作同一种
Graph/Tensor，不暴露 send/recv。

### C0/P0 — hosted reproducibility and release

本地已使用校验 SHA256 的官方 LLVM/MLIR 22.1.8 SDK 完成 clean build；当前为 61/61 lit、220 个
Python tests、strict docs、manylinux_2_38 wheel audit、无 Torch smoke 和 sdist→wheel rebuild。
hosted compiler run `31793915112` 的 clean-build 与独立 Torch compatibility jobs 均已通过。
repository、issue 和 maintainer metadata 已进入 PEP 621，并由安装后的 wheel smoke 读取校验；
当前 sdist 也已在独立目录重建、repair、strict twine 并通过无 Torch clean-venv smoke。仍不能由
本机替代的是完整 CPython 3.10–3.12 Linux/macOS release matrix、PyPI trusted-publishing
environment 与首次发布。

### B0/B1/B2 — vendor plugins

ABI/conformance 已就绪，但厂商 toolchain 与真机不是当前工作区可伪造的本地 TODO。每个 plugin
独立分发，输入/产物/async runtime capability 由 entry point 声明；不满足 conformance 时 core
必须明确报 unsupported。

## 性能与覆盖 backlog

- 图算法不做 NetworkX 全覆盖；独立 G0 diagnostic track 只用 PageRank、BFS、triangle
  counting 分别驱动 loop/convergence、frontier/worklist 和 sorted-intersection 通用 IR。
  fixed repeat 已有自动 VJP correctness fallback，但反向 control loop/tape 与性能 gate 尚缺。
  算法实现留在 examples/benchmarks，core 禁止 workload-named kernel；
- 当前 N=131072/degree-tail{8,64,256} 的 power-law 已覆盖 i32/i64、local/random、hot/cold
  八个正式 auto gate，CI low 为 1.255–1.849；i32 compiler-generated CDF bucket +
  chunked-tail TTIR 的四项 CI low 为 1.337–1.495。连续 log-normal/exponential 生成路径
  也各有 hot/cold gate。结论仍不外推其他 N 与 tail 分布；
- general strides/layouts、multi-output vector projection、high-degree vector
  SpMM 和 nonlinear/fused message family；fixed/bounded-ragged F16/F64 vector
  TTIR 已登记；
- exact kNN 已关闭 N=8192/D3/k32 build+weighted-consume bucket：动态 column snapshot
  重绑 compiler-generated TTIR 且严格 gate 通过；覆盖结论仍不外推更多 N/D/k；
- FLA/FSA 只作为 examples/benchmarks 中的匹配 workload，不成为 GraphForge core 算子；
- dense Cartesian/lower-triangular relation 已覆盖 exact、causal 与 grouped-query lane
  映射；Hq=16/Hkv=4 的正式 case 对 external Flash SDPA 为 1.084x（CI low 1.078）；
- multi-device build/communication/consume 的 overlap-accounted roofline；
- GPU-native visualization 是独立 V0 track，不阻塞 compiler 1.0。

每一项只有在测试或可复现 benchmark artifact 存在时才能从 PARTIAL/PENDING 改为 DONE。
