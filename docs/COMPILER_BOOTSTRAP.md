# GraphForge compiler bootstrap

状态：native alpha vertical slice  
固定发布工具链：LLVM/MLIR 22.1.8 (`llvmorg-22.1.8`)

## 构建

GraphForge 是标准 out-of-tree MLIR project，不下载或修改用户的 LLVM。开发构建需要一个
预构建 MLIR SDK：

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/opt/llvm-22.1.8/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gf-opt gf-translate
cmake --build build --target check-graphforge
```

默认 `GRAPHFORGE_STRICT_LLVM_VERSION=ON`。关闭它只用于显式 API compatibility test，
不会改变 release ABI 或 compiler cache identity。

## Native frontend

GraphForge 不在 Python 中拼 MLIR source。捕获结果是普通 typed descriptors；扩展模块用
C++ OpBuilder 构造并验证：

```text
Tensor DAG
  → gf_tensor
  → SCF/MemRef
  → LLVM dialect
  → MLIR ExecutionEngine

MessagePassing + Graph + Reducer
  → gf.relation | gf.cartesian | gf.generated_radius
  → gf.apply + gf.reducer regions
  → gf_iter.traverse
  → gf_kernel.launch | dense_launch | generated_launch
```

当前 native Domain capture 覆盖 scalar CSR、implicit dense、generated radius、optional node，
以及 vector-message dense online reducer。`gf.reducer` 始终保存
identity/lift/combine/finalize；
优化器从 region 类型与算子结构识别 scalar sum 和 stable online-softmax，不匹配 Python
类名、字段名或 workload 名。

`tensor_mlir.py` 只是 native inspection API；`cpu_tensor.py` 调用同一 in-process compiler 和
ExecutionEngine。二者都不生成 C/C++ 字符串，也不调用系统 C++ compiler。

## Pass pipeline

```text
gf-verify-domain
gf-form-apply-fusion-groups
gf-fuse-compatible-applies
gf-lower-domain-to-iter
gf-lower-iter-to-kernel
gf-select-kernel-schedule
gf-plan-distributed-tasks       # distributed placement 时生成 sidecar task graph
```

Domain→Iter→Kernel lowering 保留 reducer symbol、region、input role/name、snapshot version、
node-input index/segment ABI、effect、determinism 与 distributed placement。native capture 后的
Domain→Iter→Kernel→Task passes 在同一个 MLIRContext 内执行；打印的文本只是 debug snapshot，
不参与 pass 间传递。横向 fusion 的 legality 由共享 analysis 实现，
并拒绝 finalized-result dependency、version mismatch、非只读 effect 或 determinism mismatch。

## Provider boundary

`gf-translate -gf-kernel-to-ttir` 把已验证的 Kernel IR 序列化为 provider TTIR，并附带 launch
manifest。这个进程边界是有意保留的：厂商 Triton/MLIR 版本可能与 GraphForge 固定版本不兼容，
不能在同一地址空间共享 MLIR C++ 对象。之后由当前 provider 的 `IRSource` 继续产生
TTGIR/LLVM IR/PTX/cubin。

已接通的 direct translator slice 包括 fixed/bounded-ragged scalar CSR sum、generated-radius
distance sum、dense Cartesian stable streaming reducer，以及解释 scalar FP32
edge/reducer/optional-node regions 的 generic dense/CSR reducer。后者已用 tuple-state mean
和 diffusion 在 vendor Triton 上实际编译、执行。其他 operation 必须明确报告
unsupported 或选择经过语义证明的外部 library dispatch，不能伪装成 generated backend。

## Wheel

`python -m build --wheel` 的 CMake install component 包含：

- native Python compiler extension；
- `gf-opt` 与 `gf-translate`；
- GraphForge runtime；
- shared-SDK 构建所需的真实 `libMLIR`/`libLLVM` SONAME 文件；
- extension/tool 的相对 RPATH。

本地 wheel 已在全新 venv 中验证 native Tensor capture 和 bundled tools。正式发布仍须在
manylinux/macOS CI 使用 pinned SDK 重建，执行 dependency audit、lit、Python、docs 与
clean-install smoke test；本机 compatibility SDK 结果不能冒充发布认证。

## 当前缺口

- multi-output apply 与通用 vector field capture；
- vector tuple-state reducer 的通用 TTIR lowering，以及 generic algebra 到并行 tile 的优化；
- Tensor/graph CPU vectorization 与 parallel mapping；
- CUDA Driver runtime plugin，以及 ROCm/DCU/Metal/PPU provider；
- distributed task graph 的 owner/ghost map、真实 transport、byte sizing 与执行；
- manylinux/macOS release CI 和 trusted publishing。
