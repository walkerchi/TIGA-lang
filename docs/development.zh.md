# 构建与贡献 { #development }

前置阅读：[IR 实例教程](ir-walkthrough.md)解释编译器模型；
仅需安装时从[入门教程](getting-started.md)开始。本页面向修改项目的开发工作，
不是运行第一个程序的前置要求。

## 完成第一次编译器修改

1. 用最小 Python 测试或 MLIR 输入复现问题。
2. 使用 IR 教程中的 `gf-opt` 命令，定位最早出现错误的阶段。
3. 修改相关 verifier 或 [pass](https://en.wikipedia.org/wiki/Compiler_pass)，补充回归测试。
4. 执行下方 Python 和 MLIR 检查；仅在编译输出有意变化时重新生成文档快照。

| 修改内容 | 实现入口 |
|---|---|
| 公共字段与调用检查 | `python/tiga/message_passing/` |
| Domain 操作与验证 | `include/tiga/Dialect/Domain/`、`lib/Dialect/Domain/` |
| 关系遍历 | `lib/Transforms/LowerDomainToIter.cpp` |
| Kernel 表示与调度 | `lib/Transforms/LowerIterToKernel.cpp`、`lib/Transforms/SelectKernelSchedule.cpp` |
| 分布式任务依赖 | `lib/Transforms/PlanDistributedTasks.cpp` |
| 目标 TTIR 输出 | `lib/Target/Triton/Translate.cpp` |

## 仓库结构

| 目录 | 职责 |
|---|---|
| `CMakeLists.txt`、`cmake/` | 原生构建入口与共享配置，包括 LLVM SDK 的固定版本 |
| `include/tiga/Dialect/`、`lib/Dialect/` | 操作定义、约束与 verifier |
| `lib/Transforms/`、`lib/Target/` | 编译转换与目标代码生成 |
| `python/tiga/` | 公共 Python API，按 tensor、graph、autograd、message_passing 等子包组织 |
| `python_bindings/`、`lib/Runtime/` | 原生编译绑定与不依赖 Torch 的运行时 |
| `tools/` | 编译工具入口与验证脚本 |
| `examples/` | 用户程序 |
| `benchmarks/` | 工作负载、测量协议与报告 |
| `benchmarks/kernels/` | 仅用于性能对照的手写实现，不由核心代码导入 |
| `tests/mlir/`、`tests/python/` | 编译器与 Python/运行时回归测试 |
| `third_party/licenses/` | 随原生 wheel 分发的许可证，不包含第三方源码 |
| `assets/` | 正式 logo 的源文件与生成资源 |
| `docs/archive/` | 历史设计记录，不是当前 API 规范 |
| `output/roofline/` | 本地测量输出，不纳入 Git 跟踪 |

`include/` 与 `lib/` 分别存放声明和实现。TableGen 读取 `.td` 操作定义，
在构建目录生成 C++ 头文件；生成文件不作为源码提交。`cmake/` 是根构建的
配置辅助目录，不是另一个项目。Python 打包通过 scikit-build-core 调用同一套原生构建。

当前依赖不是 Git submodule：CMake 查找固定版本的 LLVM/MLIR SDK，Python extras
单独安装可选包。`third_party/licenses/` 只保存随二进制分发的许可证，不包含 LLVM
或 Torch 源码；从没有 Git 元数据的源码包和 wheel 安装时也必须包含这些许可文本。

C++ 命名空间和头文件路径使用 `tiga`，原生扩展名为 `_tiga_compiler`，Linux
运行时库名为 `libtiga_runtime`。现有 `gf.*` IR 语法、`gf-*` 工具与 `gfrt_*`
C ABI 符号保持稳定；历史实验快照保留原名称与校验和。

历史规划存放在 `docs/archive/`，不进入使用文档站点。
现行边界以 [API reference](api.zh.md)、[支持矩阵](roadmap.zh.md) 和实现测试为准。
`SECURITY.md` 是面向外部贡献者的安全漏洞私密报告政策。

## 构建与检查 { #building-and-checks }

先按[源码安装](getting-started.md#source-build)配置 LLVM SDK 与
`CMAKE_ARGS`，将安装命令替换为：

```bash
python -m pip install -e ".[test,docs]"
python -m pytest tests/python -q
python -m mkdocs build --strict
python tools/render_api_reference.py --check
python tools/check_docs_links.py site
```

### MLIR 测试 { #mlir-tests }

editable 安装默认关闭编译器测试。为 [lit](https://en.wikipedia.org/wiki/LLVM)
测试使用独立 CMake 构建目录；先将 `TIGA_LLVM_ROOT` 指向 LLVM/MLIR 22.1.8 SDK。

部分二进制 SDK 不含 FileCheck、not、count 和 lit。下方辅助脚本下载固定版本的
LLVM 源码、验证校验和，并仅将测试工具构建到独立目录，不修改 SDK。
已有源码归档可通过 `--archive` 指定。

```bash
python tools/bootstrap_llvm_test_tools.py \
  --llvm-root "$TIGA_LLVM_ROOT" \
  --output "$PWD/build/llvm-test-tools"

cmake -S . -B build/compiler -G Ninja \
  -DMLIR_DIR="$TIGA_LLVM_ROOT/lib/cmake/mlir" \
  -DLLVM_DIR="$TIGA_LLVM_ROOT/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTIGA_INCLUDE_TESTS=ON \
  -DLLVM_EXTERNAL_LIT="$PWD/build/llvm-test-tools/lit/lit.py" \
  -DTIGA_LLVM_TEST_TOOLS_DIR="$PWD/build/llvm-test-tools/bin"
cmake --build build/compiler --target check-tiga --parallel 2
python tools/render_ir_docs.py --gf-opt build/compiler/bin/gf-opt --check
```

发布工具链固定为 LLVM/MLIR 22.1.8（`llvmorg-22.1.8`）。
`TIGA_STRICT_LLVM_VERSION` 默认开启；关闭仅用于显式 API 兼容性实验，
不会改变发布 ABI 或编译缓存身份。

目标工具链通过进程边界衔接：
`gf-translate -gf-kernel-to-ttir` 将验证后的 Kernel IR 序列化为 TTIR 与
launch manifest。Triton 使用的 MLIR 版本可能不同，不能在同一地址空间共享
两个版本的 MLIR C++ 对象。

### CUDA 与包检查 { #cuda-and-package-checks }

```bash
python -m pip install -e ".[cuda,test]"
python tools/gpu_gate.py
python -m pip install build
python -m build
```

上述命令沿用安装时的 `CMAKE_ARGS`。CUDA 检查需要受支持的 NVIDIA GPU、
驱动、Triton 与已构建的编译工具，不是 CPU 入门的安装要求。

`tools/gpu_gate.py` 检查 CUDA 前向、反向、缓存与诊断。
skip、deselect 和 expected failure 均令 gate 失败。

### 发布验证与上传 { #release-validation-and-publication }

`release.yml` 在 GitHub 托管的 CPU runner 上从同一 sdist 构建 wheel，
并完成 CPU、MLIR 和文档检查；不等待 GPU runner，也不上传 PyPI。
编译使用缓存，MLIR 测试与质量检查用的 wheel 共用一次原生构建，
发布 wheel 与质量检查并行执行。

GPU 验证在本地针对下载的、已修复的 CPython 3.12 wheel 执行，
不使用 editable 安装或另行重建的 wheel。在版本 tag 对应的干净 checkout 中，
将 `TIGA_RELEASE_RUN_ID` 设为成功的 `release.yml` run ID，
将 `TIGA_RELEASE_TAG` 设为该版本 tag。
[GitHub CLI](https://cli.github.com/manual/) 完成认证后执行：

```bash
set -euo pipefail
: "${TIGA_RELEASE_RUN_ID:?Set the successful release workflow run ID}"
: "${TIGA_RELEASE_TAG:?Set the version tag}"
test "$(git rev-parse HEAD)" = "$(git rev-parse "${TIGA_RELEASE_TAG}^{commit}")"
validation_dir=$(mktemp -d)
gh api "repos/walkerchi/TIGA-lang/actions/runs/$TIGA_RELEASE_RUN_ID" \
  > "$validation_dir/build-run.json"
gh run download "$TIGA_RELEASE_RUN_ID" --repo walkerchi/TIGA-lang \
  --name wheel-manylinux_2_38_x86_64-py3.12 --dir "$validation_dir/wheel"
python3.12 -m venv "$validation_dir/venv"
py="$validation_dir/venv/bin/python"
"$py" -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
"$py" -m pip install pytest numpy pillow matplotlib packaging 'triton==3.6.0'
"$py" -m pip install --no-deps "$validation_dir"/wheel/*.whl
unset PYTHONPATH LD_LIBRARY_PATH TIGA_OPT TIGA_TRANSLATE TIGA_RUNTIME_LIBRARY MLIR_DIR LLVM_DIR
"$py" -m tools.gpu_gate > "$validation_dir/gpu.log" 2>&1
sha256sum "$validation_dir"/wheel/*.whl > "$validation_dir/SHA256SUMS"
gpu_wheel_sha256=$(cut -d ' ' -f 1 "$validation_dir/SHA256SUMS")
"$py" tools/verify_release_evidence.py "$validation_dir/wheel" \
  --run-metadata "$validation_dir/build-run.json" --commit "$(git rev-parse HEAD)" \
  --repository walkerchi/TIGA-lang --tag "$TIGA_RELEASE_TAG" \
  --gpu-wheel-sha256 "$gpu_wheel_sha256"
```

将构建记录、`gpu.log` 和 `SHA256SUMS` 保留为发布记录。
校验和仅在 GPU gate 成功后写入；确认日志后再批准上传。
校验和与手动确认代表维护者对本地验证的确认，并非独立的自动 GPU 认证。

GPU 验证通过后，才在同一 tag 上显式触发 `publish.yml`：

```bash
sha256sum --check "$validation_dir/SHA256SUMS"
gh workflow run publish.yml --repo walkerchi/TIGA-lang --ref "$TIGA_RELEASE_TAG" \
  -f build_run_id="$TIGA_RELEASE_RUN_ID" \
  -f gpu_wheel_sha256="$gpu_wheel_sha256" -F publish_pypi=true
```

上传复用已有构建产物，不重新编译。必须满足：同一仓库、tag 和提交的
release workflow 已成功，本地 GPU 验证的 wheel 校验和一致，
tag、源码和产物元数据精确匹配。
`pypi` environment 和指向 `publish.yml` 的 PyPI trusted publisher 需要单独配置；
本地构建和推送版本 tag 均不会上传 PyPI。
首发范围为 Linux x86-64、CPython 3.11/3.12、glibc ≥ 2.38。
此流程要求版本 tag 包含上述 workflow；已发布的历史 tag 不重写。

Read the Docs 使用 `.readthedocs.yaml` 和 `docs/requirements.txt`，无需安装 Tiga 或 LLVM。
GitHub 仓库公开后再导入 RTD；`READTHEDOCS_CANONICAL_URL` 提供文档根地址。
本地预览使用 `mkdocs serve`。

wheel 包含原生 Python 编译扩展、`gf-opt`、`gf-translate`、Tiga 运行时，
使用共享 SDK 构建时还包含实际 `libMLIR`/`libLLVM` SONAME 文件；静态 SDK
则将这些库链接进二进制。共享依赖使用相对 RPATH。

发布流程仍需在 manylinux 环境复现构建，执行 `auditwheel show/repair`
或对应平台工具，并检查无开发 SDK 路径的安装。先在无 Torch 的全新环境中
安装修复后的 wheel 与声明依赖，再运行下方 smoke。该检查覆盖 Tensor、
MessagePassing 前向与反向，并断言原生 CPU JIT。

执行前清除开发用 `PYTHONPATH`、`LD_LIBRARY_PATH`、`TIGA_OPT`、
`TIGA_TRANSLATE`、`MLIR_DIR` 与 `LLVM_DIR` 覆盖：

```bash
# Run outside the checkout, in a fresh environment without Torch or SDK paths.
cd /tmp
/path/to/clean/venv/bin/python -I /path/to/tiga-lang/tests/smoke/wheel_without_torch.py
```

## 贡献约定 { #contribution-policy }

- 核心 `python/tiga/` 不加入工作负载专属的 `@triton.jit`；
  手写性能对照仅放在 `benchmarks/kernels/`。
- lowering 不匹配 Python 类名或字段名。每个 rewrite 明确合法条件，
  保留 reducer 与 effect 信息。
- 外部库 dispatch 必须在 `explain()` 中标记；参考求值不得宣称为生成的 TTIR。
- Tensor/autograd 扩展属于通用 IR 或运行时设施，不加入优化器、数据集或模型专用实现。
- 大子系统保持独立子包和小型 `__init__.py`，不恢复原来的平铺大文件。
- 性能结论仅适用于登记的形状、dtype、图分布、缓存状态与硬件。
  单 GPU NCCL 绑定证明集成，不证明跨设备链路性能。
- 公开支持状态变化时，同步更新能力记录、测试、性能证据与 roadmap。

## 文档维护 { #documentation }

- Python 示例统一使用 `import tiga as tg` 与 `tg.*`；保留编译器 IR 名称及兼容锚点。
- 首页和用户教程先回答如何使用；编译器概念与实现进入独立开发路径。
- 每个入门页交代前置知识和下一步。IR 示例采用可复现的真实编译输出。
- 英文与中文页面保持一致，使用中性表述，不写第二人称。
- 固定技术术语保留英文；首次出现链接 Wikipedia，中文术语链接百度百科。
- 公式使用独立的 `$$` 块；示例运行命令链接到对应源码。
- 每张图只有一个重点；改图后检查实际截图。页面结构优先语义 HTML，
  架构图优先原生 SVG。
- 性能视图来自 `benchmarks/evidence_manifest.json` 与登记的 JSON，
  不在新生成器中手抄测量数据。provider 颜色保持稳定，条件差异用线型或标记表示。
- 图表提供有效 alt 文本、说明与完整尺寸或备用链接。交互确有价值时使用 HTML，
  保留提交到仓库的 SVG 备用图。
- 发布前运行 `mkdocs build --strict`，并检查桌面与窄屏排版。
