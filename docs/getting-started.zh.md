# 安装与第一个程序 { #getting-started }

本教程的目标：安装 Tiga，在 NVIDIA GPU 上运行邻居聚合，得到数值结果与梯度。
直接使用普通 Torch Tensor，无需编译器 IR 知识。
需要 Python 基础；仅源码安装额外要求可用的构建工具。

## 安装 { #install }

当前 alpha 正在准备首次发布，先使用源码或本地构建的 wheel。
发行名是 `tiga-lang`，import 名是 `tiga`。

### 从 PyPI 安装（首次发布后） { #pypi-install }

以下命令用于即将发布的版本，不表示当前已有正式发布包。在虚拟环境中，
先安装下文示例所需的 Torch；已有 Torch 或仅使用原生执行时跳过第一条命令：

```bash
python -m pip install torch
python -m pip install tiga-lang
python -m tiga
```

GPU 场景用匹配硬件的 Torch 安装命令替代上面的通用命令。
默认安装 Tiga 不安装或升级 Torch。需要可选 Triton provider 时，将 Tiga 安装命令替换为：

```bash
python -m pip install "tiga-lang[cuda]"
```

此 extra 不安装 Torch 或 GPU 驱动。匹配平台的预编译 wheel 包含编译器工具和
原生库，安装时无需单独准备 LLVM/MLIR SDK 或 C++ 构建工具。
首发 wheel 矩阵为 Linux x86-64、CPython 3.11/3.12、
`manylinux_2_38_x86_64`（glibc ≥ 2.38）；macOS、Windows 和其他 Python 版本不在首发范围内。
Torch adapter 面向 Torch 2.11.x，CUDA provider 面向 Triton 3.6.x；Torch 仍由使用环境单独安装。
没有匹配的 wheel 时，按[源码安装](#source-build)准备依赖；
源码分发包不等于预编译 wheel。

发布前可直接安装本地构建的 wheel，使用实际文件路径：

```bash
python -m pip install "/path/to/tiga_lang-0.1.0-<python>-<abi>-<platform>.whl"
python -m tiga
```

将占位文件名替换为匹配 Python 版本与平台的 wheel。基础 wheel 的本地安装无需访问 PyPI。

### 从源码安装 { #source-build }

需要 Python 3.11–3.12、C++ 编译器、CMake、Ninja 和预构建的
[LLVM](https://en.wikipedia.org/wiki/LLVM)/MLIR 22.1.8 SDK。
执行前替换 SDK 路径。以下命令使用受支持的 Linux x86-64 / CPython 3.11–3.12 构建环境。

```bash
git clone https://github.com/walkerchi/TIGA-lang.git tiga-lang
cd tiga-lang
python -m venv .venv
source .venv/bin/activate
# Torch 示例先装 Torch；仅使用原生 Tiga 时跳过此行。
python -m pip install torch
export TIGA_LLVM_ROOT=/path/to/llvm-22.1.8
export CMAKE_ARGS="-DMLIR_DIR=$TIGA_LLVM_ROOT/lib/cmake/mlir -DLLVM_DIR=$TIGA_LLVM_ROOT/lib/cmake/llvm"
python -m pip install -e .
python -m tiga
```

Torch 是可选依赖。GPU 场景先自行安装匹配硬件的 Torch，再安装 Tiga；
已有 Torch 时直接使用原环境。默认安装 Tiga 不会安装或升级 Torch。
无 Torch 环境可使用[原生 Tensor 与 CPU JIT 路径](execution.md#native-execution)。

`python -m tiga` 应显示包版本、原生运行时状态和找到的编译器工具。
工具未找到时先解决安装问题，参见[执行与排错](execution.md#diagnose-a-failure)。

## 第一个可微 JIT 程序 { #first-differentiable-jit-program }

从三节点、五条边开始：每条边发送“源节点温度 × 导热系数”，
每个目标节点求和并加上 bias。
[reducer](https://en.wikipedia.org/wiki/Fold_(higher-order_function)) 指消息的合并方式，
本例使用求和。

按顺序运行以下代码。输入和输出直接使用 `torch.Tensor`，
不需要学习额外的 Tensor 包装或转换接口。

```python
--8<-- "examples/torch_quickstart.py:quickstart"
```

读取结果和梯度：

```python
print(output.tolist())        # 约为 [11.1, 8.2, 17.3]
print(d_temperature.tolist()) # [7.0, 10.0, 3.0]
```

预期结果中的 11.1 来自目标节点 0 收到的两条消息，加上它的 bias。
[编程模型](programming-model.md)会逐条解释这些边、数组和字段的对应关系。

无需显式 compile，也无需手写反向函数。输出可继续参与 Torch 运算，
通过 `torch.autograd.grad` 或 `.backward()` 求导。
数值正确不等于所有步骤都已原生编译；计划与产物检查见[执行与排错](execution.md)。

完整源码运行命令：
[`python examples/torch_quickstart.py`](#first-differentiable-jit-program)。
同一小规模程序也可将 `torch.device("cuda")` 改为 `torch.device("cpu")` 运行。
没有 Torch 的环境使用[原生执行例子](execution.md#native-execution)。

## 可选安装方式与依赖 { #optional-installation }

- Torch 示例先单独安装 Torch，再安装 Tiga。原生 Tiga 无需 Torch；
  `tiga-lang[torch]` 仅作为显式选择安装依赖的 extra。
- PyPI extra 使用 `tiga-lang[cuda]` 或 `tiga-lang[nccl-cu12]`（随附 NCCL）。
  源码目录中则使用 `python -m pip install -e ".[cuda]"` 或
  `python -m pip install -e ".[nccl-cu12]"`。GPU 驱动仍需另行安装。

## 开发检查 { #run-tests }

运行程序不要求先配置编译器测试。修改项目源码时，再按
[构建与贡献](development.md)安装测试依赖并配置独立的 MLIR 测试目录。

## 下一步 { #where-next }

1. [编程模型](programming-model.md)：理解图、字段、消息与聚合。
2. [执行与排错](execution.md)：确认执行方式并定位失败原因。
3. [完整示例](examples.md)：把接口用于实际工作负载。

编译器开发者可随后进入[编译器入门](compiler-pipeline.md)与[IR 实例教程](ir-walkthrough.md)。
