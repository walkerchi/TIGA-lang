<div align="center">
  <img src="assets/tiga-logo.png" alt="Tiga — TG 标志" width="380">
  <p><a href="README.md">English</a> · 简体中文</p>
  <p><strong>面向 graph message-passing 程序的可微 JIT 编译器。</strong></p>
  <p><a href="LICENSE">Apache-2.0</a> · Alpha · 安装包：<code>tiga-lang</code> · Python import：<code>tiga</code></p>
  <p>
    <a href="docs/getting-started.zh.md">快速开始</a> ·
    <a href="docs/api.zh.md">Python API</a> ·
    <a href="docs/examples.zh.md">示例</a> ·
    <a href="docs/roadmap.zh.md">支持范围与路线图</a>
  </p>
</div>

Tiga 将图的连接关系、edge function 和
[reducer](https://en.wikipedia.org/wiki/Fold_(higher-order_function)) 保留为
[compiler IR](https://en.wikipedia.org/wiki/Intermediate_representation)，
利用这些结构选择遍历方式、融合消息计算与聚合，并自动推导梯度。
已支持的路径通过 [JIT](https://en.wikipedia.org/wiki/Just-in-time_compilation)
运行在 CPU 或 NVIDIA GPU 上。日常接口使用普通 PyTorch Tensor；
独立的原生运行时也能在没有 PyTorch 的环境中运行。

## 安装

### 从 PyPI 安装（首次发布后）

首次 PyPI 发布仍在准备中，以下命令用于发布后：

```bash
# 运行 Torch 示例时先安装 Torch；已有兼容版本或使用原生路径时跳过。
python -m pip install torch
python -m pip install tiga-lang
python -m tiga
```

安装包名为 `tiga-lang`，Python import 名为 `tiga`。
**Torch 不是默认安装依赖**：GPU 使用场景先自行安装匹配硬件的 Torch。
安装 Tiga 不会默认安装或升级 Torch；没有 Torch 时可使用
[原生 Tensor、CPU MessagePassing 与 autograd](docs/execution.zh.md#native-execution)。

可选 Triton provider 使用 `python -m pip install "tiga-lang[cuda]"`，
该 extra 不安装 Torch 或 GPU 驱动。兼容的预编译 wheel 包含 Tiga 编译器工具和
原生库，无需另装 LLVM/MLIR SDK。wheel 面向 Linux x86-64、CPython 3.11/3.12
和 glibc ≥ 2.38，见
[安装说明](docs/getting-started.zh.md#pypi-install)。

### 从源码安装

需要 Python 3.11–3.12、C++ 编译器、CMake、Ninja，以及预编译
LLVM/MLIR **22.1.8** SDK。支持的构建环境为 Linux x86-64、CPython 3.11/3.12。
可选 CUDA adapter 使用 Torch 2.11.x 与 Triton 3.6.x。

```bash
git clone https://github.com/walkerchi/TIGA-lang.git tiga-lang
cd tiga-lang
python -m venv .venv
source .venv/bin/activate
# Torch 示例需要；也可使用已有的兼容 Torch 环境。
python -m pip install torch
export TIGA_LLVM_ROOT=/path/to/llvm-22.1.8
export CMAKE_ARGS="-DMLIR_DIR=$TIGA_LLVM_ROOT/lib/cmake/mlir -DLLVM_DIR=$TIGA_LLVM_ROOT/lib/cmake/llvm"
python -m pip install -e .
python -m tiga
```

将 SDK 路径替换为实际位置。

安装失败、输入误用和 bug 报告见[排错与反馈](docs/support.zh.md)。
唯一维护者为 **walkerchi**，Independent Developer，联系邮箱为
[walker.chi.000@gmail.com](mailto:walker.chi.000@gmail.com)。
安全漏洞按 [SECURITY.md](SECURITY.md) 私下报告，不公开提交利用细节。

源码开发可用 `python -m pip install -e ".[test,docs]"` 安装测试与文档依赖。
这些显式选择的开发 extras 包含 Torch，基础安装不包含。

## 一个完整的可微程序

目标节点按行存储的
[CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
描述连接关系，edge function 定义每条边的计算。下面的 NVIDIA GPU 示例完全使用 Torch
输入、输出和 autograd：

```python
import torch
import tiga as tg

device = torch.device("cuda")

class WeightedSum(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x

graph = tg.Graph.from_csr(
    torch.tensor([0, 2, 3, 5], dtype=torch.int64, device=device),
    torch.tensor([0, 2, 1, 0, 1], dtype=torch.int64, device=device),
    num_src=3,
)
x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
weight = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], device=device, requires_grad=True)

kernel = WeightedSum()
out = kernel(graph=graph, src={"x": x}, dst={}, edge={"weight": weight})
dx, dw = torch.autograd.grad(out.sum(), (x, weight))

print(out.tolist())              # [11.0, 8.0, 17.0]
print(dx.tolist())               # [7.0, 10.0, 3.0]
print(dw.tolist())               # [1.0, 3.0, 2.0, 1.0, 2.0]
```

无需显式 compile、转换 Tensor 或手写 backward。
`out` 是 `torch.Tensor`，可直接接入普通 Torch 运算与求导。
通过 `kernel.explain()` 查看所选执行计划，
通过 `kernel.cache_info` 查看 capture variant 缓存统计。
node 更新、自定义 reducer、动态关系和 Torch interop 见
[编程模型](docs/programming-model.zh.md)与[示例](docs/examples.zh.md)。

## 当前支持范围

| 路径 | 已支持范围 | 边界 |
|---|---|---|
| 原生 Tensor | 无 Torch 的 CPU 运行时、已支持 Tensor 运算、CSR message passing 与自动梯度 | `auto` 下小程序可能使用 oracle；不支持的原生操作会报错 |
| 默认 Torch 接口 | 额外的 dense/generated relation 与 neural-module 路径 | 覆盖和融合取决于 relation、shape、dtype 和 provider |
| CPU / NVIDIA GPU | LLVM CPU JIT 与 serialized TTIR → Triton → PTX | 尚未验证 AMD 或 Intel GPU 执行 |
| 内存 / 分布式 | 原生 Tensor 预算、可选 LRU 换出/恢复、paged graph、MPI 与 CUDA TCP/NCCL halo/VJP | CUDA 分页仅支持前向，输出驻留设备；图分区使用固定 ownership |

[支持矩阵](docs/roadmap.zh.md)区分原生与 adapter 覆盖；能构建某种图，
不代表每种 reducer 或梯度都能在其上执行。

内存策略入口为 `tg.execution(...)`；`.spill()` 改变驻留位置，`.to(device)`
复制数据，`tg.save/load` 持久化数值快照。见[完整内存示例与限制](docs/memory.zh.md)。
分布式执行先完成 halo 通信，再计算本地输出。

## 性能证据

[性能与扩展性](docs/experiments.zh.md)提供速度/内存对比、1B 单卡容量与
分布式开销。

<img src="docs/assets/compiler-performance-overview.svg"
     alt="六个已登记工作负载与各自匹配基线的历史测量" width="1100">

上图汇总指定工作负载与硬件上的存档测量。
[基准报告](docs/benchmark-results.zh.md)说明各项基线、计时范围和不确定性；
原始数据和绘图命令见[复现指南](docs/benchmark-reproducibility.zh.md)。

## 编译器与开发

连接关系的语义贯穿 Domain、Iter、Kernel 与 Task IR。
编译器选择实际遍历、内存放置与目标
[lowering](https://en.wikipedia.org/wiki/Compiler#Back_end)，
梯度由自动 [VJP](https://en.wikipedia.org/wiki/Automatic_differentiation) 表达。
[编译流程](docs/compiler-pipeline.zh.md)解释各层职责；
[用实例读懂 IR](docs/ir-walkthrough.zh.md)展示真实输入与编译输出，并提供复现命令。
内部 `gf-*` 工具名和 `graphforge` C++ 目录名保留，公开安装包与 import 名为
`tiga-lang` 和 `tiga`。

[开发指南](docs/development.zh.md)包含 Python、MLIR、文档和 GPU 测试环境，
以及部分 LLVM SDK 缺少的测试工具。贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，
变更记录见 [CHANGELOG.md](CHANGELOG.md)。

采用 [Apache-2.0](LICENSE) 许可证。
