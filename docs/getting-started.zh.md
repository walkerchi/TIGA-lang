# 快速上手 { #getting-started }

## 安装 { #install }

```bash
pip install tiga-lang   # PyPI's bare "tiga" is an unrelated project
```

```python
import tiga as gf
gf.__version__  # e.g. "0.1.0"
```

wheel 自带预编译的编译器工具链；只有开发时才需要从源码构建。

## 源码构建 { #source-build }

```bash
git clone https://github.com/walkerchi/TIGA-lang.git
cd tiga-lang
export CMAKE_ARGS="-DMLIR_DIR=/path/to/llvm-22.1.8/lib/cmake/mlir"
python -m pip install -e ".[cuda,dev]"
python -m tiga
```

`python -m tiga` 会打印包版本、原生运行时状态和解析到的编译器
工具；装了可选组件时还会显示 [Torch](https://en.wikipedia.org/wiki/PyTorch)
适配器与 [CUDA](https://en.wikipedia.org/wiki/CUDA) provider 状态。`.[cuda]`
会装 Triton，但不装 Torch；需要框架互操作时再加 `.[torch]`，CUDA 部署
需要随附 NCCL 库时再加 `.[nccl-cu12]`。

## 第一个可微 JIT 程序 { #first-differentiable-jit-program }

```python
--8<-- "examples/message_passing_autograd.py"
```

首次调用会根据图的来源（provenance）、[张量](https://baike.baidu.com/item/张量)
dtype、shape/stride、编译目标和 provider 标识生成特化版本，并在第一次
读取结果（`tolist()`、`to_numpy()`、`realize()`）时编译。守卫条件不变的
后续调用复用同一份可执行文件。`kernel.cache_info` 如实报告这一点：
`miss` 是一次编译，`hit` 是一次复用；`kernel.explain()` 打印 lowering
与同样的计数，`kernel.ir(stage)` / `kernel.code("ptx")` 暴露各阶段产物。

## 运行测试 { #run-tests }

```bash
python -m pytest
cmake --build build --target check-tiga
```

Python 测试套件覆盖捕获、运行时和[自动微分](https://baike.baidu.com/item/自动微分)行为；lit
测试套件检查 dialect
验证器、变换以及 provider 翻译。

## 延伸阅读 { #where-next }

- [编程模型](programming-model.md) — 一张图看懂关系、UDF 和 reducer
- [示例](examples.md) — 按类别组织的可运行程序
- [基准测试结果](benchmark-results.md) — 实测数据
