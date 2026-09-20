# 执行与排错 { #execution-and-troubleshooting }

[跑通第一个程序](getting-started.md)之后，通过本页了解计算触发时机、确认 JIT
执行路径，并定位运行错误。不要求预先理解编译器 IR。

## 默认 Torch 路径 { #torch-execution }

Torch 输入得到 Torch 输出，使用 Torch 的求导接口。计划诊断在 kernel 上：

```python
--8<-- "examples/torch_quickstart.py:quickstart"
print(kernel.explain())
```

Torch 结果没有 Tiga 的 `.execution`、`.realize()` 或 `.generated_code()`。
`TIGA_TENSOR_BACKEND` 控制原生 Tensor 求值策略，不是所有 Torch provider 的总开关。
Torch 适配器允许的参考执行与编译路径取决于具体关系和 provider；数值正确不等于已编译。

当前 weighted-sum 与 scalar CSR 编译前向的 Torch 求导桥使用固定拓扑的语义重放，
`kernel.explain()` 会注明 backward 并非生成的 TTIR kernel。feature 宽度不满足直接
weighted tile 的 2 的幂次条件时使用 sparse provider；这不是切换 Tensor 类型。

## 要求原生执行（高级） { #native-execution }

原生 Tensor 记录延迟表达式，读取结果时才按选定策略执行。
Python [oracle](https://en.wikipedia.org/wiki/Test_oracle) 是用于核对结果的
参考求值器，不是编译生成的机器代码。

```python
import os
os.environ["TIGA_TENSOR_BACKEND"] = "native"

import tiga as tg

x = tg.tensor([1.0, 2.0, 3.0], requires_grad=True)
out = (x * x).sum()
dx = tg.autograd.grad(out, x)

print(out.tolist())                 # 14.0
print(dx.tolist())                  # [2.0, 4.0, 6.0]
print(out.execution["backend"])     # cpu-llvm-jit
```

`native` 要求原生执行，无法编译时会报告失败。默认 `auto` 策略可能为小
表达式选择 `python-oracle`。更改策略不会使原本不支持的操作或设备获得支持。

## 正确解读诊断信息 { #read-the-right-diagnostics }

| 现象 | 含义 |
|---|---|
| 原生程序返回了 Tensor | 可能只捕获了表达式，尚未实际计算 |
| `tolist()`、`to_numpy()` 或 `realize()` 完成 | 值已可用；实际如何得到该值需要查执行信息 |
| `out.execution["backend"] == "cpu-llvm-jit"` | 这个结果通过 CPU JIT 路径执行 |
| `out.execution` 为 `None` | 可能尚未执行，也可能是无需执行的输入、常量或 view |
| `kernel.cache_info` 显示命中 | 找到了 capture 变体，不证明发生过编译或执行 |
| 原生 `out.execution` 包含 `cache_hit` / `compile_ms` | 描述实际编译缓存行为，不是 capture 查询次数 |

[backend](https://en.wikipedia.org/wiki/Compiler#Back_end) 字段标识实际执行路径。
`kernel.explain()` 描述规划；编译后用 `out.generated_code()` 查看原生产物。
调用成功或 capture 缓存命中都不是性能测量。

## 排查问题 { #diagnose-a-failure }

| 现象 | 首先检查 |
|---|---|
| import 失败或找不到原生工具 | 运行 `python -m tiga`，检查当前环境与[源码安装](getting-started.md#source-build) |
| 调用报缺少字段或参数 | 对照[消息传递契约](message-passing.md#field-namespaces)检查 `src`、`dst`、`edge` 和首维 |
| 图能构造但不能执行 | 在[支持矩阵](roadmap.md#feature-support)中检查具体入口 |
| 小示例报告 `python-oracle` | 在 `auto` 下可能正常；验证 JIT 时显式要求 `native` |
| CUDA 或 Torch 互操作不可用 | 单独安装适合硬件的 Torch，按需安装扩展依赖并检查 GPU/驱动；`cuda` 不会安装 Torch |
| `kernel.ir("iter")` 抛出 `KeyError` | 当前变体可能没有该产物，参见[检查边界](compiler-pipeline.md#inspecting-a-compiled-program) |
| 首次运行比重复运行慢很多 | 分别测量编译、传输与执行，参见[测量协议](performance.md) |

不能把参考求值结果当作编译结果来掩盖原生编译失败。有效的问题记录包含
最小程序、异常、`python -m tiga` 输出与已有执行诊断，不应包含私有数据。

## 下一步 { #next }

- [示例](examples.md)：将 API 用于具体工作负载。
- [Python API](api.md)：查询签名与精确约束。
- [编译器入门](compiler-pipeline.md)：理解内部决策。
