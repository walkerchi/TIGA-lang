# 可运行 API 例子 { #runnable-api-examples }

常规入口直接使用 `torch.Tensor` 和 PyTorch autograd。每节代码独立运行，通过断言给出结果，不依赖前一节变量。后面的原生 runtime 例子用于编译器与存储的高级场景，不是使用 MessagePassing 的前置要求。依赖与构建见[快速上手](getting-started.zh.md)。这些代码纳入文档执行测试，不只是伪代码。

| 任务 | API 契约 | 完整例子 |
|---|---|---|
| 默认 Torch 输入、输出与梯度 | [Torch 接口契约](api.zh.md#torch-default) | [本页 Torch](#torch-default) |
| Tensor、view、梯度 | [默认 Torch 接口](api.zh.md#torch-default) | [本页 Tensor](#tensor) |
| CSR 聚合 | [Graph](api.zh.md#graph) / [MessagePassing](api.zh.md#messagepassing) | [本页 message passing](#message-passing) |
| 预算、spill、持久化 | [执行与存储](api.zh.md#spill-and-disk) | [本页存储](#storage) |
| 物理层搬运 | [编译器内存](memory-and-distributed.zh.md#capacity-and-version-accounting) | [本页物理实例](#physical-instances) |
| 选择 backend、排错 | [执行指南](execution.zh.md) | [执行诊断代码](execution.zh.md#native-execution) |
| IR 与生成代码 | [IR 实例](ir-walkthrough.zh.md) | [完整阶段产物](ir-walkthrough.zh.md#computation) |
| Reducer、自定义语义 | [Reducer](reducers.zh.md) | [图算法例子](examples/message-passing.zh.md) |
| Torch / NN 互操作 | [互操作](examples/programs-and-interop.zh.md) | [Attention 例子](examples/attention.zh.md) |
| 循环、隐式 VJP | [求解器](linear-solvers.zh.md) | [求解器例子](examples/solvers.zh.md) |
| 多进程与分页图 | [支持范围](roadmap.zh.md) | [分布式例子](examples/distributed-memory.zh.md) |

## 默认入口：Torch 输入、输出与梯度 { #torch-default }

本例先单独安装 Torch，再安装 Tiga；CPU 即可运行，不需要 `tg.from_torch` 包装。
后面的原生存储/runtime 例子不要求 Torch。

```python
import torch
import tiga as tg

class WeightedSum(tg.MessagePassing):
    reducer = tg.sum()
    def edge(self, src, dst, edge):
        return src.x * edge.w

graph = tg.Graph.from_csr(torch.tensor([0, 2, 3, 3]),
                          torch.tensor([0, 1, 1]), num_src=2, validate="full")
x = torch.tensor([2., 3.], requires_grad=True)
w = torch.tensor([4., 5., 2.], requires_grad=True)
out = WeightedSum()(graph=graph, src={"x": x}, dst={}, edge={"w": w})
assert isinstance(out, torch.Tensor)
torch.testing.assert_close(out, torch.tensor([23., 6., 0.]))
dx, dw = torch.autograd.grad(out.sum(), (x, w))
torch.testing.assert_close(dx, torch.tensor([4., 7.]))
torch.testing.assert_close(dw, torch.tensor([2., 3., 3.]))
```

该例验证数值与求导契约，不表示所有 provider 都生成了机器代码；实际执行路线见 kernel diagnostics。

## Torch Tensor、view 与梯度 { #tensor}

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
view = x.transpose(0, 1)
loss = (view * view).sum()
dx, = torch.autograd.grad(loss, x)
assert loss.tolist() == 30.
assert dx.tolist() == [[2., 4.], [6., 8.]]
```

## CSR 消息传递 { #message-passing}

```python
import torch
import tiga as tg

class WeightedSum(tg.MessagePassing):
    reducer = tg.sum()
    def edge(self, src, dst, edge):
        return src.x * edge.w

graph = tg.Graph.from_csr(torch.tensor([0, 2, 3], dtype=torch.int64),
                          torch.tensor([0, 1, 0], dtype=torch.int64), num_src=2)
x = torch.tensor([2., 3.], requires_grad=True)
w = torch.tensor([1., 2., 4.], requires_grad=True)
out = WeightedSum()(graph=graph, src={"x": x}, dst={}, edge={"w": w})
assert out.tolist() == [8., 8.]
assert torch.autograd.grad(out.sum(), x)[0].tolist() == [5., 2.]
```

## 存储与分配策略 { #storage}

```python
--8<-- "examples/hierarchical_memory.py:core"
```

## 编译器物理实例往返 { #physical-instances}

```python
import tiga as tg

with tg.runtime.HierarchyRuntime(budgets={"ram": 8, "nvme": 4}) as rt:
    source = rt.allocate("x", 0, tier="ram", capacity_bytes=4)
    source.buffer.write(b"tiga")
    page = rt.allocate("x", 0, tier="nvme", capacity_bytes=4)
    rt.transfer(source, page).wait()
    destination = rt.allocate("x", 0, tier="ram", capacity_bytes=4)
    rt.transfer(page, destination).wait()
    assert destination.buffer.read() == b"tiga"
```
