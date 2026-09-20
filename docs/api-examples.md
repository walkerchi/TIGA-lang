# Runnable API examples

Start with ordinary `torch.Tensor` inputs and PyTorch gradients. Each recipe runs independently; assertions specify results and no variables are borrowed from other sections. The later native-runtime recipes are advanced compiler/storage examples, not a prerequisite for MessagePassing. See [getting started](getting-started.md) for installation. Documentation tests execute these recipes, not just syntax-check them.

| Task | Contract | Complete example |
|---|---|---|
| Default Torch input/output and gradients | [Torch-facing contract](api.md#torch-default) | [Torch recipe](#torch-default) |
| Tensor, views, gradients | [Default Torch interface](api.md#torch-default) | [Tensor recipe](#tensor) |
| CSR aggregation | [Graph](api.md#graph) / [MessagePassing](api.md#messagepassing) | [Message passing](#message-passing) |
| Budgets, spill, persistence | [Execution and storage](api.md#spill-and-disk) | [Storage](#storage) |
| Physical transfers | [Compiler memory](memory-and-distributed.md#capacity-and-version-accounting) | [Physical instances](#physical-instances) |
| Backend selection and failures | [Execution guide](execution.md) | [Diagnostic code](execution.md#native-execution) |
| IR and generated code | [IR walkthrough](ir-walkthrough.md) | [Stage artifacts](ir-walkthrough.md#computation) |
| Reducers and semantics | [Reducers](reducers.md) | [Graph algorithms](examples/message-passing.md) |
| Torch / NN interop | [Interop](examples/programs-and-interop.md) | [Attention](examples/attention.md) |
| Loops and implicit VJP | [Solvers](linear-solvers.md) | [Solver examples](examples/solvers.md) |
| Multiple processes and paged graphs | [Support status](roadmap.md) | [Distributed examples](examples/distributed-memory.md) |

## Default: Torch inputs, outputs and gradients { #torch-default }

Install Torch separately before Tiga for this recipe; no CUDA or `tg.from_torch`
wrapper is needed. Native storage/runtime recipes do not require Torch.

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

This validates the public value/gradient contract, not a claim that every provider
uses generated code. Inspect the kernel's diagnostics for the selected execution route.

## Torch Tensor, views and gradients { #tensor}

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
view = x.transpose(0, 1)
loss = (view * view).sum()
dx, = torch.autograd.grad(loss, x)
assert loss.tolist() == 30.
assert dx.tolist() == [[2., 4.], [6., 8.]]
```

## CSR message passing { #message-passing}

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

## Storage and allocation policy { #storage}

```python
--8<-- "examples/hierarchical_memory.py:core"
```

## Compiler physical-instance round trip { #physical-instances}

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
