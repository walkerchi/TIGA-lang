"""Torch-facing CSR contracts must hold across provider and cache paths."""
import pytest
import torch
import tiga as tg


class WeightedSum(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x * edge.w


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("features", [1, 3, 4])
@pytest.mark.parametrize("degrees,num_src", [([2, 2, 2], 2), ([2, 1, 0], 5)])
def test_torch_fields_train_infer_and_reuse(device, features, degrees, num_src):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    counts = torch.tensor(degrees, device=device)
    row = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    col = torch.arange(sum(degrees), device=device) % num_src
    dst = torch.repeat_interleave(torch.arange(len(degrees), device=device), counts)
    graph = tg.Graph.from_csr(row, col, num_src=num_src, validate="full")
    shape = (num_src,) if features == 1 else (num_src, features)
    x = torch.arange(1, 1 + num_src * features, device=device, dtype=torch.float32)
    x = x.reshape(shape).requires_grad_()
    w = torch.linspace(.5, 1.5, col.numel(), device=device)
    if features > 1:
        w = w[:, None]
    w.requires_grad_()
    kernel = WeightedSum()

    def call():
        return kernel(graph=graph, src={"x": x}, dst={}, edge={"w": w})

    def oracle():
        messages = x[col] * w
        return messages.new_zeros((len(degrees), *x.shape[1:])).index_add(0, dst, messages)

    # Inference-first caching must not detach later training calls.
    with torch.no_grad():
        saved_view = call().view(-1)
        saved_values = saved_view.clone()
    for _ in range(2):
        with torch.no_grad():
            x.add_(.125)
        out = call()
        torch.testing.assert_close(out, oracle())
        torch.testing.assert_close(saved_view, saved_values)
        actual = torch.autograd.grad(out.square().sum(), (x, w))
        expected = torch.autograd.grad(oracle().square().sum(), (x, w))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
    with torch.no_grad():
        torch.testing.assert_close(call(), oracle())
    if device == "cuda" and features != 3:
        assert "weighted-sum" in kernel.last_variant.lowering
        assert "Torch gradients replay" in kernel.explain()
    elif features == 3:
        assert kernel.last_variant.provider == "torch.sparse.mm"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_node_update_backward(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    class BiasedSum(WeightedSum):
        def node(self, dst, incoming):
            return incoming + dst.bias

    graph = tg.Graph.from_csr(torch.tensor([0, 2, 3, 3], device=device),
                              torch.tensor([0, 1, 1], device=device), num_src=2)
    x = torch.tensor([2., 3.], device=device, requires_grad=True)
    w = torch.tensor([4., 5., 2.], device=device, requires_grad=True)
    bias = torch.tensor([.1, .2, .3], device=device, requires_grad=True)
    kernel = BiasedSum()
    for grad_mode in (False, True, True):
        with torch.set_grad_enabled(grad_mode):
            out = kernel(graph=graph, src={"x": x}, dst={"bias": bias}, edge={"w": w})
            torch.testing.assert_close(out, x.new_tensor([23.1, 6.2, .3]))
            if grad_mode:
                dx, dw, db = torch.autograd.grad(out.sum(), (x, w, bias))
                torch.testing.assert_close(dx, x.new_tensor([4., 7.]))
                torch.testing.assert_close(dw, w.new_tensor([2., 3., 3.]))
                torch.testing.assert_close(db, torch.ones_like(bias))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_weighted_sum_checks_saved_torch_versions():
    graph = tg.Graph.from_csr(torch.tensor([0, 1, 2], device="cuda"),
                              torch.tensor([1, 0], device="cuda"), num_src=2)
    x = torch.tensor([2., 3.], device="cuda", requires_grad=True)
    w = torch.ones(2, device="cuda", requires_grad=True)
    out = WeightedSum()(graph=graph, src={"x": x}, dst={}, edge={"w": w})
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


@pytest.mark.parametrize("node_update", [False, True])
@pytest.mark.parametrize("dependent_weight", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_torch_alias_and_dependent_arguments(node_update, dependent_weight):
    class BiasedSum(WeightedSum):
        def node(self, dst, incoming):
            return incoming + dst.bias

    col = torch.tensor([1, 0], device="cuda")
    graph = tg.Graph.from_csr(torch.tensor([0, 1, 2], device="cuda"), col, num_src=2)
    x = torch.tensor([2., 3.], device="cuda", requires_grad=True)
    w = x * 2 if dependent_weight else x
    kernel = BiasedSum() if node_update else WeightedSum()
    out = kernel(graph=graph, src={"x": x}, dst={"bias": x} if node_update else {},
                 edge={"w": w})
    expected = x[col] * w + (x if node_update else 0)
    actual_grad, = torch.autograd.grad(out.square().sum(), x, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected.square().sum(), x)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("requested", ["source", "weight", "both"])
def test_high_degree_bipartite_gradients(device, requested):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    row = torch.tensor([0, 65, 66, 66], device=device)
    col = torch.arange(66, device=device)
    destination = torch.cat((col.new_zeros(65), col.new_ones(1)))
    graph = tg.Graph.from_csr(row, col, num_src=68, validate="full")
    x = torch.linspace(.1, .8, 68, device=device, requires_grad=requested != "weight")
    w = torch.linspace(.2, .9, 66, device=device, requires_grad=requested != "source")
    kernel = WeightedSum()
    inputs = [v for v in (x, w) if v.requires_grad]
    for _ in range(2):
        out = kernel(graph=graph, src={"x": x}, dst={}, edge={"w": w})
        expected = x.new_zeros(3).index_add(0, destination, x[col] * w)
        torch.testing.assert_close(out, expected)
        actual_grad = torch.autograd.grad(out.sum(), inputs)
        expected_grad = torch.autograd.grad(expected.sum(), inputs)
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_strided_torch_fields_use_supported_provider(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    x = torch.arange(1., 11., device=device)[::2].requires_grad_()
    w = torch.arange(1., 7., device=device)[::2].requires_grad_()
    row = torch.tensor([0, 2, 3, 3], device=device)
    col = torch.tensor([0, 1, 1], device=device)
    graph = tg.Graph.from_csr(row, col, num_src=5)
    kernel = WeightedSum()
    out = kernel(graph=graph, src={"x": x}, dst={}, edge={"w": w})
    torch.testing.assert_close(out, x.new_tensor([10., 15., 0.]))
    dx, dw = torch.autograd.grad(out.sum(), (x, w))
    torch.testing.assert_close(dx, x.new_tensor([1., 8., 0., 0., 0.]))
    torch.testing.assert_close(dw, w.new_tensor([1., 3., 3.]))
    assert kernel.last_variant.provider == "torch.sparse.mm"
