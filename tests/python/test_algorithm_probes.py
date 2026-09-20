from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_pagerank_probe():
    path = (Path(__file__).parents[2] / "examples" / "compiler_probes" /
            "pagerank.py")
    spec = importlib.util.spec_from_file_location("graphforge_pagerank_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pagerank_probe_is_correct_and_uses_bounded_control_ir(monkeypatch):
    probe = _load_pagerank_probe()
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    values, expression_nodes, canonical_ir = probe.run(iterations=5)

    assert abs(sum(values) - 1.0) < 2.0e-6
    assert expression_nodes < 16
    assert canonical_ir.count("gf_control.repeat") == 1
    assert "iterations = 5 : i64" in canonical_ir

    _, larger_nodes, larger_ir = probe.run(iterations=50)
    assert larger_nodes == expression_nodes
    assert larger_ir.count("gf_control.repeat") == 1
    assert "iterations = 50 : i64" in larger_ir


def test_fixed_repeat_uses_automatic_existing_tensor_vjps(monkeypatch):
    import tiga as tg

    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    initial = tg.tensor(
        [2.0, 3.0], dtype=tg.float32, requires_grad=True)
    scale = tg.tensor(0.5, dtype=tg.float32, requires_grad=True)
    output = tg.repeat(
        initial, lambda state: state * scale + 1.0, iterations=3)
    cotangent = tg.tensor([1.0, 1.0], dtype=tg.float32)

    initial_gradient, scale_gradient = tg.autograd.grad(
        output, (initial, scale), grad_output=cotangent)

    assert initial_gradient.tolist() == pytest.approx([0.125, 0.125])
    assert scale_gradient.tolist() == pytest.approx(7.75)
    assert output.mlir().count("gf_control.repeat") == 1


def test_fixed_repeat_composes_message_passing_vjp(monkeypatch):
    import tiga as tg

    class Step(tg.MessagePassing):
        reducer = tg.sum()

        def edge(self, src, dst, edge):
            del dst
            return src.rank * edge.weight

        def node(self, dst, incoming, damping, base):
            del dst
            return base + damping * incoming

    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    nodes, degree, iterations = 4, 2, 3
    graph = tg.Graph.from_csr(
        tg.tensor([0, 2, 4, 6, 8], dtype=tg.int64),
        tg.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=tg.int64),
        num_src=nodes,
    )
    initial = tg.tensor(
        [0.1, 0.2, 0.3, 0.4], dtype=tg.float32, requires_grad=True)
    weight = tg.tensor([0.5] * 8, dtype=tg.float32)
    output = tg.repeat(
        initial,
        lambda state: Step()(
            graph=graph,
            src={"rank": state},
            dst={},
            edge={"weight": weight},
            damping=0.85,
            base=0.15 / nodes,
        ),
        iterations=iterations,
    )

    gradient = tg.autograd.grad(output.sum(), initial)

    assert gradient.tolist() == pytest.approx([0.85 ** iterations] * nodes)


def test_cuda_repeat_fuses_csr_message_reduction_and_node_epilogue(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    import tiga as tg

    class Step(tg.MessagePassing):
        reducer = tg.sum()

        def edge(self, src, dst, edge):
            del dst
            return src.rank * edge.weight

        def node(self, dst, incoming, damping, base):
            del dst
            return base + damping * incoming

    nodes, degree = 128, 4
    row = torch.arange(
        0, nodes * degree + 1, degree, dtype=torch.int64, device="cuda")
    column = (
        torch.arange(nodes, device="cuda")[:, None]
        + torch.arange(degree, device="cuda")[None, :]
    ).remainder(nodes).flatten()
    rank = tg.from_torch(torch.full((nodes,), 1.0 / nodes, device="cuda"))
    weight = tg.from_torch(torch.full(
        (nodes * degree,), 1.0 / degree, device="cuda"))
    graph = tg.Graph.from_csr(
        tg.from_torch(row), tg.from_torch(column), num_src=nodes)
    output = tg.repeat(
        rank,
        lambda current: Step()(
            graph=graph,
            src={"rank": current},
            dst={},
            edge={"weight": weight},
            damping=0.85,
            base=0.15 / nodes,
        ),
        iterations=5,
    )

    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    actual = output.to_torch()
    torch.testing.assert_close(actual, torch.full_like(actual, 1.0 / nodes))
    assert output.execution["backend"] == "cuda-control-loop-ttir-triton"
    assert "gf_tensor_csr_sum_epilogue" in output.generated_code("ttir")
    assert output.generated_code("gf_control").count("gf_control.repeat") == 1
