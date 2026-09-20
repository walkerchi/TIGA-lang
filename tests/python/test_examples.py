"""Execute the public examples and verify their results.

Examples show only the main flow; every correctness check that used to live
inline in the example files lives here instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

EXAMPLES = Path(__file__).parents[2] / "examples"

CUDA = torch.cuda.is_available()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(name: str, filename: str):
    return _load_module(f"graphforge_example_{name}", EXAMPLES / filename)


# --- Tensor and GPU kernels -------------------------------------------------


def test_tensor_matmul_example_results():
    module = _load("tensor_matmul", "tensor_matmul.py")
    assert module.output.tolist() == [[14.0, 4.0], [32.0, 13.0]]
    assert module.dlhs.tolist() == [[4.0, 6.0, 2.0], [-1.5, 1.5, -4.5]]
    assert module.drhs.tolist() == [[-3.0, 4.0], [-3.0, 6.5], [-3.0, 9.0]]


def test_complex_autograd_example_results():
    module = _load("complex_autograd", "complex_autograd.py")
    assert module.y.tolist() == [1.0 + 2.0j, 2.0 + 0.5j, 3.0 - 4.0j, -1.0 + 3.0j]
    assert module.dx.tolist() == [
        [2.0 + 4.0j, 6.0 - 8.0j],
        [4.0 + 1.0j, -2.0 + 6.0j],
    ]


@pytest.mark.skipif(not CUDA, reason="CUDA is unavailable")
def test_linear_recurrence_example_matches_torch_oracle():
    module = _load("linear_recurrence", "linear_recurrence.py")
    expected = (
        module.q_torch.unsqueeze(-1)
        * (module.k_torch.unsqueeze(-1) * module.v_torch.unsqueeze(-2)).cumsum(1)
    ).sum(dim=2)
    torch.testing.assert_close(module.actual, expected, rtol=2e-5, atol=2e-4)


def test_gpu_heatmap_example_writes_a_png(tmp_path, monkeypatch):
    module = _load("gpu_heatmap", "gpu_heatmap.py")
    destination = tmp_path / "heatmap.png"
    monkeypatch.setattr(
        sys, "argv",
        ["gpu_heatmap.py", "--size", "64", "--output", str(destination)])
    module.main()
    assert destination.exists() and destination.stat().st_size > 0


# --- Message passing and graph algorithms -----------------------------------


def test_message_passing_autograd_example_results():
    module = _load("message_passing_autograd", "message_passing_autograd.py")
    assert module.output.tolist() == pytest.approx([11.1, 8.2, 17.3])
    assert module.d_temperature.tolist() == [7.0, 10.0, 3.0]
    assert module.d_conductivity.tolist() == [1.0, 3.0, 2.0, 1.0, 2.0]
    assert module.d_bias.tolist() == [1.0, 1.0, 1.0]
    assert isinstance(module.output, torch.Tensor)
    assert isinstance(module.d_temperature, torch.Tensor)


def test_native_checkpoint_example_results():
    module = _load("native_checkpoint", "native_checkpoint.py")
    assert module.saved_d_conductivity.tolist() == [1., 3., 2., 1., 2.]
    assert "checkpoint" in module.saved_d_conductivity.expression()


def test_torch_quickstart_results():
    if not CUDA:
        pytest.skip("Quickstart requires CUDA; exercised by GPU release gate")
    module = _load("torch_quickstart", "torch_quickstart.py")
    torch.testing.assert_close(module.output, module.output.new_tensor([11.1, 8.2, 17.3]))
    torch.testing.assert_close(module.d_temperature, module.output.new_tensor([7., 10., 3.]))
    torch.testing.assert_close(module.d_conductivity, module.output.new_tensor([1., 3., 2., 1., 2.]))
    torch.testing.assert_close(module.d_bias, torch.ones_like(module.d_bias))


def test_ir_csr_walkthrough_results():
    module = _load("ir_csr_walkthrough", "ir_csr_walkthrough.py")
    output, gradient = module.main()
    torch.testing.assert_close(output, torch.tensor([23., 6., 0.]))
    torch.testing.assert_close(gradient, torch.tensor([4., 7.]))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_default_torch_interface_backward_and_reuse(device):
    if device == "cuda" and not CUDA:
        pytest.skip("CUDA is unavailable")
    import tiga as tg

    class WeightedSum(tg.MessagePassing):
        reducer = tg.sum()

        def edge(self, src, dst, edge):
            return src.x * edge.weight

    graph = tg.Graph.from_csr(
        torch.tensor([0, 2, 3, 3], device=device),
        torch.tensor([0, 1, 1], device=device),
        num_src=2, validate="full",
    )
    kernel = WeightedSum()
    # Reusing the kernel must bind current Torch inputs, including their graph.
    for scale in (1., 2.):
        x = torch.tensor([2. * scale, 3. * scale], device=device, requires_grad=True)
        weight = torch.tensor([4., 5., 2.], device=device, requires_grad=True)
        out = kernel(graph=graph, src={"x": x}, dst={}, edge={"weight": weight})
        assert isinstance(out, torch.Tensor)
        assert out.device == x.device and out.dtype == x.dtype
        torch.testing.assert_close(out, x.new_tensor([23., 6., 0.]) * scale)
        # Ordinary Torch operations remain connected to both input gradients.
        (out * 2).sum().backward()
        torch.testing.assert_close(x.grad, x.new_tensor([8., 14.]))
        torch.testing.assert_close(weight.grad, x.new_tensor([4., 6., 6.]) * scale)


def test_gcn_example_results():
    module = _load("gcn", "gcn.py")
    assert module.output.tolist() == [[5.5, 7.0], [6.0, 8.0], [2.25, 4.0]]
    assert module.dx.tolist() == [[2.0, 2.0], [2.25, 2.25], [1.0, 1.0]]
    assert module.dweight.tolist() == [[3.0], [11.0], [7.0], [3.0], [7.0]]


def test_diffusion_example_results():
    module = _load("diffusion", "diffusion.py")
    assert module.next_u.tolist() == pytest.approx(
        [0.05, 0.85, 0.45, -0.35], abs=1e-6)
    assert module.du.tolist() == pytest.approx([1.0] * 4, abs=1e-6)
    assert module.d_conductivity.tolist() == pytest.approx(
        [-0.05, 0.1, -0.1, -0.05, 0.05, -0.1, 0.1, 0.05], abs=1e-6)


def test_custom_reducer_example_results():
    module = _load("custom_reducer", "custom_reducer.py")
    output, gradient = module.main()
    assert output.tolist() == [2.5, 2.0, 1.5]
    assert gradient.tolist() == [1.0, 1.5, 0.5]


def _pagerank_reference(iterations: int, damping: float) -> list[float]:
    """Framework-independent oracle for the example's 4-node graph."""
    rank = [0.25] * 4
    incoming = ((2,), (0,), (1,), ())
    out_degree = (1, 1, 1, 0)
    for _ in range(iterations):
        dangling_mass = rank[3]
        base = (1.0 - damping) / 4.0 + damping * dangling_mass / 4.0
        rank = [
            base + damping * sum(
                rank[source] / out_degree[source] for source in sources)
            for sources in incoming
        ]
    return rank


def test_pagerank_probe_values_and_rolled_loop():
    module = _load("pagerank", "compiler_probes/pagerank.py")
    actual, nodes, canonical_ir = module.run()
    assert actual == pytest.approx(
        _pagerank_reference(20, 0.85), abs=2.0e-6)
    assert sum(actual) == pytest.approx(1.0, abs=2.0e-6)
    assert nodes == 10
    assert canonical_ir.count("gf_control.repeat") == 1


# --- Dynamic and generated relations -----------------------------------------


def test_radius_autograd_example_results():
    module = _load("radius_autograd", "radius_autograd.py")
    assert module.output.tolist() == pytest.approx([0.9, 3.1, 1.5], abs=1e-6)
    flat = [v for row in module.position_grad.tolist() for v in row]
    assert flat == pytest.approx(
        [-7.0, 0.0, -12.0, 0.0, 19.0, 0.0], abs=1e-6)
    assert module.source_grad.tolist() == pytest.approx([0.6, 1.8, 1.0], abs=1e-6)


@pytest.mark.skipif(not CUDA, reason="CUDA is unavailable")
def test_knn_message_passing_example_contract():
    module = _load("knn_message_passing", "knn_message_passing.py")
    assert module.result.shape == (1024,)
    assert module.result.dtype == torch.float32
    assert torch.isfinite(module.result).all()


# --- Programs, autograd and Torch interop -------------------------------------


def test_graph_program_example_fuses_two_leaves():
    module = _load("graph_program", "graph_program.py")
    assert "applies=2, post_fusion=1" in module.first.program.explain()
    if CUDA:
        out0, out1 = module.first.program.run()
        assert out0.shape == (module.nodes,) and out1.shape == (module.nodes,)
        assert out0.dtype == torch.float32 and out1.dtype == torch.float32


def test_joint_autograd_example_results():
    module = _load("joint_autograd", "joint_autograd.py")
    assert module.value.tolist() == pytest.approx(13.0, abs=1e-6)
    assert module.dx.tolist() == [4.0, 6.0]


def test_torch_interop_example_results():
    module = _load("torch_interop", "torch_interop.py")
    assert torch.allclose(
        module.output, module.x.roll(1) + module.x.roll(-1), atol=1e-6)
    assert module.round_trip.data_ptr() == module.x.data_ptr()


def test_torch_library_example_results():
    module = _load("torch_library", "torch_library.py")
    assert module.operation.qualified_name.startswith(
        "tiga::message_passing")
    assert torch.allclose(module.compiled_output, torch.tensor([8.0, 38.0]))
    assert torch.allclose(module.dx, torch.tensor([2.0, 8.0, 7.0]))
    assert torch.allclose(module.dweight, torch.tensor([1.0, 2.0, 2.0, 4.0]))
    opcheck = module.operation.opcheck(
        module.x.detach(), module.weight.detach())
    assert all(result == "SUCCESS" for result in opcheck.values())


def test_edge_nn_example_matches_eager_oracle():
    module = _load("edge_nn_message_passing", "edge_nn_message_passing.py")
    reference = module.kernel.reference(
        graph=module.graph, src={"x": module.x}, dst={})
    assert torch.allclose(module.output, reference, atol=1e-3, rtol=1e-3)
    reference2 = module.kernel2.reference(
        graph=module.graph, src={"x": module.x}, dst={})
    assert torch.allclose(module.output2, reference2, atol=1e-3, rtol=1e-3)
    assert module.xg.grad is not None and torch.isfinite(module.xg.grad).all()
    if CUDA:
        lowerings = {variant.lowering for variant in module.kernel.variants}
        assert "gf-python-emit-edge-nn-tile" in lowerings
        assert "gf-python-emit-edge-nn-tile-vjp" in lowerings


def test_gat_edge_attention_example_matches_oracle():
    module = _load("gat_edge_attention", "gat_edge_attention.py")
    reference = module.kernel.reference(
        graph=module.graph, src={"x": module.x}, dst={"x": module.x})
    assert torch.allclose(module.output, reference, atol=1e-3, rtol=1e-3)
    assert module.xg.grad is not None and torch.isfinite(module.xg.grad).all()
    assert torch.isfinite(module.positions_g.grad).all()
    if CUDA:
        lowerings = {variant.lowering for variant in module.kernel.variants}
        assert "gf-python-emit-edge-nn-attention-tile" in lowerings
        assert "gf-python-emit-edge-nn-attention-vjp" in lowerings


# --- Distributed and memory hierarchy -----------------------------------------


def test_distributed_halo_example_shards(monkeypatch):
    module = _load("distributed_halo", "distributed_halo.py")
    # fork inherits this importlib-loaded module; spawn would re-import it by
    # a name the child cannot resolve.
    import multiprocessing

    original = multiprocessing.get_context
    monkeypatch.setattr(
        multiprocessing, "get_context", lambda method: original("fork"))
    shards = module.main()
    # 8-node ring, x[i] = i: out[i] = x[i-1] + x[i] + x[i+1]; every node is
    # read by exactly three rows, so the gradient is 3 everywhere.
    expected_values = [8.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 13.0]
    for rank in (0, 1):
        values, gradients = shards[rank]
        assert values == pytest.approx(
            expected_values[4 * rank:4 * rank + 4], abs=1e-6)
        assert gradients == pytest.approx([3.0] * 4, abs=1e-6)


def test_hierarchical_memory_example_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("TIGA_SPILL_DIR", str(tmp_path))
    module = _load("hierarchical_memory", "hierarchical_memory.py")
    assert module.back.tolist() == [float(i) for i in range(8)]
    assert module.gradients.tolist() == [float(2 * i) for i in range(8)]
    assert module.restored.tolist() == module.back.tolist()


def test_paged_giant_graph_example_round_trip(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TIGA_SPILL_DIR", str(tmp_path))
    module = _load("paged_giant_graph", "paged_giant_graph.py")
    # Small grid keeps the test fast; main() asserts the disk round trip
    # (paged execution + named-spill reattach checksum) internally.
    module.main(grid=(40, 30))
    summary = capsys.readouterr().out.strip()
    assert summary.startswith("paged 1,200 nodes / 4,660 edges")
    assert "round-tripped to disk" in summary


def test_auto_offload_example_pages_over_budget(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TIGA_SPILL_DIR", str(tmp_path))
    module = _load("auto_offload", "auto_offload.py")
    # main() asserts the stencil lands as paged_csr under the small budget.
    module.main(grid=(40, 30), budget=16 << 10)
    summary = capsys.readouterr().out.strip()
    assert summary.startswith("auto-offloaded 1,200 nodes / 4,660 edges")
