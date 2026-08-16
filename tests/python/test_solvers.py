from __future__ import annotations

import importlib.util
from pathlib import Path

import graphforge as gf
import pytest

EXAMPLES = Path(__file__).parents[2] / "examples"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solvers = _load_module(
    "graphforge_example_solvers", EXAMPLES / "solvers.py")


class _ScaledLaplacian(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.value * src.u


def _laplacian_operator():
    graph = gf.Graph.from_csr(
        gf.tensor([0, 2, 4], dtype=gf.int64),
        gf.tensor([0, 1, 0, 1], dtype=gf.int64),
        num_src=2,
    )
    weights = gf.tensor([2.0, -1.0, -1.0, 2.0], dtype=gf.float32)
    return _ScaledLaplacian(), graph, weights


def test_message_passing_kernel_is_the_operator(monkeypatch):
    monkeypatch.setenv("GRAPHFORGE_TENSOR_BACKEND", "native")
    kernel, graph, weights = _laplacian_operator()
    rhs = gf.tensor([1.0, 1.0], dtype=gf.float32)

    solution = solvers.cg(
        kernel,
        rhs,
        graph=graph,
        field="u",
        edge={"value": weights},
        iterations=1,
    )

    assert solution.tolist() == pytest.approx([1.0, 1.0], abs=2.0e-6)
    assert solution.mlir().count("gf_control.repeat") == 1
    assert "gf_tensor.csr_segment_sum" in solution.mlir()


def test_kernel_operator_rejects_bad_bindings():
    kernel, graph, weights = _laplacian_operator()
    rhs = gf.tensor([1.0, 1.0], dtype=gf.float32)

    with pytest.raises(TypeError, match="graph= and field="):
        solvers.cg(kernel, rhs, graph=graph, iterations=1)
    with pytest.raises(ValueError, match="must not also"):
        solvers.cg(
            kernel,
            rhs,
            graph=graph,
            field="u",
            src={"u": rhs},
            edge={"value": weights},
            iterations=1,
        )
    with pytest.raises(ValueError, match="rhs shape"):
        solvers.cg(
            kernel,
            gf.tensor([1.0], dtype=gf.float32),
            graph=graph,
            field="u",
            edge={"value": weights},
            iterations=1,
        )

    rectangular = gf.Graph.from_csr(
        gf.tensor([0, 1], dtype=gf.int64),
        gf.tensor([0], dtype=gf.int64),
        num_src=3,
    )
    with pytest.raises(ValueError, match="square relation"):
        solvers.cg(kernel, rhs, graph=rectangular, field="u", iterations=1)

    with pytest.raises(TypeError, match="plain"):
        solvers.cg(
            lambda value: 2.0 * value,
            rhs,
            graph=graph,
            field="u",
            iterations=1,
        )


def test_callable_operator_applies_without_wrapper():
    diagonal = gf.tensor([2.0, 3.0], dtype=gf.float32)
    actual = solvers.cg(
        lambda value: diagonal * value,
        gf.tensor([4.0, 6.0], dtype=gf.float32),
        iterations=2,
    )
    assert actual.tolist() == pytest.approx([2.0, 2.0], abs=2.0e-6)

    with pytest.raises(ValueError, match="square"):
        solvers.cg(
            lambda value: gf.tensor([1.0], dtype=gf.float32),
            gf.tensor([1.0, 1.0], dtype=gf.float32),
            iterations=1,
        )


def test_dot_and_norm_are_compiler_visible_tensor_algebra():
    left = gf.tensor([1.0, 2.0, 2.0], dtype=gf.float32)
    right = gf.tensor([3.0, 4.0, 5.0], dtype=gf.float32)
    product = solvers.dot(left, right)
    length = solvers.vector_norm(left)

    assert product.tolist() == pytest.approx(21.0)
    assert length.tolist() == pytest.approx(3.0)
    assert "gf_tensor.reduce_sum" in product.mlir()
    assert "gf_tensor.sqrt" in length.mlir()


def test_richardson_is_one_bounded_control_region_and_differentiable(monkeypatch):
    monkeypatch.setenv("GRAPHFORGE_TENSOR_BACKEND", "native")
    diagonal = gf.tensor([2.0, 4.0], dtype=gf.float32)
    rhs = gf.tensor(
        [2.0, 8.0], dtype=gf.float32, requires_grad=True)
    solution = solvers.richardson(
        lambda value: diagonal * value,
        rhs,
        iterations=3,
        relaxation=0.25,
    )

    assert solution.tolist() == pytest.approx([0.875, 2.0], abs=2.0e-6)
    assert solution.mlir().count("gf_control.repeat") == 1
    gradient = gf.autograd.grad(solution.sum(), rhs)
    assert gradient.tolist() == pytest.approx([0.4375, 0.25], abs=2.0e-6)


def test_multi_state_repeat_lowers_to_typed_cpu_double_buffers(monkeypatch):
    monkeypatch.setenv("GRAPHFORGE_TENSOR_BACKEND", "native")
    vector = gf.tensor([1.0, 2.0], dtype=gf.float32)
    scalar = gf.tensor(0.0, dtype=gf.float32)
    result = gf.repeat(
        (vector, scalar),
        lambda current, total: (current + 1.0, total + current.sum()),
        iterations=3,
    )
    assert isinstance(result, tuple)
    shifted, total = result
    combined = shifted.sum() + total
    assert combined.tolist() == pytest.approx(24.0)
    assert combined.mlir().count("gf_control.repeat") == 1
    assert shifted.tolist() == pytest.approx([4.0, 5.0])
    assert total.tolist() == pytest.approx(15.0)
    ir = shifted.mlir(verify=True)
    assert '%3:2 = "gf_control.repeat"' in ir
    assert "num_carried = 2" in ir
    assert '"gf_control.yield"' in ir
    assert (shifted.execution or {})["backend"] == "cpu-llvm-jit"


def test_bounded_while_is_device_control_and_respects_iteration_limit(monkeypatch):
    monkeypatch.setenv("GRAPHFORGE_TENSOR_BACKEND", "native")
    vector = gf.tensor([1.0, 2.0], dtype=gf.float32)
    counter = gf.tensor(0.0, dtype=gf.float32)

    shifted, final_counter = gf.while_loop(
        (vector, counter),
        lambda _value, step: step < 3.0,
        lambda value, step: (value + 1.0, step + 1.0),
        max_iterations=10,
    )
    assert shifted.tolist() == pytest.approx([4.0, 5.0])
    assert final_counter.tolist() == pytest.approx(3.0)
    semantic = shifted.mlir(verify=True)
    assert semantic.count("gf_control.while") == 1
    assert 'predicate = "lt"' in semantic
    lowered = (shifted.execution or {})["artifacts"]["cpu_loop"]
    assert "scf.while" in lowered
    assert "scf.condition" in lowered
    assert "graphforge.cpu.max_iterations = 10" in lowered

    limited = gf.while_loop(
        counter,
        lambda step: step < 10.0,
        lambda step: step + 1.0,
        max_iterations=2,
    )
    assert limited.tolist() == pytest.approx(2.0)


def test_class_based_repeat_and_while_lower_like_the_functional_forms(
    monkeypatch,
):
    monkeypatch.setenv("GRAPHFORGE_TENSOR_BACKEND", "native")

    class Integrate(gf.control.Repeat):
        def __init__(self, rate):
            self.rate = rate

        def body(self, value, total):
            return value * (1.0 + self.rate), total + value.sum()

    vector = gf.tensor([1.0, 2.0], dtype=gf.float32)
    total = gf.tensor(0.0, dtype=gf.float32)
    value_out, total_out = Integrate(0.5)((vector, total), iterations=3)
    assert value_out.tolist() == pytest.approx([3.375, 6.75])
    assert total_out.tolist() == pytest.approx(14.25)
    ir = value_out.mlir(verify=True)
    assert ir.count("gf_control.repeat") == 1
    assert "num_carried = 2" in ir

    class Countdown(gf.control.While):
        def __init__(self, threshold):
            self.threshold = threshold

        def condition(self, value):
            return value > self.threshold

        def body(self, value):
            return value - 1.0

    counter = gf.tensor(5.0, dtype=gf.float32)
    final = Countdown(2.5)(counter, max_iterations=10)
    assert final.tolist() == pytest.approx(2.0)
    semantic = final.mlir(verify=True)
    assert semantic.count("gf_control.while") == 1
    lowered = (final.execution or {})["artifacts"]["cpu_loop"]
    assert "scf.while" in lowered
    assert "graphforge.cpu.max_iterations = 10" in lowered

    with pytest.raises(NotImplementedError):
        gf.control.Repeat()(counter, iterations=1)
    with pytest.raises(NotImplementedError):
        gf.control.While()(counter, max_iterations=1)


def test_fixed_cg_is_one_multi_state_region_and_automatically_differentiable(
    monkeypatch,
):
    monkeypatch.setenv("GRAPHFORGE_TENSOR_BACKEND", "native")
    diagonal = gf.tensor([2.0, 4.0], dtype=gf.float32)
    rhs = gf.tensor(
        [2.0, 8.0], dtype=gf.float32, requires_grad=True)

    solution = solvers.cg(
        lambda value: diagonal * value, rhs, iterations=2)

    assert solution.tolist() == pytest.approx([1.0, 2.0], abs=2.0e-6)
    lowered = (solution.execution or {})["artifacts"]["cpu_loop"]
    assert "graphforge.cpu.loop_scalar_temporaries = 4" in lowered
    assert "graphforge.cpu.loop_tensor_temporaries = 2" in lowered
    # One outer repeat, the initial residual product, and exactly two body
    # reductions. A saved scalar/vector SSA must be loaded from scratch rather
    # than recursively expanding a reduction inside an elementwise loop.
    assert lowered.count("iter_args(") == 4
    assert solution.mlir().count("gf_control.repeat") == 1
    assert "num_carried = 4" in solution.mlir()
    gradient = gf.autograd.grad(solution.sum(), rhs)
    assert gradient.tolist() == pytest.approx([0.5, 0.25], abs=2.0e-5)

    inverse_diagonal = gf.tensor([0.5, 0.25], dtype=gf.float32)
    preconditioned = solvers.cg(
        lambda value: diagonal * value,
        rhs,
        iterations=1,
        preconditioner=lambda residual: inverse_diagonal * residual,
    )
    assert preconditioned.tolist() == pytest.approx([1.0, 2.0], abs=2.0e-6)


def test_tolerance_cg_lowers_to_bounded_while_without_host_polling(monkeypatch):
    monkeypatch.setenv("GRAPHFORGE_TENSOR_BACKEND", "native")
    diagonal = gf.tensor([2.0, 4.0], dtype=gf.float32)
    rhs = gf.tensor(
        [2.0, 8.0], dtype=gf.float32, requires_grad=True)

    solution = solvers.cg(
        lambda value: diagonal * value,
        rhs,
        tolerance=1.0e-6,
        max_iterations=10,
    )
    assert solution.tolist() == pytest.approx([1.0, 2.0], abs=2.0e-6)
    semantic = solution.mlir()
    assert semantic.count("gf_control.while") == 1
    lowered = (solution.execution or {})["artifacts"]["cpu_loop"]
    assert "scf.while" in lowered
    assert "arith.cmpf ogt" in lowered

    already_converged = solvers.cg(
        lambda value: diagonal * value,
        rhs,
        tolerance=100.0,
        max_iterations=10,
    )
    assert already_converged.tolist() == pytest.approx([0.0, 0.0])
    with pytest.raises(NotImplementedError, match="structured control VJP"):
        gf.autograd.grad(solution.sum(), rhs)


def test_fem_poisson_example_is_matrix_free_and_accurate(monkeypatch):
    monkeypatch.setenv("GRAPHFORGE_TENSOR_BACKEND", "native")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    module = _load_module("graphforge_fem_poisson", EXAMPLES / "fem_poisson.py")

    solution, _exact, error = module.run(interior_nodes=6, iterations=3)
    assert error < 2.0e-5
    ir = solution.mlir()
    assert ir.count("gf_control.repeat") == 1
    assert "num_carried = 4" in ir
    assert "gf_tensor.csr_segment_sum" in ir
    assert module.load_gradient().tolist() == pytest.approx(
        [0.4, 0.6, 0.6, 0.4], abs=2.0e-5)
    converged, _exact, converged_error = module.run_until_converged(
        interior_nodes=6, tolerance=1.0e-6, max_iterations=20)
    assert converged_error < 2.0e-5
    assert converged.mlir().count("gf_control.while") == 1


def test_dynamic_radius_linear_solve_keeps_operator_inside_bounded_while(
    monkeypatch,
):
    monkeypatch.setenv("GRAPHFORGE_TENSOR_BACKEND", "native")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    module = _load_module(
        "graphforge_meshfree_linear_solve", EXAMPLES / "meshfree_linear_solve.py")

    solution, residual, graph, kernel = module.solve(points=8)
    assert residual.tolist() < 2.0e-5
    semantic = solution.mlir()
    assert semantic.count("gf_control.while") == 1
    assert "gf_tensor.csr_segment_sum" in semantic
    assert graph.schema.lifecycle == "dynamic"
    assert graph.build_info["builder"] == "uniform_cell_list"
    assert graph.build_info["candidate_pairs"] < 8 * 7
    assert kernel.last_variant is not None
    lowered = solution.generated_code("cpu_loop")
    assert "scf.while" in lowered
