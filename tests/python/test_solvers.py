from __future__ import annotations

import importlib.util
from pathlib import Path

import tiga as tg
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


class _ScaledLaplacian(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.value * src.u


def _laplacian_operator():
    graph = tg.Graph.from_csr(
        tg.tensor([0, 2, 4], dtype=tg.int64),
        tg.tensor([0, 1, 0, 1], dtype=tg.int64),
        num_src=2,
    )
    weights = tg.tensor([2.0, -1.0, -1.0, 2.0], dtype=tg.float32)
    return _ScaledLaplacian(), graph, weights


def test_message_passing_kernel_is_the_operator(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    kernel, graph, weights = _laplacian_operator()
    rhs = tg.tensor([1.0, 1.0], dtype=tg.float32)

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
    rhs = tg.tensor([1.0, 1.0], dtype=tg.float32)

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
            tg.tensor([1.0], dtype=tg.float32),
            graph=graph,
            field="u",
            edge={"value": weights},
            iterations=1,
        )

    rectangular = tg.Graph.from_csr(
        tg.tensor([0, 1], dtype=tg.int64),
        tg.tensor([0], dtype=tg.int64),
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
    diagonal = tg.tensor([2.0, 3.0], dtype=tg.float32)
    actual = solvers.cg(
        lambda value: diagonal * value,
        tg.tensor([4.0, 6.0], dtype=tg.float32),
        iterations=2,
    )
    assert actual.tolist() == pytest.approx([2.0, 2.0], abs=2.0e-6)

    with pytest.raises(ValueError, match="square"):
        solvers.cg(
            lambda value: tg.tensor([1.0], dtype=tg.float32),
            tg.tensor([1.0, 1.0], dtype=tg.float32),
            iterations=1,
        )


def test_dot_and_norm_are_compiler_visible_tensor_algebra():
    left = tg.tensor([1.0, 2.0, 2.0], dtype=tg.float32)
    right = tg.tensor([3.0, 4.0, 5.0], dtype=tg.float32)
    product = solvers.dot(left, right)
    length = solvers.vector_norm(left)

    assert product.tolist() == pytest.approx(21.0)
    assert length.tolist() == pytest.approx(3.0)
    assert "gf_tensor.reduce_sum" in product.mlir()
    assert "gf_tensor.sqrt" in length.mlir()


def test_richardson_is_one_bounded_control_region_and_differentiable(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    diagonal = tg.tensor([2.0, 4.0], dtype=tg.float32)
    rhs = tg.tensor(
        [2.0, 8.0], dtype=tg.float32, requires_grad=True)
    solution = solvers.richardson(
        lambda value: diagonal * value,
        rhs,
        iterations=3,
        relaxation=0.25,
    )

    assert solution.tolist() == pytest.approx([0.875, 2.0], abs=2.0e-6)
    assert solution.mlir().count("gf_control.repeat") == 1
    gradient = tg.autograd.grad(solution.sum(), rhs)
    assert gradient.tolist() == pytest.approx([0.4375, 0.25], abs=2.0e-6)


def test_multi_state_repeat_lowers_to_typed_cpu_double_buffers(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    vector = tg.tensor([1.0, 2.0], dtype=tg.float32)
    scalar = tg.tensor(0.0, dtype=tg.float32)
    result = tg.repeat(
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
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    vector = tg.tensor([1.0, 2.0], dtype=tg.float32)
    counter = tg.tensor(0.0, dtype=tg.float32)

    shifted, final_counter = tg.while_loop(
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
    assert "tiga.cpu.max_iterations = 10" in lowered

    limited = tg.while_loop(
        counter,
        lambda step: step < 10.0,
        lambda step: step + 1.0,
        max_iterations=2,
    )
    assert limited.tolist() == pytest.approx(2.0)


def test_class_based_repeat_and_while_lower_like_the_functional_forms(
    monkeypatch,
):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")

    class Integrate(tg.control.Repeat):
        def __init__(self, rate):
            self.rate = rate

        def body(self, value, total):
            return value * (1.0 + self.rate), total + value.sum()

    vector = tg.tensor([1.0, 2.0], dtype=tg.float32)
    total = tg.tensor(0.0, dtype=tg.float32)
    value_out, total_out = Integrate(0.5)((vector, total), iterations=3)
    assert value_out.tolist() == pytest.approx([3.375, 6.75])
    assert total_out.tolist() == pytest.approx(14.25)
    ir = value_out.mlir(verify=True)
    assert ir.count("gf_control.repeat") == 1
    assert "num_carried = 2" in ir

    class Countdown(tg.control.While):
        def __init__(self, threshold):
            self.threshold = threshold

        def condition(self, value):
            return value > self.threshold

        def body(self, value):
            return value - 1.0

    counter = tg.tensor(5.0, dtype=tg.float32)
    final = Countdown(2.5)(counter, max_iterations=10)
    assert final.tolist() == pytest.approx(2.0)
    semantic = final.mlir(verify=True)
    assert semantic.count("gf_control.while") == 1
    lowered = (final.execution or {})["artifacts"]["cpu_loop"]
    assert "scf.while" in lowered
    assert "tiga.cpu.max_iterations = 10" in lowered

    with pytest.raises(NotImplementedError):
        tg.control.Repeat()(counter, iterations=1)
    with pytest.raises(NotImplementedError):
        tg.control.While()(counter, max_iterations=1)


def test_fixed_cg_is_one_multi_state_region_and_automatically_differentiable(
    monkeypatch,
):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    diagonal = tg.tensor([2.0, 4.0], dtype=tg.float32)
    rhs = tg.tensor(
        [2.0, 8.0], dtype=tg.float32, requires_grad=True)

    solution = solvers.cg(
        lambda value: diagonal * value, rhs, iterations=2)

    assert solution.tolist() == pytest.approx([1.0, 2.0], abs=2.0e-6)
    lowered = (solution.execution or {})["artifacts"]["cpu_loop"]
    assert "tiga.cpu.loop_scalar_temporaries = 4" in lowered
    assert "tiga.cpu.loop_tensor_temporaries = 2" in lowered
    # One outer repeat, the initial residual product, and exactly two body
    # reductions. A saved scalar/vector SSA must be loaded from scratch rather
    # than recursively expanding a reduction inside an elementwise loop.
    assert lowered.count("iter_args(") == 4
    assert solution.mlir().count("gf_control.repeat") == 1
    assert "num_carried = 4" in solution.mlir()
    gradient = tg.autograd.grad(solution.sum(), rhs)
    assert gradient.tolist() == pytest.approx([0.5, 0.25], abs=2.0e-5)

    inverse_diagonal = tg.tensor([0.5, 0.25], dtype=tg.float32)
    preconditioned = solvers.cg(
        lambda value: diagonal * value,
        rhs,
        iterations=1,
        preconditioner=lambda residual: inverse_diagonal * residual,
    )
    assert preconditioned.tolist() == pytest.approx([1.0, 2.0], abs=2.0e-6)


def test_tolerance_cg_lowers_to_bounded_while_without_host_polling(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    diagonal = tg.tensor([2.0, 4.0], dtype=tg.float32)
    rhs = tg.tensor(
        [2.0, 8.0], dtype=tg.float32, requires_grad=True)

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
        tg.autograd.grad(solution.sum(), rhs)


def test_fem_poisson_example_is_matrix_free_and_accurate(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    module = _load_module("graphforge_fem_poisson", EXAMPLES / "fem_poisson.py")

    solution, _exact, error = module.solve(interior_nodes=6, iterations=3)
    assert error < 2.0e-5
    ir = solution.mlir()
    assert ir.count("gf_control.repeat") == 1
    assert "num_carried = 4" in ir
    assert "gf_tensor.csr_segment_sum" in ir
    assert module.load_gradient().tolist() == pytest.approx(
        [0.4, 0.6, 0.6, 0.4], abs=2.0e-5)
    converged, _exact, converged_error = module.solve(
        interior_nodes=6, tolerance=1.0e-6, max_iterations=20)
    assert converged_error < 2.0e-5
    assert converged.mlir().count("gf_control.while") == 1


def test_dynamic_radius_linear_solve_keeps_operator_inside_bounded_while(
    monkeypatch,
):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
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


class _LowerBidiagonal(tg.MessagePassing):
    """A[i,i] = 2, A[i,i-1] = -1 — a nonsymmetric matrix-free operator."""

    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.w * src.u


def _lower_bidiagonal_operator(nodes: int = 8):
    row_ptr, col_idx, weights = [0], [], []
    for dst in range(nodes):
        if dst > 0:
            col_idx.append(dst - 1)
            weights.append(-1.0)
        col_idx.append(dst)
        weights.append(2.0)
        row_ptr.append(len(col_idx))
    graph = tg.Graph.from_csr(
        tg.tensor(row_ptr, dtype=tg.int64),
        tg.tensor(col_idx, dtype=tg.int64),
        num_src=nodes,
    )
    return _LowerBidiagonal(), graph, tg.tensor(weights, dtype=tg.float32)


def test_bicgstab_solves_a_nonsymmetric_operator(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    kernel, graph, weights = _lower_bidiagonal_operator(8)
    rhs = tg.tensor([1.0] * 8, dtype=tg.float32)
    # Exact solution by forward substitution: x₀ = 1/2, xᵢ = (1 + xᵢ₋₁)/2.
    exact = [0.5]
    for _ in range(7):
        exact.append((1.0 + exact[-1]) / 2.0)

    solution = solvers.bicgstab(
        kernel, rhs, graph=graph, field="u", edge={"w": weights},
        tolerance=1.0e-6, max_iterations=64)
    assert solution.tolist() == pytest.approx(exact, abs=1.0e-5)
    assert solution.mlir().count("gf_control.while") == 1

    fixed = solvers.bicgstab(
        kernel, rhs, graph=graph, field="u", edge={"w": weights},
        iterations=8)
    assert fixed.tolist() == pytest.approx(exact, abs=1.0e-5)
    assert fixed.mlir().count("gf_control.repeat") == 1


def test_linear_solve_dispatches_method_and_validates(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    diagonal = tg.tensor([2.0, 4.0], dtype=tg.float32)
    rhs = tg.tensor([2.0, 8.0], dtype=tg.float32)
    operator = lambda value: diagonal * value

    via_dispatch = solvers.linear_solve(operator, rhs, iterations=2)
    direct = solvers.cg(operator, rhs, iterations=2)
    assert via_dispatch.tolist() == pytest.approx(direct.tolist(), abs=1e-6)

    via_bicgstab = solvers.linear_solve(
        lambda value: tg.tensor([2.0, 3.0, 5.0, 7.0]) * value,
        tg.tensor([1.0, 1.0, 1.0, 1.0], dtype=tg.float32),
        method="bicgstab", tolerance=1.0e-6, max_iterations=16)
    expected = [0.5, 1.0 / 3.0, 0.2, 1.0 / 7.0]
    assert via_bicgstab.tolist() == pytest.approx(expected, abs=1.0e-5)

    via_richardson = solvers.linear_solve(
        operator, rhs, method="richardson", iterations=60, relaxation=0.2)
    assert via_richardson.tolist() == pytest.approx([1.0, 2.0], abs=1.0e-3)

    with pytest.raises(ValueError, match="unknown linear_solve method"):
        solvers.linear_solve(operator, rhs, method="gmres", iterations=1)
    with pytest.raises(ValueError, match="no preconditioner"):
        solvers.linear_solve(
            operator, rhs, method="bicgstab", iterations=1,
            preconditioner=lambda residual: residual)
    with pytest.raises(ValueError, match="fixed-count"):
        solvers.linear_solve(
            operator, rhs, method="richardson",
            tolerance=1.0e-6, max_iterations=10)
