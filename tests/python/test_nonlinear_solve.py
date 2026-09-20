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


@pytest.fixture()
def module(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return _load_module(
        "graphforge_example_nonlinear_solve", EXAMPLES / "nonlinear_solve.py")


def test_fixed_picard_converges_with_small_residual(module):
    solution, _residual, error = module.solve(interior_nodes=8, iterations=300)

    assert error < 1.0e-4
    ir = solution.mlir()
    assert ir.count("gf_control.repeat") == 1
    assert "gf_tensor.csr_segment_sum" in ir
    # k(u) = 1 + u² >= 1 stiffens the operator, so the nonlinear solution
    # sits just below the linear Poisson parabola u(x) = x(1-x)/2.
    spacing = 1.0 / 9
    parabola = [
        0.5 * (index + 1) * spacing * (1.0 - (index + 1) * spacing)
        for index in range(8)
    ]
    for actual, linear in zip(solution.tolist(), parabola, strict=True):
        assert actual == pytest.approx(linear, rel=0.02)


def test_load_gradient_matches_finite_difference(module):
    # The fixed-repeat VJP unrolls the loop, so the differentiated map keeps
    # the iteration count in the low single digits; the finite-difference
    # check differentiates the same three-step map.
    gradient = module.load_gradient(interior_nodes=4, iterations=3)

    kernel, graph, spacing, boundary = module.nonlinear_operator(4)

    def apply(state):
        return kernel(
            graph=graph,
            src={"u": state},
            dst={"u": state, "boundary": boundary},
            inverse_spacing=1.0 / spacing,
        )

    def solve_sum(load):
        solution = module.nonlinear_solve(
            apply, load, omega=0.05, iterations=3, tolerance=None)
        return solution.sum().tolist()

    base = [spacing] * 4
    epsilon = 1.0e-3
    plus, minus = base[:], base[:]
    plus[0] += epsilon
    minus[0] -= epsilon
    finite_difference = (
        solve_sum(tg.tensor(plus, dtype=tg.float32))
        - solve_sum(tg.tensor(minus, dtype=tg.float32))
    ) / (2.0 * epsilon)
    assert gradient.tolist()[0] == pytest.approx(finite_difference, abs=1.0e-3)


def test_tolerance_driver_converges_within_max_iterations(module):
    solution, _residual, error = module.solve(
        interior_nodes=8, tolerance=1.0e-6, max_iterations=2_000)

    assert error < 1.0e-4
    assert solution.mlir().count("gf_control.while") == 1
    assert "gf_control.repeat" not in solution.mlir()


def test_stopping_contracts_are_exclusive(module):
    rhs = tg.tensor([1.0, 1.0], dtype=tg.float32)
    operator = lambda value: 2.0 * value

    with pytest.raises(ValueError, match="exactly one stopping contract"):
        module.nonlinear_solve(operator, rhs, omega=0.1, iterations=2)
    with pytest.raises(ValueError, match="exactly one stopping contract"):
        module.nonlinear_solve(
            operator, rhs, omega=0.1, tolerance=None, max_iterations=2)
