from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import graphforge as gf


def test_linear_operator_validates_matrix_free_application():
    diagonal = gf.tensor([2.0, 3.0], dtype=gf.float32)
    operator = gf.linalg.LinearOperator(
        (2, 2), matvec=lambda value: diagonal * value,
        parameters=(diagonal,), symmetric=True, name="diagonal",
    )
    actual = operator(gf.tensor([4.0, 5.0], dtype=gf.float32))
    assert actual.tolist() == pytest.approx([8.0, 15.0])

    with pytest.raises(ValueError, match="expected input shape"):
        operator(gf.tensor([1.0], dtype=gf.float32))


def test_dot_and_norm_are_compiler_visible_tensor_algebra():
    left = gf.tensor([1.0, 2.0, 2.0], dtype=gf.float32)
    right = gf.tensor([3.0, 4.0, 5.0], dtype=gf.float32)
    product = gf.linalg.dot(left, right)
    length = gf.linalg.vector_norm(left)

    assert product.tolist() == pytest.approx(21.0)
    assert length.tolist() == pytest.approx(3.0)
    assert "gf_tensor.reduce_sum" in product.mlir()
    assert "gf_tensor.sqrt" in length.mlir()


def test_richardson_is_one_bounded_control_region_and_differentiable(monkeypatch):
    monkeypatch.setenv("GRAPHFORGE_TENSOR_BACKEND", "native")
    diagonal = gf.tensor([2.0, 4.0], dtype=gf.float32)
    rhs = gf.tensor(
        [2.0, 8.0], dtype=gf.float32, requires_grad=True)
    operator = gf.linalg.LinearOperator(
        (2, 2), matvec=lambda value: diagonal * value,
        symmetric=True,
    )
    solution = gf.linalg.richardson(
        operator, rhs, iterations=3, relaxation=0.25)

    assert solution.tolist() == pytest.approx([0.875, 2.0], abs=2.0e-6)
    assert solution.mlir().count("gf_control.repeat") == 1
    gradient = gf.autograd.grad(solution.sum(), rhs)
    assert gradient.tolist() == pytest.approx([0.4375, 0.25], abs=2.0e-6)


def test_fem_poisson_example_is_matrix_free_and_accurate(monkeypatch):
    monkeypatch.setenv("GRAPHFORGE_TENSOR_BACKEND", "native")
    path = Path(__file__).parents[2] / "examples" / "fem_poisson.py"
    spec = importlib.util.spec_from_file_location("graphforge_fem_poisson", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    solution, _exact, error = module.run(interior_nodes=6, iterations=240)
    assert error < 2.0e-5
    ir = solution.mlir()
    assert ir.count("gf_control.repeat") == 1
    assert "gf_tensor.csr_segment_sum" in ir
