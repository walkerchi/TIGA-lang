from __future__ import annotations

import tiga as tg
import pytest


def test_for_loop_matches_functional_repeat(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")

    @tg.jit
    def integrate(value, rate):
        total = tg.zeros_like(value)
        for _ in range(3):
            total = total + value
            value = value * rate
        return value, total

    value = tg.tensor([1.0, 2.0], dtype=tg.float32)
    value_out, total_out = integrate(value, 2.0)
    assert value_out.tolist() == pytest.approx([8.0, 16.0])
    assert total_out.tolist() == pytest.approx([7.0, 14.0])
    ir = value_out.mlir(verify=True)
    assert ir.count("gf_control.repeat") == 1
    assert "num_carried = 2" in ir

    reference = tg.repeat(
        (value, tg.zeros_like(value)),
        lambda current, total: (current * 2.0, total + current),
        iterations=3,
    )
    assert total_out.tolist() == pytest.approx(reference[1].tolist())


def test_while_loop_matches_functional_form_and_zero_trip(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")

    @tg.jit(max_iterations=10)
    def countdown(value, threshold):
        while value > threshold:
            value = value - 1.0
        return value

    final = countdown(tg.tensor(5.0, dtype=tg.float32), 2.5)
    assert final.tolist() == pytest.approx(2.0)
    semantic = final.mlir(verify=True)
    assert semantic.count("gf_control.while") == 1
    lowered = (final.execution or {})["artifacts"]["cpu_loop"]
    assert "scf.while" in lowered
    assert "tiga.cpu.max_iterations = 10" in lowered

    already = countdown(tg.tensor(1.0, dtype=tg.float32), 2.5)
    assert already.tolist() == pytest.approx(1.0)


def test_for_with_leading_break_is_bounded_while(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")

    @tg.jit
    def fixed_point(value, tolerance):
        for _ in range(100):
            if value * value <= tolerance:
                break
            value = value * 0.5
        return value

    final = fixed_point(tg.tensor(8.0, dtype=tg.float32), 0.1)
    assert final.tolist() == pytest.approx(0.25)
    assert final.mlir().count("gf_control.while") == 1
    lowered = (final.execution or {})["artifacts"]["cpu_loop"]
    assert "tiga.cpu.max_iterations = 100" in lowered


def test_loop_variable_desugars_to_carried_index(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")

    @tg.jit
    def sweep(value):
        total = tg.zeros_like(value)
        for i in range(4):
            total = total + i * value
        return total

    total = sweep(tg.tensor([1, 2], dtype=tg.int64))
    # sum(i for i in range(4)) == 6
    assert total.tolist() == [6, 12]
    assert "num_carried = 2" in total.mlir()

    @tg.jit
    def mixed(value):
        for i in range(2):
            value = value + i
        return value

    # The carried index is rank-zero int64; the minimal Tensor frontend has
    # no cast op yet, so mixing it with float states fails closed.
    with pytest.raises(ValueError, match="identical device and dtype"):
        mixed(tg.tensor([1.0], dtype=tg.float32))


def test_body_locals_stay_out_of_carried_state(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")

    @tg.jit
    def step(value):
        for _ in range(3):
            temporary = value * 2.0
            value = temporary + 1.0
        return value

    result = step(tg.tensor([1.0], dtype=tg.float32))
    assert result.tolist() == pytest.approx([15.0])
    assert "num_carried = 1" in result.mlir()


class _ScaledLaplacian(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.value * src.u


def test_message_passing_inside_jit_loop(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    graph = tg.Graph.from_csr(
        tg.tensor([0, 2, 4], dtype=tg.int64),
        tg.tensor([0, 1, 0, 1], dtype=tg.int64),
        num_src=2,
    )
    weights = tg.tensor([2.0, -1.0, -1.0, 2.0], dtype=tg.float32)
    kernel = _ScaledLaplacian()

    @tg.jit
    def smooth(value):
        for _ in range(2):
            value = kernel(
                graph=graph, src={"u": value}, dst={"u": value},
                edge={"value": weights})
        return value

    result = smooth(tg.tensor([1.0, 0.0], dtype=tg.float32))
    assert result.tolist() == pytest.approx([5.0, -4.0])
    ir = result.mlir()
    assert ir.count("gf_control.repeat") == 1
    assert "gf_tensor.csr_segment_sum" in ir


def test_unsupported_constructs_fail_closed():
    with pytest.raises(SyntaxError, match="resource bound"):
        @tg.jit
        def unbounded_while(value):
            while value > 0.0:
                value = value - 1.0
            return value

    with pytest.raises(SyntaxError, match="while True"):
        @tg.jit(max_iterations=4)
        def while_true(value):
            while True:
                value = value - 1.0
            return value

    with pytest.raises(SyntaxError, match="range"):
        @tg.jit
        def non_range(value, items):
            for _ in items:
                value = value + 1.0
            return value

    with pytest.raises(SyntaxError, match="first loop statement"):
        @tg.jit
        def mid_break(value):
            for _ in range(4):
                value = value + 1.0
                if value > 2.0:
                    break
            return value

    with pytest.raises(SyntaxError, match="must not be"):
        @tg.jit
        def reassign_index(value):
            for i in range(4):
                i = i + 1
                value = value + i
            return value

    with pytest.raises(SyntaxError, match="initialized before the loop"):
        @tg.jit
        def late_binding(value):
            for _ in range(4):
                fresh = value * 2.0
            return fresh

    with pytest.raises(SyntaxError, match="no Tensor state"):
        @tg.jit
        def empty_loop(value):
            for _ in range(4):
                pass
            return value

    with pytest.raises(TypeError, match="single function definition"):
        tg.jit(lambda value: value)


def test_tensor_if_is_rejected_outside_loops(monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")

    @tg.jit
    def branchy(value):
        if value > 0.0:
            value = value + 1.0
        return value

    with pytest.raises(TypeError, match="Python boolean"):
        branchy(tg.tensor(1.0, dtype=tg.float32))
