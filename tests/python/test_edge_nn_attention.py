"""Fused edge-NN attention (phase 3): nn-produced scores + online softmax.

The row-centric tile kernel streams each CSR row in chunks, evaluating the
nn score chain in registers and carrying the online-softmax state in
``scf.for`` iter_args — no O(E) score or weight tensor exists.  Every check
compares against a manual scatter-softmax reference; grad-mode calls must
fall back to the eager oracle and stay correct.
"""

from __future__ import annotations

import unittest

import tiga as tg
import torch
from torch import nn


def _cuda_available() -> bool:
    return torch.cuda.is_available()


class GAT(tg.MessagePassing):
    reducer = tg.online_softmax()

    def __init__(self, attn, value="src", **trace_kwargs):
        super().__init__()
        self.attn = tg.nn.trace(attn, **trace_kwargs)
        self._value = value

    def edge(self, src, dst, edge):
        score = self.attn(edge.displacement, src.x, dst.x)
        if self._value == "dst":
            return self.reducer(score, dst.v)
        if self._value == "edge":
            return self.reducer(score, edge.feat)
        return self.reducer(score, src.x)


def _problem(nodes=192, dim=3, width=8, cutoff=0.30, seed=7, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    positions = torch.rand(nodes, dim, device=device, generator=generator)
    x = torch.randn(nodes, width, device=device, generator=generator)
    graph = tg.Graph.radius(positions, cutoff=cutoff)
    return graph, positions, x


def _reference(graph, positions, fields, attn, value_role, value_name,
               out_width):
    """Manual per-row softmax reference with ordinary torch ops."""
    row_ptr, col_idx = graph.resolve_csr()
    num_dst = row_ptr.numel() - 1
    rows = torch.repeat_interleave(
        torch.arange(num_dst, device=row_ptr.device),
        row_ptr[1:] - row_ptr[:-1])
    displacement = positions[col_idx] - positions[rows]
    x = fields["x"]
    score = attn(torch.cat([displacement, x[col_idx], x[rows]], dim=-1))
    score = score.squeeze(-1)
    if value_role == "dst":
        value = fields["v"][rows]
    elif value_role == "edge":
        value = fields["feat"]
    else:
        value = fields["x"][col_idx]
    maximum = torch.full(
        (num_dst,), float("-inf"), device=rows.device
    ).scatter_reduce_(0, rows, score, reduce="amax", include_self=True)
    weight = torch.exp(score - maximum[rows])
    denominator = torch.zeros(num_dst, device=rows.device)
    denominator.index_add_(0, rows, weight)
    numerator = torch.zeros(num_dst, out_width, device=rows.device)
    numerator.index_add_(0, rows, weight.unsqueeze(-1) * value)
    return torch.where(
        denominator.unsqueeze(-1) > 0,
        numerator / denominator.unsqueeze(-1).clamp(min=1e-30),
        torch.zeros_like(numerator))


def _reference_trainable(graph, positions, x, attn, value_role="src",
                         extra=None):
    """Differentiable per-row softmax reference (max detached, as in flash
    attention — same function value and the same Jacobian)."""
    row_ptr, col_idx = graph.resolve_csr()
    num_dst = row_ptr.numel() - 1
    rows = torch.repeat_interleave(
        torch.arange(num_dst, device=row_ptr.device),
        row_ptr[1:] - row_ptr[:-1])
    displacement = positions[col_idx] - positions[rows]
    score = attn(torch.cat([displacement, x[col_idx], x[rows]], dim=-1))
    score = score.squeeze(-1)
    if value_role == "dst":
        value = extra[rows]
    elif value_role == "edge":
        value = extra
    else:
        value = x[col_idx]
    with torch.no_grad():
        maximum = torch.full(
            (num_dst,), float("-inf"), device=rows.device
        ).scatter_reduce_(0, rows, score.detach(), reduce="amax",
                          include_self=True)
    weight = torch.exp(score - maximum[rows])
    denominator = torch.zeros(num_dst, device=rows.device)
    denominator = denominator.index_add(0, rows, weight)
    numerator = torch.zeros(num_dst, value.shape[-1], device=rows.device)
    numerator = numerator.index_add(
        0, rows, weight.unsqueeze(-1) * value)
    return torch.where(
        denominator.unsqueeze(-1) > 0,
        numerator / denominator.unsqueeze(-1).clamp(min=1e-30),
        torch.zeros_like(numerator))


@unittest.skipUnless(_cuda_available(), "edge nn attention requires CUDA")
class EdgeNNAttentionTest(unittest.TestCase):
    def _run(self, attn, value_role="src", value_name="x", out_width=8,
             nodes=192, cutoff=0.30, extra_fields=None, seed=7,
             **trace_kwargs):
        graph, positions, x = _problem(
            nodes=nodes, cutoff=cutoff, seed=seed)
        fields = {"x": x}
        if extra_fields:
            fields.update(extra_fields)
        src = {"x": x}
        dst = {"x": x}
        edge = {}
        if value_role == "dst":
            dst["v"] = fields["v"]
        if value_role == "edge":
            edge["feat"] = fields["feat"]
        program = GAT(attn, value_role, **trace_kwargs)
        with torch.no_grad():
            out = program(
                graph=graph, src=src, dst=dst, edge=edge)
        self.assertIn(
            "gf-python-emit-edge-nn-attention-tile", program.explain(),
            "fused lowering did not engage; numeric check would be vacuous")
        reference = _reference(
            graph, positions, fields, attn, value_role, value_name,
            out_width)
        self.assertTrue(
            torch.allclose(out, reference, rtol=1e-4, atol=1e-5),
            f"attention mismatch: max abs "
            f"{(out - reference).abs().max().item():.3e}")
        return program

    def test_gat_style_radius(self):
        attn = nn.Sequential(
            nn.Linear(19, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1)).cuda()
        self._run(attn)

    def test_multi_chunk_rows(self):
        # block_n=32 with mean degree ~50 exercises the scf.for rescale.
        attn = nn.Sequential(
            nn.Linear(19, 16), nn.Tanh(), nn.Linear(16, 1)).cuda()
        self._run(attn, nodes=256, cutoff=0.35, block_e=32, num_warps=2)

    def test_layernorm_in_score_net(self):
        attn = nn.Sequential(
            nn.Linear(19, 16), nn.LayerNorm(16), nn.GELU(),
            nn.Linear(16, 1)).cuda()
        self._run(attn)

    def test_scalar_temperature_chain(self):
        class Scaled(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(19, 1)

            def forward(self, h):
                return self.fc(torch.sigmoid(h) * 2.0) / 4.0

        self._run(Scaled().cuda())

    def test_dst_value_field(self):
        attn = nn.Sequential(nn.Linear(19, 1)).cuda()
        generator = torch.Generator(device="cuda").manual_seed(21)
        v = torch.randn(192, 4, device="cuda", generator=generator)
        self._run(attn, value_role="dst", value_name="v", out_width=4,
                  extra_fields={"v": v})

    def test_edge_value_field(self):
        attn = nn.Sequential(nn.Linear(19, 1)).cuda()
        graph, _positions, _x = _problem()
        num_edges = graph.resolve_csr()[1].numel()
        generator = torch.Generator(device="cuda").manual_seed(33)
        feat = torch.randn(num_edges, 4, device="cuda", generator=generator)
        self._run(attn, value_role="edge", value_name="feat", out_width=4,
                  extra_fields={"feat": feat})

    def test_empty_rows(self):
        # An isolated node (degree 0) must produce a zero row.
        generator = torch.Generator(device="cuda").manual_seed(5)
        positions = torch.rand(64, 3, device="cuda", generator=generator)
        positions[0] += 10.0  # far away: no neighbors
        x = torch.randn(64, 8, device="cuda", generator=generator)
        graph = tg.Graph.radius(positions, cutoff=0.2)
        attn = nn.Sequential(nn.Linear(19, 1)).cuda()
        program = GAT(attn)
        with torch.no_grad():
            out = program(graph=graph, src={"x": x}, dst={"x": x})
        self.assertIn("gf-python-emit-edge-nn-attention-tile",
                      program.explain())
        self.assertTrue(torch.all(out[0] == 0))
        self.assertTrue(torch.isfinite(out).all())

    def test_fused_training_gradients(self):
        attn = nn.Sequential(
            nn.Linear(19, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1)).cuda()
        graph, positions, x = _problem()
        positions = positions.requires_grad_(True)
        x = x.requires_grad_(True)
        program = GAT(attn)
        out = program(graph=graph, src={"x": x}, dst={"x": x})
        self.assertIn(
            "gf-python-emit-edge-nn-attention-vjp", program.explain(),
            "fused training path did not engage")
        coeff = torch.randn(
            out.shape, device=out.device,
            generator=torch.Generator(device=out.device).manual_seed(11))
        (out * coeff).sum().backward()
        fused = [x.grad.clone(), positions.grad.clone()]
        fused += [p.grad.clone() for p in attn.parameters()]

        for tensor in (x, positions, *attn.parameters()):
            tensor.grad = None
        reference = _reference_trainable(graph, positions, x, attn)
        self.assertTrue(
            torch.allclose(out.detach(), reference.detach(),
                           rtol=1e-4, atol=1e-5),
            f"forward mismatch: max abs "
            f"{(out.detach() - reference.detach()).abs().max().item():.3e}")
        (reference * coeff).sum().backward()
        eager = [x.grad, positions.grad]
        eager += [p.grad for p in attn.parameters()]
        for name, got, want in zip(
                ["x", "positions"]
                + [n for n, _ in attn.named_parameters()], fused, eager):
            # Cross-program atomic accumulation over E edges sets the noise
            # floor; real bugs are orders of magnitude larger.
            self.assertTrue(
                torch.allclose(got, want, rtol=2e-3, atol=2e-4),
                f"{name} gradient mismatch: "
                f"max abs {(got - want).abs().max().item():.3e}")

    def test_fused_training_dst_value(self):
        # value = dst.v: the value grad is a row-local direct accumulation,
        # and the score chain sees src/dst/positions as before.
        attn = nn.Sequential(nn.Linear(19, 1)).cuda()
        graph, positions, x = _problem()
        generator = torch.Generator(device="cuda").manual_seed(21)
        v = torch.randn(192, 4, device="cuda", generator=generator,
                        requires_grad=True)
        x = x.requires_grad_(True)
        program = GAT(attn, "dst")
        out = program(graph=graph, src={"x": x}, dst={"x": x, "v": v})
        self.assertIn("gf-python-emit-edge-nn-attention-vjp",
                      program.explain())
        coeff = torch.randn(
            out.shape, device=out.device,
            generator=torch.Generator(device=out.device).manual_seed(11))
        (out * coeff).sum().backward()
        self.assertIsNotNone(v.grad)
        reference = _reference_trainable(graph, positions, x, attn,
                                         value_role="dst", extra=v)
        v_grad_fused = v.grad.clone()
        for tensor in (x, v, *attn.parameters()):
            tensor.grad = None
        (reference * coeff).sum().backward()
        self.assertTrue(
            torch.allclose(v_grad_fused, v.grad, rtol=2e-3, atol=2e-4),
            f"value gradient mismatch: max abs "
            f"{(v_grad_fused - v.grad).abs().max().item():.3e}")

    def test_out_of_envelope_falls_back_to_oracle(self):
        # float64 fields are outside the fused envelope (f32 only): the call
        # falls back to the eager oracle and stays correct.
        attn = nn.Sequential(
            nn.Linear(19, 16), nn.ReLU(), nn.Linear(16, 1)).cuda().double()
        graph, positions, x = _problem()
        positions = positions.double()
        x = x.double()
        graph = tg.Graph.radius(positions, cutoff=0.30)
        program = GAT(attn)
        out = program(graph=graph, src={"x": x}, dst={"x": x})
        self.assertNotIn("gf-python-emit-edge-nn-attention",
                         program.explain())
        row_ptr, col_idx = graph.resolve_csr()
        rows = torch.repeat_interleave(
            torch.arange(192, device="cuda"), row_ptr[1:] - row_ptr[:-1])
        disp = positions[col_idx] - positions[rows]
        s = attn(torch.cat([disp, x[col_idx], x[rows]], dim=-1)).squeeze(-1)
        m = torch.full((192,), float("-inf"), device="cuda",
                       dtype=torch.float64)
        m.scatter_reduce_(0, rows, s, reduce="amax", include_self=True)
        w = torch.exp(s - m[rows])
        den = torch.zeros(192, device="cuda", dtype=torch.float64)
        den.index_add_(0, rows, w)
        num = torch.zeros(192, 8, device="cuda", dtype=torch.float64)
        num.index_add_(0, rows, w.unsqueeze(-1) * x[col_idx])
        self.assertTrue(
            torch.allclose(out, num / den.unsqueeze(-1),
                           rtol=1e-4, atol=1e-6))

    def test_plain_score_does_not_take_nn_path(self):
        # A non-nn score (bare field expression) stays on the oracle path
        # and remains correct.
        class PlainAttention(tg.MessagePassing):
            reducer = tg.online_softmax()

            def edge(self, src, dst, edge):
                return self.reducer(edge.sim, src.x)

        graph, positions, x = _problem()
        row_ptr, col_idx = graph.resolve_csr()
        rows = torch.repeat_interleave(
            torch.arange(192, device="cuda"), row_ptr[1:] - row_ptr[:-1])
        disp = positions[col_idx] - positions[rows]
        sim = -disp.norm(dim=-1)
        program = PlainAttention()
        out = program(graph=graph, src={"x": x}, dst={}, edge={"sim": sim})
        self.assertNotIn("edge-nn-attention", program.explain())
        m = torch.full((192,), float("-inf"), device="cuda")
        m.scatter_reduce_(0, rows, sim, reduce="amax", include_self=True)
        w = torch.exp(sim - m[rows])
        den = torch.zeros(192, device="cuda").index_add_(0, rows, w)
        num = torch.zeros(192, 8, device="cuda").index_add_(
            0, rows, w.unsqueeze(-1) * x[col_idx])
        ref = num / den.unsqueeze(-1)
        self.assertTrue(
            torch.allclose(out, ref, rtol=1e-4, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
