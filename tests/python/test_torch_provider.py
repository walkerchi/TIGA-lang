from __future__ import annotations

import unittest

import torch
import graphforge as gf

from graphforge.interop.torch.provider import (
    reusable_dense_output,
    reusable_vector_output,
)


class _VectorOwner:
    def __init__(self) -> None:
        self.output = None

    def acquire(self, reference):
        self.output = reusable_vector_output(
            self.output, reference, rows=reference.shape[0])
        return self.output


class _DenseOwner:
    def __init__(self) -> None:
        self.storage = None

    def acquire(self, reference):
        self.storage, output = reusable_dense_output(
            self.storage, reference, lanes=2, rows=3, width=4)
        return output


class _Weighted(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.weight * src.x


class TorchProviderOutputTest(unittest.TestCase):
    def test_graph_degree_analysis_uses_strided_storage_provider_hook(self):
        storage = torch.tensor([0, -9, 2, -9, 5], dtype=torch.int64)
        row_ptr = gf.from_torch(storage[::2])
        col_idx = gf.from_torch(torch.tensor(
            [0, 1, 1, 2, 0], dtype=torch.int64))
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=3)

        self.assertEqual(graph.degree_bounds(), (2, 3))
        self.assertEqual(graph.degree_bounds(), (2, 3))
        self.assertAlmostEqual(
            graph.source_index_span_ratio(samples=5),
            (0 + 1 + 0 + 1 + 1) / 5 / 3,
        )

    def test_vector_reuses_unobserved_output(self):
        owner = _VectorOwner()
        reference = torch.empty(5)
        owner.acquire(reference)
        first_pointer = owner.output.data_ptr()

        # The previous return value is not observable, so a warm invocation
        # may safely reuse its storage.
        owner.acquire(reference)
        self.assertEqual(owner.output.data_ptr(), first_pointer)

    def test_vector_does_not_overwrite_held_output(self):
        owner = _VectorOwner()
        reference = torch.empty(5)
        held = owner.acquire(reference)
        replacement = owner.acquire(reference)

        self.assertNotEqual(replacement.data_ptr(), held.data_ptr())

    def test_dense_view_keeps_backing_storage_alive(self):
        owner = _DenseOwner()
        reference = torch.empty(3, 4)
        held = owner.acquire(reference)
        replacement = owner.acquire(reference)

        self.assertIsNot(replacement._base, held._base)

    def test_dynamic_torch_library_op_has_fake_autograd_and_opcheck(self):
        row_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        col_idx = torch.tensor([0, 1, 1, 2], dtype=torch.int64)
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=3)
        x = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
        weight = torch.tensor([2.0, 3.0, 5.0, 7.0], requires_grad=True)
        registered = gf.interop.torch.register_message_passing(
            _Weighted(), graph=graph, src={"x": x}, dst={},
            edge={"weight": weight})

        output = registered(src={"x": x}, dst={}, edge={"weight": weight})
        self.assertEqual(output.tolist(), [8.0, 38.0])
        dx, dweight = torch.autograd.grad(output.sum(), (x, weight))
        torch.testing.assert_close(dx, torch.tensor([2.0, 8.0, 7.0]))
        torch.testing.assert_close(dweight, torch.tensor([1.0, 2.0, 2.0, 4.0]))

        checked = registered.opcheck(x.detach(), weight.detach())
        self.assertTrue(all(result == "SUCCESS" for result in checked.values()))
        compiled = torch.compile(
            lambda a, b: registered(src={"x": a}, dst={}, edge={"weight": b}),
            fullgraph=True)
        compiled_x = x.detach().clone().requires_grad_(True)
        compiled_weight = weight.detach().clone().requires_grad_(True)
        compiled_output = compiled(compiled_x, compiled_weight)
        compiled_gradients = torch.autograd.grad(
            compiled_output.sum(), (compiled_x, compiled_weight))
        torch.testing.assert_close(compiled_output, output)
        torch.testing.assert_close(compiled_gradients[0], dx)
        torch.testing.assert_close(compiled_gradients[1], dweight)


if __name__ == "__main__":
    unittest.main()
