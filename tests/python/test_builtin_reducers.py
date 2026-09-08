from __future__ import annotations

import unittest

import tiga as gf


class MeanAggregation(gf.MessagePassing):
    reducer = gf.mean()

    def edge(self, src, dst, edge):
        del dst, edge
        return self.reducer(src.x)


class ProductAggregation(gf.MessagePassing):
    reducer = gf.prod()

    def edge(self, src, dst, edge):
        del dst, edge
        return self.reducer(src.x)


class BuiltinReducerNativeTest(unittest.TestCase):
    def setUp(self):
        self.graph = gf.Graph.from_csr(
            gf.tensor([0, 2, 3, 5], dtype=gf.int64),
            gf.tensor([0, 2, 1, 0, 1], dtype=gf.int64),
            num_src=3,
            validate="full",
        )

    def test_builtin_mean_forward_and_vjp(self):
        x = gf.tensor([1.0, 2.0, 4.0], requires_grad=True)
        kernel = MeanAggregation()

        output = kernel(graph=self.graph, src={"x": x}, dst={})
        dx = gf.autograd.grad(output.sum(), x)

        self.assertEqual(output.tolist(), [2.5, 2.0, 1.5])
        self.assertEqual(dx.tolist(), [1.0, 1.5, 0.5])
        self.assertIn("proved-componentwise-additive-udf", kernel.explain())

    def test_builtin_prod_forward_and_vjp(self):
        graph = gf.Graph.from_csr(
            gf.tensor([0, 2, 2, 5], dtype=gf.int64),
            gf.tensor([0, 1, 0, 1, 2], dtype=gf.int64),
            num_src=3,
            validate="full",
        )
        x = gf.tensor([2.0, 3.0, 5.0], requires_grad=True)
        kernel = ProductAggregation()

        output = kernel(graph=graph, src={"x": x}, dst={})
        dx = gf.autograd.grad(output.sum(), x)

        self.assertEqual(output.tolist(), [6.0, 1.0, 30.0])
        self.assertEqual(dx.tolist(), [18.0, 12.0, 6.0])
        self.assertIn("proved-product-monoid", kernel.explain())

    def test_builtin_mean_degree_zero_row_is_nan(self):
        graph = gf.Graph.from_csr(
            gf.tensor([0, 2, 2, 4], dtype=gf.int64),
            gf.tensor([0, 2, 0, 1], dtype=gf.int64),
            num_src=3,
            validate="full",
        )
        output = MeanAggregation()(
            graph=graph, src={"x": gf.tensor([2.0, 4.0, 8.0])}, dst={})
        values = output.tolist()
        self.assertEqual(values[0], 5.0)
        self.assertTrue(values[1] != values[1])
        self.assertEqual(values[2], 3.0)


class BuiltinReducerTorchTest(unittest.TestCase):
    """The optional torch oracle evaluates the same built-in algebras."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch interop is optional")

    def setUp(self):
        import torch

        self.torch = torch
        self.graph = gf.Graph.from_csr(
            torch.tensor([0, 2, 3, 5]),
            torch.tensor([0, 2, 1, 0, 1]),
            num_src=3,
            validate="full",
        )

    def test_torch_mean_matches_native(self):
        torch = self.torch
        x = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)

        output = MeanAggregation()(graph=self.graph, src={"x": x}, dst={})
        output.sum().backward()

        self.assertEqual(output.detach().tolist(), [2.5, 2.0, 1.5])
        self.assertEqual(x.grad.tolist(), [1.0, 1.5, 0.5])

    def test_torch_prod_matches_native(self):
        torch = self.torch
        graph = gf.Graph.from_csr(
            torch.tensor([0, 2, 2, 5]),
            torch.tensor([0, 1, 0, 1, 2]),
            num_src=3,
            validate="full",
        )
        x = torch.tensor([2.0, 3.0, 5.0], requires_grad=True)

        output = ProductAggregation()(graph=graph, src={"x": x}, dst={})
        output.sum().backward()

        self.assertEqual(output.detach().tolist(), [6.0, 1.0, 30.0])
        self.assertEqual(x.grad.tolist(), [18.0, 12.0, 6.0])

    def test_torch_mean_feature_matrix(self):
        torch = self.torch
        x = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)

        output = MeanAggregation()(graph=self.graph, src={"x": x}, dst={})
        output.sum().backward()

        self.assertEqual(
            output.detach().tolist(), [[3.0, 4.0], [3.0, 4.0], [2.0, 3.0]])
        self.assertEqual(
            x.grad.tolist(), [[1.0, 1.0], [1.5, 1.5], [0.5, 0.5]])

    def test_torch_empty_row_semantics(self):
        torch = self.torch
        graph = gf.Graph.from_csr(
            torch.tensor([0, 2, 2, 4]),
            torch.tensor([0, 2, 0, 1]),
            num_src=3,
            validate="full",
        )
        x = torch.tensor([2.0, 4.0, 8.0])

        mean = MeanAggregation()(graph=graph, src={"x": x}, dst={}).tolist()
        prod = ProductAggregation()(graph=graph, src={"x": x}, dst={}).tolist()

        self.assertEqual(mean[0], 5.0)
        self.assertTrue(mean[1] != mean[1])
        self.assertEqual(mean[2], 3.0)
        self.assertEqual(prod, [16.0, 1.0, 8.0])


if __name__ == "__main__":
    unittest.main()
