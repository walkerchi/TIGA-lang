from __future__ import annotations

import unittest

import torch

from benchmarks.sparse_compute.cases import degree_statistics, make_graph


class SparseBenchmarkCasesTest(unittest.TestCase):
    def test_continuous_degree_families_are_reproducible_and_heavy_tailed(self):
        summaries = {}
        for family in ("lognormal", "exponential"):
            first = make_graph(
                8192, 16, family, "random", torch.device("cpu"), torch.int32)
            second = make_graph(
                8192, 16, family, "random", torch.device("cpu"), torch.int32)
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertTrue(torch.equal(first[1], second[1]))
            self.assertEqual(first[1].numel(), int(first[0][-1]))
            summary = degree_statistics(first[0])
            summaries[family] = summary
            self.assertGreater(float(summary["mean"]), 15.0)
            self.assertLess(float(summary["mean"]), 16.5)
            self.assertGreater(float(summary["p99"]), 3 * 16)
            self.assertGreater(int(summary["maximum"]), int(summary["p99"]))
            self.assertGreater(float(summary["zero_fraction"]), 0.0)

        self.assertGreater(
            float(summaries["lognormal"]["coefficient_of_variation"]),
            float(summaries["exponential"]["coefficient_of_variation"]),
        )

    def test_empty_degree_statistics_are_well_defined(self):
        summary = degree_statistics(torch.tensor([0], dtype=torch.int64))
        self.assertEqual(summary["family"], "empty")
        self.assertEqual(summary["maximum"], 0)


if __name__ == "__main__":
    unittest.main()
