from __future__ import annotations

import unittest

from benchmarks.common.perf_protocol import evaluate_sota_gates


def result(provider: str, milliseconds: float, features: int = 16):
    return {
        "topology": "regular",
        "locality": "local",
        "cache": "hot",
        "provider": provider,
        "nodes": 1024,
        "edges": 16384,
        "features": features,
        "index_dtype": "i64",
        "milliseconds": milliseconds,
        "samples_ms": [milliseconds] * 5,
    }


class PerfProtocolTest(unittest.TestCase):
    def test_fastest_peer_is_selected(self):
        gates = evaluate_sota_gates(
            [
                result("torch.sparse.mm", 1.0),
                result("pyg", 2.0),
                result("tiga.triton", 0.9),
                result("tiga.reference", 0.1),
            ],
            ["tiga.triton"],
        )
        self.assertEqual(gates[0].baseline, "torch.sparse.mm")
        self.assertTrue(gates[0].passed)
        self.assertAlmostEqual(gates[0].speedup_vs_sota, 1.0 / 0.9)
        self.assertGreater(gates[0].speedup_ci_low, 1.0)

    def test_every_bucket_is_gated_independently(self):
        results = [
            result("torch.sparse.mm", 1.0, features=16),
            result("tiga.triton", 0.9, features=16),
            result("torch.sparse.mm", 2.0, features=64),
            result("tiga.triton", 2.1, features=64),
        ]
        gates = evaluate_sota_gates(results, ["tiga.triton"])
        self.assertEqual([gate.passed for gate in gates], [True, False])

    def test_missing_candidate_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "was not measured"):
            evaluate_sota_gates(
                [result("torch.sparse.mm", 1.0)], ["tiga.triton"])

    def test_missing_raw_samples_fails_conservatively(self):
        candidate = result("tiga.triton", 0.9)
        del candidate["samples_ms"]
        gates = evaluate_sota_gates(
            [result("torch.sparse.mm", 1.0), candidate],
            ["tiga.triton"],
        )
        self.assertFalse(gates[0].passed)
        self.assertIn("raw samples", gates[0].reason)


if __name__ == "__main__":
    unittest.main()
