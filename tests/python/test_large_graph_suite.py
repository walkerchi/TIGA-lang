from __future__ import annotations

import unittest

from benchmarks.large_graphs.plan import choose_tier, load_suite, profile_bytes


class LargeGraphSuiteTest(unittest.TestCase):
    def test_manifest_distinguishes_source_edges_from_adjacency(self):
        suite = load_suite()
        friendster = next(
            case for case in suite["cases"] if case["id"] == "snap-friendster")
        self.assertEqual(
            friendster["adjacency_entries"], 2 * friendster["source_edges"])

    def test_working_set_is_component_sum(self):
        case = load_suite()["cases"][0]
        sizes = profile_bytes(case, case["profiles"][0])
        self.assertEqual(
            sizes["minimum_working_set"],
            sizes["topology"] + sizes["edge_values"] + sizes["node_state"],
        )

    def test_capacity_tier_respects_headroom(self):
        gib = 1 << 30
        self.assertEqual(
            choose_tier(7 * gib, hbm=10 * gib, ram=20 * gib,
                        ssd=30 * gib, headroom=0.8),
            "hbm-resident",
        )
        self.assertEqual(
            choose_tier(9 * gib, hbm=10 * gib, ram=20 * gib,
                        ssd=30 * gib, headroom=0.8),
            "ram-resident/hbm-streamed",
        )


if __name__ == "__main__":
    unittest.main()
