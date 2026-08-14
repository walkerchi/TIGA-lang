from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib missing")
class PlotBenchmarksTest(unittest.TestCase):
    def test_plot_files_are_created(self):
        from benchmarks.common.plotting import (
            plot_latency,
            plot_radius_build,
            plot_radius_pipeline,
            plot_roofline,
        )

        payload = {
            "roof": {
                "dram_bandwidth_gbs": 800.0,
                "l2_bandwidth_gbs": 1600.0,
                "fp32_gflops": 30000.0,
            },
            "results": [
                {
                    "provider": "baseline", "cache": "hot", "features": 16,
                    "milliseconds": 0.1, "achieved_gflops": 100.0,
                    "memory_roof": "L2",
                    "optimistic_roof_gflops": 1000.0,
                    "arithmetic_intensity_flop_per_byte": 0.625,
                },
                {
                    "provider": "candidate", "cache": "hot", "features": 16,
                    "milliseconds": 0.08, "achieved_gflops": 125.0,
                    "memory_roof": "L2",
                    "optimistic_roof_gflops": 1000.0,
                    "arithmetic_intensity_flop_per_byte": 0.625,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plot_roofline(payload, output)
            plot_latency(payload, output)
            plot_radius_build({
                "config": {
                    "particles": 1024,
                    "dimensions": 3,
                    "target_degree": 16,
                },
                "results": [{
                    "provider": "graphforge.cell_list",
                    "milliseconds": 0.5,
                    "candidate_pairs": 65536,
                    "accepted_edges": 16384,
                }],
            }, output)
            plot_radius_pipeline({
                "config": {"particles": 32, "dimensions": 3},
                "graph": {"accepted_edges": 96},
                "results": [
                    {"phase": "build-only", "provider": "graphforge.radius_build",
                     "milliseconds": 0.4},
                    {"phase": "consume-only", "provider": "graphforge.auto",
                     "milliseconds": 0.1},
                    {"phase": "consume-only", "provider": "torch.sparse.mm",
                     "milliseconds": 0.2},
                    {"phase": "build+consume", "provider": "graphforge.dynamic.auto",
                     "milliseconds": 0.5},
                    {"phase": "build+consume",
                     "provider": "graphforge.builder+torch.sparse.mm",
                     "milliseconds": 0.6},
                ],
                "sota_gates": [
                    {"cache": "consume-only", "speedup_vs_sota": 2.0,
                     "speedup_ci_low": 1.8, "speedup_ci_high": 2.2,
                     "passed": True},
                    {"cache": "build+consume", "speedup_vs_sota": 1.2,
                     "speedup_ci_low": 1.1, "speedup_ci_high": 1.3,
                     "passed": True},
                ],
            }, output)
            for name in ("roofline.png", "provider_latency.png",
                         "radius_build.png", "radius_pipeline.png"):
                path = output / name
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
