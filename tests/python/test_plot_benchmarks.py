from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib missing")
class PlotBenchmarksTest(unittest.TestCase):
    @staticmethod
    def _roof_payload(*, intensity=0.625, performance=100.0, nodes=1024):
        return {
            "operation": "test_operation",
            "workload": "same semantics",
            "roof": {
                "dram_bandwidth_gbs": 800.0,
                "l2_bandwidth_gbs": 1600.0,
                "fp32_gflops": 30000.0,
            },
            "results": [
                {
                    "provider": "graphforge.compiler_ttir",
                    "cache": "hot", "features": 16,
                    "milliseconds": 0.1, "achieved_gflops": performance,
                    "memory_roof": "L2", "optimistic_roof_gflops": 1000.0,
                    "arithmetic_intensity_flop_per_byte": intensity,
                    "topology": "regular", "locality": "local",
                    "index_dtype": "i64", "nodes": nodes,
                    "edges": nodes * 16,
                },
                {
                    "provider": "torch.sparse.mm",
                    "cache": "hot", "features": 16,
                    "milliseconds": 0.12, "achieved_gflops": performance * 0.8,
                    "memory_roof": "L2", "optimistic_roof_gflops": 1000.0,
                    "arithmetic_intensity_flop_per_byte": intensity,
                    "topology": "regular", "locality": "local",
                    "index_dtype": "i64", "nodes": nodes,
                    "edges": nodes * 16,
                },
            ],
        }

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

    def test_provider_colors_are_stable_and_collection_plot_is_created(self):
        from benchmarks.common.collection_plotting import (
            plot_compiler_report,
            plot_manifest_dashboard,
            plot_operation_summary,
        )
        from benchmarks.common.plotting import provider_color

        self.assertEqual(
            provider_color("graphforge.compiler_ttir"),
            provider_color("graphforge.compiler_ttir"),
        )
        self.assertNotEqual(
            provider_color("graphforge.compiler_ttir"),
            provider_color("torch.sparse.mm"),
        )
        first = self._roof_payload()
        second = self._roof_payload(
            intensity=1.25, performance=180.0, nodes=2048)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = plot_operation_summary([first, second], output)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
            self.assertTrue((output / "SUMMARY.md").exists())
            dashboard = plot_manifest_dashboard({
                "operations": {
                    "test_operation": {
                        "status": "performance-ready-test", "cases": ["a", "b"]
                    }
                }
            }, output)
            self.assertTrue(dashboard.exists())
            self.assertTrue((output / "README.md").exists())

            case = output / "test_operation" / "case"
            case.mkdir(parents=True)
            import json
            (case / "roofline.json").write_text(json.dumps(first))
            report = plot_compiler_report({
                "report_device": "test device",
                "report_panels": [{
                    "title": "Sparse test kernel",
                    "operation": "test_operation", "case": "case",
                    "filters": {"features": 16, "cache": "hot"},
                    "providers": [
                        "graphforge.compiler_ttir", "torch.sparse.mm"],
                    "baseline": "torch.sparse.mm",
                }],
            }, output, output / "compiler-report.png")
            self.assertTrue(report.exists())
            self.assertTrue(report.with_suffix(".svg").exists())

    def test_knn_roofline_uses_specialized_overlap_view(self):
        from benchmarks.common.plotting import plot_roofline

        payload = self._roof_payload()
        payload["operation"] = "knn_graph"
        payload["workload"] = "exact k-nearest-neighbor build"
        payload["config"] = {"nodes": 1024, "dimensions": 3, "k": 16}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plot_roofline(payload, output)
            path = output / "roofline.png"
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)

    def test_non_roofline_diagnostic_json_gets_a_human_plot(self):
        import json

        from benchmarks.common.diagnostic_plotting import plot_json

        payload = {
            "schema": "graphforge.memory-hierarchy.v1",
            "gate": "PASS", "h2d_ms": 1.0, "d2h_ms": 1.1,
            "nvme_spill_ms": 8.0, "nvme_restore_ms": 6.0,
            "h2d_GBps": 16.0, "d2h_GBps": 15.0,
            "nvme_spill_GBps": 2.0, "nvme_restore_GBps": 2.5,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps(payload))
            image = plot_json(path)
            self.assertEqual(image, path.with_suffix(".png"))
            self.assertTrue(image.exists())
            self.assertGreater(image.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
