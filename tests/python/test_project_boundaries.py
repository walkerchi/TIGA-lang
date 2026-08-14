from __future__ import annotations

from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "python" / "graphforge"


class ProjectBoundaryTest(unittest.TestCase):
    def test_core_contains_no_handwritten_triton_kernel(self):
        violations = []
        decorator = re.compile(r"^\s*@triton\.jit\b", re.MULTILINE)
        for path in CORE.rglob("*.py"):
            if decorator.search(path.read_text()):
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])

    def test_core_does_not_import_benchmark_oracles(self):
        violations = []
        for path in CORE.rglob("*.py"):
            text = path.read_text()
            if "benchmarks.kernels" in text or "codegen.triton" in text:
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])

    def test_torch_bridge_does_not_assemble_domain_mlir_source(self):
        bridge = (
            CORE / "interop" / "torch" / "compiler_bridge.py"
        ).read_text()
        self.assertNotIn("module {", bridge)
        self.assertNotIn('"gf.generated_radius"', bridge)
        self.assertNotIn("_streaming_reducer_mlir", bridge)

    def test_handwritten_oracles_are_isolated(self):
        oracle_dir = ROOT / "benchmarks" / "kernels"
        self.assertTrue((oracle_dir / "dense_streaming_reducer.py").is_file())
        self.assertTrue((oracle_dir / "sparse_triton_oracles.py").is_file())

    def test_high_level_framework_packages_are_out_of_scope(self):
        forbidden = ("optim", "optimizer", "nn", "dataset", "datasets")
        violations = [
            name for name in forbidden
            if (CORE / name).exists() or (CORE / f"{name}.py").exists()
        ]
        self.assertEqual(violations, [])

    def test_torch_is_an_optional_adapter(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertNotIn("torch>=2.1", project["project"]["dependencies"])
        self.assertIn("torch>=2.1", project["project"]["optional-dependencies"]["torch"])

    def test_reducer_semantics_do_not_embed_torch_execution(self):
        core = (CORE / "reducer" / "core.py").read_text()
        self.assertNotRegex(core, r"(^|\n)import torch\b")
        self.assertNotIn("index_add_", core)
        self.assertNotIn("scatter_reduce_", core)
        self.assertTrue(
            (CORE / "interop" / "torch" / "reducer_oracle.py").is_file()
        )

    def test_native_runtime_uses_wheel_install_component(self):
        cmake = (ROOT / "lib" / "Runtime" / "CMakeLists.txt").read_text()
        self.assertRegex(
            cmake,
            r"LIBRARY DESTINATION graphforge/lib\s+COMPONENT GraphForgeWheel",
        )

    def test_major_subsystems_are_packages_not_flat_modules(self):
        subsystems = (
            "autograd",
            "compiler",
            "distributed",
            "graph",
            "kernel",
            "message_passing",
            "reducer",
            "runtime",
            "tensor",
        )
        for name in subsystems:
            self.assertTrue((CORE / name / "__init__.py").is_file(), name)
            self.assertFalse((CORE / f"{name}.py").exists(), name)


if __name__ == "__main__":
    unittest.main()
