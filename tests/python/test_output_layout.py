from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from benchmarks.common.check_outputs import (
    MANIFEST, human_visualization_violations, is_formal_operation,
    semantic_x_violations,
)
from benchmarks.common.output_layout import OPERATIONS, artifact_path, operation_dir


class OutputLayoutTest(unittest.TestCase):
    def test_canonical_evidence_manifest_is_tracked_outside_output(self):
        self.assertEqual(MANIFEST, Path("benchmarks/evidence_manifest.json"))
        self.assertTrue(MANIFEST.is_file())

    def test_every_operation_has_an_isolated_roofline_directory(self):
        directories = [
            operation_dir(operation, "regular-i64")
            for operation in OPERATIONS
        ]
        self.assertEqual(len(set(directories)), len(OPERATIONS))
        for directory in directories:
            self.assertEqual(directory.parts[:2], ("output", "roofline"))

    def test_artifact_path_is_case_local(self):
        path = artifact_path(
            "weighted_aggregation", "Regular / I64", "roofline.json")
        self.assertEqual(
            path,
            Path("output/roofline/weighted_aggregation/regular_i64/roofline.json"),
        )

    def test_same_operation_providers_must_share_semantic_x(self):
        common = {
            "topology": "radius", "nodes": 32, "edges": 64, "features": 1,
        }
        valid = {"results": [
            {**common, "provider": "a",
             "arithmetic_intensity_flop_per_byte": 2.0},
            {**common, "provider": "b",
             "arithmetic_intensity_flop_per_byte": 2.0},
        ]}
        invalid = {"results": [
            {**common, "provider": "a",
             "arithmetic_intensity_flop_per_byte": 2.0},
            {**common, "provider": "b",
             "arithmetic_intensity_flop_per_byte": 3.0},
        ]}
        self.assertEqual(semantic_x_violations(valid), [])
        self.assertEqual(len(semantic_x_violations(invalid)), 1)

    def test_formal_registration_is_not_tied_to_status_wording(self):
        self.assertTrue(is_formal_operation({
            "status": "measured-generated", "cases": ["case"],
        }))
        self.assertTrue(is_formal_operation({
            "status": "performance-ready-gate-passed", "cases": ["case"],
        }))
        self.assertFalse(is_formal_operation({
            "status": "pending-matched-semantics", "cases": [],
        }))

    def test_measured_json_requires_a_human_visualization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / "case"
            case.mkdir()
            (case / "results.json").write_text("{}")
            self.assertEqual(human_visualization_violations(root), [str(case)])
            (case / "results.png").write_bytes(b"image")
            self.assertEqual(human_visualization_violations(root), [])


if __name__ == "__main__":
    unittest.main()
