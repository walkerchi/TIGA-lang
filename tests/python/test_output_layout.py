from __future__ import annotations

from pathlib import Path
import unittest

from benchmarks.common.check_outputs import is_formal_operation, semantic_x_violations
from benchmarks.common.output_layout import OPERATIONS, artifact_path, operation_dir


class OutputLayoutTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
