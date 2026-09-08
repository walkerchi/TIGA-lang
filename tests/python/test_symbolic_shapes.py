from __future__ import annotations

import unittest

import tiga as gf


class SymbolicShapeTest(unittest.TestCase):
    def test_shared_dimension_guards_and_concrete_specialization_cache(self):
        rows = gf.compiler.Dim("rows", minimum=2, maximum=32, multiple_of=2)
        features = gf.compiler.Dim("features", multiple_of=4)
        specs = (
            gf.compiler.TensorSpec((rows, features)),
            gf.compiler.TensorSpec((rows, 1)),
        )
        compiled_bindings = []

        def compile_for(binding):
            compiled_bindings.append(binding.specialization_key())
            return lambda x, bias: x + bias

        function = gf.compiler.ShapeSpecializer(specs, compile_for)
        first = function(gf.empty((4, 8)), gf.empty((4, 1)))
        second = function(gf.empty((4, 8)), gf.empty((4, 1)))
        third = function(gf.empty((6, 12)), gf.empty((6, 1)))
        self.assertEqual(first.shape, (4, 8))
        self.assertEqual(second.shape, (4, 8))
        self.assertEqual(third.shape, (6, 12))
        self.assertEqual(function.specialization_count, 2)
        self.assertEqual((function.hits, function.misses), (1, 2))
        self.assertEqual(len(compiled_bindings), 2)

    def test_symbolic_guards_reject_conflicts_and_bounds(self):
        rows = gf.compiler.Dim("rows", minimum=2, multiple_of=2)
        specs = (
            gf.compiler.TensorSpec((rows, 4)),
            gf.compiler.TensorSpec((rows, 4)),
        )
        with self.assertRaisesRegex(ValueError, "bound to both"):
            gf.compiler.bind_shapes(
                specs, (gf.empty((4, 4)), gf.empty((6, 4)))
            )
        with self.assertRaisesRegex(ValueError, "violates"):
            gf.compiler.bind_shapes(
                specs, (gf.empty((3, 4)), gf.empty((3, 4)))
            )


if __name__ == "__main__":
    unittest.main()
