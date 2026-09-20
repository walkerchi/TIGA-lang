from __future__ import annotations

import unittest

import tiga as tg


class SymbolicShapeTest(unittest.TestCase):
    def test_invalid_symbolic_bounds_fail_before_compilation(self):
        for invalid in (True, 1.5, "2", float("nan")):
            for name in ("minimum", "maximum", "multiple_of"):
                with self.subTest(name=name, value=invalid):
                    with self.assertRaisesRegex(ValueError, "integers"):
                        tg.compiler.Dim("N", **{name: invalid})
        with self.assertRaisesRegex(ValueError, "identifier"):
            tg.compiler.Dim(7)
        with self.assertRaisesRegex(ValueError, "shape"):
            tg.compiler.TensorSpec((True,))

    def test_shared_dimension_guards_and_concrete_specialization_cache(self):
        rows = tg.compiler.Dim("rows", minimum=2, maximum=32, multiple_of=2)
        features = tg.compiler.Dim("features", multiple_of=4)
        specs = (
            tg.compiler.TensorSpec((rows, features)),
            tg.compiler.TensorSpec((rows, 1)),
        )
        compiled_bindings = []

        def compile_for(binding):
            compiled_bindings.append(binding.specialization_key())
            return lambda x, bias: x + bias

        function = tg.compiler.ShapeSpecializer(specs, compile_for)
        first = function(tg.empty((4, 8)), tg.empty((4, 1)))
        second = function(tg.empty((4, 8)), tg.empty((4, 1)))
        third = function(tg.empty((6, 12)), tg.empty((6, 1)))
        self.assertEqual(first.shape, (4, 8))
        self.assertEqual(second.shape, (4, 8))
        self.assertEqual(third.shape, (6, 12))
        self.assertEqual(function.specialization_count, 2)
        self.assertEqual((function.hits, function.misses), (1, 2))
        self.assertEqual(len(compiled_bindings), 2)

    def test_symbolic_guards_reject_conflicts_and_bounds(self):
        rows = tg.compiler.Dim("rows", minimum=2, multiple_of=2)
        specs = (
            tg.compiler.TensorSpec((rows, 4)),
            tg.compiler.TensorSpec((rows, 4)),
        )
        with self.assertRaisesRegex(ValueError, "bound to both"):
            tg.compiler.bind_shapes(
                specs, (tg.empty((4, 4)), tg.empty((6, 4)))
            )
        with self.assertRaisesRegex(ValueError, "violates"):
            tg.compiler.bind_shapes(
                specs, (tg.empty((3, 4)), tg.empty((3, 4)))
            )


if __name__ == "__main__":
    unittest.main()
