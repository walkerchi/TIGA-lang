from __future__ import annotations

import importlib.util
import os
import unittest
from unittest.mock import patch

import graphforge as gf


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class VisualizationTest(unittest.TestCase):
    def test_heatmap_is_ordinary_tensor_ir_and_exports_rgb(self):
        import numpy as np

        values = gf.tensor([[0.0, 0.5], [1.0, 0.25]], dtype=gf.float32)
        raster = gf.visualize.heatmap(
            values, low=(0.0, 0.0, 0.0), high=(1.0, 0.5, 0.25))
        actual = raster.to_numpy()
        expected = np.array([
            [[0.0, 0.0, 0.0], [0.5, 0.25, 0.125]],
            [[1.0, 0.5, 0.25], [0.25, 0.125, 0.0625]],
        ], dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        mlir = raster.mlir(verify=True)
        self.assertIn('"gf_tensor.broadcast"', mlir)
        self.assertIn('"gf_tensor.mul"', mlir)
        self.assertNotIn("heatmap", mlir)

    def test_to_numpy_preserves_strided_cpu_view(self):
        import numpy as np

        value = gf.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(
            value.transpose(0, 1).to_numpy(),
            np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32),
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable")
    def test_cuda_heatmap_uses_compiler_emitted_ttir(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        source = torch.linspace(
            0.0, 1.0, 128 * 128, device="cuda", dtype=torch.float32
        ).reshape(128, 128)
        raster = gf.visualize.heatmap(
            gf.from_torch(source), low=(0.0, 0.25, 1.0), high=(1.0, 0.5, 0.0))
        with patch.dict(os.environ, {"GRAPHFORGE_TENSOR_BACKEND": "native"}):
            raster.realize()
        expected = torch.stack(
            (source, 0.25 + source * 0.25, 1.0 - source), dim=-1
        ).cpu().numpy()
        import numpy as np

        np.testing.assert_allclose(raster.to_numpy(), expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(raster.execution["backend"], "cuda-ttir-triton")
        self.assertIn("tt.store", raster.generated_code("ttir"))
        redraw = raster.prepare()
        source.fill_(0.25)
        redraw()
        torch.cuda.synchronize()
        expected.fill(0.0)
        expected[..., 0] = 0.25
        expected[..., 1] = 0.3125
        expected[..., 2] = 0.75
        np.testing.assert_allclose(
            raster.to_numpy(), expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
