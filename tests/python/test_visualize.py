from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tiga as tg


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class VisualizationTest(unittest.TestCase):
    def test_heatmap_is_ordinary_tensor_ir_and_exports_rgb(self):
        import numpy as np

        values = tg.tensor([[0.0, 0.5], [1.0, 0.25]], dtype=tg.float32)
        raster = tg.visualize.heatmap(
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

    @staticmethod
    def _piecewise_linear(t, positions, colors):
        import numpy as np

        t = np.clip(np.asarray(t), positions[0], positions[-1])
        index = np.clip(np.searchsorted(positions, t, side="right") - 1,
                        0, len(positions) - 2)
        span = positions[index + 1] - positions[index]
        weight = (t - positions[index]) / span
        colors = np.asarray(colors)
        return colors[index] + weight[:, None] * (
            colors[index + 1] - colors[index])

    def test_heatmap_named_colormap_matches_piecewise_linear_stops(self):
        import numpy as np
        from tiga.visualize import _COLORMAPS

        flat = np.linspace(-0.5, 1.5, 48, dtype=np.float32)
        values = tg.tensor(flat.reshape(6, 8).tolist(), dtype=tg.float32)
        positions = np.linspace(0.0, 1.0, 8)
        for name, stops in _COLORMAPS.items():
            actual = tg.visualize.heatmap(
                values, cmap=name).to_numpy().reshape(-1, 3)
            expected = self._piecewise_linear(flat, positions, stops)
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        mlir = tg.visualize.heatmap(values, cmap="viridis").mlir(verify=True)
        self.assertIn('"gf_tensor.sqrt"', mlir)
        self.assertNotIn("heatmap", mlir)

    def test_heatmap_custom_and_positioned_colormaps(self):
        import numpy as np

        flat = np.linspace(-0.5, 1.5, 48, dtype=np.float32)
        values = tg.tensor(flat.reshape(6, 8).tolist(), dtype=tg.float32)
        custom = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        actual = tg.visualize.heatmap(
            values, cmap=custom).to_numpy().reshape(-1, 3)
        expected = self._piecewise_linear(flat, np.array([0.0, 0.5, 1.0]), custom)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        stops = [(0.2, (0.0, 0.0, 0.0)), (0.6, (1.0, 1.0, 1.0)),
                 (0.9, (1.0, 0.0, 0.0))]
        actual = tg.visualize.heatmap(
            values, cmap=stops).to_numpy().reshape(-1, 3)
        expected = self._piecewise_linear(
            flat, np.array([stop[0] for stop in stops]),
            [stop[1] for stop in stops])
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_heatmap_two_stop_colormap_matches_low_high(self):
        import numpy as np

        values = tg.tensor([[0.0, 0.5], [1.5, 0.25]], dtype=tg.float32)
        colormap = tg.visualize.heatmap(
            values, cmap=[(0.1, 0.2, 0.3), (0.9, 0.8, 0.7)]).to_numpy()
        ramp = tg.visualize.heatmap(
            values, low=(0.1, 0.2, 0.3), high=(0.9, 0.8, 0.7)).to_numpy()
        np.testing.assert_allclose(colormap, ramp, rtol=1e-6, atol=1e-6)

    def test_heatmap_default_is_full_viridis_not_endpoint_lerp(self):
        import numpy as np

        values = tg.tensor([[0.0, 0.5], [1.5, 0.25]], dtype=tg.float32)
        default = tg.visualize.heatmap(values).to_numpy()
        explicit = tg.visualize.heatmap(values, cmap="viridis").to_numpy()
        np.testing.assert_allclose(default, explicit, rtol=1e-6, atol=1e-6)
        lerp = tg.visualize.heatmap(
            values, low=(0.267, 0.005, 0.329),
            high=(0.993, 0.906, 0.144)).to_numpy()
        # the midpoint differs: full viridis passes through teal, not brown
        self.assertFalse(np.allclose(default[0, 1], lerp[0, 1], atol=1e-3))
        np.testing.assert_allclose(  # clamped to the end color out of range
            default[1, 0], explicit[1, 0], rtol=1e-6, atol=1e-6)

    def test_heatmap_colormap_rejects_invalid_specs(self):
        values = tg.tensor([[0.0, 0.5]], dtype=tg.float32)
        with self.assertRaises(ValueError):
            tg.visualize.heatmap(values, cmap="not-a-colormap")
        with self.assertRaises(ValueError):
            tg.visualize.heatmap(values, cmap=[(1.0, 0.0, 0.0)])
        with self.assertRaises(ValueError):
            tg.visualize.heatmap(
                values, cmap=[(0.5, (0, 0, 0)), (0.4, (1, 1, 1))])
        with self.assertRaises(ValueError):
            tg.visualize.heatmap(values, cmap="viridis", vmin=1.0, vmax=1.0)

    def test_to_numpy_preserves_strided_cpu_view(self):
        import numpy as np

        value = tg.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
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
        raster = tg.visualize.heatmap(
            tg.from_torch(source), low=(0.0, 0.25, 1.0), high=(1.0, 0.5, 0.0))
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
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

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable")
    def test_cuda_heatmap_colormap_matches_stops(self):
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        source = torch.linspace(
            -0.2, 1.2, 128 * 128, device="cuda", dtype=torch.float32
        ).reshape(128, 128)
        raster = tg.visualize.heatmap(tg.from_torch(source), cmap="plasma")
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            raster.realize()
        self.assertEqual(raster.execution["backend"], "cuda-ttir-triton")
        from tiga.visualize import _COLORMAPS

        positions = np.linspace(0.0, 1.0, 8)
        expected = self._piecewise_linear(
            source.cpu().numpy().reshape(-1), positions, _COLORMAPS["plasma"]
        ).reshape(128, 128, 3)
        np.testing.assert_allclose(
            raster.to_numpy(), expected, rtol=2e-4, atol=2e-4)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class CameraTest(unittest.TestCase):
    def test_auto_planar_cloud_looks_down_z(self):
        import numpy as np

        rng = np.random.default_rng(0)
        positions = rng.normal(size=(64, 2))
        camera = tg.visualize.Camera.auto(positions)
        direction = np.asarray(camera.position) - np.asarray(camera.target)
        direction /= np.linalg.norm(direction)
        np.testing.assert_allclose(direction, [0.0, 0.0, 1.0], atol=1e-12)

    def test_auto_line_cloud_views_perpendicular_to_line(self):
        import numpy as np

        rng = np.random.default_rng(1)
        line = np.array([1.0, 2.0, 3.0])
        line /= np.linalg.norm(line)
        t = np.linspace(-2.0, 2.0, 128)
        positions = t[:, None] * line + rng.normal(scale=0.01, size=(128, 3))
        camera = tg.visualize.Camera.auto(positions)
        direction = np.asarray(camera.position) - np.asarray(camera.target)
        direction /= np.linalg.norm(direction)
        self.assertAlmostEqual(abs(float(direction @ line)), 0.0, delta=0.05)

    def test_auto_projects_inside_unit_ndc_and_centers_centroid(self):
        import numpy as np

        rng = np.random.default_rng(2)
        positions = rng.normal(size=(200, 3)) + np.array([5.0, -3.0, 1.0])
        camera = tg.visualize.Camera.auto(positions)
        ndc, _, visible = camera.world_to_ndc(positions)
        self.assertTrue(visible.all())
        self.assertLessEqual(np.abs(ndc).max(), 1.0)
        ndc_centroid, _, _ = camera.world_to_ndc(
            positions.mean(axis=0, keepdims=True))
        np.testing.assert_allclose(ndc_centroid, [[0.0, 0.0]], atol=1e-9)

    def test_world_to_ndc_matches_handwritten_reference(self):
        import numpy as np

        camera = tg.visualize.Camera(
            position=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0),
            fov=90.0, up=(0.0, 1.0, 0.0))
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.5, 0.0],
            [0.0, 0.0, 5.0],  # behind the camera
        ])
        # reference: forward=(0,0,-1), right=(1,0,0), up=(0,1,0), f=1
        relative = points - np.array([0.0, 0.0, 3.0])
        w = relative @ np.array([0.0, 0.0, -1.0])
        reference = np.column_stack([relative[:, 0] / w, relative[:, 1] / w])
        ndc, depth, visible = camera.world_to_ndc(points)
        np.testing.assert_allclose(depth, w, rtol=1e-12)
        np.testing.assert_allclose(ndc[:3], reference[:3], rtol=1e-12)
        np.testing.assert_array_equal(visible, [True, True, True, False])


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class ParticlesTest(unittest.TestCase):
    def test_particles_peak_lands_on_brightest_point(self):
        import numpy as np

        positions = [[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]]
        raster = tg.visualize.particles(
            positions, values=[10.0, 1.0, 1.0],
            width=64, height=64, point_radius=1.0)
        image = raster.to_numpy()
        peak_y, peak_x = np.unravel_index(
            image.sum(axis=-1).argmax(), image.shape[:2])
        # (-1, -1) maps to the lower-left quadrant of the image
        self.assertLess(peak_x, 32)
        self.assertGreater(peak_y, 32)
        self.assertIn('"gf_tensor.broadcast"', raster.mlir())

    def test_particles_density_field_without_values(self):
        positions = [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]
        raster = tg.visualize.particles(
            positions, width=32, height=32, point_radius=1.0)
        self.assertEqual(raster.to_numpy().shape, (32, 32, 3))

    def test_non_finite_values_are_dropped_not_fatal(self):
        import numpy as np

        positions = [[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]]
        values = [float("nan"), 0.5, 1.0]  # e.g. a degree-0 mean reduction
        raster = tg.visualize.particles(
            positions, values=values, width=64, height=64, point_radius=1.0)
        image = raster.to_numpy()
        self.assertTrue(np.isfinite(image).all())
        peak_y, peak_x = np.unravel_index(
            image.sum(axis=-1).argmax(), image.shape[:2])
        # the brightest point (value 1.0) sits top-center at (0, 1); the NaN
        # point at (-1, -1) (lower-left quadrant) contributes nothing
        self.assertLess(peak_y, 32)
        self.assertFalse(peak_x < 32 and peak_y > 32)
        raster = tg.visualize.delaunay(
            positions, values=values, width=64, height=64)
        self.assertTrue(np.isfinite(raster.to_numpy()).all())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable")
    def test_particles_accepts_torch_cuda_positions(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        positions = torch.randn(32, 3, device="cuda")
        values = torch.rand(32, device="cuda")
        raster = tg.visualize.particles(
            positions, values=values, width=32, height=32)
        self.assertEqual(raster.to_numpy().shape, (32, 32, 3))

    def test_cmap_recolors_particles_delaunay_and_mesh(self):
        import numpy as np

        positions = [[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]]
        faces = [[0, 1, 2]]
        renderers = (
            lambda **color: tg.visualize.particles(
                positions, width=32, height=32, point_radius=1.0, **color),
            lambda **color: tg.visualize.delaunay(
                positions, width=32, height=32, **color),
            lambda **color: tg.visualize.mesh(
                positions, faces, width=32, height=32, **color),
        )
        for render in renderers:
            default = render()
            gray = render(cmap="gray")  # brightest pixel maps to the last stop
            peak = np.unravel_index(
                gray.to_numpy().sum(axis=-1).argmax(), (32, 32))
            np.testing.assert_allclose(
                gray.to_numpy()[peak], [1.0, 1.0, 1.0], atol=2e-2)
            self.assertFalse(  # the default viridis ramp peaks yellow instead
                np.allclose(default.to_numpy()[peak], [1.0, 1.0, 1.0],
                            atol=2e-2))


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class DelaunayTest(unittest.TestCase):
    _CORNERS = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, 0.0]]
    _VALUES = [0.0, 0.25, 0.75, 1.0, 0.5]

    def test_delaunay_fills_triangles_with_gradient(self):
        import numpy as np

        raster = tg.visualize.delaunay(
            self._CORNERS, values=self._VALUES, width=64, height=64)
        image = raster.to_numpy()
        luminance = image.sum(axis=-1)
        self.assertGreater((luminance > 0).mean(), 0.5)  # triangle coverage
        self.assertGreater(len(np.unique(luminance)), 2)  # color gradient
        self.assertIn('"gf_tensor.broadcast"', raster.mlir())

    def test_delaunay_wireframe_and_3d_projection(self):
        import numpy as np

        wireframe = tg.visualize.delaunay(
            self._CORNERS, values=self._VALUES,
            width=64, height=64, wireframe=True)
        wire_image = wireframe.to_numpy()
        self.assertGreater((wire_image.sum(axis=-1) > 0).mean(), 0.0)
        rng = np.random.default_rng(3)
        raster = tg.visualize.delaunay(
            rng.normal(size=(40, 3)), width=64, height=64)
        self.assertEqual(raster.to_numpy().shape, (64, 64, 3))


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class LoadObjTest(unittest.TestCase):
    _MINI = """\
# comment rows are skipped
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1/1/1 2/2/2 3/3/3
f 1 3 4 2
"""

    def test_parses_slashed_tokens_and_fans_quads(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quad.obj"
            path.write_text(self._MINI)
            positions, faces = tg.visualize.load_obj(path)
        np.testing.assert_allclose(
            positions,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
             [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual(positions.dtype, np.float64)
        np.testing.assert_array_equal(
            faces, [[0, 1, 2], [0, 2, 3], [0, 3, 1]])  # quad fan-triangulated

    def test_bad_files_raise_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "empty.obj"
            empty.write_text("# nothing here\n")
            with self.assertRaises(ValueError):
                tg.visualize.load_obj(empty)
            no_faces = Path(directory) / "points.obj"
            no_faces.write_text("v 0.0 0.0 0.0\nv 1.0 0.0 0.0\n")
            with self.assertRaises(ValueError):
                tg.visualize.load_obj(no_faces)
            out_of_range = Path(directory) / "oob.obj"
            out_of_range.write_text(
                "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 7\n")
            with self.assertRaises(ValueError):
                tg.visualize.load_obj(out_of_range)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class MeshTest(unittest.TestCase):
    _TRIANGLE = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]

    def test_single_triangle_coverage(self):
        import numpy as np

        raster = tg.visualize.mesh(
            self._TRIANGLE, [[0, 1, 2]], width=64, height=64)
        image = raster.to_numpy()
        background = image[0, 0]  # uniform value -> one fill + background
        self.assertGreater(
            (np.abs(image - background).sum(axis=-1) > 1e-6).mean(), 0.1)
        self.assertIn('"gf_tensor.broadcast"', raster.mlir())

    def test_painters_algorithm_near_triangle_wins(self):
        import numpy as np

        camera = tg.visualize.Camera(
            position=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0),
            fov=45.0, up=(0.0, 1.0, 0.0))
        positions = [
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0],  # near
            [-1.5, -1.5, 0.0], [1.5, -1.5, 0.0], [0.0, 1.5, 0.0],  # far
        ]
        faces = [[0, 1, 2], [3, 4, 5]]
        values = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # near bright, far dark
        raster = tg.visualize.mesh(
            positions, faces, values=values, camera=camera,
            width=64, height=64, vmin=0.0, vmax=1.0)
        image = raster.to_numpy()
        center = image[32, 32]
        # the nearer (brighter) triangle occludes the darker one behind it
        self.assertGreater(float(center.sum()), 2.0)
        corner = image[4, 4]
        self.assertLess(float(corner.sum()), float(center.sum()))

    def test_wireframe_and_non_finite_values(self):
        import numpy as np

        raster = tg.visualize.mesh(
            self._TRIANGLE, [[0, 1, 2]], width=64, height=64,
            wireframe=True)
        self.assertGreater((raster.to_numpy().sum(axis=-1) > 0).mean(), 0.0)
        values = [float("nan"), 0.5, 1.0]  # e.g. a degree-0 mean reduction
        raster = tg.visualize.mesh(
            self._TRIANGLE, [[0, 1, 2]], values=values,
            width=64, height=64)
        image = raster.to_numpy()
        self.assertTrue(np.isfinite(image).all())
        # all-non-finite triangles are skipped without failing
        raster = tg.visualize.mesh(
            self._TRIANGLE, [[0, 1, 2]],
            values=[float("nan")] * 3, width=64, height=64)
        self.assertTrue(np.isfinite(raster.to_numpy()).all())


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class SaveVideoTest(unittest.TestCase):
    def _frames(self, count=4, size=16):
        import numpy as np

        for index in range(count):
            yield np.full((size, size, 3), index / (count - 1))

    def test_gif_roundtrip_via_pillow(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = tg.visualize.save_video(
                self._frames(), Path(directory) / "frames.gif", fps=10)
            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 4)
                self.assertEqual(image.size, (16, 16))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is unavailable")
    def test_mp4_streams_through_ffmpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            path = tg.visualize.save_video(
                self._frames(), Path(directory) / "frames.mp4", fps=10)
            self.assertGreater(path.stat().st_size, 0)

    def test_unknown_suffix_and_mismatched_shapes_are_rejected(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                tg.visualize.save_video(self._frames(), Path(directory) / "x.avi")
            frames = [np.zeros((8, 8, 3)), np.zeros((4, 4, 3))]
            with self.assertRaises(ValueError):
                tg.visualize.save_video(frames, Path(directory) / "x.gif")

    def test_invalid_frame_rate_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "not-created" / "frame.gif"
            for fps in (0, -1, True, float("nan"), float("inf"), "30"):
                with self.subTest(fps=fps):
                    with self.assertRaisesRegex(ValueError, "finite positive"):
                        tg.visualize.save_video([], target, fps=fps)
                    self.assertFalse(target.parent.exists())


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class ExchangeTest(unittest.TestCase):
    _POSITIONS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [1.0, 1.0, 0.25], [0.0, 1.0, 0.25]]
    _FACES = [[0, 1, 2], [0, 2, 3]]
    _VALUES = [0.0, 0.5, 1.0, 0.75]

    @staticmethod
    def _parse_ply(path):
        """Minimal binary_little_endian PLY reader for round-trip checks."""
        import numpy as np

        raw = path.read_bytes()
        header, _, body = raw.partition(b"end_header\n")
        lines = header.decode("ascii").splitlines()
        assert lines[0] == "ply"
        assert "format binary_little_endian 1.0" in lines
        elements = {}
        order = []
        for line in lines:
            parts = line.split()
            if parts[0] == "element":
                elements[parts[1]] = {"count": int(parts[2]), "props": []}
                order.append(parts[1])
            elif parts[0] == "property":
                elements[order[-1]]["props"].append(parts[1:])
        result = {}
        sizes = {"float": ("<f4", 4), "uchar": ("<u1", 1), "int": ("<i4", 4)}
        offset = 0
        for name in order:
            element = elements[name]
            list_prop = next(
                (p for p in element["props"] if p[0] == "list"), None)
            if list_prop is None:
                dtype = np.dtype([
                    (p[1], sizes[p[0]][0]) for p in element["props"]])
                result[name] = np.frombuffer(
                    body, dtype=dtype, count=element["count"],
                    offset=offset)
                offset += element["count"] * dtype.itemsize
            else:
                count_dtype, count_size = sizes[list_prop[1]]
                item_dtype, item_size = sizes[list_prop[2]]
                rows = []
                for _ in range(element["count"]):
                    row_count = int(np.frombuffer(
                        body, dtype=count_dtype, count=1, offset=offset)[0])
                    offset += count_size
                    rows.append(np.frombuffer(
                        body, dtype=item_dtype, count=row_count,
                        offset=offset))
                    offset += row_count * item_size
                result[name] = rows
        return result

    def test_ply_roundtrip_with_faces_and_values(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = tg.visualize.export_ply(
                Path(directory) / "mesh.ply", self._POSITIONS,
                faces=self._FACES, values=self._VALUES)
            parsed = self._parse_ply(path)
        vertex = parsed["vertex"]
        self.assertEqual(len(vertex), 4)
        np.testing.assert_allclose(
            np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1),
            self._POSITIONS, atol=1e-6)
        self.assertIn("red", vertex.dtype.names)
        self.assertIn("scalar_value", vertex.dtype.names)
        np.testing.assert_allclose(
            vertex["scalar_value"], self._VALUES, atol=1e-6)
        # lowest value maps to the low palette color, highest to the high one
        low_rgb = [round(255 * c) for c in (0.267, 0.005, 0.329)]
        high_rgb = [round(255 * c) for c in (0.993, 0.906, 0.144)]
        self.assertEqual(
            [int(vertex["red"][0]), int(vertex["green"][0]),
             int(vertex["blue"][0])], low_rgb)
        self.assertEqual(
            [int(vertex["red"][2]), int(vertex["green"][2]),
             int(vertex["blue"][2])], high_rgb)
        np.testing.assert_array_equal(parsed["face"], self._FACES)

    def test_ply_without_values_has_no_color(self):
        with tempfile.TemporaryDirectory() as directory:
            path = tg.visualize.export_ply(
                Path(directory) / "points.ply", self._POSITIONS)
            parsed = self._parse_ply(path)
        self.assertEqual(len(parsed["vertex"]), 4)
        self.assertNotIn("red", parsed["vertex"].dtype.names)
        self.assertNotIn("face", parsed)

    def test_ply_2d_positions_get_zero_z(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = tg.visualize.export_ply(
                Path(directory) / "flat.ply",
                [[1.0, 2.0], [3.0, 4.0]])
            parsed = self._parse_ply(path)
        np.testing.assert_allclose(parsed["vertex"]["z"], [0.0, 0.0])

    def test_ply_million_points_exports_fast(self):
        import time

        import numpy as np

        rng = np.random.default_rng(0)
        positions = rng.random((1_000_000, 3))
        values = rng.random(1_000_000)
        with tempfile.TemporaryDirectory() as directory:
            start = time.monotonic()
            path = tg.visualize.export_ply(
                Path(directory) / "cloud.ply", positions, values=values)
            elapsed = time.monotonic() - start
            parsed = self._parse_ply(path)
        self.assertLess(elapsed, 2.0)
        self.assertEqual(len(parsed["vertex"]), 1_000_000)

    def test_obj_roundtrip_is_identity(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = tg.visualize.export_obj(
                Path(directory) / "mesh.obj", self._POSITIONS, self._FACES)
            positions, faces = tg.visualize.load_obj(path)
        np.testing.assert_allclose(positions, self._POSITIONS, atol=1e-15)
        np.testing.assert_array_equal(faces, self._FACES)

    @unittest.skipIf(importlib.util.find_spec("pyopenvdb") is not None,
                     "pyopenvdb is installed")
    def test_export_vdb_requires_pyopenvdb(self):
        with (tempfile.TemporaryDirectory() as directory,
              self.assertRaises(ModuleNotFoundError) as caught):
            tg.visualize.export_vdb(
                Path(directory) / "cloud.vdb", self._POSITIONS, self._VALUES)
        self.assertIn("pyopenvdb", str(caught.exception))

    def test_load_ply_binary_roundtrip_with_extras(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = tg.visualize.export_ply(
                Path(directory) / "mesh.ply", self._POSITIONS,
                faces=self._FACES, values=self._VALUES)
            positions, faces, extras = tg.visualize.load_ply(path)
        np.testing.assert_allclose(positions, self._POSITIONS, atol=1e-6)
        np.testing.assert_array_equal(faces, self._FACES)
        np.testing.assert_allclose(
            extras["scalar_value"], self._VALUES, atol=1e-6)
        self.assertIn("red", extras)

    def test_load_ply_ascii_with_faces_and_extra_properties(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.ply"
            path.write_text(
                "ply\nformat ascii 1.0\n"
                "element vertex 3\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property float f_dc_0\n"
                "element face 1\n"
                "property list uchar int vertex_indices\n"
                "end_header\n"
                "0 0 0 0.1\n1 0 0 0.2\n0 1 0 0.3\n3 0 1 2\n")
            positions, faces, extras = tg.visualize.load_ply(path)
        self.assertEqual(positions.shape, (3, 3))
        np.testing.assert_array_equal(faces, [[0, 1, 2]])
        np.testing.assert_allclose(extras["f_dc_0"], [0.1, 0.2, 0.3])

    def test_load_ply_rejects_bad_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.ply"
            path.write_text("not a ply\n")
            with self.assertRaises(ValueError):
                tg.visualize.load_ply(path)
            path.write_text("ply\nformat ascii 1.0\nend_header\n")
            with self.assertRaises(ValueError):
                tg.visualize.load_ply(path)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class SplatsTest(unittest.TestCase):
    _CAMERA = None

    def test_gaussians_public_name_and_legacy_alias(self):
        self.assertIn("gaussians", tg.visualize.__all__)
        self.assertIs(tg.visualize.gaussians, tg.visualize.splats)
        self.assertEqual(tg.visualize.gaussians.__name__, "gaussians")

    def setUp(self):
        self._camera = tg.visualize.Camera(
            position=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0))

    def test_single_gaussian_blobs_at_center(self):
        import numpy as np

        raster = tg.visualize.gaussians(
            [[0.0, 0.0, 0.0]], [(1.0, 0.0, 0.0)], [0.3],
            opacities=[0.9], camera=self._camera, width=64, height=64)
        image = raster.to_numpy()
        self.assertGreater(image[32, 32, 0], 0.8)
        self.assertLess(image[2, 2, 0], 0.01)
        self.assertEqual(image.shape, (64, 64, 3))

    def test_near_gaussian_occludes_far_one(self):
        raster = tg.visualize.splats(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
            [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
            [0.4, 0.4], opacities=[1.0, 1.0],
            camera=tg.visualize.Camera(
                position=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.25),
                up=(0.0, 1.0, 0.0)),
            width=64, height=64)
        center = raster.to_numpy()[32, 32]
        self.assertGreater(center[0], 0.9)
        self.assertLess(center[2], 0.05)

    def test_anisotropic_rotated_gaussian_renders(self):
        import numpy as np

        raster = tg.visualize.splats(
            [[0.0, 0.0, 0.0]], [(0.0, 1.0, 0.0)], [(0.5, 0.05, 0.05)],
            rotations=[[0.9238795, 0.0, 0.0, 0.3826834]],  # 45° about x
            camera=self._camera, width=64, height=64)
        image = raster.to_numpy()
        self.assertTrue(np.isfinite(image).all())
        self.assertGreater(image[32, 32, 1], 0.5)

    def test_splats_validate_shapes(self):
        with self.assertRaises(ValueError):
            tg.visualize.splats(
                [[0.0, 0.0, 0.0]], [(1.0, 0.0)], [0.3], camera=self._camera)
        with self.assertRaises(ValueError):
            tg.visualize.splats(
                [[0.0, 0.0, 0.0]], [(1.0, 0.0, 0.0)], [0.0],
                camera=self._camera)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class VolumeTest(unittest.TestCase):
    def test_constant_density_fully_occludes(self):
        import numpy as np

        raster = tg.visualize.volume(
            np.ones((16, 16, 16)), camera=(30, -60), width=64, height=64,
            steps=64, cmap="gray", vmin=0.0, vmax=1.0, scale=8.0)
        image = raster.to_numpy()
        self.assertGreater(image[32, 32, 0], 0.95)  # chord length 2 × σ=8
        self.assertLess(image[2, 2, 0], 0.01)       # ray misses the cube

    def test_empty_volume_shows_background(self):
        import numpy as np

        raster = tg.visualize.volume(
            np.zeros((8, 8, 8)), camera=(30, -60), width=32, height=32,
            steps=16, background=(1.0, 0.0, 0.0))
        np.testing.assert_allclose(
            raster.to_numpy(),
            np.broadcast_to([1.0, 0.0, 0.0], (32, 32, 3)))

    def test_volume_validates_input(self):
        import numpy as np

        with self.assertRaises(ValueError):
            tg.visualize.volume(np.zeros((4, 4)))
        with self.assertRaises(ValueError):
            tg.visualize.volume(np.zeros((4, 4, 4)), steps=0)


if __name__ == "__main__":
    unittest.main()
