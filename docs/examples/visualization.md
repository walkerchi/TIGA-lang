# Visualization

Fields become images and videos without leaving the Tensor IR: the colormap
is a compiled kernel, geometry rasterization (splat, triangulation) is a
host-side preparation feeding the same color expression, and video export
streams rasters through Pillow or ffmpeg.

## GPU heatmap visualization prep { #gpu-heatmap-visualization-prep }

A heatmap turns a grid of numbers into an image by mapping each value to a
color. Here the grid is a two-dimensional sinusoidal pattern with values in
[0, 1]; a colormap converts every scalar into a red-green-blue (RGB) triple,
yielding one row of pixel data per grid point.

![Direct output of the program below: the 256×256 sinusoidal field rendered by the compiled colormap kernel — bright yellow where sin·cos approaches 1, dark purple where it approaches −1.](../assets/examples/gpu-heatmap-output.png)

$$\mathrm{field}[y, x] = 0.5 + 0.5\,\sin\!\left(\tfrac{x}{17}\right)
\cos\!\left(\tfrac{y}{23}\right)$$

`gf.visualize.heatmap` composes the broadcast arithmetic and the colormap
into a single scalar-to-RGB kernel. The generated MLIR/provider artifact is
exposed for inspection, and the optional PNG encoding stays outside the
compiler core.

```python
--8<-- "examples/gpu_heatmap.py:core"
```

The default colormap is the built-in `viridis`; `cmap` switches to another
multi-stop colormap — a name, a list of RGB colors, or `(position, RGB)`
stops — and `low`/`high` give a plain two-color ramp, always in the same
fused kernel:

```python
raster = gf.visualize.heatmap(field, cmap="magma")   # see gf.visualize.colormaps()
```

![Direct output of heatmap itself: the seven built-in colormaps, top to bottom coolwarm, gray, inferno, jet, magma, plasma, viridis — each band is one compiled kernel evaluated on a [0, 1] ramp.](../assets/examples/heatmap-colormaps.png)

??? example "Full source: examples/gpu_heatmap.py (runs as-is)"

    ```python
    --8<-- "examples/gpu_heatmap.py"
    ```

## Particles and video from positions { #particles-and-video }

**What it is.** `visualize_fields.py` runs a tiny FEM-style simulation:
steady heat conduction on a disc, discretized as a jittered polar point
set. The core (r ≤ 0.2) is held at T = 1 and the rim (r ≥ 0.95) at T = 0;
every interior node relaxes toward its neighbor mean — a
[Jacobi](https://en.wikipedia.org/wiki/Jacobi_method) sweep for the
Laplace equation, expressed as message passing with a mean reducer and a
Dirichlet mask:

$$T_i \leftarrow \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} T_j \;\; \text{(interior)}, \qquad T = 1 \;\text{on}\; \Gamma_{\mathrm{core}}, \quad T = 0 \;\text{on}\; \Gamma_{\mathrm{rim}}$$

The neighbor relation comes from the Delaunay triangulation of the same
points, so the graph and the rendered mesh agree exactly. Each node
renders as one disk splat, and `gf.visualize.save_video` streams the
relaxation into a GIF or MP4.

![Particles: every mesh node splatted as a disk, colored by temperature — the hot core glows while the concentric discretization rings stay visible.](../assets/examples/fields_particles.png)

![Animation: Jacobi relaxation of the heat equation — the initial step profile smooths into the steady logarithmic gradient.](../assets/examples/fields_diffusion.gif)

```python
--8<-- "examples/visualize_fields.py:core"
```

??? example "Full source: examples/visualize_fields.py (runs as-is)"

    ```python
    --8<-- "examples/visualize_fields.py"
    ```

## Meshes: Delaunay from points or loaded OBJ { #simulation-meshes-load-shade-orbit }

**What it is.** Rendering a continuous surface needs faces, not just
positions — and faces arrive two ways. When a simulation produces only
scattered positions, [Delaunay](https://en.wikipedia.org/wiki/Delaunay_triangulation)
triangulation computes them: `gf.visualize.delaunay` turns the disc's point
set into a mesh and fills each triangle with its mean vertex value — the
steady temperature field drawn as a classic FEM contour plot.

![Delaunay: the disc triangulated and filled with mean vertex values — the same steady state as a continuous temperature field.](../assets/examples/fields_delaunay.png)

```python
--8<-- "examples/visualize_fields.py:delaunay"
```

When the triangulation is part of the data — a solver's surface mesh, a
sculpted 3-D model — `visualize_mesh.py` loads it directly: a subdivision-2
[icosphere](https://en.wikipedia.org/wiki/Geodesic_polyhedron)
(162 vertices, 320 faces) from an [OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
file via `gf.visualize.load_obj`, its unique mesh edges turned into an
undirected `Graph.from_csr` relation, a Gaussian bump diffused from one
vertex through a mean reducer. The mesh then renders from a fixed camera,
plus a twelve-frame azimuth orbit exported as GIF.

`gf.visualize.mesh` takes the faces as given — where `gf.visualize.delaunay`
computes a triangulation from points, `mesh` renders the triangles a
simulation already produced. Overlap resolves through the
[painter's algorithm](https://en.wikipedia.org/wiki/Painter%27s_algorithm):
triangles fill far-to-near by mean vertex depth, so the near side of the
sphere cleanly occludes the far side. A fixed `vmin`/`vmax` keeps the colors
comparable across orbit frames.

The same script also exports the geometry for [Blender](https://en.wikipedia.org/wiki/Blender_(software)):
`gf.visualize.export_ply` writes a binary PLY whose vertices carry the
diffused field both as vertex colors and as a raw `scalar_value` attribute,
and `gf.visualize.export_obj` writes the plain positions/faces text format
that `load_obj` reads back identically.

![Shaded mesh: the icosphere filled triangle by triangle, the diffused bump bright on the right limb — the far side is fully occluded by the painter's algorithm.](../assets/examples/mesh_flat.png){ width="49%" }
![Wireframe: the same mesh drawn as triangle edges — the full subdivision-2 topology, front and back.](../assets/examples/mesh_wireframe.png){ width="49%" }

![Animation: a twelve-frame azimuth orbit around the mesh — the bright bump swings to the far side and disappears behind the occluding near hemisphere.](../assets/examples/mesh_orbit.gif)

```python
--8<-- "examples/visualize_mesh.py:core"
```

??? example "Full source: examples/visualize_mesh.py (runs as-is)"

    ```python
    --8<-- "examples/visualize_mesh.py"
    ```

## Gaussian splats and volumes { #gaussian-splats-and-volumes }

**What it is.** Some scenes are neither grids nor meshes but a cloud of
anisotropic 3-D [Gaussians](https://en.wikipedia.org/wiki/Gaussian_splatting) —
the representation 3-D Gaussian Splatting trains from photographs.
`gf.visualize.splats` renders them directly: each Gaussian's covariance
(quaternion rotation, per-axis scales) projects through the perspective
Jacobian to a 2-D ellipse, and colors composite near-to-far with
transmittance:

$$\Sigma = R\,\mathrm{diag}(s^2)\,R^\top, \qquad
C = \sum_{i} T_i\,\alpha_i\,c_i, \qquad
\alpha_i = o_i \, e^{-\frac{1}{2}\,\delta^\top \Sigma'^{-1} \delta}, \qquad
T_i = \prod_{j<i} (1 - \alpha_j)$$

`visualize_gaussians.py` builds a colored torus of tangent-stretched
Gaussians and renders it twice: as explicit splats, and as a voxel density
grid ray-marched by `gf.visualize.volume` — the two standard readings of
the same scene, with the emission-absorption coefficient

$$\alpha = 1 - e^{-\sigma \Delta t}$$

![Gaussian splatting: the torus rendered as explicit anisotropic Gaussians — soft elliptical footprints, colors composited through transmittance.](../assets/examples/gaussians_splats.png){ width="49%" }
![Volume rendering: the same scene accumulated into a density grid and ray-marched — the emissive torus occludes itself through the emission-absorption integral.](../assets/examples/gaussians_volume.png){ width="49%" }

![Animation: a twelve-frame azimuth orbit of the splatted torus.](../assets/examples/gaussians_orbit.gif)

```python
--8<-- "examples/visualize_gaussians.py:core"
```

??? example "Full source: examples/visualize_gaussians.py (runs as-is)"

    ```python
    --8<-- "examples/visualize_gaussians.py"
    ```

A PLY exported by a 3-D Gaussian Splatting training run loads through
`gf.visualize.load_ply`: vertex extras carry `f_dc_*`, `opacity`,
`scale_*` and `rot_*` quaternions. The standard conversions are

$$\mathrm{RGB} = 0.5 + 0.2821 \cdot f_{dc}, \qquad
o = \mathrm{sigmoid}(\text{opacity}), \qquad
s = \exp(\text{scale})$$

— one `np.column_stack` per field and the result feeds `splats` unchanged.
