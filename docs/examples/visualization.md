# Visualization

These are native visualization/compiler examples: `heatmap` currently requires
`tg.Tensor`, not a Torch tensor. Install the `visualization` extra; CPU is the
default, and CUDA additionally needs the `cuda` extra. Geometry preparation and
encoding are outside compiler IR; expanded full sources include snippet setup.


Fields become images and videos: the colormap
is a compiled kernel, geometry rasterization (splat, triangulation) is a
host-side preparation feeding the same color expression, and video export
encodes rasters through Pillow (GIF buffers all frames) or ffmpeg (MP4 streams frames).

- [`python examples/gpu_heatmap.py --device cpu --size 64 --output output/examples/heatmap.png`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/gpu_heatmap.py)
- [`python examples/visualize_fields.py --rings 8 --spokes 12 --steps 10 --size 128 --output output/examples`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/visualize_fields.py)
- [`python examples/visualize_mesh.py --model examples/assets/icosphere.obj --steps 2 --size 128 --output output/examples`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/visualize_mesh.py)
- [`python examples/visualize_gaussians.py --count 100 --size 128 --output output/examples`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/visualize_gaussians.py)

Inspect the images under `output/examples/`; MP4 export additionally requires ffmpeg.

## GPU heatmap visualization prep { #gpu-heatmap-visualization-prep }

A heatmap turns a grid of numbers into an image by mapping each value to a
color. Here the grid is a two-dimensional sinusoidal pattern with values in
[0, 1]; a colormap converts every scalar into a red-green-blue (RGB) triple,
yielding one row of pixel data per grid point.

![Direct output of the program below: the 256×256 sinusoidal field rendered by the compiled colormap kernel — bright yellow where sin·cos approaches 1, dark purple where it approaches −1.](../assets/examples/gpu-heatmap-output.png)

$$
\mathrm{field}[y, x] = 0.5 + 0.5\,\sin\!\left(\tfrac{x}{17}\right)
\cos\!\left(\tfrac{y}{23}\right)
$$

`tg.visualize.heatmap` composes the broadcast arithmetic and the colormap
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
raster = tg.visualize.heatmap(field, cmap="magma")   # see tg.visualize.colormaps()
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

$$
T_i \leftarrow \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} T_j \;\; \text{(interior)}, \qquad T = 1 \;\text{on}\; \Gamma_{\mathrm{core}}, \quad T = 0 \;\text{on}\; \Gamma_{\mathrm{rim}}
$$

The neighbor relation comes from the Delaunay triangulation of the same
points, so the graph and the rendered mesh agree exactly. Each node
renders as one disk splat, and `tg.visualize.save_video` exports the
relaxation into a GIF (host-buffered) or MP4 (streamed).

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
triangulation computes them: `tg.visualize.delaunay` turns the disc's point
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
file via `tg.visualize.load_obj`, its unique mesh edges turned into an
undirected `Graph.from_csr` relation, a Gaussian bump diffused from one
vertex through a mean reducer. The mesh then renders from a fixed camera,
plus a twelve-frame azimuth orbit exported as GIF.

`tg.visualize.mesh` takes the faces as given — where `tg.visualize.delaunay`
computes a triangulation from points, `mesh` renders the triangles a
simulation already produced. Overlap resolves through the
[painter's algorithm](https://en.wikipedia.org/wiki/Painter%27s_algorithm):
triangles fill far-to-near by mean vertex depth, so the near side of the
sphere cleanly occludes the far side. A fixed `vmin`/`vmax` keeps the colors
comparable across orbit frames.

The same script also exports the geometry for [Blender](https://en.wikipedia.org/wiki/Blender_(software)):
`tg.visualize.export_ply` writes a binary PLY whose vertices carry the
diffused field both as vertex colors and as a raw `scalar_value` attribute,
and `tg.visualize.export_obj` writes the plain positions/faces text format
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

One set of 3-D [Gaussians](https://en.wikipedia.org/wiki/Gaussian_splatting), two views:
`tg.visualize.gaussians` shows their individual colors and shapes;
`tg.visualize.volume` shows the density they accumulate in space.

<div class="tg-render-comparison">
  <figure>
    <img src="../../assets/examples/gaussians_splats.png" alt="A blue-and-gold Gaussian torus with elliptical footprints aligned to its tangent." loading="lazy" width="512" height="512">
    <figcaption><strong>Gaussian splats</strong><br>Composite the colored elliptical footprints directly.</figcaption>
  </figure>
  <figure>
    <img src="../../assets/examples/gaussians_volume.png" alt="The same torus rendered as a voxel density field, with brighter colors indicating denser regions." loading="lazy" width="512" height="512">
    <figcaption><strong>Volume</strong><br>Sample density and accumulate emission and absorption along each ray.</figcaption>
  </figure>
</div>

This reproducible synthetic scene contains **2,200 Gaussians**, not a model trained
from photographs. Both images share positions and camera. The left image has
fixed lighting baked into its RGB; the right maps density to color, so this is
not a pixel-equivalence comparison. Projection and rendering currently run in
host NumPy; this is not a GPU rendering benchmark.

### Render Gaussians

`positions`, `colors` and `scales` are `(N, 3)` arrays; `rotations` contains
`(N, 4)` quaternions in `(w, x, y, z)` order, and `opacities` is `(N,)`.
The long axes follow the ring tangent; `scales` specifies standard deviations
along the three principal axes.

```python
--8<-- "examples/visualize_gaussians.py:core"
```

The script's `torus_gaussians` helper creates the scene arrays.
`args.size` and `output` come from the command line; `background` is a dark
blue-gray RGB value. Complete initialization appears in the source below.
Both renderers return a `Raster` whose `.save()` method writes PNG.

??? example "Render the same scene as a volume"

    The example-specific `splat_density` helper samples the tangent-aligned,
    anisotropic density into a `64 × 64 × 64` grid spanning `[-1, 1]³`.
    Individual RGB values are not retained: `cmap` colors the density instead.
    The camera is unchanged.

    ```python
    --8<-- "examples/visualize_gaussians.py:volume"
    ```

### Orbit the scene

<figure class="tg-render-motion">
  <video controls loop muted playsinline preload="none" width="384" height="384"
         poster="../../assets/examples/gaussians_splats.png"
         aria-label="Camera orbit around the Gaussian torus">
    <source src="../../assets/examples/gaussians_orbit.mp4" type="video/mp4">
    <a href="../../assets/examples/gaussians_orbit.mp4">Download the orbit video</a>
  </video>
  <figcaption>One orbit at a fixed elevation. 36 frames at 12 fps; play on demand, with no autoplay.</figcaption>
</figure>

Reproduce the images and video (MP4 encoding requires ffmpeg):

[`python examples/visualize_gaussians.py --count 2200 --size 512 --frames 36 --video-format mp4 --output output/examples`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/visualize_gaussians.py)

The random seed is fixed. Omit `--video-format mp4` to export GIF instead;
rendering itself does not require ffmpeg.

??? example "Full source: examples/visualize_gaussians.py"

    ```python
    --8<-- "examples/visualize_gaussians.py"
    ```

??? info "How rendering works: projection, occlusion and density"

    A Gaussian has a center, orientation and three axis scales. With rotation
    matrix `R`, its covariance is:

    $$
    \Sigma = R\,\mathrm{diag}(s^2)\,R^\top
    $$

    The perspective Jacobian projects it to a screen-space ellipse. For pixel
    offset `δ`, projected covariance `Σ′` and opacity `oᵢ`:

    $$
    \alpha_i = o_i \exp\!\left(-\tfrac12\delta^\top\Sigma'^{-1}\delta\right)
    $$

    Gaussians are processed near-to-far by center depth. Transmittance is the
    fraction remaining after earlier layers; the final color includes the
    unobscured background:

    $$
    T_i = \prod_{j<i}(1-\alpha_j),\qquad
    C = \sum_i T_i\alpha_i c_i + T_{\mathrm{end}}c_{\mathrm{background}}
    $$

    Volume rendering instead samples density along each ray. Sample density,
    `scale` and step length determine opacity:

    $$
    \alpha = 1-\exp(-\sigma\,\mathrm{scale}\,\Delta t)
    $$

??? info "Import trained Gaussians"

    `tg.visualize.load_ply` returns positions, optional faces and extra attributes.
    A common 3DGS PLY layout stores DC color coefficients in `f_dc_*`, quaternion
    components in `rot_*`, and unactivated opacity and log-scale. For that layout:

    $$
    \mathrm{RGB}=\mathrm{clip}(0.5+0.2821\,f_{\mathrm{dc}},0,1)
    $$

    $$
    o=\mathrm{sigmoid}(\mathrm{opacity}),\qquad s=\exp(\mathrm{scale})
    $$

    Stack properties in numerical suffix order and pass the arrays to
    `tg.visualize.gaussians`. This DC-only display does not evaluate higher-order,
    view-dependent spherical harmonics. Check the conventions of each exporter.
