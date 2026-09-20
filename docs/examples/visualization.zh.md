# 可视化

这是原生可视化/编译器例子：当前 `heatmap` 要求 `tg.Tensor`，尚不接受 Torch Tensor。
安装 `visualization` extra；默认 CPU 可运行，选择 CUDA 时另需 `cuda` extra。
几何准备和编码不在编译器 IR 内，核心片段的初始化见折叠完整源码。


由场生成图像和视频：colormap 是编译出的 kernel，
几何光栅化（溅射、三角剖分）是 host 端的准备步骤，复用同一个颜色表达式；
视频导出经 Pillow（GIF 缓存全部帧）或 ffmpeg（MP4 逐帧流式编码）处理 raster。

- [`python examples/gpu_heatmap.py --device cpu --size 64 --output output/examples/heatmap.png`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/gpu_heatmap.py)
- [`python examples/visualize_fields.py --rings 8 --spokes 12 --steps 10 --size 128 --output output/examples`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/visualize_fields.py)
- [`python examples/visualize_mesh.py --model examples/assets/icosphere.obj --steps 2 --size 128 --output output/examples`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/visualize_mesh.py)
- [`python examples/visualize_gaussians.py --count 100 --size 128 --output output/examples`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/visualize_gaussians.py)

运行后检查 `output/examples/` 下的图像；MP4 导出额外依赖 ffmpeg。

## GPU 热力图可视化准备 { #gpu-heatmap-visualization-prep }

[热力图](https://baike.baidu.com/item/热力图)把数字网格变成图像：每个数值映射为一种颜色。这里的
网格是取值在 [0, 1] 内的二维正弦图案；colormap 把每个标量
转换成红-绿-蓝（RGB）三元组，每个网格点对应一行像素数据。

![下方案例程序的直接输出：256×256 正弦场经编译出的 colormap kernel 渲染——sin·cos 接近 1 处为亮黄，接近 −1 处为暗紫。](../assets/examples/gpu-heatmap-output.png)

$$
\mathrm{field}[y, x] = 0.5 + 0.5\,\sin\!\left(\tfrac{x}{17}\right)
\cos\!\left(\tfrac{y}{23}\right)
$$

`tg.visualize.heatmap` 把广播算术和 colormap 组合成单个
“标量到 RGB”的 kernel。生成的 MLIR/provider 产物对外暴露，
可以直接查看；可选的 PNG 编码则留在编译器核心之外。

```python
--8<-- "examples/gpu_heatmap.py:core"
```

默认 colormap 是内置的 `viridis`；`cmap` 切换到其他多停靠点 colormap——
名称、RGB 颜色列表或 `(position, RGB)` 停靠点——`low`/`high` 则给出
朴素的双色 ramp，始终是同一个融合 kernel：

```python
raster = tg.visualize.heatmap(field, cmap="magma")   # 名称见 tg.visualize.colormaps()
```

![heatmap 自身的直接输出：七个内置 colormap，自上而下 coolwarm、gray、inferno、jet、magma、plasma、viridis——每条色带都是编译出的 kernel 在 [0, 1] ramp 上的求值结果。](../assets/examples/heatmap-colormaps.png)

??? example "完整源码：examples/gpu_heatmap.py（可直接运行）"

    ```python
    --8<-- "examples/gpu_heatmap.py"
    ```

## 从位置出发的粒子与视频 { #particles-and-video }

**它是什么。** `visualize_fields.py` 跑了一个极简 FEM 式仿真：圆盘上的
稳态热传导，离散为带抖动的极坐标点集。核区（r ≤ 0.2）固定 T = 1，
外缘（r ≥ 0.95）固定 T = 0，每个内部节点向邻居均值松弛——即
[拉普拉斯方程](https://baike.baidu.com/item/拉普拉斯方程)的
[Jacobi](https://en.wikipedia.org/wiki/Jacobi_method) 迭代，用 mean
reducer 的 message passing 加 Dirichlet 掩码表达：

$$
T_i \leftarrow \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} T_j \;\; \text{（内部节点）}, \qquad T = 1 \;\text{于}\; \Gamma_{\mathrm{core}}, \quad T = 0 \;\text{于}\; \Gamma_{\mathrm{rim}}
$$

邻居关系来自同一组点的 Delaunay 三角剖分，因此图与渲染网格严格一致。
每个节点渲染为一个圆盘，松弛过程由 `tg.visualize.save_video` 导出
为 GIF（host 缓存全部帧）或 MP4（逐帧流式编码）。

![粒子渲染：每个网格节点溅射为一个圆盘，按温度着色——热核发亮，同心离散环清晰可辨。](../assets/examples/fields_particles.png)

![动画：热方程的 Jacobi 松弛——初始的阶跃温度分布逐渐平滑为稳态的对数梯度。](../assets/examples/fields_diffusion.gif)

```python
--8<-- "examples/visualize_fields.py:core"
```

??? example "完整源码：examples/visualize_fields.py（可直接运行）"

    ```python
    --8<-- "examples/visualize_fields.py"
    ```

## 网格：从散点 Delaunay 或加载 OBJ { #simulation-meshes-load-shade-orbit }

**它是什么。** 渲染连续表面需要 faces 而不只是 positions——faces 有两种
来源。仿真只给出离散位置时，由 [Delaunay](https://en.wikipedia.org/wiki/Delaunay_triangulation)
三角剖分现算：`tg.visualize.delaunay` 把圆盘点集变成网格，每个三角形
按顶点值均值填充——稳态温度场由此呈现为经典的 FEM 云图。

![Delaunay 渲染：圆盘三角剖分后按顶点均值填充——同一稳态呈现为连续温度场。](../assets/examples/fields_delaunay.png)

```python
--8<-- "examples/visualize_fields.py:delaunay"
```

当三角剖分本身就是数据的一部分——求解器的表面网格、建模好的 3-D
模型——`visualize_mesh.py` 直接加载：用 `tg.visualize.load_obj` 从
[OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file) 文件读入一个
细分 2 级的 icosphere（162 顶点、320 面），把无重复的网格边变成无向
`Graph.from_csr` 关系，并让一个高斯包从某个顶点出发经 mean reducer
扩散。随后同一 mesh 在固定 camera 下渲染，外加一段 12 帧 azimuth 环绕
导出的 GIF。

`tg.visualize.mesh` 直接使用给定的 faces——`tg.visualize.delaunay`
是从点现算三角剖分，`mesh` 则渲染仿真已经产出的三角形。重叠关系由
[painter's algorithm](https://en.wikipedia.org/wiki/Painter%27s_algorithm)
解决：三角形按顶点深度均值从远到近填充，球面近侧干净地遮挡远侧。
固定的 `vmin`/`vmax` 让各帧之间的颜色可比。

同一脚本还把几何体导出给 [Blender](https://en.wikipedia.org/wiki/Blender_(software))
使用：`tg.visualize.export_ply` 写出 binary PLY，顶点同时携带扩散场的
顶点颜色与原始 `scalar_value` 属性；`tg.visualize.export_obj` 写出纯
positions/faces 文本格式，`load_obj` 可恒等读回。

![着色 mesh：icosphere 逐三角形填充，扩散后的高斯包在右侧边缘最亮——远侧被 painter's algorithm 完全遮挡。](../assets/examples/mesh_flat.png){ width="49%" }
![wireframe：同一 mesh 画成三角形边线——细分 2 级的完整拓扑，正面与背面尽收眼底。](../assets/examples/mesh_wireframe.png){ width="49%" }

![动画：绕 mesh 的 12 帧 azimuth 环绕——亮包转去远侧后被近侧半球遮挡消失。](../assets/examples/mesh_orbit.gif)

```python
--8<-- "examples/visualize_mesh.py:core"
```

??? example "完整源码：examples/visualize_mesh.py（可直接运行）"

    ```python
    --8<-- "examples/visualize_mesh.py"
    ```

## Gaussian splats 与体渲染 { #gaussian-splats-and-volumes }

同一组 3-D [Gaussian](https://en.wikipedia.org/wiki/Gaussian_splatting)，两种观察方式：
`tg.visualize.gaussians` 展示每个 Gaussian 的颜色与形状；
`tg.visualize.volume` 展示它们在空间中叠加形成的密度。

<div class="tg-render-comparison">
  <figure>
    <img src="../../../assets/examples/gaussians_splats.png" alt="蓝金双色的 Gaussian 圆环，椭圆足迹沿环的切向排列。" loading="lazy" width="512" height="512">
    <figcaption><strong>Gaussian splats</strong><br>直接合成带颜色的椭圆足迹。</figcaption>
  </figure>
  <figure>
    <img src="../../../assets/examples/gaussians_volume.png" alt="同一圆环的体素密度渲染，较高密度呈现较亮的颜色。" loading="lazy" width="512" height="512">
    <figcaption><strong>Volume</strong><br>采样密度网格，沿视线累积颜色与吸收。</figcaption>
  </figure>
</div>

这是可复现的合成场景，不是从照片训练的模型。圆环由 **2,200 个 Gaussian**
组成，两图共用相机与位置；左图的 RGB 预先加入了固定光照，右图按密度着色，
因此不是像素级等价对照。渲染与投影当前由 host NumPy 执行，不是 GPU 渲染 benchmark。

### 渲染 Gaussian

`positions`、`colors` 和 `scales` 均为 `(N, 3)` 数组；`rotations` 是
`(N, 4)` 的 `(w, x, y, z)` quaternion，`opacities` 是 `(N,)`。
本例的长轴沿圆环切向排列，`scales` 给出三个主轴的标准差。

```python
--8<-- "examples/visualize_gaussians.py:core"
```

`positions` 等场景数组由脚本中的 `torus_gaussians` 生成；
`args.size`、`output` 来自命令行，`background` 为深蓝灰色 RGB。
完整初始化见下方源码。两种输出均为 `Raster`，通过 `.save()` 写出 PNG。

??? example "同一场景改用体渲染"

    `splat_density` 是这个示例的辅助函数：把切向 Gaussian 的各向异性密度采样到
    `64 × 64 × 64` 网格，覆盖 `[-1, 1]³`。它不保留逐 Gaussian 的 RGB；
    `cmap` 重新按密度着色。相机与上方完全一致。

    ```python
    --8<-- "examples/visualize_gaussians.py:volume"
    ```

### 旋转查看

<figure class="tg-render-motion">
  <video controls loop muted playsinline preload="none" width="384" height="384"
         poster="../../../assets/examples/gaussians_splats.png"
         aria-label="Gaussian 圆环的相机环绕演示">
    <source src="../../../assets/examples/gaussians_orbit.mp4" type="video/mp4">
    <a href="../../../assets/examples/gaussians_orbit.mp4">下载圆环旋转视频</a>
  </video>
  <figcaption>固定仰角，相机环绕一周。36 帧、12 fps；点击播放，不自动播放。</figcaption>
</figure>

复现本页图片与视频（MP4 编码需要 ffmpeg）：

[`python examples/visualize_gaussians.py --count 2200 --size 512 --frames 36 --video-format mp4 --output output/examples`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/visualize_gaussians.py)

默认使用固定随机种子。省略 `--video-format mp4` 时导出 GIF；渲染不需要 ffmpeg。

??? example "完整源码：examples/visualize_gaussians.py"

    ```python
    --8<-- "examples/visualize_gaussians.py"
    ```

??? info "渲染原理：投影、遮挡与密度"

    一个 Gaussian 由中心、方向和三个轴向尺度定义。旋转矩阵为 `R` 时，其协方差为：

    $$
    \Sigma = R\,\mathrm{diag}(s^2)\,R^\top
    $$

    透视投影的 Jacobian 将其变为屏幕上的二维椭圆。令像素距椭圆中心的偏移为
    `δ`，投影协方差为 `Σ′`，不透明度为 `oᵢ`，则：

    $$
    \alpha_i = o_i \exp\!\left(-\tfrac12\delta^\top\Sigma'^{-1}\delta\right)
    $$

    Gaussian 按中心深度从近到远处理。`transmittance` 表示前面各层之后剩余的透射率；
    最终颜色包含未被遮挡的背景：

    $$
    T_i = \prod_{j<i}(1-\alpha_j),\qquad
    C = \sum_i T_i\alpha_i c_i + T_{\mathrm{end}}c_{\mathrm{background}}
    $$

    体渲染改为沿每条视线采样密度。每一步的不透明度由采样密度、`scale` 与步长决定：

    $$
    \alpha = 1-\exp(-\sigma\,\mathrm{scale}\,\Delta t)
    $$

??? info "从训练结果导入 Gaussian"

    `tg.visualize.load_ply` 返回位置、可选 faces 和附加属性。常见 3DGS PLY
    将颜色 DC 系数存在 `f_dc_*`，旋转存在 `rot_*`，并保存未经激活的 opacity 与 log-scale。
    对这种布局，转换关系为：

    $$
    \mathrm{RGB}=\mathrm{clip}(0.5+0.2821\,f_{\mathrm{dc}},0,1)
    $$

    $$
    o=\mathrm{sigmoid}(\mathrm{opacity}),\qquad s=\exp(\mathrm{scale})
    $$

    按编号顺序堆叠各属性后，传给 `tg.visualize.gaussians`。
    这是仅使用 DC 颜色的显示方式，不求值高阶、随视角变化的 spherical harmonics；
    不同导出工具的属性约定应先核对。
