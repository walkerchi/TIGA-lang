# 线性求解器与控制流 { #linear-solvers-and-control-flow }

先定义一个把向量映射到向量的算子，再把它交给求解器。
从下面的三步 Poisson 示例开始；更复杂的动态邻域、迭代算法与非线性方程放在后面。

- [`python examples/fem_poisson_minimal.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/fem_poisson_minimal.py)
- [`python examples/meshfree_linear_solve.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/meshfree_linear_solve.py)
- [`python examples/nonlinear_solve.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/nonlinear_solve.py)

??? note "求解器的原生 Tensor 与 Torch 互操作"

    kernel 调用两种都接受——`src`/`dst`/`edge` 字段可以直接传 torch
    tensor，不需要包装。本页的高级原生
    求解驱动器不同：`linear_solve` / `nonlinear_solve` 会把迭代循环捕获为
    设备端控制流（`gf_control.repeat` / `gf_control.while`），因此向量
    必须是 `tg.Tensor`。在边界处用 `tg.from_torch(x)` 把 torch 存储零
    拷贝包进来；只有仍持有 Torch 存储的值才能用 `.to_torch()` 零拷贝返回。
    原生求解结果需要 `.to_torch(copy=True)`，且要求连续存储。
    共享存储本身不会把原生梯度图接入 PyTorch autograd。

<span id="matrix-free-linear-solves"></span>

## 用邻居求和求解 Poisson 方程 { #matrix-free-fem-operator-and-solver-loop }

目标很简单：在一条线上求解 [Poisson 方程](https://en.wikipedia.org/wiki/Poisson%27s_equation)，
两端固定为零，中间放 7 个未知节点。

$$
-u''(x)=1,\qquad u(0)=u(1)=0
$$

只需要定义“给定 `u`，如何算出 `A(u)`”。每个节点读取左右邻居，
自身取两倍，再除以网格间距；**迭代循环交给求解器，不需要手写 CG**。

下面是完整计算过程。`linear_solve` 来自仓库中的
[examples/solvers.py](https://github.com/walkerchi/TIGA-lang/blob/main/examples/solvers.py)，不是 `tg` 的内置 API。

```python
--8<-- "examples/fem_poisson_minimal.py:core"
```

运行 [`python examples/fem_poisson_minimal.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/fem_poisson_minimal.py)，得到：

```text
[0.0546875, 0.09375, 0.1171875, 0.125, 0.1171875, 0.09375, 0.0546875]
max error: 0.00e+00
```

这里的“无矩阵”指不构造刚度矩阵：`Graph.stencil` 给出邻接，
`edge()` 取负的邻居值，`node()` 加上自身的两倍。
图仍有 CSR 拓扑，省去的是单独的刚度矩阵与逐边刚度数组。

本例使用 `tg.Tensor`，因为当前辅助求解器捕获的是原生控制流；
这不是普通消息传递必须更换 Torch Tensor 的要求。

??? info "为什么这是 FEM？边界怎么处理？"

    一维均匀网格上的分段线性 P1 [有限元](https://baike.baidu.com/item/有限元法)
    给出以下离散方程。`h` 是节点间距，`b` 是单位载荷对应的右端项：

    $$
    (Au)_i=\frac{2u_i-u_{i-1}-u_{i+1}}{h},\qquad b_i=h
    $$

    图只存内部节点，越界邻居被省略。被省略的边界值本来就是零，
    但 `node()` 中的对角系数仍是 `2`，**不能改成有效邻居数**。

    ![一维 P1 stencil：内部节点读取左右邻居，边界固定为零。](../assets/examples/fem-poisson.svg)

    解析解用于检查节点结果：

    $$
    u_{\mathrm{exact}}(x)=\frac{x(1-x)}{2}
    $$

??? info "停止条件、固定迭代与求导"

    `tolerance=1e-6` 检查绝对残差范数，`max_iterations=32` 给出迭代上界；
    达到上界不等于保证收敛，因此示例额外检查解析解误差。

    固定次数迭代和对载荷求导保留在
    [进阶示例 fem_poisson.py](https://github.com/walkerchi/TIGA-lang/blob/main/examples/fem_poisson.py)。
    控制流与梯度契约见[线性求解器](../linear-solvers.md)。

## 动态半径图上的无矩阵线性求解 { #dynamic-radius-matrix-free-linear-solve }

**是什么。** 这里完全没有网格文件——图是从数据生成的。给定平面上
N 个点，距离不超过截断半径 r 的两个不同点成为邻居（半径图）。在该图上，
示例求解 (m·I + L) u = b，其中 L 是图拉普拉斯算子：对每个点，把它
自身的值与每个邻居的值之差求和。质量偏移 m = 1 保证算子正定，
右端项是周期模式 b_i = 1 + (i mod 3)。

![半径关系：平面上的两个不同点距离不超过截断半径 r 时成为邻居；点 i 处的算子是 m·u_i 加上对所有邻居的 u_i − u_j 求和](../assets/examples/meshfree-radius.svg)

$$
\big((mI + L)\,u\big)_i
\;=\;
m\,u_i \;+ \sum_{j\ne i \,:\, \lVert p_i - p_j \rVert \le r} (u_i - u_j)
\;=\;
b_i,
\qquad
b_i = 1 + (i \bmod 3)
$$

示例把点放在间距为 h 的直线上，并设 r = 1.01·h，因此每个点的邻域
恰好是相邻的两个点，生成的关系是一条路径。

求解器和算子接口与 FEM 情形完全相同：拓扑由 `Graph.radius` 产生，
关系的实现与复用是 MessagePassing UDF 之下 provider 层的决策。
CG 在一个 `gf_control.while` 中捕获收敛判断和四个随迭代携带的状态；
用户不需要写任何邻接结构或[稀疏矩阵](https://baike.baidu.com/item/稀疏矩阵)。

```python
--8<-- "examples/meshfree_linear_solve.py:core"
```

??? example "完整源码：examples/meshfree_linear_solve.py（可直接运行）"

    ```python
    --8<-- "examples/meshfree_linear_solve.py"
    ```

??? info "本机编译产物（Ryzen 7 255 · RTX 5070 Ti 实测）"

    === "CPU"

        ```text
        ### ShiftedRadiusLaplacian
        backend: cpu:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: materialize-fixed-radius-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        variant cache: hits=0, misses=3
        ```

    === "CUDA"

        ```text
        ### ShiftedRadiusLaplacian
        backend: cuda:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: materialize-fixed-radius-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        variant cache: hits=0, misses=3
        ```

<span id="shared-solver-entry"></span>

## 共享求解器语法糖：`linear_solve` { #shared-solver-sugar-linear_solve }

**是什么。** 上面两个示例都用定常迭代求解 A·u = b：从一个初始猜测出发，
反复应用 A 来缩小残差 r = b − A·u。提供三种方法，由一个 `method=`
参数选择：

| `method=` | 算法 | 算子契约 |
|---|---|---|
| `"cg"`（默认） | 共轭梯度 | 对称正定 |
| `"bicgstab"` | [BiCGStab](https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method) | 非对称 |
| `"richardson"` | 阻尼定步长迭代 | 任意算子，最慢 |

Richardson 采用朴素的校正步 u ← u + ω·r。当 A 对称正定时，共轭梯度
做得更好：沿一系列 A-正交搜索方向前进，在迭代之间携带四个状态
——解 u、残差 r、搜索方向 p，以及内积 ρ = ⟨r, z⟩，其中 z 是经过
可选预条件处理的残差。BiCGStab 以每步两次算子应用为代价放弃了
对称性要求，额外携带影子残差 r̂ = r₀、算子像 v = A·p 和 t = A·s，
以及标量 ρ、α、ω。

![定常求解器循环：算子、右端项与停止契约进入只捕获一次的循环体；状态 u、r、p 与 ρ 在迭代之间传递](../assets/examples/solver-loops.svg)

$$
u_{k+1} \;=\; u_k + \omega\,(b - A u_k)
\qquad\text{(Richardson)}
$$

$$
\alpha_k = \frac{\langle r_k, z_k \rangle}{p_k^{\top} A\, p_k},
\qquad
u_{k+1} = u_k + \alpha_k p_k,
\qquad
r_{k+1} = r_k - \alpha_k A p_k,
\qquad
p_{k+1} = z_{k+1} + \frac{\langle r_{k+1}, z_{k+1} \rangle}{\langle r_k, z_k \rangle}\, p_k
\quad\text{(CG)}
$$

$$
\rho_k = \langle \hat r, r_{k-1} \rangle,
\qquad
p_k = r_{k-1} + \underbrace{\tfrac{\rho_k}{\rho_{k-1}} \tfrac{\alpha_{k-1}}{\omega_{k-1}}}_{\beta}\,(p_{k-1} - \omega_{k-1} v_{k-1}),
\qquad
\alpha_k = \frac{\rho_k}{\langle \hat r, A p_k \rangle}
$$

$$
s = r_{k-1} - \alpha_k A p_k,
\qquad
\omega_k = \frac{\langle A s,\, s \rangle}{\langle A s,\, A s \rangle},
\qquad
u_k = u_{k-1} + \alpha_k p_k + \omega_k s,
\qquad
r_k = s - \omega_k A s
\quad\text{(BiCGStab)}
$$

`solvers.py` 用 `gf_control` 原语组合这些方法，循环写成 `@tg.jit` 下
普通的 Python `for`/`while`。每个示例把算子构造包在一个局部
`solve()` 里；求解器本身就是一次
`linear_solve(operator, rhs, method="cg", …)` 调用——没有别的叫
solve 的东西。该模块刻意不属于核心包——是语法糖，不是原语。

??? example "examples/solvers.py（完整源码）"

    ```python
    --8<-- "examples/solvers.py"
    ```

<span id="nonlinear-solves"></span>

## Picard 迭代的非线性扩散 { #nonlinear-diffusion-by-picard-iteration }

**是什么。** 同样的 P1 网格、同样的停止契约——但算子现在是非线性的：
扩散系数依赖于解本身，k(u) = 1 + u²。方程 −∇·(k(u)∇u) = 1 描述热导率
随温度升高的稳态热传导。由于组装形式 F(u) = A(u)·u 随迭代变化，
Krylov 方法不再直接适用；最简单的[不动点迭代](https://baike.baidu.com/item/不动点迭代)格式是
阻尼 Picard（非线性 Richardson）步，只需要算子像 F(u)——仍然是
一个 MessagePassing kernel，永远不是组装好的矩阵。阻尼 ω 必须让
F 的 [Jacobi 矩阵](https://baike.baidu.com/item/雅可比矩阵)满足 ω·λ_max < 2；
这里 λ_max ≈ 4·max(k)/h。

$$
F(u)_i \;=\; \frac{1}{h}\sum_{j} k_e\,(u_i - u_j)
\;+\; \frac{\delta_i}{h}\Bigl(1 + \tfrac{u_i^2}{2}\Bigr)\,u_i,
\qquad
k_e \;=\; 1 + \frac{u_i^2 + u_j^2}{2}
$$

$$
u \;\leftarrow\; u + \omega\,\bigl(b - F(u)\bigr)
$$

每条边的 conductance k_e 对当前迭代两个端点的 k 取平均，因此 edge UDF
同时读取 `src.u` 和 `dst.u`；两个边界单元（端点处 δ_i = 1，内部为 0）
通过 node UDF 贡献一个对角项，消掉了纯差分 stencil 否则会有的常数零空间。
一个 `nonlinear_solve(operator, b, omega=…)` 驱动两种契约：
`iterations=k` 捕获固定次数、可微分的 `gf_control.repeat`，
`tolerance=eps` 捕获有界的设备端 `gf_control.while`——无论哪种都是
单个平坦循环，因为控制区域不支持嵌套。`load_gradient()` 沿展开的
repeat 对 sum(u) 关于载荷求导；迭代次数保持在个位数，因为展开 VJP
的编译时间随迭代数增长很快，测试用同一个映射的有限差分对梯度做了
交叉验证。

```python
--8<-- "examples/nonlinear_solve.py:core"
```

??? example "完整源码：examples/nonlinear_solve.py（可直接运行）"

    ```python
    --8<-- "examples/nonlinear_solve.py"
    ```

??? info "本机编译产物（Ryzen 7 255 · RTX 5070 Ti 实测）"

    === "CPU"

        ```text
        ### NonlinearDiffusionApply
        backend: cpu:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        variant cache: hits=0, misses=3

        ### NonlinearDiffusionApply
        backend: cpu:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        variant cache: hits=0, misses=1
        ```
