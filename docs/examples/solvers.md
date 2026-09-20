# Linear solvers and control flow

Define an operator that maps one vector to another, then pass it to a solver.
Start with the three-step Poisson example below; dynamic neighborhoods,
iteration algorithms and nonlinear equations follow afterward.

- [`python examples/fem_poisson_minimal.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/fem_poisson_minimal.py)
- [`python examples/meshfree_linear_solve.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/meshfree_linear_solve.py)
- [`python examples/nonlinear_solve.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/nonlinear_solve.py)

??? note "Native solver tensors and Torch interoperability"

    A kernel call accepts either — `src`/`dst`/`edge` fields may be torch
    tensors directly, without a wrapper. The advanced solver drivers on this page are different:
    `linear_solve` / `nonlinear_solve` capture the iteration loop as
    device-side control flow (`gf_control.repeat` / `gf_control.while`), so
    their vectors must be `tg.Tensor`. Wrap torch storage at the boundary with
    `tg.from_torch(x)` when entering native expression capture. Only a value
    that still owns Torch storage supports zero-copy `.to_torch()`; a native
    solver result requires `.to_torch(copy=True)` and contiguous storage.
    Sharing storage does not bridge native gradients into PyTorch autograd.

<span id="matrix-free-linear-solves"></span>

## Solve Poisson with neighbor sums { #matrix-free-fem-operator-and-solver-loop }

Solve the [Poisson equation](https://en.wikipedia.org/wiki/Poisson%27s_equation)
on a line, with both endpoints fixed to zero and seven unknown interior nodes.

$$
-u''(x)=1,\qquad u(0)=u(1)=0
$$

Define only how to compute `A(u)`: each node takes twice its own value minus
its two neighbors, then divides by the grid spacing. **The solver owns the
iteration loop; no handwritten CG loop is needed.**

The complete computation follows. `linear_solve` is the repository helper
[examples/solvers.py](https://github.com/walkerchi/TIGA-lang/blob/main/examples/solvers.py),
not a built-in `tg` API.

```python
--8<-- "examples/fem_poisson_minimal.py:core"
```

Run [`python examples/fem_poisson_minimal.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/fem_poisson_minimal.py):

```text
[0.0546875, 0.09375, 0.1171875, 0.125, 0.1171875, 0.09375, 0.0546875]
max error: 0.00e+00
```

“Matrix-free” means no assembled stiffness matrix: `Graph.stencil` supplies
adjacency, `edge()` negates the neighbor value, and `node()` adds twice the
center. The graph still has CSR topology; there is no separate stiffness
matrix or per-edge stiffness array.

This example uses `tg.Tensor` because the current solver helper captures native
control flow, not because ordinary message passing requires replacing Torch tensors.

??? info "Why FEM, and how are boundaries handled?"

    Piecewise-linear P1 [finite elements](https://en.wikipedia.org/wiki/Finite_element_method)
    on a uniform 1-D grid produce the following equation. `h` is the spacing
    and `b` is the load vector for a unit source:

    $$
    (Au)_i=\frac{2u_i-u_{i-1}-u_{i+1}}{h},\qquad b_i=h
    $$

    The graph contains only interior nodes and omits out-of-grid neighbors.
    Those boundary values are zero, but the diagonal coefficient in `node()`
    remains `2`. **Do not replace it with the valid-neighbor count.**

    ![One-dimensional P1 stencil: interior nodes read their two neighbors; boundary values remain zero.](../assets/examples/fem-poisson.svg)

    The exact solution checks the nodal values:

    $$
    u_{\mathrm{exact}}(x)=\frac{x(1-x)}{2}
    $$

??? info "Stopping, fixed iterations and gradients"

    `tolerance=1e-6` checks the absolute residual norm; `max_iterations=32`
    bounds the loop. Reaching the limit does not guarantee convergence, so the
    example also checks the analytic-solution error.

    Fixed-count iteration and load gradients remain in the
    [advanced fem_poisson.py example](https://github.com/walkerchi/TIGA-lang/blob/main/examples/fem_poisson.py).
    See [linear solvers](../linear-solvers.md) for control-flow and differentiation contracts.

## Dynamic-radius matrix-free linear solve { #dynamic-radius-matrix-free-linear-solve }

**What it is.** Here there is no mesh file at all — the graph is generated from
data. Given N points in the plane, any two distinct points at distance at most r
become neighbors (a radius graph). On that graph the example solves
(m·I + L) u = b, where L is the graph Laplacian: for each point, the sum over
its neighbors of the difference between the point's own value and the
neighbor's value. The mass shift m = 1 keeps the operator positive definite,
and the right-hand side is the periodic pattern b_i = 1 + (i mod 3).

![Radius relation: distinct points at distance at most r become neighbors; the operator at point i is m·u_i plus the sum over neighbors of u_i − u_j](../assets/examples/meshfree-radius.svg)

$$
\big((mI + L)\,u\big)_i
\;=\;
m\,u_i \;+ \sum_{j\ne i \,:\, \lVert p_i - p_j \rVert \le r} (u_i - u_j)
\;=\;
b_i,
\qquad
b_i = 1 + (i \bmod 3)
$$

The example places the points on a line with spacing h and sets r = 1.01·h, so
each point's neighborhood is exactly its two immediate neighbors and the
generated relation is a path.

The solver and operator interfaces are unchanged from the FEM case: the
topology is produced by `Graph.radius`, and relation realization and reuse are
provider decisions below the MessagePassing UDF. CG captures its convergence
test and four carried states in one `gf_control.while`; no adjacency or sparse
matrix is written by the user.

```python
--8<-- "examples/meshfree_linear_solve.py:core"
```

??? example "Full source: examples/meshfree_linear_solve.py (runs as-is)"

    ```python
    --8<-- "examples/meshfree_linear_solve.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

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

## Shared solver sugar: `linear_solve` { #shared-solver-sugar-linear_solve }

**What it is.** Both examples above solve A·u = b by stationary iteration:
start from a guess and repeatedly apply A to shrink the residual r = b − A·u.
Three methods are provided, selected by one `method=` argument:

| `method=` | Algorithm | Operator contract |
|---|---|---|
| `"cg"` (default) | conjugate gradient | symmetric positive-definite |
| `"bicgstab"` | BiCGStab | nonsymmetric |
| `"richardson"` | damped fixed-step iteration | any, slowest |

Richardson takes the naive correction step u ← u + ω·r. Conjugate gradient
does better when A is symmetric positive-definite: it walks a sequence of
A-orthogonal search directions, carrying four states between iterations —
the solution u, the residual r, the search direction p, and the inner product
ρ = ⟨r, z⟩, where z is the (optionally preconditioned) residual. BiCGStab
drops the symmetry requirement at the price of two operator applications per
step, carrying additionally the shadow residual r̂ = r₀, the operator images
v = A·p and t = A·s, and the scalars ρ, α, ω.

![Stationary solver loop: operator, right-hand side and stopping contract enter a loop body captured once; states u, r, p and ρ are carried between iterations](../assets/examples/solver-loops.svg)

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

`solvers.py` composes these from `gf_control` primitives, with the loops
written as natural Python `for`/`while` under `@tg.jit`. Each example wraps
its operator construction in a local `solve()`; the solver itself is one
`linear_solve(operator, rhs, method="cg", …)` call — there is nothing else
named solve. The module is deliberately not part of the core package — it is
grammar sugar, not a primitive.

??? example "examples/solvers.py (full source)"

    ```python
    --8<-- "examples/solvers.py"
    ```

<span id="nonlinear-solves"></span>

## Nonlinear diffusion by Picard iteration { #nonlinear-diffusion-by-picard-iteration }

**What it is.** Same P1 grid, same stopping contracts — but the operator is
now nonlinear: the diffusion coefficient depends on the solution itself,
k(u) = 1 + u². The equation −∇·(k(u)∇u) = 1 is the steady state of heat
flow whose conductivity grows with temperature. Because the assembled form
F(u) = A(u)·u changes with the iterate, Krylov methods no longer apply
directly; the simplest fixed-point scheme is the damped Picard (nonlinear
Richardson) step, which only ever needs the operator image F(u) — still one
MessagePassing kernel, never an assembled matrix. The damping ω must keep
ω·λ_max of the Jacobian of F below 2; here λ_max ≈ 4·max(k)/h.

$$
F(u)_i \;=\; \frac{1}{h}\sum_{j} k_e\,(u_i - u_j)
\;+\; \frac{\delta_i}{h}\Bigl(1 + \tfrac{u_i^2}{2}\Bigr)\,u_i,
\qquad
k_e \;=\; 1 + \frac{u_i^2 + u_j^2}{2}
$$

$$
u \;\leftarrow\; u + \omega\,\bigl(b - F(u)\bigr)
$$

The per-edge conductance k_e averages k at the two endpoints of the current
iterate, so the edge UDF reads both `src.u` and `dst.u`; the two boundary
elements (δ_i = 1 at the end nodes, 0 inside) contribute a diagonal term
through the node UDF, which removes the constant null mode the pure
difference stencil would otherwise have. One
`nonlinear_solve(operator, b, omega=…)` drives both contracts:
`iterations=k` captures a fixed, differentiable `gf_control.repeat`, and
`tolerance=eps` captures a bounded device-side `gf_control.while` — one
flat loop either way, since control regions do not nest.
`load_gradient()` differentiates sum(u) with respect to the load through
the unrolled repeat; the count stays in the low single digits because the
unrolled VJP's compile time grows quickly with iterations, and the test
cross-checks the gradient against a finite difference of the same map.

```python
--8<-- "examples/nonlinear_solve.py:core"
```

??? example "Full source: examples/nonlinear_solve.py (runs as-is)"

    ```python
    --8<-- "examples/nonlinear_solve.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

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
