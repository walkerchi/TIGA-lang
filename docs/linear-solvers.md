# Matrix-free solvers and implicit differentiation

Global solvers should consume GraphForge kernels without being flattened into
an unstructured list of MessagePassing calls. The local operator, loop state,
convergence test, reductions, communication, and differentiation rule all carry
structure that the compiler can use.

## The first executable slice

`gf.linalg.LinearOperator` wraps a compiler-visible `matvec`; it does not own a
dense or sparse matrix. The callback may invoke Tensor code or MessagePassing
over any `Graph` realization.

```python
class StiffnessApply(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.value * src.u

A = gf.linalg.LinearOperator(
    (n, n),
    matvec=lambda u: StiffnessApply()(
        graph=mesh_relation,
        src={"u": u},
        dst={},
        edge={"value": element_coefficients},
    ),
    parameters=(element_coefficients,),
    symmetric=True,
)

x = gf.linalg.cg(A, b, iterations=k)
```

Both Richardson and CG are deliberately fixed-count. CG carries
`(x, r, p, rᴴz)` as four typed SSA values in one `gf_control.repeat`; iteration
count does not grow forward IR, and CPU lowering allocates two reusable buffers
per carried value. Single-state CUDA repeat already reuses two buffers and one
prepared operator launch per iteration; multi-state CUDA scheduling remains an
explicit open performance item rather than silently returning to Python.
CPU control lowering also hoists rank-zero body SSA (including dot-product
reductions and dependent scalar algebra) ahead of element loops and stores each
once per iteration; this prevents a scalar reduction referenced by three CG
vector updates from being recomputed once per vector element.
Ordinary downstream Tensor algebra remains Pythonic (`loss = x.sum()`): the
current single-output CPU executable ABI automatically stages a nested control
result, while a future multi-output program ABI may fuse the loop epilogue.
Changing coordinates, coefficients, a radius relation, or a halo snapshot
changes the operator's normal guards and versions—the solver API does not
change.

The complete runnable example is [examples/fem_poisson.py](https://github.com/walkerchi/graphforge/blob/main/examples/fem_poisson.py).
It applies a one-dimensional P1 Poisson stiffness operator through
MessagePassing without assembling a sparse matrix and compares against the
closed-form solution.

### FEM topology is not always a radius graph

The example's mesh connectivity is frozen while its stiffness coefficients may
change. That is the common moving-mesh case: geometry is dynamic, but element
incidence is not. Remeshing, fracture, contact, or adaptive refinement can
replace the logical relation snapshot; the `LinearOperator` call remains the
same. A `Graph.radius` operator is more natural for meshfree/particle methods
than for ordinary conforming FEM.

Pairwise MessagePassing covers edge-decomposable scalar P1 operators. General
FEM additionally needs a retained element relation:

```text
element -> gather local dofs -> quadrature/Jacobian/local tensor algebra
        -> local residual or JVP -> constrained scatter/reduction to dofs
```

That requires element-to-dof hyperrelations, local tensor-valued state,
quadrature/layout metadata, and explicit essential-constraint semantics. It
should lower through the same gather/UDF/reducer machinery, but must not be
faked as pairwise edges when doing so loses the element tensor structure.

## Fixed CG exists; convergence-driven CG needs bounded while

`gf.linalg.cg(..., iterations=k)` is an executable matrix-free primitive, not a
Python iteration helper. It accepts an optional compiler-visible
preconditioner. It deliberately does not accept a tolerance yet: exact early
convergence can otherwise make a later fixed CG step divide by zero, and a host
residual check would insert a synchronization into every iteration.

| Primitive | Why the compiler must see it | Status |
|---|---|---|
| `LinearOperator.apply` / `adjoint_apply` | preserve matrix-free graph/stencil application and differentiable parameters | executable frontend |
| element gather/local tensor/scatter | retain higher-order and mixed FEM structure beyond pairwise P1 edges | design exists as hyperrelation; executable solver slice pending |
| boundary/constraint projection | enforce essential constraints consistently in primal, adjoint, and distributed ownership | pending |
| dot, norm, scalar comparison | expose reductions and their distributed collective boundary | `dot`/`vector_norm` Tensor algebra exists; comparison and solver-level collective semantics pending |
| multi-value loop-carried SSA | keep CG vectors/scalars in one bounded region with reusable buffers | executable frontend + CPU LLVM lowering |
| bounded `gf_control.while` | device-side convergence with `max_iterations` as a mandatory safety/resource bound | pending |
| preconditioner operator | permit Jacobi/block/multigrid or provider-library choice without changing CG semantics | callable/`LinearOperator` frontend; schedule/performance gates pending |
| multi-state CUDA loop plan | reuse all carried buffers and choose host command graph vs persistent/cooperative execution | pending |
| loop memory/checkpoint plan | choose saved states, recomputation, or hierarchy spill for reverse mode | fixed-repeat correctness fallback exists; structured reverse loop pending |

The intended control form is structurally similar to:

```mlir
%x, %r, %p, %rr = gf_control.while
    max_iterations = 1000
    (%x0, %r0, %p0, %rr0) {
  cond(%x, %r, %p, %rr):
    %continue = arith.cmpf ogt, %rr, %tolerance_squared
    gf_control.condition %continue
  body(%x, %r, %p, %rr):
    // A(p), dot products, vector updates, and optional collectives
    gf_control.yield %x1, %r1, %p1, %rr1
}
```

`max_iterations` is part of specialization/resource planning even when the
condition exits earlier. On distributed graphs, dot products become explicit
collective tasks; the planner may select pipelined CG and overlap reductions
with halo/interior work only when dependency analysis proves it legal.

## Backward has two distinct contracts

Differentiating through iterations and differentiating the converged equation
are not interchangeable.

### Algorithmic or unrolled VJP

The derivative is for exactly the executed finite iteration algorithm. Current
multi-state `gf_control.repeat` reuses existing Tensor/MessagePassing VJP rules, but its
correctness path specializes the reverse body per iteration. Backward IR and
compile work therefore grow with iteration count. This is useful for testing
and truncated optimization, but it is not a performance-complete solver VJP.

A control-autodiff pass must instead emit a reverse `gf_control.repeat/while`
plus a compiler-planned tape, checkpoint, recomputation, or hierarchy spill.

### Implicit VJP

For a converged system, the implicit rule is:

```text
primal:       A(theta) x = b
adjoint:      A(theta)^T lambda = x_bar
rhs VJP:      b_bar = lambda
parameter VJP: theta_bar = -lambda^T (dA(theta)/dtheta) x
```

The user should not write this backward. `LinearOperator.parameters` declares
which captured fields belong to the operator; GraphForge can apply normal
MessagePassing VJP to the scalar contraction `λᵀ A(θ)x`. The forward solution
and adjoint may use different tolerances or preconditioners, but the API and IR
must record those choices. Implicit VJP is valid only under a declared
convergence/residual contract and must fail closed when the adjoint operator is
missing or the solve did not converge.

## JIT, fusion, and variants

There is no separate user-facing `autofuse` or `jit` module. Calling the solver
creates the same lazy program boundary as calling a kernel. Whole-program passes
can fuse pointwise updates with operator epilogues, reuse relation snapshots,
plan ping-pong buffers, and overlap halo/collective tasks. `@gf.program` remains
an optional export/prewarm boundary, not an optimization hint.

Likewise, `gf.variant` should not be a numerical primitive. A compiled variant
is a guarded implementation selected from graph statistics, shapes, dtype,
target, provider, memory budget, and distributed topology. Kernel objects expose
the selected `last_variant` and the guarded `variants` collection; they may
eventually accept a narrow policy constraint, but solver code should not branch
on named kernels.

## Performance gates

A solver claim must report more than kernel latency:

- residual and solution error under the same stopping rule;
- iteration count and preconditioner;
- operator, dot/norm, collective, and exposed-communication time;
- cold compile separately from warm solve;
- peak solver state, tape/checkpoint bytes, and hierarchy traffic;
- backward comparison against both an unrolled hand-written implementation and
  a matched implicit/adjoint implementation.

Until bounded while, structured reverse lowering, implicit VJP and matched
artifacts exist, the FEM/CG example is a compiler vertical slice—not a
CG/PETSc/FEniCS performance claim.
