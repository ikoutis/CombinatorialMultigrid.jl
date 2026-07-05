
# CombinatorialMultigrid.jl


This package implements the Combinatorial Multigrid Preconditioner *[1]*. The code handles input matrices that are symmetric diagonally dominant with negative off-diagonal entries, a class of matrices that includes graph Laplacians. This work has been supported by NSF grant CCF-#1149048.


In order to install the package simply do the following:
```
Pkg.add("CombinatorialMultigrid")
using CombinatorialMultigrid
```

## Quick start: the classic CMG preconditioner (default)

Given a Laplacian (or SDD) matrix `LX` — or an adjacency matrix `X` — build the preconditioner:

```julia
(pfunc, h) = cmg_preconditioner_lap(LX);   # from a Laplacian / SDD matrix
(pfunc, h) = cmg_preconditioner_adj(X);    # or directly from an adjacency matrix
```

`pfunc` applies one multigrid cycle: `x = pfunc(b)` returns an approximate solution of `LX * x = b`. It is a fixed **linear** operator, so it can serve as the preconditioner inside any standard preconditioned conjugate gradient implementation. The second output `h` is the hierarchy of graphs that is implicitly used in `pfunc`; it is exposed for its potential in other applications, and it can be passed back to the solver below to avoid rebuilding.

To solve a system directly, the package ships its own conjugate-gradient solver:

```julia
(x, stats) = cmg_solve(LX, b; cycle = :vcycle, tol = 1e-8);
# or, reusing the hierarchy built above:
(x, stats) = cmg_solve(h, b; cycle = :vcycle);

stats.iterations, stats.relres, stats.converged
```

With `cycle = :vcycle` this is plain PCG with the classic CMG preconditioner.

## The K-cycle solver

The package also provides a **K-cycle** variant of the preconditioner: the stationary repeats of the classic cycle are replaced by a few inner flexible-CG (Krylov) iterations at each coarse level, preconditioned by the recursive K-cycle below. On anisotropic and high-contrast problems this typically reduces both iterations and wall time; on regular 3D grids it often wins on time even at equal iteration counts, because a work-budget rule caps the coarse-level effort. The classic cycle remains the default everywhere; the K-cycle is selected explicitly.

Using it is one call — `cmg_solve` defaults to the K-cycle:

```julia
(x, stats) = cmg_solve(LX, b);            # K-cycle solve
(x, stats) = cmg_solve(h, b);             # same, reusing a prebuilt hierarchy
```

Knobs of `cmg_solve`:

| keyword | default | meaning |
|---|---|---|
| `cycle` | `:kcycle` | `:kcycle` or `:vcycle` (classic stationary cycle) |
| `tol` | `1e-8` | relative residual tolerance |
| `maxit` | `500` | maximum outer iterations |
| `theta` | `0.75` | work-budget cap for the inner iterations; `0.0` opts out into fixed local repeats |
| `inner_tol` | `0.25` | adaptive early-stopping for the inner FCG; `0.0` disables |
| `collect_stats` | `false` | record per-level K-cycle visit counts in `stats.level_visits` |

A single K-cycle application is also available from the preconditioner constructors via a knob:

```julia
(pfunc_k, h) = cmg_preconditioner_lap(LX; cycle = :kcycle);
x ≈ pfunc_k(b)   # one K-cycle sweep
```

> **Warning.** The K-cycle operator is *nonlinear* (it runs inner Krylov iterations), so `pfunc_k` must **not** be used as the preconditioner inside a standard PCG — drive it with `cmg_solve`, whose flexible-CG outer loop supports it. The default (`cycle = :vcycle`) preconditioner remains a fixed linear operator and is safe in any PCG.

`example/bench_kcycle.jl` compares the two cycles on uniform and anisotropic grids.

**Citations:**

[1] Ioannis Koutis, Gary L. Miller, David Tolliver, Combinatorial preconditioners and multilevel solvers for problems in computer vision and image processing, Computer Vision and Image Understanding, Volume 115, Issue 12, 2011, Pages 1638-1646, ISSN 1077-3142, https://doi.org/10.1016/j.cviu.2011.05.013.*
