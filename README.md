# CombinatorialMultigrid.jl

This package implements the Combinatorial Multigrid (CMG) preconditioner and
solver *[1]*. It handles symmetric diagonally dominant matrices with
non-positive off-diagonal entries (SDDM) — a class that includes graph
Laplacians and the SDD systems that arise from them. This work has been
supported by NSF grant CCF-#1149048.

## Install

```julia
using Pkg
Pkg.add("CombinatorialMultigrid")
using CombinatorialMultigrid
```

## Quick start

To solve `A x = b` for a Laplacian / SDDM matrix `A`, just call `cmg_solve`:

```julia
(x, stats) = cmg_solve(A, b)

stats.iterations   # outer flexible-CG iterations
stats.relres       # final relative residual ‖A x − b‖ / ‖b‖
stats.converged    # whether tol was reached
```

**The default is the K-cycle with degree-1/2 elimination** (`cmg-k-elim`) — in
our experience the fastest, most reliable configuration across a wide range of
problems. You can change either choice with the keywords below.

## `cmg_solve` — the solver

```julia
(x, stats) = cmg_solve(A, b; kwargs...)   # from a matrix; builds the hierarchy
(x, stats) = cmg_solve(H, b; kwargs...)   # reuse a prebuilt hierarchy (see below)
```

| keyword | default | meaning |
|---|---|---|
| `eliminate` | `true` | first exactly factor out degree-1 and degree-2 vertices (partial Cholesky / Schur complement), then solve the smaller "core". Matrix input only. |
| `cycle` | `:kcycle` | `:kcycle` (Krylov-accelerated) or `:legacy` (the classic CMG cycle). |
| `tol` | `1e-8` | relative residual tolerance. |
| `maxit` | `500` | maximum outer iterations. |
| `theta` | `0.75` | K-cycle work-budget cap; `0.0` opts out into fixed local repeats. |
| `inner_tol` | `0.25` | adaptive early-stopping for the inner iterations; `0.0` disables. |
| `collect_stats` | `false` | record per-level K-cycle visit counts in `stats.level_visits`. |

### Dropping elimination when there's no tree structure

`eliminate = true` exactly removes degree-1 and degree-2 vertices before CMG is
built. This is a **large win when the graph has tree-like / low-degree
structure** — paths, near-trees, meshes with low-degree boundaries, anything
with many chains or leaves — because those vertices are factored out for free.

The elimination is **adaptive**: a cheap allocation-free scan runs first, and
when the graph has (almost) no degree-1/2 vertices — dense graphs, expanders,
cliques, interior-dominated grids — the elimination machinery is skipped
automatically, so the default costs only that single pass. To skip even the
scan:

```julia
(x, stats) = cmg_solve(A, b; eliminate = false)
```

The result is the same solution either way — elimination is exact and the
adaptive skip only avoids work — so this is purely a performance choice.

### Legacy CMG

The classic CMG cycle *[1]* is available as `cycle = :legacy`:

```julia
(x, stats) = cmg_solve(A, b; cycle = :legacy)                    # legacy cycle + elimination
(x, stats) = cmg_solve(A, b; cycle = :legacy, eliminate = false) # the original CMG, unchanged
```

Unlike the K-cycle, the legacy cycle is a **fixed linear operator**, so it — and
only it — can be handed to your own PCG (see the next section). `:vcycle` is
accepted as a deprecated alias for `:legacy`; note the legacy cycle is a
*stationary* iteration, not a true geometric V-cycle, so we spell it `:legacy`.

## Using CMG as a preconditioner in your own solver

If you want the preconditioner operator itself (for example to drive
`Laplacians.pcgSolver` or another Krylov method), use `cmg_preconditioner_lap`
(or `cmg_preconditioner_adj` to pass an adjacency matrix directly):

```julia
(pfunc, H) = cmg_preconditioner_lap(A)          # from a Laplacian / SDDM matrix
(pfunc, H) = cmg_preconditioner_adj(Adj)        # from an adjacency matrix

x = pfunc(b)                                     # apply one cycle: x ≈ A \ b
```

`cmg_preconditioner_lap` defaults to `cycle = :legacy`, which returns a **linear**
`pfunc` that is safe inside any standard PCG. The second return value `H` is the
graph hierarchy; pass it back to `cmg_solve(H, b; ...)` to avoid rebuilding.

```julia
(pfunc_k, H) = cmg_preconditioner_lap(A; cycle = :kcycle)   # one K-cycle sweep
```

> **Warning.** The K-cycle operator is *nonlinear* (it runs inner Krylov
> iterations), so `pfunc_k` must **not** be used as the preconditioner inside a
> standard PCG. Drive it with `cmg_solve`, whose flexible-CG outer loop supports
> it. Only the `:legacy` preconditioner is a fixed linear operator.

With `eliminate = true` the second return value is an `EliminatedHierarchy`
(rather than a `Vector{HierarchyLevel}`); pass it to `cmg_solve`, which handles
the exact forward/back-substitution around the reduced solve.

## Input matrices

Inputs must be symmetric with non-positive off-diagonals (Laplacians and SDDM
matrices). Purely Laplacian systems are singular; supply a right-hand side in the
range (e.g. mean-centered, `b .-= sum(b)/length(b)`), and the returned solution
is fixed to the corresponding null-space reference.

## Notes

- The preconditioner/solver closures reuse internal workspace across calls and
  are **not** reentrant or thread-safe; use one per thread.
- `example/bench_kcycle.jl` compares the cycles on uniform and anisotropic grids.
- This package and the Python sibling **pycmg** are a verified mirror pair; the
  feature ledger (and the cmg++ extension roadmap) is `docs/PARITY.md` in the
  CMG-python repository.

## Citation

[1] Ioannis Koutis, Gary L. Miller, David Tolliver, *Combinatorial
preconditioners and multilevel solvers for problems in computer vision and image
processing*, Computer Vision and Image Understanding, Volume 115, Issue 12,
2011, Pages 1638–1646. https://doi.org/10.1016/j.cviu.2011.05.013
