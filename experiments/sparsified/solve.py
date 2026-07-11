#!/usr/bin/env python3
"""
Unified solve entry for a (possibly sparsified) CMG hierarchy -- three cycle
branches, kept separate on purpose. Maps to the pending Julia
`sparsified_solve.jl`.

Pick the cycle with `cycle=`:

  "l-cycle"  (default) -- the legacy *stationary* cycle (pycmg
      method="legacy-cmg"; it is NOT a geometric V-cycle, see PYTHON-IMPL.md).
      RECOMMENDED for a sparsified hierarchy: it runs the same-size sparsify
      levels unchanged (identity transfer, one stationary apply) and is the
      fewest matvecs / fastest. Correct on a non-sparsified hierarchy too.

  "k-cycle" -- the stock Notay K-cycle (pycmg method="kcycle"). Use it for
      NON-sparsified hierarchies (build with sparsify_on_stall=False). It
      DEGRADES on same-size sparsify levels -- there its inner FCG minimizes the
      step over levels[level+1].A (the sparsifier), not the level's own
      operator -- so on a sparsified hierarchy use "ks-cycle" instead.

  "ks-cycle" -- the Ks-cycle (operator mode): the K-cycle ADAPTED to same-size
      sparsify levels. At a sparsify level its inner FCG minimizes over the
      level's OWN operator A (the correct residual equation, since the transfer
      is the identity) and runs `samesize_nu` inner iterations, using the
      sub-hierarchy as an accelerated inner solver; at normal levels it is the
      ordinary K-cycle. This is a ROBUSTNESS fallback for sparsified
      hierarchies: it fixes the stock K-cycle's degradation and reliably
      converges, but it does ~2-3x more total work than the L-cycle (its inner
      FCG nests across stacked sparsify levels), so it is a fallback, not the
      default. Keep it for inputs where the L-cycle underperforms.

`samesize_nu` (default 8) is the number of inner FCG iterations at each sparsify
level in "ks-cycle"; higher -> fewer outer iterations, more work per iteration.
Operators are SPD (Laplacian + slack), so is_sdd is False. Returns
(x, iterations, relres, converged).
"""
from pycmg._precond import Preconditioner

from kscycle import fcg_solve_ks

_L = ("l-cycle", "l", "legacy", "legacy-cmg")
_K = ("k-cycle", "k", "kcycle")
_KS = ("ks-cycle", "ks", "kscycle")


def sparsified_solve(levels, b, cycle="l-cycle", tol=1e-8, maxiter=500,
                     theta=0.75, inner_tol=0.25, samesize_nu=8):
    """Solve levels[0].A x = b with the chosen cycle (see module docstring)."""
    c = cycle.lower()
    if c in _L:
        return Preconditioner(levels, False).solve(
            b, method="legacy-cmg", tol=tol, maxiter=maxiter,
            theta=theta, inner_tol=inner_tol)
    if c in _K:
        return Preconditioner(levels, False).solve(
            b, method="kcycle", tol=tol, maxiter=maxiter,
            theta=theta, inner_tol=inner_tol)
    if c in _KS:
        return fcg_solve_ks(levels, b, tol=tol, maxiter=maxiter, theta=theta,
                            inner_tol=inner_tol, mode="operator",
                            samesize_nu=samesize_nu)
    raise ValueError(
        f"unknown cycle {cycle!r}; use 'l-cycle', 'k-cycle', or 'ks-cycle'")


if __name__ == "__main__":
    import random

    import numpy as np

    from build import build_sparsified_hierarchy
    from graphs import blob_chain, spd_operator

    random.seed(1)
    np.random.seed(1)
    n, edges = blob_chain(6, 150, avgdeg=40, seed=7, wbridge=1e-2)
    A = spd_operator(n, edges, slack=1e-8)
    b = np.random.default_rng(0).standard_normal(n)
    b -= b.mean()

    def tr(x):
        return np.linalg.norm(A @ x - b) / np.linalg.norm(b)

    print("Sparsified hierarchy (sparsify_on_stall=True), three cycle branches:")
    on, _ = build_sparsified_hierarchy(A, sparsify_on_stall=True)
    for cyc in ("l-cycle", "k-cycle", "ks-cycle"):
        x, its, _, ok = sparsified_solve(on, b, cycle=cyc, tol=1e-9, maxiter=1000)
        note = {"l-cycle": "recommended", "k-cycle": "degrades on sparsify levels",
                "ks-cycle": "robustness fallback"}[cyc]
        print(f"  {cyc:9}: its={its:3d} conv={ok} true_res={tr(x):.1e}   ({note})")

    print("\nNon-sparsified hierarchy (sparsify_on_stall=False):")
    off, _ = build_sparsified_hierarchy(A, sparsify_on_stall=False)
    for cyc in ("l-cycle", "k-cycle"):
        x, its, _, ok = sparsified_solve(off, b, cycle=cyc, tol=1e-9, maxiter=1000)
        print(f"  {cyc:9}: its={its:3d} conv={ok} true_res={tr(x):.1e}")
