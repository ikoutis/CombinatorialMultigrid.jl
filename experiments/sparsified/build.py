#!/usr/bin/env python3
"""
Stall-forked CMG hierarchy build (sparsify-on-stall).

This is the ONLY file that imports cmg-python internals. It mirrors
`pycmg._hierarchy.build_hierarchy`'s loop verbatim and forks ONLY the stall
branch: where the production build gives up (a level that will not coarsen),
this injects a SAME-SIZE spanner+uniform sparsifier level and continues on the
sparser operator, so aggregation resumes. The resulting `list[HierarchyLevel]`
is consumed by the real solver unchanged -- a sparsify level has `cI = arange(n)`,
`R = identity`, so `_cycles.vcycle`/`kcycle` (`R @ r`, `R.T @ z`) flow through
with no special case, and `_inner_fcg` picks up the sparsifier as
`levels[level+1].A`.

Requires `pip install -e /path/to/CMG-python` (editable pycmg on sys.path).
"""
import math

import numpy as np

from pycmg._hierarchy import (HierarchyLevel, contract_coo, nnz_lower,
                              _inv_diag, _restriction)
from pycmg._ldl import ldl_factorize
from pycmg._steiner import steiner_group_arrays

from sparsify import edges_of, slack_of, sdd_from_edges, sparsify


def _set_repeats(levels, flag_iterative):
    """Repeat counts, identical to build_hierarchy (lines 164-179):
    repeat[k] = max(floor(nnz_k/nnz_{k+1} - 1), 1); terminal level counted as
    nnz(L factor) when the base is a direct LDL solve."""
    L = len(levels)
    if L >= 2:
        for k in range(L - 2):
            ratio = levels[k].nnz_lo / levels[k + 1].nnz_lo
            levels[k].repeat = max(int(math.floor(ratio - 1.0)), 1)
        k = L - 2
        last = levels[-1]
        if not flag_iterative and last.ldl is not None:
            next_nnz = last.ldl.nnz_L
        else:
            next_nnz = last.nnz_lo
        ratio = levels[k].nnz_lo / next_nnz
        levels[k].repeat = max(int(math.floor(ratio - 1.0)), 1)


def build_sparsified_hierarchy(A, sparsify_on_stall=True, keep_frac=0.5,
                               bundles=1, t=None, stall_ratio=0.9,
                               max_inject=10, base=700, bundles_growth=False,
                               rng=None):
    """Build a CMG hierarchy that injects a sparsifier level on aggregation
    stall. Returns (levels, is_sdd=False).

    Parameters mirror the guideline knobs: `keep_frac` (target keep-fraction of
    the adaptive sparsifier), `bundles` (spanner bundles), `t` (spanner stretch,
    default log2(n) per level), `stall_ratio` (a level is "productive" iff
    nc <= stall_ratio*n; above it is a stall), `max_inject` (hard cap on
    injections -> termination), `base` (direct-solve threshold, 700 as in CMG).
    Operators are treated as SPD (Laplacian + slack); is_sdd is False.
    """
    A = A.tocsr()
    levels = []
    injected = 0
    flag_iterative = False

    while True:
        n = A.shape[0]

        # (a) small matrix -> terminal direct LDL  (identical to build_hierarchy)
        if n < base:
            levels.append(HierarchyLevel(
                A=A, inv_diag=_inv_diag(A), nnz_lo=nnz_lower(A),
                is_last=True, iterative=False,
                ldl=ldl_factorize(A[:n - 1, :n - 1]) if n > 1 else None,
            ))
            break

        cI, nc = steiner_group_arrays(A.indptr, A.indices, A.data, n)
        level = HierarchyLevel(A=A, inv_diag=_inv_diag(A), nnz_lo=nnz_lower(A),
                               cluster_indices=cI, num_clusters=nc)

        # (b) full contraction -> Jacobi base  (identical)
        if nc == 1:
            level.is_last = True
            level.iterative = True
            levels.append(level)
            flag_iterative = True
            break

        # (c) productive coarsening -> normal level  (fork replaces the
        # production `nc >= n-1` stall guard with the aggressive stall_ratio)
        if nc <= stall_ratio * n:
            level.R = _restriction(cI, nc, n)
            levels.append(level)
            A = contract_coo(A, cI, nc)
            continue

        # (d) STALL. inject a sparsifier level instead of giving up.
        if sparsify_on_stall and injected < max_inject:
            e0 = edges_of(A)
            B = bundles + injected if bundles_growth else bundles
            sp_e, p = sparsify(n, e0, t=t, bundles=B, keep_frac=keep_frac,
                               rng=rng)
            if len(sp_e) < 0.98 * len(e0):        # sparsifier actually reduced
                A_sp = sdd_from_edges(n, sp_e, slack_of(A))
                levels.append(HierarchyLevel(
                    A=A, inv_diag=_inv_diag(A), nnz_lo=nnz_lower(A),
                    is_last=False, iterative=False,
                    cluster_indices=np.arange(n, dtype=np.uint32),
                    num_clusters=n, R=_restriction(np.arange(n), n, n),
                ))
                A = A_sp
                injected += 1
                continue

        # (e) no fix (flag off / cap hit / couldn't reduce) -> terminal
        # iterative base (a shallow, poor preconditioner -- the stalled CMG)
        level.is_last = True
        level.iterative = True
        levels.append(level)
        flag_iterative = True
        break

    _set_repeats(levels, flag_iterative)
    return levels, False
