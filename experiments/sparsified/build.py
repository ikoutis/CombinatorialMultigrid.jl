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
                               max_inject=10, base=700, nnz_budget=5.0,
                               bundles_growth=False, rng=None):
    """Build a CMG hierarchy, optionally injecting a sparsifier level on stall.
    Returns (levels, is_sdd=False).

    The `sparsify_on_stall` knob (True/False) turns sparsification on or off.
    Recursion stops on BOTH criteria (as in pycmg's build, generalized):

      A. per-level EDGE stall -- a level is "productive" iff contraction reduces
         the edge count below `stall_ratio*m` (m = lower-tri off-diagonals). At
         or above it the level densified (contraction fill / expander) and did
         not get cheaper. With `sparsify_on_stall=True` a stall injects a
         same-size sparsifier and continues (up to `max_inject`); with False it
         terminates, as stock CMG does at a stall.
      B. cumulative OPERATOR-COMPLEXITY budget -- if the stored work summed over
         all levels exceeds `nnz_budget * nnz(input)` (default 5x, the pycmg
         value), stop. This bounds per-cycle work/memory and backstops
         sparsification that is too timid to control density (e.g. keep_frac
         near 1): rather than stacking `max_inject` barely-reducing levels, the
         build stops. Set `nnz_budget=inf` to disable this criterion.

    Other knobs mirror the guidelines: `keep_frac` (adaptive sparsifier target
    keep-fraction), `bundles` (spanner bundles), `t` (spanner stretch, default
    log2(n)), `base` (direct-solve threshold, 700 as in CMG). Operators are
    treated as SPD (Laplacian + slack); is_sdd is False.
    """
    A = A.tocsr()
    levels = []
    injected = 0
    flag_iterative = False
    initial_nnz = nnz_lower(A)                     # criterion B baseline
    cumulative_nnz = 0

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

        # --- criterion B: cumulative operator-complexity budget (both modes) ---
        # If total stored work over all levels exceeds nnz_budget x the input's
        # nnz, stop. Sparsification keeps this ~2 at keep_frac=0.5; it only trips
        # when sparsification is too timid (keep_frac near 1) or off on a
        # densifying input -- exactly when giving up is right.
        cumulative_nnz += level.nnz_lo
        if cumulative_nnz > nnz_budget * initial_nnz:
            level.is_last = True
            level.iterative = True
            levels.append(level)
            flag_iterative = True
            break

        # --- criterion A: per-level EDGE stall (not node count) ---
        # Densification is what CMG cannot escape: contraction fill keeps ~as
        # many edges even as nodes merge, so the coarse operator is no cheaper.
        # A level is productive iff contraction actually reduces the edge count.
        A_c = contract_coo(A, cI, nc)
        m = nnz_lower(A) - n                       # lower-tri off-diagonals = edges
        m_c = nnz_lower(A_c) - nc
        if m == 0 or m_c <= stall_ratio * m:       # edges dropped enough
            level.R = _restriction(cI, nc, n)
            levels.append(level)
            A = A_c
            continue

        # (d) STALL: contraction did not reduce edges. inject a sparsifier.
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
