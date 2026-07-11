#!/usr/bin/env python3
"""
Isolated spanner + adaptive spectral sparsifier for sparsified-CMG.

Self-contained numeric module (NO cmg-python imports) so it ports 1:1 to
Julia (spanner.jl / sparsify.jl). Edges are `(u, v, w)` tuples with w > 0 a
conductance and u < v -- identical to reference.py, so `greedy_spanner`,
`spanner_bundle`, and `sparsify` are line-for-line the reference algorithm.
The only additions over reference.py are:

  * `t` defaults to `log2(n)` (reference.py hardcodes t=9.0 for n~400-960,
    which IS log2(n) there, NOT ln(n) -- with ln(n) the greedy spanner keeps
    too many edges and the 0.98 reduction gate never lets the fork fire);
  * a scipy<->edge-list bridge (`edges_of`, `slack_of`, `sdd_from_edges`) that
    replaces reference.py's dense `lap_dense`/`edges_of`/`slack_of`, since the
    real CMG operators are sparse.

See experiments/sparsified/PYTHON-GUIDELINES.md and reference.py.
"""
import heapq
import math
import random

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# spanner (greedy, resistance metric: edge length = 1/conductance) + bundle
# ---------------------------------------------------------------------------
def greedy_spanner(n, edges, t):
    """Greedy (Althofer) t-spanner in the resistance metric.

    Keep edge (u, v) unless the current spanner already connects u, v with
    resistance <= t / w_uv. A t-spanner bounds every off-spanner edge's
    effective resistance by t * r_uv, i.e. leverage <= t, which is what makes
    uniform sampling concentrate. `edges`: list of (u, v, w) conductances.
    """
    adj = [[] for _ in range(n)]
    span = []

    def within(s, tgt, cap):
        # bounded-radius Dijkstra: is there an s->tgt spanner path of length <= cap?
        d = {s: 0.0}
        pq = [(0.0, s)]
        while pq:
            du, u = heapq.heappop(pq)
            if u == tgt:
                return du <= cap
            if du > cap:
                return False
            if du > d.get(u, 1e18):
                continue
            for (v, l) in adj[u]:
                nd = du + l
                if nd < d.get(v, 1e18):
                    d[v] = nd
                    heapq.heappush(pq, (nd, v))
        return False

    for (u, v, w) in sorted(edges, key=lambda e: 1.0 / e[2]):   # stiffest first
        L = 1.0 / w
        if not within(u, v, t * L):
            adj[u].append((v, L))
            adj[v].append((u, L))
            span.append((u, v, w))
    return span


def spanner_bundle(n, edges, t, bundles):
    """Peel `bundles` spanners: extract a spanner, remove its edges, repeat.
    Returns (bundle_edges, off_bundle_edges). More bundles -> lower effective
    stretch -> aggregation resumes harder at the same edge budget."""
    key = lambda e: (min(e[0], e[1]), max(e[0], e[1]))
    kept = []
    rem = list(edges)
    for _ in range(bundles):
        s = greedy_spanner(n, rem, t)
        kept += s
        sset = set(key(e) for e in s)
        rem = [e for e in rem if key(e) not in sset]
    return kept, rem


# ---------------------------------------------------------------------------
# adaptive sparsifier: keep the bundle at weight 1; sample off-bundle edges at
# a probability CHOSEN from the bundle size to hit a target keep-fraction.
# ---------------------------------------------------------------------------
def sparsify(n, edges, t=None, bundles=1, keep_frac=0.5, spanner="greedy",
             rng=None):
    """Return (kept_edges, p). `t=None` -> t = max(2.0, log2(n)).

    p = clamp((keep_frac*m - S)/(m - S), 0, 1); the bundle (S edges) is kept at
    weight 1, each off-bundle edge is kept w.p. p and reweighted by 1/p (an
    unbiased estimator of the off-bundle part). A sparse level (S ~ m) drives
    p -> 0 (keep just the bundle, no over-sparsifying); a dense/fill level
    (S << m) gives p ~ keep_frac (a gentle geometric drop).
    """
    if t is None:
        t = max(2.0, math.log2(n)) if n > 1 else 2.0
    rand = (rng.random if rng is not None else random.random)
    if spanner == "greedy":
        bundle, off = spanner_bundle(n, edges, t, bundles)
    else:
        raise ValueError(f"unknown spanner {spanner!r}; only 'greedy' so far "
                         "(Baswana-Sen is the scale/production option)")
    m, S = len(edges), len(bundle)
    if m <= S or not off:                       # bundle already ~ whole graph
        return list(edges), 0.0
    p = min(1.0, max(0.0, (keep_frac * m - S) / (m - S)))
    kept = list(bundle) + [(u, v, w / p) for (u, v, w) in off
                           if p > 0.0 and rand() < p]
    return kept, p


# ---------------------------------------------------------------------------
# scipy <-> edge-list bridge (SDD operators: L(graph) + diag(slack), slack >= 0)
# ---------------------------------------------------------------------------
def edges_of(A):
    """Off-diagonal edges (u, v, w) with u < v, w = -A[u, v] > 0, from a sparse
    SDD/Laplacian operator. Explicit zeros are skipped (they are not edges)."""
    T = sp.triu(sp.csr_matrix(A), k=1).tocoo()
    return [(int(i), int(j), float(-v))
            for i, j, v in zip(T.row, T.col, T.data) if v != 0.0]


def slack_of(A):
    """Per-row SDD slack: diag(A) - (sum of incident conductances). >= 0 for an
    SDD operator; the diagonal excess over the graph Laplacian."""
    A = sp.csr_matrix(A)
    d = A.diagonal()
    offdiag = A - sp.diags(d)
    off_rowsum = -np.asarray(offdiag.sum(axis=1)).ravel()   # sum of conductances
    return d - off_rowsum


def sdd_from_edges(n, edges, slack=0.0):
    """Build the SDD operator L(edges) + diag(slack) as canonical CSR float64.
    `slack` may be a scalar or a length-n vector (e.g. `slack_of(A)`)."""
    if edges:
        u = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
        v = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
        w = np.fromiter((e[2] for e in edges), dtype=np.float64, count=len(edges))
        rows = np.concatenate([u, v])
        cols = np.concatenate([v, u])
        vals = np.concatenate([-w, -w])
        W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        deg = np.asarray(-W.sum(axis=1)).ravel()            # sum of conductances
    else:
        W = sp.csr_matrix((n, n))
        deg = np.zeros(n)
    slack = np.full(n, float(slack)) if np.isscalar(slack) else np.asarray(slack)
    A = (W + sp.diags(deg + slack)).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    return A
