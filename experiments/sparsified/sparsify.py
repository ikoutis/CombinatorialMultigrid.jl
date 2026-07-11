#!/usr/bin/env python3
"""
Isolated spanner + adaptive spectral sparsifier for sparsified-CMG.

Self-contained numeric module (NO cmg-python imports -- only numpy / scipy /
numba / stdlib) so it ports 1:1 to Julia (spanner.jl / sparsify.jl). Edges are
`(u, v, w)` tuples with w > 0 a conductance and u < v.

Two spanners (pick with `spanner=`):
  * `"greedy"` (default) -- the Althofer greedy t-spanner in the resistance
    metric, line-for-line reference.py. Unambiguous for correctness, but
    O(m * Dijkstra) and pure Python: fine at validation scale (n <= ~1000),
    far too slow beyond.
  * `"baswana-sen"` -- the Baswana-Sen randomized (2k-1)-spanner, O(k*m)
    expected, **numba-jitted** (array-based, no heap/dict): the scalable /
    production spanner named in the guidelines. Same (u, v, w) output.

Other notes: `t` defaults to `log2(n)` (reference.py hardcodes t=9.0 for
n~400-960, which IS log2(n), NOT ln(n)); a scipy<->edge-list bridge
(`edges_of` / `slack_of` / `sdd_from_edges`) replaces reference.py's dense
helpers since the real CMG operators are sparse.

See experiments/sparsified/PYTHON-GUIDELINES.md and reference.py.
"""
import heapq
import math
import random

import numpy as np
import scipy.sparse as sp

try:                                        # numba njit for the Baswana-Sen
    from numba import njit as _numba_njit   # kernel (array-based, jits well);
    def _jit(fn):                           # pure-Python fallback if absent
        return _numba_njit(cache=True)(fn)
except Exception:                           # pragma: no cover
    def _jit(fn):
        return fn


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


# ---------------------------------------------------------------------------
# Baswana-Sen (2k-1)-spanner (numba-jitted): O(k*m) expected, no heap/dict.
# ---------------------------------------------------------------------------
def _bs_grow(a):
    b = np.empty(2 * a.shape[0], a.dtype)
    b[:a.shape[0]] = a
    return b


_bs_grow = _jit(_bs_grow)


def _baswana_sen_impl(n, indptr, indices, elen, econd, k, p, samp):
    """One full Baswana-Sen run. `elen` = edge lengths (1/conductance) and
    `econd` = conductances, both aligned with the symmetric CSR (indptr,
    indices). `samp[i, c]` is a uniform draw deciding if cluster-center c is
    sampled in phase i (sampled iff < p). Returns (su, sv, sw): spanner edges
    at original conductance (with duplicates; the caller dedups)."""
    center = np.arange(n).astype(np.int64)      # each vertex its own cluster
    new_center = np.empty(n, np.int64)
    best_len = np.empty(n)                       # per-cluster scratch (epoch-stamped)
    best_u = np.empty(n, np.int64)
    best_cond = np.empty(n)
    seen = np.zeros(n, np.int64)
    clusters = np.empty(n, np.int64)
    epoch = 0

    cap = indptr[n] + 16                         # ~2m; doubles on demand
    su = np.empty(cap, np.int64)
    sv = np.empty(cap, np.int64)
    sw = np.empty(cap)
    sc = 0

    for i in range(k - 1):                       # phases 1..k-1: form clusters
        for v in range(n):
            cv = center[v]
            if cv == -1:                         # already finalized
                new_center[v] = -1
                continue
            if samp[i, cv] < p:                  # v's cluster sampled -> v stays
                new_center[v] = cv
                continue
            # least-length edge to each incident (other, live) cluster
            epoch += 1
            ncl = 0
            for r in range(indptr[v], indptr[v + 1]):
                u = indices[r]
                cu = center[u]
                if cu == -1 or cu == cv:         # finalized / intra-cluster -> skip
                    continue
                L = elen[r]
                if seen[cu] != epoch:
                    seen[cu] = epoch
                    best_len[cu] = L
                    best_u[cu] = u
                    best_cond[cu] = econd[r]
                    clusters[ncl] = cu
                    ncl += 1
                elif L < best_len[cu]:
                    best_len[cu] = L
                    best_u[cu] = u
                    best_cond[cu] = econd[r]
            cstar = -1                           # nearest SAMPLED cluster
            lstar = np.inf
            for j in range(ncl):
                c = clusters[j]
                if samp[i, c] < p and best_len[c] < lstar:
                    lstar = best_len[c]
                    cstar = c
            while sc + ncl > cap:                # ensure room for up to ncl edges
                su = _bs_grow(su)
                sv = _bs_grow(sv)
                sw = _bs_grow(sw)
                cap = su.shape[0]
            if cstar == -1:                      # no sampled neighbor: finalize v,
                for j in range(ncl):             # edge to every incident cluster
                    c = clusters[j]
                    su[sc] = v
                    sv[sc] = best_u[c]
                    sw[sc] = best_cond[c]
                    sc += 1
                new_center[v] = -1
            else:                                # join nearest sampled cluster,
                su[sc] = v                       # edge to it + to strictly-closer ones
                sv[sc] = best_u[cstar]
                sw[sc] = best_cond[cstar]
                sc += 1
                new_center[v] = cstar
                for j in range(ncl):
                    c = clusters[j]
                    if best_len[c] < lstar:
                        su[sc] = v
                        sv[sc] = best_u[c]
                        sw[sc] = best_cond[c]
                        sc += 1
        for v in range(n):
            center[v] = new_center[v]

    for v in range(n):                           # final phase: connect survivors
        cv = center[v]
        if cv == -1:
            continue
        epoch += 1
        ncl = 0
        for r in range(indptr[v], indptr[v + 1]):
            u = indices[r]
            cu = center[u]
            if cu == -1 or cu == cv:
                continue
            L = elen[r]
            if seen[cu] != epoch:
                seen[cu] = epoch
                best_len[cu] = L
                best_u[cu] = u
                best_cond[cu] = econd[r]
                clusters[ncl] = cu
                ncl += 1
            elif L < best_len[cu]:
                best_len[cu] = L
                best_u[cu] = u
                best_cond[cu] = econd[r]
        while sc + ncl > cap:
            su = _bs_grow(su)
            sv = _bs_grow(sv)
            sw = _bs_grow(sw)
            cap = su.shape[0]
        for j in range(ncl):
            c = clusters[j]
            su[sc] = v
            sv[sc] = best_u[c]
            sw[sc] = best_cond[c]
            sc += 1

    return su[:sc].copy(), sv[:sc].copy(), sw[:sc].copy()


_baswana_sen_impl = _jit(_baswana_sen_impl)


def spanner_baswana_sen(n, edges, k=None, rng=None):
    """Baswana-Sen randomized (2k-1)-spanner in the resistance metric (edge
    length = 1/conductance). `k` defaults to `max(2, ceil(log2 n))` (stretch
    2k-1 ~ 2 log2 n; more k -> sparser, looser). O(k*m) expected, numba-jitted.
    Returns kept spanner edges `(u, v, w)` (u < v) at original conductance -- a
    connected subgraph for connected input, so a drop-in for greedy_spanner."""
    if not edges:
        return []
    if k is None:
        k = max(2, math.ceil(math.log2(n))) if n > 2 else 1
    u = np.fromiter((e[0] for e in edges), np.int64, len(edges))
    v = np.fromiter((e[1] for e in edges), np.int64, len(edges))
    w = np.fromiter((e[2] for e in edges), np.float64, len(edges))
    # symmetric CSR of conductances (each edge stored both ways, no duplicates)
    A = sp.csr_matrix((np.concatenate((w, w)),
                       (np.concatenate((u, v)), np.concatenate((v, u)))),
                      shape=(n, n))
    econd = A.data.astype(np.float64)
    p = float(n ** (-1.0 / k)) if k >= 1 else 1.0
    samp = (rng if rng is not None else np.random).random((max(k - 1, 0), n))
    su, sv, sw = _baswana_sen_impl(
        n, A.indptr.astype(np.int64), A.indices.astype(np.int64),
        1.0 / econd, econd, int(k), p, samp)
    kept = {}                                    # dedup undirected (same conductance)
    for a, b, ww in zip(su.tolist(), sv.tolist(), sw.tolist()):
        kept[(a, b) if a < b else (b, a)] = ww
    return [(a, b, ww) for (a, b), ww in kept.items()]


def spanner_bundle(n, edges, t, bundles, spanner="greedy", k=None, rng=None):
    """Peel `bundles` spanners: extract a spanner, remove its edges, repeat.
    Returns (bundle_edges, off_bundle_edges). More bundles -> lower effective
    stretch -> aggregation resumes harder at the same edge budget. `spanner` is
    "greedy" or "baswana-sen"."""
    def _one(rem_edges):
        if spanner == "greedy":
            return greedy_spanner(n, rem_edges, t)
        if spanner == "baswana-sen":
            return spanner_baswana_sen(n, rem_edges, k=k, rng=rng)
        raise ValueError(f"unknown spanner {spanner!r}; "
                         "use 'greedy' or 'baswana-sen'")

    key = lambda e: (min(e[0], e[1]), max(e[0], e[1]))
    kept = []
    rem = list(edges)
    for _ in range(bundles):
        s = _one(rem)
        kept += s
        sset = set(key(e) for e in s)
        rem = [e for e in rem if key(e) not in sset]
    return kept, rem


# ---------------------------------------------------------------------------
# adaptive sparsifier: keep the bundle at weight 1; sample off-bundle edges at
# a probability CHOSEN from the bundle size to hit a target keep-fraction.
# ---------------------------------------------------------------------------
def sparsify(n, edges, t=None, bundles=1, keep_frac=0.5, spanner="greedy",
             k=None, rng=None):
    """Return (kept_edges, p). `t=None` -> t = max(2.0, log2(n)).

    `spanner` selects the spanner: "greedy" (default, correctness) or
    "baswana-sen" (scalable, numba; `k` defaults to ceil(log2 n)).

    p = clamp((keep_frac*m - S)/(m - S), 0, 1); the bundle (S edges) is kept at
    weight 1, each off-bundle edge is kept w.p. p and reweighted by 1/p (an
    unbiased estimator of the off-bundle part). A sparse level (S ~ m) drives
    p -> 0 (keep just the bundle, no over-sparsifying); a dense/fill level
    (S << m) gives p ~ keep_frac (a gentle geometric drop).
    """
    if t is None:
        t = max(2.0, math.log2(n)) if n > 1 else 2.0
    rand = (rng.random if rng is not None else random.random)
    bundle, off = spanner_bundle(n, edges, t, bundles, spanner=spanner, k=k,
                                 rng=rng)
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
