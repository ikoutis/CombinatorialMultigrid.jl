#!/usr/bin/env python3
"""
SPD test-graph generators for the sparsified-CMG validation.

Ported from reference.py (`dense_blob`, `blob_chain`) and prototype.py
(`two_dense_weak_bridge` -> `dense_blob_pair_bridge`), retuned DENSER: the real
`pycmg` aggregation coarsens more aggressively than reference.py's stylized
gate, so a blob must be avgdeg >= ~32 (not 16) to actually stall it (real
steiner nc/n: 0.375@deg8, 0.88@deg16, 0.958@deg32). Operators are SDD
(graph Laplacian + a positive diagonal slack) so everything is SPD -- no
nullspace/grounding, and A^-1 b is well defined for the correctness check.
"""
import random

from sparsify import sdd_from_edges


def dense_blob(n, avgdeg=32, seed=0, wjitter=False):
    """Connected Erdos-Renyi blob (spanning tree + random edges to avgdeg),
    unit conductances by default. Dense enough (avgdeg>=32) that the real CMG
    aggregation stalls on it."""
    rng = random.Random(seed)
    es = set((min(v, p), max(v, p)) for v in range(1, n) for p in [rng.randrange(v)])
    while len(es) < n * avgdeg // 2:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            es.add((min(u, v), max(u, v)))
    return [(u, v, (0.2 + rng.random() if wjitter else 1.0)) for (u, v) in es]


def blob_chain(nblobs, blobn, avgdeg=40, seed=7, wbridge=1e-2):
    """A chain of dense blobs, consecutive blobs joined by one weak edge. Each
    blob is dense (aggregation stalls); the nblobs-1 weak cuts give that many
    small eigenvalues, so the system genuinely needs a multilevel
    preconditioner. Returns (n, edges)."""
    rng = random.Random(seed)
    edges = []
    for k in range(nblobs):
        off = k * blobn
        edges += [(u + off, v + off, w)
                  for (u, v, w) in dense_blob(blobn, avgdeg, seed + k)]
        if k > 0:
            edges.append(((k - 1) * blobn + rng.randrange(blobn),
                          off + rng.randrange(blobn), wbridge))
    return nblobs * blobn, edges


def dense_blob_pair_bridge(n, avgdeg=32, seed=2, wbridge=1e-3):
    """Two dense blobs joined by ONE weak (low-conductance) edge. Uniform-only
    sampling drops the single high-resistance bridge w.p. ~(1-p) -> disconnects;
    the spanner always keeps it. Isolates 'the spanner protects high-resistance
    edges'. Returns (n, edges)."""
    h = n // 2
    e1 = dense_blob(h, avgdeg, seed)
    e2 = [(u + h, v + h, w) for (u, v, w) in dense_blob(n - h, avgdeg, seed + 100)]
    return n, e1 + e2 + [(0, h, wbridge)]


def spd_operator(n, edges, slack=1e-8):
    """SPD operator L(edges) + slack*I (constant positive diagonal slack)."""
    return sdd_from_edges(n, edges, slack)
