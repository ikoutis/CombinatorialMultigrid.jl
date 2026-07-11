#!/usr/bin/env python3
"""
Spectral prototype gate for sparsified-CMG (see experiments/sparsified/README).

Question: does a spanner + uniform-1/4 sample give a spectral sparsifier M that
is a good preconditioner for L (bounded kappa(M^-1 L), few PCG iterations), on
the kind of graphs where CMG's aggregation stalls (dense / heterogeneous)?

We compare three preconditioners on each test graph:
  - L      : the graph itself (baseline, kappa = 1)
  - M_span : spanner (kept at weight 1) + off-spanner edges kept w.p. 1/4,
             reweighted x4 (unbiased)               <-- the proposed sparsifier
  - M_unif : NO spanner; every edge kept w.p. 1/4, reweighted x4
             (the control: uniform sampling without the spanner)

If the construction is sound, M_span has bounded kappa and low iters on ALL
graphs, while M_unif blows up on graphs with heterogeneous effective
resistances (a weak bridge it can drop) — showing the spanner is what makes
uniform sampling safe. Edge-count reduction is reported too: on dense levels the
sparsifier collapses to ~spanner size, which is the density control CMG needs.

Spanner here is the greedy (Althofer) spanner in the RESISTANCE metric
(edge length = 1/conductance): keep edge (u,v) unless the current spanner already
connects u,v with resistance <= t * r_uv. A t-spanner bounds every off-spanner
edge's effective resistance by t*r_uv, i.e. leverage score <= t, which is exactly
what makes uniform sampling concentrate. (Production would use Baswana-Sen for
near-linear time; greedy is simplest and unambiguous for the gate.)
"""
import numpy as np, scipy.sparse as sp, heapq, random
from scipy.linalg import eigh

def greedy_spanner(n, edges, t):
    """edges: list of (u,v,w) conductances. Returns kept spanner edges."""
    E = sorted(edges, key=lambda e: 1.0 / e[2])          # shortest (stiffest) first
    adj = [[] for _ in range(n)]                          # spanner adjacency (nbr, length)
    span = []
    def within(s, tgt, cap):
        d = {s: 0.0}; pq = [(0.0, s)]
        while pq:
            du, u = heapq.heappop(pq)
            if u == tgt: return du <= cap
            if du > cap: return False
            if du > d.get(u, float('inf')): continue
            for (v, l) in adj[u]:
                nd = du + l
                if nd < d.get(v, float('inf')):
                    d[v] = nd; heapq.heappush(pq, (nd, v))
        return False
    for (u, v, w) in E:
        L = 1.0 / w
        if not within(u, v, t * L):
            adj[u].append((v, L)); adj[v].append((u, L))
            span.append((u, v, w))
    return span

def sample_sparsify(n, edges, t, p=0.25, use_spanner=True):
    if use_spanner:
        span = greedy_spanner(n, edges, t)
        span_set = set((min(u, v), max(u, v)) for (u, v, w) in span)
        kept = list(span)                                 # spanner at original weight
        for (u, v, w) in edges:
            if (min(u, v), max(u, v)) in span_set: continue
            if random.random() < p: kept.append((u, v, w / p))
    else:
        kept = [(u, v, w / p) for (u, v, w) in edges if random.random() < p]
    return kept

def lap(n, edges):
    r, c, val = [], [], []
    deg = np.zeros(n)
    for (u, v, w) in edges:
        r += [u, v]; c += [v, u]; val += [-w, -w]; deg[u] += w; deg[v] += w
    return (sp.csr_matrix((val, (r, c)), shape=(n, n)) + sp.diags(deg)).toarray()

def grounded(M):                                          # pin last node -> SPD
    return M[:-1, :-1]

def kappa(L, M):
    """kappa(M^-1 L) on the grounded systems. inf if M is singular (a sampled
    sparsifier that disconnected the graph — the failure mode the spanner avoids)."""
    try:
        ev = eigh(grounded(L), grounded(M), eigvals_only=True)
    except np.linalg.LinAlgError:
        return float('inf')
    ev = ev[ev > 1e-9]
    return ev[-1] / ev[0]

def pcg_iters(L, M, tol=1e-8, maxit=500):
    from numpy.linalg import solve, LinAlgError
    Lg, Mg = grounded(L), grounded(M)
    n = Lg.shape[0]
    b = np.random.default_rng(0).standard_normal(n)
    try:
        x = np.zeros(n); r = b - Lg @ x; z = solve(Mg, r); p = z.copy()
    except LinAlgError:
        return maxit                                       # singular preconditioner
    rz = r @ z; bn = np.linalg.norm(b)
    for k in range(1, maxit + 1):
        Ap = Lg @ p; a = rz / (p @ Ap); x += a * p; r -= a * Ap
        if np.linalg.norm(r) / bn < tol: return k
        z = solve(Mg, r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
    return maxit

# ---- test graphs (small, so dense generalized-eig is cheap) ----
# All start from a random spanning tree so the input Laplacian is connected
# (grounded -> SPD); random edges are added on top.
def _tree(n, rng):
    return set((min(v, p), max(v, p))
               for v in range(1, n) for p in [rng.randrange(v)])

def expander(n, d=3, seed=0):
    """connected random d-regular-ish expander, unit conductances."""
    rng = random.Random(seed); es = _tree(n, rng)
    while len(es) < n * d // 2:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v: es.add((min(u, v), max(u, v)))
    return [(u, v, 1.0) for (u, v) in es]

def dense_blob(n, avgdeg=12, seed=1):
    """denser connected Erdos-Renyi -> mimics a fill-densified aggregate level."""
    rng = random.Random(seed); es = _tree(n, rng)
    while len(es) < n * avgdeg // 2:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v: es.add((min(u, v), max(u, v)))
    return [(u, v, 1.0) for (u, v) in es]

def two_dense_weak_bridge(n, avgdeg=16, seed=2):
    """two DENSE blobs joined by ONE weak (low-conductance) edge. Uniform-only
    keeps the dense halves connected but drops the single high-resistance bridge
    w.p. 3/4 -> disconnects there; the spanner always keeps the bridge (infinite
    stretch otherwise). Isolates the 'spanner protects high-resistance edges' effect."""
    h = n // 2
    e1 = dense_blob(h, avgdeg, seed)
    e2 = [(u + h, v + h, w) for (u, v, w) in dense_blob(n - h, avgdeg, seed + 100)]
    return e1 + e2 + [(0, h, 1e-3)]

random.seed(0); np.random.seed(0)
t = 9.0                                                   # stretch ~ log n for n~400
N = 400

# 1. Density sweep on a fill-like dense blob: sparsifier reduction grows with
#    density while kappa stays bounded (the spanner size is ~fixed in n, so the
#    denser the stalled level, the more it collapses).
print("=== density sweep (dense blob, spanner+uniform-1/4) ===")
print(f"{'avg_deg':>7} {'m':>7} -> {'m_sparse':>8} {'reduction':>9} {'avgdeg_out':>10} {'kappa':>7} {'iters':>6}")
for D in (8, 16, 32, 64):
    edges = dense_blob(N, D)
    L = lap(N, edges)
    M_edges = sample_sparsify(N, edges, t, use_spanner=True)
    M = lap(N, M_edges)
    print(f"{D:>7} {len(edges):>7} -> {len(M_edges):>8} {len(edges)/len(M_edges):>8.2f}x "
          f"{2*len(M_edges)/N:>10.1f} {kappa(L, M):>7.2f} {pcg_iters(L, M):>6}")

# 2. Spanner vs no-spanner where a weak bridge decides connectivity.
print("\n=== spanner vs no-spanner: two dense blobs + one weak bridge ===")
n, edges = N, two_dense_weak_bridge(N, 16)
L = lap(n, edges)
for label, span in (("M_span (spanner+unif)", True), ("M_unif (unif only)", False)):
    reps = [kappa(L, lap(n, sample_sparsify(n, edges, t, use_spanner=span))) for _ in range(5)]
    disc = sum(1 for k in reps if k == float('inf'))
    finite = [k for k in reps if k != float('inf')]
    kbar = sum(finite) / len(finite) if finite else float('inf')
    print(f"  {label:24} disconnected {disc}/5 draws; mean kappa(finite) = {kbar:.2f}")
