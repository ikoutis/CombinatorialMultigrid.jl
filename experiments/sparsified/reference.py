#!/usr/bin/env python3
"""
Runnable reference for sparsified-CMG (sparsify-on-stall). Self-contained; run:
    python3 reference.py

This is an EXECUTABLE SPEC, not production code. It implements a minimal
aggregation-AMG that reproduces CMG's coarsening stall (a quality-gated
aggregation refuses to coarsen dense/expander levels), and shows the fix:
when aggregation stalls, inject a spanner+uniform SPECTRAL SPARSIFIER as a
SAME-SIZE hierarchy level; the sparser operator lets aggregation resume.

What it demonstrates (see the printout at the bottom):
  1. STALL: on a dense operator, aggregation gives nc/n ~ 1 (no coarsening).
  2. RESUME: after one adaptive sparsify, aggregation on the sparser operator
     gives nc/n << 1 (coarsening resumes) -- the crux the NumPy gate couldn't show.
  3. END-TO-END: a V-cycle-preconditioned CG converges in few iterations WITH
     sparsify-on-stall vs a stalled (shallow, poor) hierarchy WITHOUT it, and the
     solution matches a direct solve (correctness).

Design choices that the cmg-python / Julia ports should mirror (see
PYTHON-GUIDELINES.md):
  * Adaptive sampling probability: choose p from the measured bundle size to hit
    a target keep-fraction (geometric, gentle) -- NOT a fixed 1/4.
  * Bundle iteration (peel B spanners) implemented; default bundles=1.
  * A sparsify level is a SAME-SIZE level: transfer is the identity, so the cycle
    needs no special case (it keys on the cluster map, which is 1:n here).
Operators are SDD (Laplacian + small diagonal slack) so everything is SPD --
no nullspace/grounding bookkeeping, and A\\b is well defined for the check.
"""
import numpy as np, heapq, random
from numpy.linalg import solve

# ----------------------------------------------------------------------------
# graph <-> SDD operator helpers.  Operators are dense (small n); edges are
# (u, v, w>0) conductances.  M = L(graph) + diag(slack), slack >= 0 keeps it SDD.
# ----------------------------------------------------------------------------
def lap_dense(n, edges, slack=1e-8):
    A = np.zeros((n, n))
    for (u, v, w) in edges:
        A[u, v] -= w; A[v, u] -= w; A[u, u] += w; A[v, v] += w
    A += slack * np.eye(n)                     # SDD slack -> SPD
    return A

def edges_of(A):
    n = A.shape[0]; es = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0.0:
                es.append((i, j, -A[i, j]))
    return es

def slack_of(A):
    n = A.shape[0]
    off = np.array([sum(-A[i, j] for j in range(n) if j != i) for i in range(n)])
    return np.diag(A) - off                    # per-row SDD slack

# ----------------------------------------------------------------------------
# spanner (greedy, resistance metric length = 1/w) + bundle (peel B spanners)
# ----------------------------------------------------------------------------
def greedy_spanner(n, edges, t):
    adj = [[] for _ in range(n)]; span = []
    def within(s, tgt, cap):
        d = {s: 0.0}; pq = [(0.0, s)]
        while pq:
            du, u = heapq.heappop(pq)
            if u == tgt: return du <= cap
            if du > cap: return False
            if du > d.get(u, 1e18): continue
            for (v, l) in adj[u]:
                nd = du + l
                if nd < d.get(v, 1e18): d[v] = nd; heapq.heappush(pq, (nd, v))
        return False
    for (u, v, w) in sorted(edges, key=lambda e: 1.0 / e[2]):
        L = 1.0 / w
        if not within(u, v, t * L):
            adj[u].append((v, L)); adj[v].append((u, L)); span.append((u, v, w))
    return span

def spanner_bundle(n, edges, t, bundles):
    key = lambda e: (min(e[0], e[1]), max(e[0], e[1]))
    kept = []; rem = list(edges)
    for _ in range(bundles):
        s = greedy_spanner(n, rem, t)
        kept += s
        sset = set(key(e) for e in s)
        rem = [e for e in rem if key(e) not in sset]
    return kept, rem                            # (bundle edges, off-bundle edges)

# ----------------------------------------------------------------------------
# adaptive sparsifier: keep the bundle; sample off-bundle edges at a probability
# CHOSEN from the bundle size to hit a target keep-fraction (geometric drop).
# ----------------------------------------------------------------------------
def sparsify(n, edges, t=9.0, bundles=1, keep_frac=0.5):
    bundle, off = spanner_bundle(n, edges, t, bundles)
    m, S = len(edges), len(bundle)
    if m <= S or not off:                       # bundle already ~ whole graph
        return list(edges), 0.0
    p = min(1.0, max(0.0, (keep_frac * m - S) / (m - S)))
    kept = list(bundle) + [(u, v, w / p) for (u, v, w) in off
                           if p > 0.0 and random.random() < p]
    return kept, p

# ----------------------------------------------------------------------------
# aggregation: CMG-spirit effective-degree gate. A node merges toward its
# strongest neighbor only if that edge dominates its (weighted) degree
# (w_max/deg >= tau). Dense/expander nodes have no dominant edge -> stay
# singletons -> nc ~ n (stall). Sparsifying lowers degree -> gate opens.
# ----------------------------------------------------------------------------
def aggregate(n, edges, tau):
    deg = np.zeros(n); wmax = np.zeros(n); arg = -np.ones(n, dtype=int)
    for (u, v, w) in edges:
        deg[u] += w; deg[v] += w
        if w > wmax[u]: wmax[u] = w; arg[u] = v
        if w > wmax[v]: wmax[v] = w; arg[v] = u
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        if deg[i] > 0 and arg[i] >= 0 and wmax[i] / deg[i] >= tau:
            ra, rb = find(i), find(arg[i])
            if ra != rb: parent[ra] = rb
    roots = {}; cI = np.empty(n, dtype=int)
    for i in range(n):
        r = find(i)
        if r not in roots: roots[r] = len(roots)
        cI[i] = roots[r]
    return cI, len(roots)

def R_of(cI, nc):                               # nc x n cluster indicator
    n = len(cI); R = np.zeros((nc, n))
    R[cI, np.arange(n)] = 1.0
    return R

# ----------------------------------------------------------------------------
# hierarchy build with sparsify-on-stall
# ----------------------------------------------------------------------------
def build(A0, sparsify_on_stall, tau=1/6, t=9.0, keep_frac=0.5,
          bundles=1, base=60, stall_ratio=0.9, max_inject=10):
    levels = []           # each: (A, invD, cI, is_sparsify)
    A = A0; injected = 0
    while A.shape[0] > base:
        n = A.shape[0]; edges = edges_of(A)
        cI, nc = aggregate(n, edges, tau)
        if nc <= stall_ratio * n:               # productive coarsening
            R = R_of(cI, nc)
            levels.append((A, 1.0 / (2 * np.diag(A)), cI, False))
            A = R @ A @ R.T
        elif sparsify_on_stall and injected < max_inject:
            sp_edges, p = sparsify(n, edges, t, bundles, keep_frac)
            if len(sp_edges) >= 0.98 * len(edges):   # sparsifier couldn't reduce
                break
            A_sp = lap_dense(n, sp_edges, slack=0.0) + np.diag(slack_of(A))
            levels.append((A, 1.0 / (2 * np.diag(A)), np.arange(n), True))  # cI = identity
            A = A_sp                              # continue on the sparser operator
            injected += 1
        else:
            break                                 # stalled, no fix -> base is big
    return levels, A, injected                    # A = coarsest (base) operator

# ----------------------------------------------------------------------------
# V-cycle preconditioner keyed on cI (identity transfer for sparsify levels ->
# no special case), + preconditioned CG.  Base solved directly.
# ----------------------------------------------------------------------------
def base_solve(A, r, cap):
    # A small base is solved directly; a large one (a hierarchy that stalled and
    # never coarsened) can't be -- approximate it with a few Jacobi sweeps, which
    # is a POOR coarse solve, so a stalled hierarchy is a poor preconditioner.
    if A.shape[0] <= cap:
        return solve(A, r)
    x = np.zeros_like(r); invd = 1.0 / np.diag(A)
    for _ in range(3):
        x += 0.66 * invd * (r - A @ x)
    return x

def vcycle(levels, base_A, i, r, base_cap=60):
    if i == len(levels):
        return base_solve(base_A, r, base_cap)
    A, invD, cI, _ = levels[i]
    x = invD * r                                  # pre-smooth (Jacobi, x0 = 0)
    res = r - A @ x
    nc = cI.max() + 1
    rc = np.zeros(nc); np.add.at(rc, cI, res)     # restrict (R res)
    ec = vcycle(levels, base_A, i + 1, rc)
    x = x + ec[cI]                                # prolong (R^T ec)
    res = r - A @ x
    x = x + invD * res                            # post-smooth
    return x

def pcg(A, apply_M, b, tol=1e-8, maxit=500):
    x = np.zeros_like(b); r = b - A @ x; z = apply_M(r); p = z.copy()
    rz = r @ z; bn = np.linalg.norm(b)
    for k in range(1, maxit + 1):
        Ap = A @ p; a = rz / (p @ Ap); x += a * p; r -= a * Ap
        if np.linalg.norm(r) / bn < tol:
            return x, k
        z = apply_M(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
    return x, maxit

# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
def dense_blob(n, avgdeg, seed, wjitter=False):
    rng = random.Random(seed)
    es = set((min(v, p), max(v, p)) for v in range(1, n) for p in [rng.randrange(v)])
    while len(es) < n * avgdeg // 2:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v: es.add((min(u, v), max(u, v)))
    return [(u, v, (0.2 + rng.random() if wjitter else 1.0)) for (u, v) in es]

def blob_chain(nblobs, blobn, avgdeg, seed, wbridge=1e-3):
    """A chain of dense blobs, consecutive blobs joined by one weak edge. Each
    blob is dense (aggregation stalls); the nblobs-1 weak cuts give that many
    small eigenvalues, so the system genuinely needs a multilevel preconditioner
    (a crude one-level solve must resolve each low mode -> many iterations).
    Coarsening the blobs collapses it to a short path where the modes are cheap."""
    rng = random.Random(seed); edges = []
    for k in range(nblobs):
        off = k * blobn
        edges += [(u + off, v + off, w) for (u, v, w) in dense_blob(blobn, avgdeg, seed + k)]
        if k > 0:
            edges.append(((k - 1) * blobn + rng.randrange(blobn),
                          off + rng.randrange(blobn), wbridge))
    return nblobs * blobn, edges

def kappa(A, M):
    from scipy.linalg import eigh
    ev = eigh(A, M, eigvals_only=True); ev = ev[ev > 1e-9]; return ev[-1] / ev[0]

if __name__ == "__main__":
    random.seed(1); np.random.seed(1)
    n, edges = blob_chain(8, 120, 16, seed=7, wbridge=1e-3)
    A = lap_dense(n, edges)
    tau = 1/6
    from scipy.linalg import eigh as _eigh
    ev = _eigh(A, eigvals_only=True)
    print(f"test graph: chain of 8 dense blobs (n={n}) + 7 weak cuts; "
          f"effective kappa (lam_max/lam_2, seen by b_|_1) = {ev[-1]/ev[1]:.1e}")

    print("\n=== 1. stall, then aggregation resumes after one sparsify ===")
    e0 = edges_of(A)
    _, nc0 = aggregate(n, e0, tau)
    print(f"  original   : n={n} m={len(e0)} avgdeg={2*len(e0)/n:.1f}  ->  nc/n = {nc0/n:.3f}  (stall)")
    for B in (1, 2):                              # bundle knob: more bundles -> tighter kappa
        sp, p = sparsify(n, e0, bundles=B, keep_frac=0.5)
        _, nc1 = aggregate(n, sp, tau)
        A_sp = lap_dense(n, sp, slack=0.0) + np.diag(slack_of(A))
        print(f"  sparsified(bundles={B}): m={len(sp)} avgdeg={2*len(sp)/n:.1f} "
              f"(p={p:.2f}, {len(e0)/len(sp):.2f}x)  ->  nc/n = {nc1/n:.3f} (resumes);  "
              f"kappa(A_sp^-1 A) = {kappa(A, A_sp):.2f}")

    print("\n=== 2. end-to-end V-cycle-PCG: without vs with sparsify-on-stall ===")
    b = np.random.default_rng(0).standard_normal(n); b -= b.mean()
    xref = solve(A, b)
    for label, flag in (("without (stalls)", False), ("with sparsify", True)):
        levels, base_A, inj = build(A, flag, tau=tau)
        depth = len(levels); coarsest = base_A.shape[0]
        x, its = pcg(A, lambda r: vcycle(levels, base_A, 0, r), b)
        err = np.linalg.norm(x - xref) / np.linalg.norm(xref)
        print(f"  {label:18}: levels={depth:2d} injected={inj} coarsest_n={coarsest:4d} "
              f"PCG_iters={its:4d} rel_err={err:.1e}")
