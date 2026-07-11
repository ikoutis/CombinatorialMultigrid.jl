#!/usr/bin/env python3
"""
Ks-cycle: a K-cycle that adapts to the injected same-size sparsify level.

Diagnosis (see PYTHON-IMPL.md): the stock K-cycle degrades on a sparsified
hierarchy because its inner FCG minimizes the step length over
``levels[level+1].A``. For a normal level that is the Galerkin coarse operator
R A Rᵀ (consistent with the transfer); for a same-size sparsify level the
transfer is the identity, so the consistent operator is the level's own ``A``,
but ``levels[level+1].A`` is the *sparsifier* (κ≈4–6 away) -- the inner FCG
optimizes against the wrong operator.

The Ks-cycle special-cases a same-size level (``num_clusters == n`` and not
last) in one of two ways:

  * mode="operator" (default): keep the inner FCG but minimize over the level's
    OWN operator ``A = levels[level].A`` (the correct residual equation, since
    the transfer is the identity), preconditioned by the sub-hierarchy -- which
    approximates the sparsifier, hence A, inverse. With a few inner iterations
    (``samesize_nu``) this uses the sparsifier hierarchy as an *accelerated
    inner solver* for A and can beat the L-cycle. Never catastrophic.
  * mode="stationary": skip the inner FCG and take ONE recursive sub-hierarchy
    apply as the correction -- the stationary L-cycle treatment at the sparsify
    level, K-cycle inner-FCG below. Recovers the L-cycle when the coarse levels
    are benign, but mixing a linear top with nonlinear coarse levels can
    destabilize the outer FCG (see PYTHON-IMPL.md).

Self-contained: imports the UNCHANGED pycmg cycle primitives; nothing in
``src/pycmg`` is modified. Run the comparison with ``python3 kscycle.py``.
"""
import numpy as np

from pycmg._cycles import _base_case, compute_kcycle_repeats


def _is_samesize(H):
    """A sparsify level: identity transfer (num_clusters == n) and not the base."""
    return (not H.is_last) and H.num_clusters == H.A.shape[0]


def kscycle(levels, b, level, krepeat, inner_tol, mode="operator",
            samesize_nu=4, level_visits=None):
    """One Ks-cycle sweep (pre-smooth, adaptive coarse solve, post-smooth)."""
    H = levels[level]
    if level_visits is not None:
        level_visits[level] += 1
    if H.is_last:
        return _base_case(H, b)

    A = H.A
    invD = H.inv_diag
    R = H.R

    x = invD * b                       # pre-smooth from zero
    bc = R @ (b - A @ x)

    same = _is_samesize(H)
    if same and mode == "stationary":
        # stationary correction: one sub-hierarchy apply, no inner FCG
        z = kscycle(levels, bc, level + 1, krepeat, inner_tol, mode,
                    samesize_nu, level_visits)
    else:
        # inner FCG; for a same-size level in "operator" mode minimize over this
        # level's own A (correct residual equation) and run samesize_nu inner
        # iterations, using the sub-hierarchy as an accelerated inner solver;
        # otherwise the standard K-cycle over the (Galerkin) next operator.
        ac_level = level if (same and mode == "operator") else level + 1
        nu = samesize_nu if (same and mode == "operator") else krepeat[level]
        z = _inner_fcg_ks(levels, level, ac_level, nu, bc, krepeat, inner_tol,
                          mode, samesize_nu, level_visits)

    x = x + R.T @ z
    x = x + invD * (b - A @ x)         # post-smooth
    return x


def _inner_fcg_ks(levels, level, ac_level, nu, bc, krepeat, inner_tol, mode,
                  samesize_nu, level_visits):
    """FCG(1) coarse solve, preconditioned by the recursive Ks-cycle, minimizing
    over ``levels[ac_level].A`` (the Galerkin next operator for a normal level,
    or this level's own operator for a same-size 'operator'-mode level), capped
    at ``nu`` inner iterations."""
    Ac = levels[ac_level].A

    z = np.zeros(bc.shape[0])
    r = bc.copy()
    bnorm2 = r @ r
    if bnorm2 == 0.0:
        return z
    tau2 = inner_tol * inner_tol
    stop2 = max(1e-28, tau2) * bnorm2

    dr_prev = 0.0
    r_prev = None
    p = None
    for i in range(nu):
        if r @ r <= stop2:
            break
        d = kscycle(levels, r, level + 1, krepeat, inner_tol, mode,
                    samesize_nu, level_visits)
        if i == 0:
            p = d.copy()
        else:
            num = d @ (r - r_prev)
            beta = num / dr_prev if dr_prev != 0.0 else 0.0
            p = d + beta * p
        q = Ac @ p
        pq = p @ q
        if pq <= 0.0:
            break
        rd = r @ d
        alpha = rd / pq
        dr_prev = rd
        r_prev = r.copy()
        z += alpha * p
        r -= alpha * q
    return z


def fcg_solve_ks(levels, b, tol=1e-8, maxiter=500, theta=0.75, inner_tol=0.25,
                 mode="operator", samesize_nu=8):
    """Outer flexible-CG driven by the Ks-cycle (a port of pycmg._solve.fcg_solve
    that calls kscycle). Operators are treated as SPD (is_sdd handled by the
    caller / not needed for the Laplacian+slack experiment). Returns
    (x, iterations, relres, converged)."""
    A = levels[0].A
    n = A.shape[0]
    krepeat = compute_kcycle_repeats(levels, theta)

    x = np.zeros(n)
    r = b.copy()
    bnorm = np.sqrt(r @ r)
    if bnorm == 0.0:
        return x, 0, 0.0, True

    iterations = 0
    converged = False
    dr_prev = 0.0
    r_prev = None
    p = None
    rnorm = bnorm
    for it in range(maxiter):
        rnorm = np.sqrt(r @ r)
        if rnorm <= tol * bnorm:
            converged = True
            break
        d = kscycle(levels, r, 0, krepeat, inner_tol, mode, samesize_nu)
        if it == 0:
            p = d.copy()
        else:
            num = d @ (r - r_prev)
            beta = num / dr_prev if dr_prev != 0.0 else 0.0
            p = d + beta * p
        q = A @ p
        pq = p @ q
        if pq <= 0.0:
            break
        rd = r @ d
        alpha = rd / pq
        dr_prev = rd
        r_prev = r.copy()
        x += alpha * p
        r -= alpha * q
        iterations += 1
    if not converged and np.sqrt(r @ r) <= tol * bnorm:
        converged = True
    relres = np.sqrt(r @ r) / bnorm
    return x, iterations, relres, converged


class _Counter:
    """Wrap an operator to count matvecs (delegates every other attribute)."""

    def __init__(self, A):
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "n", 0)

    def __matmul__(self, x):
        object.__setattr__(self, "n", self.n + 1)
        return self.A @ x

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "A"), name)


def _count_finest_matvecs(levels, run):
    """Total applies of the finest (size-n) operators during `run` -- the
    dominant cost, and the honest work metric (outer iteration count is not:
    the Ks-cycle's inner FCG hides many finest-level applies)."""
    n0 = levels[0].A.shape[0]
    saved = [L.A for L in levels]
    counters = []
    for L in levels:
        if L.A.shape[0] == n0:
            c = _Counter(L.A)
            counters.append(c)
            L.A = c
    try:
        run()
    finally:
        for L, a in zip(levels, saved):
            L.A = a
    return sum(c.n for c in counters)


if __name__ == "__main__":
    import random
    import time

    from build import build_sparsified_hierarchy
    from graphs import blob_chain, spd_operator
    from pycmg._precond import Preconditioner

    def _time(run, reps=3):
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            run()
            ts.append(time.perf_counter() - t)
        return min(ts) * 1e3

    print("Cycles on a sparsified hierarchy -- iterations are misleading; the\n"
          "honest cost is finest-level matvecs and wall-clock. (tol=1e-9)\n")
    header = (f"{'config':13}| {'cycle':18}| {'outer_its':>9} {'ms':>6} "
              f"{'size-n matvecs':>14}")
    print(header)
    print("-" * len(header))
    for c in [(6, 150, 40, 1e-2, 7), (10, 100, 40, 1e-2, 5),
              (4, 250, 56, 1e-3, 9), (8, 140, 52, 1e-3, 11)]:
        random.seed(c[4])
        np.random.seed(c[4])
        n, edges = blob_chain(c[0], c[1], avgdeg=c[2], seed=c[4], wbridge=c[3])
        A = spd_operator(n, edges, slack=1e-8)
        b = np.random.default_rng(0).standard_normal(n)
        b -= b.mean()
        on, _ = build_sparsified_hierarchy(A, sparsify_on_stall=True)
        inj = sum(1 for L in on if _is_samesize(L))
        M = Preconditioner(on, False)
        runs = [
            ("L-cycle", lambda: M.solve(b, method="legacy-cmg", tol=1e-9, maxiter=3000)),
            ("K-cycle (stock)", lambda: M.solve(b, method="kcycle", tol=1e-9, maxiter=3000)),
            ("Ks-cycle operator", lambda: fcg_solve_ks(on, b, tol=1e-9, maxiter=3000, mode="operator", samesize_nu=8)),
            ("Ks-cycle stationary", lambda: fcg_solve_ks(on, b, tol=1e-9, maxiter=3000, mode="stationary")),
        ]
        tag = f"{c[0]}x{c[1]} d{c[2]} i{inj}"
        for name, run in runs:
            its = run()[1]
            print(f"{tag:13}| {name:18}| {its:>9} {_time(run):>6.0f} "
                  f"{_count_finest_matvecs(on, run):>14}")
            tag = ""
        print("-" * len(header))
    print("\nTakeaway: the L-cycle uses the FEWEST finest-level matvecs and the\n"
          "least wall-clock. The Ks-cycle (operator) cuts outer iterations below\n"
          "the L-cycle but its nested inner FCG does 2-3x more total work: with\n"
          "spectrally-accurate sparsifiers (kappa~4-6) there is nothing for the\n"
          "K-cycle to accelerate. Drive sparsified hierarchies with the L-cycle.")
