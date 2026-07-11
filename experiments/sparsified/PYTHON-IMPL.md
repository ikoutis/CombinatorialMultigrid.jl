# Sparsified-CMG — cmg-python implementation

Python-first implementation of **sparsify-on-stall**, built against the real
[`cmg-python`](https://github.com/ikoutis/CMG-python) (`pycmg`) aggregation and
solver, per [`PYTHON-GUIDELINES.md`](PYTHON-GUIDELINES.md). Validated here in the
sandbox; the algorithm is now ready to port to the Julia CMG (the pending
`spanner.jl` / `sparsify.jl` / `build_sparsified_hierarchy.jl` in this directory).

This is an **isolated experiment**: it `import`s `pycmg` internals but touches
**neither repo's `src/`**. It reuses the real `steiner_group` aggregation,
`contract_coo`, `HierarchyLevel`, and `Preconditioner.solve` verbatim; only the
on-stall build branch and the spanner/sparsify module are new.

## Files
| file | role |
|---|---|
| `sparsify.py` | two spanners — `greedy` (resistance metric) and `baswana-sen` (numba-jitted, scalable) — + `spanner_bundle` + adaptive `sparsify` + scipy⇄edge-list SDD bridge. **No pycmg imports** (numpy/scipy/numba only) — ports 1:1 to Julia. |
| `graphs.py` | SPD test-graph generators (`dense_blob`, `blob_chain`, `dense_blob_pair_bridge`), retuned dense enough to stall the *real* aggregation. |
| `build.py` | `build_sparsified_hierarchy` — mirrors `pycmg._hierarchy.build_hierarchy`, forking only the stall guard (on the EDGE ratio) to inject a same-size sparsifier level. Imports pycmg internals. |
| `validate.py` | the three guideline checks, run against the real aggregation/solver. |
| `kscycle.py` | the adaptive **Ks-cycle** and an honest cycle comparison (`python3 kscycle.py`) — see the Cycles section. Imports the unchanged pycmg cycle primitives. |
| `solve.py` | `sparsified_solve(levels, b, cycle=...)` — the three cycle branches (`l-cycle` default / `k-cycle` / `ks-cycle`). Maps to the pending Julia `sparsified_solve.jl`. |
| `run.py` | thin entry point (`python3 run.py` → `validate.py`). |

## How to run
```bash
pip install -e /path/to/CMG-python        # editable pycmg on sys.path
python3 experiments/sparsified/validate.py
```

## Validated numbers (this sandbox, real pycmg 0.4.0)
```
stall->resume : dense_blob(400, deg24)  edge_ratio 0.962 -> 0.869 after one
                sparsify (node_ratio 0.920 -> 0.627, 2.02x fewer edges),
                kappa(A_sp^-1 A) = 6.5
spanner-ess.  : two blobs + weak bridge  spanner+unif kappa 4.0  vs  unif-only 1.1e11
end-to-end    : chain of 6x150 deg-40 blobs (L-cycle PCG, method="legacy-cmg")
                without fix : 1 level  (stalls/gives up)  -> 48 iters
                with   fix  : 4 levels (inject=2)          -> 16 iters   [3.0x fewer]
correctness   : chain + single blob, both methods, true residual < 1e-8
                (centered solution error 7e-11 / 6e-11)
```

## Design decisions & findings (the port-back notes)
- **`t = log2(n)`, not `ln(n)`.** reference.py's `t=9.0` at n≈400–960 is `log2`. With
  `ln(n)` the greedy spanner keeps too many edges, the 0.98 reduction gate trips, and
  the fork never fires. `sparsify(t=None)` defaults to `max(2, log2(n))`.
- **Stall is decided on the EDGE ratio, not the node ratio.** The stall CMG cannot
  escape is *densification*: contraction fill (or an expander-like level) keeps ~as
  many edges even as nodes merge, so the coarse operator is no cheaper — that is what
  balloons hierarchy depth and per-iteration cost, and what the sparsifier targets. A
  node-based test (`nc ≤ stall_ratio·n`) can call such a level "productive" while its
  density, hence work, does not fall. So the build contracts, then keeps the level iff
  it thinned edges: **productive iff `m_c ≤ stall_ratio·m`** (`m = nnz_lower − n` = the
  lower-triangle off-diagonals), else inject. On a dense blob contraction barely thins
  edges (edge_ratio ≈ 0.96–0.99); one `keep_frac=0.5` sparsify **halves** the edges and
  lets aggregation coarsen so the next contraction thins them (edge_ratio drops below
  the 0.9 default). Denser blobs are injected repeatedly until productive — the build
  handles this automatically (bounded by `max_inject` and the 0.98 reduction gate →
  termination; each productive step strictly drops edges, so the loop terminates).
- **Two stopping criteria, both applied (`sparsify_on_stall` on/off knob).** Recursion
  stops on EITHER: **(A)** the per-level edge stall above — with sparsification on, a
  stall injects a sparsifier and continues (with it off, a stall terminates, as stock
  CMG does); or **(B)** a cumulative operator-complexity budget `cumulative_nnz >
  nnz_budget · nnz(input)` (default `nnz_budget = 5`, the pycmg value). (B) bounds
  per-cycle work/memory and **backstops timid sparsification**: at `keep_frac = 0.5`
  each sparsifier ~halves the edges so the injected chain sums to C_op ≈ 2 (geometric
  `1/(1−keep_frac)`) and (B) never fires; as `keep_frac → 1` the injections barely thin
  the graph, C_op climbs toward `1/(1−keep_frac)`, and (B) stops the build instead of
  stacking `max_inject` useless levels (measured: `keep_frac 0.9` → C_op ≈ 5.3, stops at
  6 injections vs 10). `nnz_budget = inf` disables (B). This restores pycmg's
  `cumulative_nnz > 5·initial_nnz` guard (which the first cut had dropped), now counting
  the sparsifiers' nnz and sitting beside the edge-stall as the two termination rules.
- **Same-size identity-transfer level needs zero solver changes.** A sparsify level has
  `cluster_indices = arange(n)`, `num_clusters = n`, `R = n×n identity`; `pycmg`'s
  cycles restrict/prolong with `R @ r` / `R.T @ z` (identity matmul) and pick up the
  sparsifier as `levels[level+1].A`. Confirmed end-to-end through both cycles.
  **`src/pycmg` is untouched** — see the Cycles section below.

## The cycles (and why "V-cycle" is a misnomer → call it the L-cycle)

`pycmg` (following the C++/MATLAB CMG) exposes two cycles. The stationary one is named
`vcycle` in the source and selected by `method="legacy-cmg"` (aliases `"legacy"`,
`"vcycle"`). It is **not a geometric V-cycle**: it is the classic CMG *stationary*
cycle — Jacobi pre-smooth, restrict, recurse `H.repeat` times (a count from
lower-triangle nnz ratios, not a geometric V/W schedule), prolong, Jacobi post-smooth,
with a direct-LDL or Jacobi base. Calling it a V-cycle is misleading; **the L-cycle
(L for legacy)** is the honest name. `docs` and comments here use L-cycle.

**Did anything change in it? No.** This experiment imports `pycmg._cycles.vcycle` /
`kcycle` and calls them verbatim; `src/pycmg/_cycles.py` is untouched. A sparsify level
is just an ordinary `HierarchyLevel` with identity transfer, so both cycles run it with
no special case. The only new code is the build-loop fork and the spanner/sparsify
module.

**L-cycle on an injected sparsify level (works — the 3× win).** At the injected level
(`R = I`, `A =` the original dense operator, `levels[level+1].A =` the sparsifier), the
L-cycle Jacobi-smooths on the dense `A`, restricts the residual by identity
(`bc = R @ r = r`), recurses into the sub-hierarchy built on the *sparsifier* (which
coarsens normally), prolongs by identity (`x += R.T @ z = z`), and post-smooths on `A`.
Because the sparsifier ≈ `A` spectrally (κ(A_sp⁻¹A) ≈ 4–6), that stationary coarse
correction is accurate, and the outer PCG converges — chain: **48 → 16 iters**. The
L-cycle stays a fixed linear operator (scipy-`M` safe).

**K-cycle on an injected sparsify level (degrades — a port-back note).** The K-cycle is
worse *specifically* at an injected level (chain: 44 vs the L-cycle's 16; on a
non-injecting Galerkin grid the two are identical, 27 vs 27 — so it is the injection,
not the problem). Mechanism, traced in `_cycles.py`: the K-cycle replaces each level's
stationary correction with an inner FCG (`_inner_fcg`) that **minimizes the step length
over `Ac = levels[level+1].A`** (`alpha = (r·d)/(p·Ac p)`). For a *normal* level that
`Ac` is the Galerkin coarse operator `R A Rᵀ`, consistent with the transfer. For a
*sparsify* level the transfer is the identity, so the consistent coarse operator would
be `A` itself — but `levels[level+1].A` is the **sparsifier**, which differs from `A`
by κ ≈ 4–6. So the inner FCG optimizes its step against the wrong operator while the
level's residual/smoother use `A`; the nonlinear FCG recursion propagates that
mismatch. (It is *not* excessive inner iterations — the injected level's `krepeat` is
already 1; forcing it to 1 changes nothing.)

**The Ks-cycle: an adaptive K-cycle for the sparsify level (`kscycle.py`).** We
implemented the two fixes above and measured them (`python3 kscycle.py`), which turned
up the real conclusion. `kscycle.py` special-cases a same-size level in two modes:
*stationary* (skip the inner FCG, one sub-hierarchy apply) and *operator* (inner FCG,
but minimize over the level's own `A` and run `samesize_nu` inner iterations, using the
sub-hierarchy as an accelerated inner solver). Nothing in `src/pycmg` changes — the
outer FCG driver is ported into the experiment (verified to reproduce pycmg's
`legacy-cmg` to the iteration when forced all-stationary).

**Iteration count is a trap here; the honest metric is finest-level matvecs / wall
time.** Measured on chains of dense blobs (tol 1e-9):

| cycle | outer its | size-n matvecs | wall-clock |
|---|---|---|---|
| **L-cycle** | 15–17 | **105–119** | **11–17 ms** |
| K-cycle (stock) | 26–171 | 234–1539 | 19–173 ms |
| Ks-cycle *operator* (nu=8) | **6–8** | 192–348 | 29–45 ms |
| Ks-cycle *stationary* | 17 / 85 | 119 / 595 | 11 / 108 ms |

- The **stock K-cycle degrades badly** (up to 171 iters / 1539 matvecs) — the operator
  inconsistency, worse the more sparsify levels stack.
- The **Ks-cycle *operator* mode cuts outer iterations below the L-cycle** (6–8 vs
  15–17) — the operator-consistent fix works — **but does 2–3× more total work**,
  because with two stacked same-size levels its `samesize_nu` inner iterations nest and
  it applies the finest operators far more often. Fewer iterations, more matvecs, slower.
- The **Ks-cycle *stationary* mode** (L-cycle locally at the sparsify levels, K-cycle
  at the normal levels) equals the L-cycle in some configs but is up to ~5× slower in
  others (85 vs 15 iters). It does **not** destabilize — no breakdown, `r·d > 0` every
  iteration; it just converges slowly. Cause, isolated by turning levels stationary one
  at a time: the culprit is the **first normal (Galerkin) level directly below the
  sparsify chain**. With that level on the K-cycle's inner FCG while the sparsify levels
  above are stationary, the stationary/Krylov-accelerated *adjacency* composes badly
  under the outer FCG(1); making just that one level stationary too restores 15 iters.
  So "local L at the sparsify level" is the wrong granularity — it leaves the boundary
  with the K-cycle exactly where it hurts.

### Choosing the cycle (`solve.py` — three branches)

`sparsified_solve(levels, b, cycle=...)` exposes the three as separate branches; the
default is the L-cycle. `python3 solve.py` runs all three.

| `cycle=` | use it for | notes |
|---|---|---|
| **`"l-cycle"`** (default) | **sparsified hierarchies** (and non-sparsified) | recommended: fewest matvecs, fastest; runs sparsify levels unchanged |
| `"k-cycle"` | **non-sparsified** hierarchies (`sparsify_on_stall=False`) | the stock Notay K-cycle; **degrades on sparsify levels** — don't use it there |
| `"ks-cycle"` | **sparsified hierarchies, as a robustness fallback** | operator mode, `samesize_nu=8`; fixes the stock K-cycle's degradation, reliably converges, but ~2–3× the L-cycle's work |

**Conclusion / port-back recommendation.** Default to the **L-cycle** (`cycle = :legacy`
in Julia) for sparsified hierarchies — fewest matvecs, fastest — because the injected
sparsifiers are already spectrally accurate (κ(A_sp⁻¹A) ≈ 4–6), so there is **nothing
for the K-cycle's inner FCG to accelerate**: a single stationary apply per level is
optimal, and every inner iteration is wasted work. **Keep the stock K-cycle for the
non-sparsified path** (`sparsify_on_stall=False`), where it is unchanged and correct.
**Keep the Ks-cycle (operator mode) available as a robustness fallback** for sparsified
hierarchies: it special-cases the same-size level — its inner FCG minimizes over that
level's *own* operator (not the sparsifier) and runs `samesize_nu` inner iterations —
so it fixes the stock K-cycle's degradation and converges reliably even where the
L-cycle might underperform, at the cost of more work. So three distinct branches, not
one: `l-cycle` (default), `k-cycle` (non-sparsified), `ks-cycle` (sparsified fallback).
The stock K-cycle is never the right tool *on* a sparsified hierarchy — use `ks-cycle`
there instead. (The Ks-cycle *stationary* mode — "L locally at the sparsify level" — is
kept in `kscycle.py` for study but is not a branch here: it can converge slowly, above.)
- **Production CMG breaks down on the stall.** On the ill-conditioned chain, the
  standard `precondition(A).solve` FCG breaks down (`conv=False`, true_res 8.7e-5) — the
  "loses accuracy to the ill-conditioning" the reference describes — whereas the
  sparsified hierarchy converges. This is the motivating failure.
- **Operators are near-Laplacian SPD (Laplacian + tiny slack), `is_sdd=False`.**
  `pycmg`'s base solve grounds the Laplacian null space (`ldl` on `A[:-1,:-1]`, pins
  `x[last]=0`), so the experiment uses tiny-slack operators with `b ⊥ 1` (as
  reference.py does) and measures solution error modulo the constant null space.
  `pycmg` is a *residual-reducing* iterative solver, so correctness is the true
  residual `< tol`; solution error is the conditioning-limited `κ·residual` (not
  machine-precision as in reference.py's exact base solve).

## Efficiency — two spanners (`sparsify(spanner=...)`, threaded through `build`)

The spanner is the whole cost of the build (profiled: ~99% of build time). Two are
provided:

- **`"greedy"`** (default) — the Althofer greedy t-spanner, pure Python (`heapq`/`dict`
  Dijkstra), O(m·Dijkstra). Unambiguous for correctness and it ports 1:1 to Julia, but
  it does **not** scale: ~1s per call at n=900, **12s at n=3600**, and a full build there
  was ~72s. Fine only at validation scale (n ≤ ~1000).
- **`"baswana-sen"`** — the Baswana–Sen randomized (2k−1)-spanner (`k = ⌈log₂ n⌉`),
  O(k·m) expected and **numba-jitted** (array-based, epoch-stamped cluster accumulator,
  no heap/dict — unlike greedy, which is numba-hostile). This is the scalable/production
  spanner the guidelines name. Measured **~150× faster** and it scales: the n=3600
  spanner is **82 ms** (vs greedy's 12 s), and a full n=3600 build is **309 ms** (vs
  ~72 s). It is a genuine drop-in — the spanner is connected and its sparsifier has the
  same bounded κ(A_sp⁻¹A) ≈ 3–6 as greedy (validated in `test_baswana_sen`), and
  end-to-end solves converge identically (~15–17 L-cycle iters). Its spanner is denser
  than greedy's (stretch 2k−1 ~ 2 log₂ n vs log₂ n), but the adaptive `p` compensates so
  the per-injection reduction stays ~2×.

Recommendation: `greedy` for small validation and unambiguous correctness; `baswana-sen`
for scale (the ~8400-node dumped operator, larger synthetics) and as the version to port
to Julia. Both live in `sparsify.py` and stay pycmg-free (only numpy/scipy/numba).

## Port-back plan
Port `sparsify.py` (spanner + adaptive sparsify + SDD bridge) and the `build.py` stall
fork to Julia (`spanner.jl` / `sparsify.jl` / `build_sparsified_hierarchy.jl` here),
carrying the findings above (esp. the `t=log2`, `stall_ratio` fork, and the V-cycle vs
K-cycle note). The Julia solve side needs no changes for the same-size level (verified
in the guidelines); confirm the K-cycle interaction and bench on Wulver vs
`cmg-k-elim` / `ac` on `uni_chimera(1e6,10)` / `wted_chimera(1e6,5)`.
