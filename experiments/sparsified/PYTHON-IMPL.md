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
| `sparsify.py` | greedy spanner (resistance metric) + `spanner_bundle` + adaptive `sparsify` + scipy⇄edge-list SDD bridge. **No pycmg imports** — ports 1:1 to Julia. |
| `graphs.py` | SPD test-graph generators (`dense_blob`, `blob_chain`, `dense_blob_pair_bridge`), retuned dense enough to stall the *real* aggregation. |
| `build.py` | `build_sparsified_hierarchy` — mirrors `pycmg._hierarchy.build_hierarchy`, forking only the stall guard to inject a same-size sparsifier level. The only file importing pycmg internals. |
| `validate.py` | the three guideline checks, run against the real aggregation/solver. |
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

**Port-back recommendation.** Drive a sparsified hierarchy with the **L-cycle**
(`cycle = :legacy` in Julia) — it is the natural fit, the sparsify correction being
stationary. If the K-cycle is wanted, special-case a same-size level in `_inner_fcg`:
either skip the inner FCG and add one sub-hierarchy apply (a stationary step), or use
the level's own operator `A` (not `levels[level+1].A`) for the step-length
minimization. The guidelines' "solve side needs no changes" holds for *correctness* and
for the L-cycle; this is a K-cycle *efficiency* caveat.
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

## Efficiency
Greedy spanner (pure Python, `heapq`/`dict` Dijkstra) — sub-second per call at the
validation scale (n ≤ ~1000) and it ports 1:1 to Julia. **Baswana–Sen** (`k=⌈log₂ n⌉`,
O(m) expected) is the scale/production spanner named in the guidelines; add it behind
`sparsify(spanner="baswana-sen")` when a large (e.g. the ~8400-node dumped) operator
needs it. Prefer Baswana–Sen over numba-ifying greedy — it's the algorithm that scales
and the one Julia needs; greedy's Dijkstra is numba-hostile anyway.

## Port-back plan
Port `sparsify.py` (spanner + adaptive sparsify + SDD bridge) and the `build.py` stall
fork to Julia (`spanner.jl` / `sparsify.jl` / `build_sparsified_hierarchy.jl` here),
carrying the findings above (esp. the `t=log2`, `stall_ratio` fork, and the V-cycle vs
K-cycle note). The Julia solve side needs no changes for the same-size level (verified
in the guidelines); confirm the K-cycle interaction and bench on Wulver vs
`cmg-k-elim` / `ac` on `uni_chimera(1e6,10)` / `wted_chimera(1e6,5)`.
