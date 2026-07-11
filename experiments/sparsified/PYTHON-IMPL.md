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
stall->resume : dense_blob(400, deg32)  nc/n 0.953 -> 0.807 after one sparsify
                (2.02x fewer edges), kappa(A_sp^-1 A) = 4.0
spanner-ess.  : two blobs + weak bridge  spanner+unif kappa 4.0  vs  unif-only 1.1e11
end-to-end    : chain of 6x150 deg-40 blobs (V-cycle PCG, method="legacy-cmg")
                without fix : 1 level  (stalls/gives up)  -> 48 iters
                with   fix  : 5 levels (inject=1)          -> 16 iters   [3.0x fewer]
correctness   : chain + single blob, both methods, true residual < 1e-8
                (centered solution error 4e-11 / 6e-11)
```

## Design decisions & findings (the port-back notes)
- **`t = log2(n)`, not `ln(n)`.** reference.py's `t=9.0` at n≈400–960 is `log2`. With
  `ln(n)` the greedy spanner keeps too many edges, the 0.98 reduction gate trips, and
  the fork never fires. `sparsify(t=None)` defaults to `max(2, log2(n))`.
- **Fork on `nc ≤ stall_ratio·n` (0.9), not `nc ≥ n−1`.** The real `steiner_group`
  coarsens the mild plateau that CMG actually suffers (18–23 levels), so the fork keys
  on the aggressive `stall_ratio`. One `keep_frac=0.5` sparsify **halves** the degree,
  so a deg-32 blob resumes in one injection (nc/n 0.953→0.807, crossing 0.9) while
  denser blobs are injected repeatedly until productive — the build loop handles this
  automatically (bounded by `max_inject` and the 0.98 reduction gate → termination).
- **Same-size identity-transfer level needs zero solver changes.** A sparsify level has
  `cluster_indices = arange(n)`, `num_clusters = n`, `R = n×n identity`; `pycmg`'s
  cycle restricts/prolongs with `R @ r` / `R.T @ z` (identity matmul) and picks up the
  sparsifier as `levels[level+1].A`. Confirmed end-to-end through both `vcycle` and
  `kcycle`. **`src/pycmg` is untouched.**
- **The win is on the V-cycle PCG (`method="legacy-cmg"`), not the K-cycle.** The
  same-size level is a *stationary* correction; the K-cycle wraps every level in an
  inner FCG, and running that inner FCG over a *same-size* sparsifier operator adds
  cost rather than removing it (chain: legacy-cmg 48→16, but k-cycle 48→57). **Port-back
  note:** either drive the sparsified hierarchy with the V-cycle, or special-case a
  same-size level in the K-cycle (skip the inner FCG, apply the sub-hierarchy once).
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
