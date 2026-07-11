# Julia port notes — sparsify-on-stall

> **Post-benchmark changes (supersede the 1:1 port below).** The 10⁶-chimera
> benchmark showed the ported trigger fired on normal levels, adding a large
> fixed overhead on non-stalling graphs. Three fixes were applied to the Julia
> port; the current behavior is:
> 1. **Trigger** (`src/cmgAlg.jl` `build_hierarchy`) — **nodes-vs-edges port
>    bug.** The validated reference (`reference.py`, `PYTHON-GUIDELINES.md`) uses a
>    NODE-coarsening stall test — productive iff `nc ≤ stall_ratio·n` — but the
>    Julia port tested EDGES (`m_c ≤ stall_ratio·m`), which fires on normal
>    chimera levels that coarsen nodes well yet keep edges (contraction fill).
>    Corrected to the node ratio. A graph that coarsens normally never sparsifies
>    → **zero** overhead, ~the stock hierarchy; only a genuine node plateau
>    (`nc > stall_ratio·n`) injects. `nnz_budget` remains a hard density cap.
> 2. **Defaults** (`src/sparsify.jl`): `keep_frac = 0.25` (was 0.5), `bundles = 2`
>    (was 1).
> 3. **kscycle** (`src/kcycle.jl`): the injected same-size level uses the standard
>    `krepeat[lvl]` work-budget iterations (like every other level), not a fixed
>    `_KSCYCLE_NU = 8`; the only same-size special-case left is which operator the
>    inner FCG minimizes over (`H[lvl].A`). `_KSCYCLE_NU` was removed.
>
> The narrative and expected numbers below describe the *original* port and are
> kept as historical record.

This is the **production Julia port** of sparsify-on-stall, ported 1:1 from the
validated, merged CMG-python package (`pycmg` ≥ 0.5.0, opt-in
`precondition(A, sparsify_on_stall=True)`). Unlike the Python-first *experiment*
files in this directory (`prototype.py` / `reference.py` / `PYTHON-GUIDELINES.md`),
the port lives in the package `src/` and is wired into `cmg_solve` behind a flag,
off by default.

**Validation mode: author & hand off.** Julia was not runnable in the porting
sandbox, so the code was written against the validated Python and the live Julia
source. Correctness is established by `Pkg.test()` (the new `@testset
"sparsify-on-stall"`) and the benchmarks below, run on your machine / Wulver.
The numbers to expect are listed so results diff cleanly.

## Where the code lives (production `src/`)

| Julia | ports from (CMG-python) | what |
|---|---|---|
| `src/spanner.jl` | `_sparsify.py` | greedy + Baswana–Sen spanners |
| `src/sparsify.jl` | `_sparsify.py` | `SparsifyOptions`, `spanner_bundle`, `sparsify`, `edges_of`/`slack_of`/`sdd_from_edges` |
| `src/cmgAlg.jl` `build_hierarchy` (forked) | `build_sparsified_hierarchy` (`_hierarchy.py`) | edge-stall + injection + nnz budget, behind `sparsify_on_stall` |
| `src/kcycle.jl` `kscycle!`/`inner_fcg_ks!`, `cycle=:kscycle` | `_kscycle.py` | sparsify-aware K-cycle (operator mode) |
| flag threading in `cmg_solve` / `cmg_preconditioner_lap` / elimination / disconnected | `precondition()` wiring (`_precond.py`) | `sparsify_on_stall` / `sparsify_opts` |
| `test/runtests.jl` `@testset "sparsify-on-stall"` | `tests/test_sparsify.py` | the checks |

## Function map (spot-check against the Python)

| Julia | Python |
|---|---|
| `spanner_baswana_sen`, `_baswana_sen_impl` | `spanner_baswana_sen`, `_baswana_sen_impl` |
| `greedy_spanner`, `_within` | `greedy_spanner` (`within`) |
| `spanner_bundle`, `sparsify` | same |
| `edges_of`, `slack_of`, `sdd_from_edges` | same (`slack_of` == row sums here) |
| `SparsifyOptions` | `SparsifyOptions` (defaults identical; `base=500` not 700, the Julia direct threshold) |
| `build_hierarchy(...; sparsify_on_stall)` fork | `build_sparsified_hierarchy` |
| `kscycle!` / `inner_fcg_ks!` | `kscycle` / `_inner_fcg_ks` |

## Expected numbers (reproduce from CMG-python; RNGs differ so not bit-exact)

- **stall → resume** (`bench_sparsify.jl`): a dense blob has `edge_ratio ≳ 0.95`
  (stall); one `keep_frac=0.5` sparsify drops it below 0.9, ~2× fewer edges,
  `kappa(A_sp⁻¹A) ≈ 4–15`.
- **spanner essential**: spanner+uniform `kappa ≈ 4–6`; uniform-only `kappa`
  orders of magnitude larger (Python: ~1e5–1e11) — the assertion is `≥ 20×` worse.
- **efficiency**: Baswana–Sen ~**150× faster** than greedy at n≈3600 (greedy ~12 s,
  BS ~0.08 s); both give the same bounded κ.
- **end-to-end** (`bench_cycles.jl`, chain of 6×150 dense blobs): `:legacy`
  converges in ~15–17 iters and is the fastest wall-clock; `:kcycle` degrades;
  `:kscycle` cuts to ~6–8 iters but does 2–3× the work.
- **bundles** (`bench_bundles.jl`): at `keep_frac=0.25`, greedy B=2 does **not**
  beat B=1 (neutral-to-worse median κ over seeds).
- **SDD path** (`test` #5): the original SDD system is solved **exactly**
  (`‖x − A⁻¹b‖/‖A⁻¹b‖ < 1e-6`, not up to the null-space constant).
- **off == stock** (`test` #3): `sparsify_on_stall=false` reproduces the stock
  hierarchy level-for-level.

## How to run

```
julia --project=. -e 'using Pkg; Pkg.test()'                 # the suite (incl. sparsify)
julia --project=. experiments/sparsified/bench_sparsify.jl [--quick]
julia --project=. experiments/sparsified/bench_cycles.jl   [--quick]
julia --project=. experiments/sparsified/bench_bundles.jl  [--quick]
# bench_wulver.jl needs Laplacians.jl + a chimera generator; run on Wulver
```

## Port caveats to verify on first run

- **`DataStructures.PriorityQueue`** (greedy spanner's Dijkstra): added as a new
  package dependency. Baswana–Sen (the production default) is heap-free and
  needs no external dep; greedy is the reference used by the efficiency bench.
- **Repeat counts on an injected level**: the existing `init_LevelAux` /
  `compute_kcycle_repeats` derive per-level repeats from adjacent `nnz` ratios
  and are used unchanged; a same-size injected level gets a sensible repeat the
  same way every level does (verified analogous in Python). Expect the L-cycle
  iteration counts to land in the Python ballpark (~15–17 on the 6×150 chain).
- **Finest-level matvec counting** (Python's honest cross-cycle metric) is
  proxied here by wall-clock in `bench_cycles.jl`; wall-clock already shows the
  L-cycle winning. Adding an operator-instrumentation pass is a small extension.
- **`sdd_from_edges`** relies on `sparse` summing duplicate triplets and keeping
  the explicit diagonal; on a connected Laplacian (every diagonal positive) this
  matches the Python bridge exactly.
