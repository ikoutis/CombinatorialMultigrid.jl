# Experimental sparsified-CMG (dev) — sparsify on aggregation stall

Isolated experiment (NOT wired into the package, NOT exported) exploring a fix
for CMG's one remaining hierarchy weakness: the **coarsening stall**. CMG never
sparsifies, so on levels that densify from contraction fill (or are
expander-like) aggregation cannot coarsen — producing 18–23-level hierarchies
(~2× slower than ApproxChol on a minority of connected 10⁶ chimeras).

**Idea (Y. Koutis; backed by his spanner-based spectral-sparsification result):**
when aggregation stalls, inject a **spanner + uniform-sample spectral sparsifier**
as the next level (keep_frac ≈ 1/2 of the off-spanner edges, reweighted). The
sparser graph lets standard CMG aggregation resume, controlling the density that
contraction generates. Sparsification is the general density-control primitive
CMG lacks; degree-1/2 elimination is a cheap special case of it. (The current
default spanner is a maximum-spanning-forest bundle — a cut sparsifier — not
Baswana–Sen; see **Current design** below.)

## Key structural finding: the solver needs zero changes

The K-cycle is already size-agnostic. Transfer is done by `cI`-keyed
scatter/gather (`interpolate!`, `prolongate_add!`), which **degrade to the
identity** when `cI = 1:n, nc = n`; `nc` is only a loose CG cap in
`compute_kcycle_repeats` (all denominators `max(·,1)`), and buffers size on `n`.
So a **sparsify level is just a normal `HierarchyLevel`** with
`cI = collect(1:n)`, `nc = n`, `A =` the sparsifier, `invD = 1 ./ (2·diag)`,
`islast=false, iterative=true` — and the existing `cmg_solve(H, b)` runs it
unchanged. The *only* strict-coarsening assumption in the whole codebase is the
build's stagnation guard `nc >= n-1` (`src/cmgAlg.jl:243-249`), which currently
means "give up." That single predicate is the fork point: on stall, inject a
sparsify level and continue instead of breaking.

Consequence: the experiment is a **forked build loop + spanner/sparsify code**,
reusing the production solve verbatim. Production `src/` is untouched.

## Files
- `prototype.py` — NumPy/SciPy **spectral gate** (below): validates that
  spanner+uniform is a bounded-κ sparsifier and that the spanner is essential.
  Run: `python3 prototype.py`.
- `reference.py` — **runnable executable spec** of the full algorithm: a minimal
  aggregation-AMG that reproduces the stall and demonstrates sparsify-on-stall
  end-to-end (adaptive-p, bundle iteration, same-size identity-transfer level).
  Run: `python3 reference.py`.
- `PYTHON-GUIDELINES.md` — spec/guidelines for implementing this in a Python CMG
  (cmg-python), written against `reference.py`.
- **Julia port (production — done, this branch).** The validated algorithm now
  lives in the package `src/`, behind `cmg_solve(A, b; sparsify_on_stall=true)`
  (off by default, stock behavior byte-identical): `src/spanner.jl` (greedy +
  Baswana–Sen), `src/sparsify.jl` (`SparsifyOptions`, bundle, adaptive sparsify,
  edge bridge), the `build_hierarchy` fork in `src/cmgAlg.jl`, and the
  sparsify-aware `cycle=:kscycle` in `src/kcycle.jl`. Tests:
  `test/runtests.jl`'s `@testset "sparsify-on-stall"`. Benchmarks (this dir):
  `bench_sparsify.jl`, `bench_cycles.jl`, `bench_bundles.jl`, `bench_wulver.jl`
  (+ shared `graphs.jl`). Mapping and expected numbers: `PORT-NOTES.md`.
  Baswana–Sen (`k=⌈log₂ n⌉`) is the production spanner; greedy is the reference.

## Gate results (prototype.py)
Spanner in the resistance metric (edge length = 1/conductance): a t-spanner
bounds every off-spanner edge's effective resistance by `t·r`, i.e. leverage
≤ t, which makes uniform sampling concentrate.

- **Density sweep** (dense blob, n=400, spanner t≈log n + uniform-1/4): edge
  reduction **grows with density** (2.0× at avg-deg 8 → 3.5× at 64) while κ(M⁻¹L)
  **improves** with density (39 → 4.5); avg degree drops 16→6, 32→10. This is the
  fill regime the stall lives in, and it's where the sparsifier is strongest.
- **Spanner is essential** (two dense blobs + one weak bridge): uniform sampling
  *without* a spanner disconnects 5/5 draws (drops the bridge → κ=∞);
  spanner+uniform disconnects 0/5 and stays bounded.

Conclusion: spanner+uniform is a connected, bounded-κ sparsifier with
density-growing reduction — the primitive works. κ at moderate density (~30–40)
is a tuning surface (sampling rate `p`, spanner stretch, number of spanner
bundles) that the exact construction tightens; it does not block the approach.

## Reference results (reference.py)
On a chain of 8 dense blobs joined by weak cuts (effective κ ≈ 2.6e7):
- **Stall → resume:** aggregation `nc/n` = 0.998 (stall) → 0.63 (bundles=1) /
  0.39 (bundles=2) after one adaptive sparsify (2× fewer edges), with the
  sparsifier κ(M⁻¹A) ≈ 21. More bundles → aggregation resumes harder.
- **End-to-end:** V-cycle-PCG stalls without the fix (~50 iters, accuracy lost to
  the ill-conditioning); with sparsify-on-stall it coarsens 960→~20 and converges
  in ~12 iters. The gap grows with κ/scale. Solution matches a direct solve.

## Current design (as shipped on this branch)

Three fixes landed after the first 10⁶-chimera benchmark exposed the ported
edge-ratio trigger as a net regression; the current behavior:

- **Trigger — NODE ratio, not edges.** A level is productive iff aggregation
  drops the node count below `stall_ratio·n` (default 0.9); only a genuine node
  plateau (`nc > stall_ratio·n`) injects. The port had mistranslated the
  validated `reference.py` node criterion to edges (`m_c ≤ stall_ratio·m`), which
  fired on normal levels and added a **12–14× fixed overhead** on non-stalling
  graphs. `nnz_budget` stays a hard density cap.
- **Spanner — maximum-spanning-forest bundle** (default `spanner=:mst,
  bundles=3`). A `k`-forest bundle is a Nagamochi–Ibaraki cut certificate; on the
  high-conductance stall levels (the only place sparsify fires) cut ≈ spectral,
  so it is a good spectral preconditioner there at `O(m)` build vs Baswana–Sen's
  `O(km)`, with a fixed density-independent `k·(n-1)`-edge bundle. Baswana–Sen
  and greedy remain available. (Python gate: κ 3.0–5.5, comparable-to-better than
  Baswana–Sen; same reduction; same end-to-end iterations.)
- **`keep_frac = 0.5`** (2× per injection → L-cycle injected-level repeat = 1);
  **`:kscycle`** runs the standard `krepeat` at injected levels (not a fixed ×8).

## Benchmark results (Wulver, 10⁶ chimeras, reps=3)

Full writeup — the vs-`ac` tables (stalled/clean), MST-vs-Baswana build cost,
iteration analysis, and the `:legacy`-vs-`:kscycle` comparison — is in
**[`README-sparsify.md`](README-sparsify.md)**. Headline:

- **Clean draws (~70–90% majority):** every CMG variant beats `ac` **1.3–1.9×**;
  sparsify is inert and matches `cmg-k-elim` — the 12–14× edge-trigger regression
  is fixed.
- **Stalled draws (minority):** `cmg-k-elim` is the *only* place CMG loses to `ac`
  (up to ~2× on the worst draws); `cmg-sparsify-l` lifts it to **~`ac` parity**.
- **MST vs Baswana–Sen:** MST **halves** the spanner build overhead on the large
  stalled core (`uni_bndry`, +3.8s → +1.9s) and gives comparable-or-fewer
  iterations; neutral where the core is small.
- **`-l` vs `-ks`:** within ~10% wall-clock; `-ks` takes the fewest iterations
  (and dips below `cmg-k-elim` on stalled cores), `-l` the fewest matvecs.
- **Net: with sparsify-on-stall, CMG matches or beats `ac` across the whole
  10⁶-chimera distribution.**

## Status
Gate + executable reference **done and validated in Python**; the algorithm is
**shipped in CMG-python** (`pycmg` ≥ 0.5.0, opt-in `precondition(A,
sparsify_on_stall=True)`) and now **ported to this Julia package** as a
production opt-in (this branch — `src/` + `test/` + benches, off by default and
byte-identical when off). The **10⁶-chimera benchmark on Wulver is done** (via
the `laplacian-bench` harness, `reps=3`) — see **Benchmark results** above:
sparsify-on-stall is safe on the clean majority and lifts CMG's stall minority
from ~2×-slower-than-`ac` to `ac` parity. Remaining: re-run with the
MST-bundle-3 spanner default (should cut the residual Baswana–Sen build cost on
the largest stalled cores, e.g. `uni_bndry`'s +3.8s), then open a PR. The
synthetic benches (`bench_sparsify.jl` / `bench_cycles.jl` / `bench_bundles.jl`)
need no chimera generator — dense blobs / chains reproduce the stall and run in
any Julia env.
