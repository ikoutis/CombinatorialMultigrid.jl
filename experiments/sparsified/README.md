# Experimental sparsified-CMG (dev) — sparsify on aggregation stall

Isolated experiment (NOT wired into the package, NOT exported) exploring a fix
for CMG's one remaining hierarchy weakness: the **coarsening stall**. CMG never
sparsifies, so on levels that densify from contraction fill (or are
expander-like) aggregation cannot coarsen — producing 18–23-level hierarchies
(~2× slower than ApproxChol on a minority of connected 10⁶ chimeras).

**Idea (Y. Koutis; backed by his spanner-based spectral-sparsification result):**
when aggregation stalls, inject a **spanner + uniform-sample spectral sparsifier**
as the next level (~1/4 of the off-spanner edges kept, reweighted). The sparser
graph lets standard CMG aggregation resume, controlling the density that
contraction generates. Sparsification is the general density-control primitive
CMG lacks; degree-1/2 elimination is a cheap special case of it.

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
- (pending) Julia `spanner.jl` / `sparsify.jl` / `build_sparsified_hierarchy.jl`
  / `sparsified_solve.jl` / `runtests.jl` / `bench.jl`. Baswana–Sen (`k=log n`)
  is the production spanner; the Python uses a greedy spanner for unambiguous
  correctness.

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

## Status
Gate + executable reference **done and validated in Python**. Next: implement in
cmg-python per `PYTHON-GUIDELINES.md` (fast iteration; I can run Python here,
not Julia), then port the validated algorithm to the Julia CMG and bench on
Wulver vs `cmg-k-elim`/`ac` on `uni_chimera(1e6,10)`, `wted_chimera(1e6,5)`.
No chimera generator is needed on the Python side — synthetic stalls suffice, or
dump one real stalled operator from a Julia run.
