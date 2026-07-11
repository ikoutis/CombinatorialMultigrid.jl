# Sparsified-CMG — guidelines for the cmg-python implementation

Spec for adding **sparsify-on-stall** to a Python CMG. The executable ground
truth is [`reference.py`](reference.py) in this directory (self-contained, runs
with `python3 reference.py`); this doc explains what it does, the design
decisions, and how to port it into a real CMG. The Julia port + real-chimera
benchmark happen separately in this repo / on Wulver.

## The problem and the idea

CMG's aggregation stalls on levels that **densify from contraction fill** (or are
expander-like): a quality-gated aggregation refuses to coarsen high-effective-
degree nodes, so `nc ≈ n` for many levels (18–23-level hierarchies, ~2× slower
than ApproxChol on some 10⁶ chimeras). CMG **never sparsifies**, so it can't
control that density.

**Fix:** when aggregation stalls, inject a **spanner + uniform-sample spectral
sparsifier** as the next level (a same-size level, ~2× fewer edges). The sparser
operator lets standard aggregation resume. Sparsification is the general
density-control primitive; degree-1/2 elimination is a cheap special case.

The reference demonstrates, on a chain of dense blobs with weak cuts (κ≈2.6e7):
- **stall → resume:** aggregation `nc/n` goes 0.998 → 0.63 (bundles=1) / 0.39
  (bundles=2) after one sparsify, with a spectrally-close sparsifier (κ(M⁻¹A)≈21);
- **end-to-end:** a V-cycle-PCG that *stalls without the fix* needs ~50 iters (and
  loses accuracy to the ill-conditioning), *with the fix* coarsens 960→~20 and
  converges in ~12 iters. The gap grows with κ/scale.

## Algorithm (mirror `reference.py`)

### 1. Spanner + bundle
- **Reference** uses a greedy spanner in the **resistance metric** (edge length =
  1/conductance): keep edge (u,v) unless the current spanner already connects
  them with resistance ≤ t·r_uv. A t-spanner bounds every off-spanner edge's
  effective resistance by t·r, i.e. **leverage ≤ t**, which is what makes uniform
  sampling concentrate (this is the spanner-based spectral sparsifier).
- **Production:** use **Baswana–Sen** (k = ceil(log n) → stretch 2k−1, O(m)
  expected, cluster-based). Greedy is O(m·Dijkstra) — fine for the reference/small
  graphs, too slow at scale.
- **Bundle iteration** (`spanner_bundle`): peel B spanners (extract spanner →
  remove its edges → repeat). Default **B=1**; more bundles lower the effective
  stretch, so aggregation resumes harder and κ tightens at the same edge budget
  (reference: bundles 1→2 drops nc/n 0.63→0.39). Keep it as a knob.

### 2. Adaptive sparsifier (`sparsify`)
**Do not** use a fixed keep-probability. Choose `p` from the measured bundle size
to hit a target keep-fraction (a controlled, gentle geometric drop):

```
bundle, off = spanner_bundle(n, edges, t, bundles)   # S = |bundle|, m = |edges|
p = clamp((keep_frac*m - S) / (m - S), 0, 1)
kept = bundle(weight 1) + [ (u,v, w/p) for e in off if rand() < p ]   # unbiased reweight
```
- Sparse level (S ≈ m) → p → 0 → keep just the bundle (no over-sparsifying).
- Dense/fill level (S ≪ m) → p ≈ keep_frac → geometric drop by keep_frac.
- **keep_frac ≈ 0.5** ("not overly done"): each injection is mild → better κ; inject
  a few times if one pass isn't enough (total reduction keep_fracᵏ).
- Spanner kept at weight 1, only sampled edges reweighted by 1/p → unbiased
  estimator of the off-bundle part. (An ultrasparsifier variant that *boosts* the
  bundle is also valid but is a preconditioner, not ≈A — pick one and be explicit.)

### 3. Stall detector + build loop (`build`)
Reuse the existing aggregation. Change only what happens **on stall**:

```
while n > base:
    cI, nc = aggregate(A)                         # existing CMG aggregation
    if nc <= stall_ratio * n:                     # productive coarsening (e.g. stall_ratio=0.9)
        push normal level (cI); A = R A Rᵀ        # contract, continue
    elif sparsify_on_stall and injected < max_inject:
        A_sp = build_sparsifier(A, keep_frac, bundles, t)   # keep the SDD slack
        if |edges(A_sp)| >= 0.98*|edges(A)|: break # couldn't reduce -> stop
        push SAME-SIZE level (cI = identity); A = A_sp; injected += 1
    else:
        break                                     # stalled, no fix -> this is the base
```
Cap `max_inject` (termination guarantee) and keep any existing nnz-growth guard,
counting the sparsifier's nnz.

### 4. The same-size level and the cycle — **the key integration check**
A sparsify level has `cI = identity` (`nc = n`), so its restriction/prolongation
are the **identity**. In the Julia CMG the cycle keys on `cI` (scatter/gather),
so a same-size level flows through with **no special case**. **Verify the same in
cmg-python:** if its cycle builds an explicit `R`/`P` matrix or asserts `nc < n`,
add a small branch so a same-size level uses identity transfer and recurses into
the sub-hierarchy of the sparser operator (which then coarsens normally). The base
solve and smoother are unchanged (`invD` from the sparsifier's diagonal).

## Isolation & knobs
- Put the spanner/sparsify code in its **own module**, not entangled with the
  solver (as we kept it isolated here). Two Claude instances on cmg-python →
  clean module boundary to avoid merge pain.
- Knobs (with reference defaults): `bundles=1`, `keep_frac=0.5`, spanner stretch
  `t≈log n`, `stall_ratio=0.9`, `max_inject≈10`, `base` = existing direct-solve
  threshold. `sparsify_on_stall` is the on/off flag (default off until validated,
  then on).

## What to validate on the Python side (no chimera generator needed)
1. **Stall → resume:** on a synthetic dense/stall graph, `nc/n ≈ 1` before,
   `≪ 1` after one sparsify; sparsifier κ(M⁻¹A) bounded. (reference Part 1)
2. **End-to-end:** V-cycle-PCG iterations with vs without `sparsify_on_stall` on
   an ill-conditioned stall graph (chain of dense blobs + weak cuts); assert the
   solution matches a direct solve. (reference Part 2)
3. **Correctness:** `rel_err` vs `A\b` < 1e-8-ish on SDD test systems.

You do **not** need `uni_chimera` in Python. Synthetic stalls (dense blobs /
chains) reproduce the phenomenon. For a real-data check, dump **one** stalled
level operator from a Julia CMG run on Wulver (e.g. the ~8400-node plateau of
`uni_chimera(1e6,10)`) to `.npz`/`.mtx` and load that single matrix — no generator
port. The full chimera-vs-`ac` benchmark stays in Julia/Wulver.

## Porting back
Once validated in Python: port the (small) build-loop fork + the spanner/sparsify
module to the Julia CMG (`experiments/sparsified/` here), then bench on Wulver
against `cmg-k-elim` and `ac` on `uni_chimera(1e6,10)` / `wted_chimera(1e6,5)` —
hierarchy depth, iterations, build+solve time. The solve side needs no changes in
Julia (verified); confirm the same for cmg-python per §4.

## Caveats
- The reference's aggregation gate (`w_max/deg ≥ τ`) is a **stylized** stand-in
  for CMG's effective-degree clustering — enough to reproduce stall→resume, not a
  replacement for the real rule. Use cmg-python's actual aggregation; only the
  on-stall branch is new.
- κ at moderate density is ~20 in the reference (single greedy bundle, keep_frac
  0.5). Tighten via bundles and keep_frac; the exact spanner-based construction
  (yours) sets the real constants.
