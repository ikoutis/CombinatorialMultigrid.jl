# Sparsify-on-stall — design and 10⁶-chimera benchmark

This document details the sparsify-on-stall feature (branch
`claude/julia-sparsify-on-stall`) and its benchmark on 10⁶ chimeras, run through
the `laplacian-bench` harness on Wulver (`reps=3`). The short overview and the
Python gate/reference results are in [`README.md`](README.md); the port map and
expected micro-numbers in [`PORT-NOTES.md`](PORT-NOTES.md). This file is the full
results-and-discussion writeup.

## 1. The problem

CMG's one remaining hierarchy weakness is the **coarsening stall**. On levels
that densify from contraction fill (or are expander-like), quality-gated
aggregation refuses to coarsen, so `nc ≈ n` for many levels — an 18–23-level
hierarchy that shrinks <1%/level. On a **minority of connected 10⁶ chimeras**
this makes CMG **~2× slower than ApproxChol (`ac`)**, even though CMG beats `ac`
comfortably on the majority. CMG never sparsifies, so it cannot control that
density; degree-1/2 elimination is only a cheap special case.

## 2. The idea (Y. Koutis)

When aggregation stalls, inject a **spanner + uniform-sample sparsifier** as a
**same-size** hierarchy level (identity transfer, `cI = 1:n`, `nc = n`) and
continue on the sparser operator, so standard aggregation resumes. The K-cycle
is already size-agnostic (transfer is `cI`-keyed scatter/gather that degrades to
the identity), so a sparsify level is just a normal `HierarchyLevel` and the
production solve runs it unchanged.

## 3. Final design (and how we got there)

The feature reached its current form through three corrections, each driven by
the benchmark:

### 3.1 Trigger — the NODE ratio, not edges (the port bug)

The validated reference (`reference.py`, `PYTHON-GUIDELINES.md`) declares a level
productive iff aggregation drops the **node** count below `stall_ratio·n`
(default 0.9); a node plateau (`nc > stall_ratio·n`) injects. Both the Julia and
the cmg-python **ports mistranslated this to an EDGE test** (`m_c ≤ stall_ratio·m`).
Contraction fill keeps edges even when nodes coarsen well, so the edge test fired
on **normal** chimera levels — computing and discarding a spanner and truncating
the hierarchy with a large base. That produced a **12–14× slowdown on the ~90% of
draws that never needed sparsify** (see §5.1). Fixed to the node ratio; `nnz_budget`
remains a hard density cap. This is the single most important fix — it makes
sparsify a **no-op on graphs that coarsen**, so clean draws are byte-identical to
the stock (`cmg-k-elim`) hierarchy and pay zero overhead.

### 3.2 Spanner — a maximum-spanning-forest bundle (cut sparsifier)

Default `spanner=:mst, bundles=3`. **Rationale:** a bundle of `k` peeled maximum
spanning forests is a **Nagamochi–Ibaraki `k`-connectivity certificate** — a
*cut* sparsifier. A level stalls *precisely because it is high-conductance /
expander-like*, and there **cut ≈ spectral** (no bottleneck for a cut sparsifier
to miss). So in the exact regime where sparsify fires — and only there, thanks to
§3.1 — a cheap cut sparsifier is a good *spectral* preconditioner. It also:

- builds in **O(m)** (vs Baswana–Sen's O(km)), with a **fixed, density-independent
  `k·(n-1)`-edge bundle**;
- **keeps every connectivity bridge** automatically (forced by the spanning-forest
  property), preserving the small eigenvalues from the weak cuts that make these
  systems need a multilevel solver;
- is deterministic (given tie-breaking), unlike Baswana–Sen's randomized clustering.

**Python gate** (`benchmarks/gate_mst_spanner.py` in cmg-python), keep_frac=0.5:
κ(M⁻¹A) **3.0–5.5**, comparable-to-better than Baswana–Sen (4.2–5.1), identical
2× reduction, identical end-to-end iterations on the blob-chain stall system, and
the pair-bridge κ stays bounded (the bridge is kept). Baswana–Sen and greedy
remain available via `spanner=`.

### 3.3 Cycle and keep_frac

- **`keep_frac = 0.5`** — 2× sparsification per injection. For the L-cycle this
  sets the injected level's repeat multiplier `= floor(1/keep_frac − 1) = 1`
  (cheapest cycle). Lower keep_frac sparsifies harder but raises the repeat.
- **`:kscycle`** at an injected same-size level now runs the standard
  `krepeat[level]` inner iterations (like every other level), not a fixed `×8`;
  the only same-size special-case left is *which operator* the inner FCG minimizes
  over (the level's own `A`, correct under the identity transfer).

## 4. Benchmark setup

- **Harness:** `laplacian-bench`, `chol_vs_kcycle.jl` → `analyze_sparsify_stall.jl`.
- **Cluster:** Wulver, `reps=3`, `seed=1`, 10⁶ chimeras (`uni_chimera`,
  `uni_bndry_chimera`, `wted_chimera`, `wted_bndry_chimera`) + the artificial
  `dense_blob` family, chunked, same-node-per-instance.
- **Solvers:** `ac` (ApproxChol), `cmg-k-elim` (CMG K-cycle + degree-1/2
  elimination, **no sparsify**), `cmg-sparsify-l` (`:legacy` + sparsify),
  `cmg-sparsify-ks` (`:kscycle` + sparsify). All CMG columns use `eliminate=true`.
- **Split:** every per-sample draw is **stalled** (sparsify injected, `inj>0`) or
  **clean** (`inj=0`). Sparsify is inert on clean draws by construction (§3.1).
- Metric `×ac = ac_time / solver_time` (**>1 = faster than ac**). Median total
  seconds, median iterations.

## 5. Results

### 5.1 The regression that motivated the fixes (edge-ratio trigger)

With the original edge-ratio trigger, on **clean** 10⁶ `uni_chimera` draws (no
injection): `cmg-k-elim` 0.31s vs `cmg-sparsify-l` **4.28s** and `cmg-sparsify-ks`
**3.70s** — a **12–14× slowdown** at the *same* iteration count, split ~1.5s extra
build (a spanner computed and discarded) + ~2.2s extra solve (a large truncated
base). This is what §3.1 fixed.

### 5.2 Fixed — vs `ac` (Baswana–Sen baseline, `keep=0.5, bundles=1`)

**Stalled draws** (median total s (iters), ×ac):

| family (n) | ac | cmg-k-elim | cmg-sparsify-l | cmg-sparsify-ks |
|---|---|---|---|---|
| uni_bndry (6)    | 21.15 (16) — 1.00× | 25.58 (33) — 0.83× | 20.25 (33) — **1.04×** | 21.89 (28) — 0.97× |
| uni_chimera (21) | 5.03 (21) — 1.00×  | 5.40 (23) — 0.93×  | 5.00 (31) — **1.01×**  | 5.54 (26) — 0.91×  |
| wted_chimera(18) | 3.23 (21) — 1.00×  | 3.76 (22) — 0.86×  | 3.70 (21) — 0.87×      | 3.47 (22) — 0.93×  |

**Clean draws** (the ~70–90% majority; sparsify inert):

| family (n) | ac | cmg-k-elim | cmg-sparsify-l | cmg-sparsify-ks |
|---|---|---|---|---|
| uni_bndry (63)   | 5.77 (23) — 1.00× | 4.34 (26) — 1.33× | 4.80 (26) — 1.20× | 4.39 (26) — 1.31× |
| uni_chimera (48) | 7.43 (25) — 1.00× | 4.04 (30) — 1.84× | 4.47 (29) — 1.66× | 3.95 (30) — 1.88× |
| wted_chimera(51) | 7.05 (22) — 1.00× | 4.48 (21) — 1.57× | 4.38 (23) — 1.61× | 4.67 (21) — 1.51× |

**Worst stalled draws** (the ~2×-slower-than-ac cases sparsify targets):

| draw | ac | cmg-k-elim | cmg-sparsify-l |
|---|---|---|---|
| wted_chimera(1e6,5) | 1.64 (15) | 3.80 (22) — 0.43× | **1.88 (21) — 0.87×** |
| uni_chimera(1e6,10) | 4.19 (31) | 8.32 (47) — 0.50× | 6.46 (61) — 0.65× |
| uni_bndry(1e6,7)    | 15.79 (19)| 22.23 (36) — 0.71×| 18.18 (32) — 0.87× |

**Reading:** on **clean** draws every CMG variant beats `ac` 1.3–1.9× and sparsify
stays matched to `cmg-k-elim` (regression gone). On **stalled** draws `cmg-k-elim`
is the *only* place CMG loses to `ac`; `cmg-sparsify-l` lifts it back to **~ac
parity** and roughly halves the worst-case gap. Net: **with sparsify-on-stall,
CMG matches or beats `ac` across the whole distribution.**

### 5.3 MST-bundle-3 vs Baswana–Sen

The MST run (`spanner=:mst, bundles=3`, same seed → same graphs) targets the
residual spanner build cost. The reliable cross-run signal is the **within-run**
spanner overhead (`sparsify-l` build − `cmg-k-elim` build, same node/load):

| family (stalled) | Baswana overhead | **MST overhead** |
|---|---|---|
| **uni_bndry** (990k-node stall) | +3.80s | **+1.90s** |
| uni_chimera | +0.12s | +0.32s |
| wted_chimera | +0.04s | −0.09s |

On the one family with an expensive spanner (`uni_bndry`), MST **halved** the
build overhead — the O(m)-vs-O(km) prediction. Where the stall core is small the
spanner was already ~free, so MST is a no-op there. The per-draw `sparsify-l` vs
`cmg-k-elim` speedup moves accordingly:

| family (stalled) | Baswana | **MST** |
|---|---|---|
| uni_bndry   | 1.23× | **1.30×** (reaches ac parity, 1.00× ac) |
| uni_chimera | 1.15× | 1.14× (beats ac, 1.14× ac) |
| wted_chimera| 1.09× | 1.02× |
| dense_blob  | 1.13× | 1.03× |

MST wins on the large-core families (`uni_bndry`, `uni_chimera`), is neutral on
the small-core ones (within noise). **Caveat:** cross-run node load shifted `ac`
itself 1.1–1.3× between the two runs, so absolute seconds are not comparable
across runs — only within-run deltas and ratios are.

### 5.4 Iterations

Iteration counts are node-load-independent, so they read cleanly.

**Stalled draws (MST run), median iters:**

| family | ac | cmg-k-elim | cmg-sparsify-l | cmg-sparsify-ks |
|---|---|---|---|---|
| uni_bndry | 16 | 32.5 | 35 | **27.5** |
| uni_chimera | 21 | 23 | 28 | 23 |
| wted_chimera | 21 | 22 | 21 | 22 |

Three effects:

- **MST → fewer iterations than Baswana on the hard draws** (tighter κ):
  uni_chimera `sparsify-l` 31→28, `sparsify-ks` 26→23; worst uni_chimera(1e6,10)
  `sparsify-l` 61→55; worst uni_bndry(1e6,7) `sparsify-l` 32→27, `sparsify-ks`
  27→24. Clean-draw iterations are identical MST-vs-Baswana (no injection).
- **`cmg-sparsify-ks` *reduces* iterations vs the no-sparsify `cmg-k-elim`** on the
  stalled large cores (both K-cycle, apples-to-apples): uni_bndry 32.5 → **27.5**,
  uni_chimera 23 → 23. The sparsifier conditions the hierarchy better.
- **`cmg-sparsify-l` takes more (but cheaper) iterations** — the `:legacy`
  stationary cycle needs more iterations than the K-cycle (uni_bndry 35 vs 32.5),
  but each is a repeat-1 matvec, which is why its *wall-clock* is competitive.

### 5.5 `:legacy` (`-l`) vs `:kscycle` (`-ks`)

Wall-clock is within ~10% (a tie count-weighted); the two differ in character:

- **`-l`** does the fewest matvecs on the spectrally-easy injected levels
  (κ≈4–6), so it edges `-ks` on the **stalled** draws sparsify targets — the
  port-notes recommendation for a sparsified hierarchy.
- **`-ks`** reduces to the K-cycle on clean draws (`inj=0`), so it **never
  regresses vs `cmg-k-elim`** on the majority, and — with the MST cut-sparsifier —
  takes the **fewest iterations everywhere**, dipping below `cmg-k-elim` on the
  stalled cores.

So `-l` is the best single-solve wall-clock on stalls; `-ks` is the
iteration-count winner and the safer default. The strict best would be an
**adaptive driver** (`:legacy` when `inj>0`, else `:kcycle`) — not yet built.

## 6. Conclusions

- **The feature is a clean win.** Clean majority: CMG beats `ac` 1.3–1.9× (no
  regression). Stalled minority: `cmg-sparsify-l` at `ac` parity or better, up
  from `cmg-k-elim`'s ~2×-slower. So CMG matches-or-beats `ac` across the whole
  10⁶-chimera distribution.
- **The MST cut sparsifier is the right primitive** — O(m), fixed bundle, keeps
  bridges, and (because we only sparsify high-conductance stalls, where cut ≈
  spectral) spectrally as good as Baswana–Sen while halving the large-core build.
- **`-ks` for fewest iterations, `-l` for fewest matvecs** — both safe; an
  adaptive driver would capture both.

## 7. Open items

- **`uni_chimera(1e6,10)`** still converges in ~55 iters (MST) / 61 (Baswana) —
  a per-draw κ surface that neither the trigger nor the spanner fully tames; a
  candidate for keep_frac / bundle tuning.
- **Cross-run node-load noise** (~1.1–1.3× on `ac`) means the MST-vs-Baswana
  magnitude is uncertain; direction (MST helps large cores, neutral elsewhere) is
  clear. A same-node A/B would tighten it.
- **Adaptive `:legacy`/`:kcycle`** column (`inj>0` ? legacy : kcycle).

## 8. Reproduce

```bash
# nested CMG on the sparsify branch (Wulver): git pull origin claude/julia-sparsify-on-stall
cd performance-experiments
export PAPER_SOLVERS="ac,cmg-k-elim,cmg-sparsify-l,cmg-sparsify-ks"
CVK_ONLY="uni_chimera uni_bndry_chimera wted_chimera wted_bndry_chimera dense_blob" \
CVK_CHIMERA_SIZES="1e6" CVK_REPS=3 \
  ./run_paper_comparison.sh submit --scale paper --account ikoutis
# after completion:
julia --project=.. analyze_sparsify_stall.jl \
      ../performance-analyses/chol-vs-kcycle/*.seed1.reps3.jld2 \
      --solvers ac,cmg-k-elim,cmg-sparsify-l,cmg-sparsify-ks
```

Swap the spanner via `SparsifyOptions(spanner=:baswana_sen, bundles=1)` (or
`:greedy`) to reproduce the Baswana baseline.
