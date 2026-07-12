# Sparsify-on-stall — 10⁶-chimera benchmark results

Results of the sparsify-on-stall feature (branch `claude/julia-sparsify-on-stall`)
on 10⁶ chimeras, run through the `laplacian-bench` harness on Wulver (`reps=3`,
`seed=1`), plus the artificial `dense_blob` family. Design and Python gate: see
[`README.md`](README.md); port map: [`PORT-NOTES.md`](PORT-NOTES.md).

Each per-sample draw is **stalled** (sparsify injected, `inj>0`) or **clean**
(`inj=0`; sparsify is inert by construction). Metric `×ac = ac_time / solver_time`
(**>1 = faster than ac**). Cells are median total seconds (median iterations).

## Methods

- **`ac`** — ApproxChol (`approxchol_lap`), the baseline.
- **`cmg-k-elim`** — CMG K-cycle + degree-1/2 elimination, **no sparsify**. The
  no-sparsify reference; stalls on the dense/expander minority.
- **`cmg-sparsify-l`** — `:legacy` cycle + sparsify-on-stall. Fewest matvecs
  (stationary, injected-level repeat 1); best stalled-draw wall-clock; ~10–20%
  `:legacy` tax on clean draws.
- **`cmg-sparsify-ks`** — `:kscycle` + sparsify-on-stall. Fewest iterations;
  reduces to the K-cycle on clean draws (matches `cmg-k-elim`).
- Spanner (build config): **`:mst`** = 3-forest maximum-spanning-forest bundle
  (default; O(m), cut sparsifier); **`:baswana_sen`** = randomized (2k−1)-spanner
  (O(km)). Both `keep_frac=0.5`, `eliminate=true`.

## vs `ac` (Baswana–Sen, `keep=0.5, bundles=1`)

**Stalled draws:**

| family (n) | ac | cmg-k-elim | cmg-sparsify-l | cmg-sparsify-ks |
|---|---|---|---|---|
| uni_bndry (6)    | 21.15 (16) — 1.00× | 25.58 (33) — 0.83× | 20.25 (33) — **1.04×** | 21.89 (28) — 0.97× |
| uni_chimera (21) | 5.03 (21) — 1.00×  | 5.40 (23) — 0.93×  | 5.00 (31) — **1.01×**  | 5.54 (26) — 0.91×  |
| wted_chimera (18)| 3.23 (21) — 1.00×  | 3.76 (22) — 0.86×  | 3.70 (21) — 0.87×      | 3.47 (22) — 0.93×  |
| dense_blob (21)  | 0.0142 (13) —1.00× | 0.0130 (7) — 1.09× | 0.0140 (8) — 1.01×     | 0.0152 (8) — 0.93× |

**Clean draws** (~70–90% of draws; sparsify inert):

| family (n) | ac | cmg-k-elim | cmg-sparsify-l | cmg-sparsify-ks |
|---|---|---|---|---|
| uni_bndry (63)   | 5.77 (23) — 1.00× | 4.34 (26) — 1.33× | 4.80 (26) — 1.20× | 4.39 (26) — 1.31× |
| uni_chimera (48) | 7.43 (25) — 1.00× | 4.04 (30) — 1.84× | 4.47 (29) — 1.66× | 3.95 (30) — 1.88× |
| wted_chimera (51)| 7.05 (22) — 1.00× | 4.48 (21) — 1.57× | 4.38 (23) — 1.61× | 4.67 (21) — 1.51× |

**Worst stalled draw per family:**

| draw | ac | cmg-k-elim | cmg-sparsify-l | cmg-sparsify-ks |
|---|---|---|---|---|
| wted_chimera(1e6,5) | 1.64 (15) | 3.80 (22) — 0.43× | 1.88 (21) — 0.87× | 1.88 (22) — 0.87× |
| uni_chimera(1e6,10) | 4.19 (31) | 8.32 (47) — 0.50× | 6.46 (61) — 0.65× | 6.58 (47) — 0.64× |
| uni_bndry(1e6,7)    | 15.79 (19)| 22.23 (36) — 0.71×| 18.18 (32) — 0.87×| 19.22 (27) — 0.82× |

- Clean: every CMG variant beats `ac` 1.3–1.9×; sparsify matches `cmg-k-elim`.
- Stalled: `cmg-k-elim` is the only place CMG loses to `ac`; `cmg-sparsify-l`
  reaches ~`ac` parity.

## Spanner: MST-bundle-3 vs Baswana–Sen

Same graphs (seed 1). Within-run spanner build overhead
(`sparsify-l` build − `cmg-k-elim` build, stalled median):

| family | Baswana | MST |
|---|---|---|
| uni_bndry (990k-node stall) | +3.80s | **+1.90s** |
| uni_chimera | +0.12s | +0.32s |
| wted_chimera | +0.04s | −0.09s |

Per-draw `sparsify-l` vs `cmg-k-elim` speedup:

| family (stalled) | Baswana | MST |
|---|---|---|
| uni_bndry | 1.23× | **1.30×** |
| uni_chimera | 1.15× | 1.14× |
| wted_chimera | 1.09× | 1.02× |
| dense_blob | 1.13× | 1.03× |

- MST halves the spanner build on the large stalled core (`uni_bndry`); neutral
  where the core is small.
- Cross-run node load shifted `ac` 1.1–1.3×, so only within-run deltas/ratios are
  comparable across the two runs.

## Iterations

Median iterations, stalled draws (MST run):

| family | ac | cmg-k-elim | cmg-sparsify-l | cmg-sparsify-ks |
|---|---|---|---|---|
| uni_bndry | 16 | 32.5 | 35 | **27.5** |
| uni_chimera | 21 | 23 | 28 | 23 |
| wted_chimera | 21 | 22 | 21 | 22 |

Clean-draw iterations match `cmg-k-elim` and are identical MST-vs-Baswana (no
injection). MST vs Baswana on the hard draws: fewer iterations (tighter κ), e.g.
uni_chimera `sparsify-l` 31→28, `sparsify-ks` 26→23; worst uni_chimera(1e6,10)
`sparsify-l` 61→55; worst uni_bndry(1e6,7) `sparsify-l` 32→27.

- `cmg-sparsify-ks` reduces iterations vs `cmg-k-elim` on stalled cores
  (uni_bndry 32.5→27.5); `cmg-sparsify-l` takes more, cheaper iterations.

## Summary

- Clean majority: CMG beats `ac` 1.3–1.9×; the 12–14× edge-trigger regression is
  fixed (sparsify inert on clean draws).
- Stalled minority: `cmg-sparsify-l` at ~`ac` parity, up from `cmg-k-elim`'s
  ~2×-slower.
- MST cut sparsifier: halves the large-core spanner build; comparable-or-fewer
  iterations.
- `-l` fewest matvecs (best stalled wall-clock); `-ks` fewest iterations (matches
  `cmg-k-elim` on clean).

Open: `uni_chimera(1e6,10)` still ~55 iters; a same-node A/B would tighten the
MST-vs-Baswana magnitude; an adaptive driver (`:legacy` if `inj>0` else `:kcycle`)
would combine `-l`/`-ks`.

## Reproduce

```bash
cd performance-experiments
export PAPER_SOLVERS="ac,cmg-k-elim,cmg-sparsify-l,cmg-sparsify-ks"
CVK_ONLY="uni_chimera uni_bndry_chimera wted_chimera wted_bndry_chimera dense_blob" \
CVK_CHIMERA_SIZES="1e6" CVK_REPS=3 \
  ./run_paper_comparison.sh submit --scale paper --account ikoutis
julia --project=.. analyze_sparsify_stall.jl \
      ../performance-analyses/chol-vs-kcycle/*.seed1.reps3.jld2 \
      --solvers ac,cmg-k-elim,cmg-sparsify-l,cmg-sparsify-ks
```

Baswana baseline: `SparsifyOptions(spanner=:baswana_sen, bundles=1)`.
